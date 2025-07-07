#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import openslide
from utils.logging_utils import setup_logging
from utils.stain_norm_function import preprocess_patch_with_stain_normalization
from utils.path_utils import extract_hospital_info
from utils.tsne_plot_function import (
    create_tsne_visualization,
    create_hospital_tsne_visualization,
    create_label_hospital_tsne_visualization
)

# Phikon Model Imports
try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: transformers library not available. Please install transformers.")
    print("Installation: pip install transformers")
    sys.exit(1)

# Configuration class
class Config:
    """Configuration parameters"""
    phikon_model_name = "phikon-v2"  # Options: "phikon", "phikon-v2"
    feature_dim = 768  # Will be updated based on model
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
    enable_stain_normalization = True
    stain_normalization_method = "macenko"  # Options: "macenko", "vahadane", "histaug"
    stain_target_image_path = None
    histaug_probability = 0.5
    histaug_intensity = 0.3
    max_tsne_points = 10000
    image_size = 224
    random_seed = 42

# Import stain normalization functions from utils

def setup_logging(output_dir):
    return setup_logging(output_dir, "phikon_feature_tsne_stain_norm.log")

# Phikon model loading and feature extraction

def initialize_phikon_model(config, device, logger=None):
    if logger:
        logger.info(f"Initializing Phikon model: {config.phikon_model_name}...")
    print(f"Initializing Phikon model: {config.phikon_model_name}...")
    try:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")
        if config.phikon_model_name.lower() == "phikon":
            model_path = "owkin/phikon"
            print("Loading Phikon (original) model...")
        elif config.phikon_model_name.lower() == "phikon-v2":
            model_path = "owkin/phikon-v2"
            print("Loading Phikon-v2 (enhanced) model...")
        else:
            raise ValueError(f"Unsupported Phikon model: {config.phikon_model_name}")
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        model.to(device)
        with torch.no_grad():
            dummy_image = Image.new('RGB', (config.image_size, config.image_size), color='white')
            inputs = processor(dummy_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs[0][:, 0, :] if len(outputs[0].shape) > 2 else outputs[0]
            actual_feature_dim = features.shape[1]
            config.feature_dim = actual_feature_dim
        if logger:
            logger.info(f"Phikon model {config.phikon_model_name} loaded successfully")
            logger.info(f"Feature dimension: {config.feature_dim}")
        print(f"Phikon model {config.phikon_model_name} loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Feature dimension: {config.feature_dim}")
        return model, processor
    except Exception as e:
        error_msg = f"Failed to initialize Phikon model: {e}"
        print(error_msg)
        if logger:
            logger.error(error_msg)
        raise e

def extract_features_with_phikon(model, processor, image, device):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0, :]
        else:
            features = outputs[0][:, 0, :] if len(outputs[0].shape) > 2 else outputs[0]
    return features

# Main feature extraction pipeline

def extract_features_from_patches(config, filtered_patches_df, output_dir, logger=None):
    print("\n" + "="*60)
    print(f"Feature Extraction with Phikon Model: {config.phikon_model_name}")
    if config.enable_stain_normalization:
        print(f"Stain Normalization: {config.stain_normalization_method}")
    print("="*60)
    if logger:
        logger.info(f"Feature Extraction with Phikon Model: {config.phikon_model_name}")
        if config.enable_stain_normalization:
            logger.info(f"Stain Normalization: {config.stain_normalization_method}")
        logger.info("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, processor = initialize_phikon_model(config, device, logger)
    model_name = config.phikon_model_name.upper().replace("-", "")
    if config.enable_stain_normalization:
        features_dir = os.path.join(output_dir, f"features_{model_name}_{config.stain_normalization_method}")
    else:
        features_dir = os.path.join(output_dir, f"features_{model_name}")
    os.makedirs(features_dir, exist_ok=True)
    wsi_groups = filtered_patches_df.groupby('wsi_id')
    desc = f"Extracting {config.phikon_model_name.upper()} features"
    if config.enable_stain_normalization:
        desc += f" ({config.stain_normalization_method})"
    for wsi_id, wsi_patches in tqdm(wsi_groups, desc=desc):
        features_list = []
        for idx, row in wsi_patches.iterrows():
            patch_x = row["patch_x"]
            patch_y = row["patch_y"]
            patch_size = row["patch_size"]
            wsi_file = os.path.join(config.wsi_dir, f"{wsi_id}.svs")
            if not os.path.exists(wsi_file):
                continue
            try:
                slide = openslide.OpenSlide(wsi_file)
                patch = slide.read_region((patch_x, patch_y), 0, (patch_size, patch_size))
                patch = patch.convert('RGB')
                slide.close()
                patch = preprocess_patch_with_stain_normalization(patch, config)
                features = extract_features_with_phikon(model, processor, patch, device)
                features_list.append(features.cpu().numpy())
            except Exception as e:
                if logger:
                    logger.error(f"Error extracting features for patch {wsi_id}_{patch_x}_{patch_y}: {e}")
                continue
        if features_list:
            features_array = np.vstack(features_list)
            features_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
            torch.save(torch.from_numpy(features_array), features_file)
            if logger:
                logger.info(f"Features saved for {wsi_id}: {len(features_list)} patches")
    print(f"Feature extraction completed. Features saved in: {features_dir}")
    if logger:
        logger.info(f"Feature extraction completed. Features saved in: {features_dir}")
    return features_dir

# t-SNE visualization
def create_tsne_visualizations(config, output_dir, features_dir, filtered_patches_df, logger=None):
    """Create t-SNE visualizations using unified functions
    
    Args:
        config: Configuration object
        output_dir: Output directory
        features_dir: Directory containing feature files
        filtered_patches_df: DataFrame with original patch information
        logger: Logger instance
    """
    print("\n" + "="*60)
    print("t-SNE Visualization")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all feature files
    feature_files = glob.glob(os.path.join(features_dir, "*_features.pt"))
    print(f"Found {len(feature_files)} feature files")
    
    if logger:
        logger.info(f"Found {len(feature_files)} feature files")
    
    all_features = []
    all_labels = []
    all_hospitals = []
    
    for pt_file in tqdm(feature_files, desc="Loading features"):
        try:
            # Load features
            features = torch.load(pt_file, weights_only=True)
            
            # Get WSI ID from filename
            wsi_id = os.path.basename(pt_file).replace("_features.pt", "")
            
            # Get patch info from original DataFrame
            wsi_patches = filtered_patches_df[filtered_patches_df['wsi_id'] == wsi_id]
            if len(wsi_patches) == 0:
                print(f"Warning: No patch info found for {wsi_id} in original data")
                continue
            
            hospital = extract_hospital_info(wsi_id)
            
            # Add features and metadata
            all_features.append(features)
            all_labels.extend(wsi_patches['patch_label'].tolist())
            all_hospitals.extend([hospital] * len(features))
                
        except Exception as e:
            print(f"Error loading file {pt_file}: {e}")
            continue
    
    if not all_features:
        print("No valid feature files found")
        return
    
    # Merge all features
    all_features = np.vstack(all_features)
    print(f"Total features: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    
    # Random sampling for acceleration
    if len(all_features) > config.max_tsne_points:
        np.random.seed(config.random_seed)
        indices = np.random.choice(len(all_features), config.max_tsne_points, replace=False)
        all_features = all_features[indices]
        all_labels = [all_labels[i] for i in indices]
        all_hospitals = [all_hospitals[i] for i in indices]
        print(f"Sampled features: {len(all_features)}")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Model name for filename
    model_name = config.phikon_model_name.upper().replace("-", "")
    if config.enable_stain_normalization:
        filename_suffix = f"{model_name}_{config.stain_normalization_method}"
    else:
        filename_suffix = model_name
    
    # Create title with model and stain normalization info
    title = f"t-SNE Visualization ({model_name})"
    if config.enable_stain_normalization:
        title += f" [StainNorm: {config.stain_normalization_method}]"
    
    # Create t-SNE visualizations using unified functions
    print("Creating t-SNE visualizations...")
    
    # 1. By label
    label_viz_file = os.path.join(viz_dir, f"tsne_{filename_suffix}_by_label.png")
    create_tsne_visualization(all_features, all_labels, all_hospitals, label_viz_file, title)
    print(f"Label-grouped t-SNE plot saved: {label_viz_file}")
    
    # 2. By hospital
    hosp_viz_file = os.path.join(viz_dir, f"tsne_{filename_suffix}_by_hospital.png")
    create_hospital_tsne_visualization(all_features, all_labels, all_hospitals, hosp_viz_file, title)
    print(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
    
    # 3. Combined (label + hospital)
    combined_viz_file = os.path.join(viz_dir, f"tsne_{filename_suffix}_combined.png")
    create_label_hospital_tsne_visualization(all_features, all_labels, all_hospitals, combined_viz_file, title)
    print(f"Combined t-SNE plot saved: {combined_viz_file}")
    
    # Statistics
    print(f"\n=== Visualization statistics ===")
    print(f"Total patches: {len(all_features)}")
    print(f"Hospital count: {len(set(all_hospitals))}")
    print(f"Hospital distribution: {dict(pd.Series(all_hospitals).value_counts())}")
    print(f"Label distribution: {dict(pd.Series(all_labels).value_counts())}")
    
    if logger:
        logger.info(f"Label-grouped t-SNE plot saved: {label_viz_file}")
        logger.info(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
        logger.info(f"Combined t-SNE plot saved: {combined_viz_file}")

# Main pipeline

def main():
    parser = argparse.ArgumentParser(description='Phikon Feature Extraction and t-SNE Visualization with Stain Normalization')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing conch_filtered_patches.csv')
    parser.add_argument('--phikon_model', type=str, default='phikon-v2', choices=['phikon', 'phikon-v2'], help='Phikon model choice: phikon/phikon-v2')
    parser.add_argument('--stain_norm', type=str, default='macenko', choices=['none', 'macenko', 'vahadane', 'histaug'], help='Stain normalization method: none/macenko/vahadane/histaug')
    parser.add_argument('--histaug_prob', type=float, default=0.5, help='HistAug application probability (0.0-1.0)')
    parser.add_argument('--histaug_intensity', type=float, default=0.3, help='HistAug intensity (0.0-1.0)')
    args = parser.parse_args()
    Config.phikon_model_name = args.phikon_model
    Config.enable_stain_normalization = (args.stain_norm != 'none')
    Config.stain_normalization_method = args.stain_norm if args.stain_norm != 'none' else 'macenko'
    Config.histaug_probability = args.histaug_prob
    Config.histaug_intensity = args.histaug_intensity
    print("Phikon Feature Extraction and t-SNE Visualization Pipeline Starting")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Phikon model: {args.phikon_model}")
    if Config.enable_stain_normalization:
        print(f"Stain normalization: {Config.stain_normalization_method}")
        if Config.stain_normalization_method == "histaug":
            print(f"  - HistAug probability: {Config.histaug_probability}")
            print(f"  - HistAug intensity: {Config.histaug_intensity}")
    else:
        print("Stain normalization: disabled")
    if args.phikon_model == "phikon":
        print("Model info: Phikon (original, ViT-B/16, specialized for histopathology)")
    elif args.phikon_model == "phikon-v2":
        print("Model info: Phikon-v2 (enhanced, improved performance for pathology)")
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library not available for Phikon model loading.")
        print("Please install transformers: pip install transformers")
        return
    logger = setup_logging(args.input_dir)
    try:
        filtered_patches_file = os.path.join(args.input_dir, "conch_filtered_patches.csv")
        if not os.path.exists(filtered_patches_file):
            raise FileNotFoundError(f"Filtered patches file not found: {filtered_patches_file}")
        filtered_patches_df = pd.read_csv(filtered_patches_file)
        print(f"Loaded {len(filtered_patches_df)} filtered patches")
        if logger:
            logger.info(f"Loaded {len(filtered_patches_df)} filtered patches")
        features_dir = extract_features_from_patches(Config, filtered_patches_df, args.input_dir, logger)
        create_tsne_visualizations(Config, args.input_dir, features_dir, filtered_patches_df, logger)
        print("\n" + "="*60)
        print("Pipeline completed successfully")
        print(f"Phikon model used: {Config.phikon_model_name}")
        if Config.enable_stain_normalization:
            print(f"Stain normalization used: {Config.stain_normalization_method}")
        else:
            print("Stain normalization: disabled")
        print(f"Total patches processed: {len(filtered_patches_df)}")
        print(f"Results saved in: {args.input_dir}")
        print("="*60)
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully")
            logger.info(f"Phikon model used: {Config.phikon_model_name}")
            if Config.enable_stain_normalization:
                logger.info(f"Stain normalization used: {Config.stain_normalization_method}")
                if Config.stain_normalization_method == "histaug":
                    logger.info(f"HistAug probability: {Config.histaug_probability}, intensity: {Config.histaug_intensity}")
            else:
                logger.info("Stain normalization: disabled")
            logger.info(f"Total patches processed: {len(filtered_patches_df)}")
            logger.info(f"Results saved in: {args.input_dir}")
            logger.info("="*60)
    except Exception as e:
        error_msg = f"Pipeline error: {e}"
        print(f"\n{error_msg}")
        if logger:
            logger.error(error_msg)
        import traceback
        traceback.print_exc()
        if logger:
            logger.error("Full traceback:", exc_info=True)

if __name__ == "__main__":
    main() 
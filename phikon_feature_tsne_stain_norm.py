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
        filename_prefix = f"tsne_{model_name}_{config.stain_normalization_method}"
    else:
        filename_prefix = f"tsne_{model_name}"
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
    labels = all_labels
    model_name = config.phikon_model_name.upper().replace("-", "")
    if config.enable_stain_normalization:
        filename_prefix = f"tsne_{model_name}_{config.stain_normalization_method}"
    else:
        filename_prefix = f"tsne_{model_name}"
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    # 1. 全部patch整体t-SNE，按医院上色
    print("Starting t-SNE dimensionality reduction (all patches)...")
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, random_state=getattr(config, 'random_seed', 42), perplexity=getattr(config, 'tsne_perplexity', 30), max_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    df_tsne = pd.DataFrame({
        'TSNE1': features_2d[:, 0],
        'TSNE2': features_2d[:, 1],
        'Hosp': all_hospitals,
        'Label': labels
    })
    unique_hosps = sorted(set(all_hospitals))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_hosps)))
    color_map = dict(zip(unique_hosps, colors))
    # by_hosp
    plt.figure(figsize=(12, 10))
    for hosp in unique_hosps:
        mask = df_tsne['Hosp'] == hosp
        plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'], c=[color_map[hosp]], label=hosp, alpha=0.7, s=50)
    plt.title(f't-SNE Visualization ({model_name}, {config.stain_normalization_method}) - Grouped by Hospital', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    hosp_viz_file = os.path.join(viz_dir, f"{filename_prefix}_by_hosp.png")
    plt.savefig(hosp_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
    if logger:
        logger.info(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
    # 2. by_label
    unique_labels = sorted(set(labels))
    label_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, label_colors))
    plt.figure(figsize=(12, 10))
    for label in unique_labels:
        mask = df_tsne['Label'] == label
        plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'], c=[label_color_map[label]], label=label, alpha=0.7, s=50)
    plt.title(f't-SNE Visualization ({model_name}, {config.stain_normalization_method}) - Grouped by Label', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    label_viz_file = os.path.join(viz_dir, f"{filename_prefix}_by_label.png")
    plt.savefig(label_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Label-grouped t-SNE plot saved: {label_viz_file}")
    if logger:
        logger.info(f"Label-grouped t-SNE plot saved: {label_viz_file}")
    # 3. by_hosp_IDC/ILC
    for subtype in ["IDC", "ILC"]:
        mask = [l == subtype for l in labels]
        if sum(mask) == 0:
            continue
        features_sub = all_features[mask]
        hosps_sub = [all_hospitals[i] for i, flag in enumerate(mask) if flag]
        tsne = TSNE(n_components=2, random_state=getattr(config, 'random_seed', 42), perplexity=getattr(config, 'tsne_perplexity', 30), max_iter=1000)
        features_2d = tsne.fit_transform(features_sub)
        df = pd.DataFrame({
            'TSNE1': features_2d[:, 0],
            'TSNE2': features_2d[:, 1],
            'Hosp': hosps_sub
        })
        unique_hosps_sub = sorted(set(hosps_sub))
        colors_sub = plt.cm.Set3(np.linspace(0, 1, len(unique_hosps_sub)))
        color_map_sub = dict(zip(unique_hosps_sub, colors_sub))
        plt.figure(figsize=(12, 10))
        for hosp in unique_hosps_sub:
            mask_h = df['Hosp'] == hosp
            plt.scatter(df[mask_h]['TSNE1'], df[mask_h]['TSNE2'], c=[color_map_sub[hosp]], label=hosp, alpha=0.7, s=50)
        plt.title(f't-SNE ({model_name}, {config.stain_normalization_method}) - {subtype} by Hospital', fontsize=16, fontweight='bold')
        plt.xlabel('TSNE1', fontsize=12)
        plt.ylabel('TSNE2', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        out_file = os.path.join(viz_dir, f"{filename_prefix}_by_hosp_{subtype}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{subtype} by hospital t-SNE plot saved: {out_file}")
        if logger:
            logger.info(f"{subtype} by hospital t-SNE plot saved: {out_file}")
    
    # Statistics
    print(f"\n=== Visualization statistics ===")
    print(f"Total patches: {len(all_features)}")
    print(f"Hospital count: {len(set(all_hospitals))}")
    print(f"Hospital distribution: {dict(pd.Series(all_hospitals).value_counts())}")
    print(f"Label distribution: {dict(pd.Series(all_labels).value_counts())}")
    
    # 删除combined_viz_file相关logger语句
    # if logger:
    #     logger.info(f"Label-grouped t-SNE plot saved: {label_viz_file}")
    #     logger.info(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
    #     logger.info(f"Combined t-SNE plot saved: {combined_viz_file}")

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
    logger = setup_logging(args.input_dir, "phikon_feature_tsne_stain_norm.log")
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
#!/usr/bin/env python3
"""
CONCH Feature Extraction and t-SNE Visualization Pipeline

This script extracts features from filtered patches using CONCH model
and creates t-SNE visualizations grouped by hospital and label.

Model: CONCH (Contrastive Learning for Computational Histopathology)
Features both visual feature extraction and zero-shot capabilities.

Author: AI Assistant
Date: 2025-06-28
"""

# ===================== Standard Library Imports =====================
import os
import sys
import argparse
import glob
import logging
from datetime import datetime

# 添加本地 CONCH 模块到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CONCH'))

# ===================== Third-party Library Imports =====================
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import openslide
from sklearn.manifold import TSNE

# ===================== CONCH Model Imports =====================
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False
    print("Warning: CONCH not available. Please install CONCH first.")
    print("Installation: pip install git+https://github.com/mahmoodlab/CONCH.git")

# ===================== Configuration =====================
class Config:
    """Configuration class for the pipeline"""
    
    # CONCH model parameters
    conch_checkpoint_path = './checkpoints/conch/pytorch_model.bin'
    conch_model_cfg = 'conch_ViT-B-16'
    feature_dim = 512  # CONCH visual encoder output dimension
    
    # WSI path
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
    
    # t-SNE parameters
    tsne_perplexity = 30
    tsne_max_iter = 1000
    max_tsne_points = 10000  # Maximum sampling points for t-SNE

# ===================== Utility Functions =====================
def extract_hosp_from_filename(filename):
    """Extract hospital information from filename
    
    Args:
        filename (str): WSI filename
        
    Returns:
        str: Hospital code or "Unknown"
    """
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[1]
    return "Unknown"

def setup_logging(output_dir):
    """Setup logging configuration
    
    Args:
        output_dir (str): Directory to save log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    log_file = os.path.join(output_dir, "conch_feature_tsne.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("CONCH Feature Extraction and t-SNE Visualization Started")
    logger.info("Note: CONCH supports both feature extraction and zero-shot capabilities")
    logger.info("=" * 60)
    
    return logger

# ===================== CONCH Model Functions =====================
def initialize_conch_model(config, device, logger=None):
    """Initialize CONCH model
    
    Args:
        config: Configuration object
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        tuple: (model, preprocess)
    """
    if not CONCH_AVAILABLE:
        raise ImportError("CONCH is not available. Please install CONCH first.")
    
    if logger:
        logger.info("Initializing CONCH model...")
    
    print("Initializing CONCH model...")
    print("Note: CONCH supports both visual feature extraction and zero-shot classification")
    
    # Check if checkpoint exists
    if not os.path.exists(config.conch_checkpoint_path):
        raise FileNotFoundError(f"CONCH checkpoint not found: {config.conch_checkpoint_path}")
    
    # Load CONCH model
    model, preprocess = create_model_from_pretrained(
        config.conch_model_cfg, 
        config.conch_checkpoint_path, 
        device=device
    )
    model.eval()
    
    if logger:
        logger.info("CONCH model initialized successfully")
    print("CONCH model initialized successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Feature dimension: {config.feature_dim}")
    
    return model, preprocess

def extract_features_with_conch(model, preprocess, image, device):
    """Extract features using CONCH model
    
    Args:
        model: CONCH model
        preprocess: Image preprocessing function
        image: PIL Image
        device: PyTorch device
        
    Returns:
        torch.Tensor: Extracted features
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    return features

# ===================== Feature Extraction Functions =====================
def extract_features_from_patches(config, filtered_patches_df, output_dir, logger=None):
    """Extract features from filtered patches using CONCH model
    
    Args:
        config: Configuration object
        filtered_patches_df: DataFrame with patch information
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        str: Path to features directory
    """
    print("\n" + "="*60)
    print("Feature Extraction with CONCH Model")
    print("="*60)
    
    if logger:
        logger.info("Feature Extraction with CONCH Model")
        logger.info("="*60)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, preprocess = initialize_conch_model(config, device, logger)
    
    # Create features directory with model name
    features_dir = os.path.join(output_dir, "features_CONCH")
    os.makedirs(features_dir, exist_ok=True)
    
    # Group patches by WSI
    wsi_groups = filtered_patches_df.groupby('wsi_id')
    
    desc = "Extracting CONCH features"
    
    for wsi_id, wsi_patches in tqdm(wsi_groups, desc=desc):
        features_list = []
        
        for idx, row in wsi_patches.iterrows():
            patch_x = row["patch_x"]
            patch_y = row["patch_y"]
            patch_size = row["patch_size"]
            
            # Load WSI
            wsi_file = os.path.join(config.wsi_dir, f"{wsi_id}.svs")
            if not os.path.exists(wsi_file):
                continue
            
            try:
                # Open WSI and extract patch
                slide = openslide.OpenSlide(wsi_file)
                patch = slide.read_region((patch_x, patch_y), 0, (patch_size, patch_size))
                patch = patch.convert('RGB')
                slide.close()
                
                # Extract features
                features = extract_features_with_conch(model, preprocess, patch, device)
                
                # Store features
                features_list.append(features.cpu().numpy())
                
            except Exception as e:
                if logger:
                    logger.error(f"Error extracting features for patch {wsi_id}_{patch_x}_{patch_y}: {e}")
                continue
        
        if features_list:
            # Save features
            features_array = np.vstack(features_list)
            features_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
            torch.save(torch.from_numpy(features_array), features_file)
            
            # Note: patch_info is already available in the original filtered_patches_df
            # No need to save separate patch_info.csv files
            
            if logger:
                logger.info(f"Features saved for {wsi_id}: {len(features_list)} patches")
    
    print(f"Feature extraction completed. Features saved in: {features_dir}")
    if logger:
        logger.info(f"Feature extraction completed. Features saved in: {features_dir}")
    
    return features_dir

# ===================== t-SNE Visualization Functions =====================
def create_tsne_visualizations(config, output_dir, features_dir, filtered_patches_df, logger=None):
    """Create t-SNE visualizations
    
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
    all_hosps = []
    all_wsi_ids = []
    all_patch_info = []
    
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
            
            hosp = extract_hosp_from_filename(wsi_id)
            
            all_features.append(features)
            all_hosps.extend([hosp] * len(features))
            all_wsi_ids.extend([wsi_id] * len(features))
            
            for _, row in wsi_patches.iterrows():
                all_patch_info.append({
                    'wsi_id': wsi_id,
                    'hosp': hosp,
                    'patch_x': row['patch_x'],
                    'patch_y': row['patch_y'],
                    'patch_label': row['patch_label'],
                    'patch_probability': row['patch_probability']
                })
                
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
        indices = np.random.choice(len(all_features), config.max_tsne_points, replace=False)
        all_features = all_features[indices]
        all_hosps = [all_hosps[i] for i in indices]
        all_wsi_ids = [all_wsi_ids[i] for i in indices]
        all_patch_info = [all_patch_info[i] for i in indices]
        print(f"Sampled features: {len(all_features)}")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # t-SNE dimensionality reduction
    print("Starting t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=config.tsne_perplexity, max_iter=config.tsne_max_iter)
    features_2d = tsne.fit_transform(all_features)
    
    # Model name for filename
    model_name = "CONCH"
    
    # Visualize by hospital
    df_tsne = pd.DataFrame({
        'TSNE1': features_2d[:, 0],
        'TSNE2': features_2d[:, 1],
        'Hosp': all_hosps
    })
    
    unique_hosps = sorted(set(all_hosps))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_hosps)))
    color_map = dict(zip(unique_hosps, colors))
    
    plt.figure(figsize=(12, 10))
    for hosp in unique_hosps:
        mask = df_tsne['Hosp'] == hosp
        plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'], 
                   c=[color_map[hosp]], label=hosp, alpha=0.7, s=50)
    
    plt.title(f't-SNE Visualization ({model_name}) - Grouped by Hospital', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    hosp_viz_file = os.path.join(viz_dir, f"tsne_{model_name}_by_hosp.png")
    plt.savefig(hosp_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
    
    # Draw t-SNE for each label separately (different color for each hospital)
    for label in sorted(set([info['patch_label'] for info in all_patch_info])):
        # Only keep patches with the current label
        mask = [info['patch_label'] == label for info in all_patch_info]
        features_label = all_features[mask]
        patch_info_label = [info for i, info in enumerate(all_patch_info) if mask[i]]
        hosps_label = [info['hosp'] for info in patch_info_label]
        if len(features_label) == 0:
            continue
        
        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=config.tsne_perplexity, max_iter=config.tsne_max_iter)
        features_2d = tsne.fit_transform(features_label)
        
        # Plotting
        df = pd.DataFrame({
            'TSNE1': features_2d[:, 0],
            'TSNE2': features_2d[:, 1],
            'Hosp': hosps_label
        })
        unique_hosps = sorted(set(hosps_label))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_hosps)))
        color_map = dict(zip(unique_hosps, colors))
        
        plt.figure(figsize=(12, 10))
        for hosp in unique_hosps:
            mask_h = df['Hosp'] == hosp
            plt.scatter(df[mask_h]['TSNE1'], df[mask_h]['TSNE2'],
                        c=[color_map[hosp]], label=hosp, alpha=0.7, s=50)
        
        plt.title(f't-SNE ({model_name}, {label}) - Grouped by Hospital', fontsize=22, fontweight='bold')
        plt.xlabel('TSNE1', fontsize=18)
        plt.ylabel('TSNE2', fontsize=18)
        plt.legend(loc='upper right', fontsize=16)
        plt.tight_layout()
        
        label_hosp_viz_file = os.path.join(viz_dir, f"tsne_{model_name}_by_hosp_{label}.png")
        plt.savefig(label_hosp_viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Label {label} hospital-grouped t-SNE plot saved: {label_hosp_viz_file}")
        if logger:
            logger.info(f"Label {label} hospital-grouped t-SNE plot saved: {label_hosp_viz_file}")
    
    # Visualize by label (all labels together, color by label)
    df_tsne['Label'] = [info['patch_label'] for info in all_patch_info]
    unique_labels = sorted(set(df_tsne['Label']))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(12, 10))
    for label in unique_labels:
        mask = df_tsne['Label'] == label
        plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'],
                   c=[color_map[label]], label=label, alpha=0.7, s=50)
    
    plt.title(f't-SNE Visualization ({model_name}) - Grouped by Label', fontsize=22, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=18)
    plt.ylabel('TSNE2', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    
    label_viz_file = os.path.join(viz_dir, f"tsne_{model_name}_by_label.png")
    plt.savefig(label_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Label-grouped t-SNE plot saved: {label_viz_file}")
    if logger:
        logger.info(f"Label-grouped t-SNE plot saved: {label_viz_file}")
    
    # Statistics
    print(f"\n=== Visualization statistics ===")
    print(f"Total patches: {len(all_features)}")
    print(f"WSI count: {len(set(all_wsi_ids))}")
    print(f"Hospital count: {len(set(all_hosps))}")
    print(f"Hospital distribution: {dict(pd.Series(all_hosps).value_counts())}")
    print(f"Label distribution: {dict(pd.Series([info['patch_label'] for info in all_patch_info]).value_counts())}")

# ===================== Main Pipeline =====================
def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description='CONCH Feature Extraction and t-SNE Visualization')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing conch_filtered_patches.csv')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/conch/pytorch_model.bin',
                       help='Path to CONCH model checkpoint (default: ./checkpoints/conch/pytorch_model.bin)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.conch_checkpoint_path = args.checkpoint_path
    
    print("CONCH Feature Extraction and t-SNE Visualization Pipeline Starting")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"CONCH checkpoint: {args.checkpoint_path}")
    print("Model info: CONCH (ViT-B/16, supports feature extraction + zero-shot)")
    
    # Check CONCH availability
    if not CONCH_AVAILABLE:
        print("Error: CONCH is not available. Please install CONCH first.")
        print("Installation: pip install git+https://github.com/mahmoodlab/CONCH.git")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(Config.conch_checkpoint_path):
        print(f"Error: CONCH checkpoint not found: {Config.conch_checkpoint_path}")
        print("Please ensure the checkpoint file exists or specify the correct path with --checkpoint_path")
        return
    
    # Setup logging
    logger = setup_logging(args.input_dir)
    
    try:
        # Load filtered patches
        filtered_patches_file = os.path.join(args.input_dir, "conch_filtered_patches.csv")
        if not os.path.exists(filtered_patches_file):
            raise FileNotFoundError(f"Filtered patches file not found: {filtered_patches_file}")
        
        filtered_patches_df = pd.read_csv(filtered_patches_file)
        print(f"Loaded {len(filtered_patches_df)} filtered patches")
        if logger:
            logger.info(f"Loaded {len(filtered_patches_df)} filtered patches")
        
        # Extract features
        features_dir = extract_features_from_patches(Config, filtered_patches_df, args.input_dir, logger)
        
        # Create t-SNE visualizations
        create_tsne_visualizations(Config, args.input_dir, features_dir, filtered_patches_df, logger)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("Model used: CONCH (supports feature extraction + zero-shot)")
        print(f"Total patches processed: {len(filtered_patches_df)}")
        print(f"Results saved in: {args.input_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
            logger.info("Model used: CONCH (supports feature extraction + zero-shot)")
            logger.info(f"Total patches processed: {len(filtered_patches_df)}")
            logger.info(f"Results saved in: {args.input_dir}")
            logger.info("="*60)
        
    except Exception as e:
        error_msg = f"Pipeline error: {e}"
        print(f"\n{error_msg}")
        if logger:
            logger.error(error_msg)
        # Import traceback here for error handling
        import traceback
        traceback.print_exc()
        if logger:
            logger.error("Full traceback:", exc_info=True)

if __name__ == "__main__":
    main() 
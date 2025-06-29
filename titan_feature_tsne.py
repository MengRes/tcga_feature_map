#!/usr/bin/env python3
"""
TITAN Feature Extraction and t-SNE Visualization Pipeline

This script extracts features from filtered patches using TITAN model
and creates t-SNE visualizations grouped by hospital and label.

Model: TITAN (Transformer-based pathology Image and Text Alignment Network)
A multimodal whole slide foundation model for pathology.

Author: AI Assistant
Date: 2025-06-28
"""

# ===================== Standard Library Imports =====================
import os
import sys
import argparse
import glob
import logging

# 添加本地 TITAN 模块到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TITAN'))

# ===================== Third-party Library Imports =====================
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import openslide
from sklearn.manifold import TSNE

# ===================== TITAN Model Imports =====================
try:
    from transformers import AutoModel
    TITAN_AVAILABLE = True
except ImportError:
    TITAN_AVAILABLE = False
    print("Warning: TITAN dependencies not available. Please install transformers.")
    print("Installation: pip install transformers")

# ===================== Configuration =====================
class Config:
    """Configuration class for the pipeline"""
    
    # WSI parameters
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
    log_file = os.path.join(output_dir, "titan_feature_tsne.log")
    
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
    logger.info("TITAN Feature Extraction and t-SNE Visualization Started")
    logger.info("Note: TITAN is a multimodal whole slide foundation model")
    logger.info("=" * 60)
    
    return logger

# ===================== TITAN Model Functions =====================
def initialize_titan_model(config, device, logger=None):
    """Initialize TITAN model
    
    Args:
        config: Configuration object
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        tuple: (model, feature_extractor, eval_transform)
    """
    if not TITAN_AVAILABLE:
        raise ImportError("TITAN is not available. Please check local installation or online dependencies.")
    
    if logger:
        logger.info("Initializing TITAN model...")
    
    print("Initializing TITAN model...")
    print("Note: TITAN is a multimodal whole slide foundation model")
    
    try:
        # Load TITAN model using the standard approach
        print("Loading TITAN model from Hugging Face...")
        print("This may require Hugging Face authentication...")
        
        # Load TITAN model
        model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model = model.to(device)
        
        # Get feature extractor for patch feature extraction
        # Note: return_conch() is TITAN's API method name
        feature_extractor, eval_transform = model.return_conch()
        feature_extractor = feature_extractor.to(device)
        
        # Set to evaluation mode
        model.eval()
        feature_extractor.eval()
        
        if logger:
            logger.info("TITAN model initialized successfully")
        print("TITAN model initialized successfully")
        
        return model, feature_extractor, eval_transform
        
    except Exception as e:
        error_msg = f"Failed to initialize TITAN model: {e}"
        if logger:
            logger.error(error_msg)
        print(f"Error: {error_msg}")
        print("Make sure you have:")
        print("1. Valid Hugging Face authentication (huggingface-cli login)")
        print("2. Access to MahmoodLab/TITAN model")
        print("3. Required dependencies installed")
        raise

def extract_patch_features_with_titan(feature_extractor, eval_transform, patches, device):
    """Extract patch features using TITAN feature extractor
    
    Args:
        feature_extractor: TITAN feature extractor
        eval_transform: Image preprocessing function
        patches: List of PIL Images
        device: PyTorch device
        
    Returns:
        torch.Tensor: Extracted patch features
    """
    features_list = []
    
    for patch in patches:
        # Preprocess patch
        patch_tensor = eval_transform(patch).unsqueeze(0).to(device)
        
        # Extract features using TITAN feature extractor
        with torch.no_grad():
            # TITAN feature extractor uses direct call or forward method
            features = feature_extractor(patch_tensor)
            features_list.append(features)
    
    return torch.cat(features_list, dim=0)



# ===================== Feature Extraction Functions =====================
def extract_features_from_patches(config, filtered_patches_df, output_dir, logger=None):
    """Extract features from filtered patches using TITAN model (patch-level)
    
    Args:
        config: Configuration object
        filtered_patches_df: DataFrame with patch information
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        str: Path to features directory
    """
    print("\n" + "="*60)
    print("Feature Extraction with TITAN Model (Patch-level)")
    print("="*60)
    
    if logger:
        logger.info("Feature Extraction with TITAN Model (Patch-level)")
        logger.info("="*60)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, feature_extractor, eval_transform = initialize_titan_model(config, device, logger)
    
    # Create features directory with model name
    features_dir = os.path.join(output_dir, "features_TITAN")
    os.makedirs(features_dir, exist_ok=True)
    
    # Group patches by WSI
    wsi_groups = filtered_patches_df.groupby('wsi_id')
    
    desc = "Extracting TITAN features (patch-level)"
    
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
                
                # Extract patch-level features using TITAN feature extractor
                features = extract_patch_features_with_titan(feature_extractor, eval_transform, [patch], device)
                
                # Store features
                features_list.append(features.cpu().numpy())
                
            except Exception as e:
                if logger:
                    logger.error(f"Error extracting features for patch {wsi_id}_{patch_x}_{patch_y}: {e}")
                continue
        
        if features_list:
            # Save features (patch-level)
            features_array = np.vstack(features_list)
            features_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
            torch.save(torch.from_numpy(features_array), features_file)
            
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
            # Load patch-level features
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
            
            # Use original patch information
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
    
    # Merge all features (patch-level)
    all_features = np.vstack(all_features)
    print(f"Total patch features: {len(all_features)}")
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
    model_name = "TITAN"
    
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
                   c=[color_map[hosp]], label=hosp, alpha=0.7, s=100)
    
    plt.title(f't-SNE Visualization ({model_name}) - Grouped by Hospital', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    hosp_viz_file = os.path.join(viz_dir, f"tsne_{model_name}_by_hosp.png")
    plt.savefig(hosp_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hospital-grouped t-SNE plot saved: {hosp_viz_file}")
    
    # Visualize by patch label
    df_tsne['Label'] = [info['patch_label'] for info in all_patch_info]
    unique_labels = sorted(set(df_tsne['Label']))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(12, 10))
    for label in unique_labels:
        mask = df_tsne['Label'] == label
        plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'],
                   c=[color_map[label]], label=label, alpha=0.7, s=100)
    
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
    
    # Draw t-SNE for each label separately (different color for each hospital)
    for label in unique_labels:
        # Only keep patches with the current label
        mask = [info['patch_label'] == label for info in all_patch_info]
        features_label = all_features[mask]
        info_label = [info for i, info in enumerate(all_patch_info) if mask[i]]
        hosps_label = [info['hosp'] for info in info_label]
        if len(features_label) == 0:
            continue
        
        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(config.tsne_perplexity, len(features_label)-1), max_iter=config.tsne_max_iter)
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
                        c=[color_map[hosp]], label=hosp, alpha=0.7, s=100)
        
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
    parser = argparse.ArgumentParser(description='TITAN Feature Extraction and t-SNE Visualization')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing filtered patches CSV file')
    
    args = parser.parse_args()
    
    print("TITAN Feature Extraction and t-SNE Visualization Pipeline Starting")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print("Model info: TITAN (Multimodal whole slide foundation model)")
    
    # Check TITAN availability
    if not TITAN_AVAILABLE:
        print("Error: TITAN dependencies are not available.")
        print("Please install required packages:")
        print("pip install transformers huggingface_hub")
        return
    
    # Setup logging
    logger = setup_logging(args.input_dir)
    
    try:
        # Load filtered patches - try different possible filenames
        possible_files = [
            "conch_filtered_patches.csv",
            "filtered_patches.csv",
            "extracted_patches.csv"
        ]
        
        filtered_patches_file = None
        for filename in possible_files:
            filepath = os.path.join(args.input_dir, filename)
            if os.path.exists(filepath):
                filtered_patches_file = filepath
                break
        
        if filtered_patches_file is None:
            raise FileNotFoundError(f"No filtered patches file found in {args.input_dir}. Expected one of: {possible_files}")
        
        filtered_patches_df = pd.read_csv(filtered_patches_file)
        print(f"Loaded {len(filtered_patches_df)} filtered patches from {os.path.basename(filtered_patches_file)}")
        if logger:
            logger.info(f"Loaded {len(filtered_patches_df)} filtered patches from {os.path.basename(filtered_patches_file)}")
        
        # Extract features
        features_dir = extract_features_from_patches(Config, filtered_patches_df, args.input_dir, logger)
        
        # Create t-SNE visualizations
        create_tsne_visualizations(Config, args.input_dir, features_dir, filtered_patches_df, logger)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("Model used: TITAN (Multimodal whole slide foundation model)")
        print(f"Total patches processed: {len(filtered_patches_df)}")
        print(f"Results saved in: {args.input_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
            logger.info("Model used: TITAN (Multimodal whole slide foundation model)")
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
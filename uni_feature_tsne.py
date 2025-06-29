#!/usr/bin/env python3
"""
UNI Feature Extraction and t-SNE Visualization Pipeline

This script extracts features from filtered patches using UNI models
and creates t-SNE visualizations grouped by hospital and label.

Models supported: UNI (original), UNI2-H (enhanced version)
Both models are used for feature extraction only.
"""

# ===================== Standard Library Imports =====================
import os
import sys
import argparse
import glob
import logging
from datetime import datetime

# ===================== Third-party Library Imports =====================
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import openslide
from sklearn.manifold import TSNE

# ===================== UNI Model Imports =====================
# Add UNI path
sys.path.append('./UNI')

# UNI model imports
try:
    from uni import get_encoder
    UNI_AVAILABLE = True
except ImportError:
    UNI_AVAILABLE = False
    print("Warning: UNI package not available. Will try direct timm loading.")

# UNI timm imports (for direct loading)
try:
    import timm
    try:
        from timm.data.config import resolve_data_config
    except ImportError:
        from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Error: timm not available. UNI loading will fail.")
    sys.exit(1)

# ===================== Configuration =====================
class Config:
    """Configuration class for the pipeline"""
    
    # UNI model parameters
    uni_model_name = "uni2-h"  # Options: "uni", "uni2-h"
    uni_checkpoint_path = "./UNI/assets/ckpts/"
    
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
    log_file = os.path.join(output_dir, "uni_feature_tsne.log")
    
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
    logger.info("UNI Feature Extraction and t-SNE Visualization Started")
    logger.info("=" * 60)
    
    return logger

# ===================== UNI Model Functions =====================
def initialize_uni_model(config, device, logger=None):
    """Initialize UNI model (uni or uni2-h)
    
    Args:
        config: Configuration object
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        tuple: (model, transform)
    """
    if logger:
        logger.info(f"Initializing UNI model: {config.uni_model_name}...")
    
    print(f"Initializing UNI model: {config.uni_model_name}...")
    
    try:
        # Method 1: Try using timm with Hugging Face Hub (recommended)
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available")
            
        print(f"Attempting to load {config.uni_model_name} via timm (Hugging Face Hub)...")
        
        if config.uni_model_name.lower() == "uni":
            # Original UNI model
            print("Loading original UNI model...")
            model = timm.create_model(
                "hf-hub:MahmoodLab/uni", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=True
            )
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            
        elif config.uni_model_name.lower() == "uni2-h":
            # UNI2-H model configuration
            print("Loading UNI2-H model...")
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            
        else:
            raise ValueError(f"Unsupported UNI model: {config.uni_model_name}")
        
        model.eval()
        model.to(device)
        
        if logger:
            logger.info(f"UNI model {config.uni_model_name} loaded via timm successfully")
        print(f"UNI model {config.uni_model_name} loaded via timm successfully")
        
        return model, transform
        
    except Exception as e:
        print(f"Failed to load via timm: {e}")
        if logger:
            logger.warning(f"Failed to load via timm: {e}")
        
        # Method 2: Fallback to local get_encoder method (if available)
        if UNI_AVAILABLE:
            print("Falling back to local get_encoder method...")
            try:
                model, transform = get_encoder(
                    enc_name=config.uni_model_name,
                    device=device,
                    assets_dir=config.uni_checkpoint_path
                )
                
                if model is None:
                    raise ValueError(f"Failed to initialize UNI model: {config.uni_model_name}")
                
                if logger:
                    logger.info(f"UNI model {config.uni_model_name} loaded via get_encoder successfully")
                print(f"UNI model {config.uni_model_name} loaded via get_encoder successfully")
                
                return model, transform
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load UNI model with both methods. timm error: {e}, get_encoder error: {e2}")
        else:
            raise RuntimeError(f"Failed to load UNI model via timm: {e}. Local UNI package not available as fallback.")

def extract_features_with_uni(model, transform, image, device):
    """Extract features using UNI model
    
    Args:
        model: UNI model
        transform: Image transform function
        image: PIL Image
        device: PyTorch device
        
    Returns:
        torch.Tensor: Extracted features
    """
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_tensor)
    return features

# ===================== Feature Extraction Functions =====================
def extract_features_from_patches(config, filtered_patches_df, output_dir, logger=None):
    """Extract features from filtered patches using UNI model
    
    Args:
        config: Configuration object
        filtered_patches_df: DataFrame with patch information
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        str: Path to features directory
    """
    print("\n" + "="*60)
    print(f"Feature Extraction with UNI Model: {config.uni_model_name}")
    print("="*60)
    
    if logger:
        logger.info(f"Feature Extraction with UNI Model: {config.uni_model_name}")
        logger.info("="*60)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, transform = initialize_uni_model(config, device, logger)
    
    # Create features directory with model name
    model_name = config.uni_model_name.upper()  # UNI or UNI2-H
    features_dir = os.path.join(output_dir, f"features_{model_name}")
    os.makedirs(features_dir, exist_ok=True)
    
    # Group patches by WSI
    wsi_groups = filtered_patches_df.groupby('wsi_id')
    
    desc = f"Extracting {config.uni_model_name.upper()} features"
    
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
                features = extract_features_with_uni(model, transform, patch, device)
                
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
    model_name = config.uni_model_name.upper()
    
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
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
    parser = argparse.ArgumentParser(description='UNI Feature Extraction and t-SNE Visualization')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing conch_filtered_patches.csv')
    parser.add_argument('--uni_model', type=str, default='uni2-h', choices=['uni', 'uni2-h'],
                       help='UNI model to use: uni (original) or uni2-h (enhanced). Default: uni2-h')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.uni_model_name = args.uni_model
    
    print("UNI Feature Extraction and t-SNE Visualization Pipeline Starting")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"UNI model: {args.uni_model}")
    
    # Model information
    if args.uni_model == "uni":
        print("Model info: UNI (ViT-L/16, 303M params, 1024-dim features)")
    elif args.uni_model == "uni2-h":
        print("Model info: UNI2-H (ViT-H/14, 681M params, 1536-dim features)")
    
    # Check model availability
    if not UNI_AVAILABLE and not TIMM_AVAILABLE:
        print("Error: Neither UNI package nor timm available for UNI model loading.")
        print("Please install timm: pip install timm")
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
        print(f"UNI model used: {Config.uni_model_name}")
        print(f"Total patches processed: {len(filtered_patches_df)}")
        print(f"Results saved in: {args.input_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"UNI model used: {Config.uni_model_name}")
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
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import matplotlib.pyplot as plt
import h5py
import openslide
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from CONCH.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from sklearn.manifold import TSNE
import seaborn as sns
import glob
import shutil
from datetime import datetime
import json
import logging
import sys
from torchvision import models
from torchvision import transforms

# Add UNI path
sys.path.append('./UNI')
try:
    from uni import get_encoder
    UNI_AVAILABLE = True
except ImportError:
    UNI_AVAILABLE = False
    print("Warning: UNI not available. Only CONCH model will be used.")

# Parameter Settings
class Config:
    # Dataset parameters
    dataset_name = "tcga-brca"
    
    # WSI filtering parameters - considering hosp, label and other conditions
    accept_label = ["IDC", "ILC"]            # Labels to be balanced
    accept_hosp_list = ["AR", "A2", "D8", "BH"]   # Only accept these hosp sources
    n_per_hosp = 10                          # Number of WSIs to select per hosp
    num_sampled_patches = 100                # Maximum number of patches per WSI
    patch_size = 256
    
    # Additional filtering conditions
    accept_age_groups = ["60-69", "70-79"]  # Age groups to accept (None for all)
    accept_gender = ["female"]               # Gender to accept (None for all)
    accept_race = ["white"]                  # Race to accept (None for all)
    
    # Path configuration
    coord_dir = "/raid/mengliang/wsi_process/tcga-brca_patch/patches/"
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
    label_file = "files/tcga-brca_label.csv"
    
    # Model configuration
    # Zero-shot classification uses CONCH uniformly, feature extraction can choose different models
    zero_shot_model = "CONCH"  # Zero-shot classification model (fixed as CONCH)
    feature_model = "CONCH"    # Feature extraction model (options: CONCH, UNI, UNI2-H, RESNET50)
    
    # CONCH model parameters
    checkpoint_path = './checkpoints/conch/pytorch_model.bin'
    model_cfg = 'conch_ViT-B-16'
    probability_threshold = 0.8  # Probability threshold
    skip_zero_shot = False  # Default to perform zero-shot classification
    
    # Zero-shot classification parameters
    classes = ['invasive ductal carcinoma', 'invasive lobular carcinoma']
    prompts = ['an H&E image of invasive ductal carcinoma', 'an H&E image of invasive lobular carcinoma']
    
    # UNI model parameters (for feature extraction only)
    uni_model_name = "uni2-h"  # Options: "uni", "uni2-h"
    uni_checkpoint_path = "./UNI/assets/ckpts/"
    
    # TSNE parameters
    tsne_perplexity = 30
    tsne_n_iter = 1000
    max_tsne_points = 10000  # Maximum sampling points for TSNE

# Utility Functions
def age_group(age):
    try:
        age = int(age)
        return f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
    except:
        return "unknown"

def extract_hosp_from_filename(filename):
    """Extract hospital information from filename"""
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[1]
    return "Unknown"

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, "pipeline.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TCGA-BRCA Pipeline Log Started")
    logger.info("=" * 60)
    
    return logger

def save_parameters(config, output_dir, logger=None):
    """Save parameters to txt file and log them"""
    param_file = os.path.join(output_dir, "parameters.txt")
    
    # Prepare parameter content
    param_content = []
    param_content.append("=" * 50)
    param_content.append("TCGA-BRCA Pipeline Parameters")
    param_content.append("=" * 50)
    param_content.append(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    param_content.append("")
    
    param_content.append("Model configuration:")
    param_content.append(f"  Zero-shot model: {config.zero_shot_model} (fixed)")
    param_content.append(f"  Feature extraction model: {config.feature_model}")
    param_content.append("")
    
    param_content.append("CONCH model parameters:")
    param_content.append(f"  CONCH model config: {config.model_cfg}")
    param_content.append(f"  CONCH checkpoint: {config.checkpoint_path}")
    param_content.append(f"  Probability threshold: {config.probability_threshold}")
    param_content.append(f"  Classes: {config.classes}")
    param_content.append(f"  Prompts: {config.prompts}")
    param_content.append("")
    
    if config.feature_model.upper() in ["UNI", "UNI2-H"]:
        param_content.append("UNI model parameters (for feature extraction):")
        param_content.append(f"  UNI model name: {config.uni_model_name}")
        param_content.append(f"  UNI checkpoint path: {config.uni_checkpoint_path}")
        param_content.append("")
    
    param_content.append("WSI filtering parameters (considering hosp, label and other conditions):")
    param_content.append(f"  Labels: {config.accept_label}")
    param_content.append(f"  Hosp sources: {config.accept_hosp_list}")
    param_content.append(f"  WSIs per hosp: {config.n_per_hosp}")
    param_content.append(f"  Patches per WSI: {config.num_sampled_patches}")
    param_content.append(f"  Patch size: {config.patch_size}")
    param_content.append(f"  Age groups: {config.accept_age_groups}")
    param_content.append(f"  Gender: {config.accept_gender}")
    param_content.append(f"  Race: {config.accept_race}")
    param_content.append("")
    
    param_content.append("Path configuration:")
    param_content.append(f"  Coordinate directory: {config.coord_dir}")
    param_content.append(f"  WSI directory: {config.wsi_dir}")
    param_content.append(f"  Label file: {config.label_file}")
    
    # Save to file
    with open(param_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(param_content))
    
    # Log parameters
    if logger:
        logger.info("Pipeline Parameters:")
        for line in param_content:
            logger.info(line)
    
    print(f"Parameters saved to: {param_file}")
    if logger:
        logger.info(f"Parameters saved to: {param_file}")

# Step 1: WSI Selection and Patch Extraction
def step1_wsi_selection(config, output_dir, logger=None):
    """Step 1: WSI selection and patch extraction - considering hosp, label and other conditions"""
    step_title = "Step 1: WSI Selection and Patch Extraction"
    print("\n" + "="*60)
    print(step_title)
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if logger:
        logger.info(step_title)
        logger.info("="*60)
    
    # Load label data
    if logger:
        logger.info("Loading label data...")
    df_label = pd.read_csv(config.label_file)
    df_label["source"] = df_label["filename"].str.extract(r"TCGA-([A-Z0-9]{2})-")
    df_label["age_group"] = df_label["age"].apply(age_group)
    
    if logger:
        logger.info(f"Loaded {len(df_label)} label records")
    print(f"Loaded {len(df_label)} label records")
    
    # Filter by label
    df_filtered = df_label[df_label["label"].isin(config.accept_label)].copy()
    if logger:
        logger.info(f"After label filtering: {len(df_filtered)} records")
    print(f"After label filtering: {len(df_filtered)} records")
    
    # Filter by hospital
    df_filtered = df_filtered[df_filtered["source"].isin(config.accept_hosp_list)].copy()
    if logger:
        logger.info(f"After hospital filtering: {len(df_filtered)} records")
    print(f"After hospital filtering: {len(df_filtered)} records")
    
    # Filter by age group
    if config.accept_age_groups:
        df_filtered = df_filtered[df_filtered["age_group"].isin(config.accept_age_groups)].copy()
        if logger:
            logger.info(f"After age group filtering: {len(df_filtered)} records")
        print(f"After age group filtering: {len(df_filtered)} records")
    
    # Filter by gender
    if config.accept_gender:
        df_filtered = df_filtered[df_filtered["gender"].isin(config.accept_gender)].copy()
        if logger:
            logger.info(f"After gender filtering: {len(df_filtered)} records")
        print(f"After gender filtering: {len(df_filtered)} records")
    
    # Filter by race
    if config.accept_race:
        df_filtered = df_filtered[df_filtered["race"].isin(config.accept_race)].copy()
        if logger:
            logger.info(f"After race filtering: {len(df_filtered)} records")
        print(f"After race filtering: {len(df_filtered)} records")
    
    # Balance by hospital
    selected_wsis = []
    for hosp in config.accept_hosp_list:
        hosp_wsis = df_filtered[df_filtered["source"] == hosp]
        if len(hosp_wsis) > 0:
            # Select up to n_per_hosp WSIs per hospital
            n_select = min(config.n_per_hosp, len(hosp_wsis))
            selected = hosp_wsis.sample(n=n_select, random_state=42)
            selected_wsis.append(selected)
            if logger:
                logger.info(f"Selected {len(selected)} WSIs from hospital {hosp}")
            print(f"Selected {len(selected)} WSIs from hospital {hosp}")
    
    if not selected_wsis:
        raise ValueError("No WSIs selected after filtering")
    
    df_selected = pd.concat(selected_wsis, ignore_index=True)
    if logger:
        logger.info(f"Total selected WSIs: {len(df_selected)}")
    print(f"Total selected WSIs: {len(df_selected)}")
    
    # Extract patches
    all_patches = []
    
    for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Extracting patches"):
        wsi_id = row["filename"]
        label = row["label"]
        hosp = row["source"]
        
        # Load patch coordinates
        coord_file = os.path.join(config.coord_dir, f"{wsi_id}.h5")
        if not os.path.exists(coord_file):
            if logger:
                logger.warning(f"Coordinate file not found: {coord_file}")
            continue
        
        try:
            with h5py.File(coord_file, 'r') as f:
                coords = f['coords'][:]
            
            # Randomly sample patches
            if len(coords) > config.num_sampled_patches:
                indices = np.random.choice(len(coords), config.num_sampled_patches, replace=False)
                coords = coords[indices]
            
            # Create patch records
            for coord in coords:
                patch_record = {
                    'wsi_id': wsi_id,
                    'label': label,
                    'hosp': hosp,
                    'patch_x': int(coord[0]),
                    'patch_y': int(coord[1]),
                    'patch_size': config.patch_size
                }
                all_patches.append(patch_record)
                
        except Exception as e:
            if logger:
                logger.error(f"Error processing {wsi_id}: {e}")
            continue
    
    # Create DataFrame
    patch_df = pd.DataFrame(all_patches)
    
    if logger:
        logger.info(f"Total patches extracted: {len(patch_df)}")
        logger.info(f"Patches per label: {dict(patch_df['label'].value_counts())}")
        logger.info(f"Patches per hospital: {dict(patch_df['hosp'].value_counts())}")
    
    print(f"Total patches extracted: {len(patch_df)}")
    print(f"Patches per label: {dict(patch_df['label'].value_counts())}")
    print(f"Patches per hospital: {dict(patch_df['hosp'].value_counts())}")
    
    # Save patch information
    patch_file = os.path.join(output_dir, "extracted_patches.csv")
    patch_df.to_csv(patch_file, index=False)
    print(f"Patch information saved: {patch_file}")
    
    if logger:
        logger.info(f"Patch information saved: {patch_file}")
    
    return patch_df

# Step 2: Zero-shot Classification
def step2_conch_zero_shot(config, output_dir, patch_df, logger=None):
    """Step 2: Zero-shot classification using CONCH model"""
    step_title = "Step 2: Zero-shot Classification"
    print("\n" + "="*60)
    print(step_title)
    print("="*60)
    
    if logger:
        logger.info(step_title)
        logger.info("="*60)
    
    if config.skip_zero_shot:
        if logger:
            logger.info("Skipping zero-shot classification as configured")
        print("Skipping zero-shot classification as configured")
        return patch_df
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = initialize_model(config, device, logger, model_type="zero_shot")
    
    # Process patches
    results = []
    
    for idx, row in tqdm(patch_df.iterrows(), total=len(patch_df), desc="Zero-shot classification"):
        wsi_id = row["wsi_id"]
        patch_x = row["patch_x"]
        patch_y = row["patch_y"]
        patch_size = row["patch_size"]
        
        # Load WSI
        wsi_file = os.path.join(config.wsi_dir, f"{wsi_id}.svs")
        if not os.path.exists(wsi_file):
            if logger:
                logger.warning(f"WSI file not found: {wsi_file}")
            continue
        
        try:
            # Open WSI and extract patch
            slide = openslide.OpenSlide(wsi_file)
            patch = slide.read_region((patch_x, patch_y), 0, (patch_size, patch_size))
            patch = patch.convert('RGB')
            slide.close()
            
            # Perform zero-shot classification
            pred_label, pred_prob = zero_shot_classification_with_model(
                model, preprocess, patch, device, config.zero_shot_model, config
            )
            
            # Filter by probability threshold
            if pred_prob >= config.probability_threshold:
                result = {
                    'wsi_id': wsi_id,
                    'label': row["label"],
                    'hosp': row["hosp"],
                    'patch_x': patch_x,
                    'patch_y': patch_y,
                    'patch_size': patch_size,
                    'patch_label': pred_label,
                    'patch_probability': pred_prob
                }
                results.append(result)
                
        except Exception as e:
            if logger:
                logger.error(f"Error processing patch {wsi_id}_{patch_x}_{patch_y}: {e}")
            continue
    
    # Create filtered DataFrame
    filtered_patches_df = pd.DataFrame(results)
    
    if logger:
        logger.info(f"Patches after zero-shot filtering: {len(filtered_patches_df)}")
        logger.info(f"Filtered patches per label: {dict(filtered_patches_df['patch_label'].value_counts())}")
        logger.info(f"Filtered patches per hospital: {dict(filtered_patches_df['hosp'].value_counts())}")
    
    print(f"Patches after zero-shot filtering: {len(filtered_patches_df)}")
    print(f"Filtered patches per label: {dict(filtered_patches_df['patch_label'].value_counts())}")
    print(f"Filtered patches per hospital: {dict(filtered_patches_df['hosp'].value_counts())}")
    
    # Save filtered patch information
    filtered_patch_file = os.path.join(output_dir, "filtered_patches.csv")
    filtered_patches_df.to_csv(filtered_patch_file, index=False)
    print(f"Filtered patch information saved: {filtered_patch_file}")
    
    if logger:
        logger.info(f"Filtered patch information saved: {filtered_patch_file}")
    
    return filtered_patches_df

# Step 3: Feature Extraction
def step3_conch_feature_extraction(config, output_dir, filtered_patches_df, logger=None):
    """Step 3: Feature extraction using selected model"""
    step_title = "Step 3: Feature Extraction"
    print("\n" + "="*60)
    print(step_title)
    print("="*60)
    
    if logger:
        logger.info(step_title)
        logger.info("="*60)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = initialize_model(config, device, logger, model_type="feature")
    
    # Create features directory
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Group patches by WSI
    wsi_groups = filtered_patches_df.groupby('wsi_id')
    
    for wsi_id, wsi_patches in tqdm(wsi_groups, desc="Extracting features"):
        features_list = []
        patch_info_list = []
        
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
                features = extract_features_with_model(
                    model, preprocess, patch, device, config.feature_model
                )
                
                # Store features and patch info
                features_list.append(features.cpu().numpy())
                patch_info_list.append({
                    'patch_x': patch_x,
                    'patch_y': patch_y,
                    'patch_label': row["patch_label"],
                    'patch_probability': row["patch_probability"]
                })
                
            except Exception as e:
                if logger:
                    logger.error(f"Error extracting features for patch {wsi_id}_{patch_x}_{patch_y}: {e}")
                continue
        
        if features_list:
            # Save features
            features_array = np.vstack(features_list)
            features_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
            torch.save(torch.from_numpy(features_array), features_file)
            
            # Save patch info
            patch_info_df = pd.DataFrame(patch_info_list)
            patch_info_file = os.path.join(features_dir, f"{wsi_id}_patch_info.csv")
            patch_info_df.to_csv(patch_info_file, index=False)
            
            if logger:
                logger.info(f"Features saved for {wsi_id}: {len(features_list)} patches")
    
    print(f"Feature extraction completed. Features saved in: {features_dir}")
    if logger:
        logger.info(f"Feature extraction completed. Features saved in: {features_dir}")
    
    return features_dir

# Step 4: TSNE Visualization
def step4_tsne_visualization(config, output_dir, features_dir, logger=None):
    """Step 4: TSNE Visualization"""
    print("\n" + "="*60)
    print("Step 4: TSNE Visualization")
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
            features = torch.load(pt_file)
            
            # Get WSI ID from filename
            wsi_id = os.path.basename(pt_file).replace("_features.pt", "")
            
            # Load corresponding patch info
            patch_info_file = os.path.join(features_dir, f"{wsi_id}_patch_info.csv")
            if os.path.exists(patch_info_file):
                patch_info = pd.read_csv(patch_info_file)
            else:
                print(f"Warning: No patch info file for {wsi_id}")
                continue
            
            hosp = extract_hosp_from_filename(wsi_id)
            
            all_features.append(features)
            all_hosps.extend([hosp] * len(features))
            all_wsi_ids.extend([wsi_id] * len(features))
            
            for _, row in patch_info.iterrows():
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
    
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # TSNE dimensionality reduction
    print("Starting TSNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=config.tsne_perplexity, n_iter=config.tsne_n_iter)
    features_2d = tsne.fit_transform(all_features)
    
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
    
    plt.title('TSNE Visualization - Grouped by Hospital', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    hosp_viz_file = os.path.join(viz_dir, "tsne_by_hosp.png")
    plt.savefig(hosp_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hospital-grouped TSNE plot saved: {hosp_viz_file}")
    
    # ========== Draw t-SNE for each label separately (different color for each hospital, larger font) ==========
    for label in sorted(set([info['patch_label'] for info in all_patch_info])):
        # Only keep patches with the current label
        mask = [info['patch_label'] == label for info in all_patch_info]
        features_label = all_features[mask]
        patch_info_label = [info for i, info in enumerate(all_patch_info) if mask[i]]
        hosps_label = [info['hosp'] for info in patch_info_label]
        if len(features_label) == 0:
            continue
        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=config.tsne_perplexity, n_iter=config.tsne_n_iter)
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
        plt.title(f't-SNE ({label}) - Grouped by Hospital', fontsize=22, fontweight='bold')
        plt.xlabel('TSNE1', fontsize=18)
        plt.ylabel('TSNE2', fontsize=18)
        plt.legend(loc='upper right', fontsize=16)
        plt.tight_layout()
        label_hosp_viz_file = os.path.join(viz_dir, f"tsne_by_hosp_{label}.png")
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
    plt.title('t-SNE Visualization - Grouped by Label', fontsize=22, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=18)
    plt.ylabel('TSNE2', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    label_viz_file = os.path.join(viz_dir, "tsne_by_label.png")
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

# Model Initialization Functions
def initialize_model(config, device, logger=None, model_type="feature"):
    """Initialize model based on model_type"""
    if model_type == "zero_shot":
        # Zero-shot classification uses CONCH uniformly
        return initialize_conch_model(config, device, logger)
    elif model_type == "feature":
        # Feature extraction can choose different models
        if config.feature_model.upper() == "CONCH":
            return initialize_conch_model(config, device, logger)
        elif config.feature_model.upper() in ["UNI", "UNI2-H"]:
            return initialize_uni_model(config, device, logger)
        elif config.feature_model.upper() == "RESNET50":
            return initialize_resnet50_model(config, device, logger)
        else:
            raise ValueError(f"Unsupported feature model: {config.feature_model}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_conch_model(config, device, logger=None):
    """Initialize CONCH model"""
    if logger:
        logger.info("Initializing CONCH model...")
    
    print("Initializing CONCH model...")
    model, preprocess = create_model_from_pretrained(config.model_cfg, config.checkpoint_path, device=device)
    model.eval()
    
    if logger:
        logger.info("CONCH model initialized successfully")
    print("CONCH model initialized successfully")
    
    return model, preprocess

def initialize_uni_model(config, device, logger=None):
    """Initialize UNI model"""
    if not UNI_AVAILABLE:
        raise ImportError("UNI is not available. Please install UNI first.")
    
    if logger:
        logger.info(f"Initializing UNI model: {config.uni_model_name}...")
    
    print(f"Initializing UNI model: {config.uni_model_name}...")
    
    # Determine UNI model name
    uni_model_name = config.uni_model_name
    if config.feature_model.upper() == "UNI2-H":
        uni_model_name = "uni2-h"
    
    # Initialize UNI model
    model, transform = get_encoder(
        enc_name=uni_model_name,
        device=device,
        assets_dir=config.uni_checkpoint_path
    )
    
    if model is None:
        raise ValueError(f"Failed to initialize UNI model: {uni_model_name}")
    
    if logger:
        logger.info(f"UNI model {uni_model_name} initialized successfully")
    print(f"UNI model {uni_model_name} initialized successfully")
    
    return model, transform

def initialize_resnet50_model(config, device, logger=None):
    model = models.resnet50(pretrained=True)
    # Remove the final fc layer, keep only 2048-dimensional features
    modules = list(model.children())[:-1]  # Remove fc
    model = torch.nn.Sequential(*modules)
    model.eval()
    model.to(device)
    # ImageNet standard preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess

def extract_features_with_model(model, preprocess, image, device, model_type):
    """Extract features using the specified model"""
    if model_type.upper() == "CONCH":
        # CONCH model feature extraction
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image_tensor)
        return features
    
    elif model_type.upper() in ["UNI", "UNI2-H"]:
        # UNI model feature extraction
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor)
        return features
    
    elif model_type.upper() == "RESNET50":
        # ResNet50 feature extraction
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor)  # [1, 2048, 1, 1]
            features = features.view(features.size(0), -1)  # [1, 2048]
        return features
    
    else:
        raise ValueError(f"Unsupported model type for feature extraction: {model_type}")

def zero_shot_classification_with_model(model, preprocess, image, device, model_type, config):
    """Perform zero-shot classification using the specified model"""
    if model_type.upper() == "CONCH":
        # CONCH zero-shot classification
        tokenizer = get_tokenizer()
        tokenized_prompts = tokenize(texts=config.prompts, tokenizer=tokenizer).to(device)
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_prompts)
            sim_scores = (image_features @ text_features.T * model.logit_scale.exp()).softmax(dim=-1)
        
        pred_idx = torch.argmax(sim_scores, dim=-1).item()
        pred_label = config.classes[pred_idx]
        pred_prob = sim_scores[0][pred_idx].item()
        
        label_map = {
            'invasive ductal carcinoma': 'IDC',
            'invasive lobular carcinoma': 'ILC'
        }
        
        return label_map.get(pred_label, pred_label), pred_prob
    
    else:
        raise ValueError(f"Unsupported model type for zero-shot classification: {model_type}")

# Main Function
def main():
    """Main function"""
    print("TCGA-BRCA Pipeline starting")
    print("="*60)
    
    # Display model information
    print(f"Zero-shot model: {Config.zero_shot_model} (fixed)")
    print(f"Feature extraction model: {Config.feature_model}")
    
    # Check if UNI model is available (if using UNI for feature extraction)
    if Config.feature_model.upper() in ["UNI", "UNI2-H"]:
        if not UNI_AVAILABLE:
            print("Error: UNI is not available. Please install UNI first.")
            return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{Config.dataset_name}_{Config.feature_model.lower()}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save parameters
    logger = setup_logging(output_dir)
    save_parameters(Config, output_dir, logger)
    
    try:
        # Step 1: WSI Selection and Patch Extraction
        patch_df = step1_wsi_selection(Config, output_dir, logger)
        
        # Step 2: Zero-shot Classification (using CONCH)
        filtered_patches_df = step2_conch_zero_shot(Config, output_dir, patch_df, logger)
        
        # Step 3: Feature Extraction (using selected model)
        features_dir = step3_conch_feature_extraction(Config, output_dir, filtered_patches_df, logger)
        
        # Step 4: TSNE Visualization
        step4_tsne_visualization(Config, output_dir, features_dir, logger)
        
        print("\n" + "="*60)
        print("Pipeline completed")
        print(f"Zero-shot model: {Config.zero_shot_model}")
        print(f"Feature extraction model: {Config.feature_model}")
        print(f"All results saved: {output_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully")
            logger.info(f"Zero-shot model: {Config.zero_shot_model}")
            logger.info(f"Feature extraction model: {Config.feature_model}")
            logger.info(f"All results saved: {output_dir}")
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
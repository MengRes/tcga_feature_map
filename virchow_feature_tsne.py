#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VIRCHOW'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import openslide
from sklearn.manifold import TSNE

# VIRCHOW Model Imports
import timm
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Configuration
class Config:
    """Configuration class for the pipeline"""
    
    # WSI parameters
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
    
    # t-SNE parameters
    tsne_perplexity = 30
    tsne_max_iter = 1000
    max_tsne_points = 10000  # Maximum sampling points for t-SNE

# Utility Functions
def extract_hosp_from_filename(filename):
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[1]
    return "Unknown"

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "virchow_feature_tsne.log")
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
    logger.info("VIRCHOW Feature Extraction and t-SNE Visualization Started")
    logger.info("Note: VIRCHOW is a multimodal whole slide foundation model")
    logger.info("=" * 60)
    return logger

# VIRCHOW Model Functions
def initialize_virchow_model(device, logger=None):
    if logger:
        logger.info("Initializing VIRCHOW model (timm)...")
    print("Initializing VIRCHOW model (timm)...")
    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.to(device)
    model.eval()
    config = resolve_data_config(model.pretrained_cfg, model=model)
    eval_transform = create_transform(**config)
    if logger:
        logger.info("VIRCHOW model initialized successfully")
    print("VIRCHOW model initialized successfully")
    return model, eval_transform

def extract_patch_features_with_virchow(model, eval_transform, patches, device):
    features_list = []
    for patch in patches:
        image = eval_transform(patch).unsqueeze(0).to(device)  # 1 x 3 x 224 x 224
        with torch.no_grad():
            output = model(image)  # 1 x 257 x 1280
            class_token = output[:, 0]    # 1 x 1280
            patch_tokens = output[:, 1:]  # 1 x 256 x 1280
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # 1 x 2560
            features_list.append(embedding.cpu())
    return torch.cat(features_list, dim=0)

# Feature Extraction Functions
def extract_features_from_patches(config, filtered_patches_df, output_dir, logger=None):
    print("\n" + "="*60)
    print("Feature Extraction with VIRCHOW Model (Patch-level)")
    print("="*60)
    if logger:
        logger.info("Feature Extraction with VIRCHOW Model (Patch-level)")
        logger.info("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, eval_transform = initialize_virchow_model(device, logger)
    features_dir = os.path.join(output_dir, "features_VIRCHOW")
    os.makedirs(features_dir, exist_ok=True)
    wsi_groups = filtered_patches_df.groupby('wsi_id')
    desc = "Extracting VIRCHOW features (patch-level)"
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
                features = extract_patch_features_with_virchow(model, eval_transform, [patch], device)
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

# t-SNE Visualization Functions
def create_tsne_visualizations(config, output_dir, features_dir, filtered_patches_df, logger=None):
    print("\n" + "="*60)
    print("t-SNE Visualization")
    print("="*60)
    os.makedirs(output_dir, exist_ok=True)
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
            features = torch.load(pt_file, weights_only=True)
            wsi_id = os.path.basename(pt_file).replace("_features.pt", "")
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
    all_features = np.vstack(all_features)
    print(f"Total patch features: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    if len(all_features) > config.max_tsne_points:
        indices = np.random.choice(len(all_features), config.max_tsne_points, replace=False)
        all_features = all_features[indices]
        all_hosps = [all_hosps[i] for i in indices]
        all_wsi_ids = [all_wsi_ids[i] for i in indices]
        all_patch_info = [all_patch_info[i] for i in indices]
        print(f"Sampled features: {len(all_features)}")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("Starting t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=config.tsne_perplexity, max_iter=config.tsne_max_iter)
    features_2d = tsne.fit_transform(all_features)
    model_name = "VIRCHOW"
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
    for label in unique_labels:
        mask = [info['patch_label'] == label for info in all_patch_info]
        features_label = all_features[mask]
        info_label = [info for i, info in enumerate(all_patch_info) if mask[i]]
        hosps_label = [info['hosp'] for info in info_label]
        if len(features_label) == 0:
            continue
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(config.tsne_perplexity, len(features_label)-1), max_iter=config.tsne_max_iter)
        features_2d = tsne.fit_transform(features_label)
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
    print(f"\n=== Visualization statistics ===")
    print(f"Total patches: {len(all_features)}")
    print(f"WSI count: {len(set(all_wsi_ids))}")
    print(f"Hospital count: {len(set(all_hosps))}")
    print(f"Hospital distribution: {dict(pd.Series(all_hosps).value_counts())}")
    print(f"Label distribution: {dict(pd.Series([info['patch_label'] for info in all_patch_info]).value_counts())}")

# Main Pipeline
def main():
    parser = argparse.ArgumentParser(description='VIRCHOW Feature Extraction and t-SNE Visualization')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing filtered patches CSV file')
    args = parser.parse_args()
    print("VIRCHOW Feature Extraction and t-SNE Visualization Pipeline Starting")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print("Model info: VIRCHOW (Multimodal whole slide foundation model)")
    logger = setup_logging(args.input_dir)
    try:
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
        features_dir = extract_features_from_patches(Config, filtered_patches_df, args.input_dir, logger)
        create_tsne_visualizations(Config, args.input_dir, features_dir, filtered_patches_df, logger)
        print("\n" + "="*60)
        print("Pipeline completed successfully")
        print("Model used: VIRCHOW (Multimodal whole slide foundation model)")
        print(f"Total patches processed: {len(filtered_patches_df)}")
        print(f"Results saved in: {args.input_dir}")
        print("="*60)
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully")
            logger.info("Model used: VIRCHOW (Multimodal whole slide foundation model)")
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
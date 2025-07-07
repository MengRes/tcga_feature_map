#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import openslide
from sklearn.manifold import TSNE
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tsne_plot_function import create_tsne_visualization

# Import CONCH model related modules
from CONCH.conch.open_clip_custom import create_model_from_pretrained
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Configuration class
class Config:
    def __init__(self, args):
        self.wsi_dir = args.wsi_dir
        self.tsne_perplexity = args.tsne_perplexity
        self.tsne_max_iter = args.tsne_max_iter
        self.max_tsne_points = args.max_tsne_points
        self.conch_ckpt = args.conch_ckpt
        self.conch_cfg = args.conch_cfg

# Logging setup
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "conch_feature_tsne_gender.log")
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
    logger.info("CONCH Feature Extraction and t-SNE Visualization (Gender) Started")
    logger.info("=" * 60)
    return logger

# CONCH model loading
def initialize_conch_model(config, device, logger=None):
    if logger:
        logger.info("Initializing CONCH model...")
    print("Initializing CONCH model...")
    model, preprocess = create_model_from_pretrained(
        config.conch_cfg,
        config.conch_ckpt,
        device=device
    )
    model.eval()
    if logger:
        logger.info("CONCH model initialized successfully")
    print("CONCH model initialized successfully")
    return model, preprocess

def extract_features_with_conch(model, preprocess, image, device):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CONCH Feature Extraction and t-SNE Visualization by Gender")
    parser.add_argument('--input_dir', type=str, required=True, help='Input result folder, e.g. results/wsi_patch_filter_luad_20250701_231340')
    parser.add_argument('--wsi_dir', type=str, required=True, help='WSI image folder')
    parser.add_argument('--conch_ckpt', type=str, required=True, help='Path to CONCH checkpoint, e.g. ./checkpoints/conch/pytorch_model.bin')
    parser.add_argument('--conch_cfg', type=str, required=True, help='CONCH model config, e.g. conch_ViT-B-16')
    parser.add_argument('--tsne_perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--tsne_max_iter', type=int, default=1000, help='t-SNE max iterations')
    parser.add_argument('--max_tsne_points', type=int, default=10000, help='Max points for t-SNE')
    args = parser.parse_args()
    config = Config(args)
    output_dir = args.input_dir
    logger = setup_logging(output_dir)

    # Read patch information
    patch_csv = os.path.join(args.input_dir, 'conch_filtered_patches.csv')
    patches_df = pd.read_csv(patch_csv)
    # Read gender information
    wsi_meta_csv = os.path.join(args.input_dir, 'selected_wsis.csv')
    wsi_meta_df = pd.read_csv(wsi_meta_csv)
    wsi2gender = dict(zip(wsi_meta_df['filename'], wsi_meta_df['gender']))

    # ========== Feature extraction and save to features_conch folder ==========
    features_dir = os.path.join(output_dir, "features_conch")
    os.makedirs(features_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = initialize_conch_model(config, device, logger)
    wsi_groups = patches_df.groupby('wsi_id')
    for wsi_id, wsi_patches in tqdm(wsi_groups, desc="Extracting CONCH features"):
        features_list = []
        for idx, row in wsi_patches.iterrows():
            patch_x = row['patch_x']
            patch_y = row['patch_y']
            patch_size = row['patch_size']
            wsi_file = os.path.join(args.wsi_dir, f"{wsi_id}.svs")
            if not os.path.exists(wsi_file):
                continue
            try:
                slide = openslide.OpenSlide(wsi_file)
                patch = slide.read_region((patch_x, patch_y), 0, (patch_size, patch_size))
                patch = patch.convert('RGB')
                slide.close()
                features = extract_features_with_conch(model, preprocess, patch, device)
                features_list.append(features.cpu().numpy().squeeze())
            except Exception as e:
                logger.error(f"Error extracting features for {wsi_id}_{patch_x}_{patch_y}: {e}")
                continue
        if features_list:
            features_array = np.vstack(features_list)
            features_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
            torch.save(torch.from_numpy(features_array), features_file)
            logger.info(f"Features saved for {wsi_id}: {len(features_list)} patches")
    logger.info(f"Feature extraction completed. Features saved in: {features_dir}")

    # ========== t-SNE aggregate all features, color by gender ==========
    feature_files = glob.glob(os.path.join(features_dir, "*_features.pt"))
    all_features = []
    all_genders = []
    all_wsi_ids = []
    for pt_file in tqdm(feature_files, desc="Loading features"):
        try:
            features = torch.load(pt_file, weights_only=True)
            wsi_id = os.path.basename(pt_file).replace("_features.pt", "")
            gender = wsi2gender.get(wsi_id, 'unknown')
            all_features.append(features)
            all_genders.extend([gender] * len(features))
            all_wsi_ids.extend([wsi_id] * len(features))
        except Exception as e:
            logger.error(f"Error loading file {pt_file}: {e}")
            continue
    if not all_features:
        logger.error("No valid feature files found")
        sys.exit(1)
    all_features = np.vstack(all_features)
    if len(all_features) > config.max_tsne_points:
        indices = np.random.choice(len(all_features), config.max_tsne_points, replace=False)
        all_features = all_features[indices]
        all_genders = [all_genders[i] for i in indices]
        all_wsi_ids = [all_wsi_ids[i] for i in indices]
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    model_name = "conch"
    tsne_out_csv = os.path.join(viz_dir, f'tsne_gender_result_{model_name}.csv')
    tsne_fig_path = os.path.join(viz_dir, f'tsne_by_gender_{model_name}.png')
    df_tsne = create_tsne_visualization(all_features, all_genders, all_wsi_ids, tsne_fig_path, title=f"t-SNE by Gender ({model_name.upper()})")
    df_tsne.to_csv(tsne_out_csv, index=False)
    logger.info(f"t-SNE result saved to {tsne_out_csv}")
    logger.info(f"t-SNE plot saved to {tsne_fig_path}") 
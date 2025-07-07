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
from timm.layers import SwiGLUPacked

# Import t-SNE plotting function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tsne_plot_function import create_tsne_visualization

# Configuration class
class Config:
    def __init__(self, args):
        self.uni_model_name = args.uni_model_name
        self.uni_checkpoint_path = args.uni_checkpoint_path
        self.wsi_dir = args.wsi_dir
        self.tsne_perplexity = args.tsne_perplexity
        self.tsne_max_iter = args.tsne_max_iter
        self.max_tsne_points = args.max_tsne_points

# Logging setup
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "uni_feature_tsne_gender.log")
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
    logger.info("UNI Feature Extraction and t-SNE Visualization (Gender) Started")
    logger.info("=" * 60)
    return logger

# UNI model loading (reuse existing logic)
def initialize_uni_model(config, device, logger=None):
    try:
        import timm
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
    except ImportError:
        print("timm not available. Please install timm.")
        sys.exit(1)
    if config.uni_model_name.lower() == "uni2-h":
        timm_kwargs = {
            'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
            'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
            'num_classes': 0, 'no_embed_class': True,
            'mlp_layer': SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8, 'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    model.to(device)
    return model, transform

def extract_features_with_uni(model, transform, image, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_tensor)
    return features

# Main process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNI Feature Extraction and t-SNE Visualization by Gender")
    parser.add_argument('--input_dir', type=str, required=True, help='Input result folder, e.g. results/wsi_patch_filter_luad_20250701_231340')
    parser.add_argument('--wsi_dir', type=str, required=True, help='WSI image folder')
    parser.add_argument('--uni_model_name', type=str, default='uni2-h', help='UNI model name (uni or uni2-h)')
    parser.add_argument('--uni_checkpoint_path', type=str, default='./UNI/assets/ckpts/', help='UNI checkpoint path')
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

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = initialize_uni_model(config, device, logger)

    # ========== Feature extraction and save to features_UNI2-H folder ==========
    model_name = config.uni_model_name.lower().replace('-', '')
    features_dir = os.path.join(output_dir, f"features_{model_name}")
    os.makedirs(features_dir, exist_ok=True)
    wsi_groups = patches_df.groupby('wsi_id')
    for wsi_id, wsi_patches in tqdm(wsi_groups, desc=f"Extracting {model_name} features"):
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
                features = extract_features_with_uni(model, transform, patch, device)
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
    import glob
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
    tsne_out_csv = os.path.join(viz_dir, f'tsne_gender_result_{model_name}.csv')
    tsne_fig_path = os.path.join(viz_dir, f'tsne_by_gender_{model_name}.png')
    df_tsne = create_tsne_visualization(all_features, all_genders, all_wsi_ids, tsne_fig_path, title=f"t-SNE by Gender ({model_name.upper()})")
    df_tsne.to_csv(tsne_out_csv, index=False)
    logger.info(f"t-SNE result saved to {tsne_out_csv}")
    logger.info(f"t-SNE plot saved to {tsne_fig_path}") 
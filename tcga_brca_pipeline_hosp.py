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

# 添加UNI路径
sys.path.append('./UNI')
try:
    from uni import get_encoder
    UNI_AVAILABLE = True
except ImportError:
    UNI_AVAILABLE = False
    print("Warning: UNI not available. Only CONCH model will be used.")

# ===================== Parameter Settings =====================
class Config:
    # Dataset parameters
    dataset_name = "tcga-brca"
    
    # WSI filtering parameters - 考虑hosp、label和其他条件
    accept_label = ["IDC", "ILC"]            # Labels to be balanced
    accept_hosp_list = ["AR", "A2", "D8", "BH"]   # Only accept these hosp sources
    n_per_hosp = 10                          # Number of WSIs to select per hosp
    num_sampled_patches = 100                # Maximum number of patches per WSI
    patch_size = 256
    
    # Additional filtering conditions
    accept_age_groups = ["60-69", "70-79"]  # Age groups to accept (None for all)
    accept_sex = ["female"]                  # Gender to accept (None for all)
    accept_race = ["white"]                  # Race to accept (None for all)
    
    # Path configuration
    coord_dir = "/raid/mengliang/wsi_process/tcga-brca_patch/patches/"
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
    label_file = "files/tcga-brca_label.csv"
    
    # Model configuration
    # 零样本分类统一使用CONCH，特征提取可选择不同模型
    zero_shot_model = "CONCH"  # 零样本分类模型（固定为CONCH）
    feature_model = "UNI"    # 特征提取模型（可选：CONCH, UNI, UNI2-H）
    
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

# ===================== Utility Functions =====================
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
    
    param_content.append("WSI filtering parameters (考虑hosp、label和其他条件):")
    param_content.append(f"  Labels: {config.accept_label}")
    param_content.append(f"  Hosp sources: {config.accept_hosp_list}")
    param_content.append(f"  WSIs per hosp: {config.n_per_hosp}")
    param_content.append(f"  Patches per WSI: {config.num_sampled_patches}")
    param_content.append(f"  Patch size: {config.patch_size}")
    param_content.append(f"  Age groups: {config.accept_age_groups}")
    param_content.append(f"  Sex: {config.accept_sex}")
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

# ===================== Step 1: WSI Selection and Patch Extraction =====================
def step1_wsi_selection(config, output_dir, logger=None):
    """Step 1: WSI selection and patch extraction - 考虑hosp、label和其他条件"""
    step_title = "Step 1: WSI Selection and Patch Extraction"
    print("\n" + "="*60)
    print(step_title)
    print("="*60)
    
    # 确保输出目录存在
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
        logger.info(f"Total WSI records loaded: {len(df_label)}")
    
    # 构建过滤条件
    if logger:
        logger.info("Applying multi-condition filtering...")
    
    # 基础过滤条件
    mask = (
        df_label["source"].isin(config.accept_hosp_list) &
        df_label["label"].isin(config.accept_label)
    )
    
    # 添加年龄组过滤
    if config.accept_age_groups is not None:
        mask = mask & df_label["age_group"].isin(config.accept_age_groups)
        if logger:
            logger.info(f"Filtering by age groups: {config.accept_age_groups}")
    
    # 添加性别过滤
    if config.accept_sex is not None:
        mask = mask & df_label["gender"].isin(config.accept_sex)
        if logger:
            logger.info(f"Filtering by sex: {config.accept_sex}")
    
    # 添加种族过滤
    if config.accept_race is not None:
        mask = mask & df_label["race"].isin(config.accept_race)
        if logger:
            logger.info(f"Filtering by race: {config.accept_race}")
    
    df_filtered = df_label[mask].copy()
    
    print(f"Filtered data count: {len(df_filtered)} WSIs")
    if logger:
        logger.info(f"Filtered data count: {len(df_filtered)} WSIs")
    
    # 显示过滤条件的统计信息
    print(f"\n=== Filtering conditions applied ===")
    if logger:
        logger.info("=== Filtering conditions applied ===")
    
    if config.accept_age_groups is not None:
        age_counts = df_filtered["age_group"].value_counts()
        print(f"Age groups: {dict(age_counts)}")
        if logger:
            logger.info(f"Age groups: {dict(age_counts)}")
    
    if config.accept_sex is not None:
        sex_counts = df_filtered["gender"].value_counts()
        print(f"Gender: {dict(sex_counts)}")
        if logger:
            logger.info(f"Gender: {dict(sex_counts)}")
    
    if config.accept_race is not None:
        race_counts = df_filtered["race"].value_counts()
        print(f"Race: {dict(race_counts)}")
        if logger:
            logger.info(f"Race: {dict(race_counts)}")
    
    # 显示每个hosp下每个label的可用数量
    print(f"\n=== Available data distribution ===")
    if logger:
        logger.info("=== Available data distribution ===")
    
    for hosp in config.accept_hosp_list:
        hosp_data = df_filtered[df_filtered["source"] == hosp]
        if len(hosp_data) > 0:
            label_counts = hosp_data["label"].value_counts()
            print(f"{hosp}: {dict(label_counts)}")
            if logger:
                logger.info(f"{hosp}: {dict(label_counts)}")
        else:
            print(f"{hosp}: No available data")
            if logger:
                logger.warning(f"{hosp}: No available data")
    
    # Group by hosp and balanced sampling
    selected_rows = []
    total_selected = 0
    
    for hosp in config.accept_hosp_list:
        print(f"\n--- Processing hosp: {hosp} ---")
        if logger:
            logger.info(f"Processing hosp: {hosp}")
        
        hosp_data = df_filtered[df_filtered["source"] == hosp].copy()
        
        if len(hosp_data) == 0:
            print(f"  ✗ {hosp}: No available data")
            if logger:
                logger.warning(f"{hosp}: No available data")
            continue
        
        # Count available data for each label
        label_counts = hosp_data["label"].value_counts()
        print(f"  {hosp} available data distribution:")
        if logger:
            logger.info(f"{hosp} available data distribution:")
        
        for label, count in label_counts.items():
            print(f"    {label}: {count} samples")
            if logger:
                logger.info(f"  {label}: {count} samples")
        
        # 计算每个标签的目标数量，确保平衡
        n_labels = len(config.accept_label)
        target_per_label = config.n_per_hosp // n_labels
        remainder = config.n_per_hosp % n_labels
        
        hosp_selected = []
        
        # 为每个标签采样
        for i, label in enumerate(config.accept_label):
            label_data = hosp_data[hosp_data["label"] == label].copy()
            available_count = len(label_data)
            
            # 计算当前标签的目标数量
            current_target = target_per_label
            if i < remainder:
                current_target += 1
            
            actual_count = min(available_count, current_target)
            print(f"    {label}: available{available_count}, target{current_target}, actual sampling{actual_count}")
            if logger:
                logger.info(f"  {label}: available{available_count}, target{current_target}, actual sampling{actual_count}")
            
            if actual_count > 0:
                if actual_count == available_count:
                    hosp_selected.append(label_data)
                else:
                    hosp_selected.append(label_data.sample(n=actual_count, random_state=42))
        
        # 合并当前医院的采样结果
        if hosp_selected:
            hosp_df = pd.concat(hosp_selected, ignore_index=True)
            selected_rows.append(hosp_df)
            total_selected += len(hosp_df)
            print(f"  ✓ {hosp}: Successfully sampled {len(hosp_df)} WSIs")
            if logger:
                logger.info(f"✓ {hosp}: Successfully sampled {len(hosp_df)} WSIs")
        else:
            print(f"  ✗ {hosp}: No valid sampling")
            if logger:
                logger.warning(f"✗ {hosp}: No valid sampling")

    print(f"\n=== Sampling completed ===")
    print(f"Total sampled: {total_selected} WSIs")
    if logger:
        logger.info("=== Sampling completed ===")
        logger.info(f"Total sampled: {total_selected} WSIs")

    if not selected_rows:
        error_msg = "No matched WSI sets found with the given criteria."
        print(error_msg)
        if logger:
            logger.error(error_msg)
        exit()

    df_selected = pd.concat(selected_rows, ignore_index=True)

    # 统计每个来源选择的WSI数量
    source_counts = df_selected["source"].value_counts()
    print(f"\n=== WSIs selected for each source ===")
    if logger:
        logger.info("=== WSIs selected for each source ===")
    
    for source, count in source_counts.items():
        print(f"{source}: {count} WSIs")
        if logger:
            logger.info(f"{source}: {count} WSIs")

    # 统计每个标签的分布
    label_counts = df_selected["label"].value_counts()
    print(f"\n=== Distribution for each label ===")
    if logger:
        logger.info("=== Distribution for each label ===")
    
    for label, count in label_counts.items():
        print(f"{label}: {count} WSIs")
        if logger:
            logger.info(f"{label}: {count} WSIs")

    # 统计每个医院内标签的分布
    print(f"\n=== Label distribution within each hosp ===")
    if logger:
        logger.info("=== Label distribution within each hosp ===")
    
    for hosp in config.accept_hosp_list:
        hosp_data = df_selected[df_selected["source"] == hosp].copy()
        if len(hosp_data) > 0:
            hosp_label_counts = hosp_data["label"].value_counts()
            print(f"{hosp}: {dict(hosp_label_counts)}")
            if logger:
                logger.info(f"{hosp}: {dict(hosp_label_counts)}")

    # 显示选中样本的人口统计学信息
    print(f"\n=== Selected samples demographics ===")
    if logger:
        logger.info("=== Selected samples demographics ===")
    
    if "age_group" in df_selected.columns:
        age_dist = df_selected["age_group"].value_counts()
        print(f"Age distribution: {dict(age_dist)}")
        if logger:
            logger.info(f"Age distribution: {dict(age_dist)}")
    
    if "gender" in df_selected.columns:
        gender_dist = df_selected["gender"].value_counts()
        print(f"Gender distribution: {dict(gender_dist)}")
        if logger:
            logger.info(f"Gender distribution: {dict(gender_dist)}")
    
    if "race" in df_selected.columns:
        race_dist = df_selected["race"].value_counts()
        print(f"Race distribution: {dict(race_dist)}")
        if logger:
            logger.info(f"Race distribution: {dict(race_dist)}")

    df_selected["coord_path"] = df_selected["filename"].apply(lambda f: os.path.join(config.coord_dir, f"{f}.h5"))
    df_selected["wsi_path"] = df_selected["filename"].apply(lambda f: os.path.join(config.wsi_dir, f"{f}.svs"))

    output_csv = os.path.join(output_dir, "selected_sources.csv")
    df_selected.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    if logger:
        logger.info(f"Saved selected sources: {output_csv}")

    # ===================== Patch sampling and saving =====================
    print(f"\n=== Start extracting patches ===")
    print(f"Maximum patches per WSI: {config.num_sampled_patches}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    if logger:
        logger.info("=== Start extracting patches ===")
        logger.info(f"Maximum patches per WSI: {config.num_sampled_patches}")
        logger.info(f"Patch size: {config.patch_size}x{config.patch_size}")

    # Clear output directory (but keep log files)
    if os.path.exists(output_dir):
        # Only clear patches subdirectory if it exists
        patches_dir = os.path.join(output_dir, "patches")
        if os.path.exists(patches_dir):
            import shutil
            shutil.rmtree(patches_dir)
            print(f"Cleared patches directory: {patches_dir}")
            if logger:
                logger.info(f"Cleared patches directory: {patches_dir}")
    
    # Create patches directory
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    print(f"Created patches directory: {patches_dir}")
    if logger:
        logger.info(f"Created patches directory: {patches_dir}")

    patch_rows = []
    successful_wsi = 0
    failed_wsi = 0

    for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Extracting patches"):
        filename = str(row["filename"])
        coord_path = str(row["coord_path"])
        wsi_path = str(row["wsi_path"])
        slide_id = os.path.splitext(filename)[0]
        slide_output_dir = os.path.join(patches_dir, slide_id)
        os.makedirs(slide_output_dir, exist_ok=True)
        
        if not os.path.isfile(coord_path) or not os.path.isfile(wsi_path):
            print(f"[Skip] Missing file for {slide_id}")
            if logger:
                logger.warning(f"Missing file for {slide_id}")
            failed_wsi += 1
            continue
            
        try:
            with h5py.File(coord_path, "r") as f:
                if "coords" in f:
                    coords_dataset = f["coords"]
                    if isinstance(coords_dataset, h5py.Dataset):
                        coords = np.array(coords_dataset[:])
                    else:
                        print(f"[Skip] coords is not a dataset in {slide_id}")
                        if logger:
                            logger.warning(f"coords is not a dataset in {slide_id}")
                        failed_wsi += 1
                        continue
                else:
                    print(f"[Skip] No coords dataset found in {slide_id}")
                    if logger:
                        logger.warning(f"No coords dataset found in {slide_id}")
                    failed_wsi += 1
                    continue
        except Exception as e:
            print(f"[Skip] Error reading coords for {slide_id}: {e}")
            if logger:
                logger.error(f"Error reading coords for {slide_id}: {e}")
            failed_wsi += 1
            continue
            
        if len(coords) == 0:
            print(f"[Skip] No coords in {slide_id}")
            if logger:
                logger.warning(f"No coords in {slide_id}")
            failed_wsi += 1
            continue
            
        coords_list = coords.tolist() if hasattr(coords, 'tolist') else list(coords)
        sampled_coords = coords_list if len(coords_list) <= config.num_sampled_patches else random.sample(coords_list, config.num_sampled_patches)
        
        try:
            slide = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"[Skip] Error opening slide for {slide_id}: {e}")
            if logger:
                logger.error(f"Error opening slide for {slide_id}: {e}")
            failed_wsi += 1
            continue
            
        successful_patches = 0
        for coord in sampled_coords:
            try:
                x, y = map(int, coord)
                patch = slide.read_region((x, y), 0, (config.patch_size, config.patch_size)).convert("RGB")
                patch_np = np.array(patch)
                patch_name = f"{x}_{y}.npy"
                npy_path = os.path.join(slide_output_dir, patch_name)
                np.save(npy_path, patch_np)
                patch_row = row.to_dict()
                patch_row["patch_x"] = x
                patch_row["patch_y"] = y
                patch_row["npy_path"] = npy_path
                patch_rows.append(patch_row)
                successful_patches += 1
            except Exception as e:
                print(f"[Skip] Error processing patch {coord} for {slide_id}: {e}")
                if logger:
                    logger.error(f"Error processing patch {coord} for {slide_id}: {e}")
                continue
        slide.close()
        
        successful_wsi += 1
        print(f"  {slide_id}: Successfully extracted {successful_patches}/{len(sampled_coords)} patches")
        if logger:
            logger.info(f"{slide_id}: Successfully extracted {successful_patches}/{len(sampled_coords)} patches")

    patch_df = pd.DataFrame(patch_rows)
    patch_csv = os.path.join(output_dir, "all_patches.csv")
    patch_df.to_csv(patch_csv, index=False)

    print(f"\n=== Patch extraction completed ===")
    print(f"Successfully processed WSIs: {successful_wsi}")
    print(f"Failed WSIs: {failed_wsi}")
    print(f"Total extracted patches: {len(patch_df)}")
    print(f"Saved patch_df to {patch_csv}")
    
    if logger:
        logger.info("=== Patch extraction completed ===")
        logger.info(f"Successfully processed WSIs: {successful_wsi}")
        logger.info(f"Failed WSIs: {failed_wsi}")
        logger.info(f"Total extracted patches: {len(patch_df)}")
        logger.info(f"Saved patch_df to {patch_csv}")

    return patch_df

# ===================== Step 2: CONCH Zero-shot Classification =====================
def step2_conch_zero_shot(config, output_dir, patch_df, logger=None):
    """Step 2: Zero-shot Classification (using CONCH)"""
    print("\n" + "="*60)
    print("Step 2: Zero-shot Classification")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize CONCH model for zero-shot classification
    model, preprocess = initialize_model(config, device, logger, "zero_shot")
    
    print(f"Processing {len(patch_df)} patches...")
    
    labels = []
    probabilities = []
    failed_indices = []
    
    for idx, row in tqdm(patch_df.iterrows(), total=len(patch_df), desc="Zero-shot classification"):
        try:
            npy_path = row["npy_path"]
            if not os.path.exists(npy_path):
                failed_indices.append(idx)
                continue
            
            # Load patch image
            patch_np = np.load(npy_path)
            patch_image = Image.fromarray(patch_np)
            
            # Perform zero-shot classification using CONCH
            pred_label, pred_prob = zero_shot_classification_with_model(
                model, preprocess, patch_image, device, "CONCH", config
            )
            
            # Debug information (print every 100 patches)
            if idx % 100 == 0:
                print(f"Patch {idx}: pred={pred_label}, prob={pred_prob:.3f}")
            
            labels.append(pred_label)
            probabilities.append(pred_prob)
                
        except Exception as e:
            print(f"Error processing patch {idx}: {e}")
            failed_indices.append(idx)
            labels.append("Error")
            probabilities.append(0.0)
    
    # Add prediction results to DataFrame
    patch_df["patch_label"] = labels
    patch_df["patch_probability"] = probabilities
    
    # Filter high-quality patches
    wsi_label_consistent = patch_df["label"] == patch_df["patch_label"]
    high_probability = patch_df["patch_probability"] >= config.probability_threshold
    valid_patches = wsi_label_consistent & high_probability
    
    # Filter patches with both label consistency and high probability
    df_filtered = patch_df[valid_patches].copy()
    
    print(f"\n=== Zero-shot classification results ===")
    print(f"Total patches: {len(patch_df)}")
    print(f"Label consistent patches: {wsi_label_consistent.sum()}")
    print(f"High probability patches (>= {config.probability_threshold}): {high_probability.sum()}")
    print(f"Filtered patches (consistent + high prob): {len(df_filtered)}")
    
    # Count label consistency
    print(f"\nLabel consistency statistics:")
    consistency_stats = patch_df.groupby(['label', 'patch_label']).size().unstack(fill_value=0)
    print(consistency_stats)
    
    # Count distribution of filtered patches
    if len(df_filtered) > 0:
        print(f"\nFiltered patches distribution:")
        print(f"By WSI label distribution: {dict(df_filtered['label'].value_counts())}")
        print(f"By patch label distribution: {dict(df_filtered['patch_label'].value_counts())}")
        print(f"Probability range: {df_filtered['patch_probability'].min():.3f} - {df_filtered['patch_probability'].max():.3f}")
        print(f"Average probability: {df_filtered['patch_probability'].mean():.3f}")
    else:
        print(f"\nWarning: No patches passed filtering conditions!")
        print(f"Suggest checking label consistency or adjusting prompts")
    
    # Save results
    all_patches_file = os.path.join(output_dir, "all_patches_with_predictions.csv")
    filtered_patches_file = os.path.join(output_dir, "filtered_patches.csv")
    
    patch_df.to_csv(all_patches_file, index=False)
    df_filtered.to_csv(filtered_patches_file, index=False)
    
    print(f"Saved all patches prediction results: {all_patches_file}")
    print(f"Saved filtered patches: {filtered_patches_file}")
    
    return df_filtered

# ===================== Step 3: CONCH Feature Extraction =====================
def step3_conch_feature_extraction(config, output_dir, filtered_patches_df, logger=None):
    """Step 3: Feature Extraction"""
    print("\n" + "="*60)
    print("Step 3: Feature Extraction")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model, preprocess = initialize_model(config, device, logger, "feature")
    
    # Process by WSI
    wsi_groups = filtered_patches_df.groupby("filename")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    successful_wsi = 0
    failed_wsi = 0
    
    for wsi_id, group in tqdm(wsi_groups, desc="Extracting features"):
        try:
            features_list = []
            patch_info_list = []
            
            for _, row in group.iterrows():
                try:
                    npy_path = row["npy_path"]
                    if not os.path.exists(npy_path):
                        continue
                    
                    # Load patch image
                    patch_np = np.load(npy_path)
                    patch_image = Image.fromarray(patch_np)
                    
                    # Extract features using the specified model
                    features = extract_features_with_model(model, preprocess, patch_image, device, config.feature_model)
                    features = features.cpu()
                    
                    features_list.append(features)
                    
                    # Save patch information
                    patch_info = {
                        'patch_x': row['patch_x'],
                        'patch_y': row['patch_y'],
                        'patch_label': row['patch_label'],
                        'patch_probability': row['patch_probability']
                    }
                    patch_info_list.append(patch_info)
                    
                except Exception as e:
                    print(f"Error processing patch in {wsi_id}: {e}")
                    continue
            
            if len(features_list) > 0:
                # Concatenate features
                features_tensor = torch.cat(features_list, dim=0)
                
                # Save features
                features_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
                torch.save(features_tensor, features_file)
                
                # Save patch info
                patch_info_df = pd.DataFrame(patch_info_list)
                patch_info_file = os.path.join(features_dir, f"{wsi_id}_patch_info.csv")
                patch_info_df.to_csv(patch_info_file, index=False)
                
                successful_wsi += 1
                print(f"  ✓ {wsi_id}: Saved {len(features_list)} features (dim: {features_tensor.shape[1]})")
                if logger:
                    logger.info(f"✓ {wsi_id}: Saved {len(features_list)} features (dim: {features_tensor.shape[1]})")
            else:
                failed_wsi += 1
                print(f"  ✗ {wsi_id}: No valid features")
                if logger:
                    logger.warning(f"✗ {wsi_id}: No valid features")
                
        except Exception as e:
            failed_wsi += 1
            print(f"  ✗ {wsi_id}: Error - {e}")
            if logger:
                logger.error(f"✗ {wsi_id}: Error - {e}")
    
    print(f"\n=== Feature extraction completed ===")
    print(f"Model type: {config.feature_model}")
    print(f"Successful WSIs: {successful_wsi}")
    print(f"Failed WSIs: {failed_wsi}")
    print(f"Features saved to: {features_dir}")
    
    if logger:
        logger.info("=== Feature extraction completed ===")
        logger.info(f"Model type: {config.feature_model}")
        logger.info(f"Successful WSIs: {successful_wsi}")
        logger.info(f"Failed WSIs: {failed_wsi}")
        logger.info(f"Features saved to: {features_dir}")
    
    return features_dir

# ===================== Step 4: TSNE Visualization =====================
def step4_tsne_visualization(config, output_dir, features_dir, logger=None):
    """Step 4: TSNE Visualization"""
    print("\n" + "="*60)
    print("Step 4: TSNE Visualization")
    print("="*60)
    
    # 确保输出目录存在
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
                   c=[color_map[hosp]], label=hosp, alpha=0.7, s=20)
    
    plt.title('TSNE Visualization - Grouped by Hospital', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
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
                        c=[color_map[hosp]], label=hosp, alpha=0.7, s=20)
        plt.title(f't-SNE ({label}) - Grouped by Hospital', fontsize=22, fontweight='bold')
        plt.xlabel('TSNE1', fontsize=18)
        plt.ylabel('TSNE2', fontsize=18)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
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
                   c=[color_map[label]], label=label, alpha=0.7, s=20)
    plt.title('t-SNE Visualization - Grouped by Label', fontsize=22, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=18)
    plt.ylabel('TSNE2', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
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

# ===================== Model Initialization Functions =====================
def initialize_model(config, device, logger=None, model_type="feature"):
    """Initialize model based on model_type"""
    if model_type == "zero_shot":
        # 零样本分类统一使用CONCH
        return initialize_conch_model(config, device, logger)
    elif model_type == "feature":
        # 特征提取可选择不同模型
        if config.feature_model.upper() == "CONCH":
            return initialize_conch_model(config, device, logger)
        elif config.feature_model.upper() in ["UNI", "UNI2-H"]:
            return initialize_uni_model(config, device, logger)
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
    
    # 确定UNI模型名称
    uni_model_name = config.uni_model_name
    if config.feature_model.upper() == "UNI2-H":
        uni_model_name = "uni2-h"
    
    # 初始化UNI模型
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

# ===================== Main Function =====================
def main():
    """Main function"""
    print("TCGA-BRCA Pipeline starting")
    print("="*60)
    
    # 显示模型信息
    print(f"Zero-shot model: {Config.zero_shot_model} (fixed)")
    print(f"Feature extraction model: {Config.feature_model}")
    
    # 检查UNI模型是否可用（如果使用UNI进行特征提取）
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
        print("Pipeline completed!")
        print(f"Zero-shot model: {Config.zero_shot_model}")
        print(f"Feature extraction model: {Config.feature_model}")
        print(f"All results saved: {output_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
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
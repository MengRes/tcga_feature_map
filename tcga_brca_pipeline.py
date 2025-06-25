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

# ===================== Parameter Settings =====================
class Config:
    # Dataset parameters
    dataset_name = "tcga-brca"
    
    # WSI filtering parameters
    accept_age_groups = ["60-69", "70-79"]  # Only accept these age groups
    accept_sex = ["female"]                  # Only accept these genders
    accept_race = ["white"]                  # Only accept these races
    accept_label = ["IDC", "ILC"]            # Labels to be balanced
    accept_hosp_list = ["AR", "A2", "D8"]   # Only accept these hosp sources
    n_per_hosp = 10                          # Number of WSIs to select per hosp
    num_sampled_patches = 100                # Maximum number of patches per WSI
    patch_size = 256
    
    # Path configuration
    coord_dir = "/raid/mengliang/wsi_process/tcga-brca_patch/patches/"
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
    label_file = "files/tcga-brca_label.csv"
    
    # CONCH model parameters
    checkpoint_path = './checkpoints/conch/pytorch_model.bin'
    model_cfg = 'conch_ViT-B-16'
    probability_threshold = 0.8  # Probability threshold
    skip_zero_shot = False  # Default to perform zero-shot classification
    
    # Zero-shot classification parameters
    classes = ['invasive ductal carcinoma', 'invasive lobular carcinoma']
    prompts = ['an H&E image of invasive ductal carcinoma', 'an H&E image of invasive lobular carcinoma']
    
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
    
    param_content.append("WSI filtering parameters:")
    param_content.append(f"  Age groups: {config.accept_age_groups}")
    param_content.append(f"  Gender: {config.accept_sex}")
    param_content.append(f"  Race: {config.accept_race}")
    param_content.append(f"  Label: {config.accept_label}")
    param_content.append(f"  Hosp sources: {config.accept_hosp_list}")
    param_content.append(f"  WSIs per hosp: {config.n_per_hosp}")
    param_content.append(f"  Patches per WSI: {config.num_sampled_patches}")
    param_content.append(f"  Patch size: {config.patch_size}")
    param_content.append("")
    
    param_content.append("CONCH model parameters:")
    param_content.append(f"  Model config: {config.model_cfg}")
    param_content.append(f"  Probability threshold: {config.probability_threshold}")
    param_content.append(f"  Classes: {config.classes}")
    param_content.append(f"  Prompts: {config.prompts}")
    param_content.append("")
    
    param_content.append("Path configuration:")
    param_content.append(f"  Coordinate directory: {config.coord_dir}")
    param_content.append(f"  WSI directory: {config.wsi_dir}")
    param_content.append(f"  Label file: {config.label_file}")
    param_content.append(f"  Model checkpoint: {config.checkpoint_path}")
    
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
    """Step 1: WSI selection and patch extraction"""
    step_title = "Step 1: WSI Selection and Patch Extraction"
    print("\n" + "="*60)
    print(step_title)
    print("="*60)
    
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
    
    # Multi-condition filtering
    if logger:
        logger.info("Applying multi-condition filtering...")
    mask = (
        df_label["age_group"].isin(config.accept_age_groups) &
        df_label["gender"].isin(config.accept_sex) &
        df_label["race"].isin(config.accept_race) &
        df_label["label"].isin(config.accept_label) &
        df_label["source"].isin(config.accept_hosp_list)
    )
    df_filtered = df_label[mask].copy()
    
    print(f"Filtered data count: {len(df_filtered)} WSIs")
    if logger:
        logger.info(f"Filtered data count: {len(df_filtered)} WSIs")
    
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
        
        # Calculate target number for each label (try to balance)
        n_labels = len(config.accept_label)
        target_per_label = config.n_per_hosp // n_labels
        remainder = config.n_per_hosp % n_labels
        
        hosp_selected = []
        
        # Sample for each label
        for i, label in enumerate(config.accept_label):
            label_data = hosp_data[hosp_data["label"] == label].copy()
            available_count = len(label_data)
            
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
        
        # Merge sampling results for current hosp
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

    # Count WSIs selected for each source
    source_counts = df_selected["source"].value_counts()
    print(f"\n=== WSIs selected for each source ===")
    if logger:
        logger.info("=== WSIs selected for each source ===")
    
    for source, count in source_counts.items():
        print(f"{source}: {count} WSIs")
        if logger:
            logger.info(f"{source}: {count} WSIs")

    # Count distribution for each label
    label_counts = df_selected["label"].value_counts()
    print(f"\n=== Distribution for each label ===")
    if logger:
        logger.info("=== Distribution for each label ===")
    
    for label, count in label_counts.items():
        print(f"{label}: {count} WSIs")
        if logger:
            logger.info(f"{label}: {count} WSIs")

    # Count label distribution within each hosp
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
    """Step 2: CONCH Zero-shot Classification"""
    print("\n" + "="*60)
    print("Step 2: CONCH Zero-shot Classification")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model, preprocess = create_model_from_pretrained(config.model_cfg, config.checkpoint_path, device=device)
    model.eval()
    
    tokenizer = get_tokenizer()
    tokenized_prompts = tokenize(texts=config.prompts, tokenizer=tokenizer).to(device)
    
    label_map = {
        'invasive ductal carcinoma': 'IDC',
        'invasive lobular carcinoma': 'ILC'
    }
    
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
            
            # Preprocess image
            image_tensor = preprocess(patch_image).unsqueeze(0).to(device)
            
            # Get image features
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(tokenized_prompts)
                
                # Calculate similarity using logit scale (same as 03_conch_zero-shot.py)
                sim_scores = (image_features @ text_features.T * model.logit_scale.exp()).softmax(dim=-1)
                
                # Save raw similarity scores for debugging
                raw_scores = sim_scores.cpu().numpy().flatten()
                
                # Get prediction results
                pred_idx = torch.argmax(sim_scores, dim=-1).item()
                pred_label = config.classes[pred_idx]
                pred_prob = sim_scores[0][pred_idx].item()
                
                # Debug information (print every 100 patches)
                if idx % 100 == 0:
                    print(f"Patch {idx}: raw_scores={raw_scores}, probs={sim_scores.cpu().numpy().flatten()}, pred={pred_label}, prob={pred_prob:.3f}")
                
                labels.append(label_map[pred_label])
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
    # Select patch_label with the highest probability, then filter patch_label and wsi_label consistent patches
    wsi_label_consistent = patch_df["label"] == patch_df["patch_label"]
    
    # Add probability threshold filtering
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
    """Step 3: CONCH Feature Extraction"""
    print("\n" + "="*60)
    print("Step 3: CONCH Feature Extraction")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model, preprocess = create_model_from_pretrained(config.model_cfg, config.checkpoint_path, device=device)
    model.eval()
    
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
                    
                    # Preprocess image
                    image_tensor = preprocess(patch_image).unsqueeze(0).to(device)
                    
                    # Extract features
                    with torch.no_grad():
                        image_features = model.encode_image(image_tensor)
                        features = image_features.cpu()
                    
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
            
            if features_list:
                # Merge features
                all_features = torch.cat(features_list, dim=0)
                patch_info_df = pd.DataFrame(patch_info_list)
                
                # Save feature file
                feature_file = os.path.join(features_dir, f"{wsi_id}_features.pt")
                patch_info_file = os.path.join(features_dir, f"{wsi_id}_patch_info.csv")
                
                torch.save({
                    'features': all_features,
                    'wsi_id': wsi_id,
                    'patch_info': patch_info_df
                }, feature_file)
                
                patch_info_df.to_csv(patch_info_file, index=False)
                
                successful_wsi += 1
                print(f"  {wsi_id}: Successfully extracted {len(features_list)} patch features")
            else:
                failed_wsi += 1
                print(f"  {wsi_id}: No valid features")
                
        except Exception as e:
            failed_wsi += 1
            print(f"Error processing WSI {wsi_id}: {e}")
    
    print(f"\n=== Feature extraction completed ===")
    print(f"Successfully processed WSIs: {successful_wsi}")
    print(f"Failed WSIs: {failed_wsi}")
    print(f"Feature file save directory: {features_dir}")
    
    return features_dir

# ===================== Step 4: TSNE Visualization =====================
def step4_tsne_visualization(config, output_dir, features_dir, logger=None):
    """Step 4: TSNE Visualization"""
    print("\n" + "="*60)
    print("Step 4: TSNE Visualization")
    print("="*60)
    
    # Load all features
    all_features = []
    all_hosps = []
    all_wsi_ids = []
    all_patch_info = []
    
    pt_files = glob.glob(os.path.join(features_dir, "*_features.pt"))
    print(f"Found {len(pt_files)} feature files")
    
    for pt_file in tqdm(pt_files, desc="Loading feature files"):
        try:
            data = torch.load(pt_file, map_location='cpu')
            features = data['features'].cpu().numpy()
            wsi_id = data['wsi_id']
            patch_info = data['patch_info']
            
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

# ===================== Main Function =====================
def main():
    """Main function"""
    print("TCGA-BRCA Pipeline starting")
    print("="*60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{Config.dataset_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save parameters
    logger = setup_logging(output_dir)
    save_parameters(Config, output_dir, logger)
    
    try:
        # Step 1: WSI Selection and Patch Extraction
        patch_df = step1_wsi_selection(Config, output_dir, logger)
        
        # Step 2: CONCH Zero-shot Classification
        filtered_patches_df = step2_conch_zero_shot(Config, output_dir, patch_df, logger)
        
        # Step 3: CONCH Feature Extraction
        features_dir = step3_conch_feature_extraction(Config, output_dir, filtered_patches_df, logger)
        
        # Step 4: TSNE Visualization
        step4_tsne_visualization(Config, output_dir, features_dir, logger)
        
        print("\n" + "="*60)
        print("Pipeline completed!")
        print(f"All results saved: {output_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
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
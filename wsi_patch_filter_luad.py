#!/usr/bin/env python3
"""
WSI and Patch Filtering for TCGA-LUAD
=====================================

This script filters WSIs from TCGA-LUAD dataset based on specific criteria:
- Disease label: LUAD
- Hospital source: "55"
- Age: 60-69
- Race: white
- Gender: male and female

Then extracts patches and filters them using CONCH model to ensure
patch-level and slide-level label consistency.
"""

import os
import numpy as np
import pandas as pd
import h5py
import openslide
from tqdm import tqdm
import torch
from CONCH.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from datetime import datetime
import logging
import yaml
import argparse

# Configuration
class Config:
    """Configuration class for TCGA-LUAD processing"""
    
    # Dataset parameters
    dataset_name = "tcga-luad"
    
    # WSI filtering parameters
    accept_label = ["LUAD"]                  # Labels to be balanced
    accept_hosp_list = ["55"]                # Only accept hospital "55"
    n_per_hosp = 20                          # Number of WSIs to select per hosp
    num_sampled_patches = 100                # Maximum number of patches per WSI
    patch_size = 256
    
    # Additional filtering conditions
    accept_age_groups = ["60-69"]            # Age groups to accept
    accept_gender = ["male", "female"]       # Gender to accept
    accept_race = ["white"]                  # Race to accept
    
    # Path configuration
    coord_dir = "/raid/mengliang/wsi_process/tcga-luad_patch/patches/"
    wsi_dir = "/home/mxz3935/dataset_folder/tcga-luad/"
    label_file = "files/tcga-luad_label.csv"
    
    # CONCH model parameters
    checkpoint_path = './checkpoints/conch/pytorch_model.bin'
    model_cfg = 'conch_ViT-B-16'
    probability_threshold = 0.8  # Probability threshold
    
    # Zero-shot classification parameters for LUAD
    classes = ['lung adenocarcinoma', 'other', 'normal lung tissue']
    prompts = [
        'an H&E image of lung adenocarcinoma',
        'an H&E image of other lung tumor',
        'an H&E image of normal lung tissue'
    ]
    
    # Label mapping from model prediction to standard labels
    label_map = {
        'lung adenocarcinoma': 'LUAD',
        'other': 'OTHER',
        'normal lung tissue': 'NORMAL'
    }
    
    @classmethod
    def from_yaml(cls, yaml_file):
        """Load configuration from YAML file"""
        config = cls()
        
        if not os.path.exists(yaml_file):
            print(f"Warning: Config file {yaml_file} not found. Using default configuration.")
            return config
        
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update configuration with values from YAML
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"Config updated: {key} = {value}")
                else:
                    print(f"Warning: Unknown config parameter '{key}' ignored")
                    
        except Exception as e:
            print(f"Error loading config file {yaml_file}: {e}")
            print("Using default configuration.")
        
        return config
    
    def save_yaml(self, yaml_file):
        """Save current configuration to YAML file"""
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                config_dict[attr] = getattr(self, attr)
        
        try:
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            print(f"Configuration saved to {yaml_file}")
        except Exception as e:
            print(f"Error saving config file {yaml_file}: {e}")
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("Current Configuration:")
        print("="*60)
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                print(f"  {attr}: {value}")
        print("="*60)

# Utility Functions
def age_group(age):
    """Convert age to age group"""
    try:
        age = int(age)
        return f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
    except:
        return "unknown"

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, "wsi_patch_filter_luad.log")
    
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
    logger.info("TCGA-LUAD WSI and Patch Filtering Log Started")
    logger.info("=" * 60)
    
    return logger

# WSI Selection
def select_wsis(config, output_dir, logger=None):
    """Step 1: WSI selection - considering hosp, label and other conditions"""
    print("\n" + "="*60)
    print("Step 1: WSI Selection for TCGA-LUAD")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if logger:
        logger.info("Step 1: WSI Selection for TCGA-LUAD")
        logger.info("="*60)
    
    # Load label data
    if logger:
        logger.info("Loading TCGA-LUAD label data...")
    df_label = pd.read_csv(config.label_file)
    df_label["source"] = df_label["filename"].str.extract(r"TCGA-([A-Z0-9]{2})-")
    df_label["age_group"] = df_label["age"].apply(age_group)
    
    print(f"Loaded {len(df_label)} label records")
    if logger:
        logger.info(f"Loaded {len(df_label)} label records")
    
    # Filter by label
    df_filtered = df_label[df_label["label"].isin(config.accept_label)].copy()
    print(f"After label filtering (LUAD): {len(df_filtered)} records")
    
    # Filter by hospital
    df_filtered = df_filtered[df_filtered["source"].isin(config.accept_hosp_list)].copy()
    print(f"After hospital filtering (55): {len(df_filtered)} records")
    
    # Filter by age group
    if config.accept_age_groups:
        df_filtered = df_filtered[df_filtered["age_group"].isin(config.accept_age_groups)].copy()
        print(f"After age group filtering (60-69): {len(df_filtered)} records")
    
    # Filter by gender
    if config.accept_gender:
        df_filtered = df_filtered[df_filtered["gender"].isin(config.accept_gender)].copy()
        print(f"After gender filtering (male, female): {len(df_filtered)} records")
    
    # Filter by race
    if config.accept_race:
        df_filtered = df_filtered[df_filtered["race"].isin(config.accept_race)].copy()
        print(f"After race filtering (white): {len(df_filtered)} records")
    
    # Balance by hospital and gender
    selected_wsis = []
    for hosp in config.accept_hosp_list:
        hosp_wsis = df_filtered[df_filtered["source"] == hosp]
        if len(hosp_wsis) > 0:
            # Balance by gender within each hospital
            male_wsis = hosp_wsis[hosp_wsis["gender"] == "male"]
            female_wsis = hosp_wsis[hosp_wsis["gender"] == "female"]
            
            # Calculate how many to select from each gender
            n_per_gender = config.n_per_hosp // 2  # Split equally between male and female
            
            selected_male = []
            selected_female = []
            
            # Select male WSIs
            if len(male_wsis) > 0:
                n_male = min(n_per_gender, len(male_wsis))
                selected_male = male_wsis.sample(n=n_male, random_state=42)
                print(f"Selected {len(selected_male)} male WSIs from hospital {hosp}")
                if logger:
                    logger.info(f"Selected {len(selected_male)} male WSIs from hospital {hosp}")
            
            # Select female WSIs
            if len(female_wsis) > 0:
                n_female = min(n_per_gender, len(female_wsis))
                selected_female = female_wsis.sample(n=n_female, random_state=42)
                print(f"Selected {len(selected_female)} female WSIs from hospital {hosp}")
                if logger:
                    logger.info(f"Selected {len(selected_female)} female WSIs from hospital {hosp}")
            
            # Combine selected WSIs
            if len(selected_male) > 0 or len(selected_female) > 0:
                selected = pd.concat([selected_male, selected_female], ignore_index=True)
                selected_wsis.append(selected)
                print(f"Total selected {len(selected)} WSIs from hospital {hosp} (Male: {len(selected_male)}, Female: {len(selected_female)})")
                if logger:
                    logger.info(f"Total selected {len(selected)} WSIs from hospital {hosp} (Male: {len(selected_male)}, Female: {len(selected_female)})")
    
    if not selected_wsis:
        raise ValueError("No WSIs selected after filtering")
    
    df_selected = pd.concat(selected_wsis, ignore_index=True)
    print(f"Total selected WSIs: {len(df_selected)}")
    if logger:
        logger.info(f"Total selected WSIs: {len(df_selected)}")
    
    # Print gender balance statistics
    gender_counts = df_selected['gender'].value_counts()
    print(f"Gender distribution: {dict(gender_counts)}")
    if logger:
        logger.info(f"Gender distribution: {dict(gender_counts)}")
    
    # Check gender balance
    if len(gender_counts) >= 2:
        male_count = gender_counts.get('male', 0)
        female_count = gender_counts.get('female', 0)
        balance_ratio = min(male_count, female_count) / max(male_count, female_count) if max(male_count, female_count) > 0 else 0
        print(f"Gender balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")
        if logger:
            logger.info(f"Gender balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")
    
    # Save selected WSIs
    selected_wsi_file = os.path.join(output_dir, "selected_wsis.csv")
    df_selected.to_csv(selected_wsi_file, index=False)
    print(f"Selected WSIs saved: {selected_wsi_file}")
    
    return df_selected

# Patch Extraction
def extract_patches(config, df_selected, output_dir, logger=None):
    """Step 2: Extract patches from selected WSIs"""
    print("\n" + "="*60)
    print("Step 2: Patch Extraction")
    print("="*60)
    
    if logger:
        logger.info("Step 2: Patch Extraction")
        logger.info("="*60)
    
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
    
    print(f"Total patches extracted: {len(patch_df)}")
    print(f"Patches per label: {dict(patch_df['label'].value_counts())}")
    print(f"Patches per hospital: {dict(patch_df['hosp'].value_counts())}")
    
    if logger:
        logger.info(f"Total patches extracted: {len(patch_df)}")
        logger.info(f"Patches per label: {dict(patch_df['label'].value_counts())}")
        logger.info(f"Patches per hospital: {dict(patch_df['hosp'].value_counts())}")
    
    # Save patch information
    patch_file = os.path.join(output_dir, "extracted_patches.csv")
    patch_df.to_csv(patch_file, index=False)
    print(f"Patch information saved: {patch_file}")
    
    return patch_df

# CONCH Model Functions
def initialize_conch_model(config, device, logger=None):
    """Initialize CONCH model"""
    if logger:
        logger.info("Initializing CONCH model...")
    
    print("Initializing CONCH model...")
    model, preprocess = create_model_from_pretrained(config.model_cfg, config.checkpoint_path, device=device)
    model.eval()
    
    print("CONCH model initialized successfully")
    if logger:
        logger.info("CONCH model initialized successfully")
    
    return model, preprocess

def zero_shot_classification_conch(model, preprocess, image, device, config):
    """Perform zero-shot classification using CONCH model"""
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
    
    # Use label mapping from configuration
    return config.label_map.get(pred_label, pred_label), pred_prob

# Patch Filtering with CONCH
def filter_patches_with_conch(config, patch_df, output_dir, logger=None):
    """Step 3: Filter patches using CONCH zero-shot classification
    
    IMPORTANT: This function includes label consistency checking.
    Only patches where CONCH prediction matches WSI ground truth label are kept.
    This ensures patch-level and slide-level label consistency.
    """
    print("\n" + "="*60)
    print("Step 3: CONCH Zero-shot Patch Filtering")
    print("="*60)
    
    if logger:
        logger.info("Step 3: CONCH Zero-shot Patch Filtering")
        logger.info("="*60)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = initialize_conch_model(config, device, logger)
    
    # Process patches
    results = []
    total_processed = 0
    high_confidence_patches = 0
    label_consistent_patches = 0
    
    for idx, row in tqdm(patch_df.iterrows(), total=len(patch_df), desc="CONCH filtering"):
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
            pred_label, pred_prob = zero_shot_classification_conch(
                model, preprocess, patch, device, config
            )
            
            total_processed += 1
            
            # Check if patch meets probability threshold
            if pred_prob >= config.probability_threshold:
                high_confidence_patches += 1
                
                # Check if patch label matches WSI label (consistency check)
                if pred_label == row["label"]:
                    label_consistent_patches += 1
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
    
    # Print detailed filtering statistics
    print(f"\n=== CONCH Filtering Statistics ===")
    print(f"Total patches processed: {total_processed}")
    print(f"High confidence patches (prob >= {config.probability_threshold}): {high_confidence_patches}")
    print(f"Label consistent patches (patch_label == wsi_label): {label_consistent_patches}")
    print(f"Final filtered patches: {len(filtered_patches_df)}")
    print(f"Filtering rate: {len(filtered_patches_df)/total_processed*100:.1f}%")
    print(f"Label consistency rate: {label_consistent_patches/high_confidence_patches*100:.1f}%" if high_confidence_patches > 0 else "Label consistency rate: N/A")
    
    print(f"\nFiltered patches per label: {dict(filtered_patches_df['patch_label'].value_counts())}")
    print(f"Filtered patches per hospital: {dict(filtered_patches_df['hosp'].value_counts())}")
    
    if logger:
        logger.info("=== CONCH Filtering Statistics ===")
        logger.info(f"Total patches processed: {total_processed}")
        logger.info(f"High confidence patches: {high_confidence_patches}")
        logger.info(f"Label consistent patches: {label_consistent_patches}")
        logger.info(f"Final filtered patches: {len(filtered_patches_df)}")
        logger.info(f"Filtering rate: {len(filtered_patches_df)/total_processed*100:.1f}%")
        logger.info(f"Label consistency rate: {label_consistent_patches/high_confidence_patches*100:.1f}%" if high_confidence_patches > 0 else "Label consistency rate: N/A")
    
    # Save filtered patches
    filtered_patch_file = os.path.join(output_dir, "conch_filtered_patches.csv")
    filtered_patches_df.to_csv(filtered_patch_file, index=False)
    print(f"Filtered patches saved: {filtered_patch_file}")
    
    if logger:
        logger.info(f"Filtered patches saved: {filtered_patch_file}")
    
    return filtered_patches_df

def create_default_config_file(config_file):
    """Create a default configuration file"""
    config = Config()
    config.save_yaml(config_file)
    print(f"Default configuration file created: {config_file}")

def main(config_file=None):
    """Main pipeline function"""
    print("TCGA-LUAD WSI and Patch Filtering Pipeline")
    print("="*60)
    
    # Load configuration
    if config_file and os.path.exists(config_file):
        config = Config.from_yaml(config_file)
    else:
        config = Config()
        if config_file:
            create_default_config_file(config_file)
    
    # Print configuration
    config.print_config()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/wsi_patch_filter_luad_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    try:
        # Step 1: WSI Selection
        df_selected = select_wsis(config, output_dir, logger)
        
        # Step 2: Patch Extraction
        patch_df = extract_patches(config, df_selected, output_dir, logger)
        
        # Step 3: CONCH Patch Filtering
        filtered_patches_df = filter_patches_with_conch(config, patch_df, output_dir, logger)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print(f"Results saved in: {output_dir}")
        print("="*60)
        
        if logger:
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved in: {output_dir}")
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TCGA-LUAD WSI and Patch Filtering')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config) 
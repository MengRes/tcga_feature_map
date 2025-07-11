from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import os
import glob

def perform_stain_normalization(features, method='standard'):
    """Perform stain normalization on features
    Args:
        features (np.ndarray): Feature matrix
        method (str): Normalization method ('standard', 'minmax', or None)
    Returns:
        np.ndarray: Normalized features
    """
    if method == 'standard':
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = features
    
    return normalized_features


def convert_pt_features_to_csv(feature_dir, output_file):
    """
    Convert all .pt feature files in a directory to a single CSV file
    
    Args:
        feature_dir (str): Directory containing .pt feature files
        output_file (str): Output CSV file path
        
    Returns:
        pd.DataFrame: Combined feature DataFrame
    """
    print(f"Converting features from {feature_dir} to {output_file}")
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(feature_dir, "*.pt"))
    print(f"Found {len(pt_files)} feature files")
    
    all_features = []
    
    for pt_file in pt_files:
        # Extract WSI ID from filename
        filename = os.path.basename(pt_file)
        wsi_id = filename.split('_features.pt')[0]
        
        # Load features
        features = torch.load(pt_file)
        
        # Convert to numpy if needed
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        
        # Create DataFrame for this WSI
        feature_df = pd.DataFrame(features)
        feature_df['wsi_id'] = wsi_id
        feature_df['patch_id'] = [f"{wsi_id}_patch_{i}" for i in range(len(feature_df))]
        
        # Move wsi_id and patch_id to front
        cols = ['wsi_id', 'patch_id'] + [col for col in feature_df.columns if col not in ['wsi_id', 'patch_id']]
        feature_df = feature_df[cols]
        
        all_features.append(feature_df)
        
        print(f"  Processed {wsi_id}: {features.shape}")
    
    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"Saved combined features to {output_file}")
    print(f"Total shape: {combined_df.shape}")
    
    return combined_df


def load_features_from_csv(feature_file):
    """
    Load features from CSV file
    
    Args:
        feature_file (str): Path to feature CSV file
        
    Returns:
        tuple: (features_df, feature_columns)
    """
    print(f"Loading features from {feature_file}")
    features_df = pd.read_csv(feature_file)
    
    # Identify feature columns (exclude metadata columns)
    exclude_cols = ['wsi_id', 'patch_id', 'hospital', 'gender', 'age', 'diagnosis']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"Loaded {len(features_df)} samples with {len(feature_columns)} features")
    
    return features_df, feature_columns 
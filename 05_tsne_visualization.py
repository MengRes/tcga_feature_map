import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import glob

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Configure paths
features_dir = "tcga-brca_filtered_features"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

def extract_hosp_from_filename(filename):
    """Extract hospital information from filename"""
    # Filename format: TCGA-A2-A0CY-01Z-00-DX1.xxx_features.pt
    # Hospital is the A2 part in TCGA-A2-A0CY
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[1]  # Return hospital code
    return "Unknown"

def load_all_features():
    """Load features from all pt files"""
    all_features = []
    all_hosps = []
    all_wsi_ids = []
    all_patch_info = []
    
    # Get all pt files
    pt_files = glob.glob(os.path.join(features_dir, "*_features.pt"))
    print(f"Found {len(pt_files)} feature files")
    
    for pt_file in tqdm(pt_files, desc="Loading feature files"):
        try:
            # Load pt file
            data = torch.load(pt_file, map_location='cpu')
            features = data['features'].cpu().numpy()
            wsi_id = data['wsi_id']
            patch_info = data['patch_info']
            
            # Extract hospital information
            hosp = extract_hosp_from_filename(wsi_id)
            
            # Add data
            all_features.append(features)
            all_hosps.extend([hosp] * len(features))
            all_wsi_ids.extend([wsi_id] * len(features))
            
            # Add patch information
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
    
    # Combine all features
    if all_features:
        all_features = np.vstack(all_features)
        print(f"Total features: {len(all_features)}")
        print(f"Feature dimension: {all_features.shape[1]}")
        print(f"Unique hospitals: {len(set(all_hosps))}")
        print(f"Hospital distribution: {pd.Series(all_hosps).value_counts()}")
        
        return all_features, all_hosps, all_wsi_ids, all_patch_info
    else:
        print("No valid feature files found")
        return None, None, None, None

def plot_tsne_by_hosp(features, hosps, output_path):
    """Plot TSNE visualization grouped by hospital"""
    print("Starting TSNE dimensionality reduction...")
    
    # Perform TSNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Create DataFrame for plotting
    df_tsne = pd.DataFrame({
        'TSNE1': features_2d[:, 0],
        'TSNE2': features_2d[:, 1],
        'Hosp': hosps
    })
    
    # Set color mapping
    unique_hosps = sorted(set(hosps))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_hosps)))
    color_map = dict(zip(unique_hosps, colors))
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot scatter points
    for hosp in unique_hosps:
        mask = df_tsne['Hosp'] == hosp
        plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'], 
                   c=[color_map[hosp]], label=hosp, alpha=0.7, s=20)
    
    plt.title('TSNE Visualization - Grouped by Hospital', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TSNE plot saved to: {output_path}")
    plt.show()

def plot_tsne_by_label(features, patch_info, output_path):
    """Plot TSNE visualization grouped by patch label"""
    print("Plotting TSNE by label...")
    
    # Perform TSNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Get patch labels
    patch_labels = [info['patch_label'] for info in patch_info]
    
    # Create DataFrame
    df_tsne = pd.DataFrame({
        'TSNE1': features_2d[:, 0],
        'TSNE2': features_2d[:, 1],
        'Label': patch_labels
    })
    
    # Set colors
    colors = {'IDC': 'red', 'ILC': 'blue'}
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    for label in ['IDC', 'ILC']:
        mask = df_tsne['Label'] == label
        if mask.sum() > 0:
            plt.scatter(df_tsne[mask]['TSNE1'], df_tsne[mask]['TSNE2'], 
                       c=colors[label], label=label, alpha=0.7, s=20)
    
    plt.title('TSNE Visualization - Grouped by Patch Label', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE1', fontsize=12)
    plt.ylabel('TSNE2', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Label TSNE plot saved to: {output_path}")
    plt.show()

def main():
    """Main function"""
    print("Starting to load feature data...")
    
    # Load all features
    features, hosps, wsi_ids, patch_info = load_all_features()
    
    if features is None:
        print("Unable to load feature data")
        return
    
    # Plot TSNE grouped by hospital
    hosp_output_path = os.path.join(output_dir, "tsne_by_hosp.png")
    plot_tsne_by_hosp(features, hosps, hosp_output_path)
    
    # Plot TSNE grouped by label
    label_output_path = os.path.join(output_dir, "tsne_by_label.png")
    plot_tsne_by_label(features, patch_info, label_output_path)
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total patches: {len(features)}")
    print(f"WSI count: {len(set(wsi_ids))}")
    print(f"Hospital count: {len(set(hosps))}")
    print(f"Hospital distribution:")
    hosp_counts = pd.Series(hosps).value_counts()
    for hosp, count in hosp_counts.items():
        print(f"  {hosp}: {count}")
    
    print(f"\nLabel distribution:")
    label_counts = pd.Series([info['patch_label'] for info in patch_info]).value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main() 
import os
import numpy as np
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import h5py
import openslide

plt.rcParams['font.family'] = 'DejaVu Sans'
random.seed(42)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_fig(fig, filename, show=False, dpi=300):
    save_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Image saved: {save_path}")

def pie_topN(counter, N=10):
    items = counter.most_common(N)
    other_count = sum([v for k, v in counter.items()]) - sum([v for k, v in items])
    labels = [k for k, v in items]
    counts = [v for k, v in items]
    if other_count > 0:
        labels.append('Other')
        counts.append(other_count)
    return labels, counts

class DataExplorer:
    def __init__(self, dataset_type="tcga-brca", label_file=None, wsi_dir=None, patch_coord_dir=None):
        self.dataset_type = dataset_type
        
        # Set default paths based on dataset type
        if label_file is None:
            label_file = f"files/{dataset_type}_label.csv"
        if wsi_dir is None:
            wsi_dir = f"/home/mxz3935/dataset_folder/{dataset_type}/"
        if patch_coord_dir is None:
            patch_coord_dir = f"/raid/mengliang/wsi_process/{dataset_type}_patch/patches/"
            
        self.label_file = label_file
        self.wsi_dir = wsi_dir
        self.patch_coord_dir = patch_coord_dir
        self.df = None
        self.load_data()
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.label_file)
            print(f"Successfully loaded data with {len(self.df)} samples")
            print(f"Data columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            return
    
    def extract_hospital_code(self):
        if self.df is None:
            return
        self.df["hospital_code"] = self.df["filename"].str.extract(r"TCGA-([A-Z0-9]{2})-")
        return self.df["hospital_code"]
    
    def analyze_distribution(self, column_name, title=None, figsize=(8, 6), show=False):
        if self.df is None:
            print("Data not loaded")
            return
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' does not exist")
            return
        counter = Counter(self.df[column_name])
        labels = list(counter.keys())
        counts = list(counter.values())
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(labels)), counts, color='skyblue', alpha=0.7)
        ax.set_xlabel(column_name)
        ax.set_ylabel('Count')
        ax.set_title(f'{title or column_name} Distribution')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=7)
        plt.tight_layout()
        save_fig(fig, f"{self.dataset_type}_{column_name}_distribution.png", show=show)
        print(f"\n{title or column_name} Distribution Statistics:")
        for label, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.df)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    def analyze_hospital_distribution(self, show=False):
        hospital_codes = self.extract_hospital_code()
        if hospital_codes is not None:
            self.analyze_distribution("hospital_code", "Hospital Source Distribution", show=show)
    
    def analyze_demographics(self, show=False):
        demographic_cols = ['gender', 'race', 'ethnicity', 'age']
        for col in demographic_cols:
            if col in self.df.columns:
                if col == 'age':
                    self.analyze_age_distribution(show=show)
                else:
                    self.analyze_distribution(col, f"{col.title()} Distribution", show=show)
    
    def analyze_age_distribution(self, show=False):
        if 'age' not in self.df.columns:
            return
        
        # Clean age data - convert to numeric and remove non-numeric values
        age_data = pd.to_numeric(self.df['age'], errors='coerce').dropna()
        
        if len(age_data) == 0:
            print("\nNo valid numeric age data found.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(age_data, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Age Distribution Histogram')
        axes[0].grid(True, alpha=0.3)
        axes[1].boxplot(age_data)
        axes[1].set_ylabel('Age')
        axes[1].set_title('Age Distribution Boxplot')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, f"{self.dataset_type}_age_distribution.png", show=show)
        
        print(f"\nAge Statistics:")
        print(f"  Valid age samples: {len(age_data)}/{len(self.df)} ({len(age_data)/len(self.df)*100:.1f}%)")
        print(f"  Mean age: {age_data.mean():.1f}")
        print(f"  Median age: {age_data.median():.1f}")
        print(f"  Standard deviation: {age_data.std():.1f}")
        print(f"  Min age: {age_data.min()}")
        print(f"  Max age: {age_data.max()}")
        
        # Show some examples of invalid age data if any
        invalid_ages = self.df[pd.to_numeric(self.df['age'], errors='coerce').isna()]['age'].unique()
        if len(invalid_ages) > 0:
            print(f"  Invalid age values found: {list(invalid_ages)[:5]}{'...' if len(invalid_ages) > 5 else ''}")
    
    def analyze_label_distribution(self, show=False):
        if 'label' in self.df.columns:
            self.analyze_distribution('label', 'Pathology Type Distribution', show=show)
    
    def visualize_random_patches(self, num_patches=16, grid_size=(4, 4), show=False, max_trials=50):
        if self.df is None or len(self.df) == 0:
            print("Label file is empty.")
            return
        found = False
        for _ in range(max_trials):
            row = self.df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
            wsi_name = row['filename']
            wsi_path = os.path.join(self.wsi_dir, f"{wsi_name}.svs")
            coord_path = os.path.join(self.patch_coord_dir, f"{wsi_name}.h5")
            if os.path.isfile(wsi_path) and os.path.isfile(coord_path):
                found = True
                break
        if not found:
            print(f"After {max_trials} trials, no sample with both .svs and .h5 was found.")
            return
        with h5py.File(coord_path, 'r') as f:
            coords = np.array(f['coords'][:])
        if len(coords) < num_patches:
            print(f"This WSI only has {len(coords)} patches, less than requested {num_patches}.")
            num_patches = len(coords)
        sampled_coords = coords if len(coords) <= num_patches else coords[np.random.choice(len(coords), num_patches, replace=False)]
        slide = openslide.OpenSlide(wsi_path)
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        fig.suptitle(f'Random {num_patches} Patches\nFrom: {wsi_name}', fontsize=14)
        for i, coord in enumerate(sampled_coords):
            x, y = map(int, coord)
            patch = slide.read_region((x, y), 0, (256, 256)).convert("RGB")
            row_idx = i // cols
            col_idx = i % cols
            ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
            ax.imshow(patch)
            ax.axis('off')
            ax.set_title(f"{x}_{y}", fontsize=8)
        for j in range(num_patches, rows*cols):
            row_idx = j // cols
            col_idx = j % cols
            ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
            ax.axis('off')
        plt.tight_layout()
        save_fig(fig, f"{self.dataset_type}_random_patches_{wsi_name}.png", show=show)
        slide.close()
        print(f"Patch example saved: {wsi_name}")
    
    def comprehensive_analysis(self, show=False):
        print("=" * 50)
        print(f"{self.dataset_type.upper()} Dataset Comprehensive Analysis")
        print("=" * 50)
        print(f"\nDataset Basic Information:")
        if self.df is not None:
            print(f"  Total samples: {len(self.df)}")
            print(f"  Number of columns: {len(self.df.columns)}")
        self.analyze_hospital_distribution(show=show)
        self.analyze_demographics(show=show)
        self.analyze_label_distribution(show=show)
        self.visualize_random_patches(show=show)
        print("\nAnalysis completed!")

def main(dataset_type="tcga-brca"):
    """
    Main function to run data exploration
    
    Args:
        dataset_type (str): Dataset type, e.g., "tcga-brca" or "tcga-luad"
    """
    print(f"Running data exploration for {dataset_type.upper()} dataset...")
    
    explorer = DataExplorer(dataset_type=dataset_type)
    explorer.comprehensive_analysis(show=False)

def run_tcga_brca():
    """Run analysis for TCGA-BRCA dataset"""
    main("tcga-brca")

def run_tcga_luad():
    """Run analysis for TCGA-LUAD dataset"""
    main("tcga-luad")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        if dataset_type in ["tcga-brca", "tcga-luad"]:
            main(dataset_type)
        else:
            print("Usage: python 01_data_exploration.py [tcga-brca|tcga-luad]")
            print("Available datasets: tcga-brca, tcga-luad")
            print("Using default: tcga-brca")
            main("tcga-brca")
    else:
        # Default to TCGA-BRCA
        main("tcga-brca") 
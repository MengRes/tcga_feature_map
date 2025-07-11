#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import os
import sys
import glob
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

class GCCA:
    """
    Generalized Canonical Correlation Analysis (GCCA)
    用于移除多视图数据中的视图特定变异
    """
    
    def __init__(self, n_components=128):
        """
        初始化GCCA模型
        
        Args:
            n_components (int): 输出特征维度
        """
        self.n_components = n_components
        self.projections = {}
        self.pca_models = {}
        self.pca_views = {}
        
    def fit(self, views):
        """
        训练GCCA模型
        
        Args:
            views (dict): {hospital_name: feature_matrix}
        """
        self.views = views
        self.n_views = len(views)
        
        print(f"Training GCCA with {self.n_views} views...")
        
        # 1. 对每个视图进行PCA降维
        for hospital, data in views.items():
            print(f"  Processing {hospital}: {data.shape}")
            pca = PCA(n_components=min(data.shape[1], data.shape[0]-1))
            pca_data = pca.fit_transform(data)
            self.pca_models[hospital] = pca
            self.pca_views[hospital] = pca_data
        
        # 2. 构建H矩阵 (H = [H1, H2, ..., Hn])
        H_list = []
        for hospital, data in self.pca_views.items():
            # 中心化
            data_centered = data - np.mean(data, axis=0)
            H_list.append(data_centered)
        
        H = np.hstack(H_list)
        print(f"  H matrix shape: {H.shape}")
        
        # 3. 计算SVD: H = U * S * V^T
        U, S, Vt = linalg.svd(H, full_matrices=False)
        print(f"  SVD computed, singular values: {S[:10]}...")
        
        # 4. 选择前n_components个奇异值对应的右奇异向量
        V = Vt[:self.n_components].T
        
        # 5. 计算每个视图的投影矩阵
        start_idx = 0
        for i, (hospital, data) in enumerate(self.pca_views.items()):
            end_idx = start_idx + data.shape[1]
            V_i = V[start_idx:end_idx, :]
            
            # 投影矩阵: W_i = (X_i^T * X_i)^(-1) * X_i^T * U_i
            X_i = data
            U_i = X_i @ V_i
            
            # 使用伪逆避免奇异矩阵问题
            W_i = np.linalg.pinv(X_i.T @ X_i) @ X_i.T @ U_i
            self.projections[hospital] = W_i
            
            start_idx = end_idx
            
        print("GCCA training completed!")
    
    def transform(self, view_name, data):
        """
        转换单个视图的数据
        
        Args:
            view_name (str): 医院名称
            data (np.array): 特征矩阵
            
        Returns:
            np.array: 转换后的特征
        """
        if view_name not in self.projections:
            raise ValueError(f"View {view_name} not found in training data")
        
        # 先进行PCA转换
        pca_data = self.pca_models[view_name].transform(data)
        
        # 然后进行GCCA投影
        transformed = pca_data @ self.projections[view_name]
        
        return transformed
    
    def transform_all(self, data_dict):
        """
        转换所有视图的数据
        
        Args:
            data_dict (dict): {hospital_name: feature_matrix}
            
        Returns:
            dict: {hospital_name: transformed_features}
        """
        transformed_dict = {}
        for hospital, data in data_dict.items():
            transformed_dict[hospital] = self.transform(hospital, data)
        return transformed_dict


def load_pt_features(feature_dir, hospital_info_file=None):
    """
    Load feature data from .pt files directory
    If hospital_info_file is provided, only load filtered patches
    Each row will have wsi_id, patch_x, patch_y for unique patch identification
    """
    print(f"Loading .pt features from {feature_dir}")
    import torch
    import glob
    import os
    import pandas as pd
    
    # Load hospital info if provided
    patch_filter = None
    if hospital_info_file:
        print(f"Loading filtered patch info from {hospital_info_file}")
        patch_filter = pd.read_csv(hospital_info_file)
    
    pt_files = glob.glob(os.path.join(feature_dir, "*.pt"))
    print(f"Found {len(pt_files)} feature files")
    all_features = []
    for pt_file in pt_files:
        wsi_id = os.path.basename(pt_file).split('_features.pt')[0]
        features = torch.load(pt_file, weights_only=True)
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        # If patch_filter is provided, only keep filtered patches
        if patch_filter is not None:
            wsi_patch_info = patch_filter[patch_filter['wsi_id'] == wsi_id]
            if len(wsi_patch_info) == 0:
                continue
            # Assume patch_x, patch_y are in the same order as features
            # If not, must match by coordinates
            if len(wsi_patch_info) != features.shape[0]:
                # Try to match by patch_x, patch_y
                # Assume patch_x, patch_y columns exist in both
                # features: N x D, wsi_patch_info: N x ...
                # We need to align features with patch_x, patch_y
                # For now, assume order is correct
                min_len = min(len(wsi_patch_info), features.shape[0])
                features = features[:min_len]
                wsi_patch_info = wsi_patch_info.iloc[:min_len]
            feature_df = pd.DataFrame(features)
            feature_df['wsi_id'] = wsi_id
            feature_df['patch_x'] = wsi_patch_info['patch_x'].values
            feature_df['patch_y'] = wsi_patch_info['patch_y'].values
            # Optionally copy other columns
            for col in wsi_patch_info.columns:
                if col not in ['wsi_id', 'patch_x', 'patch_y']:
                    feature_df[col] = wsi_patch_info[col].values
            all_features.append(feature_df)
        else:
            feature_df = pd.DataFrame(features)
            feature_df['wsi_id'] = wsi_id
            all_features.append(feature_df)
    if all_features:
        combined_df = pd.concat(all_features, ignore_index=True)
        print(f"Total features loaded: {combined_df.shape}")
        return combined_df
    else:
        print("No features loaded!")
        return pd.DataFrame()


def prepare_hospital_data(feature_source, hospital_info_file):
    """
    准备按医院分组的特征数据
    
    Args:
        feature_source (str): 特征文件路径（CSV）或特征目录路径（.pt文件）
        hospital_info_file (str): 医院信息文件路径
        
    Returns:
        tuple: (hospital_groups, merged_df, feature_columns)
    """
    # 判断是CSV文件还是.pt目录
    if feature_source.endswith('.csv'):
        print(f"Loading features from CSV: {feature_source}")
        features_df = pd.read_csv(feature_source)
    else:
        # 假设是.pt文件目录，传递医院信息文件以只加载过滤后的patch
        features_df = load_pt_features(feature_source, hospital_info_file)
    
    # 数据已经合并完成
    merged_df = features_df
    
    # 识别特征列（排除非特征列）
    exclude_cols = ['wsi_id', 'patch_id', 'hosp', 'gender', 'age', 'diagnosis', 'label', 'patch_label']
    feature_columns = [col for col in merged_df.columns if col not in exclude_cols]
    print(f"Feature columns: {len(feature_columns)}")
    
    # Group by hospital and sample equal number of patches for GCCA
    hospital_groups = {}
    hospital_dfs = {}
    hospital_counts = merged_df['hosp'].value_counts()
    min_count = hospital_counts.min()
    print(f"Sampling {min_count} patches per hospital for GCCA...")
    for hospital in merged_df['hosp'].unique():
        hospital_data = merged_df[merged_df['hosp'] == hospital]
        if len(hospital_data) > min_count:
            hospital_data = hospital_data.sample(n=min_count, random_state=42)
        hospital_groups[hospital] = hospital_data[feature_columns].values
        hospital_dfs[hospital] = hospital_data.reset_index(drop=True)
        print(f"  {hospital}: {hospital_groups[hospital].shape}")
    return hospital_groups, hospital_dfs, feature_columns


def remove_hospital_effects_gcca(feature_source, hospital_info_file, output_file, n_components=128):
    """
    Remove hospital effects from patch features using GCCA.
    Args:
        feature_source (str): Path to feature CSV or .pt directory
        hospital_info_file (str): Path to hospital info CSV
        output_file (str): Output CSV path
        n_components (int): GCCA output dimension
    Returns:
        tuple: (sampled_result_df, gcca_model, all_patches_df, feature_columns)
    """
    print("=== GCCA Hospital Effect Removal ===")
    # 1. Load all patches (for later evaluation)
    if feature_source.endswith('.csv'):
        all_patches_df = pd.read_csv(feature_source)
    else:
        # Load all patches with hospital info
        print("Loading all patches (not sampled)...")
        import torch, glob, os
        hospital_info = pd.read_csv(hospital_info_file)
        pt_files = sorted(glob.glob(os.path.join(feature_source, "*.pt")))
        all_features = []
        for pt_file in pt_files:
            wsi_id = os.path.basename(pt_file).split('_features.pt')[0]
            features = torch.load(pt_file, weights_only=True)
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            wsi_patch_info = hospital_info[hospital_info['wsi_id'] == wsi_id]
            if len(wsi_patch_info) == 0:
                continue
            min_len = min(len(wsi_patch_info), features.shape[0])
            features = features[:min_len]
            wsi_patch_info = wsi_patch_info.iloc[:min_len]
            feature_df = pd.DataFrame(features)
            feature_df['wsi_id'] = wsi_id
            feature_df['patch_x'] = wsi_patch_info['patch_x'].values
            feature_df['patch_y'] = wsi_patch_info['patch_y'].values
            feature_df['hosp'] = wsi_patch_info['hosp'].values
            all_features.append(feature_df)
        all_patches_df = pd.concat(all_features, ignore_index=True)
        print(f"All patches loaded: {all_patches_df.shape}")
    # 2. Identify feature columns
    exclude_cols = ['wsi_id', 'patch_id', 'hosp', 'gender', 'age', 'diagnosis', 'label', 'patch_label', 'patch_x', 'patch_y']
    feature_columns = [col for col in all_patches_df.columns if col not in exclude_cols]
    print(f"Feature columns: {len(feature_columns)}")
    # 3. Sample equal number of patches per hospital for GCCA training
    hospital_counts = all_patches_df['hosp'].value_counts()
    min_count = hospital_counts.min()
    print(f"Sampling {min_count} patches per hospital for GCCA...")
    sampled_groups = {}
    sampled_dfs = {}
    for hospital in sorted(all_patches_df['hosp'].unique()):
        hospital_data = all_patches_df[all_patches_df['hosp'] == hospital]
        if len(hospital_data) > min_count:
            hospital_data = hospital_data.sample(n=min_count, random_state=42)
        sampled_groups[hospital] = hospital_data[feature_columns].values
        sampled_dfs[hospital] = hospital_data.reset_index(drop=True)
        print(f"  {hospital}: {sampled_groups[hospital].shape}")
    # 4. Train GCCA on sampled patches
    print("Using original features without standardization...")
    gcca = GCCA(n_components=n_components)
    gcca.fit(sampled_groups)
    # 5. Transform sampled patches for output (可选)
    print("Transforming sampled features...")
    transformed_groups = gcca.transform_all(sampled_groups)
    transformed_features = []
    for hospital, data in transformed_groups.items():
        hospital_df = sampled_dfs[hospital].copy()
        feature_cols = [f'gcca_feature_{i}' for i in range(data.shape[1])]
        gcca_df = pd.DataFrame(data, columns=feature_cols, index=hospital_df.index)
        hospital_df = pd.concat([hospital_df, gcca_df], axis=1)
        transformed_features.append(hospital_df)
    result_df = pd.concat(transformed_features, ignore_index=True)
    # 6. Save sampled GCCA features (可选)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Saved GCCA-transformed features to {output_file}")
    return result_df, gcca, all_patches_df, feature_columns


def evaluate_hospital_effect_removal(gcca_model, all_patches_df, feature_columns, visualize=False, output_dir="visualizations"):
    """
    Evaluate hospital effect removal on all patches using trained GCCA model.
    Args:
        gcca_model: Trained GCCA model
        all_patches_df (pd.DataFrame): All patch data
        feature_columns (list): Feature column names
        visualize (bool): Whether to generate t-SNE visualizations
        output_dir (str): Output directory for visualizations
    """
    print("=== Evaluating Hospital Effect Removal (on all patches) ===")
    # 1. Group all patches by hospital
    all_hospital_groups = {}
    for hospital in sorted(all_patches_df['hosp'].unique()):
        mask = all_patches_df['hosp'] == hospital
        all_hospital_groups[hospital] = all_patches_df.loc[mask, feature_columns].values
    # 2. Transform all patches using trained GCCA
    all_transformed_groups = gcca_model.transform_all(all_hospital_groups)
    all_gcca_features = []
    all_hospital_labels = []
    all_wsi_ids = []
    for hospital, data in all_transformed_groups.items():
        mask = (all_patches_df['hosp'] == hospital)
        wsi_ids = all_patches_df.loc[mask, 'wsi_id'].values
        all_gcca_features.append(data)
        all_hospital_labels.extend([hospital] * len(data))
        all_wsi_ids.extend(wsi_ids)
    all_gcca_features = np.vstack(all_gcca_features)
    all_hospital_labels = np.array(all_hospital_labels)
    all_wsi_ids = np.array(all_wsi_ids)

    # 3. WSI-level split for hospital prediction
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    le = LabelEncoder()
    y = le.fit_transform(all_hospital_labels)
    wsi_to_hosp = {}
    for wsi, hosp in zip(all_wsi_ids, all_hospital_labels):
        if wsi not in wsi_to_hosp:
            wsi_to_hosp[wsi] = hosp
    wsi_list = np.array(list(wsi_to_hosp.keys()))
    wsi_hosp_labels = np.array(list(wsi_to_hosp.values()))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []
    for wsi_train_idx, wsi_test_idx in skf.split(wsi_list, wsi_hosp_labels):
        train_wsi = wsi_list[wsi_train_idx]
        test_wsi = wsi_list[wsi_test_idx]
        train_mask = np.isin(all_wsi_ids, train_wsi)
        test_mask = np.isin(all_wsi_ids, test_wsi)
        X_train, X_test = all_gcca_features[train_mask], all_gcca_features[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        if y_proba.shape[1] == 2:
            aucs.append(roc_auc_score(y_test, y_proba[:,1]))
        else:
            aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
    print(f"GCCA features (WSI split): Hospital prediction accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}, AUC: {np.mean(aucs):.3f}")

    # 4. Baseline: original features, also WSI-level split
    print("Evaluating original features on all patches (WSI split)...")
    X = all_patches_df[feature_columns].values
    all_wsi_ids_orig = all_patches_df['wsi_id'].values
    all_hospital_labels_orig = all_patches_df['hosp'].values
    y_orig = le.transform(all_hospital_labels_orig)
    wsi_to_hosp_orig = {}
    for wsi, hosp in zip(all_wsi_ids_orig, all_hospital_labels_orig):
        if wsi not in wsi_to_hosp_orig:
            wsi_to_hosp_orig[wsi] = hosp
    wsi_list_orig = np.array(list(wsi_to_hosp_orig.keys()))
    wsi_hosp_labels_orig = np.array(list(wsi_to_hosp_orig.values()))
    accs, aucs = [], []
    for wsi_train_idx, wsi_test_idx in skf.split(wsi_list_orig, wsi_hosp_labels_orig):
        train_wsi = wsi_list_orig[wsi_train_idx]
        test_wsi = wsi_list_orig[wsi_test_idx]
        train_mask = np.isin(all_wsi_ids_orig, train_wsi)
        test_mask = np.isin(all_wsi_ids_orig, test_wsi)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_orig[train_mask], y_orig[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        if y_proba.shape[1] == 2:
            aucs.append(roc_auc_score(y_test, y_proba[:,1]))
        else:
            aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
    print(f"Original features (WSI split): Hospital prediction accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}, AUC: {np.mean(aucs):.3f}")

    # 5. t-SNE visualization if requested
    if visualize:
        create_tsne_visualizations(all_patches_df, all_gcca_features, all_hospital_labels, feature_columns, output_dir)


def create_tsne_visualizations(all_patches_df, gcca_features, hospital_labels, feature_columns, output_dir="visualizations"):
    """
    Create t-SNE visualizations for GCCA features.
    Args:
        all_patches_df (pd.DataFrame): All patch data
        gcca_features (np.array): GCCA transformed features
        hospital_labels (np.array): Hospital labels
        feature_columns (list): Original feature column names
        output_dir (str): Output directory for visualizations
    """
    print("=== Creating t-SNE Visualizations ===")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os
    
    # Create visualizations directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. All patches t-SNE (by hospital)
    print("Computing t-SNE for all patches (by hospital)...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    tsne_coords = tsne.fit_transform(gcca_features)
    
    plt.figure(figsize=(12, 8))
    unique_hospitals = np.unique(hospital_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_hospitals)))
    
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    
    plt.title('GCCA Features - All Patches by Hospital', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gcca_all_patches_by_hospital.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'gcca_all_patches_by_hospital.png')}")
    
    # 2. By disease type (if available)
    if 'diagnosis' in all_patches_df.columns or 'label' in all_patches_df.columns:
        diagnosis_col = 'diagnosis' if 'diagnosis' in all_patches_df.columns else 'label'
        diagnosis_labels = all_patches_df[diagnosis_col].values
        
        print(f"Computing t-SNE for all patches (by {diagnosis_col})...")
        plt.figure(figsize=(12, 8))
        unique_diagnoses = np.unique(diagnosis_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_diagnoses)))
        
        for i, diagnosis in enumerate(unique_diagnoses):
            mask = diagnosis_labels == diagnosis
            plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                       label=diagnosis, alpha=0.7, s=20, c=[colors[i]])
        
        plt.title(f'GCCA Features - All Patches by {diagnosis_col.title()}', fontsize=16)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.legend(title=diagnosis_col.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gcca_all_patches_by_{diagnosis_col}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, f'gcca_all_patches_by_{diagnosis_col}.png')}")
    
    # 3. Original features vs GCCA features comparison
    print("Computing t-SNE for original features comparison...")
    original_features = all_patches_df[feature_columns].values
    tsne_original = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    tsne_coords_original = tsne_original.fit_transform(original_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original features
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        ax1.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    ax1.set_title('Original Features by Hospital', fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # GCCA features
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        ax2.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    ax2.set_title('GCCA Features by Hospital', fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gcca_vs_original_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'gcca_vs_original_features.png')}")
    
    print("=== t-SNE Visualizations Completed ===")


def save_gcca_features_by_wsi(result_df, feature_columns, output_dir):
    """
    Save GCCA features for all patches, grouped by WSI, as .pt files in output_dir.
    Each file: <wsi_id>_features.pt
    """
    import torch
    os.makedirs(output_dir, exist_ok=True)
    for wsi_id, group in result_df.groupby('wsi_id'):
        features = group[feature_columns].values.astype('float32')
        # Convert numpy array to PyTorch tensor for proper saving
        features_tensor = torch.from_numpy(features)
        torch.save(features_tensor, os.path.join(output_dir, f'{wsi_id}_features.pt'))
    print(f"Saved GCCA features for all WSIs to {output_dir}")


def main():
    import argparse, sys, os
    parser = argparse.ArgumentParser(description='Remove hospital effects from WSI patch features using GCCA')
    parser.add_argument('--feature_source', type=str, required=True,
                       help='Path to feature CSV file or directory containing .pt feature files')
    parser.add_argument('--hospital_info', type=str, required=True,
                       help='Path to hospital information CSV file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output path for GCCA-processed features')
    parser.add_argument('--n_components', type=int, default=128,
                       help='Number of GCCA components (default: 128)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate hospital effect removal')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate t-SNE visualizations')
    args = parser.parse_args()
    result_df, gcca_model, all_patches_df, feature_columns = remove_hospital_effects_gcca(
        feature_source=args.feature_source,
        hospital_info_file=args.hospital_info,
        output_file=args.output_file,
        n_components=args.n_components
    )
    # 新增：保存GCCA特征为.pt文件
    # 目标目录与输入features_*目录同级，名为features_GCCA
    if os.path.isdir(args.feature_source):
        parent_dir = os.path.dirname(args.feature_source.rstrip('/'))
        gcca_feat_dir = os.path.join(parent_dir, 'features_GCCA')
        gcca_feature_cols = [col for col in result_df.columns if isinstance(col, str) and col.startswith('gcca_feature_')]
        save_gcca_features_by_wsi(result_df, gcca_feature_cols, gcca_feat_dir)
    if args.evaluate:
        # 设置可视化输出目录为特征源目录下的visualizations文件夹
        if os.path.isdir(args.feature_source):
            parent_dir = os.path.dirname(args.feature_source.rstrip('/'))
            viz_dir = os.path.join(parent_dir, 'visualizations')
        else:
            viz_dir = "visualizations"
        evaluate_hospital_effect_removal(gcca_model, all_patches_df, feature_columns, args.visualize, viz_dir)
    print("=== GCCA processing completed successfully! ===")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Cross-validation evaluation of original and adversarial UNI features
1. 5-fold CV for hospital/disease classification using original UNI features
2. 5-fold CV for hospital/disease classification using adversarial features
All outputs and comments are in English.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')  # 保证可import adversarial_deconfounding.py中的类
from adversarial_deconfounding import AdversarialProjectionModel, PatchDataset, train_epoch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_features_and_metadata(features_dir, patches_csv_path):
    """
    Load features and metadata
    """
    print(f"Loading features from {features_dir}")
    patches_df = pd.read_csv(patches_csv_path)
    def extract_hosp_from_filename(filename):
        parts = filename.split('-')
        if len(parts) >= 3:
            return parts[1]
        return "Unknown"
    patches_df['hosp'] = patches_df['wsi_id'].apply(extract_hosp_from_filename)
    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith('_features.pt')])
    if len(feature_files) == 0:
        raise ValueError(f"No feature files found in {features_dir}")
    all_features = []
    all_metadata = []
    for feature_file in tqdm(feature_files, desc="Loading features"):
        try:
            features = torch.load(os.path.join(features_dir, feature_file), weights_only=True)
            wsi_id = feature_file.replace("_features.pt", "")
            wsi_patches = patches_df[patches_df['wsi_id'] == wsi_id]
            if len(wsi_patches) == 0:
                continue
            min_count = min(len(features), len(wsi_patches))
            all_features.append(features[:min_count].numpy())
            all_metadata.append(wsi_patches.iloc[:min_count])
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
            continue
    if not all_features:
        raise ValueError("No valid features loaded")
    features_array = np.vstack(all_features)
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    print(f"Loaded {len(features_array)} samples with {features_array.shape[1]} dimensions")
    return features_array, metadata_df

def cross_val_classification_wsi_level(features, metadata_df, label_col, n_splits=5, random_state=42):
    """
    5-fold cross-validation for classification (LogisticRegression), WSI-level split
    features: all patch features (N, D)
    metadata_df: DataFrame with columns ['wsi_id', label_col]
    label_col: 'hosp' or 'patch_label'
    """
    # 1. Get WSI-level label
    wsi_ids = np.asarray(metadata_df['wsi_id'].values)
    wsi_to_label = metadata_df.groupby('wsi_id')[label_col].first().to_dict()
    unique_wsi = np.asarray(list(wsi_to_label.keys()))
    wsi_labels = np.asarray(list(wsi_to_label.values()))
    le = LabelEncoder()
    wsi_labels_encoded = le.fit_transform(wsi_labels)
    # 2. StratifiedKFold on WSI
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, aucs = [], []
    for train_wsi_idx, test_wsi_idx in skf.split(unique_wsi, wsi_labels_encoded):
        train_wsi = unique_wsi[train_wsi_idx]
        test_wsi = unique_wsi[test_wsi_idx]
        train_mask = np.isin(wsi_ids, train_wsi)
        test_mask = np.isin(wsi_ids, test_wsi)
        X_train, X_test = features[train_mask], features[test_mask]
        y_train, y_test = np.asarray(metadata_df[label_col].values[train_mask]), np.asarray(metadata_df[label_col].values[test_mask])
        le_fold = LabelEncoder()
        y_train_enc = le_fold.fit_transform(y_train)
        y_test_enc = le_fold.transform(y_test)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train_enc)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        accs.append(acc)
        # For AUC, need one-vs-rest for multiclass
        if len(np.unique(y_train_enc)) > 2:
            y_score = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test_enc, y_score, multi_class='ovr')
        else:
            y_score = clf.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test_enc, y_score)
        aucs.append(auc)
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)

def cross_val_adversarial_wsi_level(features, metadata_df, le_disease, le_hosp, n_splits=5, random_state=42, num_epochs=20, batch_size=32, learning_rate=1e-3, lambda_adv=1.0, grl_alpha=1.0, proj_dim=512, nonlinear=True):
    """
    5-fold WSI-level CV: 每fold内训练adversarial模型并提取特征，仅用val特征做分类评估
    返回hosp/disease分类的acc/auc均值±std
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    wsi_ids = np.asarray(metadata_df['wsi_id'].values)
    wsi_to_disease = metadata_df.groupby('wsi_id')['patch_label'].first().to_dict()
    unique_wsi = np.asarray(list(wsi_to_disease.keys()))
    wsi_disease_labels = np.asarray(list(wsi_to_disease.values()))  # 原始字符串label
    wsi_disease_encoded = le_disease.transform(wsi_disease_labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs_hosp, aucs_hosp, accs_dis, aucs_dis = [], [], [], []
    for fold, (train_wsi_idx, val_wsi_idx) in enumerate(skf.split(unique_wsi, wsi_disease_encoded)):
        train_wsi = unique_wsi[train_wsi_idx]
        val_wsi = unique_wsi[val_wsi_idx]
        train_mask = np.isin(wsi_ids, train_wsi)
        val_mask = np.isin(wsi_ids, val_wsi)
        train_features = features[train_mask]
        val_features = features[val_mask]
        train_meta = metadata_df.iloc[train_mask].reset_index(drop=True)
        val_meta = metadata_df.iloc[val_mask].reset_index(drop=True)
        train_disease_labels = le_disease.transform(np.asarray(train_meta['patch_label']))
        train_hospital_labels = le_hosp.transform(np.asarray(train_meta['hosp']))
        val_disease_labels = le_disease.transform(np.asarray(val_meta['patch_label']))
        val_hospital_labels = le_hosp.transform(np.asarray(val_meta['hosp']))
        train_dataset = PatchDataset(train_features, train_disease_labels, train_hospital_labels, np.asarray(train_meta['wsi_id'].values))
        val_dataset = PatchDataset(val_features, val_disease_labels, val_hospital_labels, np.asarray(val_meta['wsi_id'].values))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdversarialProjectionModel(
            feature_dim=features.shape[1],
            proj_dim=proj_dim,
            num_diseases=len(le_disease.classes_),
            num_hospitals=len(le_hosp.classes_),
            grl_alpha=grl_alpha,
            nonlinear=nonlinear
        ).to(device)
        disease_criterion = nn.CrossEntropyLoss()
        hospital_criterion = nn.CrossEntropyLoss()
        projection_optimizer = Adam(model.projection.parameters(), lr=learning_rate)
        hospital_optimizer = Adam(model.hospital_classifier.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            train_epoch(model, train_loader, disease_criterion, hospital_criterion, projection_optimizer, hospital_optimizer, device, lambda_adv)
        model.eval()
        adv_val_features = []
        with torch.no_grad():
            for batch in val_loader:
                features_batch = batch['features'].to(device)
                proj_features = model.projection(features_batch)
                adv_val_features.append(proj_features.cpu().numpy())
        adv_val_features = np.vstack(adv_val_features)
        clf_hosp = LogisticRegression(max_iter=1000)
        y_hosp = np.asarray(val_hospital_labels)
        clf_hosp.fit(adv_val_features, y_hosp)
        y_pred = clf_hosp.predict(adv_val_features)
        acc_hosp = accuracy_score(y_hosp, y_pred)
        if len(np.unique(y_hosp)) > 2:
            y_score = clf_hosp.predict_proba(adv_val_features)
            auc_hosp = roc_auc_score(y_hosp, y_score, multi_class='ovr')
        else:
            y_score = clf_hosp.predict_proba(adv_val_features)[:,1]
            auc_hosp = roc_auc_score(y_hosp, y_score)
        accs_hosp.append(acc_hosp)
        aucs_hosp.append(auc_hosp)
        clf_dis = LogisticRegression(max_iter=1000)
        y_dis = np.asarray(val_disease_labels)
        clf_dis.fit(adv_val_features, y_dis)
        y_pred = clf_dis.predict(adv_val_features)
        acc_dis = accuracy_score(y_dis, y_pred)
        if len(np.unique(y_dis)) > 2:
            y_score = clf_dis.predict_proba(adv_val_features)
            auc_dis = roc_auc_score(y_dis, y_score, multi_class='ovr')
        else:
            y_score = clf_dis.predict_proba(adv_val_features)[:,1]
            auc_dis = roc_auc_score(y_dis, y_score)
        accs_dis.append(acc_dis)
        aucs_dis.append(auc_dis)
        print(f"Fold {fold+1}: Hosp ACC={acc_hosp:.4f}, AUC={auc_hosp:.4f} | Disease ACC={acc_dis:.4f}, AUC={auc_dis:.4f}")
    return (np.mean(accs_hosp), np.std(accs_hosp), np.mean(aucs_hosp), np.std(aucs_hosp),
            np.mean(accs_dis), np.std(accs_dis), np.mean(aucs_dis), np.std(aucs_dis))

def plot_tsne(features, labels, label_names, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10,8))
    for l in np.unique(labels):
        mask = labels == l
        plt.scatter(features_2d[mask,0], features_2d[mask,1], label=l, alpha=0.7, s=20)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='5-fold CV for hospital/disease classification using UNI features (original & adversarial)')
    parser.add_argument('--features_dir', type=str, required=True, help='Directory containing *_features.pt')
    parser.add_argument('--patches_csv', type=str, required=True, help='CSV file with patch metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load original features
    features, metadata_df = load_features_and_metadata(args.features_dir, args.patches_csv)
    # Encode labels for reporting
    le_hosp = LabelEncoder()
    le_disease = LabelEncoder()
    le_hosp.fit(metadata_df['hosp'])
    le_disease.fit(metadata_df['patch_label'])
    # 1. 原始UNI特征 hosp 分类
    print("\n[Original UNI features] 5-fold CV (WSI-level): Hospital classification")
    acc_hosp, acc_hosp_std, auc_hosp, auc_hosp_std = cross_val_classification_wsi_level(features, metadata_df, 'hosp')
    print(f"Hospital ACC: {acc_hosp:.4f} ± {acc_hosp_std:.4f}, AUC: {auc_hosp:.4f} ± {auc_hosp_std:.4f}")
    # 2. 原始UNI特征 disease 分类
    print("\n[Original UNI features] 5-fold CV (WSI-level): Disease classification")
    acc_dis, acc_dis_std, auc_dis, auc_dis_std = cross_val_classification_wsi_level(features, metadata_df, 'patch_label')
    print(f"Disease ACC: {acc_dis:.4f} ± {acc_dis_std:.4f}, AUC: {auc_dis:.4f} ± {auc_dis_std:.4f}")
    # 3/4. adversarial特征评估（每fold内独立训练、提取、评估）
    lambda_adv = 0.01
    print(f"\n[Adversarial features] 5-fold CV (WSI-level): Hospital & Disease classification (strict, per-fold training, lambda_adv={lambda_adv})")
    adv_acc_hosp, adv_acc_hosp_std, adv_auc_hosp, adv_auc_hosp_std, adv_acc_dis, adv_acc_dis_std, adv_auc_dis, adv_auc_dis_std = cross_val_adversarial_wsi_level(features, metadata_df, le_disease, le_hosp, n_splits=5, random_state=42, num_epochs=20, batch_size=32, learning_rate=1e-3, lambda_adv=lambda_adv, grl_alpha=1.0, proj_dim=512, nonlinear=True)
    print(f"[Adversarial] Hospital ACC: {adv_acc_hosp:.4f} ± {adv_acc_hosp_std:.4f}, AUC: {adv_auc_hosp:.4f} ± {adv_auc_hosp_std:.4f}")
    print(f"[Adversarial] Disease ACC: {adv_acc_dis:.4f} ± {adv_acc_dis_std:.4f}, AUC: {adv_auc_dis:.4f} ± {adv_auc_dis_std:.4f}")
    # Save results
    with open(os.path.join(args.output_dir, 'cv_results.txt'), 'w') as f:
        f.write('[Original UNI features] 5-fold CV (WSI-level): Hospital classification\n')
        f.write(f'Hospital ACC: {acc_hosp:.4f} ± {acc_hosp_std:.4f}, AUC: {auc_hosp:.4f} ± {auc_hosp_std:.4f}\n')
        f.write('[Original UNI features] 5-fold CV (WSI-level): Disease classification\n')
        f.write(f'Disease ACC: {acc_dis:.4f} ± {acc_dis_std:.4f}, AUC: {auc_dis:.4f} ± {auc_dis_std:.4f}\n')
        f.write(f'[Adversarial features] 5-fold CV (WSI-level): Hospital classification (strict, per-fold training, lambda_adv={lambda_adv})\n')
        f.write(f'Hospital ACC: {adv_acc_hosp:.4f} ± {adv_acc_hosp_std:.4f}, AUC: {adv_auc_hosp:.4f} ± {adv_auc_hosp_std:.4f}\n')
        f.write(f'[Adversarial features] 5-fold CV (WSI-level): Disease classification (strict, per-fold training, lambda_adv={lambda_adv})\n')
        f.write(f'Disease ACC: {adv_acc_dis:.4f} ± {adv_acc_dis_std:.4f}, AUC: {adv_auc_dis:.4f} ± {adv_auc_dis_std:.4f}\n')
    print(f"\nResults saved to {os.path.join(args.output_dir, 'cv_results.txt')}")

    # 原始UNI特征 t-SNE
    print("\nPlotting t-SNE for original UNI features...")
    plot_tsne(features, metadata_df['hosp'].values, le_hosp.classes_, 'Original UNI features (colored by hospital)', os.path.join(args.output_dir, 'tsne_uni_by_hosp.png'))
    plot_tsne(features, metadata_df['patch_label'].values, le_disease.classes_, 'Original UNI features (colored by disease)', os.path.join(args.output_dir, 'tsne_uni_by_disease.png'))
    # adversarial特征 t-SNE（取全部fold的val投影特征拼接）
    print("\nPlotting t-SNE for adversarial projection features...")
    adv_proj_features = []
    adv_proj_hosp = []
    adv_proj_disease = []
    # 重新跑一遍CV，收集所有val的投影特征和标签
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    wsi_ids = np.asarray(metadata_df['wsi_id'].values)
    wsi_to_disease = metadata_df.groupby('wsi_id')['patch_label'].first().to_dict()
    unique_wsi = np.asarray(list(wsi_to_disease.keys()))
    wsi_disease_labels = np.asarray(list(wsi_to_disease.values()))  # 原始字符串label
    wsi_disease_encoded = le_disease.transform(wsi_disease_labels)
    for fold, (train_wsi_idx, val_wsi_idx) in enumerate(skf.split(unique_wsi, wsi_disease_encoded)):
        train_wsi = unique_wsi[train_wsi_idx]
        val_wsi = unique_wsi[val_wsi_idx]
        train_mask = np.isin(wsi_ids, train_wsi)
        val_mask = np.isin(wsi_ids, val_wsi)
        train_features = features[train_mask]
        val_features = features[val_mask]
        train_meta = metadata_df.iloc[train_mask].reset_index(drop=True)
        val_meta = metadata_df.iloc[val_mask].reset_index(drop=True)
        train_disease_labels = le_disease.transform(np.asarray(train_meta['patch_label']))
        train_hospital_labels = le_hosp.transform(np.asarray(train_meta['hosp']))
        train_dataset = PatchDataset(train_features, train_disease_labels, train_hospital_labels, np.asarray(train_meta['wsi_id'].values))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdversarialProjectionModel(
            feature_dim=features.shape[1],
            proj_dim=512,
            num_diseases=len(le_disease.classes_),
            num_hospitals=len(le_hosp.classes_),
            grl_alpha=1.0,
            nonlinear=True
        ).to(device)
        disease_criterion = nn.CrossEntropyLoss()
        hospital_criterion = nn.CrossEntropyLoss()
        projection_optimizer = Adam(model.projection.parameters(), lr=1e-3)
        hospital_optimizer = Adam(model.hospital_classifier.parameters(), lr=1e-3)
        for epoch in range(20):
            train_epoch(model, train_loader, disease_criterion, hospital_criterion, projection_optimizer, hospital_optimizer, device, lambda_adv)
        # 提取val投影特征
        val_dataset = PatchDataset(val_features, le_disease.transform(np.asarray(val_meta['patch_label'])), le_hosp.transform(np.asarray(val_meta['hosp'])), np.asarray(val_meta['wsi_id'].values))
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch in val_loader:
                features_batch = batch['features'].to(device)
                proj_features = model.projection(features_batch)
                adv_proj_features.append(proj_features.cpu().numpy())
                adv_proj_hosp.append(batch['hospital_label'].cpu().numpy())
                adv_proj_disease.append(batch['disease_label'].cpu().numpy())
    adv_proj_features = np.vstack(adv_proj_features)
    adv_proj_hosp = np.concatenate(adv_proj_hosp)
    adv_proj_disease = np.concatenate(adv_proj_disease)
    plot_tsne(adv_proj_features, adv_proj_hosp, le_hosp.classes_, 'Adversarial projection features (by hospital)', os.path.join(args.output_dir, 'tsne_advproj_by_hosp.png'))
    plot_tsne(adv_proj_features, adv_proj_disease, le_disease.classes_, 'Adversarial projection features (by disease)', os.path.join(args.output_dir, 'tsne_advproj_by_disease.png'))

if __name__ == "__main__":
    main() 
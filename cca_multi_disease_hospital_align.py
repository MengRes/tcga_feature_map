import os
import torch
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import umap.umap_ as umap

META_CSV = 'results/wsi_patch_filter_20250628_163736/conch_filtered_patches.csv'
FEATURE_DIR = 'results/wsi_patch_filter_20250628_163736/features_UNI'
SAVE_DIR = 'results/wsi_patch_filter_20250628_163736/features_UNI_CCA_multi'
os.makedirs(SAVE_DIR, exist_ok=True)

meta = pd.read_csv(META_CSV)

all_features_cca = []
all_labels = []
all_hosps = []
all_wsi_ids = []

for disease in meta['label'].unique():
    meta_d = meta[meta['label'] == disease]
    hosp_list = meta_d['hosp'].unique()
    if len(hosp_list) < 2:
        continue  # skip if only one hospital

    # Choose the largest hosp as reference
    ref_hosp = meta_d['hosp'].value_counts().index[0]
    ref_meta = meta_d[meta_d['hosp'] == ref_hosp]
    ref_feats_list = []
    ref_idx_list = []
    for wsi_id in ref_meta['wsi_id'].unique():
        feature_path = os.path.join(FEATURE_DIR, f"{wsi_id}_features.pt")
        if os.path.exists(feature_path):
            feats = torch.load(feature_path, weights_only=True)
            mask = ref_meta['wsi_id'] == wsi_id
            n = mask.sum()
            ref_feats_list.append(feats[:n])
            ref_idx_list.extend(ref_meta.index[mask].tolist())
    ref_feats = np.vstack(ref_feats_list)
    min_n = len(ref_feats)

    # Collect all hosp features, align to ref_hosp
    disease_feats_cca = []
    disease_labels = []
    disease_hosps = []
    disease_wsi_ids = []

    for hosp in hosp_list:
        meta_h = meta_d[meta_d['hosp'] == hosp]
        feats_list = []
        idx_list = []
        for wsi_id in meta_h['wsi_id'].unique():
            feature_path = os.path.join(FEATURE_DIR, f"{wsi_id}_features.pt")
            if os.path.exists(feature_path):
                feats = torch.load(feature_path, weights_only=True)
                mask = meta_h['wsi_id'] == wsi_id
                n = mask.sum()
                feats_list.append(feats[:n])
                idx_list.extend(meta_h.index[mask].tolist())
        feats = np.vstack(feats_list)
        n_h = len(feats)
        n_cca = min(min_n, n_h)
        if n_cca < 2:
            continue
        # CCA between ref_hosp and this hosp
        cca_dim = min(20, feats.shape[1], ref_feats.shape[1], n_cca)
        cca = CCA(n_components=cca_dim)
        X_c, Y_c = cca.fit_transform(ref_feats[:n_cca], feats[:n_cca])
        if hosp == ref_hosp:
            disease_feats_cca.append(X_c)
        else:
            disease_feats_cca.append(Y_c)
        disease_labels.extend([disease]*n_cca)
        disease_hosps.extend([hosp]*n_cca)
        disease_wsi_ids.extend([meta_h['wsi_id'].unique()[0]]*n_cca)  # for demo, assign by wsi

    # Merge all CCA features for this disease
    all_features_cca.append(np.vstack(disease_feats_cca))
    all_labels.extend(disease_labels)
    all_hosps.extend(disease_hosps)
    all_wsi_ids.extend(disease_wsi_ids)

# Concatenate all diseases' CCA features
features_cca = np.vstack(all_features_cca)
labels = np.array(all_labels)
hosps = np.array(all_hosps)
wsi_ids = np.array(all_wsi_ids)

# Save by wsi_id
for wsi_id in np.unique(wsi_ids):
    idx = wsi_ids == wsi_id
    torch.save(torch.from_numpy(features_cca[idx]), os.path.join(SAVE_DIR, f'{wsi_id}_features.pt'))

print(f"Multi-disease, multi-hospital CCA alignment finished. Features saved to {SAVE_DIR}")

# Prepare data for visualization and classification
features_cca = np.vstack(all_features_cca)
hosps = np.array(all_hosps)
labels = np.array(all_labels)

# For raw features, collect all disease/hosp original features (min_n per disease/hosp)
all_features_raw = []
all_hosps_raw = []
for disease in meta['label'].unique():
    meta_d = meta[meta['label'] == disease]
    hosp_list = meta_d['hosp'].unique()
    if len(hosp_list) < 2:
        continue
    ref_hosp = meta_d['hosp'].value_counts().index[0]
    min_n = None
    feats_by_hosp = {}
    for hosp in hosp_list:
        meta_h = meta_d[meta_d['hosp'] == hosp]
        feats_list = []
        for wsi_id in meta_h['wsi_id'].unique():
            feature_path = os.path.join(FEATURE_DIR, f"{wsi_id}_features.pt")
            if os.path.exists(feature_path):
                feats = torch.load(feature_path, weights_only=True)
                mask = meta_h['wsi_id'] == wsi_id
                n = mask.sum()
                feats_list.append(feats[:n])
        if feats_list:
            feats = np.vstack(feats_list)
            feats_by_hosp[hosp] = feats
            if min_n is None or len(feats) < min_n:
                min_n = len(feats)
    for hosp in hosp_list:
        if hosp in feats_by_hosp and min_n >= 2:
            all_features_raw.append(feats_by_hosp[hosp][:min_n])
            all_hosps_raw.extend([hosp]*min_n)
if all_features_raw:
    features_raw = np.vstack(all_features_raw)
    hosps_raw = np.array(all_hosps_raw)
else:
    features_raw = np.empty((0, features_cca.shape[1]))
    hosps_raw = np.array([])

# t-SNE before CCA
if len(features_raw) > 0:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features_raw)
    plt.figure(figsize=(10,8))
    for h in np.unique(hosps_raw):
        mask = hosps_raw == h
        plt.scatter(features_tsne[mask,0], features_tsne[mask,1], label=h, alpha=0.7, s=20)
    plt.title('t-SNE of Original Features (All Diseases)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tsne_multi_disease_raw.png', dpi=200)
    plt.close()
    print('Saved: tsne_multi_disease_raw.png')

# t-SNE after CCA
if len(features_cca) > 0:
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30)
    features_cca_tsne = tsne2.fit_transform(features_cca)
    plt.figure(figsize=(10,8))
    for h in np.unique(hosps):
        mask = hosps == h
        plt.scatter(features_cca_tsne[mask,0], features_cca_tsne[mask,1], label=h, alpha=0.7, s=20)
    plt.title('t-SNE of CCA-aligned Features (All Diseases)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tsne_multi_disease_cca.png', dpi=200)
    plt.close()
    print('Saved: tsne_multi_disease_cca.png')

# UMAP before CCA
if len(features_raw) > 0:
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    features_umap = umap_reducer.fit_transform(features_raw)
    plt.figure(figsize=(10,8))
    for h in np.unique(hosps_raw):
        mask = hosps_raw == h
        plt.scatter(features_umap[mask,0], features_umap[mask,1], label=h, alpha=0.7, s=20)
    plt.title('UMAP of Original Features (All Diseases)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('umap_multi_disease_raw.png', dpi=200)
    plt.close()
    print('Saved: umap_multi_disease_raw.png')

# UMAP after CCA
if len(features_cca) > 0:
    umap_reducer2 = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    features_cca_umap = umap_reducer2.fit_transform(features_cca)
    plt.figure(figsize=(10,8))
    for h in np.unique(hosps):
        mask = hosps == h
        plt.scatter(features_cca_umap[mask,0], features_cca_umap[mask,1], label=h, alpha=0.7, s=20)
    plt.title('UMAP of CCA-aligned Features (All Diseases)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('umap_multi_disease_cca.png', dpi=200)
    plt.close()
    print('Saved: umap_multi_disease_cca.png')

# Classification: hosp prediction before CCA
if len(features_raw) > 0:
    clf_raw = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(features_raw, hosps_raw, test_size=0.2, random_state=42, stratify=hosps_raw)
    clf_raw.fit(X_train, y_train)
    y_pred = clf_raw.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Before CCA, Logistic Regression hospital classification accuracy: {acc:.4f}')

# Classification: hosp prediction after CCA
if len(features_cca) > 0:
    clf_cca = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(features_cca, hosps, test_size=0.2, random_state=42, stratify=hosps)
    clf_cca.fit(X_train, y_train)
    y_pred = clf_cca.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'After CCA, Logistic Regression hospital classification accuracy: {acc:.4f}')

# Disease classification before CCA
if len(features_raw) > 0:
    all_labels_raw = []
    for disease in meta['label'].unique():
        meta_d = meta[meta['label'] == disease]
        hosp_list = meta_d['hosp'].unique()
        if len(hosp_list) < 2:
            continue
        ref_hosp = meta_d['hosp'].value_counts().index[0]
        min_n = None
        feats_by_hosp = {}
        for hosp in hosp_list:
            meta_h = meta_d[meta_d['hosp'] == hosp]
            feats_list = []
            for wsi_id in meta_h['wsi_id'].unique():
                feature_path = os.path.join(FEATURE_DIR, f"{wsi_id}_features.pt")
                if os.path.exists(feature_path):
                    feats = torch.load(feature_path, weights_only=True)
                    mask = meta_h['wsi_id'] == wsi_id
                    n = mask.sum()
                    feats_list.append(feats[:n])
            if feats_list:
                feats = np.vstack(feats_list)
                feats_by_hosp[hosp] = feats
                if min_n is None or len(feats) < min_n:
                    min_n = len(feats)
        for hosp in hosp_list:
            if hosp in feats_by_hosp and min_n >= 2:
                all_labels_raw.extend([disease]*min_n)
    all_labels_raw = np.array(all_labels_raw)
    clf_raw = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(features_raw, all_labels_raw, test_size=0.2, random_state=42, stratify=all_labels_raw)
    clf_raw.fit(X_train, y_train)
    y_pred = clf_raw.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Before CCA, Logistic Regression disease classification accuracy: {acc:.4f}')

# Disease classification after CCA
if len(features_cca) > 0:
    clf_cca = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(features_cca, labels, test_size=0.2, random_state=42, stratify=labels)
    clf_cca.fit(X_train, y_train)
    y_pred = clf_cca.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'After CCA, Logistic Regression disease classification accuracy: {acc:.4f}') 
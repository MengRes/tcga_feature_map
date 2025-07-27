import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from dtaidistance import dtw
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ========== 配置 ==========
META_CSV = 'results/wsi_patch_filter_20250628_163736/conch_filtered_patches.csv'
FEATURE_DIR = 'results/wsi_patch_filter_20250628_163736/features_UNI'
SAVE_DIR = 'results/wsi_patch_filter_20250628_163736/features_UNI_CCA_DTW'
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 加载元信息和特征 ==========
meta = pd.read_csv(META_CSV)
hosp_list = meta['hosp'].unique()
hosp_features = {}
hosp_wsi_ids = {}
for hosp in hosp_list:
    meta_h = meta[meta['hosp'] == hosp]
    feats_list = []
    wsi_ids = []
    for wsi_id in meta_h['wsi_id'].unique():
        feature_path = os.path.join(FEATURE_DIR, f"{wsi_id}_features.pt")
        if os.path.exists(feature_path):
            feats = torch.load(feature_path, weights_only=True)
            mask = meta_h['wsi_id'] == wsi_id
            n = mask.sum()
            feats_list.append(feats[:n])
            wsi_ids.extend([wsi_id]*n)
    if feats_list:
        hosp_features[hosp] = np.vstack(feats_list)
        hosp_wsi_ids[hosp] = np.array(wsi_ids)

# ========== 构建patch级别数据 ==========
all_feats_raw = []
for hosp in hosp_list:
    arr = hosp_features[hosp]
    all_feats_raw.append(arr)
all_feats_raw = np.vstack(all_feats_raw)
all_hosps_raw = meta['hosp'].values
all_diseases_raw = meta['patch_label'].values
wsi_id_arr = meta['wsi_id'].values

# ========== 构建WSI->disease/hosp映射及分层标签 ==========
wsi_to_disease = {}
wsi_to_hosp = {}
for wsi, d, h in zip(wsi_id_arr, all_diseases_raw, all_hosps_raw):
    if wsi not in wsi_to_disease:
        wsi_to_disease[wsi] = d
    if wsi not in wsi_to_hosp:
        wsi_to_hosp[wsi] = h
unique_wsi = np.array(list(wsi_to_disease.keys()))
wsi_disease_labels = np.array(list(wsi_to_disease.values()))
wsi_hosp_labels = np.array(list(wsi_to_hosp.values()))
le_disease = LabelEncoder()
le_hosp = LabelEncoder()
le_disease.fit(wsi_disease_labels)
le_hosp.fit(wsi_hosp_labels)
wsi_disease_labels_encoded = le_disease.transform(wsi_disease_labels)
wsi_hosp_labels_encoded = le_hosp.transform(wsi_hosp_labels)

# ========== 动态调整n_splits ==========
wsi_disease_counts = Counter(wsi_disease_labels)
min_disease_wsi_count = min(wsi_disease_counts.values())
n_splits = min(5, min_disease_wsi_count)
if n_splits < 2:
    print("每个disease类别的WSI数量太少，无法做交叉验证。")
    exit()

# ========== 分类评估函数 ==========
def eval_clf(X, y, wsi_id_arr, unique_wsi, stratify_labels, label_name):
    accs = []
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_wsi_idx, test_wsi_idx) in enumerate(skf.split(unique_wsi, stratify_labels)):
        train_wsi = unique_wsi[train_wsi_idx]
        test_wsi = unique_wsi[test_wsi_idx]
        train_mask = np.isin(wsi_id_arr, train_wsi)
        test_mask = np.isin(wsi_id_arr, test_wsi)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_enc[train_mask], y_enc[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[{label_name}] Fold {fold+1} skipped: only one class in train or test.")
            continue
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
    if len(accs) == 0:
        print(f"{label_name}: No valid folds (all folds have only one class).")
        return
    print(f'{label_name}: ACC={np.mean(accs):.4f}±{np.std(accs):.4f}')

# ========== CCA对齐 ==========
ref_hosp = meta['hosp'].value_counts().index[0]
ref_feats = hosp_features[ref_hosp]
min_n = min([len(f) for f in hosp_features.values()])
cca_dim = min(5, ref_feats.shape[1], min_n)
cca = CCA(n_components=cca_dim)
cca_proj = {}
for hosp, feats in hosp_features.items():
    n_cca = min(len(ref_feats), len(feats))
    if hosp == ref_hosp:
        X_c, _ = cca.fit_transform(ref_feats[:n_cca], ref_feats[:n_cca])
        cca_proj[hosp] = X_c
    else:
        _, Y_c = cca.fit_transform(ref_feats[:n_cca], feats[:n_cca])
        cca_proj[hosp] = Y_c

# ========== CCA+DTW分布对齐 ==========
all_feats_cca = []
all_hosps_cca = []
all_diseases_cca = []
all_feats_dtw = []
all_hosps_dtw = []
all_diseases_dtw = []
for hosp in hosp_list:
    arr = cca_proj[hosp]
    all_feats_cca.append(arr)
    all_hosps_cca.extend([hosp]*len(arr))
    all_diseases_cca.extend(meta[meta['hosp'] == hosp]['patch_label'].values)
    n = arr.shape[0]
    arr_dtw = np.zeros_like(arr)
    for axis in range(arr.shape[1]):
        arr_sorted = np.sort(arr[:, axis])
        ref_curve = np.sort(cca_proj[ref_hosp][:, axis])
        if hosp == ref_hosp:
            arr_dtw[:, axis] = ref_curve
        else:
            path = dtw.warping_path(ref_curve, arr_sorted)
            aligned = np.zeros_like(ref_curve)
            for idx, (r, t) in enumerate(path):
                aligned[r] = arr_sorted[t]
            arr_dtw[:, axis] = np.sort(aligned)[:n]
    all_feats_dtw.append(arr_dtw)
    all_hosps_dtw.extend([hosp]*n)
    all_diseases_dtw.extend(meta[meta['hosp'] == hosp]['patch_label'].values)
all_feats_cca = np.vstack(all_feats_cca)
all_hosps_cca = np.array(all_hosps_cca)
all_diseases_cca = np.array(all_diseases_cca)
all_feats_dtw = np.vstack(all_feats_dtw)
all_hosps_dtw = np.array(all_hosps_dtw)
all_diseases_dtw = np.array(all_diseases_dtw)

# ========== 分类评估 ==========
print("[Classification] Original features:")
eval_clf(all_feats_raw, all_hosps_raw, wsi_id_arr, unique_wsi, wsi_hosp_labels_encoded, 'Hospital (raw)')
eval_clf(all_feats_raw, all_diseases_raw, wsi_id_arr, unique_wsi, wsi_disease_labels_encoded, 'Disease (raw)')
print("[Classification] CCA features:")
eval_clf(all_feats_cca, all_hosps_cca, wsi_id_arr, unique_wsi, wsi_hosp_labels_encoded, 'Hospital (CCA)')
eval_clf(all_feats_cca, all_diseases_cca, wsi_id_arr, unique_wsi, wsi_disease_labels_encoded, 'Disease (CCA)')
print("[Classification] CCA+DTW features:")
eval_clf(all_feats_dtw, all_hosps_dtw, wsi_id_arr, unique_wsi, wsi_hosp_labels_encoded, 'Hospital (CCA+DTW)')
eval_clf(all_feats_dtw, all_diseases_dtw, wsi_id_arr, unique_wsi, wsi_disease_labels_encoded, 'Disease (CCA+DTW)')

# ========== t-SNE可视化 ==========
def plot_tsne(features, labels, label_names, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(8,6))
    for l in np.unique(labels):
        mask = labels == l
        plt.scatter(features_2d[mask,0], features_2d[mask,1], label=l, alpha=0.7, s=30)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

print("Plotting t-SNE for original features...")
plot_tsne(all_feats_raw, all_hosps_raw, hosp_list, 'Original features (by hospital)', os.path.join(SAVE_DIR, 'tsne_raw_by_hosp.png'))
plot_tsne(all_feats_raw, all_diseases_raw, np.unique(all_diseases_raw), 'Original features (by disease)', os.path.join(SAVE_DIR, 'tsne_raw_by_disease.png'))
print("Plotting t-SNE for CCA features...")
plot_tsne(all_feats_cca, all_hosps_cca, hosp_list, 'CCA features (by hospital)', os.path.join(SAVE_DIR, 'tsne_cca_by_hosp.png'))
plot_tsne(all_feats_cca, all_diseases_cca, np.unique(all_diseases_cca), 'CCA features (by disease)', os.path.join(SAVE_DIR, 'tsne_cca_by_disease.png'))
print("Plotting t-SNE for CCA+DTW features...")
plot_tsne(all_feats_dtw, all_hosps_dtw, hosp_list, 'CCA+DTW features (by hospital)', os.path.join(SAVE_DIR, 'tsne_ccadtw_by_hosp.png'))
plot_tsne(all_feats_dtw, all_diseases_dtw, np.unique(all_diseases_dtw), 'CCA+DTW features (by disease)', os.path.join(SAVE_DIR, 'tsne_ccadtw_by_disease.png'))

print(f"CCA+DTW distribution alignment finished. Curves and results saved to {SAVE_DIR}") 
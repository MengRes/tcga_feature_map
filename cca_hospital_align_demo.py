import argparse
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

parser = argparse.ArgumentParser(description="CCA alignment demo for hospital/source domain")
parser.add_argument('--disease', type=str, default=None, help='Specify disease label')
parser.add_argument('--hosp1', type=str, default=None, help='Specify first hospital/source')
parser.add_argument('--hosp2', type=str, default=None, help='Specify second hospital/source')
args = parser.parse_args()

META_CSV = 'results/wsi_patch_filter_20250628_163736/conch_filtered_patches.csv'
FEATURE_DIR = 'results/wsi_patch_filter_20250628_163736/features_UNI'
SAVE_DIR = 'results/wsi_patch_filter_20250628_163736/features_UNI_CCA_demo'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load metadata
meta = pd.read_csv(META_CSV)

# Select disease
if args.disease is not None:
    chosen_disease = args.disease
else:
    chosen_disease = meta['label'].value_counts().index[0]

meta_d = meta[meta['label'] == chosen_disease]
hosp_counts = meta_d['hosp'].value_counts()

# Select hospitals
if args.hosp1 is not None and args.hosp2 is not None:
    hosp1, hosp2 = args.hosp1, args.hosp2
else:
    if len(hosp_counts) < 2:
        raise ValueError('Not enough hospitals for the selected disease!')
    hosp1, hosp2 = hosp_counts.index[:2]

print(f"Using disease: {chosen_disease}, hosp1: {hosp1}, hosp2: {hosp2}")

meta_dh1 = meta_d[meta_d['hosp'] == hosp1]
meta_dh2 = meta_d[meta_d['hosp'] == hosp2]

# Load features for the selected disease and hospitals
features_list = []
labels_list = []
hosps_list = []
wsi_ids_list = []
for submeta in [meta_dh1, meta_dh2]:
    for wsi_id in submeta['wsi_id'].unique():
        feature_path = os.path.join(FEATURE_DIR, f"{wsi_id}_features.pt")
        if os.path.exists(feature_path):
            feats = torch.load(feature_path, weights_only=True)
            mask = submeta['wsi_id'] == wsi_id
            n = mask.sum()
            features_list.append(feats[:n])
            labels_list.extend(submeta.loc[mask, 'label'])
            hosps_list.extend(submeta.loc[mask, 'hosp'])
            wsi_ids_list.extend([wsi_id]*n)

features = np.vstack(features_list)
hosps = np.array(hosps_list)
wsi_ids = np.array(wsi_ids_list)

# Get feature indices for the two hospitals
X_idx = np.where(hosps == hosp1)[0]
Y_idx = np.where(hosps == hosp2)[0]
n = min(len(X_idx), len(Y_idx))
X_idx = X_idx[:n]
Y_idx = Y_idx[:n]
X = features[X_idx]
Y = features[Y_idx]

# CCA alignment
cca_dim = min(20, X.shape[1], Y.shape[1], n)
cca = CCA(n_components=cca_dim)
X_c, Y_c = cca.fit_transform(X, Y)

# Create aligned feature array (others are zero)
features_cca = np.zeros((len(hosps), X_c.shape[1]), dtype=np.float32)
features_cca[X_idx] = X_c
features_cca[Y_idx] = Y_c

# Save aligned features by wsi_id
for wsi_id in np.unique(wsi_ids):
    idx = wsi_ids == wsi_id
    torch.save(torch.from_numpy(features_cca[idx]), os.path.join(SAVE_DIR, f'{wsi_id}_features.pt'))
print(f"CCA alignment finished. Features saved to {SAVE_DIR}")

# t-SNE visualization (original features)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_tsne = tsne.fit_transform(np.vstack([X, Y]))
hosp_labels = np.array([hosp1]*len(X) + [hosp2]*len(Y))
plt.figure(figsize=(8,6))
for h, color in zip([hosp1, hosp2], ['red', 'blue']):
    mask = hosp_labels == h
    plt.scatter(features_tsne[mask,0], features_tsne[mask,1], label=h, alpha=0.7, s=30, c=color)
plt.title('t-SNE of Original Features')
plt.legend()
plt.tight_layout()
plt.savefig('tsne_original.png', dpi=200)
plt.close()

# t-SNE visualization (CCA-aligned features)
tsne2 = TSNE(n_components=2, random_state=42, perplexity=30)
features_cca_tsne = tsne2.fit_transform(np.vstack([X_c, Y_c]))
plt.figure(figsize=(8,6))
for h, color in zip([hosp1, hosp2], ['red', 'blue']):
    mask = hosp_labels == h
    plt.scatter(features_cca_tsne[mask,0], features_cca_tsne[mask,1], label=h, alpha=0.7, s=30, c=color)
plt.title('t-SNE of CCA-aligned Features')
plt.legend()
plt.tight_layout()
plt.savefig('tsne_cca.png', dpi=200)
plt.close()

# Evaluate hospital classification before CCA
X_raw = np.vstack([X, Y])
y_raw = np.array([0]*len(X) + [1]*len(Y))
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
clf_raw = LogisticRegression(max_iter=1000)
clf_raw.fit(X_train_raw, y_train_raw)
y_pred_raw = clf_raw.predict(X_test_raw)
acc_raw = accuracy_score(y_test_raw, y_pred_raw)
print(f'Before CCA, Logistic Regression hospital classification accuracy: {acc_raw:.4f}')

# Evaluate hospital classification after CCA
X_eval = np.vstack([X_c, Y_c])
y_eval = np.array([0]*len(X_c) + [1]*len(Y_c))
X_train, X_test, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'After CCA, Logistic Regression hospital classification accuracy: {acc:.4f}') 
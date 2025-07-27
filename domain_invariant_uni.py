import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Function
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DomainInvariantModel(nn.Module):
    def __init__(self, frozen_encoder, z_dim, num_classes=14, num_domains=2, lambda_grl=1.0):
        super().__init__()
        self.encoder = frozen_encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )
        self.disease_classifier = nn.Linear(z_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            GRL(lambda_=lambda_grl),
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )
    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        z_proj = self.projection(z)
        disease_logits = self.disease_classifier(z_proj)
        domain_logits = self.domain_classifier(z_proj)
        return disease_logits, domain_logits

def compute_losses(disease_logits, domain_logits, y_true, domain_labels, lambda_grl=1.0):
    loss_disease = nn.BCEWithLogitsLoss() if y_true.ndim == 2 else nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()
    L_disease = loss_disease(disease_logits, y_true)
    L_domain = loss_domain(domain_logits, domain_labels)
    total_loss = L_disease + lambda_grl * L_domain
    return total_loss, L_disease.item(), L_domain.item()

# Example training loop (pseudo-code, assumes dataloader yields images, disease_labels, domain_labels)
def train_domain_invariant_model(model, dataloader, optimizer, device, lambda_grl=1.0, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, disease_labels, domain_labels in dataloader:
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            domain_labels = domain_labels.to(device)
            disease_logits, domain_logits = model(images)
            loss, ld, ldom = compute_losses(disease_logits, domain_logits, disease_labels, domain_labels, lambda_grl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Disease={ld:.4f}, Domain={ldom:.4f}")

# Inference: only use projection + disease classifier
def predict_disease(model, encoder, x, device):
    model.eval()
    with torch.no_grad():
        z = encoder(x.to(device))
        z_proj = model.projection(z)
        logits = model.disease_classifier(z_proj)
        probs = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
        return probs 

def wsi_level_cross_val(features, wsi_ids, disease_labels, hospital_labels, encoder, num_classes, num_domains, device, lambda_grl=1.0, num_epochs=10, batch_size=32, output_dir="model_ckpt"):
    """
    features: (N, D) all patch features
    wsi_ids: (N,) patch对应的WSI id
    disease_labels: (N,) patch的disease标签（字符串）
    hospital_labels: (N,) patch的hospital标签（字符串）
    encoder: 冻结的UNI encoder
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    # WSI-level分组
    wsi_to_disease = {}
    for wsi, label in zip(wsi_ids, disease_labels):
        if wsi not in wsi_to_disease:
            wsi_to_disease[wsi] = label
    unique_wsi = np.array(list(wsi_to_disease.keys()))
    wsi_disease_labels = np.array(list(wsi_to_disease.values()))
    le_disease = LabelEncoder()
    le_hosp = LabelEncoder()
    le_disease.fit(disease_labels)
    le_hosp.fit(hospital_labels)
    wsi_disease_encoded = le_disease.transform(wsi_disease_labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_disease, accs_hosp = [], []
    for fold, (train_wsi_idx, val_wsi_idx) in enumerate(skf.split(unique_wsi, wsi_disease_encoded)):
        train_wsi = unique_wsi[train_wsi_idx]
        val_wsi = unique_wsi[val_wsi_idx]
        train_mask = np.isin(wsi_ids, train_wsi)
        val_mask = np.isin(wsi_ids, val_wsi)
        X_train, X_val = features[train_mask], features[val_mask]
        y_train_dis = le_disease.transform(np.array(disease_labels)[train_mask])
        y_val_dis = le_disease.transform(np.array(disease_labels)[val_mask])
        y_train_hosp = le_hosp.transform(np.array(hospital_labels)[train_mask])
        y_val_hosp = le_hosp.transform(np.array(hospital_labels)[val_mask])
        # 构建dataloader
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_dis), torch.tensor(y_train_hosp))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_dis), torch.tensor(y_val_hosp))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # 构建模型
        model = DomainInvariantModel(encoder, z_dim=features.shape[1], num_classes=num_classes, num_domains=num_domains, lambda_grl=lambda_grl).to(device)
        optimizer = Adam(list(model.projection.parameters()) + list(model.disease_classifier.parameters()) + list(model.domain_classifier.parameters()), lr=1e-3)
        # 训练并保存最佳权重
        best_val_acc = 0
        best_epoch = -1
        for epoch in range(num_epochs):
            train_domain_invariant_model(model, train_loader, optimizer, device, lambda_grl=lambda_grl, num_epochs=1)
            # Validation
            model.eval()
            correct_dis, total_dis = 0, 0
            with torch.no_grad():
                for x, y_dis, y_hosp in val_loader:
                    x = x.to(device)
                    y_dis = y_dis.to(device)
                    logits_dis, _ = model(x)
                    pred_dis = logits_dis.argmax(dim=1)
                    correct_dis += (pred_dis == y_dis).sum().item()
                    total_dis += len(y_dis)
            val_acc = correct_dis / total_dis
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(best_state_dict, os.path.join(output_dir, f"best_model_fold{fold+1}.pth"))
            print(f"Epoch {epoch}: val disease acc={val_acc:.4f} (best={best_val_acc:.4f} at epoch {best_epoch})")
        # 加载最佳权重
        model.load_state_dict(torch.load(os.path.join(output_dir, f"best_model_fold{fold+1}.pth"), weights_only=True))
        # 验证
        model.eval()
        correct_dis, total_dis = 0, 0
        correct_hosp, total_hosp = 0, 0
        with torch.no_grad():
            for x, y_dis, y_hosp in val_loader:
                x = x.to(device)
                y_dis = y_dis.to(device)
                y_hosp = y_hosp.to(device)
                logits_dis, logits_hosp = model(x)
                pred_dis = logits_dis.argmax(dim=1)
                pred_hosp = logits_hosp.argmax(dim=1)
                correct_dis += (pred_dis == y_dis).sum().item()
                total_dis += len(y_dis)
                correct_hosp += (pred_hosp == y_hosp).sum().item()
                total_hosp += len(y_hosp)
        acc_d = correct_dis / total_dis
        acc_h = correct_hosp / total_hosp
        accs_disease.append(acc_d)
        accs_hosp.append(acc_h)
        print(f"Fold {fold+1}: Disease ACC={acc_d:.4f}, Hospital ACC={acc_h:.4f}")
    print(f"Mean Disease ACC: {np.mean(accs_disease):.4f} ± {np.std(accs_disease):.4f}")
    print(f"Mean Hospital ACC: {np.mean(accs_hosp):.4f} ± {np.std(accs_hosp):.4f}")

def plot_tsne(features, labels, label_names, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10,8))
    for l in np.unique(labels):
        mask = labels == l
        plt.scatter(features_2d[mask,0], features_2d[mask,1], label=label_names[l] if label_names is not None else l, alpha=0.7, s=20)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    import pandas as pd
    import glob
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    # 1. 加载patch元信息
    meta = pd.read_csv("results/wsi_patch_filter_20250628_163736/conch_filtered_patches.csv")
    # 2. 加载所有patch特征
    feature_dir = "results/wsi_patch_filter_20250628_163736/features_UNI"
    feature_files = sorted(glob.glob(f"{feature_dir}/*_features.pt"))
    all_features, all_wsi_ids, all_disease, all_hosp = [], [], [], []
    for f in feature_files:
        wsi_id = f.split("/")[-1].replace("_features.pt", "")
        feats = torch.load(f, map_location="cpu", weights_only=True)
        mask = meta['wsi_id'] == wsi_id
        n = mask.sum()
        if n == 0: continue
        all_features.append(feats[:n].numpy())
        all_wsi_ids.extend([wsi_id]*n)
        all_disease.extend(meta[mask]['patch_label'].tolist())
        all_hosp.extend(meta[mask]['hosp'].tolist())
    features = np.vstack(all_features)
    wsi_ids = np.array(all_wsi_ids)
    disease_labels = np.array(all_disease)
    hospital_labels = np.array(all_hosp)
    # 3. 原始UNI特征五折交叉验证（WSI-level）
    le_disease = LabelEncoder()
    le_hosp = LabelEncoder()
    le_disease.fit(disease_labels)
    le_hosp.fit(hospital_labels)
    # t-SNE: 原始UNI特征
    os.makedirs("visualizations", exist_ok=True)
    print("Plotting t-SNE for original UNI features...")
    plot_tsne(features, le_hosp.transform(hospital_labels), le_hosp.classes_, 'Original UNI features (by hospital)', os.path.join("visualizations", 'tsne_uni_by_hosp.png'))
    plot_tsne(features, le_disease.transform(disease_labels), le_disease.classes_, 'Original UNI features (by disease)', os.path.join("visualizations", 'tsne_uni_by_disease.png'))
    # 4. 用nn.Identity模拟冻结UNI encoder（特征直通）
    import torch.nn as nn
    encoder = nn.Identity()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(disease_labels))
    num_domains = len(np.unique(hospital_labels))
    # --- 提取对抗投影特征 ---
    print("Extracting adversarial projection features for all patches...")
    model = DomainInvariantModel(encoder, z_dim=features.shape[1], num_classes=num_classes, num_domains=num_domains, lambda_grl=0.05).to(device)
    # 加载所有fold的最佳权重并提取特征（这里简单用最后一个fold权重，或可循环所有fold）
    best_model_path = os.path.join("model_ckpt", f"best_model_fold5.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        feats_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        z_proj = model.projection(feats_tensor).cpu().numpy()
    # t-SNE: 对抗投影特征
    print("Plotting t-SNE for adversarial projection features...")
    plot_tsne(z_proj, le_hosp.transform(hospital_labels), le_hosp.classes_, 'Adversarial projection features (by hospital)', os.path.join("visualizations", 'tsne_advproj_by_hosp.png'))
    plot_tsne(z_proj, le_disease.transform(disease_labels), le_disease.classes_, 'Adversarial projection features (by disease)', os.path.join("visualizations", 'tsne_advproj_by_disease.png'))
    wsi_level_cross_val(
        features, wsi_ids, disease_labels, hospital_labels,
        encoder, num_classes, num_domains, device,
        lambda_grl=0.4, num_epochs=20, batch_size=32, output_dir="model_ckpt"
    ) 
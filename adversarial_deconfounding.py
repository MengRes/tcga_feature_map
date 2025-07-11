#!/usr/bin/env python3
"""
对抗训练去混淆模型
目标：
- maximize 疾病预测准确率
- minimize 医院分类准确率（对抗）

结构：
patch image → frozen encoder → feature f → disease head
                              ↘ GRL → domain classifier (hospital)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转函数的实现
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层
    """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DiseaseHead(nn.Module):
    """
    疾病预测头
    """
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(DiseaseHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class HospitalClassifier(nn.Module):
    """
    医院分类器（对抗）
    """
    def __init__(self, input_dim, num_hospitals, hidden_dim=256):
        super(HospitalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_hospitals)
        )

    def forward(self, x):
        return self.classifier(x)


class AdversarialModel(nn.Module):
    """
    对抗训练模型
    """
    def __init__(self, feature_dim, num_diseases, num_hospitals, grl_alpha=1.0):
        super(AdversarialModel, self).__init__()
        
        # 疾病预测头
        self.disease_head = DiseaseHead(feature_dim, num_diseases)
        
        # 梯度反转层
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        
        # 医院分类器（对抗）
        self.hospital_classifier = HospitalClassifier(feature_dim, num_hospitals)
        
    def forward(self, features):
        # 疾病预测
        disease_logits = self.disease_head(features)
        
        # 医院对抗分类
        reversed_features = self.grl(features)
        hospital_logits = self.hospital_classifier(reversed_features)
        
        return disease_logits, hospital_logits


class PatchDataset(Dataset):
    """
    补丁数据集
    """
    def __init__(self, features, disease_labels, hospital_labels, wsi_ids):
        self.features = torch.FloatTensor(features)
        self.disease_labels = torch.LongTensor(disease_labels)
        self.hospital_labels = torch.LongTensor(hospital_labels)
        self.wsi_ids = wsi_ids
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'disease_label': self.disease_labels[idx],
            'hospital_label': self.hospital_labels[idx],
            'wsi_id': self.wsi_ids[idx]
        }


def load_features_and_metadata(features_dir, patches_csv_path):
    """
    加载特征和元数据
    """
    print(f"Loading features from {features_dir}")
    
    # 加载补丁信息
    patches_df = pd.read_csv(patches_csv_path)
    
    # 从文件名提取医院信息
    def extract_hosp_from_filename(filename):
        parts = filename.split('-')
        if len(parts) >= 3:
            return parts[1]
        return "Unknown"
    
    patches_df['hosp'] = patches_df['wsi_id'].apply(extract_hosp_from_filename)
    
    # 加载特征文件
    feature_files = sorted(glob.glob(os.path.join(features_dir, "*_features.pt")))
    
    if len(feature_files) == 0:
        raise ValueError(f"No feature files found in {features_dir}")
    
    print(f"Found {len(feature_files)} feature files")
    
    all_features = []
    all_metadata = []
    
    for feature_file in tqdm(feature_files, desc="Loading features"):
        try:
            features = torch.load(feature_file, weights_only=True)
            wsi_id = os.path.basename(feature_file).replace("_features.pt", "")
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


def train_epoch(model, dataloader, disease_criterion, hospital_criterion, 
                disease_optimizer, hospital_optimizer, device, lambda_adv=1.0):
    """
    训练一个epoch
    """
    model.train()
    total_disease_loss = 0
    total_hospital_loss = 0
    total_disease_correct = 0
    total_hospital_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        disease_labels = batch['disease_label'].to(device)
        hospital_labels = batch['hospital_label'].to(device)
        
        # 前向传播
        disease_logits, hospital_logits = model(features)
        
        # 计算损失
        disease_loss = disease_criterion(disease_logits, disease_labels)
        hospital_loss = hospital_criterion(hospital_logits, hospital_labels)
        
        # 总损失：疾病损失 - λ * 医院损失（对抗）
        total_loss = disease_loss - lambda_adv * hospital_loss
        
        # 反向传播
        disease_optimizer.zero_grad()
        hospital_optimizer.zero_grad()
        total_loss.backward()
        disease_optimizer.step()
        hospital_optimizer.step()
        
        # 统计
        total_disease_loss += disease_loss.item()
        total_hospital_loss += hospital_loss.item()
        
        _, disease_preds = torch.max(disease_logits, 1)
        _, hospital_preds = torch.max(hospital_logits, 1)
        
        total_disease_correct += (disease_preds == disease_labels).sum().item()
        total_hospital_correct += (hospital_preds == hospital_labels).sum().item()
        total_samples += len(features)
    
    avg_disease_loss = total_disease_loss / len(dataloader)
    avg_hospital_loss = total_hospital_loss / len(dataloader)
    disease_acc = total_disease_correct / total_samples
    hospital_acc = total_hospital_correct / total_samples
    
    return avg_disease_loss, avg_hospital_loss, disease_acc, hospital_acc


def evaluate(model, dataloader, disease_criterion, hospital_criterion, device):
    """
    评估模型
    """
    model.eval()
    total_disease_loss = 0
    total_hospital_loss = 0
    all_disease_preds = []
    all_disease_labels = []
    all_hospital_preds = []
    all_hospital_labels = []
    all_disease_probs = []
    all_hospital_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            disease_labels = batch['disease_label'].to(device)
            hospital_labels = batch['hospital_label'].to(device)
            
            disease_logits, hospital_logits = model(features)
            
            disease_loss = disease_criterion(disease_logits, disease_labels)
            hospital_loss = hospital_criterion(hospital_logits, hospital_labels)
            
            total_disease_loss += disease_loss.item()
            total_hospital_loss += hospital_loss.item()
            
            disease_probs = F.softmax(disease_logits, dim=1)
            hospital_probs = F.softmax(hospital_logits, dim=1)
            
            _, disease_preds = torch.max(disease_logits, 1)
            _, hospital_preds = torch.max(hospital_logits, 1)
            
            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(disease_labels.cpu().numpy())
            all_hospital_preds.extend(hospital_preds.cpu().numpy())
            all_hospital_labels.extend(hospital_labels.cpu().numpy())
            all_disease_probs.extend(disease_probs.cpu().numpy())
            all_hospital_probs.extend(hospital_probs.cpu().numpy())
    
    avg_disease_loss = total_disease_loss / len(dataloader)
    avg_hospital_loss = total_hospital_loss / len(dataloader)
    
    disease_acc = accuracy_score(all_disease_labels, all_disease_preds)
    hospital_acc = accuracy_score(all_hospital_labels, all_hospital_preds)
    
    # 计算AUC
    try:
        disease_auc = roc_auc_score(all_disease_labels, all_disease_probs, multi_class='ovr', average='macro')
    except:
        disease_auc = 0.5
    
    try:
        hospital_auc = roc_auc_score(all_hospital_labels, all_hospital_probs, multi_class='ovr', average='macro')
    except:
        hospital_auc = 0.5
    
    return {
        'disease_loss': avg_disease_loss,
        'hospital_loss': avg_hospital_loss,
        'disease_acc': disease_acc,
        'hospital_acc': hospital_acc,
        'disease_auc': disease_auc,
        'hospital_auc': hospital_auc
    }


def train_adversarial_model(features, metadata_df, output_dir, 
                          num_epochs=50, batch_size=32, learning_rate=1e-3, 
                          lambda_adv=1.0, grl_alpha=1.0):
    """
    训练对抗模型
    """
    print("Training Adversarial Model")
    print("="*50)
    
    # 数据预处理
    le_disease = LabelEncoder()
    le_hospital = LabelEncoder()
    
    disease_labels = le_disease.fit_transform(metadata_df['patch_label'])
    hospital_labels = le_hospital.fit_transform(metadata_df['hosp'])
    wsi_ids = metadata_df['wsi_id'].values
    
    print(f"Disease classes: {le_disease.classes_}")
    print(f"Hospital classes: {le_hospital.classes_}")
    print(f"Number of diseases: {len(le_disease.classes_)}")
    print(f"Number of hospitals: {len(le_hospital.classes_)}")
    
    # 创建数据集
    dataset = PatchDataset(features, disease_labels, hospital_labels, wsi_ids)
    
    # WSI级别的分层交叉验证
    wsi_to_disease = metadata_df.groupby('wsi_id')['patch_label'].first().to_dict()
    wsi_list = np.array(list(wsi_to_disease.keys()))
    wsi_disease_labels = np.array(list(wsi_to_disease.values()))
    wsi_disease_encoded = le_disease.transform(wsi_disease_labels)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    
    for fold, (train_wsi_idx, val_wsi_idx) in enumerate(skf.split(wsi_list, wsi_disease_encoded)):
        print(f"\nFold {fold + 1}/5")
        print("-" * 30)
        
        train_wsi = wsi_list[train_wsi_idx]
        val_wsi = wsi_list[val_wsi_idx]
        
        # 创建训练和验证掩码
        train_mask = np.isin(wsi_ids, train_wsi)
        val_mask = np.isin(wsi_ids, val_wsi)
        
        # 分割数据
        train_features = features[train_mask]
        val_features = features[val_mask]
        train_disease_labels = disease_labels[train_mask]
        val_disease_labels = disease_labels[val_mask]
        train_hospital_labels = hospital_labels[train_mask]
        val_hospital_labels = hospital_labels[val_mask]
        train_wsi_ids = wsi_ids[train_mask]
        val_wsi_ids = wsi_ids[val_mask]
        
        # 创建数据加载器
        train_dataset = PatchDataset(train_features, train_disease_labels, 
                                   train_hospital_labels, train_wsi_ids)
        val_dataset = PatchDataset(val_features, val_disease_labels, 
                                 val_hospital_labels, val_wsi_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdversarialModel(
            feature_dim=features.shape[1],
            num_diseases=len(le_disease.classes_),
            num_hospitals=len(le_hospital.classes_),
            grl_alpha=grl_alpha
        ).to(device)
        
        # 损失函数和优化器
        disease_criterion = nn.CrossEntropyLoss()
        hospital_criterion = nn.CrossEntropyLoss()
        
        disease_optimizer = optim.Adam(model.disease_head.parameters(), lr=learning_rate)
        hospital_optimizer = optim.Adam(model.hospital_classifier.parameters(), lr=learning_rate)
        
        # 训练循环
        best_disease_acc = 0
        best_hospital_acc = 1.0  # 对抗目标：最小化医院分类准确率
        patience = 10
        patience_counter = 0
        
        train_history = []
        
        for epoch in range(num_epochs):
            # 训练
            train_disease_loss, train_hospital_loss, train_disease_acc, train_hospital_acc = \
                train_epoch(model, train_loader, disease_criterion, hospital_criterion,
                          disease_optimizer, hospital_optimizer, device, lambda_adv)
            
            # 验证
            val_results = evaluate(model, val_loader, disease_criterion, hospital_criterion, device)
            
            train_history.append({
                'epoch': epoch,
                'train_disease_loss': train_disease_loss,
                'train_hospital_loss': train_hospital_loss,
                'train_disease_acc': train_disease_acc,
                'train_hospital_acc': train_hospital_acc,
                'val_disease_loss': val_results['disease_loss'],
                'val_hospital_loss': val_results['hospital_loss'],
                'val_disease_acc': val_results['disease_acc'],
                'val_hospital_acc': val_results['hospital_acc'],
                'val_disease_auc': val_results['disease_auc'],
                'val_hospital_auc': val_results['hospital_auc']
            })
            
            # 早停检查
            if val_results['disease_acc'] > best_disease_acc and val_results['hospital_acc'] < best_hospital_acc:
                best_disease_acc = val_results['disease_acc']
                best_hospital_acc = val_results['hospital_acc']
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_fold_{fold}.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Disease Acc={val_results['disease_acc']:.3f}, "
                      f"Hospital Acc={val_results['hospital_acc']:.3f}, "
                      f"Disease AUC={val_results['disease_auc']:.3f}, "
                      f"Hospital AUC={val_results['hospital_auc']:.3f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型进行评估
        model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_fold_{fold}.pth')))
        final_results = evaluate(model, val_loader, disease_criterion, hospital_criterion, device)
        
        all_results.append(final_results)
        
        print(f"Fold {fold + 1} Final Results:")
        print(f"  Disease Accuracy: {final_results['disease_acc']:.3f}")
        print(f"  Hospital Accuracy: {final_results['hospital_acc']:.3f}")
        print(f"  Disease AUC: {final_results['disease_auc']:.3f}")
        print(f"  Hospital AUC: {final_results['hospital_auc']:.3f}")
        
        # 保存训练历史
        train_df = pd.DataFrame(train_history)
        train_df.to_csv(os.path.join(output_dir, f'training_history_fold_{fold}.csv'), index=False)
    
    # 计算平均结果
    avg_results = {}
    for key in all_results[0].keys():
        avg_results[key] = np.mean([result[key] for result in all_results])
        avg_results[f'{key}_std'] = np.std([result[key] for result in all_results])
    
    print(f"\nCross-Validation Results:")
    print(f"Disease Accuracy: {avg_results['disease_acc']:.3f} ± {avg_results['disease_acc_std']:.3f}")
    print(f"Hospital Accuracy: {avg_results['hospital_acc']:.3f} ± {avg_results['hospital_acc_std']:.3f}")
    print(f"Disease AUC: {avg_results['disease_auc']:.3f} ± {avg_results['disease_auc_std']:.3f}")
    print(f"Hospital AUC: {avg_results['hospital_auc']:.3f} ± {avg_results['hospital_auc_std']:.3f}")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)
    
    # 保存标签编码器
    import pickle
    with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump({'disease': le_disease, 'hospital': le_hospital}, f)
    
    # 提取并保存对抗训练后的特征
    print("\nExtracting and saving adversarial features...")
    adv_features = save_adversarial_features(features, metadata_df, output_dir, le_disease, le_hospital)
    
    # 创建t-SNE可视化
    print("\nCreating t-SNE visualizations...")
    create_tsne_visualizations(features, adv_features, metadata_df, output_dir)
    
    return avg_results, all_results


def save_adversarial_features(features, metadata_df, output_dir, le_disease, le_hospital):
    """
    提取并保存对抗训练后的特征
    """
    print("Extracting adversarial features...")
    
    # 创建特征保存目录
    parent_dir = os.path.dirname(output_dir.rstrip('/'))
    adv_feat_dir = os.path.join(parent_dir, 'features_ADV')
    os.makedirs(adv_feat_dir, exist_ok=True)
    
    # 创建最终的对抗模型（使用所有数据训练）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model = AdversarialModel(
        feature_dim=features.shape[1],
        num_diseases=len(le_disease.classes_),
        num_hospitals=len(le_hospital.classes_),
        grl_alpha=1.0
    ).to(device)
    
    # 准备数据
    disease_labels = le_disease.transform(metadata_df['patch_label'])
    hospital_labels = le_hospital.transform(metadata_df['hosp'])
    wsi_ids = metadata_df['wsi_id'].values
    
    # 创建数据集和数据加载器
    dataset = PatchDataset(features, disease_labels, hospital_labels, wsi_ids)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 训练最终模型（使用所有数据）
    print("Training final model on all data...")
    disease_criterion = nn.CrossEntropyLoss()
    hospital_criterion = nn.CrossEntropyLoss()
    disease_optimizer = optim.Adam(final_model.disease_head.parameters(), lr=1e-3)
    hospital_optimizer = optim.Adam(final_model.hospital_classifier.parameters(), lr=1e-3)
    
    # 简单训练几个epoch
    for epoch in range(20):
        train_epoch(final_model, dataloader, disease_criterion, hospital_criterion,
                   disease_optimizer, hospital_optimizer, device, lambda_adv=1.0)
        if epoch % 5 == 0:
            print(f"Final training epoch {epoch}/20")
    
    # 提取特征
    print("Extracting features...")
    final_model.eval()
    all_extracted_features = []
    all_wsi_ids = []
    all_patch_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            features_batch = batch['features'].to(device)
            wsi_ids_batch = batch['wsi_id']
            
            # 获取疾病预测头的中间特征（在最后的线性层之前）
            disease_features = final_model.disease_head.classifier[:-1](features_batch)
            all_extracted_features.append(disease_features.cpu().numpy())
            all_wsi_ids.extend(wsi_ids_batch)
    
    all_extracted_features = np.vstack(all_extracted_features)
    
    # 按WSI分组保存特征
    print("Saving features by WSI...")
    wsi_groups = {}
    for i, wsi_id in enumerate(all_wsi_ids):
        if wsi_id not in wsi_groups:
            wsi_groups[wsi_id] = []
        wsi_groups[wsi_id].append(all_extracted_features[i])
    
    for wsi_id, wsi_features in wsi_groups.items():
        wsi_features = np.array(wsi_features, dtype=np.float32)
        torch.save(torch.from_numpy(wsi_features), 
                  os.path.join(adv_feat_dir, f'{wsi_id}_features.pt'))
    
    print(f"Saved {len(wsi_groups)} WSI feature files to {adv_feat_dir}")
    print(f"Feature dimension: {all_extracted_features.shape[1]}")
    
    return all_extracted_features


def create_tsne_visualizations(original_features, adv_features, metadata_df, output_dir):
    """
    创建t-SNE可视化，比较原始特征和对抗训练后的特征
    """
    print("Creating t-SNE visualizations...")
    
    from sklearn.manifold import TSNE
    
    # 创建可视化目录
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 准备标签
    hospital_labels = metadata_df['hosp'].values
    disease_labels = metadata_df['patch_label'].values
    
    # 计算t-SNE
    print("Computing t-SNE for original features...")
    tsne_original = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    tsne_coords_original = tsne_original.fit_transform(original_features)
    
    print("Computing t-SNE for adversarial features...")
    tsne_adv = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    tsne_coords_adv = tsne_adv.fit_transform(adv_features)
    
    # 1. 按医院的可视化对比
    print("Creating hospital-based t-SNE comparison...")
    unique_hospitals = np.unique(hospital_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_hospitals)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始特征按医院
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        ax1.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    ax1.set_title('Original Features by Hospital', fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 对抗特征按医院
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        ax2.scatter(tsne_coords_adv[mask, 0], tsne_coords_adv[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    ax2.set_title('Adversarial Features by Hospital', fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_hospital_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(viz_dir, 'tsne_hospital_comparison.png')}")
    
    # 2. 按疾病的可视化对比
    print("Creating disease-based t-SNE comparison...")
    unique_diseases = np.unique(disease_labels)
    colors_disease = plt.cm.tab10(np.linspace(0, 1, len(unique_diseases)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始特征按疾病
    for i, disease in enumerate(unique_diseases):
        mask = disease_labels == disease
        ax1.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=disease, alpha=0.7, s=20, c=[colors_disease[i]])
    ax1.set_title('Original Features by Disease', fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 对抗特征按疾病
    for i, disease in enumerate(unique_diseases):
        mask = disease_labels == disease
        ax2.scatter(tsne_coords_adv[mask, 0], tsne_coords_adv[mask, 1], 
                   label=disease, alpha=0.7, s=20, c=[colors_disease[i]])
    ax2.set_title('Adversarial Features by Disease', fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_disease_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(viz_dir, 'tsne_disease_comparison.png')}")
    
    # 3. 单独的医院可视化
    print("Creating individual hospital t-SNE plots...")
    
    # 原始特征按医院
    plt.figure(figsize=(12, 8))
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        plt.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    plt.title('Original Features - All Patches by Hospital', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_original_by_hospital.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(viz_dir, 'tsne_original_by_hospital.png')}")
    
    # 对抗特征按医院
    plt.figure(figsize=(12, 8))
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        plt.scatter(tsne_coords_adv[mask, 0], tsne_coords_adv[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    plt.title('Adversarial Features - All Patches by Hospital', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_adversarial_by_hospital.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(viz_dir, 'tsne_adversarial_by_hospital.png')}")
    
    # 4. 单独的疾病可视化
    print("Creating individual disease t-SNE plots...")
    
    # 原始特征按疾病
    plt.figure(figsize=(12, 8))
    for i, disease in enumerate(unique_diseases):
        mask = disease_labels == disease
        plt.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=disease, alpha=0.7, s=20, c=[colors_disease[i]])
    plt.title('Original Features - All Patches by Disease', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_original_by_disease.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(viz_dir, 'tsne_original_by_disease.png')}")
    
    # 对抗特征按疾病
    plt.figure(figsize=(12, 8))
    for i, disease in enumerate(unique_diseases):
        mask = disease_labels == disease
        plt.scatter(tsne_coords_adv[mask, 0], tsne_coords_adv[mask, 1], 
                   label=disease, alpha=0.7, s=20, c=[colors_disease[i]])
    plt.title('Adversarial Features - All Patches by Disease', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_adversarial_by_disease.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(viz_dir, 'tsne_adversarial_by_disease.png')}")
    
    print("t-SNE visualizations completed!")


def main():
    parser = argparse.ArgumentParser(description='Adversarial Training for Hospital Deconfounding')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing feature files')
    parser.add_argument('--patches_csv', type=str, required=True,
                       help='Path to patches CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                       help='Adversarial loss weight')
    parser.add_argument('--grl_alpha', type=float, default=1.0,
                       help='Gradient reversal layer alpha')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    features, metadata_df = load_features_and_metadata(args.features_dir, args.patches_csv)
    
    # 训练模型
    avg_results, all_results = train_adversarial_model(
        features, metadata_df, args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_adv=args.lambda_adv,
        grl_alpha=args.grl_alpha
    )
    
    print(f"\nTraining completed! Results saved in {args.output_dir}")


if __name__ == "__main__":
    main() 
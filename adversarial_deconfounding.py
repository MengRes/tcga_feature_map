#!/usr/bin/env python3
"""
Adversarial training for deconfounding model
Objective:
- maximize disease prediction accuracy
- minimize hospital classification accuracy (adversarial)

Structure:
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
    Gradient reversal function implementation
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
    Gradient reversal layer
    """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DiseaseHead(nn.Module):
    """
    Disease prediction head
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
    Hospital classifier (adversarial)
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


class ProjectionHead(nn.Module):
    """
    Projection head to map frozen features to a new space (trainable)
    """
    def __init__(self, input_dim, proj_dim=256, hidden_dim=512, nonlinear=True):
        super(ProjectionHead, self).__init__()
        if nonlinear:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim)
            )
        else:
            self.proj = nn.Linear(input_dim, proj_dim)
    def forward(self, x):
        return self.proj(x)

class AdversarialProjectionModel(nn.Module):
    """
    Adversarial model with trainable projection head on frozen features
    Only projection head and domain classifier are trainable
    """
    def __init__(self, feature_dim, proj_dim, num_diseases, num_hospitals, grl_alpha=1.0, nonlinear=True):
        super(AdversarialProjectionModel, self).__init__()
        self.projection = ProjectionHead(feature_dim, proj_dim, nonlinear=nonlinear)
        self.disease_head = DiseaseHead(proj_dim, num_diseases)
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.hospital_classifier = HospitalClassifier(proj_dim, num_hospitals)
    def forward(self, features):
        proj = self.projection(features)
        disease_logits = self.disease_head(proj)
        reversed_proj = self.grl(proj)
        hospital_logits = self.hospital_classifier(reversed_proj)
        return disease_logits, hospital_logits


class PatchDataset(Dataset):
    """
    Patch dataset
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
    Load features and metadata
    """
    print(f"Loading features from {features_dir}")
    
    # Load patch information
    patches_df = pd.read_csv(patches_csv_path)
    
    # Extract hospital information from filename
    def extract_hosp_from_filename(filename):
        parts = filename.split('-')
        if len(parts) >= 3:
            return parts[1]
        return "Unknown"
    
    patches_df['hosp'] = patches_df['wsi_id'].apply(extract_hosp_from_filename)
    
    # Load feature files
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
    Train one epoch
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
        
        # Forward pass
        disease_logits, hospital_logits = model(features)
        
        # Calculate loss
        disease_loss = disease_criterion(disease_logits, disease_labels)
        hospital_loss = hospital_criterion(hospital_logits, hospital_labels)
        
        # Total loss: disease loss - λ * hospital loss (adversarial)
        total_loss = disease_loss - lambda_adv * hospital_loss
        
        # Backward pass
        disease_optimizer.zero_grad()
        hospital_optimizer.zero_grad()
        total_loss.backward()
        disease_optimizer.step()
        hospital_optimizer.step()
        
        # Statistics
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
    Evaluate model
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
    
    # Calculate AUC
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
    Train adversarial model
    """
    print("Training Adversarial Model")
    print("="*50)
    
    # Data preprocessing
    le_disease = LabelEncoder()
    le_hospital = LabelEncoder()
    
    disease_labels = le_disease.fit_transform(metadata_df['patch_label'])
    hospital_labels = le_hospital.fit_transform(metadata_df['hosp'])
    wsi_ids = metadata_df['wsi_id'].values
    
    print(f"Disease classes: {le_disease.classes_}")
    print(f"Hospital classes: {le_hospital.classes_}")
    print(f"Number of diseases: {len(le_disease.classes_)}")
    print(f"Number of hospitals: {len(le_hospital.classes_)}")
    
    # Create dataset
    dataset = PatchDataset(features, disease_labels, hospital_labels, wsi_ids)
    
    # WSI-level stratified cross-validation
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
        
        # Create training and validation masks
        train_mask = np.isin(wsi_ids, train_wsi)
        val_mask = np.isin(wsi_ids, val_wsi)
        
        # Split data
        train_features = features[train_mask]
        val_features = features[val_mask]
        train_disease_labels = disease_labels[train_mask]
        val_disease_labels = disease_labels[val_mask]
        train_hospital_labels = hospital_labels[train_mask]
        val_hospital_labels = hospital_labels[val_mask]
        train_wsi_ids = wsi_ids[train_mask]
        val_wsi_ids = wsi_ids[val_mask]
        
        # Create data loaders
        train_dataset = PatchDataset(train_features, train_disease_labels, 
                                   train_hospital_labels, train_wsi_ids)
        val_dataset = PatchDataset(val_features, val_disease_labels, 
                                 val_hospital_labels, val_wsi_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdversarialProjectionModel(
            feature_dim=features.shape[1],
            proj_dim=512, # Example projection dimension
            num_diseases=len(le_disease.classes_),
            num_hospitals=len(le_hospital.classes_),
            grl_alpha=grl_alpha
        ).to(device)
        
        # Loss functions and optimizers
        disease_criterion = nn.CrossEntropyLoss()
        hospital_criterion = nn.CrossEntropyLoss()
        
        # Only optimize projection head and domain classifier
        projection_optimizer = optim.Adam(model.projection.parameters(), lr=learning_rate)
        hospital_optimizer = optim.Adam(model.hospital_classifier.parameters(), lr=learning_rate)
        
        # Training loop
        best_disease_acc = 0
        best_hospital_acc = 1.0  # Adversarial objective: minimize hospital classification accuracy
        patience = 10
        patience_counter = 0
        
        train_history = []
        
        for epoch in range(num_epochs):
            # Training
            train_disease_loss, train_hospital_loss, train_disease_acc, train_hospital_acc = \
                train_epoch(model, train_loader, disease_criterion, hospital_criterion,
                          projection_optimizer, hospital_optimizer, device, lambda_adv)
            
            # Validation
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
            
            # Early stopping check
            if val_results['disease_acc'] > best_disease_acc and val_results['hospital_acc'] < best_hospital_acc:
                best_disease_acc = val_results['disease_acc']
                best_hospital_acc = val_results['hospital_acc']
                patience_counter = 0
                # Save best model
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
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_fold_{fold}.pth')))
        final_results = evaluate(model, val_loader, disease_criterion, hospital_criterion, device)
        
        all_results.append(final_results)
        
        print(f"Fold {fold + 1} Final Results:")
        print(f"  Disease Accuracy: {final_results['disease_acc']:.3f}")
        print(f"  Hospital Accuracy: {final_results['hospital_acc']:.3f}")
        print(f"  Disease AUC: {final_results['disease_auc']:.3f}")
        print(f"  Hospital AUC: {final_results['hospital_auc']:.3f}")
        
        # Save training history
        train_df = pd.DataFrame(train_history)
        train_df.to_csv(os.path.join(output_dir, f'training_history_fold_{fold}.csv'), index=False)
    
    # Calculate average results
    avg_results = {}
    for key in all_results[0].keys():
        avg_results[key] = np.mean([result[key] for result in all_results])
        avg_results[f'{key}_std'] = np.std([result[key] for result in all_results])
    
    print(f"\nCross-Validation Results:")
    print(f"Disease Accuracy: {avg_results['disease_acc']:.3f} ± {avg_results['disease_acc_std']:.3f}")
    print(f"Hospital Accuracy: {avg_results['hospital_acc']:.3f} ± {avg_results['hospital_acc_std']:.3f}")
    print(f"Disease AUC: {avg_results['disease_auc']:.3f} ± {avg_results['disease_auc_std']:.3f}")
    print(f"Hospital AUC: {avg_results['hospital_auc']:.3f} ± {avg_results['hospital_auc_std']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)
    
    # Save label encoders
    import pickle
    with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump({'disease': le_disease, 'hospital': le_hospital}, f)
    
    # Extract and save adversarial features
    print("\nExtracting and saving adversarial features...")
    adv_features = save_adversarial_features(features, metadata_df, output_dir, le_disease, le_hospital)
    
    # Create t-SNE visualizations
    print("\nCreating t-SNE visualizations...")
    create_tsne_visualizations(features, adv_features, metadata_df, output_dir)
    
    return avg_results, all_results


def save_adversarial_features(features, metadata_df, output_dir, le_disease, le_hospital):
    """
    Extract and save adversarial features
    """
    print("Extracting adversarial features...")
    
    # Create feature save directory
    parent_dir = os.path.dirname(output_dir.rstrip('/'))
    adv_feat_dir = os.path.join(parent_dir, 'features_ADV')
    os.makedirs(adv_feat_dir, exist_ok=True)
    
    # Create final adversarial model (trained on all data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model = AdversarialProjectionModel(
        feature_dim=features.shape[1],
        proj_dim=512, # Example projection dimension
        num_diseases=len(le_disease.classes_),
        num_hospitals=len(le_hospital.classes_),
        grl_alpha=1.0
    ).to(device)
    
    # Prepare data
    disease_labels = le_disease.transform(metadata_df['patch_label'])
    hospital_labels = le_hospital.transform(metadata_df['hosp'])
    wsi_ids = metadata_df['wsi_id'].values
    
    # Create dataset and data loader
    dataset = PatchDataset(features, disease_labels, hospital_labels, wsi_ids)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Train final model (on all data)
    print("Training final model on all data...")
    disease_criterion = nn.CrossEntropyLoss()
    hospital_criterion = nn.CrossEntropyLoss()
    projection_optimizer = optim.Adam(final_model.projection.parameters(), lr=1e-3)
    hospital_optimizer = optim.Adam(final_model.hospital_classifier.parameters(), lr=1e-3)
    
    # Simple training for a few epochs
    for epoch in range(20):
        train_epoch(final_model, dataloader, disease_criterion, hospital_criterion,
                   projection_optimizer, hospital_optimizer, device, lambda_adv=1.0)
        if epoch % 5 == 0:
            print(f"Final training epoch {epoch}/20")
    
    # Extract features
    print("Extracting features...")
    final_model.eval()
    all_extracted_features = []
    all_wsi_ids = []
    all_patch_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            features_batch = batch['features'].to(device)
            wsi_ids_batch = batch['wsi_id']
            
            # Get intermediate features of the disease prediction head (before the last linear layer)
            # These are the features after the projection head
            proj_features = final_model.projection(features_batch)
            all_extracted_features.append(proj_features.cpu().numpy())
            all_wsi_ids.extend(wsi_ids_batch)
    
    all_extracted_features = np.vstack(all_extracted_features)
    
    # Save features by WSI
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
    Create t-SNE visualizations, comparing original features with adversarial training features
    """
    print("Creating t-SNE visualizations...")
    
    from sklearn.manifold import TSNE
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Prepare labels
    hospital_labels = metadata_df['hosp'].values
    disease_labels = metadata_df['patch_label'].values
    
    # Compute t-SNE
    print("Computing t-SNE for original features...")
    tsne_original = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    tsne_coords_original = tsne_original.fit_transform(original_features)
    
    print("Computing t-SNE for adversarial features...")
    tsne_adv = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
    tsne_coords_adv = tsne_adv.fit_transform(adv_features)
    
    # 1. Visualization comparison by hospital
    print("Creating hospital-based t-SNE comparison...")
    unique_hospitals = np.unique(hospital_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_hospitals)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original features by hospital
    for i, hospital in enumerate(unique_hospitals):
        mask = hospital_labels == hospital
        ax1.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=hospital, alpha=0.7, s=20, c=[colors[i]])
    ax1.set_title('Original Features by Hospital', fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Hospital', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adversarial features by hospital
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
    
    # 2. Visualization comparison by disease
    print("Creating disease-based t-SNE comparison...")
    unique_diseases = np.unique(disease_labels)
    colors_disease = plt.cm.tab10(np.linspace(0, 1, len(unique_diseases)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original features by disease
    for i, disease in enumerate(unique_diseases):
        mask = disease_labels == disease
        ax1.scatter(tsne_coords_original[mask, 0], tsne_coords_original[mask, 1], 
                   label=disease, alpha=0.7, s=20, c=[colors_disease[i]])
    ax1.set_title('Original Features by Disease', fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adversarial features by disease
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
    
    # 3. Individual hospital visualization
    print("Creating individual hospital t-SNE plots...")
    
    # Original features by hospital
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
    
    # Adversarial features by hospital
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
    
    # 4. Individual disease visualization
    print("Creating individual disease t-SNE plots...")
    
    # Original features by disease
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
    
    # Adversarial features by disease
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    features, metadata_df = load_features_and_metadata(args.features_dir, args.patches_csv)
    
    # Train model
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
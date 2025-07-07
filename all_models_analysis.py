#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

def extract_hosp_from_filename(filename):
    """Extract hospital information from filename"""
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[1]
    return "Unknown"

def load_features_and_metadata(features_dir, patches_csv_path, model_name):
    """Load features and metadata for a specific model"""
    print(f"\nLoading {model_name} features...")
    
    patches_df = pd.read_csv(patches_csv_path)
    patches_df['hosp'] = patches_df['wsi_id'].apply(extract_hosp_from_filename)
    
    feature_files = glob.glob(os.path.join(features_dir, "*_features.pt"))
    
    if len(feature_files) == 0:
        print(f"  Warning: No feature files found in {features_dir}")
        return None, None
    
    print(f"  Found {len(feature_files)} feature files")
    
    all_features = []
    all_metadata = []
    
    for feature_file in tqdm(feature_files, desc=f"Loading {model_name}", leave=False):
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
            print(f"  Error loading {feature_file}: {e}")
            continue
    
    if not all_features:
        print(f"  Warning: No valid features loaded for {model_name}")
        return None, None
    
    features_array = np.vstack(all_features)
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    print(f"  Loaded {len(features_array)} samples with {features_array.shape[1]} dimensions")
    
    return features_array, metadata_df

def analyze_single_model(features, metadata_df, model_name):
    """Complete analysis for a single model"""
    print(f"\n{'='*60}")
    print(f"Complete Analysis: {model_name}")
    print('='*60)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    le_hosp = LabelEncoder()
    hospital_encoded = le_hosp.fit_transform(metadata_df['hosp'])
    le_diag = LabelEncoder()
    diagnosis_encoded = le_diag.fit_transform(metadata_df['patch_label'])
    
    results = {'hospital_classification': {}}
    
    # Hospital Source Classification by Diagnosis (5-Fold CV)
    print("\nHospital Source Classification by Diagnosis (5-Fold CV)")
    print("-" * 50)
    
    diagnoses = metadata_df['patch_label'].unique()
    
    for diagnosis in diagnoses:
        print(f"\n  {diagnosis} diagnosis...")
        
        diag_mask = metadata_df['patch_label'] == diagnosis
        diag_features = features_scaled[diag_mask]
        diag_hospitals = hospital_encoded[diag_mask]
        
        unique_hospitals = np.unique(diag_hospitals)
        if len(unique_hospitals) < 2 or len(diag_features) < 20:
            print(f"    Warning: Insufficient data")
            continue
        
        print(f"    Debug: {len(diag_features)} samples, {len(unique_hospitals)} hospitals: {unique_hospitals}")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        all_true_labels = []
        all_predictions = []
        all_pred_proba = []
        
        for train_idx, test_idx in skf.split(diag_features, diag_hospitals):
            X_train = diag_features[train_idx]
            X_test = diag_features[test_idx]
            y_train = diag_hospitals[train_idx]
            y_test = diag_hospitals[test_idx]
            
            clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            cv_scores.append(accuracy_score(y_test, y_pred))
            
            all_true_labels.extend(y_test)
            all_predictions.extend(y_pred)
            all_pred_proba.extend(y_pred_proba)
        
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        
        # Calculate detailed per-hospital metrics
        cm = confusion_matrix(all_true_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_labels, all_predictions, average=None, zero_division=0
        )
        
        # Calculate AUC metrics using probability predictions
        if len(all_pred_proba) > 0:
            try:
                # Convert list of arrays to single array
                pred_proba_array = np.vstack(all_pred_proba)
                
                if len(unique_hospitals) == 2:
                    # Binary classification - use probability of positive class
                    roc_auc = roc_auc_score(all_true_labels, pred_proba_array[:, 1])
                    avg_precision = average_precision_score(all_true_labels, pred_proba_array[:, 1])
                else:
                    # Multi-class classification - use one-vs-rest
                    roc_auc = roc_auc_score(all_true_labels, pred_proba_array, multi_class='ovr', average='macro')
                    avg_precision = average_precision_score(all_true_labels, pred_proba_array, average='macro')
            except Exception as e:
                print(f"      Warning: AUC calculation failed: {e}")
                roc_auc = None
                avg_precision = None
        else:
            print(f"      Warning: No probability predictions available for AUC calculation")
            roc_auc = None
            avg_precision = None
        
        # Get hospital names from label encoder
        hospital_names = le_hosp.classes_
        
        # Create detailed metrics dictionary
        hospital_metrics = {}
        for i, hosp_name in enumerate(hospital_names):
            if i < len(precision):  # Check if this hospital exists in predictions
                hospital_metrics[hosp_name] = {
                    'precision': precision[i],
                    'recall': recall[i], 
                    'f1_score': f1[i],
                    'support': support[i]
                }
        
        results['hospital_classification'][diagnosis] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'cv_scores': cv_scores,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'n_samples': len(diag_features),
            'n_hospitals': len(unique_hospitals),
            'y_true': all_true_labels,
            'y_pred': all_predictions,
            'confusion_matrix': cm,
            'hospital_metrics': hospital_metrics,
            'hospital_names': hospital_names
        }
        
        auc_info = f" (AUC: {roc_auc:.3f})" if roc_auc is not None else " (AUC: N/A)"
        print(f"    {diagnosis}: {mean_accuracy:.3f} ± {std_accuracy:.3f}{auc_info}")
    
    # Store label encoders for later use
    results['label_encoders'] = {'hospital': le_hosp, 'diagnosis': le_diag}
    
    return results

def analyze_all_models(input_dir):
    """Analyze all available models"""
    print("Complete Analysis of All Models")
    print("="*60)
    
    patches_csv = os.path.join(input_dir, "conch_filtered_patches.csv")
    if not os.path.exists(patches_csv):
        raise FileNotFoundError(f"Patches CSV not found: {patches_csv}")
    
    feature_dirs = glob.glob(os.path.join(input_dir, "features_*"))
    feature_dirs = [d for d in feature_dirs if os.path.isdir(d)]
    
    if not feature_dirs:
        raise ValueError(f"No feature directories found in {input_dir}")
    
    print(f"\nFound {len(feature_dirs)} feature directories:")
    for feature_dir in feature_dirs:
        model_name = os.path.basename(feature_dir).replace("features_", "")
        print(f"  - {model_name}")
    
    all_results = {}
    
    for feature_dir in feature_dirs:
        model_name = os.path.basename(feature_dir).replace("features_", "")
        
        try:
            features, metadata_df = load_features_and_metadata(feature_dir, patches_csv, model_name)
            
            if features is None:
                print(f"Warning: Skipping {model_name}: failed to load data")
                continue
            
            model_results = analyze_single_model(features, metadata_df, model_name)
            all_results[model_name] = model_results
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            continue
    
    return all_results

def create_visualizations(all_results, output_dir):
    """Create comprehensive visualizations"""
    print(f"\n{'='*60}")
    print("Creating Visualizations")
    print('='*60)
    
    if not all_results:
        print("Warning: No results to visualize")
        return
    
    viz_dir = os.path.join(output_dir, "all_models_analysis")
    os.makedirs(viz_dir, exist_ok=True)
    
    models = list(all_results.keys())
    diagnoses = set()
    
    for model_results in all_results.values():
        if 'hospital_classification' in model_results:
            diagnoses.update(model_results['hospital_classification'].keys())
    diagnoses = sorted(list(diagnoses))
    
    # 1. Hospital classification heatmap
    plt.figure(figsize=(12, 8))
    
    accuracy_matrix = np.full((len(models), len(diagnoses)), np.nan)
    for i, model in enumerate(models):
        for j, diagnosis in enumerate(diagnoses):
            if ('hospital_classification' in all_results[model] and 
                diagnosis in all_results[model]['hospital_classification']):
                accuracy_matrix[i, j] = all_results[model]['hospital_classification'][diagnosis]['mean_accuracy']
    
    mask = np.isnan(accuracy_matrix)
    sns.heatmap(accuracy_matrix, 
                xticklabels=diagnoses, 
                yticklabels=models,
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                mask=mask,
                cbar_kws={'label': 'Hospital Classification Accuracy'},
                vmin=0, vmax=1)
    
    plt.title('Hospital Source Classification Accuracy Across All Models\n(Higher = More Hospital Information in Features)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Diagnosis Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    heatmap_file = os.path.join(viz_dir, "hospital_source_classification_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {heatmap_file}")
    
    # 2. Model ranking
    plt.figure(figsize=(12, 6))
    
    model_avg_accuracy = {}
    for model in models:
        if 'hospital_classification' in all_results[model]:
            accuracies = [result['mean_accuracy'] 
                         for result in all_results[model]['hospital_classification'].values()]
            if accuracies:
                model_avg_accuracy[model] = np.mean(accuracies)
    
    sorted_models = sorted(model_avg_accuracy.items(), key=lambda x: x[1])
    model_names = [item[0] for item in sorted_models]
    avg_accuracy = [item[1] for item in sorted_models]
    
    bars = plt.bar(range(len(model_names)), avg_accuracy, color='lightcoral', alpha=0.8)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Hospital Source Classification Accuracy', fontsize=12)
    plt.title('Model Ranking by Average Hospital Source Classification Accuracy\n(Lower = Less Hospital Information)', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars, avg_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    ranking_file = os.path.join(viz_dir, "model_ranking.png")
    plt.savefig(ranking_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ranking plot saved: {ranking_file}")
    
    # 3. Generate confusion matrices for each model and diagnosis
    for model in models:
        if 'hospital_classification' not in all_results[model]:
            continue
            
        for diagnosis, result in all_results[model]['hospital_classification'].items():
            if 'confusion_matrix' not in result or 'hospital_names' not in result:
                continue
                
            plt.figure(figsize=(8, 6))
            cm = result['confusion_matrix']
            hospital_names = result['hospital_names']
            
            # Create confusion matrix heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=hospital_names,
                       yticklabels=hospital_names,
                       square=True)
            
            plt.title(f'{model} - {diagnosis} Diagnosis\nHospital Source Classification Confusion Matrix\n' + 
                     f'Overall Accuracy: {result["mean_accuracy"]:.3f} ± {result["std_accuracy"]:.3f}', 
                     fontsize=12, fontweight='bold')
            plt.xlabel('Predicted Hospital', fontsize=10)
            plt.ylabel('Actual Hospital', fontsize=10)
            
            # Add per-hospital metrics as text
            if 'hospital_metrics' in result:
                metrics_text = ""
                for hosp, metrics in result['hospital_metrics'].items():
                    metrics_text += f"{hosp}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1_score']:.2f}\n"
                
                plt.figtext(0.02, 0.02, metrics_text, fontsize=8, family='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            cm_file = os.path.join(viz_dir, f"{model}_{diagnosis}_confusion_matrix.png")
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved: {cm_file}")
    
    print(f"All visualizations completed")

def save_results(all_results, output_dir):
    """Save comprehensive results"""
    print(f"\n{'='*60}")
    print("Saving Results")
    print('='*60)
    
    results_dir = os.path.join(output_dir, "all_models_analysis")
    os.makedirs(results_dir, exist_ok=True)
    
    # Summary report
    summary_file = os.path.join(results_dir, "complete_analysis_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("COMPLETE MODEL ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Model rankings
        models = list(all_results.keys())
        model_avg_accuracy = {}
        
        for model in models:
            if 'hospital_classification' in all_results[model]:
                accuracies = [result['mean_accuracy'] 
                             for result in all_results[model]['hospital_classification'].values()]
                if accuracies:
                    model_avg_accuracy[model] = np.mean(accuracies)
        
        sorted_models = sorted(model_avg_accuracy.items(), key=lambda x: x[1])
        
        f.write("Model Rankings (Hospital Source Classification Accuracy):\n")
        f.write("-" * 50 + "\n")
        for rank, (model, avg_accuracy) in enumerate(sorted_models, 1):
            f.write(f"{rank:2d}. {model:<15} {avg_accuracy:.3f}\n")
        
        # Add AUC rankings
        model_avg_auc = {}
        for model in models:
            if 'hospital_classification' in all_results[model]:
                aucs = [result['roc_auc'] 
                       for result in all_results[model]['hospital_classification'].values()
                       if result['roc_auc'] is not None]
                if aucs:
                    model_avg_auc[model] = np.mean(aucs)
        
        if model_avg_auc:
            sorted_models_auc = sorted(model_avg_auc.items(), key=lambda x: x[1], reverse=True)
            f.write("\nModel Rankings (Hospital Source Classification AUC):\n")
            f.write("-" * 50 + "\n")
            for rank, (model, avg_auc) in enumerate(sorted_models_auc, 1):
                f.write(f"{rank:2d}. {model:<15} {avg_auc:.3f}\n")
        
        f.write("\n\nDetailed Results:\n")
        f.write("="*30 + "\n")
        
        for model, model_results in all_results.items():
            f.write(f"\n{model}:\n")
            f.write("-" * len(model) + "\n")
            
            if 'hospital_classification' in model_results:
                f.write("  Hospital Source Classification by Diagnosis:\n")
                for diagnosis, result in model_results['hospital_classification'].items():
                    auc_info = f", AUC: {result['roc_auc']:.3f}" if result['roc_auc'] is not None else ", AUC: N/A"
                    f.write(f"    {diagnosis}: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}{auc_info}\n")
                    f.write(f"      (n_samples={result['n_samples']}, n_hospitals={result['n_hospitals']})\n")
                    
                    # Add detailed per-hospital metrics
                    if 'hospital_metrics' in result:
                        f.write(f"      Detailed Hospital Classification Results:\n")
                        for hosp, metrics in result['hospital_metrics'].items():
                            f.write(f"        {hosp}: precision={metrics['precision']:.3f}, ")
                            f.write(f"recall={metrics['recall']:.3f}, ")
                            f.write(f"f1={metrics['f1_score']:.3f}, ")
                            f.write(f"support={metrics['support']}\n")
                        
                        # Add confusion matrix
                        if 'confusion_matrix' in result and 'hospital_names' in result:
                            f.write(f"      Confusion Matrix:\n")
                            cm = result['confusion_matrix']
                            hosp_names = result['hospital_names']
                            
                            # Header
                            f.write(f"                Predicted\n")
                            f.write(f"        Actual  ")
                            for name in hosp_names:
                                f.write(f"{name:>6}")
                            f.write(f"\n")
                            
                            # Matrix rows
                            for i, actual_hosp in enumerate(hosp_names):
                                f.write(f"        {actual_hosp:<6}  ")
                                for j in range(len(hosp_names)):
                                    if i < cm.shape[0] and j < cm.shape[1]:
                                        f.write(f"{cm[i,j]:>6}")
                                    else:
                                        f.write(f"{'0':>6}")
                                f.write(f"\n")
                    f.write("\n")
                f.write("\n")
    
    print(f"Summary report saved: {summary_file}")
    
    # CSV files - Overall results
    hospital_csv_data = []
    for model, model_results in all_results.items():
        if 'hospital_classification' in model_results:
            for diagnosis, result in model_results['hospital_classification'].items():
                hospital_csv_data.append({
                    'model': model,
                    'diagnosis': diagnosis,
                    'mean_accuracy': result['mean_accuracy'],
                    'std_accuracy': result['std_accuracy'],
                    'roc_auc': result['roc_auc'],
                    'average_precision': result['average_precision'],
                    'n_samples': result['n_samples'],
                    'n_hospitals': result['n_hospitals']
                })
    
    hospital_csv = os.path.join(results_dir, "hospital_source_classification_results.csv")
    pd.DataFrame(hospital_csv_data).to_csv(hospital_csv, index=False)
    print(f"Hospital source classification CSV saved: {hospital_csv}")
    
    # CSV files - Detailed per-hospital metrics
    detailed_csv_data = []
    for model, model_results in all_results.items():
        if 'hospital_classification' in model_results:
            for diagnosis, result in model_results['hospital_classification'].items():
                if 'hospital_metrics' in result:
                    for hosp, metrics in result['hospital_metrics'].items():
                        detailed_csv_data.append({
                            'model': model,
                            'diagnosis': diagnosis,
                            'hospital': hosp,
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1_score': metrics['f1_score'],
                            'support': metrics['support']
                        })
    
    detailed_csv = os.path.join(results_dir, "detailed_hospital_metrics.csv")
    pd.DataFrame(detailed_csv_data).to_csv(detailed_csv, index=False)
    print(f"Detailed hospital metrics CSV saved: {detailed_csv}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete Analysis of All Models')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing multiple features_* subdirectories')
    
    args = parser.parse_args()
    
    print("COMPLETE ANALYSIS OF ALL MODELS")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    
    try:
        # Analyze all models
        all_results = analyze_all_models(args.input_dir)
        
        if not all_results:
            print("Warning: No valid model results obtained")
            return
        
        # Create visualizations
        create_visualizations(all_results, args.input_dir)
        
        # Save results
        save_results(all_results, args.input_dir)
        
        print(f"\n{'='*60}")
        print("COMPLETE ANALYSIS FINISHED!")
        print(f"Results saved in: {os.path.join(args.input_dir, 'all_models_analysis')}")
        print('='*60)
        
        # Final summary
        print(f"\nFINAL MODEL RANKINGS:")
        
        models = list(all_results.keys())
        model_avg_accuracy = {}
        
        for model in models:
            if 'hospital_classification' in all_results[model]:
                accuracies = [result['mean_accuracy'] 
                             for result in all_results[model]['hospital_classification'].values()]
                if accuracies:
                    model_avg_accuracy[model] = np.mean(accuracies)
        
        sorted_models = sorted(model_avg_accuracy.items(), key=lambda x: x[1])
        
        for rank, (model, avg_accuracy) in enumerate(sorted_models, 1):
            print(f"  {rank:2d}. {model:<15} {avg_accuracy:.3f}")
        
        # Print AUC rankings
        model_avg_auc = {}
        for model in models:
            if 'hospital_classification' in all_results[model]:
                aucs = [result['roc_auc'] 
                       for result in all_results[model]['hospital_classification'].values()
                       if result['roc_auc'] is not None]
                if aucs:
                    model_avg_auc[model] = np.mean(aucs)
        
        if model_avg_auc:
            sorted_models_auc = sorted(model_avg_auc.items(), key=lambda x: x[1], reverse=True)
            print(f"\nFINAL MODEL RANKINGS (by AUC):")
            for rank, (model, avg_auc) in enumerate(sorted_models_auc, 1):
                print(f"  {rank:2d}. {model:<15} {avg_auc:.3f}")
        
        print(f"\nANALYSIS COMPLETED:")
        print(f"  - {len(all_results)} models analyzed")
        print(f"  - Hospital source classification by diagnosis")
        print(f"  - 5-fold cross-validation for each diagnosis")
        print(f"  - AUC and Average Precision metrics included")
        print(f"  - Comprehensive visualizations and reports generated")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

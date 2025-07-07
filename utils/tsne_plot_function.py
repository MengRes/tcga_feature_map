import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

def create_tsne_visualization(features, labels, hospitals, output_path, title="t-SNE Visualization"):
    """Create t-SNE visualization with labels and hospitals
    Args:
        features (np.ndarray): Feature matrix
        labels (list): List of labels
        hospitals (list): List of hospital codes
        output_path (str): Path to save the plot
        title (str): Plot title
    Returns:
        pd.DataFrame: DataFrame with t-SNE results and metadata
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features)
    df = pd.DataFrame({
        'tSNE1': tsne_result[:, 0],
        'tSNE2': tsne_result[:, 1],
        'Label': labels,
        'Hospital': hospitals
    })
    plt.figure(figsize=(12, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = df['Label'] == label
        plt.scatter(df[mask]['tSNE1'], df[mask]['tSNE2'], 
                   c=[colors[i]], label=f'Label {label}', alpha=0.7, s=50)
    plt.title(f"{title} - By Label")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return df

def create_hospital_tsne_visualization(features, labels, hospitals, output_path, title="t-SNE Visualization"):
    """Create t-SNE visualization colored by hospital
    Args:
        features (np.ndarray): Feature matrix
        labels (list): List of labels
        hospitals (list): List of hospital codes
        output_path (str): Path to save the plot
        title (str): Plot title
    Returns:
        pd.DataFrame: DataFrame with t-SNE results and metadata
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features)
    df = pd.DataFrame({
        'tSNE1': tsne_result[:, 0],
        'tSNE2': tsne_result[:, 1],
        'Label': labels,
        'Hospital': hospitals
    })
    plt.figure(figsize=(12, 8))
    unique_hospitals = sorted(set(hospitals))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_hospitals)))
    for i, hospital in enumerate(unique_hospitals):
        mask = df['Hospital'] == hospital
        plt.scatter(df[mask]['tSNE1'], df[mask]['tSNE2'], 
                   c=[colors[i]], label=hospital, alpha=0.7, s=50)
    plt.title(f"{title} - By Hospital")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return df

def create_label_hospital_tsne_visualization(features, labels, hospitals, output_path, title="t-SNE Visualization"):
    """Create t-SNE visualization with both label and hospital information
    Args:
        features (np.ndarray): Feature matrix
        labels (list): List of labels
        hospitals (list): List of hospital codes
        output_path (str): Path to save the plot
        title (str): Plot title
    Returns:
        pd.DataFrame: DataFrame with t-SNE results and metadata
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features)
    df = pd.DataFrame({
        'tSNE1': tsne_result[:, 0],
        'tSNE2': tsne_result[:, 1],
        'Label': labels,
        'Hospital': hospitals
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = df['Label'] == label
        ax1.scatter(df[mask]['tSNE1'], df[mask]['tSNE2'], 
                   c=[colors[i]], label=f'Label {label}', alpha=0.7, s=50)
    ax1.set_title(f"{title} - By Label")
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    unique_hospitals = sorted(set(hospitals))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_hospitals)))
    for i, hospital in enumerate(unique_hospitals):
        mask = df['Hospital'] == hospital
        ax2.scatter(df[mask]['tSNE1'], df[mask]['tSNE2'], 
                   c=[colors[i]], label=hospital, alpha=0.7, s=50)
    ax2.set_title(f"{title} - By Hospital")
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return df 
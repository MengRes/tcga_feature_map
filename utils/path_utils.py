import re
import os
import numpy as np

def extract_hospital_info(file_path):
    """Extract hospital information from file path
    Args:
        file_path (str): Path containing hospital info (e.g. .../TCGA-XX-XXXX-01Z-00-DX1.XXXXX.svs)
    Returns:
        str: Hospital code or "Unknown"
    """
    match = re.search(r'TCGA-([A-Z0-9]+)-', file_path)
    if match:
        return match.group(1)
    return "Unknown"

def extract_label_from_path(file_path):
    """Extract label information from file path
    Args:
        file_path (str): Path containing label info (e.g. .../label_0/...)
    Returns:
        int or None: Label value if found, else None
    """
    match = re.search(r'label_(\d+)', file_path)
    if match:
        return int(match.group(1))
    return None

def load_features_from_folder(feature_folder):
    """Load features from folder containing .npy files
    Args:
        feature_folder (str): Path to folder with .npy feature files
    Returns:
        tuple: (features, file_paths, labels, hospitals)
    """
    features = []
    file_paths = []
    labels = []
    hospitals = []
    
    for root, dirs, files in os.walk(feature_folder):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                feature = np.load(file_path)
                features.append(feature)
                file_paths.append(file_path)
                
                # Extract label and hospital info
                label = extract_label_from_path(file_path)
                hospital = extract_hospital_info(file_path)
                
                labels.append(label)
                hospitals.append(hospital)
    
    return np.array(features), file_paths, labels, hospitals 
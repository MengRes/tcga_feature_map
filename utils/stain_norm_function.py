#!/usr/bin/env python3
"""
Stain Normalization Functions
============================

This module contains functions for stain normalization in histopathology images.
Includes Macenko, Vahadane, and HistAug methods.
"""

import numpy as np
import cv2
from skimage import exposure, color
from PIL import Image
import random


def rgb_to_od(image):
    """Convert RGB image to optical density space
    Args: image: PIL Image or numpy array   
    Returns: numpy.ndarray: Optical density image
    """
    image = np.asarray(image, dtype=np.float64)
    image = np.maximum(image, 1)  # Avoid log(0)
    od = -np.log(image / 255.0)
    return od


def od_to_rgb(od):
    """Convert optical density image back to RGB space
    Args: od: numpy.ndarray, Optical density image   
    Returns: numpy.ndarray: RGB image (uint8)
    """
    od = np.maximum(od, 0)
    rgb = np.exp(-od) * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def normalize_stain_macenko(image, target_concentrations=None, target_stains=None):
    """Stain normalization using Macenko method
    Args:
        image: PIL Image or numpy array
        target_concentrations: Target stain concentrations (if None, use defaults)
        target_stains: Target stain matrix (if None, use defaults)
    Returns:
        PIL Image: Normalized image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Default H&E stain vectors (from typical H&E staining)
    if target_stains is None:
        target_stains = np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11]   # Eosin
        ]).T
    
    if target_concentrations is None:
        target_concentrations = np.array([
            [1.9705, 1.0308],  # 99th percentiles for H&E
        ]).T
    
    # Convert to optical density
    od = rgb_to_od(image)
    
    # Reshape for matrix operations
    od_reshaped = od.reshape(-1, 3).T
    
    # SVD on optical density
    U, s, Vt = np.linalg.svd(od_reshaped, full_matrices=False)
    
    # Extract stain vectors (first two components)
    stain_vectors = U[:, :2]
    
    # Normalize stain vectors
    stain_vectors[:, 0] = stain_vectors[:, 0] / np.linalg.norm(stain_vectors[:, 0])
    stain_vectors[:, 1] = stain_vectors[:, 1] / np.linalg.norm(stain_vectors[:, 1])
    
    # Project OD onto stain space
    concentrations = np.linalg.lstsq(stain_vectors, od_reshaped, rcond=None)[0]
    
    # Normalize concentrations
    max_concentrations = np.percentile(concentrations, 99, axis=1, keepdims=True)
    concentrations = concentrations / max_concentrations * target_concentrations
    
    # Reconstruct with target stains
    normalized_od = target_stains @ concentrations
    normalized_od = normalized_od.T.reshape(od.shape)
    
    # Convert back to RGB
    normalized_rgb = od_to_rgb(normalized_od)
    
    return Image.fromarray(normalized_rgb)


def normalize_stain_vahadane(image):
    """Stain normalization using Vahadane method (simplified implementation)
    Args:
        image: PIL Image or numpy array
    Returns:
        PIL Image: Normalized image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to LAB color space for better separation
    lab = color.rgb2lab(image / 255.0)
    
    # Apply histogram matching on L channel
    target_l = 0.7  # Target lightness
    lab[:, :, 0] = exposure.match_histograms(lab[:, :, 0], 
                                           np.full_like(lab[:, :, 0], target_l))
    
    # Convert back to RGB
    normalized_rgb = color.lab2rgb(lab)
    normalized_rgb = (normalized_rgb * 255).astype(np.uint8)
    
    return Image.fromarray(normalized_rgb)


def apply_histaug(image, intensity=0.3):
    """Apply HistAug: Histogram-based stain augmentation
    Args:
        image: PIL Image or numpy array
        intensity: Augmentation intensity (0.0 - 1.0)
    Returns:
        PIL Image: Augmented image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Random hue shift (simulate different staining variations)
    hue_shift = np.random.uniform(-intensity * 20, intensity * 20)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    # Random saturation change
    saturation_factor = np.random.uniform(1 - intensity * 0.3, 1 + intensity * 0.3)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    # Random value (brightness) change
    value_factor = np.random.uniform(1 - intensity * 0.2, 1 + intensity * 0.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_factor, 0, 255)
    # Convert back to RGB
    hsv = hsv.astype(np.uint8)
    augmented_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(augmented_rgb)


def preprocess_patch_with_stain_normalization(patch, config):
    """Apply stain normalization or HistAug to patch
    Args:
        patch: PIL Image
        config: Configuration object with stain normalization parameters
    Returns:
        PIL Image: Processed patch
    """
    if not config.enable_stain_normalization:
        return patch
    try:
        if config.stain_normalization_method == "macenko":
            return normalize_stain_macenko(patch)
        elif config.stain_normalization_method == "vahadane":
            return normalize_stain_vahadane(patch)
        elif config.stain_normalization_method == "histaug":
            # Apply HistAug with probability
            if random.random() < config.histaug_probability:
                return apply_histaug(patch, config.histaug_intensity)
            else:
                return patch
        else:
            print(f"Warning: Unknown stain normalization method: {config.stain_normalization_method}")
            return patch
    except Exception as e:
        print(f"Warning: Stain normalization failed, using original patch: {e}")
        return patch 
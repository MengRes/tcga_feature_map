from sklearn.preprocessing import StandardScaler

def perform_stain_normalization(features, method='standard'):
    """Perform stain normalization on features
    Args:
        features (np.ndarray): Feature matrix
        method (str): Normalization method ('standard', 'minmax', or None)
    Returns:
        np.ndarray: Normalized features
    """
    if method == 'standard':
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = features
    
    return normalized_features 
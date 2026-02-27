"""
Plasticity-based connectivity updates
"""
import numpy as np


def apply_plasticity_updates(adjacency_matrix, activation_ratios, scaling=1.0):
    """
    Update connectivity matrix based on node activation changes (plasticity).
    
    Each edge E(i,j) is updated based on the activation changes of both nodes:
        E_new(i,j) = E(i,j) * (R_i * R_j)^scaling
    
    Where R_i and R_j are the activation ratios for nodes i and j.
    
    This implements Hebbian-like plasticity: connections between co-activated
    nodes are strengthened.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n_nodes, n_nodes)
        Original connectivity matrix
    activation_ratios : ndarray, shape (n_nodes,)
        Ratio of activation change for each node
        (1.0 = no change, >1.0 = increase, <1.0 = decrease)
    scaling : float
        Scaling factor for plasticity strength (default: 1.0)
        - scaling = 1.0: full plasticity effect
        - scaling < 1.0: reduced plasticity
        - scaling > 1.0: amplified plasticity
        
    Returns
    -------
    updated_matrix : ndarray, shape (n_nodes, n_nodes)
        Updated connectivity matrix after plasticity
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Create matrix of pairwise activation products
    # plasticity_factor[i,j] = (R_i * R_j)^scaling
    ratio_matrix = np.outer(activation_ratios, activation_ratios)
    plasticity_factor = np.power(ratio_matrix, scaling)
    
    # Apply plasticity: element-wise multiplication
    updated_matrix = adjacency_matrix * plasticity_factor
    
    return updated_matrix


def normalize_connectivity_matrix(matrix, method='minmax'):
    """
    Normalize connectivity matrix to [0, 1] range.
    
    Parameters
    ----------
    matrix : ndarray, shape (n_channels, n_channels)
        Connectivity matrix
    method : str
        Normalization method:
        - 'minmax': normalize to [0, 1] using min-max scaling
        - 'maxabs': divide by maximum absolute value
        - 'zscore': z-score normalization (may have negative values)
        
    Returns
    -------
    normalized : ndarray
        Normalized connectivity matrix
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        
        if max_val - min_val < 1e-10:
            # If all values are the same, return zeros
            return np.zeros_like(matrix)
        
        normalized = (matrix - min_val) / (max_val - min_val)
        
    elif method == 'maxabs':
        # Normalize by maximum absolute value
        max_abs = np.max(np.abs(matrix))
        
        if max_abs < 1e-10:
            return np.zeros_like(matrix)
        
        normalized = matrix / max_abs
        
    elif method == 'zscore':
        # Z-score normalization
        mean_val = np.mean(matrix)
        std_val = np.std(matrix)
        
        if std_val < 1e-10:
            return np.zeros_like(matrix)
        
        normalized = (matrix - mean_val) / std_val
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def compute_plasticity_effect(adjacency_matrix, activation_ratios, 
                              normalize=True, scaling=1.0):
    """
    Complete pipeline for applying plasticity and normalizing connectivity.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n_nodes, n_nodes)
        Original connectivity matrix
    activation_ratios : ndarray, shape (n_nodes,)
        Ratio of activation change for each node
    normalize : bool
        Whether to normalize the updated matrix (default: True)
    scaling : float
        Plasticity scaling factor (default: 1.0)
        
    Returns
    -------
    updated_matrix : ndarray, shape (n_nodes, n_nodes)
        Updated (and optionally normalized) connectivity matrix
    """
    # Apply plasticity updates
    updated_matrix = apply_plasticity_updates(adjacency_matrix, activation_ratios, scaling)
    
    # Normalize if requested
    if normalize:
        updated_matrix = normalize_connectivity_matrix(updated_matrix, method='minmax')
    
    return updated_matrix


def analyze_plasticity_changes(original_matrix, updated_matrix):
    """
    Analyze the changes in connectivity after plasticity.
    
    Parameters
    ----------
    original_matrix : ndarray, shape (n_nodes, n_nodes)
        Original connectivity matrix
    updated_matrix : ndarray, shape (n_nodes, n_nodes)
        Updated connectivity matrix after plasticity
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'mean_change': Mean absolute change in connectivity
        - 'max_increase': Maximum increase in connectivity
        - 'max_decrease': Maximum decrease in connectivity
        - 'n_increased': Number of connections that increased
        - 'n_decreased': Number of connections that decreased
        - 'relative_change': Mean relative change (as ratio)
    """
    # Compute changes
    absolute_change = updated_matrix - original_matrix
    relative_change = (updated_matrix - original_matrix) / (original_matrix + 1e-10)
    
    # Get statistics
    stats = {
        'mean_change': np.mean(np.abs(absolute_change)),
        'max_increase': np.max(absolute_change),
        'max_decrease': np.min(absolute_change),
        'n_increased': np.sum(absolute_change > 0),
        'n_decreased': np.sum(absolute_change < 0),
        'relative_change': np.mean(relative_change[np.abs(original_matrix) > 1e-10])
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    # Create example connectivity matrix
    n_nodes = 5
    A = np.random.rand(n_nodes, n_nodes) * 0.5
    A = (A + A.T) / 2  # Make symmetric
    
    # Create activation ratios (some nodes increase, some decrease)
    activation_ratios = np.array([1.2, 0.8, 1.5, 1.0, 0.9])
    
    print("Original matrix:")
    print(A)
    print("\nActivation ratios:")
    print(activation_ratios)
    
    # Apply plasticity
    updated = compute_plasticity_effect(A, activation_ratios, normalize=True)
    
    print("\nUpdated matrix:")
    print(updated)
    
    # Analyze changes
    stats = analyze_plasticity_changes(A, updated)
    print("\nPlasticity statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

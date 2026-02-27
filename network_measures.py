"""
Network measures computation using Brain Connectivity Toolbox (bctpy)
"""

import numpy as np
import bct
from config import NETWORK_MEASURES

def compute_global_efficiency(adjacency_matrix):
    """
    Compute global efficiency of the network.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    efficiency : float
        Global efficiency
    """
    return bct.efficiency_wei(adjacency_matrix)


def compute_local_efficiency(adjacency_matrix):
    """
    Compute local efficiency (average across nodes).
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    efficiency : float
        Average local efficiency
    """
    local_eff = bct.efficiency_wei(adjacency_matrix, local=True)
    return np.mean(local_eff)


def compute_clustering_coefficient(adjacency_matrix):
    """
    Compute average clustering coefficient.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    clustering : float
        Average clustering coefficient
    """
    cc = bct.clustering_coef_wu(adjacency_matrix)
    return np.mean(cc)


def compute_transitivity(adjacency_matrix):
    """
    Compute transitivity of the network.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    transitivity : float
        Network transitivity
    """
    return bct.transitivity_wu(adjacency_matrix)


def compute_modularity(adjacency_matrix):
    """
    Compute modularity using Louvain algorithm.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    modularity : float
        Modularity Q value
    """
    try:
        _, Q = bct.community_louvain(adjacency_matrix)
        return Q
    except:
        # If modularity computation fails, return NaN
        return np.nan


def compute_degree(adjacency_matrix):
    """
    Compute average weighted degree.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    degree : float
        Average degree
    """
    degrees = bct.degrees_und(adjacency_matrix)
    return np.mean(degrees)


def compute_betweenness_centrality(adjacency_matrix):
    """
    Compute average betweenness centrality.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    betweenness : float
        Average betweenness centrality
    """
    # Convert to connection-length matrix (inverse weights)
    # Avoid division by zero
    length_matrix = np.where(adjacency_matrix > 0, 1.0 / adjacency_matrix, 0)
    bc = bct.betweenness_wei(length_matrix)
    return np.mean(bc)


def compute_rich_club(adjacency_matrix, k=None):
    """
    Compute rich club coefficient.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
    k : int, optional
        Degree threshold. If None, uses median degree.
        
    Returns
    -------
    rich_club : float
        Rich club coefficient at degree k
    """
    try:
        degrees = bct.degrees_und(adjacency_matrix)
        if k is None:
            k = int(np.median(degrees))
        
        # Binarize for rich club
        binary_matrix = (adjacency_matrix > 0).astype(int)
        rc = bct.rich_club_wu(adjacency_matrix, klevel=k)
        
        if len(rc) > 0 and not np.isnan(rc[0]):
            return rc[0]
        else:
            return np.nan
    except:
        return np.nan


def compute_assortativity(adjacency_matrix):
    """
    Compute assortativity coefficient.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    assortativity : float
        Assortativity coefficient
    """
    try:
        return bct.assortativity_wei(adjacency_matrix, flag=0)
    except:
        return np.nan


def compute_spectral_radius(adjacency_matrix):
    """
    Compute spectral radius (largest eigenvalue).
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    spectral_radius : float
        Largest eigenvalue magnitude
    """
    eigenvalues = np.linalg.eigvals(adjacency_matrix)
    return np.max(np.abs(eigenvalues))


def compute_small_worldness(adjacency_matrix):
    """
    Compute small-worldness coefficient.
    
    Small-worldness = (C/C_random) / (L/L_random)
    where C is clustering and L is path length
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    small_worldness : float
        Small-worldness coefficient
    """
    try:
        # Real network measures
        C_real = np.mean(bct.clustering_coef_wu(adjacency_matrix))
        
        # Convert to length matrix for path length computation
        length_matrix = np.where(adjacency_matrix > 0, 1.0 / adjacency_matrix, np.inf)
        D = bct.distance_wei(length_matrix)[0]
        L_real = np.mean(D[D != np.inf])
        
        # Generate random network with same density
        n_nodes = adjacency_matrix.shape[0]
        n_edges = np.sum(adjacency_matrix > 0) // 2
        density = n_edges / (n_nodes * (n_nodes - 1) / 2)
        
        # Random network
        rand_matrix = np.random.rand(n_nodes, n_nodes)
        rand_matrix = (rand_matrix + rand_matrix.T) / 2  # Symmetrize
        threshold = np.percentile(rand_matrix, (1 - density) * 100)
        rand_matrix = np.where(rand_matrix > threshold, rand_matrix, 0)
        np.fill_diagonal(rand_matrix, 0)
        
        C_rand = np.mean(bct.clustering_coef_wu(rand_matrix))
        
        rand_length = np.where(rand_matrix > 0, 1.0 / rand_matrix, np.inf)
        D_rand = bct.distance_wei(rand_length)[0]
        L_rand = np.mean(D_rand[D_rand != np.inf])
        
        # Small-worldness
        gamma = C_real / C_rand if C_rand > 0 else np.nan
        lambda_ = L_real / L_rand if L_rand > 0 else np.nan
        sigma = gamma / lambda_ if lambda_ > 0 else np.nan
        
        return sigma
    except:
        return np.nan


def compute_diameter(adjacency_matrix):
    """
    Compute network diameter (longest shortest path).
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    diameter : float
        Network diameter
    """
    try:
        # Convert to length matrix
        length_matrix = np.where(adjacency_matrix > 0, 1.0 / adjacency_matrix, np.inf)
        D = bct.distance_wei(length_matrix)[0]
        
        # Get maximum finite distance
        finite_distances = D[D != np.inf]
        if len(finite_distances) > 0:
            return np.max(finite_distances)
        else:
            return np.nan
    except:
        return np.nan


def compute_all_network_measures(adjacency_matrix):
    """
    Compute all network measures for a single connectivity matrix.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted connectivity matrix
        
    Returns
    -------
    measures : dict
        Dictionary of measure names and values
    """
    measures = {}
    
    measure_functions = {
        'global_efficiency': compute_global_efficiency,
        'local_efficiency': compute_local_efficiency,
        'clustering_coefficient': compute_clustering_coefficient,
        'transitivity': compute_transitivity,
        'modularity': compute_modularity,
        'degree': compute_degree,
        'betweenness_centrality': compute_betweenness_centrality,
        'rich_club': compute_rich_club,
        'assortativity': compute_assortativity,
        'spectral_radius': compute_spectral_radius,
        'small_worldness': compute_small_worldness,
        'diameter': compute_diameter
    }
    
    for measure_name, func in measure_functions.items():
        try:
            measures[measure_name] = func(adjacency_matrix)
        except Exception as e:
            print(f"  Warning: Failed to compute {measure_name}: {e}")
            measures[measure_name] = np.nan
    
    return measures


def compute_network_measures_for_subjects(connectivity_matrices_dict, band_names):
    """
    Compute network measures for all subjects, bands, and methods.
    
    Parameters
    ----------
    connectivity_matrices_dict : dict
        Dictionary with structure:
        {group: {subject_id: {method: {band: matrix}}}}
    band_names : list
        List of frequency band names
        
    Returns
    -------
    network_measures : dict
        Dictionary with structure:
        {group: {subject_id: {method: {band: {measure: value}}}}}
    """
    network_measures = {}
    
    for group in connectivity_matrices_dict.keys():
        network_measures[group] = {}
        
        for subject_id, subject_data in connectivity_matrices_dict[group].items():
            print(f"\nComputing network measures for {subject_id} ({group})")
            network_measures[group][subject_id] = {}
            
            for method in subject_data.keys():
                network_measures[group][subject_id][method] = {}
                
                for band in band_names:
                    conn_matrix = subject_data[method][band]
                    
                    print(f"  {method} - {band}...", end=' ')
                    measures = compute_all_network_measures(conn_matrix)
                    network_measures[group][subject_id][method][band] = measures
                    print("✓")
    
    return network_measures

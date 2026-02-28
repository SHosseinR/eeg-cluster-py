"""
Network measures computation using Brain Connectivity Toolbox (bctpy)
"""

import numpy as np
import bct
from config import NETWORK_MEASURES

def _prepare_weighted_directed_matrix(adjacency_matrix):
    """
    Prepare a weighted directed adjacency matrix for metric computation.
    """
    W = np.array(adjacency_matrix, dtype=float, copy=True)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(W, 0.0)
    # Most connectivity measures are non-negative; clip tiny negatives/noise.
    W[W < 0] = 0.0
    return W


def _to_length_matrix(weight_matrix):
    """
    Convert weights to connection lengths for shortest-path metrics.
    """
    L = np.full_like(weight_matrix, np.inf, dtype=float)
    positive = weight_matrix > 0
    L[positive] = 1.0 / weight_matrix[positive]
    np.fill_diagonal(L, 0.0)
    return L


def _global_efficiency_from_distance(distance_matrix):
    """
    Compute global efficiency from a shortest-path distance matrix.
    """
    D = np.array(distance_matrix, dtype=float, copy=True)
    np.fill_diagonal(D, np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_D = 1.0 / D
    inv_D[~np.isfinite(inv_D)] = 0.0

    n = D.shape[0]
    if n <= 1:
        return np.nan
    return np.sum(inv_D) / (n * (n - 1))


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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    L = _to_length_matrix(W)
    D = bct.distance_wei(L)[0]
    return _global_efficiency_from_distance(D)


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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    n = W.shape[0]
    if n <= 2:
        return np.nan

    local_eff = np.full(n, np.nan, dtype=float)

    for i in range(n):
        # Directed neighborhood: nodes with incoming OR outgoing edge to i.
        nbr_mask = (W[i, :] > 0) | (W[:, i] > 0)
        nbr_mask[i] = False
        nbr_idx = np.where(nbr_mask)[0]

        if nbr_idx.size < 2:
            continue

        subW = W[np.ix_(nbr_idx, nbr_idx)]
        subL = _to_length_matrix(subW)
        subD = bct.distance_wei(subL)[0]
        local_eff[i] = _global_efficiency_from_distance(subD)

    if np.all(np.isnan(local_eff)):
        return np.nan
    return np.nanmean(local_eff)


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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    cc = bct.clustering_coef_wd(W)
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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    return bct.transitivity_wd(W)


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
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        _, Q = bct.modularity_dir(W)
        return Q
    except:
        try:
            W = _prepare_weighted_directed_matrix(adjacency_matrix)
            _, Q = bct.community_louvain(W)
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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    strengths = bct.strengths_dir(W)
    return np.mean(strengths)


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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    length_matrix = np.zeros_like(W, dtype=float)
    positive = W > 0
    length_matrix[positive] = 1.0 / W[positive]
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
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        _, _, degrees = bct.degrees_dir(W)
        if k is None:
            k = int(np.median(degrees))

        rc = bct.rich_club_wd(W, klevel=k)
        
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
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        A = (W > 0).astype(float)
        # Directed assortativity: average over out-in, in-out, out-out, in-in.
        vals = []
        for flag in (1, 2, 3, 4):
            try:
                vals.append(float(bct.assortativity_bin(A, flag=flag)))
            except:
                continue
        if len(vals) == 0:
            return np.nan
        return np.mean(vals)
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
    W = _prepare_weighted_directed_matrix(adjacency_matrix)
    eigenvalues = np.linalg.eigvals(W)
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
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        C_real = np.mean(bct.clustering_coef_wd(W))
        
        # Convert to length matrix for path length computation
        length_matrix = _to_length_matrix(W)
        D = bct.distance_wei(length_matrix)[0]
        finite_D = D[np.isfinite(D) & (D > 0)]
        L_real = np.mean(finite_D) if finite_D.size > 0 else np.nan
        
        # Generate random network with same density
        n_nodes = adjacency_matrix.shape[0]
        n_edges = np.sum(W > 0)
        density = n_edges / (n_nodes * (n_nodes - 1))
        density = float(np.clip(density, 0.0, 1.0))
        
        # Random network
        rand_matrix = np.random.rand(n_nodes, n_nodes)
        threshold = np.percentile(rand_matrix, (1 - density) * 100)
        rand_matrix = np.where(rand_matrix > threshold, rand_matrix, 0.0)
        np.fill_diagonal(rand_matrix, 0)
        
        C_rand = np.mean(bct.clustering_coef_wd(rand_matrix))
        
        rand_length = _to_length_matrix(rand_matrix)
        D_rand = bct.distance_wei(rand_length)[0]
        finite_D_rand = D_rand[np.isfinite(D_rand) & (D_rand > 0)]
        L_rand = np.mean(finite_D_rand) if finite_D_rand.size > 0 else np.nan
        
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
        W = _prepare_weighted_directed_matrix(adjacency_matrix)
        length_matrix = _to_length_matrix(W)
        D = bct.distance_wei(length_matrix)[0]
        
        # Get maximum finite distance
        finite_distances = D[np.isfinite(D) & (D > 0)]
        if len(finite_distances) > 0:
            return np.max(finite_distances)
        else:
            return np.nan
    except:
        return np.nan


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

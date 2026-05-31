"""
Connectivity analysis using various methods
"""

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time, phase_slope_index
from config import CONNECTIVITY_METHODS, FMIN, FMAX

def compute_plv(epochs, fs, fmin, fmax):
    """
    Compute Phase Locking Value (PLV) connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        PLV connectivity matrix
    """
    # Convert to MNE Epochs object
    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    # Compute PLV
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='plv',
        mode='multitaper',
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose='ERROR'
    )
    
    # Get connectivity matrix (average across frequency)
    connectivity_matrix = con.get_data(output='dense')
    # print(f'{connectivity_matrix.shape=}')
    connectivity_matrix = np.mean(connectivity_matrix, axis=2)  # Average across freqs
    connectivity_matrix += connectivity_matrix.T

    return connectivity_matrix


def compute_psi(epochs, fs, fmin, fmax):
    """
    Compute Phase Slope Index (PSI) connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        PSI connectivity matrix (directed)
    """
    # Convert to MNE Epochs object
    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    # Compute PSI
    con = phase_slope_index(
        epochs_mne,
        mode='multitaper',
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        verbose='ERROR'
    )

    # Get connectivity matrix
    connectivity_matrix = con.get_data(output='dense')
    connectivity_matrix = np.mean(connectivity_matrix, axis=2)  # Average across freqs

    psi_pos = np.zeros_like(connectivity_matrix)
    neg = connectivity_matrix < 0
    psi_pos[neg.T] = -connectivity_matrix[neg]   
    pos = connectivity_matrix > 0
    psi_pos[pos] = connectivity_matrix[pos]

    return psi_pos


def compute_granger_causality(epochs, fs, fmin, fmax):
    """
    Compute Spectral Granger Causality connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        Granger Causality connectivity matrix (directed)
    """
    # Convert to MNE Epochs object
    n_ch = epochs.shape[1]

    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    sources, targets = np.where(~np.eye(n_ch, dtype=bool))
    seeds   = [[int(i)] for i in sources]
    targs   = [[int(j)] for j in targets]
    indices = (seeds, targs)
    # indices = (sources.tolist(), targets.tolist())
    # print(f'{indices=}')
    
    # Compute Granger Causality
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='gc',
        mode='multitaper',
        indices=indices,
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose='ERROR'
    )
    
    # Get connectivity matrix
    vals = con.get_data()
    vals = vals[:, 0] if vals.ndim == 2 else vals  # handle (n_conn, 1)

    gc_mat = np.full((n_ch, n_ch), np.nan, float)
    gc_mat[sources, targets] = vals
    np.fill_diagonal(gc_mat, 0.0)
    return gc_mat

def compute_granger_causality_tr(epochs, fs, fmin, fmax):
    """
    Compute Time Teversed Spectral Granger Causality connectivity.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    connectivity : ndarray, shape (n_channels, n_channels)
        Time Reversed Granger Causality connectivity matrix (directed)
    """
    # Convert to MNE Epochs object
    n_ch = epochs.shape[1]

    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    sources, targets = np.where(~np.eye(n_ch, dtype=bool))
    seeds   = [[int(i)] for i in sources]
    targs   = [[int(j)] for j in targets]
    indices = (seeds, targs)
    # indices = (sources.tolist(), targets.tolist())
    # print(f'{indices=}')
    
    # Compute Time Reversed Granger Causality
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='gc_tr',
        mode='multitaper',
        indices=indices,
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose='ERROR'
    )
    
    # Get connectivity matrix
    vals = con.get_data()
    vals = vals[:, 0] if vals.ndim == 2 else vals  # handle (n_conn, 1)

    gc_mat = np.full((n_ch, n_ch), np.nan, float)
    gc_mat[sources, targets] = vals
    np.fill_diagonal(gc_mat, 0.0)
    return gc_mat

def compute_pdc(epochs, fs, fmin, fmax):
    raise NotImplementedError


def compute_connectivity_for_band(filtered_epochs, band_name, fs, method='plv'):
    """
    Compute connectivity for a specific frequency band and method.
    
    Parameters
    ----------
    filtered_epochs : dict
        Dictionary of filtered epochs per band
    band_name : str
        Name of frequency band
    fs : float
        Sampling frequency
    method : str
        Connectivity method ('plv', 'psi', 'gc', 'pdc')
        
    Returns
    -------
    connectivity_matrix : ndarray, shape (n_channels, n_channels)
        Average connectivity matrix across epochs
    """
    epochs = filtered_epochs[band_name]
    
    # Get frequency range for this band
    from config import FREQUENCY_BANDS
    fmin, fmax = FREQUENCY_BANDS[band_name]
    
    # Compute connectivity based on method
    if method == 'plv':
        connectivity = compute_plv(epochs, fs, fmin, fmax)
    elif method == 'psi':
        connectivity = compute_psi(epochs, fs, fmin, fmax)
    elif method == 'gc':
        connectivity = compute_granger_causality(epochs, fs, fmin, fmax)
    elif method == 'gc_tr':
        connectivity = compute_granger_causality_tr(epochs, fs, fmin, fmax)
    elif method == 'pdc':
        connectivity = compute_pdc(epochs, fs, fmin, fmax)
    else:
        raise ValueError(f"Unknown connectivity method: {method}")
    
    return connectivity


def normalize_connectivity_matrix(matrix):
    """
    Normalize connectivity matrix to [0, 1] range.
    
    Parameters
    ----------
    matrix : ndarray, shape (n_channels, n_channels)
        Connectivity matrix
        
    Returns
    -------
    normalized : ndarray
        Normalized connectivity matrix
    """
    matrix = np.asarray(matrix, dtype=float)
    finite_mask = np.isfinite(matrix)
    if not np.any(finite_mask):
        return np.zeros_like(matrix, dtype=float)

    # Min-max normalization over valid entries only. This prevents one NaN edge
    # from turning a whole subject/band matrix into NaNs.
    min_val = np.min(matrix[finite_mask])
    max_val = np.max(matrix[finite_mask])
    
    if max_val - min_val > 0:
        normalized = (matrix - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(matrix, dtype=float)

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    
    return normalized


def compute_all_connectivity(filtered_epochs, fs, methods=CONNECTIVITY_METHODS):
    """
    Compute connectivity for all methods and all frequency bands.
    
    Parameters
    ----------
    filtered_epochs : dict
        Dictionary of filtered epochs per band
    fs : float
        Sampling frequency
    methods : list
        List of connectivity methods to compute
        
    Returns
    -------
    connectivity_results : dict
        Nested dictionary: {method: {band: normalized_matrix}}
    """
    from config import FREQUENCY_BANDS
    
    connectivity_results = {}
    
    for method in methods:
        print(f"\nComputing {method.upper()} connectivity:")
        connectivity_results[method] = {}
        
        for band_name in FREQUENCY_BANDS.keys():
            print(f"  {band_name}...", end=' ')
            
            try:
                conn_matrix = compute_connectivity_for_band(
                    filtered_epochs, band_name, fs, method
                )
                
                # Normalize
                conn_matrix_normalized = normalize_connectivity_matrix(conn_matrix)
                
                connectivity_results[method][band_name] = conn_matrix_normalized
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                # Store zeros as placeholder
                n_channels = filtered_epochs[band_name].shape[1]
                connectivity_results[method][band_name] = np.zeros((n_channels, n_channels))
    
    return connectivity_results

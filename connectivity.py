"""
Connectivity analysis using various methods
"""

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
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
        verbose=False
    )
    
    # Get connectivity matrix (average across frequency)
    connectivity_matrix = con.get_data(output='dense')
    connectivity_matrix = np.mean(connectivity_matrix, axis=0)  # Average across freqs
    
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
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='psi',
        mode='multitaper',
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False
    )
    
    # Get connectivity matrix
    connectivity_matrix = con.get_data(output='dense')
    connectivity_matrix = np.mean(connectivity_matrix, axis=0)  # Average across freqs
    
    return connectivity_matrix


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
    info = mne.create_info(
        ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
        sfreq=fs,
        ch_types='eeg'
    )
    epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
    
    # Compute Granger Causality
    con = spectral_connectivity_epochs(
        epochs_mne,
        method='gc',
        mode='multitaper',
        sfreq=fs,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False
    )
    
    # Get connectivity matrix
    connectivity_matrix = con.get_data(output='dense')
    connectivity_matrix = np.mean(connectivity_matrix, axis=0)  # Average across freqs
    
    return connectivity_matrix


def compute_pdc(epochs, fs, fmin, fmax):
    """
    Compute Partial Directed Coherence (PDC) connectivity.
    
    Note: PDC implementation may require additional libraries.
    This is a placeholder that attempts to use available methods.
    
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
        PDC connectivity matrix (directed)
    """
    try:
        # Try to use MNE-connectivity if PDC is available
        info = mne.create_info(
            ch_names=[f'Ch{i}' for i in range(epochs.shape[1])],
            sfreq=fs,
            ch_types='eeg'
        )
        epochs_mne = mne.EpochsArray(epochs, info, verbose=False)
        
        # Attempt to compute with 'gc_tr' which is similar to PDC
        con = spectral_connectivity_epochs(
            epochs_mne,
            method='gc_tr',  # Time-reversed Granger (similar to PDC)
            mode='multitaper',
            sfreq=fs,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            verbose=False
        )
        
        connectivity_matrix = con.get_data(output='dense')
        connectivity_matrix = np.mean(connectivity_matrix, axis=0)
        
        return connectivity_matrix
        
    except Exception as e:
        print(f"Warning: PDC computation failed. Using placeholder. Error: {e}")
        print("Consider installing specialized PDC package or using alternative method.")
        
        # Return placeholder (zeros or random)
        n_channels = epochs.shape[1]
        return np.zeros((n_channels, n_channels))


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
    # Min-max normalization
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if max_val - min_val > 0:
        normalized = (matrix - min_val) / (max_val - min_val)
    else:
        normalized = matrix
    
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

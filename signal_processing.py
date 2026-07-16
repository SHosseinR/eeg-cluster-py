"""
Signal processing utilities for EEG data
"""

import numpy as np
from scipy import signal
from config import EPOCH_DURATION, OVERLAP, FREQUENCY_BANDS

def create_epochs(data, fs, epoch_duration=EPOCH_DURATION, overlap=OVERLAP):
    """
    Chunk continuous data into fixed-duration epochs.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Continuous EEG data
    fs : float
        Sampling frequency
    epoch_duration : float
        Duration of each epoch in seconds
    overlap : float
        Overlap between epochs in seconds
        
    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples_per_epoch)
        Epoched data
    """
    n_channels, n_samples = data.shape
    
    # Calculate epoch parameters
    samples_per_epoch = int(epoch_duration * fs)
    step_size = int((epoch_duration - overlap) * fs)
    
    # Calculate number of epochs
    n_epochs = int((n_samples - samples_per_epoch) / step_size) + 1
    
    epochs = np.zeros((n_epochs, n_channels, samples_per_epoch))
    
    for i in range(n_epochs):
        start_idx = i * step_size
        end_idx = start_idx + samples_per_epoch
        
        if end_idx <= n_samples:
            epochs[i] = data[:, start_idx:end_idx]
        else:
            # Pad last epoch if necessary
            available_samples = n_samples - start_idx
            epochs[i, :, :available_samples] = data[:, start_idx:]
            # Zero-pad the rest
            epochs[i, :, available_samples:] = 0
    
    print(f"Created {n_epochs} epochs of {epoch_duration}s duration")
    print(f"Epoch shape: {epochs[0].shape}")
    
    return epochs


def bandpass_filter(data, fs, low_freq, high_freq, order=4):
    """
    Apply bandpass filter to data.
    
    Parameters
    ----------
    data : ndarray
        Input data (can be 2D or 3D)
    fs : float
        Sampling frequency
    low_freq : float
        Low cutoff frequency
    high_freq : float
        High cutoff frequency
    order : int
        Filter order
        
    Returns
    -------
    filtered_data : ndarray
        Filtered data with same shape as input
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter along the last axis (time)
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data


def filter_epochs_by_bands(epochs, fs, frequency_bands=FREQUENCY_BANDS):
    """
    Filter epochs into different frequency bands.
    
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, n_samples)
        Epoched data
    fs : float
        Sampling frequency
    frequency_bands : dict
        Dictionary mapping band names to (low, high) frequency tuples
        
    Returns
    -------
    filtered_epochs : dict
        Dictionary mapping band names to filtered epochs
        Each value has shape (n_epochs, n_channels, n_samples)
    """
    filtered_epochs = {}
    
    print(f"\nFiltering epochs into frequency bands:")
    
    for band_name, (low_freq, high_freq) in frequency_bands.items():
        print(f"  {band_name}: {low_freq}-{high_freq} Hz")
        filtered_epochs[band_name] = bandpass_filter(epochs, fs, low_freq, high_freq)
    
    return filtered_epochs


def process_subject_epochs(data, fs, *, return_broadband=False):
    """
    Complete epoch processing pipeline for a single subject.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Continuous EEG data
    fs : float
        Sampling frequency
        
    Returns
    -------
    filtered_epochs : dict
        Dictionary mapping band names to filtered epochs
        Each value has shape (n_epochs, n_channels, n_samples)
    broadband_epochs : ndarray, optional
        Returned with ``filtered_epochs`` only when ``return_broadband=True``.
        These unfiltered epochs support spectral estimation without a second
        band-pass restriction.
    """
    # Create epochs
    epochs = create_epochs(data, fs)
    
    # Filter into frequency bands
    filtered_epochs = filter_epochs_by_bands(epochs, fs)
    
    if return_broadband:
        return filtered_epochs, epochs
    return filtered_epochs


def prepare_epochs_for_connectivity(filtered_epochs, band_name):
    """
    Prepare epochs from a specific band for connectivity analysis.
    
    Parameters
    ----------
    filtered_epochs : dict
        Output from filter_epochs_by_bands
    band_name : str
        Name of frequency band
        
    Returns
    -------
    epochs_array : ndarray, shape (n_epochs, n_channels, n_samples)
        Epochs ready for connectivity analysis
    """
    return filtered_epochs[band_name]

"""
Data loading utilities for EEG data
"""

import os
import glob
import numpy as np
import mne
from config import CHANNELS_TO_DROP

def load_subject_epochs(subject_folder):
    """
    Load all .set files from a subject folder and combine them.
    
    Parameters
    ----------
    subject_folder : str
        Path to subject folder containing .set files
        
    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
        Combined EEG data from all files
    fs : float
        Sampling frequency
    channel_names : list
        Channel names
    """
    set_files = sorted(glob.glob(os.path.join(subject_folder, '*.set')))
    
    if not set_files:
        raise ValueError(f"No .set files found in {subject_folder}")
    
    print(f"Loading {len(set_files)} file(s) from {subject_folder}")
    
    all_data = []
    fs = None
    channel_names = None
    
    for set_file in set_files:
        print(f"  Loading: {os.path.basename(set_file)}")
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        
        # Drop specified channels
        existing_to_drop = [ch for ch in CHANNELS_TO_DROP if ch in raw.ch_names]
        if existing_to_drop:
            raw.drop_channels(existing_to_drop)
            print(f"    Dropped channels: {existing_to_drop}")
        
        # Get data
        data = raw.get_data()
        
        # Store sampling frequency and channel names from first file
        if fs is None:
            fs = raw.info['sfreq']
            channel_names = raw.ch_names
        else:
            # Verify consistency across files
            if fs != raw.info['sfreq']:
                raise ValueError(f"Sampling frequency mismatch in {set_file}")
            if channel_names != raw.ch_names:
                raise ValueError(f"Channel names mismatch in {set_file}")
        
        all_data.append(data)
    
    # Concatenate all data along time axis
    combined_data = np.concatenate(all_data, axis=1)
    
    print(f"  Total data shape: {combined_data.shape}")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Number of channels: {len(channel_names)}")
    
    return combined_data, fs, channel_names


def load_group_data(data_path, group_name="Group"):
    """
    Load data from all subjects in a group.
    
    Parameters
    ----------
    data_path : str
        Path to directory containing subject folders
    group_name : str
        Name of the group (for logging)
        
    Returns
    -------
    subjects_data : list of dict
        List of dictionaries containing data for each subject
        Each dict has keys: 'data', 'fs', 'channels', 'subject_id'
    """
    subject_folders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    subject_folders = sorted(subject_folders)
    
    if not subject_folders:
        raise ValueError(f"No subject folders found in {data_path}")
    
    print(f"\n{'='*80}")
    print(f"Loading {group_name} data from: {data_path}")
    print(f"Found {len(subject_folders)} subjects")
    print(f"{'='*80}\n")
    
    subjects_data = []
    
    for i, subject_folder in enumerate(subject_folders):
        subject_id = os.path.basename(subject_folder)
        print(f"\n[{i+1}/{len(subject_folders)}] Processing {subject_id}")
        
        try:
            data, fs, channels = load_subject_epochs(subject_folder)
            
            subjects_data.append({
                'data': data,
                'fs': fs,
                'channels': channels,
                'subject_id': subject_id,
                'group': group_name
            })
            
        except Exception as e:
            print(f"  ERROR loading {subject_id}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Successfully loaded {len(subjects_data)}/{len(subject_folders)} subjects")
    print(f"{'='*80}\n")
    
    return subjects_data


def verify_data_consistency(subjects_data):
    """
    Verify that all subjects have consistent sampling frequency and channels.
    
    Parameters
    ----------
    subjects_data : list of dict
        Subject data from load_group_data
        
    Returns
    -------
    bool
        True if all subjects are consistent
    """
    if not subjects_data:
        return False
    
    reference_fs = subjects_data[0]['fs']
    reference_channels = subjects_data[0]['channels']
    
    for subject in subjects_data:
        if subject['fs'] != reference_fs:
            print(f"WARNING: Subject {subject['subject_id']} has different sampling frequency")
            return False
        if subject['channels'] != reference_channels:
            print(f"WARNING: Subject {subject['subject_id']} has different channels")
            return False
    
    print(f"✓ All subjects consistent:")
    print(f"  Sampling frequency: {reference_fs} Hz")
    print(f"  Number of channels: {len(reference_channels)}")
    print(f"  Channels: {reference_channels}")
    
    return True

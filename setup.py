#!/usr/bin/env python
"""
Setup and verification script for EEG connectivity analysis pipeline
Run this after installing requirements to verify everything works
"""

import sys
import subprocess

def check_python_version():
    """Check if Python version is adequate."""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_package(package_name, import_name=None):
    """Check if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} not found")
        return False


def install_requirements():
    """Install requirements from requirements.txt."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False


def verify_installation():
    """Verify all required packages are installed."""
    print("\n" + "="*80)
    print("VERIFYING INSTALLATION")
    print("="*80 + "\n")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'sklearn'),
        ('mne', 'mne'),
        ('mne-connectivity', 'mne_connectivity'),
        ('bctpy', 'bct'),
        ('statsmodels', 'statsmodels'),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    return all_installed


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n" + "="*80)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*80 + "\n")
    
    try:
        print("Testing NumPy...")
        import numpy as np
        arr = np.random.randn(10, 10)
        print(f"✓ NumPy array operations work")
        
        print("\nTesting MNE...")
        import mne
        info = mne.create_info(ch_names=['Ch1', 'Ch2'], sfreq=100, ch_types='eeg')
        print(f"✓ MNE basic functions work")
        
        print("\nTesting BCT...")
        import bct
        # Test a simple BCT function
        test_matrix = np.random.rand(10, 10)
        test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
        eff = bct.efficiency_wei(test_matrix)
        print(f"✓ BCT network measures work (efficiency = {eff:.4f})")
        
        print("\nTesting MNE-Connectivity...")
        from mne_connectivity import spectral_connectivity_epochs
        print(f"✓ MNE-Connectivity imports work")
        
        print("\nTesting scikit-learn...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        print(f"✓ Scikit-learn imports work")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False


def create_test_data():
    """Create minimal test data for verification."""
    print("\n" + "="*80)
    print("CREATING TEST DATA")
    print("="*80 + "\n")
    
    try:
        import numpy as np
        import os
        
        test_dir = "./test_data"
        os.makedirs(test_dir, exist_ok=True)
        
        # Create dummy EEG data
        n_channels = 10
        n_samples = 10000
        fs = 100
        
        data = np.random.randn(n_channels, n_samples) * 10  # Simulate EEG amplitudes
        
        print(f"Created test data:")
        print(f"  Channels: {n_channels}")
        print(f"  Samples: {n_samples}")
        print(f"  Duration: {n_samples/fs} seconds")
        print(f"  Sampling rate: {fs} Hz")
        
        # Save
        np.save(os.path.join(test_dir, "test_eeg.npy"), data)
        print(f"\n✓ Test data saved to {test_dir}/test_eeg.npy")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating test data: {e}")
        return False


def run_minimal_test():
    """Run a minimal connectivity analysis test."""
    print("\n" + "="*80)
    print("RUNNING MINIMAL CONNECTIVITY TEST")
    print("="*80 + "\n")
    
    try:
        import numpy as np
        import mne
        from mne_connectivity import spectral_connectivity_epochs
        
        # Create synthetic EEG data
        n_epochs = 5
        n_channels = 4
        n_times = 1000
        sfreq = 100
        
        data = np.random.randn(n_epochs, n_channels, n_times)
        info = mne.create_info(
            ch_names=[f'Ch{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )
        epochs = mne.EpochsArray(data, info, verbose=False)
        
        print("Computing PLV connectivity...")
        con = spectral_connectivity_epochs(
            epochs,
            method='plv',
            mode='multitaper',
            sfreq=sfreq,
            fmin=8,
            fmax=13,
            faverage=True,
            verbose=False
        )
        
        conn_matrix = con.get_data(output='dense')
        print(f"✓ Connectivity computed successfully")
        print(f"  Matrix shape: {conn_matrix.shape}")
        print(f"  Mean connectivity: {np.mean(conn_matrix):.4f}")
        
        # Test network measure
        import bct
        conn_2d = np.mean(conn_matrix, axis=0)
        eff = bct.efficiency_wei(conn_2d)
        print(f"  Global efficiency: {eff:.4f}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during connectivity test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup and verification routine."""
    print("\n" + "="*80)
    print("EEG CONNECTIVITY ANALYSIS - SETUP & VERIFICATION")
    print("="*80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Verify installation
    installed = verify_installation()
    
    if not installed:
        print("\n⚠️  Some packages are missing.")
        response = input("Do you want to install requirements now? (y/n): ")
        if response.lower() == 'y':
            if install_requirements():
                print("\nRe-verifying installation...")
                installed = verify_installation()
            else:
                print("\n❌ Installation failed. Please install manually:")
                print("   pip install -r requirements.txt")
                sys.exit(1)
    
    if not installed:
        print("\n❌ Setup incomplete. Please install missing packages.")
        sys.exit(1)
    
    # Test functionality
    print("\n")
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed.")
        sys.exit(1)
    
    # Create test data
    create_test_data()
    
    # Run minimal test
    if not run_minimal_test():
        print("\n❌ Connectivity test failed.")
        sys.exit(1)
    
    # Success
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE - ALL SYSTEMS GO!")
    print("="*80)
    print("\nYou can now run the analysis pipeline:")
    print("  python main.py")
    print("\nOr explore the examples:")
    print("  python examples.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

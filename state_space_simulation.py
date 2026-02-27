"""
State-space simulation for EEG dynamics with stimulation
"""
import numpy as np
from numpy.linalg import eig


def normalize_adjacency_matrix(A, stability_constant=0.01):
    """
    Normalize adjacency matrix to ensure stable dynamics.
    
    The normalization ensures the maximum eigenvalue is less than 1
    to prevent unstable dynamics.
    
    Parameters
    ----------
    A : ndarray, shape (n_nodes, n_nodes)
        Adjacency matrix (connectivity matrix)
    stability_constant : float
        Constant added to denominator for stability (default: 0.01)
        
    Returns
    -------
    A_norm : ndarray, shape (n_nodes, n_nodes)
        Normalized adjacency matrix with stable dynamics
    """
    # Compute eigenvalues
    w, _ = eig(A)
    lambda_max = np.abs(w).max()
    
    # Normalize to ensure stability
    A_norm = A / (stability_constant + lambda_max)
    
    # Remove self-connections (set diagonal to 0)
    A_norm = A_norm - np.diag(np.diag(A_norm))
    
    return A_norm


def create_control_matrix(n_nodes, stimulation_node):
    """
    Create control input matrix B for single-node stimulation.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network
    stimulation_node : int
        Index of the node to stimulate (0-indexed)
        
    Returns
    -------
    B : ndarray, shape (n_nodes, n_nodes)
        Control matrix (diagonal with 1 at stimulation node, 0 elsewhere)
    """
    B = np.zeros((n_nodes, n_nodes))
    B[stimulation_node, stimulation_node] = 1.0
    return B


def create_stimulation_signal(n_nodes, stimulation_node, duration, dt, amplitude=1.0):
    """
    Create stimulation signal U.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes
    stimulation_node : int
        Index of node to stimulate
    duration : float
        Duration of stimulation in seconds
    dt : float
        Time step in seconds
    amplitude : float
        Stimulation amplitude (default: 1.0)
        
    Returns
    -------
    U : ndarray, shape (n_nodes, n_timesteps)
        Stimulation signal (1 at stimulation node for duration, 0 elsewhere)
    """
    n_timesteps = int(duration / dt)
    U = np.zeros((n_nodes, n_timesteps))
    U[stimulation_node, :] = amplitude
    return U


def simulate_eeg_dynamics(A, x0, xbar, B, U, dt=0.001):
    """
    Simulate EEG dynamics with state-space model:
    
        dx/dt = A*x - A*xbar + B*u
        x(0) = xbar
    
    Where:
    - A is the (normalized) adjacency matrix
    - x is the state (node activations)
    - xbar is the baseline activation (average over time)
    - B is the control matrix
    - u is the stimulation input
    
    Parameters
    ----------
    A : ndarray, shape (n_nodes, n_nodes)
        Normalized adjacency matrix
    x0 : ndarray, shape (n_nodes,) or (n_nodes, 1)
        Initial state (should be xbar)
    xbar : ndarray, shape (n_nodes,) or (n_nodes, 1)
        Baseline activation
    B : ndarray, shape (n_nodes, n_nodes)
        Control matrix
    U : ndarray, shape (n_nodes, n_timesteps)
        Stimulation signal
    dt : float
        Time step for simulation (default: 0.001 seconds)
        
    Returns
    -------
    x : ndarray, shape (n_nodes, n_timesteps)
        Simulated state trajectory
    x_final : ndarray, shape (n_nodes,)
        Final state after stimulation
    """
    # Convert to float if boolean
    if x0.dtype == np.bool_:
        x0 = x0.astype(float)
    if xbar.dtype == np.bool_:
        xbar = xbar.astype(float)
    
    # Ensure proper dimensions
    if x0.ndim == 1:
        x0 = x0.reshape(-1, 1)
    if xbar.ndim == 1:
        xbar = xbar.reshape(-1, 1)
    
    # Get dimensions
    n_nodes = A.shape[0]
    n_timesteps = U.shape[1]
    
    # Initialize trajectory
    x = np.zeros((n_nodes, n_timesteps))
    xt = x0.copy()
    
    # Simulate dynamics
    for t in range(n_timesteps):
        # Store current state
        x[:, t] = xt[:, 0]
        
        # Compute derivative: dx/dt = A*x - A*xbar + B*u
        u_t = U[:, t].reshape(-1, 1)
        dxdt = np.matmul(A, xt) - np.matmul(A, xbar) + np.matmul(B, u_t)
        
        # Euler integration
        xt = xt + dxdt * dt
    
    # Return full trajectory and final state
    x_final = x[:, -1]
    
    return x, x_final


def compute_activation_changes(x_baseline, x_final):
    """
    Compute relative change in node activations after stimulation.
    
    Parameters
    ----------
    x_baseline : ndarray, shape (n_nodes,)
        Baseline node activations (before stimulation)
    x_final : ndarray, shape (n_nodes,)
        Final node activations (after stimulation)
        
    Returns
    -------
    activation_ratios : ndarray, shape (n_nodes,)
        Ratio of change for each node (1.0 = no change, >1.0 = increase, <1.0 = decrease)
        Values are clipped to avoid extreme ratios
    """
    # Compute ratio: final / baseline
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    activation_ratios = (x_final + epsilon) / (x_baseline + epsilon)
    
    # Clip extreme values to reasonable range [0.1, 10.0]
    activation_ratios = np.clip(activation_ratios, 0.1, 10.0)
    
    return activation_ratios


def run_full_simulation(adjacency_matrix, baseline_activation, stimulation_node,
                       stimulation_duration=1.0, stimulation_amplitude=1.0,
                       dt=0.001, stability_constant=0.01):
    """
    Complete pipeline for simulating EEG dynamics with stimulation.
    
    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n_nodes, n_nodes)
        Original connectivity matrix
    baseline_activation : ndarray, shape (n_nodes,)
        Baseline node activations (average over time)
    stimulation_node : int
        Index of node to stimulate (0-indexed)
    stimulation_duration : float
        Duration of stimulation in seconds (default: 1.0)
    stimulation_amplitude : float
        Amplitude of stimulation (default: 1.0)
    dt : float
        Time step for simulation (default: 0.001)
    stability_constant : float
        Constant for matrix normalization (default: 0.01)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'trajectory': Full state trajectory, shape (n_nodes, n_timesteps)
        - 'final_state': Final state after stimulation, shape (n_nodes,)
        - 'activation_ratios': Ratio of change for each node, shape (n_nodes,)
        - 'normalized_matrix': Normalized adjacency matrix used in simulation
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Normalize adjacency matrix for stable dynamics
    A_norm = normalize_adjacency_matrix(adjacency_matrix, stability_constant)
    
    # Create control matrix
    B = create_control_matrix(n_nodes, stimulation_node)
    
    # Create stimulation signal
    U = create_stimulation_signal(n_nodes, stimulation_node, 
                                  stimulation_duration, dt, stimulation_amplitude)
    
    # Initial state is baseline
    x0 = baseline_activation.copy()
    xbar = baseline_activation.copy()
    
    # Run simulation
    trajectory, x_final = simulate_eeg_dynamics(A_norm, x0, xbar, B, U, dt)
    
    # Compute activation changes
    activation_ratios = compute_activation_changes(baseline_activation, x_final)
    
    # Package results
    results = {
        'trajectory': trajectory,
        'final_state': x_final,
        'activation_ratios': activation_ratios,
        'normalized_matrix': A_norm,
        'baseline': baseline_activation
    }
    
    return results


# Example usage
if __name__ == "__main__":
    # Create example adjacency matrix
    n_nodes = 5
    A = np.random.rand(n_nodes, n_nodes)
    A = (A + A.T) / 2  # Make symmetric
    
    # Create baseline activation
    xbar = np.random.rand(n_nodes)
    
    # Run simulation
    results = run_full_simulation(
        adjacency_matrix=A,
        baseline_activation=xbar,
        stimulation_node=2,
        stimulation_duration=1.0,
        dt=0.001
    )
    
    print("Baseline activation:", results['baseline'])
    print("Final state:", results['final_state'])
    print("Activation ratios:", results['activation_ratios'])

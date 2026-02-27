import numpy as np
# initialize A matrix
np.random.seed(42)  # for reproducibility
n_nodes = 5
A = np.random.rand(n_nodes, n_nodes)
print(A)

from nctpy.utils import matrix_normalization
system = 'continuous'
A_norm = matrix_normalization(A=A, c=1, system=system)
print(A_norm)
print(f'{np.linalg.eig(A_norm).eigenvalues}')
from nctpy.energies import sim_state_eq
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='paper', font_scale=1)

T = 20*1000  # time horizon
U = np.zeros((n_nodes, T))  # the input to the system
U[:, 0:1000] = 1  # impulse, 1 input at the first time point delivered to all nodes
B = np.eye(n_nodes)  # uniform full control set
x0 = np.ones((n_nodes, 1))  # initial state, all nodes set to 1 unit of neural activity
x = sim_state_eq(A_norm=A_norm, B=B, x0=x0, U=U, system=system)

# plot
f, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(x.T)
ax.set_ylabel('Simulated neural activity (arbitrary units)')
ax.set_xlabel('Time (arbitrary units)')
# f.savefig('A_stable.png', dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.show()
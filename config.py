# For core functionality
datasets = ['train', 'validate', 'test', 'test_Hesthaven_Ubbiali']
num_snapshots = {
    'train':    100,
    'validate':  50,
    'test':      50,
    'test_Hesthaven_Ubbiali': 3,
    }

# We treat velocity (u,v) as a single vector component u.
components = ['u', 'p']
num_basis = {'u': 30, 'p': 30, }

# A mask for the snapshots in the dataset describing how the components are
# located in the single smallest pattern. Here we treat the velocity vector
# as a single component, so we mask both u and v with 'u'.
mask = ['u','u','p']


# For plotting
import numpy as np
mu_names = ['lx', 'ly', 'angle']
mu_range = {'lx': (1,2), 'ly': (1,2), 'angle': (np.pi/6, 5*np.pi/6)}
P = len(mu_names)

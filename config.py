"""
"""

datasets = ['train', 'validate', 'test', 'test_Hesthaven_Ubbiali']
num_snapshots = {
    'train':    100,
    'validate':  50,
    'test':      50,
    'test_Hesthaven_Ubbiali': 3,
    }

components = ['u', 'p']
num_basis = {'u': 30, 'p': 30, }

mask = ['u','u','p']


# For plotting
import numpy as np
mu_names = ['lx', 'ly', 'angle']
mu_range = {'lx': (1,2), 'ly': (1,2), 'angle': (np.pi/6, 5*np.pi/6)}
P = len(mu_names)
unsteady = False

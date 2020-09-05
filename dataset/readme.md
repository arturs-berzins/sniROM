
### Data structure
Current `utils` expects the dataset to be structured as follows. For each sub-set there must be a sub-directory. The minimum required sub-sets are:
- `train`: used for computing the POD i.e. finding the reduced basis, as well as training the regression models.
- `validate`: used solely for evaluating the performance during the hypertuning of the ANN.
- `test`: used for evaluating the perfomance of the best ANN configuration and the other two regression models, as well as the projection error.

Each sub-set must contain:
- `parameters.txt`: containing an array of parameter samples of dimension `N_set x N_d`.
- `truth_i` for `i=0,..,N_subset`: snapshots in binary consisting of `numpy.float64` data types (C doubles) using big-endian byteorder. Each snapshot can be loaded into a `numpy` vector via e.g. `np.load('dataset/train/truth_0', np.float64()).byteswap()` (see `utils.load_snapshot()`). Each vector contains all degrees of freedom. E.g. for a steady problem and components `u`, `v`, `p` (such as the skewed lid-driven-cavity) the vector is as follows:
`u` \ 
`v` &nbsp;&nbsp; node `0`
`p` /
`u`  \ 
`v` &nbsp;&nbsp; node `1`
`p` /
.
.
.
`u`  \ 
`v` &nbsp;&nbsp; node `#nodes-1`
`p` /

For time-dependent problems, this pattern is repeated for every time step, starting at the initial condition. This corresponds to a `#components*#nodes*(#timesteps+1)` vector.

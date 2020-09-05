"""Gaussian processes regression. """

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.optimize
from sklearn.utils.optimize import _check_optimize_result

from sklearn.gaussian_process import GaussianProcessRegressor

class myGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super().__init__(kernel=kernel, alpha=alpha,
                 optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                 normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True,
                bounds=bounds, options={'maxfun':150000, 'maxiter':150000, 'maxls':50, 'disp':False})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min

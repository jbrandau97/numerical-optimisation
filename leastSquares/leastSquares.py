from functools import lru_cache
import numdifftools as nd
import numpy as np
import pandas as pd
from typing import Type
from scipy.linalg import qr
import warnings

import os
import sys

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/lineSearch")

# Now you can import your module
from utils import absObjective, Phi, visualise

from lineSearch import _lineSearch


class leastSquares(_lineSearch):
    """
    Base class for solving the least squares problem using the line search method
    """

    def gaussNewton(
        self, res: callable, jac: callable = None, decomp: str = "SVD"
    ) -> None:
        """
        Solve the least squares problem using the Gauss-Newton method

        Args:
            res (callable): The residual function
            Optional:
                jac (callable): The Jacobian function
                decomp (str): The decomposition method to use, either "SVD" (default), "QR" or "Cholesky"

        Returns:
            None
        """
        self.res = res
        self.n = len(self.x_init)
        # Use the numerical Jacobian if the user does not provide one
        self.jacobian = jac if jac is not None else nd.Jacobian(self.res)
        self.alpha_init = self.alpha_max
        self.data = pd.DataFrame(columns=["x", "f", "df", "J", "alpha"])
        self.data.loc[0] = [
            self.x_init,
            self.func.f(self.x_init),
            self.func.df(self.x_init),
            self.jacobian(self.x_init),
            self.alpha_init,
        ]

        for i in range(self.max_iter):
            # Compute the residuals
            r = self.res(self.data.loc[i, "x"])
            # Compute the Jacobian matrix of the residuals
            J = self.jacobian(self.data.loc[i, "x"])

            # Perform the decomposition of the Jacobian
            match decomp:
                case "SVD":
                    U, s, V = np.linalg.svd(J, full_matrices=False)
                    J_pinv = V.T @ np.diag(1 / s) @ U.T
                case "QR":
                    Q, R = np.linalg.qr(J)
                    J_pinv = np.linalg.solve(R, Q.T)
                case "Cholesky":
                    R = np.linalg.cholesky(J.T @ J)
                    J_pinv = np.linalg.solve(R.T, np.linalg.solve(R, J.T))
                case other:
                    raise ValueError("Invalid decomposition method")

            # Compute the search direction
            p = -J_pinv @ r

            # Perform the line search
            match self.params["ls_method"]:
                case "strong_wolfe":
                    self.alpha = self.strongWolfe(self.data.loc[i, "x"], p)
                case "backtracking":
                    self.alpha = self.backtracking(self.data.loc[i, "x"], p)
                case other:
                    raise ValueError("Invalid line search method")

            # Update the data frame
            self.data.loc[i + 1] = [
                self.data.loc[i, "x"] + self.alpha * p,
                self.func.f(self.data.loc[i, "x"] + self.alpha * p),
                self.func.df(self.data.loc[i, "x"] + self.alpha * p),
                self.jacobian(self.data.loc[i, "x"] + self.alpha * p),
                self.alpha,
            ]

            # Check for convergence
            if np.linalg.norm(self.data.loc[i + 1, "J"] @ p) < self.tol:
                break
            if i == self.max_iter - 1:
                warnings.warn("Maximum number of iterations reached")

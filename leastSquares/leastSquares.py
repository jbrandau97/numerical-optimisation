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
sys.path.append(parent_dir + "/trustRegion")

# Now you can import your module
from utils import absObjective, Phi, visualise

from lineSearch import _lineSearch
from trustRegion import _trustRegion


class _leastSquares(_lineSearch):
    """
    Base class for solving the least squares problem using the line search method
    """

    def solve(
        self,
        res: callable,
        jac: callable = None,
        delta_max: float = 3,
        eta: float = 0.1,
        params: dict = None,
    ) -> None:
        """
        Solve the least squares problem using the Gauss-Newton method

        Args:
            res (callable): The residual function
            Optional:
                jac (callable): The Jacobian function
                params (dict): The parameters for the solver
        """

        if params is not None:
            self.params = params
        else:
            self.params = {
                "algorithm": "gauss_newton",
                "decomp": "SVD",
                "ls_method": "strong_wolfe",
            }

        print(self.params)

        self.res_func = res
        self.n = len(self.x_init)
        # Use the numerical Jacobian if the user does not provide one
        self.jac_func = jac if jac is not None else nd.Jacobian(self.res)
        self.alpha_init = self.alpha_max
        self.data = pd.DataFrame(columns=["x", "f", "df", "J", "alpha"])
        self.data.loc[0] = [
            self.x_init,
            self.func.f(self.x_init),
            self.func.df(self.x_init),
            self.jac_func(self.x_init),
            self.alpha_init,
        ]

        if self.params["algorithm"] == "levenberg_marquardt":
            self.delta_max = delta_max
            self.eta = eta
            self.delta = 0.5 * self.delta_max
            self.D, self.D_inv = np.eye(self.n), np.eye(self.n)

        for i in range(self.max_iter):
            # Compute the residuals
            self.res = self.res_func(self.data["x"].iloc[-1])
            # Compute the Jacobian matrix of the residuals
            self.J = self.jac_func(self.data["x"].iloc[-1])

            # Compute search direction and step length
            match self.params["algorithm"]:
                case "gauss_newton":
                    self.p, self.alpha = self.gaussNewton()
                case "levenberg_marquardt":
                    self.levenbergMarquardt()
                    self.alpha = 1  # The step length is always 1 for the Levenberg-Marquardt method
                case other:
                    raise ValueError("Invalid algorithm")

            if self.params["algorithm"] != "levenberg_marquardt":
                # Update the data frame
                self.data.loc[i + 1] = [
                    self.data["x"].iloc[-1] + self.alpha * self.p,
                    self.func.f(self.data["x"].iloc[-1] + self.alpha * self.p),
                    self.func.df(self.data["x"].iloc[-1] + self.alpha * self.p),
                    self.jac_func(self.data["x"].iloc[-1] + self.alpha * self.p),
                    self.alpha,
                ]

            # Check for convergence
            if np.linalg.norm(self.data["J"].iloc[-1] @ self.p) < self.tol:
                break
            if i == self.max_iter - 1:
                warnings.warn("Maximum number of iterations reached")

    def gaussNewton(self) -> np.ndarray | float:
        """
        Solve the least squares problem using the Gauss-Newton method
        """

        # Perform the decomposition of the Jacobian
        match self.params["decomp"]:
            case "SVD":
                U, s, V = np.linalg.svd(self.J, full_matrices=False)
                self.J_pinv = V.T @ np.diag(1 / s) @ U.T
            case "QR":
                Q, R = np.linalg.qr(self.J)
                self.J_pinv = np.linalg.solve(R, Q.T)
            case "Cholesky":
                R = np.linalg.cholesky(self.J.T @ self.J)
                self.J_pinv = np.linalg.solve(R.T, np.linalg.solve(R, self.J.T))
            case other:
                raise ValueError("Invalid decomposition method")

        # Compute the search direction
        p = -self.J_pinv @ self.res

        # Perform the line search
        match self.params["ls_method"]:
            case "strong_wolfe":
                alpha = self.strongWolfe(self.data["x"].iloc[-1], p)
            case "backtracking":
                alpha = self.backtracking(self.data["x"].iloc[-1], p)
            case other:
                raise ValueError("Invalid line search method")

        return p, alpha

    def levenbergMarquardt(self, scaling: bool = False) -> None:
        """
        Solve the least squares problem using the Levenberg-Marquardt method
        """

        # Define the model function
        self.M = (
            lambda p: 0.5 * np.linalg.norm(self.res, ord=2) ** 2
            + p.T @ self.J.T @ self.res
            + 0.5 * p.T @ self.J.T @ self.J @ p
        )

        # Construct scaling matrices
        if scaling:
            for i in range(self.n):
                self.D[i, i] = np.max([self.D[i, i], np.linalg.norm(self.J[:, i])])
                self.D_inv[i, i] = 1 / self.D[i, i]

        # Compute the Gauss-Newton search direction and check if it lies inside the trust region
        self.p_GN, _ = self.gaussNewton()
        self.Dp = np.linalg.norm(self.D @ self.p_GN)
        if np.linalg.norm(self.D @ self.p_GN) <= 1.1 * self.delta:
            self.p = self.p_GN
        else:
            # Compute optimal lambda
            self.J_scaled = self.J @ self.D_inv
            self.lmbda = 1e-3 * np.max(np.diag(self.J_scaled.T @ self.J_scaled))
            while self.Dp > 1.1 * self.delta:  # or self.Dp < 0.9 * self.delta:
                R = np.linalg.cholesky(self.J.T @ self.J + self.lmbda * np.eye(self.n))
                p = -np.linalg.solve(R @ R.T, self.J.T @ self.res)
                q = np.linalg.solve(R, p)
                self.lmbda += (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * (
                    (np.linalg.norm(p) - self.delta) / self.delta
                )
                self.lmbda = max(
                    1e-3, self.lmbda
                )  # Ensure that lambda is positive and not too small such that R exists
                R = np.linalg.cholesky(self.J.T @ self.J + self.lmbda * np.eye(self.n))
                self.p = -np.linalg.solve(R @ R.T, self.J.T @ self.res)
                self.Dp = np.linalg.norm(self.D @ self.p)
                print(self.lmbda, self.Dp)

        # Compute rho
        self.rho = (
            self.data["f"].iloc[-1] - self.func.f(self.data["x"].iloc[-1] + self.p)
        ) / (self.M(np.zeros_like(self.p)) - self.M(self.p))

        # Update the trust region radius
        if self.rho < 0.25:
            self.delta *= 0.25
        elif self.rho > 0.75 and np.abs(np.linalg.norm(self.p) - self.delta) < self.tol:
            self.delta = min(2 * self.delta, self.delta_max)

        # Update the data frame if the step is accepted
        if self.rho > self.eta:
            self.data.loc[self.data.index.max() + 1] = [
                self.data["x"].iloc[-1] + self.p,
                self.func.f(self.data["x"].iloc[-1] + self.p),
                self.func.df(self.data["x"].iloc[-1] + self.p),
                self.jac_func(self.data["x"].iloc[-1] + self.p),
                1,
            ]
        elif self.delta < 1e-10 * self.delta_max:
            warnings.warn("The trust region radius is too small")

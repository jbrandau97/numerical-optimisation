from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.linalg import orth
from typing import Type
import warnings

import os
import sys

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import your module
from utils import absObjective, Phi, visualise


class _trustRegion(visualise):
    """
    Base class for trust region methods.
    """

    def __init__(
        self,
        func: Type[absObjective],
        x_init: np.ndarray,
        delta_max: float = 3.0,
        eta: float = 0.1,
        tol: float = 1e-6,
        max_iter: int = 100,
        params: dict = None,
    ):
        """
        Constructor for the _trustRegion class.

        Args:
            radius (float): initial trust region radius
            maxIter (int): maximum number of iterations
            tol (float): tolerance for the stopping criterion
        """

        self.func = func
        self.x_init = x_init
        self.delta_max = delta_max
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        if params is None:
            self.params = {
                "ls_method": "strong_wolfe",
                "solver": "2d_subspace",
            }
        else:
            self.params = params

    def trustRegion(self) -> None:
        """
        Run the trust region algorithm.

        Args:
            None

        Returns:
            None
        """

        self.delta = 0.5 * self.delta_max
        self.data = pd.DataFrame(
            columns=[
                "x",
                "xind",
                "f",
                "df",
                "d2f",
                "delta",
                "rho",
                "active_TR",
                "gnorm",
            ]
        )
        self.data.loc[0] = [
            self.x_init,
            0,
            self.func.f(self.x_init),
            self.func.df(self.x_init),
            self.func.d2f(self.x_init),
            self.delta,
            0,
            True,
            np.linalg.norm(self.func.df(self.x_init)),
        ]

        for i in range(self.max_iter):
            # Define the model function
            self.M = (
                lambda p: self.data["f"].iloc[-1]
                + p @ self.data["df"].iloc[-1]
                + 0.5 * p @ self.data["d2f"].iloc[-1] @ p
            )

            # Compute the search direction
            match self.params["solver"]:
                case "2d_subspace":
                    self.p = self.solve2dSubspace()
                case "dogleg":
                    self.p = self.solveDogleg()
                case "cauchy":
                    self.p = self.solveCauchy()
                case _:
                    raise ValueError("Invalid solver")

            # Compute the actual reduction
            self.rho = (
                self.data["f"].iloc[-1] - self.func.f(self.data["x"].iloc[-1] + self.p)
            ) / (self.M(np.zeros_like(self.p)) - self.M(self.p))

            # Update the trust region radius
            if self.rho < 0.25:
                self.delta *= 0.25
            elif (
                self.rho > 0.75
                and np.abs(np.linalg.norm(self.p) - self.delta) < self.tol
            ):
                self.delta = min(2 * self.delta, self.delta_max)

            # Update the data frame if the step is accepted
            if self.rho > self.eta:
                self.data.loc[i + 1] = [
                    self.data["x"].iloc[-1] + self.p,
                    i + 1,
                    self.func.f(self.data["x"].iloc[-1] + self.p),
                    self.func.df(self.data["x"].iloc[-1] + self.p),
                    self.func.d2f(self.data["x"].iloc[-1] + self.p),
                    self.delta,
                    self.rho,
                    np.linalg.norm(self.p) > self.delta - np.finfo(float).eps,
                    np.linalg.norm(self.func.df(self.data["x"].iloc[-1] + self.p)),
                ]

                # Check for convergence
                if np.linalg.norm(self.data["df"].iloc[-1], ord=np.inf) < self.tol * (
                    1 + np.abs(self.data["f"].iloc[-1])
                ):
                    break
            elif self.delta < 1e-10 * self.delta_max:
                warnings.warn("The trust region radius is too small")
                break

            if i == self.max_iter - 1:
                warnings.warn("The maximum number of iterations has been reached")

    def solve2dSubspace(self) -> np.ndarray:
        """
        Solve the 2D subspace trust region problem.

        Args:
            None

        Returns:
            p (np.ndarray): the search direction
        """

        self.g = self.data["df"].iloc[-1]
        self.B = self.data["d2f"].iloc[-1]

        # Get an orthonormal basis for the subspace
        self.V = orth(np.array([self.g, np.linalg.solve(self.B, self.g)]))

        # Check if the subspace is rank-deficient
        if self.V.shape[1] > np.linalg.matrix_rank(self.V):
            warnings.warn("The subspace is rank-deficient")
            gBg = self.g.T @ self.B @ self.g
            self.tau = (
                1
                if gBg <= 0
                else min(np.linalg.norm(self.g) ** 3 / (self.delta * gBg), 1)
            )
            return -self.tau * self.delta / np.linalg.norm(self.g) * self.g
        else:
            self.gv = self.V.T @ self.g
            self.Bv = self.V.T @ self.B @ self.V
            self.a = -np.linalg.solve(self.Bv, self.gv)
            if np.linalg.norm(self.a) <= self.delta:
                return self.V @ self.a
            else:
                self.D, self.Q = np.linalg.eig(self.Bv)  # Eigendeomposition of Bv
                self.Qg = self.Q.T @ self.gv
                self.r = [
                    self.delta**2,
                    2 * self.delta**2 * (self.D[0] + self.D[1]),
                    self.delta**2
                    * (self.D[0] ** 2 + 4 * self.D[0] * self.D[1] + self.D[1] ** 2)
                    - self.Qg[0] ** 2
                    - self.Qg[1] ** 2,
                    2
                    * (
                        self.delta**2
                        * (self.D[0] ** 2 * self.D[1] + self.D[0] * self.D[1] ** 2)
                        - self.Qg[0] ** 2 * self.D[1]
                        - self.Qg[1] ** 2 * self.D[0]
                    ),
                    self.delta**2 * (self.D[0] ** 2 * self.D[1] ** 2)
                    - self.Qg[0] ** 2 * self.D[1] ** 2
                    - self.Qg[1] ** 2 * self.D[0] ** 2,
                ]  # Coefficients of the 4th degree polynomial
                self.t = np.roots(self.r)
                self._lambda = np.min(self.t[self.t + np.min(self.D) > 0])
                self.a = -np.linalg.solve(self.Bv + self._lambda * np.eye(2), self.gv)
                return self.V @ self.a

    def solveDogleg(self) -> np.ndarray:
        """
        Solve the dogleg trust region problem.

        Args:
            None

        Returns:
            p (np.ndarray): the search direction
        """

        self.g = self.data["df"].iloc[-1]
        self.B = self.data["d2f"].iloc[-1]

        # Compute the Newton step
        D, _ = np.linalg.eig(self.B)
        self.pN = -np.linalg.solve(self.B, self.g)

        # Check for positive definiteness of the Hessian and whether the Newton step is inside the trust region
        if np.all(D > 0) and np.linalg.norm(self.pN) <= self.delta:
            return self.pN
        else:
            self.gBg = self.g.T @ self.B @ self.g

            # Check for collinearity of the gradient and the Newton step
            if orth(np.array([self.g, self.pN])).shape[1] == 1:
                self.tau = (
                    1
                    if self.gBg <= 0
                    else min(np.linalg.norm(self.g) ** 3 / (self.delta * self.gBg), 1)
                )
                return (
                    -self.tau * self.delta / np.linalg.norm(self.g) * self.g
                )  # Return the Cauchy point
            else:
                self.pU = -((self.g.T @ self.g) / self.gBg) * self.g
                if np.linalg.norm(self.pU) >= self.delta:
                    self.tau = self.delta / np.linalg.norm(self.pU)
                else:
                    # Compute the intersection of the trust region with the line connecting the Cauchy and Newton points
                    self.r = [
                        np.linalg.norm(self.pN - self.pU) ** 2,
                        2 * (self.pU.T @ (self.pN - self.pU)),
                        np.linalg.norm(self.pU) ** 2 - self.delta**2,
                    ]
                    self.tau = np.max(np.roots(self.r) + 1)
                    return (
                        self.tau * self.pU
                        if self.tau < 1
                        else self.pU + (self.tau - 1) * (self.pN - self.pU)
                    )

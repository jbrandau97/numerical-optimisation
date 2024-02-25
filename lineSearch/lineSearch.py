from functools import lru_cache
import numpy as np
import pandas as pd
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


class lineSearch(visualise):
    """
    Base class for line search methods
    """

    def __init__(
        self,
        func: Type[absObjective],
        x_init: np.ndarray,
        alpha_max: float = 1,
        c1: float = 1e-4,
        c2: float = 0.9,
        rho: float = 0.2,
        tol: float = 1e-6,
        max_iter: int = 1000,
        params: dict = None,
    ) -> None:
        """
        Initialize the line search object

        Args:
            func (Type[absObjective]): the objective function
            x_init (np.ndarray): initial guess
            alpha_max (float): maximum step size
            c1 (float): constant for Armijo condition
            c2 (float): constant for curvature condition
            rho (float): step size reduction factor for backtracking
            tol (float): tolerance for the stopping criterion
            max_iter (int): maximum number of iterations
            params (dict): dictionary of method-specific parameters
                ls_method (str): line search method, either "backtracking", "strong_wolfe" or "constant"
                descent_method (str): descent method, either "steepest", "newton" or "quasi_newton"
                Optional parameters:
                    inv_hessian_update (str): quasi-Newton method for updating the inverse Hessian approximation H, either "bfgs" or "dfp"
                    inv_hessian_init (np.ndarray): initial inverse Hessian approximation H

        Returns:
            None
        """

        # Error handling
        if not isinstance(func, absObjective):
            raise TypeError("func must be an instance of absObjective")
        if not isinstance(x_init, np.ndarray):
            raise TypeError("x_init must be a numpy array")
        if not isinstance(alpha_max, (int, float)):
            raise TypeError("alpha_max must be a number")
        if not isinstance(c1, (int, float)) and c1 > 0 and c1 < c2 < 1:
            raise TypeError("c1 must be a positive number")
        if not isinstance(c2, (int, float)) and c1 < c2 < 1:
            raise TypeError("c2 must be a positive number")
        if not isinstance(rho, (int, float)) and 0 < rho < 1:
            raise TypeError("rho must be a number between 0 and 1")
        if not isinstance(tol, (int, float)) and tol > 0:
            raise TypeError("tol must be a positive number")
        if not isinstance(max_iter, int) and max_iter > 0:
            raise TypeError("max_iter must be a positive integer")

        # Set default parameters
        if params is None:
            self.params: dict = {
                "ls_method": "strong_wolfe",
                "descent_method": "steepest",
            }
        else:
            self.params: dict = params
        # Set attributes
        self.func = func
        self.x_init = x_init
        self.alpha_max = alpha_max
        self.c1 = c1
        self.c2 = c2
        self.rho = rho
        self.tol = tol
        self.max_iter = max_iter

    def steepestDescent(self) -> None:
        """
        Steepest descent method

        Args:
            None

        Returns:
            None
        """

        self.alpha_init = self.alpha_max
        self.data = pd.DataFrame(columns=["x", "f", "df", "d2f", "alpha", "gnorm"])
        self.data.loc[0] = [
            self.x_init,
            self.func.f(self.x_init),
            self.func.df(self.x_init),
            self.func.d2f(self.x_init),
            0,
            np.linalg.norm(self.func.df(self.x_init)),
        ]

        for i in range(self.max_iter):
            p = -self.data.loc[i, "df"]

            # Update the initial guess for step length alpha
            if i > 0:
                self.alpha_init = (
                    self.data.loc[i - 1, "alpha"]
                    * self.data.loc[i - 1, "gnorm"] ** 2
                    / self.data.loc[i, "gnorm"] ** 2
                )

            match self.params["ls_method"]:
                case "backtracking":
                    self.alpha = self.backtracking(self.data.loc[i, "x"], p)
                case "strong_wolfe":
                    self.alpha = self.strongWolfe(self.data.loc[i, "x"], p)
                case "constant":
                    self.alpha = self.alpha_max
                case other:
                    raise ValueError("Invalid line search method")

            self.data.loc[i + 1] = [
                self.data.loc[i, "x"] + self.alpha * p,
                self.func.f(self.data.loc[i, "x"] + self.alpha * p),
                self.func.df(self.data.loc[i, "x"] + self.alpha * p),
                self.func.d2f(self.data.loc[i, "x"] + self.alpha * p),
                self.alpha,
                np.linalg.norm(self.func.df(self.data.loc[i, "x"] + self.alpha * p)),
            ]

            if np.linalg.norm(self.data.loc[i + 1, "df"], ord=np.inf) < self.tol * (
                1 + np.abs(self.data.loc[i + 1, "f"])
            ):
                break

            if i == self.max_iter - 1:
                warnings.warn("Maximum number of iterations reached")

    def newton(self) -> None:
        """
        Newton-type methods:
            Newton's method using exact Hessian
            Quasi-Newton method using inverse Hessian approximations

        Args:
            None

        Returns:
            None
        """

        self.alpha_init = self.alpha_max
        match self.params["descent_method"]:
            case "newton":
                self.data = pd.DataFrame(
                    columns=["x", "f", "df", "d2f", "alpha", "gnorm"]
                )
                self.data.loc[0] = [
                    self.x_init,
                    self.func.f(self.x_init),
                    self.func.df(self.x_init),
                    self.func.d2f(self.x_init),
                    0,
                    np.linalg.norm(self.func.df(self.x_init)),
                ]
            case "quasi_newton":
                self.data = pd.DataFrame(
                    columns=["x", "f", "df", "d2f", "H", "alpha", "gnorm"]
                )
                self.data.loc[0] = [
                    self.x_init,
                    self.func.f(self.x_init),
                    self.func.df(self.x_init),
                    self.func.d2f(self.x_init),
                    (
                        self.params["inv_hessian_init"]
                        if "hessian_init" in self.params
                        else np.eye(len(self.x_init))
                    ),
                    0,
                    np.linalg.norm(self.func.df(self.x_init)),
                ]
            case other:
                raise ValueError("Invalid descent method")

        for i in range(self.max_iter):
            match self.params["descent_method"]:
                case "newton":
                    p = -np.linalg.solve(
                        self.data.loc[i, "d2f"], self.data.loc[i, "df"]
                    )
                case "quasi_newton":
                    p = -self.data.loc[i, "H"] @ self.data.loc[i, "df"]

            match self.params["ls_method"]:
                case "backtracking":
                    self.alpha = self.backtracking(self.data.loc[i, "x"], p)
                case "strong_wolfe":
                    self.alpha = self.strongWolfe(self.data.loc[i, "x"], p)
                case "constant":
                    self.alpha = self.alpha_max
                case other:
                    raise ValueError("Invalid line search method")

            match self.params["descent_method"]:
                case "newton":
                    self.data.loc[i + 1] = [
                        self.data.loc[i, "x"] + self.alpha * p,
                        self.func.f(self.data.loc[i, "x"] + self.alpha * p),
                        self.func.df(self.data.loc[i, "x"] + self.alpha * p),
                        self.func.d2f(self.data.loc[i, "x"] + self.alpha * p),
                        self.alpha,
                        np.linalg.norm(
                            self.func.df(self.data.loc[i, "x"] + self.alpha * p)
                        ),
                    ]
                case "quasi_newton":
                    y = np.reshape(
                        self.func.df(self.data.loc[i, "x"] + self.alpha * p)
                        - self.data.loc[i, "df"],
                        (2, 1),
                    )
                    s = np.reshape(self.alpha * p, (2, 1))
                    match self.params["inv_hessian_update"]:
                        case (
                            "dfp"
                        ):  # Davidon-Fletcher-Powell (DFP) formula with inverse Hessian approximation obtained by Sherman-Morrison-Woodbury formula
                            H = (
                                self.data.loc[i, "H"]
                                - (
                                    (self.data.loc[i, "H"] @ y)
                                    @ y.T
                                    @ self.data.loc[i, "H"]
                                )
                                / (y.T @ self.data.loc[i, "H"] @ y)
                                + (s @ s.T) / (y.T @ s)
                            )
                        case (
                            "bfgs"
                        ):  # Broyden-Fletcher-Goldfarb-Shanno (BFGS) formula with inverse Hessian approximation obtained by Sherman-Morrison-Woodbury formula
                            r = 1 / (y.T @ s)
                            H = (
                                np.eye(len(self.x_init)) - (r * (s @ y.T))
                            ) @ self.data.loc[i, "H"] @ (
                                np.eye(len(self.x_init)) - (r * (y @ s.T))
                            ) + (
                                r * (s @ s.T)
                            )

                    self.data.loc[i + 1] = [
                        self.data.loc[i, "x"] + self.alpha * p,
                        self.func.f(self.data.loc[i, "x"] + self.alpha * p),
                        self.func.df(self.data.loc[i, "x"] + self.alpha * p),
                        self.func.d2f(self.data.loc[i, "x"] + self.alpha * p),
                        H,
                        self.alpha,
                        np.linalg.norm(
                            self.func.df(self.data.loc[i, "x"] + self.alpha * p)
                        ),
                    ]

            if np.linalg.norm(self.data.loc[i + 1, "df"], ord=np.inf) < self.tol * (
                1 + np.abs(self.data.loc[i + 1, "f"])
            ):
                break

            if i == self.max_iter - 1:
                warnings.warn("Maximum number of iterations reached")

    def strongWolfe(self, x: np.ndarray, p) -> float:
        """
        Strong Wolfe line search method

        Args:
            x (np.ndarray): current iterate
            p (np.ndarray): search direction

        Returns:
            alpha (float): step size that satisfies the strong Wolfe conditions
        """

        w: float = 0.9  # parameter for linear interpolation
        Phi_ = Phi(self.func, x, p)  # create a Phi object --> linear model function
        alpha: list = [0, self.alpha_max]

        for i in range(self.max_iter):
            f = Phi_.f(alpha[-1])
            g = Phi_.df(alpha[-1])
            f0 = Phi_.f(0)
            g0 = Phi_.df(0)

            # Check the Armijo condition
            if f > f0 + self.c1 * alpha[-1] * g0 or (i > 1 and f >= alpha[-2]):
                return self.zoom(Phi_, alpha[-2], alpha[-1])
            # Check the curvature condition
            elif np.abs(g) <= -self.c2 * g0:
                return alpha[-1]
            # Check if the derivative is positive
            elif g >= 0:
                return self.zoom(Phi_, alpha[-1], alpha[-2])

            # Update alpha
            alpha.append(w * alpha[-1] + (1 - w) * self.alpha_max)

        return alpha[-1]

    def backtracking(self, x: np.ndarray, p: np.ndarray) -> float:
        """
        Backtracking line search method

        Args:
            x (np.ndarray): current iterate
            p (np.ndarray): search direction

        Returns:
            alpha (float): step size that satisfies the Armijo condition
        """

        alpha = self.alpha_max
        Phi_ = Phi(self.func, x, p)

        for i in range(self.max_iter):
            if Phi_.f(alpha) <= Phi_.f(0) + self.c1 * alpha * Phi_.df(0):
                return alpha
            else:
                alpha *= self.rho

        return alpha

    def zoom(self, Phi_: Type[Phi], alpha_lo: float, alpha_hi: float) -> float:
        """
        Zoom method for the strong Wolfe line search

        Args:
            Phi (Type[Phi]): linear model function
            alpha_lo (float): lower bound
            alpha_hi (float): upper bound

        Returns:
            alpha (float): step size that satisfies the strong Wolfe conditions
        """

        eps = np.finfo(float).eps  # machine epsilon

        for i in range(self.max_iter):
            alpha = (alpha_lo + alpha_hi) / 2
            f = Phi_.f(alpha)
            g = Phi_.df(alpha)
            f0 = Phi_.f(0)
            g0 = Phi_.df(0)

            if np.abs(alpha_hi - alpha_lo) < eps:
                warnings.warn("Interval is too small")
                return alpha

            # Check the Armijo condition
            if f > f0 + self.c1 * alpha * g0 or f >= Phi_.f(alpha_lo):
                alpha_hi = alpha
            else:
                # Check the curvature condition
                if np.abs(g) <= -self.c2 * g0:
                    return alpha
                # Check if the derivative is positive
                if g * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha

        return alpha

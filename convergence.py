import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, Type

from utils import absObjective


class _convergence:
    """
    Base class for convergence analysis.
    """

    def __init__(
        self,
        data: object,
        func: Type[absObjective] = None,
        p: Union[int, str] = 2,
        x_min: np.ndarray = None,
        hess: np.ndarray = None,
    ):
        """
        Initialize the class and compute error norms.

        Args:
            data (object): The data object.
            Optional:
                func (Type[absObjective]): The objective function.
                p (int or str): The order of the error norm.
                x_min (np.ndarray): The minimiser.
                hess (np.ndarray): The Hessian matrix at the minimiser.
        """

        self.info = data
        self.func = func
        self.data = pd.DataFrame(columns=["x_err", "f_err", "df_err"])

        # Check if the true minimiser is provided and take the last iterate if not
        self.x_min = x_min if x_min is not None else self.info["x"].iloc[-1]

        # Compute error norms for the iterates
        if isinstance(p, str) and p.upper() == "M":
            self.p = 2
            self.M = hess if hess is not None else self.func.d2f(self.x_min)
            x_err = [
                self.info.data["x"].iloc[i] - self.x_min
                for i in range(len(self.info.data["x"]))
            ]
            x_err = [np.sqrt(x @ self.M @ x) for x in x_err]
        else:
            x_err = [
                self.info.data["x"].iloc[i] - self.x_min
                for i in range(len(self.info.data["x"]))
            ]
            x_err = [np.linalg.norm(x, ord=p) for x in x_err]
        self.data["x_err"] = x_err

        # Compute error norms for the function values
        if self.func is not None:
            f_err = [np.abs(f - self.func.f(self.x_min)) for f in self.info.data["f"]]
            self.data["f_err"] = f_err

        # Compute error norms for the gradient values
        df_err = [np.linalg.norm(df, ord=p) for df in self.info.data["df"]]
        self.data["df_err"] = df_err

    def plot(self, q: int = 1.4) -> None:
        """
        Plot the error norms.

        Args:
            q (int): Exponent for Q-convergence.

        Returns:
            None
        """

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Q-convergence plot
        self.data["x_err_q"] = np.divide(
            self.data["x_err"].iloc[1:], np.power(self.data["x_err"].iloc[:-1], q)
        )
        self.data.plot(
            ax=ax[0],
            y="x_err_q",
            title=f"Q-convergence for q = {q}",
            xlabel="Iteration",
            ylabel=r"$\frac{\|x_{k+1}-x^\star\|}{\|x_k-x^\star\|^q}$",
        )

        # Algebraic convergence plot - log-log scale
        self.data.plot(
            ax=ax[1],
            y="x_err",
            title="Algebraic convergence",
            xlabel="Iteration",
            ylabel=r"$\|x_k-x^\star\|$",
        )
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")

        # Exponential convergence plot - semi-log scale
        self.data.plot(
            ax=ax[2],
            y="x_err",
            title="Exponential convergence",
            xlabel="Iteration",
            ylabel=r"$\|x_k-x^\star\|$",
        )
        ax[2].set_yscale("log")

        plt.tight_layout()
        plt.show()

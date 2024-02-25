import abc
import matplotlib.pyplot as plt
import numpy as np
from typing import Type


class absObjective:
    """implements abstract base class for objective function"""

    def _init_(self):
        pass

    @abc.abstractclassmethod
    def f(self, x: np.array):
        """objective function"""
        return NotImplementedError

    @abc.abstractclassmethod
    def df(self, x: np.array):
        """objective function gradient"""
        return NotImplementedError

    @abc.abstractclassmethod
    def d2f(self, x: np.array):
        """ojective function Hessian"""
        return NotImplementedError


class Phi:
    def __init__(self, func: Type[absObjective], x: np.ndarray, p: np.ndarray):
        self.func = func
        self.x = x
        self.p = p

    def f(self, alpha: float):
        """Return the value of the function at x + alpha * p"""
        return self.func.f(self.x + alpha * self.p)

    def df(self, alpha: float):
        """Return the value of the derivative of the function at x + alpha * p"""
        return self.func.df(self.x + alpha * self.p) @ self.p


class visualise:
    def plot_contour(
        self, grid: np.ndarray = [-2, 2, -2, 2], npoints: int = 100, ncontours: int = 20
    ):
        """
        Plot contour of the objective function
        """
        x = np.linspace(grid[0], grid[1], npoints)
        y = np.linspace(grid[2], grid[3], npoints)
        X, Y = np.meshgrid(x, y)
        Z = self.func.f(np.array([X, Y]))
        fig, ax = plt.subplots()
        ax.contour(X, Y, Z, levels=ncontours, cmap="viridis")
        ax.plot(
            self.data["x"].apply(lambda x: x[0]),
            self.data["x"].apply(lambda x: x[1]),
            "r.-",
        )
        ax.set_title("Contour plot of the objective function")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()

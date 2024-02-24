import abc
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

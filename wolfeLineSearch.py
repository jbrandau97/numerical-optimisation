import numpy as np
import pandas as pd
from typing import Type
import warnings


class _wolfeLineSearch:
    """
    This class implements the strong Wolfe Line Search algorithm.
    """

    def __init__(
        self, c1: float = 1e-4, c2: float = 0.9, w: float = 0.9, maxIter: int = 100
    ):
        """
        Constructor for the _wolfeLineSearch class.

        Args:
            c1 (float): constant for Armijo condition
            c2 (float): constant for curvature condition
            w (float): constant for the weight of the previous step size
            maxIter (int): maximum number of iterations
        """

        if w < 0 or w > 1:
            raise ValueError("w must be between 0 and 1")

        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.maxIter = maxIter

    def strongWolfe(self, x: np.ndarray, p: np.ndarray) -> float:
        """
        Strong Wolfe Line Search algorithm.

        Args:
            x (np.ndarray): current point
            p (np.ndarray): search direction

        Returns:
            alpha (float): step size
        """

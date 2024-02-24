import numdifftools as nd
import numpy as np

from lineSearch import *
from utils import absObjective


class shitFunction(absObjective):
    def f(self, x: np.array):
        return x[0] ** 2 + x[1] ** 2

    def df(self, x: np.array):
        return np.array([2 * x[0], 2 * x[1]])

    def d2f(self, x: np.array):
        return np.array([[2, 0], [0, 2]])


l1 = lineSearch(shitFunction(), np.array([1, 1]))
l1.steepestDescent()
print(l1.data)

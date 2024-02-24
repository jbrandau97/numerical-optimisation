import numdifftools as nd
import numpy as np

from lineSearch import lineSearch
from utils import absObjective


class shitFunction(absObjective):
    def f(self, x: np.array):
        return (x[1] + np.log(x[0])) ** 2 + (x[1] - x[0]) ** 2

    def df(self, x: np.array):
        return nd.Gradient(self.f)(x)

    def d2f(self, x: np.array):
        return nd.Hessian(self.f)(x)


l1 = lineSearch(shitFunction(), np.array([1, 1]))
l1.steepestDescent()
print(l1.data)
l1.plot_contour()

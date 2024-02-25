import sys
import os
import numdifftools as nd
import numpy as np

from lineSearch import lineSearch

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import your module
from utils import absObjective, Phi


class shitFunction(absObjective):
    def f(self, x: np.array):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def df(self, x: np.array):
        return nd.Gradient(self.f)(x)

    def d2f(self, x: np.array):
        return nd.Hessian(self.f)(x)


l1 = lineSearch(
    shitFunction(),
    np.array([1.0, -2.0]),
    params={
        "ls_method": "strong_wolfe",
        "descent_method": "quasi_newton",
        "inv_hessian_update": "bfgs",
    },
)
l1.newton()

print(f"The minimum is at {l1.data["x"].iloc[-1]} with value {l1.data["f"].iloc[-1]} reached in {len(l1.data)-1} iterations.")

l1.plot_contour()

import sys
import os
import numdifftools as nd
import numpy as np

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import your module
from utils import absObjective, Phi
from trustRegion import _trustRegion


class shitFunction(absObjective):
    def f(self, x: np.array):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def df(self, x: np.array):
        return nd.Gradient(self.f)(x)

    def d2f(self, x: np.array):
        return nd.Hessian(self.f)(x)


l1 = _trustRegion(
    shitFunction(),
    np.array([1.0, -2.0]),
    params={
        "solver": "2d_subspace",
    },
)
l1.trustRegion()

print(f"The minimum is at {l1.data["x"].iloc[-1]} with value {l1.data["f"].iloc[-1]} reached in {len(l1.data)-1} iterations.")

l1.plot_contour(np.array([-2, 2, -2.5, 4]), npoints=100, ncontours=20)

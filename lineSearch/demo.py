import sys
import os
import numdifftools as nd
import numpy as np

from lineSearch import *

import os
import sys

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import your module
from utils import absObjective, Phi


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

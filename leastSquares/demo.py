import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import os
import sys
from time import time

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/lineSearch")

# Now you can import your module
from utils import absObjective, Phi
from lineSearch import _lineSearch
from leastSquares import _leastSquares
from convergence import _convergence


# Define the model
def model(t: np.ndarray, x: np.ndarray = np.array([3, 150, 2])):
    return (x[0] + x[1] * t**2) * np.exp(-x[2] * t)


# Define the grid of time points
t = np.linspace(0, 4, 200)
# Maximises the absolute value of the model function over the interval (0, 4]
t_max = t[np.argmax(np.abs(model(t)))]
# Generate the noisy measurements
np.random.seed(0)
y = model(t) + np.random.normal(size=t.size, loc=0, scale=0.05 * np.abs(model(t_max)))

# Define the initial guess
x = np.array([1.0, 1.0, 1.0])


# Define the residual function
def res(x: np.array):
    return y - (x[0] + x[1] * t**2) * np.exp(-x[2] * t)


# Define the Jacobian function
def jac(x: np.array):
    return nd.Gradient(lambda x: res(x))(x)


# Define the objective function
class leastSquaresObjective(absObjective):
    def __init__(self, t: np.array, y: np.array):
        self.t = t
        self.y = y

    def f(self, x: np.array):
        return 0.5 * np.linalg.norm(res(x), ord=2) ** 2

    def df(self, x: np.array):
        return jac(x).T @ res(x)

    def d2f(self, x: np.array):
        return jac(x).T @ jac(x)  # Approximation of the Hessian


# Generate the least squares object
l1 = _leastSquares(
    leastSquaresObjective(t, y),
    x,
    max_iter=100,
    params={
        "ls_method": "strong_wolfe",
    },
    c2=0.1,
)

# Compare the different decomposition methods
chol_start = time()
l1.gaussNewton(res, jac, "Cholesky")
chol_end = time()
qr_start = time()
l1.gaussNewton(res, jac, "QR")
qr_end = time()
svd_start = time()
l1.gaussNewton(res, jac, "SVD")
svd_end = time()

# Print the time taken for each decomposition method
print(f"SVD: {svd_end - svd_start}")
print(f"QR: {qr_end - qr_start}")
print(f"Cholesky: {chol_end - chol_start}")

# Print the estimated model parameters
print(l1.data["x"].iloc[-1])

# Overplot the data with the estimated model
plt.figure()
plt.plot(t, y, "b.", label="Noisy measurements")
plt.plot(
    t,
    model(t, l1.data["x"].iloc[-1]),
    "r-",
    label="Estimated model",
)
plt.legend()
plt.show()

conv = _convergence(l1, leastSquaresObjective(t, y), x_min=np.array([3, 150, 2]))
conv.plot()

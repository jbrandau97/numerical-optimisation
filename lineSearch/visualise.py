import matplotlib.pyplot as plt
import numpy as np


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

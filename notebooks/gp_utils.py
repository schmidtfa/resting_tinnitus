import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def plot_kernel_function(
    kernel_function,
    max_distance=1,
    resolution=100,
    label=None,
    ax=None,
    **line_kwargs
):
    

    def _plot_line(xs, ys, **plot_kwargs):
        """Plot line with consistent style (e.g. bordered lines)"""
        linewidth = plot_kwargs.get("linewidth", 3)
        plot_kwargs["linewidth"] = linewidth

        # Copy settings for background
        background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
        background_plot_kwargs["linewidth"] = linewidth + 2
        background_plot_kwargs["color"] = "white"
        del background_plot_kwargs["label"]  # no legend label for background

        plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
        plt.plot(xs, ys, **plot_kwargs, zorder=31)

    """Helper to plot a kernel function"""
    X = np.linspace(0, max_distance, resolution)[:, None]
    covariance = kernel_function(X, X)
    distances = np.linspace(0, max_distance, resolution)
    if ax is not None:
        plt.sca(ax)
    _plot_line(distances, covariance[0, :], label=label, **line_kwargs)
    plt.xlim([0, max_distance])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("|X1-X2|")
    plt.ylabel("covariance")
    if label is not None:
        plt.legend()




def quadratic_distance_kernel(X0, X1, eta=1, sigma=.5):
    # Use linear algebra identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    X0_norm = np.sum(X0 ** 2, axis=-1)
    X1_norm = np.sum(X1 ** 2, axis=-1)
    squared_distances = X0_norm[:, None] + X1_norm[None, :] - 2 * X0 @ X1.T
    rho = 1 / sigma ** 2
    return eta ** 2 * np.exp(-rho * squared_distances)



def ornstein_uhlenbeck_kernel(X0, X1, eta=1, rho=4):
    distances = np.abs(X1[None, :] - X0[:, None])
    return eta * np.exp(-rho * distances)




def periodic_kernel(X0, X1, eta=1, sigma=1, periodicity=.5):
    distances = np.sin((X1[None, :] - X0[:, None]) * periodicity) ** 2
    rho = 2 / sigma ** 2
    return eta ** 2 * np.exp(-rho * distances)



def plot_predictive_covariance(predictive, n_samples=50, color='C0', label=None, max_distance=16):

    eta_samples = predictive['eta'].values[0, :n_samples] ** .5
    sigma_samples = 1 / predictive['rho'].values[0, :n_samples] ** .5
    
    for ii, (eta, sigma) in enumerate(zip(eta_samples, sigma_samples)):
        label = label if ii == 0 else None

        kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
        plot_kernel_function(kernel_function, color=color, label=label, alpha=.5, linewidth=5, max_distance=max_distance)


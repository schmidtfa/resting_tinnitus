#%%
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pandas as pd
import xarray as xr
from sklearn.preprocessing import scale
import scipy
from scipy.special import expit
import pytensor.tensor as pt
import graphviz

import seaborn as sns
import rdata
from os.path import join
from functools import partial

#%%

data_dir = 'https://github.com/kpjmcg/Statistical-Rethinking-2023-Python-Notes/tree/master/Data'

kline2 = pd.read_csv('kline2.csv',sep=';')
kline2
# %%
parsed = rdata.parser.parse_file('islandsDistMatrix.rda')
converted = rdata.conversion.convert(parsed)
islandsDistMatrix = converted['islandsDistMatrix']
islandsDistMatrix.to_numpy()
# %%
# Data / coords
CULTURE_ID, CULTURE = pd.factorize(kline2.culture.values)
ISLAND_DISTANCES = islandsDistMatrix.to_numpy()
TOOLS = kline2.total_tools.values.astype(int)
coords = {"culture": CULTURE}

#%%

with pm.Model(coords=coords) as distance_model:
        
    # Priors
    alpha_bar = pm.Normal("alpha_bar", 3, 0.5)
    eta_squared = pm.Exponential("eta_squared", 2)
    rho_squared = pm.Exponential("rho_squared", 0.5)
    
    # Gaussian Process
    kernel_function = eta_squared * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_squared)
    GP = pm.gp.Latent(cov_func=kernel_function)
    alpha = GP.prior("alpha", X=ISLAND_DISTANCES, dims="culture")
    
    # Likelihood
    lambda_T = pm.math.exp(alpha_bar + alpha[CULTURE_ID])
    pm.Poisson("T", lambda_T, dims='culture', observed=TOOLS)


#%%
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



def plot_predictive_covariance(predictive, n_samples=30, color='C0', label=None):

    eta_samples = predictive['eta_squared'].values[0, :n_samples] ** .5
    sigma_samples = 1 / predictive['rho_squared'].values[0, :n_samples] ** .5
    
    for ii, (eta, sigma) in enumerate(zip(eta_samples, sigma_samples)):
        label = label if ii == 0 else None

        kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
        plot_kernel_function(kernel_function, color=color, label=label, alpha=.5, linewidth=5, max_distance=7)


with distance_model:
    prior_predictive = pm.sample_prior_predictive(random_seed=12).prior

plot_predictive_covariance(prior_predictive, label='prior')
plt.ylim([0, 2]);
plt.title("Prior Covariance Functions");


# %%

with distance_model:
    distance_inference = pm.sample(target_accept=.99)
# %%
sns.set_style('ticks')
sns.set_context('poster')
plot_predictive_covariance(prior_predictive, color='k', label='prior')
plot_predictive_covariance(distance_inference.posterior, label='posterior')
plt.ylim([0, 2]);
# %%

#%%
import arviz as az

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pytensor.tensor.var import Variable
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.stats import norm, pearsonr
from xarray import concat

from .tree import Tree

TensorLike = Union[npt.NDArray[np.float_], pt.TensorVariable]

#%%

def _sample_posterior(
    all_trees: List[List[Tree]],
    X: TensorLike,
    rng: np.random.Generator,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    excluded: Optional[List[int]] = None,
    shape: int = 1,
) -> npt.NDArray[np.float_]:
    """
    Generate samples from the BART-posterior.

    Parameters
    ----------
    all_trees : list
        List of all trees sampled from a posterior
    X : tensor-like
        A covariate matrix. Use the same used to fit BART for in-sample predictions or a new one for
        out-of-sample predictions.
    rng : NumPy RandomGenerator
    size : int or tuple
        Number of samples.
    excluded : Optional[npt.NDArray[np.int_]]
        Indexes of the variables to exclude when computing predictions
    """
    stacked_trees = all_trees

    if isinstance(X, Variable):
        X = X.eval()

    if size is None:
        size_iter: Union[List, Tuple] = (1,)
    elif isinstance(size, int):
        size_iter = [size]
    else:
        size_iter = size

    flatten_size = 1
    for s in size_iter:
        flatten_size *= s

    idx = rng.integers(0, len(stacked_trees), size=flatten_size)

    trees_shape = len(stacked_trees[0])
    leaves_shape = shape // trees_shape

    pred = np.zeros((flatten_size, trees_shape, leaves_shape, X.shape[0]))

    for ind, p in enumerate(pred):
        for odim, odim_trees in enumerate(stacked_trees[idx[ind]]):
            for tree in odim_trees:
                p[odim] += tree.predict(x=X, excluded=excluded, shape=leaves_shape)

    # pred.reshape((*size_iter, shape, -1))
    return pred.transpose((0, 3, 1, 2)).reshape((*size_iter, -1, shape))

#%%

def plot_variable_importance(
    idata: az.InferenceData,
    bartrv: Variable,
    X: npt.NDArray[np.float_],
    labels: Optional[List[str]] = None,
    sort_vars: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    samples: int = 100,
    random_seed: Optional[int] = None,
) -> Tuple[npt.NDArray[np.int_], List[plt.Axes]]:
    """
    Estimates variable importance from the BART-posterior.

    Parameters
    ----------
    idata: InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray[np.float_]
        The covariate matrix.
    labels : Optional[List[str]]
        List of the names of the covariates. If X is a DataFrame the names of the covariables will
        be taken from it and this argument will be ignored.
    sort_vars : bool
        Whether to sort the variables according to their variable importance. Defaults to True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    samples : int
        Number of predictions used to compute correlation for subsets of variables. Defaults to 100
    random_seed : Optional[int]
        random_seed used to sample from the posterior. Defaults to None.

    Returns
    -------
    idxs: indexes of the covariates from higher to lower relative importance
    axes: matplotlib axes
    """

    if bartrv.ndim == 1:  # type: ignore
        shape = 1
    else:
        shape = bartrv.eval().shape[0]

    if hasattr(X, "columns") and hasattr(X, "values"):
        labels = X.columns
        X = X.values

    n_draws = idata["posterior"].dims["draw"]
    half = n_draws // 2
    f_half = idata["sample_stats"]["variable_inclusion"].sel(draw=slice(0, half - 1))
    s_half = idata["sample_stats"]["variable_inclusion"].sel(draw=slice(half, n_draws))

    var_imp_chains = concat([f_half, s_half], dim="chain", join="override").mean(("draw")).values
    var_imp = idata["sample_stats"]["variable_inclusion"].mean(("chain", "draw")).values
    if labels is None:
        labels_ary = np.arange(len(var_imp))
    else:
        labels_ary = np.array(labels)

    rng = np.random.default_rng(random_seed)

    ticks = np.arange(len(var_imp), dtype=int)
    idxs = np.argsort(var_imp)
    subsets = [idxs[:-i].tolist() for i in range(1, len(idxs))]
    subsets.append(None)  # type: ignore

    if sort_vars:
        indices = idxs[::-1]
    else:
        indices = np.arange(len(var_imp))

    chains_mean = (var_imp / var_imp.sum())[indices]
    chains_hdi = az.hdi((var_imp_chains.T / var_imp_chains.sum(axis=1)).T)[indices]

    all_trees = bartrv.owner.op.all_trees

    predicted_all = _sample_posterior(
        all_trees, X=X, rng=rng, size=samples, excluded=None, shape=shape
    )

    ev_mean = np.zeros(len(var_imp))
    ev_hdi = np.zeros((len(var_imp), 2))
    for idx, subset in enumerate(subsets):
        predicted_subset = _sample_posterior(
            all_trees=all_trees,
            X=X,
            rng=rng,
            size=samples,
            excluded=subset,
            shape=shape,
        )
        pearson = np.zeros(samples)
        for j in range(samples):
            pearson[j] = (
                pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0]
            ) ** 2

    return pearsonr

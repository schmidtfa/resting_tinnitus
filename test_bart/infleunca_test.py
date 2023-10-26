#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc_bart as pmb
# %%
RANDOM_SEED = 8457
rng = np.random.RandomState(RANDOM_SEED)
az.style.use("arviz-white")
plt.rcParams["figure.dpi"] = 300
# %%
sin = np.loadtxt("space_influenza.csv", skiprows=1, delimiter=",")

X = sin[:, 1][:, None]
Y = sin[:, 2]
Y_jittered = np.random.normal(Y, 0.02)
# %%
idatas = {}
ms = ["10", "20", "50", "100", "200"]

for m in ms:
    with pm.Model() as model:
        μ = pmb.BART("μ", X, Y, m=int(m))
        p = pm.Deterministic("p", pm.math.sigmoid(μ))
        y = pm.Bernoulli("y", p=p, observed=Y)
        idata = pm.sample(idata_kwargs={"log_likelihood": True})

    idatas[m] = idata
# %%
idatas["20"]
# %%
X.shape
# %%

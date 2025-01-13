#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('ticks')
sns.set_context('poster')

cmap = sns.color_palette('deep').as_hex()

tin_c = cmap[1]
no_tin_c = cmap[0]
#%%



df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
df_pe = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params.csv')

df = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
df_nt = df.query('tinnitus == False')
# %%
n_bins = 1000
f, ax = plt.subplots(figsize=(5,5))
ax.hist(df_nt['Offset'], bins=n_bins, color=no_tin_c, edgecolor=no_tin_c)
sns.despine()
# %%
f, ax = plt.subplots(figsize=(5,5))
ax.hist(df_nt['Knee Frequency (Hz)'][df_nt['Knee Frequency (Hz)'] < 100], bins=n_bins, edgecolor=no_tin_c, color=no_tin_c,)
sns.despine()

#%%
f, ax = plt.subplots(figsize=(5,5))
ax.hist(df_nt['tau'][df_nt['Knee Frequency (Hz)'] < 100], bins=n_bins, edgecolor=no_tin_c, color=no_tin_c,)
ax.set_xlim(0.0, .25)
sns.despine()

#%%
f, ax = plt.subplots(figsize=(5,5))
ax.hist(df_nt['Exponent_1'], bins=n_bins, color=no_tin_c, edgecolor=no_tin_c)
ax.set_xlim(-0.01, .05)
sns.despine()

#%%
f, ax = plt.subplots(figsize=(5,5))
ax.hist(df_nt['Exponent_2'], bins=n_bins, color=no_tin_c, edgecolor=no_tin_c)
#ax.set_xlim(0.5, 3)
sns.despine()

# %%
f, ax = plt.subplots(figsize=(5,5))
ax.hist(df_nt['n_peaks'], bins=25, edgecolor=no_tin_c, color=no_tin_c,)
ax.set_xlim(0, 10)
sns.despine()

# %%

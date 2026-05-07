# %%
from neurodsp.sim import sim_synaptic_current, set_random_seed
from neurodsp.utils import create_times
import pandas as pd

import scipy.signal as dsp
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')

set_random_seed(43)

# %% based on info from gao 2017)
def make_gao_ei(n_secs, fs, ie_ratio=4):
    """Do lfp gao style parameters as in Gao et al. 2017"""

    mv_rest = -65
    mv_rev_e, mv_rev_i = 0, -80
    ampa = sim_synaptic_current(n_secs, fs=fs, n_neurons=8000, firing_rate=2, tau_r=0.0001, tau_d=0.002, t_ker=1) * (
        mv_rest - mv_rev_e
    )
    gaba = sim_synaptic_current(n_secs, fs=fs, n_neurons=2000, firing_rate=5, tau_r=0.0005, tau_d=0.01, t_ker=1) * (
        mv_rest - mv_rev_i
    ) * ie_ratio #use ie instead of ei to make the ratios simpler

    lfp = ampa + gaba

    return lfp


nsecs = 5*60
fs = 1000
times = create_times(nsecs, fs)

ratios = np.arange(2., 6., step=0.01) #1/
lfps = [make_gao_ei(n_secs=nsecs, fs=fs, ie_ratio=i) for i in ratios]
freqs, psds = zip(*([dsp.welch(lfp, 
                               fs=fs, 
                               nperseg=4 * fs, 
                               noverlap=2 * fs) for lfp in lfps]))

# %%
cmap = sns.color_palette('magma_r', np.shape(lfps)[0])
f, ax = plt.subplots(figsize=(6, 6))
for ix, psd in enumerate(psds):
    ax.loglog(freqs[0], psd, color=cmap[ix])

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (a.u.)')
sns.despine()

#f.savefig('ei_changes_spec.svg')
# %%
from pyrasa import irasa
from pyrasa.utils.aperiodic_utils import compute_aperiodic_model

#%%
param_list, param_list_gao = [], []
for psd, ratio in zip(psds, ratios):
    params = compute_aperiodic_model(psd, freqs[0], 
                                     fit_func='knee', 
                                     fit_bounds=(1, 100)).aperiodic_params 
    params['I:E'] = ratio

    #% get gao style effect
    param_list.append(params)

    params_gao = compute_aperiodic_model(psd, freqs[0], 
                                     fit_func='fixed', 
                                     fit_bounds=(30, 50)).aperiodic_params 
    
    params_gao['I:E'] = ratio
    param_list_gao.append(params_gao)

#%%
df_gao = pd.concat(param_list_gao)[['Exponent', 'I:E']]

df_no_irasa = pd.concat(param_list)

df_mg = df_no_irasa.merge(df_gao, on='I:E')

#%%
titles = ['Offset', 'Exponent_1', 'Exponent_2', 
          'Knee Frequency (Hz)', 'tau', 'Exponent']



f, axes = plt.subplots(figsize=(12, 12), ncols=3, nrows=2)
for ax, title in zip(axes.flatten(), titles): 

    sns.scatterplot(df_mg, x='I:E', 
                    y=title, 
                    hue='I:E', 
                    palette='magma', 
                    legend=False,
                    ax=ax)
    
    if title == 'Exponent_1':
        ax.set_ylabel('Exponent (pre Knee)')
    elif title == 'Exponent_2':
        ax.set_ylabel('Exponent (post Knee)')
    elif title == 'Exponent':
        ax.set_ylabel('Exponent \n (30-50Hz; Gao et al. 2017)')

sns.despine()
f.tight_layout()
 
#f.savefig('ei_spec_param_change_norasa.svg')

#%%

aps, aps_gao = [], []
for lfp, ratio in zip(lfps, ratios):

    ap = irasa(lfp, fs=fs, nperseg=4 * fs, 
               noverlap=2 * fs, 
               band=(1, 100)).fit_aperiodic_model('knee')

    ap_df = ap.aperiodic_params
    ap_df['I:E'] = ratio
    aps.append(ap_df)

# %%
df_ap = pd.concat(aps)

titles = ['Offset', 'Exponent_1', 'Exponent_2', 
          'Knee Frequency (Hz)', 'tau']


f, axes = plt.subplots(figsize=(12, 12), ncols=3, nrows=2)
for ax, title in zip(axes.flatten(), titles): 

    sns.scatterplot(df_ap, x='I:E', 
                    y=title, 
                    hue='I:E', 
                    palette='magma', 
                    legend=False,
                    ax=ax)
    
    if title == 'Exponent_1':
        ax.set_ylabel('Exponent (pre Knee)')
    elif title == 'Exponent_2':
        ax.set_ylabel('Exponent (post Knee)')
    elif title == 'Exponent':
        ax.set_ylabel('Exponent \n (30-50Hz; Gao et al. 2017)')

sns.despine()
f.tight_layout()
 
 
#f.savefig('ei_spec_param_change_irasa.svg')

#%%
df_no_irasa['type'] = 'no_irasa'
df_ap['type'] = 'irasa'


import scipy.stats as st

rs = {}
for feat in ['Offset', 'Exponent_1', 'Exponent_2', 
          'Knee Frequency (Hz)', 'tau',]:

    r, p = st.pearsonr(df_no_irasa[feat], df_ap[feat])
    rs.update({feat: r})


rs


# %%
df_ap.corr('spearman', numeric_only=True)


#%%
df_no_irasa.corr('spearman', numeric_only=True)

# %%
df_ap_gao.corr('spearman', numeric_only=True)

f, axes = plt.subplots(figsize=(5, 5))
sns.scatterplot(df_ap_gao, 
                x='I:E', 
                y='Exponent', 
                hue='I:E', 
                palette='magma', 
                legend=False,
                    #ax=ax
                )
sns.despine()
f.tight_layout()

f.savefig('ei_expo_change_30_60.svg')
#%%
# %%
titles = ['Exponent_1', 'Exponent_2', 'Knee Frequency (Hz)', 'tau']

f, axes = plt.subplots(figsize=(12, 12), ncols=2, nrows=2)
for ax, title in zip(axes.flatten(), titles): 

    sns.scatterplot(df_ap, x='I:E', 
                    y=title, 
                    hue='I:E', 
                    palette='magma', 
                    legend=False,
                    ax=ax)
    ax.set_title(title)

sns.despine()
f.tight_layout()

#f.savefig('ei_spec_param_change.svg')
# %%
df_neural_info = pd.read_csv('../data/synaptic_kernels_fullcites.csv')
# %%
df_neural_info
# %%
mv_rest = -65
n_secs = 60*5
ie_ratio = 2

rev_e, rev_gaba_a, rev_gaba_b = 0, -70, -100 # reversal potentials

#10_000 neurons 80/20 rule for excitatory and inhibitory

# ampa + nmda = 8000 Neurons
ampa = sim_synaptic_current(n_secs, fs=fs, n_neurons=6000, firing_rate=.8, tau_r=0.0005, tau_d=0.002, t_ker=1) * (
    mv_rest - rev_e
)
nmda = sim_synaptic_current(n_secs, fs=fs, n_neurons=2000, firing_rate=.8, tau_r=0.0077, tau_d=0.07, t_ker=1) * (
    mv_rest - rev_e
)

# gaba_a + gaba_b = 2000 Neurons
gaba_a = sim_synaptic_current(n_secs, fs=fs, n_neurons=1700, firing_rate=5, tau_r=0.0005, tau_d=0.01, t_ker=1) * (
    mv_rest - rev_gaba_a
) * ie_ratio 

gaba_b = sim_synaptic_current(n_secs, fs=fs, n_neurons=300, firing_rate=5, tau_r=0.015, tau_d=0.04, t_ker=1) * (
    mv_rest - rev_gaba_b
) #* ie_ratio 
times = create_times(n_secs, fs)

# %%
df = pd.DataFrame({'ampa': ampa,
                   'nmda': nmda,
                   'gaba_a': gaba_a,
                   'gaba_b': gaba_b,
                   'time': times}).melt(id_vars='time', var_name='current', value_name='amplitude')
# %%
from matplotlib.lines import Line2D

currents = df['current'].unique()
palette  = sns.color_palette('deep', n_colors=len(currents))

# 2 — prepare the grid of sub-plots
fig, axes = plt.subplots(nrows=4, figsize=(12, 6), sharex=True)

for ix, (ax, current) in enumerate(zip(axes.flatten(), currents)):
    sns.lineplot(
        data = df.query('time < .8 and current == @current'),
        x    = 'time',
        y    = 'amplitude',
        color= palette[ix],
        ax   = ax,
        #lw   = 1.8
    )
    sns.despine(ax=ax)
    ax.set_ylabel('')          # tidy y-labels

# 3 — craft explicit legend handles (never fails)
legend_handles = [
    Line2D([0], [0], color=palette[ix], lw=2, label=str(cur))
    for ix, cur in enumerate(currents)
]

# 4 — add a single legend to the whole figure
fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=len(currents),
    bbox_to_anchor=(0.5, 1.05),
    frameon=False
)

fig.tight_layout()
fig.savefig('ts_all_currents.svg')
plt.show()
# %%
df_freqs = []
for current in currents:

    cur_c = df.query(f'current == "{current}"')

    freqs, psd = dsp.welch(cur_c['amplitude'], 
                           fs=fs, 
                           nperseg=8*fs, 
                           noverlap=4*fs)

    cur_psd = pd.DataFrame({'Frequency (Hz)': freqs,
                            'Power (a.u.)': psd,
                            'current': current})

    df_freqs.append(cur_psd)
# %%
df_freq = pd.concat(df_freqs)
# %%
f, ax = plt.subplots(figsize=(5,5))
sns.lineplot(df_freq, 
             x='Frequency (Hz)', 
             y='Power (a.u.)',
             hue='current',
             palette='deep',
             ax=ax)

sns.despine()
ax.set_yscale('log')
ax.set_xscale('log')
f.savefig('psd_all_currents.svg')
# %%
combined_sig = ampa + gaba_a + gaba_b + nmda

comb_freq, comb_psd = dsp.welch(combined_sig, 
                           fs=fs, 
                           nperseg=8*fs, 
                           noverlap=4*fs)

f_mask = np.logical_and(comb_freq >= 1, comb_freq <= 100)

f, ax = plt.subplots(figsize=(14,7), ncols=2)
ax[0].plot(times[:500], combined_sig[:500])
ax[1].plot(comb_freq[f_mask], comb_psd[f_mask])
ax[1].set_yscale('log')
ax[1].set_xscale('log')

ax[1].set_ylabel('Power (a.u.)')
ax[1].set_xlabel('Frequency (Hz)')

ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('Amplitude (a.u.)')

f.tight_layout()
f.savefig('meg_like_sim.svg')
# %%

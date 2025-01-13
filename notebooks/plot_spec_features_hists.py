#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

# %%

df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
df_pe = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params.csv')

df = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
osc_cols = ['tinnitus', 'ch_name', 'delta_osc', 'theta_osc', 'alpha_osc','beta_osc']
df_osc = (df[osc_cols].melt(id_vars=['tinnitus', 'ch_name'],
                   var_name='Frequency Band',
                   value_name='Parcel-Wise Oscillations (%)')
             .groupby(['Frequency Band','tinnitus', 'ch_name'])
             .mean()
             .reset_index())


df_osc['Frequency Band'] = df_osc['Frequency Band'].replace({
                                'delta_osc': 'Delta',
                                'theta_osc': 'Theta',
                                'alpha_osc': 'Alpha',
                                'beta_osc': 'Beta',
                                #'gamma_osc': 'Gamma',
                                })
#%%
f, ax = plt.subplots(figsize=(10, 5))

order= ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

sns.stripplot(data=df_osc,
             x='Frequency Band',
             y='Parcel-Wise Oscillations (%)',
             order=order,
             dodge=True,
             ax=ax,
             hue='tinnitus',
             palette='deep')
sns.despine()


f.savefig('/home/schmidtfa/experiments/resting_tinnitus/results/parcelwise_oscillations.svg')

# %%

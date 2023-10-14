#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import arviz as az
import bambi as bmb

#%%
data_all = joblib.load('/mnt/obob/staff/fschmidt/resting_tinnitus/data/connectivity/exponent.dat')

#%%
tin = np.abs(data_all['tinnitus'])
no_tin = np.abs(data_all['no_tinnitus'])

tin_ave = tin.groupby('ch_name').mean()
no_tin_ave = no_tin.groupby('ch_name').mean()

contrast = no_tin_ave - tin_ave

#%%
import seaborn as sns
import matplotlib.pyplot as plt


# Set up the matplotlib figure
f, ax = plt.subplots(ncols=3, figsize=(30, 10))

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(tin_ave, dtype=bool))

# Draw the heatmap with the mask and correct aspect ratio
vmax = 0.4

sns.heatmap(no_tin_ave, mask=mask, cmap='magma', vmax=vmax, vmin=0,
            square=True, cbar_kws={"shrink": .5}, ax=ax[0])

sns.heatmap(tin_ave, mask=mask, cmap='magma', vmax=vmax, vmin=0,
            square=True, cbar_kws={"shrink": .5}, ax=ax[1])

sns.heatmap(contrast, mask=mask, cmap='RdBu_r', vmax=0.1, vmin=-0.1,
            square=True, cbar_kws={"shrink": .5}, ax=ax[2])

#%% stack and clean
tin_stack = (tin.stack()
                .rename_axis(('subject_id', 'ch_a', 'ch_b'))
                .reset_index(name='connectivity'))
tin_stack['tinnitus'] = True

no_tin_stack = (no_tin.stack()
                      .rename_axis(('subject_id', 'ch_a', 'ch_b'))
                      .reset_index(name='connectivity'))
no_tin_stack['tinnitus'] = False

df_stack = pd.concat([tin_stack, no_tin_stack])
df_stack['ch_cmb'] = df_stack['ch_a'].to_numpy() + '___' + df_stack['ch_b'].to_numpy()
#checked this. number should be correct but wo nans 
#%% remove duplicates a==b & b==a or a==a
stacks = []

for subject_id in df_stack['subject_id']:

    cur_stack = df_stack.query(f'subject_id == "{subject_id}"')
    mask_t = (cur_stack[['ch_a', 'ch_b']].apply(frozenset, axis=1).duplicated()) | (cur_stack['ch_a']==cur_stack['ch_b'])
    cur_stack = cur_stack[~mask_t]
    stacks.append(cur_stack)

#%%
clean_df =  pd.concat(stacks)

#%%
df_stack.query('ch_a == "???"')



#%%
import bambi as bmb


#%%
md = bmb.Model(data=df_stack, formula='connectivity ~ 1 + (1 + tinnitus|ch_cmb)', dropna=True)

#%%
mdf = md.fit()


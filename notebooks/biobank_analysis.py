#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import scipy.stats as stats
from scipy.special import expit

sns.set_style('ticks')
sns.set_context('poster')

df = pd.read_csv('big_bio_cc.csv')

# %%

# %%
df.columns = ['subject_id',
              'sex',
              
              'age_T2',
              'age_T3',
              
              'wmv_n_T2',
              'wmv_n_T3',

              'Tinnitus_T2',
              'Tinnitus_T3',
              'Tinnitus_sev',
              
       'SRT_l_T2',
       'SRT_l_T3',

       'SRT_r_T2',
       'SRT_r_T3',

       'gmv_n_T2',
       'gmv_n_T3',

              'Tinnitus_T0',
              'Tinnitus_T1',

              'cc_central_T2',
              'cc_central_T3',
              'cc_mid_ant_T2',
              'cc_mid_ant_T3',
              'cc_mid_pos_T2',
              'cc_mid_pos_T3',
 
       ]

df = df.query('subject_id != XXXX') #extreme grey matter increase from T2 to T3: SubjectID Masked to confirm to UKB rules 
# %%
df['SRT'] = np.nanmean(df[['SRT_l_T3', 'SRT_r_T3']], axis=1) #'SRT_l_T2', 'SRT_r_T2',
df['SRT_delta'] = np.nanmean(df[['SRT_l_T3', 'SRT_r_T3']], axis=1) - np.nanmean(df[['SRT_l_T2', 'SRT_r_T2']], axis=1)
df['age_delta'] = df['age_T3'] - df['age_T2']
df['age_ave'] = df[['age_T2', 'age_T3']].mean(axis=1)


no_t_list = ['No, never',
             'Do not know',
             np.nan 
          ] 

df_no_t_tmp = (df.query('Tinnitus_T0 == @no_t_list')
             .query('Tinnitus_T1 == @no_t_list')
             .query('Tinnitus_T2 == @no_t_list')
             .query('Tinnitus_T3 == @no_t_list'))

df_no_t_tmp['Tinnitus'] = 'Control'
# %%
t_list = ['Yes, but not now, but have in the past', 
          'Yes, now most or all of the time', 
          'Yes, now a lot of the time', 
          'Yes, now some of the time'] 


df_t_tmp = (df.query('Tinnitus_T0 == @no_t_list')
              .query('Tinnitus_T1 == @no_t_list')
              .query('Tinnitus_T2 == @no_t_list')
              .query('Tinnitus_T3 == @t_list'))
df_t_tmp['Tinnitus'] = 'Tinnitus'

#%%


df_cut = pd.concat([df_no_t_tmp, df_t_tmp])

#%%
df_cut = df_cut[['subject_id', 'age_delta', 
                 'age_ave', 'age_T2', 'age_T3', 
                 'sex', 'SRT', 'Tinnitus',  'SRT_delta',
                 
                 'gmv_n_T2',
                 'gmv_n_T3',
       
                 'wmv_n_T2',
                 'wmv_n_T3',
                               'cc_central_T2',
              'cc_central_T3',
              'cc_mid_ant_T2',
              'cc_mid_ant_T3',
              'cc_mid_pos_T2',
              'cc_mid_pos_T3',

       ]].dropna()

df_t = df_cut[df_cut['Tinnitus'] == 'Tinnitus']
df_no_t = df_cut[df_cut['Tinnitus'] == 'Control'].copy()#.dropna()

#%%
matched_subs = []

for cur_id in df_t['subject_id'].unique():

    try:
       cur_sub = df_t.query(f'subject_id == {cur_id}')

       #first filter sex
       cur_sex = cur_sub['sex'].to_numpy()[0]

       df_no_t_sex = df_no_t.query(f'sex == "{cur_sex}"')
    

       cur_matched_subs = df_no_t_sex[
                                      np.logical_and(
                                                 np.logical_and(np.isclose(df_no_t_sex['age_T3'].to_numpy(), cur_sub['age_T3'].to_numpy(), atol=1), 
                                                                np.isclose(df_no_t_sex['age_T2'].to_numpy(), cur_sub['age_T2'].to_numpy(), atol=1)          
                                                        ),
                                                        np.isclose(df_no_t_sex['SRT'].to_numpy(), cur_sub['SRT'].to_numpy(), atol=.5),)]

       cur_match = cur_matched_subs['subject_id'].to_numpy()[0]

       matched_subs.append(cur_matched_subs.query(f'subject_id == {cur_match}'))

       df_no_t = df_no_t.query(f'subject_id != {cur_match}')
    except IndexError:
       df_t = df_t.query(f'subject_id != {cur_id}')

#%%
df_matched = pd.concat([pd.concat(matched_subs), df_t]).reset_index()

#%%
f, axes = plt.subplots(figsize=(8, 5), ncols=2)

cols = sns.color_palette('deep')
cmap = [cols[2], cols[4]]

sns.swarmplot(df_matched, x='Tinnitus', y='age_ave', palette=cmap, hue='Tinnitus', ax=axes[0], alpha=0.5)
sns.pointplot(df_matched, x='Tinnitus', y='age_ave', palette=cmap, hue='Tinnitus', ax=axes[0], markers='_')
axes[0].set_ylabel('Age (years)')


sns.swarmplot(df_matched, x='Tinnitus', y='SRT', palette=cmap, hue='Tinnitus', ax=axes[1], alpha=0.5)
sns.pointplot(df_matched, x='Tinnitus', y='SRT', palette=cmap, hue='Tinnitus', ax=axes[1], markers='_')
axes[1].set_ylabel('Hearing Ability (SRT)')

sns.despine()

f.tight_layout()

for ax in axes:
    ax.set_xlabel('')

#f.savefig('./new_fig_match_bb.svg')


#%%

df_matched['age_ave'].mean()


#%%


#%% check again for nans

df_matched[['age_delta','age_ave','age_T2',
            'age_T3','sex', 'SRT',
            'index', 'subject_id',
            'wmv_n_T2',
            'wmv_n_T3'
            ]].isna().sum()


# %%
df_matched['wm_delta'] = df_matched['wmv_n_T3'] - df_matched['wmv_n_T2']
df_matched['cc_cent_delta'] = df_matched['cc_central_T3'] - df_matched['cc_central_T2']
df_matched['cc_mid_ant_delta'] = df_matched['cc_mid_ant_T3'] - df_matched['cc_mid_ant_T2']
df_matched['cc_mid_pos_delta'] = df_matched['cc_mid_pos_T3'] - df_matched['cc_mid_pos_T2']
df_matched['gm_delta'] = df_matched['gmv_n_T3'] - df_matched['gmv_n_T2']


#%%
stats.spearmanr(df_matched.query('Tinnitus == "Tinnitus"').dropna()['age_T3'], 
                df_matched.query('Tinnitus == "Tinnitus"').dropna()['gmv_n_T3'])

#%%
stats.spearmanr(df_matched.query('Tinnitus == "Control"').dropna()['age_T3'], 
                df_matched.query('Tinnitus == "Control"').dropna()['gmv_n_T3'])

#%%

f, ax = plt.subplots(figsize=(5,5))

ax.scatter(df_matched.query('Tinnitus == "Tinnitus"')['age_T3'], 
            df_matched.query('Tinnitus == "Tinnitus"')['gmv_n_T3'], label='Tinnitus')

ax.scatter(df_matched.query('Tinnitus == "Control"')['age_T3'], 
            df_matched.query('Tinnitus == "Control"')['gmv_n_T3'], label='Control')

plt.legend()

sns.despine()

#%% sanity check

f, ax = plt.subplots(figsize=(5,5))

ax.scatter(df_matched.query('Tinnitus == "Tinnitus"')['age_T3'], 
            df_matched.query('Tinnitus == "Tinnitus"')['wmv_n_T3'], label='Tinnitus')

ax.scatter(df_matched.query('Tinnitus == "Control"')['age_T3'], 
            df_matched.query('Tinnitus == "Control"')['wmv_n_T3'], label='Control')

plt.legend()

sns.despine()


#%%
stats.spearmanr(df_matched.query('Tinnitus == "Tinnitus"').dropna()['age_T3'], 
                df_matched.query('Tinnitus == "Tinnitus"').dropna()['wmv_n_T3'])

#%%
stats.spearmanr(df_matched.query('Tinnitus == "Control"').dropna()['age_T3'], 
                df_matched.query('Tinnitus == "Control"').dropna()['wmv_n_T3'])

# %%
f, ax = plt.subplots(figsize=(4,4))
df_matched['wm_delta_norm'] = df_matched['wm_delta'] / 1000 #scale to cm3
sns.pointplot(df_matched, x='Tinnitus', y='wm_delta_norm', hue='Tinnitus', markers='_')

ax.set_ylabel('White Matter Volume')

sns.despine()


stats.ttest_ind(df_matched.query('Tinnitus == "Tinnitus"')['wm_delta'].dropna(),
                df_matched.query('Tinnitus == "Control"')['wm_delta'].dropna())



def run_logreg(X, y):

       m_X, sd_X = X.mean(), X.std()
       X_scaled = (X - m_X) / sd_X

       with pm.Model() as m:
       
              alpha_z = pm.Normal('alpha_z', 0, 1)
              beta_z = pm.Normal('beta_z', 0, 1)

              #transform the data to the original scale for plotting
              alpha = pm.Deterministic(
                     "alpha",
                     alpha_z - beta_z * m_X / sd_X
              )
              beta = pm.Deterministic("beta", beta_z / sd_X)

              pm.Bernoulli('y',
                            logit_p=alpha_z + beta_z * X_scaled,
                            observed=y
                            )


              idata = pm.sample()

       summ = az.summary(idata,  hdi_prob=.89)

       return summ, idata
#%%

sum_wm, idata_wm = run_logreg(X=df_matched['wm_delta_norm'], y=df_matched['Tinnitus'] == 'Tinnitus')

np.exp(sum_wm.loc['beta_z'])


#%%


df_matched_sorted = df_matched.sort_values('wm_delta_norm')


posterior_slope = np.array([expit(idata_wm.posterior['alpha'].values + idata_wm.posterior['beta'].values * val) for val in df_matched_sorted['wm_delta_norm']])

sns.set_theme(context='poster',
              style='ticks')

f, ax = plt.subplots(figsize=(5,5))

ax.plot(df_matched_sorted['wm_delta_norm'], posterior_slope.mean(axis=(1,2)))
ax.fill_between(df_matched_sorted['wm_delta_norm'], np.percentile(posterior_slope, q=5.5, axis=(1,2)), 
                          np.percentile(posterior_slope, q=94.5, axis=(1,2)), 
                           alpha=0.25, color=cols[0])

y = (df_matched_sorted["Tinnitus"] == 'Tinnitus').astype(int)

ax.scatter(df_matched_sorted['wm_delta_norm'][y==0], y[y==0] + 0.025, color=cmap[0], alpha=0.5, s=100)
ax.scatter(df_matched_sorted['wm_delta_norm'][y==1], y[y==1] - 0.025, color=cmap[1], alpha=0.5, s=100)

ax.set_xlabel("white matter volume \n change (T3-T2; cm3)")
ax.set_ylabel("p (Tinnitus)")

ax.set_ylim(0.0, 1.0)

sns.despine()

f.savefig('white_matter_reduction_tinn.svg')


#%%
df_matched['gm_delta_norm'] = df_matched['gm_delta'] / 1000 #just scale to cm3

sum_gm, idata_gm = run_logreg(X=df_matched['gm_delta_norm'], y=df_matched['Tinnitus'] == 'Tinnitus')

np.exp(sum_gm.loc['beta_z'])



#%%
df_matched_sorted = df_matched.sort_values('gm_delta_norm')


posterior_slope = np.array([expit(idata_gm.posterior['alpha'].values + idata_gm.posterior['beta'].values * val) for val in df_matched_sorted['gm_delta_norm']])


f, ax = plt.subplots(figsize=(5,5))

ax.plot(df_matched_sorted['gm_delta_norm'], posterior_slope.mean(axis=(1,2)))
ax.fill_between(df_matched_sorted['gm_delta_norm'], np.percentile(posterior_slope, q=5.5, axis=(1,2)), 
                          np.percentile(posterior_slope, q=94.5, axis=(1,2)), 
                           alpha=0.25, color=cols[0])

y = (df_matched_sorted["Tinnitus"] == 'Tinnitus').astype(int)

ax.scatter(df_matched_sorted['gm_delta_norm'][y==0], y[y==0] + 0.025, color=cmap[0], alpha=0.5, s=100)
ax.scatter(df_matched_sorted['gm_delta_norm'][y==1], y[y==1] - 0.025, color=cmap[1], alpha=0.5, s=100)

ax.set_xlabel("grey matter volume \n change (T3-T2; cm3)")
ax.set_ylabel("p (Tinnitus)")

ax.set_ylim(0.0, 1.0)

sns.despine()

f.savefig('gm_matter_reduction_tinn.svg')

#%%


sum_srt, idata_srt = run_logreg(X=df_matched['SRT_delta'], y=df_matched['Tinnitus'] == 'Tinnitus')

np.exp(sum_srt.loc['beta_z'])

#%%

sum_gmt2, idata_gmt2 = run_logreg(X=df_matched['gmv_n_T2'], y=df_matched['Tinnitus'] == 'Tinnitus')

np.exp(sum_gmt2.loc['beta_z'])

#%%
sum_wmt2, idata_wmt2 = run_logreg(X=df_matched['wmv_n_T2'], y=df_matched['Tinnitus'] == 'Tinnitus')

np.exp(sum_wmt2.loc['beta_z'])
# %%

#%% combine hearing information with resting state information
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')

#%% get shoebox info
p10_data = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_hearing_no_sb.csv')
p10_data.rename(columns={'REFERRER_ID': 'subject_id'}, inplace=True)

event_dictionary ={'none' : False, 'tinnitus' : True, 
                   'hearingloss' : False, 'both': True}
p10_data['tinnitus_a'] = p10_data['hearing'].map(event_dictionary)
p10_tinnitus = p10_data[['subject_id', 'age', 'tinnitus_a', 'gender',
                         'ssq_speechhearing','ssq_spatialhearing', 
                         'ssq_qualitiesofhearing', 'ssq_listeningeffort',
                         'tinnitus_distress',]]
p10_tinnitus['subject_id'] = p10_tinnitus['subject_id'].str.lower()

df_dB = pd.read_csv('../data/pta_all_subs.csv')

p10_tinnitus_db = p10_tinnitus.merge(df_dB, on='subject_id')
#%% tinnitus martha
['ID', 'age at date of acq', 'Unnamed: 2']
resting_martha = pd.read_excel('../data/subject_lists/resting_stat_tinn_IDs.xlsx')
resting_martha_pta = pd.read_csv('../data/subjecttable1119.csv')[['age', 'external_patient_id','hl_r_al_avg', 'hl_l_al_avg', 'tq_score']]

#%%

tinnitus_subs = ['19980617ATLN', '19930709CRGL', '19800508CRBU', '19741020CRAZ',
                 '19900203SSSH', '19670105CASI', '19620826WLTI', '19590315MRLI',
                 '19590423MRBR', '19610805EIGA',]


martha_sel = resting_martha_pta.query('external_patient_id == @tinnitus_subs').copy()
martha_sel['dB'] = np.nanmean(martha_sel[['hl_r_al_avg', 'hl_l_al_avg']], axis=1)
martha_sel.columns = ['measurement_age', 'subject_id', 'hl_r', 'hl_l', 'tinnitus_distress', 'dB']
df_martha_clean = martha_sel[['subject_id', 'measurement_age', 'tinnitus_distress', 'dB']].copy()
df_martha_clean['tinnitus_b'] = True

#%%

lisa_tinnitus = pd.read_csv('../data/subject_lists/tinnitus.csv')
lisa_tinnitus.columns = ['subject_id', 'tinnitus_c']
lisa_tinnitus['tinnitus_c'] = lisa_tinnitus['tinnitus_c'] == 'yes'
lisa_tinnitus['subject_id'] = lisa_tinnitus['subject_id'].str.lower()

#%%
df_tinn_cmb = pd.concat([p10_tinnitus_db, df_martha_clean]) #lisa_tinnitus
df_tinn_cmb.drop_duplicates('subject_id', inplace=True, keep='last')
df_tinn_cmb['tinnitus'] = df_tinn_cmb[['tinnitus_a', 'tinnitus_b']].sum(axis=1) > 0 #'tinnitus_c'

#%%
df_tinn_cmb = df_tinn_cmb[['subject_id', 'age', 'tinnitus', 'gender', 'tinnitus_distress', 'dB',]]
#%% get pandas data for resting
df_tinn_cmb['subject_id'] = [s_id.lower() for s_id in df_tinn_cmb['subject_id']]
resting_list = pd.read_csv('../notebooks/resting_list_single.csv')

#%% merge both
df_cmb = pd.merge(resting_list, df_tinn_cmb,  on='subject_id')
df_cmb.drop_duplicates('subject_id', inplace=True, keep='last')
df_cmb.to_csv('../data/subject_lists/df_tinnitus.csv')


#%%
experiments = np.array([subject[1]['path'].split('/')[4] for subject in df_cmb.iterrows()])

#%%
sns.catplot(data=df_cmb, x='tinnitus', y='measurement_age', kind='swarm')

#%%
sns.catplot(data=df_cmb, x='tinnitus', y='dB', kind='swarm')

#%% 
from scipy.stats import zscore

df_cmb['age_z'] = zscore(df_cmb['measurement_age'])
df_cmb['dB_z'] = zscore(df_cmb['dB'])

#%%
df_cmb




#%%
df_no_tinn = df_cmb.query('tinnitus == False')[['subject_id', 'tinnitus', 'fs_1k', 
                                                'measurement_age', 'dB', 'age_z', 'dB_z',
                                                'gender', 'path', 'tinnitus_distress',]].copy().reset_index()
df_tinn = df_cmb.query('tinnitus == True')[['subject_id', 'tinnitus', 'fs_1k', 
                                            'measurement_age', 'dB', 'age_z', 'dB_z',
                                            'gender', 'path', 'tinnitus_distress',]].copy().reset_index()

#%
#TODO: get hearing level from all missing subjects
#df_tinn_db = df_tinn[np.isnan(df_tinn['dB']) != True]
#df_no_tinn_db = df_no_tinn[np.isnan(df_no_tinn['dB']) != True].copy()

age_col = ['age_z']


best_match_list = []
match_order = 'age'

df_tinn = df_tinn.query('dB < 50') #<37 for nice overlap
#df_no_tinn = df_no_tinn.query('dB > 5')

for subject in df_tinn['subject_id']:

    cur_df = df_tinn.query(f'subject_id == "{subject}"')

    #get first a decent age match
    if match_order == 'age':
        ave_diff = np.abs(np.subtract(cur_df[age_col].to_numpy(), # diff in years
                                      df_no_tinn[age_col].to_numpy(),
                                            ))
        
        candidates = df_no_tinn[ave_diff >= np.quantile(ave_diff, 0.2)]

        #get next the best fitting db subject
        cur_best = candidates.iloc[np.abs(np.subtract(candidates['dB'].to_numpy(), 
                                                    cur_df['dB'].to_numpy())).argmin()]
    
    elif match_order == 'db':
        ave_diff = np.abs(np.subtract(cur_df['dB'].to_numpy(), # diff in years
                                      df_no_tinn['dB'].to_numpy(),
                                            ))
        
        candidates = df_no_tinn[ave_diff <= np.quantile(ave_diff, 0.2)]

        #get next the best fitting db subject
        cur_best = candidates.iloc[np.abs(np.subtract(candidates[age_col].to_numpy(), 
                                                      cur_df[age_col].to_numpy())).argmin()]

    best_match_list.append(cur_best)


    df_no_tinn = df_no_tinn.query(f'subject_id != "{cur_best.subject_id}"')


#%%
df_matched = pd.concat([pd.concat(best_match_list, axis=1).T, df_tinn]).reset_index()
df_matched.drop_duplicates('subject_id', inplace=True, keep='last')
df_matched.to_csv('../data/tinnitus_match.csv')
#%%
fig, ax = plt.subplots()

sns.swarmplot(data=df_matched, x='tinnitus', y='measurement_age',
              hue='tinnitus', ax=ax, size=10, alpha=0.4, )#legend=False)
sns.pointplot(data=df_matched, x='tinnitus', y='measurement_age', hue='tinnitus',
              ax=ax, estimator='mean', markers='+', scale=1.5)
fig.set_size_inches(4,6)
ax.set_ylabel('age (years)')
sns.despine()

#fig.savefig('../results/age_match.svg')

#%%
from scipy.stats import mannwhitneyu

mannwhitneyu(df_matched.query('tinnitus == True')['measurement_age'].to_numpy().astype(int),
             df_matched.query('tinnitus == False')['measurement_age'].to_numpy().astype(int),
         )
#%%
fig, ax = plt.subplots()

sns.swarmplot(data=df_matched, x='tinnitus', y='dB',
              hue='tinnitus', ax=ax, size=10, alpha=0.4,)# legend=False)
sns.pointplot(data=df_matched, x='tinnitus', y='dB', hue='tinnitus', 
              ax=ax, estimator='mean', markers='+', scale=1.5)
fig.set_size_inches(4,6)
ax.set_ylabel('Hearing Threshold (dB)')
sns.despine()

fig.savefig('../results/pta_match.svg')

mannwhitneyu(df_matched.query('tinnitus == True')['dB'].dropna().to_numpy().astype(int),
             df_matched.query('tinnitus == False')['dB'].dropna().to_numpy().astype(int),
         )

#np.sum([np.isnan(n) for n in df_matched['dB'].to_numpy()])
#%%
from scipy.stats.contingency import chi2_contingency

cross_tab = pd.crosstab(index=df_matched['tinnitus'], columns=df_matched['gender'])
c, p, dof, expected = chi2_contingency(cross_tab.to_numpy())

#%%
cross_tab_pct = pd.crosstab(index=df_matched['tinnitus'], columns=df_matched['gender'], normalize='index')


fig, ax = plt.subplots() 
cross_tab_pct.plot(kind='bar', stacked=True, ax=ax, color=[sns.color_palette('deep')[3],
                                                            sns.color_palette('deep')[4]])
fig.set_size_inches(4,6)
ax.set_xticklabels(labels=[False, True],rotation=0)
ax.set_ylabel('Proportion')
sns.despine()
fig.savefig('../results/gender_match.svg')

# %%
plt.hist(df_matched['tinnitus_distress'])
# %%
np.unique(df_matched['tinnitus'], return_counts=True)
# %%

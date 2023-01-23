#%% combine hearing information with resting state information
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('ticks')
sns.set_context('poster')


#%% get shoebox info
p10_data = pd.read_csv('/mnt/obob/staff/fschmidt/neurogram/data/p10/df_hearing_no_sb.csv')
p10_data.rename(columns={'REFERRER_ID': 'subject_id'}, inplace=True)

event_dictionary ={'none' : False, 'tinnitus' : True, 
                   'hearingloss' : False, 'both': True}
p10_data['tinnitus_a'] = p10_data['hearing'].map(event_dictionary)
p10_tinnitus = p10_data[['subject_id', 'age', 'tinnitus_a']]
p10_tinnitus['subject_id'] = p10_tinnitus['subject_id'].str.lower()

#%% tinnitus martha
['ID', 'age at date of acq', 'Unnamed: 2']
resting_martha = pd.read_excel('../data/subject_lists/resting_stat_tinn_IDs.xlsx')


tinnitus_subs = pd.DataFrame(['19980617ATLN', '19980213BASH', '19930709CRGL', '19930927PTBU', '19930428SBOE',
'19871011EILC', '19840920GASA', '19800508CRBU','19730306PISH', '19700131CRIS', 
'19680613ANME', '19631229RSAT', '19900203SSSH', '19820906ANFI', '19741020CRAZ',
'19740628BRKE', '19631206CRRI', '19620826WLTI', '19590315MRLI', '19590423MRBR',
'19520716MRTE', '19881117GBSR', '19760807ZHCU', '19670607MRPA', '19670105CASI',
'1960911EIKN', '19610805EIGA', '19600518HDEM', '19600406JHSH'])
tinnitus_subs.columns = ['ID']

df_martha_clean = pd.merge(tinnitus_subs, resting_martha, on='ID')[['ID', 'age at date of acq']]
df_martha_clean['ID'] = [subject.lower() for subject in df_martha_clean ['ID']]
df_martha_clean.columns = ['subject_id', 'age']
df_martha_clean['tinnitus_b'] = True


#%%

lisa_tinnitus = pd.read_csv('../data/subject_lists/tinnitus.csv')
lisa_tinnitus.columns = ['subject_id', 'tinnitus_c']
lisa_tinnitus['tinnitus_c'] = lisa_tinnitus['tinnitus_c'] == 'yes'
lisa_tinnitus['subject_id'] = lisa_tinnitus['subject_id'].str.lower()

#%%
df_tinn_cmb = pd.concat([p10_tinnitus, df_martha_clean, lisa_tinnitus])
df_tinn_cmb.drop_duplicates('subject_id', inplace=True)
df_tinn_cmb['tinnitus'] = df_tinn_cmb[['tinnitus_a', 'tinnitus_b', 'tinnitus_c']].sum(axis=1) > 0
df_tinn_cmb = df_tinn_cmb[['subject_id', 'age', 'tinnitus']]
#%% get pandas data for resting
resting_list = pd.read_csv('../data/subject_lists/resting_list.csv')



#%% merge both
df_cmb = pd.merge(df_tinn_cmb, resting_list, on='subject_id')
df_cmb.drop_duplicates('subject_id', inplace=True, keep='last')


df_cmb.to_csv('../data/subject_lists/df_tinnitus.csv')


#%%
sns.catplot(data=df_cmb.query('measurement_age > 22'), x='tinnitus', y='measurement_age', kind='point')


#%%

ids, counts = np.unique(df_cmb['subject_id'], return_counts=True)
id_list = ids[counts > 1]

df_multi = df_cmb.query('subject_id in @id_list')
df_multi.reset_index(inplace=True)

#%%
df_multi['subject_id'].unique().shape

df_multi.drop_duplicates('subject_id', inplace=True)
# %%
(df_multi['hearing'] == 'tinnitus').sum()
# %%
(df_multi['hearing'] == 'hearing loss + tinnitus').sum()


# %%
(df_cmb['hearing'] == 'tinnitus').sum()
# %%
(df_cmb['hearing'] == 'hearing loss + tinnitus').sum()
# %%

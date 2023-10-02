#%%
import pandas as pd
# %%
left_aud = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/aud_all_left.csv')
left_aud['ear'] = 'left'
right_aud = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/aud_all_right.csv')
right_aud['ear'] = 'right'
# %% combine both ears
df_cmb = pd.concat([left_aud, right_aud]).groupby(['subject_id', 
                                                   'test_date', 
                                                   #'Frequency (Hz)'
                                                   ]).mean().reset_index()



df_cmb.drop_duplicates(subset=['subject_id', 
                               #'Frequency (Hz)'
                               ], keep="last")

df_cmb[['subject_id', 'dB']].to_csv('../data/pta_all_subs.csv')

#%%
df_nathan = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/irasa_sub.csv')
df_merged = df_cmb[['subject_id', 'Frequency (Hz)', 'dB', 'ear']].merge(df_nathan, on='subject_id')
df_merged.to_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/nathan_fabi_merge.csv')




df_cmb
# %%
path_nathan = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/df_Lisa_EO_all_ah.csv'
df_lisa = pd.read_csv(path_nathan, delimiter='\t')
# %%

df_merged = df_cmb.merge(df_lisa, on='subject_id')
# %%

# %%
df_merged
# %%

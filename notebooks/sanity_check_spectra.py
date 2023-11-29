#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import joblib
import numpy as np


import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')

#%%
df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = sorted(df_all['subject_id'].unique())
#__fft_method_multitaper.dat
#bad subjects

#%%
subject_id =  '19910612crke'#subject_ids[4]
#%
INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__fft_method_irasa.dat'))[0])

freq_cut = cur_data['freq'] < 100

#plt.loglog(cur_data['freq'][freq_cut], cur_data['src']['label_tc_ap'].T[freq_cut]);
#plt.plot(cur_data['freq'][freq_cut], cur_data['src']['label_tc'].T[freq_cut])
freq_cut = cur_data['freq'] < 20
plt.plot(cur_data['freq'][freq_cut], cur_data['src']['label_tc_p'].T[freq_cut].mean(axis=1));


#%%

plt.loglog(cur_data['freq'][freq_cut], cur_data['src']['label_tc'].T[freq_cut].mean(axis=1))




#%%
spectra_dfs = []

for subject_id in subject_ids:

    #%
    print(subject_id)
    INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
    cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__fft_method_multitaper.dat'))[0])

    freq_cut = cur_data['freq'] < 100

    spectra_dfs.append(pd.DataFrame({'Frequency (Hz)': cur_data['freq'][freq_cut],
                                     'Power (a.u.)': cur_data['src']['label_tc'].T[freq_cut].mean(axis=1),
                                     'subject_id': subject_id}))

#%%
df_spectra = pd.concat(spectra_dfs).merge(df_all[['subject_id', 'tinnitus']], on='subject_id')

f, ax = plt.subplots(figsize=(5,5))


good_ix = np.logical_and(df_spectra['Frequency (Hz)'] > 3, df_spectra['Frequency (Hz)'] < 20)

sns.lineplot(data=df_spectra[good_ix], 
             x='Frequency (Hz)', y='Power (a.u.)', 
             hue='subject_id',
            ax=ax, palette='deep', legend=False)
#ax.set_xscale('log')
#ax.set_yscale('log')
sns.despine()

#f.savefig('../results/alpha_inset.svg')

#%%
f, ax = plt.subplots(figsize=(5,5))

sns.lineplot(data=df_spectra, x='Frequency (Hz)', y='Power (a.u.)', 
hue='subject_id', ax=ax, palette='deep', legend=False)
ax.set_xscale('log')
ax.set_yscale('log')
sns.despine()

#f.savefig('../results/spectra_ave.svg')


#%% plot the alpha spectrum per source
spectra_dfs_chs = []

for subject_id in subject_ids:

    INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
    cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__src_type_beamformer.dat'))[0])

    freq_cut = np.logical_and(cur_data['freq'] > 3, cur_data['freq'] < 20)

    cur_df = pd.DataFrame(cur_data['src']['label_tc'].T[freq_cut], 
                       columns=cur_data['src']['label_info']['names_order_mne'],
                       index=cur_data['freq'][freq_cut])
    cur_df_tidy = cur_df.reset_index().melt(id_vars='index',
                                            var_name='ch_name',
                                            value_name='Power (a.u.)')
    cur_df_tidy.rename(columns={'index' :'Frequency (Hz)'}, inplace=True)
    cur_df_tidy['subject_id'] = subject_id
    
    spectra_dfs_chs.append(cur_df_tidy)

#%%

df_s_ch = pd.concat(spectra_dfs_chs).merge(df_all[['subject_id', 'tinnitus']], on='subject_id')


#%%

grid = sns.FacetGrid(data=df_s_ch, col='ch_name', hue='tinnitus', 
                  col_wrap=20, palette='deep')

grid.map(sns.lineplot, 'Frequency (Hz)', 'Power (a.u.)')

grid.fig.tight_layout(w_pad=1)

grid.fig.savefig('../results/large_grid_alpha_plot.svg')



#%%
okayish_ids = [8, 15, 20, 22, 25, 32, 37, 49, 50, 64, 91, 93]
bad_sub_ids = [6,11,13, 19, 31, 57, 61, 72, 79] + okayish_ids
bad_subs = [sid for ix, sid in enumerate(subject_ids) if ix in bad_sub_ids]


# %%
#freq_cut_low = cur_data['freq'] < 40
#plt.plot(cur_data['freq'][freq_cut_low], cur_data['src']['label_tc'].T[freq_cut_low])
#plt.loglog(cur_data['freq'][freq_cut_low], cur_data['src']['label_tc'].T[freq_cut_low])
# %%
df_all.query('subject_id == @bad_subs').mean()
# %%
df_all.query('subject_id != @bad_subs').mean()
# %%

bad_data_list = []

for ix in bad_sub_ids:

    subject_id = subject_ids[ix]

    INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
    cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__src_type_beamformer.dat'))[0])

    bad_data_list.append(cur_data['src']['label_tc'].T[freq_cut])

# %%
bad_array = np.array(bad_data_list)
# %%

f, axes = plt.subplots(ncols=4, nrows=6, figsize=(12, 18))

for ix, ax in enumerate(axes.flatten()):

    if ix < np.shape(bad_array)[0]:
        ax.set_title(bad_subs[ix])
        ax.loglog(cur_data['freq'][freq_cut], bad_array[ix])

sns.despine()
plt.tight_layout()

f.savefig('../results/bad_subs.png')
# %%

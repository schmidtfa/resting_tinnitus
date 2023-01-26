#%%
import joblib
from pathlib import Path, PurePath
import pandas as pd
import pingouin as pg
import mne

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import eelbrain as eb


#Plotting setup

sns.set_style('ticks')
sns.set_context('poster')

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


#%% plot demographics and select an age group
df_tinnitus_demographics = pd.read_csv('../data/subject_lists/df_tinnitus.csv')

tinn_match = df_tinnitus_demographics.query('measurement_age > 23')

g = sns.stripplot(data=tinn_match, x='tinnitus', y='measurement_age', hue='tinnitus', size=10, alpha=0.25)
g = sns.pointplot(data=tinn_match, x='tinnitus', y='measurement_age', hue='tinnitus', markers="_", scale=1.7,)
g.legend_.remove()

g.set_ylabel('Age (Years)')
g.set_xlabel('Tinnitus')

g.figure.set_size_inches(5,4)
sns.despine()



# %% Get all tinnitus subjects

path2data = Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg')

eye_thresh, heart_thresh = 0.5, 0.5

my_path_ending = '*/*.dat'#f'*/*freq_range_\[0.5, 45]__meg_type_mag__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}.dat'

all_files = [str(sub_path) for sub_path in path2data.glob(my_path_ending) if sub_path.is_file()]
print(len(all_files))

# %%

#TODO: add recording time information to preprocessing

all_f_len = len(all_files)

all_info_dfs, data4ds = [], []

 
for idx, file in enumerate(all_files):

    print(f'Loading file {idx}/{all_f_len}')
    cur_file = joblib.load(file)

    if cur_file['blink_epochs'].get_data().shape[0] > 50: #minimum trial requirement

        #evoked
        evoked = (cur_file['blink_epochs'].average()
                                          .resample(sfreq=100)
                                          .crop(tmin=0, tmax=0.5))

        evoked_nd = eb.load.fiff.evoked_ndvar(evoked,
                                  connectivity='grid'
                                  )

        # trial based irasa
        #overwrite info field for spectrum
        info_psd = mne.Info(cur_file['blink_epochs'].info, sfreq=2)
        

        periodic_irasa_evo_nd = eb.load.fiff.evoked_ndvar(mne.EvokedArray(cur_file['epoch_irasa']['periodic'], info=info_psd),
                                  name='periodic_evoked',
                                  connectivity='grid'
                                  )

        aperiodic_irasa_evo_nd = eb.load.fiff.evoked_ndvar(mne.EvokedArray(cur_file['epoch_irasa']['aperiodic'], info=info_psd),
                                  name='aperiodic_evoked',
                                  connectivity='grid'
                                  )
        
        cur_file['eye_eve_trf'].h.x = np.abs(cur_file['eye_eve_trf'].h.x)

        #prepare data for eelbrain dataset
        data4ds.append([cur_file['subject_id'], int(cur_file['tinnitus'].to_numpy()[0]), int(cur_file['age']),
                        cur_file['blink_epochs'].get_data().shape[0], (np.abs(cur_file['eog_scores']) > 0.5).sum(),
                        evoked_nd, periodic_irasa_evo_nd, aperiodic_irasa_evo_nd, cur_file['eye_eve_trf'].h])


#%%
ds = eb.Dataset.from_caselist(['subject_id', 'tinnitus', 'age', 'n_blinks', 'n_eog_comps', 'evoked', 'periodic_evoked', 'aperiodic_evoked', 'trf'], data4ds, random='subject_id')

#%%
ds_eog = ds[np.logical_and(ds['n_eog_comps'] > 0, ds['n_eog_comps'] < 3)]
ds_age = ds_eog[ds_eog['age'] > 26]
ds2test = ds_age[np.logical_and(ds_age['n_blinks'] < 1000, ds_age['n_blinks'] > 50)]

#%%
tinn_perc = ds2test['tinnitus'].x.mean() * 100
print(f' {tinn_perc}% of the group have tinnitus')


#%%
tinn_match = ds2test.as_dataframe()

g = sns.stripplot(data=tinn_match, x='tinnitus', y='age', hue='tinnitus', size=10, alpha=0.25)
g = sns.pointplot(data=tinn_match, x='tinnitus', y='age', hue='tinnitus', markers="_", scale=1.7,)
g.legend_.remove()

g.set_ylabel('Age (Years)')
g.set_xlabel('Tinnitus')

g.figure.set_size_inches(5,4)
sns.despine()

# %% Test if groups differ
# TODO: replace or add an equivalence test
pg.ttest(x=tinn_match.query('tinnitus == True')['age'],
         y=tinn_match.query('tinnitus == False')['age'],)

#%%
g = sns.stripplot(data=tinn_match, x='tinnitus', y='n_eog_comps', hue='tinnitus', size=10, alpha=0.25)
g = sns.pointplot(data=tinn_match, x='tinnitus', y='n_eog_comps', hue='tinnitus', markers="_", scale=1.7,)
g.legend_.remove()

g.set_ylabel('n_eog_comps')
g.set_xlabel('Tinnitus')

g.figure.set_size_inches(5,4)
sns.despine()


# %% Test if groups differ
# TODO: replace or add an equivalence test
pg.ttest(x=tinn_match.query('tinnitus == True')['n_eog_comps'],
         y=tinn_match.query('tinnitus == False')['n_eog_comps'],)




#%% compare blink patterns
g = sns.stripplot(data=tinn_match, x='tinnitus', y='n_blinks', hue='tinnitus', size=10, alpha=0.25)
g = sns.pointplot(data=tinn_match, x='tinnitus', y='n_blinks', hue='tinnitus', markers="_", scale=1.7,)
g.legend_.remove()

g.set_ylabel('n_blinks')
g.set_xlabel('Tinnitus')

g.figure.set_size_inches(5,4)
sns.despine()

#%%
# TODO: replace or add an equivalence test
pg.ttest(x=tinn_match.query('tinnitus == True')['n_blinks'],
         y=tinn_match.query('tinnitus == False')['n_blinks'],)

#%%
test_trf = eb.testnd.TTestIndependent(y='trf', x='tinnitus', ds=ds2test,)# tfce=True)
p_trf = eb.plot.TopoButterfly(test_trf, clip='circle')

#%%
test_aperiodic = eb.testnd.TTestIndependent(y='aperiodic_evoked', x='tinnitus', ds=ds2test)#, tfce=True)
p_aperiodic_evoked = eb.plot.TopoButterfly(test_aperiodic, clip='circle')
#%%
test_periodic = eb.testnd.TTestIndependent(y='periodic_evoked', x='tinnitus', ds=ds2test,)#, tfce=True)
p_periodic_evoked = eb.plot.TopoButterfly(test_periodic, clip='circle')


#%%


#info dataframe
all_info_dfs.append(pd.DataFrame({'subject_id': cur_file['subject_id'],
            'tinnitus': cur_file['tinnitus'],
            'age': cur_file['age'],
            'n_blinks': cur_file['blink_epochs'].get_data().shape[0],
            }))

# %%
df_blink_info = pd.concat(all_info_dfs)

g = sns.stripplot(data=df_blink_info, x='tinnitus', y='n_blinks', hue='tinnitus', size=10, alpha=0.05)
g = sns.pointplot(data=df_blink_info, x='tinnitus', y='n_blinks', hue='tinnitus', markers="_", scale=1.7,)
g.legend_.remove()

g.set_ylabel('n eye events')
g.set_xlabel('Tinnitus')

g.figure.set_size_inches(5,4)
sns.despine()

# %%
(df_blink_info['n_blinks'] > 50).mean()
# %%

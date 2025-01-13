#%%
import pandas as pd
import mne
from pathlib import Path
# %%
df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

# %%
cur_sub = 'mrsh'
#%%
cur_p = list(Path('/home/schmidtfa/experiments/resting_tinnitus/data/sinuhe').glob(f'*{cur_sub}*.fif'))[0]

raw = mne.io.read_raw(cur_p)
# %%
raw
# %%

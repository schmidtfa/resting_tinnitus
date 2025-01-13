#%%
import pandas as pd

df_ti = pd.read_csv('tinnitus_match.csv')
# %%

with open('all_paths.txt', 'a') as file:
    for f in df_ti['path']:
        str(f)


[str(f) for f in ]
# %%

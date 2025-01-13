#%%
import pandas as pd
import bambi as bmb
import arviz as az
# %%

sample_kwargs = {'draws': 2000,
               'tune': 2000,
               'chains': 4,
               'target_accept': 0.95,}

df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
# %%
md_age = bmb.Model(formula='tinnitus ~ 1 + scale(measurement_age)',
          data=df_all,
          family='bernoulli',
          link='logit')

md_db = bmb.Model(formula='tinnitus ~ 1 + scale(dB)',
          data=df_all,
          family='bernoulli',
          link='logit')

md_gender = bmb.Model(formula='tinnitus ~ 1 + gender',
          dropna=True,
          data=df_all,
          family='bernoulli',
          link='logit')
# %%
mdf_age = md_age.fit(**sample_kwargs)

#%%
az.summary(mdf_age)

# %%
mdf_db = md_db.fit(**sample_kwargs)
# %%
az.summary(mdf_db)
# %%
mdf_gender = md_gender.fit(**sample_kwargs)

#%%
az.summary(mdf_gender)
# %%

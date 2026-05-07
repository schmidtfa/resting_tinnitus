#%%
from pathlib import Path
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

import seaborn as sns

sns.set_theme(context='poster', style='ticks')


#%%debug

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()
duration = 6
dfs = []
recompute = False
get_feature = 'raw' #raw

if recompute:
    for subject_id in subject_ids:

        try:
            query_string = f'__duration_{duration}__hmax_2__source_surface__atlas_glasser.dat'
            INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'
            cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))

            #% src based analysis
            def add_info(df, df_info):
                df['subject_id'] = df_info['subject_id'].iloc[0]
                df['tinnitus'] = df_info['tinnitus'].iloc[0]
                df['dB'] = df_info['dB'].iloc[0]
                df['age'] = df_info['measurement_age'].iloc[0]
                df['tinnitus_distress'] = df_info['tinnitus_distress'].iloc[0]

                return df
            
            if get_feature == 'raw':
                curf = cur_data['src']['irasa_label_stc'].raw_spectrum.T
            elif get_feature == 'periodic':
                curf = cur_data['src']['irasa_label_stc'].periodic.T
            elif get_feature == 'aperiodic':
                curf = cur_data['src']['irasa_label_stc'].aperiodic.T

            df = pd.DataFrame(curf, 
                            columns=cur_data['src']['label_info']['names_order_mne'])
            df['freqs'] = cur_data['src']['irasa_label_stc'].freqs
            df = df.melt(id_vars='freqs', var_name='roi', value_name='Power')

            df = add_info(df, cur_data['subject_info'])

            df = df.query('roi != "???"')

            dfs.append(df)
        except IndexError:
            print(subject_id)


    df = pd.concat(dfs)

    df.to_csv(f'../data/full_spectra_sbg_tinnitus_{get_feature}.csv')

if get_feature == 'raw':
    df = pd.read_csv('../data/full_spectra_sbg_tinnitus.csv')
else:
    df= pd.read_csv(f'../data/full_spectra_sbg_tinnitus_{get_feature}.csv')

#%% plot all spectra
fmin, fmax=1, 100

left_rois = [r for r in df['roi'].unique() if r.startswith('L')]
right_rois = [r for r in df['roi'].unique() if r.startswith('R')]

#%%

df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
df_regions_info['roi'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
df_regions_info['roi'] = df_regions_info['roi'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                    'R_7Pl_ROI': 'R_7PL_ROI',})

df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']
df_cmb = df_regions_info.merge(df, on='roi')

#%%
df_regions_info


#%% plot all rois

cur_hemi = 'L'
df_all2plot = df_cmb.query(f'freqs >= {fmin} and freqs <= {fmax}').groupby(['LR', 'Lobe', 'freqs', 'tinnitus']).mean(numeric_only=True).reset_index()
g = sns.FacetGrid(df_all2plot.query(f'LR == "{cur_hemi}"'), col='Lobe', col_wrap=5, 
                  hue='tinnitus', sharey=False, height=4)
g.map_dataframe(sns.lineplot, x='freqs', y='Power', errorbar='se')

for ax in g.axes.flatten():
    ax.set_ylabel('Power')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_yscale('log')
    ax.set_xscale('log')

sns.despine()

#g.figure.savefig(f'../results/final_shit/raw_spectra_all_cortices_hemi_{cur_hemi}.svg',  bbox_inches='tight')


#%% showcase lobes
from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface


gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')

gdf2plot = gdf.merge(df_regions_info, on='roi')


f, axes = plot_surface(gdf2plot,#.query('values == 1'),
                    column='Lobe',
                    cmap='Set2', 
                    show_cbar=True,
                    #vmin=0,
                    #vmax=1
                    )
#f.savefig(f'../results/final_shit/raw_spectra_brain_lobes.svg')




#%%

cur_feat = 'alpha_osc' # Exponent_1, Exponent_2, alpha_cf, alpha_pw, alpha_bw
model = 'tinnitus'
rope = True
ave = False
all_rois = True


if cur_feat == 'alpha_osc':
    data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_rkf')
else:
    data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_final')

if 'alpha' in cur_feat:
    df_stats = pd.read_csv(data_f / f'{cur_feat}_hi__peak_threshold_{2.0}.csv', index_col = 0)
else:
    df_stats = pd.read_csv(data_f / f'{cur_feat}_hi__peak_threshold_{3.0}.csv', index_col = 0)

interaction = np.array([1 if 'age:tinnitus' in i else 0 for i in df_stats.index])  == 1
age = np.array([1 if 'age' in i else 0 for i in df_stats.index]) - interaction.copy() == 1
tinnitus = np.array([1 if 'tinnitus' in i else 0 for i in df_stats.index]) - interaction.copy() == 1


df_inter = df_stats.iloc[interaction].copy().reset_index()
df_inter['roi'] = [i[8:-15] for i in df_inter['index']]
df_age = df_stats.iloc[age].copy().reset_index()
df_age['roi'] = [i[8:-6] for i in df_age['index']]
df_tinnitus = df_stats.iloc[tinnitus].copy().reset_index()
df_tinnitus['roi'] = [i[8:-11] for i in df_tinnitus['index']]


if model == 'tinnitus':
    cur_df = df_tinnitus.copy()
elif model == 'inter':
    cur_df = df_inter.copy()
elif model == 'age':
    cur_df = df_age.copy()

if rope:
    if cur_feat == 'alpha_osc':
        log_rope = 0.05 * sp.constants.pi / np.sqrt(3)
        rope_l, rope_h = -1*log_rope, log_rope #for logistic model 0.1 * pi / np.sqrt(3)
    else:
        rope_l, rope_h = -.05, .05

else:
    rope_l, rope_h = -.0, .0
#
mask_neg = np.logical_and(cur_df['hdi_94.5%'] < rope_l, cur_df['hdi_5.5%'] < rope_l)
mask_pos = np.logical_and(cur_df['hdi_94.5%'] > rope_h, cur_df['hdi_5.5%'] > rope_h)

mask = np.logical_or(mask_pos, mask_neg)

if cur_feat == 'alpha_osc':
    cur_df['mean'] = np.exp(cur_df['mean'])
cur_df['mean_mask'] = cur_df['mean'] * mask


#%
if ave == False:
    if cur_feat == 'alpha_osc':
        sig_ch = cur_df['roi'][cur_df['hdi_94.5%'].argmin()]
    else:
        if cur_df['hdi_5.5%'].max() > np.abs(cur_df['hdi_94.5%'].min()):
            sig_ch = cur_df['roi'][cur_df['hdi_5.5%'].argmax()]
        else:
            sig_ch = cur_df['roi'][cur_df['hdi_94.5%'].argmin()]
#% L_A1_ROI

if 'alpha' in cur_feat:
    fmin = 2
    fmax = 16
elif 'Exponent_2' in cur_feat:
    fmin, fmax=20, 200
else: 
    fmin, fmax=1, 100

#sig_ch = 'L_A1_ROI'#'L_A1_ROI' # 'L_LBelt_ROI'

if ave:
    mask_list = cur_df.loc[mask]['roi'].values.tolist()
    a1 = df.query('roi == @mask_list').groupby(['subject_id', 'freqs']).mean(numeric_only=True).reset_index().query(f'freqs >= {fmin} and freqs <= {fmax}')
else:
    a1 = df.query(f'roi == "{sig_ch}"').query(f'freqs >= {fmin} and freqs <= {fmax}')

if all_rois:
    mask_list = cur_df.loc[mask]['roi'].values.tolist()
    a1 = df.query('roi == @mask_list').query(f'freqs >= {fmin} and freqs <= {fmax}')
    g = sns.FacetGrid(a1, col='roi', col_wrap=5, hue='tinnitus', sharey=False)
    g.map_dataframe(sns.lineplot, x='freqs', y='Power', errorbar='ci')
    
    for ax in g.axes.flatten():
        ax.set_ylabel('Power')
        ax.set_xlabel('Frequency (Hz)')

        if 'alpha' not in cur_feat:
            ax.set_yscale('log')
            ax.set_xscale('log')

    sns.despine()
    #g.figure.savefig(f'../results/final_shit/raw_spectra_{cur_feat}_all_{all_rois}.svg')

else:
    f, ax = plt.subplots(figsize=(5,5))
    sns.lineplot(a1, x='freqs', y='Power', 
                hue='tinnitus', 
                errorbar='ci',
                ax=ax)
    ax.set_ylabel('Power')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f'{sig_ch}')

    if 'alpha' not in cur_feat:
        ax.set_yscale('log')
        ax.set_xscale('log')

    sns.despine()
    #f.savefig(f'../results/final_shit/raw_spectra_{cur_feat}.svg')


#%%

cand = 'L_A1_ROI'
fmin, fmax = 2, 15#15, 100
f, ax = plt.subplots(figsize=(5,5))
sns.lineplot((df.query(f'roi == "{cand}"')
             .query(f'freqs > {fmin} and freqs < {fmax}'))
             , x='freqs', y='Power', 
            hue='tinnitus', 
            #errorbar='se',
            ax=ax)
ax.set_ylabel('Power')
ax.set_xlabel('Frequency (Hz)')
ax.set_title(f'{cand}')

#if 'alpha' not in cur_feat:
ax.set_yscale('log')
ax.set_xscale('log')
import matplotlib.ticker as mticker

fmt = mticker.FuncFormatter(lambda v, pos: f"{v:g}")
ax.xaxis.set_major_formatter(fmt)
ax.xaxis.set_minor_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
ax.yaxis.set_minor_formatter(fmt)
#ax.xaxis.set_minor_formatter(mticker.NullFormatter())
#ax.yaxis.set_minor_formatter(mticker.NullFormatter())


xmin, xmax = ax.get_xlim()

loc = mticker.LogLocator(base=10, subs=np.arange(2, 10))
ticks = loc.tick_values(xmin, xmax)

# keep only minor ticks inside view
minor_ticks = [t for t in ticks if xmin <= t <= xmax]

if minor_ticks:
    ax.xaxis.set_minor_locator(mticker.FixedLocator([minor_ticks[0]]))
else:
    ax.xaxis.set_minor_locator(mticker.NullLocator())

ymin, ymax = ax.get_ylim()

loc = mticker.LogLocator(base=10, subs=np.arange(2, 10))
ticks = loc.tick_values(ymin, ymax)

# keep only minor ticks inside view
minor_ticks = [t for t in ticks if ymin <= t <= ymax]

if minor_ticks:
    ax.yaxis.set_minor_locator(mticker.FixedLocator([minor_ticks[0]]))
else:
    ax.yaxis.set_minor_locator(mticker.NullLocator())
# else:
#     ax.set_yscale('log')

sns.despine()



#f.savefig(f'../results/final_shit/raw_spectra_{cur_feat}.svg')

#%%
from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface


gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')

gdf['values'] = 0
gdf.loc[gdf['roi'] == cand, 'values'] = 1


f, axes = plot_surface(gdf,#.query('values == 1'),
                    column='values',
                    cmap='Reds', 
                    show_cbar=False,
                    vmin=0,
                    vmax=1
                    )
f.savefig(f'../results/final_shit/raw_spectra_{cur_feat}_region.svg')


#%%
from pyrasa.utils.peak_utils import get_peak_params

dfs = []
for subject in a1['subject_id'].unique():


    curs = a1.query(f'subject_id == "{subject}"')
    cur_peaks = get_peak_params(curs['Power'].values[np.newaxis,:], 
                                curs['freqs'].values,
                                smooth=True,
                                smoothing_window=2,
                                peak_threshold=2.,
                                min_peak_height=0.01,
                                peak_width_limits=[1, 8])
    cur_peaks = cur_peaks.query('cf <= 13 and cf >= 6')
    cur_peaks.sort_values('pw').drop_duplicates(keep='first')

    if cur_peaks.shape[0] == 0:
        cur_peaks = pd.DataFrame({'ch_name': 0,
                      'cf': 0,
                      'bw': 0,
                      'pw': 0,}, index=[0])
    cur_peaks['subject_id'] = subject
    cur_peaks['tinnitus'] = curs['tinnitus'].unique()[0]
    dfs.append(cur_peaks)

dff = pd.concat(dfs)   

dff['peak_detected'] = dff['pw'] > 0 


#%%
sns.pointplot(dff, x='tinnitus', y='pw')

import scipy.stats as st

st.ttest_ind(dff.query('tinnitus == False')['pw'],
             dff.query('tinnitus == True')['pw'],)


#%%
for subject in a1['subject_id'].unique():


    fmin = 0
    fmax = 20

    curs = a1.query(f'subject_id == "{subject}"').query(f'freqs >= {fmin} and freqs <= {fmax}')

    if curs['tinnitus'].unique()[0] == False:
        cur_p = dff.query(f'subject_id == "{subject}"')

        f, ax = plt.subplots(figsize=(5,5))
        sns.lineplot(curs, x='freqs', y='Power', ax=ax)
        ax.set_title(subject)

        try:
            ax.axhline(cur_p['pw'].values[0], color='r')
            ax.axvline(cur_p['cf'].values[0], color='r')
        except IndexError:
            print(subject)
# %%



# %%
df_tmp = (#roi == "L_A1_ROI" and
    a1.query('freqs >= 4 and freqs <= 15')
      .copy()
)

sns.set_context('paper')

def ridge_lines(
    data, *,
    spacing=1,
    alpha=0.35,
    lw=1.0,
    fill=True,
    sort_desc=True,
    sort_asc=None,
    palette="magma_r",
    color_reverse=False,
    peak_col="peak_detected",   # name of your column
    grey="0.6",                 # matplotlib grey (string "0.6") or tuple
    **kwargs
):
    ax = plt.gca()

    if sort_asc is not None:
        sort_desc = not sort_asc

    # sort subjects by average Power within this facet (7–13 Hz)
    subj_mean = (
        data.query("freqs >= 7 and freqs <= 13")
            .groupby("subject_id")["Power"]
            .mean()
    )
    subjects = subj_mean.sort_values(ascending=not sort_desc).index.tolist()

    # palette colors follow the sorted order
    colors = sns.color_palette(palette, n_colors=len(subjects))
    if color_reverse:
        colors = list(reversed(colors))

    if spacing is None:
        p01, p99 = np.nanpercentile(data["Power"], [1, 99])
        spacing = (p99 - p01) * 1.15 if (p99 - p01) > 0 else 1.0

    for i, (sub, c) in enumerate(zip(subjects, colors)):
        d = data.loc[data["subject_id"] == sub].sort_values("freqs")

        # determine whether this subject has a detected peak
        # (works if peak_col is boolean, 0/1, or True/False; and constant per subject)
        has_peak = bool(d[peak_col].iloc[0]) if peak_col in d.columns else True

        ridge_color = c if has_peak else grey

        base = i * spacing
        y = d["Power"].to_numpy() + base

        ax.plot(d["freqs"], y, alpha=alpha, lw=lw, color=ridge_color)
        if fill:
            ax.fill_between(d["freqs"], base, y, alpha=alpha * 0.6, color=ridge_color)

    ax.set_yticks([])
    ax.spines.left.set_visible(False)


df_tmp = df_tmp.merge(dff[['subject_id', 'peak_detected']], on='subject_id')


g = sns.FacetGrid(
    data=df_tmp,
    col="tinnitus",
    height=5,
    aspect=.25,
    sharex=True,
    sharey=True
)

g.map_dataframe(ridge_lines, sort_asc=False, palette="viridis", peak_col="peak_detected", grey="0.7")
g.set_axis_labels("freqs", f"Power {sig_ch}")
g.set_titles("tinnitus = {col_name}")

 #  ax.set_yscale('log')
   #ax.set_xscale('log')
#plt.tight_layout()
# %%

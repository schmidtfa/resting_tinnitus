#imports
from os import listdir
from os.path import join
from datetime import datetime
import numpy as np
import mne
import joblib
from pathlib import Path
from preproc_utils import preproc_data
import matplotlib as mpl



def get_nearest_empty_room(info):
    """
    This function finds the empty room file with the closest date to the current measurement.
    The file is used for the noise covariance estimation.
    """
    empty_room_path = '/mnt/sinuhe/data_raw/empty_room/subject_subject'
    all_empty_room_dates = np.array([datetime.strptime(date, '%y%m%d') for date in listdir(empty_room_path)])

    cur_date = info['meas_date']
    cur_date_truncated = datetime(cur_date.year, cur_date.month, cur_date.day)  # necessary to truncate

    def _nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    while True:
        nearest_date_datetime = _nearest(all_empty_room_dates, cur_date_truncated)
        nearest_date = nearest_date_datetime.strftime("%y%m%d")

        cur_empty_path = join(empty_room_path, nearest_date)

        # do not use 210115 (styrofoam head fake measurement)
        if cur_empty_path == '/mnt/sinuhe/data_raw/empty_room/subject_subject/210115':
            cur_empty_path = '/mnt/sinuhe/data_raw/empty_room/subject_subject/210114'
        # do not use 210321 (does not start with file id tag)
        elif '220321' in cur_empty_path:
            cur_empty_path = '/mnt/sinuhe/data_raw/empty_room/subject_subject/220322'
        elif '220728' in cur_empty_path:
            cur_empty_path = '/mnt/sinuhe/data_raw/empty_room/subject_subject/220721'

        if 'supine' in listdir(cur_empty_path)[0]:
            all_empty_room_dates = np.delete(all_empty_room_dates,
                                                all_empty_room_dates == nearest_date_datetime)
        elif np.logical_and('68' in listdir(cur_empty_path)[0],
                            'sss' not in listdir(cur_empty_path)[0].lower()):
            break

    fname_empty_room = join(cur_empty_path, listdir(cur_empty_path)[0])

    return fname_empty_room



def raw2source(raw, subject_id, subjects_dir, preproc_settings, parc='HCPMMP1'):

    # %Compute a covariance matrix
    ###### ESTIMATE NOISE COVARIANCE MATRIX
    #select only meg channels from raw
    raw.pick(picks=['meg'])

    info = raw.info

    fname_empty_room = get_nearest_empty_room(info)
    empty_room = preproc_data(fname_empty_room, **preproc_settings)

    noise_cov = mne.compute_raw_covariance(empty_room, rank=None, picks='meg', method='auto')
    true_rank = mne.compute_rank(noise_cov, info=empty_room.info)  # inferring true rank

    ###### MAKE FORWARD SOLUTION AND INVERSE OPERATOR
    # The files live in:
    trans_path = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/headmodels/'

    fs_path = join(subjects_dir, 'fsaverage')
    fname_trans = join(trans_path, subject_id, subject_id + '-trans.fif')

    src_file = join(fs_path, 'bem', 'fsaverage-ico-4-src.fif')
    bem_file = join(fs_path, 'bem', 'fsaverage-5120-bem-sol-single-layer.fif')

    fwd = mne.make_forward_solution(info=info, trans=fname_trans, src=src_file, bem=bem_file)
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, rank=true_rank, loose=0, fixed=True, depth=0.8)

    snr = 3
    lambda2 = 1 / snr ** 2  # = default value

    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=lambda2, method='MNE')

    #% get tc from parcellation and return
    src = mne.read_source_spaces(src_file)
    labels_mne = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir)

    names_order_mne = np.array([label.name[:-3] for label in labels_mne])

    rh = [True if label.hemi == 'rh' else False for label in labels_mne]
    lh = [True if label.hemi == 'lh' else False for label in labels_mne]

    label_info = {'lh': lh,
                  'rh': rh,
                  'parc': parc,
                  'names_order_mne': names_order_mne}

    label_tc = mne.extract_label_time_course(stc, labels_mne, src, mode='mean') #TODO: Maybe try PCA

    data_dict = {'label_tc': label_tc,
                 'label_info': label_info}


    return data_dict



def plot_parc(stc_parc, stc_mask, labels_mne, 
                subjects_dir, cmap, clevels, plot_kwargs, 
                parc='HCPMMP1'):

    mpl.use('Qt5Agg')

    labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

    names_order_mne = np.array([label.name[:-3] for label in labels_mne])

    rh = [True if label.hemi == 'rh' else False for label in labels_mne]
    lh = [True if label.hemi == 'lh' else False for label in labels_mne]

    import nibabel as nib
    Brain = mne.viz.get_brain_class() #doesnt work directly from pysurfer

    brain = Brain("fsaverage", **plot_kwargs)

    #mask locations based on percentile
    for hemi in ["lh", "rh"]:

        annot_file = subjects_dir + f'/fsaverage/label/{hemi}.{parc}.annot'
        labels, _, nib_names = nib.freesurfer.read_annot(annot_file)

        names_order_nib = np.array([str(name)[2:-1] for name in nib_names])

        if hemi == "lh":
            names_mne = names_order_mne[lh]
            cur_stc = stc_parc[lh]#, tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[lh]
        else:
            names_mne = names_order_mne[rh]
            cur_stc = stc_parc[rh]#, tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[rh]

        # Create a dictionary to map strings to their indices in array1
        index_dict = {value: index for index, value in enumerate(names_mne)}

        # Find the indices of strings in array1 corresponding to array2
        right_order = [index_dict[value] for value in names_order_nib]

        cur_stc_ordered = cur_stc[right_order]
        cur_mask_ordered = cur_mask[right_order]
        
        cur_stc_ordered[cur_mask_ordered] = np.nan

        vtx_data = cur_stc_ordered[labels]
        vtx_data[labels == -1] = -1

        brain.add_data(vtx_data, hemi=hemi, fmin=clevels[0], fmid=clevels[1],
                       fmax=clevels[2], colormap=cmap, #np.nanmax(stc_parc)
                       colorbar=False, alpha=.8)

    
    screenshot = brain.screenshot()
    #brain.close()
    
    return screenshot
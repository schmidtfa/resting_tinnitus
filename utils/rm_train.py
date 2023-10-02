from trf import trf_corr, trf_reg
import numpy as np
import scipy.stats as stats


def rm_train_ica(raw, ica, lam=100):

        #get info for train timeseries
        n_samples=raw.get_data().shape[1]
        fs=raw.info['sfreq']
        t = n_samples/fs
        samples = np.linspace(0, t, int(fs*t), endpoint=False)
        train = np.sin(2*np.pi*16.6666*samples)

        ic_signal = ica.get_sources(raw).get_data()

        # First get XX^T and XY
        xxt, xy, t_trf = trf_corr(train[np.newaxis, :], ic_signal, fs,
                                -0.1, 0.1)

        #% Now do inverse with some regularization
        w = trf_reg(xxt, xy, 1, lam, reg_type='ridge')

        rs = []
        for ch in np.arange(ic_signal.shape[0]):
                
                pred = np.convolve(np.squeeze(w)[ch,:], ic_signal[ch,:], mode='same') 

                r, p = stats.pearsonr(pred, ic_signal[ch,:])
                
                rs.append(r)

        return np.array(rs)
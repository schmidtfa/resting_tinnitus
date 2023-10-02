import numpy as np
import pandas as pd
import fitz
from os.path import split
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.pyplot as plt 


def audiogram_scraper(infile, remove_weird=True):
    '''
    The function simply takes the filename of a pdf and will return two pandas dataframes.

    Note: This function is super specific so it only works on the typical pdf style we have for our audiograms.
    If you want to use it on different pdfs you need to adapt at least the indices.
    '''

    doc = fitz.open(infile)
    page = doc.load_page(0)

    get_obj_height = lambda rect: (rect[1] + rect[3]) / 2

    height_120 = get_obj_height(page.search_for('120')[1])
    height_10 = get_obj_height(page.search_for('-10')[1])

    y_ticks = np.linspace(height_10, height_120, 27) # compute the height of the y ticks
    y_labels = np.arange(-10, 125, 5)


    blk_list = []
    blocks = np.array(page.get_text('rawdict')['blocks']) # to support smart indexing

    for block in blocks:
        if block['type'] == 1 and block['width'] == block['height']: # only the dots have this property
            blk_list.append(block['number'])
            
    list_switch = int(len(blk_list) / 2)

    left_ear = [get_obj_height(block['bbox']) for block in blocks[blk_list[:11]]]
    right_ear = [get_obj_height(block['bbox']) for block in blocks[blk_list[list_switch:list_switch+11]]]

    get_dbs = lambda ear : [y_labels[np.abs(y_ticks - cur_val).argmin()] for cur_val in ear]

    pandas_idx = split(infile)[-1][0:12]
    khz = [0.125, 0.25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8]
    khz = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    
    if left_ear == right_ear:
        print('subject %s has exactly the same values for both ears this is extremely unlikely'  % pandas_idx)

        if remove_weird == True:
            print('The subjects data with weird values is removed')
            left_ear = []
            right_ear = []
    
    audiogram_left = pd.DataFrame(dict(zip(khz, get_dbs(left_ear)), index=[pandas_idx]))
    audiogram_right = pd.DataFrame(dict(zip(khz, get_dbs(right_ear)), index=[pandas_idx]))

    return audiogram_left, audiogram_right



def plot_audiogram(audio, order_names, title):
    sns.set_style("ticks")
    #sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_context("poster")

    g = sns.lineplot(data=audio,
                 x='Frequency (Hz)',
                 y='dB',
                 hue='subject_id',
                 hue_order=order_names,
                 palette='magma_r',
                 sort=True,
                 #legend=labels,
                 alpha=0.4)

    g = sns.lineplot(data=audio,
                 x='Frequency (Hz)',
                 y='dB',
                 color='white',#'#4d004b',
                 lw=4,
                 alpha=1)
    
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles[1:], labels=labels[1:],loc=[1.1,0], fontsize=8)
    g.get_legend().remove()
    g.axes.set_ylabel("Hearing Level (dB)")
    #g.axes.set_aspect(0.05)
    g.axes.set_title(title)
    g.axes.set_xscale('log')
    #sns.despine()
    
    g.axes.set_xticks([125, 250, 500, 1000, 2000, 4000, 8000])
    g.axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylim(-20, 120)
    plt.xlim(125, 8000)
    plt.gca().invert_yaxis()
    g.yaxis.set_major_locator(ticker.MultipleLocator(20))
    return g

import os
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from linedet_utils import narrow_line_search,plot_line_ratio,coltour_chi2map,plot_std_ener

from xspec_config_multisp import allmodel_data,xPlot,delcomp,model_load,reset,Plot

from xspec import Xset,AllData,Plot,AllModels,Fit

reset()
Plot.xLog=False

plt.ioff()

os.chdir('/media/parrama/SSD/Observ/BHLMXB/Chandra/post_process/tgcat/4U1630-47/bigbatch')

Xset.chatter=5

Xset.restore('lineplots_opt/13717_heg_-1_mod_broadband_linecont.xcm')

data_mod_high=allmodel_data()

Xset.restore('lineplots_opt/13717_heg_-1_mod_autofit.xcm')

data_autofit=allmodel_data()

while 'gaussian' in AllModels(1).expression:
    delcomp('gaussian')

data_autofit_noabs=allmodel_data()

model_load(data_autofit)


line_search_e=np.array('4 10 0.05'.split()).astype(float)

line_search_norm=np.array('0.01 10 500'.split(' ')).astype(float)

chi_dict_init=np.load('/home/parrama/Documents/Work/PhD/docs/papers/Wind review/4U_obs/chi_dict_init.npy',
                      allow_pickle=True).item()
chi_dict_autofit=np.load('/home/parrama/Documents/Work/PhD/docs/papers/Wind review/4U_obs/chi_dict_autofit.npy',
                        allow_pickle=True).item()

# chi_dict_init=narrow_line_search(data_mod_high)
# chi_dict_autofit=narrow_line_search(data_autofit)

addcomps_cont=['diskbb']
absline_addcomp_position=None

mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'legend.fontsize': 8})

def paper_plot(fig_paper, chi_dict_init, chi_dict_postauto, title=None):
    line_cont_range = chi_dict_init['line_cont_range']
    ax_paper = np.array([None] * 4)
    fig_paper.suptitle(title)

    # gridspec creates a grid of spaces for subplots. We use 4 rows for the 4 plots
    # Second column is there to keep space for the colorbar. Hspace=0. sticks the plots together
    gs_paper = GridSpec(4, 1, figure=fig_paper,height_ratios=[1,1,1,1.3], hspace=0.)

    # first plot is the data with additive components
    ax_paper[0] = plt.subplot(gs_paper[0])
    prev_plot_add = Plot.add
    Plot.add = True

    # reloading the pre-autofit continuum for display
    data_mod_high.load()

    xPlot('ldata', axes_input=ax_paper[0])

    # loading the no abs autofit
    data_autofit_noabs.load()

    Plot.add = prev_plot_add

    # second plot is the first blind search coltour
    ax_paper[1] = plt.subplot(gs_paper[1], sharex=ax_paper[0])
    ax_colorbar = None
    coltour_chi2map(fig_paper,ax_paper[1],chi_dict_init,combined='paper',ax_bar=ax_colorbar)
    ax_paper[1].set_xlim(line_cont_range)

    ax_paper[2] = plt.subplot(gs_paper[2], sharex=ax_paper[0])
    # third plot is the autofit ratio with lines added
    plot_line_ratio(ax_paper[2],data_autofit=data_autofit,data_autofit_noabs=data_autofit_noabs,
                    n_addcomps_cont=len(addcomps_cont), mode='paper',
                    line_position=absline_addcomp_position,
                    line_search_e=line_search_e, line_cont_range=line_cont_range,plot_ener=False)

    # fourth plot is the second blind search coltour
    ax_paper[3] = plt.subplot(gs_paper[3], sharex=ax_paper[0])
    ax_colorbar = None

    coltour_chi2map(fig_paper,ax_paper[3],chi_dict_postauto,combined='nolegend',ax_bar='bottom',norm=(251.5,12.6))

    #coltour_chi2map(fig_paper, ax_paper[3], chi_dict_postauto, combined='nolegend', ax_bar=ax_colorbar)

    ax_paper[3].set_xlim(line_cont_range)

    plot_std_ener(ax_paper[1],mode='chimap', plot_em=True,exclude_last=True)
    plot_std_ener(ax_paper[2], plot_em=True,exclude_last=True)
    plot_std_ener(ax_paper[3],mode='chimap', plot_em=True,exclude_last=True)

    #taking off the x axis of the first 3 axis to avoid ugly stuff
    for ax in ax_paper[:3]:
            ax.xaxis.set_visible(False)

    #adding panel names
    for ax,name in zip(ax_paper,['A','B','C','D']):
        ax.text(0.02, 0.05, name, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes,fontsize=25)

    plt.tight_layout(pad=0)

fig_paper = plt.figure(figsize=(12, 15))

paper_plot(fig_paper, chi_dict_init, chi_dict_autofit)

Xset.chatter=10

plt.savefig('./4U_paper_plot.pdf')
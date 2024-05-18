import os,sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob

def plot_scurve(x,y,ax=None,ion_range=None,ion_range_stable=True,color=None,
                 color_ion_range=None,return_plot=False,label=''):
    '''
    Plots a stability curve onto an already existing ax (or creates it if it doesn't exist

    arguments:

        x: x axis data (log xi/T)
        y: y axis data (log T)
        z: z axis data if 3d plot

        ax: already existing axe. Creates a new figure if set to None

        ion_range: None or [xi_min,xi_max]
            if not None, overplot the region of a given ion in a given color (by default the same as the main color)

        ion_range_stable:
                    boolean, to hash the overplotted range to highlight it is unstable

        color: color of the main curve (matplotlib default if None)
        color_ion_range: color of the overplotted range (same as color if None)
    '''

    if ax is None:
        fig_use,ax_use=plt.subplots(1,figsize=(10,8))
        ax_use.set_xlabel(r'log($\xi$/T)')
        ax_use.set_ylabel(r'log(T)')
    else:
        ax_use=ax

    curve_plot=ax_use.plot(x,y,color=color)

    main_color=curve_plot[0].get_color()

    if ion_range is not None:
        ion_range_mask=(x+y>=ion_range[0]) & (x+y<=ion_range[1])
        ion_plot = ax_use.plot(np.array(x)[ion_range_mask], np.array(y)[ion_range_mask],
                           color=main_color if color_ion_range is None else color_ion_range,lw=5,
                           alpha=0.5 if not ion_range_stable else 1.,label=label)

    if return_plot:
        return curve_plot
def plot_4U_curves(save_path_curves=None,save_path_SEDs=None,label=False):

    os.chdir('/home/parrama/Documents/Work/PhD/docs/papers/wind_4U/global/SEDs/stability/2021/curves/scurves')


    scurves_path=glob.glob('**_scurve.dat')
    scurves_path.sort()
    s_curves=np.array([np.loadtxt(elem).T for elem in scurves_path])

    dates_range_str=np.loadtxt('dates.dat',dtype='str').T
    dates_range=mpl.dates.date2num(dates_range_str[1])
    ion_ranges = np.loadtxt('Fe.dat', dtype='str').T
    unstable = np.loadtxt('unstable_seds.dat', dtype='str')

    # making a dates colormap
    color_cmap = mpl.cm.plasma
    c_norm= mpl.colors.Normalize(vmin=min(dates_range),
                                         vmax=max(dates_range))

    colors_func = mpl.cm.ScalarMappable(norm=c_norm, cmap=color_cmap)

    fig_use = plt.figure(figsize=(10, 8))
    ax_use=fig_use.add_subplot(111)
    ax_use.set_xlabel(r'log($\xi$/T)')
    ax_use.set_ylabel(r'log(T)')
    ax_use.set_xlim(-4.9,-2)
    ax_use.set_ylim(4.1,8.5)
    for i in range(len(s_curves)):

        if ion_ranges[0][i]=='4130010109-001_mod_broader_SED':
            indiv_color=colors_func.to_rgba(mpl.dates.date2num(['2021-09-21T05:02:03']))
        else:
            indiv_color=colors_func.to_rgba(dates_range[dates_range_str[0]==ion_ranges[0][i].split('_')[0]][0])


        plot=plot_scurve(s_curves[i][0],s_curves[i][1],ax=ax_use,
                    ion_range=ion_ranges.T[i][1:].astype(float),
                    ion_range_stable=not ion_ranges[0][i] in unstable,
                    color=indiv_color,return_plot=True,label= ion_ranges[0][i] if label else None)

    sm = plt.cm.ScalarMappable(cmap=color_cmap, norm=c_norm)

    date_format=mpl.dates.DateFormatter('%Y-%m-%d')
    plt.colorbar(sm,ticks=mpl.dates.AutoDateLocator(),
                               format=date_format)
    if label:
        plt.legend()

    plt.tight_layout()

    if save_path_curves is not None:
        plt.savefig(save_path_curves)


    #switching to the SEDs directory
    os.chdir('/home/parrama/Documents/Work/PhD/docs/papers/wind_4U/global/SEDs/stability/2021/SEDs')

    #loading the SEDs

    SEDs_path=glob.glob('**SED.xcm')
    SEDs_path.sort()

    SEDs=np.array([np.loadtxt(elem).T for elem in SEDs_path])

    fig_sed = plt.figure(figsize=(10, 8))
    ax_sed=fig_sed.add_subplot(111)
    ax_sed.set_xlabel(r'E (keV)')
    ax_sed.set_ylabel(r'L$_{\nu}$ (erg/s/Hz)')
    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')

    ax_sed.set_xlim(1e-1,2e2)
    ax_sed.set_ylim(1e-6,3e-3)

    x_axis_bins=np.logspace(-2,3,2000)

    for i in range(len(SEDs)):

        if ion_ranges[0][i]=='4130010109-001_mod_broader_SED':
            indiv_color=colors_func.to_rgba(mpl.dates.date2num(['2021-09-21T05:02:03']))
        else:
            indiv_color=colors_func.to_rgba(dates_range[dates_range_str[0]==ion_ranges[0][i].split('_')[0]][0])

        #note:
        plt.plot(x_axis_bins[1:],SEDs[i][2][1:],color=indiv_color)

    plt.tight_layout()

    if save_path_SEDs is not None:
        plt.savefig(save_path_SEDs)

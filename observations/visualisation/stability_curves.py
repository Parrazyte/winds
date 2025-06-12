import os,sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob

import getpass
username=getpass.getuser()

def xi_i(log_xi_div_T,log_T):

    '''
    Function for the pcolormesh
    '''

    return log_xi_div_T+log_T

def plot_scurve(x,y,ax=None,ion_range=None,ion_range_stable=True,color=None,
                 color_ion_range=None,return_plot=False,label_ion='',label_main='',ls_main='-',lw_main=None):
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

        return plot:
        gives back the plot object

        ls_main:
        linestyle of the main lines
    '''

    if ax is None:
        fig_use,ax_use=plt.subplots(1,figsize=(6,6))
        ax_use.set_xlabel(r'log($\xi$/T)')
        ax_use.set_ylabel(r'log(T)')
    else:
        ax_use=ax

    curve_plot=ax_use.plot(x,y,color=color,ls=ls_main,lw=lw_main,label=label_main)

    main_color=curve_plot[0].get_color()

    if ion_range is not None:
        ion_range_mask=(x+y>=ion_range[0]) & (x+y<=ion_range[1])
        ion_plot = ax_use.plot(np.array(x)[ion_range_mask], np.array(y)[ion_range_mask],
                           color=main_color if color_ion_range is None else color_ion_range,lw=5,
                           alpha=0.5 if not ion_range_stable else 1.,ls=':' if not ion_range_stable else '-',
                               label=label_ion)

    if return_plot:
        return curve_plot
def plot_4U_2021_curves(save_path_curves=None,save_path_SEDs=None,label=False,colormap='plasma',restrict_range=None,
                        plot_hmxt=True,plot_inset=True):

    os.chdir('/home/'+username+'/Documents/Work/PhD/docs/papers/wind_4U/global/SEDs/stability/2021/curves/scurves')


    scurves_path=[elem for elem in glob.glob('**_scurve.dat') if not "insight_SED" in elem]

    scurve_hmxt_path=[elem for elem in glob.glob('**_scurve.dat') if "insight_SED" in elem][0]
    scurve_hmxt=np.loadtxt(scurve_hmxt_path).T

    scurves_path.sort()
    s_curves=np.array([np.loadtxt(elem).T for elem in scurves_path])

    dates_range_str=np.loadtxt('dates.dat',dtype='str').T
    dates_range=mpl.dates.date2num(dates_range_str[1])
    ion_ranges = np.loadtxt('Fe.dat', dtype='str').T
    unstable = np.loadtxt('unstable_seds.dat', dtype='str')

    # making a dates colormap
    color_cmap = getattr(mpl.cm,colormap)
    c_norm= mpl.colors.Normalize(vmin=min(dates_range),
                                         vmax=max(dates_range))

    colors_func = mpl.cm.ScalarMappable(norm=c_norm, cmap=color_cmap)

    fig_use = plt.figure(figsize=(6, 4.5))
    ax_use=fig_use.add_subplot(111)
    ax_use.set_xlabel(r'log($\xi$/T)')
    ax_use.set_ylabel(r'log(T)')
    ax_use.set_xlim(-4.9,-2)
    ax_use.set_ylim(4.1,8.5)

    plot_scurves=[]
    for i in range(len(s_curves)):

        if restrict_range is not None:
            if i not in range(restrict_range[0],restrict_range[1]+1):
                continue

        if ion_ranges[0][i]=='4130010109-001_mod_broader_SED':
            indiv_color=colors_func.to_rgba(mpl.dates.date2num(['2021-09-21T05:02:03']))
        else:
            indiv_color=colors_func.to_rgba(dates_range[dates_range_str[0]==ion_ranges[0][i].split('_')[0]][0])


        plot_scurves+=[plot_scurve(s_curves[i][0],s_curves[i][1],ax=ax_use,
                    ion_range=ion_ranges.T[i][1:].astype(float),
                    ion_range_stable=not ion_ranges[0][i] in unstable,
                    color=indiv_color,return_plot=True,label_ion= ion_ranges[0][i] if label else None)]

        if '4130010115' in ion_ranges[0][i]:
            plot_scurve(s_curves[i][0],s_curves[i][1],ax=ax_use,
                        ion_range=None,
                        ion_range_stable=not ion_ranges[0][i] in unstable,
                        color='black',ls_main=':')

        #adding the hue for the last jet state
        if '4130010108-002' in ion_ranges[0][i]:
            plot_scurve(s_curves[i][0],s_curves[i][1],ax=ax_use,
                        ion_range=None,
                        ion_range_stable=not ion_ranges[0][i] in unstable,
                        color='black',ls_main='--')

    if plot_hmxt:
        #adding the HMXT SED
        plot_scurve(scurve_hmxt[0], scurve_hmxt[1], ax=ax_use,
                    ion_range=None,
                    ion_range_stable=True,
                    color='dodgerblue', return_plot=True, label_main='HMXT \n2021-09-25')

    if label or plot_hmxt:
        plt.legend(loc='lower right',bbox_to_anchor=(1.45, -0.15))

    sm = plt.cm.ScalarMappable(cmap=color_cmap, norm=c_norm)

    date_format=mpl.dates.DateFormatter('%Y-%m-%d')
    plt.colorbar(sm,ticks=mpl.dates.AutoDateLocator(),
                               format=date_format)

    if plot_inset:

        # inset Axes....
        x1, x2, y1, y2 = -3.4,-3.05,6.1,6.9  # subregion of the original image
        axins = ax_use.inset_axes(
            [0.65, 0., 0.35, 0.45],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

        #replotting everything
        for i in range(len(s_curves)):

            if restrict_range is not None:
                if i not in range(restrict_range[0],restrict_range[1]+1):
                    continue

            if ion_ranges[0][i]=='4130010109-001_mod_broader_SED':
                indiv_color=colors_func.to_rgba(mpl.dates.date2num(['2021-09-21T05:02:03']))
            else:
                indiv_color=colors_func.to_rgba(dates_range[dates_range_str[0]==ion_ranges[0][i].split('_')[0]][0])


            plot_scurves+=[plot_scurve(s_curves[i][0],s_curves[i][1],ax=axins,
                        ion_range=ion_ranges.T[i][1:].astype(float),
                        ion_range_stable=not ion_ranges[0][i] in unstable,
                        color=indiv_color,return_plot=True)]

            if '4130010115' in ion_ranges[0][i]:
                plot_scurve(s_curves[i][0],s_curves[i][1],ax=axins,
                            ion_range=None,
                            ion_range_stable=not ion_ranges[0][i] in unstable,
                            color='black',ls_main=':',label_main='ejecta')

            #adding the hue for the last jet state
            if '4130010108-002' in ion_ranges[0][i]:
                plot_scurve(s_curves[i][0],s_curves[i][1],ax=axins,
                            ion_range=None,
                            ion_range_stable=not ion_ranges[0][i] in unstable,
                            color='black',ls_main='--',label_main='jet')


        # axins.legend(loc='lower left',bbox_to_anchor=(-1.85,0.2),title='radio')

        #adding the HMXT SED
        plot_scurve(scurve_hmxt[0], scurve_hmxt[1], ax=axins,
                    ion_range=None,
                    ion_range_stable=True,
                    color='dodgerblue', return_plot=True)

        ax_use.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()

    if save_path_curves is not None:
        plt.savefig(save_path_curves)


    #switching to the SEDs directory
    os.chdir('/home/'+username+'/Documents/Work/PhD/docs/papers/wind_4U/global/SEDs/stability/2021/SEDs')

    #loading the SEDs

    SEDs_path=glob.glob('**SED.xcm')
    SEDs_path.sort()

    SEDs=np.array([np.loadtxt(elem).T for elem in SEDs_path])

    SED_hmxt=np.loadtxt('insight_SED_epoch1.xcm').T

    fig_sed = plt.figure(figsize=(6,4.5))
    ax_sed=fig_sed.add_subplot(111)
    ax_sed.set_xlabel(r'E (keV)')
    ax_sed.set_ylabel(r'$\nu$F$_{\nu}$ (erg/s/Hz)')
    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')

    ax_sed.set_xlim(1,2e2)
    ax_sed.set_ylim(1e-5,1e-2)

    x_axis_bins=np.logspace(-2,3,2000)

    for i in range(len(SEDs)):

        if ion_ranges[0][i]=='4130010109-001_mod_broader_SED':
            indiv_color=colors_func.to_rgba(mpl.dates.date2num(['2021-09-21T05:02:03']))
        else:
            indiv_color=colors_func.to_rgba(dates_range[dates_range_str[0]==ion_ranges[0][i].split('_')[0]][0])

        #note:
        plt.plot(x_axis_bins[1:],SEDs[i][2][1:]*x_axis_bins[1:],color=indiv_color)

        if '4130010115' in ion_ranges[0][i]:
            plt.plot(x_axis_bins[1:], SEDs[i][2][1:]*x_axis_bins[1:], color='black',ls=':')

        #adding the hue for the last jet state
        if '4130010108' in ion_ranges[0][i]:
            plt.plot(x_axis_bins[1:], SEDs[i][2][1:]*x_axis_bins[1:], color='black',ls='--',zorder=10)


    plt.plot(x_axis_bins[1:],SED_hmxt[2][1:]*x_axis_bins[1:],color='dodgerblue')
    plt.tight_layout()

    if save_path_SEDs is not None:
        plt.savefig(save_path_SEDs)


def plot_4U_state_curves(
        indir_curves='/home/'+username+'/Documents/Work/PhD/docs/papers/wind_4U/global/SEDs/stability/states',
                         save_path_curves=None,save_path_SEDs=None,label=False,colormap='plasma',restrict_range=None,
                        plot_hmxt=True,plot_zoom=False,mode='paper'):

    '''

    mode=paper for Parra24 style
    mode=proposal for the XRISM AO style

    '''

    os.chdir(indir_curves)


    scurves_path=[elem for elem in glob.glob('**scurve.dat')]
    SEDs_path=[elem for elem in glob.glob('**') if elem not in scurves_path]

    scurves_path.sort()
    SEDs_path.sort()

    #just in case
    # ['4130010101-001_mod_broader_SED_scurve.dat', '4130010108-002_mod_broader_SED_scurve.dat',
    #  '4130010112-002_mod_broader_SED_scurve.dat', '4130010119-003_SED_scurve.dat', '4130010126-001_SED_scurve.dat',
    #  '4130010131-001_SED_scurve.dat', 'SED_broad_1p5e-1Ledd_scurve.dat', 'SED_broad_2p5e-2Ledd_scurve.dat',
    #  'SED_broad_3p5e-2Ledd_scurve.dat', 'SED_diagonal_upper_high_highE_flux_s_scurve.dat',
    #  'nu80801327002A01_nth_SED_scurve.dat', 'nu80902312004A01-005_nth_SED_scurve.dat']

    colors=['blue','blue','blue',
            'red','red',
            'orange','orange','orange','orange',
            'green',
            'red','red']

    ls_list=['-','--',':',
             '--','-',
             '-',':','--','-.',
             '-',
             '-.',':']

    s_curves=np.array([np.loadtxt(elem).T for elem in scurves_path])


    fig_use = plt.figure(figsize=(6, 4.5))
    ax_use=fig_use.add_subplot(111)
    ax_use.set_xlabel(r'log($\xi$/T)')
    ax_use.set_ylabel(r'log(T)')
    ax_use.set_xlim(-4.5,-2)
    ax_use.set_ylim(4.1,8.)

    mesh = [xi_i(np.linspace(-4.5, -2., 50), elem) for elem in np.linspace(4.1, 8., 60)]

    if mode=='paper':

        contours_nolabel= ax_use.contour(np.linspace(-4.5, -2., 50), np.linspace(4.1, 8., 60), mesh,
                                   levels=[0., 5.], colors='grey', linewidths=1.)


        contours= ax_use.contour(np.linspace(-4.5, -2., 50), np.linspace(4.1, 8., 60), mesh,
                                   levels=[1., 2., 3, 4., ], colors='grey', linewidths=1.)
    elif mode=='proposal':

        contours = ax_use.contour(np.linspace(-4.5, -2., 50), np.linspace(4.1, 8., 60), mesh,
                                  levels=[ 3, 4., ], colors='grey', linewidths=1.)

    plot_scurves=[]
    for i in range(len(s_curves)):

        if restrict_range is not None:
            if i not in range(restrict_range[0],restrict_range[1]+1):
                continue

        plot_scurves+=[plot_scurve(s_curves[i][0],s_curves[i][1],ax=ax_use,
                color=colors[i],return_plot=True,ls_main=ls_list[i],
                               ion_range=[3.35 if i==2 else 3., 3.95 if i==2 else 3.75 if i==0 else 3.85 if i==1 else 4] if mode == 'proposal' and colors[i] == 'blue' else None,
                               color_ion_range='grey' if mode == 'proposal' and colors[i] == 'blue' else None,
                                   label_ion='no wind' if i==0 and mode=='proposal' else '')]

    if plot_zoom:
        # inset Axes....
        x1, x2, y1, y2 = -3.4,-3.05,6.1,6.9  # subregion of the original image
        axins = ax_use.inset_axes(
            [0.65, 0., 0.35, 0.45],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

        plot_scurves=[]
        for i in range(len(s_curves)):

            if restrict_range is not None:
                if i not in range(restrict_range[0],restrict_range[1]+1):
                    continue

            plot_scurves+=[plot_scurve(s_curves[i][0],s_curves[i][1],ax=axins,
                        color=colors[i],return_plot=True,ls_main=ls_list[i],
                                       ion_range=[3,4] if mode=='proposal' else None,
                                       color_ion_range='grey' if mode=='proposal' and colors[i]=='blue' else None)]

        ax_use.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()

    fmt_clabels = {}

    label_names = [ r'log($\xi$)=1', r'log($\xi$)=2'] if mode=='paper' else [] +[r'log($\xi$)=3',r'log($\xi$)=4',]
    for i,(l, s) in enumerate(zip(contours.levels, label_names)):

        fmt_clabels[l] = s

    manual_locs=[(-4.15,5.15),(-2.8,4.8)] if mode=='paper' else [] + [(-4.,7.),(-2.45,6.45)]

    contour_labels=ax_use.clabel(contours, inline=True, fontsize=10,fmt=fmt_clabels,manual=manual_locs)

    if mode=='proposal':
        plt.legend(loc='lower right')

    if save_path_curves is not None:
        plt.savefig(save_path_curves)


    #loading the SEDs

    #to be ok with different grids we load it this way
    SEDs=np.repeat(None,len(SEDs_path))
    for i_path,path in enumerate(SEDs_path):
        SEDs[i_path]=np.loadtxt(path).T

    fig_sed = plt.figure(figsize=(6,4.5))
    ax_sed=fig_sed.add_subplot(111)
    ax_sed.set_xlabel(r'E (keV)')
    ax_sed.set_ylabel(r'$\nu$F$_{\nu}$ (erg/s/Hz)')
    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')

    ax_sed.set_xlim(1,1e2)
    ax_sed.set_ylim(1e-5,1.5e-2)

    x_axis_bins=np.logspace(-2,3,2000)

    # old SED binning
    x_axis_bins_old=np.logspace(-1,2,1000)

    for i in range(len(SEDs)):

        if 0 or 'mod_broader' in SEDs_path[i]:
            plt.plot(x_axis_bins[1:],SEDs[i][2][1:]*x_axis_bins[1:],color=colors[i],ls=ls_list[i])
        else:
            plt.plot(x_axis_bins_old[1:],SEDs[i][2][1:]*x_axis_bins_old[1:],color=colors[i],ls=ls_list[i])

    plt.tight_layout()

    if save_path_SEDs is not None:
        plt.savefig(save_path_SEDs)

    return fig_use,fig_sed
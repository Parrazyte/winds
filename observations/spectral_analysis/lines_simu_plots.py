import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import glob
import pandas as pd
from general_tools import ravel_ragged
ap = argparse.ArgumentParser(description='Script to plot line detectability from various instruments.\n)')

'''GENERAL OPTIONS'''

mpl.rcParams.update({'font.size': 18})

ap.add_argument('-vert',nargs=1,help='plot vertical figure',default=False)

ap.add_argument('-dir',nargs=1,help='simulations and plotting directory',
                default='/media/parrama/crucial_SSD/Observ/highres/linedet_compa/AO2/4U1957',type=str)

ap.add_argument('-all_subdirs',nargs=1,help='make the plots from all simulations in the subdirectories',
                default=True,type=bool)

ap.add_argument('-display_data_ul',nargs=1,help='display upper limits from existing data using CSV files',
                default=True,type=bool)

ap.add_argument('-ul_line_csv',nargs=1,help='CSV of lines to overplot line constraints',
                default='line_table_4U1957.csv',type=str)

ap.add_argument('-ul_obs_csv',nargs=1,help='CSV of flux values to overplot line constraints',
                default='observ_table_4U1957.csv',type=str)

#currently can be EW or bshift
ap.add_argument("-mode",nargs=1,help='plot the results of the simulations for different modes',
                default='combi',type=str)

ap.add_argument("-fakestats",nargs=1,help='use run with or without fakeit statistical fluctuations',
                default=True,type=str)

ap.add_argument('-n_iter',nargs=1,help='number of iterations of each flux level for XRISM/XMM/Chandra',
                default=[10,100,100])

ap.add_argument('-expos_ew',nargs=1,help='Exposure time in ks for XRISM/XMM/Chandra',
                default=['50','50','50'])

ap.add_argument('-expos_bshift',nargs=1,help='Exposure time in ks for XRISM/XMM/Chandra',
                default=['50','50','50'])

ap.add_argument('-line',nargs=1,help='line to use for XRISM/XMM/Chandra',
                default=['FeKa26abs+SKa16abs','FeKa26abs','FeKa26abs'])


#for default subdir listing
ap.add_argument('-subdir_list',nargs=1,help='manual subdir list for indexing',
                default=['epoch4_XRISM_FeKa26_40ks','epoch4_XRISM_SKa16_40ks',
                          'epoch8_XRISM_FeKa26_40ks','epoch8_XRISM_SKa16_40ks'])
ap.add_argument('-man_label_list',nargs=1,help='manual label list',
                default=['x','x','x','x'])
ap.add_argument('-man_label_plot_list',nargs=1,help='manual label list',
                default=['Soft SED - FeKa26','Soft SED - SKa16',
                         'Inter SED - FeKa26','Inter SED - SKa16'])
ap.add_argument('-man_marker_list',nargs=1,help='manual label list',
                default=['+','x','+','x'])
ap.add_argument('-man_ls_list',nargs=1,help='manual label list',
                default=[':','--',':','--'])
ap.add_argument('-man_lw_list',nargs=1,help='manual label list',
                default=[2,2,2,2])
ap.add_argument('-man_color_list',nargs=1,help='manual label list',
                default=['black','black','purple','purple'])


ap.add_argument('-flux_str',nargs=1,help='flux logspace parameters for file fetching',default='1_100_10',type=str)

ap.add_argument('-mask',nargs=1,help='mask part of the values',default='1/1')

#EW arguments
ap.add_argument('-width_inter',nargs=1,help='width interval bounds',default=[5e-3,5e-3])

#bshift arguments
ap.add_argument('-bshift_EW_val',nargs=1,help='EW for which to simulate the blueshift in eVs',default=20)

ap.add_argument('-bshift_width_val',nargs=1,help='bshift for which to simulate the bshfit errors in eVs', default=0.005)

ap.add_argument('-sigmas',nargs=1,help='sigmas to plot',default=[3])

args=ap.parse_args()
vert=args.vert
all_subdirs=args.all_subdirs
ul_line_csv=args.ul_line_csv
ul_obs_csv=args.ul_obs_csv
display_data_ul=args.display_data_ul

dir=args.dir
mode=args.mode
fakestats=args.fakestats
n_iter=args.n_iter
expos_ew=args.expos_ew
expos_bshift=args.expos_bshift
line=args.line

subdir_list=args.subdir_list
man_label_list=args.man_label_list
man_label_plot_list=args.man_label_plot_list
man_marker_list=args.man_marker_list
man_ls_list=args.man_ls_list
man_lw_list=args.man_lw_list
man_color_list=args.man_color_list

flux_str=args.flux_str

mask_vals=np.array(args.mask.split('/')).astype(float)
width_inter=args.width_inter
bshift_EW_val=args.bshift_EW_val
bshift_width_val=args.bshift_width_val

width_str='_'.join([str(elem) for elem in width_inter])

#to get back to index
id_sigmas=np.array(args.sigmas)-1


def plot_indiv_panel(ax_use,panel_mode,show_yaxis=True,legend=True,
                     base_dir='/media/parrama/crucial_SSD/Observ/highres/linedet_compa/AO2/',
                     subdir='auto',secax=None,
                     display_lims=True,line_csv='line_table_AO2.csv',obs_csv='observ_table_AO2.csv',
                     #custom commands for non-auto subdir
                     man_label=None,man_label_plot=None,man_marker=None,man_ls=None,man_lw=1,man_color=None,
                     label_constraints=True,lock_lims=False):

    def loglog_regressor(X,Y,ax,color,label='',marker='',ls='',lw=1,label_plot=True,secax=None):

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import linregress

        # Transform to log-log space
        log_X = np.log10(X)
        log_Y = np.log10(Y)

        # Perform linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = linregress(log_X, log_Y)

        # Regression line in log-log space
        log_Y_pred = slope * log_X + intercept

        log_X_pred= (log_Y - intercept)/slope

        # Plot
        # ax.plot(X, 10 ** log_Y_pred, color=color)
        if secax is not None:
            ax_scatter=secax
        else:
            ax_scatter=ax


        ax_scatter.scatter(10**log_X_pred, Y,label=(label.split(' ')[-1]) if 'XRISM' in label
                else '',marker=marker,color=color,s=100)

        ax.plot(10**log_X_pred, Y,ls=ls,color=color,lw=lw,alpha=1.,label=label_plot)

        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('X (log scale)')
        # plt.ylabel('Y (log scale)')
        # plt.title('Linear Regression in Log-Log Space')
        # plt.legend()
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.tight_layout()
        # plt.show()

    # ax_use.set_xscale('linear')
    ax_use.set_yscale('log')
    ax_use.set_xscale('log')

    if secax is not None:
        secax.set_yscale('log')
        secax.set_xscale('log')
    if show_yaxis:
        ax_use.set_ylabel('Observed [3-10] keV flux (ergs/s/cm²)')

    if panel_mode=='ew':
        ax_use.set_xlabel('EW treshold for '+str(id_sigmas[0]+1) +r'$\sigma$ detection (eV)')
    elif panel_mode=='bshift':
        ax_use.set_xlabel((str(id_sigmas[0]+1) +r'$\sigma$ ' if len(id_sigmas)==1 else '')+'velocity shift error (km/s)')

    if legend:
        ax_use.legend()

    expos_arg=expos_ew if panel_mode=='ew' else expos_bshift
    marker_indiv=['.','x','+']
    # marker_indiv=['','','']

    ls_indiv=[':','--','-']
    # ls_indiv=[(2, (2, 4)),(0,(2,4))]

    lw_indiv=[2,2]

    expos_list=np.array(["10","35","50"])
    #loading elements

    color_tel=[['black','purple'],
                ['blue','cyan'],
                ['green','lightgreen']]

    def plot_subdir(file_path,label,label_plot,marker,ls,color,lw,secax):


        arr_tel = np.loadtxt(file_path)

        mask_vals_tel = [mask_vals[0] * i % mask_vals[1] == 0 for i in range(len(arr_tel))]

        arr_tel = arr_tel[mask_vals_tel]

        for i_sigma in id_sigmas:
            loglog_regressor(arr_tel.T[1 + i_sigma], arr_tel.T[0], ax_use,
                             label=label,
                             label_plot=label_plot,
                             marker=marker, ls=ls,
                             color=color,
                             lw=lw, secax=secax)

    if subdir=='auto':

        for i_tel,telescope in enumerate(['XRISM','Chandra','XMM']):
            for i_line,elem_line in enumerate(line[i_tel].split('+')):
                if elem_line=='':
                    continue
                for i_expos,expos in enumerate(expos_arg[i_tel].split('+')):

                    id_expos=np.argwhere(expos_list==expos)[0][0]
                    file_id=os.path.join(base_dir, telescope+'_simu_'+expos+'ks_'+elem_line.replace('abs','') ,
                                           'ew_lim**' if panel_mode=='ew' else 'bshift_err**' if panel_mode=='bshift'
                                           else 'thiswonteverhappen')

                    file_path_load=[elem for elem in glob.glob(file_id) if 'ks' in elem.split('/')[-1]]
                    if len(file_path_load)==0:
                        continue

                    panel_label=(telescope + ' ' + str(expos) + 'ks') if (telescope=='XRISM' and i_line==0)\
                                             else ''
                    panel_label_plot=(telescope+' '+elem_line.replace('abs','')) if \
                                             i_expos==len(expos_arg[i_tel].split('+'))-1 else ''
                    panel_marker=marker_indiv[id_expos]
                    panel_ls=ls_indiv[i_line]
                    panel_color=color_tel[i_tel][i_line]
                    panel_lw=lw_indiv[i_line]
                    plot_subdir(file_path_load[0],label=panel_label,label_plot=panel_label_plot,
                                marker=panel_marker,ls=panel_ls,color=panel_color,lw=panel_lw,secax=secax)
    else:

        file_id = os.path.join(base_dir,  subdir,
                               'ew_lim**' if panel_mode == 'ew' else 'bshift_err**' if panel_mode == 'bshift'
                               else 'thiswonteverhappen')

        file_path_load = [elem for elem in glob.glob(file_id) if 'ks' in elem.split('/')[-1]]
        if len(file_path_load) == 0:
            breakpoint()


        plot_subdir(file_path_load[0], label=man_label, label_plot=man_label_plot,
                     marker=man_marker, ls=man_ls, color=man_color, lw=man_lw, secax=secax)


    if panel_mode=='ew' and display_lims:
        obs_line_csv=pd.read_csv(os.path.join(base_dir,line_csv),skiprows=3,header=None)
        obs_UL=obs_line_csv[[18]]
        obs_data_csv=pd.read_csv(os.path.join(base_dir,obs_csv),skiprows=3,header=None)
        obs_flux_3_10=obs_data_csv[[19]]
        obs_HR=obs_data_csv[[4]]

        if lock_lims:
            ax_use.set_xlim(ax_use.get_xlim())
            ax_use.set_ylim(ax_use.set_ylim())

        #doubling the flux values for Hard obs since the diagram is for a soft state
        obs_flux_readj=ravel_ragged([elem*(2 if elem2>0.5 else 1) for (elem,elem2)\
                                     in zip(obs_flux_3_10.values,obs_HR.values) ])

        obs_ul_vals=ravel_ragged(obs_UL.values)
        # breakpoint()
        ax_use.errorbar(obs_ul_vals,obs_flux_readj,xerr=obs_ul_vals/10,xuplims=True,ls='',
                        label='FeKa26 constraints' if label_constraints else '',color='grey',marker='')

    if legend:
        ax_use.legend(loc='lower left')

        if secax is not None:
            secax.legend(loc='upper right')

if mode in ['ew','bshift']:
    fig_full,ax_full=plt.subplots(figsize=(8,8))


    if mode=='ew':
        plot_indiv_panel(ax_full,panel_mode=mode)

        plt.rcParams['figure.constrained_layout.use'] = True

        # necessary to have the constrained layout saved
        plt.show()

        plt.savefig('/media/parrama/crucial_SSD/Observ/highres/linedet_compa/AO1/EW_detec_compa'+
                ('_nostat' if not fakestats else '')+
                '_'+str(expos_ew)+
                '_' + str(n_iter) + '_iter' +
                    '_flux_' + str(flux_str) +
                '_width_' + width_str +'.png')

    elif mode=='bshift':
        plot_indiv_panel(ax_full,panel_mode=mode)

        plt.rcParams['figure.constrained_layout.use'] = True

        # necessary to have the constrained layout saved
        plt.show()

        plt.savefig('/media/parrama/crucial_SSD/Observ/highres/linedet_compa/AO1/bshift_err_compa'+
                ('_nostat' if not fakestats else '')+
                '_' + str(expos_bshift) +
                '_' + str(n_iter) + '_iter' +
                    '_flux_' + str(flux_str) +
                'EW_' + str(bshift_EW_val) +
                '_width_' + str(bshift_width_val) + '.png')

elif mode=='combi':

    vert=False

    fig_xull,ax_full=plt.subplots(figsize=(8,14) if vert else (14,7))

    if vert:
        gs1 = gridspec.GridSpec(2,1)

    else:
        gs1 = gridspec.GridSpec(1,2)
    gs1.update(wspace=0.25 if args.vert else 0.05, hspace=0.25 if args.vert else 0.025)
    axes=[None,None]
    for i in range(2):
       # i = i + 1 # grid spec indexes from 0
        axes[i]= plt.subplot(gs1[i])
        if i==1:
           axes[i].yaxis.tick_right()

    vert=args.vert

    mpl.rcParams['legend.fontsize'] = 10 if vert else 14

    secax_0=axes[0].twinx()

    if all_subdirs:
        # list_subdirs=glob.glob(os.path.join(dir,'**/'))

        for i_subdir,subdir in enumerate(subdir_list):


            last_subdir=i_subdir==len(subdir_list)-1

            plot_indiv_panel(axes[0], panel_mode='ew', base_dir=dir,subdir=subdir, secax=secax_0,
                             legend=last_subdir,display_lims=display_data_ul,
                             line_csv=ul_line_csv,obs_csv=ul_obs_csv,
                             man_label=man_label_list[i_subdir],
                             man_label_plot=man_label_plot_list[i_subdir],
                             man_marker=man_marker_list[i_subdir],
                             man_ls=man_ls_list[i_subdir],
                             man_color=man_color_list[i_subdir],
                             man_lw=man_lw_list[i_subdir],
                             label_constraints=last_subdir,
                             lock_lims=last_subdir)

            plot_indiv_panel(axes[1], panel_mode='bshift', base_dir=dir,subdir=subdir, show_yaxis=vert, legend=False,
                             man_label=man_label_list[i_subdir],
                             man_label_plot=man_label_plot_list[i_subdir],
                             man_marker=man_marker_list[i_subdir],
                             man_ls=man_ls_list[i_subdir],
                             man_color=man_color_list[i_subdir],
                             man_lw=man_lw_list[i_subdir],
                             lock_lims = last_subdir)

        secax_0.set_yticks([])
        secax_0.set_yticklabels([])
        secax_0.tick_params(labelright='off')

        #plotting the flux limits
        axes[0].axhline(y=4.7922e-10,color='grey',ls=':',lw=1)
        axes[0].axhline(y=1.2681e-09,color='grey',ls=':',lw=1)

        axes[1].axhline(y=4.7922e-10,color='grey',ls=':',lw=1)
        axes[1].axhline(y=1.2681e-09,color='grey',ls=':',lw=1)

    else:
        plot_indiv_panel(axes[0],panel_mode='ew',base_dir=dir,legend=True,secax=secax_0,
                         display_lims=display_data_ul,
                             line_csv=ul_line_csv,obs_csv=ul_obs_csv)

        secax_0.set_yticks([])
        secax_0.set_yticklabels([])
        secax_0.tick_params(labelright='off')


        if vert:
            axes[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[0].xaxis.tick_top()
            axes[0].tick_params(labeltop=True)
            axes[0].xaxis.set_label_position('top')
        plt.legend()

        plot_indiv_panel(axes[1],panel_mode='bshift',base_dir=dir,show_yaxis=vert,legend=False)

        if vert:
            axes[1].yaxis.tick_left()
            axes[1].tick_params(labelleft=True)
            # axes[0].yaxis.set_label_position('top')
        plt.legend()

    if not vert:
        plt.rcParams['figure.constrained_layout.use'] = True

    #necessary to have the constrained layout saved

    #for figsize=(16,8)
    #plt.subplots_adjust(left=0.07, right=0.95, top=0.99, bottom=0.09)
    #
    # if vert:
    #     plt.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.08)
    # else:
    #     plt.subplots_adjust(left=0.1, right=0.93, top=0.99, bottom=0.12)

    # plt.show()

    plt.savefig(os.path.join(dir,
                ('all_subdirs_' if all_subdirs else '')+
                'combi_err_compa' +
                ('_nostat' if not fakestats else '') +
                '_' + '_'.join(expos_ew) +
                '__' + '_'.join(expos_bshift) +
                '__' + '_'.join(np.array(n_iter,dtype=str)) + '_iter' +
                '_'+'_'.join(line)+
                '_flux_' + str(flux_str) +
                'EW_' + str(bshift_EW_val) +
                '_width_' + str(bshift_width_val) +
                ('_vert' if vert else '')+'.png'))

    plt.savefig(os.path.join(dir,
                ('all_subdirs_' if all_subdirs else '')+
                'combi_err_compa' +
                ('_nostat' if not fakestats else '') +
                '_' + '_'.join(expos_ew) +
                '__' + '_'.join(expos_bshift) +
                '__' + '_'.join(np.array(n_iter,dtype=str)) + '_iter' +
                '_' + '_'.join(line)+
                '_flux_' + str(flux_str) +
                'EW_' + str(bshift_EW_val) +
                '_width_' + str(bshift_width_val) +
                ('_vert' if vert else '')+'.pdf'))


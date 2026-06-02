import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import glob
import pandas as pd
from general_tools import ravel_ragged
from line_simus_tools import loglog_regressor
import matplotlib.ticker as tkr
from scipy.stats import linregress
from matplotlib.collections import LineCollection
import getpass
username=getpass.getuser()

ap = argparse.ArgumentParser(description='Script to plot line detectability from various instruments.\n)')

'''GENERAL OPTIONS'''

mpl.rcParams.update({'font.size': 18})

ap.add_argument('-vert',nargs=1,help='plot vertical figure',default=False)

ap.add_argument('-dir',nargs=1,help='simulations and plotting directory',
                default='/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/4U1957',type=str)

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




    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('X (log scale)')
    # plt.ylabel('Y (log scale)')
    # plt.title('Linear Regression in Log-Log Space')
    # plt.legend()
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    # plt.show()

def plot_indiv_panel(ax_use,panel_mode,show_yaxis=True,legend=True,
                     base_dir='/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/',
                     subdir='auto',secax=None,
                     display_lims=True,line_csv='line_table_AO2.csv',obs_csv='observ_table_AO2.csv',
                     #custom commands for non-auto subdir
                     man_label=None,man_label_plot=None,man_marker=None,man_ls=None,man_lw=1,man_color=None,
                     label_constraints=True,lock_lims=False):



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

        plt.savefig('/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO1/EW_detec_compa'+
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

        plt.savefig('/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO1/bshift_err_compa'+
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


def compa_plots_XMM_A025():
    #handmade version for XMM AO25

    mpl.rcParams.update({'font.size': 14})


    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/Propal/XMM/AO25/mid-i/simu/plot_NH')

    plt.figure(figsize=(4.5,4),layout='constrained')

    ax=plt.gca()

    load_NH_RGS=np.loadtxt('photo_nh_lim_mod_fit_hard_2e21_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    load_NH_XRISM=np.loadtxt('photo_nh_lim_mod_mod_bs_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'NH threshold for 3$\sigma$ detection (cm$^{-2}$)')
    plt.ylabel(r'Observed Flux (erg s$^{-1}$ cm$^{-2}$)')

    loglog_regressor(load_NH_XRISM.T[-1]*1e22, load_NH_XRISM.T[0], ax=plt.gca(), color='orange', marker='+',ls='--',label_plot='XRISM hot wind')

    interp_vals=loglog_regressor(load_NH_RGS.T[-1][:-1]*1e22, load_NH_RGS.T[0][:-1], ax=plt.gca(), color='blue', marker='+',ls='--',label_plot='RGS warm wind',return_vals=True)

    #final scatter for the systematics dominated zone
    stat_y=load_NH_RGS.T[0][-2:]
    stat_x = [interp_vals[0][-1],interp_vals[0][-1]]

    ax.scatter(stat_x[-1],stat_y[-1],marker='+', color='blue', s=100)


    ax.plot(stat_x,stat_y,ls='--', color='blue', lw=1, alpha=1., label='')

    mpl.rcParams.update({'font.size': 10})

    plt.legend()
    plt.savefig('fig_linedet_NH.pdf')
    plt.close()



    ##VSHIFT ERROR
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/Propal/XMM/AO25/mid-i/simu/plot_verr')

    mpl.rcParams.update({'font.size': 14})
    plt.figure(figsize=(4.5,4),layout='constrained')

    ax=plt.gca()

    load_verr_RGS=np.loadtxt('photo_vshift_err_mod_fit_hard_2e21_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    load_verr_XRISM=np.loadtxt('photo_vshift_err_mod_mod_bs_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'3$\sigma$ velocity shift error (km s$^{-1}$)')
    plt.ylabel(r'Observed Flux (erg s$^{-1}$ cm$^{-2}$)')

    loglog_regressor(load_verr_XRISM.T[-1], load_verr_RGS.T[0], ax=plt.gca(), color='orange', marker='+',ls='--',label_plot='XRISM hot wind')

    loglog_regressor(load_verr_RGS.T[-1], load_verr_RGS.T[0], ax=plt.gca(), color='blue', marker='+',ls='--',label_plot='RGS warm wind')
    mpl.rcParams.update({'font.size': 10})

    plt.legend()
    plt.savefig('fig_linedet_verr.pdf')
    plt.close()


def compa_plots_SQUDE_XRB():
    #handmade version for SQUDE XRB proposal

    mpl.rcParams.update({'font.size': 14})


    os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_SQUDE_propal/plot_NH')

    plt.figure(figsize=(4.5,4),layout='constrained')

    ax=plt.gca()

    load_NH_RGS=np.loadtxt('RGS_photo_nh_lim_mod_fit_hard_2e21_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    load_NH_XRISM=np.loadtxt('XRISM_photo_nh_lim_mod_mod_bs_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    load_NH_SQUDE_hot=np.loadtxt('SQUDE_hot_photo_nh_lim_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_0.1_1000_9_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')
    load_NH_SQUDE_warm=np.loadtxt('SQUDE_warm_photo_nh_lim_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_0.1_1000_9_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'NH threshold for 3$\sigma$ detection (cm$^{-2}$)')
    plt.ylabel(r'Observed Flux (erg s$^{-1}$ cm$^{-2}$)')

    plt.axvline(1e21, color='grey', lw=0.5)
    plt.axvline(1e22, color='grey', lw=0.5)
    plt.axvline(1e23, color='grey', lw=0.5)



    #for SQUDE warm
    loglog_regressor(load_NH_SQUDE_warm.T[-1]*1e22, load_NH_SQUDE_warm.T[0],sampl_Y=load_NH_XRISM.T[0],
                     ax=plt.gca(), color='cyan',
                     marker='+',ls='--',label_plot='SQUDE\nwarm')

    #for SQUDE hot we do the regression only from the brightest point with NH=10 and upwards
    loglog_regressor(load_NH_SQUDE_hot.T[-1][5:]*1e22, load_NH_SQUDE_hot.T[0][5:],sampl_Y=load_NH_XRISM.T[0],
                     ax=plt.gca(), color='orange',
                     marker='+',ls='--',label_plot='SQUDE\nhot')

    #for XRISM
    loglog_regressor(load_NH_XRISM.T[-1]*1e22, load_NH_XRISM.T[0], ax=plt.gca(), color='red',
                     marker='+',ls='--',label_plot='XRISM\nhot')


    interp_vals_RGS=loglog_regressor(load_NH_RGS.T[-1][:-1]*1e22, load_NH_RGS.T[0][:-1], ax=plt.gca(),
                                 color='blue', marker='+',ls='--',label_plot='RGS\nwarm',return_vals=True)

    #final scatter for the RGS systematics dominated zone
    stat_y=load_NH_RGS.T[0][-2:]
    stat_x = [interp_vals_RGS[0][-1],interp_vals_RGS[0][-1]]

    ax.scatter(stat_x[-1],stat_y[-1],marker='+', color='blue', s=100)
    ax.plot(stat_x,stat_y,ls='--', color='blue', lw=1, alpha=1., label='')

    # mpl.rcParams.update({'font.size': 8})
    # plt.legend()
    # plt.xlim(1e20,6.8e23)
    plt.savefig('fig_linedet_NH_SQUDE.pdf')
    plt.savefig('fig_linedet_NH_SQUDE.png',dpi=300)
    plt.close()



    ##VSHIFT ERROR
    os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/simu_SQUDE_propal/plot_verr')

    mpl.rcParams.update({'font.size': 14})
    plt.figure(figsize=(4.5,4),layout='constrained')

    ax=plt.gca()

    load_verr_RGS=np.loadtxt('RGS_photo_vshift_err_mod_fit_hard_2e21_50ks_20_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    load_verr_XRISM=np.loadtxt('XRISM_1e23_photo_vshift_err_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')

    load_verr_SQUDE_hot=np.loadtxt('SQUDE_hot_photo_vshift_err_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')
    load_verr_SQUDE_warm=np.loadtxt('SQUDE_warm_photo_vshift_err_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_1_100_5_in_2p0_10p0_keV_mod_pion_abs_NS_0.005.txt')


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'3$\sigma$ velocity shift error (km s$^{-1}$)')
    plt.ylabel(r'Observed Flux (erg s$^{-1}$ cm$^{-2}$)')

    #for SQUDE warm
    loglog_regressor(load_verr_SQUDE_warm.T[-1], load_verr_SQUDE_warm.T[0],
                     ax=plt.gca(), color='cyan',
                     marker='+',ls='--',label_plot='SQUDE\nwarm')

    #for SQUDE hot we do the regression only from the brightest point with NH=10 and upwards
    loglog_regressor(load_verr_SQUDE_hot.T[-1][2:], load_verr_SQUDE_hot.T[0][2:],
                     ax=plt.gca(), color='orange',
                     marker='+',ls='--',label_plot='SQUDE\nhot')

    loglog_regressor(load_verr_XRISM.T[-1], load_verr_RGS.T[0], ax=plt.gca(), color='red', marker='+',ls='--',label_plot='XRISM\nhot')

    loglog_regressor(load_verr_RGS.T[-1], load_verr_RGS.T[0], ax=plt.gca(), color='blue', marker='+',ls='--',label_plot='RGS\nwarm')
    mpl.rcParams.update({'font.size': 9})
    plt.legend()

    plt.savefig('fig_linedet_verr_SQUDE.pdf')
    plt.savefig('fig_linedet_verr_SQUDE.png',dpi=300)

    plt.close()



def colored_line(x, y, c, ax, **lc_kwargs):

    """
    from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def Q_shape_col(val_csv_warm=None,val_csv_hot=None,
                factor_soft=2,
                col_warm='magma_r',
                col_hot='viridis_r',
                d_kpc=8,M_CO=8,
                ledd_lowbranch=0.01,ledd_highbranch=0.1,lw=10,
                transi_width_margin=0.0065,
                figsize=(6,4)):

    '''
    Computes the Q-shape evolution of a given parameter according to a sampling of values for a given flux

    val_csv_warm: csv containing the values of the constraints with warm absorber with the hard state SED,
    val_csv_hot: csv containing the values of the constraints with a hot absorber with a hard state SED,

    factor_soft: flux multiplier for soft state SEDs. Used to draw the vertical line of the soft state and
                the horizontal lines linking the two together.
    col: colormap
    d_kpc,M_CO: used to normalize the position of the Q-shape

    example:
    Q_shape_col('photo_nh_lim_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_0.1_1000_9_in_2p0_10p0_keV_mod_pion_abs_NS_0.005_warm.txt',
    'photo_nh_lim_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_0.1_1000_9_in_2p0_10p0_keV_mod_pion_abs_NS_0.005_hot.txt',
    figsize=(6,4))

    '''


    val_warm=np.loadtxt(val_csv_warm)
    val_3sig_warm=val_warm.T[-1]
    val_flux_warm=val_warm.T[0]

    if val_csv_hot is not None:
        val_hot=np.loadtxt(val_csv_hot)
        val_3sig_hot=val_hot.T[-1]
        val_flux_hot=val_hot.T[0]


    edd_conv=edd_factor(d_kpc,M_CO).value
    # Perform linear regression in log-log space with y in eddington units
    logslope_warm, logintercept_warm, r_value, p_value, std_err = linregress(np.log10(val_flux_warm*edd_conv),
                                                                             np.log10(val_3sig_warm))

    #defining the y axis flux range for the warm hard state
    eddrange_warm_hard=np.logspace(-5,np.log10(ledd_highbranch),200)

    val_pred_warm_hard=10**(logslope_warm*np.log10(eddrange_warm_hard)+logintercept_warm)


    #defining the y axis flux range for the warm soft state
    eddrange_warm_soft=np.logspace(np.log10(ledd_lowbranch),np.log10(ledd_highbranch),100)

    #and computing the predicted limits considering  the soft factor
    val_pred_warm_soft=10**(logslope_warm*np.log10(eddrange_warm_soft*factor_soft)+logintercept_warm)


    #defining the x axis hr range for the state transitions
    hr_warm_transi=np.logspace(-1-transi_width_margin,0.+transi_width_margin,100)
    val_pred_warm_transi_high=10**(logslope_warm*\
                            #normalizing for constant logarithmic step to *factor soft in soft and *1 in hard
                            np.log10((abs(np.log10(hr_warm_transi))*(factor_soft-1)+1)*ledd_highbranch)
                            +logintercept_warm)

    val_pred_warm_transi_low=10**(logslope_warm*\
                            #normalizing for constant logarithmic step to *factor soft in soft and *1 in hard
                            np.log10((abs(np.log10(hr_warm_transi))*(factor_soft-1)+1)*ledd_lowbranch)
                            +logintercept_warm)

    vrange_warm=np.log10(np.array([min(val_pred_warm_hard.tolist()+val_pred_warm_soft.tolist()),
                 max(val_pred_warm_hard.tolist() + val_pred_warm_soft.tolist())]))


    '''
    #hot
    '''
    if val_csv_hot is not None:
        logslope_hot, logintercept_hot, r_value, p_value, std_err = linregress(np.log10(val_flux_hot*edd_conv),
                                                                                 np.log10(val_3sig_hot))

        #defining the y axis flux range for the hot hard state
        eddrange_hot_hard=np.logspace(-5,np.log10(ledd_highbranch),200)

        val_pred_hot_hard=10**(logslope_hot*np.log10(eddrange_hot_hard)+logintercept_hot)


        #defining the y axis flux range for the hot soft state
        eddrange_hot_soft=np.logspace(np.log10(ledd_lowbranch),np.log10(ledd_highbranch),100)

        #and computing the predicted limits considering  the soft factor
        val_pred_hot_soft=10**(logslope_hot*np.log10(eddrange_hot_soft*factor_soft)+logintercept_hot)


        #defining the x axis hr range for the state transitions
        hr_hot_transi=np.logspace(-1-transi_width_margin,0.+transi_width_margin,100)
        val_pred_hot_transi_high=10**(logslope_hot*\
                                #normalizing for constant logarithmic step to *factor soft in soft and *1 in hard
                                np.log10((abs(np.log10(hr_hot_transi))*(factor_soft-1)+1)*ledd_highbranch)
                                +logintercept_hot)

        val_pred_hot_transi_low=10**(logslope_hot*\
                                #normalizing for constant logarithmic step to *factor soft in soft and *1 in hard
                                np.log10((abs(np.log10(hr_hot_transi))*(factor_soft-1)+1)*ledd_lowbranch)
                                +logintercept_hot)


    # fig,ax= plt.subplots(1, 3, figsize=figsize, width_ratios=[1,19, 1])
    fig,ax= plt.subplots(figsize=figsize)

    ax.set_xscale('log')
    ax.set_xlabel(r'Spectral Hardness ([6-10]/[3-6] keV flux)')
    ax.set_xlim(0.05,2)

    ax.set_yscale('log')
    ax.set_ylabel(r'Luminosity ($L/L_{Edd}$)')
    ax.set_ylim(1e-5,1)


    combined_x_warm=np.repeat(1,len(val_pred_warm_hard)).tolist()+hr_warm_transi.tolist()[::-1]+\
               np.repeat(0.1,len(val_pred_warm_soft)).tolist()[::-1]+hr_warm_transi.tolist()

    combined_y_warm=eddrange_warm_hard.tolist()+np.repeat(ledd_highbranch,len(hr_warm_transi)).tolist()[::-1]+\
               eddrange_warm_soft.tolist()[::-1]+np.repeat(ledd_lowbranch,len(hr_warm_transi)).tolist()

    combined_c_warm=np.log10(val_pred_warm_hard).tolist()+np.log10(val_pred_warm_transi_high).tolist()[::-1]+ \
               np.log10(val_pred_warm_soft).tolist()[::-1]+np.log10(val_pred_warm_transi_low).tolist()

    mask_use_warm = np.array(combined_c_warm) < 0
    vrange_warm=np.log10(np.array([min(10**np.array(combined_c_warm)[mask_use_warm]),
                                  max(10**np.array(combined_c_warm)[mask_use_warm])])).round(1)

    lines_warm=colored_line(np.array(combined_x_warm)[mask_use_warm],
                           np.array(combined_y_warm)[mask_use_warm],
                           np.array(combined_c_warm)[mask_use_warm],
                           ax=ax,cmap=col_warm,lw=lw,clim=vrange_warm)

    # lines_warm=colored_line(combined_x_warm,
    #                         combined_y_warm,
    #                         combined_c_warm,ax=ax,cmap=col_warm,lw=lw,clim=vrange_warm)

    if val_csv_hot is not None:

        margin_x_hot=0.22
        combined_x_hot=np.repeat(1*(1+margin_x_hot),len(val_pred_hot_hard)).tolist()+\
                        [1+margin_x_hot]+\
                        hr_hot_transi.tolist()[::-1]+\
                        [0.1/(1+margin_x_hot)] + \
                        np.repeat(0.1/(1+margin_x_hot),len(val_pred_hot_soft)).tolist()[::-1]+\
                        hr_hot_transi.tolist()

        combined_y_hot=(1.*eddrange_hot_hard).tolist()+\
                        [ledd_highbranch*1.8]+\
                        np.repeat(ledd_highbranch*1.8,len(hr_hot_transi)).tolist()[::-1] +\
                        [ledd_highbranch * 1.8] +\
                        ((1.)*eddrange_hot_soft).tolist()[::-1]+\
                       np.repeat(ledd_lowbranch/2,len(hr_hot_transi)).tolist()

        combined_c_hot=(np.log10(val_pred_hot_hard).tolist()+
                        [np.log10(val_pred_hot_hard[-1])]+
                        np.log10(val_pred_hot_transi_high).tolist()[::-1]+
                        [np.log10(val_pred_hot_soft[-1])]+\
                   np.log10(val_pred_hot_soft).tolist()[::-1]+np.log10(val_pred_hot_transi_low).tolist())



        mask_use_hot = np.array(combined_c_hot) < 1

        # breakpoint()


        vrange_hot=np.log10(np.array([min(10**np.array(combined_c_hot)[mask_use_hot]),
                                      max(10**np.array(combined_c_hot)[mask_use_hot])])).round(1)

        lines_hot=colored_line(np.array(combined_x_hot)[mask_use_hot],
                               np.array(combined_y_hot)[mask_use_hot],
                               np.array(combined_c_hot)[mask_use_hot],
                               ax=ax,cmap=col_hot,lw=lw,clim=vrange_hot)


        cbar_hot=fig.colorbar(lines_hot,location='left',pad=0.16)
        cbar_hot.ax.invert_yaxis()
        cbar_hot_ticks_adj=['%.1e'%elem for elem in 10 ** (22 + cbar_hot.get_ticks())]
        cbar_hot.set_ticklabels(cbar_hot_ticks_adj)
        cbar_hot.set_label('hot wind \n NH',y=1.,rotation=0,labelpad=-55)
        cbar_hot.ax.yaxis.set_ticks_position('left')


    # cbar_warm=fig.colorbar(val_warm_hard)
    cbar_warm=fig.colorbar(lines_warm,pad=0.04,)
    cbar_warm.ax.invert_yaxis()
    cbar_warm_ticks_adj=['%.1e'%elem for elem in 10 ** (22 + cbar_warm.get_ticks())]
    cbar_warm.set_ticklabels(cbar_warm_ticks_adj)
    cbar_warm.set_label('warm wind \n NH',y=1.1,rotation=0,labelpad=-57)

    plt.tight_layout()



    # plt.tight_layout()
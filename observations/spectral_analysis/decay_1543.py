import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import pandas as pd
from general_tools import combo_legend
import matplotlib.dates as mdates
vals_counts_Swift=[
    [],
    [],
    [],
    [],
    []
]

#pulled from the archive
rxte_dates_str=\
'''
2011-05-10 02:04:04.2
2011-05-11 01:48:18.1
2011-05-12 01:29:29.8
2011-05-13 02:11:25.7
2011-05-13 21:00:07.8
2011-05-14 15:51:17.6
2011-05-15 15:23:02.4
2011-05-17 00:26:54.8
2011-05-18 04:37:44.8
2011-05-19 01:04:35.9
2011-05-19 05:42:44.9
2011-05-20 20:55:24.4
2011-05-21 18:48:42
2011-05-22 23:06:00
2011-05-24 01:38:07.3
2011-05-24 06:22:44.3
2011-05-25 07:30:45.8
2011-05-26 14:50:41.9
2011-05-27 01:41:30.3
2011-05-27 09:39:04.9
2011-05-28 12:16:44
2011-05-29 16:31:21.2
2011-05-30 14:24:51
2011-05-31 10:41:20
2011-06-02 14:20:36.1
2011-06-03 12:24:15.8
2011-06-04 15:00:07.8
2011-06-05 17:59:58.3
2011-06-06 17:29:00.7
2011-06-07 08:47:43.9
2011-06-08 14:34:09.1
2011-06-09 08:16:38.5
2011-06-11 06:45:49.2
2011-06-12 12:34:29.3
2011-06-13 07:21:30.2
2011-06-14 09:58:57.2
2011-06-15 20:55:01.9
2011-06-17 16:49:03.1
2011-06-18 07:58:55.8
2011-06-18 22:46:04.2
2011-06-19 01:53:43.9
2011-06-20 04:26:11.9
2011-06-21 10:02:28.9
2011-06-22 08:29:10.2
2011-06-23 04:48:38.9
2011-06-24 08:04:24.1
2011-06-25 05:54:06.6
2011-06-26 04:01:23.2
2011-06-27 03:46:25.5
2011-06-28 05:59:17.7
2011-06-29 05:32:38.4
2011-06-30 20:48:52.1
2011-07-01 17:25:10.8
2011-07-02 18:20:28.6
2011-07-03 21:20:17.4
2011-07-05 15:33:48.7
2011-07-06 15:59:16.2
2011-07-07 21:28:17.8
2011-07-09 04:06:41.2
2011-07-10 02:10:31.3
2011-07-10 20:54:15.3
2011-07-12 04:12:23.3
2011-07-14 03:18:38.9
2011-07-15 09:37:29.9
2011-07-16 06:55:40.2
2011-07-17 03:18:05.2
2011-07-18 12:11:09.6
2011-07-19 08:33:31.1
2011-07-20 06:56:37.2
2011-07-21 10:39:56.2
2011-07-22 07:02:28.9
2011-07-23 08:49:28.4
2011-07-24 02:47:37.8
2011-07-26 08:09:14.4
2011-07-28 02:37:39.1
2011-07-29 05:05:54.8
2011-07-30 06:10:50.6
2011-07-31 15:31:19.2
2011-08-01 01:11:39.3
2011-08-02 06:08:53.1
2011-08-04 02:38:34.4
2011-08-04 20:46:40.8
2011-08-06 01:48:01.7
2011-08-08 19:23:43.3
2011-08-10 23:00:34.3
2011-08-12 17:11:54.2
2011-08-15 23:10:39.1
2011-08-18 23:08:21.7
2011-08-19 00:00:06.9
2011-08-21 05:18:39.5
2011-08-23 08:17:40.7
2011-08-25 01:23:36.4
2011-08-27 09:36:13.8
2011-08-29 21:22:56.4
2011-08-30 17:41:41.9
2011-09-03 03:12:42
2011-09-06 18:37:26.4
2011-09-14 02:32:46.2
2011-09-22 15:47:47.6
2011-09-30 00:04:28.7'''

#first one is from NuSTAR, rest is from Swift
monit_rxte_counts_str=\
'''
2025-12-26
44.1
38.0
32.7
21.3
8.7
10.4
2026-01-04
55.3
15.7
42.2
7.7
9.5
4.9
2026-01-07
57.7
9.3
43.8
4.5
10.2
2.9
2026-01-09
58.5
11
44.2
5.4
10.6
3.5
2026-01-10
64.4
19.8
50.2
9.7
9.9
6.3
2026-01-11
46.7
25.1
36.2
12.3
7.3
7.9
2026-01-12
51
22.8
38.3
11.2
9.5
7.2
2026-01-17
60.1
7.9
45.7
3.9
10.6
2.5
2026-01-19
67.7
1.7
49.7
0.8
13.9
0.5
2026-01-20
68.5
0
50.6
0
13.8
0
2026-01-21
63.8
24
48.9
11.7
10.8
7.6
'''

monit_rxte_counts=np.array(monit_rxte_counts_str.split('\n')[1:-1]).reshape(11,7).T



rxte_times=Time(rxte_dates_str.split('\n')[1:],format='iso')

def plot_stiele():

    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/Swift/MAXIJ1543-564')
    vals_2011=pd.read_csv('RXTE_compa_2011_stiele.csv',header=None)

    plt.figure(figsize=(8,6))
    ax=plt.gca()
    plt.yscale('log')
    plt.ylim(5,180)
    plt.xlim(0.08,0.8)
    plt.xlabel('Hardness Ratio [5.7-9.5]/[2.9-5.7]keV')
    plt.ylabel('Count Rate in [2-15] keV')
    c_val_norm=((rxte_times-rxte_times[0]).value)
    c_val_norm_20weeks=((rxte_times-rxte_times[0]).value)%140
    c_val_norm_10weeks=((rxte_times-rxte_times[0]).value)%70

    plt.plot(vals_2011[0],vals_2011[1],color='grey',alpha=0.5/3,label='days 0-70')
    test=plt.scatter(vals_2011[0][c_val_norm<70],vals_2011[1][c_val_norm<70],
                marker='+',c=c_val_norm_10weeks[c_val_norm<70],s=100,vmin=0,vmax=70,cmap='tab10',
                label='days 0-70',alpha=0.5)


    plt.plot(vals_2011[0],vals_2011[1],color='grey',alpha=0.5/3,label='days 70-140')
    plt.scatter(vals_2011[0][c_val_norm>70][:-1],vals_2011[1][c_val_norm>70][:-1],
                marker='x',c=c_val_norm_10weeks[c_val_norm>70][:-1],s=100,vmin=0,vmax=70,cmap='tab10',
                label='days 70-140',alpha=0.5)

    plt.plot(vals_2011[0],vals_2011[1],color='grey',alpha=0.5/3,label='day 143')
    plt.scatter(vals_2011[0][c_val_norm>140],vals_2011[1][c_val_norm>140],
                marker='d',c=c_val_norm_10weeks[c_val_norm>140],s=100,vmin=0,vmax=70,cmap='tab10',
                label='day 143',alpha=0.5)

    main_leg_hd, main_leg_labli = combo_legend(plt.gca())

    ax_monit=plt.twinx()
    ax_monit.set_ylim(5,180)
    ax_monit.set_xlim(0.08,0.8)
    ax_monit.set_yscale('log')
    ax_monit.yaxis.set_visible(False)

    ticklabels=np.arange(0, 71, 7).astype(str)+'/'+np.arange(70, 141, 7).astype(str)
    ticklabels[0]+='/140'
    ticklabels[1]+='/147'
    monit_counts=monit_rxte_counts[1:3].astype(float).sum(0)
    monit_HR=monit_rxte_counts[5:].astype(float).sum(0)/monit_rxte_counts[3:5].astype(float).sum(0)
    monit_c_val=(Time(monit_rxte_counts[0])-Time(monit_rxte_counts[0][0])).value

    ax_monit.plot(monit_HR,monit_counts,color='red',label='Swift')
    ax_monit.plot([],[],color='red',label='NuSTAR')

    ax_monit.scatter(monit_HR,monit_counts,c=monit_c_val,marker='o',s=70,vmin=0,vmax=70,cmap='tab10',alpha=1,zorder=1000,
                label='NuSTAR')

    ax_monit.scatter(monit_HR[0],monit_counts[0],
                     marker='+',s=100,color='black',alpha=1,zorder=2000,
                label='NuSTAR')

    ax_monit.scatter(monit_HR,monit_counts,c=monit_c_val,marker='o',s=70,vmin=0,vmax=70,cmap='tab10',alpha=1,zorder=1000,
                label='Swift')

    monit_leg_hd, monit_leg_labli = combo_legend(ax_monit)

    monit_order=np.array(monit_leg_labli).argsort()
    monit_leg_hd=np.array(monit_leg_hd,dtype=object)[monit_order]
    monit_leg_labli=np.array(monit_leg_labli)[monit_order]

    main_order=np.array(main_leg_labli).argsort()
    main_leg_hd=np.array(main_leg_hd,dtype=object)[main_order][[1,2,0]].tolist()
    main_leg_labli=np.array(main_leg_labli)[main_order][[1,2,0]]

    main_leg_hd=[tuple(elem) for elem in main_leg_hd]

    ax.legend(main_leg_hd,main_leg_labli,loc='upper right',title='2011')
    ax_monit.legend(monit_leg_hd,monit_leg_labli,loc='upper left',title='2026')


    plt.suptitle('evolution of MAXI J1543 in a Hardness Intensity Diagram in          and  ')
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.text(x=0.755,y=213,s='2011',color='grey',weight='bold')
    plt.text(x=0.87,y=213,s='2026',color='red',weight='bold')

    caxspe = plt.axes([0.88, 0.095, 0.04, 0.825])
    caxspe2 = plt.axes([0.88, 0.095, 0.04, 0.825])

    cbar=plt.colorbar(test,cax=caxspe,ticks=np.arange(0,71,7))
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.set_title('days since \n first observation',fontsize=10,y=-0.09)

    cbar_right=plt.colorbar(test,cax=caxspe2,ticks=np.arange(0,71,7),)
    cbar_right.ax.yaxis.set_ticks_position('right')
    cbar_right.ax.set_yticklabels(np.arange(70, 141, 7).astype(str))

    plt.savefig('fig_RXTE.pdf')
    plt.savefig('fig_RXTE.png')


def loader_mod_for_flux():
    Xset.restore('mod_disk_pow_2p3.xcm')
    AllData.ignore('**-1.')
    Fit.perform()
    AllModels(1)(6).values=0.
    AllModels.calcFlux("1. 10.")
    Fit.perform()
    AllModels(1)(8).values=0.
    AllModels.calcFlux("1. 10.")


def plot_bs():
    xrism_ls_loader('mod_em6keV_noabsline_narrow_out_4.0_10.0.dill', [4., 10.], force_ylog_ratio=False,
                    ratio_bounds=[0.8, 1.2], squished_mode=False, show_indiv_transi=False, force_side_lines='abs')

def fake_bright_endtransi_FeKaXXVI():
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_baseline_XRISM_endtransi_FeKaXXVIduet.xcm')
    settings=FakeitSettings(response='rsl_Hp_L_2025.rmf',arf='rsl_pntsrc_GVC_2025.arf',
                            exposure=5e4,fileName='test_50k_endtransi.pi')
    AllData.fakeit(settings=settings,applyStats=True)

def fake_bright_endtransi_pion1e23():
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_pion_1e23_8p5e-10.xcm')
    settings=FakeitSettings(response='rsl_Hp_L_2025.rmf',arf='rsl_pntsrc_GVC_2025.arf',
                            exposure=5e4,fileName='test_50k_endtransi_pion_1e23.pi')
    AllData.fakeit(settings=settings,applyStats=True)

def fake_bright_endtransi_pion2e23():
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_pion_2p5e23_narrow_8p5e-10.xcm')
    settings=FakeitSettings(response='rsl_Hp_L_2025.rmf',arf='rsl_pntsrc_GVC_2025.arf',
                            exposure=5e4,fileName='test_50k_endtransi_pion_2p5e23.pi')
    AllData.fakeit(settings=settings,applyStats=True)

def make_plot_spectrum():

    #bright
    rebinv_xrism(1, 7)

    #mid
    rebinv_xrism(2, 4)

    #faint
    rebinv_xrism(3, 3)

    AllData.notice('all')
    AllData.ignore('**-6.55 8.5-**')
    xPlot('ldata')
    plot_std_ener(plt.gca(),plot_indiv_transi=True,mode='',alpha_line=0.3,noname=True)
    plot_std_ener(plt.gca(),plot_indiv_transi=False,mode='misc',alpha_line=0.5,noline=True)
    plt.text(7.645,0.67,r'NiXXVII K$\alpha$',color='blue')

def make_plot_spectrum_all_highE():

    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_compa_all3.xcm')
    #bright
    rebinv_xrism(1, 7)

    #mid
    rebinv_xrism(2, 4)

    #faint
    rebinv_xrism(3, 3)

    Plot.add=False
    AllData.notice('all')
    AllData.ignore('**-6.55 8.5-**')
    xPlot('ldata', group_names=[r'50ks - 35mCrab - log$\xi$=5.2', r'100ks - 15mCrab - log$\xi$=4.8',
                                r'150ks - 5mCrab - log$\xi$=4.3'])
    plot_std_ener(plt.gca(), plot_indiv_transi=True, mode='', alpha_line=0.3, noname=True)
    plot_std_ener(plt.gca(), plot_indiv_transi=False, mode='misc', alpha_line=0.5, noline=True)
    plt.text(7.645, 0.7, r'NiXXVII K$\alpha$', color='blue')
    plt.suptitle('simulated XRISM spectra at 3 different flux levels using a photionized model fit from the NuSTAR spectrum')

    #dont forget to zoom the plot first
    # plt.tight_layout()
    # plt.savefig('mod_compa_all3_lines_highe.pdf')
    # plt.savefig('mod_compa_all3_lines_highe.png')

def plot_decay_HSS():
    fig_lc,ax_lc=plt.subplots()
    init_datenum = mdates.date2num(["2026-01-17"])
    # plt.yscale('log')
    plt.ylim(3,50)
    plt.xlabel('Date')
    plt.ylabel('1-10 keV absorbed flux (mCrabs)')
    xrism_window=mdates.date2num(["2026-01-20","2026-04-05"])

    #first limit for 1/3 of logarithmic decay to 5mCrab
    xrism_limit_1=mdates.date2num(["2026-02-12"])[0]

    #second limit for 2/3 of logarithmic decay to 5mCrab
    xrism_limit_2=mdates.date2num(["2026-03-09"])[0]


    plt.scatter(init_datenum,33,s=150,marker='x',color='black',label='last Swift observation \n (17-01)')

    x_decay=np.arange(100)+init_datenum
    y_decay=33*np.exp(-np.arange(100)/43)
    plt.plot(x_decay[y_decay>5],y_decay[y_decay>5],color='black',
             label='expected decay from\n 2011 E-folding time',ls=':')

    plt.axhline(5,0,1,color='purple',ls='--',label='lowest flux before\n 2011 Soft-to-Hard transition')

    #
    # plt.axvspan(xrism_window[0],xrism_window[1],0,1,color='grey',alpha=0.2,
    #             label="XRISM visibility window")
    plt.axvline(xrism_window[0],0,1,color='blue',alpha=1,label='XRISM visibility window')
    plt.axvline(xrism_window[1],0,1,color='blue',alpha=1)

    plt.axvspan(xrism_window[0],xrism_limit_1,0,1,color='black',alpha=0.4,
                label="first third of flux decay")

    plt.axvspan(xrism_limit_1,xrism_limit_2,0,1,color='red',alpha=0.4,
                label="second third of flux decay")

    plt.axvspan(xrism_limit_2,xrism_window[1],0,1,color='green',alpha=0.4,
                label="second third of flux decay")


    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax_lc.xaxis.set_major_formatter(date_format)
    ax_lc.set_xticks(init_datenum + np.arange(0, 100, 7))
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.1))
    for label in ax_lc.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='center')

    #convenient x axis limits
    plt.xlim(mdates.date2num(["2026-01-12"])[0] - 5, mdates.date2num(["2026-04-07"])[0])

    #zoom to fullscreen
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/Swift/MAXIJ1543-564')
    # plt.savefig('lc_decay.pdf')
    # plt.savefig('lc_decay.png')


def make_plot_spectrum_all_lowE():

    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_compa_all3.xcm')

    #bright
    rebinv_xrism(1, 15)

    #mid
    rebinv_xrism(2, 15)

    #faint
    rebinv_xrism(3, 15)

    Plot.add=False
    AllData.notice('all')
    AllData.ignore('**-2.4 5.-**')
    xPlot('ldata', group_names=[r'50ks - 35mCrab - log$\xi$=5.2', r'100ks - 15mCrab - log$\xi$=4.8',
                                r'150ks - 5mCrab - log$\xi$=4.3'],ylims=[9e-2,5])
    plot_std_ener(plt.gca(), plot_indiv_transi=False, mode='', alpha_line=0.3, noname=True)
    plot_std_ener(plt.gca(), plot_indiv_transi=False, mode='misc', alpha_line=0.5, noline=True)
    plt.suptitle('simulated XRISM spectra at 3 different flux levels using a photionized model fit from the NuSTAR spectrum')

    # plt.tight_layout()
    # plt.savefig('mod_compa_all3_lines_lowe.pdf')
    # plt.savefig('mod_compa_all3_lines_lowe.png')

def fake_faint_softfloor_pion2e23():
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_pion_2p5e23_narrow_1p2e-10_adaptedlogxi.xcm')
    settings=FakeitSettings(response='rsl_Hp_L_2025.rmf',arf='rsl_pntsrc_GVC_2025.arf',
                            exposure=1.5e5,fileName='test_150k_softfloor_pion_2p5e23_adaptedlogxi.pi')
    AllData.fakeit(settings=settings,applyStats=True)

def fake_faint_softmid_pion2e23():
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/simu_DDT')
    Xset.restore('mod_pion_2p5e23_narrow_3p6e-10_adaptedlogxi.xcm')
    settings=FakeitSettings(response='rsl_Hp_L_2025.rmf',arf='rsl_pntsrc_GVC_2025.arf',
                            exposure=1e5,fileName='test_100k_softfmid_pion_2p5e23_adaptedlogxi.pi')
    AllData.fakeit(settings=settings,applyStats=True)
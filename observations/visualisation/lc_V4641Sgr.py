#general imports
import os,sys
import glob

import argparse
import warnings


import numpy as np
import pandas as pd
from decimal import Decimal

import streamlit as st
#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple
from matplotlib.gridspec import GridSpec

from matplotlib.lines import Line2D
from matplotlib.ticker import Locator,MaxNLocator,AutoMinorLocator
from scipy.stats import norm as scinorm

import matplotlib.dates as mdates

#pickle for the rxte lightcurve dictionnary
import pickle

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astroquery.simbad import Simbad

from copy import deepcopy

#needed to fetch the links to the lightcurves for MAXI data
import requests
from bs4 import BeautifulSoup

#correlation values and trend plots with MC distribution from the uncertainties
from custom_pymccorrelation import pymccorrelation,perturb_values
from lmplot_uncert import lmplot_uncert_a

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their
#'perturb values' function

from ast import literal_eval
# import time

lc_dir='/home/parrama/Documents/Work/PhD/docs/Observations/V4641Sgr/'

MAXI_psf_csv=pd.read_csv(os.path.join(lc_dir,'MAXI_megumi/v4640_241111/gsc_final','V4641_Sgr_1d.qdp'),
                              skiprows=7,
                              sep=' ',names=['MJD','fit','f24','f24err-','f24err+','f410','f410err-','f410err+'])

MAXI_od_csv=pd.read_csv(os.path.join(lc_dir,'MAXI_megumi','v4641sgr_2.0-10.0keV_gsclc_ondemand.dat'),skiprows=1,
                                      names=['MJD_start','MJD_end','counts','counts_err'],sep=' ')

NICER_dates_csv=pd.read_csv(os.path.join(lc_dir,'NICER','NICER_epoch_bounds.txt'),skiprows=1,
                            names=['tstart','tstop','mjdstart','mjdstop'],sep='\t')

NICER_mjds=Time(NICER_dates_csv['mjdstart'],format='mjd')
NICER_mjde=Time(NICER_dates_csv['mjdstop'],format='mjd')

NICER_mjd_err=(NICER_mjde-NICER_mjds)/2
NICER_mjd=NICER_mjds+NICER_mjd_err
NICER_dates=mdates.date2num(NICER_mjd.datetime)
NICER_dateserr=NICER_mjd_err.to_value('jd')

NICER_flux_csv=pd.read_csv(os.path.join(lc_dir,'NICER','NICER_line_values_4_10_0.02_0.01_10_500.txt'),sep='\t')

NICER_flux_arr=np.array([literal_eval(elem) for elem in NICER_flux_csv['broad_flux']])

#assuming an HR of 0.5 here
MAXI_od_210_f=MAXI_od_csv['counts']*1/(1.065+1.172)*2*2.4e-8

#errors, adding an average systematics
MAXI_od_210_err=MAXI_od_csv['counts_err']*1/(1.065+1.172)*(1.1+1.03)/2*2*2.4e-8

#dates
MAXI_od_ds=Time(MAXI_od_csv.MJD_start,format='mjd')
MAXI_od_de=Time(MAXI_od_csv.MJD_end,format='mjd')

MAXI_od_mjd_err=(MAXI_od_de-MAXI_od_ds)/2

MAXI_od_mjd=(MAXI_od_ds+MAXI_od_mjd_err)

MAXI_od_dates=mdates.date2num(MAXI_od_mjd.datetime)

MAXI_od_daterr=MAXI_od_mjd_err.to_value('jd')


#values for psf
crab_conv_24=1/1.065*(2.4e-8)
crab_conv_410=1/1.172*(2.4e-8)

MAXI_psf_210_f=MAXI_psf_csv['f24']*crab_conv_24+MAXI_psf_csv['f410']*crab_conv_410

#using the systematics as a multiplicator otherwise this doesn't make sense

MAXI_psf_210_errl=np.sqrt((MAXI_psf_csv['f24err-']*crab_conv_24)**2*1.165/1.065+
                          (MAXI_psf_csv['f410err-']*crab_conv_410)**2*1.203/1.72)
                          # #2-4 systematics
                          # (0.1*2.4e-8)**2+
                          #   #4-10 systematics
                          # (0.03*2.4e-8)**2)

MAXI_psf_210_errh=np.sqrt((MAXI_psf_csv['f24err+']*crab_conv_24)**2*1.1+
                          (MAXI_psf_csv['f410err+']*crab_conv_410)**2*1.03)
                          # #2-4 systematics
                          # (0.1*2.4e-8)**2+
                          #   #4-10 systematics
                          # (0.03*2.4e-8)**2)

MAXI_psf_210_err=np.array([MAXI_psf_210_errl.values,MAXI_psf_210_errh.values])



MAXI_psf_mjd=Time(MAXI_psf_csv.MJD,format='mjd')
MAXI_psf_dates=mdates.date2num(MAXI_psf_mjd.datetime)








####PLOT
date_format = mdates.DateFormatter('%Y-%m-%d')
#     date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
fig_lc,ax_lc=plt.subplots(4,1,sharex=True,figsize=(10,6))
plt.subplots_adjust(hspace=0)

ax_lc[0].xaxis.set_major_formatter(date_format)




ax_lc[0].set_ylabel('MAXI\n[2-10]keV flux\n(ergs/s/cm²)')
ax_lc[1].set_ylabel('NICER\n[3-10]keV flux\n(erg/s/cm²)')
ax_lc[2].set_ylabel('NICER\n[6-10]/[3-10]keV\nHR')

ax_lc[3].set_ylabel('Einstein Probe\n(tbd)')

ax_lc[0].errorbar(x=MAXI_psf_dates,xerr=0.5,y=MAXI_psf_210_f,yerr=MAXI_psf_210_err,linewidth=0.5,
                  elinewidth=1.,ls=':',
                  color='black',label='PSF fitting')

min_vis_psf=MAXI_psf_dates[MAXI_psf_210_f.values>0.][0]
max_vis_psf=MAXI_psf_dates[MAXI_psf_210_f.values>0.][-1]

ax_lc[0].errorbar(x=MAXI_od_dates,xerr=MAXI_od_daterr,y=MAXI_od_210_f,yerr=MAXI_od_210_err,linewidth=0.5,
                  elinewidth=1.,ls=':',
                  color='grey',alpha=0.3,label='On-demand tool')

min_vis_od=MAXI_od_dates[MAXI_od_210_f.values>0.][0]
max_vis_od=MAXI_od_dates[MAXI_od_210_f.values>0.][-1]

ax_lc[0].set_xlim(min(min_vis_psf,min_vis_od)-1,max(max_vis_psf,max_vis_od)+1)

# ax_lc[0].set_ylim(0,ax_lc[0].get_ylim()[1])
ax_lc[0].set_ylim(0.,2e-9)


#adding the XRISM obs
xrism_interval=Time(['2024-09-30 09:42:04','2024-09-30 17:03:04'])

#adding the radio obs
radio_obs_ul=Time('2024-09-29 12:57:36')
radio_obs_jet=Time('2024-10-06 12:28:48')

ax_lc_02 = ax_lc[0].twinx()
ax_lc_12 = ax_lc[1].twinx()
ax_lc_22 = ax_lc[2].twinx()
ax_lc_32 = ax_lc[3].twinx()

for i_ax,ax in enumerate([ax_lc_02,ax_lc_12,ax_lc_22,ax_lc_32]):
    ax.axvspan(mdates.date2num(xrism_interval[0].datetime),mdates.date2num(xrism_interval[1].datetime),
                     color='green',alpha=0.5,label='XRISM observation')

    ax.axvline(mdates.date2num(radio_obs_ul.datetime),color='blue',ls='-',alpha=0.5,label='Radio non-detection')
    ax.axvline(mdates.date2num(radio_obs_jet.datetime),color='red',ls='-',alpha=0.5,label='Radio detection')

    ax.yaxis.set_visible(False)
    if i_ax==3:
        ax.legend(loc='upper left')

ax_lc[0].legend(loc='upper right')

ax_lc[1].errorbar(NICER_dates,xerr=NICER_dateserr,y=NICER_flux_arr.T[4][0],yerr=NICER_flux_arr.T[4][1:],
                  linewidth=0.5,
                  elinewidth=1., ls=':',color='black')

NICER_HR=NICER_flux_arr.T[2][0]/NICER_flux_arr.T[1][0]

NICER_HR_err = np.array([((NICER_flux_arr.T[2][i] / NICER_flux_arr.T[2][0]) ** 2 + \
                        (NICER_flux_arr.T[1][i] / NICER_flux_arr.T[1][0]) ** 2) ** (1 / 2) * NICER_HR for i in [1, 2]])

ax_lc[1].errorbar(NICER_dates,xerr=NICER_dateserr,y=NICER_flux_arr.T[4][0],yerr=NICER_flux_arr.T[4][1:],
                  linewidth=0.5,
                  elinewidth=1., ls=':',color='black')

ax_lc[2].errorbar(NICER_dates,xerr=NICER_dateserr,y=NICER_HR,yerr=NICER_HR_err,
                  linewidth=0.5,
                  elinewidth=1., ls=':',color='black')

ax_lc[1].set_ylim(0,2e-9)
ax_lc[2].set_ylim(0.1,2)

# ax_lc[0].set_xlim(NICER_dates[0]-0.5,NICER_dates[-1]+0.5)
plt.show()



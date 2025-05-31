# general imports
import os, sys
import glob

import argparse
import warnings
import numpy as np
import pandas as pd
from mpdaf.obj import sexa2deg
from decimal import Decimal

import streamlit as st
# matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple
from matplotlib.gridspec import GridSpec

from matplotlib.lines import Line2D
from matplotlib.ticker import Locator, MaxNLocator, AutoMinorLocator
from scipy.stats import norm as scinorm

import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

# pickle for the rxte lightcurve dictionnary
import pickle

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astroquery.simbad import Simbad

from general_tools import interval_extract,edd_factor
from xspec_config_multisp import xPlot

from copy import deepcopy

# needed to fetch the links to the lightcurves for MAXI data
import requests
from bs4 import BeautifulSoup

# correlation values and trend plots with MC distribution from the uncertainties
from custom_pymccorrelation import pymccorrelation, perturb_values
from lmplot_uncert import lmplot_uncert_a

# Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their
# 'perturb values' function

from ast import literal_eval

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

# import time

def lc_internal_1():
    fig, ax = plt.subplots(figsize=(10, 6))
    lc = fits.open('xa901001010xtd_0p5to10keV_b128_src_lc.fits')
    lc_bg = fits.open('xa901001010xtd_0p5to10keV_b128_bgd_lc.fits')
    plt.errorbar(lc[1].data['TIME']+64, xerr=128,
                 y=lc[1].data["RATE"] - lc_bg[1].data['RATE'],
                 yerr=lc[1].data['ERROR'], marker='s', ls=':')

    plt.ylabel('net [0.5-10] keV rate (counts/s')
    plt.xlabel('Time')
    plt.suptitle('Xtend lightcurve of the source in bins of 128s')

def lc_internal_paper():

    binning=128

    figdir='/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/V4641Sgr/timeres/v4641dat/xtd/lc'
    lc_03_4 = fits.open(figdir+'/xa901001010xtd_src_0p3to4p0_bin128s.lc')
    lc_03_4_bg = fits.open(figdir+'/xa901001010xtd_bgd_0p3to4p0_bin128s.lc')

    lc_4_10 = fits.open(figdir+'/xa901001010xtd_src_4p0to10p0_bin128s.lc')
    lc_4_10_bg = fits.open(figdir+'/xa901001010xtd_bgd_4p0to10p0_bin128s.lc')

    fig_lc_intra, ax_lc_intra = plt.subplots(3, 1, sharex=True, figsize=(6,10))
    plt.subplots_adjust(hspace=0)

    xcolors_grp = ['black', 'red', 'limegreen', 'blue', 'cyan']

    inter_arr=np.array(list(interval_extract(lc_03_4[1].data['TIME']/128)))*128

    ax_lc_intra[0].set_ylabel('net [0.5-4.0]\nkeV rate\n(counts/s')
    ax_lc_intra[-1].set_xlabel('Time after the start of the observation (s)')
    ax_lc_intra[0].set_xlim(lc_03_4[1].data['TIME'][0],lc_03_4[1].data['TIME'][-1]+binning)
    ax_lc_intra[1].set_ylabel('net [4.0-10]\nkeV rate\n(counts/s)')

    ax_lc_intra[2].set_ylabel('net\n[4.0-10]/[0.5-4.0] keV\nHardness Ratio')

    for i_inter,elem_inter in enumerate(inter_arr):

        inter_mask=(lc_03_4[1].data['TIME']>=elem_inter[0]) & (lc_03_4[1].data['TIME']<=elem_inter[1])

        net_03_4=lc_03_4[1].data["RATE"][inter_mask] - lc_03_4_bg[1].data['RATE'][inter_mask]
        err_03_4=lc_03_4[1].data['ERROR'][inter_mask]
        net_4_10=lc_4_10[1].data["RATE"][inter_mask] - lc_4_10_bg[1].data['RATE'][inter_mask]
        err_4_10=lc_4_10[1].data['ERROR'][inter_mask]

        ax_lc_intra[0].errorbar(lc_03_4[1].data['TIME'][inter_mask]+binning/2, xerr=binning,
                     y=net_03_4,
                     yerr=err_03_4, marker='s',markersize=3, ls=':',color=xcolors_grp[i_inter])

        ax_lc_intra[1].errorbar(lc_4_10[1].data['TIME'][inter_mask]+binning/2, xerr=binning,
                     y=net_4_10,
                     yerr=err_4_10, marker='s',markersize=3, ls=':',color=xcolors_grp[i_inter])

        xtd_HR = net_4_10/net_03_4

        xtd_HR_err = ((err_03_4/net_03_4) ** 2 + (err_4_10/net_4_10) ** 2) ** (1 / 2) * xtd_HR

        ax_lc_intra[2].errorbar(lc_03_4[1].data['TIME'][inter_mask]+binning/2, xerr=128,
                     y=xtd_HR,
                     yerr=xtd_HR_err, marker='s',markersize=3, ls=':',color=xcolors_grp[i_inter])

    ax_lc_intra[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=True,
        direction='out')

    plt.subplots_adjust(left=0.16, right=0.96, top=0.97, bottom=0.05)

    #
    # ax_lc_intra[0].xaxis.tick_top()
    # ax_lc_intra[0].xaxis.set_ticklabels(ax_lc_intra[-1].xaxis.get_ticklabels())
    # plt.suptitle('Xtend lightcurve of the source in bins of 128s')

def lc_optical_monit():

    lc_optical_dir='/media/parrazyte/crucial_SSD/Observ/BHLMXB/Optical/V4641Sgr'

    lc_V=pd.read_csv(os.path.join(lc_optical_dir,'v4641Vpe-ccd_clean.dat'),
                     sep=' ',names=['MJD','V','ST'])



    #from the Mcdonalds14
    to_jd_mc14=2452423.641782407
    #this one is from orosz but used in the mcdonalds
    period_mc14=2.81730

    #from the Goranskij
    to_jd_gor24=2459410.4080208335
    period_gor24=2.81727

    def phase_time(time,to,period):
        phased_time=(time-to)%period/period

        return phased_time

    time_mjd=Time(lc_V['MJD'],format='mjd')
    time_phased_mc=phase_time(time_mjd.jd,to_jd_mc14,period_mc14)
    time_phased_Gor=phase_time(time_mjd.jd,to_jd_gor24,period_gor24)

    phase_obs_mc=phase_time(Time('2024-09-30').jd,to_jd_mc14,period_mc14)
    phase_obs_Gor=phase_time(Time('2024-09-30').jd,to_jd_gor24,period_gor24)

    #note: the obs is on MJD 60583
    mask_around=(time_mjd.value>60560) & (time_mjd.value<60600)

    #unfolded
    #plt.figure
    plt.plot(lc_V.MJD,lc_V.V,ls='',marker='d')
    plt.plot(lc_V.MJD[mask_around],lc_V.V[mask_around],ls='',marker='d',color='red')
    plt.gca().invert_yaxis()

    #folded
    plt.figure()
    plt.plot(time_phased_mc,lc_V.V,ls='',marker='d',color='blue')
    plt.plot(time_phased_mc+1,lc_V.V,ls='',marker='d',color='blue')

    plt.plot(time_phased_mc[mask_around],lc_V.V[mask_around],ls='',marker='d',color='red')
    plt.plot(time_phased_mc[mask_around]+1,lc_V.V[mask_around],ls='',marker='d',color='red')

    plt.gca().invert_yaxis()


def vrad_V4641():

    '''
    Here the result is the radial velocity of the system, and thus should be substracted
    from the observed velocity measurement

    '''
    #from vizier, slightly different than the ones of Gaia for some reason.
    #in [l,b], l is longitude (in plane), b is latitude (vertical)
    #reminder: https://en.wikipedia.org/wiki/Galactic_coordinate_system
    galcoords=[006.7739158,-4.7888593]
    l,d=galcoords
    #deprojecting the distance to the galactic plane
    #pm0.7
    d_full=6.2
    d_galplane=6.2*np.cos(-4.7888593*2*np.pi/360)

    #assuming a galactic center distance of 8.5 kpc
    d_galc=8.5

    #distance from the galactic center to the source
    #see https://www.omnicalculator.com/math/triangle-side
    d_galc_s=(d_galc**2 + d_galplane**2 -2*d_galc*d_galplane*np.cos(6.7739158*2*np.pi/360))**(1/2)

     #our angular velocity
    w_0=240
    #w velocity of the source from https://www.aanda.org/articles/aa/abs/2017/05/aa30540-17/aa30540-17.html
    w_source=1.022*(d_galc_s/d_galc)**(0.0803)*w_0

    #projection angle of the object compared to us is also beta (aka the angle between d_galc_s and d_galc)
    beta=np.arcsin(d_galplane/(d_galc_s/np.sin(l*2*np.pi/360)))

    #projected w
    w_los=np.sin(beta)*w_source
    #final value: 65.42 (coming away from us because the rotation curve of the galaxy is clockwise)



def vrot_earth(source='V4641sgr',date='2024-09-30'):

    '''
    Gives the radial velocity offset due to earth's rotation around the sun for a given source
    (by fetching Simbad for coordinates)
    and a given time

    result should be added to an observed velocity measurement
    '''

    north_p = EarthLocation.from_geodetic(lat=0*u.deg, lon=90*u.deg,height=0*u.m)

    sc_source_vals=sexa2deg([Simbad.query_object(source)[0]['RA'],Simbad.query_object(source)[0]['DEC']][::-1])[::-1]

    sc = SkyCoord(ra=sc_source_vals[0]*u.deg, dec=sc_source_vals[1]*u.deg)

    heliocorr = sc.radial_velocity_correction('heliocentric',obstime=Time(date),location=north_p)
    return heliocorr.to(u.km/u.s)


def lc_paper_monit(spec_mode='Eddington'):
    lc_dir = '/home/parrazyte/Documents/Work/PostDoc/docs/docs_XRISM/V4641Sgr_docs/'

    MAXI_psf_csv = pd.read_csv(os.path.join(lc_dir, 'MAXI_megumi/v4640_241111/gsc_final', 'V4641_Sgr_1d.qdp'),
                               skiprows=7,
                               sep=' ',
                               names=['MJD', 'fit', 'f24', 'f24err-', 'f24err+', 'f410', 'f410err-', 'f410err+'])

    MAXI_od_csv = pd.read_csv(os.path.join(lc_dir, 'MAXI_megumi', 'v4641sgr_2.0-10.0keV_gsclc_ondemand.dat'),
                              skiprows=1,
                              names=['MJD_start', 'MJD_end', 'counts', 'counts_err'], sep=' ')

    EP_csv =pd.read_csv(os.path.join(lc_dir,'EP','V4641_wxt_lc_0515_154_ord_v2_0.txt'),skiprows=4,header=None,
                                  names=['mjd', 'exposure', 'rate_tot','rate_tot_err',
                                         'rate_soft','rate_soft_err',
                                         'rate_hard','rate_hard_err'],sep=' ')

    sp_anal_dir='/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/V4641Sgr/simultaneous'
    swift_csv=pd.read_csv(os.path.join(sp_anal_dir,'infos_fit_deabs_minus_highbg_Swift_NH_015.txt'),
                          header=0,sep='\t')
    NICER_csv=pd.read_csv(os.path.join(sp_anal_dir,'infos_fit_deabs_minus_highbg_NICER_NH_015.txt'),
                          header=0,sep='\t')
    swift_f_1_10=np.array(swift_csv[swift_csv.columns[5:12]].sum(1))
    swift_f_3_6=np.array(swift_csv[swift_csv.columns[5:8]].sum(1))
    swift_f_6_10=np.array(swift_csv[swift_csv.columns[8:12]].sum(1))
    swift_HR=np.array(swift_f_6_10/swift_f_3_6)
    swift_dates=np.array(mdates.date2num(swift_csv['t_start']))

    swift_dates_order=swift_dates.argsort()
    swift_f_1_10=swift_f_1_10[swift_dates_order]
    swift_f_3_6=swift_f_3_6[swift_dates_order]
    swift_f_6_10=swift_f_6_10[swift_dates_order]
    swift_HR=swift_HR[swift_dates_order]
    swift_dates=swift_dates[swift_dates_order]

    NICER_f_1_10=np.array(NICER_csv[NICER_csv.columns[5:12]].sum(1))
    NICER_f_3_6=np.array(NICER_csv[NICER_csv.columns[5:8]].sum(1))
    NICER_f_6_10=np.array(NICER_csv[NICER_csv.columns[8:12]].sum(1))
    NICER_HR=np.array(NICER_f_6_10/NICER_f_3_6)
    NICER_dates=np.array(mdates.date2num(NICER_csv['t_start']))

    NICER_dates_order=NICER_dates.argsort()
    NICER_f_1_10=NICER_f_1_10[NICER_dates_order]
    NICER_f_3_6=NICER_f_3_6[NICER_dates_order]
    NICER_f_6_10=NICER_f_6_10[NICER_dates_order]
    NICER_HR=NICER_HR[NICER_dates_order]
    NICER_dates=NICER_dates[NICER_dates_order]

    edd_factor_source=edd_factor(6.2,6.4)

    # assuming an HR of 0.5 here
    MAXI_od_210_f = MAXI_od_csv['counts'] * 1 / (1.065 + 1.172) * 2 * 2.4e-8

    # errors, adding an average systematics
    MAXI_od_210_err = MAXI_od_csv['counts_err'] * 1 / (1.065 + 1.172) * (1.1 + 1.03) / 2 * 2 * 2.4e-8

    # dates
    MAXI_od_ds = Time(MAXI_od_csv.MJD_start, format='mjd')
    MAXI_od_de = Time(MAXI_od_csv.MJD_end, format='mjd')

    MAXI_od_mjd_err = (MAXI_od_de - MAXI_od_ds) / 2

    MAXI_od_mjd = (MAXI_od_ds + MAXI_od_mjd_err)

    MAXI_od_dates = mdates.date2num(MAXI_od_mjd.datetime)

    MAXI_od_daterr = MAXI_od_mjd_err.to_value('jd')

    # values for psf
    crab_conv_24 = 1 / 1.065 * (2.4e-8)
    crab_conv_410 = 1 / 1.172 * (2.4e-8)

    MAXI_psf_210_f = MAXI_psf_csv['f24'] * crab_conv_24 + MAXI_psf_csv['f410'] * crab_conv_410

    # using the systematics as a multiplicator otherwise this doesn't make sense

    MAXI_psf_210_errl = np.sqrt((MAXI_psf_csv['f24err-'] * crab_conv_24) ** 2 * 1.165 / 1.065 +
                                (MAXI_psf_csv['f410err-'] * crab_conv_410) ** 2 * 1.203 / 1.72)
    # #2-4 systematics
    # (0.1*2.4e-8)**2+
    #   #4-10 systematics
    # (0.03*2.4e-8)**2)

    MAXI_psf_210_errh = np.sqrt((MAXI_psf_csv['f24err+'] * crab_conv_24) ** 2 * 1.1 +
                                (MAXI_psf_csv['f410err+'] * crab_conv_410) ** 2 * 1.03)
    # #2-4 systematics
    # (0.1*2.4e-8)**2+
    #   #4-10 systematics
    # (0.03*2.4e-8)**2)

    MAXI_psf_210_err = np.array([MAXI_psf_210_errl.values, MAXI_psf_210_errh.values])

    MAXI_psf_mjd = Time(MAXI_psf_csv.MJD, format='mjd')
    MAXI_psf_dates = mdates.date2num(MAXI_psf_mjd.datetime)

    ####PLOT
    date_format = mdates.DateFormatter('%Y-%m-%d')
    #     date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
    # fig_lc,ax_lc=plt.subplots(4,1,sharex=True,figsize=(10,6))
    fig_lc, ax_lc = plt.subplots(4, 1, sharex=True, figsize=(6, 10))

    def numtomjd(x):
        '''
        the direct conversion doesn't seem to work so using  an hardwritten workaround
        '''
        return x + 40587

    #    return  Time(mdates.num2date(x)).mjd

    def mjdtonum(x):
        '''
        the direct conversion doesn't seem to work so using  an hardwritten workaround
        '''
        return x - 40587

        # return mdates.date2num(Time(x,format='mjd').datetime)

    # more axis for better visualisation
    secax_0 = ax_lc[0].secondary_xaxis('top', functions=(numtomjd, mjdtonum))

    #swapped because we swapped the NICER and EP lc
    secax_1 = ax_lc[2].secondary_xaxis('top', functions=(numtomjd, mjdtonum))
    secax_2 = ax_lc[3].secondary_xaxis('top', functions=(numtomjd, mjdtonum))
    secax_3 = ax_lc[1].secondary_xaxis('top', functions=(numtomjd, mjdtonum))
    secax_4 = ax_lc[1].secondary_xaxis('bottom', functions=(numtomjd, mjdtonum))

    for ax in [secax_0, secax_1, secax_2, secax_3, secax_4]:
        ax.yaxis.set_visible(False)

    secax_4.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction='in')

    secax_0.xaxis.set_minor_locator(MultipleLocator(1))
    secax_1.xaxis.set_minor_locator(MultipleLocator(1))
    secax_2.xaxis.set_minor_locator(MultipleLocator(1))
    secax_3.xaxis.set_minor_locator(MultipleLocator(1))
    secax_4.xaxis.set_minor_locator(MultipleLocator(1))

    secax_1.xaxis.set_ticklabels('')
    secax_2.xaxis.set_ticklabels('')
    secax_3.xaxis.set_ticklabels('')
    secax_4.xaxis.set_ticklabels('')

    ax_lc_02 = ax_lc[0].twinx()
    ax_lc_12 = ax_lc[1].twinx()
    ax_lc_22 = ax_lc[2].twinx()
    ax_lc_32 = ax_lc[3].twinx()

    ax_lc_32.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,
        labeltop=False,
        direction='out')

    ax_lc[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction='in')

    ax_lc[1].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction='in')

    ax_lc[2].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction='in')

    ax_lc[3].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction='in')

    plt.subplots_adjust(hspace=0)

    ax_lc[1].set_ylim(0.0, 0.29)

    flux_convert=1
    if spec_mode=='flux':
        flux_convert=1
        ax_lc[2].set_ylabel('[3-10]keV unabsorbed flux\n($10^{-9}$ erg/s/cm²)')
        ax_lc[2].set_ylim(0, 2e-9 * 1e9)

    elif spec_mode=='Eddington':
        flux_convert=edd_factor_source
        ax_lc[2].set_yscale('log')
        ax_lc[2].set_ylabel('unabsorbed luminosity \n in [1-10]keV (L/L$_{Edd}$)')
        # ax_lc[2].set_ylim(6e-5, 5e-3)

    ax_lc[0].xaxis.set_major_formatter(date_format)

    ax_lc[0].set_ylabel('MAXI [2-10]keV flux\n($10^{-9}$ ergs/s/cm²)')
    ax_lc[3].set_ylabel('unabsorbed HR \n [6-10]/[3-10]keV')

    ax_lc[1].set_ylabel('Einstein Probe \n[0.5-4] keV count rate ')

    ax_lc[1].errorbar(x=mdates.date2num(Time(EP_csv['mjd'],format='mjd').datetime),
                      y=EP_csv['rate_tot'],yerr=EP_csv['rate_tot_err'],
                      marker='s', markersize=5,
                      linewidth=0.5,elinewidth=1.,ls=':',color='black')

    ax_lc[0].errorbar(x=MAXI_psf_dates, xerr=0.5, y=MAXI_psf_210_f * 1e9, yerr=MAXI_psf_210_err * 1e9, linewidth=0.5,
                      elinewidth=1., ls=':',
                      color='black', label='PSF fitting')

    min_vis_psf = MAXI_psf_dates[MAXI_psf_210_f.values > 0.][0]
    max_vis_psf = MAXI_psf_dates[MAXI_psf_210_f.values > 0.][-1]

    ax_lc[0].errorbar(x=MAXI_od_dates, xerr=MAXI_od_daterr, y=MAXI_od_210_f * 1e9, yerr=MAXI_od_210_err * 1e9,
                      linewidth=0.5,
                      elinewidth=1., ls=':',
                      color='grey', alpha=0.3, label='On-demand tool')

    min_vis_od = MAXI_od_dates[MAXI_od_210_f.values > 0.][0]
    max_vis_od = MAXI_od_dates[MAXI_od_210_f.values > 0.][-1]

    # ax_lc[0].set_xlim(min(min_vis_psf, min_vis_od) - 10, max(max_vis_psf, max_vis_od) + 1)

    # ax_lc[0].set_ylim(0,ax_lc[0].get_ylim()[1])
    ax_lc[0].set_ylim(0., 2e-9 * 1e9)

    # adding the XRISM obs
    xrism_interval = Time(['2024-09-30 09:42:04', '2024-09-30 17:03:04'])

    # adding the radio obs
    radio_obs_ul = Time('2024-09-29 12:57:36')
    radio_obs_jet = Time('2024-10-06 12:28:48')
    radio_obs_firstdet = Time('2024-09-16 15:05:00')

    for i_ax, ax in enumerate([ax_lc_02, ax_lc_12, ax_lc_22, ax_lc_32]):
        ax.axvspan(mdates.date2num(xrism_interval[0].datetime), mdates.date2num(xrism_interval[1].datetime),
                   color='green', alpha=0.2, label='XRISM observation')

        ax.axvline(mdates.date2num(radio_obs_firstdet.datetime), color='red', ls='-', alpha=0.5, label='')
        ax.axvline(mdates.date2num(radio_obs_ul.datetime), color='blue', ls='-', alpha=0.5, label='Radio non-detection')
        ax.axvline(mdates.date2num(radio_obs_jet.datetime), color='red', ls='-', alpha=0.5, label='Radio detection')

        ax.yaxis.set_visible(False)
        if i_ax==3:
            ax.xaxis.set_visible(True)
            ax.legend(loc='upper left')

    ax_lc[0].legend(loc='upper right')


    #NICER points
    NICER_y=(NICER_f_1_10 * flux_convert)
    ax_lc[2].errorbar(NICER_dates,y=NICER_y,xerr=0,yerr=0,marker='d',markersize=5,
                      linewidth=0.5,
                      elinewidth=1., ls=':', color='black',label="NICER")

    ax_lc[3].errorbar(NICER_dates,y=NICER_HR,xerr=0,yerr=0,marker='d',markersize=5,
                      linewidth=0.5,
                      elinewidth=1., ls=':', color='black')

    #Swift points
    swift_y=(swift_f_1_10 * flux_convert)[swift_dates>19500]
    ax_lc[2].errorbar(swift_dates[swift_dates>19500],y=swift_y,
                      xerr=0,yerr=0,
                      marker='d',markersize=5,
                      linewidth=0.5,
                      elinewidth=1., ls=':', color='brown',label=r"Swift")
    ax_lc[3].errorbar(swift_dates[swift_dates>19500],y=swift_HR[swift_dates>19500],
                      xerr=0,yerr=0,
                      marker='d',markersize=5,
                      linewidth=0.5,
                      elinewidth=1., ls=':', color='brown')



    #for adding the XRISM point
    #from the full_lines model
    # Model Flux 0.0055709 photons (2.5823e-11 ergs/cm^2/s) range (1.0000 - 10.000 keV)
    # Model Flux 0.0015144 photons (1.0085e-11 ergs/cm^2/s) range (3.0000 - 6.0000 keV)
    # Model Flux 0.00046412 photons (5.4538e-12 ergs/cm^2/s) range (6.0000 - 10.000 keV)

    ax_lc[2].errorbar((mdates.date2num((xrism_interval[0]+TimeDelta(3.5*3600,format='sec')).datetime)),
                      2.5823e-11*edd_factor_source,marker='X',color='green',markersize=10,label="XRISM")

    ax_lc[3].errorbar((mdates.date2num((xrism_interval[0]+TimeDelta(3.5*3600,format='sec')).datetime)),
                      5.4538e-12/1.0085e-11,marker='X',color='green',markersize=10)

    ax_lc[2].legend(loc='lower left')


    ax_lc[2].set_ylim(6e-5,2e-3)
    ax_lc[3].set_ylim(0.1, 1.5)

    plt.show()

    ax_lc[0].set_xlim(NICER_dates[0] - 15.5, NICER_dates[-1] + 7.5)

    plt.subplots_adjust(left=0.15, right=0.96, top=0.97, bottom=0.03)

    plt.show()

    # needs to be done last
    plt.setp(ax_lc[2].get_yticklabels()[-1], visible=False)
    plt.setp(ax_lc[3].get_yticklabels()[-1], visible=False)

def xcustom_eeuf(mode='ratio'):
    fig,ax=plt.subplots(2, 1, figsize=(10, 6),)
    if mode=='ratio':
        xPlot('eeuf,ratio',axes_input=ax.tolist(),force_ylog_ratio=True,ylims=[[5e-4, 7e-2],[3e-1, 9.1]])
    if mode=='delchi':
        xPlot('eeuf,delchi',axes_input=ax.tolist(),)
        ax[0].set_yscale('log')
        ax[0].set_ylim([5e-4, 7e-2])
        ax[1].tick_params(
            axis='x', which='both', bottom=True, labelbottom=True)
    plt.subplots_adjust(hspace=0)


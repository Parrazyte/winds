#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:33:19 2022

@author: parrama
"""

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

'''Astro'''

#local
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/')

#online
sys.path.append('/mount/src/winds/observations/spectral_analysis/')
sys.path.append('/mount/src/winds/general/')

#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std,c_light,lines_std_names,lines_e_dict,ang2kev

from general_tools import ravel_ragged,MinorSymLogLocator,rescale_flex,expand_epoch

#Catalogs and manipulation
from astroquery.vizier import Vizier


telescope_list=('Chandra','NICER','NuSTAR','Suzaku','XMM','Swift')

#making a "fake" instrument (colorblind friendly) colormap to keep individual identification
color_cmap_instru = mpl.cm.plasma

norm_pow=1.
c_norm_instru = mpl.colors.Normalize(vmin=0.,
                              vmax=len(telescope_list)**(norm_pow)-1.5)

colors_func_instru = mpl.cm.ScalarMappable(norm=c_norm_instru, cmap=color_cmap_instru)

# computing the actual color array for the detections
telescope_colors={'Chandra':colors_func_instru.to_rgba(0),
                  'NICER':colors_func_instru.to_rgba(1),
                  'NuSTAR':colors_func_instru.to_rgba(2**(norm_pow)),
                  'Suzaku':colors_func_instru.to_rgba(3**(norm_pow)),
                   'XMM':colors_func_instru.to_rgba(4**(norm_pow)),
                  'Swift':colors_func_instru.to_rgba(5**(norm_pow))}


#previous non colorblind-friendly mode
# telescope_colors={'XMM':'red',
#                   'Chandra':'blue',
#                   'NICER':'green',
#                   'Suzaku':'magenta',
#                   'NuSTAR':'orange',
#                   'Swift':'cyan'}

#inclination, mass and other values

#note: in order to make sure we can use our overlap function, we add a very slight uncertainty to inclinations without uncertainties, 
#to make sur the overlap result with the inclination constraints is not 0
# incl_dic={
#         '1H1659-487':[57.5,20.5,20.5],
#         'GX339-4':[57.5,20.5,20.5],
#         #grs with both ortographs to swap colors sometimes
#         'GRS1915+105':[64,4,4],
#         'GRS 1915+105':[64,4,4],
#         'MAXIJ1820+070':[74,7,7],
#         'H1743-322':[75,3,3],
#         '4U1630-472':[67.5,7.5,7.5],
#         '4U1630-47':[67.5,7.5,7.5],
#         'SwiftJ1658.2-4242':[64,3,2],
#         'IGRJ17091-3624':[70,0,0.001],
#         'MAXIJ1535-571':[45,45,0],
#         '1E1740.7-2942':[50,0,40],
#         'GROJ1655-40':[69,2,2],
#         'SwiftJ1753.5-0127':[55,7,2],
#          #mixed both reflection measurements here
#         'EXO1846-031':[56.5,16.5,16.5],
#         '4U1957+115':[13,0,0.001],
#         'MAXIJ1348-630':[28,3,44],
#         'XTEJ1752-223':[49,49,0],
#         'MAXIJ1659-152':[70,10,10],
#         'GS1354-64':[70,0,0.001],
#         'XTEJ1650-500':[47,0,43],
#         #for 17451 we only have eclipses constraints
#         'IGRJ17451-3022':[70,0,20],
#         'SwiftJ1357.2-0933':[80,0,10],
#         'XTEJ1652-453':[32,32,0],
#         'MAXIJ1803-298':[70,0,20],
#         'V4641Sgr':[72,4,4],
#         'V404Cyg':[67,1,3],
#         'XTEJ1550-564':[75,4,4],
#         'MAXIJ1305-704':[72,8,5],
#         'AT2019wey':[30,30,0],
#         '4U1543-475':[21,2,2],
#         'GRS1739-278':[33,0,0.001],
#         'XTEJ1118+480':[72,2,2],
#         'GRS1716-249':[50,10,10],
#         'MAXIJ0637-430':[64,6,6]
#          }

incl_dyn_dict={
        '4U1543-475':[20.7,1.5,1.5,1],
        '4U1630-47':[67.5,7.5,7.5,1],
        '4U1957+115':[13,0,0,0],
        'A0620-00':[52.6,2.5,2.5,1],
        'A1524-61':[57,13,13,1],
        'GROJ0422+32':[56,4,4,1],
        'GROJ1655-40':[69,2,2,1],
        'GRS1009-45':[59,22,22,0],
        'GRS1716-249':[61,15,15,1],
        'GRS1915+105':[64,4,4,1],
        'GS1354-64':[79,79,0,0],
        'GS2000+251':[68,6,6,1],
        'GX339-4':[57,20,20,0],
        'H1705-250':[64,16,16,1],
        'IGRJ17451-3022':[70,0,20,1],
        'MAXIJ1305-704':[72,8,5,1],
        'MAXIJ1659-152':[70,10,10,1],
        'MAXIJ1803-298':[67,8,8,1],
        'MAXIJ1820+070':[74,7,7,1],
        'MAXIJ1836-194':[9,5,6,1],
        'MAXIJ1848-015':[77,2,2,1],
        'NovaMuscae1991':[43,3,2,1],
        'SwiftJ1357.2-0933':[81,12,9,1],
        'V404Cyg':[67,1,2,1],
        'V4641Sgr':[72,4,4,1],
        'XTEJ1118+480':[72,2,2,1],
        'XTEJ1550-564':[75,4,4,1],
        'XTEJ1650-500':[47,0,43,0],
        'XTEJ1859+226':[67,4,4,1]
}

incl_jet_dict={'H1743-322':[75,3,3,1],
              'MAXIJ1348-630':[28,3,3,1],
              'MAXIJ1535-571':[45,45,0,0],
              'XTEJ1752-223':[49,49,0,0],}

incl_misc_dict={'IGRJ17091-3624':[70,0,0,0],
               'MAXIJ1348-630':[65,7,7,0]
               }

incl_refl_dict={
        '1E1740.7-2942':[31,18,29,1],
        '4U1543-475':[67,8,7,1],
        '4U1630-47':[55,11,8,1],
        '4U1957+115':[52,13,12,1],
        'AT2019wey':[14,10,12,1],
        'EXO1846-031':[62,9,10,1],
        'GRS1716-249':[59,12,7,1],
        'GRS1739-278':[70,11,5,1],
        'GRS1758-258':[67,13,8,1],
        'GRS1915+105':[60,8,8,1],
        'GS1354-64':[47,10,11,1],
        'GX339-4':[49,14,14,1],
        'H1743-322':[54,13,12,1],
        'IGRJ17091-3624':[47,11,10,1],
        'IGRJ17454-3019':[54,14,15,1],
        'MAXIJ0637-430':[63,10,9,1],
        'MAXIJ1348-630':[52,11,8,1],
        'MAXIJ1535-571':[44,19,17,1],
        'MAXIJ1631-479':[22,12,10,1],
        'MAXIJ1727-203':[65,14,11,1],
        'MAXIJ1803-298':[72,9,6,1],
        'MAXIJ1813-095':[42,13,11,1],
        'MAXIJ1820+070':[64,9,8,1],
        'MAXIJ1848-015':[29,10,13,1],
        'SwiftJ1658.2-4242':[50,10,9,1],
        'SwiftJ1728.9-3613':[7,3,8,1],
        'SwiftJ174540.2-290037':[31,9,8,1],
        'SwiftJ174540.7-290015':[63,8,10,1],
        'SwiftJ1753.5-0127':[73,8,8,1],
        'V404Cyg':[37,8,9,0],
        'V4641Sgr':[66,11,7,1],
        'XTEJ1550-564':[40,10,10,1],
        'XTEJ1652-453':[32,32,0,0],
        'XTEJ1752-223':[35,4,4,1],
        'XTEJ1859+226':[71,1,1,1],
        'XTEJ1908+094':[28,11,11,1],
        'XTEJ2012+381':[68,11,6,1]
         }

Porb_dict={'1E1740.7-2942':[303,2,1,1],
           '4U1543-475':[26.8,0,0,1],
           '4U1957+115':[9.33,0,0,1],
           'A0620-00':[7.75,0,0,1],
           'A1524-61':[6.2,2,2,1],
           'CXOGC174540.0-290031':[7.8,0,0,1],
           'GROJ0422+32':[5.09,0,0,1],
           'GROJ1655-40':[62.9,0,0,1],
           'GRS1009-45':[6.85,0,0,1],
           'GRS1716-249':[6.67,0,0,1],
           'GRS1915+105':[812,4,4,1],
           'GS1354-64':[61.1,0,0,1],
           'GS2000+251':[8.26,0,0,1],
           'GX339-4':[42.2,0,0,1],
           'H1705-250':[12.51,0,0,1],
           'IGRJ17451-3022':[6.3,0,0,1],
           'MAXIJ0637-430':[2.2,1,1,0],
           'MAXIJ1305-704':[9.5,0.1,0.1,1],
           'MAXIJ1659-152':[2.4,0,0,1],
           'MAXIJ1803-298':[7,0.2,0.2,1],
           'MAXIJ1820+270':[16.5,0,0,1],
           'NovaMuscae1991':[10.4,0,0,1],
           'SwiftJ1357.2-0933':[2.8,0.3,0.3,1],
           'SwiftJ1727.8-1613':[7.6,0.2,0.2,1],
           'SwiftJ174510.8-262411':[11.3,11.3,0,0],
           'SwiftJ1753.5-0127':[3.2,0,0,1],
           'SwiftJ1910.2-0546':[2.4,0.1,0.1,1],
           'V404Cyg':[155.3,0,0,1],
           'V4641Sgr':[67.6,0,0,1],
           'XTEJ1118+480':[4.1,0,0,1],
           'XTEJ1550-564':[37.0,0,0,1],
           'XTEJ1650-500':[7.7,0,0,1],
           'XTEJ1752-223':[7,7,0,0],
           'XTEJ1859+226':[6.6,0,0,1]}

dist_dict={
    '4U1543-475':[7.5,0.5,0.5,1],
    '4U1630-47':[8.1,3.4,3.4,1],
    '4U1755-388':[6.5,2.5,2.5,1],
    'A0620-00':[1.06,1,1,1],
    'A1524-61':[8,0.9,0.9,1],
    'EXO1846-031':[7,0,0,0],
    'GROJ0422+32':[2.5,0.3,0.3,1],
    'GROJ1655-40':[3.2,0.2,0.2,1],
    'GRS1009-45':[3.8,0.3,0.3,1],
    'GRS1716-249':[6.9,1.1,1.1,1],
    'GRS1739-278':[7.3,1.3,1.3,1],
    'GRS1915+105':[9.4,1.6,1.6,1],
    'GS1354-64':[25,0,0,0],
    'GS2000+251':[2.7,0.7,0.7,1],
    'H1705-250':[8.6,2.1,2.1,1],
    'H1743-322':[8.5,0.8,0.8,1],
    'IGRJ17098-3628':[10.5,0,0,0],
    'MAXIJ1305-704':[7.5,1.4,1.8,1],
    'MAXIJ1348-630':[3.4,0.4,0.4,1],
    'MAXIJ1535-571':[4.1,0.5,0.6,1],
    'MAXIJ1659-152':[8.6,3.7,3.7,1],
    'MAXIJ1820+070':[2.96,0.33,0.33,1],
    'MAXIJ1836-194':[7,3,3,1],
    'MAXIJ1848-015':[3.4,0.3,0.3,1],
    'NovaMuscae1991':[5,0.7,0.7,1],
    'SwiftJ1727.8-1613':[2.7,0.3,0.3,1],
    'SwiftJ1728.9-3613':[8.4,0.8,0.8,1],
    'SwiftJ174510.8-262411':[3.7,1.1,1.1,0],
    'SwiftJ1753.5-0127':[5.6,2.8,1.6,1],
    'V404Cyg':[2.4,0.2,0.2,1],
    'V4641Sgr':[6.2,0.7,0.7,1],
    'XTEJ1118+480':[1.7,0.1,0.1,1],
    'XTEJ1550-564':[4.4,0.4,0.6,1],
    'XTEJ1650-500':[2.6,0.7,0.7,1],
    'XTEJ1720-318':[6.5,3.5,3.5,1],
    'XTEJ1752-223':[6,2,2,1],

    #here there is no estimate (the one quoted in BlackCAT is utter garbage) so we keep this at 8kpc
    'XTEJ1817-330':[8,8,8,1],

    'XTEJ1818-245':[3.6,0.8,0.8],
    'XTEJ1859+226':[12.5,1.5,1.5],
    'XTEJ1908+094':[6.5,3.5,3.5]
}

mass_dict={
    '4U1543-475':[8.4,1,1,1],
    '4U1957+115':[3,1,2.5,1],
    'A0620-00':[6.6,0.3,0.3,1],
    'A1524-61':[5.8,2.4,3,1],
    'GROJ0422+32':[2.7,0.5,0.7,1],
    'GROJ1655-40':[5.4,0.3,0.3,1],
    'GRS1716-249':[6.4,2,3.2,1],
    'GRS1915+105':[11.2,1.8,2,1],
    'GS2000+251':[7.2,1.7,1.7,1],
    'GX339-4':[5.9,3.6,3.6,1],
    'H1705-250':[5.4,1.5,1.5,1],
    'MAXIJ1305-704':[8.9,1.,1.6,1],
    'MAXIJ1820+070':[6.9,1.2,1.2,1],
    'NovaMuscae1991':[11,1.4,2.1,1],
    'SwiftJ1357.2-0933':[11.6,1.9,2.5,1],
    'V404Cyg':[9,0.6,0.2,1],
    'V4641Sgr':[6.4,0.6,0.6,1],
    'XTEJ1118+480':[7.1,0.1,0.1,1],
    'XTEJ1550-564':[11.7,3.9,3.9,1],
    'XTEJ1859+226':[8,2,2,1]}

dippers_list=['4U1543-475',
              '4U1630-47',
              'A0620-00',
              'CXOGC174540.0-290031',
              'GROJ1655-40',
              'GRS1915+105',
              'GRS 1915+105',
              'H1743-322',
              'IGRJ17091-3624',
              'IGRJ17451-3022',
              'MAXIJ1305-704',
              'MAXIJ1659-152',
              'MAXIJ1803-298',
              'MAXIJ1820+070',
              'SwiftJ1357.2-0933',
              'SwiftJ1658.2-4242',
              'XTEJ1817-330',
              'XTEJ1859+226']


class wind_det:
    def __init__(self,state,reflmix=False,trust=True):
        self.state=state
        self.reflmix=bool(int(reflmix))
        self.trust=bool(int(trust))

class band_det:
    def __init__(self,wind_det_str):
        '''
        order should be 1:soft, 2:hard, 3+: weird if needed
        '''

        n_wind_det=wind_det_str.split(',')
        self.soft=None
        self.hard=None
        self.trust=False
        self.trust_noem=False

        for i_det,elem_wind_det in enumerate(wind_det_str.split(',')):
            wind_det_state=elem_wind_det.split('_')[0]
            wind_det_reflmix=False if len(elem_wind_det.split('_'))<=1 else elem_wind_det.split('_')[1]
            wind_det_trust=True if len(elem_wind_det.split('_'))<=2 else elem_wind_det.split('_')[2]

            setattr(self,wind_det_state,wind_det(wind_det_state,wind_det_reflmix,wind_det_trust))

            wind_obj=getattr(self,elem_wind_det.split('_')[0])
            #setting a global trust for the band if at least one detection is ok
            if wind_obj.trust:
                self.trust=True

            #setting a global trust with no reflection for the band
            if wind_obj.trust and not wind_obj.reflmix:
                self.trust_noem=True

class source_wind:
    '''
    Source wind detections class

    form of the values for each band: state_bool1_bool2

    with bool1 for reflmix, bool2 for trust
    '''
    def __init__(self,iron_band=None,soft_x=None,visible=None,infrared=None):

        self.trust=False
        self.trust_noem=False
        if iron_band is None:
            self.iron_band=False
        else:
            self.iron_band=band_det(iron_band)

            if self.iron_band.trust:
                self.trust = True

            if self.iron_band.trust_noem:
                self.trust_noem=True


        if soft_x is None:
            self.soft_x=False
        else:
            self.soft_x=band_det(soft_x)

            if self.soft_x.trust:
                self.trust = True

            if self.soft_x.trust_noem:
                self.trust_noem=True

        if visible is None:
            self.visible=False
        else:
            self.visible=band_det(visible)

            if self.visible.trust:
                self.trust = True

            if self.visible.trust_noem:
                self.trust_noem=True

        if infrared is None:
            self.infrared=False
        else:
            self.infrared=band_det(infrared)

            if self.infrared.trust:
                self.trust = True

            if self.infrared.trust_noem:
                self.trust_noem=True

#note: we don't list SwiftJ1658.2-4242 because for now all of its NuSTAR wind signatures have been reclassified as edges
#otherwise we would have
# 'SwiftJ1658.2-4242':source_wind('soft,hard'),

wind_det_dict={'4U1543-47':source_wind('soft_1_0','soft_1_0'),
              '4U1630-47':source_wind('soft','soft'),
              'EXO1846-031':source_wind('hard_1'),
              'GROJ1655-40':source_wind('soft','soft'),
              'GRS1716-249':source_wind(visible='hard'),
              'GRS1758-258':source_wind('hard_0_0'),
              'GRS1915+105':source_wind('soft,hard','soft',infrared='hard_0_0'),
              'GX339-4':source_wind(soft_x='hard',visible='soft,hard'),
              'H1743-322':source_wind('soft'),
              'IGRJ17091-3624':source_wind('soft,hard_1_0',soft_x='hard'),
              'IGRJ17451-3022':source_wind('soft','soft'),
              'MAXIJ1305-704':source_wind('soft,hard_1','soft,hard'),
              'MAXIJ1348-630':source_wind('soft_1,hard_1',soft_x='hard_0_0',visible='hard',infrared='soft,hard'),
              'MAXIJ1535-571':source_wind('soft_1,hard_1'),
              'MAXIJ1631-479':source_wind('soft_1'),
              'MAXIJ1803-298':source_wind('soft',visible='hard'),
              'MAXIJ1810-222':source_wind(soft_x='soft_0_1,hard_0_1'),
              'MAXIJ1820+070':source_wind('soft',visible='hard',infrared='soft,hard'),
              'SwiftJ1727.8-1613':source_wind(visible='soft_0_0,hard'),
              'SwiftJ1357.2-0933':source_wind(visible='hard'),
              'SwiftJ151857.0-572147':source_wind('soft'),
              'SwiftJ174540.7-290015':source_wind('soft_1_0'),
              'V404Cyg':source_wind('hard','hard','hard'),
              'V4641Sgr':source_wind(soft_x='soft',visible='hard'),
              'XTEJ1550-564':source_wind('soft_1'),
              'XTEJ1652-453':source_wind('soft_1_0')
              }

wind_det_sources=list(wind_det_dict.keys())
#custom distande dictionnary for measurements which are not up to date in blackcat/watchdog
# dist_dic={
#     'MAXIJ1535-571':[4.1,0.6,0.5],
#     'GRS 1915+105':[9.4,1.4,1.4],
#     'GRS1915+105':[9.4,1.4,1.4],
#     'MAXIJ1348-630':[3.39,0.385,0.382],
#     'H1743-322':[8.5,0.8,0.8],
#     'SwiftJ1357.2-0933':[8,0,0],
#     }

#note : only dynamical measurements for BHs
# mass_dic={
#     '1H1659-487':[5.9,3.6,3.6],
#     'GRS 1915+105':[11.2,1.8,2],
#     'GRS1915+105':[11.2,1.8,2],
#     'MAXIJ1820+070':[6.9,1.2,1.2],
#     'GROJ1655-40':[5.4,0.3,0.3],
#     'SAXJ1819.3-2525':[6.4,0.6,0.6],
#     'GS2023+338':[9,0.6,0.2],
#     'XTEJ1550-564':[11.7,3.9,3.9],
#     'MAXIJ1305-704':[8.9,1.,1.6],
#     '4U1543-475':[8.4,1,1],
#     'XTEJ1118+480':[7.55,0.65,0.65],
#     # 'IGRJ17451-3022':[1.5,0,0],
#     #NS:
#     'XTEJ1701-462':[1.4,0,0]
#     }

#BAT conversion factors for 1 cts/s in 15-50 keV counts to 15-50keV flux
convert_BAT_count_flux={
                        #this one assumes 11e22nH (but that doesn't change anything) and 2.5 gamma powerlaw
                        '4U1630-47':3.597E-07
                        }

sources_det_dic=['GRS1915+105','GRS 1915+105','GROJ1655-40','H1743-322','4U1630-47','IGRJ17451-3022']

#should be moved somewhere for online
rxte_lc_path='/media/parrama/crucial_SSD/Observ/BHLMXB/RXTE/RXTE_lc_dict.pickle'

if os.path.exists(rxte_lc_path):
    with open(rxte_lc_path,'rb') as rxte_lc_file:
        dict_lc_rxte=pickle.load(rxte_lc_file)
else:
    dict_lc_rxte=None

#current number of informations in abslines_infos
n_infos=9

info_str=['equivalent widths','velocity shifts','energies','line flux','time','width']

info_hid_str=['6-10/3-6 Hardness Ratio','3-10 Luminosity','Time',
              'nthcomp Gamma','15-50 Luminosity','15-50/3-6 Hardness Ratio']

axis_str=['Line EW (eV)','Line velocity shift (km/s)','Line energy (keV)',r'line flux (erg/s/cm$^{-2}$)',None,r'Line FWHM (km/s)']

axis_hid_str=['Hardness Ratio ([6-10]/[3-6] keV bands)',r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units',None,
              r'nthcomp $\Gamma$',r'Luminosity in the [15-50] keV band in (L/L$_{Edd}$) units',
              'Hardness Ratio ([15-50]/[3-6] keV bands)']

#indexes for ratio choices in lines between specific ions/complexes
ratio_choices={
                '25':[3,0],
                '26':[4,1],
                'Ka':[1,0],
                'Kb':[4,3],
              }

#theoretical ratio values
ratio_vals={'Ka':0.545
    
            }

ratio_choices_str={'25': lines_std[lines_std_names[6]]+' / '+lines_std[lines_std_names[3]],
                   '26': lines_std[lines_std_names[7]]+' / '+lines_std[lines_std_names[4]],
                   'Ka': lines_std[lines_std_names[4]]+' / '+lines_std[lines_std_names[3]],
                   'Kb': lines_std[lines_std_names[7]]+' / '+lines_std[lines_std_names[6]]}

#minimum nH value to consider for the colormaps
min_nh=5e-1

def corr_factor_lbat(lbat_init):

    '''

    TAILORED FOR 4U1630-47 !

    without uncertainties for now. Fetched manually from the linear regression
    requires lbat_init in Eddington units to work
    '''

    return (lbat_init/10**(1e-4))**(1.01)*10**(-0.77)

def int_rate_to_flux(int_rate,int_rate_err=None,error_percent=68.26):

    '''

    TAILORED FOR 4U1630-47 !

    Without uncertainties for now. Fetched manually from the linear regression.
    returns FLUX values.

    The input is in 30-50keV, and the output in 15-50keV (since the goal is to convert \
    the native ISGRI rate into a 15-50 keV flux usable direcly in the broad HID

    Note that the errors on the regression parameters are extendable to different values
    (mainly because the regression itself assumes gaussian perturbations on the values themselves)
    assuming a gaussian evolution (this works pretty well in practice)

    '''

    #this gives the conversion for the gaussian error parameters
    err_conv=scinorm.ppf((1 + error_percent/100) / 2)


    int_flux_main=(int_rate / 10) ** (1.16) * 10 ** (-8.84)

    if int_rate_err is not None:
        assert np.array(int_rate_err).size in [1,2], 'This function only accepts individual values'
        if np.array(int_rate_err).size==1:
            int_rate_err_use=[int_rate_err,int_rate_err]
        else:
            int_rate_err_use=int_rate_err

        int_rate_low=int_rate-int_rate_err_use[0]
        int_rate_high = int_rate + int_rate_err_use[1]

        #note here that we test both sides of the uncertainty for the power to make things simpler
        int_flux_low=min((int_rate_low/10)**(1.16+0.14*err_conv),(int_rate_low/10)**(1.16-0.15*err_conv))\
                     *10**(-(8.84+0.04*err_conv))
        int_flux_high=max((int_rate_high/10)**(1.16+0.14*err_conv),(int_rate_high/10)**(1.16-0.15*err_conv))\
                      *10**(-(8.84-0.04*err_conv))

        return int_flux_main,int_flux_main-int_flux_low,int_flux_high-int_flux_main
    else:
        return int_flux_main

def bat_rate_to_gamma(rate):
    '''
    without uncertainties for now. Fetched manually from the linear regression
    '''
    return ((np.log10(rate)+3.03)/1.00)+1

def forceAspect(ax,aspect=1):
    
    '''Forces an aspect ratio of 1 in matplotlib axes'''
    
    # lims=(abs(ax.get_xlim()[1]-ax.get_xlim()[0]),abs(ax.get_ylim()[1]-ax.get_ylim()[0]))
    # ax.set_aspect(2)
    

def silent_Simbad_query(source_name):

    '''just here to avoid seing the Simbad warning for when no source is found everytime'''
    
    #converting strings into lists to avoid slicing unique names since we're using the query for lists
    if type(source_name) in [str,np.str_]:
        source_id=[source_name]
    else:
        source_id=source_name
        
    with warnings.catch_warnings():
        # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
        # warnings.filterwarnings('ignore','.*Identifier not found.*',)
        warnings.filterwarnings('ignore',category=UserWarning)
        result=Simbad.query_objects(source_id)
        
    return result
    
@st.cache_data
def load_catalogs():
    
    '''
    Load various catalogs to use for the visualisation script
    '''
    
    print('\nLoading BH catalogs...')
    #for the distance measurements, we import two black hole catalogs
    Vizier.ROW_LIMIT=-1
    ctl_watchdog=Vizier.get_catalogs('watchdog')[1]
    try:
        ctl_blackcat=pd.read_html('https://www.astro.puc.cl/BlackCAT/transients.php')[0]
    except:
        st.warning('BlackCAT webpage offline. Using saved equivalent from 03-2023...')
        ctl_blackcat=pd.read_html('https://web.archive.org/web/20230311033338/https://astro.puc.cl/BlackCAT/transients.php')[1]

    print('\nBH catalogs loading complete.')
    ctl_watchdog_obj=np.array(ctl_watchdog['Name'])
    ctl_watchdog_obj=np.array([elem.replace(' ','') for elem in ctl_watchdog_obj])
    
    ctl_blackcat_obj=ctl_blackcat['Name (Counterpart)'].to_numpy()
    ctl_blackcat_obj=np.array([elem.replace(' ','') for elem in ctl_blackcat_obj])

    '''Snippet adapted from https://stackoverflow.com/questions/65042243/adding-href-to-panda-read-html-df'''

    with st.spinner('Loading MAXI sources...'):
        maxi_url = "http://maxi.riken.jp/top/slist.html"

        with requests.get(maxi_url) as maxi_r:

            maxi_html_table = BeautifulSoup(maxi_r.text,features='lxml').find('table')

        ctl_maxi_df = pd.read_html(str(maxi_html_table), header=0)[0]

        #we create the request for the Simbad names of all MAXI sources here
        #so that it is done only once per script launch as it takes some time to run

        ctl_maxi_df['standard'] = [link.get('href').replace('..',maxi_url[:maxi_url.find('/top')]).replace('.html','_g_lc_1day_all.dat')\
                               for link in [elem for elem in maxi_html_table.find_all('a') if 'star_data' in elem.get('href')]]

        ctl_maxi_simbad=silent_Simbad_query(ctl_maxi_df['source name'])

    with st.spinner('Loading Swift-BAT transients...'):

        bat_url = "https://swift.gsfc.nasa.gov/results/transients/"

        with requests.get(bat_url) as bat_r:
            bat_html_table = BeautifulSoup(bat_r.text, features='lxml').find('table')

        ctl_bat_df = pd.read_html(str(bat_html_table), header=0)[0]

        # we create the request for the Simbad names of all bat sources here
        # so that it is done only once per script launch as it takes some time to run

        #the links are much simpler
        ctl_bat_df['standard'] = [
            'https://swift.gsfc.nasa.gov/results/transients/weak/'+source_name.replace(' ','')+'.lc.txt' \
            for source_name in ctl_bat_df['Source Name']]

        ctl_bat_types=np.array(ctl_bat_df['Source Type'],dtype=str)
        #note: the catalog is way too long to load each time, so we restrict the object categories
        #to avoid that, we restrict the object in certain categories
        excluded_types=['AGN', 'AXP', 'Algol type binary', 'BL Lac',
           'BL Lac LPQ', 'BL Lac/HPQ', 'BL Radio galaxy', 'BY Dra variable',
           'Be Star', 'Beta Lyra binary', 'Blazar', 'Blazar HP',
           'Blazar/Sy1', 'Blazar?', 'CV', 'CV/AM Her', 'CV/DQ Her',
           'CV/Dwarf N', 'CV?', 'Cluster of galaxies', 'Double star',
           'Dwarf nova', 'FSRQ', 'Flare star', 'Galactic center', 'Galaxy',
           'Galaxy cluster', 'Galaxy in group', 'Gamma-ray source',
           'Globular cluster', 'HMXB', 'HMXB/BH', 'HMXB/BHC', 'HMXB/NS',
           'HMXB/Pulsar', 'HMXB/SFXT', 'HMXB/SFXT candidate', 'HMXB/SFXT?',
           'HPQ', 'HXMB/NS', 'HXMB/SFXT', 'Interacting galaxies',
           'LMC source',
           'LMXB/NS', 'LMXB/msPSR', 'LPQ', 'Liner', 'Mira Cet variable',
           'Molecular cloud', 'Multiple star', 'Nova', 'PSR/PWN', 'Pulsar',
           'QSO', 'QSO/FSRQ', 'Quasar', 'Quasar FS', 'RS CVn',
           'RS CVn variable', 'Radio galaxy', 'Radio source', 'SGR', 'SN',
           'SNR', 'SNR/PWN', 'SRC/Gamma', 'SRC/X-ray', 'SRC/gamma', 'Star',
           'Star/Be', 'Sy', 'Sy1', 'Sy1 NL', 'Sy1.2', 'Sy1.5', 'Sy1.5/LPQ',
           'Sy1.8', 'Sy1.9', 'Sy1/LINER', 'Sy1Sy2/Merger', 'Sy2', 'Sy2 HII',
           'Sy2/LINER', 'Symb/WD', 'T Tauri star', 'TDF', 'Transient',
           'Variable star', 'W UMa binary', 'X-ray source',
           'X-ray transient', 'XRB/NS','uQUASAR']

        included_mask=[str(elem) not in excluded_types for elem in ctl_bat_types]

        ctl_bat_simbad = silent_Simbad_query(np.array(ctl_bat_df['Source Name'])[included_mask])

        ctl_bat_df=ctl_bat_df[included_mask]

        #resetting the index to avoid issues
        ctl_bat_df=ctl_bat_df.reset_index()

    return ctl_blackcat,ctl_watchdog,ctl_blackcat_obj,ctl_watchdog_obj,ctl_maxi_df,ctl_maxi_simbad,ctl_bat_df,\
            ctl_bat_simbad

@st.cache_data
def load_integral():

    integral_files = glob.glob('INTEGRAL/Sample/**',recursive=True)

    lc_sw_list=[elem for elem in integral_files if 'ibis_lc_sw.csv' in elem]

    fit_revol_list=[elem for elem in integral_files if 'ibis_fit_revol.csv' in elem]

    lc_sw_dict={}

    #incremental addition to ensure we don't have several lightcurve files for each object
    for elem_lc_sw in lc_sw_list:

        source_name=elem_lc_sw.split('Sample/')[1].split('/')[0].split('/')[-1]

        assert elem_lc_sw not in list(lc_sw_dict.keys()),\
            'Error: several integral lc_sw files for single source'+source_name

        lc_sw_dict[source_name]=pd.read_csv(elem_lc_sw, dtype={'scw': str})

    fit_revol_dict = {}

    # incremental addition to ensure we don't have several lightcurve files for each object
    for elem_fit_revol in fit_revol_list:
        source_name = elem_fit_revol.split('Sample/')[1].split('/')[0].split('/')[-1]

        assert elem_fit_revol not in list(lc_sw_dict.keys()), \
            'Error: several integral lc_sw files for single source' + source_name

        fit_revol_dict[source_name]=pd.read_csv(elem_fit_revol,dtype={'revolution': str})

    #not needed for now, maybe later
    # ctl_integral_simbad=silent_Simbad_query(list(lc_sw_dict.keys()))

    return lc_sw_dict,fit_revol_dict

@st.cache_data
def fetch_bat_lightcurve(ctl_bat_df,_ctl_bat_simbad,name,binning='day'):
    '''

    note: arguments starting with _ are not hashed by st.cache_data

    Attempt to identify a BAT source corresponding to the name given through Simbad identification
    If the attempt is successful, loads a dataframe containing 
    the 1 day or orbit BAT lightcurve for this object, from the BAT website ascii files

    mode gives the type of lightcurve:
        -day for the day average
        -orbit for the individual orbits

    see at https://swift.gsfc.nasa.gov/results/transients/Transient_synopsis.html for the info of each column

    Note: it could be possible to go further using the 8 band snapshot lightcurves of
    https://swift.gsfc.nasa.gov/results/bs157mon/
    '''

    simbad_query = silent_Simbad_query([name[0].split('_')[0]])

    if simbad_query is None:
        return None

    if simbad_query['MAIN_ID'][0] not in _ctl_bat_simbad['MAIN_ID']:
        return None


    # we fetch the script id instead of the direct column number because Simbad erases the columns with no match
    # (-1 because the id starts at 1)
    source_id = _ctl_bat_simbad['SCRIPT_NUMBER_ID'][_ctl_bat_simbad['MAIN_ID'] == simbad_query['MAIN_ID'][0]][0] - 1

    bat_link = ctl_bat_df['standard'][source_id]

    if binning == 'day':
        col_names=['TIME', 'RATE', 'ERROR', 'YEAR', 'DAY', 'STAT_ERR', 'SYS_ERR',
       'DATA_FLAG', 'TIMEDEL_EXPO', 'TIMEDEL_CODED', 'TIMEDEL_DITHERED']
    elif binning =='orbit':
        col_names=['TIME', 'RATE', 'ERROR', 'YEAR', 'DAY', 'MJD', 'TIMEDEL',
       'STAT_ERR', 'SYS_ERR', 'PCODEFR', 'DATA_FLAG', 'DITHER_FLAG']
        
        bat_link = bat_link.replace('.lc.txt', '.orbit.lc.txt')

    #the urls are modified because they can't contain special characters
    bat_link=bat_link.replace('+','p')

    try:
        source_lc = pd.read_csv(bat_link,
                                     skiprows=[0,1,2,4],header=0,names=col_names,
                                     delim_whitespace=True,usecols=range(len(col_names)))
    except:
        try:
            source_lc = pd.read_csv(bat_link.replace('weak/',''),
                                skiprows=[0, 1, 2, 4], header=0, names=col_names,
                                delim_whitespace=True, usecols=range(len(col_names)))
        except:
            st.error('Issue when fetching BAT lightcurve at '+bat_link)
            raise ValueError

    return source_lc

@st.cache_data
def fetch_maxi_lightcurve(ctl_maxi_df,_ctl_maxi_simbad,name,binning='day'):
    
    '''

    note: arguments starting with _ are not hashed by st.cache_data

    Attempt to identify a MAXI source corresponding to the name given through Simbad identification
    If the attempt is successful, loads a dataframe contaiting the 1 day MAXI lightcurve for this object, from the MAXI website

    binning gives the type of lightcurve:
        -day for the day average
        -orbit for the individual orbits

    '''

    simbad_query=silent_Simbad_query([name[0].split('_')[0]])
    
    if simbad_query is None:
        return None
    
    if simbad_query['MAIN_ID'][0] not in _ctl_maxi_simbad['MAIN_ID']:
        return None
    
    #we fetch the script id instead of the direct column number because Simbad erases the columns with no match 
    #(-1 because the id starts at 1)
    source_id=_ctl_maxi_simbad['SCRIPT_NUMBER_ID'][_ctl_maxi_simbad['MAIN_ID']==simbad_query['MAIN_ID']][0]-1

    maxi_link=ctl_maxi_df['standard'][source_id]

    if binning=='orbit':
        maxi_link=maxi_link.replace('lc_1day','lc_1orb')


    source_lc=pd.read_csv(maxi_link,names=['MJDcenter','2-20keV[ph/s/cm2]','err_2-20',
                                              '2-4keV','err_2-4','4-10keV','err_4-10','10-20keV','err_10-20'],sep=' ')

    return source_lc

@st.cache_data
def fetch_rxte_lightcurve(name,dict_rxte=dict_lc_rxte):
    
    '''
    Attempts to fetch a lightcurve for the given namen in the lightcurve rxte dictionnary
    
    the dictionnary has dataframes of all the main band 1-day averaged lightcurves found in the RXTE archive for the sample,
    with simbad ids as keys
    '''
    
    simbad_query=silent_Simbad_query([name[0].split('_')[0]])
    
    
    if simbad_query is None:
        return None

    if simbad_query[0]['MAIN_ID'] not in dict_rxte.keys():
        return None
    
    return dict_rxte[simbad_query[0]['MAIN_ID']]

def plot_lightcurve(dict_linevis,ctl_maxi_df,ctl_maxi_simbad,name,ctl_bat_df,ctl_bat_simbad,
                    lc_integral_sw_dict,fit_integral_revol_dict,dist_factor=None,
                    dict_rxte=dict_lc_rxte,
                    mode='full',display_hid_interval=True,superpose_ew=False,binning='day'):

    '''
    plots various  lightcurves for sources in the Sample if a match is found in RXTE, MAXI or BAT source lists

    mode
    full : full lightcurve in 2-20 keV
    HR_soft : HR in 4-10/2-4 bands
    HR_hard: HR in 10-20/2-4 bands
    BAT: BAT-only high energy lightcurve
    INTEGRAL_band: INTEGRAL, in the "band" band . Can be 30-100,30-50 or 50-100 for equivalent keV bands

    binning:
    for RXTE/MAXI/BAT:
        -day: daily averages
        -orbit: orbit averages (only for MAXI and BAT), 1.5h for MAXI, and the column TIMDEL for BAT
    for INTEGRAL:
        -sw: science window
        -revol: revolution (2.X days)
    BAT orbit lightcurve times have a TIME in seconds after 1/01/2001

    '''
    
    slider_date=dict_linevis['slider_date']
    zoom_lc=dict_linevis['zoom_lc']
    mask_obj=dict_linevis['mask_obj']
    no_obs=dict_linevis['no_obs']
    use_obsids=dict_linevis['use_obsids']

    if no_obs:
        date_list=[]
        instru_list=None
    else:
        #for the two variables below we fetch the first element because they are 1 size arrays containing the info for the only source displayed
        date_list=dict_linevis['date_list'][mask_obj][0]
        instru_list=dict_linevis['instru_list'][mask_obj][0]
    
    #for the EW superposition
    abslines_plot_restrict=dict_linevis['abslines_plot_restrict']
    mask_lines=dict_linevis['mask_lines']
    conf_thresh=dict_linevis['slider_sign']

    maxi_lc_df=fetch_maxi_lightcurve(ctl_maxi_df,ctl_maxi_simbad,name,binning=binning)
    rxte_lc_df=fetch_rxte_lightcurve(name, dict_rxte)

    bat_lc_df=None
    if mode=='BAT':
        bat_lc_df =fetch_bat_lightcurve(ctl_bat_df, ctl_bat_simbad, name, binning=binning)
        if bat_lc_df is None:
            return None

    if mode not in ['full','HR_soft','HR_hard','BAT'] and maxi_lc_df is None and rxte_lc_df is None:
        return None
    
    fig_lc,ax_lc=plt.subplots(figsize=(12,4))

    str_binning_monit=' daily' if binning=='day' else ' individual orbit'
    str_binning_int=' revolution' if binning=='revol' else 'science window ' if binning=='sw' else ''
    #main axis definitions
    # ax_lc.set_xlabel('Time')
    if mode=='full':
        # ax_lc.set_title(name[0]+str_binning_monit+' broad band monitoring')

        # full name is maxi_lc_df.columns[1]
        maxi_y_str='MAXI 2-20 keV rate' if maxi_lc_df is not None else ''
        #full name is rxte_lc_df.columns[1]
        rxte_y_str='RXTE 1.5-12 keV rate/'+str(20 if name[0]=='4U1630-47' else 25) if rxte_lc_df is not None else ''
        ax_lc.set_ylabel(maxi_y_str+(' | ' if maxi_lc_df is not None and rxte_lc_df is not None else '')+rxte_y_str)

    elif mode=='HR_soft':
        ax_lc.set_title(name[0]+str_binning_monit+' Soft Hardness Ratio monitoring')

        maxi_y_str='MAXI counts HR in [4-10]/[2-4] keV' if maxi_lc_df is not None else ''
        rxte_y_str='RXTE band C/(B+A) [5-12]/[1.5-5] keV' if rxte_lc_df is not None else ''
        ax_lc.set_ylabel(maxi_y_str+\
                         ("/" if maxi_lc_df is not None and rxte_lc_df is not None else '')+\
                         rxte_y_str,fontsize=8 if maxi_lc_df is not None and rxte_lc_df is not None else None)
    elif mode=='HR_hard':
        ax_lc.set_title(name[0]+str_binning_monit+' Hard Hardness Ratio monitoring')

        maxi_y_str = 'MAXI counts HR in [10-20]/[2-4] keV' if maxi_lc_df is not None else ''
        ax_lc.set_ylabel(maxi_y_str)

    elif mode=='BAT':
        ax_lc.set_title(name[0]+str_binning_monit+' BAT monitoring')
        bat_y_str = 'BAT count rate (15-50 keV)'
        ax_lc.set_ylabel(bat_y_str)

    elif 'INTEGRAL' in mode:
        ax_lc.set_title(name[0]+(' science window' if binning=='sw' else ' revolution')\
                               +' INTEGRAL monitoring')
        int_y_str = 'INTEGRAL '+('Flux (erg/s)'if binning=='revol' else 'IBIS counts')+' ('+mode.split('_')[1]+' keV)'
        ax_lc.set_ylabel(int_y_str)


    '''MAXI'''
    
    #empty list initialisation to append easily the dates list afterwards no matter 
    num_maxi_dates=[]
    num_rxte_dates=[]
    
    if rxte_lc_df is not None:
            
        #creating a variable for the dates
        num_rxte_dates=mdates.date2num(Time(rxte_lc_df[rxte_lc_df.columns[0]],format='mjd').datetime)
        
        if mode=='full':    
            ax_lc.set_yscale('log')
            
            #plotting the full lightcurve
            #note: the conversion factor is here to convert(ish) the rxte counts into maxi counts
            ax_lc.errorbar(num_rxte_dates,rxte_lc_df[rxte_lc_df.columns[1]]/(20 if name[0]=='4U1630-47' else 25) ,
                           xerr=0.5,yerr=rxte_lc_df[rxte_lc_df.columns[2]]/(20 if name[0]=='4U1630-47' else 25),
                        linestyle='',color='sienna',marker='',elinewidth=0.5,label='RXTE standard counts')
            ax_lc.set_ylim(0.1,ax_lc.get_ylim()[1]*1.1)
    
        if mode=='HR_soft':
            
            '''
            for RXTE, 
            band A is [1.5-3] keV
            band B is [3-5] keV
            band C is [5-12] keV
            
            for the HR, we do C/A+B
            '''
            
            #computing the HR evolution and uncertainties
            
            rxte_hr_denom=rxte_lc_df[rxte_lc_df.columns[3]]+rxte_lc_df[rxte_lc_df.columns[5]]
            rxte_hr=rxte_lc_df[rxte_lc_df.columns[7]]/rxte_hr_denom
            
            rxte_hr_err=abs((rxte_lc_df[rxte_lc_df.columns[8]]/rxte_lc_df[rxte_lc_df.columns[7]]+(rxte_lc_df[rxte_lc_df.columns[4]]+rxte_lc_df[rxte_lc_df.columns[6]])/rxte_hr_denom)*rxte_hr)
            
            ax_lc.set_yscale('log')
            
            ax_lc.set_ylim(0.1,2)
            
            #plotting the full lightcurve
            rxte_hr_errbar=ax_lc.errorbar(num_rxte_dates,rxte_hr,xerr=0.5,yerr=rxte_hr_err,
                        linestyle='',color='sienna',marker='',elinewidth=0.5,label='RXTE HR')
    
            #adapting the transparency to hide the noisy elements
            rxte_hr_alpha_val=1/(20*abs(rxte_hr_err/rxte_hr)**2)
            rxte_hr_alpha=[min(1,elem) for elem in rxte_hr_alpha_val]
            
            #replacing indiviudally the alpha values for each point but the line
            for elem_children in rxte_hr_errbar.get_children()[1:]:
    
                elem_children.set_alpha(rxte_hr_alpha)
                
    if maxi_lc_df is not None:
            
        #creating a variable for the dates
        num_maxi_dates=mdates.date2num(Time(maxi_lc_df[maxi_lc_df.columns[0]],format='mjd').datetime)

        #note that in day binning, the maxi dates values are always a day+0.5 and so represent "actual" days
        xerr_maxi=0.5 if binning=='day' else 1.5/24/2
        
        if mode=='full':    
            ax_lc.set_yscale('log')

            #plotting the full lightcurve
            ax_lc.errorbar(num_maxi_dates,maxi_lc_df[maxi_lc_df.columns[1]],xerr=xerr_maxi,yerr=maxi_lc_df[maxi_lc_df.columns[2]],
                        linestyle='',color='black',marker='',elinewidth=0.5,label='MAXI standard counts')

            #ax_lc.set_ylim(0.05,ax_lc.get_ylim()[1])

            ax_lc.set_yscale('symlog', linthresh=0.05, linscale=0.1)
            ax_lc.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.05))
            ax_lc.set_ylim(0,max(ax_lc.get_ylim()[1],5e-2)*1.1)

        if mode=='HR_soft':
            #computing the HR evolution and uncertainties
            maxi_hr=maxi_lc_df[maxi_lc_df.columns[5]]/maxi_lc_df[maxi_lc_df.columns[3]]
            
            maxi_hr_err=abs((maxi_lc_df[maxi_lc_df.columns[6]]/maxi_lc_df[maxi_lc_df.columns[5]]+maxi_lc_df[maxi_lc_df.columns[4]]/maxi_lc_df[maxi_lc_df.columns[3]])*maxi_hr)
            ax_lc.set_yscale('log')
            
            #ax_lc.set_ylim(0.3,2)
            
            #plotting the full lightcurve
            maxi_hr_errbar=ax_lc.errorbar(num_maxi_dates,maxi_hr,xerr=xerr_maxi,yerr=maxi_hr_err,
                        linestyle='',color='black',marker='',elinewidth=0.5,label='MAXI HR')
    
            #adapting the transparency to hide the noisy elements
            maxi_hr_alpha_val=1/(20*abs(maxi_hr_err/maxi_hr)**2)
            maxi_hr_alpha=[min(1,elem) for elem in maxi_hr_alpha_val]
            
            #replacing indiviudally the alpha values for each point but the line
            for elem_children in maxi_hr_errbar.get_children()[1:]:
    
                elem_children.set_alpha(maxi_hr_alpha)
        if mode=='HR_hard':
            # computing the HR evolution and uncertainties
            maxi_hr = maxi_lc_df[maxi_lc_df.columns[7]] / maxi_lc_df[maxi_lc_df.columns[3]]

            maxi_hr_err = abs((maxi_lc_df[maxi_lc_df.columns[8]] / maxi_lc_df[maxi_lc_df.columns[7]] + maxi_lc_df[
                maxi_lc_df.columns[4]] / maxi_lc_df[maxi_lc_df.columns[3]]) * maxi_hr)
            ax_lc.set_yscale('log')

            ax_lc.set_ylim(0.03, 2)

            # plotting the full lightcurve
            maxi_hr_errbar = ax_lc.errorbar(num_maxi_dates, maxi_hr, xerr=xerr_maxi, yerr=maxi_hr_err,
                                            linestyle='', color='black', marker='', elinewidth=0.5, label='MAXI HR')

            # adapting the transparency to hide the noisy elements
            maxi_hr_alpha_val = 1 / (20 * abs(maxi_hr_err / maxi_hr) ** 2)
            maxi_hr_alpha = [min(1, elem) for elem in maxi_hr_alpha_val]

            # replacing indiviudally the alpha values for each point but the line
            for elem_children in maxi_hr_errbar.get_children()[1:]:
                elem_children.set_alpha(maxi_hr_alpha)

        # ax_lc.set_ylim(0.1,ax_lc.get_ylim()[1])
        
    if bat_lc_df is not None:
        '''
        creating a variable for the dates
        The time construction is different for orbit and day lightcurves in swift
        
        In the day lightcurve, the first column ('TIME') simply gives the MJD and we just need to add .5 to get the full
        day error range
        
        In the orbit lightcurve, the first column ('TIME') gives a number of seconds after 01-01-2001 (aka MJD 51910)
        and the TIME_DEL is the exposure time and
        '''

        if binning=='day':
            num_bat_dates = mdates.date2num(Time(bat_lc_df['TIME'], format='mjd').datetime)+0.5
        elif binning=='orbit':

            #base time value + TIME in seconds + half of the exposure to center the point
            base_date=Time('51910',format='mjd')+TimeDelta(bat_lc_df['TIME'],format='sec')+\
                                                 TimeDelta(bat_lc_df['TIMEDEL'],format='sec')/2
            base_date=base_date.datetime
            num_bat_dates=mdates.date2num(base_date)

        if mode=='BAT':
            # note that in day binning, the bat dates values are always a day+0.5 and so represent "actual" days
            #in orbit we divide the exposure time by two and offset the main value by xerr_bat
            xerr_bat = 0.5 if binning == 'day' else bat_lc_df['TIMEDEL']/86400/2


            ax_lc.set_yscale('symlog', linthresh=0.001, linscale=0.1)
            ax_lc.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.001))

            # plotting the lightcurve
            ax_lc.errorbar(num_bat_dates,
                           bat_lc_df[bat_lc_df.columns[1]], xerr=xerr_bat,
                           yerr=bat_lc_df[bat_lc_df.columns[2]],
                           linestyle='', color='black', marker='', elinewidth=0.5, label='bat standard counts')

            ax_lc.set_ylim(0,max(max(bat_lc_df[bat_lc_df.columns[1]])*1.1,1e-2))

    lc_integral_sw_df=None
    fit_integral_revol_df=None
    if name[0] in list(lc_integral_sw_dict.keys()):
        lc_integral_sw_df=lc_integral_sw_dict[name[0]]
    if name[0] in list(fit_integral_revol_dict):
        fit_integral_revol_df=fit_integral_revol_dict[name[0]]

    if 'INTEGRAL' in mode:

        int_band=mode.split('_')[1]

        if binning=='sw' and lc_integral_sw_df is not None:
            #fetching or computing the count rate values for the given band
            if int_band=='30-100':
                #need to sum two bands here
                counts_int=np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[1]])+\
                           np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[5]])

                counts_err_int=np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[2]])+\
                               np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[6]])

            if int_band=='30-50':
                counts_int=np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[1]])
                counts_err_int=np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[2]])

            if int_band=='50-100':
                counts_int=np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[5]])
                counts_err_int=np.nan_to_num(lc_integral_sw_df[lc_integral_sw_df.columns[6]])

            #creating a different variable for the times
            integral_sw_dates=[Time(elem).datetime for elem in lc_integral_sw_df['ISOT']]
            num_int_sw_dates=[mdates.date2num(elem) for elem in integral_sw_dates]

            ax_lc.set_yscale('symlog', linthresh=1, linscale=0.1)
            ax_lc.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=1))

            # plotting the lightcurve
            ax_lc.errorbar(num_int_sw_dates, counts_int, yerr=counts_err_int,
                           linestyle='', color='black', marker='', elinewidth=0.5, label='ibis standard counts')

            ax_lc.set_ylim(0, max(ax_lc.get_ylim()[1], 1e-2))

        if binning=='revol' and fit_integral_revol_df is not None:

            # fetching or computing the count rate values for the given band
            #note: here we use the same variable counts_int but we plot the FLUX

            if int_band == '30-100':
                # need to sum two bands here
                counts_int = np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[27]]) + \
                             np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[29]])

                counts_err_int = np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[28]]) + \
                                 np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[30]])

            if int_band == '30-50':
                counts_int = np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[27]])
                counts_err_int = np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[28]])

            if int_band == '50-100':
                counts_int = np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[29]])
                counts_err_int = np.nan_to_num(fit_integral_revol_df[fit_integral_revol_df.columns[30]])

            mask_int_ok = ~np.isnan(fit_integral_revol_df['RATE_30.0-50.0'])

            #multiplying by the Eddington factor to get the actual flux
            counts_int=counts_int[mask_int_ok]*dist_factor
            counts_err_int=counts_err_int[mask_int_ok]*dist_factor

            # creating a different variable for the times (centered on the middle of the revolution)
            integral_revol_dates=[(Time(elem)+TimeDelta(1.5,format='jd')).datetime\
                                  for elem in fit_integral_revol_df['ISOT'][mask_int_ok]]
            num_int_revol_dates=np.array([mdates.date2num(elem) for elem in integral_revol_dates])

            ax_lc.set_yscale('symlog', linthresh=float('%.1e'%(1e-10*dist_factor)), linscale=0.1)
            ax_lc.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=float('%.1e'%(1e-10*dist_factor))))

            #see https://www.sciencedirect.com/science/article/pii/S1387647321000166 per new orbit duration
            # plotting the lightcurve
            ax_lc.errorbar(num_int_revol_dates, counts_int, xerr=1.35,yerr=counts_err_int,
                           linestyle='', color='black', marker='', elinewidth=0.5, label='ibis flux')

            #limiting the uncertainties
            ax_lc.set_ylim(0, max(max(2*counts_int), 1e-10))


    #displaying observations with other instruments
    
    label_tel_list=[]
        
    if superpose_ew and not no_obs:
        #creating a second y axis with common x axis
        ax_lc_ew=ax_lc.twinx()
        ax_lc_ew.set_yscale('log')
        ax_lc_ew.set_ylabel('absorption line EW (eV)')
        
        #plotting the detection and upper limits following what we do for the scatter graphs
    
        date_list_repeat=np.array([date_list for repeater in range(sum(mask_lines))])

        instru_list_repeat=np.array([instru_list for repeater in range(sum(mask_lines))])
        
        #these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others            
        val_sign=ravel_ragged(abslines_plot_restrict[4][0]).astype(float)
        
        #standard detection mask (we don't need the intime here since the graph bounds will be cut if needed
        bool_detsign=(val_sign>=conf_thresh) & (~np.isnan(val_sign))
        
        #mask used for upper limits only
        bool_nondetsign=((val_sign<conf_thresh) | (np.isnan(val_sign)))

        #makers for different lines
        marker_style_lines=np.array(['+','x','.','P','X','*'])

        markers_arr=np.array([np.repeat(marker_style_lines[mask_lines][i_line],
                                        len(ravel_ragged(abslines_plot_restrict[4][0][i_line])))\
                              for i_line in range(sum(mask_lines))])

        markers_arr_det=ravel_ragged(markers_arr)[bool_detsign]
        markers_arr_ul=ravel_ragged(markers_arr)[bool_nondetsign]

        x_data_det=mdates.date2num(ravel_ragged(date_list_repeat))[bool_detsign]
        y_data_det=ravel_ragged(abslines_plot_restrict[0][0])[bool_detsign]
        
        y_error_det=np.array([ravel_ragged(abslines_plot_restrict[0][1])[bool_detsign],
                              ravel_ragged(abslines_plot_restrict[0][2])[bool_detsign]]).T
        
        x_data_ul=mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign]
        y_data_ul=ravel_ragged(abslines_plot_restrict[5][0])[bool_nondetsign]
                    
        color_det=[telescope_colors[elem] for elem in ravel_ragged(instru_list_repeat)[bool_detsign]]
        
        color_ul=[telescope_colors[elem] for elem in ravel_ragged(instru_list_repeat)[bool_nondetsign]]

        ax_lc_ew.set_ylim(min(4,min(ravel_ragged(abslines_plot_restrict[0][0])[bool_detsign])),
                          max(100,max(ravel_ragged(abslines_plot_restrict[0][0])[bool_detsign])))

        markers_legend_done_list=[]
        #zipping the errorbars to allow different colors
        for x_data,y_data,y_err,color,marker in zip(x_data_det,y_data_det,y_error_det,color_det,markers_arr_det):

            line_name=lines_std_names[3+np.argwhere(marker_style_lines==marker)[0][0]]
            #not putting the time of the obs as an xerr to avoid display issues
            ax_lc_ew.errorbar(x_data,y_data,xerr=0.,yerr=np.array([y_err]).T,color=color,marker=marker,markersize=4,elinewidth=1,label=lines_std[line_name] if marker not in markers_legend_done_list else '')

            if marker not in markers_legend_done_list:
                markers_legend_done_list+=[marker]

        for x_data,y_data,color,marker in zip(x_data_ul,y_data_ul,color_ul,markers_arr_ul):

            #not putting the time of the obs as an xerr to avoid display issues
            ax_lc_ew.errorbar(x_data,y_data,xerr=0.,yerr=0.05*y_data,marker=marker,color=color,uplims=True,markersize=4,elinewidth=1,capsize=2,alpha=1.,label=lines_std[line_name] if marker not in markers_legend_done_list else '')

            if marker not in markers_legend_done_list:
                markers_legend_done_list+=[marker]
                
    for i_obs,date_obs in enumerate(date_list):
        
        num_date_obs=mdates.date2num(Time(date_obs).datetime)
        
        #we add a condition for the label to only plot each instrument once
        ax_lc.axvline(x=num_date_obs,ymin=0,ymax=1,color=telescope_colors[instru_list[i_obs]],
                        label=instru_list[i_obs]+' exposure' if instru_list[i_obs] not in label_tel_list else '',ls=':',lw=0.5)

        if instru_list[i_obs] not in label_tel_list:
            label_tel_list+=[instru_list[i_obs]]

    #resizing the x axis and highlighting depending on wether we are zooming on a restricted time interval or not
    
    tot_dates_list=[]
    tot_dates_list+=[] if maxi_lc_df is None else num_maxi_dates.tolist()
    tot_dates_list+=[] if rxte_lc_df is None else num_rxte_dates.tolist()
    tot_dates_list+=[] if bat_lc_df is None else  num_bat_dates.tolist()
    if zoom_lc:

        time_range=min(mdates.date2num(slider_date[1]),max(tot_dates_list))-max(mdates.date2num(slider_date[0]),min(tot_dates_list))

        ax_lc.set_xlim(max(mdates.date2num(slider_date[0]),min(tot_dates_list))-time_range/50,
                                             min(mdates.date2num(slider_date[1]),max(tot_dates_list))+time_range/50)
        time_range=min(mdates.date2num(slider_date[1]),max(tot_dates_list))-max(mdates.date2num(slider_date[0]),min(tot_dates_list))
        
    else:
        ax_lc.set_xlim(min(tot_dates_list),max(tot_dates_list))
        time_range=max(tot_dates_list)-min(tot_dates_list)
        
        #highlighting the time interval in the main HID
        
        if display_hid_interval:
            plt.axvspan(mdates.date2num(slider_date[0]),mdates.date2num(slider_date[1]),0,1,color='grey',alpha=0.3,
                        label='HID interval')
        
    #creating an appropriate date axis
    #manually readjusting for small durations because the AutoDateLocator doesn't work well
    if time_range<10:
        date_format=mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    elif time_range<365:
        date_format=mdates.DateFormatter('%Y-%m-%d')
    else:
        date_format=mdates.DateFormatter('%Y-%m')
    # else:
    #     date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())

    ax_lc.xaxis.set_major_formatter(date_format)

    #forcing 8 xticks along the ax
    ax_lc.set_xlim(ax_lc.get_xlim())

    #putting an interval in minutes (to avoid imprecisions when zooming)
    date_tick_inter=int((ax_lc.get_xlim()[1]-ax_lc.get_xlim()[0])*24*60/10)

    #above 10 days ticks
    if date_tick_inter>60*24*10:
        ax_lc.xaxis.set_major_locator(mdates.DayLocator(interval=int(date_tick_inter/(24*60))))
    #above 1 day ticks
    elif date_tick_inter>60*24:
        ax_lc.xaxis.set_major_locator(mdates.HourLocator(interval=int(date_tick_inter/60)))
    else:
        ax_lc.xaxis.set_major_locator(mdates.MinuteLocator(interval=date_tick_inter))

    #and offsetting if they're too close to the bounds because otherwise the ticks can be missplaced
    if ax_lc.get_xticks()[0]-ax_lc.get_xlim()[0]>date_tick_inter/(24*60)*3/4:
        ax_lc.set_xticks(ax_lc.get_xticks()-date_tick_inter/(2*24*60))

    if ax_lc.get_xticks()[0]-ax_lc.get_xlim()[0]<date_tick_inter/(24*60)*1/4:
        ax_lc.set_xticks(ax_lc.get_xticks()+date_tick_inter/(2*24*60))

    # ax_lc.set_xticks(ax_lc.get_xticks()[::2])
                    
    for label in ax_lc.get_xticklabels(which='major'):
        label.set(rotation=0 if date_tick_inter>60*24*10 else 45, horizontalalignment='center')

        #prettier but takes too much space
        # label.set(rotation=45, horizontalalignment='right')

    #contamination zone for 4U
    if name[0]=='4U1630-47' and mode in ['full','HR_soft','HR_hard']:
        conta_start=mdates.date2num(Time('2018-12-10').datetime)
        conta_end=mdates.date2num(Time('2019-06-20').datetime)
        plt.axvspan(xmin=conta_start,xmax=conta_end,color='grey',zorder=1000,alpha=0.3)

    ax_lc.legend(loc='upper left',ncols=2)

    if superpose_ew:
        ax_lc_ew.legend(loc='upper right')
    
    return fig_lc

# @st.cache_data
def dist_mass(dict_linevis,use_unsure_mass_dist=True):

    '''
    Fetches local data and blackcat/watchdog to retrieve the mass and distances of sources.
    Local is > BC/WD because it is (currently) more up to date.

    -use_unsure_mass_dist: use mass and distance measurements set as unsure in the local dictionnary
        (with 0 for the last element of their measurement array)
    '''
    
    ctl_blackcat=dict_linevis['ctl_blackcat']
    ctl_blackcat_obj=dict_linevis['ctl_blackcat_obj']
    ctl_watchdog=dict_linevis['ctl_watchdog']
    ctl_watchdog_obj=dict_linevis['ctl_watchdog_obj']
    names=dict_linevis['obj_list']
    
    d_obj=np.array([None]*len(names))
    m_obj=np.array([None]*len(names))
    
    for i in range(len(names)):
        d_obj[i]='nan'

        if names[i] in dist_dict:
            try:
                if dist_dict[names[i]][3]==1 or use_unsure_mass_dist:
                    #putting manual/updated distance values first
                    d_obj[i]=dist_dict[names[i]][0]
            except:
                breakpoint()
                pass
        else:
            
            obj_row=None
            #searching for the distances corresponding to the object namess in the first (most recently updated) catalog
            for elem in ctl_blackcat_obj:
                if names[i] in elem:
                    obj_row=np.argwhere(ctl_blackcat_obj==elem)[0][0]
                    break                    

            # breakpoint()
            if obj_row is not None:

                obj_d_key = ctl_blackcat.iloc[obj_row]['d [kpc]']

                if not (type(obj_d_key)==str or np.isnan(obj_d_key)) and \
                    ('≥' not in obj_d_key and '>' not in obj_d_key):

                    print('New measurement found in BlackCAT, not found in the biblio. Please check.')
                    breakpoint()
                    d_obj[i]=ctl_blackcat.iloc[obj_row]['d [kpc]']

                    #formatting : using only the main values + we do not want to use this catalog's results if they are simply upper/lower limits
                    d_obj[i]=str(d_obj[i])
                    d_obj[i]=d_obj[i].split('/')[-1].split('±')[0].split('~')[-1].split('∼')[-1]

                    if '≥' in d_obj[i] or '>' in d_obj[i] or '<' in d_obj[i] or '≤' in d_obj[i]:
                        d_obj[i]='nan'

                    if '-' in d_obj[i]:
                        if '+' in d_obj[i]:
                            #taking the mean value if it's an uncertainty
                            d_obj[i]=float(d_obj[i].split('+')[0].split('-')[0])
                        else:
                            #taking the mean if it's an interval
                            d_obj[i]=(float(d_obj[i].split('-')[0])+float(d_obj[i].split('-')[-1]))/2
            
            
            #searching in the second catalog if nothing was found in the first one
            if d_obj[i]=='nan':
                if len(np.argwhere(ctl_watchdog_obj==names[i]))!=0:
                    
                    #watchdog assigns by default 5+-3 kpc to sources with no distance estimate so we need to check for that
                    #(there is no source with an actual 5kpc distance)
                    watchdog_d_val=float(ctl_watchdog[np.argwhere(ctl_watchdog_obj==names[i])[0][0]]['Dist1'])

                    #these ones are false/outdated
                    # here same, the lower limit quoted in WATCHDOG has been disproved in Charles19
                    watchdog_d_exclu=['SwiftJ1357.2-0933']

                    if names[i] not in watchdog_d_exclu and watchdog_d_val not in [5.,8.]:

                        print('New measurement found in WATCHDOG, not found in the biblio. Please check.')
                        breakpoint()
                        d_obj[i]=watchdog_d_val
                        
        if d_obj[i]=='nan':
            #giving a default value of 8kpc to the objects for which we do not have good distance measurements
            d_obj[i]=8

        else:
            d_obj[i]=float(d_obj[i])

        #fixing the source mass at 8 solar Masses if not in the local list since we have very few reliable estimates
        # of the BH masses anyway except for NS whose masses are in a dictionnary
        if names[i] in mass_dict and (mass_dict[names[i]][3]==1 or use_unsure_mass_dist):
            m_obj[i]=mass_dict[names[i]][0]
        else:
            m_obj[i]=8
    
    return d_obj,m_obj

#@st.cache_data
def obj_values(file_paths,E_factors,dict_linevis):
    
    '''
    Extracts the stored data from each value line_values file. 
    Merges all the files with the same object name
    
    the visual_line option give more information but requires a higher position in the directory structure
    '''
    
    obj_list=dict_linevis['obj_list']
    cameras=dict_linevis['args_cam']
    expmodes=dict_linevis['expmodes']
    multi_obj=dict_linevis['multi_obj']
    visual_line=True
    
    obs_list=np.array([None]*len(obj_list))
    lval_list=np.array([None]*len(obj_list))
    l_list=np.array([None]*len(obj_list))
    
    date_list=np.array([None]*len(obj_list))
    instru_list=np.array([None]*len(obj_list))
    exptime_list=np.array([None]*len(obj_list))
    fitmod_broadband_list=np.array([None]*len(obj_list))
    epoch_obs_list=np.array([None]*len(obj_list))
    flux_high_list=np.array([None]*len(obj_list))
    # ind_links=np.array([None]*len(obj_list))

    for i in range(len(obj_list)):
        
        #matching the line paths corresponding to each object
        if visual_line:    
            curr_obj_paths=[elem for elem in file_paths if '/'+obj_list[i]+'/' in elem]
        else:
            curr_obj_paths=file_paths
            
        curr_E_factor=E_factors[i]
        
        store_lines=[]

        lineval_paths_arr=[]
        
        for elem_path in curr_obj_paths:
            
            #opening the values file
            with open(elem_path) as store_file:

                store_lines_single_full=store_file.readlines()[1:]

                #only keeping the lines with selected cameras and exposures
                #for NICER observation, there's no camera to check so we pass directly

                if cameras=='all':
                    store_lines_single=store_lines_single_full
                    store_lines+=store_lines_single_full
                else:
                    #will need improvement for NuSTAR probably
                    store_lines_single=[elem for elem in store_lines_single_full if\
                                        '_' in elem.split('\t')[0] and \
                                        np.any([elem_cam in elem.split('\t')[0].split('_')[1] for elem_cam in cameras])]

                    try:
                        #checking if it's an XMM file and selecting exposures if so
                        if 'NICER' not in elem_path and store_lines_single[0].split('\t')[0].split('_')[1] in ['pn','mos1','mos2']:
                            store_lines_single=[elem for elem in store_lines_single if elem.split('\t')[0].split('_')[3]\
                                                in expmodes]
                    except:
                        breakpoint()

                    store_lines+=store_lines_single

                lineval_paths_arr+=[np.repeat(elem_path,len(store_lines_single))]

        store_lines=ravel_ragged(np.array(store_lines))

        # used to allow separate observation folders at the bigbatch level
        lineval_paths_arr=ravel_ragged(lineval_paths_arr)

        #and storing the observation ids and values 
        curr_obs_list=np.array([None]*len(store_lines))
        curr_lval_list=np.array([None]*len(store_lines))
        curr_l_list=np.array([None]*len(store_lines))

        for l,line in enumerate(store_lines):

            curr_line=line.split('\t')
                
            curr_obs_list[l]=curr_line[0]
            #converting the lines into something usable
            curr_lval_list[l]=[literal_eval(elem.replace(',','').replace(' ',',')) if elem!='' else [] for elem in curr_line[1:]]
                
            #separing the flux values for easier plotting
            curr_l_list[l]=np.array(curr_lval_list[l][-1])*curr_E_factor
            
            #taking them off from the values lists
            curr_lval_list[l]=curr_lval_list[l][:-1]
        
        #creating indexes links
        obs_list[i]=curr_obs_list
        lval_list[i]=curr_lval_list
        l_list[i]=np.array([elem for elem in curr_l_list])

        curr_date_list=np.array([None]*len(obs_list[i]))
        curr_instru_list=np.array([None]*len(obs_list[i]))
        curr_exptime_list=np.array([None]*len(obs_list[i]))
        curr_fitmod_broadband_list=np.array([None]*len(obs_list[i]))
        curr_epoch_obs_list=np.array([None]*len(obs_list[i]))
        curr_flux_high_list=np.array([None]*len(obs_list[i]))
        #fetching spectrum informations

        if visual_line:

            '''
            importing xspec while online can be very complicated, so we only import it here to make the dumps
            The dumps will never be made online so that there shouldn't be a problem
            
            
            Note that xspec still needs to be imported to load the pickles, but we give the results in arrays
            so that the visual_line dumps don't need it afterwards
            '''
            from xspec_config_multisp import load_fitmod,parse_xlog

            for i_obs,obs in enumerate(obs_list[i]):

                lineval_path=lineval_paths_arr[i_obs]

                '''Storing the list of exposures'''

                # fetching the full list of epoch obs to get the exposure later
                with open('/'.join(lineval_path.split('/')[:-1]) + '/summary_line_det.log') as summary:
                    summary_lines = summary.readlines()[1:]

                if 'XMM/' in lineval_path or 'Chandra/' in lineval_path:
                    #safeguard for the previous way the summaries were handled
                    #kept separate from the rest so this is done consciously
                    summary_obs_line = [elem for elem in summary_lines if obs in ('_').join(elem.split('\t'))]
                else:
                    summary_obs_line = [elem for elem in summary_lines if obs in elem]

                assert len(summary_obs_line) == 1, 'Error in observation summary matching'

                epoch_tab = summary_obs_line[0].split('\t')[0]
                # note: this is for backward compatiblity with old NICER reduction without gti-level split

                if '[' not in epoch_tab:
                    elem_epoch_obs_list = [epoch_tab]
                    multi_sat=False
                else:
                    elem_epoch_obs_list = expand_epoch(literal_eval(summary_obs_line[0].split('\t')[0]))

                    #the multi sat flag is here to avoid fetching informations which assume that a single suffix
                    #works for all the spectra

                    #note: this method is more complicated and won't need to be used as long as we don't only
                    #allow for multi instrument spectral analysis in multi folders
                    #failsafe against summaries (such a for nicer) without extension information
                    # if summary_obs_line[0].count('\t')<2 or 'Suzaku/' in lineval_path:
                    #     multi_sat=False
                    # else:
                    #     multi_sat=len(literal_eval(summary_obs_line[0].split('\t')[1]))>1

                    multi_sat='/multi/' in lineval_path


                curr_epoch_obs_list[i_obs] = elem_epoch_obs_list

                # the path+ obs prefix for all the stored files in the lineplots_X folders
                obs_path_prefix = lineval_path[:lineval_path.rfind('/') + 1] + obs

                #storing the high flux array if it exists
                if os.path.isfile(obs_path_prefix+'_main_spflux_high.txt'):
                    curr_flux_high_list[i_obs]=np.loadtxt(obs_path_prefix+'_main_spflux_high.txt')

                '''Storing the fitmod'''

                '''
                Here, we need both the txt for the main values and the fitmod for the errors stored in the
                .errors method
                
                Extremely inneficient but this should work for everything
                
                Note that since we take the values and the errors from the broadband fits, the errors are not
                from a chain
                '''

                txtmod_path=obs_path_prefix+'_mod_broadband_post_auto.txt'

                #safeguard for the few nustar runs where I deleted the products by mistake
                if os.path.isfile(txtmod_path):

                    with open(txtmod_path) as txt_file:
                        txt_lines=txt_file.readlines()

                    #fetching the main values of the parameters
                    mainmod_mainpars=parse_xlog(txt_lines,return_pars=True)[0]

                    fitmod_path=obs_path_prefix+'_fitmod_broadband_post_auto.pkl'

                    assert os.path.isfile(fitmod_path),'broadband_post_auto fitmod missing'

                    #storing the post_hid fitmod to have access to the full model
                    #note that the last directory has to be unchanged since the save otherwise there's a
                    #logfile trace somewhere who's crashing

                    try:
                        elem_fitmod=load_fitmod(fitmod_path)
                    except:

                        print('logfile bug in '+fitmod_path)
                        print('fixing...')
                        #Annoying bug when at some point the continuum fitmod was saved in the autofit with the
                        # io of the first xspec log still on it. This needs to be cleaned otherwise they can only load
                        # in the directory where they were created and if this file exists

                        currdir=os.getcwd()

                        os.chdir('/'.join(fitmod_path.split('/')[:-2]))

                        try:
                            elem_fitmod=load_fitmod('/'.join(fitmod_path.split('/')[-2:]))
                        except:
                            breakpoint()
                            print("if the pyxspec path is loaded that shouldn't happen, need to investigate")

                        from xspec import Xset
                        #we also need to restore the data to resave the fitmod afterwards

                        Xset.restore('/'.join(fitmod_path.split('/')[-2:]).replace('fitmod','mod').replace('.pkl','.xcm'))

                        #updating the component fitmod
                        for elem_comp in elem_fitmod.complist:
                            elem_comp.fitmod=elem_fitmod

                        #and resaving
                        elem_fitmod.dump('/'.join(fitmod_path.split('/')[-2:]))

                        #returning to the current directory
                        os.chdir(currdir)

                        #and reloading the fitmod
                        elem_fitmod=load_fitmod(fitmod_path)

                    #errors ravelled on the datagroup dimension to make things easier
                    elem_fitmod_errors=[subelem.tolist() for elem in elem_fitmod.errors for subelem in elem]

                    '''
                    There's two "wrong" xspec errors to readjust:
                    The upper error can appear as a negative value (the main val) it's pegged
                    The lower error can be equal the main value  when it' pegged at 1 
                    We need to correct for both so we edit the array directly
                    '''

                    tot_error_arr=np.array([[mainmod_mainpars[i_par]]+elem_fitmod_errors[i_par]\
                                           for i_par in range(len(mainmod_mainpars))]).T

                    tot_error_arr[1]=np.where(tot_error_arr[0]==tot_error_arr[1],0,tot_error_arr[1])
                    tot_error_arr[2]=tot_error_arr[2].clip(0)

                    tot_error_arr=tot_error_arr.T

                    #creating a dictionnary of all component names and asociated parameter values
                    #note that we offset the parlist per 1 to get back to parameters ids
                    comp_par_fitmod_broadband_dict={comp.compname:tot_error_arr[np.array(comp.parlist)-1] for comp in\
                                              [elem for elem in elem_fitmod.includedlist if elem is not None]}

                    curr_fitmod_broadband_list[i_obs]=comp_par_fitmod_broadband_dict

                if multi_sat:
                    #skipping the computations below because we don't care for now
                    curr_instru_list[i_obs]='multi'
                    #loading the first obs to get some data
                    # note: the first item of the literal_eval list here is the suffix of the first file
                    filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+('_gti_event_spec' if 'xis' in obs else '_sp')+ \
                             literal_eval(summary_obs_line[0].split('\t')[1])[0]

                    #ensuring no issue with suzaku files without a header
                    filepath=filepath.replace('xis0_xis2_xis3','xis1').replace('xis0_xis3','xis1')


                else:

                    if len(obs.split('_'))<=1:

                        if 'source' in obs.split('_')[0]:
                            #this means a Swift observation
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_grp_opt.pi'
                            curr_instru_list[i_obs]='Swift'

                        elif obs.startswith('nu'):
                            #this means a NuSTAR observation
                            #we take the directory structure from the according file in curr_obj_paths
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_sp_src_grp_opt.pha'
                            curr_instru_list[i_obs]='NuSTAR'
                        else:
                            #this means a NICER observation
                            #we take the directory structure from the according file in curr_obj_paths
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_sp_grp_opt.pha'
                            curr_instru_list[i_obs]='NICER'

                            epoch_sp_list=[elem+'_sp_grp_opt.pha' for elem in elem_epoch_obs_list]


                    else:

                        if obs.split('_')[1] in ['pn','mos1','mos2']:

                            #we take the directory structure from the according file in curr_obj_paths
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_sp_src_grp_20.ds'
                            curr_instru_list[i_obs]='XMM'

                        elif obs.split('_')[1]=='heg':

                            #we take the directory structure from the according file in curr_obj_paths
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_grp_opt.pha'
                            curr_instru_list[i_obs]='Chandra'

                        elif 'xis' in obs:
                            #we take the directory structure from the according file in curr_obj_paths
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_gti_event_spec_src_grp_opt.pha'

                            #getting non stacked files for megumi files to be able to read the header
                            filepath=filepath.replace('xis0_xis2_xis3','xis_1').replace('xis0_xis3','xis_1')

                            curr_instru_list[i_obs]='Suzaku'

                        elif obs.split('_')[1] in ['0','1']:
                            #note that this only works for xis1 files as xis0_xis2 are merged and the header had been removed
                            #we take the directory structure from the according file in curr_obj_paths
                            filepath='/'.join(lineval_path.split('/')[:-2])+'/'+obs+'_sp_grp_opt.pha'
                            curr_instru_list[i_obs]='Suzaku'

                        #failsafe if the file is an XMM spectrum a _full folder instead
                        if not os.path.isfile(filepath):

                            filepath='/'.join(lineval_path.split('/')[:-3])+'_full/'+lineval_path.split('/')[-3]+'/'+obs+'_sp_src_grp_20.ds'

                            if not os.path.isfile(filepath):
                                #should not happen
                                breakpoint()
                                print('issue with identifying obs sp path')

                #summing the exposures for NICER
                if curr_instru_list[i_obs]=='NICER':

                    curr_exptime_list[i_obs] = 0

                    for elem_file in epoch_sp_list:
                        with fits.open('/'.join(lineval_path.split('/')[:-2]) + '/' + elem_file) as hdul:
                            curr_exptime_list[i_obs] += hdul[1].header['EXPOSURE']

                try:
                    with fits.open(filepath) as hdul:

                        if curr_instru_list[i_obs]!='NICER':

                            curr_exptime_list[i_obs]=hdul[1].header['EXPOSURE']

                        if curr_instru_list[i_obs] in ['NICER','NuSTAR']:

                            start_obs_s = hdul[1].header['TSTART'] +\
                                          (hdul[1].header['TIMEZERO'] if curr_instru_list[i_obs]=='NICER' else 0)
                            # saving for titles later
                            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

                            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

                            curr_date_list[i_obs] = str(obs_start.isot)

                        else:
                            try:
                                curr_date_list[i_obs]=hdul[0].header['DATE-OBS']
                            except:
                                try:
                                    curr_date_list[i_obs]=hdul[1].header['DATE-OBS']
                                except:
                                    curr_date_list[i_obs]=Time(hdul[1].header['MJDSTART'],format='mjd').isot
                except:
                    breakpoint()
                    print('issue with obs fits handling')

        date_list[i]=curr_date_list
        instru_list[i]=curr_instru_list
        exptime_list[i]=curr_exptime_list
        fitmod_broadband_list[i]=curr_fitmod_broadband_list
        epoch_obs_list[i]=curr_epoch_obs_list
        flux_high_list[i]=curr_flux_high_list
        # ind_links[i]=os.path.join(os.getcwd(),obj_dir)+'/'+np.array(obs_list[i])+'_recap.pdf'
    
    if multi_obj:
        l_list=np.array([elem for elem in l_list],dtype=object)
    else:
         l_list=np.array([elem for elem in l_list])

    return obs_list,lval_list,l_list,date_list,instru_list,exptime_list,fitmod_broadband_list,epoch_obs_list,\
        flux_high_list

#@st.cache_data
def abslines_values(file_paths,dict_linevis,only_abs=False,obsid=None):

    '''
    Extracts the stored data from each autofit_values file

    if obsid is set to an obsid string or identifier, only returns the values of the line(s)
    containing that obsid
    '''

    #converting non-arrays (notably a single string) into iterables
    if type(file_paths) not in (np.array,list,tuple):
        file_paths_use=[file_paths]
    else:
        file_paths_use=file_paths

    cameras=dict_linevis['args_cam']
    expmodes=dict_linevis['expmodes']
    visual_line=dict_linevis['visual_line']
    if visual_line:
        obj_list = dict_linevis['obj_list']
        abslines_inf=np.array([None]*len(obj_list))
        autofit_inf=np.array([None]*len(obj_list))
    else:
        obj_list=['current']
        abslines_inf=np.array([None])
        autofit_inf=np.array([None])

    for i in range(len(obj_list)):
            
        #matching the line paths corresponding to each object
        if visual_line:    
            curr_obj_paths=[elem for elem in file_paths_use if '/'+obj_list[i]+'/' in elem]
        else:
            curr_obj_paths=file_paths_use
        
        store_lines=[]

        for elem_path in curr_obj_paths:
            
            #opening the values file
            with open(elem_path) as store_file:
                
                store_lines_single_full=store_file.readlines()[1:]
                
                #only keeping the lines with selected cameras and exposures
                #for NICER observation, there's no camera to check so we pass directly

                #restricting to the given obsid if asked to
                if obsid is not None:
                    store_lines_single_full=[elem_line for elem_line in store_lines_single_full if obsid in elem_line]

                if cameras=='all':
                    store_lines_single=store_lines_single_full
                    store_lines+=store_lines_single_full
                else:
                    #will need improvement for NuSTAR probably
                    store_lines_single=[elem for elem in store_lines_single_full if\
                                        '_' in elem.split('\t')[0] and \
                                        np.any([elem_cam in elem.split('\t')[0].split('_')[1] for elem_cam in cameras])]

                    try:
                        #checking if it's an XMM file and selecting exposures if so
                        if 'NICER' not in elem_path and store_lines_single[0].split('\t')[0].split('_')[1] in ['pn','mos1','mos2']:
                            store_lines_single=[elem for elem in store_lines_single if elem.split('\t')[0].split('_')[3]\
                                                in expmodes]
                    except:
                        breakpoint()

                    store_lines+=store_lines_single


        store_lines=ravel_ragged(np.array(store_lines))
        
        #and storing the absline values and fit parameters
        curr_abslines_infos=np.array([None]*len(store_lines))
        curr_autofit_infos=np.array([None]*len(store_lines))
        
        for l,line in enumerate(store_lines):
            
            curr_line=line[:-1]
            curr_line=curr_line.split('\t')

            #converting the lines into something usable
            curr_abslines_infos[l]=np.array([None]*len(curr_line[1:-2]))
            curr_autofit_infos[l]=np.array([None,None])
            
            for  m in range(len(curr_line[1:-2])):


                elem_array=np.array(literal_eval(curr_line[m+1].replace(',','').replace(' ',',')))\
                    if curr_line[m+1]!='' else None
                
                #inverting the sign of the blueshift values                    
                if m==1:

                    #note: o is the line number, p is the uncertaintiy
                    for o in range(elem_array.shape[0]):
                        
                        if elem_array[o][0] is not None:
                            #to avoid negative uncertainties we swap the positive and negative uncertainties and 
                            elem_array[o]=elem_array[o][[0,2,1]]

                            elem_array[o][0]*=-1
                
                curr_abslines_infos[l][m]=elem_array
                    
                # except:
                #     
                    
                #     #covering the detections before I fixed the error of not converting the array to a list
                #     
                #     curr_abslines_infos[l][m]=np.array(literal_eval(','.join(curr_line[m+1].split())))\
                #         if curr_line[m+1]!='' else None

            #separing the flux values for easier plotting
            for m in [0,1]:
                curr_autofit_infos[l][m]=np.array(literal_eval(curr_line[-2+m].replace(',','').replace(' ',',')))\
                    if curr_line[-2+m]!='' else None
                
        abslines_inf[i]=curr_abslines_infos
        autofit_inf[i]=curr_autofit_infos

    if only_abs:
        return abslines_inf
    else:
        return abslines_inf,autofit_inf


#@st.cache_data
def values_manip(abslines_infos,dict_linevis,autofit_infos,lum_list_infos,mask_include=None):
    
    range_absline=dict_linevis['range_absline']
    n_infos=dict_linevis['n_infos']
    obj_list=dict_linevis['obj_list']
    multi_obj=dict_linevis['multi_obj']
    incl_dict_use=dict_linevis['incl_dict_use']

    n_obj=len(obj_list)

    if mask_include is None:
        #default ragged full True array
        mask_include_use=np.array([np.array([True]*len(abslines_infos[i])) for i in range(n_obj)],dtype=bool)
    else:
        #this is necessary for cases with a single telescope and a single object to avoid issues
        if len(mask_include)==1:
            mask_include_use=mask_include.astype(bool)
        else:
            mask_include_use=mask_include

    abslines_inf=np.array([abslines_infos[i_obj][mask_include_use[i_obj]] for i_obj in range(n_obj)],dtype=object)
    autofit_inf = np.array([autofit_infos[i_obj][mask_include_use[i_obj]] for i_obj in range(n_obj)], dtype=object)
    lum_list= np.array([lum_list_infos[i_obj][mask_include_use[i_obj]] for i_obj in range(n_obj)], dtype=object)

    #so we can translate these per lines instead
    abslines_inf_line=np.array([None]*len(range_absline))
    
    if len([elem for elem in abslines_inf if elem is not None])>0:
        for i_line in range(len(range_absline)):
            arr_part_line=np.array([None]*len(abslines_inf))
            for i_obj in range(len(abslines_inf)):

                arr_part_obs=np.array([None]*len(abslines_inf[i_obj]))
                    
                for i_obs in range(len(abslines_inf[i_obj])):
                    #testing if the line detection was skipped
                    if np.array(abslines_inf[i_obj][i_obs])[1] is not None:
                        
                        #here to have an easier time using the data, we add nan uncertainties to the significance in order to keep 
                        #regular shaped arrays that we can tranpose at will
                        array_obs=np.array([elem if len(np.shape(elem))==2\
                                            else np.array([[elem[i],None,None] for i in range(len(elem))]).astype(float)\
                                                for elem in abslines_inf[i_obj][i_obs]])
                        
                        arr_part_obs[i_obs]=np.transpose(array_obs,axes=[1,0,2])[i_line]
                        
                    else:
                        arr_part_obs[i_obs]=np.repeat(np.nan,repeats=n_infos*3).reshape((n_infos,3))
                        
                arr_part_line[i_obj]=np.array([elem for elem in arr_part_obs])
            abslines_inf_line[i_line]=np.array([elem for elem in arr_part_line],dtype=object)
        
        abslines_inf_line=np.array([elem for elem in abslines_inf_line])

        #reorganizing the array to ensure it's non regular
        abslines_inf_line_use=np.array([[None]*len(abslines_inf)]*len(range_absline))

        for i_1 in range(len(range_absline)):
            for i_2 in range(len(abslines_inf)):
                abslines_inf_line_use[i_1][i_2]=abslines_inf_line[i_1][i_2]

        abslines_inf_line=abslines_inf_line_use
    
    abslines_inf_obj=np.array([None]*len(obj_list))
    
    for i_obj in range(len(obj_list)):
        try:
            abslines_inf_obj[i_obj]=np.transpose(np.array([elem for elem in abslines_inf_line.T[i_obj]]),axes=(3,2,0,1))
        except:
            breakpoint()
            print('should not happen')

    #creating absline_plot

    abslines_plt=np.array([None]*n_infos)
    for i_info in range(len(abslines_plt)):

        arr_part_uncert=np.array([None]*3)
        for i_uncert in range(len(arr_part_uncert)):

            arr_part_line=np.array([None]*len(range_absline))
            for i_line in range(len(arr_part_line)):

                arr_part_obj=np.array([None]*len(abslines_inf))
                for i_obj in range(len(arr_part_obj)):

                    arr_part_obs=np.array([None]*len(abslines_inf[i_obj]))
                    for i_obs in range(len(arr_part_obs)):

                        arr_part_obs[i_obs]=abslines_inf_line[i_line][i_obj][i_obs][i_info][i_uncert]

                    arr_part_obj[i_obj]=arr_part_obs

                arr_part_line[i_line]=arr_part_obj

            arr_part_uncert[i_uncert]=np.array([elem for elem in arr_part_line])

        abslines_plt[i_info]=arr_part_uncert
    
    #re-organizing the array to ensure it's non regular
    abslines_plt_true=np.array([[[[None]*len(abslines_inf)]*len(range_absline)]*3]*n_infos)

    if len([elem for elem in abslines_inf_line if elem is not None])>0:

        for i_1 in range(n_infos):
            for i_2 in range(3):
                for i_3 in range(len(range_absline)):
                    for i_4 in range(len(abslines_inf)):
                        abslines_plt_true[i_1][i_2][i_3][i_4]=abslines_plt[i_1][i_2][i_3][i_4]

        abslines_plt=abslines_plt_true
    else:
        sys.exit()
    
    '''
    in the plt form, the new order is:
        -the info (5 rows, ew/bshift/Del-C/sign/flux)
        -it's uncertainty (3 rows, main value/neg uncert/pos uncert,useless for the Del-C and sign)
        -each absorption line
        -the number of sources
        -the number of obs for each source
    '''

    #### Energy
    #creating the energy distribution from the blueshifts        
    abslines_e=deepcopy(abslines_plt[1])
    
    for i_line in range_absline:
        for i_obj in range(len(abslines_e[0][i_line])):
            for i_obs in range(len(abslines_e[0][i_line][i_obj])):
                
                conv_line_e=lines_e_dict[lines_std_names[3+i_line]][0]
                
                #replacing the bshift value by the energy
                abslines_e[0][i_line][i_obj][i_obs]=None if abslines_plt[1][0][i_line][i_obj][i_obs]==None else\
                                                        conv_line_e*(1+abslines_plt[1][0][i_line][i_obj][i_obs]/c_light)
                    
                #same with the uncertainties 
                abslines_e[1][i_line][i_obj][i_obs]=\
                    None if abslines_plt[1][1][i_line][i_obj][i_obs]==None else\
                    abslines_plt[1][1][i_line][i_obj][i_obs] if type(abslines_plt[1][1][i_line][i_obj][i_obs])==str else\
                    abslines_e[0][i_line][i_obj][i_obs]-conv_line_e*\
                    (1+(abslines_plt[1][0][i_line][i_obj][i_obs]-abslines_plt[1][1][i_line][i_obj][i_obs])/c_light)
                    
                abslines_e[2][i_line][i_obj][i_obs]=\
                    None if abslines_plt[1][2][i_line][i_obj][i_obs]==None else\
                        abslines_plt[1][1][i_line][i_obj][i_obs] if type(abslines_plt[1][2][i_line][i_obj][i_obs])==str else\
                    conv_line_e*(1+(abslines_plt[1][0][i_line][i_obj][i_obs]+abslines_plt[1][2][i_line][i_obj][i_obs])/c_light)-\
                    abslines_e[0][i_line][i_obj][i_obs]
            
    #### Flux
    #creating a flux values variable with the same shape than the others
    '''
    Reminder that the flux array indexing is:
    0: full HID band (depends ond the instrument and analysis so not used)
    
    1: 3.-6. keV
    2: 6.-10. keV
    3: 1.-3. keV
    4: 3.-10. keV
    
    '''

    if multi_obj:

        lum_plt=np.array([None]*3)
        for i_uncert in range(len(lum_plt)):
            
            arr_part_band=np.array([None]*5)
            for i_band in range(len(arr_part_band)):
                
                arr_part_obj=np.array([None]*len(abslines_inf))
                for i_obj in range(len(arr_part_obj)):
                    
                    arr_part_obs=np.array([None]*len(abslines_inf[i_obj]))
                    for i_obs in range(len(arr_part_obs)):

                        arr_part_obs[i_obs]=lum_list[i_obj][i_obs][i_uncert][i_band]

                    #avoiding negative uncertainties (shouldn't happen)
                    if i_uncert!=0:
                        arr_part_obs=arr_part_obs.clip(0)
                        
                    arr_part_obj[i_obj]=arr_part_obs
                    
                arr_part_band[i_band]=arr_part_obj
            
            lum_plt[i_uncert]=arr_part_band
    else:


        lum_plt=deepcopy(lum_list.transpose(2,3,0,1))
    
        #avoiding negative uncertainties
        lum_plt[1:]=lum_plt[1:].clip(0)

    #restructuring the array
    lum_plt_use=np.array([[[None]*len(abslines_inf)]*3]*5)

    for i_1 in range(5):
        for i_2 in range(3):
            for i_3 in range(len(abslines_inf)):
                lum_plt_use[i_1][i_2][i_3]=lum_plt[i_2][i_1][i_3]

    lum_plt=lum_plt_use

    lum_plt=np.array([[[subsubelem for subsubelem in subelem] for subelem in elem]\
                       for elem in lum_plt],dtype=object)

    #note that all these arrays are linked together through their elements so any modifications should be made on
    #copy of their values and not on the arrays themselves

    #We then use uncertainty composition for the HID

    hid_plt_vals=lum_plt[2][0]/lum_plt[1][0]

    hid_errors=np.array([((lum_plt[2][i]/lum_plt[2][0])**2+\
                          (lum_plt[1][i]/lum_plt[1][0])**2)**(1/2)*hid_plt_vals for i in [1,2]])
    
    #capping the error lower limits proportions at 1
    for i_obj in range(len(hid_errors[0])):
        hid_errors[0][i_obj]=hid_errors[0][i_obj].clip(0,1)

    hid_plt=np.array([[hid_plt_vals,hid_errors[0],hid_errors[1]],[lum_plt[4][i] for i in range(3)]])

    #computing an array of the object inclinations
    incl_plt=np.array([[np.nan,np.nan,np.nan] if elem not in incl_dict_use else incl_dict_use[elem] for elem in obj_list])

    #computing an array of the line widths from the autofit computations
    
    #### fit parameters
    
    width_plt=np.array([[[None]*len(autofit_inf)]*6]*3)
    nh_plt=np.array([[None]*len(autofit_inf)]*3)
    
    kt_plt=np.array([[None]*len(autofit_inf)]*3)
    
    for i_uncert in range(3):
        for i_line in range(6):
            for i_obj in range(len(autofit_inf)):
                width_part_obj=np.zeros(len(autofit_inf[i_obj]))
                nh_part_obj=np.repeat(min_nh,len(autofit_inf[i_obj]))
                kt_part_obj=np.repeat(np.nan,len(autofit_inf[i_obj]))
                
                for i_obs in range(len(autofit_inf[i_obj])):
                    
                    #empty values for when there was no autofit performed
                    if autofit_inf[i_obj][i_obs][0] is None:
                        width_part_obj[i_obs]=np.nan
                        nh_part_obj[i_obs]=np.nan
                        kt_part_obj[i_obs]=np.nan
                        continue
                    
                    #note: the last index at 0 here is the datagroup. always use the data from the first datagroup
                    mask_sigma=[lines_std_names[3+i_line] in elem and 'Sigma' in elem for elem in autofit_inf[i_obj][i_obs][1][0]]
                    mask_nh=['phabs.nH' in elem for elem in autofit_inf[i_obj][i_obs][1][0]]

                    mask_kt=['diskbb.Tin' in elem for elem in autofit_inf[i_obj][i_obs][1][0]]
                    
                    #insuring there is no problem by putting nans where there is no line
                    if sum(mask_sigma)==0:
                        width_part_obj[i_obs]=np.nan
                    else:                  
                        
                        '''
                        If the width is compatible with 0 at 3 sigma (confirmed by checking the width value in absline_plot),
                        we put 0 as main value and lower uncertainty and use the 90% upper limit 
                        Else we use the standard value multiplied by the fwhm/1 sigma conversion factor (2.3548)
                        '''
                        
                        #here we fetch the correct line by applying the mask. The 0 after is there to avoid a len 1 nested array
                        #note: should be replaced by the line actual energy if using big bshifts
                        width_vals=autofit_inf[i_obj][i_obs][0][0][mask_sigma][0].astype(float)[i_uncert]\
                                              /lines_e_dict[lines_std_names[3+i_line]][0]*c_light*2.3548
                                                  
                        #index 7 is the width
                        if abslines_plt[7][0][i_line][i_obj][i_obs]==0:
                            if i_uncert in [0,1]:
                                #0 for the main value and the lower limit
                                width_part_obj[i_obs]=0
                            else:
                                #the upper limit is thus the sum of the 1sigma main value and the upper error 
                                #if the value is unconstrained and thus frozen, we put the upper limit at 0
                                #to act as if the point didn't exist
                                if autofit_inf[i_obj][i_obs][0][0][mask_sigma][0].astype(float)[2]==0:
                                    width_part_obj[i_obs]=0
                                else:
                                    #the numerical factor transforms the 1 sigma width into a FWHM
                                    width_part_obj[i_obs]=autofit_inf[i_obj][i_obs][0][0][mask_sigma][0].astype(float)[[0,2]].sum()\
                                                      /lines_e_dict[lines_std_names[3+i_line]][0]*c_light*2.3548
                        else:
                            width_part_obj[i_obs]=width_vals
                        
                    if sum(mask_nh)!=0:
                        nh_part_obj[i_obs]=autofit_inf[i_obj][i_obs][0][0][mask_nh][0].astype(float)[i_uncert]
                        
                    if sum(mask_kt)!=0:
                        kt_part_obj[i_obs]=autofit_inf[i_obj][i_obs][0][0][mask_kt][0].astype(float)[i_uncert]
                        
                width_plt[i_uncert][i_line][i_obj]=width_part_obj
                nh_plt[i_uncert][i_obj]=nh_part_obj
                kt_plt[i_uncert][i_obj]=kt_part_obj

    if mask_include is not None:
        #also returning the updated lum_list
        return abslines_inf_line,abslines_inf_obj,abslines_plt,abslines_e,\
               lum_plt,hid_plt,incl_plt,width_plt,nh_plt,kt_plt,lum_list
    else:
        return abslines_inf_line, abslines_inf_obj, abslines_plt, abslines_e, \
            lum_plt, hid_plt, incl_plt, width_plt, nh_plt, kt_plt

def values_manip_high_E(val_high_list):

    '''
    transposing list like structures into plot-like structures with the uncertainty as first dimension,
    then objects, then observation
    '''

    n_obj=len(val_high_list)
    val_high_plot=np.array([None]*3)

    for i_incert in range(3):
        #this
        val_high_plot_incert=np.array([None]*n_obj)
        for i_obj in range(n_obj):
            val_high_plot_incert[i_obj]=val_high_list[i_obj].T[i_incert]

        val_high_plot[i_incert]=val_high_plot_incert

    val_high_plot=np.array([[subelem for subelem in elem] for elem in val_high_plot],dtype=object)
    return val_high_plot

def hid_graph(ax_hid,dict_linevis,
              display_single=False,display_nondet=True,display_upper=False,
              cyclic_cmap_nondet=False,cyclic_cmap_det=False,cyclic_cmap=False,
              cmap_incl_type=None,cmap_incl_type_str=None,
              radio_info_label=None,
              ew_ratio_ids=None,
              color_nondet=True,
              restrict_threshold=False,display_nonsign=False,display_central_abs=False,
              display_incl_inside=False,dash_noincl=False,
              display_hid_error=False,display_edgesource=False,split_cmap_source=True,
              display_evol_single=False,display_dicho=False,
              global_colors=True,alpha_abs=1,
              paper_look=False,bigger_text=True,square_mode=True,zoom=False,
              broad_mode=False,
              restrict_match_INT=False,broad_binning='day',orbit_bin_lim=1):

    '''

    in broad mode, replaces the standard HID axis by adding theoretical flues estimated from the bat catalog
    '''

    abslines_infos_perobj=dict_linevis['abslines_infos_perobj']
    abslines_plot=dict_linevis['abslines_plot']
    nh_plot=dict_linevis['nh_plot']
    kt_plot_restrict = dict_linevis['Tin_diskbb_plot_restrict']
    kt_plot=dict_linevis['Tin_diskbb_plot']
    hid_plot=dict_linevis['hid_plot']
    incl_plot=dict_linevis['incl_plot']
    mask_obj=dict_linevis['mask_obj']
    mask_obj_base=dict_linevis['mask_obj_base']
    mask_lines=dict_linevis['mask_lines']
    mask_lines_ul=dict_linevis['mask_lines_ul']
    obj_list=dict_linevis['obj_list']
    date_list=dict_linevis['date_list']
    instru_list=dict_linevis['instru_list']
    lum_list=dict_linevis['lum_list']
    choice_telescope=dict_linevis['choice_telescope']
    telescope_list=dict_linevis['telescope_list']
    bool_incl_inside=dict_linevis['bool_incl_inside']
    bool_noincl=dict_linevis['bool_noincl']
    slider_date=dict_linevis['slider_date']
    slider_sign=dict_linevis['slider_sign']
    radio_info_cmap=dict_linevis['radio_info_cmap']
    radio_cmap_i=dict_linevis['radio_cmap_i']
    cmap_color_source=dict_linevis['cmap_color_source']
    cmap_color_det=dict_linevis['cmap_color_det']
    cmap_color_nondet=dict_linevis['cmap_color_nondet']
    exptime_list=dict_linevis['exptime_list']
    display_minorticks=dict_linevis['display_minorticks']

    diago_color=dict_linevis['diago_color']
    custom_states_color=dict_linevis['custom_states_color']
    custom_outburst_color=dict_linevis['custom_outburst_color']
    custom_outburst_number=dict_linevis['custom_outburst_number']

    hr_high_plot_restrict=dict_linevis['hr_high_plot_restrict']
    hid_log_HR=dict_linevis['hid_log_HR']
    flag_single_obj=dict_linevis['flag_single_obj']

    if not broad_mode==False:
        HR_broad_bands=dict_linevis['HR_broad_bands']
        lum_broad_bands= dict_linevis['lum_broad_bands']
        Edd_factor_restrict=dict_linevis['Edd_factor_restrict']
        lum_plot = dict_linevis['lum_plot']

    if broad_mode=='BAT':
        catal_bat_df=dict_linevis['catal_bat_df']
        catal_bat_simbad=dict_linevis['catal_bat_simbad']
        sign_broad_hid_BAT=dict_linevis['sign_broad_hid_BAT']

    if restrict_match_INT:
        lc_int_sw_dict = dict_linevis['lc_int_sw_dict']
        fit_int_revol_dict = dict_linevis['fit_int_revol_dict']

    #note that these one only get the significant BAT detections so no need to refilter
    lum_high_1sig_plot_restrict=dict_linevis['lum_high_1sig_plot_restrict']
    lum_high_sign_plot_restrict=dict_linevis['lum_high_sign_plot_restrict']
    # global normalisations values for the points
    norm_s_lin = 5
    norm_s_pow = 1.15

    #extremal allowed values for kT in the fitting procedure(in keV)
    kt_min = 0.5
    kt_max = 3.

    # parameters independant of the presence of lines
    type_1_cm = ['Inclination', 'Time', 'nH', 'kT']

    # parameters without actual colorbars
    type_1_colorcode = ['Source', 'Instrument','custom_line_struct','custom_acc_states','custom_outburst']

    fig_hid=ax_hid.get_figure()

    hid_plot_use = deepcopy(hid_plot)

    if restrict_match_INT:
        #currently limited to 4U1630-47
        int_lc_df = fit_int_revol_dict['4U1630-47']

        int_lc_mjd=np.array([Time(elem).mjd.astype(float) for elem in int_lc_df['ISOT']])

        obs_dates=Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd.astype(float)

        #computing which observations are within the timeframe of an integral revolution (assuming 3days-long)
        mask_withtime_INT=[min((int_lc_mjd-elem)[(int_lc_mjd-elem)>=0])<3 for elem in obs_dates[0]]

    elif broad_mode=='BAT':

        HR_broad_6_10=HR_broad_bands=='([6-10]+[BAND])/[3-6]'
        lum_broad_soft=lum_broad_bands=='[3-10]+[BAND]'

        ax_hid.set_yscale('log')

        #no need for symlog now that we use upper limits
        # if not HR_broad_6_10:
        #     broad_x_linthresh=0.01
        #     ax_hid.set_xscale('symlog', linthresh=broad_x_linthresh, linscale=0.1)
        #     ax_hid.xaxis.set_minor_locator(MinorSymLogLocator(linthresh=broad_x_linthresh))
        # else:

        broad_x_linthresh=0
        ax_hid.set_xscale('log')

        ax_hid.set_xlabel('Hardness Ratio in '+HR_broad_bands.replace('BAND','15-50')+' keV bands)')
        ax_hid.set_ylabel(r'Luminosity in the '+lum_broad_bands.replace('BAND','15-50')+' keV band in (L/L$_{Edd}$) units')

        # #currently limited to 4U1630-47
        # bat_lc_df_init = fetch_bat_lightcurve(catal_bat_df, catal_bat_simbad,['4U1630-47'], binning=broad_binning)
        #
        # #restricting to significant BAT detections or not
        #
        # mask_sign_bat = bat_lc_df_init[bat_lc_df_init.columns[1]] - bat_lc_df_init[
        #     bat_lc_df_init.columns[2]] * 2 > 0
        #
        # if sign_broad_hid_BAT:
        #     # significance test to only get good bat data
        #
        #     # applying the mask. Reset index necessary to avoid issues when calling indices later.
        #     # drop to avoid creating an index column that will ruin the column calling
        #     bat_lc_df = bat_lc_df_init[mask_sign_bat].reset_index(drop=True)
        # else:
        #     bat_lc_df=bat_lc_df_init
        #
        # if broad_binning=='day':
        #     bat_lc_mjd=np.array(bat_lc_df[bat_lc_df.columns[0]])
        #
        # elif broad_binning=='orbit':
        #     bat_lc_tstart=Time('51910',format='mjd')+TimeDelta(bat_lc_df['TIME'],format='sec')
        #     bat_lc_tend=Time('51910',format='mjd')+TimeDelta(bat_lc_df['TIMEDEL'],format='sec')
        #
        # #converting to 15-50keV luminosity in Eddington units, removing negative values
        # bat_lc_lum_nocorr=np.array([bat_lc_df[bat_lc_df.columns[1]],bat_lc_df[bat_lc_df.columns[2]],
        #                             bat_lc_df[bat_lc_df.columns[2]]]).clip(0).T\
        #             *convert_BAT_count_flux['4U1630-47']*Edd_factor_restrict
        #
        # #and applying the correction
        # bat_lc_lum=corr_factor_lbat(bat_lc_lum_nocorr)
        #
        # if broad_binning=='day':
        #     obs_dates=Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd.astype(int)
        #
        #
        #     mask_withtime_BAT=[elem in bat_lc_mjd for elem in obs_dates[0]]
        #
        #     #this one only considers BAT but also considers non-significant points
        #
        #     #not necessary now that we create different arrays beforehand
        #     # #getting an array with the bat flux of each observation date
        #     # lum_broad_single_BAT=np.array([np.array([np.nan, np.nan,np.nan]) if obs_dates[0][i_obs] not in bat_lc_mjd else bat_lc_lum[bat_lc_mjd==obs_dates[0][i_obs]][0]\
        #     #                           for i_obs in range(len(obs_dates[0]))]).T
        #
        # elif broad_binning=='orbit':
        #
        #     obs_tstart=Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd
        #     obs_tend=(Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str))+\
        #             TimeDelta(np.array([exptime_list[mask_obj][0] for i in range(sum(mask_lines))],format='sec'))).mjd
        #
        #     #TO BE IMPROVED FOR MORE PRECISION
        #     # # getting an array with the bat flux of each observation date
        #     # lum_broad_single_BAT = np.array([np.array([np.nan, np.nan]) if obs_dates[0][i_obs] not in bat_lc_mjd else
        #     #                              bat_lc_lum[bat_lc_mjd == obs_dates[0][i_obs]][0] \
        #     #                              for i_obs in range(len(obs_dates[0]))]).T

        #now we combine the BAT non-significant elements and the already existing significant elements in
        #lum_high_plot_restrict

        if flag_single_obj:
            lum_broad_single=np.array([elem for elem in np.transpose(lum_high_1sig_plot_restrict,(1,0,2))[0]])

            #mask to highlight non-significant high Energy detections
            mask_sign_high_E=~np.isnan(np.array([elem for elem in np.transpose(lum_high_sign_plot_restrict,(1,0,2))[0]],
                                                dtype=float)[0])

        else:
            lum_broad_single=np.array([elem for elem in lum_high_1sig_plot_restrict.T[0]])

            #mask to highlight non-significant high Energy detections
            mask_sign_high_E=~np.isnan(np.array([elem for elem in lum_high_sign_plot_restrict.T[0]])[0])

        lum_broad_single=lum_broad_single.T

        #transforming the BAT non-significant detections in 1 sigma upper limits or removing them
        for i_obs in range(len(lum_broad_single)):
            if not mask_sign_high_E[i_obs]:
                if not sign_broad_hid_BAT:
                    lum_broad_single[i_obs]=np.array([lum_broad_single[i_obs][0]+lum_broad_single[i_obs][1],
                                            lum_broad_single[i_obs][0]+lum_broad_single[i_obs][1],0.])
                else:
                    lum_broad_single[i_obs]=np.repeat(np.nan,3)

        lum_broad_single=lum_broad_single.T

        #creating the mask to avoid plotting nans everywhere
        mask_with_broad=~np.isnan(lum_broad_single[0])

        #this is the quantity that needs to be added if the numerator is broad+6-10 and not just broad
        hid_broad_add=lum_plot[2][0][mask_obj][0].astype(float) if HR_broad_6_10 else 0

        hid_broad_vals = (lum_broad_single[0] + hid_broad_add)\
                         / lum_plot[1][0][mask_obj][0].astype(float)

        #here the numerator is the quadratic uncertainty addition and then the fraction is for the quadratic ratio uncertainty
        #composition
        hid_broad_err= np.array([((((lum_plot[2][i][mask_obj][0] if HR_broad_6_10 else 0)**2+lum_broad_single[i]**2)**(1/2)\
                           /(lum_broad_single[0]+hid_broad_add)) ** 2 + \
                                (lum_plot[1][i][mask_obj][0] / lum_plot[1][0][mask_obj][0]) ** 2) ** (1 / 2) * hid_broad_vals\
                               for i in [1, 2]])

        #overwriting hid_plot's individual elements because overwriting the full obs array doesn't work
        #there's lot of issues if using mask_obj directly but here we should be in single object mode only
        #so we can do it differently
        i_obj_single=np.argwhere(mask_obj).T[0][0]

        hid_plot_use[0][0][i_obj_single]=hid_broad_vals
        hid_plot_use[0][1][i_obj_single] = hid_broad_err[0]
        hid_plot_use[0][2][i_obj_single] = hid_broad_err[1]

        if lum_broad_soft:
            hid_plot_use[1][0][i_obj_single] += lum_broad_single[0]

            hid_plot_use[1][1][i_obj_single] = ((hid_plot_use[1][1][i_obj_single]) ** 2 + \
                                                      (lum_broad_single[1]) ** 2) ** (1 / 2)
            hid_plot_use[1][2][i_obj_single] = ((hid_plot_use[1][2][i_obj_single]) ** 2 + \
                                                      (lum_broad_single[1]) ** 2) ** (1 / 2)

    else:

        # log x scale for an easier comparison with Ponti diagrams
        if hid_log_HR:
            ax_hid.set_xscale('log')

            if display_minorticks:
                if ax_hid.get_xlim()[0]>0.1:
                    ax_hid.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
                    ax_hid.xaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
                else:
                    ax_hid.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))
                    ax_hid.xaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))

        ax_hid.set_xlabel('Hardness Ratio ([6-10]/[3-6] keV bands)')
        ax_hid.set_ylabel(r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units')
        ax_hid.set_yscale('log')

    #recreating some variables

    mask_obs_intime_repeat = np.array(
        [np.repeat(((np.array([Time(subelem) for subelem in elem]) >= Time(slider_date[0])) & \
                    (np.array([Time(subelem) for subelem in elem]) <= Time(slider_date[1]))), sum(mask_lines)) for elem
         in date_list], dtype=object)

    # checking which sources have no detection in the current combination

    global_displayed_sign = np.array(
        [ravel_ragged(elem)[mask.astype(bool)] for elem, mask in zip(abslines_plot[4][0][mask_lines].T, mask_obs_intime_repeat)],
        dtype=object)

    incl_cmap = np.array([incl_plot.T[0], incl_plot.T[0] - incl_plot.T[1], incl_plot.T[0] + incl_plot.T[2]]).T
    incl_cmap_base = incl_cmap[mask_obj]
    incl_cmap_restrict = incl_cmap[mask_obj]

    nh_plot_restrict = deepcopy(nh_plot)

    nh_plot_restrict = nh_plot_restrict.T[mask_obj].T

    if len(mask_obj) == 1 and np.ndim(hid_plot_use) == 4:
        hid_plot_restrict=hid_plot_use
    else:
        hid_plot_restrict = hid_plot_use.T[mask_obj].T

    incl_plot_restrict = incl_plot[mask_obj]

    if display_nonsign:
        mask_obj_withdet = np.array([(elem > 0).any() for elem in global_displayed_sign])
    else:
        mask_obj_withdet = np.array([(elem >= slider_sign).any() for elem in global_displayed_sign])

    # storing the number of objects with detections
    n_obj_withdet = sum(mask_obj_withdet & mask_obj)

    # computing the extremal values of the whole sample/plotted sample to get coherent colormap normalisations, and creating the range of object colors
    if global_colors:
        global_plotted_sign = abslines_plot[4][0].ravel()
        global_plotted_data = abslines_plot[radio_cmap_i][0].ravel()

        # objects colormap for common display
        norm_colors_obj = mpl.colors.Normalize(vmin=0,
                                               vmax=max(0, len(abslines_infos_perobj) + (-1 if not cyclic_cmap else 0)))
        colors_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_source)

        norm_colors_det = mpl.colors.Normalize(vmin=0, vmax=max(0,
                                                                n_obj_withdet + (-1 if not cyclic_cmap_det else 0) + (
                                                                    1 if n_obj_withdet == 0 else 0)))
        colors_det = mpl.cm.ScalarMappable(norm=norm_colors_det, cmap=cmap_color_det)

        norm_colors_nondet = mpl.colors.Normalize(vmin=0, vmax=max(0, len(abslines_infos_perobj) - n_obj_withdet + (
            -1 if not cyclic_cmap_nondet else 0)))
        colors_nondet = mpl.cm.ScalarMappable(norm=norm_colors_nondet, cmap=cmap_color_nondet)

        # the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
        global_plotted_datetime = np.array([elem for elem in date_list for i in range(len(mask_lines))], dtype='object')

        global_mask_intime = np.repeat(True, len(ravel_ragged(global_plotted_datetime)))

        global_mask_intime_norepeat = np.repeat(True, len(ravel_ragged(date_list)))

    else:
        global_plotted_sign = abslines_plot[4][0][mask_lines].T[mask_obj].ravel()
        global_plotted_data = abslines_plot[radio_cmap_i][0][mask_lines].T[mask_obj].ravel()

        # objects colormap
        norm_colors_obj = mpl.colors.Normalize(vmin=0, vmax=max(0, len(abslines_infos_perobj[mask_obj]) + (
            -1 if not cyclic_cmap else 0)))
        colors_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_source)

        norm_colors_det = mpl.colors.Normalize(vmin=0, vmax=max(0, n_obj_withdet + (-1 if not cyclic_cmap_det else 0)))
        colors_det = mpl.cm.ScalarMappable(norm=norm_colors_det, cmap=cmap_color_det)

        norm_colors_nondet = mpl.colors.Normalize(vmin=0,
                                                  vmax=max(0, len(abslines_infos_perobj[mask_obj]) - n_obj_withdet + (
                                                      -1 if not cyclic_cmap_nondet else 0)))
        colors_nondet = mpl.cm.ScalarMappable(norm=norm_colors_nondet, cmap=cmap_color_nondet)

        # adapting the plotted data in regular array for each object in order to help
        # global masks to take off elements we don't want in the comparison

        # the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
        global_plotted_datetime = np.array([elem for elem in date_list[mask_obj] for i in range(sum(mask_lines))],
                                           dtype='object')

        global_mask_intime = (Time(ravel_ragged(global_plotted_datetime)) >= Time(slider_date[0])) & \
                             (Time(ravel_ragged(global_plotted_datetime)) <= Time(slider_date[1]))

        global_mask_intime_norepeat = (Time(ravel_ragged(date_list[mask_obj])) >= Time(slider_date[0])) & \
                                      (Time(ravel_ragged(date_list[mask_obj])) <= Time(slider_date[1]))

    # global_nondet_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])<slider_sign) & (global_mask_intime)

    global_det_mask = (np.array([subelem for elem in global_plotted_sign for subelem in elem]) > 0) & (
        global_mask_intime)

    global_sign_mask = (np.array([subelem for elem in global_plotted_sign for subelem in elem]) >= slider_sign) & (
        global_mask_intime)

    global_det_data = np.array([subelem for elem in global_plotted_data for subelem in elem])[global_det_mask]

    # this second array is here to restrict the colorbar scalings to take into account significant detections only
    global_sign_data = np.array([subelem for elem in global_plotted_data for subelem in elem])[global_sign_mask]

    # same for the color-coded infos
    cmap_info = mpl.cm.plasma_r.copy() if radio_info_cmap not in ['Time', 'nH', 'kT'] else mpl.cm.plasma.copy()

    cmap_info.set_bad(color='grey')

    # normalisation of the colormap
    if radio_cmap_i == 1 or radio_info_cmap == 'EW ratio':
        gamma_colors = 1 if radio_cmap_i == 1 else 0.5
        cmap_norm_info = colors.PowerNorm(gamma=gamma_colors)

    elif radio_info_cmap not in ['Inclination', 'Time', 'kT']:
        cmap_norm_info = colors.LogNorm()
    else:
        # keeping a linear norm for the inclination
        cmap_norm_info = colors.Normalize(0,90)


    # putting the axis limits at standard bounds or the points if the points extend further
    lum_list_ravel = np.array([subelem for elem in lum_list for subelem in elem])
    bounds_x = [min(lum_list_ravel.T[2][0] / lum_list_ravel.T[1][0]),
                max(lum_list_ravel.T[2][0] / lum_list_ravel.T[1][0])]
    bounds_y = [min(lum_list_ravel.T[4][0]), max(lum_list_ravel.T[4][0])]

    if zoom=='auto' or broad_mode!=False:

        #the nan values are the BAT non-matching points and thus need to be removed
        broad_notBAT_mask=~np.isnan(ravel_ragged(hid_plot_restrict[0][0]))

        xlims=(min(ravel_ragged(hid_plot_restrict[0][0])[broad_notBAT_mask]),
                        max(ravel_ragged(hid_plot_restrict[0][0])[broad_notBAT_mask]))

        ylims=(min(ravel_ragged(hid_plot_restrict[1][0])[broad_notBAT_mask]),
                        max(ravel_ragged(hid_plot_restrict[1][0])[broad_notBAT_mask]))

        rescale_flex(ax_hid,xlims,ylims,0.05)

    if type(zoom)==list:
        ax_hid.set_xlim(zoom[0][0],zoom[0][1])
        ax_hid.set_ylim(zoom[1][0], zoom[1][1])

    if not zoom and not broad_mode:

        rescale_flex(ax_hid,bounds_x,bounds_y,0.05,std_x=[0.1,2],std_y=[1e-5,1])

    # creating space for the colorbar
    if radio_info_cmap not in type_1_colorcode:
        ax_cb = plt.axes([0.92, 0.105, 0.02, 0.775])

        # giving a default value to the colorbar variable so we can test if a cb has been generated later on
        cb = None

    # markers
    marker_abs = 'o'
    marker_nondet = 'd'
    marker_ul = 'h'
    marker_ul_top = 'H'

    alpha_ul = 0.5

    # note: the value will finish at false for sources with no non-detections
    label_obj_plotted = np.repeat(False, len(abslines_infos_perobj[mask_obj]))

    is_colored_scat = False

    # creating the plotted colors variable#defining the mask for detections and non detection
    plotted_colors_var = []

    #### detections HID

    id_obj_det = 0

    #### Still issues with colormapping when restricting time

    # loop on the objects for detections (restricted or not depending on if the mode is detection only)
    for i_obj, abslines_obj in enumerate(abslines_infos_perobj[mask_obj]):

        # defining the index of the object in the entire array if asked to, in order to avoid changing colors
        if global_colors:
            i_obj_glob = np.argwhere(obj_list == obj_list[mask_obj][i_obj])[0][0]
        else:
            i_obj_glob = i_obj

        '''
        # The shape of each abslines_obj is (uncert,info,line,obs)
        '''

        # defining the hid positions of each point
        if broad_mode!=False:
            x_hid=hid_plot_use[0][0][mask_obj][i_obj]

            #similar structure for the rest
            x_hid_err=np.array([hid_plot_use[0][1][mask_obj][i_obj],hid_plot_use[0][2][mask_obj][i_obj]])

            #we do it this way because a += on y_hid will overwite lum_list, which is extremely dangerous
            if lum_broad_soft:
                y_hid=lum_list[mask_obj][i_obj].T[4][0] + lum_broad_single[0]
                y_hid_err=(lum_list[mask_obj][i_obj].T[4][1:]**2+lum_broad_single[1:]**2)**(1/2)
            else:
                y_hid = lum_list[mask_obj][i_obj].T[4][0]
                y_hid_err=lum_list[mask_obj][i_obj].T[4][1:]
        else:
            x_hid = lum_list[mask_obj][i_obj].T[2][0] / lum_list[mask_obj][i_obj].T[1][0]

            x_hid_err=(((lum_list[mask_obj][i_obj].T[2][1:]/lum_list[mask_obj][i_obj].T[2][0])**2 + \
                      (lum_list[mask_obj][i_obj].T[1][1:]/lum_list[mask_obj][i_obj].T[1][0])**2)**(1/2)*x_hid)

            y_hid = lum_list[mask_obj][i_obj].T[4][0]

            y_hid_err=lum_list[mask_obj][i_obj].T[4][1:]


        # defining the masks and shapes of the markers for the rest

        # defining the mask for the time interval restriction
        datelist_obj = Time(np.array([date_list[mask_obj][i_obj] for i in range(sum(mask_lines))]).astype(str))
        mask_intime = (datelist_obj >= Time(slider_date[0])) & (datelist_obj <= Time(slider_date[1]))

        if broad_mode=='BAT':
            mask_intime=(mask_intime) & mask_with_broad

        # defining the mask for detections and non detection        
        mask_det = (abslines_obj[0][4][mask_lines] > 0.) & (mask_intime)

        # defining the mask for significant detections
        mask_sign = (abslines_obj[0][4][mask_lines] >= slider_sign) & (mask_intime)

        # these ones will only be used if the restrict values chexbox is checked

        obj_val_cmap_sign = np.array(
            [np.nan if len(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_sign.T[i_obs]]) == 0 else \
                 (max(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_sign.T[i_obs]]) \
                      if radio_info_cmap != 'EW ratio' else \
                      np.nan if abslines_obj[0][radio_cmap_i][ew_ratio_ids[0]].T[i_obs] < slider_sign or \
                                abslines_obj[0][radio_cmap_i][ew_ratio_ids[1]].T[i_obs] < slider_sign else \
                          abslines_obj[0][radio_cmap_i][ew_ratio_ids[1]].T[i_obs] / \
                          abslines_obj[0][radio_cmap_i][ew_ratio_ids[0]].T[i_obs]) \
             for i_obs in range(len(abslines_obj[0][radio_cmap_i][mask_lines].T))])

        # the size is always tied to the EW
        obj_size_sign = np.array([np.nan if len(abslines_obj[0][0][mask_lines].T[i_obs][mask_sign.T[i_obs]]) == 0 else \
                                      max(abslines_obj[0][0][mask_lines].T[i_obs][mask_sign.T[i_obs]]) \
                                  for i_obs in range(len(abslines_obj[0][0][mask_lines].T))])

        # and we can create the plot mask from it (should be the same wether we take obj_size_sign or the size)
        obj_val_mask_sign = ~np.isnan(obj_size_sign)

        # creating a display order which is the reverse of the EW size order to make sure we do not hide part the detections
        obj_order_sign = obj_size_sign[obj_val_mask_sign].argsort()[::-1]

        if 'custom' in radio_info_cmap:
            if radio_info_cmap=='custom_line_struct':
                #reordering to have the substructures above the rest
                colors_data_restrict=diago_color[mask_obj][i_obj][obj_val_mask_sign]
                obj_order_sign_mainstruct=obj_size_sign[obj_val_mask_sign][colors_data_restrict=='grey'].argsort()[::-1]
                obj_order_sign_substruct=obj_size_sign[obj_val_mask_sign][colors_data_restrict=='orange'].argsort()[::-1]
                obj_order_sign_outliers=obj_size_sign[obj_val_mask_sign][colors_data_restrict=='blue'].argsort()[::-1]

                len_arr=np.arange(len(obj_order_sign))
                obj_order_sign=np.concatenate([len_arr[colors_data_restrict=='grey'][obj_order_sign_mainstruct],
                                                    len_arr[colors_data_restrict=='orange'][obj_order_sign_substruct],
                                                    len_arr[colors_data_restrict=='blue'][obj_order_sign_outliers]])

            elif radio_info_cmap=='custom_acc_states':
                # reordering to have the substructures above the rest
                colors_data_restrict = custom_states_color[mask_obj][i_obj][obj_val_mask_sign]

                states_zorder=['grey','red','orange','green','blue','purple']
                obj_order_sign_states=[obj_size_sign[obj_val_mask_sign][colors_data_restrict == elem_state_zorder]\
                                              .argsort()[::-1] for elem_state_zorder in states_zorder]

                len_arr = np.arange(len(obj_order_sign))
                obj_order_sign = np.concatenate([len_arr[colors_data_restrict == states_zorder[i_state]]\
                                                     [obj_order_sign_states[i_state]]\
                                                 for i_state in range(len(states_zorder))])

        # same thing for all detections
        obj_val_cmap = np.array(
            [np.nan if len(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                 max(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
             for i_obs in range(len(abslines_obj[0][radio_cmap_i][mask_lines].T))])

        obj_size = np.array([np.nan if len(abslines_obj[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                                 max(abslines_obj[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                             for i_obs in range(len(abslines_obj[0][0][mask_lines].T))])

        obj_val_mask = ~np.isnan(obj_size)

        # creating a display order which is the reverse of the EW size order to make sure we show as many detections as possible
        obj_order = obj_size[obj_val_mask].argsort()[::-1]

        # not used for now                
        # else:
        #     #in single line mode we can directly fetch the single lines values and the mask for the specific line
        #     obj_val_cmap_sign=abslines_obj[0][radio_cmap_i][mask_lines][mask_sign[mask_lines]].astype(float)

        #     #and the mask
        #     obj_val_mask_sign=mask_sign[mask_lines]

        #     #in single line mode we can directly fetch the single lines values and the mask for the specific line
        #     obj_val_cmap=abslines_obj[0][radio_cmap_i][mask_lines][mask_det[mask_lines]].astype(float)

        #     obj_val_mask=mask_det[mask_lines]

        # this mask is used to plot 'unsignificant only' detection points
        obj_val_mask_nonsign = (obj_val_mask) & (~obj_val_mask_sign)

        # plotting everything

        # we put the color mapped scatter into a list to clim all of them at once at the end
        if i_obj == 0:
            scat_col = []

        # plotting the detection centers if asked for

        if len(x_hid[obj_val_mask]) > 0 and display_central_abs:

            if display_hid_error:
                ax_hid.errorbar(x_hid[obj_val_mask], y_hid[obj_val_mask],
                               xerr=x_hid_err.T[obj_val_mask].T,yerr=y_hid_err.T[obj_val_mask].T,
                               marker='',ls='',
                           color=colors_obj.to_rgba(i_obj_glob) \
                               if radio_info_cmap == 'Source' else 'grey',
                               label='', zorder=1000)

            ax_hid.scatter(x_hid[obj_val_mask], y_hid[obj_val_mask], marker=marker_abs,
                       color=colors_obj.to_rgba(i_obj_glob) \
                           if radio_info_cmap == 'Source' else 'grey', label='', zorder=1000, edgecolor='black',
                       plotnonfinite=True)

        #### detection scatters
        # plotting statistically significant absorptions before values

        if radio_info_cmap == 'Instrument':
            color_instru = [telescope_colors[elem] for elem in
                        instru_list[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]]

            if display_nonsign:
                color_instru_nonsign = [telescope_colors[elem] for elem in
                                        instru_list[mask_obj][i_obj][obj_val_mask_nonsign]]

        # note: there's no need to reorder for source level informations (ex: inclination) since the values are the same for all the points                    
        c_scat = None if radio_info_cmap == 'Source' else \
            mdates.date2num(
                date_list[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]) if radio_info_cmap == 'Time' else \
                np.repeat(incl_cmap_restrict[i_obj][cmap_incl_type],
                          len(x_hid[obj_val_mask_sign])) if radio_info_cmap == 'Inclination' else \
                    color_instru if radio_info_cmap == 'Instrument' else \
                        nh_plot_restrict[0][i_obj][obj_val_mask_sign][obj_order_sign] if radio_info_cmap == 'nH' else \
                            kt_plot_restrict[0][i_obj][obj_val_mask_sign][
                                obj_order_sign] if radio_info_cmap == 'kT' else \
                                diago_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]\
                                    if radio_info_cmap=='custom_line_struct' else \
                                custom_states_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign] \
                                        if radio_info_cmap == 'custom_acc_states' else \
                                custom_outburst_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign] \
                                        if radio_info_cmap=='custom_outburst' else\
                                obj_val_cmap_sign[obj_val_mask_sign][obj_order_sign]

        #### TODO : test the dates here with just IGRJ17451 to solve color problem

        # adding a failsafe to avoid problems when nothing is displayed
        if c_scat is not None and len(c_scat) == 0:
            c_scat = None

        if restrict_threshold:

            color_val_detsign=(colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                                             colors_det.to_rgba(
                                                                 id_obj_det)) if radio_info_cmap == 'Source' else None

            if broad_mode=='BAT' and not sign_broad_hid_BAT:

                alpha_abs_sign_full=np.where(mask_sign_high_E,1,0.3)

                alpha_abs_detsign=alpha_abs_sign_full[obj_val_mask_sign][obj_order_sign]

                #in the case where alpha is not singular the len of the color keyword needs to match it
                color_val_detsign=None if color_val_detsign is None else np.array([color_val_detsign]*len(alpha_abs_detsign))
            else:
                alpha_abs_detsign=None

            if len(x_hid[obj_val_mask_sign][obj_order_sign])>0:
                # displaying "significant only" cmaps/sizes
                scat_col += [
                    ax_hid.scatter(x_hid[obj_val_mask_sign][obj_order_sign], y_hid[obj_val_mask_sign][obj_order_sign],
                                   marker=marker_abs, color=color_val_detsign,
                                   c=c_scat, s=norm_s_lin * obj_size_sign[obj_val_mask_sign][obj_order_sign] ** norm_s_pow,
                                   edgecolor='black' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                                   linewidth=1 + int(display_edgesource) / 2,
                                   norm=cmap_norm_info,
                                   label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and \
                                            (radio_info_cmap == 'Source' or display_edgesource) and len(
                                       x_hid[obj_val_mask_sign]) > 0 else '',
                                   cmap=cmap_info, alpha=alpha_abs_detsign,
                                   plotnonfinite=True)]

            if (radio_info_cmap == 'Source' or display_edgesource) and len(x_hid[obj_val_mask_sign]) > 0:
                label_obj_plotted[i_obj] = True

        # plotting the maximum value and hatch coding depending on if there's a significant abs line in the obs
        else:

            # displaying "all" cmaps/sizes but only where's at least one significant detection (so we don't hatch)
            scat_col += [
                ax_hid.scatter(x_hid[obj_val_mask_sign][obj_order_sign], y_hid[obj_val_mask_sign][obj_order_sign],
                               marker=marker_abs,
                               color=(colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                          colors_det.to_rgba(id_obj_det)) if radio_info_cmap == 'Source' else None,
                               c=c_scat, s=norm_s_lin * obj_size[obj_val_mask_sign][obj_order_sign] ** norm_s_pow,
                               edgecolor='black' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                               linewidth=1 + int(display_edgesource),
                               norm=cmap_norm_info,
                               label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and \
                                                                  (
                                                                              radio_info_cmap == 'Source' or display_edgesource) and len(
                                   x_hid[obj_val_mask_sign]) > 0 else '',
                               cmap=cmap_info, alpha=alpha_abs,
                               plotnonfinite=True)]

            if (radio_info_cmap == 'Source' or display_edgesource) and len(x_hid[obj_val_mask_sign]) > 0:
                label_obj_plotted[i_obj] = True

        # adding the plotted colors into a list to create the ticks from it at the end
        plotted_colors_var += [elem for elem in
                               (incl_cmap_restrict.T[cmap_incl_type] if radio_info_cmap == 'Inclination' else \
                                    (obj_val_cmap_sign[obj_val_mask_sign][obj_order_sign] if restrict_threshold \
                                         else obj_val_cmap[obj_val_mask_sign][obj_order_sign]).tolist()) if
                               not np.isnan(elem)]

        if display_nonsign:

            c_scat_nonsign = None if radio_info_cmap == 'Source' else \
                mdates.date2num(date_list[mask_obj][i_obj][obj_val_mask_nonsign]) if radio_info_cmap == 'Time' else \
                    np.repeat(incl_cmap_restrict[i_obj][cmap_incl_type],
                              len(x_hid[obj_val_mask_nonsign])) if radio_info_cmap == 'Inclination' else \
                        nh_plot_restrict[0][i_obj][obj_val_mask_nonsign] if radio_info_cmap == 'nH' else \
                            kt_plot_restrict[0][i_obj][obj_val_mask_nonsign] if radio_info_cmap == 'kT' else \
                                diago_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                    if radio_info_cmap == 'custom_line_struct' else \
                                    custom_states_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                        if radio_info_cmap == 'custom_acc_states' else \
                                        custom_outburst_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                        if radio_info_cmap=='custom_outburst' else \
                    obj_val_cmap[obj_val_mask_nonsign]

            # adding a failsafe to avoid problems when nothing is displayed
            if c_scat is not None and len(c_scat) == 0:
                c_scat = None

            # and "unsignificant only" in any case is hatched. Edgecolor sets the color of the hatch
            scat_col += [ax_hid.scatter(x_hid[obj_val_mask_nonsign], y_hid[obj_val_mask_nonsign], marker=marker_abs,
                                        color=(colors_obj.to_rgba(
                                            i_obj_glob) if not split_cmap_source else colors_det.to_rgba(
                                            id_obj_det)) if radio_info_cmap == 'Source' else None,
                                        c=c_scat_nonsign, s=norm_s_lin * obj_size[obj_val_mask_nonsign] ** norm_s_pow,
                                        hatch='///',
                                        edgecolor='grey' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                                        linewidth=1 + int(display_edgesource),
                                        norm=cmap_norm_info,
                                        label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and \
                                                                           (
                                                                                       radio_info_cmap == 'Source' or display_edgesource) else '',
                                        cmap=cmap_info,
                                        alpha=alpha_abs,
                                        plotnonfinite=True)]
            if (radio_info_cmap == 'Source' or display_edgesource) and len(x_hid[obj_val_mask_nonsign]) > 0:
                label_obj_plotted[i_obj] = True

            plotted_colors_var += [elem for elem in (
                incl_cmap_restrict.T[cmap_incl_type] if radio_info_cmap == 'Inclination' else obj_val_cmap[
                    obj_val_mask_nonsign].tolist()) if not np.isnan(elem)]

        if len(x_hid[obj_val_mask_sign]) > 0 or (len(x_hid[obj_val_mask_nonsign]) > 0 and display_nonsign):
            id_obj_det += 1

        # resizing all the colors and plotting the colorbar, only done at the last iteration
        if radio_info_cmap not in type_1_colorcode and i_obj == len(abslines_infos_perobj[mask_obj]) - 1 and len(
                plotted_colors_var) > 0:

            is_colored_scat = False

            for elem_scatter in scat_col:

                # standard limits for the inclination and Time
                if radio_info_cmap == 'Inclination':
                    elem_scatter.set_clim(vmin=0, vmax=90)
                elif radio_info_cmap == 'Time':

                    if global_colors:
                        elem_scatter.set_clim(
                            vmin=min(mdates.date2num(ravel_ragged(date_list))),
                            vmax=max(mdates.date2num(ravel_ragged(date_list))))
                    else:
                        elem_scatter.set_clim(
                            vmin=max(
                                min(mdates.date2num(ravel_ragged(date_list\
                                    [mask_obj])[global_mask_intime_norepeat])),
                                mdates.date2num(slider_date[0])),
                            vmax=min(
                                max(mdates.date2num(ravel_ragged(date_list\
                                    [mask_obj])[global_mask_intime_norepeat])),
                                mdates.date2num(slider_date[1])))

                elif radio_info_cmap == 'nH':

                    if global_colors:
                        elem_scatter.set_clim(vmin=min(ravel_ragged(nh_plot_restrict[0])),
                                              vmax=max(ravel_ragged(nh_plot_restrict[0])))
                    else:
                        elem_scatter.set_clim(vmin=min(ravel_ragged(nh_plot_restrict[0])[global_mask_intime_norepeat]),
                                              vmax=max(ravel_ragged(nh_plot_restrict[0])[global_mask_intime_norepeat]))
                elif radio_info_cmap == 'kT':

                    elem_scatter.set_clim(vmin=0.5, vmax=3)

                else:

                    # dynamical limits for the rest
                    if global_colors and radio_info_cmap not in ('EW ratio', 'Inclination', 'Time', 'nH', 'kT'):
                        if display_nonsign:
                            elem_scatter.set_clim(vmin=min(global_det_data), vmax=max(global_det_data))
                        else:
                            elem_scatter.set_clim(vmin=min(global_sign_data), vmax=max(global_sign_data))
                    else:
                        elem_scatter.set_clim(vmin=min(plotted_colors_var), vmax=max(plotted_colors_var))

                if len(elem_scatter.get_sizes()) > 0:
                    is_colored_scat = True

                    # keeping the scatter to create the colorbar from it
                    elem_scatter_forcol = elem_scatter

                # ax_cb.set_axis_off()

            # defining the ticks from the currently plotted objects

            if radio_cmap_i == 1 or radio_info_cmap == 'EW ratio':

                cmap_min_sign = 1 if min(plotted_colors_var) == 0 else min(plotted_colors_var) / abs(
                    min(plotted_colors_var))

                cmap_max_sign = 1 if min(plotted_colors_var) == 0 else max(plotted_colors_var) / abs(
                    max(plotted_colors_var))

                # round numbers for the Velocity shift                
                if radio_info_cmap == 'Velocity shift':
                    bshift_step = 250 if choice_telescope == ['Chandra'] else 500

                    # the +1 are here to ensure we see the extremal ticks

                    cmap_norm_ticks = np.arange(((min(plotted_colors_var) // bshift_step) + 1) * bshift_step,
                                                ((max(plotted_colors_var) // bshift_step) + 1) * bshift_step,
                                                2 * bshift_step)
                    elem_scatter.set_clim(vmin=min(cmap_norm_ticks), vmax=max(cmap_norm_ticks))

                else:
                    cmap_norm_ticks = np.linspace(cmap_min_sign * abs(min(plotted_colors_var)) ** (gamma_colors),
                                                  max(plotted_colors_var) ** (gamma_colors), 7, endpoint=True)

                # adjusting to round numbers

                if radio_info_cmap == 'EW ratio':
                    cmap_norm_ticks = np.concatenate((cmap_norm_ticks, np.array([1])))

                    cmap_norm_ticks.sort()

                if radio_cmap_i == 1 and min(plotted_colors_var) < 0:
                    # cmap_norm_ticks=np.concatenate((cmap_norm_ticks,np.array([0])))
                    # cmap_norm_ticks.sort()
                    pass

                if radio_info_cmap != 'Velocity shift':
                    # maintaining the sign with the square norm
                    cmap_norm_ticks = cmap_norm_ticks ** (1 / gamma_colors)

                    cmap_norm_ticks = np.concatenate((np.array([min(plotted_colors_var)]), cmap_norm_ticks))

                    cmap_norm_ticks.sort()


            else:
                cmap_norm_ticks = None

            # only creating the colorbar if there is information to display
            if is_colored_scat and radio_info_cmap not in type_1_colorcode:

                if radio_info_cmap == 'Time':

                    low_bound_date = max(
                        min(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj]) \
                                                [global_mask_intime_norepeat])),
                        mdates.date2num(slider_date[0]))

                    high_bound_date = min(
                        max(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj]) \
                                                [global_mask_intime_norepeat])),
                        mdates.date2num(slider_date[1]))

                    # manually readjusting for small durations because the AutoDateLocator doesn't work well
                    time_range = high_bound_date - low_bound_date

                    if time_range < 150:
                        date_format = mdates.DateFormatter('%Y-%m-%d')
                    elif time_range < 1825:
                        date_format = mdates.DateFormatter('%Y-%m')
                    else:
                        date_format = mdates.AutoDateFormatter(mdates.AutoDateLocator())

                    cb = plt.colorbar(elem_scatter_forcol, cax=ax_cb, ticks=mdates.AutoDateLocator(),
                                      format=date_format,)
                else:
                    cb = plt.colorbar(elem_scatter_forcol, cax=ax_cb, extend='min' if radio_info_cmap == 'nH' else None)
                    if cmap_norm_ticks is not None:
                        cb.set_ticks(cmap_norm_ticks)

                # cb.ax.minorticks_off()

                if radio_cmap_i == 1:
                    cb_add_str = ' (km/s)'
                else:
                    cb_add_str = ''

                if radio_info_cmap == 'Inclination':
                    cb.set_label(cmap_incl_type_str + ' of the source inclination (°)', labelpad=10)
                elif radio_info_cmap == 'Time':
                    cb.set_label('Observation date', labelpad=30)
                elif radio_info_cmap == 'nH':
                    cb.set_label(r'nH ($10^{22}$ cm$^{-2}$)', labelpad=10)
                elif radio_info_cmap == 'kT':
                    cb.set_label(r'disk temperature (keV)', labelpad=10)
                else:
                    if restrict_threshold:
                        cb.set_label(((
                                          'minimal ' if radio_cmap_i == 1 else 'maximal ') if radio_info_cmap != 'EW ratio' else '') + (
                                         radio_info_label[radio_cmap_i - 1].lower() if radio_info_cmap != 'Del-C' else
                                         radio_info_label[radio_cmap_i - 1]) +
                                     ' in significant detections\n for each observation' + cb_add_str, labelpad=10)
                    else:
                        cb.set_label(((
                                          'minimal ' if radio_cmap_i == 1 else 'maximal ') if radio_info_cmap != 'EW ratio' else '') + (
                                         radio_info_label[radio_cmap_i - 1].lower() if radio_info_cmap != 'Del-C' else
                                         radio_info_label[radio_cmap_i - 1]) +
                                     ' in all detections\n for each observation' + cb_add_str, labelpad=10)

    label_obj_plotted = np.repeat(False, len(abslines_infos_perobj[mask_obj]))

    #### non detections HID

    id_obj_det = 0
    id_obj_nondet = 0

    scatter_nondet = []

    # loop for non detection, separated to be able to restrict the color range in case of non detection
    for i_obj_base, abslines_obj_base in enumerate(abslines_infos_perobj[mask_obj]):

        # skipping everything if we don't plot nondetections
        if not display_nondet:
            continue

        # defining the index of the object in the entire array if asked to, in order to avoid changing colors
        if global_colors:
            i_obj_glob = np.argwhere(obj_list == obj_list[mask_obj][i_obj_base])[0][0]
        else:
            i_obj_glob = i_obj_base

        '''
        # The shape of each abslines_obj is (uncert,info,line,obs)
        '''

        # we use non-detection-masked arrays for non detection to plot them even while restricting the colors to a part of the sample

        x_hid_base = lum_list[mask_obj][i_obj_base].T[2][0] / lum_list[mask_obj][i_obj_base].T[1][0]
        y_hid_base = lum_list[mask_obj][i_obj_base].T[4][0]

        if len(mask_obj)==1 and np.ndim(hid_plot)==4:
            x_hid_uncert=hid_plot.transpose(2,0,1,3)[i_obj][0]
            y_hid_uncert=hid_plot.transpose(2,0,1,3)[i_obj][1]
        else:
            x_hid_uncert = hid_plot_use.T[mask_obj][i_obj_base].T[0]
            y_hid_uncert = hid_plot_use.T[mask_obj][i_obj_base].T[1]

        # reconstructing standard arrays
        x_hid_uncert = np.array([[subelem for subelem in elem] for elem in x_hid_uncert])
        y_hid_uncert = np.array([[subelem for subelem in elem] for elem in y_hid_uncert])

        if broad_mode!=False:

            x_hid_base = hid_plot_use[0][0][mask_obj][i_obj_base]

            #done this way to avoid overwriting lum_list if using += on y_hid_base
            if lum_broad_soft:
                y_hid_base=lum_list[mask_obj][i_obj_base].T[4][0]+lum_broad_single[0]

        # defining the non detection as strictly non detection or everything below the significance threshold
        if display_nonsign:
            mask_det = abslines_obj_base[0][4][mask_lines] > 0.

        else:
            mask_det = abslines_obj_base[0][4][mask_lines] >= slider_sign

        # defining the mask for the time interval restriction
        datelist_obj = Time(np.array([date_list[mask_obj][i_obj_base] \
                                      for i in range(sum(mask_lines_ul if display_upper else mask_lines))]).astype(str))
        mask_intime = (datelist_obj >= Time(slider_date[0])) & (datelist_obj <= Time(slider_date[1]))

        mask_intime_norepeat = (Time(date_list[mask_obj][i_obj_base].astype(str)) >= Time(slider_date[0])) & (
                    Time(date_list[mask_obj][i_obj_base].astype(str)) <= Time(slider_date[1]))

        if broad_mode=='BAT':
            mask_intime=np.array([(elem) & mask_with_broad for elem in mask_intime])
            mask_intime_norepeat=(mask_intime_norepeat) & mask_with_broad
        if restrict_match_INT:
            mask_intime=np.array([(elem) & mask_withtime_INT for elem in mask_intime])
            mask_intime_norepeat=(mask_intime_norepeat) & mask_withtime_INT

        # defining the mask
        prev_mask_nondet = np.isnan(
            np.array([np.nan if len(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                          max(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                      for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))]))

        mask_nondet = (np.isnan(
            np.array([np.nan if len(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                          max(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                      for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))]))) & (mask_intime_norepeat)

        # testing if the source has detections with current restrictions to adapt the color when using source colors, if asked to
        if obj_list[mask_obj][i_obj_base] not in obj_list[mask_obj_withdet]:
            source_nondet = True

        else:
            source_nondet = False

            # increasing the counter for sources with no non detections but detections
            if len(x_hid_base[mask_nondet]) == 0:
                id_obj_det += 1

        if len(x_hid_base[mask_nondet]) > 0:
            # note: due to problems with colormapping of the edgecolors we directly compute the color of the edges with a normalisation
            norm_cmap_incl = mpl.colors.Normalize(0, 90)

            if global_colors:
                norm_cmap_time = mpl.colors.Normalize(
                    min(mdates.date2num(ravel_ragged(date_list)[global_mask_intime_norepeat])),
                    max(mdates.date2num(ravel_ragged(date_list)[global_mask_intime_norepeat])))
            else:
                norm_cmap_time = mpl.colors.Normalize(
                    min(mdates.date2num(ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                    max(mdates.date2num(ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])))
            if display_upper:

                # we define the upper limit range of points independantly to be able to have a different set of lines used for detection and
                # upper limits if necessary

                mask_det_ul = (abslines_obj_base[0][4][mask_lines_ul] > 0.) & (mask_intime)
                mask_det_ul = (abslines_obj_base[0][4][mask_lines_ul] >= slider_sign) & (mask_intime)

                mask_nondet_ul = np.isnan(np.array( \
                    [np.nan if len(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]]) == 0 else \
                         max(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]]) \
                     for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))])) & (mask_intime_norepeat)

                # defining the sizes of upper limits (note: capped to 75eV)
                obj_size_ul = np.array(
                    [np.nan if len(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]]) != 0 else \
                         min(max(abslines_obj_base[0][5][mask_lines_ul].T[i_obs][~mask_det_ul.T[i_obs]]), 75) \
                     for i_obs in range(len(abslines_obj_base[0][0][mask_lines_ul].T))])

                # creating a display order which is the reverse of the EW size order to make sure we do not hide part the ul
                # not needed now that the UL are not filled colorwise
                # obj_order_sign_ul=obj_size_ul[mask_nondet_ul].argsort()[::-1]

                # there is no need to use different markers unless we display source per color, so we limit the different triangle to this case
                marker_ul_curr = marker_abs if display_single else marker_ul_top if \
                    ((id_obj_nondet if source_nondet else id_obj_det) if split_cmap_source else i_obj_base) % 2 != 0 and \
                    radio_info_cmap == 'Source' else marker_ul

                if radio_info_cmap == 'Instrument':
                    color_data = [telescope_colors[elem] for elem in
                                  instru_list[mask_obj][i_obj_base][mask_nondet_ul]]

                    edgec_scat = [colors.to_rgba(elem) for elem in color_data]
                else:

                    edgec_scat = (colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                      (colors_nondet.to_rgba(id_obj_nondet) if source_nondet else \
                                           colors_det.to_rgba(
                                               id_obj_det))) if radio_info_cmap == 'Source' and color_nondet else \
                        cmap_info(norm_cmap_incl(incl_cmap_base[i_obj_base][cmap_incl_type])) \
                            if radio_info_cmap == 'Inclination' else \
                            cmap_info(
                                norm_cmap_time(mdates.date2num(date_list[mask_obj][i_obj_base][mask_nondet_ul]))) \
                                if radio_info_cmap == 'Time' else \
                                cmap_info(cmap_norm_info(nh_plot.T[mask_obj].T[0][i_obj_base][
                                                             mask_nondet_ul])) if radio_info_cmap == 'nH' else \
                                    cmap_info(
                                        cmap_norm_info(kt_plot.T[mask_obj].T[0][i_obj_base][mask_nondet_ul])) if (
                                                1 and radio_info_cmap == 'kT') else \
                                        diago_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                            if radio_info_cmap == 'custom_line_struct' else \
                                            custom_states_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                                if radio_info_cmap == 'custom_acc_states' else \
                                            custom_outburst_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                                if radio_info_cmap=='custom_outburst' else \
                                            'grey'

                # adding a failsafe to avoid problems when nothing is displayed
                if len(edgec_scat) == 0:
                    edgec_scat = None

                if broad_mode == 'BAT' and not sign_broad_hid_BAT:
                    alpha_nondet_ul_full = np.where( mask_sign_high_E, 1, 0.3)

                    alpha_nondet_ul = alpha_nondet_ul_full[mask_nondet_ul]


                    # in the case where alpha is not singular the len of the color keyword needs to match it
                    color_val_nondet_ul =np.array([[0.,0.,0.,0.]]* len(alpha_nondet_ul))

                else:
                    alpha_nondet_ul=alpha_abs-0.2

                    color_val_nondet_ul=[0.,0.,0.,0.]

                elem_scatter_nondet = ax_hid.scatter(
                    x_hid_base[mask_nondet_ul], y_hid_base[mask_nondet_ul], marker=marker_ul_curr,
                    facecolor=color_val_nondet_ul, edgecolor=edgec_scat,
                    s=norm_s_lin * obj_size_ul[mask_nondet_ul] ** norm_s_pow,
                    label='' if not color_nondet else (
                        obj_list[mask_obj][i_obj_base] if not label_obj_plotted[i_obj_base] and \
                                                          (radio_info_cmap == 'Source' or display_edgesource) else ''),
                    zorder=500, alpha=alpha_nondet_ul,
                    cmap=cmap_info if radio_info_cmap in ['Inclination', 'Time'] else None, ls='--' if (
                                display_incl_inside and not bool_incl_inside[mask_obj][
                            i_obj_base] or dash_noincl and bool_noincl[mask_obj][i_obj_base]) else 'solid',
                    plotnonfinite=True)

                #we re-enforce the facecolor after otherwise because it can be overwirtten by alpha modifications
                elem_scatter_nondet.set_facecolor('none')

                scatter_nondet += [elem_scatter_nondet]

            else:

                if radio_info_cmap == 'Instrument':
                    color_data = [telescope_colors[elem] for elem in
                                  instru_list[mask_obj][i_obj_base][mask_nondet]]

                    c_scat_nondet = [colors.to_rgba(elem) for elem in color_data]
                else:

                    c_scat_nondet = np.array([(colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                                   (colors_nondet.to_rgba(id_obj_nondet) if source_nondet else \
                                                        colors_det.to_rgba(
                                                            id_obj_det)))]) if radio_info_cmap == 'Source' and color_nondet else \
                        np.repeat(incl_cmap_base[i_obj_base][cmap_incl_type], sum(mask_nondet)) \
                            if radio_info_cmap == 'Inclination' else \
                            mdates.date2num(date_list[mask_obj][i_obj_base][mask_nondet]) \
                                if radio_info_cmap == 'Time' else \
                                nh_plot.T[mask_obj].T[0][i_obj_base][mask_nondet] if radio_info_cmap == 'nH' else \
                                kt_plot.T[mask_obj].T[0][i_obj_base][mask_nondet] if radio_info_cmap == 'kT' else \
                                    diago_color[mask_obj][i_obj_base][mask_nondet] \
                                        if radio_info_cmap == 'custom_line_struct' else \
                                    custom_states_color[mask_obj][i_obj_base][mask_nondet] \
                                        if radio_info_cmap == 'custom_acc_states' else \
                                    custom_outburst_color[mask_obj][i_obj_base][mask_nondet] \
                                        if radio_info_cmap == 'custom_outburst' else \
                                        'grey'

                if broad_mode == 'BAT' and not sign_broad_hid_BAT:
                    alpha_nondet_full = np.where(mask_sign_high_E, 1, 0.3)

                    alpha_nondet= alpha_nondet_full[mask_nondet]



                else:
                    alpha_nondet=alpha_abs-0.2


                elem_scatter_nondet = ax_hid.scatter(x_hid_base[mask_nondet], y_hid_base[mask_nondet],
                                                     marker=marker_nondet,
                                                     c=c_scat_nondet, cmap=cmap_info, norm=cmap_norm_info,
                                                     label='' if not color_nondet else (
                                                         obj_list[mask_obj][i_obj_base] if not label_obj_plotted[
                                                             i_obj_base] and \
                                                        (radio_info_cmap == 'Source' or display_edgesource) else ''),
                                                     zorder=1000,
                                                     edgecolor='black', alpha=alpha_nondet,
                                                     plotnonfinite=True)

                # note: the plot non finite allows to plot the nan values passed to the colormap with the color predefined as bad in
                # the colormap

                if display_hid_error:

                    # in order to get the same clim as with the standard scatter plots, we manually readjust the rgba values of the colors before plotting
                    # the errorbar "empty" and changing its color manually (because as of now matplotlib doesn't like multiple color inputs for errbars)
                    if radio_info_cmap in type_1_cm:
                        if radio_info_cmap == 'Inclination':
                            cmap_norm_info.vmin = 0
                            cmap_norm_info.vmax = 90
                        elif radio_info_cmap == 'Time':
                            cmap_norm_info.vmin = max(min(mdates.date2num(
                                ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                                      mdates.date2num(slider_date[0]))
                            cmap_norm_info.vmax = min(max(mdates.date2num(
                                ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                                      mdates.date2num(slider_date[1]))

                        elif radio_info_cmap == 'nH':
                            cmap_norm_info.vmin = min(
                                ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat])
                            cmap_norm_info.vmax = max(
                                ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat])
                        elif radio_info_cmap == 'kT':
                            cmap_norm_info.vmin = kt_min
                            cmap_norm_info.vmax = kt_max

                        colors_func = mpl.cm.ScalarMappable(norm=cmap_norm_info, cmap=cmap_info)

                        c_scat_nondet_rgba_clim = colors_func.to_rgba(c_scat_nondet)

                    elem_err_nondet = ax_hid.errorbar(x_hid_uncert[0][mask_nondet], y_hid_uncert[0][mask_nondet],
                                                      xerr=x_hid_uncert[1:].T[mask_nondet].T,
                                                      yerr=y_hid_uncert[1:].T[mask_nondet].T, marker='None',
                                                      linestyle='None', linewidth=0.5,
                                                      c=c_scat_nondet if radio_info_cmap not in type_1_cm else None,
                                                      label='', zorder=1000, alpha=1.)

                    if radio_info_cmap in type_1_cm:
                        for elem_children in elem_err_nondet.get_children()[1:]:
                            elem_children.set_colors(c_scat_nondet_rgba_clim)

            if radio_info_cmap == 'Source' and color_nondet:
                label_obj_plotted[i_obj_base] = True

            if radio_info_cmap in type_1_cm:

                if radio_info_cmap == 'Inclination':
                    elem_scatter_nondet.set_clim(vmin=0, vmax=90)

                    # if display_hid_error:
                    #     elem_err_nondet.set_clim(vmin=0,vmax=90)

                elif radio_info_cmap == 'Time':
                    if global_colors:
                        elem_scatter_nondet.set_clim(
                            vmin=min(mdates.date2num(ravel_ragged(date_list))),
                            vmax=max(mdates.date2num(ravel_ragged(date_list))))
                    else:
                        elem_scatter_nondet.set_clim(
                            vmin=max(min(mdates.date2num(
                                ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                     mdates.date2num(slider_date[0])),
                            vmax=min(max(mdates.date2num(
                                ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                     mdates.date2num(slider_date[1])))
                        # if display_hid_error:
                    #     elem_err_nondet.set_clim(
                    #     vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj][global_mask_intime_norepeat]))),mdates.date2num(slider_date[0])),
                    #     vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj][global_mask_intime_norepeat]))),mdates.date2num(slider_date[1])))

                elif radio_info_cmap == 'nH':
                    if global_colors:
                        elem_scatter_nondet.set_clim(vmin=min(ravel_ragged(nh_plot[0])),
                                                     vmax=max(ravel_ragged(nh_plot[0])))
                    else:
                        elem_scatter_nondet.set_clim(
                            vmin=min(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]),
                            vmax=max(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]))
                elif radio_info_cmap == 'kT':
                    elem_scatter_nondet.set_clim(vmin=kt_min,
                                                 vmax=kt_max)

                if len(elem_scatter_nondet.get_sizes()) > 0:
                    is_colored_scat_nondet = True

                # creating the colorbar at the end if it hasn't been created with the detections
                if i_obj_base == len(
                        abslines_infos_perobj[mask_obj]) - 1 and not is_colored_scat and is_colored_scat_nondet:

                    # creating an empty scatter with a 'c' value to serve as base for the colorbar
                    elem_scatter_empty = ax_hid.scatter(x_hid_base[mask_nondet][False], y_hid_base[mask_nondet][False],
                                                        marker=None,
                                                        c=cmap_info(norm_cmap_time(mdates.date2num(
                                                            date_list[mask_obj][i_obj_base][mask_nondet])))[False],
                                                        label='', zorder=1000, edgecolor=None, cmap=cmap_info, alpha=1.)

                    if radio_info_cmap == 'Inclination':

                        elem_scatter_empty.set_clim(vmin=0, vmax=90)

                        cb = plt.colorbar(elem_scatter_empty, cax=ax_cb)

                        cb.set_label(cmap_incl_type_str + ' of the source inclination (°)', labelpad=10)
                    elif radio_info_cmap == 'Time':

                        low_bound_date=max(min(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj])\
                                                                 [global_mask_intime_norepeat])),
                                     mdates.date2num(slider_date[0]))

                        high_bound_date=min(max(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj])\
                                                                 [global_mask_intime_norepeat])),
                                     mdates.date2num(slider_date[1]))

                        elem_scatter_empty.set_clim(vmin=low_bound_date,vmax=high_bound_date)

                        # manually readjusting for small durations because the AutoDateLocator doesn't work well
                        time_range = high_bound_date-low_bound_date

                        if time_range < 150:
                            date_format = mdates.DateFormatter('%Y-%m-%d')
                        elif time_range < 1825:
                            date_format = mdates.DateFormatter('%Y-%m')
                        else:
                            date_format = mdates.AutoDateFormatter(mdates.AutoDateLocator())

                        cb = plt.colorbar(elem_scatter_empty, cax=ax_cb, ticks=mdates.AutoDateLocator(),
                                          format=date_format)

                        cb.set_label('Observation date', labelpad=10)

                    elif radio_info_cmap == 'nH':
                        elem_scatter_empty.set_clim(
                            vmin=min(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]),
                            vmax=max(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]))
                        cb = plt.colorbar(elem_scatter_empty, cax=ax_cb, extend='min')

                        cb.set_label(r'nH ($10^{22}$ cm$^{-2}$)')

                    elif radio_info_cmap == 'kT':
                        elem_scatter_empty.set_clim(vmin=kt_min, vmax=kt_max)

                        cb = plt.colorbar(elem_scatter_empty, cax=ax_cb)

                        cb.set_label(r'disk temperature (keV)')

            # only adding to the index if there are non detections
            if source_nondet:
                id_obj_nondet += 1
            else:
                id_obj_det += 1

    # taking off the axes in the colorbar axes if no colorbar was displayed

    if radio_info_cmap not in type_1_colorcode and cb is None:
        ax_cb.axis('off')

    #### Displaying arrow evolution if needed and if there are points
    if display_single and display_evol_single and sum(global_mask_intime_norepeat) > 1 and display_nondet:

        # odering the points depending on the observation date
        date_order = datelist_obj[0][mask_intime[0]].argsort()

        # plotting the main line between all points
        ax_hid.plot(x_hid_base[mask_intime[0]][date_order], y_hid_base[mask_intime[0]][date_order], color='grey',
                    linewidth=0.5,alpha=0.5)

        # computing the position of the arrows to superpose to the lines
        xarr_start = x_hid_base[mask_intime[0]][date_order][range(len(x_hid_base[mask_intime[0]][date_order]) - 1)].astype(float)
        xarr_end = x_hid_base[mask_intime[0]][date_order][range(1, len(x_hid_base[mask_intime[0]][date_order]))].astype(float)
        yarr_start = y_hid_base[mask_intime[0]][date_order][range(len(y_hid_base[mask_intime[0]][date_order]) - 1)]
        yarr_end = y_hid_base[mask_intime[0]][date_order][range(1, len(y_hid_base[mask_intime[0]][date_order]))]

        #mask to know if we can do a log computation of the arrow positions or not (aka not broad mode or
        #x positions above the lintresh threshold
        x_arr_log_ok=np.array([not broad_mode or \
                                   (elem_x_s>=broad_x_linthresh and elem_x_e>=broad_x_linthresh)\
                                   for (elem_x_s,elem_x_e) in zip(xarr_start,xarr_end)])

        #preventing error when no point
        if sum(x_arr_log_ok)!=0:

            #linear version first
            xpos = (xarr_start + xarr_end) / 2
            ypos = (yarr_start + yarr_end) / 2

            xdir = xarr_end - xarr_start
            ydir = yarr_end - yarr_start

            #log version in the mask
            xpos[x_arr_log_ok] = 10**((np.log10(xarr_start[x_arr_log_ok]) + np.log10(xarr_end[x_arr_log_ok])) / 2)
            ypos[x_arr_log_ok] = 10**((np.log10(yarr_start[x_arr_log_ok]) + np.log10(yarr_end[x_arr_log_ok])) / 2)

            xdir[x_arr_log_ok] = 10**(np.log10(xarr_end[x_arr_log_ok]) - np.log10(xarr_start[x_arr_log_ok]))
            ydir[x_arr_log_ok] = 10**(np.log10(yarr_end[x_arr_log_ok]) - np.log10(yarr_start[x_arr_log_ok]))

            # this is the offset from the position, since we make the arrow start at the middle point of
            # the segment we don't want it to go any further so we put it almost at the same value
            arrow_size_frac=0.001

            for X, Y, dX, dY,log_ok in zip(xpos, ypos, xdir, ydir,x_arr_log_ok):
                if log_ok:
                    ax_hid.annotate("", xytext=(X, Y), xy=(10**(np.log10(X) + arrow_size_frac * np.log10(dX)),
                                                           10**(np.log10(Y) + arrow_size_frac * np.log10(dY))),
                                    arrowprops=dict(arrowstyle='->', color='grey',alpha=0.5), size=10)
                else:
                    ax_hid.annotate("", xytext=(X, Y), xy=(X+ arrow_size_frac *dX,
                                                           Y+ arrow_size_frac *dY),
                                    arrowprops=dict(arrowstyle='->', color='grey',alpha=0.5), size=10)


        # else:
        #     xpos = (xarr_start + xarr_end) / 2
        #     ypos = (yarr_start + yarr_end) / 2
        # 
        #     xdir = xarr_end - xarr_start
        #     ydir = yarr_end - yarr_start


    ####displaying the thresholds if asked to

    if display_dicho:

        if broad_mode=='BAT':
            if HR_broad_6_10:
                # vertical
                pass
                # ax_hid.axline((0.1, 1e-6), (0.1, 10), ls='--', color='grey')
            else:
                # vertical
                ax_hid.axline((0.1, 1e-6), (0.1, 10), ls='--', color='grey')

        else:
            # horizontal
            ax_hid.axline((0.01, 1e-2), (10, 1e-2), ls='--', color='grey')

            # vertical
            ax_hid.axline((0.8, 1e-6), (0.8, 10), ls='--', color='grey')

        # restricting the graph to the portion inside the thrsesolds
        # ax_hid.set_xlim(ax_hid.get_xlim()[0],0.8)
        # ax_hid.set_ylim(1e-2,ax_hid.get_ylim()[1])

    ''''''''''''''''''
    #### legends
    ''''''''''''''''''

    if radio_info_cmap == 'Source' or display_edgesource:

        # looks good considering the size of the graph
        n_col_leg_source = 4 if paper_look else (5 if sum(mask_obj) < 30 else 6)

        old_legend_size = mpl.rcParams['legend.fontsize']

        mpl.rcParams['legend.fontsize'] = (5.5 if sum(mask_obj) > 30 and radio_info_cmap == 'Source' else 7) + (
            3 if paper_look else 0)

        hid_legend = fig_hid.legend(loc='lower center', ncol=n_col_leg_source, bbox_to_anchor=(0.475, -0.11))

        elem_leg_source, labels_leg_source = plt.gca().get_legend_handles_labels()

        # selecting sources with both detections and non detections
        sources_uniques = np.unique(labels_leg_source, return_counts=True)
        sources_detnondet = sources_uniques[0][sources_uniques[1] != 1]

        # recreating the elem_leg and labels_leg with grouping but only if the colormaps are separated because then it makes sense
        if split_cmap_source:

            leg_source_gr = []
            labels_leg_source_gr = []

            for elem_leg, elem_label in zip(elem_leg_source, labels_leg_source):
                if elem_label in sources_detnondet:

                    # only doing it for the first iteration
                    if elem_label not in labels_leg_source_gr:
                        leg_source_gr += [tuple(np.array(elem_leg_source)[np.array(labels_leg_source) == elem_label])]
                        labels_leg_source_gr += [elem_label]

                else:
                    leg_source_gr += [elem_leg]
                    labels_leg_source_gr += [elem_label]

            # updating the handle list
            elem_leg_source = leg_source_gr
            labels_leg_source = labels_leg_source_gr

        n_obj_leg_source = len(elem_leg_source)

        def n_lines():
            return len(elem_leg_source) // n_col_leg_source + (1 if len(elem_leg_source) % n_col_leg_source != 0 else 0)

        # inserting blank spaces until the detections have a column for themselves
        while n_lines() < n_obj_withdet:
            # elem_leg_source.insert(5,plt.Line2D([],[], alpha=0))
            # labels_leg_source.insert(5,'')

            elem_leg_source += [plt.Line2D([], [], alpha=0)]
            labels_leg_source += ['']

        # removing the first version with a non-aesthetic number of columns
        hid_legend.remove()

        # recreating it with updated spacing
        hid_legend = fig_hid.legend(elem_leg_source, labels_leg_source, loc='lower center',
                                    ncol=n_col_leg_source,
                                    bbox_to_anchor=(0.475, -0.02 * n_lines() - (
                                        0.02 * (6 - n_lines()) if paper_look else 0) - (0.1 if paper_look else 0)),
                                    handler_map={tuple: HandlerTuple(ndivide=None, pad=1.)},
                                    columnspacing=0.5 if paper_look else 1)

        '''
        # maintaining a constant marker size in the legend (but only for markers)
        # note: here we cannot use directly legend_handles because they don't consider the second part of the legend tuples
        # We thus use the findobj method to search in all elements of the legend
        '''
        for elem_legend in hid_legend.findobj():

            #### find a way to change the size of this

            if type(elem_legend) == mpl.collections.PathCollection:
                if len(elem_legend._sizes) != 0:
                    for i in range(len(elem_legend._sizes)):
                        elem_legend._sizes[i] = 50 + (80 if paper_look else 0) + (
                            30 if n_lines() < 6 else 0) if display_upper else 30 + (40 if paper_look else 0) + (
                            10 if n_lines() < 6 else 0)

                    if paper_look and display_upper:
                        elem_legend.set_linewidth(2)

                    # changing the dash type of dashed element for better visualisation:
                    if elem_legend.get_dashes() != [(0.0, None)]:
                        elem_legend.set_dashes((0, (5, 1)))

        # old legend version
        # hid_legend=fig_hid.legend(loc='upper right',ncol=1,bbox_to_anchor=(1.11,0.895) if bigger_text and radio_info_cmap=='Source' \
        #                           and color_nondet else (0.9,0.88))

        mpl.rcParams['legend.fontsize'] = old_legend_size

    if display_single:
        hid_det_examples = [
            (Line2D([0], [0], marker=marker_abs, color='white',
                                                            markersize=50 ** (1 / 2), alpha=1., linestyle='None',
                                                            markeredgecolor='black', markeredgewidth=2)) \
                if display_upper else
            (Line2D([0], [0], marker=marker_nondet, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2)),
            (Line2D([0], [0], marker=marker_abs, color='grey', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2))]
    else:
        hid_det_examples = [
            ((Line2D([0], [0], marker=marker_ul, color='white', markersize=50 ** (1 / 2), alpha=alpha_ul, linestyle='None',
                     markeredgecolor='black', markeredgewidth=2),
              Line2D([0], [0], marker=marker_ul_top, color='white', markersize=50 ** (1 / 2), alpha=alpha_ul,
                     linestyle='None', markeredgecolor='black', markeredgewidth=2)) \
                 if radio_info_cmap == 'Source' else Line2D([0], [0], marker=marker_ul, color='white',
                                                            markersize=50 ** (1 / 2), alpha=alpha_ul, linestyle='None',
                                                            markeredgecolor='black', markeredgewidth=2)) \
                if display_upper else
            (Line2D([0], [0], marker=marker_nondet, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2)),
            (Line2D([0], [0], marker=marker_abs, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2))]

    if display_nonsign:
        hid_det_examples += [
            (Line2D([0], [0], marker=marker_abs, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='grey', markeredgewidth=2))]

    mpl.rcParams['legend.fontsize'] = 7 + (2 if paper_look and not zoom else 0)

    # marker legend

    # manual custom subplot adjust to get the same scale for the 3 sources with ULs and for the zoomed 5 sources with detection
    # to be put in the 5 sources
    custom=False

    fig_hid.legend(handles=hid_det_examples, loc='center left',
                   labels=['upper limit' if display_upper else 'non detection ',
                           'absorption line detection\n above ' + (r'3$\sigma$' if slider_sign == 0.997 else str(
                               slider_sign * 100) + '%') + ' significance',
                           'absorption line detection below ' + str(slider_sign * 100) + ' significance.'],
                   title='',
                   bbox_to_anchor=(0.125, 0.829 - (
                       0.012 if paper_look and not zoom else 0)) if bigger_text and square_mode else (
                   0.125, 0.82), handler_map={tuple: mpl.legend_handler.HandlerTuple(None)},
                   handlelength=2, handleheight=2., columnspacing=1.)

    # note: upper left anchor (0.125,0.815)
    # note : upper right anchor (0.690,0.815)
    # note: 0.420 0.815

    # size legend

    if display_upper and not display_single:
        # displaying the 
        if radio_info_cmap == 'Source':
            hid_size_examples = [(Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500),
                                  Line2D([0], [0], marker=marker_ul_top, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500),
                                  Line2D([0], [0], marker=marker_ul_top, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500),
                                  Line2D([0], [0], marker=marker_ul_top, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500))]
        else:
            hid_size_examples = [(Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500))]
    else:
        hid_size_examples = [(Line2D([0], [0], marker=marker_abs, color='black',
                                     markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None')),
                             (Line2D([0], [0], marker=marker_abs, color='black',
                                     markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None')),
                             (Line2D([0], [0], marker=marker_abs, color='black',
                                     markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None'))]

    ew_legend = fig_hid.legend(handles=hid_size_examples, loc='center left', labels=['5 eV', '20 eV', '50 eV'],
                                title='Equivalent widths',
                                bbox_to_anchor=(0.125, 0.218 + (
                                    0.028 if paper_look and not zoom else 0)) if bigger_text and square_mode else (
                                0.125, 0.218), handleheight=4, handlelength=4, facecolor='None')

    if radio_info_cmap == 'Instrument':
        instru_examples = np.array([Line2D([0], [0], marker=marker_abs, color=list(telescope_colors.values())[i],
                                           markeredgecolor='black',
                                           markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None')\
                                    for i in range(len(telescope_list))])

        telescope_choice_sort=np.array(choice_telescope)
        telescope_choice_sort.sort()

        instru_ind = [np.argwhere(np.array(telescope_list) == elem)[0][0] for elem in telescope_choice_sort]

        instru_legend = fig_hid.legend(handles=instru_examples[instru_ind].tolist(), loc='upper right',
                                       labels=telescope_choice_sort.tolist(),
                                       title=radio_info_cmap,
                                       bbox_to_anchor=(0.900, 0.88) if bigger_text and square_mode else (0.825, 0.918),
                                       handleheight=1, handlelength=4, facecolor='None')

    # manual custom subplot adjust to get the same scale for the 3 visible sources plot and for the zoomed 5 sources with detection
    # elem=fig_hid.add_axes([0.5, 0.792, 0.1, 0.1])
    # mpl.rcParams.update({'font.size': 2})
    # elem.axis('off')

    # manual custom subplot adjust to get the same scale for the 3 sources with ULs and for the zoomed 5 sources with detection
    #to be put in the 5 sources

    if custom:
        plt.subplots_adjust(top=0.863)

    # note: 0.9 0.53
    # destacked version
    # fig_hid.legend(handles=hid_size_examples,loc='center left',labels=['5 eV','20 eV','50 eV'],title='Equivalent widths',
    #             bbox_to_anchor=(0.125,0.235) if bigger_text and square_mode else (0.125,0.235),handler_map = {tuple:mpl.legend_handler.HandlerTuple(None)},handlelength=8,handleheight=5,columnspacing=5.)


def distrib_graph(data_perinfo,info,dict_linevis,data_ener=None,conf_thresh=0.99,indiv=False,save=False,close=False,streamlit=False,bigger_text=False,split=None):
    
    '''
    repartition diagram from all the observations in the current pool (i.e. all objects/obs/lines).
    
    Use the 'info' keyword to graph flux,ewidth, bshift or ener
    
    Use the 'indiv' keyword to plot for all lines simultaneously or 6 plots for the 6 single lines
    
    Non detections are filtered via 0 values in significance
    
    Detections above and below the given threshold are highlighted
    
    we ravel the last 2 dimensions with a custom function since for multi object plots the sequences can be ragged and the custom .ravel function
    doesn't work
    '''
    n_infos=dict_linevis['n_infos']
    range_absline=dict_linevis['range_absline']
    mask_lines=dict_linevis['mask_lines']
    list_id_lines=np.array(range_absline)[mask_lines]
    bins_bshift=dict_linevis['bins_bshift']
    bins_ener=dict_linevis['bins_ener']
    mask_obj=dict_linevis['mask_obj']

    
    if streamlit:
        display_nonsign=dict_linevis['display_nonsign']
        scale_log_ew=dict_linevis['scale_log_ew']
        obj_disp_list=dict_linevis['obj_list'][mask_obj]
        instru_list=dict_linevis['instru_list'][mask_obj]
        date_list=dict_linevis['date_list'][mask_obj]
        width_plot_restrict=dict_linevis['width_plot_restrict']
        glob_col_source=dict_linevis['glob_col_source']
        cmap_color_det=dict_linevis['cmap_color_det']
        split_dist_method=dict_linevis['split_dist_method']
    else:
        display_nonsign=True
        scale_log_ew=False
        glob_col_source=False
        cmap_color_det=mpl.cm.plasma
        split_dist_method=False
        
    save_dir=dict_linevis['save_dir']
    save_str_prefix=dict_linevis['save_str_prefix']
    args_cam=dict_linevis['args_cam']
    args_line_search_e=dict_linevis['args_line_search_e']
    args_line_search_norm=dict_linevis['args_line_search_norm']

    #range of the existing lines for loops
    range_line=range(len(list_id_lines))
    
    line_mode=info=='lines'
    if not line_mode:
        ind_info=np.argwhere([elem in info for elem in ['ew','bshift','ener','lineflux','time','width']])[0][0]
    else:
        #we don't care here
        ind_info=0
        
    split_off=split=='Off'
    split_source=split=='Source'
    split_instru=split=='Instrument'
    
    #main data array
    data_plot=data_perinfo[ind_info] if ind_info not in [2,4,5] else data_ener if ind_info==2\
            else date_list if ind_info==4 else width_plot_restrict

    ratio_mode='ratio' in info
    
    if indiv:
        graph_range=range_absline
    else:
        #using a list index for the global graphs allows us to keep the same structure
        #however we need to restrict it to the currently plotted lines in streamlit mode
        if streamlit:
            graph_range=[range(len([elem for elem in mask_lines if elem]))]
        else:
            graph_range=[range_absline]

    #computing the range of the ew bins from the global graph to get the same scale for all individual graphs)
    
    if scale_log_ew:
        bins_ew=np.geomspace(1,min(100,(max(ravel_ragged(data_perinfo[0][0]))//5+1)*5),20)
    else:
        bins_ew=np.arange(5, min(100, (max(ravel_ragged(data_perinfo[0][0])) // 5 + 1) * 5), 2.5)
        # bins_ew=np.linspace(5,min(100,(max(ravel_ragged(data_perinfo[0][0]))//5+1)*5),20)

        
    bins_ewratio=np.linspace(0.2,4,20)
    
    if n_infos>=5:
        if len(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0])>0])>0 and len(ravel_ragged(data_perinfo[3][0]).nonzero()[0])!=0:
            
            bins_flux=np.geomspace(max(1e-16,min(1e-13,(min(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0])>0])*0.9))),
                                   max(1e-10,(max(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0])>0])*1.1)),20)
        else:
            bins_flux=np.geomspace(1e-13,1e-10,20)
    else:
        bins_flux=None

    #for now
    bins_time=None
    
    #linear bin widths between 0 (where most widths value lie) and 0.05 which is the hard limit
    bins_width=np.linspace(0,5500,12)

    #sorting the bins in an array depending on the info asked
    bins_info=[bins_ew,bins_bshift,bins_ener,bins_flux,bins_time,bins_width]
    
    bins_ratio=[bins_ewratio]
    
    #and fetching the current one (or custom bins for other modes)
    hist_bins=bins_info[ind_info] if not ratio_mode else bins_ratio[ind_info] if not line_mode else range(graph_range)
    
    #fetching the global boolean array for the graph size definition
    bool_det_glob=ravel_ragged(data_perinfo[4][0]).astype(float)
        
    #the "and" keyword doesn't work for arrays so we use & instead (which requires parenthesis)
    bool_det_glob=(bool_det_glob!=0.) & (~np.isnan(bool_det_glob))
    
    #computing the maximal height of the graphs (in order to keep the same y axis scaling on all the individual lines graphs)
    max_height=max((np.histogram(ravel_ragged(data_plot[0])[bool_det_glob],bins=bins_info[ind_info]))[0])
       
    for i in graph_range:
    
        if ratio_mode:
            
            #we fetch the index list corresponding to the info string at the end of the info provided
            ratio_indexes_x=ratio_choices[info[-2:]]
            
        fig_hist,ax_hist=plt.subplots(1,1,figsize=(6,4) if not split_off else (6,6) if bigger_text else (10,8))
        
        if not bigger_text:
            if indiv:
                fig_hist.suptitle('Repartition of the '+info_str[ind_info]+' of the '+lines_std[lines_std_names[3+i]]+' absorption line')
            else:
                if ratio_mode:
                    fig_hist.suptitle('Repartition of the '+ratio_choices_str[info[-2:]]+' ratio in currently selected sources')
                else:
                    fig_hist.suptitle('Repartition of the '+info_str[ind_info]+
                                  (' in currently selected lines and sources' if streamlit else ' absorption lines'))
            

        #using a log x axis if the 
        ax_hist.set_ylabel(r'Detections')
        ax_hist.set_xlabel('' if line_mode else (ratio_choices_str[info[-2:]]+' ' if ratio_mode else '')+
                           axis_str[ind_info].replace(' (eV)',('' if ratio_mode else ' (eV)'))+
                           (' ratio' if ratio_mode else ''))
        
        if split_off:
            ax_hist.set_ylim([0,max_height+0.25])
            
        if ind_info in [0,3]:            
            if ind_info==3 or scale_log_ew:
                ax_hist.set_xscale('log')
                
        #these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others
        bool_sign=ravel_ragged(data_perinfo[4][0][i]).astype(float)

        #single variable for more advanced bar plots with splitting
        bool_signdet=(bool_sign>=conf_thresh) & (~np.isnan(bool_sign))
        
        #split variables for old computations with significance or not
        bool_det=(bool_sign!=0.) & (~np.isnan(bool_sign))
        bool_sign=bool_sign[bool_det]>=conf_thresh
        
        if info=='width':
            
            #creating a mask for widths that are not compatible with 0 at 3 sigma\
            #(different from the scatter here as we don't want the 0 values)
            bool_sign_width=ravel_ragged(data_perinfo[7][0][i]).astype(float)!=0
            
            bool_sign=bool_sign & bool_sign_width[bool_det]
            
        if not split_off:
            
            if split_source:
                #source by source version
                sign_det_split=np.array([ravel_ragged(data_perinfo[4][0][i].T[j]) for j in range(len(obj_disp_list))],
                                        dtype=object)
        
                #unused for now        
                # #creating the bool det and bool sign masks
                # bool_det_split=np.array([(elem!=0.) & (~np.isnan(elem)) for elem in sign_det_split])
                
                #note: here we don't link bool sign to bool det because we won't bother plotting non significant lines
                bool_sign_split=np.array([(elem>=conf_thresh) & (~np.isnan(elem)) for elem in sign_det_split],
                                         dtype=object)
                        
                if info=='width':
                    
                    #creating a mask for widths that are not compatible with 0 at 3 sigma\
                    #(different from the scatter here as we don't want the 0 values)
                    bool_sign_width_split=np.array([ravel_ragged(data_perinfo[7][0][i].T[j]).astype(float)!=0\
                                                    for j in range(len(obj_disp_list))],dtype=object)
                        

                    bool_sign_split=np.array([elem & elem2 for elem,elem2 in zip(bool_sign_split, bool_sign_width_split)],dtype=object)

                    
            if split_instru:
                
                instru_list_repeat=np.array([instru_list for repeater in (i if type(i)==range else [i])])
                
                instru_unique=np.unique(ravel_ragged(instru_list))

                #here we re split bool_sign with values depending on the instrument to restrict it
                bool_sign_split=[(bool_signdet) & (ravel_ragged(instru_list_repeat)==instru_unique[i_instru])\
                                  for i_instru in range(len(instru_unique))]
            
                
        #### creating the hist variables
        if line_mode:
            
            #computing the number of element for each line
            line_indiv_size=len(ravel_ragged(data_perinfo[4][0][0]))
            
            bool_sign_line=[]
            for id_line,line in enumerate(list_id_lines):

                bool_sign_line+=[(bool_signdet) & (np.array(ravel_ragged(np.repeat(False,line_indiv_size*id_line)).tolist()+\
                                                   ravel_ragged(np.repeat(True,line_indiv_size)).tolist()+\
                                                   ravel_ragged(np.repeat(False,line_indiv_size*(len(list_id_lines)-(id_line+1)))).tolist()))]

            #here we create a 2D array with the number of lines detected per source and per line
            hist_data_splitsource=np.array([[sum((data_perinfo[4][0][i_line][i_obj]>=conf_thresh) & (~np.isnan(data_perinfo[4][0][i_line][i_obj].astype(float))))\
                                          for i_line in range_line] for i_obj in range(len(obj_disp_list))])
            
            #suming on the sources give the global per line number of detection    
            hist_data=hist_data_splitsource.sum(axis=0)
            
            if split_source:
                hist_data_split=hist_data_splitsource
                
            if split_instru:
                hist_data_split=[[np.sum((bool_sign_line[i_line]) & (bool_sign_split[i_instru]))\
                                  for i_line in range_line] for i_instru in range(len(instru_unique))]
                
        #this time data_plot is an array
        elif ratio_mode:
            
            #we need to create restricted bool sign and bool det here, with only the index of each line
            sign_ratio_arr=np.array([ravel_ragged(data_perinfo[4][0][ratio_indexes_x[i]]).astype(float) for i in [0,1]])
            
            #here we create different mask variables to keep the standard bool_det/sign for the other axis if needed
            bool_det_ratio=(sign_ratio_arr[0]!=0.) & (~np.isnan(sign_ratio_arr[0])) & (sign_ratio_arr[1]!=0.) & (~np.isnan(sign_ratio_arr[1]))
            
            #this doesn't work with bitwise comparison
            bool_sign_ratio=(sign_ratio_arr[0]>=conf_thresh) & (~np.isnan(sign_ratio_arr[0])) & (sign_ratio_arr[1]>=conf_thresh) & (~np.isnan(sign_ratio_arr[1]))

            #making it clearer
            bool_sign_ratio=bool_sign_ratio[bool_det_ratio]
                        
            #before using them to create the data ratio

            hist_data=np.array(
                  [ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]/\
                   ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                   ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]/\
               ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],dtype=object)

            if not split_off:
                
                if split_source:
                    sign_det_ratio_split=[[ravel_ragged(data_perinfo[4][0][ratio_indexes_x[i]].T[j]).astype(float) for j in range(len(obj_disp_list))] for i in range(2)]

                    #creating the bool det and bool sign masks
                    # bool_det_split=np.array([(elem_num!=0.) & (~np.isnan(elem_num)) & (elem_denom!=0.) & (~np.isnan(elem_denom))\
                    #                              for elem_num,elem_denom in zip([elem for elem in sign_det_ratio_split])])

                    bool_sign_split=np.array([(elem_num>=conf_thresh) & (~np.isnan(elem_num)) & (elem_denom>=conf_thresh) & (~np.isnan(elem_denom))\
                                                 for elem_num,elem_denom in zip(sign_det_ratio_split[0],sign_det_ratio_split[1])],dtype=object)

                    #computing the data array for the ratio (here we don't need to transpose because we select a single line with ratio_indexes_x
                    hist_data_split=np.array(
                          [ravel_ragged(data_plot[0][ratio_indexes_x[0]][i_obj])[bool_sign_split[i_obj]]/\
                           ravel_ragged(data_plot[0][ratio_indexes_x[1]][i_obj])[bool_sign_split[i_obj]]\
                               for i_obj in range(len(obj_disp_list))],dtype=object)

                elif split_instru:
                    bool_sign_split=[(bool_sign_ratio) & ((ravel_ragged(instru_list)==instru_unique[i_instru])[bool_det_ratio])\
                                      for i_instru in range(len(instru_unique))]

                    hist_data_split=np.array(
                          [ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_split[i_instru]]/\
                           ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_split[i_instru]]\
                               for i_instru in range(len(instru_unique))],dtype=object)

                    
        else:
            hist_data=[ravel_ragged(data_plot[0][i])[bool_det][bool_sign],ravel_ragged(data_plot[0][i])[bool_det][~bool_sign]]

            if not split_off:

                #### should be changed to use the same method as split_instru
                if split_source:
                    #here we need to add a transposition
                    hist_data_split=np.array([ravel_ragged(data_plot[0][i].T[i_obj])[bool_sign_split[i_obj]] for i_obj in range(len(obj_disp_list))],dtype=object)
                elif split_instru:                    
                    #no need for transposition here
                    hist_data_split=np.array([ravel_ragged(data_plot[0][i])[bool_sign_split[i_line]] for i_line in range(len(instru_unique))],dtype=object)

                #adjusting the length to avoid a size 1 array which would cause issue
                if len(hist_data_split)==1:
                    hist_data_split=hist_data_split[0]
                    
                        
        hist_cols=['blue','grey']
        hist_labels=['detections above '+str(conf_thresh*100)+'% treshold','detections below '+str(conf_thresh*100)+'% treshold']
                    
        #### histogram plotting and colors
        
        if display_nonsign:
            ax_hist.hist(hist_data,stacked=True,color=hist_cols,label=hist_labels,bins=hist_bins)
        else:
            if not split_off:
                
                #indexes of obj_list with significant detections
                id_obj_withdet=np.argwhere(np.array([sum(elem)>0 for elem in bool_sign_split])!=0).T[0]

                if glob_col_source:
                    n_obj_withdet=sum(mask_obj)
                else:
                    #number of sources with detections
                    n_obj_withdet=len(id_obj_withdet)
                                
                #color mapping accordingly for source split
                norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=n_obj_withdet-1)
                colors_func_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=cmap_color_det)
                
                if line_mode:
                    
                    if split_source:
                                
                        #creating the range of bars (we don't use histograms here because our counting is made arbitrarily on a given axis)
                        #recreate_5cmap
                        # for i_obj_det in range(n_obj_withdet-1):
                            
                        for i_obj_det in range(n_obj_withdet):
                                
                            i_obj=i_obj_det
                            
                            #only for custom modifications to recreate the 5cmap
                            # i_obj=id_obj_withdet[i_obj_det]
                            
                            #using a range in the list_id_lines allow to resize the graph when taking off lines
                            ax_hist.bar(np.array(range(len(list_id_lines)))-1/4+i_obj_det/(4/3*n_obj_withdet),hist_data_split[i_obj],
                                         width=1/(4/3*n_obj_withdet),
                             color=colors_func_obj.to_rgba(i_obj_det),label=obj_disp_list[i_obj] if i_obj in id_obj_withdet else '')
                                                                                                                                                                                                                            
                        #changing the tick and tick names to the actual line names
                        ax_hist.set_xticks(np.array(range(len(list_id_lines))))
                        ax_hist.set_xticklabels([lines_std[lines_std_names[i_line+3]] for i_line in list_id_lines],rotation=60)
                    
                    elif split_instru:
                        #creating the range of bars (we don't use histograms here because our counting is made arbitrarily on a given axis
                        #using a range in the list_id_lines allow to resize the graph when taking off lines
                        [ax_hist.bar(np.array(range(len(list_id_lines)))-1/4+i_instru/(4/3*len(instru_unique)),hist_data_split[i_instru],
                                     width=1/(4/3*len(instru_unique)),
                         color=telescope_colors[instru_unique[i_instru]],label=instru_unique[i_instru]) for i_instru in range(len(instru_unique))]
                                                                                                                                                                                                                            
                        #changing the tick and tick names to the actual line names
                        ax_hist.set_xticks(np.array(range(len(list_id_lines))))
                        ax_hist.set_xticklabels([lines_std[lines_std_names[i_line+3]] for i_line in list_id_lines],rotation=60)
                        
                else:
                    
                    if split_source:
                            
                        #skipping displaying bars for objects with no detection
                        source_withdet_mask=[sum(elem)>0 for elem in bool_sign_split]
                        
                        #resticting histogram plotting to sources with detection to havoid plotting many useless empty source hists
                        ax_hist.hist(hist_data_split[source_withdet_mask],
                                     color=np.array([colors_func_obj.to_rgba(i_obj_det) for i_obj_det in range(n_obj_withdet)])[(source_withdet_mask if glob_col_source else np.repeat(True,n_obj_withdet))],
                                     label=obj_disp_list[source_withdet_mask],bins=hist_bins,rwidth=0.8,align='left',stacked=True)

                    elif split_instru:

                        try:
                            ax_hist.hist(hist_data_split,color=[telescope_colors[instru_unique[i_instru]] for i_instru in range(len(instru_unique))],
                                      label=instru_unique,bins=hist_bins,rwidth=0.8,align='left',stacked=not split_dist_method)
                        except:
                            breakpoint()

                        #if we want combined (not stacked) + transparent                        
                        # t=[ax_hist.hist(hist_data_split[i_instru],color=telescope_colors[instru_unique[i_instru]],
                        #              label=instru_unique[i_instru],bins=hist_bins,rwidth=0.8,align='left',alpha=0.5) for i_instru in range(len(instru_unique))]
                        
                    #adding minor x ticks only (we don't want minor y ticks because they don't make sense for distributions)
                    ax_hist.xaxis.set_minor_locator(AutoMinorLocator())
                        
            else:
                ax_hist.hist(hist_data[0],color=hist_cols[0],label=hist_labels[0],bins=hist_bins)
            
        #forcing only integer ticks on the y axis
        ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        if not bigger_text or not split_off:
            plt.legend(fontsize=10)
            
        plt.tight_layout()
        if save:
            if indiv:
                suffix_str='_'+lines_std_names[3+i]
            else:
                suffix_str='_all'
                
            plt.savefig(save_dir+'/graphs/distrib/'+save_str_prefix+'autofit_distrib_'+info+suffix_str+'_cam_'+args_cam+'_'+\
                            args_line_search_e.replace(' ','_')+'_'+args_line_search_norm.replace(' ','_')+'.png')
        if close:
            plt.close(fig_hist)
        
    #returning the graph for streamlit display
    if streamlit:
        return fig_hist
    
def correl_graph(data_perinfo,infos,data_ener,dict_linevis,mode='intrinsic',mode_vals=None,conf_thresh=0.99,indiv=False,save=False,close=False,
                 streamlit=False,compute_correl=False,bigger_text=False,show_linked=True,show_ul_ew=False):
    
    '''

    should be updated to reflect 4U paper changes

    Scatter plots for specific line parameters from all the observations in the current pool (i.e. all objects/obs/lines).
    
    infos:
        Intrinsic:
            -ewidth
            -bshift
            -ener
            -ewratio+25/26/Ka/Kb : ratio between the two 25/26/Ka/Kb lines. Always uses the highest line (in energy) as numerator
            -width Note: adding a number directly after the width is necessary to plot ewratio vs width as line for which
                            the width is plotted needs to be specified
            -nH
            
        observ:
            -HR
            -flux
            -Time
            'nthcomp-gamma' (only for high-energy obs)
            'highE-flux' (15-50 keV)
            'highE-HR' (15-50keV/3-6keV)

        source:
            -inclin
    Modes:
        -Intrinsic  Use the 'info' keyword to plot infos vs one another
        -observ     Use the 'info' keyword to plot infos vs HR or flux (observ info always on y axis)
        -source     Use the 'info' keyword to plot ewidth, bshift, ener or flux vs the source inclinations
        -ewcomp     Use the 'info' keyword to specify which the lines for which the ew will be compared
        
    In HID mode, requires the mode_vals variable to be set to the flux values
    in inclin mode, idem with the inclination values
    
    Color modes (not an argument, ruled by the dicionnary variable color_scatter):
        -None
        -Instrument
        -Sources
        -Time
        -HR
        -width
        -line structures (for 4U1630-47)
        -accretion states (for 4U1630-47)
        -outbursts

    (Note : displaying non significant results overrides the coloring)
    
    Use the 'indiv' keyword to plot for all lines simultaneously or 6 plots for the 6 single lines
    the streamlit mode uses the indiv=False since the lines are already selected in the data provided to the function
    
    Non detections are filtered via 0 values in significance
    
    Detections above and below the given threshold and linked values are highlighted if the checkbox to display non significant detectoins is set, else 
    only significant detections are displayed

    It is possible to compute the spearman and pearson coefficients,
            to compute linear regressions for significantly correlated graphs, and to restrict the range of both
            computations with x and y bounds

    The person and spearman coefficients and the regressions are computed from significant detections only,
     with MC propagation of their uncertainties through a custom library adapted from
      https://github.com/privong/pymccorrelation
    As such, the distribution of the errors are assumed to be gaussian 
    (which is not the case, but using the distributions themselves would be a nightmare)


    we ravel the last 2 dimensions with a custom function since for multi object plots the sequences can be ragged and the custom .ravel function
    doesn't work

    Note : there's an explicit call to the FeKg26 being the last of the tested abs line (and assumption that Kb25 and Kb26 are just before), 
    so the uncertainty shift part will have to be changed if this is modified
    '''
    
    range_absline=dict_linevis['range_absline']
    mask_lines=dict_linevis['mask_lines']

    if streamlit:
        display_nonsign=dict_linevis['display_nonsign']
        scale_log_hr=dict_linevis['scale_log_hr']
        scale_log_ew=dict_linevis['scale_log_ew']


        compute_regr=dict_linevis['compute_regr']
        regr_pval_threshold=dict_linevis['regr_pval_threshold']
        restrict_comput_scatter=dict_linevis['restrict_comput_scatter']
        comput_scatter_lims=dict_linevis['comput_scatter_lims']

        lock_lims_det=dict_linevis['lock_lims_det']
        display_pearson=dict_linevis['display_pearson']
        display_abserr_bshift=dict_linevis['display_abserr_bshift']
        display_std_abserr_bshift=dict_linevis['display_std_abserr_bshift']
        glob_col_source=dict_linevis['glob_col_source']
        display_th_width_ew=dict_linevis['display_th_width_ew']
        cmap_color_det=dict_linevis['cmap_color_det']
        common_observ_bounds_lines=dict_linevis['common_observ_bounds_lines']
        common_observ_bounds_dates=dict_linevis['common_observ_bounds_dates']

    else:

        display_nonsign=True
        scale_log_hr=True
        scale_log_ew=True
        compute_regr=False
        lock_lims_det=False
        display_pearson=False
        display_abserr_bshift=False
        display_std_abserr_bshift=False
        glob_col_source=False
        display_th_width_ew=False
        cmap_color_det=mpl.cm.plasma
        common_observ_bounds_lines=False
        common_observ_bounds_dates=False
        
    
    mask_obj=dict_linevis['mask_obj']
    abslines_ener=dict_linevis['abslines_ener']
    abslines_plot=dict_linevis['abslines_plot']
        
    save_dir=dict_linevis['save_dir']
    save_str_prefix=dict_linevis['save_str_prefix']
    args_cam=dict_linevis['args_cam']
    args_line_search_e=dict_linevis['args_line_search_e']
    args_line_search_norm=dict_linevis['args_line_search_norm']
    
    if streamlit:
        slider_date=dict_linevis['slider_date']
        instru_list=dict_linevis['instru_list'][mask_obj]
        diago_color=dict_linevis['diago_color'][mask_obj]
        custom_states_color = dict_linevis['custom_states_color'][mask_obj]
        custom_outburst_color=dict_linevis['custom_outburst_color'][mask_obj]
        custom_outburst_number=dict_linevis['custom_outburst_number'][mask_obj]
        custom_outburst_dict=dict_linevis['custom_outburst_dict'][mask_obj]

    if 'color_scatter' in dict_linevis:    
        color_scatter=dict_linevis['color_scatter']
        obj_disp_list=dict_linevis['obj_list'][mask_obj]
        date_list=dict_linevis['date_list'][mask_obj]
        hid_plot=dict_linevis['hid_plot_restrict']
        width_plot_restrict=dict_linevis['width_plot_restrict']
        nh_plot_restrict=dict_linevis['nh_plot_restrict']
    
    else:
        color_scatter='None'
        date_list=None
        hid_plot=None
        width_plot_restrict=None
        nh_plot_restrict=None

    lum_plot_restrict=dict_linevis['lum_plot_restrict']
    observ_list=dict_linevis['observ_list'][mask_obj]

    #note that these one only get the significant BAT detections so no need to refilter
    lum_high_sign_plot_restrict=dict_linevis['lum_high_sign_plot_restrict']
    hr_high_sign_plot_restrict=dict_linevis['hr_high_sign_plot_restrict']

    gamma_nthcomp_plot_restrict=dict_linevis['gamma_nthcomp_plot_restrict']

    mask_added_regr_sign=dict_linevis['mask_added_regr_sign']

    mask_lum_high_valid=~np.isnan(ravel_ragged(lum_high_sign_plot_restrict[0]))

    #This considers all the lines
    mask_obs_sign=np.array([ravel_ragged(abslines_plot[4][0].T[mask_obj].T[i]).astype(float)>=conf_thresh\
                                      for i in range(len(mask_lines))]).any(0)
        
    #note: quantities here are already restricted
    mask_intime_norepeat=(Time(ravel_ragged(date_list))>=slider_date[0]) & (Time(ravel_ragged(date_list))<=slider_date[1])

    mask_sign_intime_norepeat=mask_obs_sign & mask_intime_norepeat

    mask_bounds=((mask_intime_norepeat) if not common_observ_bounds_dates else  np.repeat(True,len(mask_intime_norepeat))) & \
                ((mask_obs_sign) if (common_observ_bounds_lines and not (show_ul_ew and not lock_lims_det))\
                 else np.repeat(True,len(mask_intime_norepeat)))



    if sum(mask_bounds)>0:
        bounds_hr=[0.9*min(ravel_ragged(hid_plot[0][0])[mask_bounds]-ravel_ragged(hid_plot[0][1])[mask_bounds]),
                1/0.9*max(ravel_ragged(hid_plot[0][0])[mask_bounds]+ravel_ragged(hid_plot[0][2])[mask_bounds])]
        bounds_flux=[0.9*min(ravel_ragged(hid_plot[1][0])[mask_bounds]-ravel_ragged(hid_plot[1][1])[mask_bounds]),
                1/0.9*max(ravel_ragged(hid_plot[1][0])[mask_bounds]+ravel_ragged(hid_plot[1][2])[mask_bounds])]

    else:
        bounds_hr=None
        bounds_flux=None
    
    n_obj=len(observ_list)
    
    infos_split=infos.split('_')
    alpha_ul=0.3
    
    x_error=None
    y_error=None
    
    #using time changes the way things are plotted, uncertainties are computed and significance are used
    if 'time' in infos_split[0]:
        time_mode=True
        infos_split=infos_split[::-1]
    else:
        time_mode=False
    
    #using ratios changes the way things are plotted, uncertainties are computed and significance are used
    if 'ratio' in infos:
        ratio_mode=True
        
        if 'ratio' in infos_split[0]:
            #fetching the line indexes corresponding to the names
            ratio_indexes_x=ratio_choices[infos_split[0][-2:]]
        
            #keeping the same architecture in time mode even if the ratio is plotted on the y axis for simplicity
        elif 'ratio' in infos_split[1]:
            #fetching the line indexes corresponding to the names
            ratio_indexes_x=ratio_choices[infos_split[1][-2:]]
    else:
        ratio_mode=False

    if mode=='ewcomp':
        line_comp_mode=True
    else:
        line_comp_mode=False
        
    #not showing upper limit for the width plot as there's no point currently
    if 'width' in infos:
        width_mode=True
        show_ul_ew=False
        if ratio_mode:
            #if comparing width with the ewratio, the line number for the width has to be specified for the width and is 
            #retrieved (and -1 to compare to an index)
            width_line_id=int(infos[infos.find('width')+5])-1

    else:
        width_mode=False
        
    #failsafe to prevent wrong colorings for intrinsic plots
    if (ratio_mode or line_comp_mode) and color_scatter=='width':
        color_scatter='None'
        
    data_list=[data_perinfo[0],data_perinfo[1],data_ener,data_perinfo[3],date_list,width_plot_restrict]

    high_E_mode=False
    #infos and data definition depending on the mode
    if mode=='intrinsic':

        ind_infos=[np.argwhere([elem in infos_split[i] for elem in ['ew','bshift','ener','lineflux','time','width']])[0][0] for i in [0,1]]

        data_plot=[data_list[ind] for ind in ind_infos]
        
    elif mode=='observ':
        #the 'None' is here to have the correct index for the width element
        ind_infos=[np.argwhere([elem in infos_split[0] for elem in ['ew','bshift','ener','lineflux','None','width']])[0][0],
                    np.argwhere(np.array(['HR','flux','time','nthcomp-gamma','highE-flux','highE-HR'])==infos_split[1])[0][0]]

        if ind_infos[1] in [3,4,5]:
            high_E_mode=True

        second_arr_infos=[date_list,gamma_nthcomp_plot_restrict,lum_high_sign_plot_restrict,hr_high_sign_plot_restrict]
        data_plot=[data_list[ind_infos[0]], second_arr_infos[ind_infos[1]-2] if ind_infos[1]>=2\
                                            else mode_vals[ind_infos[1]]]
        
        if time_mode:
            data_plot=data_plot[::-1]
        
    elif mode=='source':
        ind_infos=[np.argwhere([elem in infos_split[0] for elem in ['ew','bshift','ener','lineflux','None','width']])[0][0],-1]
        data_plot=[data_list[ind_infos[0]], mode_vals]
        
    elif line_comp_mode:
        ind_infos=[np.argwhere([elem in infos_split[i] for elem in lines_std_names[3:]])[0][0] for i in [0,1]]
        data_plot=[data_perinfo[0],data_perinfo[0]]
    
    if indiv:
        graph_range=range_absline
    else:
        #using a list index for the global graphs allows us to keep the same structure
        #however we need to restrict it to the currently plotted lines in streamlit mode
        if streamlit:
            graph_range=[range(len([elem for elem in mask_lines if elem]))]
        else:
            graph_range=[range_absline]

    for i in graph_range:

        figsize_val=(8.+(-0.5 if mode!='observation' else 0),5.5) if color_scatter in ['Time','HR','width','nH','L_3-10'] else (6,6)

        # if color_scatter in ['Time','HR','width','nH']:
        #
        #     fig_scat = plt.figure(figsize=figsize_val if bigger_text else (11, 10.5))
        #     gs = GridSpec(1, 2, width_ratios=[4, 0.2])
        #     ax_scat=plt.subplot(gs[0])
        #     ax_cb=plt.subplot(gs[1])
        # else:
        fig_scat,ax_scat=plt.subplots(1,1,figsize=figsize_val if bigger_text else (11,10.5))

        legend_title=''

        if 'width' in infos_split[1]:
            ax_scat.set_ylim(0,5500)
        elif 'width' in infos_split[0]:
            ax_scat.set_xlim(0,5500)

        if not bigger_text:
            if indiv:
                fig_scat.suptitle(info_str[ind_infos[0]]+' - '+(info_str[ind_infos[1]] if mode=='intrinsic' else\
                                  info_hid_str[ind_infos[1]])+' scatter for the '+lines_std[lines_std_names[3+i]]+
                                  ' absorption line')
            else:
                fig_scat.suptitle(((infos_split[0]+' - '+infos_split[1]+' equivalent widths') if line_comp_mode else\
                    ((info_str[ind_infos[0]] if not ratio_mode else infos_split[0])+' - '+(info_str[ind_infos[1]] if mode=='intrinsic' else\
                              (info_hid_str[ind_infos[1]]) if mode=='observ' else 'inclination')))\
                              +(' for currently selected '+('lines and ' if not ratio_mode else '')+'sources' if streamlit else ' for all absorption lines'))
            
        if not line_comp_mode:

            if sum(mask_lines)==1:
                line_str=lines_std[np.array(lines_std_names)[3:9][mask_lines][0]]
            else:
                line_str=''
                
            ax_scat.set_xlabel('Time' if time_mode else (ratio_choices_str[infos_split[0][-2:]]+' ' if ratio_mode else '')+
            (axis_str[ind_infos[0]].replace(' (eV)',('' if ratio_mode else ' (eV)'))+' '+(infos_split[0][-2:]+' ratio'))\
                if ratio_mode else line_str+' '+axis_str[ind_infos[0]])

        else:
            line_str=''
            ax_scat.set_xlabel(infos_split[0]+' '+axis_str[ind_infos[0]])
            ax_scat.set_ylabel(infos_split[1]+' '+axis_str[ind_infos[0]])
            
        #putting a logarithmic y scale if it shows equivalent widths or one of the hid parameters
        #note we also don't put the y log scale for a gamma y axis (of course)
        if mode!='source' and ((mode=='observ' and scale_log_hr and not time_mode and not ind_infos[1]==[3])\
                               or ((ind_infos[0 if time_mode else 1] in [0,3] or line_comp_mode) and scale_log_ew)):
                            
            ax_scat.set_yscale('log')

        #putting a logarithmic x scale if it shows equivalent widths
        if ind_infos[0] in [0,3] and scale_log_ew and not time_mode:
            ax_scat.set_xscale('log')

        
        if mode=='intrinsic':
            ax_scat.set_ylabel(line_str+' '+axis_str[ind_infos[1]])
        elif mode=='observ':
            ax_scat.set_ylabel(axis_str[ind_infos[0]] if (time_mode and not ratio_mode) else\
                               axis_str[ind_infos[0]]+' '+(infos_split[0][-2:]+' ratio') if time_mode else\
                               axis_hid_str[ind_infos[1]])
        elif mode=='source':
            ax_scat.set_ylabel('Source inclination (°)')
            ax_scat.set_ylim((0,90))

        #creating a time variable for time mode to be used later

        if time_mode:

            #creating an appropriate date axis
            #manually readjusting for small durations because the AutoDateLocator doesn't work well
            time_range=min(mdates.date2num(slider_date[1]),max(mdates.date2num(ravel_ragged(date_list))))-\
                       max(mdates.date2num(slider_date[0]),min(mdates.date2num(ravel_ragged(date_list))))

            #
            #
            # if time_range<150:
            #     date_format=mdates.DateFormatter('%Y-%m-%d')
            # elif time_range<1825:
            #     date_format=mdates.DateFormatter('%Y-%m')
            # else:
            #     date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
            # ax_scat.xaxis.set_major_formatter(date_format)
            #
            # plt.xticks(rotation=70)


        date_list_repeat=np.array([date_list for repeater in (i if type(i)==range else [i])]) if not ratio_mode else date_list

        mask_lum_high_valid_repeat = ravel_ragged([mask_lum_high_valid for repeater in (i if type(i)==range else [i])])\
            if not ratio_mode else mask_lum_high_valid

        if streamlit:
            mask_intime=(Time(ravel_ragged(date_list_repeat))>=slider_date[0]) & (Time(ravel_ragged(date_list_repeat))<=slider_date[1])

        else:
            mask_intime=True
        
            mask_intime_norepeat=True
            
        #the boolean masks for detections and significance are more complex when using ratios instead of the standard data since 
        #two lines need to be checked
                    
        if line_comp_mode or ratio_mode:
                                
            #we can use the data constructs used for the ew ratio mode to create the ratios in ratio_mode
            #we just need to get common indexing variable
            ind_ratio=ind_infos if line_comp_mode else ratio_indexes_x
                
            #in ew ratio mode we need to make sure than both lines are defined for each point so we must combine the mask of both lines

            bool_sign_x=ravel_ragged(data_perinfo[4][0][ind_ratio[0]]).astype(float)
            bool_sign_y=ravel_ragged(data_perinfo[4][0][ind_ratio[1]]).astype(float)

            #adding a width mask to ensure we don't show elements with no width
            if width_mode:
                bool_sign_ratio_width=ravel_ragged(width_plot_restrict[2][width_line_id]).astype(float)!=0 
            else:
                bool_sign_ratio_width=True
                
            bool_det_ratio=(bool_sign_x!=0.) & (~np.isnan(bool_sign_x)) & (bool_sign_y!=0.) & (~np.isnan(bool_sign_y)) &\
                           mask_intime_norepeat & (True if not high_E_mode else mask_lum_high_valid)
            
            #for whatever reason we can't use the bitwise comparison so we compute the minimum significance of the two lines before testing for a single arr
            bool_sign_ratio=np.array([bool_sign_x[bool_det_ratio],bool_sign_y[bool_det_ratio]]).T.min(1)>=conf_thresh
            
            #applying the width mask (or just a True out of width mode)
            bool_sign_ratio=bool_sign_ratio & (True if bool_sign_ratio_width is True else bool_sign_ratio_width[bool_det_ratio])

                
            #masks for upper limits (needs at least one axis to have detection and significance)
            bool_nondetsign_x=np.array(((bool_sign_x<conf_thresh).tolist() or (np.isnan(bool_sign_x).tolist()))) &\
                        (np.array((bool_sign_y>=conf_thresh).tolist()) & np.array((~np.isnan(bool_sign_y)).tolist())) \
                              & mask_intime_norepeat & (True if not high_E_mode else mask_lum_high_valid)
            bool_nondetsign_y=np.array(((bool_sign_y<conf_thresh).tolist() or (np.isnan(bool_sign_y).tolist()))) &\
                        (np.array((bool_sign_x>=conf_thresh).tolist()) & np.array((~np.isnan(bool_sign_x)).tolist())) \
                              & mask_intime_norepeat & (True if not high_E_mode else mask_lum_high_valid)

            #the boool sign and det are only used for the ratio in ratio_mode, but are global in ewratio mode
            if line_comp_mode:
                
                #converting the standard variables to the ratio ones
                bool_det=bool_det_ratio
                bool_sign=bool_sign_ratio
                
                #note: we don't care about the 'i' index of the graph range here since this graph is never made in indiv mode/not all lines mode
                
                #here we can build both axis variables and error variables identically
                x_data,y_data=[np.array([ravel_ragged(data_plot[0][0][ind_infos[i]])[bool_det][bool_sign],
                                         ravel_ragged(data_plot[0][0][ind_infos[i]])[bool_det][~bool_sign]],dtype=object) for i in [0,1]]

                mask_added_regr_sign_use=None if mask_added_regr_sign is None else \
                                        mask_added_regr_sign[bool_det][bool_sign]


                #same thing for the uncertainties                
                x_error,y_error=[np.array([[ravel_ragged(data_plot[i][l][ind_infos[i]])[bool_det][bool_sign] for l in [1,2]],                                         
                                           [ravel_ragged(data_plot[i][l][ind_infos[i]])[bool_det][~bool_sign] for l in [1,2]]],
                                          dtype=object) for i in [0,1]]
                
            #here we assume this is ratio_mode
            else:  
            #this time data_plot is an array
            #here, in the ratio X is the numerator so the values are obtained by dividing X/Y
            
                #here we can directly create the data ratio
                if time_mode:

                    x_data=np.array([mdates.date2num(ravel_ragged(date_list_repeat))[bool_det_ratio][bool_sign_ratio],mdates.date2num(ravel_ragged(date_list_repeat))[bool_det_ratio][~bool_sign_ratio]],
                                    dtype=object)

                        
                    #in this case the ew ratio is on the Y axis
                    y_data=np.array(
                          [ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]/\
                           ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                           ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]/\
                           ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],dtype=object)            
                else:
                    x_data=np.array(
                          [ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]/\
                           ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                           ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]/\
                           ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],dtype=object) 
        else:
            
            #these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others            
            bool_sign=ravel_ragged(data_perinfo[4][0][i]).astype(float)
            
            #standard detection mask
            bool_det=(bool_sign!=0.) & (~np.isnan(bool_sign)) & (mask_intime) &\
                     (True if not high_E_mode else mask_lum_high_valid_repeat)

            bool_detsign_bounds=(bool_sign!=0.) & (~np.isnan(bool_sign)) & \
                            (mask_intime if not common_observ_bounds_dates else True) &\
                            (True if not high_E_mode else mask_lum_high_valid_repeat) & (bool_sign>=conf_thresh)

            
            #mask used for upper limits only
            bool_nondetsign=((bool_sign<conf_thresh) | (np.isnan(bool_sign))) & (mask_intime) & \
                            (True if not high_E_mode else mask_lum_high_valid_repeat)
            
            #restricted significant mask, to be used in conjunction with bool_det
            bool_sign=bool_sign[bool_det]>=conf_thresh
            
            if width_mode:
                
                #creating a mask for widths that are not compatible with 0 at 3 sigma (for which this width value is pegged to 0)
                bool_sign_width=ravel_ragged(width_plot_restrict[2][i]).astype(float)!=0
                
                bool_sign=bool_sign & bool_sign_width[bool_det]
                
            if time_mode:

                x_data=np.array([mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][bool_sign],
                                 mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][~bool_sign]],
                                dtype=object)

                y_data=np.array([ravel_ragged(data_plot[1][0][i])[bool_det][bool_sign],
                                 ravel_ragged(data_plot[1][0][i])[bool_det][~bool_sign]],
                        dtype=object)

                if len(x_data[0])>0:
                    x_data_bounds=mdates.date2num(ravel_ragged(date_list_repeat))[bool_detsign_bounds]
                else:
                    x_data_bounds=None

            else:
                x_data=np.array([ravel_ragged(data_plot[0][0][i])[bool_det][bool_sign],
                                 ravel_ragged(data_plot[0][0][i])[bool_det][~bool_sign]],
                    dtype=object)

                if len(x_data[0])>0:
                    x_data_bounds=ravel_ragged(data_plot[0][0][i])[bool_detsign_bounds]
                else:
                    x_data_bounds=None

        #applying the same thing to the y axis if ratios are also plotted there
        if type(ind_infos[1]) not in [int,np.int64] and 'ratio' in ind_infos[1]:
                #note:not needed for now
                pass
            # #we fetch the index list corresponding to the info string at the end of the info provided
            # ratio_indexes_y=ratio_choices[ind_infos[1][-2:]]
            
            # #before using them to create the data ratio (here we don't bother using the non significant detections)
            # #we don't bother masking the infinites here because they are masked natively when plotting
            # y_data=np.array(
            #       [ravel_ragged(data_plot[0][0][ratio_indexes_y[0]])[bool_det[ratio_indexes_y[0]]][bool_sign[ratio_indexes_y[0]]]/\
            #        ravel_ragged(data_plot[0][0][ratio_indexes_y[1]])[bool_det[ratio_indexes_y[1]]][bool_sign[ratio_indexes_y[1]]],
            #        ravel_ragged(data_plot[0][0][ratio_indexes_y[0]])[bool_det[ratio_indexes_y[0]]][~bool_sign[ratio_indexes_y[0]]]/\
            #        ravel_ragged(data_plot[0][0][ratio_indexes_y[1]])[bool_det[ratio_indexes_y[1]]][~bool_sign[ratio_indexes_y[1]]]],dtype=object)
        else:
              
            if mode=='intrinsic':
                
                if width_mode and ratio_mode:
                    #the width needs to be changed to a single line here
                        y_data=np.array([ravel_ragged(data_plot[1][0][width_line_id])[bool_det_ratio][bool_sign_ratio],
                                         ravel_ragged(data_plot[1][0][width_line_id])[bool_det_ratio][~bool_sign_ratio]],
                                    dtype=object)
                    
                else:
                    y_data=np.array([ravel_ragged(data_plot[1][0][i])[bool_det][bool_sign],
                                     ravel_ragged(data_plot[1][0][i])[bool_det][~bool_sign]],
                                dtype=object)

            elif mode=='observ' and not time_mode:
                
                #since the hid data is the same no matter the line, we need to repeat it for the number of lines used
                # when plotting the global graph
                #index 1 since data_plot is [x_axis,y_axis], then 0 to get the main value
                y_data_repeat=np.array([data_plot[1][0] for repeater in (i if type(i)==range else [i])])
                                        
                #only then can the linked mask be applied correctly (doesn't change anything in individual mode)
                if ratio_mode:
                    #here we just select one of all the lines in the repeat and apply the ratio_mode mask onto it
                    y_data=np.array([ravel_ragged(y_data_repeat[ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio],
                                     ravel_ragged(y_data_repeat[ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]],
                                    dtype=object)

                    mask_added_regr_sign_use = None if mask_added_regr_sign is None else \
                        mask_added_regr_sign[bool_det_ratio][bool_sign_ratio]
                else:
                    #this is not implemented currently

                    y_data=np.array([ravel_ragged(y_data_repeat)[bool_det][bool_sign],
                                 ravel_ragged(y_data_repeat)[bool_det][~bool_sign]],
                                dtype=object)

                    mask_added_regr_sign_use = None if mask_added_regr_sign is None else \
                        ravel_ragged([mask_added_regr_sign \
                                        for repeater in (i if type(i)==range else [i])])[bool_det][bool_sign]


            elif mode=='source':
                y_data_repeat=np.array([data_plot[1][i_obj][0] for repeater in (i if type(i)==range else [i])\
                                        for i_obj in range(n_obj) for i_obs in range(len(data_plot[0][0][repeater][i_obj]))\
                                    ]).ravel()
                    
    
                y_data=np.array([y_data_repeat[bool_det][bool_sign],y_data_repeat[bool_det][~bool_sign]],dtype=object)
                        
        #### upper limit computation
        
        if show_ul_ew:
            
            if line_comp_mode or ratio_mode:

                #we use the same double definition here
                #in ratio_mode, x is the numerator so the ul_x case amounts to an upper limit
                y_data_ul_x=np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[0]])[bool_nondetsign_x],
                                dtype=object)                
                x_data_ul_x=np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[1]])[bool_nondetsign_x],
                                dtype=object)
                
                #here in ratio mode ul_y amounts to a lower limit
                y_data_ul_y=np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[0]])[bool_nondetsign_y],
                                dtype=object)
                x_data_ul_y=np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[1]])[bool_nondetsign_y],
                                dtype=object)
                
                #same way of defining the errors
                y_error_ul_x=np.array([ravel_ragged(data_perinfo[0][l][ind_ratio[1]])[bool_nondetsign_x] for l in [1,2]],
                                      dtype=object)
                x_error_ul_y=np.array([ravel_ragged(data_perinfo[0][l][ind_ratio[0]])[bool_nondetsign_y] for l in [1,2]],
                                      dtype=object)
                
                #computing two (upper and lower) limits for the ratios
                
                #the upper limit corresponds to points for which the numerator line is not detected, so it's the ul_y 
                #similarly, the lower limit is for ul_x
 
                #this is only in ratio_mode
                if ratio_mode:
                    if time_mode:
                        #switching x/y and ll/ul in time mode since the ratio is now on the y axis
                        
                        y_data_ll=x_data_ul_y/y_data_ul_y
                        y_data_ul=x_data_ul_x/y_data_ul_x
                        
                        x_data_ll=np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign_y],dtype=object)
                        x_data_ul=np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign_x],dtype=object)
                    else:
                        x_data_ll=y_data_ul_y/x_data_ul_y
                        y_data_ll=np.array(ravel_ragged(y_data_repeat[ind_ratio[1]])[bool_nondetsign_y],dtype=object)
                        
                        x_data_ul=y_data_ul_x/x_data_ul_x
                        y_data_ul=np.array(ravel_ragged(y_data_repeat[ind_ratio[0]])[bool_nondetsign_x],dtype=object)

                        pass
                        
            else:
                
                if time_mode:

                        y_data_ul=np.array(ravel_ragged(data_perinfo[5][0][i])[bool_nondetsign],
                                    dtype=object)
                        x_data_ul=np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign],dtype=object)

                else:
                    
                    x_data_ul=np.array(ravel_ragged(data_perinfo[5][0][i])[bool_nondetsign],
                                dtype=object)
                    

                    #we can directly create the y_data_ul from y_data_repeat no matter which one it comes from
                    y_data_ul=np.array(ravel_ragged(y_data_repeat)[bool_nondetsign],dtype=object)

                        
        #in the errors we want to distinguish the parameters which were linked so we start compute the masks of the 'linked' values
        
        #we don't really care about the linked state in ewratio mode
        if line_comp_mode:
            linked_mask=np.array([np.repeat(False,len(x_data[0])),np.repeat(False,len(x_data[1]))],dtype=object)
        else:
            if ratio_mode:
                
                linked_mask=np.array([np.array([type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[0]])\
                                    [bool_det_ratio][bool_sign_ratio]] and\
                          [type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[1]])\
                                    [bool_det_ratio][bool_sign_ratio]]).astype(object),
                      np.array([type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[0]])\
                                [bool_det_ratio][~bool_sign_ratio]] and\
                          [type(elem)==str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[1]])\
                                    [bool_det_ratio][~bool_sign_ratio]]).astype(object)],
                     dtype=object)
            else:         
                linked_mask=np.array(\
                                 [np.array([type(elem)==str for elem in \
                                            ravel_ragged(data_perinfo[1][1][i])[bool_det][bool_sign].astype(object)]),
                                  np.array([type(elem)==str for elem in \
                                            ravel_ragged(data_perinfo[1][1][i])[bool_det][~bool_sign].astype(object)])],
                                 dtype=object)
            
        #reconverting the array to bools and replacing it by true if it is empty to avoid conflicts when there is not data to plot
        #note : this doesn't work for arrays of length 1 so we add a reconversion when we use it just in case
        
        #resticting to cases where there's more than one resulting element
        if type(linked_mask[0])!=bool:
            linked_mask[0]=linked_mask[0].astype(bool)
        else:
            linked_mask[0]=np.array([linked_mask[0]])
        if type(linked_mask[1])!=bool:
            linked_mask[1]=linked_mask[1].astype(bool)
        else:
            linked_mask[1]=np.array([linked_mask[1]])
            

        #defining the jumps in indexes for linked values in the all line arrays, which is the number of lines shifted times the number of obs in 
        #all current objects
        n_obs=len(ravel_ragged(data_perinfo[0][0][0]))
        jump_Kb=3*n_obs
        jump_Kg=4*n_obs

        #unused for now        
        # #defining the number of lines to skip when not all lines are used
        # n_jump_Kg=len([elem for elem in mask_lines[:-1] if elem is True])
        # n_jump_Kb26=len([elem for elem in mask_lines[:-2] if elem is True])
        # n_jump_Kb25=len([elem for elem in mask_lines[:-3] if elem is True])
        
        #custom function so that the array indexation is not many lines long 
        def linked_uncer(ind,ind_incer,ind_info):
            
            '''
            For all lines at once it is easier to use absolute index positions after developping the array
            We do the index changes before bool_det and bool_sign else we could lose the regularity in the shifts
            linked_ind is only applied in abslines_plot (and not the restricted array) because else the information of the line the current line
            is linked to can be missing if the other line is masked
            '''
            
            linked_line_ind_restrict=ind//n_obs
            
            #here we retrieve the index of the original line and translate it into a position for the non masked array            
            linked_line_ind=np.argwhere(mask_lines)[linked_line_ind_restrict][0]

            linked_line_pos=ind%n_obs+n_obs*linked_line_ind
            
            '''
            Since the line shifts can be either positive or negative depending on if the first complexes are linked to the others or vice-versa, 
            we directly fetch all of the uncertainty values for each linked complex and pick the associated value
            We also make sure each line is linked properly by restricting the lines we check to the ones with the same base values
            '''
            
            ###TODO: update this for future linedets
            # #first we isolate the values for the same exposure
            # curr_exp_values=ravel_ragged(abslines_plot[ind_info][0].transpose(1,0,2)[mask_obj].transpose(1,0,2))[(ind//6)*6:(ind//6+1)*6]
            
            #then we identify 
            
            if linked_line_ind>2:
                val_line_pos=linked_line_pos-jump_Kg if linked_line_ind>4 else linked_line_pos-jump_Kb
            else:
                #should not happen
                breakpoint()
                
            if ind_info==2:
                #testing if there are 5 dimensions which would mean that we need to transpose with specific axes due to the array being regular
                if np.ndim(abslines_plot)==5:
                     return ravel_ragged(abslines_ener[ind_incer].transpose(1,0,2)[mask_obj].transpose(1,0,2))[val_line_pos]
                else:
                    return ravel_ragged(abslines_ener[ind_incer].T[mask_obj].T)[val_line_pos]
            else:
                if np.ndim(abslines_plot)==5:
                    return ravel_ragged(abslines_plot[ind_info][ind_incer].transpose(1,0,2)[mask_obj].transpose(1,0,2))[val_line_pos]
                else:
                    return ravel_ragged(abslines_plot[ind_info][ind_incer].T[mask_obj].T)[val_line_pos]
            
        #### error computation
        #(already done earlier in specific cases)
        
        #in ratio mode the errors are always for the ew and thus are simply computed from composing the uncertainties of each ew
        #then coming back to the ratio
        if ratio_mode:
            if time_mode:
                
                '''
                reminder of the logic since this is quite confusing
                the uncertainty is computed through the least squares because it's the correct formula for these kind of computation
                (correctly considers the skewness of the bivariate distribution)
                here, we can use ratio_indiexes_x[0] and ratio_indexes_x[1] independantly as long was we divide the uncertainty 
                ([i] which will either be [1] or [2]) by the main value [0]. However, as the uncertainties are asymetric,
                 each one must be paired with the opposite one to maintain consistency 
                (aka the + uncertaintiy of the numerator and the - incertainty of the denominator, when composed,
                 give the + uncertainty of the quotient)
                on top of that, the adressing is important as the minus uncertainty of the quotient needs to be first, 
                so the -num/+det must be first
                so for the first index in [1,2] which is i=1, we must have i for the num and 2 for the det
                the num is the one with ratio_indexes[0]
                '''

                y_error=np.array([[((ravel_ragged(data_plot[1][i][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask]/\
                                     ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask])**2+
                                   (ravel_ragged(data_plot[1][2 if i==1 else 1][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask]/\
                                    ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask])**2)**(1/2)*
                                   y_data[i_sign] for i in [1,2]] for i_sign,elem_sign_mask in enumerate([bool_sign_ratio,~bool_sign_ratio])],dtype=object)
                
            else:
                x_error=np.array([[((ravel_ragged(data_plot[0][i][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask]/\
                                     ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask])**2+
                                   (ravel_ragged(data_plot[0][2 if i==1 else 1][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask]/\
                                    ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask])**2)**(1/2)*
                                   x_data[i_sign] for i in [1,2]] for i_sign,elem_sign_mask in enumerate([bool_sign_ratio,~bool_sign_ratio])],dtype=object)
                    
                if width_mode:
                    y_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[0])\
                                       for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][width_line_id]))] for j in [1,2]])
                
                    
        #defining the errors and shifting the linked values accordingly for blueshifts
        #here we do not need the linked_ind shenanigans because this is not called during streamlit (and so all the lines are there)
        
        #THIS ALSO NEEDS TO CHANGE
        if indiv:
            x_error=np.array([[elem if type(elem)!=str else (ravel_ragged(data_plot[0][j][i-3])[ind_val] if i<5 else\
                      ravel_ragged(data_plot[0][1][i-4])[ind_val]) for ind_val,elem in enumerate(ravel_ragged(data_plot[0][j][i]))] for j in [1,2]])
                
            if mode=='intrinsic':
                y_error=np.array([[elem if type(elem)!=str else (ravel_ragged(data_plot[1][j][i-3])[ind_val] if i<5 else\
                      ravel_ragged(data_plot[0][1][i-4])[ind_val]) for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1,2]])
            
        elif not ratio_mode and not line_comp_mode:
            if time_mode:
                #swapping the error to the y axis
                y_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[1])\
                                                                for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1,2]])
            else:
                x_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[0])\
                                                            for ind_val,elem in enumerate(ravel_ragged(data_plot[0][j][i]))] for j in [1,2]])

            if mode=='intrinsic':
                y_error=np.array([[elem if type(elem)!=str else linked_uncer(ind_val,j,ind_infos[0])\
                                   for ind_val,elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1,2]])
                
        '''
        in observ and source mode, we can compute the errors in the same formula in indiv mode or not since the values are the same for each line
        '''
        #note : all the cases of use of time mode are already created before
        if mode=='observ' and not time_mode:

            #l is error index here
            y_err_repeat=np.array([[data_plot[1][l] for repeater in (i if type(i)==range else [i])] for l in [1,2]])
                                    
            #only then can the linked mask be applied correctly (doesn't change anything in individual mode)
            if ratio_mode:

                #here we just select one of all the lines in the repeat and apply the ratio_mode mask onto it
                y_error=np.array([[ravel_ragged(y_err_repeat[l][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio] for l in [0,1]],
                                [ravel_ragged(y_err_repeat[l][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio] for l in [0,1]]],dtype=object)
            else:
                y_error=np.array([[ravel_ragged(y_err_repeat[l])[bool_det][bool_sign] for l in [0,1]],
                                [ravel_ragged(y_err_repeat[l])[bool_det][~bool_sign] for l in [0,1]]],dtype=object)
            
        if mode=='source':
            y_err_repeat=np.array([[data_plot[1][i_obj][l] for j in (i if type(i)==range else [i])\
                                    for i_obj in range(n_obj) for i_obs in range(len(data_plot[0][0][j][i_obj]))]\
                                    for l in [1,2]])
            

            y_error=np.array([[ravel_ragged(y_err_repeat[l])[bool_det][bool_sign] for l in [0,1]],
                              [ravel_ragged(y_err_repeat[l])[bool_det][~bool_sign] for l in [0,1]]],
                            dtype=object)

        #maybe missing something for time mode here
        ###TODO
        if show_ul_ew and not line_comp_mode and not time_mode:
            if ratio_mode:
                #creation of y errors from both masks for upper and lower limits
                y_error_ul=np.array([ravel_ragged(y_err_repeat[l][ind_ratio[0]])[bool_nondetsign_x] for l in [0,1]],dtype=object)
                y_error_ll=np.array([ravel_ragged(y_err_repeat[l][ind_ratio[1]])[bool_nondetsign_y] for l in [0,1]],dtype=object)
            else:
                #creating y errors from the upper limit mask
                y_error_ul=np.array([ravel_ragged(y_err_repeat[l])[bool_nondetsign] for l in [0,1]],dtype=object)
                
        #applying the det and sign masks
        if not line_comp_mode and not ratio_mode:

            if time_mode:
                y_error=[np.array([y_error[0][bool_det][bool_sign],y_error[1][bool_det][bool_sign]]),
                          np.array([y_error[0][bool_det][~bool_sign],y_error[1][bool_det][~bool_sign]])]
            else:

                if len(x_data[0])>0:
                    x_err_bounds=[x_error[0][bool_detsign_bounds],x_error[1][bool_detsign_bounds]]
                else:
                    x_err_bounds=None

                x_error=[np.array([x_error[0][bool_det][bool_sign],x_error[1][bool_det][bool_sign]]),
                      np.array([x_error[0][bool_det][~bool_sign],x_error[1][bool_det][~bool_sign]])]
            
        
        if mode=='intrinsic':
            if width_mode and ratio_mode:
                y_error=[np.array([y_error[0][bool_det_ratio][bool_sign_ratio],y_error[1][bool_det_ratio][bool_sign_ratio]]).astype(float),
                          np.array([y_error[0][bool_det_ratio][~bool_sign_ratio],y_error[1][bool_det_ratio][~bool_sign_ratio]]).astype(float)]
            else:
                y_error=[np.array([y_error[0][bool_det][bool_sign],y_error[1][bool_det][bool_sign]]).astype(float),
                          np.array([y_error[0][bool_det][~bool_sign],y_error[1][bool_det][~bool_sign]]).astype(float)]

        id_point_refit=999999999999999
        # # changing the second brightest substructure point for the EW ratio plot to the manually fitted one
        # # with relaxed SAA filtering
        # #now modified directly in the files
        # if color_scatter == 'custom_line_struct' and 'ewratio' in infos_split[0]:
        #     x_point = 1.6302488583411436
        #     id_point_refit=np.argwhere(x_data[0]==x_point)[0][0]
        #
        #     #computed manually
        #     x_data[0][id_point_refit]=2.41
        #     x_error[0][0][id_point_refit]=1.09
        #     x_error[0][1][id_point_refit]=2.59

        #no need to do anything for y error in hid mode since it's already set to None
        
        data_cols=['blue','grey']
        
        data_link_cols=['dodgerblue','silver'] if show_linked else data_cols
        
        # data_labels=np.array(['detections above '+str(conf_thresh*100)+'% treshold','detections below '+str(conf_thresh*100)+'% treshold'])
        data_labels=['','']
        #### plots
                
        #plotting the unlinked and linked results with a different set of colors to highlight the differences
        #note : we don't use markers here since we cannot map their color

        if width_mode:
            #note: we only do this for significant unlinked detections
            uplims_mask=y_data[0].astype(float)[~(linked_mask[0].astype(bool))]==0
        else:
            uplims_mask=None

        errbar_list=['' if len(x_data[s])==0 else ax_scat.errorbar(x_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                                      y_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                          xerr=None if x_error is None else\
                              np.array([elem[~(linked_mask[s].astype(bool))] for elem in x_error[s]]).clip(0),
                          yerr=np.array([elem[~(linked_mask[s].astype(bool))] for elem in y_error[s]]).clip(0),
                                           linewidth=1,
                          c=data_cols[s],label=data_labels[s] if color_scatter=='None' else '',linestyle='',
                          uplims=uplims_mask,
                          marker='D' if color_scatter=='None' else None,
                          alpha=None if color_scatter=='custom_outburst' else 1,zorder=1)\
         for s in ([0,1] if display_nonsign else [0])]

           
        #note:deprecated
        errbar_list_linked=['' if len(x_data[s])==0 else ax_scat.errorbar(x_data[s].astype(float)[linked_mask[s].astype(bool)],
                                                        y_data[s].astype(float)[linked_mask[s].astype(bool)],
                          xerr=None if x_error is None else\
                              np.array([elem[linked_mask[s].astype(bool)] for elem in x_error[s]]).clip(0),
                            yerr=np.array([elem[linked_mask[s].astype(bool)] for elem in y_error[s]]).clip(0),
                                       linewidth=1,
                          color=data_link_cols[s],label='linked '+data_labels[s]  if color_scatter=='None' else '',linestyle='',
                          marker='D' if color_scatter=='None' else None,
                          uplims=False,
                          alpha=None if color_scatter=='custom_outburst' else 0.6 if show_linked else 1.0,zorder=1000)\
         for s in ([0,1] if display_nonsign else [0])]        
            
                
        #locking the graph if asked to to avoid things going haywire with upper limits
        if lock_lims_det or not show_ul_ew:
            ax_scat.set_xlim(ax_scat.get_xlim())
            ax_scat.set_ylim(ax_scat.get_ylim())
        
        #### adding the absolute blueshift uncertainties for Chandra in the blueshift graphs
        
        #adding a line at 0 blueshift
        if infos_split[0]=='bshift':
            if time_mode:
                ax_scat.axhline(y=0, xmin=0, xmax=1, color='grey', linestyle=':', lw=1.)
            else:
                ax_scat.axvline(x=0,ymin=0,ymax=1,color='grey',linestyle=':',lw=1.)
        
        #(-speed_abs_err[0],speed_abs_err[1],color='grey',label='Absolute error region',alpha=0.3)
        
        if display_std_abserr_bshift and infos_split[0]=='bshift':
            
            #plotting the distribution mean and std (without the GRS exposures)
            v_mean=-200
            v_sigma=360

            if time_mode:
                #reverses the axese so here we need to do it vertically (only case)
                # mean
                ax_scat.axhline(y=v_mean, xmin=0, xmax=1, color='brown', linestyle='-', label=r'$\overline{v}$',
                                lw=0.75)

                # span for the std
                ax_scat.axhspan(v_mean - v_sigma, v_mean + v_sigma, color='brown', label=r'$\sigma_v$', alpha=0.1)

            else:
            
                #mean
                ax_scat.axvline(x=v_mean,ymin=0,ymax=1,color='brown',linestyle='-',label=r'$\overline{v}$',lw=0.75)

                #span for the std
                ax_scat.axvspan(v_mean-v_sigma,v_mean+v_sigma,color='brown',label=r'$\sigma_v$',alpha=0.1)

        if display_abserr_bshift and infos_split[0]=='bshift':

            #computing the distribution mean and std
            if time_mode:
                bshift_data=y_data[0]
                bshift_err=y_error[0]
            else:
                bshift_data=x_data[0]
                bshift_err=x_error[0]

            #computing the mean and str
            bshift_mean=bshift_data.mean()
            bshift_sigma=bshift_data.std()
            bshift_mean_err=bshift_sigma/np.sqrt(len(bshift_data))

            # mean
            if time_mode:
                ax_scat.axhline(y=bshift_mean, xmin=0, xmax=1, color='orange', linestyle='-',
                                label=r'$\overline{v}_{curr}$',
                                lw=0.75)

                # span for the std
                ax_scat.axhspan(bshift_mean - bshift_sigma, bshift_mean - bshift_sigma, color='orange',
                                label=r'$\sigma_v_{curr}$',
                                alpha=0.1)

            else:
                ax_scat.axvline(x=bshift_mean, ymin=0, ymax=1, color='orange', linestyle='-', label=r'$\overline{v}_{curr}$',
                                lw=0.75)

                # span for the std
                ax_scat.axvspan(bshift_mean -bshift_sigma,bshift_mean +bshift_sigma, color='orange', label=r'$\sigma_{v_{curr}}$',
                                alpha=0.1)

            legend_title+=r'$\overline{v}_{curr}='+str(int(bshift_mean))+'\pm'+str(int(bshift_mean_err))+\
                          '$ \n $\sigma_{v_{curr}}='+str(int(bshift_sigma))+'$'
        #### Color definition
        
        #for the colors, we use the same logic than to create y_data with observation/source level parameters here
        
        #default colors for when color_scatter is set to None, will be overwriten
        color_arr=['blue']
        color_arr_ul='black'
        color_arr_ul_x=color_arr_ul
        color_arr_ul_y=color_arr_ul
            
        if color_scatter=='Instrument':

            #there's no need to repeat in ewratio since the masks are computed for a single line                
            if line_comp_mode or ratio_mode:
                color_data_repeat=instru_list
            else:
                color_data_repeat=np.array([instru_list for repeater in (i if type(i)==range else [i])])
            
            if ratio_mode:
                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                 ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]],dtype=object)
                                                          
            else:

                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                dtype=object)
                
                if not line_comp_mode:
                    #same thing for the upper limits
                    color_data_ul=ravel_ragged(color_data_repeat)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                #same thing for the upper limits in x and y
                color_data_ul_x=ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y=ravel_ragged(color_data_repeat)[bool_nondetsign_y]

            #computing the actual color array for the detections
            color_arr=np.array([np.array([telescope_colors[elem] for elem in color_data[s]]) for s in [0,1]],dtype=object)
            
            #and for the upper limits if needed
            if show_ul_ew:
                                    
                if ratio_mode or line_comp_mode:
                    color_arr_ul_x=np.array([telescope_colors[elem] for elem in color_data_ul_x])
        
                    color_arr_ul_y=np.array([telescope_colors[elem] for elem in color_data_ul_y])
                    
                else:
                    color_arr_ul=np.array([telescope_colors[elem] for elem in color_data_ul])

            #here we can keep a simple labeling
            label_dict=telescope_colors

        elif 'custom' in color_scatter:

            custom_color=diago_color if color_scatter=='custom_line_struct' else\
                         custom_states_color if color_scatter=='custom_acc_states' else\
                         custom_outburst_color if color_scatter=='custom_outburst' else 'grey'

            # there's no need to repeat in ewratio since the masks are computed for a single line
            if line_comp_mode or ratio_mode:
                color_data_repeat = custom_color
            else:
                color_data_repeat = np.array([custom_color for repeater in (i if type(i) == range else [i])])

            if ratio_mode:
                color_data = np.array([ravel_ragged(color_data_repeat,ragtuples=False)[bool_det_ratio][bool_sign_ratio],
                                       ravel_ragged(color_data_repeat,ragtuples=False)[bool_det_ratio][~bool_sign_ratio]], dtype=object)

            else:

                color_data = np.array([ravel_ragged(color_data_repeat,ragtuples=False)[bool_det][bool_sign],
                                       ravel_ragged(color_data_repeat,ragtuples=False)[bool_det][~bool_sign]],
                                      dtype=object)

                if not line_comp_mode:
                    # same thing for the upper limits
                    color_data_ul = ravel_ragged(color_data_repeat,ragtuples=False)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                # same thing for the upper limits in x and y
                color_data_ul_x = ravel_ragged(color_data_repeat,ragtuples=False)[bool_nondetsign_x]
                color_data_ul_y = ravel_ragged(color_data_repeat,ragtuples=False)[bool_nondetsign_y]

            # computing the actual color array for the detections
            color_arr = np.array([np.array([elem for elem in color_data[s]]) for s in [0, 1]],
                                 dtype=object)

            # and for the upper limits if needed
            if show_ul_ew:

                if ratio_mode or line_comp_mode:
                    color_arr_ul_x = np.array([elem for elem in color_data_ul_x])

                    color_arr_ul_y = np.array([elem for elem in color_data_ul_y])

                else:
                    color_arr_ul = np.array([elem for elem in color_data_ul])

            if color_scatter=='custom_line_struct':
                label_dict={'main structure':'grey','substructure':'orange','outliers':'blue'}
            elif color_scatter=='custom_acc_states':
                label_dict = {'undecided': 'grey',
                              'thermal dominated': 'green',
                              'intermediate': 'orange',
                              'SPL': 'red',
                              'canonical hard':'blue',
                              'QRM':'violet'}
            elif color_scatter=='custom_outburst':
                # note: 0 here because this is only used in display_single mode
                label_dict=custom_outburst_dict[0]

        elif color_scatter in ['Time','HR','width','nH','L_3-10']:
            
            color_var_arr=date_list if color_scatter=='Time'\
                else hid_plot[0][0] if color_scatter=='HR' else hid_plot[1][0] if color_scatter=='L_3-10'\
                else width_plot_restrict[0] if color_scatter=='width' else nh_plot_restrict[0]
                            
            #there's no need to repeat in ewratio since the masks are computed for a single line                
            if color_scatter=='width':
                color_data_repeat=color_var_arr
            else:
                if line_comp_mode or ratio_mode:
                    color_data_repeat=color_var_arr
                else:
                    color_data_repeat=np.array([color_var_arr for repeater in (i if type(i)==range else [i])])
            
            if ratio_mode:
                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                 ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]],dtype=object)
                                                          
            else:

                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                dtype=object)

                if not line_comp_mode:
                    #same thing for the upper limits
                    color_data_ul=ravel_ragged(color_data_repeat)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                
                #same thing for the upper limits in x and y
                color_data_ul_x=ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y=ravel_ragged(color_data_repeat)[bool_nondetsign_y]
                
            #here we compute a conversion of the dates to numerical values in the case of a Time colormap
            if color_scatter=='Time':
                c_arr=np.array([mdates.date2num(color_data[s]) for s in [0,1]],dtype=object)
            
                if ratio_mode or line_comp_mode:
                    c_arr_ul_x=mdates.date2num(color_data_ul_x)
                    c_arr_ul_y=mdates.date2num(color_data_ul_y)
                    
                else:
                    c_arr_ul=mdates.date2num(color_data_ul)
            else:
                c_arr=color_data
                
                if ratio_mode or line_comp_mode:
                    c_arr_ul_x=color_data_ul_x
                    c_arr_ul_y=color_data_ul_y
                else:
                    c_arr_ul=color_data_ul

            c_arr_tot=c_arr.tolist()
            
            # #adding the upper limits to the normalisation if necessary
            if show_ul_ew and ('ew' in infos or line_comp_mode):

                if line_comp_mode or ratio_mode:
                    c_arr_tot+=[c_arr_ul_x,c_arr_ul_y]
                else:
                    c_arr_tot+=[c_arr_ul]    
                        
            #differing norms for Time and HR:
            if len(ravel_ragged(c_arr_tot))>0:
                if min(ravel_ragged(c_arr_tot))==max(ravel_ragged(c_arr_tot)):
                    #safeguard to keep a middle range color when there's only one value
                    c_norm = mpl.colors.Normalize(vmin=min(ravel_ragged(c_arr_tot))*0.9,
                                                  vmax=max(ravel_ragged(c_arr_tot))*1.1)
                else:
                    if color_scatter in ['HR','nH','L_3-10']:
                        c_norm=colors.LogNorm(vmin=min(ravel_ragged(c_arr_tot)),
                                              vmax=max(ravel_ragged(c_arr_tot)))
                    else:

                        c_norm=mpl.colors.Normalize(vmin=min(ravel_ragged(c_arr_tot)),
                                                    vmax=max(ravel_ragged(c_arr_tot)))
            else:
                c_norm=None

            color_cmap=mpl.cm.plasma
            colors_func_date=mpl.cm.ScalarMappable(norm=c_norm,cmap=color_cmap)

            #computing the actual color array for the detections
            color_arr=np.array([[colors_func_date.to_rgba(elem) for elem in c_arr[s]] for s in ([0,1] if display_nonsign else [0]) ])

            #and for the upper limits
            if ratio_mode or line_comp_mode:
                
                #the axes swap in timemode requires swapping the indexes to fetch the uncertainty locations
                color_arr_ul_x=np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul_x])
                color_arr_ul_y=np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul_y])
            else:
                color_arr_ul=np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul]) 
            
        elif color_scatter=='Source':
            
            #there's no need to repeat in ewratio since the masks are computed for a single line                
            if line_comp_mode or ratio_mode:
                color_data_repeat=np.array([obj_disp_list[i_obj] for i_obj in range(n_obj)\
                                            for i_obs in range(len(data_perinfo[0][0][0][i_obj]))])
                
            else:
                color_data_repeat=np.array([obj_disp_list[i_obj] for repeater in (i if type(i)==range else [i])\
                                        for i_obj in range(n_obj) for i_obs in range(len(data_perinfo[0][0][repeater][i_obj]))\
                                    ])
            
            if ratio_mode:
                
                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                 ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]],dtype=object)

            else:

                color_data=np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],
                                     ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                            dtype=object)
                
                if not line_comp_mode:
                    #same thing for the upper limits
                    color_data_ul=ravel_ragged(color_data_repeat)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                #same thing for the upper limits in x and y
                color_data_ul_x=ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y=ravel_ragged(color_data_repeat)[bool_nondetsign_y]
                
            color_data_tot=[color_data[s] for s in ([0,1] if display_nonsign else [0])]
            #global array for unique extraction
            if show_ul_ew and ('ew' in infos or line_comp_mode):
                
                if not ratio_mode and not line_comp_mode:
                    #same thing for the upper limits
                    color_data_tot+=[color_data_ul]
                else:
                    #same thing for the upper limits in x and y
                    color_data_tot+=[color_data_ul_x,color_data_ul_y]
                    
                
            #we extract the number of objects with detection from the array of sources
            if glob_col_source:
                disp_objects=obj_disp_list                
            else:
                disp_objects=np.unique(ravel_ragged(color_data_tot))
            
            #and compute a color mapping accordingly
            norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=len(disp_objects)-1)
            colors_func_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=cmap_color_det)
        
            color_arr=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for s in ([0,1] if display_nonsign else [0]) for elem in color_data[s]])
                
            label_dict={disp_objects[i]:colors_func_obj.to_rgba(i) for i in range(len(disp_objects))}
            
            if not display_nonsign:
                color_arr=np.array([color_arr,'None'],dtype=object)

            #same for the upper limits if needed
            if show_ul_ew and ('ew' in infos or line_comp_mode):
                                    
                if ratio_mode or line_comp_mode:
                    color_arr_ul_x=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for elem in color_data_ul_x])
        
                    color_arr_ul_y=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for elem in color_data_ul_y])
                    
                else:

                    color_arr_ul=np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects==elem)[0][0]) for elem in color_data_ul])

                        
        def plot_ul_err(lims_bool,x_data,y_data,x_err,y_err,col,label=''):
            
            #this construction allows us to maintain a string color name when not using color_scatter

            for i_err,(x_data_single,y_data_single,x_err_single,y_err_single,col_single) in\
                enumerate(zip(x_data,y_data,x_err,y_err,col if color_scatter!='None' else np.repeat(col,len(x_data)))):
                
                ax_scat.errorbar(x_data_single,y_data_single,xerr=np.array([x_err_single]).T if x_err_single is not None else x_err_single,
                                 yerr=np.array([y_err_single]).T if y_err_single is not None else y_err_single,
                                 xuplims=lims_bool[0]==1,xlolims=lims_bool[0]==-1,uplims=lims_bool[1]==1,lolims=lims_bool[1]==-1,
                                 color=col_single,linestyle='',marker='.' if color_scatter=='None' else None,
                                 label=label if i_err==0 else '',alpha=alpha_ul)
                    
        ####plotting upper limits
        if show_ul_ew and ('ew' in infos or line_comp_mode):
            
            if line_comp_mode:
                #xuplims here
                plot_ul_err([1,0],x_data_ul_x,y_data_ul_x,x_data_ul_x*0.05,y_error_ul_x.T,color_arr_ul_x)
                
                #uplims here
                plot_ul_err([0,1],x_data_ul_y,y_data_ul_y,x_error_ul_y.T,y_data_ul_y*0.05,color_arr_ul_y)
                
            else:

                # else:
                #     ax_scat.set_xlim(mdates.date2num(['2012-01-01']),mdates.date2num(['2012-10-01']))
                
                if time_mode:
                    #uplims here, the upper limits display has the same construction no matter if in ratio mode or not
                    plot_ul_err([0,1],x_data_ul,y_data_ul,[None]*len(x_data_ul),
                                y_data_ul*0.05,color_arr_ul_x if ratio_mode else color_arr_ul)
                
                else:
                    
                    #xuplims here, the upper limits display has the same construction no matter if in ratio mode or not
                    plot_ul_err([1,0],x_data_ul,y_data_ul,x_data_ul*0.05,
                                y_error_ul.T,color_arr_ul_x if ratio_mode else color_arr_ul)
                
                #adding the lower limits in ratio mode
                if ratio_mode:
                    
                    if time_mode:
                          #lolims here
                          plot_ul_err([0,-1],x_data_ll,y_data_ll,[None]*len(x_data_ll),y_data_ll*0.05,color_arr_ul_y)
                    else:
                        #lolims here
                        plot_ul_err([-1,0],x_data_ll,y_data_ll,x_data_ll*0.05,y_error_ll.T,color_arr_ul_y)


        if time_mode:
            # creating an appropriate date axis
            # manually readjusting for small durations because the AutoDateLocator doesn't work well
            if time_range < 10:
                date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            elif time_range < 365:
                date_format = mdates.DateFormatter('%Y-%m-%d')
            else:
                date_format = mdates.DateFormatter('%Y-%m')

            ax_scat.xaxis.set_major_formatter(date_format)

            # forcing 8 xticks along the ax
            ax_scat.set_xlim(ax_scat.get_xlim())

            n_xticks=6
            # putting an interval in seconds (to avoid imprecisions when zooming)
            date_tick_inter = int((ax_scat.get_xlim()[1] - ax_scat.get_xlim()[0]) * 24 * 60 * 60 / n_xticks)

            # 10 days
            if date_tick_inter > 60* 60 * 24 * 10:
                ax_scat.xaxis.set_major_locator(mdates.DayLocator(interval=date_tick_inter//(24*60*60)))
            # 1 day
            elif date_tick_inter > 60* 60 * 24:
                ax_scat.xaxis.set_major_locator(mdates.HourLocator(interval=date_tick_inter//(60*60)))
            elif date_tick_inter > 60 * 60:
                ax_scat.xaxis.set_major_locator(mdates.MinuteLocator(interval=date_tick_inter//(60)))
            else:
                ax_scat.xaxis.set_major_locator(mdates.SecondLocator(interval=date_tick_inter))

            # and offsetting if they're too close to the bounds because otherwise the ticks can be missplaced
            if ax_scat.get_xticks()[0] - ax_scat.get_xlim()[0] > date_tick_inter / (24 * 60 *  60) * 3 / 4:
                ax_scat.set_xticks(ax_scat.get_xticks() - date_tick_inter / (2 * 24 * 60  * 60))

            if ax_scat.get_xticks()[0] - ax_scat.get_xlim()[0] < date_tick_inter / (24 * 60 * 60) * 1 / 4:
                ax_scat.set_xticks(ax_scat.get_xticks() + date_tick_inter / (2 * 24 * 60 * 60))

            # ax_lc.set_xticks(ax_lc.get_xticks()[::2])

            for label in ax_scat.get_xticklabels(which='major'):
                label.set(rotation=0 if date_tick_inter > 60 * 60 * 24 * 10 else 45, horizontalalignment='center')

        #### adding cropping on the EW ratio X axis to fix unknown issue
        
        #complicated restriction to take off all the elements of x_data no matter their dimension if they are empty arrays
        x_data_use=[elem for elem in x_data if len(np.array(np.shape(elem)).nonzero()[0])==np.ndim(elem)]

        if ratio_mode and len(x_data)>0:          
            
            if len(x_data_use)!=0:
                if width_mode:
                    if min(ravel_ragged(x_data_use))>0.28 and max(ravel_ragged(x_data_use))<3.5:
                        ax_scat.set_xlim(0.28,3.5)
                else:
                    if min(ravel_ragged(x_data_use))>0.28:
                        ax_scat.set_xlim(0.28,ax_scat.get_xlim()[1])
                    
        #### Color replacements in the scatter to match the colormap
        if color_scatter!='None':
            
            for s,elem_errbar in enumerate(errbar_list):
                #replacing indiviudally the colors for each point but the line
                                    
                #here the empty element is '' so we use a type comparison
                
                for elem_children in ([] if type(elem_errbar)==str else elem_errbar.get_children()[1:]):
                    
                    if type(elem_children)==mpl.collections.LineCollection:

                        elem_children.set_colors(color_arr[s][~(linked_mask[s].astype(bool))])

                        #highlighting BAT/INT projected luminosities without having to worry about how the graphs
                        #were created
                        if mode=='observ' and ind_infos[1] in [4,5] and mask_added_regr_sign_use is not None:

                            ls_dist=np.array([np.where(bool,'--','-') for bool in mask_added_regr_sign_use])

                            #older version
                            # ls_dist=np.array([np.where((elem in bat_lc_lum_scat.T[0] if ind_infos[1]==4 else bool),'--','-')\
                            #                   for elem,bool in zip(y_data[s],mask_added_regr_sign_use)])

                            ls_dist = ls_dist.tolist()

                            elem_children.set_linestyles(ls_dist)

                        #### distinguishing the 3 GRS obs
                        elif 'bshift' in infos_split:
                            
                            data_bshift=np.array([x_data[s],y_data[s]])[np.array(infos_split)=='bshift'][s]
                            
                            ls_dist=np.repeat('-',len(x_data[s]))
                            #to avoid issues due to the type of the array
                            ls_dist = ls_dist.tolist()

                            facecol_dist=np.repeat('full',len(x_data[s]))
                            

                            facecol_dist=facecol_dist.tolist()
                            
                            bshift_obscured=[948.903977133402, 1356.3406485107535, 2639.060062961778]
                            
                            index_bshift_GRS=np.argwhere([elem in bshift_obscured for elem in data_bshift]).T[0]

                            for i_obscured in index_bshift_GRS:
                                ls_dist[i_obscured]='--'
                                facecol_dist[i_obscured]='none'

                            elem_children.set_linestyles(ls_dist)

                        # deprecated
                        # #dashing the errorbar for the 4U point with adjusted manual values for the EW ratio
                        # elif color_scatter == 'custom_line_struct' and 'ewratio' in infos_split[0]:
                        #     ls_dist=np.repeat('-',len(x_data[s]))
                        #     #to avoid issues due to the type of the array
                        #     ls_dist = ls_dist.tolist()
                        #
                        #     ls_dist[id_point_refit]='--'
                        #
                        #     elem_children.set_linestyles(ls_dist)
                            
                    else:
                        elem_children.set_visible(False)
                        
                        #this case is restricted to the upper limits in width mode
                        x_ul=x_data[0].astype(float)[~(linked_mask[0].astype(bool))][uplims_mask]
                        
                        #using the max values (in yerr) as the upper limits so in y
                        y_ul=yerr_ul=np.array([elem[~(linked_mask[s].astype(bool))][uplims_mask] for elem in y_error[0]])[1]
                        
                        xerr_ul=None if x_error is None else np.array([elem[~(linked_mask[s].astype(bool))][uplims_mask] for elem in x_error[0]]).T
                        yerr_ul=y_ul*0.05
                        
                        col_ul=color_arr[0][~(linked_mask[0])][uplims_mask]
                        plot_ul_err([0,1],x_ul,y_ul,xerr_ul,yerr_ul,col_ul,label='')
                        
            for s,elem_errbar_linked in enumerate(errbar_list_linked):
                #replacing indiviudally the colors for each point but the line

                for elem_children in ([] if type(elem_errbar_linked)==str else elem_errbar_linked.get_children()[1:]):
                    
                    elem_children.set_colors(color_arr[s][(linked_mask[s].astype(bool))])
                
            if color_scatter in ['Time','HR','width','nH','L_3-10']:
                
                #adding space for the colorbar
                #breaks everything so not there currently
                #ax_cb=fig_scat.add_axes([0.99, 0.123, 0.02, 0.84 if not compute_correl else 0.73])
                #ax_cb = fig_scat.add_axes([0.99, 0.123, 0.02, 0.84])
                # divider = make_axes_locatable(ax_scat)
                # ax_cb = divider.append_axes('right', size='5%', pad=0.05)

                #scatter plot on top of the errorbars to be able to map the marker colors and create the colormap
                
                #no labels needed here
                scat_list=[ax_scat.scatter(x_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                           y_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                c=c_arr[s][~(linked_mask[s].astype(bool))],cmap=color_cmap,norm=c_norm,marker='D',alpha=1,zorder=1)\
                           for s in ([0,1] if display_nonsign else [0])]

                scat_list_linked=[ax_scat.scatter(x_data[s].astype(float)[(linked_mask[s].astype(bool))],
                                                  y_data[s].astype(float)[(linked_mask[s].astype(bool))],
                c=c_arr[s][(linked_mask[s].astype(bool))],cmap=color_cmap,norm=c_norm,marker='D',alpha=1,zorder=1)\
                                  for s in ([0,1] if display_nonsign else [0])]
                        
                #forcing a common normalisation for all the scatters
                scat_list_tot=scat_list+scat_list_linked
                        
                for elem_scat in scat_list_tot:

                    if len(ravel_ragged(c_arr_tot))>0:
                        elem_scat.set_clim(min(ravel_ragged(c_arr_tot)),max(ravel_ragged(c_arr_tot)))
                                                
                if color_scatter=='Time':
                    #creating the colormap

                    #manually readjusting for small durations because the AutoDateLocator doesn't work well
                    time_range=max(mdates.date2num(ravel_ragged(date_list)[mask_intime_norepeat]))\
                               -min(mdates.date2num(ravel_ragged(date_list)[mask_intime_norepeat]))

                    if time_range<150:
                        date_format=mdates.DateFormatter('%Y-%m-%d')
                    elif time_range<1825:
                        date_format=mdates.DateFormatter('%Y-%m')
                    else:
                        date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())

                    #test = fig_scat.colorbar(scat_list[0],cax=ax_cb,format=mdates.DateFormatter('%Y-%m'))

                    ####TODO: reintroduce cax to get constant ax size
                    test=plt.colorbar(scat_list[0],ticks=mdates.AutoDateLocator(),format=date_format)

                elif color_scatter in ['HR','width','nH','L_3-10']:

                    #creating the colormap (we add a bottom extension for nH to indicate the values cut)
                    test=plt.colorbar(scat_list[0],
                                 label=r'nH ($10^{22}$ cm$^{-2}$)' if color_scatter=='nH' else color_scatter,
                                 extend='min' if color_scatter=='nH' else None,aspect=30)


            else:
                
                #scatter plot on top of the errorbars to be able to map the marker colors
                #The loop allows to create scatter with colors according to the labels
                for s in ([0,1] if display_nonsign else [0]):


                    # for this we must split differently to ensure we keep the labeling for the
                    # first point displayed in each outburst while keeping the alpha colors for everything
                    #we do it now to make a dictionnary containing only the currently displayed points
                    if color_scatter == 'custom_outburst':

                        #creating a new dictionnary
                        label_dict_use={}

                        outburst_mask_list=[]
                        outburst_order_list=[]
                        for outburst in list(label_dict.keys()):
                            # selecting the obs part of this outburst
                            outburst_select_mask=[elem.all() for elem in (color_arr[s].T[:-1].T==label_dict[outburst][:-1])]

                            #saving the outburst masks for connectors
                            outburst_mask_list+=[outburst_select_mask]

                            #skipping outbursts with no observations
                            if sum(outburst_select_mask)==0:
                                continue

                            color_arr_outburst=color_arr[s][outburst_select_mask]
                            #computing the first point of the outburst (aka darkest) from the alpha orders
                            #we invert it to get the argosrt from the first to the last
                            outburst_order= (1-color_arr_outburst.T[-1]).argsort()

                            outburst_order_list+=[outburst_order]

                            outburst_init_id=outburst_order[0]

                            for i in range(len(color_arr_outburst)):
                                if outburst_init_id==i:
                                    #giving the outburst name
                                    label_dict_use[outburst]=color_arr_outburst[i]

                                #only adding dictionnary entries for element with different dates/colors
                                elif not np.array([(color_arr_outburst[i]==elem).all()\
                                      for elem in np.array(list(label_dict_use.values()))]).any():
                                    #giving a modified outburst name with a . to now it shouldn't be displayed
                                    #point number in the displayed ones
                                    outburst_id=np.argwhere(outburst_order==i)[0][0]
                                    label_dict_use[outburst+'.'+str(outburst_id)] = color_arr_outburst[i]

                            #putting the connector with the outburst color
                            #note that we overwrite the alpha so we don't care about which color we're taking

                            x_outburst=x_data[s].astype(float)[~(linked_mask[s].astype(bool))]\
                                [outburst_select_mask][outburst_order]
                            y_outburst=y_data[s].astype(float)[~(linked_mask[s].astype(bool))]\
                                [outburst_select_mask][outburst_order]
                            ax_scat.plot(x_outburst,y_outburst,
                                color=color_arr_outburst[0],
                                label='', alpha=0.3, zorder=0)

                            xscale_lin=ax_scat.get_xscale()=='linear'
                            yscale_lin=ax_scat.get_yscale()=='linear'
                            #and arrows, using annotate to avoid issues with ax limits and resizing

                            x_pos=(x_outburst[1:]+x_outburst[:-1])/2 if xscale_lin else \
                                10 ** ((np.log10(x_outburst[1:]) + np.log10(x_outburst[:-1])) / 2)

                            y_pos=(y_outburst[1:]+y_outburst[:-1])/2 if yscale_lin else \
                                10 ** ((np.log10(y_outburst[1:]) + np.log10(y_outburst[:-1])) / 2)

                            x_dir =x_outburst[1:]-x_outburst[:-1] if xscale_lin else \
                                10 ** (np.log10(x_outburst[1:]) - np.log10(x_outburst[:-1]))
                            y_dir = y_outburst[1:]-y_outburst[:-1] if yscale_lin else \
                                10 ** (np.log10(y_outburst[1:]) - np.log10(y_outburst[:-1]))

                            #this is the offset from the position, since we make the arrow start at the middle point of
                            #the segment we don't want it to go any further so we put it almost at the same value
                            #note that for some reason this ends up with non uniform proportions in log scale, but
                            #that remains good enough for now

                            arrow_size_frac = 0.1

                            for X, Y, dX, dY in zip(x_pos, y_pos, x_dir, y_dir):
                                ax_scat.annotate("", xytext=(X, Y),
                                                xy=(X + arrow_size_frac * dX if xscale_lin else\
                                                    10 ** (np.log10(X) + arrow_size_frac * np.log10(dX)),
                                                    Y + arrow_size_frac * dY if yscale_lin else\
                                                    10 ** (np.log10(Y) + arrow_size_frac * np.log10(dY))),
                                                arrowprops=dict(arrowstyle='->', color=color_arr_outburst[0],
                                                                alpha=0.3), size=10)


                        #replacing the dictionnary with this one
                        label_dict=label_dict_use
                        order_disp=np.array(list(label_dict.keys())).argsort()




                    else:
                        order_disp=np.arange(len(np.array(list(label_dict.keys()))))

                    for i_col,color_label in enumerate(np.array(list(label_dict.keys()))[order_disp]):
        
                        #creating a mask for the points of the right color
                        if color_scatter=='Instrument':

                            color_mask = [(elem == label_dict[color_label]).all() for elem in
                                          color_arr[s][~(linked_mask[s].astype(bool))]]

                            color_mask_linked=[(elem == label_dict[color_label]).all() for elem in
                                               color_arr[s][(linked_mask[s].astype(bool))]]

                        elif 'custom' in color_scatter:

                            if color_scatter=='custom_outburst':
                                color_mask = [(elem == label_dict[color_label]).all() for elem in
                                              color_arr[s][~(linked_mask[s].astype(bool))]]

                                color_mask_linked = [(elem == label_dict[color_label]).all() for elem in
                                                     color_arr[s][(linked_mask[s].astype(bool))]]
                            else:

                                color_mask = [elem == label_dict[color_label] for elem in
                                              color_arr[s][~(linked_mask[s].astype(bool))]]

                                color_mask_linked=[elem==label_dict[color_label] for elem in
                                                   color_arr[s][(linked_mask[s].astype(bool))]]

                        #same idea but here since the color is an RGB tuple we need to convert the element before the comparison
                        elif color_scatter=='Source':
                            color_mask=[tuple(elem) ==label_dict[color_label] for elem in color_arr[s][~(linked_mask[s].astype(bool))]]
                            color_mask_linked=[tuple(elem) ==label_dict[color_label] for elem in color_arr[s][(linked_mask[s].astype(bool))]]
        
                        #checking if there is at least one upper limit:
                        #(a bit convoluted but we cannot concatenate 0 len arrays so we add a placeholder that'll never get recognized instead)

                        col_concat=(color_arr_ul_x.tolist() if type(color_arr_ul_x)!=str else []+\
                                     color_arr_ul_y.tolist() if type(color_arr_ul_y)!=str  else [])\
                                         if (ratio_mode or line_comp_mode) else color_arr_ul

                        no_ul_displayed=np.sum([tuple(elem)==label_dict[color_label] for elem in col_concat])==0
                                
                        #not displaying color/labels that are not actually in the plot
                        if np.sum(color_mask)==0 and np.sum(color_mask_linked)==0 and (not show_ul_ew or no_ul_displayed):
                            continue
                        
                        #needs to be split to avoid indexation problem when calling color_mask behind
                        if uplims_mask is None:

                            #adding the marker color change for the BAT infered obs
                            if mode == 'observ' and ind_infos[1] in [4,5] and mask_added_regr_sign_use is not None:

                                facecol_adjust_mask=mask_added_regr_sign_use[color_mask]

                                #older version
                                # facecol_adjust_mask = np.array([(elem in bat_lc_lum_scat.T[0]) if ind_infos[1]==4\
                                #                                 else bool\
                                #                     for elem,bool in zip(y_data[s][color_mask],
                                #                                          mask_added_regr_sign_use[color_mask])])

                                for id_obs in range(len(y_data[s][color_mask])):
                                    ax_scat.scatter(
                                        x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                                        y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                                        color=label_dict[color_label] if not facecol_adjust_mask[id_obs] else None,
                                        facecolor='none' if facecol_adjust_mask[id_obs] else None,
                                        label=color_label if id_obs == 0 else '',
                                        marker='D',
                                        edgecolor=label_dict[color_label] if facecol_adjust_mask[id_obs] else None,
                                        alpha=1, zorder=1)

                            #adding the marker color change for the obscured obs
                            elif 'bshift' in infos_split:
                                
                                data_bshift_GRS=np.array([x_data[s][color_mask],y_data[s][color_mask]])[np.array(infos_split)=='bshift'][s]
                                
                                index_bshift_GRS_indiv=np.argwhere([elem in bshift_obscured for elem in data_bshift_GRS]).T[0]
                                
                                for id_GRS in range(len(data_bshift_GRS)):
                                    ax_scat.scatter(
                                        x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_GRS],
                                        y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_GRS],
                                              color=label_dict[color_label] if id_GRS not in index_bshift_GRS_indiv else None,
                                              facecolor=None if id_GRS not in index_bshift_GRS_indiv else 'none',
                                              label='' if color_scatter=='custom_outburst' and "." in color_label else\
                                                    color_label if id_GRS==0 else '',
                                              marker='D',edgecolor=None if id_GRS not in index_bshift_GRS_indiv else\
                                            label_dict[color_label],
                                              alpha=None if color_scatter=='custom_outburst' else 1,zorder=1)

                            # # changing the marker color for the 4U point with adjusted manual values for the EW ratio
                            # elif  color_scatter == 'custom_line_struct' and 'ewratio' in infos_split[0]:
                            #
                            #     for id_obs,id_obs_full in enumerate(np.arange(len(y_data[s]))[color_mask]):
                            #         ax_scat.scatter(
                            #             x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                            #             y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                            #             color=label_dict[color_label] if id_obs_full!=id_point_refit else None,
                            #             facecolor='none' if id_obs_full==id_point_refit else None,
                            #             label=color_label if id_obs == 0 else '',
                            #             marker='D',
                            #             edgecolor=label_dict[color_label] if id_obs_full==id_point_refit else None,
                            #             alpha=1, zorder=1)
                            
                            else:
                                ax_scat.scatter(
                                x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask],
                                y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask],
                                      color=label_dict[color_label],
                                label='' if color_scatter=='custom_outburst' and "." in color_label else color_label,
                                    marker='D',alpha=None if color_scatter=='custom_outburst' else 1,zorder=1)

                        else:
                            ax_scat.scatter(
                                x_data[s].astype(float)[~(linked_mask[s].astype(bool))][~uplims_mask][np.array(color_mask)[~uplims_mask]],
                                y_data[s].astype(float)[~(linked_mask[s].astype(bool))][~uplims_mask][np.array(color_mask)[~uplims_mask]],
                                      color=label_dict[color_label],
                                label='' if color_scatter=='custom_outburst' and "." in color_label else color_label,
                                marker='D',alpha=None if color_scatter=='custom_outburst' else 1,zorder=1)
                        
                        
                        #No label for the second plot to avoid repetitions
                        ax_scat.scatter(x_data[s].astype(float)[(linked_mask[s].astype(bool))][color_mask_linked],
                                                              y_data[s].astype(float)[(linked_mask[s].astype(bool))][color_mask_linked],
                                  color=label_dict[color_label],
                        label='',marker='D',alpha=1,zorder=1)
                        
        # ax_scat.set_xlim(mdates.date2num(['2012-01-01']),mdates.date2num(['2012-10-01']))

        #adjusting the axis sizes for ewratio mode to get the same scale
        if line_comp_mode:
            ax_scat.set_ylim(max(min(ax_scat.get_xlim()[0],ax_scat.get_ylim()[0]),0),max(ax_scat.get_xlim()[1],ax_scat.get_ylim()[1]))
            ax_scat.set_xlim(max(min(ax_scat.get_xlim()[0],ax_scat.get_ylim()[0]),0),max(ax_scat.get_xlim()[1],ax_scat.get_ylim()[1]))
            ax_scat.plot(ax_scat.get_xlim(),ax_scat.get_ylim(),ls='--',color='grey')

        #resizing for a square graph shape
        forceAspect(ax_scat,aspect=0.8)
        
        
        #### theoretical line drawing for ew_width
        if infos=='ew_width' and display_th_width_ew:
            
            #adding a blank line in the legend for cleaner display                                
            ax_scat.plot(np.NaN, np.NaN, '-', color='none', label='   ')
                
            line_id=np.argwhere(np.array(mask_lines))[0][0]
            if line_id==0:
                files=glob.glob('/home/parrama/Documents/Work/PhD/docs/atom/Fe25*_*.dat')
            elif line_id==1:
                files=glob.glob('/home/parrama/Documents/Work/PhD/docs/atom/Fe26*_*.dat')
                
            nh_values=np.array([elem.split('/')[-1].replace('.dat','').split('_')[1] for elem in files]).astype(float)
            
            nh_order=nh_values.argsort()
            
            nh_values.sort()
            
            ls_curve=['solid','dotted','dashed','dashdot',(0, (3, 5, 1, 5, 1, 5))]
            
            colors_curve=['black','black','black','black','black']

            for id_curve,nh_curve_path in enumerate(np.array(files)[nh_order]):
                nh_array=np.array(pd.read_csv(nh_curve_path,sep='      ',engine='python',header=None)).T
                
                ax_scat.plot(nh_array[0],nh_array[1],label=r'$N_i= $'+str(nh_values[id_curve])+' cm'+r'$^{-2}$',
                         color=colors_curve[id_curve],
                         alpha=0.5,ls=ls_curve[id_curve%len(ls_curve)])
                
        #logarithmic scale by default for ew ratios
        if ratio_mode:
        
            if time_mode:
                ax_scat.set_yscale('log')
                ax_scat.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                ax_scat.set_yticks(np.arange(0.3,0.8,0.2).tolist()+[1,3,5])
                
                ax_scat.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                
            else:
                ax_scat.set_xscale('log')
                ax_scat.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                ax_scat.set_xticks(np.arange(0.3,0.8,0.2).tolist()+[1,3,5])
                
                ax_scat.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        ####forcing common observ bounds if asked 
        #computing the common bounds if necessary
        if common_observ_bounds_lines:
            if mode=='observ' and sum(mask_bounds)>0 and not time_mode:
                plt.ylim(bounds_hr if infos_split[1]=='HR' else bounds_flux if infos_split[1]=='flux' else None)

        if common_observ_bounds_dates :
            if mode=='observ' and not ratio_mode and not time_mode and x_data_bounds is not None:

                #for linspace only for now
                delta_x=(max(x_data_bounds+x_err_bounds[1])-min(x_data_bounds-x_err_bounds[0]))*0.05

                plt.xlim(min(x_data_bounds-x_err_bounds[0])-delta_x,max(x_data_bounds+x_err_bounds[1])+delta_x)

        #### Correlation values

        restrict_comput_mask = np.repeat(True, len(x_data[0]))

        if restrict_comput_scatter:
            # each "non-applicability" condition is having the same upper and lower limit value
            if comput_scatter_lims[0][0] != comput_scatter_lims[0][1]:
                restrict_comput_mask = (restrict_comput_mask) & (x_data[0] >= comput_scatter_lims[0][0]) \
                                      & (x_data[0] <= comput_scatter_lims[0][1])
            if comput_scatter_lims[1][0] != comput_scatter_lims[1][1]:
                restrict_comput_mask = (restrict_comput_mask) & (y_data[0] >= comput_scatter_lims[1][0]) \
                                      & (y_data[0] <= comput_scatter_lims[1][1])

            if sum(restrict_comput_mask)==0:
                st.warning('No points remaining in trend bounds. Skipping bounds... ')
                restrict_comput_mask = np.repeat(True, len(x_data[0]))

        # giving default values to pearson and spearman to avoid issues when rerunning streamlit rapidly
        r_pearson = np.zeros(4).reshape(2, 2)
        r_spearman = np.zeros(4).reshape(2, 2)

        # computing the statistical coefficients for the significant portion of the sample (we don't care about the links here)
        if compute_correl and mode != 'source' and len(x_data[0]) > 1 and not time_mode:


            x_data_trend = x_data[0][restrict_comput_mask]
            y_data_trend = y_data[0][restrict_comput_mask]

            # we cannot transpose the whole arrays since in ewcomp/HID mode they are most likely ragged due to having both
            # the significant and unsignificant data, so we create regular versions manually
            x_error_sign_T = None if x_error is None else np.array([elem for elem in x_error[0]]).T[restrict_comput_mask]
            y_error_sign_T = None if y_error is None else np.array([elem for elem in y_error[0]]).T[restrict_comput_mask]

            # note: general p-score conversion: 1/scinorm.ppf((1 + error_percent/100) / 2)

            if display_pearson:
                r_pearson = np.array(
                    pymccorrelation(x_data_trend.astype(float), y_data_trend.astype(float),
                                    dx_init=x_error_sign_T / 1.65,dy_init=y_error_sign_T / 1.65,
                                    ylim_init=uplims_mask,
                                    Nperturb=1000, coeff='pearsonr', percentiles=(50, 5, 95)))

                # switching back to uncertainties from quantile values
                r_pearson = np.array([[r_pearson[ind_c][0], r_pearson[ind_c][0] - r_pearson[ind_c][1],
                                       r_pearson[ind_c][2] - r_pearson[ind_c][0]] \
                                      for ind_c in [0, 1]])

                # no need to put uncertainties now
                # str_pearson=r'$r_{Pearson}='+str(round(r_pearson[0][0],2))+'_{-'+str(round(r_pearson[0][1],2))+'}^{+'\
                #                         +str(round(r_pearson[0][2],2))+'}$ '+'| $p='+'%.1e' % Decimal(r_pearson[1][0 ])\
                #                         +'_{-'+'%.1e' % Decimal(r_pearson[1][1])+'}^{+'+'%.1e' % Decimal(r_pearson[1][2])+'}$\n'

                str_pearson = r'$r_{Pearson}=$' + str(round(r_pearson[0][0], 2)) + '$\;|\; $p=' + '%.1e' % Decimal(
                    r_pearson[1][0]) + '$\n'
            else:
                str_pearson=''

            r_spearman = np.array(
                pymccorrelation(x_data_trend.astype(float), y_data_trend.astype(float),
                                dx_init=x_error_sign_T / 1.65,dy_init=y_error_sign_T / 1.65,
                                ylim_init=uplims_mask,
                                Nperturb=1000, coeff='spearmanr', percentiles=(50, 5, 95)))

            # switching back to uncertainties from quantile values
            r_spearman = np.array([[r_spearman[ind_c][0], r_spearman[ind_c][0] - r_spearman[ind_c][1],
                                    r_spearman[ind_c][2] - r_spearman[ind_c][0]] \
                                   for ind_c in [0, 1]])


            # str_spearman=r'$r_{Spearman}='+str(round(r_spearman[0][0],2))+'_{-'+str(round(r_spearman[0][1],2))\
            #                           +'}^{+'+str(round(r_spearman[0][2],2))+'}$ '+'| $p='+'%.1e' % Decimal(r_spearman[1][0])\
            #                           +'_{-'+'%.1e' % Decimal(r_spearman[1][1])+'}^{+'+'%.1e' % Decimal(r_spearman[1][2])+'}$ '

            str_spearman = r'$r_S \,=' + str(round(r_spearman[0][0], 2)) + '$\n$p_S=' + '%.1e' % Decimal(
                r_spearman[1][0]) + '$'

            legend_title += (str_pearson if display_pearson else '') + str_spearman

        else:
            # placeholder to avoid activating the trends
            r_spearman = [[], [1]]

            legend_title += ''

        if compute_regr and compute_correl and r_spearman[1][0] < regr_pval_threshold:

            # note: doesn't consider non-significant detections

            x_data_regr = x_data[0][restrict_comput_mask]
            y_data_regr = y_data[0][restrict_comput_mask]

            # note: this is the right shape for lmpplot_uncert
            x_err_regr = (np.array([elem for elem in x_error[0]]).T)[restrict_comput_mask].T
            y_err_regr = (np.array([elem for elem in y_error[0]]).T)[restrict_comput_mask].T

            # since we can't easily change the lin to log type of the axes, we recreate a linear
            with st.spinner('Computing linear regrs'):
                lmplot_uncert_a(ax_scat, x_data_regr, y_data_regr, x_err_regr, y_err_regr, percent=90,
                                nsim=2000, return_linreg=False,percent_regions=[68,95,99.7],
                                error_percent=90,
                                intercept_pos='auto',inter_color=['lightgrey','silver','darkgrey'],
                                infer_log_scale=True, xbounds=plt.xlim(), ybounds=plt.ylim(),
                                line_color='black')

        #updating the ticks of the y axis to avoid powers of ten when unnecessary
        if ax_scat.get_yscale()=='log':
            if ax_scat.get_ylim()[0]>0.01:
                ax_scat.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))
                ax_scat.yaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))



        #### legend display
        if show_linked or (compute_correl and mode!='source' and len(x_data[0])>1 and not time_mode) or color_scatter not in ['Time','HR','width','nH','L_3-10',None]:
            
            scat_legend=ax_scat.legend(fontsize=9 if infos=='ew_width' and display_th_width_ew else 10,title=legend_title,
                                   ncol=2 if display_th_width_ew and infos=='ew_width' else 1,loc='upper right' if not ratio_mode else 'upper right')
            plt.setp(scat_legend.get_title(),fontsize='small')
                
        if len(x_data_use)>0:
            try:
                plt.tight_layout()
            except:
                st.rerun()
        else:
            #preventing log scales for graphs with no values, which crashes them
            plt.xscale('linear')
            plt.yscale('linear')

        
        #### custom things for the wind_review_global paper
        
        # #specific limits for the width graphs
        # ax_scat.set_xlim(0,90)
        
        if save:
            if indiv:
                suffix_str='_'+lines_std_names[3+i]
            else:
                suffix_str='_all'
                
            plt.savefig(save_dir+'/graphs/'+mode+'/'+save_str_prefix+'autofit_correl_'+infos+suffix_str+'_cam_'+args_cam+'_'+\
                            args_line_search_e.replace(' ','_')+'_'+args_line_search_norm.replace(' ','_')+'.png')
        if close:
            plt.close(fig_scat)
            
    #returning the graph for streamlit display
    if streamlit:
        return fig_scat

kev_to_erg = 1.60218e-9 # convertion factor from keV to erg


def flux_erg_pow(g,K,e1,e2):
    '''
    from Tristan Bouchet
    integrated flux between e1 and e2 from photon index (g) and norm (K) (with F = K*E**(-g)), in erg/s/cm2
    '''
    return (K/(2-g))*(e2**(2-g) - e1**(2-g))*kev_to_erg

def err_flux_erg_pow(g, K, dg, dK, e1,e2):
    '''
    from Tristan Bouchet
    finds the error on flux from photon index (g, dg) and norm (K,dK) in erg/s/cm2
    '''
    F12 = (K/(2-g))*(e2**(2-g) - e1**(2-g))
    if K==0. : return 0
    else : return F12 * np.sqrt( (dK/K)**2 + ((K/(F12*(2-g)))* (np.log(e2)*e2**(2-g)-np.log(e1)*e1**(2-g)) - 1/(2-g))**2 * (dg)**2 ) * kev_to_erg

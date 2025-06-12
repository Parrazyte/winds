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

import getpass
username=getpass.getuser()

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
from io import StringIO
from lmplot_uncert import lmplot_uncert_a

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their 
#'perturb values' function

from ast import literal_eval
# import time

'''Astro'''

#rough way of testing if online or not
online=os.getcwd().startswith('/mount/src')
project_dir='/'.join(__file__.split('/')[:-4])

#to be tested online
sys.path.append(os.path.join(project_dir,'observations/spectral_analysis/'))
sys.path.append(os.path.join(project_dir,'general/'))
sys.path.append(os.path.join(project_dir,'observations/visualisation/visual_line_tools'))

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
                  'SWIFT':colors_func_instru.to_rgba(5**(norm_pow))}

mpl_base_colors=mpl.colors.TABLEAU_COLORS
mpl_base_colors_list=list(mpl_base_colors.keys())
telescope_colors_inter={'Chandra':mpl_base_colors[mpl_base_colors_list[0]],
                  'NICER':mpl_base_colors[mpl_base_colors_list[1]],
                  'NuSTAR':mpl_base_colors[mpl_base_colors_list[2]],
                  'Suzaku':mpl_base_colors[mpl_base_colors_list[3]],
                   'XMM':mpl_base_colors[mpl_base_colors_list[4]],
                  'SWIFT':mpl_base_colors[mpl_base_colors_list[5]],
                  'INTEGRAL':mpl_base_colors[mpl_base_colors_list[6]]}

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
        'SwiftJ1753.5-0127': [79, 5, 5, 1],
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
        'XTEJ2012+381':[58,15,16,1]
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
           'SwiftJ1727.8-1613':[10.8,0.,0.,1],
           'SwiftJ174510.8-262411':[11.3,11.3,0,0],
           'SwiftJ1753.5-0127':[3.26,0.02,0.02,1],
           'SwiftJ1910.2-0546':[2.4,0.1,0.1,1],
           'V404Cyg':[155.3,0,0,1],
           'V4641Sgr':[67.6,0,0,1],
           'XTEJ1118+480':[4.1,0,0,1],
           'XTEJ1550-564':[37.0,0,0,1],
           'XTEJ1650-500':[7.7,0,0,1],
           'XTEJ1752-223':[7,7,0,0],
           'XTEJ1859+226':[6.6,0,0,1]}


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

#note that here we don't put the reports for the broad abslines since we don't know if they are associated to winds

wind_det_dict={'4U1543-47':source_wind('soft_1_0','soft_1_0'),
              '4U1630-47':source_wind('soft','soft'),
              'EXO1846-031':source_wind('hard_1'),
              'GROJ1655-40':source_wind('soft','soft'),
              'GRS1716-249':source_wind(visible='hard'),
              'GRS1758-258':source_wind('hard_0_0'),
              'GRS1915+105':source_wind('soft,hard','soft',infrared='hard_0_1'),
              'GX339-4':source_wind(soft_x='hard'),
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

#--------------------
#from Tristan
kev_to_erg = 1.60218e-9 # convertion factor from keV to erg

# def flux_erg_pow(g,K,e1,e2):
#     '''integrated flux between e1 and e2 from photon index (g) and norm (K) (with F = K*E**(-g)), in erg/s/cm2'''
#     return (K/(2-g))*(e2**(2-g) - e1**(2-g))*kev_to_erg
#
# def err_flux_erg_pow(g, K, dg, dK, e1,e2):
#     '''finds the error on flux from photon index (g, dg) and norm (K,dK) in erg/s/cm2'''
#     F12 = (K/(2-g))*(e2**(2-g) - e1**(2-g))
#     if K==0. : return 0
#     else : return F12 * np.sqrt( (dK/K)**2 + ((K/(F12*(2-g)))* (np.log(e2)*e2**(2-g)-np.log(e1)*e1**(2-g)) - 1/(2-g))**2 * (dg)**2 ) * kev_to_erg

# # example with Crab nebula:
# flux_erg_pow(2.09,9.9 ,30,50), err_flux_erg_pow(2.09,9.9,0.01,0.4 ,30,50)


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


#--------------------



#BAT conversion factors for 1 cts/s in 15-50 keV counts to 15-50keV flux
convert_BAT_count_flux={
                        #this one assumes 11e22nH (but that doesn't change anything) and 2.5 gamma powerlaw
                        '4U1630-47':3.597E-07
                        }

sources_det_dic=['GRS1915+105','GRS 1915+105','GROJ1655-40','H1743-322','4U1630-47','IGRJ17451-3022']

#should be moved somewhere for online
rxte_lc_path='/media/'+username+'/crucial_SSD/Observ/BHLMXB/RXTE/RXTE_lc_dict.pickle'

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
        ctl_blackcat=pd.read_html('https://web.archive.org/web/20250325064252/https://www.astro.puc.cl/BlackCAT/transients.php')[1]

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

        ctl_maxi_df = pd.read_html(StringIO(str(maxi_html_table)), header=0)[0]

        #we create the request for the Simbad names of all MAXI sources here
        #so that it is done only once per script launch as it takes some time to run

        ctl_maxi_df['standard'] = [link.get('href').replace('..',maxi_url[:maxi_url.find('/top')]).replace('.html','_g_lc_1day_all.dat')\
                               for link in [elem for elem in maxi_html_table.find_all('a') if 'star_data' in elem.get('href')]]

        ctl_maxi_simbad=silent_Simbad_query(ctl_maxi_df['source name'])

    with st.spinner('Loading Swift-BAT transients...'):

        bat_url = "https://swift.gsfc.nasa.gov/results/transients/"

        with requests.get(bat_url) as bat_r:
            bat_html_table = BeautifulSoup(bat_r.text, features='lxml').find('table')

        ctl_bat_df = pd.read_html(StringIO(str(bat_html_table)), header=0)[0]

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

    if simbad_query['main_id'][0] not in _ctl_bat_simbad['main_id']:
        return None


    # we fetch the script id instead of the direct column number because Simbad erases the columns with no match
    # (-1 because the id starts at 1)
    source_id = _ctl_bat_simbad['object_number_id'][_ctl_bat_simbad['main_id'] == simbad_query['main_id'][0]][0] - 1

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
    
    if simbad_query['main_id'][0] not in _ctl_maxi_simbad['main_id']:
        return None
    
    #we fetch the script id instead of the direct column number because Simbad erases the columns with no match 
    #(-1 because the id starts at 1)
    source_id=_ctl_maxi_simbad['object_number_id'][_ctl_maxi_simbad['main_id']==simbad_query['main_id']][0]-1

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

    if simbad_query[0]['main_id'] not in dict_rxte.keys():
        return None
    
    return dict_rxte[simbad_query[0]['main_id']]

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
                note: the 90s RXTE lightcurves are also available online, just to heavy to be loaded dynamically
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
        maxi_y_str='MAXI 2-20 keV rate' if maxi_lc_df is not None and slider_date[1].year >= 2009 else ''
        #full name is rxte_lc_df.columns[1]
        rxte_y_str='RXTE 1.5-12 keV rate/'+str(20 if name[0]=='4U1630-47' else 25) if rxte_lc_df is not None \
                    and slider_date[0].year <= 2012 else ''
        label_both = maxi_y_str!='' and rxte_y_str!=''
        ax_lc.set_ylabel(maxi_y_str+(' | ' if label_both else '')+rxte_y_str)

    elif mode=='HR_soft':
        ax_lc.set_title(name[0]+str_binning_monit+' Soft Hardness Ratio monitoring')

        maxi_y_str='MAXI counts HR in [4-10]/[2-4] keV' if maxi_lc_df is not None and slider_date[1].year >= 2009 else ''
        rxte_y_str='RXTE band C/(B+A) [5-12]/[1.5-5] keV' if rxte_lc_df is not None and slider_date[0].year <= 2012 else ''
        label_both=maxi_y_str!='' and rxte_y_str!=''
        ax_lc.set_ylabel(maxi_y_str + (" | " if label_both else '')+rxte_y_str,fontsize=6 if label_both else None)

    elif mode=='HR_hard':
        ax_lc.set_title(name[0]+str_binning_monit+' Hard Hardness Ratio monitoring')

        maxi_y_str = 'MAXI counts HR in [10-20]/[2-4] keV' if maxi_lc_df is not None else ''
        ax_lc.set_ylabel(maxi_y_str)

    elif mode=='BAT':
        # ax_lc.set_title(name[0]+str_binning_monit+' BAT monitoring')
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

            #this is a "worst case" error composition
            rxte_hr_err=abs((rxte_lc_df[rxte_lc_df.columns[8]]/rxte_lc_df[rxte_lc_df.columns[7]]+
                             (rxte_lc_df[rxte_lc_df.columns[4]]+rxte_lc_df[rxte_lc_df.columns[6]])
                             /rxte_hr_denom)*rxte_hr)
            
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
            #base value taken from MJDREFI+MJDREFF in obs with the instrument
            base_date=Time(51910.00074287037,format='mjd')+\
                                                 TimeDelta(bat_lc_df['TIME'],format='sec')+\
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

        ax_lc_ew.set_ylim(min(4,min(min(ravel_ragged(abslines_plot_restrict[0][0])[bool_detsign]),
                                    min(ravel_ragged(abslines_plot_restrict[5][0])[bool_nondetsign]))),
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
                        label=instru_list[i_obs]+' exposure' if instru_list[i_obs] not in label_tel_list else '',
                      ls=':',lw=1.)

        if instru_list[i_obs] not in label_tel_list:
            label_tel_list+=[instru_list[i_obs]]

    if name[0] == 'GROJ1655-40':
        # showing the Swift photodiode exposures for GRO J1655-40 for the manuscript plots
        mjd_arr_swift_1655 = [53448,
                              53449.2,
                              53450.2,
                              53456.4,
                              53463.5,
                              53463.7,
                              53470.4,
                              53481.9,
                              53494,
                              53504.3,
                              53505.4,
                              53506.5,
                              53511.4,
                              53512.3,
                              53512.9]
        num_date_swift_1655 = mdates.date2num(Time(mjd_arr_swift_1655, format='mjd').datetime)

        for i_obs_swift_1655,date_obs_swift_1655 in enumerate(num_date_swift_1655):
            ax_lc.axvline(x=date_obs_swift_1655, ymin=0, ymax=1, color='grey',
                          label='Swift PD exposure' if i_obs_swift_1655==0 else '', ls=':', lw=1.)

    #resizing the x axis and highlighting depending on wether we are zooming on a restricted time interval or not
    
    tot_dates_list=[]
    tot_dates_list+=[] if maxi_lc_df is None else num_maxi_dates.tolist()
    tot_dates_list+=[] if rxte_lc_df is None else num_rxte_dates.tolist()
    tot_dates_list+=[] if bat_lc_df is None else  num_bat_dates.tolist()

    # and offsetting if they're too close to the bounds because otherwise the ticks can be missplaced

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

    if time_range<=0:
        no_monit_points=True
    else:
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

        if ax_lc.get_xticks()[0]-ax_lc.get_xlim()[0]>date_tick_inter/(24*60)*3/4:
            ax_lc.set_xticks(ax_lc.get_xticks()-date_tick_inter/(2*24*60))

        if ax_lc.get_xticks()[0]-ax_lc.get_xlim()[0]<date_tick_inter/(24*60)*1/4:
            ax_lc.set_xticks(ax_lc.get_xticks()+date_tick_inter/(2*24*60))

        # ax_lc.set_xticks(ax_lc.get_xticks()[::2])

        for label in ax_lc.get_xticklabels(which='major'):
            label.set(rotation=0 if date_tick_inter>60*24*10 else 45, horizontalalignment='center')

        #prettier but takes too much space
        # label.set(rotation=45, horizontalalignment='right')

    #MAXI contamination zone for 4U1630-47
    if name[0]=='4U1630-47' and mode in ['full','HR_soft','HR_hard']:
        conta_start=mdates.date2num(Time('2018-12-10').datetime)
        conta_end=mdates.date2num(Time('2019-06-20').datetime)
        plt.axvspan(xmin=conta_start,xmax=conta_end,color='grey',zorder=1000,alpha=0.3)

    #MAXI contamination zone for H1743-322
    if name[0]=='H1743-322' and mode in ['full','HR_soft','HR_hard']:
        #contamination zone for H1743-322 in 2023: see https://www.astronomerstelegram.org/?read=15919
        conta_start=mdates.date2num(Time('2023-03-01').datetime)
        conta_end=mdates.date2num(Time('2023-05-03').datetime)
        plt.axvspan(xmin=conta_start,xmax=conta_end,color='grey',zorder=1000,alpha=0.3)

    if name[0]=='GROJ1655-40':

        #conservative limit acording to the continuous "stable" interval defined in Uttley2015
        # see https://academic.oup.com/mnras/article/451/1/475/1375790 page 3
        hypersoft_start=mdates.date2num(Time('53459',format='mjd').datetime)
        hypersoft_end=mdates.date2num(Time('53494',format='mjd').datetime)
        plt.axvspan(xmin=hypersoft_start,xmax=hypersoft_end,color='green',zorder=1000,alpha=0.3,
                    label='hypersoft state')

    # #hypersoft state for GROJ1655-40
    try:
        ax_lc.legend(loc='upper right' if name[0]=="GROJ1655-40" else 'upper left',ncols=2)
    except:
        #to avoid an issue with a different python version online
        ax_lc.legend(loc='upper right' if name[0]=="GROJ1655-40" else 'upper left')

    # #shifted position if need be
    # ax_lc.legend(loc='upper right' if name[0]=="GROJ1655-40" else 'center right',ncols=2,
    #              bbox_to_anchor=(0.5, 0.53, 0.5, 0.5))

    if superpose_ew:
        # ax_lc_ew.legend(loc='upper center')
        ax_lc_ew.legend(loc='upper right')

    
    return fig_lc


#@st.cache_data
def obj_values(file_paths,dict_linevis,local_paths=False):
    
    '''
    Extracts the stored data from each value line_values file. 
    Merges all the files with the same object name
    
    the visual_line option give more information but requires a higher position in the directory structure
    '''
    
    obj_list=dict_linevis['obj_list']
    cameras=dict_linevis['args_cam']
    expmodes=dict_linevis['expmodes']
    multi_obj=dict_linevis['multi_obj']

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
        if local_paths:
            curr_obj_paths = file_paths
        else:
            curr_obj_paths=[elem for elem in file_paths if '/'+obj_list[i]+'/' in elem]

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
                
            #splitting the flux values for easier plotting
            curr_l_list[l]=np.array(curr_lval_list[l][-1])
            
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
                obs_suffix_list=[elem for elem in literal_eval(summary_obs_line[0].split('\t')[1]) if elem.startswith('_')]

                if len(obs_suffix_list)==0:
                    #this can happen for multi epchs with only full file name in the file identifiers
                    obs_file=literal_eval(summary_obs_line[0].split('\t')[1])[0]
                else:
                    obs_file=obs+('_gti_event_spec' if 'xis' in obs else '_sp')+obs_suffix_list[0]

                filepath=os.path.join('/'.join(lineval_path.split('/')[:-2]),obs_file)


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

                try:
                    elem_array=np.array(literal_eval(curr_line[m+1].replace(',','').replace(' ',',')\
                                                     .replace('nan','315151582340293')))\
                        if curr_line[m+1]!='' else None

                    #this weird line is here to not keep the nans since they cannot be directly understood
                    #with literal_eval
                    elem_array=np.where(elem_array==315151582340293,np.nan,elem_array)

                except:
                    breakpoint()
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
                        try:
                            array_obs=np.array([elem if len(np.shape(elem))==2\
                                            else np.array([[elem[i],None,None] for i in range(len(elem))]).astype(float)\
                                                for elem in abslines_inf[i_obj][i_obs]])
                        except:
                            breakpoint()

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
               lum_plt,hid_plt,width_plt,nh_plt,kt_plt,lum_list
    else:
        return abslines_inf_line, abslines_inf_obj, abslines_plt, abslines_e, \
            lum_plt, hid_plt, width_plt, nh_plt, kt_plt

def values_manip_var(val_high_list):

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


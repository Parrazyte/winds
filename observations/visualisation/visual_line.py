#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:11:45 2022

@author: parrama
"""
from __future__ import annotations

#general imports
import io as io
import os,sys
import re
import glob

import argparse
import zipfile

import numpy as np
import pandas as pd

import streamlit as st
#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit.components.v1

#disabling the warning for many open figures because that's exactly the point of the code
plt.rcParams.update({'figure.max_open_warning': 0})

import matplotlib.colors as colors

import time
from astropy.time import Time,TimeDelta
from datetime import date

from copy import deepcopy

from scipy.stats import norm
#correlation values and uncertainties with MC distribution from the uncertainties
from custom_pymccorrelation import pymccorrelation

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their 
#'perturb values' function

from ast import literal_eval
# import time

import dill
'''
#Astro
'''

#Catalogs and manipulation
from astroquery.vizier import Vizier

#visualisation functions
from visual_line_tools import load_catalogs,dist_mass,obj_values,abslines_values,values_manip,distrib_graph,correl_graph,\
    n_infos, plot_lightcurve, hid_graph, sources_det_dic, dippers_list,telescope_list,load_integral,telescope_colors,\
    convert_BAT_count_flux,flux_erg_pow,err_flux_erg_pow,values_manip_var,corr_factor_lbat,fetch_bat_lightcurve,\
    int_rate_to_flux,incl_dyn_dict,incl_jet_dict,incl_misc_dict,incl_refl_dict,Porb_dict,wind_det_dict,wind_det_sources

from lmplot_uncert import lmplot_uncert_a
# import mpld3

# import streamlit.components.v1 as components


ap = argparse.ArgumentParser(description='')

'''
#GENERAL OPTIONS
'''


ap.add_argument("-cameras",nargs=1,help='Cameras to use for the spectral analysis',default='all',type=str)
ap.add_argument("-expmodes",nargs=1,help='restrict the analysis to a single type of exposure',default='all',type=str)
ap.add_argument("-prefix",nargs=1,help='restrict analysis to a specific prefix',default='auto',type=str)
ap.add_argument("-outdir",nargs=1,help="name of output directory for line plots",default="lineplots_opt",type=str)

'''
#DIRECTORY SPECIFICS
'''

ap.add_argument("-local",nargs=1,help='launch analysis in the current directory instead',default=True,type=bool)

'''
#MODES
'''

ap.add_argument('-multi_obj',nargs=1,help='compute the hid for multiple obj directories inside the current directory',
                default=True)

'''
#SPECTRUM PARAMETERS
'''


ap.add_argument("-line_cont_range",nargs=1,help='min and max energies of the line continuum broand band fit',default='4 10',type=str)
ap.add_argument("-line_cont_ig",nargs=1,help='min and max energies of the ignore zone in the line continuum broand band fit',
                default='6.-8.',type=str)

ap.add_argument("-line_search_norm",nargs=1,help='min, max and nsteps (for one sign)  of the line norm search (which operates in log scale)',
                default='0.01 10 500',type=str)

'''
#VISUALISATION
'''

args=ap.parse_args()

#adding the top directory to the path to avoid issues when importing fitting_tools

#local
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/')

#online
sys.path.append('/mount/src/winds/observations/spectral_analysis/')
sys.path.append('/mount/src/winds/general/')

#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std,lines_std_names,range_absline

from general_tools import ravel_ragged,MinorSymLogLocator
'''
# Notes:
# -Only works for the auto observations (due to prefix naming) for now

# -For now we fix the masses of all the objets at 10M_sol

# -Due to the way the number of steps is computed, we explore one less value for the positive side of the normalisation

# -The norm_stepval argument is for a fixed flux band, and the value is scaled in the computation depending on the line energy step
'''

cameras=args.cameras
expmodes=args.expmodes
prefix=args.prefix
outdir=args.outdir

#rough way of testing if online or not
online='parrama' not in os.getcwd()
if online:
    project_dir='/mount/src/winds/'
else:
    project_dir='/home/parrama/Documents/Work/PhD/Scripts/Python/'


line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
line_cont_ig=args.line_cont_ig
line_search_norm=np.array(args.line_search_norm.split(' ')).astype(float)

multi_obj=args.multi_obj

#don't know where the bug comes from tbf
try:
    st.set_page_config(page_icon=":hole:",layout='wide')
except:
    pass
                   
#readjusting the variables in lists
if cameras=='all':
    #note: we use nu as the fpma/fpmb alias
    cameras=['pn','mos1','mos2','heg','xis','pin','nu']

else:
    cameras=[cameras]
    if 'pn' in cameras[0]:
        cameras=cameras+['pn']
    if 'mos1' in cameras[0]:
        cameras=cameras+['mos1']
    if 'mos2' in cameras[0]:
        cameras=cameras+['mos2']    
    if 'heg' in cameras[0]:
        cameras=cameras+['heg']
    cameras=cameras[1:]

if expmodes=='all':
    expmodes=['Imaging','Timing']
else:
    expmodes=[expmodes]
    if 'timing' in expmodes[0] or 'Timing' in expmodes[0]:
        expmodes=expmodes+['Timing']
    if 'imaging' in expmodes[0] or 'Imaging' in expmodes[0]:
        expmodes=expmodes+['Imaging']
    expmodes=expmodes[1:]
    
def getoverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

    
# @st.cache
# def folder_state(folderpath='./'):
#     #fetching the previously computed directories from the summary folder file
#     try:
#         with open(os.path.join(folderpath,outdir,'summary_line_det.log')) as summary_expos:
#             launched_expos=summary_expos.readlines()
    
#             #creating variable for completed analysis only
#             completed_expos=['_'.join(elem.split('\t')[:-1]) for elem in launched_expos if 'Line detection complete.' in elem]
#             launched_expos=['_'.join(elem.split('\t')[:-1]) for elem in launched_expos]
#     except:
#         launched_expos=[]
#         completed_expos=[]
        
#     return launched_expos,completed_expos

'''
#Initialisation
'''

# #for the current directory:
# started_expos,done_expos=folder_state()
 
# #bad spectra manually taken off
bad_flags=[]

norm_par_space=np.concatenate((-np.logspace(np.log10(line_search_norm[1]),np.log10(line_search_norm[0]),int(line_search_norm[2]/2)),np.array([0]),
                                np.logspace(np.log10(line_search_norm[0]),np.log10(line_search_norm[1]),int(line_search_norm[2]/2))))
norm_nsteps=len(norm_par_space)

if not online:
    if not multi_obj:

        #assuming the last top directory is the object name
        obj_name=os.getcwd().split('/')[-2]

        #listing the exposure ids in the bigbatch directory
        bigbatch_files=glob.glob('**')

        #tacking off 'spectrum' allows to disregard the failed combined lightcurve computations of some obsids as unique exposures compared to their
        #spectra
        exposid_list=np.unique(['_'.join(elem.split('_')[:4]).replace('rate','').replace('.ds','')+'_auto' for elem in bigbatch_files\
                      if '/' not in elem and 'spectrum' not in elem and elem[:10].isdigit() and True in ['_'+elemcam+'_' in elem for elemcam in cameras]])
        #fetching the summary files for the data reduction steps
        with open('glob_summary_extract_reg.log') as sumfile:
            glob_summary_reg=sumfile.readlines()[1:]
        with open('glob_summary_extract_sp.log') as sumfile:
            glob_summary_sp=sumfile.readlines()[1:]

        #loading the diagnostic messages after the analysis has been done
        if os.path.isfile(os.path.join(outdir,'summary_line_det.log')):
            with open(os.path.join(outdir,'summary_line_det.log')) as sumfile:
                glob_summary_linedet=sumfile.readlines()[1:]

        #creating summary files for the rest of the exposures
        lineplots_files=[elem.split('/')[1] for elem in glob.glob(outdir+'/*',recursive=True)]

        aborted_exposid=[elem for elem in exposid_list if not elem+'_recap.pdf' in lineplots_files]


#Distance and Mass determination

st.sidebar.header('Sample selection')

line_search_e_str_arg= st.sidebar.radio('Mode',('visual_line','BID'))
BID_mode=line_search_e_str_arg=='BID'
line_search_e_str=line_search_e_str_arg.replace('visual_line','4 10 0.05').replace('BID','4 10 0.02')

line_search_e=np.array(line_search_e_str.split(' ')).astype(float)
#we create these variables in any case because the multi_obj plots require them
line_search_e_space=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2],line_search_e[2])
#this one is here to avoid adding one point if incorrect roundings create problem
line_search_e_space=line_search_e_space[line_search_e_space<=line_search_e[1]]


radio_epoch_split=st.sidebar.radio('Observation type',['averaged ObsID'] if BID_mode else ('averaged ObsID','individual orbits','Time Resolved'))
use_obsids=radio_epoch_split=='averaged ObsID'
use_orbit_obs=radio_epoch_split=='individual orbits'
use_time_resolved=radio_epoch_split=='Time Resolved'
time_resolved_split_avail = ['100', '30','var']

if use_time_resolved:
    time_resolved_split=st.sidebar.radio('Temporal Resolution (s)',time_resolved_split_avail)
    time_resolved_dump_str='_TR_'+time_resolved_split
else:
    time_resolved_dump_str=''
    time_resolved_split=''

use_orbit_obs_str='_indiv' if use_orbit_obs else ''

#note: here the goal is to allow to have indiv_X individual lineplots directories that won't load
#but let lineplots_opt_X directories free when not in orbit_mode
use_orbit_obs_str_path=use_orbit_obs_str+'/'


#We put the telescope option before anything else to filter which file will be used
choice_telescope=st.sidebar.multiselect('Telescopes',['NICER'] if BID_mode else (['NICER','NuSTAR'] if use_orbit_obs or time_resolved_split=='100' else ['NICER'])\
                                                        if use_orbit_obs or use_time_resolved else\
                 (['XMM','Chandra']+(['NICER','NuSTAR','Suzaku','Swift'] if online else ['NICER','NuSTAR','Suzaku','Swift'])),
                                        default=['NICER'] if BID_mode else((['NICER','NuSTAR'] if use_orbit_obs or time_resolved_split=='100' else ['NICER'])\
                                            if use_orbit_obs or use_time_resolved else \
                                            ['XMM','Chandra']))

if online:
    include_full=False
    include_untested= st.toggle('Include untested and preliminary NICER sources',value=False if line_search_e_str=='4 10 0.05' else True)

    st.info('''
    The current version is experimental as I'm currently implementing many new fonctionalities, for a new study on 4U 1630-47.
    
    See https://theses.fr/s296242 (manuscript soon up).
    
    The "about" part will be updated once the first paper on 4U 1630-47 will be accepted.
    
    Please send an email if you find bugs
    #''', icon="ℹ️")

else:
    with st.sidebar:
        if not BID_mode:
            include_full=st.toggle('Include problematic data (_full) folders')
        else:
            include_full=False
        if not BID_mode:
            include_untested= st.toggle('Include untested and preliminary NICER sources',value=False if line_search_e_str=='4 10 0.05' else True)
        else:
            include_untested=True

use_unsure_mass_dist=st.sidebar.toggle('Use tentative mass and distance measurements',value=True)

if not online:
    os.system('mkdir -p glob_batch/visual_line_dumps/')

join_telescope_str=np.array(choice_telescope)
join_telescope_str.sort()
join_telescope_str='_'.join(join_telescope_str.tolist())

if online:
    dump_path='/mount/src/winds/observations/visualisation/visual_line_dumps/dump_'+join_telescope_str+\
              use_orbit_obs_str+time_resolved_dump_str+\
              ('_full' if include_full else '')+\
              ('_untested' if include_untested else '')+ \
              ('_with_md_unsure' if use_unsure_mass_dist else '')+ \
              ('_line_search_'+line_search_e_str.replace(' ','_') if line_search_e_str!='4 10 0.05' else '')+ \
              '.pkl'

    if not os.path.isfile(dump_path):
        print(dump_path)
        st.error('Dump file not found for this configuration')

    update_dump=False
else:
    dump_path='./glob_batch/visual_line_dumps/dump_'+join_telescope_str+\
              use_orbit_obs_str+time_resolved_dump_str+\
              ('_full' if include_full else '')+\
              ('_untested' if include_untested else '')+ \
              ('_with_md_unsure' if use_unsure_mass_dist else '')+ \
              ('_line_search_' + line_search_e_str.replace(' ', '_') if line_search_e_str != '4 10 0.05' else '') + \
              '.pkl'

    update_dump=st.sidebar.button('Update dumps')

#wrapped in a function to be cachable in streamlit

dump_path_catal=project_dir+'/observations/visualisation/visual_line_dumps/dump_catalogs.pkl'

if not online:
    update_online=st.sidebar.button('Update online version')
else:
    update_online=False
    
if update_online:
    
    #updating script
    path_online=__file__.replace('visual_line','visual_line_online')
    os.system('cp '+__file__+' '+path_online)
    
    #opening the online file and replacing ''' by ### due to the magic disabler not working currently
    
    with open(path_online,'r') as online_file:
        online_lines=online_file.readlines()

    for i in range(len(online_lines)):
        
        #trick to avoid replacing the markdown where we don't want to
        #the last replace is to override this specific line (the following one) which gets fucked by the rest,
        # the pass allows a line to remain in the if
        #here this line will be commented so that's why we keep it as a single line
        online_lines[i]=online_lines[i].replace(".markdown('''",'.markdown(@@@').replace(".info('''",'.info(@@@').replace(".header('''",'.header(@@@').replace("#'''",'@@@').replace("'''#","@@@").replace("r'''",'r@@@').replace("'''",'###').replace("@@@","'''").replace('online_lines[i]','#')
        
        pass
            
    with open(path_online,'w') as online_file:
        online_file.writelines(online_lines)
        
    
    #updating dumps to one level above the script
    os.system('cp -r ./glob_batch/visual_line_dumps/ '+path_online[:path_online.rfind('/')]+'/')

if update_dump or not os.path.isfile(dump_path):
    
    with st.spinner(text='Updating dump file...' if update_dump else\
                    'Using new configuration. Creating dump file...'):
        #### file search
        
        all_files=glob.glob('**',recursive=True)
        lineval_id='line_values_'+line_search_e_str.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
        lineval_files=[elem for elem in all_files if \
                       (outdir+use_orbit_obs_str_path in elem\
                       or outdir+'_nth'+use_orbit_obs_str_path in elem and 'Suzaku' not in elem)\
                       and lineval_id in elem\
                       and ('/Sample/' in elem or 'XTEJ1701-462/' in elem)\
                       and ('/'+time_resolved_split+'s/' in elem if use_time_resolved else\
                            np.all(['/'+elem_split+'s/' not in elem for elem_split in time_resolved_split_avail]))]

        abslines_id='autofit_values_'+line_search_e_str.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
        abslines_files=[elem for elem in all_files if \
                        (outdir+use_orbit_obs_str_path in elem\
                       or outdir+'_nth'+use_orbit_obs_str_path in elem and 'Suzaku' not in elem)\
                        and abslines_id in elem\
                        and ('/Sample/' in elem or 'XTEJ1701-462/' in elem) \
                        and ('/' + time_resolved_split + 's/' in elem if use_time_resolved else \
                                 np.all(['/' + elem_split + 's/' not in elem for elem_split in time_resolved_split_avail]))]

        #telescope selection
        lineval_files=[elem for elem_telescope in choice_telescope for elem in lineval_files if elem_telescope+'/' in elem]
        abslines_files=[elem for elem_telescope in choice_telescope for elem in abslines_files if elem_telescope+'/' in elem]

        #removing individual

        #separating the multi from the rest
        lineval_files_multi=[elem for elem in lineval_files if '/multi/' in elem]
        abslines_files_multi=[elem for elem in abslines_files if '/multi/' in elem]

        lineval_files=[elem for elem in lineval_files if '/multi/' not in elem]
        abslines_files=[elem for elem in abslines_files if '/multi/' not in elem]

        #some additional removals for in progress dirs
        lineval_files = [elem for elem in lineval_files if '4U_mix' not in elem]
        abslines_files = [elem for elem in abslines_files if '4U_mix' not in elem]

        lineval_files=[elem for elem in lineval_files if outdir+'_old' not in elem]
        lineval_files_multi=[elem for elem in lineval_files_multi if outdir+'_old' not in elem]

        abslines_files = [elem for elem in abslines_files if outdir+'_old' not in elem]
        abslines_files_multi = [elem for elem in abslines_files_multi if outdir+'_old' not in elem]

        if not include_full:
            lineval_files=[elem for elem in lineval_files if '_full' not in elem]
            lineval_files_multi=[elem for elem in lineval_files_multi if '_full' not in elem]

            abslines_files=[elem for elem in abslines_files if '_full' not in elem]
            abslines_files_multi=[elem for elem in abslines_files_multi if '_full' not in elem]

        if not include_untested:
            lineval_files=[elem for elem in lineval_files if '/untested/' not in elem]
            lineval_files_multi=[elem for elem in lineval_files_multi if '/untested/' not in elem]

            abslines_files=[elem for elem in abslines_files if '/untested/' not in elem]
            abslines_files_multi=[elem for elem in abslines_files_multi if '/untested/' not in elem]


        if multi_obj:
            obj_list=np.unique(np.array([elem.split('/')[-4] for elem in lineval_files]))
        else:
            obj_list=np.array([obj_name])

        #loading integral elements
        lc_int_sw_dict,fit_int_revol_dict=load_integral()
            
        #note: there's no need to order anymore since the file values are attributed for each object of object list in the visual_line functions

        catal_blackcat, catal_watchdog, \
            catal_blackcat_obj, catal_watchdog_obj, \
            catal_maxi_df, catal_maxi_simbad, \
            catal_bat_df, catal_bat_simbad = load_catalogs()

        catal_arr=np.array([catal_blackcat,catal_watchdog,catal_blackcat_obj,catal_watchdog_obj,
                            catal_maxi_df,catal_maxi_simbad,catal_bat_df,catal_bat_simbad],dtype=object)

        with open(dump_path_catal,'wb+') as file:
            dill.dump(catal_arr,file=file)

        #creating the dictionnary for all of the arguments to pass to the visualisation functions
        dict_linevis={
            'ctl_blackcat':catal_blackcat,
            'ctl_blackcat_obj':catal_blackcat_obj,
            'ctl_watchdog':catal_watchdog,
            'ctl_watchdog_obj':catal_watchdog_obj,
            #note that this one is not used in the first array functions so it doesn't matter for lineval_files_multi
            'lineval_files':lineval_files,
            'obj_list':obj_list,
            'cameras':cameras,
            'expmodes':expmodes,
            'multi_obj':multi_obj,
            'range_absline':range_absline,
            'n_infos':n_infos,
            'args_cam':args.cameras,
            'line_search_e_str':line_search_e_str,
            'args_line_search_norm':args.line_search_norm,
            'visual_line':True,
            }
        
        #### main arrays computation
        
        #getting the single parameters
        dist_obj_list,mass_obj_list=dist_mass(dict_linevis,use_unsure_mass_dist=use_unsure_mass_dist)

        #Reading the results files
        observ_list,lineval_list,lum_list,date_list,instru_list,exptime_list,\
            fitmod_broadband_list,epoch_obs_list,flux_high_list=obj_values(lineval_files,dict_linevis)

        #loading the interesting multi infos
        temp,temp,temp,date_list_multi,temp,temp,fitmod_broadband_list_multi,epoch_obs_list_multi,flux_high_list_multi=\
            obj_values(lineval_files_multi,dict_linevis)

        #the values here are for each observation
        abslines_infos,autofit_infos=abslines_values(abslines_files,dict_linevis)
        
        dump_dict={}
        
        from visual_line_tools import dict_lc_rxte
            
        dump_dict['instru_list']=instru_list
        dump_dict['telescope_list']=telescope_list
        dump_dict['choice_telescope']=choice_telescope
        dump_dict['dist_obj_list']=dist_obj_list
        dump_dict['mass_obj_list']=mass_obj_list
        dump_dict['line_search_e']=line_search_e
        dump_dict['multi_obj']=multi_obj
        dump_dict['observ_list']=observ_list
        dump_dict['bad_flags']=bad_flags
        dump_dict['obj_list']=obj_list
        dump_dict['date_list']=date_list

        dump_dict['exptime_list']=exptime_list
        dump_dict['abslines_infos']=abslines_infos
        dump_dict['autofit_infos']=autofit_infos
        dump_dict['lum_list']=lum_list
        dump_dict['dict_linevis']=dict_linevis

        dump_dict['dict_lc_rxte']=dict_lc_rxte
        dump_dict['lc_int_sw_dict']=lc_int_sw_dict
        dump_dict['fit_int_revol_dict']=fit_int_revol_dict

        dump_dict['fitmod_broadband_list']=fitmod_broadband_list
        dump_dict['epoch_obs_list']=epoch_obs_list

        dump_dict['date_list_multi']=date_list_multi
        dump_dict['fitmod_broadband_list_multi']=fitmod_broadband_list_multi
        dump_dict['epoch_obs_list_multi']=epoch_obs_list_multi
        dump_dict['flux_high_list']=flux_high_list
        dump_dict['flux_high_list_multi']=flux_high_list_multi

        with open(dump_path,'wb+') as dump_file:
            dill.dump(dump_dict,file=dump_file)
   
with open(dump_path,'rb') as dump_file:
    dump_dict=dill.load(dump_file)
    
instru_list=dump_dict['instru_list']
telescope_list=dump_dict['telescope_list']
choice_telescope=dump_dict['choice_telescope']
dist_obj_list=dump_dict['dist_obj_list']
mass_obj_list=dump_dict['mass_obj_list']
line_search_e=dump_dict['line_search_e']
multi_obj=dump_dict['multi_obj']
observ_list=dump_dict['observ_list']
bad_flags=dump_dict['bad_flags']
obj_list=dump_dict['obj_list']
date_list=dump_dict['date_list']

abslines_infos=dump_dict['abslines_infos']
autofit_infos=dump_dict['autofit_infos']
lum_list=dump_dict['lum_list']
dict_linevis=dump_dict['dict_linevis']
exptime_list=dump_dict['exptime_list']

container_date_change=st.container(border=True)

tab_hid, tab_monitoring, tab_param,tab_source_df,tab_about=\
    st.tabs(["Hardness Luminosity Diagram","Monitoring", "Parameter analysis","Tables","About"])


try:
    catal_blackcat, catal_watchdog, \
        catal_blackcat_obj, catal_watchdog_obj, \
        catal_maxi_df, catal_maxi_simbad, \
        catal_bat_df, catal_bat_simbad = load_catalogs()
except:
    st.warning('Catalog Loading failed. Loading from dump...')

    with open(dump_path_catal, 'rb') as file:
        catal_blackcat, catal_watchdog, \
            catal_blackcat_obj, catal_watchdog_obj, \
            catal_maxi_df, catal_maxi_simbad, \
            catal_bat_df, catal_bat_simbad = dill.load(file)

dict_lc_rxte=dump_dict['dict_lc_rxte']

lc_int_sw_dict=dump_dict['lc_int_sw_dict']
fit_int_revol_dict=dump_dict['fit_int_revol_dict']

fitmod_broadband_list=dump_dict['fitmod_broadband_list']
epoch_obs_list=dump_dict['epoch_obs_list']

date_list_multi=dump_dict['date_list_multi']
fitmod_broadband_list_multi=dump_dict['fitmod_broadband_list_multi']
epoch_obs_list_multi=dump_dict['epoch_obs_list_multi']

flux_high_list=dump_dict['flux_high_list']
flux_high_list_multi=dump_dict['flux_high_list_multi']


#creating some secondary lists from fitmod_broadband
gamma_nthcomp_list=np.array([np.array([np.repeat(np.nan,3) if\
                            (subelem is None or 'disk_nthcomp' not in list(subelem.keys())) else\
                                np.array(subelem['disk_nthcomp'][0])\
             for subelem in elem]) for elem in fitmod_broadband_list],dtype=object)

gamma_nthcomp_list_multi=np.array([np.array([np.repeat(np.nan,3) if\
                            (subelem is None or 'disk_nthcomp' not in list(subelem.keys())) else\
                                np.array(subelem['disk_nthcomp'][0])\
             for subelem in elem]) for elem in fitmod_broadband_list_multi],dtype=object)

Tin_diskbb_list=np.array([np.array([np.repeat(np.nan,3) if\
                            (subelem is None or 'cont_diskbb' not in list(subelem.keys())) else\
                                np.array(subelem['cont_diskbb'][0])\
             for subelem in elem]) for elem in fitmod_broadband_list],dtype=object)

Tin_diskbb_list_multi=np.array([np.array([np.repeat(np.nan,3) if\
                            (subelem is None or 'cont_diskbb' not in list(subelem.keys())) else\
                                np.array(subelem['cont_diskbb'][0])\
             for subelem in elem]) for elem in fitmod_broadband_list_multi],dtype=object)

'''
# in the abslines_infos_perline form, the order is:
#     -each habsorption line
#     -the number of sources
#     -the number of obs for each source
#     -the info (5 rows, EW/bshift/Del-C/sign)
#     -it's uncertainty (3 rows, main value/neg uncert/pos uncert,useless for the Del-C and sign)
'''

#checking if the obsid identifiers of every index is in the bad flag list or if there's just no file
if len(observ_list.ravel())==0:
    st.write('\nNo line detection to build HID graph.')

#some naming variables for the files
save_dir='glob_batch' if multi_obj else outdir

if multi_obj:
    save_str_prefix=''
else:
    save_str_prefix=obj_list[0]+'_'

'''
#Page creation
'''
#### Streamlit page creation

line_display_str=np.array([r'FeXXV Ka (6.70 keV)',r'FeXXVI  Ka (6.97 keV)','NiXXVII Ka (7.80 keV)',
                      'FeXXV Kb (7.89 keV)','FeXXVI Kb (8.25 keV)','FeXXVI Kg (8.70 keV)'])

if multi_obj:
    radio_single=st.sidebar.radio('Display options:',('All Objects','Multiple Objects','Single Object')
                                  ,index=2 if 'NuSTAR' in choice_telescope or 'NICER' in choice_telescope or 'Suzaku'\
                                   in choice_telescope else 0)
    
    if radio_single=='Single Object':
        display_single=True
    else:
        display_single=False
    
    if radio_single=='Multiple Objects':
        display_multi=True
    else:
        display_multi=False
        
    if display_multi:
        restrict_sources_detection=st.sidebar.toggle('Restrict to sources with significant detection')
        ####source with det restriction done manually as of now, should be changed
        
    if display_multi:
        with st.sidebar.expander('Source'):
            choice_source=st.multiselect('',options=[elem for elem in obj_list if elem in sources_det_dic] if restrict_sources_detection else obj_list,default=[elem for elem in obj_list if elem in sources_det_dic] if restrict_sources_detection else obj_list)     

    if display_single:
        #switching to array to keep the same syntax later on
        choice_source=[st.sidebar.selectbox('Source',obj_list,index=0 if '4U1630-47' not in obj_list else\
                                                              int(np.argwhere(obj_list=='4U1630-47')[0][0]))]

    #first mask here to limit the display of individual obs to exclude
    if display_single or display_multi:
        mask_obj_select = np.array([elem in choice_source for elem in obj_list])
    else:
        mask_obj_select = np.repeat(True, len(obj_list))

    obj_list_select_id=np.argwhere(mask_obj_select).T[0]

    with st.sidebar.expander('Observation'):

        obs_list_str=np.array([np.array([obj_list[i]+'_'+observ_list[i][j].replace('_-1','').replace('_auto','')\
                               for j in range(len(observ_list[i]))]) for i in obj_list_select_id],dtype=object)

        sorted_choice_obs=ravel_ragged(obs_list_str)
        sorted_choice_obs.sort()

        choice_obs_restrict = st.multiselect('Restrict to a given number of observations:', sorted_choice_obs,
                                             default=None)

        restrict_Ledd=st.toggle(r'Restrict [3-10] keV $L_{Edd}$ fraction',value=False)

        if restrict_Ledd:
            restrict_Ledd_low=st.number_input('minimum allowed Eddington fraction',value=0.,format='%.3e')
            restrict_Ledd_high=st.number_input('maximum allowed Eddington fraction',value=0.,format='%.3e')
        else:
            restrict_Ledd_low=0.
            restrict_Ledd_high=0.

        restrict_HR=st.toggle(r'Restrict Hardnes Ratio',value=False)

        dict_linevis['restrict_Ledd_low']=restrict_Ledd_low
        dict_linevis['restrict_Ledd_high']=restrict_Ledd_high

        if restrict_HR:
            restrict_HR_band=st.radio('Hardness Ratio bands',('[6-10]/[3-10]','[15-50]/[3-6]'))
            restrict_HR_low=st.number_input('minimum allowed HR',value=0.,format='%.3e')
            restrict_HR_high=st.number_input('maximum allowed HR',value=0.,format='%.3e')
        else:
            restrict_HR_band='[6-10]/[3-10]'
            restrict_HR_low=0.
            restrict_HR_high=0.

        dict_linevis['restrict_HR_low']=restrict_HR_low
        dict_linevis['restrict_HR_high']=restrict_HR_high

        # directly removing all other objects is complicated for the array transpotions, so instead we
        # simply mask all objects except for the ones in which the obs_restrict are selected, and then
        # mask every other observatiosn in these

        obj_included_obs_restrict = np.unique([elem.split('_')[0] for elem in choice_obs_restrict])
        mask_obj_select_obs_restrict=np.array([elem in obj_included_obs_restrict for elem in obj_list])

        try:
            mask_obj_select=mask_obj_select & mask_obj_select_obs_restrict
        except:
            breakpoint()

        restrict_match_INT=st.toggle('Restrict to Observations with INTEGRAL coverage',value=False)

        choice_obs=st.multiselect('Exclude individual observations:',sorted_choice_obs,
                                  default=None if '4U1630-47_405051010_xis1' not in sorted_choice_obs else ['4U1630-47_405051010_xis1'])


#masking for restriction to single objects
if display_single or display_multi:
    mask_obj_select=np.array([elem in choice_source for elem in obj_list])
else:
    mask_obj_select=np.repeat(True,len(obj_list))

####Nickel display is turned off here
if not BID_mode:
    with st.sidebar.expander('Absorption lines restriction'):
        selectbox_abstype=st.multiselect('',
                    options=line_display_str[:2].tolist()+line_display_str[3:].tolist(),default=line_display_str[:2])
else:
    selectbox_abstype=line_display_str[:2].tolist()+line_display_str[3:].tolist()

#creating the line mask from that
mask_lines=np.array([elem in selectbox_abstype for elem in line_display_str])

with st.sidebar.expander('Inclination'):

    #these two should match
    inclination_options = np.array(['dynamical', 'jet', 'misc', 'reflection'])
    inclination_dict_list = [incl_dyn_dict, incl_jet_dict, incl_misc_dict, incl_refl_dict]

    inclination_prio = np.array(st.multiselect('Inclination methods and priority', options=inclination_options,
                                      default=['dynamical', 'jet', 'misc', 'reflection']))
    use_inclination_unsure = st.toggle('Use unsure inclination estimates', value=True)

    slider_inclin=st.slider('Inclination restriction (°)',min_value=0.,max_value=90.,step=0.5,value=[0.,90.])

    include_noinclin=st.toggle('Include Sources with no inclination information',value=True)
    
    incl_inside=st.toggle('Only include sources with uncertainties strictly compatible with the current limits',value=False)
    
    display_incl_inside=st.toggle('Display ULs differently for sources with uncertainties not strictly compatible with the current limits',value=False)
    
    dash_noincl=st.toggle('Display ULs differently for sources with no inclination information',value=False)
    
    radio_dipper=st.radio('Dipping sources restriction',('Off','Add dippers','Restrict to dippers','Restrict to non-dippers'))
    
# first a list of the dictionnaries selected, ordered according to the selection order
inclination_dict_list_use = [inclination_dict_list[np.argwhere(inclination_options == elem)[0][0]] for elem in
                             inclination_prio]

# then creating a global dictionnary from this order:
incl_dict_use = {}

for elem_dict in inclination_dict_list_use:
    for elem_source in list(elem_dict.keys()):
        # adding the elements not already in the global dictionnary
        # with an option according to the surety option selected
        if elem_source not in list(incl_dict_use.keys()) and (elem_dict[elem_source][3] == 1 or use_inclination_unsure):
            incl_dict_use[elem_source] = elem_dict[elem_source][:-1]

dict_linevis['incl_dict_use'] = incl_dict_use

'''
#SOURCE TABLE
'''

#The Eddington factor is modifiable
# distance factor for the flux conversion later on
dist_factor = 4 * np.pi * (dist_obj_list * 1e3 * 3.086e18) ** 2

# L_Edd unit factor
Edd_factor = dist_factor / (1.26e38 * mass_obj_list)

#creating a full, placeholder incl_to show in the graph
# computing an array of the object inclinations
incl_plot = np.array(
    [[np.nan, np.nan, np.nan] if elem not in incl_dict_use else incl_dict_use[elem] for elem in obj_list])

source_df_arr = np.array(
    [obj_list, dist_obj_list, mass_obj_list, Edd_factor, incl_plot.T[0], incl_plot.T[1], incl_plot.T[2],
     [sum(elem == 'XMM') for elem in instru_list], [sum(elem == 'Chandra') for elem in instru_list]]).astype(str).T

source_df = pd.DataFrame(source_df_arr, columns=['source', 'distance (kpc)', 'mass (M_sun)', 'Eddington ratio factor',
                                                 'inclination (°)', 'incl err -', 'incl err +',
                                                 'XMM exposures', 'Chandra exposures'])

source_df['Eddington ratio factor'] = source_df['Eddington ratio factor'].astype(float).round(3)

#replacing the lines of the new source_df_arr (to keep modified parameters if any)
# and then replacing it in the session_state

if "source_df" in st.session_state:
    for i_source,elem_source in enumerate(st.session_state['source_df']['source']):
        if elem_source in obj_list:
            i_new_table_source=np.argwhere(obj_list==elem_source)[0][0]
            source_df.iloc[i_new_table_source]=st.session_state['source_df'].iloc[i_source]


    st.session_state['source_df']=source_df

#specific command from https://discuss.streamlit.io/t/experimental-data-editor-column-basic-calculation/39837/10
#to dynamically recompute the Edd_factor column as a function of the distance and mass columns

def add_c(new_df: pd.DataFrame | None = None):
    # breakpoint()

    if new_df is not None:
        if new_df.equals(st.session_state["source_df"]):
            return
        st.session_state["source_df"] = new_df

    df = st.session_state["source_df"]
    Edd_ratio_factor = 4*np.pi*(df["distance (kpc)"].astype(float)*1e3 * 3.086e18)**2 / (1.26e38*df["mass (M_sun)"].astype(float))
    #avoids rounding issues
    Edd_ratio_factor=Edd_ratio_factor.round(3)
    df['Eddington ratio factor']=Edd_ratio_factor
    st.session_state["source_df"] = df

    st.rerun()

if "source_df" not in st.session_state:
    st.session_state.source_df = source_df
    add_c()

with tab_source_df:
    from streamlit_pdf_viewer import pdf_viewer


    with st.expander('Source parameters'):


        tab_pdf, tab_csv = st.tabs(["Full BH candidates table", "Current Sample editable array"])

        with tab_pdf:
            with st.spinner('Loading PDF'):
                pdf_viewer(os.path.join(project_dir, 'observations/visualisation/visual_line_dumps/sources_tables.pdf'),
                       pages_to_render=[1, 2, 3, 4, 7, 8], width=1400,render_text=True)

            # # st.notice('This table is similar to what was last published in ')
            #
            #
            #
            # ####PREPARING LATEX TABLES
            # source_table_tex = os.path.join(project_dir, 'observations/visualisation/source_tables.tex')
            #
            # with open(source_table_tex) as tex_file:
            #     tex_lines = tex_file.readlines()
            #
            # tex_str = ''.join(tex_lines)
            #
            # # first table
            # source_table_str = tex_str[tex_str.find('label{table:sources}'):tex_str.find('\end{table*}')]
            #
            # # second table
            # state_table_str = tex_str[tex_str.find('end{table*}') + 10:]
            # state_table_str = state_table_str[
            #                   state_table_str.find('label{table:sources_det_states}'):state_table_str.find(
            #                       '\end{table*}')]
            #
            # # taking the main element of each table
            # source_table_latex = source_table_str[
            #                      source_table_str.find('1E 1740.7'):source_table_str.find('\end{tabular}')]
            #
            # state_table_latex = state_table_str[
            #                     state_table_str.find('4U 1543-47}') - 10:state_table_str.find('\end{tabular}')]
            #
            # # replacing all of the commands that do not work, and removing math text because we're already in math mode
            # source_table_latex = source_table_latex.replace('\T', '').replace('\B', '').replace('$', '').replace(
            #     '\\textbf{dips}',
            #     '\\bm{dips}')
            #
            # state_table_latex = state_table_latex.replace('\T', '').replace('\B', '').replace('$', '') \
            #     .replace('\\textbf{', '\\bm{').replace('\\textit{', '\\mathit{').replace('*', '^*')
            #
            # # fetching the occurences of refences
            # source_table_refs_str = np.unique(re.findall('labelcref{ref_source:.*?}', source_table_latex))
            # state_table_refs_str = np.unique(re.findall('labelcref{ref_source_state:.*?}', state_table_latex))
            #
            # # and replacing with a fake reference in the latex text itself
            # for i_ref, ref_str in enumerate(source_table_refs_str):
            #     source_table_latex = source_table_latex.replace("\\" + ref_str, '\\textcolor{RoyalBlue}{\\bm{[' + str(
            #         i_ref + 1) + ']}}')
            #
            # for i_ref, ref_str in enumerate(state_table_refs_str):
            #     state_table_latex = state_table_latex.replace("\\" + ref_str, '\\textcolor{RoyalBlue}{\\bm{[' + str(
            #         i_ref + 1) + ']}}')
            #
            # # needs to be done after to avoid issues with } moving away in the loop before
            # state_table_latex = state_table_latex.replace('soft', '\\text{soft}').replace('hard', '\\text{hard}') \
            #     .replace('obscured', '\\text{obscured}')
            #
            # # and after to reset to a good state to have the italics we want
            # state_table_latex = state_table_latex.replace('\\mathit{\\text{soft}}', '\\textcolor{Goldenrod}{soft}') \
            #     .replace('\\mathit{\\text{hard}}', '\\textcolor{Goldenrod}{hard}')
            #
            # # using the #''' to avoid issues when switching to online
            # source_table_header = r'''
            #         \def\arraystretch{2.5}
            #         \begin{array}{c|c|c|c|c|ccc}
            #         \hline
            #         \hline
            #                \textrm{Name}
            #              & \textrm{mass} (M_\odot)
            #              & \textrm{distance} (kpc)
            #              & \textrm{inclination} (°)
            #              & \textrm{absorption lines}
            #              & \textrm{exposures in }
            #              &\textrm{the sample}
            #                 \\
            #
            #              &
            #              &
            #              &
            #              & \textrm{reported in the iron band}
            #              & \textrm{EPIC PN}
            #              & \textrm{HETG}
            #                 \\
            #         \hline
            #         \hline
            #         '''  #
            #
            # state_table_header = r'''
            #         \def\arraystretch{2.5}
            #         \begin{array}{c || c || c | c }
            #
            #         \hline
            #         \hline
            #              \textrm{Source}
            #              & \textrm{accretion states}
            #              & \textrm{with absorption lines reported}
            #              \\
            #
            #         \hline
            #
            #              & \textrm{this work}
            #              & \textrm{other works}
            #              & \textrm{other works}\\
            #
            #         \hline
            #              & \textrm{iron band}
            #              & \textrm{iron band}
            #              & \textrm{other energies}
            #              \\
            #         \hline
            #         \hline
            #         '''  #
            #
            # table_footer = r'''\end{array}'''  #
            #
            # # this allows to replace single line jumps into double line jumps with a horizontal dotted line in between
            # # for better visibility
            # source_table_latex = source_table_latex.replace('\\\\', '\\\\\\hdashline')
            # state_table_latex = state_table_latex.replace('\\\\', '\\\\\\hdashline')
            #
            # source_table_latex_disp = source_table_header + source_table_latex + table_footer
            # state_table_latex_disp = state_table_header + state_table_latex + table_footer
            #
            # source_table_footnotes = source_table_str[
            #                          source_table_str.find('Notes'):source_table_str.find('References') - 36]
            #
            # state_table_footnotes = state_table_str[
            #                         state_table_str.find('Notes'):state_table_str.find('References')]
            #
            # source_table_footnotes += 'the second table below'
            # source_table_footnotes = source_table_footnotes.replace('\citealt{Corral-Santana2016_blackcat}',
            #                                                         '[Corral-Santana et al. 2016](https://doi.org/10.1051/0004-6361/201527130)') \
            #     .replace('\\msun{}', '$M_{\odot}$')
            #
            # state_table_footnotes = state_table_footnotes.replace('italic', '\\textcolor{Goldenrod}{italic}')
            #
            # st.markdown(source_table_footnotes)
            # st.latex(source_table_latex_disp)
            #
            # # the second one will be displayed later
            #
            # # REFERENCES
            #
            # # fetching the actual paper names in the bib files from the references part of the table
            # source_table_references = source_table_str[source_table_str.find('References'):].split('citep[][]{')
            # state_table_references = state_table_str[state_table_str.find('References'):].split('citep[][]{')
            #
            # # matching with the surnames of the table
            # source_table_refs_bibid = [[elem.split('}')[0] for elem in source_table_references \
            #                             if elem_source_ref.replace('labelcref', 'label') in elem] \
            #                            for elem_source_ref in source_table_refs_str]
            #
            # state_table_refs_bibid = [[elem.split('}')[0] for elem in state_table_references \
            #                            if elem_state_ref.replace('labelcref', 'label') in elem] \
            #                           for elem_state_ref in state_table_refs_str]
            #
            # # checking there's no reference missing
            # if not np.all([len(elem) == 1 for elem in source_table_refs_bibid + state_table_refs_bibid]):
            #     st.error('Issue during bibliography reference matching')
            #
            # # just removing the useless dimension
            # source_table_refs_bibid = ravel_ragged(source_table_refs_bibid)
            # state_table_refs_bibid = ravel_ragged(state_table_refs_bibid)
            #
            # # loading the bib file
            # bib_path = os.path.join(project_dir, 'observations/visualisation/bib_source_tables.bib')
            #
            # with open(bib_path) as bib_file:
            #     bib_lines = bib_file.readlines()
            #
            # # and transforming it into something more useful
            # bib_str = ''.join(bib_lines)
            #
            # bib_list = bib_str.split('@')[1:]
            #
            # # getting the list of lines of biblio of each matching item
            # source_bib_list = [[elem.split('\n') for elem in bib_list if elem_bib_item + ',' in elem.split('\n')[0]] \
            #                    for elem_bib_item in source_table_refs_bibid]
            #
            # state_bib_list = [[elem.split('\n') for elem in bib_list if elem_bib_item + ',' in elem.split('\n')[0]] \
            #                   for elem_bib_item in state_table_refs_bibid]
            #
            # if not np.all([len(elem) == 1 for elem in source_bib_list + state_bib_list]):
            #     st.error('Issue during bibliography reference matching')
            #
            # # removing the useless dimension
            # source_bib_list = [elem[0] for elem in source_bib_list]
            # state_bib_list = [elem[0] for elem in state_bib_list]
            #
            # # assigning a link line to each element (by preference a doi, else an url, else an arxiv id url
            # # we also format the link line into an actual link for each link type
            # source_bib_urls = [['https://doi.org/' + elem[elem.find('{') + 1:elem.find('}')] for elem in elem_bib if
            #                     elem.startswith('doi =')] + \
            #                    [elem[elem.find('{') + 1:elem.find('}')].split()[0] for elem in elem_bib if
            #                     elem.startswith('url =')] + \
            #                    ['https://arxiv.org/abs/' + elem[elem.find('{') + 1:elem.find('}')] \
            #                     for elem in elem_bib if elem.startswith('arxivId =')] for elem_bib in source_bib_list]
            #
            # state_bib_urls = [['https://doi.org/' + elem[elem.find('{') + 1:elem.find('}')] for elem in elem_bib if
            #                    elem.startswith('doi =')] + \
            #                   [elem[elem.find('{') + 1:elem.find('}')].split()[0] for elem in elem_bib if
            #                    elem.startswith('url =')] + \
            #                   ['https://arxiv.org/abs/' + elem[elem.find('{') + 1:elem.find('}')] \
            #                    for elem in elem_bib if elem.startswith('arxivId =')] for elem_bib in state_bib_list]
            #
            # if not np.all([len(elem) >= 1 for elem in source_bib_urls + state_bib_urls]):
            #     st.error('At least one bibliography reference used in the tables has no URL')
            #
            # # selecting the "favorite" available url for each link
            # source_bib_urls = [elem[0] for elem in source_bib_urls]
            # state_bib_urls = [elem[0] for elem in state_bib_urls]
            #
            # # and finally displaying the biblio
            # source_table_biblio_str = '\n'.join(
            #     ['[[' + str(i_url + 1) + ']](' + source_bib_urls[i_url] + ')' for i_url in range(len(source_bib_urls))])
            #
            # state_table_biblio_str = '\n'.join(
            #     ['[[' + str(i_url + 1) + ']](' + state_bib_urls[i_url] + ')' for i_url in range(len(state_bib_urls))])
            #
            # st.markdown('References:')
            # st.markdown(source_table_biblio_str)

        with tab_csv:
            # st.dataframe(source_df,
            #              column_config={"Eddington ratio Factor": st.column_config.NumberColumn(format='%.3e')})
            #
            # csv_source = convert_df(source_df)

            st.info('Changing the Mass and Distance will affect the HLDs and other plots. The Eddington ratio factor is locked '
                    'and computed automatically as a function of the mass and distance')
            #the editable dataframe
            source_df_updated = st.data_editor(
                st.session_state["source_df"],
                key="data",
                column_config={"Eddington ratio factor": st.column_config.NumberColumn(format='%.3e', disabled=True)},
                hide_index=True,use_container_width=True)

            #avoids rounding issues
            source_df_updated['Eddington ratio factor']=source_df_updated['Eddington ratio factor'].astype(float).round(3)

            add_c(source_df_updated)

        csv_source=convert_df(source_df_updated)
        st.download_button(
            label="Download current sample array as CSV",
            data=csv_source,
            file_name='source_table.csv',
            mime='text/csv',
        )

#replacing the values in case this is important
dist_obj_list, mass_obj_list, Edd_factor=\
    source_df_updated[source_df_updated.keys()[[1,2,3]]].values.astype(float).T

dist_factor = 4 * np.pi * (dist_obj_list * 1e3 * 3.086e18) ** 2


#updating individually the luminosity components with the selected Eddington factor
for i_obj in range(len(Edd_factor)):

    lum_list[i_obj]*=Edd_factor[i_obj]

#not used currently
# radio_dispmode=st.sidebar.radio('HID display',('Autofit line detections','blind search peak detections'))
# if radio_dispmode=='Autofit line detections':
#     display_final=True
# else:
#     display_final=False

#     radio_info_cmap=st.sidebar.radio('Color map options:',('Source','Peak Del-C'))
#     slider_ener=st.sidebar.slider('Peak energy range',min_value=line_search_e[0],max_value=line_search_e[1],
#                                   step=line_search_e[2],value=[6.,9.])
#     display_abslines=st.sidebar.toggle('Display absorption Lines')
#     display_emlines=st.sidebar.toggle('Display emission Lines')
    
restrict_time=st.sidebar.toggle('Restrict time interval',value=True)
        
slider_sign=st.sidebar.slider('Detection significance treshold',min_value=0.9,max_value=1.,step=1e-3,value=0.997,format="%.3f")

replace_high_e_multi=st.sidebar.toggle('Use multi telescope fits for high-energy infos (whenever possible)',value=True)

add_BAT_flux_corr = st.sidebar.toggle('Use corrected BAT flux when lacking high-energy coverage', value=True)

if add_BAT_flux_corr:
    radio_BAT_binning_scat = st.sidebar.radio('BAT flux sampling', ('daily', 'single orbit'),index=0 if use_obsids else 1)

    BAT_binning_scat = 'day' if radio_BAT_binning_scat == 'daily' \
        else 'orbit' if radio_BAT_binning_scat == 'single orbit' else None
else:
    BAT_binning_scat = 'day'

add_INT_flux_corr = st.sidebar.toggle('Use projected INTEGRAL flux when lacking high-energy coverage',value=True)

if add_INT_flux_corr:
    INT_binning_scat = st.sidebar.radio('INTEGRAL flux sampling', options=['revolution'])
else:
    INT_binning_scat = 'revolution'

if replace_high_e_multi:
    for i_obj in range(len(epoch_obs_list)):
        for i_epoch,indiv_epoch in enumerate(epoch_obs_list[i_obj]):
            #searching for multi epochs with at least one matching observation within that epoch
            id_match_multi=[i_epoch_multi for i_epoch_multi in range(len(epoch_obs_list_multi[i_obj]))\
                              if np.any([elem in epoch_obs_list_multi[i_obj][i_epoch_multi]\
                                         for elem in epoch_obs_list[i_obj][i_epoch]])]

            #skipping elements with no match
            if len(id_match_multi)==0:
                continue

            #ensuring there's a single match
            assert len(id_match_multi)==1,'Issue: multiple multi epoch match found for epoch '\
                                            +str(epoch_obs_list[i_obj][i_epoch])

            #and replacing the informations of each high-energy array by the corresponding multi-telescope obs array
            flux_high_list[i_obj][i_epoch]=flux_high_list_multi[i_obj][id_match_multi[0]]
            gamma_nthcomp_list[i_obj][i_epoch]=gamma_nthcomp_list_multi[i_obj][id_match_multi[0]]
            Tin_diskbb_list[i_obj][i_epoch]=Tin_diskbb_list_multi[i_obj][id_match_multi[0]]


#creating the mask for the observation restriction (needs ot be done after editing flux_high_list for the HR cut
mask_included_selection = np.array([np.array([ \
\
    # obs to remove
    obj_list[i] + '_' + observ_list[i][j].replace('_-1', '').replace('_auto', '') \
    not in choice_obs and \
\
    # obs to restrict
    (True if len(choice_obs_restrict) == 0 or \
             obj_list[i] not in obj_included_obs_restrict else \
         obj_list[i] + '_' + observ_list[i][j].replace('_-1', '').replace('_auto', '') \
         in choice_obs_restrict) and \
\
        # luminosity bounds
    (True if restrict_Ledd_low == 0. else lum_list[i][j][0][4] >= restrict_Ledd_low) and \
    (True if restrict_Ledd_high == 0. else lum_list[i][j][0][4] <= restrict_Ledd_high) and \
\
    # HR bounds
    (True if restrict_HR_low == 0. else \
         (lum_list[i][j][0][2]/lum_list[i][j][0][1] >=restrict_HR_low) if restrict_HR_band=='[6-10]/[3-10]' else
         (flux_high_list[i][j][0]/lum_list[i][j][0][1] >=restrict_HR_low) if restrict_HR_band=='[15-50]/[3-10]' else True) and \
 \
    (True if restrict_HR_high == 0. else \
         (lum_list[i][j][0][2] / lum_list[i][j][0][1] <=restrict_HR_high) if restrict_HR_band == '[6-10]/[3-10]' else
          (flux_high_list[i][j][0] / lum_list[i][j][0][1] <=restrict_HR_high) if restrict_HR_band == '[15-50]/[3-10]' else True) \
 \
    for j in range(len(observ_list[i]))]) for i in range(len(obj_list))], dtype=object)

mask_obj_select_observ=np.array([elem.any() for elem in mask_included_selection])
mask_obj_select=(mask_obj_select) & (mask_obj_select_observ)

#reselecting the elements inside if we mask the entire object anyway, to avoid the array transposition going to shit
for i_obj,elem in enumerate(mask_obj_select_observ):
    if not elem:
        mask_included_selection[i_obj]=np.repeat(True,len(mask_included_selection[i_obj]))

if len(mask_included_selection) == 1:
    # this is necessary to avoid issues when choosing a single telescope with a single object
    mask_included_selection = mask_included_selection.astype(bool)

# creating the global inclination dictionnary depending on what options have been selected


'''
#RESTRICTION MASKING AND TRANSPOSING
'''

n_obj_init=len(obj_list)

# getting all the variations we need
abslines_infos_perline, abslines_infos_perobj, abslines_plot, abslines_ener, \
    lum_plot, hid_plot, width_plot, nh_plot, kt_plot,lum_list = values_manip(abslines_infos, dict_linevis,
                                                                                autofit_infos,
                                                                                lum_list,
                                                                                mask_include=mask_included_selection)



instru_list = np.array([instru_list[i_obj][mask_included_selection[i_obj]] for i_obj in range(n_obj_init)], dtype=object)
observ_list = np.array([observ_list[i_obj][mask_included_selection[i_obj]] for i_obj in range(n_obj_init)], dtype=object)
date_list = np.array([date_list[i_obj][mask_included_selection[i_obj]] for i_obj in range(n_obj_init)], dtype=object)

####Streamlit HID options
st.sidebar.header('HID options')

#not used right now
# else:
#     #full true mask
#     mask_lines=np.array(line_display_str!='')
            

if online:
    display_nonsign=False
else:
    display_nonsign=st.sidebar.toggle('Show detections below significance threshold',value=False)

if display_nonsign:
    restrict_threshold=st.sidebar.toggle('Prioritize showing maximal values of significant detections',value=True)
else:
    restrict_threshold=True
        
HID_options_str=np.array(['Source','Inclination','Instrument','Time',
                          r'line $V_{shift}$',r'line $\Delta$C',r'line $EW$ ratio',
                          r'$nH$',r'$T_{in}$'] \
                          + (['Custom: Line substructures', 'Custom: accretion states','Custom: ionization'] if \
                                 display_single and obj_list[mask_obj_select][0] == '4U1630-47' else []) \
                          + (['Custom: Outbursts'] if display_single else []))

radio_info_cmap_options=['Source','Inclination','Instrument','Time','Velocity shift','Del-C','EW ratio','nH','kT'] \
                        + (['custom_line_struct', 'custom_acc_states','custom_ionization'] if \
                               display_single and obj_list[mask_obj_select][0] == '4U1630-47' else []) \
                        + (['custom_outburst'] if display_single else [])

radio_info_cmap_str=st.sidebar.radio('HLD colormap',(HID_options_str[:4].tolist()+HID_options_str[7:].tolist()) if BID_mode else HID_options_str ,
                                     index=0)

radio_info_index=np.argwhere(HID_options_str==radio_info_cmap_str)[0][0]

radio_info_cmap=radio_info_cmap_options[radio_info_index]

if radio_info_cmap!='Source':
    display_edgesource=st.sidebar.toggle('Color code sources in marker edges',value=False)
else:
    display_edgesource=False

if radio_info_cmap=='Inclination':
    #order change here to have more logic in the variables
    cmap_incl_str=['Main value','Lower limit','Upper limit']
    cmap_incl_type_str=st.sidebar.radio('Inclination information to use:',cmap_incl_str)

    cmap_incl_type=[i for i in range(3) if cmap_incl_str[i]==cmap_incl_type_str][0]
else:
    cmap_inclination_limit=False
    cmap_incl_type=1
    cmap_incl_type_str=None
    
if radio_info_cmap=='EW ratio':
    with st.sidebar.expander('Lines selection for EW ratio:'):
        selectbox_ratioew=st.sidebar.multiselect('',options=line_display_str[mask_lines],default=line_display_str[mask_lines][:2])
        
else:
    selectbox_ratioew=''
    
radio_zoom_hid=st.sidebar.radio('Zoom:',('Global sample','Current selection','manual bounds'),index=0)
if radio_zoom_hid=='Global sample':
    zoom_hid=False
elif radio_zoom_hid=='Current selection':
    zoom_hid='auto'
elif radio_zoom_hid=='manual bounds':

    def format_slider(x,val_decimal=3):
        return ('%.'+str(val_decimal)+'e')%x

    values_zoom_hr=st.sidebar.select_slider('Displayed HR range',options=np.logspace(-2,1,num=100),
                                            value=[0.1,2.0092330025650478],format_func=format_slider)
    values_zoom_lum = st.sidebar.select_slider('Displayed luminosity range', options=np.logspace(-5,0,num=100),
                                        value=[1e-5, 1.],format_func=format_slider)
    zoom_hid=[values_zoom_hr,values_zoom_lum]

if BID_mode:
    display_nondet=True
else:
    display_nondet=st.sidebar.toggle('Display exposures with no detection',value=True)

if display_nondet and not BID_mode:
    with st.sidebar.expander('Upper limits'):
        display_upper=st.toggle('Display upper limits',value=True if line_search_e_str=='4 10 0.05' else False)
        if display_upper:
                selectbox_upperlines=st.multiselect('Lines selection for upper limit display:',
                                                            options=line_display_str[mask_lines],default=line_display_str[mask_lines][:2])
                mask_lines_ul=np.array([elem in selectbox_upperlines for elem in line_display_str])
        else:
            mask_lines_ul=False
else:
    display_upper=False
    mask_lines_ul=False
    
if display_single:
    display_evol_single=st.sidebar.toggle('Highlight time evolution in the HID',value=False)
else:
    display_evol_single=False
    
if not online:
    save_format=st.sidebar.radio('Graph format:',('pdf','svg','png'))
    
    def save_HID():
        '''
        # Saves the current graph in a svg (i.e. with clickable points) format.
        '''
    
        fig_hid.savefig(save_dir+'/'+save_str_prefix+'HID_cam_'+args.cameras+'_'+\
                    line_search_e_str.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'curr_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
            
    st.sidebar.button('Save current HID view',on_click=save_HID)

        
with st.sidebar.expander('Visualisation'):

    skip_HID=st.toggle('Skip HID computation',value=False)

    display_hid_error=st.toggle('Display errorbar for HID position',value=False)

    if BID_mode:
        display_dicho=False
        color_nondet=True
        display_central_abs=False
        split_cmap_source=False

    else:

        display_dicho=st.toggle('Display favourable zone',value=not display_single)

        color_nondet=st.toggle('Color non-detections',value=True)

        display_central_abs=st.toggle('Display centers for absorption detections',value=False)

        split_cmap_source=st.toggle('Use different colormaps for detections and non-detections',value=True)

    alpha_abs=st.toggle('Plot with transparency',value=False)

    global_colors=st.toggle('Normalize colors/colormaps over the entire sample',value=False)
        
    hid_log_HR=st.toggle('Use log scale for the Hardness Ratio',value=True)

    display_minorticks=st.toggle('Display minor ticks for the Hardness Ratio',value=False)

    if display_single and choice_source[0]=='4U1630-47':

        hatch_unstable=st.toggle('Hash unstable 2021 observations',value=False)
    else:
        hatch_unstable=False

    change_legend_position=st.toggle('Change legend position',value=False)

    if not online:
        paper_look=st.toggle('Paper look',value=False)

        bigger_text=st.toggle('Bigger text size',value=True)
        
        square_mode=st.toggle('Square mode',value=True)
    
        show_linked=st.toggle('Distinguish linked detections',value=False)
    else:
        paper_look=False
        bigger_text=True
        square_mode=True
        show_linked=False
        
if alpha_abs:
    alpha_abs=0.5
else:
    alpha_abs=1


def make_zip(filebites_arr,filename_arr):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a",
                         zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in zip(filename_arr,filebites_arr):
            zip_file.writestr(file_name, data.getvalue())

    return zip_buffer


with st.sidebar.expander('Broad band HID'):

    display_broad_hid_BAT = st.toggle('Display broad band HID (HARD BAND=15-50)', value=False)

    HR_broad_bands=st.radio('HID Hardness Ratio',('[BAND]/[3-6]','([6-10]+[BAND])/[3-6]'))
    lum_broad_bands=st.radio('HID Luminosity',('[3-10]','[3-10]+[BAND]'))

    sign_broad_hid_BAT=st.toggle('Restrict Broad HID to 2 sigma significant highE measurements',value=False)


expander_monit=st.sidebar.expander('Monitoring')
with expander_monit:
    
    plot_lc_monit=st.toggle('Plot MAXI/RXTE monitoring',value=False)
    plot_hr_monit=st.toggle('Plot monitoring HR',value=False)
    plot_lc_bat=st.toggle('Plot BAT monitoring',value=False)

    radio_monit_binning=st.radio('MAXI/BAT binning',('daily','single orbit'))
    monit_binning='day' if radio_monit_binning=='daily' else 'orbit' if radio_monit_binning=='single orbit' else None

    plot_lc_integral=st.toggle('Plot INTEGRAL monitoring',value=False)
    radio_integral_binning=st.radio('INTEGRAL binning',('revolution','science_window'))
    integral_binning='revol' if radio_integral_binning=='revolution' else 'sw'
    radio_integral_band=st.radio('INTEGRAL band',('30-100','30-50','50-100'),index=0)

    monit_highlight_hid=st.toggle('Highlight HID coverage',value=False)


    if plot_lc_monit or plot_hr_monit or plot_lc_bat or plot_lc_integral:
        zoom_lc=st.toggle('Zoom on the restricted time period in the lightcurve',value=True)
    else:
        zoom_lc=False
        
    fig_lc_monit=None
    fig_hr_soft_monit=None
    fig_hr_hard_monit=None
    fig_lc_bat=None
    fig_lc_int=None

    plot_maxi_ew=st.toggle('Superpose measured EW',value=False)

    if not online:
        def save_lc_local():

            '''
            # Saves the current maxi_graphs in a svg (i.e. with clickable points) format.
            '''
            if display_single:

                if fig_lc_monit is not None:
                    fig_lc_monit.savefig(save_dir+'/'+'LC_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
                if fig_hr_soft_monit is not None:
                    fig_hr_soft_monit.savefig(save_dir+'/'+'HR_soft_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')

                if fig_hr_hard_monit is not None:
                    fig_hr_hard_monit.savefig(save_dir+'/'+'HR_hard_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')

                if fig_lc_bat is not None:
                    fig_lc_bat.savefig(save_dir+'/'+'BAT_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,
                                       bbox_inches='tight')

                if fig_lc_int is not None:
                    fig_lc_int.savefig(save_dir+'/'+'INT_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,
                                       bbox_inches='tight')
        st.button('Save current MAXI curves',on_click=save_lc_local,key='save_lc_key')

compute_only_withdet=st.sidebar.toggle('Skip parameter analysis when no detection remain with the current constraints',value=True)

# if not online:
#     with st.sidebar.expander('Stacking'):
#         stack_det=st.toggle('Stack detections')
#         stack_flux_lim = st.number_input(r'Max ratio of fluxes to stack', value=2., min_value=1e-10, format='%.3e')
#         stack_HR_lim=st.number_input(r'Max ratio of HR to stack',value=2.,min_value=1e-10,format='%.3e')
#         stack_time_lim=st.number_input(r'Max time delta to stack',value=2.,min_value=1e-1,format='%.3e')
# else:
#     stack_det=False

mpl.rcParams.update({'font.size': 10+(3 if paper_look else 0)})

if not square_mode:
    fig_hid,ax_hid=plt.subplots(1,1,figsize=(8,5) if bigger_text else (12,6))
else:
    fig_hid,ax_hid=plt.subplots(1,1,figsize=(8,6))
ax_hid.clear()

'''
#Dichotomy
'''

#some warnings to avoid crashes
if radio_single !='All Objects' and len(choice_source)<1:
    st.warning('Please select at least one Source.')
    st.stop()

if len(selectbox_abstype)<1:
    st.warning('Please select at least one line.')
    st.stop()

#fetching the line indexes when plotting EW ratio as colormap
ew_ratio_ids=np.argwhere([elem in selectbox_ratioew for elem in line_display_str]).T[0]

if radio_info_cmap=='EW ratio' and len(ew_ratio_ids)<2:
    st.warning('Cannot build EW ratio colormap from current line restriction')
    st.stop()


#string of the colormap legend for the informations
radio_info_label=['Velocity shift', r'$\Delta-C$', 'Equivalent width ratio']


#masking the objects depending on inclination
mask_inclin=[include_noinclin if elem not in incl_dict_use else getoverlap((incl_dict_use[elem][0]-incl_dict_use[elem][1],incl_dict_use[elem][0]+incl_dict_use[elem][2]),slider_inclin)>0 for elem in obj_list]

#creating the mask for highlighting objects whose inclination limits go beyond the inclination restrictions if asked to
bool_incl_inside=np.array([False if elem not in incl_dict_use else\
                           round(getoverlap((incl_dict_use[elem][0]-incl_dict_use[elem][1],
                                 incl_dict_use[elem][0]+incl_dict_use[elem][2]),slider_inclin),3)==\
                           incl_dict_use[elem][1]+incl_dict_use[elem][2] and\
                           (incl_dict_use[elem][0]>=slider_inclin[0] and incl_dict_use[elem][0]<=slider_inclin[1])\
                           for elem in obj_list])

bool_noincl=np.array([True if elem not in incl_dict_use else False for elem in obj_list])


if incl_inside:
    mask_inclin=bool_incl_inside

#masking dippers/non dipper if asked to
mask_dippers=np.array([elem in dippers_list for elem in obj_list])

if radio_dipper=='Restrict to dippers':
    mask_inclin=(mask_inclin) & (mask_dippers)
if radio_dipper=='Add dippers':
    mask_inclin=(mask_inclin) | (mask_dippers)
elif radio_dipper=='Restrict to non-dippers':
    mask_inclin=(mask_inclin) & ~(mask_dippers)

#double mask taking into account both single/multiple display mode and the inclination

mask_obj_base=(mask_obj_select) & (mask_inclin)

#### Array restrictions

#time delta to add some leeway to the limits available and avoid directly cropping at the observations
delta_1y=TimeDelta(365,format='jd')
delta_1m=TimeDelta(30,format='jd')
delta_1w=TimeDelta(7,format='jd')
delta_1h=TimeDelta(3600,format='sec')
delta_1s=TimeDelta(1,format='sec')
if restrict_time:

    with container_date_change:

        slider_date_coarse=st.slider('Dates restriction',
                                     min_value=(Time(min(ravel_ragged(date_list[mask_obj_base])))-\
                                                             (delta_1y if use_obsids else delta_1m)).datetime,
                                     max_value=max((Time(max(ravel_ragged(date_list[mask_obj_base])))+\
                                             (delta_1y if use_obsids else delta_1m)),
                                            (Time(str(date.today())) if use_obsids else \
                                            Time(max(ravel_ragged(date_list[mask_obj_base]))) + delta_1m)).datetime,
                              value=[(Time(min(ravel_ragged(date_list[mask_obj_base])))-\
                                      (delta_1m if use_obsids else delta_1w)).datetime,
                                     (Time(max(ravel_ragged(date_list[mask_obj_base])))+ \
                                      (delta_1m if use_obsids else delta_1w)).datetime],
                              step=delta_1h.datetime,
                              format='YYYY-MM-DD HH:MM:ss')

        manual_date_vals=st.toggle('Manual Date bounds')

        if manual_date_vals:

            man_min_date_val = st.date_input('Minimum date', value=None)
            time_min_date_val=slider_date_coarse[0] if man_min_date_val is None else Time(man_min_date_val.isoformat()).datetime
            man_max_date_val = st.date_input('Maximum date', value=None)
            time_max_date_val=slider_date_coarse[1] if man_max_date_val is None else Time(man_max_date_val.isoformat()).datetime

            slider_date_coarse=np.array([time_min_date_val,time_max_date_val])

        fine_restrict_dates=st.toggle('Fine Dates restriction')


        if fine_restrict_dates:
            fine_range=slider_date_coarse[1]-slider_date_coarse[0]
            fine_delta=delta_1s.datetime*60
            fine_range_split=fine_range//fine_delta
            slider_date_fine=st.select_slider('Fine Dates restriction',
                                              options=[slider_date_coarse[0]+fine_delta*i for i in range(fine_range_split+1)],
                                       # max_value=slider_date_coarse[1],
                                       value=[slider_date_coarse[0],slider_date_coarse[1]],)
                            # step=TimeDelta(60,format='sec').datetime,
                            # format_func = 'YYYY-MM-DD HH:MM:ss')
            slider_date=slider_date_fine

        else:
            slider_date=slider_date_coarse
        # slider_date=st.slider('Dates restriction',min_value=(Time(min(ravel_ragged(date_list[mask_obj_base])))-delta_1y).datetime,
        #                       max_value=max((Time(max(ravel_ragged(date_list[mask_obj_base])))+delta_1y),
        #                                 Time(str(date.today()))).datetime,
        #                       value=[(Time(min(ravel_ragged(date_list[mask_obj_base])))-delta_1m).datetime,
        #                              (Time(max(ravel_ragged(date_list[mask_obj_base])))+delta_1m).datetime])
else:
    slider_date=[Time(min(ravel_ragged(date_list[mask_obj_base]))).datetime,
                                 Time(max(ravel_ragged(date_list[mask_obj_base]))).datetime]

#creating a mask according to the sources with observations in the current date restriction
mask_obj_intime=np.array([((np.array([Time(subelem) for subelem in elem])>=Time(slider_date[0])) &\
                  (np.array([Time(subelem) for subelem in elem])<=Time(slider_date[1]))).any() for elem in date_list])

#restricting mask_obj_base with the new base
mask_obj_base=mask_obj_base & mask_obj_intime

# #creating restricted ploting arrays witht the current streamlit object and lines selections
# abslines_plot_restrict=deepcopy(abslines_plot)
# for i_info in range(len(abslines_plot_restrict)):
#     for i_incer in range(len(abslines_plot_restrict[i_info])):
#         abslines_plot_restrict[i_info][i_incer]=abslines_plot_restrict[i_info][i_incer][mask_lines].T[mask_obj].T
    
mask_obs_intime_repeat=np.array([np.repeat(((np.array([Time(subelem) for subelem in elem])>=Time(slider_date[0])) &\
                  (np.array([Time(subelem) for subelem in elem])<=Time(slider_date[1]))),sum(mask_lines)) for elem in date_list],dtype=object)

#forcing bool types for single sources
if len(mask_obs_intime_repeat)==1:
    mask_obs_intime_repeat=mask_obs_intime_repeat.astype(bool)

#checking which sources have no detection in the current combination
global_displayed_sign=np.array([ravel_ragged(elem)[mask] for elem,mask in zip(abslines_plot[4][0][mask_lines].T,mask_obs_intime_repeat)],dtype=object)

#creating a mask from the ones with at least one detection 
#(or at least one significant detections if we don't consider non significant detections)
if display_nonsign:
    mask_obj_withdet=np.array([(elem>0).any() for elem in global_displayed_sign])
else:
    mask_obj_withdet=np.array([(elem>=slider_sign).any() for elem in global_displayed_sign])

#storing the number of objects with detections
n_obj_withdet=sum(mask_obj_withdet & mask_obj_base)

#we don't fuse with mask_obj_withdet anymore since the options to remove non-detections was removed
mask_obj=mask_obj_base

Edd_factor_restrict=Edd_factor[mask_obj].astype(float)
dist_factor_restrict=dist_factor[mask_obj].astype(float)

if np.ndim(hid_plot)==4:
    flag_single_obj=True
else:
    flag_single_obj=False

dict_linevis['flag_single_obj']=flag_single_obj

if flag_single_obj:
    hid_plot_restrict=hid_plot
    lum_plot_restrict=lum_plot
else:
    hid_plot_restrict=hid_plot.T[mask_obj].T
    lum_plot_restrict=lum_plot.T[mask_obj].T


incl_plot_restrict=incl_plot[mask_obj]

#creating variables with values instead of uncertainties for the inclination and nh colormaps

incl_cmap=np.array([incl_plot.T[0],incl_plot.T[0]-incl_plot.T[1],incl_plot.T[0]+incl_plot.T[2]]).T
incl_cmap_base=incl_cmap[mask_obj_base]
incl_cmap_restrict=incl_cmap[mask_obj]

nh_plot_restrict=deepcopy(nh_plot)
nh_plot_restrict=nh_plot_restrict.T[mask_obj].T

kt_plot_restrict=deepcopy(kt_plot)
kt_plot_restrict=kt_plot_restrict.T[mask_obj].T


#HIGH ENERGY TRANSPOSITIONS

flux_high_list = np.array([flux_high_list[i_obj][mask_included_selection[i_obj]] for i_obj in range(n_obj_init)], dtype=object)

fitmod_broadband_list = np.array([fitmod_broadband_list[i_obj][mask_included_selection[i_obj]] for i_obj in range(n_obj_init)], dtype=object)
gamma_nthcomp_list=np.array([gamma_nthcomp_list[i_obj][mask_included_selection[i_obj]]\
                             for i_obj in range(n_obj_init)], dtype=object)
Tin_diskbb_list=np.array([Tin_diskbb_list[i_obj][mask_included_selection[i_obj]]\
                             for i_obj in range(n_obj_init)], dtype=object)

dict_linevis['gamma_nthcomp_list']=gamma_nthcomp_list
dict_linevis['Tin_diskbb_list']=Tin_diskbb_list

#creating a luminosity version with nans instead
lum_high_list=np.array([np.array([np.repeat(np.nan,3) if flux_high_list[i_obj][i_obs] is None\
                else flux_high_list[i_obj][i_obs]*Edd_factor[i_obj] for i_obs in range(len(flux_high_list[i_obj]))])\
          for i_obj in range(len(flux_high_list))],dtype=object)

#created to avoid indexation errors, will be replaced later
lum_high_sign_list = deepcopy(lum_high_list)

#this one will only be for the HID
lum_high_1sig_list=deepcopy(lum_high_list)

#and a count rate version to store in the table
BAT_rate_list=np.array([np.array([np.repeat(np.nan,3) for i_obs in range(len(flux_high_list[i_obj]))])\
          for i_obj in range(len(flux_high_list))],dtype=object)

#exposure array
BAT_expos_list=np.array([np.repeat(np.nan,len(flux_high_list[i_obj]))\
          for i_obj in range(len(flux_high_list))],dtype=object)

#variable for checking if BAT values have been added

if sum(mask_obj)>0:
    mask_added_BAT_sign = np.repeat(False,  len(lum_high_list[mask_obj][0]))
else:
    mask_added_BAT_sign=False

#adding the significant BAT extrapolated fluxes to the lum_high_list array if asked to (for the scatter plots)
#the sum mask_obj condition skips issues when no observation is kept
if add_BAT_flux_corr and display_single and choice_source[0]=='4U1630-47' and sum(mask_obj)>0 and use_obsids:

    bat_lc_df_scat = fetch_bat_lightcurve(catal_bat_df, catal_bat_simbad, choice_source, binning=BAT_binning_scat)

    #first for all elements
    bat_lc_arr_rate=np.array([bat_lc_df_scat[bat_lc_df_scat.columns[1]],
                              bat_lc_df_scat[bat_lc_df_scat.columns[2]],
                              bat_lc_df_scat[bat_lc_df_scat.columns[2]]]).clip(0)

    bat_lc_expos_rate=np.array(bat_lc_df_scat['TIMEDEL_CODED'])

    # converting to 15-50keV luminosity in Eddington units, removing negative values
    bat_lc_lum_nocorr_scat = bat_lc_arr_rate.T \
                        * convert_BAT_count_flux['4U1630-47'] * Edd_factor_restrict

    bat_lc_lum_scat=corr_factor_lbat(bat_lc_lum_nocorr_scat)

    bat_lc_mjd_scat = np.array(bat_lc_df_scat[bat_lc_df_scat.columns[0]])

    #for the rest
    obs_dates_4U = Time(date_list[mask_obj][0].astype(str)).mjd.astype(int)

    mask_withtime_BAT = [elem in bat_lc_mjd_scat for elem in obs_dates_4U]

    id_match_BAT_scat = np.array([np.argwhere(bat_lc_mjd_scat == elem)[0][0]\
                                       for elem in obs_dates_4U[mask_withtime_BAT]])

    #first adding all of the BAT matches
    for id_obs_match,i_obs_match in enumerate(np.arange(len(lum_high_list[mask_obj][0]))[mask_withtime_BAT]):

        BAT_rate_list[mask_obj][0][i_obs_match]=np.array([bat_lc_arr_rate.T[id_match_BAT_scat[id_obs_match]][0],
                                                          bat_lc_arr_rate.T[id_match_BAT_scat[id_obs_match]][1]*1.65,
                                                          bat_lc_arr_rate.T[id_match_BAT_scat[id_obs_match]][1]*1.65])

        BAT_expos_list[mask_obj][0][i_obs_match]=bat_lc_expos_rate[id_match_BAT_scat[id_obs_match]]

        #only overwritting the values without already existing measured fluxes
        if np.isnan(lum_high_list[mask_obj][0][i_obs_match][0]):

            #note that here we multiply the uncertainties by 1.65 to have everything at 90% uncertainty
            # (which we later assume in the correlation coefficients computation)

            #in theory the uncertain converison
            # should be done before the luminosity conversion but it's almost linear so we don't care
            lum_high_list[mask_obj][0][i_obs_match]=np.array([bat_lc_lum_scat[id_match_BAT_scat[id_obs_match]][0],
                                                         bat_lc_lum_scat[id_match_BAT_scat[id_obs_match]][1]*1.65,
                                                         bat_lc_lum_scat[id_match_BAT_scat[id_obs_match]][1]*1.65])

            #1 sigma errors for the broad HID
            lum_high_1sig_list[mask_obj][0][i_obs_match]=np.array([bat_lc_lum_scat[id_match_BAT_scat[id_obs_match]][0],
                                                         bat_lc_lum_scat[id_match_BAT_scat[id_obs_match]][1],
                                                         bat_lc_lum_scat[id_match_BAT_scat[id_obs_match]][1]])
            mask_added_BAT_sign[i_obs_match]=True

    #then creating a second array for only 2 sigma significant detections (for the scatters)
    lum_high_sign_list=deepcopy(lum_high_list)
    for i_obs in range(len(lum_high_list[mask_obj][0])):
        if lum_high_1sig_list[mask_obj][0][i_obs][0]-lum_high_1sig_list[mask_obj][0][i_obs][1]*2<=0:
            lum_high_sign_list[mask_obj][0][i_obs]=np.repeat(np.nan,3)
            mask_added_BAT_sign[i_obs]=False

if sum(mask_obj)>0:
    mask_added_INT_sign=np.repeat(False,len(lum_high_list[mask_obj][0]))
else:
    mask_added_INT_sign=False

if add_INT_flux_corr and display_single and choice_source[0]=='4U1630-47' and sum(mask_obj)>0:

    # currently limited to 4U1630-47
    int_lc_df = fit_int_revol_dict[choice_source[0]]

    ####VARIABLEs FOR THE FLUX ADDITION. ONLY ADDING SIGNIFICANT FLUX FOR NOW
    mask_sign_rate_int=np.array(int_lc_df['RATE_30.0-50.0'] - 2 * int_lc_df['RATE_ERR_30.0-50.0'] > 0)

    int_fit_ratesign_30_50 = np.array(int_lc_df['RATE_30.0-50.0'][mask_sign_rate_int])
    int_fit_ratesign_30_50_err = np.array(int_lc_df['RATE_ERR_30.0-50.0'][mask_sign_rate_int]).clip(0)

    int_fit_flux_regr_15_50_1sig=np.array([int_rate_to_flux(elem,elem_err) for elem,elem_err\
                                     in zip(int_fit_ratesign_30_50,int_fit_ratesign_30_50_err)])

    int_fit_flux_regr_15_50=np.array([int_rate_to_flux(elem,elem_err,error_percent=90) for elem,elem_err\
                                     in zip(int_fit_ratesign_30_50,int_fit_ratesign_30_50_err)])

    int_fit_lum_regr_15_50=int_fit_flux_regr_15_50*Edd_factor_restrict[0]
    int_fit_lum_regr_15_50_1sig=int_fit_flux_regr_15_50_1sig*Edd_factor_restrict[0]

    int_lc_mjd = np.array([Time(elem).mjd.astype(float) for elem in int_lc_df['ISOT']])

    # to avoid mixing with the table variable
    #since this is initially the starting time, we offset it by 1.5 to then compute the argmin from the middle
    #of the osbervation
    int_lc_mjd_regr = int_lc_mjd.astype(int)[mask_sign_rate_int]+1.5

    # for the rest
    obs_dates_4U = Time(date_list[mask_obj][0].astype(str)).mjd.astype(int)


    id_int_regr_bestmatch=[abs(elem-int_lc_mjd_regr).argmin() for elem in obs_dates_4U]

    #here we put 2.5 as a max to allow being up to one day away from the center of the revolution
    mask_int_regr_match_obs=[abs(elem-int_lc_mjd_regr).min()<=2.5 for elem in obs_dates_4U]

    mask_added_INT_sign = np.repeat(False, len(lum_high_list[mask_obj][0]))


    # first adding all of the BAT matches
    for i_obs_match in np.arange(len(lum_high_list[mask_obj][0]))[mask_int_regr_match_obs]:

        # only overwritting the values without already existing measured fluxes
        #NOTE: COULD BE UPGRADED TO JUST MORE CONSTRAINED VALUES, BUT THIS COULD
        #BE AN ISSUE IN REVOLUTION MATCHING BC INTEGRAL OBS MIGHT BE FAR FROM THE OBS
        if np.isnan(lum_high_list[mask_obj][0][i_obs_match][0]):

            # using the 90% uncertainties here
            # (which we later assume in the correlation coefficients computation)
            lum_high_list[mask_obj][0][i_obs_match] = int_fit_lum_regr_15_50[id_int_regr_bestmatch[i_obs_match]]

            # 1 sigma errors for the broad HID
            lum_high_1sig_list[mask_obj][0][i_obs_match] =\
                int_fit_lum_regr_15_50_1sig[id_int_regr_bestmatch[i_obs_match]]

            mask_added_INT_sign[i_obs_match] = True

    #then creating a second array for only 2 sigma significant detections (for the scatters)
    lum_high_sign_list=deepcopy(lum_high_list)
    for i_obs in range(len(lum_high_list[mask_obj][0])):
        if lum_high_1sig_list[mask_obj][0][i_obs][0]-lum_high_1sig_list[mask_obj][0][i_obs][1]*2<=0:
            lum_high_sign_list[mask_obj][0][i_obs]=np.repeat(np.nan,3)
            mask_added_BAT_sign[i_obs]=False
            mask_added_INT_sign[i_obs] = False

#mask for dashing points in the correlations later
if display_single and choice_source[0]=='4U1630-47':
    mask_added_regr_sign=(mask_added_INT_sign) | (mask_added_BAT_sign)
else:
    mask_added_regr_sign=None

dict_linevis['mask_added_regr_sign']=mask_added_regr_sign


#for the HID
lum_high_1sig_plot=values_manip_var(lum_high_1sig_list)
gamma_nthcomp_plot=values_manip_var(gamma_nthcomp_list)

#for the tables
lum_high_plot=values_manip_var(lum_high_list)

#for the scatters
lum_high_sign_plot=values_manip_var(lum_high_sign_list)

#secondary parameters
Tin_diskbb_plot=values_manip_var(Tin_diskbb_list)

BAT_lc_plot=values_manip_var(BAT_rate_list)

#no need to change this one since it only has 2D
BAT_expos_plot=BAT_expos_list

#masking the selected objects in each array
BAT_lc_plot_restrict=deepcopy(BAT_lc_plot)

BAT_expos_plot_restrict=deepcopy(BAT_expos_plot)

Tin_diskbb_plot_restrict=deepcopy(Tin_diskbb_plot)

lum_high_1sig_plot_restrict=deepcopy(lum_high_1sig_plot)

lum_high_plot_restrict=deepcopy(lum_high_plot)

lum_high_sign_plot_restrict=deepcopy(lum_high_sign_plot)

gamma_nthcomp_plot_restrict=deepcopy(gamma_nthcomp_plot)

#to avoid issues with single object
if not flag_single_obj:
    BAT_lc_plot_restrict = BAT_lc_plot_restrict.T[mask_obj].T
    BAT_expos_plot_restrict=BAT_expos_plot_restrict.T[mask_obj].T
    Tin_diskbb_plot_restrict=Tin_diskbb_plot_restrict.T[mask_obj].T
    lum_high_1sig_plot_restrict=lum_high_1sig_plot_restrict.T[mask_obj].T
    lum_high_plot_restrict=lum_high_plot_restrict.T[mask_obj].T
    lum_high_sign_plot_restrict=lum_high_sign_plot_restrict.T[mask_obj].T
    gamma_nthcomp_plot_restrict=gamma_nthcomp_plot_restrict.T[mask_obj].T

#we create this one solely for the sake of the scatter plots
hr_high_plot_restrict=deepcopy(lum_high_plot_restrict)

#index 1 is the 3-6 band. slightly overestimated since lum_plot is in 90% uncertainties but these should be negligible
hr_high_plot_restrict[0]=lum_high_plot_restrict[0]/lum_plot_restrict[1][0]
hr_high_plot_restrict[1]=((lum_high_plot_restrict[1]/lum_high_plot_restrict[0])**2+\
                          (lum_plot_restrict[1][1]/lum_plot_restrict[1][0])**2)**(1/2)*hr_high_plot_restrict[0]

#different conditions depending on whether the high_E luminosity is 0 or not to create the upper limit
for i_obj in range(len(hr_high_plot_restrict[2])):

    #this won't be true if the values are nans
    no_det_vals=lum_high_plot_restrict[0][i_obj]==0.

    #standard computation
    hr_high_UL_whenvalue=((lum_high_plot_restrict[2][i_obj]/lum_high_plot_restrict[0][i_obj])**2+\
                              (lum_plot_restrict[1][2][i_obj]/lum_plot_restrict[1][0][i_obj])**2)**(1/2)\
                 *hr_high_plot_restrict[0][i_obj]

    #constructing conservatively from UL numerator/LLdenominator
    hr_high_UL_whennovalue=lum_high_plot_restrict[2][i_obj]/(lum_plot_restrict[1][0][i_obj]-lum_plot_restrict[1][1][i_obj])


    hr_high_plot_restrict[2][i_obj]=np.where(no_det_vals,hr_high_UL_whennovalue,hr_high_UL_whenvalue)

#we create this one solely for the sake of the scatter plots
hr_high_sign_plot_restrict=deepcopy(lum_high_sign_plot_restrict)

hr_high_sign_plot_restrict[0]=lum_high_sign_plot_restrict[0]/lum_plot_restrict[1][0]
hr_high_sign_plot_restrict[1]=((lum_high_sign_plot_restrict[1]/lum_high_sign_plot_restrict[0])**2+\
                          (lum_plot_restrict[1][1]/lum_plot_restrict[1][0])**2)**(1/2)*hr_high_sign_plot_restrict[0]

#different conditions depending on whether the high_E luminosity is 0 or not to create the upper limit
for i_obj in range(len(hr_high_sign_plot_restrict[2])):

    #this won't be true if the values are nans
    no_det_vals=lum_high_sign_plot_restrict[0][i_obj]==0.

    #standard computation
    hr_high_sign_UL_whenvalue=((lum_high_sign_plot_restrict[2][i_obj]/lum_high_sign_plot_restrict[0][i_obj])**2+\
                              (lum_plot_restrict[1][2][i_obj]/lum_plot_restrict[1][0][i_obj])**2)**(1/2)\
                 *hr_high_sign_plot_restrict[0][i_obj]

    #constructing conservatively from UL numerator/LLdenominator
    hr_high_sign_UL_whennovalue=lum_high_sign_plot_restrict[2][i_obj]/(lum_plot_restrict[1][0][i_obj]-lum_plot_restrict[1][1][i_obj])


    hr_high_sign_plot_restrict[2][i_obj]=np.where(no_det_vals,hr_high_sign_UL_whennovalue,hr_high_sign_UL_whenvalue)


dict_linevis['lum_high_1sig_plot_restrict']=lum_high_1sig_plot_restrict

dict_linevis['lum_high_sign_plot_restrict']=lum_high_sign_plot_restrict
dict_linevis['hr_high_plot_restrict']=hr_high_plot_restrict
dict_linevis['hr_high_sign_plot_restrict']=hr_high_sign_plot_restrict
dict_linevis['gamma_nthcomp_plot_restrict']=gamma_nthcomp_plot_restrict
dict_linevis['Tin_diskbb_plot_restrict']=Tin_diskbb_plot_restrict
dict_linevis['Tin_diskbb_plot']=Tin_diskbb_plot

#n_obj_restricted
n_obj_r=sum(mask_obj)

#creating an array for the intime observations
mask_intime_plot=np.array([(Time(date_list[mask_obj][i_obj_r].astype(str))>=Time(slider_date[0])) & (Time(date_list[mask_obj][i_obj_r].astype(str))<=Time(slider_date[1])) for i_obj_r in range(n_obj_r)],dtype=object)

#custom colorbar for the line substructure and outliers (needs the high energy elements)

diago_color=deepcopy(hid_plot[1][0])
if display_single and choice_source[0]=='4U1630-47' and sum(ravel_ragged(mask_intime_plot))>0:
    for i_obj in range(len(diago_color)):
        if obj_list[i_obj]=='4U1630-47':
            for i_obs in range(len(diago_color[i_obj])):

                #first rule is for luminosity and HR + HR broad or det treshold to remove things
                # that stay in intermediate states
                # second rule is for a significant detection in suzeaku above 3.8 EWratio
                is_line=(not np.isnan(abslines_plot[4][0][0][i_obj][i_obs])
                                                 and abslines_plot[4][0][0][i_obj][i_obs]>slider_sign and \
                                                 not np.isnan(abslines_plot[4][0][1][i_obj][i_obs]))

                is_below_broad=not np.isnan(hr_high_plot_restrict[0][0][i_obs]) and hr_high_plot_restrict[0][0][i_obs]<0.1

                is_substructure=(hid_plot[1][0][i_obj][i_obs]<4.2e-2 and hid_plot[1][0][i_obj][i_obs]>2e-2 \
                                 and hid_plot[0][0][i_obj][i_obs]<0.35 and (is_line or is_below_broad))

                #suzaku start of the substructure
                is_outlier_suzaku=(((is_line and abslines_plot[4][0][1][i_obj][i_obs]>slider_sign) and \
                                                 abslines_plot[0][0][1][i_obj][i_obs]\
                                                 /abslines_plot[0][0][0][i_obj][i_obs]>3.8) and\
                                                instru_list[i_obj][i_obs]=='Suzaku')

                #spl outlier with Feka25 det
                is_outlier_SPL=abslines_plot[4][0][0][i_obj][i_obs]>slider_sign and hid_plot[1][0][i_obj][i_obs]>1e-1

                #XMM transition
                is_outlier_XMM=instru_list[i_obj][i_obs]=='XMM' and \
                               hid_plot[1][0][i_obj][i_obs] < 0.09 and hid_plot[1][0][i_obj][i_obs] > 0.075

                # hid_plot[0][0][i_obj][i_obs]>0.43 and  hid_plot[0][0][i_obj][i_obs]<0.45 and \

                if is_substructure:
                    diago_color[i_obj][i_obs]='orange'
                #outlier SPL detection
                elif is_outlier_suzaku or is_outlier_SPL or is_outlier_XMM:
                    diago_color[i_obj][i_obs]='blue'
                else:
                    diago_color[i_obj][i_obs]='grey'

dict_linevis['diago_color']=diago_color

custom_states_color=deepcopy(hid_plot[1][0])
if display_single and choice_source[0]=='4U1630-47' and sum(ravel_ragged(mask_intime_plot))>0:
    for i_obj in range(len(custom_states_color)):
        if obj_list[i_obj]=='4U1630-47':
            for i_obs in range(len(custom_states_color[i_obj])):

                #first rule is for luminosity and HR + HR broad or det treshold to remove things
                # that stay in intermediate states
                # second rule is for a significant detection in suzeaku above 3.8 EWratio
                
                obs_soft_lum=hid_plot[1][0][i_obj][i_obs]
                obs_soft_HR=hid_plot[0][0][i_obj][i_obs]
                obs_broad_HR=hr_high_plot_restrict[0][0][i_obs]

                #note that the 1.65 are to convert to 1 sigma uncertainties

                obs_broad_HR_UL=hr_high_plot_restrict[2][0][i_obs]/1.65 if hr_high_plot_restrict[0][0][i_obs]==0. else\
                    hr_high_plot_restrict[0][0][i_obs]+hr_high_plot_restrict[1][0][i_obs]/1.65

                obs_broad_HR_sign=not np.isnan(hr_high_plot_restrict[0][0][i_obs]) and \
                                  not hr_high_plot_restrict[0][0][i_obs]==0. and \
                                  (lum_high_plot_restrict[0][0][i_obs]>2*lum_high_plot_restrict[1][0][i_obs]/1.65)

                # note that the 6.1 limit is done explicitely to only take the right points during the 2021 outburst
                is_standard_hard=obs_soft_HR>6.1e-1 and obs_soft_lum<1e-1

                is_QRM=obs_soft_lum>1e-1 and obs_soft_lum<1.2e-1 and obs_soft_HR>6e-1

                is_soft=obs_broad_HR<1e-1 if obs_broad_HR_sign else obs_broad_HR_UL<1e-1

                is_inter=not np.isnan(obs_broad_HR) and \
                         ((obs_broad_HR_sign and obs_broad_HR>1e-1) or (not obs_broad_HR_sign and obs_broad_HR_UL>0.1))

                is_SPL=not np.isnan(obs_broad_HR) and obs_broad_HR>2.8e-1 and obs_soft_lum>5e-2

                #no decidable state
                custom_states_color[i_obj][i_obs]='grey'

                if is_inter:
                    custom_states_color[i_obj][i_obs] = 'orange'
                if is_soft:
                    custom_states_color[i_obj][i_obs] = 'green'
                if is_SPL:
                    custom_states_color[i_obj][i_obs]='red'

                #note: the two hard points with wrong hard position due to too much BAT variability:
                # 4130010111-004 and 4130010114-004
                if is_standard_hard:
                    custom_states_color[i_obj][i_obs] = 'blue'
                if is_QRM:
                    custom_states_color[i_obj][i_obs] = 'purple'

dict_linevis['custom_states_color']=custom_states_color

custom_ionization_color = deepcopy(hid_plot[1][0])
if display_single and choice_source[0] == '4U1630-47' and sum(ravel_ragged(mask_intime_plot)) > 0:
    for i_obj in range(len(custom_states_color)):
        if obj_list[i_obj] == '4U1630-47':
            for i_obs in range(len(custom_ionization_color[i_obj])):


                # no specific state
                custom_ionization_color[i_obj][i_obs] = 'grey'

                if observ_list[i_obj][i_obs]=='400010010_xis1':
                    #outlier_diagonal_middle
                    custom_ionization_color[i_obj][i_obs] = 'red'

                if observ_list[i_obj][i_obs]=='400010060_xis1':
                    #outlier_diagonal_lower_floor
                    custom_ionization_color[i_obj][i_obs] = 'rosybrown'

                if observ_list[i_obj][i_obs]=='906008010_xis1':
                    #diagonal_lower_low_highE_flux
                    custom_ionization_color[i_obj][i_obs] = 'orange'

                if observ_list[i_obj][i_obs]=='5501010104-001':
                    #diagonal_upper_low_highE_flux
                    custom_ionization_color[i_obj][i_obs] = 'powderblue'

                if observ_list[i_obj][i_obs]=='nu40014009001A01':
                    #diagonal_upper_mid_highE_flux
                    custom_ionization_color[i_obj][i_obs] = 'turquoise'

                if observ_list[i_obj][i_obs]=='409007010_xis1':
                    #diagonal_upper_high_highE_flux
                    custom_ionization_color[i_obj][i_obs] = 'pink'

                if observ_list[i_obj][i_obs]=='nu80902312002A01':
                    #SPL_whereline
                    custom_ionization_color[i_obj][i_obs] = 'forestgreen'

dict_linevis['custom_ionization_color'] = custom_ionization_color

#outburst coloring
color_cmap_outburst = mpl.cm.tab10

#note that this means the 2010 suzaku obs doesn't have a different color, but it makes everything else better
c_norm_outburst = mpl.colors.Normalize(vmin=1,
                              vmax=9)

colors_func_date = mpl.cm.ScalarMappable(norm=c_norm_outburst,cmap=color_cmap_outburst)

# computing the actual color array for the detections for later
color_outburst_arr = np.array([colors_func_date.to_rgba(elem) for elem in range(10)])

custom_outburst_number = deepcopy(hid_plot[1][0])
custom_outburst_color = deepcopy(hid_plot[1][0])

custom_outburst_dict=np.repeat(None,len(hid_plot[1][0]))

if display_single and choice_source[0] == '4U1630-47' and sum(ravel_ragged(mask_intime_plot)) > 0:
    for i_obj in range(len(custom_states_color)):

        #Here the idea is threefold.
        # 1: split automatically each source obs into different outburst assuming
        #    a minimum time step between 2 obs means a new outburst (and store it)
        # 2: color each outburst differently
        # 3: add a decreasing alpha to represent the time evolution within the outburst, normalized to the time
        #    between the first and last observation of each outburst
        outburst_dict={}

        date_list_obj=np.array([Time(elem) for elem in date_list[i_obj]])
        date_list_obj_order=date_list_obj.argsort()
        date_list_obj_sorted=date_list_obj.copy()
        date_list_obj_sorted.sort()

        #creating the sorted outburst number with (for now) a 1 year outburst delta treshold
        outburst_number_obj_arr=[]
        n_outburst=1

        manual_add=True

        #here we go through the sorted arr
        for i in range(len(date_list_obj)):

            #for 4U1630-47, adding a single addition for the 2022 outburst start:
            if obj_list[i_obj]=='4U1630-47' and manual_add and date_list_obj_sorted[i]>Time('2022-06-01'):
                n_outburst+=1
                manual_add=False


            outburst_number_obj_arr+=[n_outburst]

            if i!=len(date_list_obj)-1 and date_list_obj_sorted[i+1]-date_list_obj_sorted[i]>=TimeDelta(365,format='jd'):
                n_outburst+=1



        outburst_number_obj_arr=np.array(outburst_number_obj_arr)
        for i_obs in range(len(date_list_obj)):

            i_obs_sorted=np.argwhere(date_list_obj_order == i_obs)[0][0]

            #storing the outburst number an dcolor
            custom_outburst_number[i_obj][i_obs]=outburst_number_obj_arr[i_obs_sorted]

            #changed to array to modify the alpha below
            custom_outburst_color[i_obj][i_obs]=np.array(colors_func_date.to_rgba(custom_outburst_number[i_obj][i_obs]))


            #computing the max time delta between the observation of the outburst to which belongs the observation
            obs_curr_outburst=date_list_obj_sorted[outburst_number_obj_arr==custom_outburst_number[i_obj][i_obs]]

            #adding the outburst to the dictionnary for the first date of the outburst
            if obs_curr_outburst[0]==date_list_obj_sorted[i_obs_sorted]:
                outburst_start_y=obs_curr_outburst[0].iso.split('-')[0]
                outburst_end_y=obs_curr_outburst[-1].iso.split('-')[0]

                if outburst_start_y==outburst_end_y:
                    outburst_str=outburst_start_y
                else:
                    outburst_str=outburst_start_y+'-'+outburst_end_y

                outburst_dict[outburst_str]=custom_outburst_color[i_obj][i_obs]

            curr_outburst_max_delta=obs_curr_outburst[-1]-obs_curr_outburst[0]

            #computing the delta fraction, with a safeguard for single observations
            curr_outburst_delta_frac=0 if len(obs_curr_outburst)==1 else \
                                (date_list_obj_sorted[i_obs_sorted]-obs_curr_outburst[0])/curr_outburst_max_delta

            #changing the alpha linearly down to 0.3 for the last observation
            custom_outburst_color[i_obj][i_obs][-1]=1.-0.7*np.sqrt(float(curr_outburst_delta_frac))

            #changing back into a tuple to avoid issues with transpositions later
            custom_outburst_color[i_obj][i_obs]=tuple(custom_outburst_color[i_obj][i_obs])

        custom_outburst_dict[i_obj]=outburst_dict

dict_linevis['custom_outburst_color'] = custom_outburst_color
dict_linevis['custom_outburst_number'] = custom_outburst_number
dict_linevis['custom_outburst_dict'] = custom_outburst_dict

#defining the dataset that will be used in the plots for the colormap limits
if radio_info_cmap in ['Velocity shift','Del-C']:
    radio_cmap_i=1 if radio_info_cmap=='Velocity shift' else 2
else:
    radio_cmap_i=0
    
#colormap when not splitting detections
cmap_color_source=mpl.cm.hsv_r.copy()

cmap_color_source.set_bad(color='grey')

cyclic_cmap=True

#colormaps when splitting detections
cmap_color_det=mpl.cm.plasma.copy()
cmap_color_det.set_bad(color='grey')

cyclic_cmap_det=False

cmap_color_nondet=mpl.cm.viridis_r.copy()
cmap_color_nondet.set_bad(color='grey')

cyclic_cmap_nondet=False

#computing the extremal values of the whole sample/plotted sample to get coherent colormap normalisations, and creating the range of object colors

if global_colors:
    global_plotted_sign=abslines_plot[4][0].ravel()
    global_plotted_data=abslines_plot[radio_cmap_i][0].ravel()

    #objects colormap for common display
    norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=max(0,len(abslines_infos_perobj)+(-1 if not cyclic_cmap else 0)))
    colors_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=cmap_color_source)

    norm_colors_det=mpl.colors.Normalize(vmin=0,vmax=max(0,n_obj_withdet+(-1 if not cyclic_cmap_det else 0)+(1 if n_obj_withdet==0 else 0)))
    colors_det=mpl.cm.ScalarMappable(norm=norm_colors_det,cmap=cmap_color_det)

    norm_colors_nondet=mpl.colors.Normalize(vmin=0,vmax=max(0,len(abslines_infos_perobj)-n_obj_withdet+(-1 if not cyclic_cmap_nondet else 0)))
    colors_nondet=mpl.cm.ScalarMappable(norm=norm_colors_nondet,cmap=cmap_color_nondet)

    #the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
    global_plotted_datetime=np.array([elem for elem in date_list for i in range(len(mask_lines))],dtype='object')

    global_mask_intime=np.repeat(True,len(ravel_ragged(global_plotted_datetime)))

    global_mask_intime_norepeat=np.repeat(True,len(ravel_ragged(date_list)))

else:
    global_plotted_sign=abslines_plot[4][0][mask_lines].T[mask_obj].ravel()
    global_plotted_data=abslines_plot[radio_cmap_i][0][mask_lines].T[mask_obj].ravel()

    #objects colormap
    norm_colors_obj=mpl.colors.Normalize(vmin=0,vmax=max(0,len(abslines_infos_perobj[mask_obj])+(-1 if not cyclic_cmap else 0)))
    colors_obj=mpl.cm.ScalarMappable(norm=norm_colors_obj,cmap=cmap_color_source)

    norm_colors_det=mpl.colors.Normalize(vmin=0,vmax=max(0,n_obj_withdet+(-1 if not cyclic_cmap_det else 0)))
    colors_det=mpl.cm.ScalarMappable(norm=norm_colors_det,cmap=cmap_color_det)

    norm_colors_nondet=mpl.colors.Normalize(vmin=0,
                        vmax=max(0,len(abslines_infos_perobj[mask_obj])-n_obj_withdet+(-1 if not cyclic_cmap_nondet else 0)))
    colors_nondet=mpl.cm.ScalarMappable(norm=norm_colors_nondet,cmap=cmap_color_nondet)

#adapting the plotted data in regular array for each object in order to help
#global masks to take off elements we don't want in the comparison

    #the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
    global_plotted_datetime=np.array([elem for elem in date_list[mask_obj] for i in range(sum(mask_lines))],dtype='object')

    if len(global_plotted_datetime)==0:
        no_obs=True
        global_mask_intime=False
        global_mask_intime_norepeat=False
    else:
        no_obs=False
        global_mask_intime=(Time(ravel_ragged(global_plotted_datetime))>=Time(slider_date[0])) &\
            (Time(ravel_ragged(global_plotted_datetime))<=Time(slider_date[1]))

        global_mask_intime_norepeat=(Time(ravel_ragged(date_list[mask_obj]))>=Time(slider_date[0])) &\
            (Time(ravel_ragged(date_list[mask_obj]))<=Time(slider_date[1]))

#global_nondet_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])<slider_sign) & (global_mask_intime)

global_det_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])>0) & (global_mask_intime)

global_sign_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])>=slider_sign) & (global_mask_intime)

global_det_data=np.array([subelem for elem in global_plotted_data for subelem in elem])[global_det_mask]

#this second array is here to restrict the colorbar scalings to take into account significant detections only
global_sign_data=np.array([subelem for elem in global_plotted_data for subelem in elem])[global_sign_mask]

#same for the color-coded infos
cmap_info=mpl.cm.plasma_r.copy() if radio_info_cmap not in ['Time','nH','kT'] else mpl.cm.plasma.copy()

cmap_info.set_bad(color='grey')

#normalisation of the colormap
if radio_cmap_i==1 or radio_info_cmap=='EW ratio':
    gamma_colors=1 if radio_cmap_i==1 else 0.5
    cmap_norm_info=colors.PowerNorm(gamma=gamma_colors)
    
elif radio_info_cmap not in ['Inclination','Time','kT']:
    cmap_norm_info=colors.LogNorm()
else:
    #keeping a linear norm for the inclination
    cmap_norm_info=colors.Normalize()

with tab_hid:

    tab_soft_hid,tab_BAT_hid,tab_add_data=st.tabs(['Soft X HID','Broad HID','Add data'])


'''
#Manual additions
'''

with tab_add_data:

    st.markdown('These data points will be added to the soft and/or broad HLDs.')
    st.info('Remember to make your Eddington ratio compatible with this work by choosing the same D and M '
              'than the ones I am using')
    df = pd.DataFrame(
        [
            {"ObsID":'', "Date (UTC)":None,"Telescope": '', "L_3-10/L_Edd": 0.,"HR_[6-10]/[3-6]":0.,'HR_[15-50]/[3-6]':0.,
             'color':''},
        ]
    )
    additional_HLD_points = st.data_editor(
        df,
        column_config={
            "ObsID": st.column_config.TextColumn(
                "ObsID of the observation",
                help="Mostly for avoiding confusion",),
            "Date (UTC)": st.column_config.DatetimeColumn(
                "Date of the observation in UTC format",
                help="Mostly for avoiding confusion"),
            "Telescope": st.column_config.TextColumn(
                "Instrument taking the observation",
                help="Mostly for avoiding confusion",),
            "L_3-10/L_Edd": st.column_config.NumberColumn(
                "3-10 keV Eddington ratio",
                help="",
                format="%.3e"),
            "HR_[6-10]/[3-6]": st.column_config.NumberColumn(
                "[6-10]/[3-6] keV HR",
                help="",
                format="%.3e",),
            "HR_[15-50]/[3-6]": st.column_config.NumberColumn(
                "[15-50]/[3-6] keV HR",
                help="",
                format="%.3e",),
            "color": st.column_config.TextColumn(
                "Color for the display",
                help="Mostly for avoiding confusion", ),

        },
        hide_index=True,num_rows="dynamic")

    # favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
    # st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

#necessary items for the hid graph run
items_list=[
abslines_infos_perobj,
abslines_plot,nh_plot,kt_plot,hid_plot, incl_plot,
mask_obj, mask_obj_base, mask_lines, mask_lines_ul,
obj_list, date_list, instru_list, lum_list, choice_telescope, telescope_list,
bool_incl_inside, bool_noincl,
slider_date, slider_sign,
radio_info_cmap, radio_cmap_i,
cmap_color_source, cmap_color_det, cmap_color_nondet,observ_list,hatch_unstable,change_legend_position]

items_str_list=['abslines_infos_perobj',
'abslines_plot','nh_plot','kt_plot','hid_plot','incl_plot',
'mask_obj','mask_obj_base', 'mask_lines', 'mask_lines_ul',
'obj_list', 'date_list', 'instru_list', 'lum_list', 'choice_telescope', 'telescope_list',
'bool_incl_inside', 'bool_noincl',
'slider_date', 'slider_sign',
'radio_info_cmap', 'radio_cmap_i',
'cmap_color_source', 'cmap_color_det', 'cmap_color_nondet','observ_list','hatch_unstable','change_legend_position']

for dict_key, dict_item in zip(items_str_list,items_list):
    dict_linevis[dict_key]=dict_item

dict_linevis['lc_int_sw_dict'] = lc_int_sw_dict
dict_linevis['fit_int_revol_dict'] = fit_int_revol_dict
dict_linevis['HR_broad_bands'] = HR_broad_bands
dict_linevis['lum_broad_bands'] = lum_broad_bands

dict_linevis['Edd_factor_restrict'] = Edd_factor_restrict
dict_linevis['lum_plot'] = lum_plot
dict_linevis['use_obsids'] = use_obsids
dict_linevis['exptime_list'] = exptime_list
dict_linevis['hid_log_HR'] = hid_log_HR
dict_linevis['display_minorticks']=display_minorticks

dict_linevis['additional_HLD_points']=additional_HLD_points

if len(global_plotted_datetime)==0:
    st.warning('No points remaining with current sample/date selection')
elif not skip_HID:
    hid_graph(ax_hid,dict_linevis,
              display_single=display_single, display_nondet=display_nondet, display_upper=display_upper,
              cyclic_cmap_nondet=cyclic_cmap_nondet, cyclic_cmap_det=cyclic_cmap_det, cyclic_cmap=cyclic_cmap,
              cmap_incl_type=cmap_incl_type, cmap_incl_type_str=cmap_incl_type_str,
              radio_info_label=radio_info_label,
              ew_ratio_ids=ew_ratio_ids,
              color_nondet=color_nondet,
              restrict_threshold=restrict_threshold, display_nonsign=display_nonsign,
              display_central_abs=display_central_abs,
              display_incl_inside=display_incl_inside, dash_noincl=dash_noincl,
              display_hid_error=display_hid_error, display_edgesource=display_edgesource,
              split_cmap_source=split_cmap_source,
              display_evol_single=display_evol_single, display_dicho=display_dicho,
              global_colors=global_colors, alpha_abs=alpha_abs,
              paper_look=paper_look, bigger_text=bigger_text, square_mode=square_mode, zoom=zoom_hid,
              restrict_match_INT=restrict_match_INT)

# fig_hid_html = mpld3.fig_to_html(fig_hid)
# components.html(fig_hid_html, height=1000)

with tab_hid:

    with tab_soft_hid:

        if not skip_HID:
            #preventing some bugs when updating too many parameters at once
            try:
                st.pyplot(fig_hid)
            except:
                st.rerun()

    broad_band_disp_ok=True

    with tab_BAT_hid:

        if not display_single:
            st.info('Broad band BAT HIDs currently restricted to single sources.')

            broad_band_disp_ok=False
        else:
            if not choice_source[0]=='4U1630-47':
                st.info('Broad band BAT HIDs currently restricted to  4U 1630-47')

                broad_band_disp_ok=False

        if broad_band_disp_ok and not skip_HID and not len(global_plotted_datetime)==0:

            if not display_broad_hid_BAT:
                st.info('Toggle BAT broad band HID option in the sidebar to display.')

            else:

                if not square_mode:
                    fig_hid_bat, ax_hid_bat = plt.subplots(1, 1, figsize=(8, 5) if bigger_text else (12, 6))
                else:
                    fig_hid_bat, ax_hid_bat = plt.subplots(1, 1, figsize=(8, 6))
                ax_hid_bat.clear()

                dict_linevis['catal_bat_df']=catal_bat_df
                dict_linevis['catal_bat_simbad']=catal_bat_simbad
                dict_linevis['Edd_factor_restrict']=Edd_factor_restrict
                dict_linevis['lum_plot']=lum_plot
                dict_linevis['HR_broad_bands']=HR_broad_bands
                dict_linevis['lum_broad_bands']=lum_broad_bands
                dict_linevis['sign_broad_hid_BAT']=sign_broad_hid_BAT

                hid_graph(ax_hid_bat, dict_linevis,
                          display_single=display_single, display_nondet=display_nondet, display_upper=display_upper,
                          cyclic_cmap_nondet=cyclic_cmap_nondet, cyclic_cmap_det=cyclic_cmap_det,
                          cyclic_cmap=cyclic_cmap,
                          cmap_incl_type=cmap_incl_type, cmap_incl_type_str=cmap_incl_type_str,
                          radio_info_label=radio_info_label,
                          ew_ratio_ids=ew_ratio_ids,
                          color_nondet=color_nondet,
                          restrict_threshold=restrict_threshold, display_nonsign=display_nonsign,
                          display_central_abs=display_central_abs,
                          display_incl_inside=display_incl_inside, dash_noincl=dash_noincl,
                          display_hid_error=display_hid_error, display_edgesource=display_edgesource,
                          split_cmap_source=split_cmap_source,
                          display_evol_single=display_evol_single, display_dicho=display_dicho,
                          global_colors=global_colors, alpha_abs=alpha_abs,
                          paper_look=paper_look, bigger_text=bigger_text, square_mode=square_mode, zoom=zoom_hid,
                          broad_mode='BAT',restrict_match_INT=restrict_match_INT)

                try:
                    st.pyplot(fig_hid_bat)
                except:
                    st.rerun()

#### About tab
with tab_about:
    st.markdown('**visual_line** is a visualisation and download tool for iron-band X-ray absorption lines signatures in Black Hole Low-Mass X-ray Binaries (BHLMXBs).')
    st.markdown('It is made to complement and give access to the results of my observational papers, and more generally, to give an overview of the sampling and X-ray evolution of the outbursts of this category of sources.')
    st.markdown('Please contact me at [maxime.parrastro@gmail.com](mailto:maxime.parrastro@gmail.com) for questions, to report bugs or request features.')
    
    with st.expander('I want an overview of the science behind this'):
        st.header('Outbursts')

        #note: st.latex doesn't support non-math latex atm, and even trying to cheat it wih katex doesn't work very well
        #see in the future if it's better
        st.markdown('''
        X-Ray Binaries are binary systems emitting mostly in the X-ray band, composed of a compact object (Neutron Star or Black Hole) and a companion star.  
        The subgroup we are focusing on here is restricted to **Black Holes** orbiting with a "low mass" star (generally in the main sequence), for which accretion happens through Robe Loche overflow and an **accretion disk**. 
        Most of these sources have a very specific behavior: they spend the bigger part of their lifes undetected, at low luminosity (in "quiescence"), 
        but undergo regular **outbursts** on the timescale of a few months.
        During these outbursts, sources brighten by several orders of magnitude, up to a significant percentage of their Eddington luminosity $L_{Edd}$, which is the "theoretical" upper limit of standard accreting regimes.  
        
        Meanwhile, these objects follow a specific pattern of spectral evolution, switching from a powerlaw dominated Spectral Energy Distribution (SED) in soft X-rays (the so-called **"hard"** state) during the initial brightening, to the **"soft"** state, similarly bright but dominated by the thermal emission of the accretion disk (most likely extending to the Innermost Stable Circular Orbit of the Black Hole). After some time spent during the soft state (and occasional back and forth in some instances), the source invariably becomes fainter, transits back to the hard state if necessary, then returns to quiescence.
        
        Figure references:  
            [Petrucci et al. 2021](https://doi.org/10.1051/0004-6361/202039524)  
            [Done et al. 2007](https://doi.org/10.1007/s00159-007-0006-1)  
        #''')
        
        col_fig1, col_fig2= st.columns(2)
        try:
            with col_fig1:
                st.image(dump_path[:dump_path.rfind('/')]+'/outburst.png',caption='Example of the evolution of GX 339-4 in a Hardness/Luminosity Diagram during its 2019 outburst. the MJD dates of each observation highlight the direction of the evolution, and colors different spectral states (independant from the right picture). From Petrucci et al. 21')
            with col_fig2:
                st.image(dump_path[:dump_path.rfind('/')]+'/xray_states.png',caption="Example of the differences between spectral shapes for the soft (red) and hard (blue) state of Cygnus X-1. From Done et al. 2007")
        except:
            pass

        st.markdown('''
        Beyond this direct spectral dichotomy, a wealth of other features have been linked to the outburst evolution:  
            -a radio component associated to **jets** is only detected during the hard state  
            -specific **Quasi-Periodic Oscillations** (QPOs) can be detected in the timing features of the spectra, with different types in different spectral states, and can be linked to a wealth of geometrical and physicla behaviors  
            -**Dipping** episodes can be observed in the lightcurve of high-inclined sources in specific portions of the outburst. (Note: this is a different behavior than eclipses).  
            -Some sources exhibit strong **reflection** components during a part of the outburst, interpreted as internal illumination of the disk by the **hot corona** believed to be the source of the hard state powerlaw component.   
            -Other sources exhibit strongly relativistic **iron-line emission** features, which can be fitted to get constraints on the spin of the BH.  
            -...
        
        As of now, the reason for the typical spectral evolution of outbursts, as well as the cause of most of the features that appear during it, remains unknown.
        #''')
        
        col_winds, col_figwinds= st.columns(2)
        
        with col_winds:
            
            st.header('Winds')
            
            st.markdown('''
            Another feature seen in outbursts is the appearance of **narrow, blueshifted absorption lines** in X-ray spectra, primarily from the very strong Ka and Kb lines of FeXXV and FeXXVI at 7-8keV.
            They are interpreted as the signature of dense material outflowing from the accretion disk of the Black Hole, and are expected to expell amounts of matter comparable to the accretion rate.
            Since the first observations 25 ago, a wealth of absorption profiles have been observed in BHLMXBS.
            Wind signatures in **X-rays** have been traditionally found only in the soft states of high-inclined sources, but recent detections in hard states or for low-inclined sources challenge this assumption.
            
            One of the most critical matters about winds is their physical origin. In X-ray Binaries, two launching mechanisms are favored, using either **thermal** or **magnetic** processes. Modeling efforts are recent and only few observations have been successfully recreated by either until now, but this has shown the limit of current instruments. Indeed, it appears impossible to directly distinguish the wind launch mechanisms by simply using the absorption signatures, even for the best quality observations of the current generation instruments. Thus, until data from the new instruments, such as XRISM, becomes available, hope lies in the constraints on the physical parameters of the model creating the winds, or a more complete model to data comparison. 
            
            The **JED-SAD model** is a complete accretion-ejection framework for magnetically threaded disks, developed at the University of Grenoble-Alpes (France). Beyond very promising results for fitting all
            parts of the outburst of Black Hole XRBs, this model has been shown to produce winds, through both theoretical solutions and simulations. We aim to compare observations to synthetic spectral signatures computed from self-similar JED-SAD solutions, in order to access the evolution of the outflow, and to further constrain the nature and physical conditions of the disk during these observations.
            
            #''')
            
            st.header('''
            The global study ([Parra et al. 2024](https://doi.org/10.1051/0004-6361/202346920))
            #''')
            st.markdown('''
                        The science community and our own modeling efforts would benefit from a global and up-to-date view of the current wind signatures in BHLMXBs. However, while detailed works have been performed on the vast majority of individual detections, there are very few larger studies for several outbursts and sources. With the goal of providing a complete view of all currently known X-ray wind signatures, we first focus on the most historically studied and constraining observations, using the XMM-Newton and Chandra-HETG instruments.  
                        
                        We identify BHLMXB candidates through the BlackCAT and WATCHDOG catalogs, for a total of 79 sources. After extracting and pre-selecting all available spectra of these sources with high-enough quality to allow the detection of line, we end up with 242 spectra in 42 sources. We refer readers to the main paper for details on the line detection procedure.
                        
                        Beyond interactive displays of our results through HID and scatter plots, we provide direct access to the results table, restricted according to user demands. We also provide a monitoring display tool, which combines RXTE and up-to-date MAXI lightcurves and HR ratio evolutions of all single sources in the sample.
                        #''')
        try:
            with col_figwinds:
                st.image(dump_path[:dump_path.rfind('/')]+'/linedet_example.jpg',caption='Steps of the fitting procedure for a standard 4U130-47 Chandra spectra. First panel: 4-10 spectrum after the first continuum fit. Second panel: ∆C map of the line blind search, restricted to positive (i.e. improvements) regions. Standard confidence intervals are highlighted with different line styles, and the colormap with the ∆C improvements of emission and absorption lines. Third panel: Ratio plot of the best fit model once absorption lines are added. Fourth panel: Remaining residuals seen through a second blind search.#')
        except:
            pass

        st.markdown('''
                    See the paper for detailed references to the points discussed above, and [Diaz Trigo et al. 2016](https://doi.org/10.1002/asna.201612315) or [Ponti et al. 2016](https://doi.org/10.1002/asna.201612339) for reviews on winds.  
                    #''')
                
    with st.expander('I want to know how to use this tool'):
        
        st.header('General information')
        
        st.markdown('''
                    Streamlit is a python library for web applications. 
                    Its interactivity is achieved by storing the status of the multiple widgets and re-running **the entire script** every time a modification is performed, allowing to recompute/redisplay/... all necessary outputs depending on the changes made to the widgets. 
                    
                    **The monitoring and parameter analysis plots take time to compute**, as the first one fetches data from the MAXI website in real time, and the second computes perturbative estimates of the correlation coefficient for each plot. As such, they are deactivated by default.  
                    
                    It is worth noting that the current version of this tool has trouble performing too many actions in a short time. This is partially covered through internal failsafes, but if you modify several options at once and something crashes, or displays in a non-standard way, either resetting an option (which reruns the script internally) or restarting the tool (either by pressing R or going in the top-right menu and clicking on Rerun) can fix the issue.
                    
                    Moreover, to avoid too much issues for data management, the data is split between each combination of instrument, and the creation of subsequent widget depends on this data.
                    
                    **This means that all subsequent widgets are reset to their default values whenever adding or removing an instrument.**

                    #''')
                    
        st.header('Data selection')

        st.markdown('''
                    
                   The following options restrict the data used in **all** of the displays.
                   
                   **Telescopes**  
                   
                   This is rather straightforward. Remember the statement above: changing this resets everything else.
                   
                   **Sources**  
                   
                   The Main source display option conditions several elements further down the line:
                       
                  -In Multi-Object mode, a toggle allows to pre-restrict the choice of sources with the 5 main objects with significant detections in the paper. This is a manual restriction, which is not affected by the choice of lines or significance threshold.
                  
                  -The display of all monitoring plots is restricted to the **single object** mode. 
                  
                  
                  **Inclination**  
                  
                  The inclination restrictions is based on the results on the informations in Table 1 of the paper, which can be shown in the "Source parameters" table in the "Tables" tab. Since not all inclination measurements are created equal, the inclination measurement primarily includes dynamical estimates if any, then jet estimates, and finally reflection measurements.
                  
                  -Dipping measurements are considered independantly, which means that a source with "only" dipping reports and no explicit inclination constraints will be ignored if the toggle for including Sources with no information is not checked.
                  
                  -The second toggle allows to disregard sources with uncertainties extending past the currently selected inclination range. Incompatible inclination measurements are considered similarly as a bigger interval.
                  
                  -The third option dashes the upper limit (hexagon) displays of sources incompatible with previous constrain.
                  
                  -The fourth option dashes all sources with no inclination measurements (including dippers with no proper inclination constraints).
                  
                  -The dipper option choice overrides the primary inclination constrain and togglees
                  
                  **Date** 
                  
                  The time interval restriction is deported on the main page for more precise changes but is also a main selection widget affecting all the following elements. Currently, the duration of the observation is not considered.
                  
                  **Significance**
                  
                  The detection significance uses the final MC significance (see the paper for details) of each line. All existing lines must have been added to the models, which means they were at least 99\% significant with the f-test during the automatic fitting procedure. This can still mean very poor MC significance in case of bad data or degenerate components. 
                  
                  The MC tests were done with 1000 computations, which means that using 1. as a significance threshold only restricts to detections above 0.999.
                   
                  #''')
                    
        st.header('Hardness Intensity Diagram (HID)')
        
        st.markdown('''
                    
                   The HID displays the observations in a Hardness/Intensity (in this case Luminosity) diagram, using the standard bands of X-ray measurements. In the default options, detections above the threshold are marked by a full circle, with a size scaling with the Equivalent Width (EW, or depth relative to the continuum) of the line. This does **NOT** necessarily means higher significance. 
                   If displayed, detections below the significance threshold are marker with grey hashed circles.
                   
                   The default options also shows EW upper limits (ULs) with empty hexagons, using two different symbols to aid visually distinguish between the many sources of the sample. In the case of upper limits, **smaller** hexagons means that the observation guarantees no detection **above** this EW value (at a 3 sigma confidence level). In other words, the smaller the hexagon, the more constraining the observation.
                   
                   The main visualisation options allow to display several line or observation parameters as a colormap or color code for the exposures.
                   The "source" option is the only one with 2 different colormaps, a jet for the sources with detections (the same one used in the parameter analysis), and a viridis for the non-detections. We advise restricting the number of sources for easier identification of individual detections.
                   
                   For all line parameters (currently velocity shift and Del-C), the value displayed is the most extremal among the significant lines in this observation. Velocity shifts are considere d from the source point of view, which means that positive values are :red[redshifts] and negative values :blue[blueshifts].
                   
                   If the upper limit option is selected, the user can choice a range of lines, from which the biggest upper limit will be displayed.
                   
                   Various visualisation options allow to zoom on the current set of points in the graph, change coloring options and display errorbars (instead of upper limits).
                   
                   #''')
        
        st.header('Monitoring')
        
        st.markdown('''
                    
                   Whenever the sample selection is restricted to single sources, long-term lightcurves and HR evolution can be displayed using both
                   RXTE-ASM and MAXI data with a 1-day binning.  
                   
                   RXTE data is taken from a local copy of the definitive products available at http://xte.mit.edu/ASM_lc.html. Lightcurves use the sum 
                   of the intensity in all bands ([1.5 − 12] keV), corrected by a factor of 25 to match (visually) MAXI values, and HR values are built 
                   as the ratio of bands C and B+A, i.e. [5.5 − 12]/[1.5 − 5] keV. MAXI data is loaded on the fly from the official website at 
                   http://maxi.riken.jp/top/slist.html, in order to use the latest dataset available. 
                   
                   MAXI lightcurves use the full [2 − 20] keV band and HR built from the [4 − 10]/[2 − 4] bands. A transparency factor proportional to 
                   the quality of the data (estimated from the ratio of the HR value to its uncertainty) is applied to both HRs to aid visibility, and 
                   the dates of exposures from the instruments used in the sample are highlighted. The date restriction selected in the sample selection 
                   can be both highlighted and used to zoom the lightcurve display, and EW values and upper limits can be displayed on a secondary axis 
                   at the date of each exposure in the sample.
                   
                   #''')
                   
        
        st.header('Parameter Analysis')
        
        st.markdown('''
                    The distribution and correlation of line parameters can be computed on the fly from the chosen data selection. 

                    Distributions are restricted to the main line parameters, as well as the number of detections for each line, and
                    can be stacked/split according to sources and instruments. 
                    
                    Scatter plots display correlations between various intrinsic parameters, as well as observation-level and source-level
                    parameters.
                    
                    Besides the 3 main parameters (EW, velocity shift and energy), several additional parameters can be added to both the 
                    distributions and the scatters. 
                    
                    In scatter plots, p-values are computed according to the perturbation method discussed in section 4.1.1 of the paper. 
                    This process is done on the fly, which means that graphs take some time to produce, and the p-values can fluctuate slightly.
                    
                    Similarly to the HID, scatter plots can be color-coded according to various informations, and EW upper
                    limits for currently selected sources can be included in the relevant plots, along with other secondary options.
                    
                    By default, parameter analysis is skipped if only upper limits remain, to avoid additional computing time in situations where it's
                    mostly unneeded. Similarly, restricting the analysis to sources with a detection avoid too much cluttering and better spread in the
                    colormap used for sources.
                    #''')
                    
        
        st.header('Data display and download (Tables)')

        st.markdown('''
                    the complete data of sources, observation and line parameters are displayed according to the current selection made in the sidebar,
                    and can be downloaded through separate csvs file that can be loaded as multi-dimensional dataframes.
                    #''')
                    
#### Transposing the tables into plot arrays

flag_noabsline=False

#bin values for all the histograms below
#for the Velocity shift and energies the range is locked so we can use a global binning for all the diagrams

bins_bshift=np.linspace(-8e3 if 'XMM' in choice_telescope or 'NuSTAR' in choice_telescope else -2e3,
                        3e3,num=26,endpoint=True)
# bins_bshift=np.linspace(-1e4,5e3,num=31,endpoint=True)
bins_ener=np.arange(6.,9.,2*line_search_e[2])

#creating restricted ploting arrays witht the current streamlit object and lines selections
abslines_plot_restrict=deepcopy(abslines_plot)
#re-regularizing the array to have an easier time selecting the axes
abslines_plot_restrict=np.array([[[sss_elem for sss_elem in ss_elem] for ss_elem in s_elem] for s_elem in abslines_plot])
abslines_plot_restrict=np.transpose(np.transpose(abslines_plot_restrict.T[mask_obj],(1,0,2,3))[mask_lines],(3,2,0,1))

#same for abslines_ener
abslines_ener_restrict=deepcopy(abslines_ener)
abslines_ener_restrict=np.array([elem for elem in abslines_ener])
abslines_ener_restrict=np.transpose(np.transpose(abslines_ener_restrict,(1,0,2))[mask_lines].T[mask_obj],(1,2,0))

#and the width was created later in the code's writing so it is already regular
width_plot_restrict=deepcopy(width_plot)
width_plot_restrict=np.transpose(np.transpose(width_plot_restrict,(1,0,2))[mask_lines].T[mask_obj],(1,2,0))

dict_linevis['zoom_lc']=zoom_lc
dict_linevis['abslines_plot_restrict']=abslines_plot_restrict
dict_linevis['bins_bshift']=bins_bshift
dict_linevis['bins_ener']=bins_ener
dict_linevis['display_nonsign']=display_nonsign
dict_linevis['save_dir']=save_dir
dict_linevis['save_str_prefix']= save_str_prefix

'''''''''''''''''''''
####Creating and plotting the dataframes
'''''''''''''''''''''

def produce_df(data,rows, columns, row_names=None, column_names=None,row_index=None,col_index=None):

    """
    rows is a list of lists that will be used to build a MultiIndex
    columns is a list of lists that will be used to build a MultiIndex
    
    Note:
    replaces row_index and col_index by the values provided instead of building them if asked so
    """
    
    if row_index is None:
        row_index_build = pd.MultiIndex.from_product(rows, names=row_names)
    else:
        row_index_build=row_index
        
    if col_index is None:
        col_index_build = pd.MultiIndex.from_product(columns, names=column_names)
    else:
        col_index_build=col_index
        
    return pd.DataFrame(data,index=row_index_build, columns=col_index_build)

        
'''
#OBS & LINE TABLES
'''


if display_single and choice_source[0]=='4U1630-47':

    int_lc_df = fit_int_revol_dict[choice_source[0]]

    int_lc_mjd = np.array([Time(elem).mjd.astype(float) for elem in int_lc_df['ISOT']])

    mask_intime_INT_revol=np.array([Time(elem,format='mjd')>=Time(slider_date[0]) \
                                  and Time(elem,format='mjd')<=Time(slider_date[1])\
                                    for elem in int_lc_mjd])

    if sum(ravel_ragged(mask_intime_plot))>0:

        obs_dates = Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd.astype(
            float)

        mask_intime_INT_withobs=[sum((elem - obs_dates[0]) >= 0)>0 and\
                                  min((elem - obs_dates[0])[(elem - obs_dates[0]) >= 0]) < 3 for elem in int_lc_mjd]


        # computing which observations are within the timeframe of an integral revolution (assuming 3days-long)
        mask_withtime_INT = [False if sum((int_lc_mjd - elem) >= 0)==0 else \
                             min((int_lc_mjd - elem)[(int_lc_mjd - elem) >= 0]) < 3 for elem in obs_dates[0]]

        if restrict_match_INT:
            mask_intime_plot[0]=(mask_intime_plot[0]) & mask_withtime_INT

#and an date order 
order_intime_plot_restrict=np.array([np.array([Time(elem) for elem in date_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)]]).argsort() for i_obj_r in range(n_obj_r)],dtype=object)

#creating  4 dimensionnal dataframes for the observ and line informations

observ_df_list=[]
observ_high_df_list=[]
line_df_list=[]

abs_plot_tr=np.array([[subelem for subelem in elem] for elem in abslines_plot_restrict]).transpose(3,2,0,1)

line_rows=np.array(lines_std_names[3:9])[mask_lines]

for i_obj_r in range(n_obj_r):
    
    n_obs_r=sum(mask_intime_plot[i_obj_r].astype(bool))

    #the transposer order is 3D regularly, and 4D if the array is more regular, aka if there's a single source
    tr_order=(2,0,1,3) if flag_single_obj else (2,0,1)

    #recreating an individual non ragged array (similar to abslines_perobj) in construction for each object
    hid_plot_indiv=np.array([[subelem for subelem in elem] for elem in hid_plot_restrict.transpose(tr_order)[i_obj_r]],
                            dtype=float)

    if flag_single_obj:
        hr_high_plot_indiv=np.array([elem for elem in np.transpose(hr_high_plot_restrict, (1, 0, 2))[i_obj_r]],
                                    dtype=float)
        lum_high_plot_indiv = np.array([elem for elem in np.transpose(lum_high_plot_restrict, (1, 0, 2))[i_obj_r]],
                                     dtype=float)
        BAT_lc_plot_indiv = np.array([elem for elem in np.transpose(BAT_lc_plot_restrict, (1, 0, 2))[i_obj_r]],
                                     dtype=float)

    else:
        hr_high_plot_indiv=np.array([elem for elem in hr_high_plot_restrict.T[i_obj_r]],dtype=float)
        lum_high_plot_indiv=np.array([elem for elem in lum_high_plot_restrict.T[i_obj_r]],dtype=float)

        BAT_lc_plot_indiv=np.array([elem for elem in BAT_lc_plot_restrict.T[i_obj_r]],dtype=float)

    #this one is always 2D
    BAT_expos_plot_indiv=BAT_expos_plot_restrict[i_obj_r]

    #this one is put back to non-Eddington values, and we remove the first flux measurement whose band can change between instruments or computations
    flux_plot_indiv=np.array([[subelem for subelem in elem] for elem in lum_plot_restrict[1:].transpose(tr_order)[i_obj_r]],dtype=float)/Edd_factor_restrict[i_obj_r]

    line_plot_indiv=np.array([[[subsubelem for subsubelem in subelem] for subelem in elem] for elem in abs_plot_tr[i_obj_r]],dtype=float)

    #applying the intime mask on each observation and sorting by date
    hid_plot_indiv=hid_plot_indiv.transpose(2,0,1)[mask_intime_plot[i_obj_r].astype(bool)]\
                    [order_intime_plot_restrict[i_obj_r].astype(int)].transpose(1,2,0)
    flux_plot_indiv=flux_plot_indiv.transpose(2,0,1)[mask_intime_plot[i_obj_r].astype(bool)]\
                    [order_intime_plot_restrict[i_obj_r].astype(int)].transpose(1,2,0)
        
    line_plot_indiv=line_plot_indiv.transpose(3,0,1,2)[mask_intime_plot[i_obj_r].astype(bool)]\
                    [order_intime_plot_restrict[i_obj_r].astype(int)].transpose(2,3,0,1)

    hr_high_plot_indiv=hr_high_plot_indiv.T[mask_intime_plot[i_obj_r].astype(bool)]\
                        [order_intime_plot_restrict[i_obj_r].astype(int)].T
    lum_high_plot_indiv = lum_high_plot_indiv.T[mask_intime_plot[i_obj_r].astype(bool)]\
                          [order_intime_plot_restrict[i_obj_r].astype(int)].T
    BAT_lc_plot_indiv = BAT_lc_plot_indiv.T[mask_intime_plot[i_obj_r].astype(bool)]\
                          [order_intime_plot_restrict[i_obj_r].astype(int)].T

    BAT_expos_plot_indiv=BAT_expos_plot_indiv[mask_intime_plot[i_obj_r].astype(bool)]\
                                             [order_intime_plot_restrict[i_obj_r].astype(int)]

    '''
    # splitting information to take off 1 dimension and only take specific information    
    # EW, bshift, width, flux, sign, upper
    '''
    used_indexes=[[0,0],[0,1],[0,2],
                  [1,0],[1,1],[1,2],
                  [7,0],[7,1],[7,2],
                  [3,0],[3,1],[3,2],
                  [4,0],
                  [5,0]]
                  
    line_plot_indiv=np.array([line_plot_indiv[elem[0]][elem[1]] for elem in used_indexes])
    
    #creating the row indexes (row: object, subrow: observation)    
    observ_list_indiv=observ_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)][order_intime_plot_restrict[i_obj_r].astype(int)].tolist()

    instru_list_indiv=instru_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)][order_intime_plot_restrict[i_obj_r].astype(int)].tolist()

    observ_list_indiv=[elem.replace('_Imaging_auto','').replace('_Timing_auto','').replace('_heg_-1','').replace('_heg_1','') for elem
                       in observ_list_indiv]
    
    iter_rows=[[obj_list[mask_obj][i_obj_r]],
               instru_list_indiv,
               observ_list_indiv,
               date_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)][order_intime_plot_restrict[i_obj_r].astype(int)].tolist()]
    
    
    #creating the iter index manually because we have two clumns (observ and time) not being dimensions of one another
    row_index_arr_obs=np.array([[iter_rows[0][0],iter_rows[1][i_obs_r],iter_rows[2][i_obs_r],iter_rows[3][i_obs_r]]\
                                for i_obs_r in range(n_obs_r)]).T
    
    row_index_arr_line=np.array([[iter_rows[0][0],iter_rows[1][i_obs_r],iter_rows[2][i_obs_r],iter_rows[3][i_obs_r],
                                  line_rows[i_line_r]]\
                                for i_obs_r in range(n_obs_r) for i_line_r in range(sum(mask_lines))]).T
    
    row_index_obs=pd.MultiIndex.from_arrays(row_index_arr_obs,names=['Source','Instrument','obsid','date'])
    
    row_index_line=pd.MultiIndex.from_arrays(row_index_arr_line,names=['Source','Instrument','obsid','date','line'])

    #you can use the standard way for columns for the observ df
    iter_columns=[['HR [6-10]/[3-10]','L_3-10/LEdd','Flux_3-6','Flux_6-10','Flux_1-3','Flux_3-10'],
                  ['main','90% err-','90% err+']]

    iter_columns_high=[['HR [15-50]/[3-6]','L_15-50/LEdd','Flux_15-50','BAT_rate_15-50','BAT_expos_coded'],
                       ['main','90% err-','90% err+']]

    #but not for the line df
    column_index_arr_line=np.array([['EW','main'],['EW','err-'],['EW','err+'],
                          ['blueshift','main'],['blueshift','err-'],['blueshift','err+'],
                          ['width','main'],['width','err-'],['width','err+'],
                          ['line flux','main'],['line flux','err-'],['line flux','err+'],
                          ['sign','main'],['EW UL','main']]).T
                          
    column_index_line=pd.MultiIndex.from_arrays(column_index_arr_line,names=['measure','value'])

    observ_col=np.concatenate([hid_plot_indiv,flux_plot_indiv]).transpose(2,0,1)
    observ_col_reshaped=observ_col.reshape(n_obs_r,len(iter_columns[0])*len(iter_columns[1]))

    observ_col_high=np.array([hr_high_plot_indiv,lum_high_plot_indiv,lum_high_plot_indiv/Edd_factor_restrict[i_obj_r],
                              BAT_lc_plot_indiv,
                              np.array([BAT_expos_plot_indiv,
                                        np.zeros(len(BAT_expos_plot_indiv)),
                                        np.zeros(len(BAT_expos_plot_indiv))])])\
                    .transpose(2,0,1)

    observ_col_high_reshaped=observ_col_high.reshape(n_obs_r,len(iter_columns_high[0])*len(iter_columns_high[1]))

    #creating both dataframes, with a reshape in 2 dimensions (one for the lines and one for the columns)
    #switching to str type allows to display low values correctly
    curr_df_observ=produce_df(observ_col_reshaped,iter_rows,iter_columns,
                              row_names=['Source','Instrument','ObsID','date'],
                                column_names=['measure','value'],row_index=row_index_obs).astype(str)

    curr_df_observ_high=produce_df(observ_col_high_reshaped, iter_rows,iter_columns_high,
                                   row_names=['Source','Instrument','ObsID','date'],
                                column_names=['measure','value'],row_index=row_index_obs).astype(str)

    #removing the last two columns of the high-E since the exposure is only a main value
    curr_df_observ_high=curr_df_observ_high[curr_df_observ_high.columns[:-2]]

    pd.set_option('display.float_format', lambda x: '%.3e' % x)

    observ_df_list+=[curr_df_observ]

    observ_high_df_list+=[curr_df_observ_high]

    line_df_list+=[produce_df(line_plot_indiv.transpose(1,2,0).reshape(n_obs_r*sum(mask_lines),14),None,None,row_names=None,
                            column_names=None,row_index=row_index_line,col_index=column_index_line).astype(str)]

if no_obs:
    observ_df=None
    line_df=None
    observ_high_df_list=None
else:
    observ_df=pd.concat(observ_df_list)
    line_df=pd.concat(line_df_list)
    observ_high_df=pd.concat(observ_high_df_list)
        
with tab_source_df:
    
    with st.expander('Observation parameters'):

        tab_observ_soft,tab_observ_high=st.tabs(('Soft X-rays','Hard X-rays'))

        if no_obs:
            st.warning('No points remaining with current sample/date selection')

        else:
            with tab_observ_soft:

                #the format is offset by 3 because we shift by the number of columns with row names
                st.dataframe(observ_df,use_container_width=True,column_config={\
                             i:st.column_config.NumberColumn(format='%.3e') for i in range(4,len(observ_df.columns)+4)})

                csv_observ= convert_df(observ_df)

                st.download_button(
                    label="Download soft X-ray infos as CSV",
                    data=csv_observ,
                    file_name='observ_table.csv',
                    mime='text/csv',
                )

            with tab_observ_high:
                # the format is offset by 3 because we shift by the number of columns with row names
                st.dataframe(observ_high_df, use_container_width=True, column_config={ \
                    i: st.column_config.NumberColumn(format='%.3e') for i in range(4, len(observ_df.columns) + 4)})

                csv_observ_high = convert_df(observ_high_df)

                st.download_button(
                    label="Download hard X-ray infos as CSV",
                    data=csv_observ_high,
                    file_name='observ_high_table.csv',
                    mime='text/csv',
                )


    if display_single:
        with st.expander('Monitoring'):

            if not choice_source[0] in lc_int_sw_dict.keys() and not choice_source[0] in fit_int_revol_dict.keys():
                st.warning('No INTEGRAL Coverage for this source')

            else:
                radio_mask_INT_monit_withtime=st.toggle\
                    ('Restrict table display to INTEGRAL revolutions simultaneous with observations')
                if radio_mask_INT_monit_withtime:
                    mask_intime_INT_revol=(mask_intime_INT_revol) & (mask_intime_INT_withobs)

                if choice_source[0] in fit_int_revol_dict.keys():
                    #note: here formatting the column is necessary to display the flux correctly
                    st.dataframe(fit_int_revol_dict[choice_source[0]][mask_intime_INT_revol],use_container_width=True,
                                 column_config={ \
                                     i: st.column_config.NumberColumn(format='%.3e')\
                                     for i in range(len(fit_int_revol_dict[choice_source[0]].columns)-5,\
                                                    len(fit_int_revol_dict[choice_source[0]].columns)+1)})

                # if choice_source[0] in lc_int_sw_dict.keys():
                #     lc_int_sw_revol=[elem[:4] for elem in np.array(lc_int_sw_dict[choice_source[0]]['scw'])]
                #
                #     mask_intime_INT_sw=[elem in np.array(fit_int_revol_dict[choice_source[0]][mask_intime_INT_revol]['revolution'],
                #                       dtype='str') for elem in lc_int_sw_revol]
                #
                #     st.dataframe(lc_int_sw_dict[choice_source[0]][mask_intime_INT_sw],use_container_width=True)


    with st.expander('Line parameters'):

        if no_obs:
            st.warning('No points remaining with current sample/date selection')

        else:
            #the format is offset by 4 because we shift by the number of columns with row names
            st.dataframe(line_df,use_container_width=True,column_config={\
                         i:st.column_config.NumberColumn(format='%.3e') for i in range(5,len(line_df.columns)+5)})

            csv_observ= convert_df(line_df)

            st.download_button(
                label="Download as CSV",
                data=csv_observ,
                file_name='line_table.csv',
                mime='text/csv',
            )

    with st.expander('Absorption lines in the literature'):
        with st.spinner('Loading PDF'):
            pdf_viewer(os.path.join(project_dir, 'observations/visualisation/visual_line_dumps/sources_tables.pdf'),
                   pages_to_render=[5,6, 7, 8],width=1400,render_text=True)

        # st.markdown(state_table_footnotes)
        # st.latex(state_table_latex_disp)
        #
        # st.markdown('References :')
        # st.markdown(state_table_biblio_str)



'''''''''''''''''''''
 ####Monitoring
'''''''''''''''''''''

dict_linevis['no_obs']=no_obs

with tab_monitoring:
    if plot_lc_monit:

        if not display_single:
            st.info('Lightcurve monitoring plots are restricted to single source mode.')

        else:
            with st.spinner('Building lightcurve...'):
                fig_lc_monit=plot_lightcurve(dict_linevis,catal_maxi_df,catal_maxi_simbad,choice_source,
                                             catal_bat_df,catal_bat_simbad,
                                             lc_integral_sw_dict=lc_int_sw_dict,fit_integral_revol_dict=fit_int_revol_dict,
                                             display_hid_interval=monit_highlight_hid,
                                                 superpose_ew=plot_maxi_ew,dict_rxte=dict_lc_rxte,
                                             binning=monit_binning)

                #wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_lc_monit is not None:
                    st.pyplot(fig_lc_monit)

    if plot_hr_monit:

        if not display_single:
            st.info('HR monitoring plots are restricted to single source mode.')

        else:
            with st.spinner('Building soft HR evolution...'):
                fig_hr_soft_monit=plot_lightcurve(dict_linevis,catal_maxi_df,catal_maxi_simbad,choice_source,
                                                  catal_bat_df, catal_bat_simbad,
                                             lc_integral_sw_dict=lc_int_sw_dict,fit_integral_revol_dict=fit_int_revol_dict,
                                             mode='HR_soft',display_hid_interval=monit_highlight_hid,
                                                 superpose_ew=plot_maxi_ew,dict_rxte=dict_lc_rxte,
                                             binning=monit_binning)
                # fig_maxi_lc_html = mpld3.fig_to_html(fig_maxi_lc)
                # components.html(fig_maxi_lc_html,height=500,width=1000)

                #wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_hr_soft_monit is not None:
                    st.pyplot(fig_hr_soft_monit)

            with st.spinner('Building hard HR evolution...'):
                fig_hr_hard_monit = plot_lightcurve(dict_linevis, catal_maxi_df, catal_maxi_simbad, choice_source,
                                                    catal_bat_df, catal_bat_simbad,
                                             lc_integral_sw_dict=lc_int_sw_dict,fit_integral_revol_dict=fit_int_revol_dict,
                                               mode='HR_hard',
                                               display_hid_interval=monit_highlight_hid,
                                               superpose_ew=plot_maxi_ew, dict_rxte=dict_lc_rxte,
                                             binning=monit_binning)
                # fig_maxi_lc_html = mpld3.fig_to_html(fig_maxi_lc)
                # components.html(fig_maxi_lc_html,height=500,width=1000)

                # wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_hr_hard_monit is not None:
                    st.pyplot(fig_hr_hard_monit)

    if plot_lc_bat:

        if not display_single:
            st.info('BAT monitoring plots are restricted to single source mode.')

        else:
            with st.spinner('Building BAT lightcurve...'):
                fig_lc_bat= plot_lightcurve(dict_linevis, catal_maxi_df, catal_maxi_simbad, choice_source,
                                               catal_bat_df, catal_bat_simbad,mode='BAT',
                                             lc_integral_sw_dict=lc_int_sw_dict,fit_integral_revol_dict=fit_int_revol_dict,
                                               display_hid_interval=monit_highlight_hid,
                                               superpose_ew=plot_maxi_ew, dict_rxte=dict_lc_rxte,
                                             binning=monit_binning)

                # wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_lc_bat is not None:
                    st.pyplot(fig_lc_bat)

    if plot_lc_integral:

        if not display_single:
            st.info('INTEGRAL monitoring plots are restricted to single source mode.')

        else:
            with st.spinner('Building Integral IBIS lightcurve...'):
                fig_lc_bat= plot_lightcurve(dict_linevis, catal_maxi_df, catal_maxi_simbad, choice_source,
                                               catal_bat_df, catal_bat_simbad,mode='INTEGRAL_'+radio_integral_band,
                                             lc_integral_sw_dict=lc_int_sw_dict,fit_integral_revol_dict=fit_int_revol_dict,
                                            dist_factor=dist_factor_restrict[0],
                                               display_hid_interval=monit_highlight_hid,
                                               superpose_ew=plot_maxi_ew, dict_rxte=dict_lc_rxte,
                                             binning=integral_binning)

                # wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_lc_bat is not None:
                    st.pyplot(fig_lc_bat)

    if not plot_lc_monit and not plot_hr_monit and not plot_lc_bat and not plot_lc_integral:
        st.info('In single source mode, select a monitoring option in the sidebar to plot lightcurves and HR evolutions of the selected object')

    if ((plot_lc_monit and fig_lc_monit is None) or (plot_hr_monit and fig_hr_soft_monit is None)) and display_single:
        st.warning('No match in MAXI/RXTE source list found.')

    if plot_lc_bat and fig_lc_bat is None:
        st.warning('No match in BAT transient list found.')

with expander_monit:

    #global lc download
    if display_single:

        #creating a list of the lc plots objects
        fig_list=[]
        fig_name_list=[]
        if fig_lc_monit is not None:
            fig_list+=[fig_lc_monit]
            fig_name_list+=[choice_source[0]+'fig_lc_monit.svg']
        if fig_hr_soft_monit is not None:
            fig_list+=[fig_hr_soft_monit]
            fig_name_list+=[choice_source[0]+'fig_hr_soft_monit.svg']
        if fig_hr_hard_monit is not None:
            fig_list+=[fig_hr_hard_monit]
            fig_name_list+=[choice_source[0]+'fig_hr_hard_monit.svg']
        if fig_lc_bat is not None:
            fig_list+=[fig_lc_bat]
            fig_name_list += [choice_source[0]+'fig_lc_bat.svg']
        if fig_lc_int is not None:
            fig_list+=[fig_lc_int]
            fig_name_list+=[choice_source[0]+'fig_lc_int.svg']

        if len(fig_list)>=1:
            #saving them into a list of byte objects
            fig_io_list=[io.BytesIO() for i in range(len(fig_list))]
            for elem_fig,elem_io in zip(fig_list,fig_io_list):
                elem_fig.savefig(elem_io,format='svg')

            #converting into a zip
            zip_io=make_zip(fig_io_list,fig_name_list)

            st.download_button(
                label="Download ZIP of monitoring figures",
                data=zip_io,
                file_name=choice_source[0]+'_monit.zip',
                mime='text/csv')

'''''''''''''''''''''
   #### Parameter analysis
'''''''''''''''''''''

with st.sidebar.expander('Parameter analysis'):
    
    display_param_withdet=st.toggle('Restrict parameter analysis to sources with significant detections',value=True)
    
    display_param=st.multiselect('Additional parameters',
                                 ('EW ratio (Line)','width (Line)','Line flux (Line)','Time (Observation)',
                                  'Line EW comparison',
                                  'High Energy parameters (Observation)'),default=None)

    use_ewratio = 'EW ratio (Line)' in display_param
    display_scat_ewcomp=  'Line EW comparison' in display_param
    if use_ewratio and sum(mask_lines)<=1:
        st.warning('Cannot build EW ratio with current line restriction')
        use_ewratio=False

    if display_scat_ewcomp and sum(mask_lines)<=1:
        st.warning('Cannot compare Line EW with current line restriction')
        display_scat_ewcomp=False

    glob_col_source=st.toggle('Normalize source colors over the entire sample',value=True)

    st.header('Distributions')
    display_distrib=st.toggle('Plot distributions',value=False)
    use_distrib_lines=st.toggle('Show line by line distribution',value=True)
    split_distrib=st.radio('Split distributions:',('Off','Source','Instrument'),index=1)
    split_dist_method=st.toggle('Split distributions horizontally',value=False)

    if split_distrib=='Source' and (display_single or sum(mask_obj)==1):
            split_distrib='Off'
            
    st.header('Correlations')
    
    display_types=st.multiselect('Main parameters',('Line','Observation','Source'),default=None)

    display_scat_intr='Line' in display_types
    display_scat_hid='Observation' in display_types
    display_scat_inclin='Source' in display_types
    
    display_scat_ewcomp='Line EW comparison' in display_param
    if display_scat_ewcomp:        
        ewratio_comp=st.multiselect('Lines to compare', [elem for elem in lines_std_names[3:9] if 'abs' in elem],default=lines_std_names[3:5])
        


    if use_ewratio:
        ewratio_strs=np.array(['Fe XXVI Ka/Fe XXV Ka','FeXXVI Kb/Fe XXV Kb','FeXXV Kb/Fe XXV Ka','FeXXVI Kb/Fe XXVI Ka'])
        ewratio_type_str=st.selectbox('Ratio to use',ewratio_strs)
        ewratio_type=str(np.array(['Ka','Kb','25','26'])[ewratio_strs==ewratio_type_str][0])

    use_width='width (Line)' in display_param
    if use_width:
        display_th_width_ew=st.toggle('Display theoretical individual width vs EW evolution',value=False)
    else:
        display_th_width_ew=False
        
    use_time_param='Time (Observation)' in display_param
    use_lineflux='Line flux (Line)' in display_param
    use_high_E_param='High Energy parameters (Observation)' in display_param

    st.header('Computations')

    compute_correl = st.toggle('Compute Pearson/Spearman for the scatter plots', value=False)
    compute_regr = st.toggle('Compute linear regression in strongly correlated graphs ', value=False)
    regr_pval_threshold=st.number_input('p-value treshold to compute the regression',
                                        format='%.3e',value=1e-5)

    restrict_comput_scatter = st.toggle('Restrict computation bounds', value=False)
    if restrict_comput_scatter:
        comput_scatter_xmin = st.number_input(r'$x_{min}$',format='%.3e')
        comput_scatter_xmax = st.number_input(r'$x_{max}$',format='%.3e')
        comput_scatter_ymin = st.number_input(r'$y_{min}$',format='%.3e')
        comput_scatter_ymax = st.number_input(r'$y_{max}$',format='%.3e')
    else:
        comput_scatter_xmin = 0
        comput_scatter_xmax = 0
        comput_scatter_ymin = 0
        comput_scatter_ymax = 0

    
    st.header('Visualisation')
    radio_color_scatter_options=np.array(['None','Source','Instrument','Time',
                                          r'line $FWHM$',r'$nH$',r'$HR_{soft}$',r'$L_{3-10}$']\
                                         + (['Custom: Line substructures','Custom: accretion states','Custom: ionization'] if \
                                         display_single and obj_list[mask_obj_select][0]=='4U1630-47' else [])\
                                         + (['Custom: Outbursts'] if display_single else []))

    color_scatter_options=['None','Source','Instrument','Time','width','nH','HR','L_3-10']\
                        + (['custom_line_struct','custom_acc_states','custom_ionization'] if\
                            display_single and obj_list[mask_obj_select][0]=='4U1630-47' else [])\
                        + (['custom_outburst'] if display_single else [])


    radio_color_scatter=st.radio('Scatter plot color options:',radio_color_scatter_options,index=1)
    color_scatter= np.array(color_scatter_options)[radio_color_scatter_options==radio_color_scatter][0]

    scale_log_ew=st.toggle('Use a log scale for the equivalent width and line fluxes')
    scale_log_hr=st.toggle('Use a log scale for the HID parameters',value=True)
    display_std_abserr_bshift=st.toggle('Display mean and std of Chandra velocity shift distribution',value=True)
    display_abserr_bshift=st.toggle('Display mean and std of current velocity shift distribution',value=False)
    common_observ_bounds_dates=st.toggle('Use common observation parameter bounds independantly of the dates restriction',
                                   value=False)
    common_observ_bounds_lines=st.toggle('Use common observation parameter bounds for all lines',value=True)
    display_pearson = st.toggle('Display Pearson rank', value=False)
    display_legend_correl=st.toggle('Display legend',value=True)

    st.header('Upper limits')
    show_scatter_ul=st.toggle('Display upper limits in EW plots',value=False)
    lock_lims_det=not(st.toggle('Include upper limits in graph bounds computations',value=True))
    use_alpha_ul=st.toggle('Use transparent upper limits',value=True)

    st.header('Non-standard plots')
    plot_gamma_correl=st.toggle('Plot powerlaw index correlations',value=False)
    plot_physical_compa=st.toggle('Plot physical parameter correlations for the global sample',value=False)

if compute_only_withdet:

    if no_obs or sum(global_mask_intime_norepeat)==0 or sum(global_sign_mask)==0:
        if no_obs:
            with tab_param:
                st.warning('No detections for current object/date selection. Cannot compute parameter analysis.')

        elif sum(global_mask_intime_norepeat)==0:
            with tab_param:
                st.warning('No point left in selected dates interval. Cannot compute parameter analysis.')
        elif sum(global_sign_mask)==0:
            with tab_param:
                st.warning('No detections for current object/date selection. Cannot compute parameter analysis.')
        st.stop()
        
#overwriting the objet mask with sources with detection for parameter analysis if asked to
if display_param_withdet:
    
    mask_obj=(np.array([elem in sources_det_dic for elem in obj_list])) & (mask_obj_base)

    if sum(mask_obj)==0:
        st.warning('No detections for current object/date selection. Cannot compute parameter analysis.')
        st.stop()
        
    #recreating restricted ploting arrays with the current streamlit object and lines selections
    abslines_plot_restrict = deepcopy(abslines_plot)
    # re-regularizing the array to have an easier time selecting the axes
    abslines_plot_restrict = np.array(
        [[[sss_elem for sss_elem in ss_elem] for ss_elem in s_elem] for s_elem in abslines_plot])
    abslines_plot_restrict = np.transpose(np.transpose(abslines_plot_restrict.T[mask_obj], (1, 0, 2, 3))[mask_lines],
                                          (3, 2, 0, 1))

    # same for abslines_ener
    abslines_ener_restrict = deepcopy(abslines_ener)
    abslines_ener_restrict = np.array([elem for elem in abslines_ener])
    abslines_ener_restrict = np.transpose(np.transpose(abslines_ener_restrict, (1, 0, 2))[mask_lines].T[mask_obj],
                                       (1, 2, 0))

    # and the width was created later in the code's writing so it is already regular
    width_plot_restrict = deepcopy(width_plot)
    width_plot_restrict = np.transpose(np.transpose(width_plot_restrict, (1, 0, 2))[mask_lines].T[mask_obj], (1, 2, 0))

    if flag_single_obj:
        hid_plot_restrict = hid_plot
        lum_plot_restrict = lum_plot
    else:
        hid_plot_restrict = hid_plot.T[mask_obj].T
        lum_plot_restrict = lum_plot.T[mask_obj].T

    incl_plot_restrict=incl_plot[mask_obj]


    #make a flux like this
    #and passing in the dictionnary for use in the functions
    dict_linevis['mask_obj']=mask_obj
    dict_linevis['abslines_plot_restrict']=abslines_plot_restrict
    dict_linevis['abslines_ener_restrict']=abslines_ener_restrict
    dict_linevis['width_plot_restrict']=width_plot_restrict
    dict_linevis['hid_plot_restrict']=hid_plot_restrict
    dict_linevis['incl_plot_restrict']=incl_plot_restrict

dict_linevis['display_pearson']=display_pearson
dict_linevis['display_abserr_bshift']=display_abserr_bshift
dict_linevis['display_std_abserr_bshift']=display_std_abserr_bshift
dict_linevis['glob_col_source']=glob_col_source
dict_linevis['display_th_width_ew']=display_th_width_ew
dict_linevis['common_observ_bounds_lines']=common_observ_bounds_lines
dict_linevis['common_observ_bounds_dates']=common_observ_bounds_dates
dict_linevis['use_alpha_ul']=use_alpha_ul
dict_linevis['display_legend_correl']=display_legend_correl

dict_linevis['split_dist_method']=split_dist_method

os.system('mkdir -p '+save_dir+'/graphs')
os.system('mkdir -p '+save_dir+'/graphs/distrib')
os.system('mkdir -p '+save_dir+'/graphs/intrinsic')
os.system('mkdir -p '+save_dir+'/graphs/hid')
os.system('mkdir -p '+save_dir+'/graphs/inclin')

'''
# AUTOFIT LINES
'''

'''
#Distributions
'''

def streamlit_distrib():
    distrib_ew=distrib_graph(abslines_plot_restrict,'ew',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text,split=split_distrib)
    distrib_bshift=distrib_graph(abslines_plot_restrict,'bshift',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text,split=split_distrib)
    distrib_ener=distrib_graph(abslines_plot_restrict,'ener',dict_linevis,abslines_ener_restrict,conf_thresh=slider_sign,streamlit=True,
                               bigger_text=bigger_text,split=split_distrib)
    if use_ewratio:
        distrib_ewratio=distrib_graph(abslines_plot_restrict,'ewratio'+ewratio_type,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                       bigger_text=bigger_text,split=split_distrib)
        
    if n_infos>=5 and use_lineflux:
        distrib_lineflux=distrib_graph(abslines_plot_restrict,'lineflux',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text,split=split_distrib)
        
    if use_width:
        distrib_width=distrib_graph(abslines_plot_restrict,'width',dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                       bigger_text=bigger_text,split=split_distrib)
        
    if use_distrib_lines:
        distrib_lines=distrib_graph(abslines_plot_restrict,'lines',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text,split=split_distrib)
        
    with tab_param:
        with st.expander('Distribution graphs'):
            
            col_list={'ew':None,'bshift':None,'ener':None}
            
            if use_ewratio:
                col_list['ewratio']=None
            if n_infos>=5 and use_lineflux:
                col_list['lineflux']=None
    
            if use_distrib_lines:
                col_list['lines']=None
                
            if use_width:
                col_list['width']=None
                
            st_cols=st.columns(len(col_list))
            
            for i_col,col_name in enumerate(list(col_list.keys())):
                col_list[col_name]=st_cols[i_col]
                
            with col_list['ew']:
                st.pyplot(distrib_ew)
    
            with col_list['bshift']:
                st.pyplot(distrib_bshift)
            with col_list['ener']:
                st.pyplot(distrib_ener)
                    
            if use_ewratio:
                with col_list['ewratio']:
                    st.pyplot(distrib_ewratio)
            
            if use_lineflux and n_infos>=5:
                with col_list['lineflux']:
                    st.pyplot(distrib_lineflux)
            
            if use_distrib_lines:
                with col_list['lines']:
                    st.pyplot(distrib_lines)
                    
            if use_width:
                with col_list['width']:
                    st.pyplot(distrib_width)
                
'''
#1-1 Correlations
'''

def streamlit_scat(mode):
    
    scat_ew=[]
    scat_bshift=[]
    scat_ener=[]
    scat_width=[]
    
    if mode=='ewcomp':
        scat_ew=[correl_graph(abslines_plot_restrict,ewratio_comp[0]+'_'+ewratio_comp[1],abslines_ener_restrict,dict_linevis,mode='ewcomp',
                               conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                               show_ul_ew=show_scatter_ul)]
        if use_time_param and use_ewratio:
            #not actually bshift but we keep the same column names
            scat_bshift=[correl_graph(abslines_plot_restrict,'time_ewratio'+ewratio_type,abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,
                                        mode='observ',conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            
    if mode=='intrinsic':
        scat_ew=[correl_graph(abslines_plot_restrict,'bshift_ew',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_bshift=[correl_graph(abslines_plot_restrict,'ener_ew',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                               compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_ener=[correl_graph(abslines_plot_restrict,'ener_bshift',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                 compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if n_infos>=5 and use_lineflux:
            scat_lineflux=[correl_graph(abslines_plot_restrict,'lineflux_bshift',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,
                                        streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
            
        if use_time_param:
            scat_ew+=[correl_graph(abslines_plot_restrict,'time_ew',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            scat_bshift+=[correl_graph(abslines_plot_restrict,'time_bshift',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            scat_ener+=[correl_graph(abslines_plot_restrict,'time_ener',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            if use_width:
                scat_width+=[correl_graph(abslines_plot_restrict,'time_width',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            
        if use_width:
            scat_ew+=[correl_graph(abslines_plot_restrict,'ew_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            scat_bshift+=[correl_graph(abslines_plot_restrict,'bshift_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            scat_ener+=[correl_graph(abslines_plot_restrict,'ener_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_ew=show_scatter_ul)]
            if use_ewratio:
                scat_width=[correl_graph(abslines_plot_restrict,'ewratio'+ewratio_type+'_width1',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                            show_linked=show_linked,show_ul_ew=show_scatter_ul)]
                scat_ewratio=[correl_graph(abslines_plot_restrict,'ewratio'+ewratio_type+'_width2',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                            show_linked=show_linked,show_ul_ew=show_scatter_ul)]
    elif mode=='observ':
        scat_ew=\
        [correl_graph(abslines_plot_restrict,'ew_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',conf_thresh=slider_sign,
                      streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,show_ul_ew=show_scatter_ul),
         correl_graph(abslines_plot_restrict,'ew_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',conf_thresh=slider_sign,
                      streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,show_ul_ew=show_scatter_ul)]

        scat_bshift=\
            [correl_graph(abslines_plot_restrict,'bshift_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'bshift_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
            
        scat_ener=\
        [correl_graph(abslines_plot_restrict,'ener_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
         correl_graph(abslines_plot_restrict,'ener_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]

        if use_high_E_param:
            scat_ew += \
                [correl_graph(abslines_plot_restrict, 'ew_nthcomp-gamma', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ', conf_thresh=slider_sign,
                              streamlit=True, compute_correl=compute_correl, bigger_text=bigger_text,
                              show_linked=show_linked, show_ul_ew=show_scatter_ul),
                 correl_graph(abslines_plot_restrict, 'ew_highE-HR', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ', conf_thresh=slider_sign,
                              streamlit=True, compute_correl=compute_correl, bigger_text=bigger_text,
                              show_linked=show_linked, show_ul_ew=show_scatter_ul),
                 correl_graph(abslines_plot_restrict, 'ew_highE-flux', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ', conf_thresh=slider_sign,
                              streamlit=True, compute_correl=compute_correl, bigger_text=bigger_text,
                              show_linked=show_linked, show_ul_ew=show_scatter_ul)]

            scat_bshift += \
                [correl_graph(abslines_plot_restrict, 'bshift_nthcomp-gamma', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ',
                              conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                              bigger_text=bigger_text, show_linked=show_linked),
                 correl_graph(abslines_plot_restrict, 'bshift_highE-HR', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ',
                              conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                              bigger_text=bigger_text, show_linked=show_linked),
                 correl_graph(abslines_plot_restrict, 'bshift_highE-flux', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ',
                              conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                              bigger_text=bigger_text, show_linked=show_linked)]

            scat_ener += \
                [correl_graph(abslines_plot_restrict, 'ener_nthcomp-gamma', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ',
                              conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                              bigger_text=bigger_text, show_linked=show_linked),
                 correl_graph(abslines_plot_restrict, 'ener_highE-HR', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ',
                              conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                              bigger_text=bigger_text, show_linked=show_linked),
                 correl_graph(abslines_plot_restrict, 'ener_highE-flux', abslines_ener_restrict, dict_linevis,
                              mode_vals=hid_plot_restrict, mode='observ',
                              conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                              bigger_text=bigger_text, show_linked=show_linked)]

        if use_width:
            
            scat_width=\
            [correl_graph(abslines_plot_restrict,'width_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'width_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
        if use_ewratio:
            scat_ewratio=\
            [correl_graph(abslines_plot_restrict,'ewratio'+ewratio_type+'_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                  conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                  show_ul_ew=show_scatter_ul),
            correl_graph(abslines_plot_restrict,'ewratio'+ewratio_type+'_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                  conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                  show_ul_ew=show_scatter_ul)]

            if use_high_E_param:
                scat_ewratio+=[correl_graph(abslines_plot_restrict,'ewratio'+ewratio_type+'_nthcomp-gamma',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                          show_ul_ew=show_scatter_ul),
                correl_graph(abslines_plot_restrict, 'ewratio' + ewratio_type + '_highE-HR',
                             abslines_ener_restrict, dict_linevis, mode_vals=hid_plot_restrict,
                             mode='observ',
                             conf_thresh=slider_sign, streamlit=True, compute_correl=compute_correl,
                             bigger_text=bigger_text, show_linked=show_linked,
                             show_ul_ew=show_scatter_ul),
                  correl_graph(abslines_plot_restrict,'ewratio'+ewratio_type+'_highE-flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                          show_ul_ew=show_scatter_ul)]

        if n_infos>=5 and use_lineflux:
            scat_lineflux=\
            [correl_graph(abslines_plot_restrict,'lineflux_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'lineflux_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
    elif mode=='source':
        scat_ew+=[correl_graph(abslines_plot_restrict,'ew_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                                show_ul_ew=show_scatter_ul)]
        
        scat_bshift+=[correl_graph(abslines_plot_restrict,'bshift_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                               conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
        scat_ener+=[correl_graph(abslines_plot_restrict,'ener_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                 conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if use_width:
            scat_width+=[correl_graph(abslines_plot_restrict,'width_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                     conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if n_infos>=5:
            scat_lineflux=[correl_graph(abslines_plot_restrict,'lineflux_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,
                                        mode='source',conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked)]
    
    with tab_param:
        with st.expander('Correlation graphs for '+('line' if mode=='intrinsic' else mode)+' parameters'):
    
            col_list={'ew':None,'bshift':None,'ener':None}
            
            if use_width:
                col_list['width']=None
            
            #defining columns for each data type
            if use_ewratio and (mode=='observ' or use_width):
                col_list['ewratio']=None
            if n_infos>=5 and use_lineflux:
                col_list['lineflux']=None
            st_cols=st.columns(len(col_list))
            
            for i_col,col_name in enumerate(list(col_list.keys())):
                col_list[col_name]=st_cols[i_col]
            
            with col_list['ew']:
                pholder_ew=[st.pyplot(elem) for elem in scat_ew]
                            
            if mode!='ewcomp' or use_time_param:
    
                with col_list['bshift']:
                    pholder_bshift=[st.pyplot(elem) for elem in scat_bshift]
    
            if mode!='ewcomp':
                with col_list['ener']:
                    pholder_ener=[st.pyplot(elem) for elem in scat_ener]
                if use_ewratio and (mode=='observ' or use_width):
                    with col_list['ewratio']:
                        pholder_ewratio=[st.pyplot(elem) for elem in scat_ewratio]
                        
            if use_lineflux and n_infos>=5:
                with col_list['lineflux']:
                    pholder_lineflux=[st.pyplot(elem) for elem in scat_lineflux]
                    
            if use_width:
                with col_list['width']:
                    pholder_width=[st.pyplot(elem) for elem in scat_width]
            
mpl.rcParams.update({'font.size': 14})

#storing arguments to reduce the number of arguments in the scatter plot functions    
dict_linevis['scale_log_hr']=scale_log_hr
dict_linevis['scale_log_ew']=scale_log_ew

dict_linevis['abslines_ener']=abslines_ener
dict_linevis['abslines_plot']=abslines_plot

dict_linevis['lock_lims_det']=lock_lims_det

dict_linevis['compute_regr']=compute_regr
dict_linevis['regr_pval_threshold']=regr_pval_threshold
dict_linevis['restrict_comput_scatter']=restrict_comput_scatter
dict_linevis['comput_scatter_lims']=[[comput_scatter_xmin,comput_scatter_xmax],[comput_scatter_ymin,comput_scatter_ymax]]

dict_linevis['color_scatter']=color_scatter
dict_linevis['lum_plot_restrict']=lum_plot_restrict
dict_linevis['hid_plot_restrict']=hid_plot_restrict
dict_linevis['width_plot_restrict']=width_plot_restrict
dict_linevis['nh_plot_restrict']=nh_plot_restrict

dict_linevis['gamma_nthcomp_list']=gamma_nthcomp_list
dict_linevis['lum_high_list']=lum_high_list

if display_distrib:
    
    if sum(global_mask_intime_norepeat)==0 or sum(global_sign_mask)==0:
        if sum(global_mask_intime_norepeat)==0:
            st.warning('No point left in selected dates interval. Cannot compute distributions.')
        else:
            st.warning('No significant detection left with current source selection. Cannot compute distributions.')
    else:
        streamlit_distrib()
    
if display_scat_intr:
    streamlit_scat('intrinsic')

if display_scat_ewcomp:
    streamlit_scat('ewcomp')
    
if display_scat_hid:
    streamlit_scat('observ')

if display_scat_inclin:
    streamlit_scat('source')
    
if not (display_scat_intr or display_scat_ewcomp or display_scat_hid or display_scat_inclin):
    with tab_param:
        st.info('Select parameters to compare or enable distributions to generate plots')

if display_single and choice_source[0]=='4U1630-47' and plot_gamma_correl:

    # currently limited to 4U1630-47
    int_lc_df = fit_int_revol_dict[choice_source[0]]

    #significance test to only get good integral FITTED data (also removes nans)
    mask_sign_int=np.array(int_lc_df['RATE_30.0-50.0'] - 2 * int_lc_df['RATE_ERR_30.0-50.0'] > 0 ) & \
                  np.array(int_lc_df['FLUX_30.0-50.0'] - 2 * int_lc_df['FLUX_ERR_30.0-50.0'] > 0 ) & \
                  np.array(int_lc_df['powerlaw_PhoIndex_lo']!=0)

    int_fit_gamma = np.array(int_lc_df['powerlaw_PhoIndex'][mask_sign_int])
    int_fit_gamma_low=np.array(int_lc_df['powerlaw_PhoIndex_lo'][mask_sign_int])
    int_fit_gamma_high=np.array(int_lc_df['powerlaw_PhoIndex_hi'][mask_sign_int])

    int_fit_norm=np.array(int_lc_df['powerlaw_norm'][mask_sign_int])
    int_fit_norm_low=np.array(int_lc_df['powerlaw_norm_lo'][mask_sign_int])
    int_fit_norm_high=np.array(int_lc_df['powerlaw_norm_hi'][mask_sign_int])

    int_fit_flux=np.array(int_lc_df['FLUX_30.0-50.0'][mask_sign_int])
    int_fit_flux_err=np.array(int_lc_df['FLUX_ERR_30.0-50.0'][mask_sign_int])
    
    int_fit_rate_30_50=np.array(int_lc_df['RATE_30.0-50.0'][mask_sign_int])
    int_fit_rate_30_50_err = np.array(int_lc_df['RATE_ERR_30.0-50.0'][mask_sign_int])

    #computing the 15-50keV theoretical flux to compare with BAT
    int_fit_flux_15_50=flux_erg_pow(int_fit_gamma,int_fit_norm,15.,50.)
    int_fit_flux_15_50_low=flux_erg_pow(int_fit_gamma_high,int_fit_norm_low,15.,50.)
    int_fit_flux_15_50_high = flux_erg_pow(int_fit_gamma_low, int_fit_norm_high, 15., 50.)

    #note that the zero clipping also cancels the "0" high and low values (which come out as negative errs otherwise)
    int_fit_flux_15_50_err=np.array([int_fit_flux_15_50-int_fit_flux_15_50_low,
                                    int_fit_flux_15_50_high-int_fit_flux_15_50]).clip(0)

    lum_int_15_50=int_fit_flux_15_50*Edd_factor_restrict[0]
    lum_int_15_50_err=int_fit_flux_15_50_err*Edd_factor_restrict[0]

    int_lc_mjd_regr = np.array([Time(elem).mjd.astype(float) for elem in int_lc_df['ISOT']])+1.5

    #to avoid mixing with the table variable. +1.5 to match at the center of the revolution
    int_lc_mjd_corr=int_lc_mjd.astype(int)[mask_sign_int]

    int_fit_gamma_err=np.array([[0 if elem_err==0 else elem_main-elem_err for elem_main,elem_err\
                                in zip(int_fit_gamma,int_fit_gamma_low)],
                               [0 if elem_err == 0 else elem_err - elem_main for elem_main, elem_err \
                                in zip(int_fit_gamma, int_fit_gamma_high)]])

    from visual_line_tools import fetch_bat_lightcurve
    from decimal import Decimal

    bat_lc_df_init = fetch_bat_lightcurve(catal_bat_df, catal_bat_simbad, choice_source, binning='day')

    #significance test to only get good bat data
    mask_sign_bat=bat_lc_df_init[bat_lc_df_init.columns[1]]-bat_lc_df_init[bat_lc_df_init.columns[2]]*2>0

    #applying the mask. Reset index necessary to avoid issues when calling indices later.
    # drop to avoid creating an index column that will ruin the column calling
    bat_lc_df=bat_lc_df_init[mask_sign_bat].reset_index(drop=True)

    bat_lc_mjd = np.array(bat_lc_df[bat_lc_df.columns[0]])

    num_bat_dates =Time(bat_lc_df['TIME'], format='mjd').mjd

    #here we put 1.5 as a max to remain in the same revolution
    mask_int_withBAT=[abs(elem-bat_lc_mjd).min()<=1.5 for elem in int_lc_mjd_corr]

    #matching the closest BAT element (aka if possible the one from the middle day of the observation)
    match_int_bat=[abs(elem-bat_lc_mjd).argmin() for elem in int_lc_mjd_corr[mask_int_withBAT]]

    count_bat_match_int=np.array(bat_lc_df[bat_lc_df.columns[1]][match_int_bat])
    count_bat_err_match_int=np.array(bat_lc_df[bat_lc_df.columns[2]][match_int_bat])

    lum_bat_match_int=count_bat_match_int * convert_BAT_count_flux['4U1630-47'] * Edd_factor_restrict
    lum_bat_err_match_int=count_bat_err_match_int * convert_BAT_count_flux['4U1630-47'] * Edd_factor_restrict

    #for the rest
    obs_dates = Time(date_list[mask_obj][0].astype(str)).mjd.astype(int)

    mask_withtime_BAT = [elem in bat_lc_mjd for elem in obs_dates]

    mask_NuSTAR=instru_list[mask_obj][0]=='NuSTAR'
    mask_NuSTAR_withBAT=(mask_NuSTAR) & (mask_withtime_BAT)
    mask_Suzaku=instru_list[mask_obj][0]=='Suzaku'
    mask_Suzaku_withBAT=(mask_Suzaku) & (mask_withtime_BAT)

    match_NuSTAR_bat=np.array([np.argwhere(bat_lc_mjd==elem)[0][0] for elem in obs_dates[mask_NuSTAR_withBAT]])

    match_Suzaku_bat=np.array([np.argwhere(bat_lc_mjd==elem)[0][0] for elem in obs_dates[mask_Suzaku_withBAT]])

    #previous outdated version
    #note that the second dimension is a len 3 array with errors included
    # obs_gamma=np.array([np.repeat(np.nan,3) if elem is None or \
    #                         'disk_nthcomp' not in list(elem.keys()) and 'cont_powerlaw' not in list(elem.keys()) else \
    #                     elem['disk_nthcomp'][0] if 'disk_nthcomp' in list(elem.keys()) else \
    #                        elem['cont_powerlaw'][0]\
    #            for elem in fitmod_broadband_list[mask_obj][0]])

    gamma_nthcomp_single=gamma_nthcomp_list[mask_obj][0]

    count_bat_match_NuSTAR=np.array(bat_lc_df[bat_lc_df.columns[1]][match_NuSTAR_bat])
    count_bat_err_match_NuSTAR=np.array(bat_lc_df[bat_lc_df.columns[2]][match_NuSTAR_bat])

    count_bat_match_Suzaku=np.array(bat_lc_df[bat_lc_df.columns[1]][match_Suzaku_bat])
    count_bat_err_match_Suzaku=np.array(bat_lc_df[bat_lc_df.columns[2]][match_Suzaku_bat])

    #gamma-BAT rate figure
    fig_gamma_bat_rate,ax_gamma_bat_rate=plt.subplots(figsize=(6,6))
    # ax_gamma_bat_rate.set_xlim(1.5, 3.5)
    # ax_gamma_bat_rate.set_ylim(1e-3, 0.25)
    ax_gamma_bat_rate.set_yscale('log')
    ax_gamma_bat_rate.set_xlabel(r'powerlaw $\Gamma$')
    ax_gamma_bat_rate.set_ylabel('BAT 15-50 keV rate')

    #setting up alpha for the colors depending on the uncertainty of the integral gamma
    int_fit_gamma_alpha=abs(int_fit_gamma[mask_int_withBAT]/int_fit_gamma_err.T[mask_int_withBAT].T.max(0))
    int_fit_gamma_alpha = np.array([0.0001 if elem == np.inf else elem for elem in int_fit_gamma_alpha])
    int_fit_gamma_alpha = int_fit_gamma_alpha / max(int_fit_gamma_alpha)
    i_max_alpha_gamma=np.argmax(int_fit_gamma_alpha)

    #NuSTAR errorbar (no need to put a NICER as all NICER epochs are there because of NuSTAR)
    ax_gamma_bat_rate.errorbar(gamma_nthcomp_single[mask_NuSTAR_withBAT].T[0].T,count_bat_match_NuSTAR,
                          xerr=gamma_nthcomp_single[mask_NuSTAR_withBAT].T[1:],
                          yerr=count_bat_err_match_NuSTAR,
                          alpha=1,
                          label='NuSTAR', color=telescope_colors['NuSTAR'], ls='')

    #Suzaku errorbar
    ax_gamma_bat_rate.errorbar(gamma_nthcomp_single[mask_Suzaku_withBAT].T[0].T,count_bat_match_Suzaku,
                          xerr=gamma_nthcomp_single[mask_Suzaku_withBAT].T[1:],
                          yerr=count_bat_err_match_Suzaku,
                          alpha=1,
                          label='Suzaku', color=telescope_colors['Suzaku'], ls='')

    #integral errorbar
    int_gamma_err=np.array([[[int_fit_gamma_err.T[mask_int_withBAT][i_int_withBAT].T[0]],
                                             [int_fit_gamma_err.T[mask_int_withBAT][i_int_withBAT].T[1]]]\
                           for i_int_withBAT in range(sum(mask_int_withBAT))])

    for i_int_withBAT in range(sum(mask_int_withBAT)):
        ax_gamma_bat_rate.errorbar(int_fit_gamma[mask_int_withBAT][i_int_withBAT],
                              count_bat_match_int[i_int_withBAT],
                              xerr=int_gamma_err[i_int_withBAT],
                              yerr=count_bat_err_match_int[i_int_withBAT],
                              # alpha=1,
                              alpha=int_fit_gamma_alpha[i_int_withBAT],
                              label='INTEGRAL' if i_int_withBAT==i_max_alpha_gamma else '',color='black',ls='')

    r_spearman_gamma_bat_rate= np.array(pymccorrelation(int_fit_gamma[mask_int_withBAT], count_bat_match_int,
                                          dx_init=int_fit_gamma_err.T[mask_int_withBAT],
                                          dy_init=count_bat_err_match_int,
                                          Nperturb=1000, coeff='spearmanr', percentiles=(50, 5, 95)))

    # switching back to uncertainties from quantile values
    r_spearman_gamma_bat_rate = np.array([[r_spearman_gamma_bat_rate[ind_c][0],
                            r_spearman_gamma_bat_rate[ind_c][0] - r_spearman_gamma_bat_rate[ind_c][1],
                            r_spearman_gamma_bat_rate[ind_c][2] - r_spearman_gamma_bat_rate[ind_c][0]] \
                           for ind_c in [0, 1]])

    str_spearman_gamma_bat_rate = r'$r_S \,=' + str(round(r_spearman_gamma_bat_rate[0][0], 2)) +\
                   '$\n$p_S=' + '%.1e' % Decimal(r_spearman_gamma_bat_rate[1][0]) + '$'

    #computing the linear regression
    #here with error percent of 68.26 because these are standard data
    bat_gamma_slope_arr,bat_gamma_intercept_arr,bat_gamma_sigma_arr,bat_gamma_x_intercept=\
        lmplot_uncert_a(ax_gamma_bat_rate,int_fit_gamma[mask_int_withBAT],count_bat_match_int,
                           int_gamma_err.T[0],count_bat_err_match_int,
                           xlim=None,ylim=None, percent=90, nsim=100,
                            intercept_pos='auto',
                            return_linreg=True, infer_log_scale=True,nanzero_err=True,
                            xbounds=[1.,3.5],ybounds=[1e-3,2e-1],
                            line_color='blue',lw=1.3, inter_color='lightgrey')

    corr_str=r'Rate$_{[15-50]}=10^{%.2f' % bat_gamma_slope_arr[0] +'_{-%.2f' % bat_gamma_slope_arr[1]\
                 + '}^{+%.2f' % bat_gamma_slope_arr[2] + '}(\Gamma-'+str(bat_gamma_x_intercept)+')'\
                 + ('+' if bat_gamma_intercept_arr[0]>0 else '')+'%.2f' % bat_gamma_intercept_arr[0]\
                 + '_{-%.2f' % bat_gamma_intercept_arr[1] + '}^{+%.2f' % bat_gamma_intercept_arr[2] + '}'\
                 +'}$'

    plt.suptitle(corr_str)

    # plt.suptitle('intercept at ' + str(bat_gamma_x_intercept)
    #              + '\n'r'a=$%.2f' % bat_gamma_slope_arr[0] + '_{-%.2f' % bat_gamma_slope_arr[1]
    #              + '}^{+%.2f' % bat_gamma_slope_arr[2] + '}$'
    #              + '\n'r'b=$%.2f' % bat_gamma_intercept_arr[0] + '_{-%.2f' % bat_gamma_intercept_arr[1]
    #              + '}^{+%.2f' % bat_gamma_intercept_arr[2] + '}$'
    #              + '\n'r'sig=$%.2f' % bat_gamma_sigma_arr[0] + '_{-%.2f' % bat_gamma_sigma_arr[1]
    #              + '}^{+%.2f' % bat_gamma_sigma_arr[2] + '}$')

    ax_gamma_bat_rate.legend(fontsize=10,title=str_spearman_gamma_bat_rate)
    fig_gamma_bat_rate.tight_layout()

    #integral gamma-flux figure
    fig_gamma_flux_int,ax_gamma_flux_int=plt.subplots(figsize=(6,6))
    ax_gamma_flux_int.set_xlim(1.5, 3.5)
    # ax_gamma_flux_int.set_ylim(3e-11, 5e-9)
    ax_gamma_flux_int.set_yscale('log')
    ax_gamma_flux_int.set_xlabel(r'powerlaw $\Gamma$')
    # plt.ylabel('30-50 keV flux (erg/s/cm²)')
    plt.ylabel('30-50 keV luminosity ($L/L_{Edd}$)')

    #setting up alpha for the colors
    int_fit_flux_alpha=abs(int_fit_flux/int_fit_flux_err)
    int_fit_flux_alpha = np.array([0.1 if elem == np.inf else elem for elem in int_fit_flux_alpha])
    int_fit_flux_alpha = int_fit_flux_alpha / max(int_fit_flux_alpha)
    i_max_alpha_flux=np.argmax(int_fit_flux_alpha)

    #note: Flux translated to Eddington ratios using the Edd factor of the source
    for i_revol in range(len(int_fit_flux)):
        ax_gamma_flux_int.errorbar(int_fit_gamma[i_revol],int_fit_flux[i_revol]*Edd_factor[mask_obj][0],
                xerr=np.array([[int_fit_gamma_err.T[i_revol].T[0]],[int_fit_gamma_err.T[i_revol].T[1]]]),
                yerr=int_fit_flux_err[i_revol]*Edd_factor[mask_obj][0],color='black',alpha=int_fit_flux_alpha[i_revol],
                label='INTEGRAL' if i_revol==i_max_alpha_flux else '',ls='')

    #adding the february/march 2024 points, handmade from Tristan's data
    #feb
    # ax_gamma_flux_int.errorbar(1.93173,0.00025141484646483593,
    #                            xerr=[[0.00926375],[0.00978644]],yerr=[[1.2151665648938697e-06],[1.1216922137481912e-06]],
    #                            color='black',marker='d')
    # #march
    # ax_gamma_flux_int.errorbar(2.07,0.0004436682181837173,
    #                            xerr=[[0.00736723],[0.00743803]],yerr=[[1.6903278498844185e-06],[1.5812744402144266e-06]],
    #                            color='black',marker='d')

    #note: using systematic uncertainties instead bc the fits give suspicious low errors
    ax_gamma_flux_int.errorbar(1.93173,0.00025141484646483593,
                               xerr=0.1,yerr=0.00025/10,
                               color='black',marker='d')
    #march
    ax_gamma_flux_int.errorbar(2.07,0.0004436682181837173,
                               xerr=0.1,yerr=0.00044/10,
                               color='black',marker='d')

    r_spearman_gamma_flux_int= np.array(pymccorrelation(int_fit_gamma, int_fit_flux,
                                          dx_init=int_fit_gamma_err.T,
                                          dy_init=int_fit_flux_err,
                                          Nperturb=1000, coeff='spearmanr', percentiles=(50, 5, 95)))

    # switching back to uncertainties from quantile values
    r_spearman_gamma_flux_int = np.array([[r_spearman_gamma_flux_int[ind_c][0],
                            r_spearman_gamma_flux_int[ind_c][0] - r_spearman_gamma_flux_int[ind_c][1],
                            r_spearman_gamma_flux_int[ind_c][2] - r_spearman_gamma_flux_int[ind_c][0]] \
                           for ind_c in [0, 1]])

    str_spearman_gamma_flux_int = r'$r_S \,=' + str(round(r_spearman_gamma_flux_int[0][0], 2)) +\
                   '$\n$p_S=' + '%.1e' % Decimal(r_spearman_gamma_flux_int[1][0]) + '$'


    ax_gamma_flux_int.legend(fontsize=10,title=str_spearman_gamma_flux_int)

    fig_gamma_flux_int.tight_layout()
    
    #integral gamma-rate figure
    fig_gamma_rate_int,ax_gamma_rate_int=plt.subplots(figsize=(6,6))
    ax_gamma_rate_int.set_xlim(1.5, 3.5)
    # ax_gamma_rate_int.set_ylim(3e-11, 5e-9)
    ax_gamma_rate_int.set_yscale('log')
    ax_gamma_rate_int.set_xlabel(r'powerlaw $\Gamma$')
    plt.ylabel('INTEGRAL rate in [30-50] keV (cts/s)')

    #setting up alpha for the colors
    int_fit_rate_30_50_alpha=abs(int_fit_rate_30_50/int_fit_rate_30_50_err)
    int_fit_rate_30_50_alpha = np.array([0.001 if elem == np.inf else elem for elem in int_fit_rate_30_50_alpha])
    int_fit_rate_30_50_alpha = (int_fit_rate_30_50_alpha / max(int_fit_rate_30_50_alpha))**(1/2)
    i_max_alpha_rate=np.argmax(int_fit_rate_30_50_alpha)

    for i_revol in range(len(int_fit_rate_30_50)):
        ax_gamma_rate_int.errorbar(int_fit_gamma[i_revol],int_fit_rate_30_50[i_revol],
                xerr=np.array([[int_fit_gamma_err.T[i_revol].T[0]],[int_fit_gamma_err.T[i_revol].T[1]]]),
                yerr=int_fit_rate_30_50_err[i_revol],color='black',alpha=int_fit_rate_30_50_alpha[i_revol],
                label='INTEGRAL' if i_revol==i_max_alpha_rate else '',ls='')

    r_spearman_gamma_rate_int= np.array(pymccorrelation(int_fit_gamma, int_fit_rate_30_50,
                                          dx_init=int_fit_gamma_err.T,
                                          dy_init=int_fit_rate_30_50_err,
                                          Nperturb=1000, coeff='spearmanr', percentiles=(50, 5, 95)))

    # switching back to uncertainties from quantile values
    r_spearman_gamma_rate_int = np.array([[r_spearman_gamma_rate_int[ind_c][0],
                            r_spearman_gamma_rate_int[ind_c][0] - r_spearman_gamma_rate_int[ind_c][1],
                            r_spearman_gamma_rate_int[ind_c][2] - r_spearman_gamma_rate_int[ind_c][0]] \
                           for ind_c in [0, 1]])

    str_spearman_gamma_rate_int = r'$r_S \,=' + str(round(r_spearman_gamma_rate_int[0][0], 2)) +\
                   '$\n$p_S=' + '%.1e' % Decimal(r_spearman_gamma_rate_int[1][0]) + '$'


    ax_gamma_rate_int.legend(fontsize=10,title=str_spearman_gamma_rate_int)

    fig_gamma_rate_int.tight_layout()
    
    #integral rate_flux figure
    fig_rate_flux_int,ax_rate_flux_int=plt.subplots(figsize=(6,6))
    # ax_rate_flux_int.set_xlim(1.5, 3.5)
    # ax_rate_flux_int.set_ylim(3e-11, 5e-9)
    ax_rate_flux_int.set_yscale('log')
    ax_rate_flux_int.set_xscale('log')
    ax_rate_flux_int.set_xlabel(r'INTEGRAL rate in [30-50] keV (cts/s)')
    ax_rate_flux_int.set_ylabel(r'30-50 keV flux (erg/s/cm²)')

    #setting up alpha for the colors
    int_fit_rate_30_50_alpha=abs(int_fit_rate_30_50/int_fit_rate_30_50_err)
    int_fit_rate_30_50_alpha = np.array([0.001 if elem == np.inf else elem for elem in int_fit_rate_30_50_alpha])
    int_fit_rate_30_50_alpha = (int_fit_rate_30_50_alpha / max(int_fit_rate_30_50_alpha))**(1/2)
    i_max_alpha_rate=np.argmax(int_fit_rate_30_50_alpha)

    for i_revol in range(len(int_fit_rate_30_50)):
        ax_rate_flux_int.errorbar(int_fit_rate_30_50[i_revol],int_fit_flux[i_revol],
                xerr=int_fit_rate_30_50_err[i_revol],
                yerr=int_fit_flux_err[i_revol],color='black',alpha=int_fit_rate_30_50_alpha[i_revol],
                label='INTEGRAL' if i_revol==i_max_alpha_rate else '',ls='')

    r_spearman_rate_flux_int= np.array(pymccorrelation(int_fit_rate_30_50,int_fit_flux,
                                          dx_init=int_fit_rate_30_50_err,
                                          dy_init=int_fit_flux_err,
                                          Nperturb=1000, coeff='spearmanr', percentiles=(50, 5, 95)))

    # switching back to uncertainties from quantile values
    r_spearman_rate_flux_int = np.array([[r_spearman_rate_flux_int[ind_c][0],
                            r_spearman_rate_flux_int[ind_c][0] - r_spearman_rate_flux_int[ind_c][1],
                            r_spearman_rate_flux_int[ind_c][2] - r_spearman_rate_flux_int[ind_c][0]] \
                           for ind_c in [0, 1]])

    str_spearman_rate_flux_int = r'$r_S \,=' + str(round(r_spearman_rate_flux_int[0][0], 2)) +\
                   '$\n$p_S=' + '%.1e' % Decimal(r_spearman_rate_flux_int[1][0]) + '$'


    ax_rate_flux_int.legend(fontsize=10,title=str_spearman_rate_flux_int)

    fig_rate_flux_int.tight_layout()

    #integral rate_flux_15_50 figure
    fig_rate_flux_15_50_int,ax_rate_flux_15_50_int=plt.subplots(figsize=(6,6))
    # ax_rate_flux_15_50_int.set_xlim(1.5, 3.5)
    # ax_rate_flux_15_50_int.set_ylim(3e-11, 5e-9)
    ax_rate_flux_15_50_int.set_yscale('log')
    ax_rate_flux_15_50_int.set_xscale('log')
    ax_rate_flux_15_50_int.set_xlabel(r'INTEGRAL rate in [30-50] keV (cts/s)')
    ax_rate_flux_15_50_int.set_ylabel(r'15-50 keV extrapolated flux  (erg/s/cm²)')

    #setting up alpha for the colors
    int_fit_rate_30_50_alpha=abs(int_fit_rate_30_50/int_fit_rate_30_50_err)
    int_fit_rate_30_50_alpha = np.array([0.001 if elem == np.inf else elem for elem in int_fit_rate_30_50_alpha])
    int_fit_rate_30_50_alpha = (int_fit_rate_30_50_alpha / max(int_fit_rate_30_50_alpha))**(1/2)
    i_max_alpha_rate=np.argmax(int_fit_rate_30_50_alpha)

    for i_revol in range(len(int_fit_rate_30_50)):
        ax_rate_flux_15_50_int.errorbar(int_fit_rate_30_50[i_revol],int_fit_flux_15_50[i_revol],
                xerr=int_fit_rate_30_50_err[i_revol],
                yerr=int_fit_flux_15_50_err.T[[i_revol]].T,color='black',alpha=int_fit_rate_30_50_alpha[i_revol],
                label='INTEGRAL' if i_revol==i_max_alpha_rate else '',ls='')

    r_spearman_rate_flux_15_50_int= np.array(pymccorrelation(int_fit_rate_30_50,int_fit_flux_15_50,
                                          dx_init=int_fit_rate_30_50_err,
                                          dy_init=int_fit_flux_15_50_err.T,
                                          Nperturb=1000, coeff='spearmanr', percentiles=(50, 5, 95)))

    # switching back to uncertainties from quantile values
    r_spearman_rate_flux_15_50_int = np.array([[r_spearman_rate_flux_15_50_int[ind_c][0],
                            r_spearman_rate_flux_15_50_int[ind_c][0] - r_spearman_rate_flux_15_50_int[ind_c][1],
                            r_spearman_rate_flux_15_50_int[ind_c][2] - r_spearman_rate_flux_15_50_int[ind_c][0]] \
                           for ind_c in [0, 1]])

    str_spearman_rate_flux_15_50_int = r'$r_S \,=' + str(round(r_spearman_rate_flux_15_50_int[0][0], 2)) +\
                   '$\n$p_S=' + '%.1e' % Decimal(r_spearman_rate_flux_15_50_int[1][0]) + '$'

    int_corr_slope_arr,int_corr_intercept_arr,int_corr_sigma_arr,int_corr_x_intercept=\
        lmplot_uncert_a(ax_rate_flux_15_50_int,int_fit_rate_30_50,int_fit_flux_15_50,
                           int_fit_rate_30_50_err,int_fit_flux_15_50_err,
                           xlim=None,ylim=None, percent=68.26, nsim=1000,
                            intercept_pos='auto',
                            return_linreg=True, infer_log_scale=True,log_sampling=True,nanzero_err=True,
                            xbounds=plt.xlim(),ybounds=plt.ylim(),
                            line_color='brown',lw=1.3,
                            inter_color='lightgrey')

    int_corr_str=r'$F_{int}^{15-50}=(\frac{R_{int}^{30-50}}{10^{%.1e'%(int_corr_x_intercept)+'}})^{%.2f'\
                    % int_corr_slope_arr[0] +'_{-%.2f' % int_corr_slope_arr[1] + '}^{+%.2f' % int_corr_slope_arr[2] + '}}'\
                 +'\\times 10^{'+('+' if int_corr_intercept_arr[0]>0 else '')+'%.2f' % int_corr_intercept_arr[0]\
                 + '_{-%.2f' % int_corr_intercept_arr[1] + '}^{+%.2f' % int_corr_intercept_arr[2] + '}'\
                 +'}$'

    plt.suptitle(int_corr_str)
    

    ax_rate_flux_15_50_int.legend(fontsize=10,title=str_spearman_rate_flux_15_50_int)

    fig_rate_flux_15_50_int.tight_layout()



    #third figure: BAT derived flux vs high energy flux
    bat_lc_lum = np.array([bat_lc_df[bat_lc_df.columns[1]], bat_lc_df[bat_lc_df.columns[2]]]).clip(0).T \
                 * convert_BAT_count_flux['4U1630-47'] * Edd_factor_restrict

    # getting an array with the bat flux of each observation date
    lum_BAT_single = np.array(
        [np.array([np.nan, np.nan]) if obs_dates[i_obs] not in bat_lc_mjd else bat_lc_lum[bat_lc_mjd == obs_dates[i_obs]][0] \
         for i_obs in range(len(obs_dates))]).T

    mask_match_BAT_main=~np.isnan(lum_BAT_single[0])

    flux_high_list_single=flux_high_list[mask_obj][0]
    flux_high_list_single_mask=[elem is not None for elem in flux_high_list_single]

    lum_high_list_single=np.array([elem for elem in flux_high_list_single[flux_high_list_single_mask]],
                                  dtype=float).clip(0).T*Edd_factor_restrict

    lum_high_list_single_withBAT=lum_high_list_single.T[mask_match_BAT_main[flux_high_list_single_mask]].T

    #BAT flux- Observed flux figure
    fig_flux_BAT_native,ax_flux_BAT_native=plt.subplots(figsize=(6,6))
    ax_flux_BAT_native.set_xlim(1e-4, 1)
    ax_flux_BAT_native.set_ylim(1e-4, 1)
    ax_flux_BAT_native.set_xscale('log')
    ax_flux_BAT_native.set_yscale('log')
    ax_flux_BAT_native.set_xlabel(r'[15-50] keV derived BAT Luminosity ($L/L_{Edd}$)')
    ax_flux_BAT_native.set_ylabel('[15-50] keV measured luminosity ($L/L_{Edd}$)')

    #integral errorbar
    for i_int_withBAT in range(sum(mask_int_withBAT)):
        ax_flux_BAT_native.errorbar(lum_bat_match_int[i_int_withBAT],
                              lum_int_15_50[mask_int_withBAT][i_int_withBAT],
                              xerr=lum_bat_err_match_int[i_int_withBAT],
                              yerr=np.array([lum_int_15_50_err.T[mask_int_withBAT][i_int_withBAT]]).T,
                              # alpha=1,
                              alpha=int_fit_gamma_alpha[i_int_withBAT],
                              label='INTEGRAL (powerlaw)' if i_int_withBAT==i_max_alpha_gamma else '',
                                    color='black',ls='')

    plt.errorbar(lum_BAT_single[0][flux_high_list_single_mask & mask_match_BAT_main],
                 lum_high_list_single_withBAT[0],
                 xerr=lum_BAT_single[1][flux_high_list_single_mask & mask_match_BAT_main],
                 yerr=lum_high_list_single_withBAT[1:],
                 ls='',color='blue',label='NuSTAR/Suzaku-PIN (nthcom)')

    #computing the linear regression

    lum_bat_regress=np.array(lum_bat_match_int.tolist()+\
                             lum_BAT_single[0][flux_high_list_single_mask & mask_match_BAT_main].tolist())
    lum_bat_regress_err=np.array(lum_bat_err_match_int.tolist()+\
                             lum_BAT_single[1][flux_high_list_single_mask & mask_match_BAT_main].tolist())


    lum_15_50_regress=np.array(lum_int_15_50[mask_int_withBAT].tolist()+lum_high_list_single_withBAT[0].tolist())
    
    #here we resize the 90% errors of our dataset assuming a gaussian distribution
    lum_15_50_regress_err=np.array([lum_int_15_50_err[i_incert].T[mask_int_withBAT].T.tolist()+\
                                    (lum_high_list_single_withBAT[1:][i_incert]\
                                   * 1/norm.ppf((1 + 90/100) / 2)).tolist() for i_incert in range(2)])

    
    bat_corr_slope_arr,bat_corr_intercept_arr,bat_corr_sigma_arr,bat_corr_x_intercept=\
        lmplot_uncert_a(ax_flux_BAT_native,lum_bat_regress,lum_15_50_regress,
                           lum_bat_regress_err,lum_15_50_regress_err,
                           xlim=None,ylim=None, percent=68.26, nsim=1000,
                            intercept_pos='auto',
                            return_linreg=True, infer_log_scale=True,log_sampling=True,nanzero_err=True,
                            xbounds=None,ybounds=None,
                            line_color='brown',lw=1.3, inter_color='lightgrey')

    corr_str=r'$L_{Obs}=(\frac{L_{BAT}}{10^{%.1e'%(bat_corr_x_intercept)+'}})^{%.2f'\
                    % bat_corr_slope_arr[0] +'_{-%.2f' % bat_corr_slope_arr[1] + '}^{+%.2f' % bat_corr_slope_arr[2] + '}}'\
                 +'\\times 10^{'+('+' if bat_corr_intercept_arr[0]>0 else '')+'%.2f' % bat_corr_intercept_arr[0]\
                 + '_{-%.2f' % bat_corr_intercept_arr[1] + '}^{+%.2f' % bat_corr_intercept_arr[2] + '}'\
                 +'}$'

    plt.suptitle(corr_str)

    # fig_flux_BAT_native.suptitle('BAT projected vs observed high energy luminosity')
    plt.legend(fontsize=10,)

    #fourth figure to look at the flux vs gamma evolution internally
    fig_native_flux_gamma,ax_native_flux_gamma=plt.subplots(figsize=(6,6))
    ax_native_flux_gamma.set_xlim(1.,3.5)
    ax_native_flux_gamma.set_ylim(1e-4, 1e-1)
    ax_native_flux_gamma.set_yscale('log')
    ax_native_flux_gamma.set_xlabel(r'nthcomp $\Gamma$')
    ax_native_flux_gamma.set_ylabel('15-50 keV luminosity ($L/L_{Edd}$)')
    
    plt.errorbar(gamma_nthcomp_single[flux_high_list_single_mask].T[0],lum_high_list_single[0],
                 xerr=gamma_nthcomp_single[flux_high_list_single_mask].T[1:],yerr=lum_high_list_single[1:],ls='',
                 color='blue',label='NuSTAR/Suzaku-PIN',marker='d')

    # fig_native_flux_gamma.suptitle(r'Observed nthcomp $\Gamma$ vs high energy luminosity')
    plt.legend(fontsize=10,)

    fig_high_soft, ax_high_soft = plt.subplots()

    plt.errorbar(lum_high_plot_restrict.T[0][0], lum_plot_restrict[1][0][0],
                 xerr=np.array([elem for elem in lum_high_plot_restrict.T[0][1:]]).clip(0),
                 yerr=np.array([elem for elem in lum_plot_restrict[1].T[0][1:]]), ls='')
    plt.xlabel('[15-50] keV lum')
    plt.ylabel('[3-6] keV lum')
    plt.suptitle('color color diagram 1')
    plt.xscale('log')
    plt.yscale('log')

    fig_kt_lum, ax_kt_lum = plt.subplots()

    plt.errorbar(kt_plot_restrict[0][0], lum_plot_restrict[4][0][0],
                 xerr=np.array([elem for elem in kt_plot_restrict.T[0][1:]]).clip(0),
                 yerr=np.array([elem for elem in lum_plot_restrict[4].T[0][1:]]), ls='')
    plt.xlabel('kT')
    plt.ylabel('[3-10] keV lum')
    plt.suptitle('kt - Luminosity ')
    plt.xscale('linear')
    plt.yscale('log')


    fig_high_softhard, ax_high_softhard = plt.subplots()

    plt.errorbar(lum_high_plot_restrict.T[0][0], lum_plot_restrict[2][0][0],
                 xerr=np.array([elem for elem in lum_high_plot_restrict.T[0][1:]]).clip(0),
                 yerr=np.array([elem for elem in lum_plot_restrict[2].T[0][1:]]), ls='')
    plt.xlabel('[15-50] keV lum')
    plt.ylabel('[6-10] keV lum')
    plt.suptitle('color color diagram 2')

    plt.xscale('log')
    plt.yscale('log')

    fig_high_softtot, ax_high_softot = plt.subplots()

    plt.errorbar(lum_high_plot_restrict.T[0][0], lum_plot_restrict[4][0][0],
                 xerr=np.array([elem for elem in lum_high_plot_restrict.T[0][1:]]).clip(0),
                 yerr=np.array([elem for elem in lum_plot_restrict[4].T[0][1:]]), ls='')
    plt.xlabel('[15-50] keV lum')
    plt.ylabel('[3-10] keV lum')
    plt.suptitle('color color diagram 3')

    plt.xscale('log')
    plt.yscale('log')

    with tab_param:
        with st.expander('High energy derivation correlations'):
            he_cols=st.columns(2)
            with he_cols[0]:
                st.pyplot(fig_gamma_bat_rate)
                st.pyplot(fig_flux_BAT_native)
                st.pyplot(fig_rate_flux_15_50_int)
                st.pyplot(fig_gamma_rate_int)

            with he_cols[1]:
                st.pyplot(fig_gamma_flux_int)
                st.pyplot(fig_native_flux_gamma)
                st.pyplot(fig_rate_flux_int)

            colordiag_cols=st.columns(3)
            with colordiag_cols[0]:
                st.pyplot(fig_high_soft)
                st.pyplot(fig_kt_lum)
            with colordiag_cols[1]:
                st.pyplot(fig_high_softhard)
            with colordiag_cols[2]:
                st.pyplot(fig_high_softtot)


            # he_cols= st.columns(3)
            # with he_cols[0]:
            #     st.pyplot(fig_native_flux_gamma)
            #     st.pyplot(fig_gamma_bat)
            # with he_cols[1]:
            #     st.pyplot(fig_gamma_flux)
            #
            # with he_cols[2]:
            #     st.pyplot(fig_flux_BAT_native)

if plot_physical_compa:

    sources_refl=np.array(list(incl_refl_dict.keys()))
    sources_dyn=np.array(list(incl_dyn_dict.keys()))
    sources_jet=np.array(list(incl_jet_dict.keys()))
    sources_misc=np.array(list(incl_misc_dict.keys()))

    sources_incl_refl_dyn=[elem for elem in sources_refl if elem in sources_dyn]
    sources_incl_refl_jet=[elem for elem in sources_refl if elem in sources_jet]
    sources_incl_refl_misc=[elem for elem in sources_refl if elem in sources_misc]

    '''
    #reflection-dynamical inclination plot
    '''
    
    fig_refl_dyn,ax_refl_dyn=plt.subplots(figsize=(6,6))

    ax_refl_dyn.set_xlim(0,90)
    ax_refl_dyn.set_ylim(0,90)

    ax_refl_dyn.set_title('Reflection vs dynamical inclination comparison',fontsize=14)
    ax_refl_dyn.set_xlabel('reflection inclination (°)')
    ax_refl_dyn.set_ylabel('dynamical inclination (°)')

    cmap_refl_dyn=mpl.cm.plasma
    norm_colors_refl_dyn = mpl.colors.Normalize(vmin=0, vmax=len(sources_incl_refl_dyn)-1)
    colors_func_refl_dyn = mpl.cm.ScalarMappable(norm=norm_colors_refl_dyn , cmap=cmap_refl_dyn)

    color_arr_refl_dyn = np.array([colors_func_refl_dyn.to_rgba(i) for i in range(len(sources_incl_refl_dyn))])

    ax_refl_dyn.plot([0,90],[0,90],color='grey',ls='--',lw=0.5)

    for i_source,elem_source in enumerate(sources_incl_refl_dyn):

        elem_xuplim=incl_refl_dict[elem_source][0] == incl_refl_dict[elem_source][1]
        elem_xlolim=incl_refl_dict[elem_source][0] + incl_refl_dict[elem_source][2] >= 90
        elem_uplim=incl_dyn_dict[elem_source][0] == incl_dyn_dict[elem_source][1]
        elem_lolim=incl_dyn_dict[elem_source][0] + incl_dyn_dict[elem_source][2] >= 90

        # axis errorbars
        err_refl_dyn=ax_refl_dyn.errorbar(incl_refl_dict[elem_source][0],incl_dyn_dict[elem_source][0],
                     xerr=5 if elem_xuplim or elem_xlolim else np.array([incl_refl_dict[elem_source][1:-1]]).T,
                     yerr=5 if elem_uplim or elem_lolim else np.array([incl_dyn_dict[elem_source][1:-1]]).T,
                     ls='',label=elem_source,color=color_arr_refl_dyn[i_source],
                      xuplims=elem_xuplim,
                      xlolims=elem_xlolim,
                      uplims=elem_uplim,
                      lolims=elem_lolim,
                      marker='d' if elem_source in dippers_list else '.')

        #dashing unsure measurements
        if incl_refl_dict[elem_source][-1]==0:
            err_refl_dyn[-1][0].set_linestyle('--')

        if incl_dyn_dict[elem_source][-1]==1:
            err_refl_dyn[-1][1].set_linestyle('--')

    ax_refl_dyn.legend(loc='lower left')


    '''
    #reflection-jet inclination plot
    '''
    fig_refl_jet,ax_refl_jet=plt.subplots(figsize=(6,6))

    ax_refl_jet.set_xlim(0,90)
    ax_refl_jet.set_ylim(0,90)

    ax_refl_jet.set_title('Reflection vs jet inclination comparison',fontsize=14)
    ax_refl_jet.set_xlabel('reflection inclination (°)')
    ax_refl_jet.set_ylabel('jet inclination (°)')

    cmap_refl_jet=mpl.cm.plasma
    norm_colors_refl_jet = mpl.colors.Normalize(vmin=0, vmax=len(sources_incl_refl_jet) - 1)
    colors_func_refl_jet = mpl.cm.ScalarMappable(norm=norm_colors_refl_jet , cmap=cmap_refl_jet)

    color_arr_refl_jet = np.array([colors_func_refl_jet.to_rgba(i) for i in range(len(sources_incl_refl_jet))])

    ax_refl_jet.plot([0,90],[0,90],color='grey',ls='--',lw=0.5)

    for i_source,elem_source in enumerate(sources_incl_refl_jet):

        elem_xuplim=incl_refl_dict[elem_source][0] == incl_refl_dict[elem_source][1]
        elem_xlolim=incl_refl_dict[elem_source][0] + incl_refl_dict[elem_source][2] >= 90
        elem_uplim=incl_jet_dict[elem_source][0] == incl_jet_dict[elem_source][1]
        elem_lolim=incl_jet_dict[elem_source][0] + incl_jet_dict[elem_source][2] >= 90

        # axis errorbars
        err_refl_jet=ax_refl_jet.errorbar(incl_refl_dict[elem_source][0],incl_jet_dict[elem_source][0],
                     xerr=5 if elem_xuplim or elem_xlolim else np.array([incl_refl_dict[elem_source][1:-1]]).T,
                     yerr=5 if elem_uplim or elem_lolim else np.array([incl_jet_dict[elem_source][1:-1]]).T,
                     ls='',label=elem_source,color=color_arr_refl_jet[i_source],
                      xuplims=elem_xuplim,
                      xlolims=elem_xlolim,
                      uplims=elem_uplim,
                      lolims=elem_lolim,
                      marker='d' if elem_source in dippers_list else '.')

        #dashing unsure measurements
        if incl_refl_dict[elem_source][-1]==0:
            err_refl_jet[-1][0].set_linestyle('--')

        if incl_jet_dict[elem_source][-1]==0:
            err_refl_jet[-1][1].set_linestyle('--')

    ax_refl_jet.legend(loc='lower left')

            
    '''
    #reflection-misc inclination plot
    '''
    
    fig_refl_misc,ax_refl_misc=plt.subplots(figsize=(6,6))

    ax_refl_misc.set_xlim(0,90)
    ax_refl_misc.set_ylim(0,90)

    ax_refl_misc.set_title('Reflection vs other inclination comparison',fontsize=14)
    ax_refl_misc.set_xlabel('reflection inclination (°)')
    ax_refl_misc.set_ylabel('other inclination (°)')

    cmap_refl_misc=mpl.cm.plasma
    norm_colors_refl_misc = mpl.colors.Normalize(vmin=0, vmax=len(sources_incl_refl_misc) - 1)
    colors_func_refl_misc = mpl.cm.ScalarMappable(norm=norm_colors_refl_misc , cmap=cmap_refl_misc)

    color_arr_refl_misc = np.array([colors_func_refl_misc.to_rgba(i) for i in range(len(sources_incl_refl_misc))])

    ax_refl_misc.plot([0,90],[0,90],color='grey',ls='--',lw=0.5)

    for i_source,elem_source in enumerate(sources_incl_refl_misc):

        elem_xuplim=incl_refl_dict[elem_source][0] == incl_refl_dict[elem_source][1]
        elem_xlolim=incl_refl_dict[elem_source][0] + incl_refl_dict[elem_source][2] >= 90
        elem_uplim=incl_misc_dict[elem_source][0] == incl_misc_dict[elem_source][1]
        elem_lolim=incl_misc_dict[elem_source][0] + incl_misc_dict[elem_source][2] >= 90

        # axis errorbars
        err_refl_misc=ax_refl_misc.errorbar(incl_refl_dict[elem_source][0],incl_misc_dict[elem_source][0],
                     xerr=5 if elem_xuplim or elem_xlolim else np.array([incl_refl_dict[elem_source][1:-1]]).T,
                     yerr=5 if elem_uplim or elem_lolim else np.array([incl_misc_dict[elem_source][1:-1]]).T,
                     ls='',label=elem_source,color=color_arr_refl_misc[i_source],
                      xuplims=elem_xuplim,
                      xlolims=elem_xlolim,
                      uplims=elem_uplim,
                      lolims=elem_lolim,
                      marker='d' if elem_source in dippers_list else '.')

        #dashing unsure measurements
        if incl_refl_dict[elem_source][-1]==0:
            err_refl_misc[-1][0].set_linestyle('--')

        if incl_misc_dict[elem_source][-1]==0:
            err_refl_misc[-1][1].set_linestyle('--')

    ax_refl_misc.legend(loc='lower left')


    '''
    #Porb_incl plot
    '''

    sources_incl_all=np.array(list(incl_dict_use.keys()))
    sources_porb=np.array(list(Porb_dict.keys()))

    sources_incl_porb=[elem for elem in sources_porb if elem in sources_incl_all]

    fig_Porb_incl, ax_Porb_incl = plt.subplots(figsize=(6, 6))

    ax_Porb_incl.set_xlim(1, 1e3)
    ax_Porb_incl.set_xscale('log')
    ax_Porb_incl.set_ylim(0, 90)

    ax_Porb_incl.set_title('Distribution of Orbital period and inclination', fontsize=14)
    ax_Porb_incl.set_xlabel('Orbital Period (h)')
    ax_Porb_incl.set_ylabel('Inclination (°)')

    # cmap_Porb_incl = mpl.cm.plasma
    # norm_colors_Porb_incl = mpl.colors.Normalize(vmin=0, vmax=len(sources_incl_Porb) - 1)
    # colors_func_Porb_incl = mpl.cm.ScalarMappable(norm=norm_colors_Porb_incl, cmap=cmap_Porb_incl)
    #
    # color_arr_Porb_incl = np.array([colors_func_Porb_incl.to_rgba(i) for i in range(len(sources_incl_Porb))])

    for i_source, elem_source in enumerate(sources_incl_porb):

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] ==incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_Porb_incl = ax_Porb_incl.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                              xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                  [Porb_dict[elem_source][1:-1]]).T,
                                              yerr=5 if elem_uplim or elem_lolim else np.array(
                                                  [incl_dict_use[elem_source][1:-1]]).T,
                                              ls='', label=elem_source, color='grey',
                                              xuplims=elem_xuplim,
                                              xlolims=elem_xlolim,
                                              uplims=elem_uplim,
                                              lolims=elem_lolim,
                                              marker='d' if elem_source in dippers_list else '.')

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_Porb_incl[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_Porb_incl[-1][1].set_linestyle('--')

    # ax_Porb_incl.legend(loc='lower left')

    '''
    #iron band wind detections
    '''

    fig_wind_ib, ax_wind_ib = plt.subplots(figsize=(6, 6))

    ax_wind_ib.set_xlim(1, 1e3)
    ax_wind_ib.set_xscale('log')
    ax_wind_ib.set_ylim(0, 90)

    ax_wind_ib.set_title('Iron band wind detections', fontsize=14)
    ax_wind_ib.set_xlabel('Orbital Period (h)')
    ax_wind_ib.set_ylabel('Inclination (°)')

    sources_wind_ib=[elem for elem in wind_det_sources if wind_det_dict[elem].iron_band is not False]

    sources_wind_ib_sure=[elem for elem in wind_det_sources if wind_det_dict[elem].iron_band is not False and\
                          wind_det_dict[elem].iron_band.trust_noem]

    sources_wind_ib_porb=[elem for elem in sources_wind_ib if elem in sources_incl_porb]

    cmap_wind_ib = mpl.cm.plasma
    norm_colors_wind_ib = mpl.colors.Normalize(vmin=0, vmax=len(sources_wind_ib_porb) - 1)
    colors_func_wind_ib = mpl.cm.ScalarMappable(norm=norm_colors_wind_ib, cmap=cmap_wind_ib)

    color_arr_wind_ib = np.array([colors_func_wind_ib.to_rgba(i) for i in range(len(sources_wind_ib_porb))])

    for i_source, elem_source in enumerate(sources_wind_ib_porb):

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] ==incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_wind_ib = ax_wind_ib.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                              xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                  [Porb_dict[elem_source][1:-1]]).T,
                                              yerr=5 if elem_uplim or elem_lolim else np.array(
                                                  [incl_dict_use[elem_source][1:-1]]).T,
                                              ls='', label=elem_source, color=color_arr_wind_ib[i_source],
                                              alpha=1 if elem_source in sources_wind_ib_sure else 0.3,
                                              xuplims=elem_xuplim,
                                              xlolims=elem_xlolim,
                                              uplims=elem_uplim,
                                              lolims=elem_lolim,
                                              marker='d' if elem_source in dippers_list else '.')

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_wind_ib[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_wind_ib[-1][1].set_linestyle('--')

    #overplotting the rest of the sources
    for i_source, elem_source in enumerate(sources_incl_porb):

        if elem_source in sources_wind_ib_porb:
            continue

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] ==incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_Porb_incl = ax_wind_ib.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                              xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                  [Porb_dict[elem_source][1:-1]]).T,
                                              yerr=5 if elem_uplim or elem_lolim else np.array(
                                                  [incl_dict_use[elem_source][1:-1]]).T,
                                              ls='', label='', color='grey',
                                              xuplims=elem_xuplim,
                                              xlolims=elem_xlolim,
                                              uplims=elem_uplim,
                                              lolims=elem_lolim,
                                              marker='d' if elem_source in dippers_list else '.',zorder=-1)

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_Porb_incl[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_Porb_incl[-1][1].set_linestyle('--')

    ax_wind_ib.legend(loc='lower left')


    '''
    #soft x wind detections
    '''

    fig_wind_softx, ax_wind_softx = plt.subplots(figsize=(6, 6))

    ax_wind_softx.set_xlim(1, 1e3)
    ax_wind_softx.set_xscale('log')
    ax_wind_softx.set_ylim(0, 90)

    ax_wind_softx.set_title('Soft X wind detections', fontsize=14)
    ax_wind_softx.set_xlabel('Orbital Period (h)')
    ax_wind_softx.set_ylabel('Inclination (°)')

    sources_wind_softx=[elem for elem in wind_det_sources if wind_det_dict[elem].soft_x is not False]

    sources_wind_softx_sure=[elem for elem in wind_det_sources if wind_det_dict[elem].iron_band is not False and\
                          wind_det_dict[elem].iron_band.trust_noem]

    sources_wind_softx_porb=[elem for elem in sources_wind_softx if elem in sources_incl_porb]

    cmap_wind_softx = mpl.cm.plasma
    norm_colors_wind_softx = mpl.colors.Normalize(vmin=0, vmax=len(sources_wind_softx_porb) - 1)
    colors_func_wind_softx = mpl.cm.ScalarMappable(norm=norm_colors_wind_softx, cmap=cmap_wind_softx)

    color_arr_wind_softx = np.array([colors_func_wind_softx.to_rgba(i) for i in range(len(sources_wind_softx_porb))])

    for i_source, elem_source in enumerate(sources_wind_softx_porb):

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] ==incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_wind_softx = ax_wind_softx.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                              xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                  [Porb_dict[elem_source][1:-1]]).T,
                                              yerr=5 if elem_uplim or elem_lolim else np.array(
                                                  [incl_dict_use[elem_source][1:-1]]).T,
                                              ls='', label=elem_source, color=color_arr_wind_softx[i_source],
                                              alpha=1 if elem_source in sources_wind_softx_sure else 0.3,
                                              xuplims=elem_xuplim,
                                              xlolims=elem_xlolim,
                                              uplims=elem_uplim,
                                              lolims=elem_lolim,
                                              marker='d' if elem_source in dippers_list else '.')

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_wind_softx[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_wind_softx[-1][1].set_linestyle('--')

    # overplotting the rest of the sources
    for i_source, elem_source in enumerate(sources_incl_porb):

        if elem_source in sources_wind_softx_porb:
            continue

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] == incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_Porb_incl = ax_wind_softx.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                            xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                [Porb_dict[elem_source][1:-1]]).T,
                                            yerr=5 if elem_uplim or elem_lolim else np.array(
                                                [incl_dict_use[elem_source][1:-1]]).T,
                                            ls='', label='', color='grey',
                                            xuplims=elem_xuplim,
                                            xlolims=elem_xlolim,
                                            uplims=elem_uplim,
                                            lolims=elem_lolim,
                                            marker='d' if elem_source in dippers_list else '.',zorder=-1)

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_Porb_incl[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_Porb_incl[-1][1].set_linestyle('--')

    ax_wind_softx.legend(loc='lower left')


    '''
     #OIR wind detections
    '''

    fig_wind_oir, ax_wind_oir = plt.subplots(figsize=(6, 6))

    ax_wind_oir.set_xlim(1, 1e3)
    ax_wind_oir.set_xscale('log')
    ax_wind_oir.set_ylim(0, 90)

    ax_wind_oir.set_title('OIR wind detections', fontsize=14)
    ax_wind_oir.set_xlabel('Orbital Period (h)')
    ax_wind_oir.set_ylabel('Inclination (°)')

    sources_wind_oir=[elem for elem in wind_det_sources if (wind_det_dict[elem].visible is not False or\
                                                           wind_det_dict[elem].infrared is not False)]

    sources_wind_oir_sure=[elem for elem in wind_det_sources if ((wind_det_dict[elem].visible is not False and\
                          wind_det_dict[elem].visible.trust_noem) or (wind_det_dict[elem].infrared is not False and\
                          wind_det_dict[elem].infrared.trust_noem))]

    sources_wind_oir_porb=[elem for elem in sources_wind_oir if elem in sources_incl_porb]

    cmap_wind_oir = mpl.cm.plasma
    norm_colors_wind_oir = mpl.colors.Normalize(vmin=0, vmax=len(sources_wind_oir_porb) - 1)
    colors_func_wind_oir = mpl.cm.ScalarMappable(norm=norm_colors_wind_oir, cmap=cmap_wind_oir)

    color_arr_wind_oir = np.array([colors_func_wind_oir.to_rgba(i) for i in range(len(sources_wind_oir_porb))])

    for i_source, elem_source in enumerate(sources_wind_oir_porb):

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] ==incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_wind_oir = ax_wind_oir.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                              xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                  [Porb_dict[elem_source][1:-1]]).T,
                                              yerr=5 if elem_uplim or elem_lolim else np.array(
                                                  [incl_dict_use[elem_source][1:-1]]).T,
                                              ls='', label=elem_source, color=color_arr_wind_oir[i_source],
                                              alpha=1 if elem_source in sources_wind_oir_sure else 0.3,
                                              xuplims=elem_xuplim,
                                              xlolims=elem_xlolim,
                                              uplims=elem_uplim,
                                              lolims=elem_lolim,
                                              marker='d' if elem_source in dippers_list else '.')

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_wind_oir[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_wind_oir[-1][1].set_linestyle('--')

    # overplotting the rest of the sources
    for i_source, elem_source in enumerate(sources_incl_porb):

        if elem_source in sources_wind_oir_porb:
            continue

        elem_xuplim = Porb_dict[elem_source][0] == Porb_dict[elem_source][1]
        elem_xlolim = False
        elem_uplim = incl_dict_use[elem_source][0] == incl_dict_use[elem_source][1]
        elem_lolim = incl_dict_use[elem_source][0] + incl_dict_use[elem_source][2] >= 90

        # axis errorbars
        err_Porb_incl = ax_wind_oir.errorbar(Porb_dict[elem_source][0], incl_dict_use[elem_source][0],
                                            xerr=5 if elem_xuplim or elem_xlolim else np.array(
                                                [Porb_dict[elem_source][1:-1]]).T,
                                            yerr=5 if elem_uplim or elem_lolim else np.array(
                                                [incl_dict_use[elem_source][1:-1]]).T,
                                            ls='', label='', color='grey',
                                            xuplims=elem_xuplim,
                                            xlolims=elem_xlolim,
                                            uplims=elem_uplim,
                                            lolims=elem_lolim,
                                            marker='d' if elem_source in dippers_list else '.',zorder=-1)

        # dashing unsure measurements
        if Porb_dict[elem_source][-1] == 0:
            err_Porb_incl[-1][0].set_linestyle('--')

        if incl_dict_use[elem_source][-1] == 0:
            err_Porb_incl[-1][1].set_linestyle('--')

    ax_wind_oir.legend(loc='lower left')

    #streamlit display

    with tab_param:
        with st.expander('Physical parameter correlations'):
            phys_cols=st.columns(3)
            with phys_cols[0]:
                st.pyplot(fig_refl_dyn)
                st.pyplot(fig_wind_ib)
                st.pyplot(fig_Porb_incl)

            with phys_cols[1]:
                st.pyplot(fig_refl_jet)
                st.pyplot(fig_wind_softx)

            with phys_cols[2]:
                st.pyplot(fig_refl_misc)
                st.pyplot(fig_wind_oir)



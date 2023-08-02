#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:11:45 2022

@author: parrama
"""

#general imports
import os,sys

import glob

import argparse

import numpy as np
import pandas as pd

import streamlit as st
#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
#disabling the warning for many open figures because that's exactly the point of the code
plt.rcParams.update({'figure.max_open_warning': 0})

import matplotlib.colors as colors

import time
from astropy.time import Time,TimeDelta
from copy import deepcopy

#correlation values and uncertainties with MC distribution from the uncertainties
from custom_pymccorrelation import pymccorrelation

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their 
#'perturb values' function

from ast import literal_eval
# import time

import dill
###Astro###

#Catalogs and manipulation
from astroquery.vizier import Vizier

#visualisation functions
from visual_line_tools import load_catalogs,dist_mass,obj_values,abslines_values,values_manip,distrib_graph,correl_graph,incl_dic,\
    n_infos, plot_lightcurve, hid_graph, sources_det_dic, dippers_list,telescope_list


# import mpld3

# import streamlit.components.v1 as components


ap = argparse.ArgumentParser(description='Script to display lines in XMM Spectra.\n)')

###GENERAL OPTIONS###


ap.add_argument("-cameras",nargs=1,help='Cameras to use for the spectral analysis',default='all',type=str)
ap.add_argument("-expmodes",nargs=1,help='restrict the analysis to a single type of exposure',default='all',type=str)
ap.add_argument("-prefix",nargs=1,help='restrict analysis to a specific prefix',default='auto',type=str)
ap.add_argument("-outdir",nargs=1,help="name of output directory for line plots",default="lineplots_opt",type=str)

###DIRECTORY SPECIFICS###

ap.add_argument("-local",nargs=1,help='launch analysis in the current directory instead',default=True,type=bool)

###MODES###

ap.add_argument('-multi_obj',nargs=1,help='compute the hid for multiple obj directories inside the current directory',
                default=True)

###SPECTRUM PARAMETERS###


ap.add_argument("-line_cont_range",nargs=1,help='min and max energies of the line continuum broand band fit',default='4 10',type=str)
ap.add_argument("-line_cont_ig",nargs=1,help='min and max energies of the ignore zone in the line continuum broand band fit',
                default='6.-8.',type=str)
ap.add_argument("-line_search_e",nargs=1,help='min, max and step of the line energy search',default='4 10 0.05',type=str)

ap.add_argument("-line_search_norm",nargs=1,help='min, max and nsteps (for one sign)  of the line norm search (which operates in log scale)',
                default='0.01 10 500',type=str)

###VISUALISATION###

args=ap.parse_args()

#adding the top directory to the path to avoid issues when importing fitting_tools

#local
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/')
#online
sys.path.append('/app/winds/observations/spectral_analysis/')
sys.path.append('/app/winds/general/')

#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std,lines_std_names,range_absline

from general_tools import ravel_ragged
###
# Notes:
# -Only works for the auto observations (due to prefix naming) for now

# -For now we fix the masses of all the objets at 10M_sol

# -Due to the way the number of steps is computed, we explore one less value for the positive side of the normalisation

# -The norm_stepval argument is for a fixed flux band, and the value is scaled in the computation depending on the line energy step
###

cameras=args.cameras
expmodes=args.expmodes
prefix=args.prefix
outdir=args.outdir

#rough way of testing if online or not
online='parrama' not in os.getcwd()


line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
line_cont_ig=args.line_cont_ig
line_search_e=np.array(args.line_search_e.split(' ')).astype(float)
line_search_norm=np.array(args.line_search_norm.split(' ')).astype(float)

multi_obj=args.multi_obj

#don't know where the bug comes from tbf
try:
    st.set_page_config(page_icon=":hole:",layout='wide')
except:
    pass
                   
#readjusting the variables in lists
if cameras=='all':
    cameras=['pn','mos1','mos2','heg']
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

###initialisation###

# #for the current directory:
# started_expos,done_expos=folder_state()
 
# #bad spectra manually taken off
bad_flags=[]


#we create these variables in any case because the multi_obj plots require them
line_search_e_space=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2],line_search_e[2])
#this one is here to avoid adding one point if incorrect roundings create problem
line_search_e_space=line_search_e_space[line_search_e_space<=line_search_e[1]]


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

#######################################
######Hardness-Luminosity Diagrams######
#######################################

#Distance and Mass determination

#wrapped in a function to be cachable in streamlit
if not online:
    catal_blackcat,catal_watchdog,catal_blackcat_obj,catal_watchdog_obj,catal_maxi_df,catal_maxi_simbad=load_catalogs()

st.sidebar.header('Sample selection')
#We put the telescope option before anything else to filter which file will be used
choice_telescope=st.sidebar.multiselect('Telescopes', ['XMM','Chandra']+([] if online else ['NICER','Suzaku','Swift']),default=('XMM','Chandra'))

if online:
    radio_ignore_full=True
else:
    radio_ignore_full=st.sidebar.radio('Include problematic data (_full) folders',('No','Yes'))=='No'

if not online:
    os.system('mkdir -p glob_batch/visual_line_dumps/')

join_telescope_str=np.array(choice_telescope)
join_telescope_str.sort()
join_telescope_str='_'.join(join_telescope_str.tolist())

if online:
    dump_path='/app/winds/observations/visualisation/visual_line_dumps/dump_'+join_telescope_str+'_'+('no' if radio_ignore_full else '')+'full.pkl'
    
    update_dump=False
else:
    dump_path='./glob_batch/visual_line_dumps/dump_'+join_telescope_str+'_'+('no' if radio_ignore_full else '')+'full.pkl'

    update_dump=st.sidebar.button('Update dump')

if not online:
    update_online=st.sidebar.button('Update online version')
else:
    update_online=False
    
if update_online:
    
    #updating script
    path_online=__file__.replace('visual_line','visual_line_online')
    os.system('cp '+__file__+' '+path_online)
    
    #opening the online file and replacing ### by ### due to the magic disabler not working currently
    
    with open(path_online,'r') as online_file:
        online_lines=online_file.readlines()

    for i in range(len(online_lines)):
        
        #trick to avoid replacing the markdown where we don't want to
        #the last replace is to override this specific line (the following one) which gets fucked by the rest, the pass allows a line to remain in the if
        #=#.replace(".markdown('''",'.markdown('''').replace("'''",''''').replace("###",'###').replace("'''","###").replace('#','#')
        
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
        lineval_id='line_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
        lineval_files=[elem for elem in all_files if outdir+'/' in elem and lineval_id in elem and ('/Sample/' in elem or 'XTEJ1701-462/' in elem)]
        
        abslines_id='autofit_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
        abslines_files=[elem for elem in all_files if outdir+'/' in elem and abslines_id in elem and ('/Sample/' in elem or 'XTEJ1701-462/' in elem)]
        
        #telescope selection
        lineval_files=[elem for elem_telescope in choice_telescope for elem in lineval_files if elem_telescope+'/' in elem]
        abslines_files=[elem for elem_telescope in choice_telescope for elem in abslines_files if elem_telescope+'/' in elem]
        
        if radio_ignore_full:
            lineval_files=[elem for elem in lineval_files if '_full' not in elem]
            abslines_files=[elem for elem in abslines_files if '_full' not in elem]
            
        if multi_obj:
            obj_list=np.unique(np.array([elem.split('/')[-4] for elem in lineval_files]))
        else:
            obj_list=np.array([obj_name])
            
        #note: there's no need to order anymore since the file values are attributed for each object of object list in the visual_line functions
        
        #creating the dictionnary for all of the arguments to pass to the visualisation functions
        dict_linevis={
            'ctl_blackcat':catal_blackcat,
            'ctl_blackcat_obj':catal_blackcat_obj,
            'ctl_watchdog':catal_watchdog,
            'ctl_watchdog_obj':catal_watchdog_obj,
            'lineval_files':lineval_files,
            'obj_list':obj_list,
            'cameras':cameras,
            'expmodes':expmodes,
            'multi_obj':multi_obj,
            'range_absline':range_absline,
            'n_infos':n_infos,
            'args_cam':args.cameras,
            'args_line_search_e':args.line_search_e,
            'args_line_search_norm':args.line_search_norm,
            'visual_line':True
            }
        
        #### main arrays computation
        
        #getting the single parameters
        dist_obj_list,mass_obj_list=dist_mass(dict_linevis)
        
        #distance factor for the flux conversion later on
        dist_factor=4*np.pi*(dist_obj_list*1e3*3.086e18)**2
        
        #L_Edd unit factor
        Edd_factor=dist_factor/(1.26e38*mass_obj_list)
        
        #Reading the results files
        observ_list,lineval_list,flux_list,date_list,instru_list,exptime_list=obj_values(lineval_files,Edd_factor,dict_linevis)
        
        dict_linevis['flux_list']=flux_list
        
        #the values here are for each observation
        abslines_infos,autofit_infos=abslines_values(abslines_files,dict_linevis)
        
        #getting all the variations we need
        abslines_infos_perline,abslines_infos_perobj,abslines_plot,abslines_ener,flux_plot,hid_plot,incl_plot,width_plot,nh_plot,kt_plot=values_manip(abslines_infos,dict_linevis,autofit_infos)

        ####(deprecated) deleting bad flags        
        # #taking of the bad files points from the HiD
        # if multi_obj:
            
        #     #in multi object mode, we loop one more time for each object   
        #     for i in range(len(observ_list)):     
                
        #         bad_index=[]
        #         #check if the obsid identifiers of every index is in the bad flag list
        #         for j in range(len(observ_list[i])):
        #             if np.any(observ_list[i][j] in bad_flags):
        #                 bad_index+=[j]
                        
        #         #and delete the resulting indexes from the arrays
        #         observ_list[i]=np.delete(observ_list[i],bad_index)
        #         lineval_list[i]=np.delete(lineval_list[i],bad_index,axis=0)
        #         flux_list[i]=np.delete(flux_list[i],bad_index,axis=0)
        #         # links_list[i]=np.delete(links_list[i],bad_index)
        
        # #same process for a single object
        # else:
        #     bad_index=[]
        
        #     #checking if the observ list isn't empty before trying to delete anything
        #     if len(observ_list)!=0:
        #         for j in range(len(observ_list[0])):
        #             if np.any(observ_list[0][j] in bad_flags):
        #                 bad_index+=[j]
                        
        #         #and delete the resulting indexes from the arrays
        #         observ_list[0]=np.delete(observ_list[0],bad_index)
        #         lineval_list[0]=np.delete(lineval_list[0],bad_index,axis=0)
        #         flux_list[0]=np.delete(flux_list[0],bad_index,axis=0)
        #         # links_list[0]=np.delete(links_list[0],bad_index)
        
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
        dump_dict['abslines_plot']=abslines_plot
        dump_dict['hid_plot']=hid_plot
        dump_dict['flux_plot']=flux_plot
        dump_dict['nh_plot']=nh_plot
        dump_dict['kt_plot']=kt_plot
        dump_dict['incl_plot']=incl_plot
        dump_dict['abslines_infos_perobj']=abslines_infos_perobj
        dump_dict['flux_list']=flux_list
        dump_dict['abslines_ener']=abslines_ener
        dump_dict['width_plot']=width_plot
        dump_dict['dict_linevis']=dict_linevis
        dump_dict['catal_maxi_df']=catal_maxi_df
        dump_dict['catal_maxi_simbad']=catal_maxi_simbad
        dump_dict['dict_lc_rxte']=dict_lc_rxte
        
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
abslines_plot=dump_dict['abslines_plot']
hid_plot=dump_dict['hid_plot']
flux_plot=dump_dict['flux_plot']
kt_plot=dump_dict['kt_plot']
nh_plot=dump_dict['nh_plot']
incl_plot=dump_dict['incl_plot']
abslines_infos_perobj=dump_dict['abslines_infos_perobj']
flux_list=dump_dict['flux_list']
abslines_ener=dump_dict['abslines_ener']
width_plot=dump_dict['width_plot']
dict_linevis=dump_dict['dict_linevis']
catal_maxi_df=dump_dict['catal_maxi_df']
catal_maxi_simbad=dump_dict['catal_maxi_simbad']
dict_lc_rxte=dump_dict['dict_lc_rxte']

###
# in the abslines_infos_perline form, the order is:
#     -each habsorption line
#     -the number of sources
#     -the number of obs for each source
#     -the info (5 rows, EW/bshift/Del-C/sign)
#     -it's uncertainty (3 rows, main value/neg uncert/pos uncert,useless for the Del-C and sign)
###

#checking if the obsid identifiers of every index is in the bad flag list or if there's just no file
if len(observ_list.ravel())==0:
    st.write('\nNo line detection to build HID graph.')

#some naming variables for the files
save_dir='glob_batch' if multi_obj else outdir

if multi_obj:
    save_str_prefix=''
else:
    save_str_prefix=obj_list[0]+'_'

###Page creation###
#### Streamlit page creation

line_display_str=np.array([r'FeXXV Ka (6.70 keV)',r'FeXXVI  Ka (6.97 keV)','NiXXVII Ka (7.80 keV)',
                      'FeXXV Kb (7.89 keV)','FeXXVI Kb (8.25 keV)','FeXXVI Kg (8.70 keV)'])

if multi_obj:
    radio_single=st.sidebar.radio('Display options:',('All Objects','Multiple Objects','Single Object'))
    
    if radio_single=='Single Object':
        display_single=True
    else:
        display_single=False
    
    if radio_single=='Multiple Objects':
        display_multi=True
    else:
        display_multi=False
        
    if display_multi:
        restrict_sources_detection=st.sidebar.checkbox('Restrict to sources with significant detection')
        ####source with det restriction done manually as of now, should be changed
        
    if display_multi:
        with st.sidebar.expander('Source'):
            choice_source=st.multiselect('',options=[elem for elem in obj_list if elem in sources_det_dic] if restrict_sources_detection else obj_list,default=[elem for elem in obj_list if elem in sources_det_dic] if restrict_sources_detection else obj_list)     
        
    if display_single:
        #switching to array to keep the same syntax later on
        choice_source=[st.sidebar.selectbox('Source',obj_list)]

####Nickel display is turned off here
with st.sidebar.expander('Absorption lines restriction'):
    selectbox_abstype=st.multiselect('',
                    options=line_display_str[:2].tolist()+line_display_str[3:].tolist(),default=line_display_str[:2])

#creating the line mask from that
mask_lines=np.array([elem in selectbox_abstype for elem in line_display_str])

with st.sidebar.expander('Inclination'):
    slider_inclin=st.slider('Inclination restriction (°)',min_value=0.,max_value=90.,step=0.5,value=[0.,90.])
    
    include_noinclin=st.checkbox('Include Sources with no inclination information',value=True)
    
    incl_inside=st.checkbox('Only include sources with uncertainties strictly compatible with the current limits',value=False)
    
    display_incl_inside=st.checkbox('Display ULs differently for sources with uncertainties not strictly compatible with the current limits',value=False)
    
    dash_noincl=st.checkbox('Display ULs differently for sources with no inclination information',value=False)
    
    radio_dipper=st.radio('Dipping sources restriction',('Off','Add dippers','Restrict to dippers','Restrict to non-dippers'))
    
    
#not used currently
# radio_dispmode=st.sidebar.radio('HID display',('Autofit line detections','blind search peak detections'))
# if radio_dispmode=='Autofit line detections':
#     display_final=True
# else:
#     display_final=False

#     radio_info_cmap=st.sidebar.radio('Color map options:',('Source','Peak Del-C'))
#     slider_ener=st.sidebar.slider('Peak energy range',min_value=line_search_e[0],max_value=line_search_e[1],
#                                   step=line_search_e[2],value=[6.,9.])
#     display_abslines=st.sidebar.checkbox('Display absorption Lines')
#     display_emlines=st.sidebar.checkbox('Display emission Lines')
    
restrict_time=st.sidebar.checkbox('Restrict time interval',value=True)
        
slider_sign=st.sidebar.slider('Detection significance treshold',min_value=0.9,max_value=1.,step=1e-3,value=0.997,format="%.3f")

####Streamlit HID options
st.sidebar.header('HID options')

#not used right now
# else:
#     #full true mask
#     mask_lines=np.array(line_display_str!='')
            


display_nonsign=st.sidebar.checkbox('Show detections below significance threshold',value=False)

if display_nonsign:
    restrict_threshold=st.sidebar.checkbox('Prioritize showing maximal values of significant detections',value=True)
else:
    restrict_threshold=True
        
HID_options_str=np.array(['Source','Velocity shift',r'Delta C','EW ratio','Inclination','Time','Instrument','Column density','Disk temperature'])
radio_info_cmap_str=st.sidebar.radio('HID colormap',HID_options_str,index=0)

radio_info_index=np.argwhere(HID_options_str==radio_info_cmap_str)[0][0]
                             
radio_info_cmap=['Source','Velocity shift','Del-C','EW ratio','Inclination','Time','Instrument','nH','kT'][radio_info_index]

if radio_info_cmap!='Source':
    display_edgesource=st.sidebar.checkbox('Color code sources in marker edges',value=False)
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
        selectbox_ratioeqw=st.sidebar.multiselect('',options=line_display_str[mask_lines],default=line_display_str[mask_lines][:2])
        
else:
    selectbox_ratioeqw=''
    
checkbox_zoom=st.sidebar.checkbox('Zoom around the displayed elements',value=False)
            
display_nondet=st.sidebar.checkbox('Display exposures with no detection',value=True)

if display_nondet:
    with st.sidebar.expander('Upper limits'):
        display_upper=st.checkbox('Display upper limits',value=True)    
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
    display_evol_single=st.sidebar.checkbox('Highlight time evolution in the HID',value=False)
else:
    display_evol_single=False
    
if not online:
    save_format=st.sidebar.radio('Graph format:',('pdf','svg','png'))
    
    def save_hld():
        ###
        # Saves the current graph in a svg (i.e. with clickable points) format.
        ###
    
        fig_hid.savefig(save_dir+'/'+save_str_prefix+'HLD_cam_'+args.cameras+'_'+\
                    args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'curr_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
            
    st.sidebar.button('Save current HID view',on_click=save_hld)

        
with st.sidebar.expander('Visualisation'):
    
    display_dicho=st.checkbox('Display favourable zone',value=True)
    
    display_obj_zerodet=st.checkbox('Color sources with no detection',value=True)
    
    display_hid_error=st.checkbox('Display errorbar for HID position',value=False)
    
    display_central_abs=st.checkbox('Display centers for absorption detections',value=False)

    alpha_abs=st.checkbox('Plot with transparency',value=False)
    
    split_cmap_source=st.checkbox('Use different colormaps for detections and non-detections',value=True)
    
    global_colors=st.checkbox('Normalize colors/colormaps over the entire sample',value=False)
        
    if not online:
        paper_look=st.checkbox('Paper look',value=False)

        bigger_text=st.checkbox('Bigger text size',value=True)
        
        square_mode=st.checkbox('Square mode',value=True)
    
        show_linked=st.checkbox('Distinguish linked detections',value=False)
    else:
        paper_look=False
        bigger_text=True
        square_mode=True
        show_linked=False
        
if alpha_abs:
    alpha_abs=0.5
else:
    alpha_abs=1
    
with st.sidebar.expander('Monitoring'):
    
    plot_lc_monit=st.checkbox('Plot monitoring lightcurve',value=False)
    plot_hr_monit=st.checkbox('Plot monitoring HR',value=False)
        
    monit_highlight_hid=st.checkbox('Highlight HID coverage',value=False)
    
    if plot_lc_monit or plot_hr_monit:
        zoom_lc=st.checkbox('Zoom on the restricted time period in the lightcurve',value=False)
    else:
        zoom_lc=False
        
    fig_lc_monit=None
    fig_hr_monit=None
    
    plot_maxi_ew=st.checkbox('Superpose measured EW',value=False)
    
    def save_lc():
        
        ###
        # Saves the current maxi_graph in a svg (i.e. with clickable points) format.
        ###
        if display_single:
            fig_lc_monit.savefig(save_dir+'/'+'LC_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
            fig_hr_monit.savefig(save_dir+'/'+'HR_'+choice_source[0]+'_'+str(round(time.time()))+'.'+save_format,bbox_inches='tight')
            
    st.button('Save current MAXI curves',on_click=save_lc,key='save_lc_key')
    
compute_only_withdet=st.sidebar.checkbox('Skip parameter analysis when no detection remain with the current constraints',value=True)

with st.sidebar.expander('Stacking'):
    stack_det=st.checkbox('Stack detections')
    stack_flux_lim = st.number_input(r'Max ratio of fluxes to stack', value=2., min_value=1e-10, format='%.3e')
    stack_HR_lim=st.number_input(r'Max ratio of HR to stack',value=2.,min_value=1e-10,format='%.3e')
    stack_time_lim=st.number_input(r'Max time delta to stack',value=2.,min_value=1e-1,format='%.3e')

mpl.rcParams.update({'font.size': 10+(3 if paper_look else 0)})

if not square_mode:
    fig_hid,ax_hid=plt.subplots(1,1,figsize=(8,5) if bigger_text else (12,6))
else:
    fig_hid,ax_hid=plt.subplots(1,1,figsize=(8,6))
ax_hid.clear()


###HID GRAPH###

#log x scale for an easier comparison with Ponti diagrams
ax_hid.set_xscale('log')
ax_hid.set_xlabel('Hardness Ratio ([6-10]/[3-6] keV bands)')
ax_hid.set_ylabel(r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units')
ax_hid.set_yscale('log')


###Dichotomy###

#some warnings to avoid crashes
if radio_single !='All Objects' and len(choice_source)<1:
    st.warning('Please select at least one Source.')
    st.stop()
    
if len(selectbox_abstype)<1:
    st.warning('Please select at least one line.')
    st.stop()

#fetching the line indexes when plotting EW ratio as colormap
eqw_ratio_ids=np.argwhere([elem in selectbox_ratioeqw for elem in line_display_str]).T[0]

if radio_info_cmap=='EW ratio' and len(eqw_ratio_ids)<2:
    st.warning('Cannot build EW ratio colormap from current line restriction')
    st.stop()


#string of the colormap legend for the informations
radio_info_label=['Velocity shift', r'$\Delta-C$', 'Equivalent width ratio']

#masking for restriction to single objects
if display_single or display_multi:
    mask_obj_select=np.array([elem in choice_source for elem in obj_list])
else:
    mask_obj_select=np.repeat(True,len(obj_list))
    
#masking the objects depending on inclination
mask_inclin=[include_noinclin if elem not in incl_dic else getoverlap((incl_dic[elem][0]-incl_dic[elem][1],incl_dic[elem][0]+incl_dic[elem][2]),slider_inclin)>0 for elem in obj_list]

#creating the mask for highlighting objects whose inclination limits go beyond the inclination restrictions if asked to
bool_incl_inside=np.array([False if elem not in incl_dic else round(getoverlap((incl_dic[elem][0]-incl_dic[elem][1],
            incl_dic[elem][0]+incl_dic[elem][2]),slider_inclin),3)==incl_dic[elem][1]+incl_dic[elem][2] for elem in obj_list])

bool_noincl=np.array([True if elem not in incl_dic else False for elem in obj_list])
    
    
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

if restrict_time:
    slider_date=st.slider('Dates restriction',min_value=(Time(min(ravel_ragged(date_list[mask_obj_base])))-delta_1y).datetime,
                          max_value=(Time(max(ravel_ragged(date_list[mask_obj_base])))+delta_1y).datetime,
                          value=[(Time(min(ravel_ragged(date_list[mask_obj_base])))-delta_1m).datetime,
                                 (Time(max(ravel_ragged(date_list[mask_obj_base])))+delta_1m).datetime])
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

#checking which sources have no detection in the current combination
global_displayed_sign=np.array([ravel_ragged(elem)[mask] for elem,mask in zip(abslines_plot[4][0][mask_lines].T,mask_obs_intime_repeat)],dtype=object)

#creating a mask from the ones with at least one detection 
#(or at least one significant detections if we don't consider non significant detections)
if display_nonsign:
    mask_obj_withdet=np.array([(elem>0).any() for elem in global_displayed_sign])
else:
    mask_obj_withdet=np.array([(elem>slider_sign).any() for elem in global_displayed_sign])

#storing the number of objects with detections
n_obj_withdet=sum(mask_obj_withdet & mask_obj_base)

if not display_obj_zerodet:
    mask_obj=mask_obj_base & mask_obj_withdet
else:
    mask_obj=mask_obj_base
    
hid_plot_restrict=hid_plot.T[mask_obj].T
incl_plot_restrict=incl_plot[mask_obj]

#creating variables with values instead of uncertainties for the inclination and nh colormaps

incl_cmap=np.array([incl_plot.T[0],incl_plot.T[0]-incl_plot.T[1],incl_plot.T[0]+incl_plot.T[2]]).T
incl_cmap_base=incl_cmap[mask_obj_base]
incl_cmap_restrict=incl_cmap[mask_obj]

nh_plot_restrict=deepcopy(nh_plot)
nh_plot_restrict=nh_plot_restrict.T[mask_obj].T

kt_plot_restrict=deepcopy(kt_plot)
kt_plot_restrict=kt_plot_restrict.T[mask_obj].T

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

    global_mask_intime=(Time(ravel_ragged(global_plotted_datetime))>=Time(slider_date[0])) &\
        (Time(ravel_ragged(global_plotted_datetime))<=Time(slider_date[1]))

    global_mask_intime_norepeat=(Time(ravel_ragged(date_list[mask_obj]))>=Time(slider_date[0])) &\
        (Time(ravel_ragged(date_list[mask_obj]))<=Time(slider_date[1]))

#global_nondet_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])<=slider_sign) & (global_mask_intime)

global_det_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])>0) & (global_mask_intime)

global_sign_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])>slider_sign) & (global_mask_intime)

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

#necessary items for the hid graph run
items_list=[
abslines_infos_perobj,
abslines_plot,nh_plot,kt_plot,hid_plot, incl_plot,
mask_obj, mask_obj_base, mask_lines, mask_lines_ul,
obj_list, date_list, instru_list, flux_list, choice_telescope, telescope_list,
bool_incl_inside, bool_noincl,
slider_date, slider_sign,
radio_info_cmap, radio_cmap_i,
cmap_color_source, cmap_color_det, cmap_color_nondet]

items_str_list=['abslines_infos_perobj',
'abslines_plot','nh_plot','kt_plot','hid_plot','incl_plot',
'mask_obj','mask_obj_base', 'mask_lines', 'mask_lines_ul',
'obj_list', 'date_list', 'instru_list', 'flux_list', 'choice_telescope', 'telescope_list',
'bool_incl_inside', 'bool_noincl',
'slider_date', 'slider_sign',
'radio_info_cmap', 'radio_cmap_i',
'cmap_color_source', 'cmap_color_det', 'cmap_color_nondet']

for dict_key, dict_item in zip(items_str_list,items_list):
    dict_linevis[dict_key]=dict_item


hid_graph(ax_hid,dict_linevis,
          display_single=display_single, display_nondet=display_nondet, display_upper=display_upper,
          cyclic_cmap_nondet=cyclic_cmap_nondet, cyclic_cmap_det=cyclic_cmap_det, cyclic_cmap=cyclic_cmap,
          cmap_incl_type=cmap_incl_type, cmap_incl_type_str=cmap_incl_type_str,
          radio_info_label=radio_info_label,
          eqw_ratio_ids=eqw_ratio_ids,
          display_obj_zerodet=display_obj_zerodet,
          restrict_threshold=restrict_threshold, display_nonsign=display_nonsign,
          display_central_abs=display_central_abs,
          display_incl_inside=display_incl_inside, dash_noincl=dash_noincl,
          display_hid_error=display_hid_error, display_edgesource=display_edgesource,
          split_cmap_source=split_cmap_source,
          display_evol_single=display_evol_single, display_dicho=display_dicho,
          global_colors=global_colors, alpha_abs=alpha_abs,
          paper_look=paper_look, bigger_text=bigger_text, square_mode=square_mode, zoom=checkbox_zoom)

# fig_hid_html = mpld3.fig_to_html(fig_hid)
# components.html(fig_hid_html, height=1000)

tab_hid, tab_monitoring, tab_param,tab_source_df,tab_about= st.tabs(["HID", "Monitoring", "Parameter analysis","Tables","About"])

with tab_hid:
    
    #preventing some bugs when updating too many parameters at once
    try:
        st.pyplot(fig_hid)
    except:
        st.experimental_rerun()
        
#### About tab
with tab_about:
    st.markdown('**visual_line** is a visualisation and download tool for iron-band X-ray absorption lines signatures in Black Hole Low-Mass X-ray Binaries (BHLMXBs).')
    st.markdown('It is made to complement and give access to the results of [Parra et al. 2023](https://arxiv.org/abs/2308.00691), and more generally, to give an overview of the sampling and X-ray evolution of the outbursts of this category of sources.')
    st.markdown('Please contact me at [maxime.parra@univ-grenoble-alpes.fr](mailto:maxime.parra@univ-grenoble-alpes.fr) for questions, to report bugs or request features.')
    
    with st.expander('I want an overview of the science behind this'):
        st.header('Outbursts')
        st.markdown('''
        X-Ray Binaries are binary systems emitting mostly in the X-ray band, composed of a compact object (Neutron Star or Black Hole) and a companion star.  
        The subgroup we are focusing on here is restricted to **Black Holes** orbiting with a "low mass" star (generally in the main sequence), for which accretion happens through Robe Loche overflow and an **accretion disk**. 
        Most of these sources have a very specific behavior: they spend the bigger part of their lifes undetected, at low luminosity (in "quiescence"), 
        but undergo regular **outbursts** on the timescale of a few months.
        During these outbursts, sources brighten by several orders of magnitude, up to a significant percentage of their Eddington luminosity $L_{Edd}$, which is the "theoretical" upper limit of standard accreting regimes.  
        
        Meanwhile, these objects follow a specific pattern of spectral evolution, switching from a powerlaw dominated Spectral Energy Distribution (SED) in soft X-rays (the so-called **"hard"** state) during the initial brightening, to the **"soft"** state, similarly bright but dominated by the thermal emission of the accretion disk (most likely extending to the Innermost Stable Circular Orbit of the Black Hole). After some time spent during the soft state (and occasional back and forth in some instances), the source invariably becomes fainter, transits back to the hard state if necessary, then returns to quiescence.
        ''')
        
        col_fig1, col_fig2= st.columns(2)
        with col_fig1:
            st.image(dump_path[:dump_path.rfind('/')]+'/outburst.png',caption='Example of the evolution of GX 339-4 in a Hardness/Luminosity Diagram during its 2019 outburst. the MJD dates of each observation highlight the direction of the evolution, and colors different spectral states (independant from the right picture). From Petrucci et al. 21')
        with col_fig2:
            st.image(dump_path[:dump_path.rfind('/')]+'/xray_states.png',caption="Example of the differences between spectral shapes for the soft (red) and hard (blue) state of Cygnus X-1. From Done et al. 2007")
            
        st.markdown('''
        Beyond this direct spectral dichotomy, a wealth of other features have been linked to the outburst evolution:  
            -a radio component associated to **jets** is only detected during the hard state  
            -specific **Quasi-Periodic Oscillations** (QPOs) can be detected in the timing features of the spectra, with different types in different spectral states, and can be linked to a wealth of geometrical and physicla behaviors  
            -**Dipping** episodes can be observed in the lightcurve of high-inclined sources in specific portions of the outburst. (Note: this is a different behavior than eclipses).  
            -Some sources exhibit strong **reflection** components during a part of the outburst, interpreted as internal illumination of the disk by the **hot corona** believed to be the source of the hard state powerlaw component.   
            -Other sources exhibit strongly relativistic **iron-line emission** features, which can be fitted to get constraints on the spin of the BH.  
            -...
        
        As of now, the reason for the typical spectral evolution of outbursts, as well as the cause of most of the features that appear during it, remains unknown.
        ''')
        
        col_winds, col_figwinds= st.columns(2)
        
        with col_winds:
            
            st.header('Winds')
            
            st.markdown('''
            Another features seen in outbursts is the appearance of **narrow, blueshifted absorption lines** in X-ray spectra, primarily from the very strong Ka and Kb lines of FeXXV and FeXXVI at 7-8keV.
            They are interpreted as the signature of dense material outflowing from the accretion disk of the Black Hole, and are expected to expell amounts of matter comparable to the accretion rate.
            Since the first observations 25 ago, a wealth of absorption profiles have been observed in BHLMXBS.
            Wind signatures in **X-rays** have been traditionally found only in the soft states of high-inclined sources, but recent detections in hard states or for low-inclined sources challenge this assumption.
            
            One of the most critical matters about winds is their physical origin. In X-ray Binaries, two launching mechanisms are favored, using either **thermal** or **magnetic** processes. Modeling efforts are recent and only few observations have been successfully recreated by either until now, but this has shown the limit of current instruments. Indeed, it appears impossible to directly distinguish the wind launch mechanisms by simply using the absorption signatures, even for the best quality observations of the current generation instruments. Thus, until data from the new instruments, such as XRISM, becomes available, hope lies in the constraints on the physical parameters of the model creating the winds, or a more complete model to data comparison. 
            
            The **JED-SAD model** is a complete accretion-ejection framework for magnetically threaded disks, developed at the University of Grenoble-Alpes (France). Beyond very promising results for fitting all
            parts of the outburst of Black Hole XRBs, this model has been shown to produce winds, through both theoretical solutions and simulations. The results of this study will be compared to synthetic spectral signatures computed from self-similar JED-SAD solutions in the future, in order to access the evolution of the outflow, and to further constrain the nature and physical conditions of the disk during these observations.
            
            ''')
            
            st.header('This work')
            st.markdown('''
                        The science community and our own modeling efforts would benefit from a global and up-to-date view of the current wind signatures in BHLMXBs. However, while detailed works have been performed on the vast majority of individual detections, there are very few larger studies for several outbursts and sources. With the goal of providing a complete view of all currently known X-ray wind signatures, we first focus on the most historically studied and constraining observations, using the XMM-Newton and Chandra-HETG instruments.  
                        
                        We identify BHLMXB candidates through the BlackCAT and WATCHDOG catalogs, for a total of 79 sources. After extracting and pre-selecting all available spectra of these sources with high-enough quality to allow the detection of line, we end up with 242 spectra in 42 sources. We refer readers to the main paper for details on the line detection procedure.
                        
                        Beyond interactive displays of our results through HID and scatter plots, we provide direct access to the results table, restricted according to user demands. We also provide a monitoring display tool, which combines RXTE and up-to-date MAXI lightcurves and HR ratio evolutions of all single sources in the sample.
                        ''')
        
        with col_figwinds:
            st.image(dump_path[:dump_path.rfind('/')]+'/linedet_example.jpg',caption='Steps of the fitting procedure for a standard 4U130-47 Chandra spectra. First panel: 4-10 spectrum after the first continuum fit. Second panel: ∆C map of the line blind search, restricted to positive (i.e. improvements) regions. Standard confidence intervals are highlighted with different line styles, and the colormap with the ∆C improvements of emission and absorption lines. Third panel: Ratio plot of the best fit model once absorption lines are added. Fourth panel: Remaining residuals seen through a second blind search.')
            
        st.markdown('''
                    See [Parra et al. 2023](https://arxiv.org/abs/2308.00691) for detailed references to the points discussed above, and [Diaz Trigo et al. 2016](https://doi.org/10.1002/asna.201612315) or [Ponti et al. 2016](https://doi.org/10.1002/asna.201612339) for reviews on winds.  
                    
                    Figure references:  
                        [Petrucci et al. 2021](https://doi.org/10.1051/0004-6361/202039524)  
                        [Done et al. 2007]((https://doi.org/10.1007/s00159-007-0006-1))  
                    ''')
                
    with st.expander('I want to know how to use this tool'):
        
        st.header('General information')
        
        st.markdown('''
                    Streamlit is a python library for web applications. 
                    Its interactivity is achieved by storing the status of the multiple widgets and re-running **the entire script** every time a modification is performed, allowing to recompute/redisplay/... all necessary outputs depending on the changes made to the widgets. 
                    
                    **The monitoring and parameter analysis plots take time to compute**, as the first one fetches data from the MAXI website in real time, and the second computes perturbative estimates of the correlation coefficient for each plot. As such, they are deactivated by default.  
                    
                    It is worth noting that the current version of this tool has trouble performing too many actions in a short time. This is partially covered through internal failsafes, but if you modify several options at once and something crashes, or displays in a non-standard way, either resettoing an option (which reruns the script internally) or restarting the tool (either by pressing R or going in the top-right menu and clicking on Rerun) can fix the issue.
                    
                    Moreover, to avoid too much issues for data management, the data is split between each combination of instrument, and the creation of subsequent widget depends on this data.
                    
                    **This means that all subsequent widgets are reset to their default values whenever adding or removing an instrument.**

                    ''')
                    
        st.header('Data selection')

        st.markdown('''
                    
                   The following options restrict the data used in **all** of the displays.
                   
                   **Telescopes**  
                   
                   This is rather straightforward. Remember the statement above: changing this resets everything else.
                   
                   **Sources**  
                   
                   The Main source display option conditions several elements further down the line:
                       
                  -In Multi-Object mode, a checkbox allows to pre-restrict the choice of sources with the 5 main objects with significant detections in the paper. This is a manual restriction, which is not affected by the choice of lines or significance threshold.
                  
                  -The display of all monitoring plots is restricted to the **single object** mode. 
                  
                  
                  **Inclination**  
                  
                  The inclination restrictions is based on the results on the informations in Table 1 of the paper, which can be shown in the "Source parameters" table in the "Tables" tab. Since not all inclination measurements are created equal, the inclination measurement primarily includes dynamical estimates if any, then jet estimates, and finally reflection measurements.
                  
                  -Dipping measurements are considered independantly, which means that a source with "only" dipping reports and no explicit inclination constraints will be ignored if the checkbox for including Sources with no information is not checked.
                  
                  -The second checkbox allows to disregard sources with uncertainties extending past the currently selected inclination range. Incompatible inclination measurements are considered similarly as a bigger interval.
                  
                  -The third option dashes the upper limit (hexagon) displays of sources incompatible with previous constrain.
                  
                  -The fourth option dashes all sources with no inclination measurements (including dippers with no proper inclination constraints).
                  
                  -The dipper option choice overrides the primary inclination constrain and checkboxes
                  
                  **Date** 
                  
                  The time interval restriction is deported on the main page for more precise changes but is also a main selection widget affecting all the following elements. Currently, the duration of the observation is not considered.
                  
                  **Significance**
                  
                  The detection significance uses the final MC significance (see the paper for details) of each line. All existing lines must have been added to the models, which means they were at least 99\% significant with the f-test during the automatic fitting procedure. This can still mean very poor MC significance in case of bad data or degenerate components. 
                  
                  The MC tests were done with 1000 computations, which means that using 1. as a significance threshold only restricts to detections above 0.999.
                   
                  ''')
                    
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
                   
                   ''')
        
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
                   
                   ''')
                   
        
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
                    ''')
                    
        
        st.header('Data display and download (Tables)')

        st.markdown('''
                    the complete data of sources, observation and line parameters are displayed according to the current selection made in the sidebar,
                    and can be downloaded through separate csvs file that can be loaded as multi-dimensional dataframes.
                    ''')
                    
#### Transposing the tables into plot arrays

flag_noabsline=False

#bin values for all the histograms below
#for the Velocity shift and energies the range is locked so we can use a global binning for all the diagrams
bins_bshift=np.linspace(-2e3,3e3,num=26,endpoint=True)
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

#####################
####Creating and plotting the dataframes
#####################

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

###
#SOURCE TABLE
###
    
source_df_arr=np.array([obj_list,dist_obj_list,mass_obj_list,incl_plot.T[0],incl_plot.T[1],incl_plot.T[2],
                       [sum(elem=='XMM') for elem in instru_list],[sum(elem=='Chandra') for elem in instru_list]]).astype(str).T

source_df=pd.DataFrame(source_df_arr,columns=['source','distance (kpc)','mass (M_sun)','inclination (°)','incl err -','incl err +',
                                              'XMM exposures','Chandra exposures'])
with tab_source_df:
    
    with st.expander('Source parameters'):
        
        st.dataframe(source_df)
    
        csv_source = convert_df(source_df)
    
        st.download_button(
            label="Download as CSV",
            data=csv_source,
            file_name='source_table.csv',
            mime='text/csv',
        )
        
        
###
#OBS & LINE TABLES
###

#n_obj_restricted
n_obj_r=sum(mask_obj)

#creating an array for the intime observations
mask_intime_plot=np.array([(Time(date_list[mask_obj][i_obj_r].astype(str))>=Time(slider_date[0])) & (Time(date_list[mask_obj][i_obj_r].astype(str))<=Time(slider_date[1])) for i_obj_r in range(n_obj_r)],dtype=object)


#and an date order 
order_intime_plot_restrict=np.array([np.array([Time(elem) for elem in date_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)]]).argsort() for i_obj_r in range(n_obj_r)],dtype=object)

#creating  4 dimensionnal dataframes for the observ and line informations

observ_df_list=[]

line_df_list=[]

abs_plot_tr=np.array([[subelem for subelem in elem] for elem in abslines_plot_restrict]).transpose(3,2,0,1)

line_rows=np.array(lines_std_names[3:9])[mask_lines]
    
for i_obj_r in range(n_obj_r):
    
    n_obs_r=sum(mask_intime_plot[i_obj_r].astype(bool))
    
    #recreating an individual non ragged array (similar to abslines_perobj) in construction for each object
    hid_plot_indiv=np.array([[subelem for subelem in elem] for elem in hid_plot_restrict.transpose(2,0,1)[i_obj_r]],dtype=float)
    
    line_plot_indiv=np.array([[[subsubelem for subsubelem in subelem] for subelem in elem] for elem in abs_plot_tr[i_obj_r]],dtype=float)
    
    #applying the intime mask on each observation and sorting by date

    hid_plot_indiv=hid_plot_indiv.transpose(2,0,1)[mask_intime_plot[i_obj_r].astype(bool)][order_intime_plot_restrict[i_obj_r].astype(int)].transpose(1,2,0)

        
    line_plot_indiv=line_plot_indiv.transpose(3,0,1,2)[mask_intime_plot[i_obj_r].astype(bool)][order_intime_plot_restrict[i_obj_r].astype(int)].transpose(2,3,0,1)
    
    ###
    # splitting information to take off 1 dimension and only take specific information    
    # EW, bshift, width, flux, sign, upper
    ###
    used_indexes=[[0,0],[0,1],[0,2],
                  [1,0],[1,1],[1,2],
                  [7,0],[7,1],[7,2],
                  [3,0],[3,1],[3,2],
                  [4,0],
                  [5,0]]
                  
    line_plot_indiv=np.array([line_plot_indiv[elem[0]][elem[1]] for elem in used_indexes])
    
    #creating the row indexes (row: object, subrow: observation)    
    observ_list_indiv=observ_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)].tolist()
    
    observ_list_indiv=[elem.replace('_Imaging_auto','').replace('_Timing_auto','').replace('_heg_-1','').replace('_heg_1','') for elem
                       in observ_list_indiv]
    
    iter_rows=[[obj_list[mask_obj][i_obj_r]],
               observ_list_indiv,
               date_list[mask_obj][i_obj_r][mask_intime_plot[i_obj_r].astype(bool)][order_intime_plot_restrict[i_obj_r].astype(int)].tolist()]
    
    
    #creating the iter index manually because we have two clumns (observ and time) not being dimensions of one another
    row_index_arr_obs=np.array([[iter_rows[0][0],iter_rows[1][i_obs_r],iter_rows[2][i_obs_r]] for i_obs_r in range(n_obs_r)]).T
    
    row_index_arr_line=np.array([[iter_rows[0][0],iter_rows[1][i_obs_r],iter_rows[2][i_obs_r],line_rows[i_line_r]]\
                                for i_obs_r in range(n_obs_r) for i_line_r in range(sum(mask_lines))]).T
    
    row_index_obs=pd.MultiIndex.from_arrays(row_index_arr_obs,names=['Source','obsid','date'])
    
    row_index_line=pd.MultiIndex.from_arrays(row_index_arr_line,names=['Source','obsid','date','line'])
    
    #you can use the standard way for columns for the observ df
    iter_columns=[['HR [6-10]/[3-10]','Lx/LEdd'],['main','err-','err+']]
    
    #but not for the line df
    column_index_arr_line=np.array([['EW','main'],['EW','err-'],['EW','err+'],
                          ['blueshift','main'],['blueshift','err-'],['blueshift','err+'],
                          ['width','main'],['width','err-'],['width','err+'],
                          ['line flux','main'],['line flux','err-'],['line flux','err+'],
                          ['sign','main'],['EW UL','main']]).T
                          
    column_index_line=pd.MultiIndex.from_arrays(column_index_arr_line,names=['measure','value'])
    
    #creating both dataframes, with a reshape in 2 dimensions (one for the lines and one for the columns)
    observ_df_list+=[produce_df(hid_plot_indiv.transpose(2,0,1).reshape(n_obs_r,6),iter_rows,iter_columns,row_names=['Source','obsid','date'],
                            column_names=['measure','value'],row_index=row_index_obs)]

    line_df_list+=[produce_df(line_plot_indiv.transpose(1,2,0).reshape(n_obs_r*sum(mask_lines),14),None,None,row_names=None,
                            column_names=None,row_index=row_index_line,col_index=column_index_line)]
    
observ_df=pd.concat(observ_df_list)
    
line_df=pd.concat(line_df_list)

        
with tab_source_df:
    
    with st.expander('Observation parameters'):
        
        st.dataframe(observ_df,use_container_width=True)
    
        csv_observ= convert_df(observ_df)
    
        st.download_button(
            label="Download as CSV",
            data=csv_observ,
            file_name='observ_table.csv',
            mime='text/csv',
        )
        
    with st.expander('Line parameters'):
        
        st.dataframe(line_df,use_container_width=True)
    
        csv_observ= convert_df(line_df)
    
        st.download_button(
            label="Download as CSV",
            data=csv_observ,
            file_name='line_table.csv',
            mime='text/csv',
        )
        

#####################
 ####Monitoring
#####################

with tab_monitoring:
    if plot_lc_monit:
        
        if not display_single:
            st.info('Lightcurve monitoring plots are restricted to single source mode.')
            
        else:
            with st.spinner('Building lightcurve...'):
                fig_lc_monit=plot_lightcurve(dict_linevis,catal_maxi_df,catal_maxi_simbad,choice_source,display_hid_interval=monit_highlight_hid,
                                                 superpose_ew=plot_maxi_ew,dict_rxte=dict_lc_rxte)
            
                #wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_lc_monit is not None:
                    st.pyplot(fig_lc_monit)
    
    if plot_hr_monit:
        
        if not display_single:
            st.info('HR monitoring plots are restricted to single source mode.')
            
        else:
            with st.spinner('Building HR evolution...'):
                fig_hr_monit=plot_lightcurve(dict_linevis,catal_maxi_df,catal_maxi_simbad,choice_source,mode='HR',display_hid_interval=monit_highlight_hid,
                                                 superpose_ew=plot_maxi_ew,dict_rxte=dict_lc_rxte)
                # fig_maxi_lc_html = mpld3.fig_to_html(fig_maxi_lc)
                # components.html(fig_maxi_lc_html,height=500,width=1000)
                
                #wrapper to avoid streamlit trying to plot a None when resetting while loading
                if fig_hr_monit is not None:
                    st.pyplot(fig_hr_monit)
    
    if not plot_lc_monit and not plot_hr_monit:
        st.info('In single source mode, select a monitoring option in the sidebar to plot lightcurves and HR evolutions of the selected object')
    
    if ((plot_lc_monit and fig_lc_monit is None) or (plot_hr_monit and fig_hr_monit is None)) and display_single:
        st.warning('No match in MAXI/RXTE source list found.')
        
#####################
   #### Parameter analysis
#####################

with st.sidebar.expander('Parameter analysis'):
    
    display_param_withdet=st.checkbox('Restrict parameter analysis to sources with significant detections',value=True)
    
    display_param=st.multiselect('Additional parameters',
                                 ('EW ratio (Line)','width (Line)','Line flux (Line)','Time (Observation)',
                                  'Line EW comparison'),default=None)
    
    glob_col_source=st.checkbox('Normalize source colors over the entire sample',value=True)
    
    st.header('Distributions')
    display_distrib=st.checkbox('Plot distributions',value=False)
    use_distrib_lines=st.checkbox('Show line by line distribution',value=True)
    split_distrib=st.radio('Split distributions:',('Off','Source','Instrument'),index=1)
    
    if split_distrib=='Source' and (display_single or sum(mask_obj)==1):
            split_distrib='Off'
            
    st.header('Correlations')
    
    display_types=st.multiselect('Main parameters',('Line','Observation','Source'),default=None)
    
    display_scat_intr='Line' in display_types
    display_scat_hid='Observation' in display_types
    display_scat_inclin='Source' in display_types
    
    display_scat_eqwcomp='Line EW comparison' in display_param
    if display_scat_eqwcomp:        
        eqwratio_comp=st.multiselect('Lines to compare', [elem for elem in lines_std_names[3:9] if 'abs' in elem],default=lines_std_names[3:5])
        
    
    use_eqwratio='EW ratio (Line)' in display_param
    if use_eqwratio:
        eqwratio_strs=np.array(['Fe XXVI Ka/Fe XXV Ka','FeXXVI Kb/Fe XXV Kb','FeXXV Kb/Fe XXV Ka','FeXXVI Kb/Fe XXVI Ka'])
        eqwratio_type_str=st.selectbox('Ratio to use',eqwratio_strs)
        eqwratio_type=str(np.array(['Ka','Kb','25','26'])[eqwratio_strs==eqwratio_type_str][0])
        
    use_width='width (Line)' in display_param
    if use_width:
        display_th_width_ew=st.checkbox('Display theoretical individual width vs EW evolution',value=False)
    else:
        display_th_width_ew=False
        
    use_time='Time (Observation)' in display_param
    use_lineflux='Line flux (Line)' in display_param
    
    compute_correl=st.checkbox('Compute Pearson/Spearman for the scatter plots',value=True)
    display_pearson=st.checkbox('Display Pearson rank',value=False)
    st.header('Visualisation')
    radio_color_scatter=st.radio('Scatter plot color options:',('None','Source','Instrument','Time','HR','width','nH'),index=1)
    scale_log_eqw=st.checkbox('Use a log scale for the equivalent width and line fluxes')
    scale_log_hr=st.checkbox('Use a log scale for the HID parameters',value=True)
    display_abserr_bshift=st.checkbox('Display mean and std of Chandra velocity shift distribution',value=True)
    
    common_observ_bounds=st.checkbox('Use common observation parameter bounds for all lines',value=True)
    
    #plot_trend=st.checkbox('Display linear trend lines in the scatter plots',value=False)
    plot_trend=False
    
    st.header('Upper limits')
    show_scatter_ul=st.checkbox('Display upper limits in EW plots',value=False)
    lock_lims_det=not(st.checkbox('Include upper limits in graph bounds computations',value=True))
        

if compute_only_withdet:
    
    if sum(global_mask_intime_norepeat)==0 or sum(global_sign_mask)==0:
        if sum(global_mask_intime_norepeat)==0:
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

    hid_plot_restrict=hid_plot.T[mask_obj].T
    incl_plot_restrict=incl_plot[mask_obj]
    
    #and passing in the dictionnary for use in the functions
    dict_linevis['mask_obj']=mask_obj
    dict_linevis['abslines_plot_restrict']=abslines_plot_restrict
    dict_linevis['abslines_ener_restrict']=abslines_ener_restrict
    dict_linevis['width_plot_restrict']=width_plot_restrict
    dict_linevis['hid_plot_restrict']=hid_plot_restrict
    dict_linevis['incl_plot_restrict']=incl_plot_restrict
    
    
dict_linevis['display_pearson']=display_pearson
dict_linevis['display_abserr_bshift']=display_abserr_bshift
dict_linevis['glob_col_source']=glob_col_source
dict_linevis['display_th_width_ew']=display_th_width_ew
dict_linevis['common_observ_bounds']=common_observ_bounds

os.system('mkdir -p '+save_dir+'/graphs')
os.system('mkdir -p '+save_dir+'/graphs/distrib')
os.system('mkdir -p '+save_dir+'/graphs/intrinsic')
os.system('mkdir -p '+save_dir+'/graphs/hid')
os.system('mkdir -p '+save_dir+'/graphs/inclin')

###
# AUTOFIT LINES
###

###Distributions###

def streamlit_distrib():
    distrib_eqw=distrib_graph(abslines_plot_restrict,'eqw',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text,split=split_distrib)
    distrib_bshift=distrib_graph(abslines_plot_restrict,'bshift',dict_linevis,conf_thresh=slider_sign,streamlit=True,bigger_text=bigger_text,split=split_distrib)
    distrib_ener=distrib_graph(abslines_plot_restrict,'ener',dict_linevis,abslines_ener_restrict,conf_thresh=slider_sign,streamlit=True,
                               bigger_text=bigger_text,split=split_distrib)
    if use_eqwratio:
        distrib_eqwratio=distrib_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type,dict_linevis,conf_thresh=slider_sign,streamlit=True,
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
            
            col_list={'eqw':None,'bshift':None,'ener':None}
            
            if use_eqwratio:
                col_list['eqwratio']=None
            if n_infos>=5 and use_lineflux:
                col_list['lineflux']=None
    
            if use_distrib_lines:
                col_list['lines']=None
                
            if use_width:
                col_list['width']=None
                
            st_cols=st.columns(len(col_list))
            
            for i_col,col_name in enumerate(list(col_list.keys())):
                col_list[col_name]=st_cols[i_col]
                
            with col_list['eqw']:
                st.pyplot(distrib_eqw)
    
            with col_list['bshift']:
                st.pyplot(distrib_bshift)
            with col_list['ener']:
                st.pyplot(distrib_ener)
                    
            if use_eqwratio:
                with col_list['eqwratio']:
                    st.pyplot(distrib_eqwratio)
            
            if use_lineflux and n_infos>=5:
                with col_list['lineflux']:
                    st.pyplot(distrib_lineflux)
            
            if use_distrib_lines:
                with col_list['lines']:
                    st.pyplot(distrib_lines)
                    
            if use_width:
                with col_list['width']:
                    st.pyplot(distrib_width)
                
###1-1 Correlations###

###Intrinsic line parameters###

def streamlit_scat(mode):
    
    scat_eqw=[]
    scat_bshift=[]
    scat_ener=[]
    scat_width=[]
    
    if mode=='eqwratio':
        scat_eqw=[correl_graph(abslines_plot_restrict,eqwratio_comp[0]+'_'+eqwratio_comp[1],abslines_ener_restrict,dict_linevis,mode='eqwratio',
                               conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                               show_ul_eqw=show_scatter_ul)]
        if use_time and use_eqwratio:
            #not actually bshift but we keep the same column names
            scat_bshift=[correl_graph(abslines_plot_restrict,'time_eqwratio'+eqwratio_type,abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,
                                        mode='observ',conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            
    if mode=='intrinsic':
        scat_eqw=[correl_graph(abslines_plot_restrict,'bshift_eqw',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_bshift=[correl_graph(abslines_plot_restrict,'ener_eqw',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                               compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_ener=[correl_graph(abslines_plot_restrict,'ener_bshift',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,streamlit=True,
                                 compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if n_infos>=5 and use_lineflux:
            scat_lineflux=[correl_graph(abslines_plot_restrict,'lineflux_bshift',abslines_ener_restrict,dict_linevis,conf_thresh=slider_sign,
                                        streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
            
        if use_time:
            scat_eqw+=[correl_graph(abslines_plot_restrict,'time_eqw',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_bshift+=[correl_graph(abslines_plot_restrict,'time_bshift',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_ener+=[correl_graph(abslines_plot_restrict,'time_ener',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            if use_width:
                scat_width+=[correl_graph(abslines_plot_restrict,'time_width',abslines_ener_restrict,dict_linevis,mode_vals=None,mode='observ',
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            
        if use_width:
            scat_eqw+=[correl_graph(abslines_plot_restrict,'eqw_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_bshift+=[correl_graph(abslines_plot_restrict,'bshift_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            scat_ener+=[correl_graph(abslines_plot_restrict,'ener_width',abslines_ener_restrict,dict_linevis,mode_vals=None,
                                        conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
            if use_eqwratio:
                scat_width=[correl_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type+'_width1',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                            show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
                scat_eqwratio=[correl_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type+'_width2',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                            show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
    elif mode=='observ':
        scat_eqw=\
        [correl_graph(abslines_plot_restrict,'eqw_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',conf_thresh=slider_sign,
                      streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,show_ul_eqw=show_scatter_ul),
         correl_graph(abslines_plot_restrict,'eqw_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',conf_thresh=slider_sign,
                      streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,show_ul_eqw=show_scatter_ul)]
        
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
    
        if use_width:
            
            scat_width=\
            [correl_graph(abslines_plot_restrict,'width_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'width_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
        if use_eqwratio:
         scat_eqwratio=\
             [correl_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type+'_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                      show_ul_eqw=show_scatter_ul),
              correl_graph(abslines_plot_restrict,'eqwratio'+eqwratio_type+'_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                      conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                      show_ul_eqw=show_scatter_ul)]
        
        if n_infos>=5 and use_lineflux:
            scat_lineflux=\
            [correl_graph(abslines_plot_restrict,'lineflux_HR',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked),
             correl_graph(abslines_plot_restrict,'lineflux_flux',abslines_ener_restrict,dict_linevis,mode_vals=hid_plot_restrict,mode='observ',
                          conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
    elif mode=='source':
        scat_eqw+=[correl_graph(abslines_plot_restrict,'eqw_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked,
                                show_ul_eqw=show_scatter_ul)]
        
        scat_bshift+=[correl_graph(abslines_plot_restrict,'bshift_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                               conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        
        scat_ener+=[correl_graph(abslines_plot_restrict,'ener_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                 conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        scat_width+=[correl_graph(abslines_plot_restrict,'width_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,mode='source',
                                 conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,show_linked=show_linked)]
        if n_infos>=5:
            scat_lineflux=[correl_graph(abslines_plot_restrict,'lineflux_inclin',abslines_ener_restrict,dict_linevis,mode_vals=incl_plot_restrict,
                                        mode='source',conf_thresh=slider_sign,streamlit=True,compute_correl=compute_correl,bigger_text=bigger_text,
                                        show_linked=show_linked)]
    
    with tab_param:
        with st.expander('Correlation graphs for '+('line' if mode=='intrinsic' else mode)+' parameters'):
    
            col_list={'eqw':None,'bshift':None,'ener':None}
            
            if use_width:
                col_list['width']=None
            
            #defining columns for each data type
            if use_eqwratio and (mode=='observ' or use_width):
                col_list['eqwratio']=None
            if n_infos>=5 and use_lineflux:
                col_list['lineflux']=None
            st_cols=st.columns(len(col_list))
            
            for i_col,col_name in enumerate(list(col_list.keys())):
                col_list[col_name]=st_cols[i_col]
            
            with col_list['eqw']:
                [st.pyplot(elem) for elem in scat_eqw]
                            
            if mode!='eqwratio' or use_time:
    
                with col_list['bshift']:
                    [st.pyplot(elem) for elem in scat_bshift]
    
            if mode!='eqwratio':
                with col_list['ener']:
                    [st.pyplot(elem) for elem in scat_ener]
                if use_eqwratio and (mode=='observ' or use_width):
                    with col_list['eqwratio']:
                        [st.pyplot(elem) for elem in scat_eqwratio]
                        
            if use_lineflux and n_infos>=5:
                with col_list['lineflux']:
                    [st.pyplot(elem) for elem in scat_lineflux]
                    
            if use_width:
                with col_list['width']:
                    [st.pyplot(elem) for elem in scat_width]
            
mpl.rcParams.update({'font.size': 14})

#storing arguments to reduce the number of arguments in the scatter plot functions    
dict_linevis['scale_log_hr']=scale_log_hr
dict_linevis['scale_log_eqw']=scale_log_eqw

dict_linevis['abslines_ener']=abslines_ener
dict_linevis['abslines_plot']=abslines_plot

dict_linevis['lock_lims_det']=lock_lims_det
dict_linevis['plot_trend']=plot_trend

dict_linevis['color_scatter']=radio_color_scatter
dict_linevis['observ_list']=observ_list
dict_linevis['hid_plot_restrict']=hid_plot_restrict
dict_linevis['width_plot_restrict']=width_plot_restrict
dict_linevis['nh_plot_restrict']=nh_plot_restrict

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

if display_scat_eqwcomp:
    streamlit_scat('eqwratio')
    
if display_scat_hid:
    streamlit_scat('observ')

if display_scat_inclin:
    streamlit_scat('source')
    
if not (display_scat_intr or display_scat_eqwcomp or display_scat_hid or display_scat_inclin):
    with tab_param:
        st.info('Select parameters to compare or enable distributions to generate plots')

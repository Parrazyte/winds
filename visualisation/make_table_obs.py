#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:11:45 2022

@author: parrama
"""

#general imports
import os

import glob

import argparse

import numpy as np

'''Astro'''


#custom script with some lines and fit utilities and variables
from fitting_tools import range_absline

#visualisation functions
from visual_line_tools import n_infos, obj_values,abslines_values


ap = argparse.ArgumentParser(description='Script to display lines in XMM Spectra.\n)')

'''GENERAL OPTIONS'''


ap.add_argument("-cameras",nargs=1,help='Cameras to use for the spectral analysis',default='all',type=str)
ap.add_argument("-expmodes",nargs=1,help='restrict the analysis to a single type of exposure',default='all',type=str)
ap.add_argument("-grouping",nargs=1,help='specfile grouping to use in [5,10,20] cts/bin',default='20',type=str)
ap.add_argument("-prefix",nargs=1,help='restrict analysis to a specific prefix',default='auto',type=str)
ap.add_argument("-outdir",nargs=1,help="name of output directory for line plots",default="lineplots_opt",type=str)

'''DIRECTORY SPECIFICS'''

ap.add_argument("-local",nargs=1,help='launch analysis in the current directory instead',default=True,type=bool)


'''SPECTRUM PARAMETERS'''


ap.add_argument("-line_cont_range",nargs=1,help='min and max energies of the line continuum broand band fit',default='4 10',type=str)
ap.add_argument("-line_cont_ig",nargs=1,help='min and max energies of the ignore zone in the line continuum broand band fit',
                default='6.-8.',type=str)
ap.add_argument("-line_search_e",nargs=1,help='min, max and step of the line energy search',default='4 10 0.05',type=str)

ap.add_argument("-line_search_norm",nargs=1,help='min, max and nsteps (for one sign)  of the line norm search (which operates in log scale)',
                default='0.01 10 500',type=str)

'''VISUALISATION'''

args=ap.parse_args()

'''
Notes:
-Only works for the auto observations (due to prefix naming) for now

-For now we fix the masses of all the objets at 10M_sol

-Due to the way the number of steps is computed, we explore one less value for the positive side of the normalisation

-The norm_stepval argument is for a fixed flux band, and the value is scaled in the computation depending on the line energy step
'''

cameras=args.cameras
expmodes=args.expmodes
grouping=args.grouping
prefix=args.prefix
local=args.local
outdir=args.outdir

line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
line_cont_ig=args.line_cont_ig
line_search_e=np.array(args.line_search_e.split(' ')).astype(float)
line_search_norm=np.array(args.line_search_norm.split(' ')).astype(float)

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
    
'''initialisation'''

# #for the current directory:
# started_expos,done_expos=folder_state()
 
#bad spectra manually taken off
bad_flags=[]


#we create these variables in any case because the multi_obj plots require them
line_search_e_space=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2],line_search_e[2])
#this one is here to avoid adding one point if incorrect roundings create problem
line_search_e_space=line_search_e_space[line_search_e_space<=line_search_e[1]]


norm_par_space=np.concatenate((-np.logspace(np.log10(line_search_norm[1]),np.log10(line_search_norm[0]),int(line_search_norm[2]/2)),np.array([0]),
                                np.logspace(np.log10(line_search_norm[0]),np.log10(line_search_norm[1]),int(line_search_norm[2]/2))))
norm_nsteps=len(norm_par_space)

'''''''''''''''''''''''''''''''''''''''
''''''Hardness-Luminosity Diagrams''''''
'''''''''''''''''''''''''''''''''''''''

'Distance and Mass determination'

#wrapped in a function to be cached in streamlit

telescope_list=('XMM','Chandra')

#later:,'NICER','Suzaku','Swift')

#We put the telescope option before anything else to filter which file will be used
choice_telescope=['XMM','Chandra']

ignore_full=True

#### file search

#### current directory set to BHLMXB
os.chdir('/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB')

all_files=glob.glob('**',recursive=True)
lineval_id='line_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
lineval_files=[elem for elem in all_files if outdir+'/' in elem and lineval_id in elem and ('/Sample/' in elem or 'XTEJ1701-462/' in elem)]

abslines_id='autofit_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
abslines_files=[elem for elem in all_files if outdir+'/' in elem and abslines_id in elem and ('/Sample/' in elem or 'XTEJ1701-462/' in elem)]

#telescope selection
lineval_files=[elem for elem_telescope in choice_telescope for elem in lineval_files if elem_telescope+'/' in elem]
abslines_files=[elem for elem_telescope in choice_telescope for elem in abslines_files if elem_telescope+'/' in elem]

if ignore_full:
    lineval_files=[elem for elem in lineval_files if '_full' not in elem]
    abslines_files=[elem for elem in abslines_files if '_full' not in elem]
    

obj_list=np.unique(np.array([elem.split('/')[-4] for elem in lineval_files]))

    
#note: there's no need to order anymore since the file values are attributed for each object of object list in the visual_line functions

#creating the dictionnary for all of the arguments to pass to the visualisation functions
dict_linevis={
    'ctl_blackcat':None,
    'ctl_blackcat_obj':None,
    'ctl_watchdog':None,
    'ctl_watchdog_obj':None,
    'lineval_files':lineval_files,
    'obj_list':obj_list,
    'cameras':cameras,
    'multi_obj':True,
    'expmodes':expmodes,
    'range_absline':range_absline,
    'n_infos':n_infos,
    'args_cam':args.cameras,
    'args_line_search_e':args.line_search_e,
    'args_line_search_norm':args.line_search_norm,
    'visual_line':True
    }

#### main arrays computation

#useless L_Edd unit factor because we don't need it here
Edd_factor=np.repeat(1,len(obj_list))

#Reading the results files
observ_list,lineval_list,flux_list,date_list,instru_list,exptime_list=obj_values(lineval_files,Edd_factor,dict_linevis)

dict_linevis['flux_list']=flux_list

#the values here are for each observation
abslines_infos,autofit_infos=abslines_values(abslines_files,dict_linevis)

#### creating the lines to be written
line_list=[]
    
for i_obj,obj in enumerate(obj_list):
    
    #fetching the order of the exposures for the current obj
    date_order=date_list[i_obj].argsort()
    
    #writing the name of the object as a multi-row of the number of exposures
    line_list+=['\multirow{'+str(len(date_order))+'}{*}{'+obj+'}']
    
    #and parsing it as an index
    for ind_exp,i_exp in enumerate(date_list[i_obj].argsort()):
        
        #writing the date's day
        line_list+=['&'+date_list[i_obj][i_exp].split('T')[0]+'']
        
        #writing the instrument
        line_list+=['&'+instru_list[i_obj][i_exp]+'']
        
        #writing the obsid and identifier
        #here we take off a few elements unneeded from the observ_list string
        observ_string=observ_list[i_obj][i_exp].replace('_pn','').replace('_Imaging','').replace('_Timing','').replace('_auto','')\
                                             .replace('_heg','').replace('_-1','').replace('_','\_')
        line_list+=['&'+observ_string+'']
        
        #writing the exposure
        line_list+=['&'+str(round(exptime_list[i_obj][i_exp]/1e3,2))+'']
        
        #writing the line EW/upper limit values
        for i_line in range(6):
            
            #skipping NiKa27
            if i_line==2:
                continue
            
            #significance threshold dichotomy
            if abslines_infos[i_obj][i_exp][4][i_line]<0.997:
                
                #adding the upper limit when no significant line is detected, if the upper limit has been computed or is too high
                if abslines_infos[i_obj][i_exp][5][i_line]==0 or abslines_infos[i_obj][i_exp][5][i_line]>=100:
                    line_list+=['&/']
                else:
                    line_list+=['&$\leq'+str(round(abslines_infos[i_obj][i_exp][5][i_line]))+'$']
            
            else:
                #displaying the line EW with associated uncertainties
                line_EW_vals=[round(abslines_infos[i_obj][i_exp][0][i_line][i_incer]) for i_incer in range(3)]

                #displaying a \pm if its relevant 
                if line_EW_vals[1]==line_EW_vals[2]:
                    line_list+=['&$\\textbf{'+str(line_EW_vals[0])+'}\pm'+str(line_EW_vals[1])+'$']
                else:
                    line_list+=['&\\textbf{'+str(line_EW_vals[0])+'}$_{-'+str(line_EW_vals[1])+'}^{+'+str(line_EW_vals[2])+'}$']
        
        #adding the end of the line
        #we add one more \ because writing the string takes one off for some reason
        line_list+=['\T \B \\\ ']
        line_list+=['\n']
        
        #adding sources separations
        if ind_exp==len(date_list[i_obj])-1:
            line_list+=['\hline\n']
            
        
os.system('mkdir -p glob_batch')

#writing the list in a file
with open('glob_batch/obs_table_'+outdir+'.txt','w+') as file:
    file.writelines(line_list)
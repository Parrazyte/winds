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
from astropy.time import Time,TimeDelta

'''Astro'''


#custom script with some lines and fit utilities and variables
from fitting_tools import range_absline

#visualisation functions
from visual_line_tools import n_infos, obj_values,abslines_values,values_manip,load_catalogs
from dist_mass_tools import dist_mass

from general_tools import ravel_ragged

ap = argparse.ArgumentParser(description='Script to display lines in XMM Spectra.\n)')

'''GENERAL OPTIONS'''


ap.add_argument("-cameras",nargs=1,help='Cameras to use for the spectral analysis',default='all',type=str)
ap.add_argument("-expmodes",nargs=1,help='restrict the analysis to a single type of exposure',default='all',type=str)
ap.add_argument("-grouping",nargs=1,help='specfile grouping to use in [5,10,20] cts/bin',default='opt',type=str)
ap.add_argument("-prefix",nargs=1,help='restrict analysis to a specific prefix',default='auto',type=str)
ap.add_argument("-outdir",nargs=1,help="name of output directory for line plots",default="lineplots_opt",type=str)

#null value is False
ap.add_argument('-restrict_obj',nargs=1,help='restrict to single object',default='4U1630-47')

ap.add_argument('-no_multi',nargs=1,help="remove multi satellite analysis",default=True)
'''DIRECTORY SPECIFICS'''

ap.add_argument("-local",nargs=1,help='launch analysis in the current directory instead',default=True,type=bool)

ap.add_argument('-sign_threshold',nargs=1,
                help='data significance used to start the upper limit procedure and estimate the detectability',
                default=0.997,type=float)

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

sign_threshold=args.sign_threshold

restrict_obj=args.restrict_obj
no_multi=args.no_multi

line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
line_cont_ig=args.line_cont_ig
line_search_e=np.array(args.line_search_e.split(' ')).astype(float)
line_search_norm=np.array(args.line_search_norm.split(' ')).astype(float)

outburst_split_dic={
    '4U1630-47':
       [[Time('2002-09-01'),Time('2004-11-30')],
        [Time('2005-10-01'),Time('2006-05-31')],
        [Time('2007-12-01'),Time('2008-06-30')],
        [Time('2009-12-01'),Time('2010-06-30')],
        [Time('2011-12-01'),Time('2013-12-31')],
        [Time('2015-01-01'),Time('2015-04-30')],
        [Time('2016-08-01'),Time('2017-01-31')],
        [Time('2018-05-01'),Time('2019-09-30')],
        [Time('2020-03-01'),Time('2020-06-30')],
        [Time('2021-09-01'),Time('2022-03-31')],
        [Time('2022-07-01'),Time('2024-02-28')]]}

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

catal_blackcat,catal_watchdog,catal_blackcat_obj,catal_watchdog_obj,catal_maxi_df,catal_maxi_simbad,\
    catal_bat_df,catal_bat_simbad=load_catalogs()

telescope_list=('XMM','Chandra','NICER','Suzaku','Swift','NuSTAR')

choice_telescope=('XMM','Chandra','NICER','Suzaku','Swift','NuSTAR')

#### current directory set to BHLMXB
tbdir=input('directory to search ?')

os.chdir(tbdir)

all_files=glob.glob('**',recursive=True)
lineval_id='line_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
lineval_files=[elem for elem in all_files if outdir+'/' in elem and lineval_id in elem and '/Sample/' in elem]

abslines_id='autofit_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
abslines_files=[elem for elem in all_files if outdir+'/' in elem and abslines_id in elem and '/Sample/' in elem]

#telescope selection
lineval_files=[elem for elem_telescope in choice_telescope for elem in lineval_files if elem_telescope+'/' in elem]
abslines_files=[elem for elem_telescope in choice_telescope for elem in abslines_files if elem_telescope+'/' in elem]

if ignore_full:
    lineval_files=[elem for elem in lineval_files if '_full' not in elem]
    abslines_files=[elem for elem in abslines_files if '_full' not in elem]

# some additional removals for in progress dirs
lineval_files = [elem for elem in lineval_files if '4U_mix' not in elem]
abslines_files = [elem for elem in abslines_files if '4U_mix' not in elem]

lineval_files = [elem for elem in lineval_files if outdir + '_old' not in elem]
abslines_files = [elem for elem in abslines_files if outdir + '_old' not in elem]

if restrict_obj!=False:
    lineval_files=[elem for elem in lineval_files if restrict_obj+'/' in elem]
    abslines_files=[elem for elem in abslines_files if restrict_obj+'/' in elem]

if no_multi:
    lineval_files=[elem for elem in lineval_files if '/multi/' not in elem]
    abslines_files=[elem for elem in abslines_files if '/multi/' not in elem]

if restrict_obj:
    obj_list=[restrict_obj]
else:
    obj_list=np.unique(np.array([elem.split('/')[-4] for elem in lineval_files]))

    
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

#getting the single parameters
dist_obj_list,mass_obj_list=dist_mass(dict_linevis)

#distance factor for the flux conversion later on
dist_factor=4*np.pi*(dist_obj_list*1e3*3.086e18)**2

#L_Edd unit factor
Edd_factor=dist_factor/(1.26e38*mass_obj_list)

#Reading the results files
observ_list,lineval_list,lum_list,date_list,instru_list,exptime_list=obj_values(lineval_files,Edd_factor,dict_linevis)

dict_linevis['lum_list']=lum_list

#the values here are for each observation
abslines_infos,autofit_infos=abslines_values(abslines_files,dict_linevis)

# getting all the variations we need

# getting all the variations we need
abslines_infos_perline, abslines_infos_perobj, abslines_plot, abslines_ener, \
    lum_plot, hid_plot, incl_plot, width_plot, nh_plot, kt_plot = values_manip(abslines_infos, dict_linevis,
                                                                                autofit_infos,
                                                                                lum_list)
#### creating the lines to be written
line_list=[]
    
det_line_list=[]

#creating an outburst splitting for individual sources
if restrict_obj:

    time_sorted=[Time(elem) for elem in date_list[0]]
    time_sorted.sort()

    outburst_split=np.array([[i for i in range(len(time_sorted)) if\
                              time_sorted[i]>outburst_split_dic[restrict_obj][j][0] and \
                              time_sorted[i]<outburst_split_dic[restrict_obj][j][1] ]\
                             for j in range(len(outburst_split_dic[restrict_obj]))],dtype='object')



for i_obj,obj in enumerate(obj_list):
    
    #fetching the order of the exposures for the current obj
    date_order=date_list[i_obj].argsort()

    if not restrict_obj:
        #writing the name of the object as a multi-row of the number of exposures
        line_list+=['\multirow{'+str(len(date_order))+'}{*}{'+obj+'}']
    
    #and parsing it as an index
    for ind_exp,i_exp in enumerate(date_list[i_obj].argsort()):

        if restrict_obj:

            if ind_exp not in ravel_ragged(outburst_split):

                # highlighting exposures out of outburst:
                line_list += ['\hline Out of outburst']

                #note: there's no line detection out of outburst currently
                # det_line_list+=['\hline Out of outburst']

            else:
                #fetching the corresponding outburst
                id_outburst=[i_out for i_out in range(len(outburst_split)) if ind_exp in outburst_split[i_out]][0]
                curr_outburst=outburst_split[id_outburst]

                if ind_exp==curr_outburst[0]:

                    #too long
                    # date_start_out=outburst_split_dic[restrict_obj][id_outburst][0].to_string()
                    # month_start_out='-'.join(date_start_out.split('-')[:2])
                    # date_end_out = outburst_split_dic[restrict_obj][id_outburst][1].to_string()
                    # month_end_out = '-'.join(date_end_out.split('-')[:2])
                    # month_out_str=month_start_out+'-'+month_end_out

                    date_start_out=outburst_split_dic[restrict_obj][id_outburst][0].to_string()
                    year_start_out='-'.join(date_start_out.split('-')[:1])
                    date_end_out = outburst_split_dic[restrict_obj][id_outburst][1].to_string()
                    year_end_out = '-'.join(date_end_out.split('-')[:1])
                    year_out_str=year_start_out+('/'+year_end_out if year_start_out!=year_end_out else '')


                    # writing the years of the outburst as a multi-row of the number of exposures in it
                    line_list += ['\hline\multirow{' + str(len(curr_outburst)) + '}{*}{' + year_out_str + '}']

                    #computing the amount of exposures with significant detections in the outburst
                    n_curr_outburst_withlines=sum([(abslines_infos[i_obj][i_expos][4][:2] >= sign_threshold).any()\
                                                    for i_expos in date_list[i_obj].argsort()[curr_outburst]])

                    if n_curr_outburst_withlines>0:
                        #writing the name of the object as a multi-row of the number of exposures
                        det_line_list+=['\hline\multirow{'+str(n_curr_outburst_withlines)+'}{*}{'+year_out_str+'}']


        #writing the date's day
        line_list+=['&'+date_list[i_obj][i_exp].split('T')[0]+'']
        
        #writing the instrument
        line_list+=['&'+instru_list[i_obj][i_exp]+'']
        
        #writing the obsid and identifier
        #here we take off a few elements unneeded from the observ_list string
        observ_string=observ_list[i_obj][i_exp].replace('_pn','').replace('_Imaging','').replace('_Timing','').replace('_auto','')\
                                             .replace('_heg','').replace('_-1','').replace('_xis1','')\
                                             .replace('_','\_').replace('nu','').replace('A01','')

        if instru_list[i_obj][i_exp]=='NICER':
            observ_string=observ_string.split('-')[0]

        line_list+=['&'+observ_string+'']
        
        #writing the exposure
        line_list+=['&'+str(round(exptime_list[i_obj][i_exp]/1e3,2))+'']
        
        #flag for adding blueshift lines for when there is at least a Ka detection
        if (abslines_infos[i_obj][i_exp][4][:2]>=sign_threshold).any():
            det_obs=True

            if not restrict_obj:
                det_line_list+=[obj]
            det_line_list+=['&'+date_list[i_obj][i_exp].split('T')[0]+'']
            det_line_list+=['&'+observ_string+'']
            
            #and the HR
            det_line_list+=['&$'+str(round(hid_plot[0][0][i_obj][i_exp],3))+'_{-'+str(round(hid_plot[0][1][i_obj][i_exp],3))+'}^{+'+str(round(hid_plot[0][2][i_obj][i_exp],3))+'}$']
            
            eddratio_list=[round(hid_plot[1][i_incert][i_obj][i_exp]*1e2,1) for i_incert in range(3)]
            
            #adding the flux
            det_line_list+=['&$'+str(eddratio_list[0])+('' if eddratio_list[1]==0 else '_{-'+str(eddratio_list[1])+'}')+('' if eddratio_list[2]==0 else '^{+'+str(eddratio_list[2])+'}')+'$']
            

        else:
            det_obs=False
            
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
            
                #blueshift line if there is a detection for the Ka complex
                if i_line<2 and det_obs:
                    det_line_list+=['&/&/&/']
                   
                    
            else:
                #displaying the line EW with associated uncertainties
                line_EW_vals=[round(abslines_infos[i_obj][i_exp][0][i_line][i_incer]) for i_incer in range(3)]

                line_bshift_vals=[round(round(abslines_infos[i_obj][i_exp][1][i_line][i_incer],-2)) for i_incer in range(3)]
                
                line_width_vals=['/' if np.isnan(elem) else round(round(elem,-2)) for elem in [width_plot[i_incer][i_line][i_obj][i_exp] for i_incer in range(3)]]
                
                #displaying a \pm if its relevant 
                if line_EW_vals[1]==line_EW_vals[2]:
                    EW_str=['&$\\textbf{'+str(line_EW_vals[0])+'}\pm'+str(line_EW_vals[1])+'$']
                else:
                    EW_str=['&\\textbf{'+str(line_EW_vals[0])+'}$_{-'+str(line_EW_vals[1])+'}^{+'+str(line_EW_vals[2])+'}$']
                
                line_list+=EW_str
                        
                #adding line infos for the line det table if there is a detection for the Ka complex
                if i_line<2 and det_obs:
                    
                    #EW
                    det_line_list+=EW_str
                    
                    #bshift
                    det_line_list+=['&\\textbf{'+str(line_bshift_vals[0])+'}$_{-'+str(line_bshift_vals[1])+'}^{+'+str(line_bshift_vals[2])+'}$']
                           
                    #width
                    det_line_list+=['&/' if len(np.nonzero([line_width_vals])[0])==0 else '&\\textbf{'+str(line_width_vals[0])+'}$'+('' if line_width_vals[1]==0 else '_{-'+str(line_width_vals[1])+'}')+('' if line_width_vals[2]==0 else '^{+'+str(line_width_vals[2])+'}')+'$']
        
        #adding the end of the line
        #we add one more \ because writing the string takes one off for some reason
        line_list+=['\T \B \\\ ']
        line_list+=['\n']
        
        if det_obs:
            #adding the end of the line
            #we add one more \ because writing the string takes one off for some reason
            det_line_list+=['\T \B \\\ ']
            det_line_list+=['\n']
            
        #adding sources separations
        if not restrict_obj and ind_exp==len(date_list[i_obj])-1:
            line_list+=['\hline\n']
            
        
os.system('mkdir -p glob_batch/line_tables')

#writing the list in a file
with open('glob_batch/line_tables/obs_table_'+outdir+'_'+('' if not restrict_obj else restrict_obj)+'.txt','w+') as file:
    file.writelines(line_list)
    
with open('glob_batch/line_tables/det_table_'+outdir+'_'+('' if not restrict_obj else restrict_obj)+'.txt','w+') as file:
    file.writelines(det_line_list)
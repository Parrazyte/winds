#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#general imports
import os,sys
import subprocess
import pexpect
import argparse
from tee import StdoutTee,StderrTee
import logging
import glob
import re
import threading
import numpy as np
import time
from ast import literal_eval

from matplotlib.widgets import Slider,Button

from astropy.stats import sigma_clip
import matplotlib as mpl

#for no_op_context
import contextlib

#using agg because qtagg still generates backends with plt.ioff()
mpl.use('agg')

#mpl.use('qt5agg')

import matplotlib as mpl

import matplotlib.pyplot as plt
plt.ioff()

from matplotlib import pyplot as plt


from astropy.time import Time,TimeDelta

from general_tools import file_edit,ravel_ragged,interval_extract,MinorSymLogLocator,str_orbit

#astro imports
from astropy.io import fits


'''peak detection'''
from findpeaks import findpeaks

from joblib import Parallel, delayed


"""
Created on Thu Sep  1 23:18:16 2022

Data reduction Script for NICER Observations

Searches for all NICER Obs type directories in the subdirectories and launches the process for each

list of possible actions : 

1. process_obsdir: run the nicerl2 script to process an obsid folder

gti. create_gtis: create custom gtis files to be used later for lightcurve and spectrum creation

fs. extract_all_spectral: runs the nicerl3-spect script to compute spectral products of an obsid folder (aka s,b,r at the same time)

l. extract_lightcurve: runs a set of nicerl3-lc scripts to compute a range of lightcurve and HR evolutions

g. group_spectra: group spectra using the optimized Kastra et al. binning

m.merge: merge all spectral products in the subdirectories to a bigbatch directory

ml. merge_lightcurve: merge all lightcurve products in the subdirectories to a lcbatch directory 

c. clean_products: clean event products in the observation's event_cl directory

fc. clean_all: clean all files including standard products and products of this script from the directory

DEPRECATED 
2. select_detector: removes specific detectors from the event file (not tested)

s. extract_spectrum: extract a pha spectrum from a process obsid folder using  Xselect

b. extract_background: extract a bg spectrum from a process obsid folder using a specific method

r. extract_response: extract a response from a processed obsid folder

 

"""

'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce NICER files.\n)')

#the basics

ap.add_argument('-load_functions',nargs=1,help="Load functions but don't launch anything",default=False,type=bool)

ap.add_argument("-dir", "--startdir", nargs='?', help="starting directory. Current by default", default='./', type=str)
ap.add_argument("-l","--local",nargs=1,help='Launch actions directly in the current directory instead',
                default=False,type=bool)
ap.add_argument('-catch','--catch_errors',help='Catch errors while running the data reduction and continue',
                default=False,type=bool)

#1 for no parallelization
ap.add_argument('-parallel',help='number of processors for parallel directories',
                default=1,type=bool)

#global choices
ap.add_argument("-a","--action",nargs=1,help='List which action(s) to run,separated by comas',
                default ='l', type = str)

#default: fc,1,gti,fs,l,g,m,ml,c

#note: should be kept to true for most complicated tasks
ap.add_argument("-over",nargs=1,help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder',default=True,type=bool)

#directory level overwrite (not active in local)
ap.add_argument('-folder_over',nargs=1,help='relaunch action through folders with completed analysis',
                default=False, type=bool)
ap.add_argument('-folder_cont',nargs=1,help='skips all but the last 2 directories in the summary folder file',
                default=False,type=bool)

ap.add_argument('-invert_subdirs',nargs=1,help='start the analysis from the subdirectories in reverse order',
                default=True,type=bool)
#note : we keep the previous 2 directories because bug or breaks can start actions on a directory following the initially stopped one

#action specific overwrite

'''1. process'''

#These arguments should be adjusted with lots of attention after looking at both the flare plots
#the summary of temporal filtering logged in process_obsdir, and the resulting spectra

#how to handle day periods after the optical leak
#use day, night, or both depending on the output desired
ap.add_argument('-day_mode',nargs=1,help='Handle day data only, night data only, or both',type=str,default='both')

#Better to be coupled with overdyn filtering
ap.add_argument('-keep_SAA',nargs=1,help='keep South Atlantic Anomaly (SAA) Periods',type=bool,default=True)

#-1 means deactivated for both over and undershoot limits

#note: should be set to -1 if overdyn is activated in the gti filtering options
ap.add_argument('-overshoot_limit',nargs=1,help='overshoot event rate limit',type=float,default=-1)

ap.add_argument('-undershoot_limit',nargs=1,help='undershoot event rate limit',type=float,default=800)

ap.add_argument('-keep_lowmem',nargs=1,help='disable the memory discarding filtering for high count rates',type=bool,
                default=False)

#default to keep the base value of NICERDAS (30 as of the writing of this)
ap.add_argument('-br_earth_min',nargs=1,help='bright earth minimum angle',type=str,default='default')

ap.add_argument('-min_gti',nargs=1,help='minimum gti size',type=float,default=1.0)

#this one is for the standard nimaketime
ap.add_argument('-erodedilate',nargs=1,help='Erodes increasingly more gtis around the excluded intervals',
                type=float,default=1.0)

'''gti creation'''
#keyword for split: split_timeinsec
#note: split is broken in the new version, needs fixing
ap.add_argument('-gti_split',nargs=1,help='GTI split method',default='orbit+flare+overdyn+underdyn+HR_flare',type=str)
ap.add_argument('-flare_method',nargs=1,help='Flare extraction method(s)',default='clip+peak',type=str)

#previous version was with a SAS, tool, which required installing SAS. Now the default is NICER directly
ap.add_argument('-gti_tool',nargs=1,help='GTI tool used to make the gti file itself',default='NICERDAS',type=str)

ap.add_argument('-add_merge_gti',nargs=1,help='Add an daily integrated DAY/NIGHT GTI for less datagroups',default=True,
                type=bool)

'''Flare methods '''
#for "normal" flare clip
ap.add_argument('-clip_sigma',nargs=1,help='clipping minimum variance treshold in sigmas',default=3.,type=float)

ap.add_argument('-flare_factor',nargs=1,help='minimum flare multiplication factor for flare clipping',
                default=2.,type=float)

#for peak. 10 for bright sources, 2 for small sources
ap.add_argument('-peak_score_thresh',nargs=1,help='topological peak score treshold for peak exclusion',
                default=2.,type=float)

#for overdyn, in s since based on the mkf
ap.add_argument('-erodedilate_overdyn',nargs=1,help='Erodes increasingly more gtis around the overshoot excluded intervals',
                type=int,default=5)

ap.add_argument('-hard_flare_segments',nargs=1,
                help='Number of segments to split the hard flare lightcurve to determine the sigma clipping',
                type=int,default=5)

ap.add_argument('-hard_flare_sigma',nargs=1,help='Number of sigmas of the sigma hard_flare sigma clipping',
                type=int,default=5)

ap.add_argument('-hard_flare_min_duration',nargs=1,help='minimum duration of a single hard flare period',type=int,
                default=30)

ap.add_argument('-erodedilate_hard_flare',nargs=1,
                help='Erodes increasingly more gtis around the hard flare excluded intervals',
                type=int,default=5)

ap.add_argument('-gti_HR_threshold',nargs=1,
                help='lower limit for the [8-12]/[2-8] HR ratio to remove flares',
                type=float,default=0.1)

ap.add_argument('-erodedilate_HR',nargs=1,
                help='Erodes increasingly more gtis around the HR flare excluded intervals',
                type=int,default=5)

ap.add_argument('-overdyn_thresh',nargs=1,help='threshold between the two modes of overdyn filtering',
                type=float,default=1.)

ap.add_argument('-overdyn_factor_low',nargs=1,help='comparison factor for the low count rate part of overdyn filtering',
                type=float,default=5.)

ap.add_argument('-overdyn_factor_high',nargs=1,help='comparison factor for the high count rate part of overdyn filtering',
                type=float,default=1.)

ap.add_argument('-underdyn_method',nargs=1,help='method for underdyn filtering',type=str,
                default='compa')

#for underdyn gradient
#also used for the leeway between both jumps
ap.add_argument('-underdyn_jump_width',nargs=1,
                help='Maximum duration and leeway of an undershoot+main counts jump',type=int,
                default=10)

ap.add_argument('-underdyn_jump_factor',nargs=1,
                help='Minimum changing factor of an undershoot+main counts jump',type=int,
                default=5)

#note: not used currently
ap.add_argument('-gti_lc_band',nargs=1,help='Band for the lightcurve used for GTI splitting',
                default='12-15',type=str)

#note: can be better to increase the soft treshold for high background moments
ap.add_argument('-int_split_band',nargs=1,help='band of the lightcurve used for GTI intensity splitting',
                type=str,default='0.3-10.')

#this should be well above the variability that's being probed to get a good sampling
ap.add_argument('-int_split_bin',nargs=1,help='binning of the light curve used for GTI intensity splitting in s',
                default=0.1)

'''lightcurves'''

ap.add_argument('-lc_bin_list',nargs=1,
                help='A list of binnings with which to generate all lightcurces/HR evolutions (in s)',
                default=[1.,60.],type=list)
#note: also defines the binning used for the gti definition

# lc_bands_list_det=['1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10']

lc_bands_list_det=['1-3','0.3-10.']
#default='3-10'+','+','.join(lc_bands_list_det)
ap.add_argument('-lc_bands_str',nargs=1,help='Gives the list of bands to create lightcurves from',
                default=','.join(lc_bands_list_det),type=str)

ap.add_argument('-hr_bands_str',nargs=1,help='Gives the list of bands to create hrs from',default='6-10/3-6',type=str)

ap.add_argument('-skip_merge_lc',nargs=1,help='Skip merge GTIs for lightcurve computations',default=True,type=bool)
'''spectra'''

#note that the default of nicerl3-spect is True
ap.add_argument('-sp_systematics',help='put systematics in the spectrum',default=True,type=str)

#(note that this doesn't work right not because the keyword isn't implemented yet)
ap.add_argument('-relax_SAA_bg',help='Increase the maximum of the nxb.saa_norm model to a higher value',
                default=False,type=str)

#if set to scorpeon_all, will clean the bg files in the directory before running since it uses the task with clobber=NO
#for scorpeon, scorpeon_script for the model and scorpeon_default for the static
ap.add_argument('-bg',"--bgmodel",help='Give the background model to use for the data reduction',
                default='scorpeon_all',type=str)

#python or default
ap.add_argument('-bg_lang',"--bg_language",
        help='Gives the language output for the script generated to load spectral data into either PyXspec or Xspec',
                default='python',type=str)

ap.add_argument('-gtype',"--grouptype",help='Give the group type to use in regroup_spectral',default='opt',type=str)

#deprecated
ap.add_argument('-baddet','--bad_detectors',help='List detectors to exclude from the data reduction',
                default='-14,-34,-54',type=str)

#set to '' to deactivate
ap.add_argument('-heasoft_init_alias',help="name of the heasoft initialisation script alias",
                default="heainit",type=str)

#set to '' to deactivate
ap.add_argument('-caldb_init_alias',help="name of the caldbinit initialisation script alias",
                default="caldbinit",type=str)

#set to '' to deactivate. Only necessary (and used) with gti_tool=SAS for gti creation
ap.add_argument('-sas_init_alias',help="name of the caldbinit initialisation script alias",
                default="sasinit",type=str)

#only necessary if using 3C50
ap.add_argument('-alias_3C50',help="bash alias for the 3C50 directory",default='$NICERBACK3C50',type=str)

'''
clean
'''

#useful for HDDs
ap.add_argument('-clean_wait_value',help='waiting time after cleaning a folder',default=10,type=float)

args=ap.parse_args()

load_functions=args.load_functions

startdir=args.startdir
action_list=args.action.split(',')
local=args.local
folder_over=args.folder_over
folder_cont=args.folder_cont

invert_subdirs=args.invert_subdirs
overwrite_glob=args.over
catch_errors=args.catch_errors
bgmodel=args.bgmodel
bglanguage=args.bg_language

keep_SAA=args.keep_SAA

relax_SAA_bg=args.relax_SAA_bg

overshoot_limit=args.overshoot_limit
undershoot_limit=args.undershoot_limit
min_gti=args.min_gti
erodedilate=args.erodedilate
keep_lowmem=args.keep_lowmem
br_earth_min=args.br_earth_min
gti_split=args.gti_split
gti_lc_band=args.gti_lc_band
flare_method=args.flare_method
add_merge_gti=args.add_merge_gti
gti_HR_threshold=args.gti_HR_threshold
erodedilate_HR=args.erodedilate_HR

gti_tool=args.gti_tool

clip_sigma=args.clip_sigma
flare_factor=args.flare_factor
peak_score_thresh=args.peak_score_thresh
int_split_band=args.int_split_band
int_split_bin=args.int_split_bin

parallel=args.parallel

lc_bin_list=args.lc_bin_list
lc_bands_str=args.lc_bands_str
hr_bands_str=args.hr_bands_str
skip_merge_lc=args.skip_merge_lc

overdyn_thresh=args.overdyn_thresh
overdyn_factor_low=args.overdyn_factor_low
overdyn_factor_high=args.overdyn_factor_high
underdyn_method=args.underdyn_method

sp_systematics=args.sp_systematics

erodedilate_overdyn=args.erodedilate_overdyn

hard_flare_segments=args.hard_flare_segments
hard_flare_sigma=args.hard_flare_sigma
hard_flare_min_duration=args.hard_flare_min_duration
erodedilate_hard_flare=args.erodedilate_hard_flare

underdyn_jump_width=args.underdyn_jump_width
underdyn_jump_factor=args.underdyn_jump_factor

grouptype=args.grouptype
bad_detectors=args.bad_detectors
heasoft_init_alias=args.heasoft_init_alias
caldb_init_alias=args.caldb_init_alias
sas_init_alias=args.sas_init_alias
alias_3C50=args.alias_3C50
clean_wait_value=args.clean_wait_value

day_mode=args.day_mode

'''''''''''''''''
''''FUNCTIONS''''
'''''''''''''''''

def set_var(spawn):
    
    '''
    Sets starting environment variables for data analysis
    '''
    if heasoft_init_alias!='':
        spawn.sendline(heasoft_init_alias)

    if caldb_init_alias!='':
        spawn.sendline(caldb_init_alias)

#to keep the same loops with or without the Tees in several functions
@contextlib.contextmanager
def no_op_context():
    yield

#function to remove (most) control chars
def _remove_control_chars(message):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)

def process_obsdir(directory,overwrite=True,keep_SAA=False,overshoot_limit=30.,undershoot_limit=500.,
                                            min_gti=5.0,erodedilate=5.0,keep_lowmem=False,
                                            br_earth_min='default',day_mode='both',thread=None,parallel=False):
    
    '''
    Processes a directory using the nicerl2 script

    options (default values are the defaults of the function)

    see https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/nimaketime.html for details of these

    -keep_SAA: remove South Atlantic Anomaly filtering (default False).
                Should only be done in specific cases, see

    -overshoot/undershoot limit: limit above which to filter the events depending on these quantities
        if set to -1, deactivates the criterium by filtering below 1e6 (which will never be reached)

    -min_gti minimum gti

    -erodedilate: erodes gtis arode the excluded intervals

    -keep_lowmem: True to disable the max_lowmem filtering
    This column indicates when the MPU is discarding events due extremely high count rates,
     in which case the calibration problems will likely arise.
     The default value of 0 disables this criterium, in favor of a similar screening done in niautoscreen.

    -br_earth_min:  Exclude times when distance to the bright earth is less than MIN_BR_EARTH.
                    default does not modify the nicerl2 default parameters

    -day_mode:      Apply screening according to the night settings of nicerl2

    -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections
    '''

    io_log=open(directory+'/process_obsdir.log','w+')

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8',logfile=io_log if parallel else None)
    
    print('\n\n\nEvent filtering...')
    
    set_var(bashproc)

    if overshoot_limit==-1:
        overshoot_limit_use=1e6
    else:
        overshoot_limit_use=overshoot_limit

    if undershoot_limit==-1:
        undershoot_limit_use=1e6
    else:
        undershoot_limit_use=undershoot_limit

    with (no_op_context() if parallel else StdoutTee(directory+'/process_obsdir.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/process_obsdir.log',buff=1,file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read=sys.stdout

        #initializing to pass errors if they aren't created
        process_state_night=1
        process_state_day=1

        if day_mode in ['night','both']:

            #first night mode filtering
            bashproc.sendline('nicerl2 indir=' + directory +' clobber=' + ('YES' if overwrite else 'FALSE') +
                              ' nicersaafilt=' + ('NO' if keep_SAA else 'YES') +
                              ' overonly_range=0-%.1f'%overshoot_limit_use +
                              ' underonly_range=0-%.1f'%undershoot_limit_use +
                              ' mingti=%.1f' % min_gti +
                              ' erodedilate=%.1f' % erodedilate+
                              (' max_lowmem=0' if keep_lowmem else '')+
                              (' br_earth='+str(br_earth_min) if br_earth_min!='default' else ''))

            process_state_night=bashproc.expect(['terminating with status','Event files written'],timeout=None)

        if day_mode in ['day','both']:
            #day mode filtering (only in screen mode if a full nicerl2 has already been performed before
            bashproc.sendline('nicerl2 indir=' + directory +' clobber=' + ('YES') +
                              ' nicersaafilt=' + ('NO' if keep_SAA else 'YES') +
                              ' overonly_range=0-%.1f'%overshoot_limit_use +
                              ' underonly_range=0-%.1f'%undershoot_limit_use +
                              ' mingti=%.1f' % min_gti +
                              ' erodedilate=%.1f' % erodedilate+
                              (' max_lowmem=0' if keep_lowmem else '')+
                              (' br_earth='+str(br_earth_min) if br_earth_min!='default' else '')+

                              #actually using task=screen tends to make nicerl2 crash so we do the whole thing instead
                              # (' tasks=SCREEN' if day_mode=='both' else '')+
                              "  threshfilter=DAY clfile='$CLDIR/ni$OBSID_0mpu7_cl_day.evt'")

            process_state_day=bashproc.expect(['terminating with status','Event files written'],timeout=None)
        
        #exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()
        
        #raising an error to stop the process if the command has crashed for some reason
        if (np.array([process_state_day,process_state_night])==0).any():
            raise ValueError

####THIS IS DEPRECATED            
def select_detector(directory,detectors='-14,-34,-54',thread=None,parallel=False):
    
    '''
    Removes specific detectors from the event file before continuing the analysis
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/fpmsel-using/
    '''


    io_log = open(directory + '/process_obsdir.log', 'w+')

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8',logfile=io_log if parallel else None)
    
    print('\n\n\n Detector selection...')
    
    #finding the event file
    evt_path=[elem for elem in glob.glob('**',recursive=True) if directory in elem and 'cl.evt' in elem]
    
    #stopping the process if there is no processed event file
    if len(evt_path)==0:
        raise ValueError
        
    evt_name=evt_path[0].split('/')[-1]
    
    set_var(bashproc)

    with (no_op_context() if parallel else StdoutTee(directory+'/select_detector.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/select_detector.log',buff=1,file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read=sys.stdout
        
        bashproc.sendline('nifpmsel '+evt_name+' '+evt_name.replace('.evt','_sel.evt')+'detlist=launch,'+detectors)
        
        select_state=bashproc.expect(['DONE','terminating with status'],timeout=None)
                
        #raising an error to stop the process if the command has crashed for some reason
        if select_state!=0:
            bashproc.sendline('exit')
            if thread is not None:
                thread.set()
            raise ValueError
        
        #replacing the previous event file by the selected event file
        bashproc.sendline('mv '+evt_name.replace('.evt','_sel.evt')+' '+evt_name)
        
        #exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()

def plot_event_diag(mode,obs_start_str,time_obs,id_gti_orbit,
                    counts_035_8,counts_035_2,counts_2_8,counts_8_12,counts_overshoot,counts_undershoot,cutoff_rigidity,
                    counts_035_8_glob=None,
                    save_path=None,
                    id_gti=None,id_flares=None,id_overcut=None,
                    id_undercut=None,
                    id_hard_flares=None,
                    id_HR_flares=None,
                    gti_nimkt_arr=None,split_arg=None,split_gti_arr=None,
                    orbit_cut_times=None):
    
    '''
    Global event diagnostic plot function
    
    Displays a lot of elements to infer how usable the information is, notably the main quantities of the mkf file
    
    modes:
        -global to display the whole obsid with basic info
        -orbit to display detailed info and excluded/included intervals on a single orbit
        -manual to select a gti interval in manual gti creation mode
    '''
    
    fig_events, ax_events = plt.subplots(1, figsize=(12, 8))

    ax_events.set_xlabel('Time (s) after ' + obs_start_str)
    ax_events.set_ylabel('Count Rate (counts/s)')
    ax_rigidity = ax_events.twinx()
    ax_rigidity.set_ylabel('Cutoff Rigidity (Gev/c)')

    # we just want something above 0 here while keeping a log scale but allowing 0 counts
    ax_events.set_yscale('symlog', linthresh=0.1, linscale=0.1)
    ax_events.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.1))

    if mode=='global':

        n_orbit=len(id_gti_orbit)
        
        for i_orbit in range(n_orbit):
                
            ax_events.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_035_8[id_gti_orbit[i_orbit]],
                               color='red', label='0.35-8 keV Count Rate' if i_orbit==0 else '')

            ax_events.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_035_2[id_gti_orbit[i_orbit]],
                               color='pink', label='0.35-2 keV Count Rate' if i_orbit==0 else '')

            ax_events.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_2_8[id_gti_orbit[i_orbit]],
                               color='magenta', label='2-8 keV Count Rate' if i_orbit==0 else '')

            ax_events.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_8_12[id_gti_orbit[i_orbit]],
                               color='blue', label='8-12 keV Count Rate' if i_orbit==0 else '')
        
            ax_events.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_overshoot[id_gti_orbit[i_orbit]],
                               color='orange', label='Overshoot Rate (>20keV)' if i_orbit==0 else '')
        
            ax_events.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_undershoot[id_gti_orbit[i_orbit]],
                               color='brown', label='Undershoot Rate' if i_orbit==0 else '')
        
            ax_rigidity.plot(time_obs[id_gti_orbit[i_orbit]], cutoff_rigidity[id_gti_orbit[i_orbit]],
                             color='green', label='Cutoff Rigidity' if i_orbit==0 else '')

        ax_rigidity.axhline(1.5, 0, 1, color='green', ls='--', label='Upper limit for risky regions')
        ax_events.axhline(30, 0, 1, color='orange', ls='--', label='Default nicerl2 flare cut')

        ax_events.legend(loc='upper left')
        ax_rigidity.legend(loc='upper right')
        ax_events.set_ylim(0, ax_events.get_ylim()[1])

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()

        return

    ax_events.errorbar(time_obs[id_gti_orbit], counts_035_8,
                       color='red', label='0.35-8 keV Count Rate')

    ax_events.errorbar(time_obs[id_gti_orbit], counts_035_2,
                       color='pink', label='0.35-2 keV Count Rate')

    ax_events.errorbar(time_obs[id_gti_orbit], counts_2_8,
                       color='magenta', label='2-8 keV Count Rate')

    ax_events.errorbar(time_obs[id_gti_orbit], counts_8_12,
                       color='blue', label='8-12 keV Count Rate')

    ax_events.errorbar(time_obs[id_gti_orbit], counts_overshoot,
                       color='orange', label='Overshoot Rate (>20keV)')

    ax_events.errorbar(time_obs[id_gti_orbit], counts_undershoot,
                       color='brown', label='Undershoot Rate')

    ax_rigidity.plot(time_obs[id_gti_orbit], cutoff_rigidity,
                     color='green', label='Cutoff Rigidity')

    # flare and gti intervals
    for id_inter, list_inter in enumerate(list(interval_extract(id_gti))):
        ax_rigidity.axvspan(time_obs[min(list_inter)]-1/2, time_obs[max(list_inter)]+1/2, color='grey', alpha=0.2,
                            label='standard gtis' if id_inter == 0 else '')

    for id_inter, list_inter in enumerate(list(interval_extract(id_flares))):
        ax_rigidity.axvspan(time_obs[min(list_inter)]-1/2, time_obs[max(list_inter)]+1/2, color='blue', alpha=0.2,
                            label='flare gtis' if id_inter == 0 else '')

    if id_overcut is not None:
        for id_inter, list_inter in enumerate(list(interval_extract(id_overcut))):
            ax_rigidity.axvspan(time_obs[min(list_inter)]-1/2, time_obs[max(list_inter)]+1/2, color='orange', alpha=0.2,
                                label='overshoot filtered gtis' if id_inter == 0 else '')

    if id_undercut is not None:
        for id_inter, list_inter in enumerate(list(interval_extract(id_undercut))):
            ax_rigidity.axvspan(time_obs[min(list_inter)]-1/2, time_obs[max(list_inter)]+1/2, color='saddlebrown',
                                alpha=0.3,
                                label='undershoot filtered gtis' if id_inter == 0 else '')

    if id_hard_flares is not None:

        for id_inter, list_inter in enumerate(list(interval_extract(id_hard_flares))):
            ax_rigidity.axvspan(time_obs[min(list_inter)] - 1 / 2, time_obs[max(list_inter)] + 1 / 2, color='cyan',
                                alpha=0.2,
                                label='hard flare gtis' if id_inter == 0 else '')

    if id_HR_flares is not None:

        for id_inter, list_inter in enumerate(list(interval_extract(id_HR_flares))):
            ax_rigidity.axvspan(time_obs[min(list_inter)] - 1 / 2, time_obs[max(list_inter)] + 1 / 2, color='purple',
                                alpha=0.2,
                                label='HR flare gtis' if id_inter == 0 else '')

    # computing the non-gti intervals from nimaketime
    id_nongti_nimkt = []

    for elem_gti in id_gti_orbit:
        # testing if the gti is in one of the gtis of the nimaketime
        if not ((time_obs[elem_gti] >= gti_nimkt_arr.T[0]) & (time_obs[elem_gti] <= gti_nimkt_arr.T[1])).any():
            # and storing if that's not the case
            id_nongti_nimkt += [elem_gti]

    # and plotting
    for id_inter, list_inter in enumerate(list(interval_extract(id_nongti_nimkt))):
        ax_rigidity.axvspan(time_obs[min(list_inter)]-1/2, time_obs[max(list_inter)]+1/2, color='red', alpha=0.1,
                            label='nicerl2 nimaketime exclusion' if id_inter == 0 else '')

    # plotting the split gti intervals if in the right mode
    if split_gti_arr is not None:
        if 'intensity' in split_arg:

            #showing splits horizontally
            for id_inter, list_inter in enumerate(split_gti_arr):

                ax_events.axhspan(min(counts_035_8_glob[list_inter]),max(counts_035_8_glob[list_inter]),
                                    color='green', alpha=0.1,
                                    label='intensity split intervals (flare cuts excluded)' \
                                        if id_inter == 0 else '')
                if id_inter!=0:
                    ax_events.axhline(min(counts_035_8_glob[list_inter]),color='green',ls=':')
        else:
            #showing splits vertically
            for id_inter, list_inter in enumerate(split_gti_arr):
                ax_rigidity.axvspan(time_obs[min(list_inter)]-1/2, time_obs[max(list_inter)]+1/2,
                                    color='green', alpha=0.1,
                                    label='split intervals ' +
                                          ('(flare cuts excluded)' if 'flare' in split_arg else '') \
                                        if id_inter == 0 else '')
                if id_inter!=0:
                    ax_rigidity.axvline(time_obs[min(list_inter)],color='green',ls=':')

    ax_rigidity.axhline(1.5, 0, 1, color='green', ls='--', label='UL of risky rigidity region')
    ax_events.axhline(30, 0, 1, color='orange', ls='--', label='Default overshoot cut')
    ax_events.axhline(500, 0, 1, color='brown', ls='--', label='Default undershoot cut')

    ax_events.legend(loc='upper left')
    ax_rigidity.legend(loc='upper right')
    ax_events.set_ylim(0, ax_events.get_ylim()[1])

    if mode=='manual':

        if len(orbit_cut_times)!=0:
            for elem_time in orbit_cut_times:
                ax_events.axvline(elem_time,color='green')

        ax_slider=fig_events.add_axes([0.2, 0.02, 0.65, 0.03])

        slid = Slider(ax_slider, label='current gti',
                      valmin=time_obs[id_gti_orbit][0] if len(orbit_cut_times)==0 else orbit_cut_times[-1],
                      valmax=time_obs[id_gti_orbit][-1],valstep=1)

        def slider_update(val):

            for elem_child in ax_events.get_children():
                if elem_child._label == 'current gti':
                    elem_child.remove()
            ax_events.axvspan(time_obs[id_gti_orbit][0] if len(orbit_cut_times)==0 else orbit_cut_times[-1],
                              slid.val, 0, 1, alpha=0.3, color='green',label='current gti')

        slid.on_changed(slider_update)

        ax_button=fig_events.add_axes([0.9, 0.025, 0.08, 0.04])

        but = Button(ax=ax_button, label='Save GTI')

        def func_button(val):
            plt.close()
            print(slid.val)

        plt.show()
        but.on_clicked(func_button)

        plt.show(block=True)

        return slid.val

    else:

        plt.tight_layout()

        # note that str_orbit adds 1 to the counter
        plt.savefig(save_path)

        plt.close()
    
def create_gtis(directory,split_arg='orbit+flare+overdyn+underdyn+HR_flare',band='3-15',flare_method='clip+peak',
                clip_method='median',
                clip_sigma=2.,clip_band='8-12',peak_score_thresh=2.,
                int_split_band='0.3-10.',int_split_bin=0.1,clip_int_delta=True,
                flare_factor=2,gti_tool='NICERDAS',erodedilate_overdyn=5,

                #for overdyn filtering
                overdyn_thresh=1,overdyn_factor_low=2,overdyn_factor_high=1,

                #for flare
                hard_flare_segments=5,hard_flare_sigma=4,erodedilate_hard_flare=5,
                hard_flare_min_duration=30,

                #for underdyn
                underdyn_method='compa',

                #for underdyn_method=gradient
                underdyn_jump_width=10,underdyn_jump_factor=5,

                #for HR
                HR_threshold=0.1,erodedilate_HR=5,

                day_mode='both',thread=None,add_merge_gti=True,parallel=False):
    '''
    wrapper for a function to split nicer obsids into indivudal portions with different methods
    the default binning is 1s because the NICER mkf file time resolution is 1s

    overwrite is always on here since we don't use a specific nicerdas task with the overwrite option
    split modes (combinable):


        filtering:

        -orbit:split each obs into each individual nicer observation period. Generally, should always be enabled.
               GTIs naming: obsid-XXX chronologically for each split

        -flare: isolates background flare periods in each observation from the main data
                GTI naming: obsid-XXXFYYY


        -HR_flare: filters observations with unrealistically high HR in the 8-12/2-8 keV band
                    HR_threshold: sets the lower limit for the filtering criteria
                    erodedilate_HR: number of s of eroding on each side above the flaring gtis

        -overdyn: dynamically filters high overshoot regions by comparing the 0.35-8keV count rate and the overhsoot rate

                  When the 2-8 count rate is <overdyn_thresh cts/s,
                   filters when the overshoot rate is >overdyn_factor_low* the count rate
                  When the 2-8 count rate is >overdyn_thresh ct/s,
                   filters when the overshoot rate is >overdyn_factor_high* the count rate

                 dilates the resulting exclusion using the erodedilate_overdyn parameter (in s)


        -underdyn: dynamically filters high undershoot regions
                    if underdyn_method is set to compa:
                        searches anormal high undershoot periods by comparing the 0.35-2, 2-8 and undershoot lightcurves

                    if underdyn_method is set to gradient:
                        by searching for jumps in both the 0.35-8. lightcurve
                        and the undershoot lightcurve within a short period
                        underdyn_jump_width defines the maximum width for a jump to take place
                        underdyn_jump_factor defines the multiplicative height of a jump
                        A jump is considered valid if an undershoot jump happens\
                         within unerdyn_jump_width of a main counts jump
                        For now, Removes the entirety of the following orbit

        -hard_flare:
                isolate hard-only background flare periods in each observation from the non-nimkt excluded data
                Uses sigma clipping from the lowest of n segments of the remaining gti at this step
                GTI naming: obsid-XXXHFYYY
                Note that the goal here is to keep the data, so hard_flare is done AFTER overdyn

                we limit to periods bigger than hard_flare_min_duration to avoid removing simple random noise

                dilates the resulting exclusion using the erodedilate_hard_flare parameter (in s)

                Note that this is not needed if the nxbprel_norm parameter of the nxb background can't get the job done


        -split_X: on top of cutting splits and flares, splits each orbit in individual periods of X seconds for
                  time-resolved spectroscopy
                  GTI naming: obsid-XXXTYYY

        -manual: provides an interactive window to make individual gti splits per orbit.
                 GTI naming: obsid-XXXMYYY

        -intensity_N: Splits the gtis depending on their count rate in N portions
                  (from the delta between min and max of the count rate of each orbit)
                  N can be a number to split in even intensity delta intervals, or a list of quantiles in percent
                  to split in portions (should be ordered and not include 0 or 100)

                  Note that this requires a first set of orbit-level gtis, so the orbit option must be enabled
                  This also means that the flare and gtis excluded by nimaketime are also not in the lc which serves
                  as the base

                  ex:
                  -intensity_4 will split between below the 25%, 25-50%, 50-75% and >75%
                  of the max_counts-min_counts delta
                  -intensity_[15,50,85] will split below 15%, 15-50%, 50-85% and above 85%

                 GTI naming: obsid-XXXIYYY
            int_split_band:
                band in which the lightcurve used to create the intensity split will be created

            int_split_bin:
                binning of the lightcurve used for the intensity split

            clip_int_delta:
                3 sigma clip the lightcurve used for the intensity split before doing the repartition:
                the effect is a tighter repartition and the each side of the 3 sigma clip ends up with the first/last
                portion

    flare detection methods:
            -clip: clips the highest count portion of the observation in a given band
                   note that all individual flare/dip periods in single orbits are grouped together

                clip_method:
                    -median or mean to clip from the mean or from the median

                clip_sigma:
                    -variance sigmas for which to apply clipping to

                clip_band:
                    -which file or info from the mkf file to use to clip the flares
                    currently implemented:
                        -8-12keV
                        -overshoot

                flare_factor:
                    -treshold value of the minimal multiplication factor of the flare in clip_band

            -peak: performs peak detection in the individual orbit using findpeaks then exclude peak regions
                   with a given "score" (see https://erdogant.github.io/findpeaks/pages/html/Topology.html)

                peak_score_thresh:
                    peak score threshold to exlcude peaks as flares in the data

    -add_merge_gti:
    Merges every single GTI at the end of the filtering process and creates a single integrated obsid-level gti
    Only applied if not in split mode or manual mode

    gti_tool:
        software used for gti creation. Can be "SAS" (first version from before nigti existed)\
        or "NICERDAS" (now default)
    NOTE: requires sas and a sasinit alias to initialize it (to use tabgtigen)

    -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections

    '''

    def create_gti_files(id_gti, data_lc, orbit_prefix, suffix, file_base, time_gtis, gti_tool='NICERDAS',
                         id_gti_multi=None):

        '''
        creates a gti file from a list of indexes of times which will be picked in data_lc

        1.creates a copy of a lightcurve_type file (file_base) with an additional column with a gti mask
                (typically this is either the mkf file for classic cuts (so 1 second resolution)
                or a lightcurve file for more flexible resolution

        2.creates the gti file itself

        gti_tool:
            -SAS         uses sas's tabgtigen
            -NICERDAS    uses nicerdas nigti tool

        if id_gti_multi is provided, id_gti is assumed to be the raveled array and id_gti_multi an array with
        an additional dimension with each required gti split (typically by orbit).
        '''

        if len(id_gti) == 0:
            return

        # Here we use the housekeeping file as the fits base for the gti mask file
        fits_gti = fits.open(file_base)

        # creating the orbit gti expression
        gti_path = os.path.join(directory, 'xti', obsid + '_gti_' + orbit_prefix + suffix) + '.gti'

        if id_gti_multi is None:
            # preparing the list of gtis to replace manually
            gti_intervals = np.array(list(interval_extract(id_gti))).T
        else:
            # doing the interval extract within each subarray to automatically get the splits
            gti_intervals_full = [list(interval_extract(id_gti_multi[i_orbit])) for i_orbit in
            range(len(id_gti_multi))]
            gti_intervals = [[], []]

            for i_orbit in range(len(id_gti_split)):
                if len(gti_intervals_full[i_orbit])==0:
                    continue
                gti_intervals[0] += np.array(gti_intervals_full[i_orbit]).T[0].tolist()
                gti_intervals[1] += np.array(gti_intervals_full[i_orbit]).T[1].tolist()


            gti_intervals = np.array(gti_intervals)

        #returning if there is no gti_intervals at all
        if len(gti_intervals)==0:
            return

        delta_time_gtis = (time_gtis[1] - time_gtis[0]) / 2

        start_obs_s = fits_gti[1].header['TSTART'] + fits_gti[1].header['TIMEZERO']
        # saving for titles later
        mjd_ref = Time(fits_gti[1].header['MJDREFI'] + fits_gti[1].header['MJDREFF'], format='mjd')

        obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

        if gti_tool=='NICERDAS':

            '''
            the task nigti doesn't accept ISOT formats with decimal seconds so we use NICER MET instead 
            (see https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/nigti.html)
            
            we still add a -0.5*delta and +0.5*delta on each side to avoid issues with losing the last bins of lightcurves
            '''

            gti_input_path=os.path.join(directory, 'xti', obsid + '_gti_input_' + orbit_prefix + suffix) + '.txt'
            with open(gti_input_path,'w+') as f_input:
                f_input.writelines([str(start_obs_s+time_gtis[gti_intervals[0][i]]-delta_time_gtis)+' '+
                                    str(start_obs_s+time_gtis[gti_intervals[1][i]]+delta_time_gtis)+'\n'\
                                    for i in range(len(gti_intervals.T))])

            bashproc.sendline('nigti @'+gti_input_path+' '+gti_path+' clobber=YES chatter=4')
            bashproc.expect('ngti=',timeout=60)

        if gti_tool=='SAS':

            '''
            NOTE: NOT SURE THE INTEGRATED GTIs WOULD WORK WITH SAS
            '''

            # creating a custom gti 'mask' file
            gti_column = fits.ColDefs([fits.Column(name='IS_GTI', format='I',
                                                   array=np.array(
                                                       [1 if i in id_gti else 0 for i in range(len(data_lc))]))])

            # replacing the hdu with a hdu containing it
            fits_gti[1] = fits.BinTableHDU.from_columns(fits_gti[1].columns[:2] + gti_column)
            fits_gti[1].name = 'IS_GTI'

            lc_mask_path = os.path.join(directory, 'xti', obsid + '_gti_mask_' + orbit_prefix + suffix) + '.fits'

            if os.path.isfile(lc_mask_path):
                os.remove(lc_mask_path)

            fits_gti.writeto(lc_mask_path)

            # waiting for the file to be created
            while not os.path.isfile(lc_mask_path):
                time.sleep(0.1)

            bashproc.sendline('tabgtigen table=' + lc_mask_path + ' expression="IS_GTI==1" gtiset=' + gti_path)

            # this shouldn't take too long so we keep the timeout
            # two expects because there's one for the start and another for the end
            bashproc.expect('tabgtigen:- tabgtigen')
            bashproc.expect('tabgtigen:- tabgtigen')

            '''
            There is an issue with the way both tabgtigen and nicer creates the exposure due to a lacking keyword
            To ensure things work correctly, we remake the contents of the file and keep the header
            '''


            # opening and modifying the content of the header in the gti file for NICER
            with fits.open(gti_path, mode='update') as hdul:

                # for some reason we don't get the right values here so we recreate them
                # creating a custom gti 'mask' file

                # storing the current header
                prev_header = hdul[1].header

                # creating a START and a STOP column in "standard" GTI fashion
                # note: the 0.5 is there to allow the initial and final second bounds

                ####note that the start_obs_s-0.5 might need to be modified

                gti_column_start = fits.ColDefs([fits.Column(name='START', format='D',
                                                             array=np.array([time_gtis[elem] + start_obs_s -delta_time_gtis for elem in
                                                                             gti_intervals[0]]))])
                gti_column_stop = fits.ColDefs([fits.Column(name='STOP', format='D',
                                                            array=np.array([time_gtis[elem] + start_obs_s + delta_time_gtis for elem in
                                                                            gti_intervals[1]]))])

                # replacing the hdu
                hdul[1] = fits.BinTableHDU.from_columns(gti_column_start + gti_column_stop)

                # replacing the header
                hdul[1].header = prev_header

                # Changing the reference times
                hdul[1].header['MJDREF'] = 56658 + 7.775925925925930E-04

                # hdul[1].header['MJDREFI']=56658
                # hdul[1].header['MJDREFF']=7.775925925925930E-04

                # and the gti keywords
                hdul[1].header['ONTIME'] = 2*delta_time_gtis*len(id_gti)
                hdul[1].header['TSTART'] = hdul[1].data['START'][0] - start_obs_s
                hdul[1].header['TSTOP'] = hdul[1].data['STOP'][-1] - start_obs_s

                hdul.flush()

    io_log=open(directory+'/create_gtis.log','w+')

    #ensuring a good obsid name even in local
    if directory=='./':
        obsid=os.getcwd().split('/')[-1]
    else:
        obsid=directory

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8',logfile=io_log if parallel else None)

    print('\n\n\nCreating gtis products...')

    set_var(bashproc)

    if os.path.isfile(os.path.join(directory + '/extract_gtis.log')):
        os.system('rm ' + os.path.join(directory + '/extract_gtis.log'))


    pi_band = '-'.join((np.array(band.split('-')).astype(int) * 100).astype(str).tolist())

    #not needed currently since we don't create lc files anymore
    # #removing old lc files
    # old_files_lc = [elem for elem in glob.glob(os.path.join(directory + '/xti/**/*'), recursive=True) if
    #                 elem.endswith('.lc') and 'bin' not in elem]
    #
    # for elem_file in old_files_lc:
    #     os.remove(elem_file)

    #removing old gti files
    old_files_gti=[elem for elem in glob.glob(os.path.join(directory,'xti/**'), recursive=True) if
                   '_gti_' in elem]

    for elem_file_gti in old_files_gti:
        os.remove(elem_file_gti)

    with (no_op_context() if parallel else StdoutTee(os.path.join(directory + '/create_gtis.log'), mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(os.path.join(directory + '/create_gtis.log'), buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        '''
        new method for the flares following https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/flares/
        '''
        #should always be in auxil but we cover rank -2 directories this way. Also testing both gunzipped and non-gunzipped
        # (one is enough assuming they're the same)
        file_mkf=(glob.glob(os.path.join(directory,'**/**.mkf'),recursive=True)+glob.glob(os.path.join(directory,'**/**.mkf.gz'),recursive=True))[0]
        with fits.open(file_mkf) as fits_mkf:
            data_mkf = fits_mkf[1].data

            #from https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time_resolved_spec/
            #this value is offset by the mjd_ref value

            #note that there's an offset of 8 seconds between this value and the actual 1st column of
            #the time vector, for some reason ???

            #note that using leapinit can create a "Dumping CFITSIO error stack", see here:
            #https: // heasarc.gsfc.nasa.gov / docs / nicer / analysis_threads / common - errors /
            #so we don't consider the leapinit
            #start_obs_s=fits_mkf[1].header['TSTART']+fits_mkf[1].header['TIMEZERO']-fits_mkf[1].header['LEAPINIT']

            start_obs_s=fits_mkf[1].header['TSTART']+fits_mkf[1].header['TIMEZERO']
            #saving for titles later
            mjd_ref=Time(fits_mkf[1].header['MJDREFI']+fits_mkf[1].header['MJDREFF'],format='mjd')

            obs_start=mjd_ref+TimeDelta(start_obs_s,format='sec')
            
            obs_start_str=str(obs_start.to_datetime())

            time_obs=data_mkf['TIME']-start_obs_s

            counts_035_8=data_mkf['FPM_XRAY_PI_0035_0200']+data_mkf['FPM_XRAY_PI_0200_0800']

            counts_035_2=data_mkf['FPM_XRAY_PI_0035_0200']

            counts_2_8=data_mkf['FPM_XRAY_PI_0200_0800']

            counts_8_12=data_mkf['FPM_XRAY_PI_0800_1200']

            counts_overshoot=data_mkf['FPM_OVERONLY_COUNT']

            counts_undershoot = data_mkf['FPM_UNDERONLY_COUNT']

            cutoff_rigidity=data_mkf['COR_SAX']

        #fetching the standard gti file applied to the events
        #note that in some cases nimaketime is not created so we fetch the GTIs from the cleaned event file instead

        #file_nimaketime=glob.glob(os.path.join(directory,'**/**/nimaketime.gti'),recursive=True)[0]

        #how this was done can be checked in process_obsdir, the details of the evolution of the gtis is written
        #see https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/nimaketime.html

        file_evt_night=np.unique(glob.glob(os.path.join(directory,'**/**/**mpu7_cl.evt'),recursive=True)).tolist()
        file_evt_day=np.unique(glob.glob(os.path.join(directory,'**/**/**mpu7_cl_day.evt'),recursive=True)).tolist()

        assert day_mode in ['both','day', 'night'], 'Error: day_mode not among the authorized values (day,night,both)'

        if day_mode!='both':
            if day_mode=='day':
                file_evt_night=[]
            elif day_mode=='night':
                file_evt_day=[]

        assert len(file_evt_night+file_evt_day)>=1, 'Error: no event file detected'

        for elem_file_evt in file_evt_night+file_evt_day:

            split_keyword=''
            if '_day.evt' in elem_file_evt:
                day_mode_str='day mode'
                split_keyword+='D'
                plot_suffix='_day'
            else:
                day_mode_str='night mode'
                split_keyword+='N'
                plot_suffix='_night'

            print('Creating gtis for '+day_mode_str+' data\n')

            with fits.open(elem_file_evt) as hdul:

                #in the gti file
                #gti_nimaketime=hdul[1].data

                #in the cleaned event file
                gti_nimaketime=hdul[3].data

                # in the nimaketime the header TSTART is the same than in the housekeeping file but the start
                # in the event file is shifted, so we use the value we retrieved before

                #array offseted form for plotting and computations later


                gti_nimkt_arr=np.array(gti_nimaketime.tolist()) - start_obs_s


            #adding gaps of more than 100s as cuts in the gtis
            #useful in all case to avoid inbetweens in the plot even if we don't cut the gtis

            #first computing the gti where the jump happens
            id_gti_split=[-1]
            #adding gaps of more than 100s as cuts in the gtis
            for i in range(len(time_obs) - 1):
                if time_obs[i + 1] - time_obs[i] > 100:
                    id_gti_split += [i]

            id_gti_orbit=[]
            if len(id_gti_split)==1:
                id_gti_orbit+=[range(len(time_obs))]
            else:
                for id_split in range(len(id_gti_split)):

                    #note:+1 at the end since we're using a range
                    id_gti_orbit+=[list(range(id_gti_split[id_split]+1,(len(time_obs)-1 if\
                        id_split==len(id_gti_split)-1 else id_gti_split[id_split+1])+1))]

            n_orbit=len(id_gti_orbit)

            split_gti_arr=np.array([None]*n_orbit)

            # computing the non-gti intervals from nimaketime
            id_nongti_nimkt = []

            for elem_gti_orbit in id_gti_orbit:

                id_nongti_nimkt_orbit = []

                for elem_gti in elem_gti_orbit:
                    # testing if the gti is in one of the gtis of the nimaketime
                    if not ((time_obs[elem_gti] >= gti_nimkt_arr.T[0]) & (
                            time_obs[elem_gti] <= gti_nimkt_arr.T[1])).any():
                        # and storing if that's not the case
                        id_nongti_nimkt_orbit += [elem_gti]

                id_nongti_nimkt+=[id_nongti_nimkt_orbit]

            if 'split' in split_arg:

                split_sampling=float([elem for elem in split_arg.split('+') if 'split' in elem][0].split('_')[1])

                for i_orbit, id_nongti_nimkt_orbit in enumerate(id_nongti_nimkt):

                    #computing the starting id, a bit convoluted but not to depend on the binning
                    start_id=np.argwhere([elem not in id_nongti_nimkt_orbit for elem in id_gti_orbit[i_orbit]])\
                        [0][0]

                    split_gti_orbit=[]
                    indiv_split=[]

                    #failsafe if the full orbit is ruled out
                    if len(id_gti_orbit[i_orbit][start_id:])==0:
                        split_gti_arr[i_orbit]=[]
                        continue

                    #and split from that
                    for id_gti in id_gti_orbit[i_orbit][start_id:]:

                        #starting a gti interval
                        if len(indiv_split)<2:
                            indiv_split+=[id_gti]
                        else:
                            #testing if adding a gti doesn't make the interval larger than the split
                            if time_obs[id_gti]-time_obs[indiv_split[0]]<=split_sampling:
                                indiv_split+=[id_gti]
                            else:
                                #storing the interval
                                split_gti_orbit+=[indiv_split]

                                #and resetting it for the next id
                                indiv_split=[id_gti]

                    #adding the last interval
                    split_gti_orbit+=[indiv_split]

                    split_gti_arr[i_orbit]=split_gti_orbit

            #plotting the global figure
            save_path_str=os.path.join(directory,obsid+'-global_flares'+plot_suffix+'.png')

            plot_event_diag('global',obs_start_str,time_obs,
                            id_gti_orbit=id_gti_orbit,
                            counts_035_8=counts_035_8,
                            counts_035_2=counts_035_2,
                            counts_2_8=counts_2_8,
                            counts_8_12=counts_8_12,
                            counts_overshoot=counts_overshoot,
                            counts_undershoot=counts_undershoot,
                            cutoff_rigidity=cutoff_rigidity,
                            save_path=save_path_str)

            #can be modified if needed

            if clip_band=='8-12':
                flare_lc=counts_8_12
            else:
                breakpoint()
                #to be implemented


            clip_lc=np.where(np.isnan(flare_lc),0,flare_lc)

            if 'flare' in split_arg:

                '''
                Note: this one is mainly for short, bright flare which completely distort the spectrum and
                need to be removed
                '''

                id_gti=[]
                id_flares=[]
                id_dips=[]

                for i_orbit,elem_gti_orbit in enumerate(id_gti_orbit):

                    elem_id_flares=[]


                    if 'clip' in flare_method:

                        # note:this one should be improved to something closer to hard_flare

                        #should clip anything weird in the data unless the flare is huge
                        clip_data = sigma_clip(clip_lc[elem_gti_orbit], sigma=2)

                        #switching to log10 for more constraining stds, the +3 should avoid any negative values
                        clip_data=np.log10(np.where(clip_data==0,0.01,clip_data))+3

                        clip_std=clip_data.std()

                        if clip_method=='mean':
                            clip_base=clip_data.mean()
                        else:
                            clip_sort=clip_data.copy()
                            clip_sort.sort()
                            clip_base=clip_sort[int(len(clip_sort)/2)]

                        #computing the gtis outside of the 3 sigma of the clipped distribution
                        #even with uncertainties
                        #note: inverted errors in the data on purpose
                        # elem_id_gti=[elem for elem in elem_gti_orbit if \
                        #              (clip_lc[elem]+clip_sigma)>=clip_base-3*clip_sigma*clip_std\
                        #              and (clip_lc[elem])<=clip_base+3*clip_std]

                        elem_id_flares+=[elem for elem in elem_gti_orbit if \
                                     (clip_lc[elem])>clip_base+clip_sigma*clip_std and clip_lc[elem]>=clip_base+np.log10(flare_factor)]

                    if 'peak' in flare_method:
                        def peak_search(array_arg,topo_score_thresh=3,id_offset=0):

                            '''
                            Searches for peaks using two different methods (topology and peakdetect)
                            from https://erdogant.github.io/findpeaks/pages/html/index.html

                            then combines the two to find strong peaks (topology score superior to a threshold)
                            and the corresponding peak size (which is accessible with peakdetect)

                            topo_score_thresh: minimum topology score to retain peaks

                            id_offset: offset the position of the indexes of the returned peaks
                            Useful for orbits > 1 where the id starts at the first id_gti_orbit instead of 0
                            '''

                            #first algo, restricted to peaks
                            peak_finder = findpeaks(method='topology', interpolate=None,whitelist=['peak'])

                            results_topo = peak_finder.fit(X=array_arg)

                            #raking of the peaks in the current algorithm, filtered to get only the strong one
                            strong_peak_pos=results_topo['persistence']['y'][results_topo['persistence']['score']>topo_score_thresh]

                            #lookahead value manually adjusted, could become a parameter
                            peak_finder = findpeaks(method='peakdetect', lookahead=5)

                            results_peakdet = peak_finder.fit(X=array_arg)

                            #this is a list of the number of each peak region the current point belongs to
                            peak_region=results_peakdet['df']['labx']

                            #now we need to merge the peak regions to avoid repeating regions in which lie
                            #several peaks
                            strong_peak_region=[]

                            #this is the list of peak number regions
                            strong_peak_number=[]

                            for i in range(len(strong_peak_pos)):

                                if peak_region[strong_peak_pos[i]] not in strong_peak_number:

                                    strong_peak_number+=[peak_region[strong_peak_pos[i]]]

                                    strong_peak_region+=[(np.argwhere(peak_region==peak_region[strong_peak_pos[i]]).T[0]\
                                                        +id_offset).tolist()]

                            return strong_peak_pos+id_offset,strong_peak_region

                        #taking off the nans to avoid issues, and offsetting by the index position of the first GTI
                        # in the orbit
                        peak_soft,peak_region_soft=peak_search(np.where(np.isnan(counts_035_8[elem_gti_orbit]),0,
                                                                        counts_035_8[elem_gti_orbit]),
                                                                        id_offset=elem_gti_orbit[0],
                                                                        topo_score_thresh=peak_score_thresh)
                        peak_hard,peak_region_hard=peak_search(np.where(np.isnan(counts_8_12[elem_gti_orbit]),0,
                                                                        counts_8_12[elem_gti_orbit]),
                                                                        id_offset=elem_gti_orbit[0],
                                                                        topo_score_thresh=peak_score_thresh)

                        peak_region_multi=[]

                        '''
                        Since we want at least a peak in hard, we fetch the peaks positions in hard 
                        which are part of a peak region in soft
                        
                        For these, we merge the corresponding peak regions in hard and soft to be conservative                    
                        '''

                        #loop in hard peaks
                        for elem_peak in peak_hard:

                            peak_region_match_soft=[elem_region for elem_region in peak_region_soft if elem_peak in elem_region]

                            if len(peak_region_match_soft)!=0:

                                peak_region_match_hard=[elem_region for elem_region in peak_region_hard if elem_peak in elem_region]

                                #adding both to the global peak regions list
                                peak_region_multi+=peak_region_match_soft[0]

                                if len(peak_region_match_hard)!=0:
                                    peak_region_multi+=peak_region_match_hard[0]

                        #cleaning potential repeats
                        peak_region_multi=np.unique(peak_region_multi).tolist()

                        #adding anerodedilate factor of 30s for safety around big peaks

                        peak_region_eroded=[]
                        if len(peak_region_multi)>0:
                            for i_region,peak_interval in enumerate(interval_extract(peak_region_multi)):
                                if peak_interval[1]-peak_interval[0]>60:
                                    peak_region_eroded+=np.arange(max(peak_interval[0]-30,elem_gti_orbit[0]),
                                                                  peak_interval[0]).tolist()+\
                                                        np.arange(peak_interval[0],peak_interval[1]+1).tolist()+ \
                                                       np.arange(peak_interval[1]+1,
                                                        min(peak_interval[1]+31,elem_gti_orbit[-1]+1)).tolist()
                                else:
                                    peak_region_eroded+=np.arange(peak_interval[0],peak_interval[1]+1).tolist()

                            elem_id_flares+=peak_region_eroded

                    #ensuring no repeats
                    elem_id_flares=np.unique(elem_id_flares).tolist()

                    #defining the gtis after the flares have been defined
                    elem_id_gti = [elem for elem in elem_gti_orbit if elem not in elem_id_flares]

                    id_gti += [elem_id_gti]
                    id_flares += [elem_id_flares]

                    if 'split' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split]=[elem for elem in split_gti_arr[i_orbit][i_split]\
                                                         if elem not in elem_id_flares]

            else:
                id_gti=id_gti_orbit
                id_flares=[]
                id_dips=[]

            id_undercut = np.array([None] * n_orbit)

            if 'underdyn' in split_arg:
                for i_orbit, (elem_gti_orbit,id_nongti_nimkt_orbit) in enumerate(zip(id_gti,id_nongti_nimkt)):

                    # note that here we're only filtering what's not in the flares

                    if underdyn_method=='compa':

                        '''
                        We look at the combination of 3 criteria to assert whether the source is contaminated.
                        1. abnormally high 0.35-2 count rate compared to the undershoots (>10%)
                        2. abnormally low HR ratio in 2-8/0.35-2 (<2%)
                        3. high undershoots (>100cts/s)
                        '''
                        # note that here we're only filtering what's not in the flares

                        # we cut with different criteria for high and low bg rates to have better filtering
                        mask_under = ((counts_035_2[elem_gti_orbit] >= 0.1* counts_undershoot[elem_gti_orbit]) & \
                                     (counts_2_8[elem_gti_orbit]/counts_035_2[elem_gti_orbit] <= 0.02) & \
                                     (counts_undershoot[elem_gti_orbit]>100))

                        id_under_orbit = np.array(elem_gti_orbit)[mask_under].tolist()

                        id_undercut[i_orbit] = id_under_orbit

                        # redefining the gtis after the flares have been defined
                        id_gti[i_orbit] = [elem for elem in elem_gti_orbit if elem not in id_under_orbit]

                        # this one for is for the automatic split
                        if 'split' in split_arg:
                            for i_split in range(len(split_gti_arr[i_orbit])):
                                split_gti_arr[i_orbit][i_split] = [elem for elem in split_gti_arr[i_orbit][i_split] \
                                                                   if elem not in id_under_orbit]


                    elif underdyn_method=='gradient':

                        '''
                        OUTDATED
                        Here we're looking for a gradient in undershoots that would be linked to a gradient in main count rate
                        and not be diagnosed
                        The jumps are usually very short
                        '''

                        assert type(underdyn_jump_width)==int,'Error: underdyn_jump_width must be an integer'

                        jump_undershoot_id =[]
                        jump_main_counts_id=[]

                        #note: here to avoid issues we only compute these starting from positions outside of nimkt,
                        # to avoid issues with starting jumps
                        for elem_gti in [gti for gti in elem_gti_orbit[:-1] if gti not in id_nongti_nimkt_orbit]:

                            #identifying the neighboring jumps that are also INSIDE the gtis and not in the nimkt excluded
                            mask_jump_main_counts=(counts_035_8[elem_gti:min(elem_gti+underdyn_jump_width,elem_gti_orbit[-1])]>\
                                                   underdyn_jump_factor*counts_035_8[elem_gti])
                            mask_okgti_main_counts=np.array([elem in elem_gti_orbit and elem not in id_nongti_nimkt_orbit for elem in \
                                                    np.arange(elem_gti,min(elem_gti+underdyn_jump_width,elem_gti_orbit[-1]))])

                            if np.any((mask_jump_main_counts) & (mask_okgti_main_counts)):
                                jump_main_counts_id+=[elem_gti]

                            #same for undershoot jumps but we don't restrict them to the current gtis

                            mask_jump_underdyn=(counts_undershoot[elem_gti:min(elem_gti+underdyn_jump_width,elem_gti_orbit[-1])]>\
                                                underdyn_jump_factor*counts_undershoot[elem_gti])

                            if np.any(mask_jump_underdyn):
                                jump_undershoot_id+=[elem_gti]

                        jump_undershoot_id=np.array(jump_undershoot_id)
                        jump_main_counts_id=np.array(jump_main_counts_id)

                        #finding the ids where there is less than a 10 second gap between the jumps

                        #note: there are some jumps with more than 10s at the end of orbits so using a higher value than
                        #underdyn_jump_wdith can help

                        if len(jump_undershoot_id)>0 and len(jump_main_counts_id)>0:
                            id_jump_both=[elem_jump_main for elem_jump_main in jump_main_counts_id\
                                            if np.any(abs(jump_undershoot_id-elem_jump_main)<3*underdyn_jump_width)]
                        else:
                            id_jump_both=[]

                        #stopping the gti at the first id
                        if len(id_jump_both)>0:
                            elem_id_undercut = np.array(elem_gti_orbit)[elem_gti_orbit>id_jump_both[0]]

                            # defining the gtis after the flares have been defined
                            elem_id_gti = [elem for elem in elem_gti_orbit if elem not in elem_id_undercut]

                            id_gti[i_orbit] = elem_id_gti
                            id_undercut [i_orbit]= elem_id_undercut



            id_overcut = np.array([None] * n_orbit)

            if 'overdyn' in split_arg:

                for i_orbit, (elem_gti_orbit,id_nongti_nimkt_orbit) in enumerate(zip(id_gti,id_nongti_nimkt)):

                    # note that here we're only filtering what's not in the flares

                    # we cut with different criteria for high and low bg rates to have better filtering
                    mask_over = ((counts_2_8[elem_gti_orbit] <= overdyn_thresh) & \
                                 (counts_overshoot[elem_gti_orbit] > overdyn_factor_low * counts_2_8[elem_gti_orbit])) | \
                                ((counts_2_8[elem_gti_orbit] > overdyn_thresh) & \
                                 (counts_overshoot[elem_gti_orbit] > overdyn_factor_high * counts_2_8[elem_gti_orbit]))

                    id_over_orbit = np.array(elem_gti_orbit)[mask_over].tolist()

                    if erodedilate_overdyn > 0:

                        assert type(erodedilate_overdyn) == int, 'Error: erodedilate_overdyn should be an integer'

                        id_over_orbit_eroded = np.unique(ravel_ragged([ \
                            np.arange(elem - erodedilate_overdyn, elem + erodedilate_overdyn + 1) \
                            for elem in id_over_orbit]))

                        # limiting to the initial range of gtis to avoid eroded beyond the bounds
                        id_over_orbit_eroded = [elem for elem in elem_gti_orbit if elem in id_over_orbit_eroded]
                    else:
                        id_over_orbit_eroded = id_over_orbit

                    # removing nimaketime exclusions
                    id_over_orbit_eroded = [elem for elem in id_over_orbit_eroded if elem not in id_nongti_nimkt_orbit]

                    id_overcut[i_orbit] = id_over_orbit_eroded

                    # redefining the gtis after the flares have been defined
                    id_gti[i_orbit] = [elem for elem in elem_gti_orbit if elem not in id_over_orbit_eroded]

                    # this one for is for the automatic split
                    if 'split' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = [elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in id_over_orbit_eroded]

            id_HR_flares=np.array([None] * n_orbit)

            if 'HR' in split_arg:

                for i_orbit, (elem_gti_orbit,id_nongti_nimkt_orbit) in enumerate(zip(id_gti,id_nongti_nimkt)):

                    # note that here we're only filtering what's not in the flares

                    # we cut with different criteria for high and low bg rates to have better filtering
                    #we only filter when the 2_8 count rate per FPM is above 0.2 to avoid issues with high uncertainties
                    #at low count rates
                    mask_HR_thresh = (counts_8_12[elem_gti_orbit]/counts_2_8[elem_gti_orbit] >= HR_threshold)\
                                    & (counts_8_12[elem_gti_orbit] >= 0.2)

                    id_HR_flare_orbit = np.array(elem_gti_orbit)[mask_HR_thresh].tolist()

                    if erodedilate_HR > 0:

                        assert type(erodedilate_HR) == int, 'Error: erodedilate_HR should be an integer'

                        id_HR_flare_orbit_eroded = np.unique(ravel_ragged([ \
                            np.arange(elem - erodedilate_HR, elem + erodedilate_HR + 1) \
                            for elem in id_HR_flare_orbit]))

                        # limiting to the initial range of gtis to avoid eroding beyond the bounds
                        id_HR_flare_orbit_eroded = [elem for elem in elem_gti_orbit if elem in id_HR_flare_orbit_eroded]
                    else:
                        id_HR_flare_orbit_eroded = id_over_orbit

                    # removing nimaketime exclusions
                    id_HR_flare_orbit_eroded = [elem for elem in id_HR_flare_orbit_eroded if elem not in id_nongti_nimkt_orbit]

                    id_HR_flares[i_orbit] = id_HR_flare_orbit_eroded

                    # redefining the gtis after the flares have been defined
                    id_gti[i_orbit] = [elem for elem in elem_gti_orbit if elem not in id_HR_flare_orbit_eroded]

                    # this one for is for the automatic split
                    if 'split' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = [elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in id_HR_flare_orbit_eroded]

            id_hard_flares=np.array([None] * n_orbit)

            if 'hard_flare' in split_arg:

                '''
                There are also a second type of less bright flares that mainly affect the sp above 8 keV.
                These ones do not correlate as much to the overshoot lightcurve, and often don't with
                the mainlightcurve
                
                The hard flares are thus defined from sigma clipping using the lowest region of the high energy
                FPM lightcurve
                '''


                for i_orbit, (elem_gti_orbit,id_nongti_nimkt_orbit) in enumerate(zip(id_gti,id_nongti_nimkt)):


                    '''
                    To avoid potential issues with too much source variability, we split the remaining data into
                    5 segments, and will compute the sigma clipping threshold from the faintest of the 4
                    
                    We restrict the segment cut to the non-nimkt excluded portion
                    '''
                    elem_id_hard_flares = []


                    #restricting the elem_gti_orbit to the non excluded periods
                    elem_gti_orbit_oknimkt=[elem for elem in elem_gti_orbit if elem not in id_nongti_nimkt_orbit]

                    if len(elem_gti_orbit_oknimkt)==0:
                        continue

                    #cutting the high E lightcurve
                    flare_lc_hardcut=flare_lc[elem_gti_orbit_oknimkt]

                    #computing median and stds for each split
                    median_sampled = [np.nanmedian(flare_lc_hardcut[int(i * len(flare_lc_hardcut) / hard_flare_segments)\
                                                              :int((i + 1) * len(flare_lc_hardcut) / hard_flare_segments)])
                                     for i in range(5)]
                    std_sampled = [np.nanstd(flare_lc_hardcut[int(i * len(flare_lc_hardcut) / hard_flare_segments)\
                                                              :int((i + 1) * len(flare_lc_hardcut) / hard_flare_segments)])
                                     for i in range(5)]

                    #4 sigma clipping for now. Should not consider the full nan periods if there are some
                    hard_flare_clip_threshold=min(median_sampled)+hard_flare_sigma*std_sampled[np.argmin(median_sampled)]

                    elem_id_hard_flares += [elem for elem in elem_gti_orbit_oknimkt\
                                            if flare_lc[elem]>hard_flare_clip_threshold]

                    if erodedilate_hard_flare>0:
                        assert type(erodedilate_hard_flare)==int,'Error: erodedilate_hard_flare should be an integer'

                        elem_id_hard_flares_eroded=np.unique(ravel_ragged([\
                                             np.arange(elem-erodedilate_hard_flare,elem+erodedilate_hard_flare+1)\
                                                                    for elem in elem_id_hard_flares]))

                        #limiting to the initial range of gtis to avoid eroding beyond the bounds
                        elem_id_hard_flares_eroded=[elem for elem in elem_gti_orbit_oknimkt if elem in elem_id_hard_flares_eroded]
                    else:
                        elem_id_hard_flares_eroded=elem_id_hard_flares

                    elem_id_hard_flares=elem_id_hard_flares_eroded

                    hard_flare_intervals = list(interval_extract(elem_id_hard_flares))
                    hard_flare_intervals_clean=[elem for elem in hard_flare_intervals if elem[1]-elem[0]>hard_flare_min_duration]

                    elem_id_hard_flares=ravel_ragged([np.arange(elem[0],elem[1]+1).tolist() for elem in hard_flare_intervals_clean])

                    # defining the gtis after the flares have been defined
                    elem_id_gti = [elem for elem in elem_gti_orbit if elem not in elem_id_hard_flares]

                    id_gti [i_orbit]=elem_id_gti
                    id_hard_flares += [elem_id_hard_flares]

                    if 'split' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = [elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in elem_id_hard_flares]


            if 'manual' in split_arg:

                split_gti_arr= np.array([None] * n_orbit)

                for i_orbit in range(n_orbit):
                    orbit_cut_times=[time_obs[id_gti[i_orbit]][0]]

                    while orbit_cut_times[-1]!=time_obs[id_gti[i_orbit]][-1]:

                        orbit_cut_times+=\
                                    [plot_event_diag(mode='manual', obs_start_str=obs_start_str, time_obs=time_obs,
                                    id_gti_orbit=id_gti_orbit[i_orbit],
                                    counts_035_8=counts_035_8[id_gti_orbit[i_orbit]],
                                     counts_035_2=counts_035_2[id_gti_orbit[i_orbit]],
                                     counts_2_8=counts_2_8[id_gti_orbit[i_orbit]],
                                    counts_8_12=counts_8_12[id_gti_orbit[i_orbit]],
                                    counts_overshoot=counts_overshoot[id_gti_orbit[i_orbit]],
                                    counts_undershoot=counts_undershoot[id_gti_orbit[i_orbit]],
                                    cutoff_rigidity=cutoff_rigidity[id_gti_orbit[i_orbit]],
                                    save_path='',
                                    id_gti=id_gti[i_orbit], id_flares=id_flares[i_orbit],
                                    id_overcut=id_overcut[i_orbit],id_undercut=id_undercut[i_orbit],
                                    id_hard_flares=id_hard_flares[i_orbit],
                                    id_HR_flares=id_HR_flares[i_orbit],
                                    gti_nimkt_arr=gti_nimkt_arr, split_arg=split_arg,
                                    split_gti_arr=split_gti_arr[i_orbit],orbit_cut_times=orbit_cut_times)]

                        print('Added gti manual split at t='+str(orbit_cut_times[-1])+' s')

                    n_cuts=len(orbit_cut_times)
                    cut_gtis = [np.argwhere(time_obs == orbit_cut_times[i_cut])[0][0] for i_cut in
                                range(n_cuts)]

                    #note that the min(i_cut+1) offsets all but the first cut's gti starts by one, and the end is always
                    #offset by one. So here we make the choice that the cut is part of the gti up to that cut
                    split_gti_arr[i_orbit]=np.array([min(i_cut,1)+np.arange(cut_gtis[i_cut],cut_gtis[i_cut+1]+max(1-i_cut,0)) for i_cut in range(n_cuts-1)],dtype=object)

                    #removing flares if necessary
                    if 'flares' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = np.array([elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in id_flares[i_orbit]])

                    #removing overdyn if necesary
                    if 'overdyn' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = np.array([elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in id_overcut[i_orbit]])

                    #removing overdyn if necesary
                    if 'underdyn' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = np.array([elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in id_undercut[i_orbit]])

                    #removing hard flares if necesary
                    if 'hard_flare' in split_arg:
                        for i_split in range(len(split_gti_arr[i_orbit])):
                            split_gti_arr[i_orbit][i_split] = np.array([elem for elem in split_gti_arr[i_orbit][i_split] \
                                                               if elem not in id_hard_flares[i_orbit]])

            #creating individual orbit figures
            for i_orbit in range(n_orbit):

                save_path_str=os.path.join(directory,obsid+'-'+str_orbit(i_orbit)+plot_suffix+'_flares.png')

                plot_event_diag(mode='orbit',obs_start_str=obs_start_str,time_obs=time_obs,
                                id_gti_orbit=id_gti_orbit[i_orbit],
                                counts_035_8=counts_035_8[id_gti_orbit[i_orbit]],
                                counts_035_2=counts_035_2[id_gti_orbit[i_orbit]],
                                counts_2_8=counts_2_8[id_gti_orbit[i_orbit]],
                                counts_8_12=counts_8_12[id_gti_orbit[i_orbit]],
                                counts_overshoot=counts_overshoot[id_gti_orbit[i_orbit]],
                                counts_undershoot=counts_undershoot[id_gti_orbit[i_orbit]],
                                cutoff_rigidity=cutoff_rigidity[id_gti_orbit[i_orbit]],
                                save_path=save_path_str,
                                id_gti=id_gti[i_orbit],id_flares=id_flares[i_orbit],
                                id_overcut=id_overcut[i_orbit],id_undercut=id_undercut[i_orbit],
                                id_hard_flares=id_hard_flares[i_orbit],
                                id_HR_flares=id_HR_flares[i_orbit],
                                gti_nimkt_arr=gti_nimkt_arr,split_arg=split_arg,
                                split_gti_arr=split_gti_arr[i_orbit])

            #creating the gti files for each part of the obsid
            # (note that this won't make anything if there's no sas keyword)
            if gti_tool=='SAS' and sas_init_alias!='':
                bashproc.sendline(sas_init_alias)

            # NOTE: this doesn't work nor it did back then with XMM_datared, so using masks instead
            # def expr_gti(time_arr,id_arr):
            #
            #     intervals_id=list(interval_extract(id_arr))
            #
            #     expr=''
            #
            #     for i_inter,elem_inter in enumerate(intervals_id):
            #         expr+='(TIME>='+str(time_arr[elem_inter[0]])+' AND TIME<='+str(time_arr[elem_inter[1]])+')'
            #
            #         if i_inter!=len(intervals_id)-1:
            #             expr+=' OR '
            #
            #     return expr

            for i_orbit in range(n_orbit):

                if 'split' in split_arg or 'manual' in split_arg:

                    split_keyword += ('S' if 'split' in split_arg else 'MAN' if 'manual' in split_arg else '')
                    #create the gti files with a "S" keyword and keeping the orbit information in the name
                    for i_split,split_gtis in enumerate(split_gti_arr[i_orbit]):
                        if len(split_gtis)>0:
                            create_gti_files(split_gtis,flare_lc,str_orbit(i_orbit),suffix=split_keyword+str_orbit(i_split),
                                             file_base=file_mkf,time_gtis=time_obs,gti_tool=gti_tool)
                else:
                    create_gti_files(id_gti[i_orbit],flare_lc,str_orbit(i_orbit),suffix=split_keyword+'',
                                     file_base=file_mkf,time_gtis=time_obs,gti_tool=gti_tool)

                if len(id_flares[i_orbit])>0:
                    create_gti_files(id_flares[i_orbit],flare_lc,str_orbit(i_orbit),suffix=split_keyword+'F',
                                     file_base=file_mkf,time_gtis=time_obs,gti_tool=gti_tool)
                if 'hard_flare' in split_arg:
                    if len(id_hard_flares[i_orbit])>0:
                        create_gti_files(id_hard_flares[i_orbit], flare_lc, str_orbit(i_orbit), suffix=split_keyword + 'HF',
                                         file_base=file_mkf, time_gtis=time_obs, gti_tool=gti_tool)


                if add_merge_gti:

                    # giving both the full id_gti and the one with the orbit splits (which will be used to ensure)
                    # the gti intervals won't be overextended between each orbit
                    create_gti_files(ravel_ragged(id_gti), flare_lc, '', suffix=split_keyword + 'MRG',
                                    file_base = file_mkf, time_gtis = time_obs, gti_tool = gti_tool,
                                    id_gti_multi = id_gti)

            if 'intensity' in split_arg:

                split_gti_arr= np.array([None] * n_orbit)

                '''
                Splitting according to a given intensity repartition 
                This is done after the first gti creation because it requires the gti files of individual
                orbits to be created
                '''

                assert 'orbit' in split_arg,"Error: intensity splitting requires orbit splitting"

                int_split_keyword=[elem for elem in split_arg.split('+') if 'intensity' in elem][0]

                int_split_type=literal_eval(int_split_keyword.split('_')[1])

                if type(int_split_type)==int:
                    int_split_quantiles=np.linspace(0,100,int_split_type+1)
                else:
                    int_split_quantiles=[0]+int_split_type+[100]


                #creating a lightcurve for all orbits with the given band and binning
                #here we assume that orbit gtis are already created
                extract_lc(directory,binning=int_split_bin,bands=int_split_band,HR=None,overwrite=True,
                           skip_merge=True,
                           day_mode='day' if day_mode_str=='day mode' else\
                                    'night' if day_mode_str=='night mode' else '')

                for i_orbit in range(n_orbit):

                    int_lc_orbit_path=os.path.join(directory,'xti',obsid+'-'+split_keyword+str_orbit(i_orbit)+'_'+str(int_split_band)+\
                                                '_bin_'+str(int_split_bin)+'.lc')

                    with fits.open(int_lc_orbit_path) as hdul:

                        int_lc_orbit_time = hdul[1].data['TIME']
                        int_lc_orbit_cts= hdul[1].data['RATE']

                        #cts_err = hdul[1].data['ERROR']


                    if clip_int_delta:
                        #sigma clipping to remove significant outliers
                        int_lc_orbit_cts_var=sigma_clip(int_lc_orbit_cts,3)
                    else:
                        int_lc_orbit_cts_var=sigma_clip(int_lc_orbit_cts,3)


                    int_lc_orbit_cts_delta = int_lc_orbit_cts_var.max() - int_lc_orbit_cts_var.min()

                    orbit_gti_int=[]

                    time_quantiles=[]

                    for i_quantile in range(len(int_split_quantiles)-1):
                        mask_quantile= (int_lc_orbit_cts > (int_lc_orbit_cts_var.min() + int_lc_orbit_cts_delta * int_split_quantiles[i_quantile]/100)) & \
                                       (int_lc_orbit_cts <= (int_lc_orbit_cts_var.min() + int_lc_orbit_cts_delta * int_split_quantiles[i_quantile+1]/100))

                        #adding the lowest value bins below the clipping for the lowest quantile
                        if i_quantile==0:
                            mask_quantile= mask_quantile | (int_lc_orbit_cts <= int_lc_orbit_cts_var.min())

                        #adding the highest value bins above the clipping for the highest quantile
                        if i_quantile==len(int_split_quantiles)-2:
                            mask_quantile = mask_quantile | (int_lc_orbit_cts > int_lc_orbit_cts_var.max())

                        #here we don't transfer into the mkf back because we'll be using the lightcurve file directly
                        #for the gtis so a range of the ids of mask_quantile is directly the gtis we want
                        orbit_gti_int+=[np.arange(len(int_lc_orbit_time))[mask_quantile]]

                        # #careful here, this can be tricky when transfered into lightcurves later depending
                        # #on the binning of said lightcurves
                        # time_quantiles+=[lc_time[mask_quantile]-0.5*float(int_split_bin)]
                        #
                        # #adding the gti indexes
                        # orbit_gti_int+=[np.array([np.argwhere(time_obs==time_quantiles[i_quantile][i_elem])[0][0] for i_elem in
                        #    range(len(time_quantiles[i_quantile]))])]

                    split_gti_arr[i_orbit]=orbit_gti_int

                    save_path_str = os.path.join(directory, obsid + '-' +split_keyword+str_orbit(i_orbit)+
                                                 plot_suffix+ '_flares.png')

                    #it's too complicated to display the time cut here so we simply don't
                    plot_event_diag(mode='orbit', obs_start_str=obs_start_str, time_obs=time_obs,
                                    id_gti_orbit=id_gti_orbit[i_orbit],
                                    counts_035_8=counts_035_8[id_gti_orbit[i_orbit]],
                                    counts_035_2=counts_035_2[id_gti_orbit[i_orbit]],
                                    counts_2_8=counts_2_8[id_gti_orbit[i_orbit]],
                                    counts_8_12=counts_8_12[id_gti_orbit[i_orbit]],
                                    counts_overshoot=counts_overshoot[id_gti_orbit[i_orbit]],
                                    counts_undershoot=counts_undershoot[id_gti_orbit[i_orbit]],
                                    cutoff_rigidity=cutoff_rigidity[id_gti_orbit[i_orbit]],
                                    counts_035_8_glob=counts_035_8,
                                    save_path=save_path_str,
                                    id_gti=id_gti[i_orbit], id_flares=id_flares[i_orbit],
                                    id_hard_flares=id_hard_flares[i_orbit],
                                    id_HR_flares=id_HR_flares[i_orbit],
                                    gti_nimkt_arr=gti_nimkt_arr, split_arg=split_arg,
                                    id_overcut=id_overcut[i_orbit],
                                    id_undercut=id_undercut[i_orbit],
                                    split_gti_arr=None)

                    split_keyword+='I'

                    #create the gti files with a "I" keyword and keeping the orbit information in the name
                    #here we use the int_lc directly to have the right resolution
                    for i_split,split_gtis in enumerate(split_gti_arr[i_orbit]):
                        if len(split_gtis)>0:
                            create_gti_files(split_gtis,int_lc_orbit_time,str_orbit(i_orbit),split_keyword+str_orbit(i_split),
                                             file_base=int_lc_orbit_path,time_gtis=int_lc_orbit_time,
                                             gti_tool=gti_tool)

                    #removing the non-split gti file
                    # (here we remove the last letter to remove the intensity keyword and come back to what the keyword
                    # was at the beginning
                    os.remove(os.path.join(directory,'xti',obsid+'_gti_'+split_keyword[:-1]+str_orbit(i_orbit)+'.gti'))

        #exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()

#### extract_all_spectral
def extract_all_spectral(directory,bkgmodel='scorpeon_script',language='python',overwrite=True,relax_SAA_bg=True,
                         sp_systematics=True,day_mode='both',thread=None,parallel=False):
    
    '''
    Wrapper for nicerl3-spect, extracts spectra, creates bkg and rmfs

    Note: can produce no output without error if no gti (in total, not files) in the event file

    if gti files created by create_gtis are present, instead of creating a full spectrum,
    creates individual spectral products for each gti
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/nicerl3-spect/

    Processes a directory using the nicerl3-spect script

    -relax_SAA_bg: options to incease the range of possible values of nxb.saa_norm to a higher value
    (By default this is at a max of 100 as of Heasoft 6.33, but this is clearly insufficient for SAA passages)
    
    bgmodel options:
        -scorpeon_script: uses scorpeon in script mode to create a variable xspec-compatible bg model
        
        -scorpeon_file: uses scorpeon in file mode to produce a static background file

        -scorpeon_all: first creates a spectrum without background, then rerun the task 3 times to create
                       the bg files in static, xspec and pyxspec modes

        -3c50: 3c50 model of Remillar et al., fetches data from the alias_3C50 argument
        
        -sw: Space weather model
        
    specific option for scorpeon_script:
        
        -language: if set to python, the generated scripts are made for Pyxspec instead of standard xspec
                   if set to default, the generated scripts are made for standard xspec

    misc:
        -sp_systematics: Add systematics to the spectra. This is the default option with NICERL3, but can be removed
                         to test things or compare with other people's DR

        -day_mode:      (day, night, both) use day, night or both products
                        (both gti and events are day/night specific)

    -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections

    '''

    io_log=open(directory+'/extract_all_spectral.log','w+')

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8',logfile=io_log if parallel else None)
    
    print('\n\n\nCreating spectral products...')
    
    set_var(bashproc)

    with (no_op_context() if parallel else StdoutTee(directory+'/extract_all_spectral.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_all_spectral.log',buff=1,file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read=sys.stdout
        
        bkg_outlang_str=''
        
        if 'scorpeon' in bkgmodel:
            bkgmodel_str=bkgmodel.split('_')[0]
            bkgmodel_mode=bkgmodel.split('_')[1]
            
            #specific option for script mode
            if bkgmodel_mode=='script':
                if language=='python':
                    bkg_outlang_str='outlang=PYTHON'
                elif language=='default':
                    bkg_outlang_str=''
                elif language!='all':
                    print(
                        'NICER_datared_error: only "python" and "default" is implemented for the language output of scorpeon in script mode')
                    return 'NICER_datared_error: only "python" and "default" is implemented for the language output of scorpeon in script mode'

        else:
            bkgmodel_str=bkgmodel
            bkgmodel_mode='file'
            

        #checking if gti files exist in the folder
        gti_files= np.array([elem for elem in glob.glob(os.path.join(directory,'xti/**'), recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        assert day_mode in ['both','day', 'night'], 'Error: day_mode not among the authorized values (day,night,both)'

        if day_mode!='both':

            gti_files=np.array([elem for elem in gti_files if ('_gti_D' in elem if day_mode=='day' else\
                                                               '_gti_N' in elem if day_mode=='night' else 0)])
        gti_files.sort()

        def extract_single_spectral(gtifile=None):

            '''
            wrapper for individual gti or full obsid spectral computations

            Note: automatically recognize if in day or night mode from the beginning of the gti keyword
            assumes night mode if no gti is provided
            '''

            gti_str='' if gtifile is None else ' gtifile='+gtifile

            if gtifile is not None:
                print('Creating spectral products with gti file '+gtifile)
            else:
                print('Creating spectral products')

            extract_mode='night'
            extract_suffix=''
            if gtifile is not None:
                if 'D' in gtifile.split('_')[-1]:
                    extract_mode='day'
                    print('Using day mode event file...\n')
                    extract_suffix='_day'
                elif 'N' in gtifile.split('_')[-1]:
                    extract_mode='night'
                    print('Using night mode event file...\n')
                else:
                    print('Error: cannot recognize day mode from gti file')
                    raise ValueError

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            relaxed_SAA_bg_str=' bkgconfigs="nxb.saa_norm.max=6000"' if relax_SAA_bg else ''

            systematics_str=' syserrfile=NONE' if not sp_systematics else ''

            #no scorpeon model at first if in scorpeon_all mode to loop the creation of all afterwards
            bkg_full_str=' bkgmodeltype='+('NONE' if bkgmodel=='scorpeon_all' else bkgmodel_str+' bkgformat='+bkgmodel_mode+' '+bkg_outlang_str)

            bashproc.sendline('nicerl3-spect indir='+directory+bkg_full_str+
                              ' clobber='+('YES' if overwrite else 'FALSE')+gti_str+relaxed_SAA_bg_str+systematics_str+
                              ((" clfile='$CLDIR/ni$OBSID_0mpu7_cl_day.evt'  suffix="+extract_suffix) if extract_mode=='day' else ''))

            process_state=bashproc.expect(['DONE','ERROR: could not find UFA file','Task aborting due to zero EXPOSURE',
                                           'Task aborting due to zero response',
                                           'PIL ERROR PIL_UNSPECIFIED_ERROR: non-specific pil error'],timeout=None)

            if process_state in [2,3]:

                #skipping the computation
                return 'skip'

            #raising an error to stop the process if the command has crashed for some reason
            if process_state>0:
                with open(directory+'/extract_all_spectral.log') as file:
                    lines=file.readlines()

                bashproc.sendline('exit')
                if thread is not None:
                    thread.set()
                return lines[-1].replace('\n','')

            #creating all types of scorpeon models afterwards if need be we don't expect bugs since the function
            #would have returned already otherwise
            if bkgmodel=='scorpeon_all':

                bkg_full_str_list=[' bkgmodeltype=scorpeon bkgformat=model outlang=PYTHON',
                                    ' bkgmodeltype=scorpeon bkgformat=model outlang=xcm',
                                    ' bkgmodeltype=scorpeon bkgformat=file']
                for elem_bkg_str in bkg_full_str_list:
                    bashproc.sendline('nicerl3-spect indir=' + directory + elem_bkg_str +
                                      ' clobber=YES' + gti_str + relaxed_SAA_bg_str +
                                      systematics_str +((" clfile='$CLDIR/ni$OBSID_0mpu7_cl_day.evt'  suffix=" +
                                                         extract_suffix) if extract_mode == 'day' else ''))

                    process_state = bashproc.expect(['DONE'], timeout=None)

            allfiles=glob.glob(os.path.join(directory,'xti/**'),recursive=True)

            #fetching the path of the spectrum and rmf file (out of pre-compiled products
            spfile=[elem for elem in allfiles if '_sr'+extract_suffix+'.pha' in elem and '/products/' not in elem]

            if len(spfile)>1:
                print('NICER_datared_error: Several output spectra detected for single computation')
                raise ValueError
            elif len(spfile)==0:
                print('NICER_datared_error: No spectral file detected for single computation')
                raise ValueError
            else:
                spfile=spfile[0]

            #storing the full observation ID for later
            file_id=spfile.split('/')[-1][2:].replace('_sr'+extract_suffix+'.pha','')
            file_dir=spfile[:spfile.rfind('/')]
            file_suffix=file_id.split(directory)[-1]

            copyfile_suffixes=['_sr.pha','_bg.rmf','_sk.arf','.arf','.rmf']+ \
                              (['_bg.py','_bg.pha','_bg.xcm'] if bkgmodel=='scorpeon_all' else [])+\
                            (['_bg.py'] if bkgmodel=='scorpeon_script' and language=='python' else [])+ \
                            (['_bg.pha'] if bkgmodel=='scorpeon_file' else [])+ \
                            (['_bg.xcm'] if bkgmodel == 'scorpeon_script' and language=='default' else [])

            copyfile_list=['ni'+file_id+elem.replace('.',extract_suffix+'.') for elem in copyfile_suffixes]

            #copying the spectral products into the main directory
            for elem_file in copyfile_list:
                os.system('cp '+os.path.join(file_dir,elem_file)+' '+os.path.join(directory,elem_file))

            #renaming all the spectral products
            prod_files=glob.glob(directory+'/ni'+directory+'**',recursive=False)

            for elem in prod_files:
                os.system('mv '+elem+' '+elem.replace('ni','').replace(file_suffix,gti_suffix)\
                          .replace('_day',''))

            #updating the file names in the bg load files
            bg_file_replace=[directory+'/'+directory+gti_suffix+'_bg.py',directory+'/'+directory+gti_suffix+'_bg.xcm']
            for elem_file_bg in bg_file_replace:

                if os.path.isfile(elem_file_bg):

                    with open(elem_file_bg) as old_bgload_file:
                        old_bgload_lines=old_bgload_file.readlines()

                    #removing the file
                    os.remove(elem_file_bg)

                    #and rewritting one with updated variables
                    with open(elem_file_bg,'w+') as new_bgload_file:
                        for line in old_bgload_lines:

                            #for python
                            if line.startswith('nicer_srcrmf'):
                                new_bgload_file.writelines('nicer_srcrmf="'+directory+gti_suffix+'.rmf"\n')
                                continue
                            elif line.startswith('nicer_skyarf'):
                                new_bgload_file.writelines('nicer_skyarf="'+directory+gti_suffix+'_sk.arf"\n')
                                continue
                            elif line.startswith('nicer_diagrmf'):
                                new_bgload_file.writelines('nicer_diagrmf="'+directory+gti_suffix+'_bg.rmf"\n')
                                continue

                            #for xcm
                            if line.startswith('set nicer_srcrmf'):
                                new_bgload_file.writelines('set nicer_srcrmf "'+directory+gti_suffix+'.rmf"\n')
                                continue
                            elif line.startswith('set nicer_skyarf'):
                                new_bgload_file.writelines('set nicer_skyarf "'+directory+gti_suffix+'_sk.arf"\n')
                                continue
                            elif line.startswith('set nicer_diagrmf'):
                                new_bgload_file.writelines('set nicer_diagrmf "'+directory+gti_suffix+'_bg.rmf"\n')
                                continue

                            new_bgload_file.writelines(line)

        if len(gti_files)==0:
            print('no gti files detected. Computing spectral products from the entire obsid...')

        else:
            print(str(len(gti_files))+' gti files detected. Computing spectral products from individual gtis...')

        for elem_gti in gti_files:

            process_state=extract_single_spectral(elem_gti)

            #stopping the loop in case of crash
            if process_state not in [None,'skip']:

                #exiting the bashproc
                bashproc.sendline('exit')
                if thread is not None:
                    thread.set()

                #raising an error to stop the process if the command has crashed for some reason
                return 'GTI '+elem_gti.split('_gti_')[1].replace('.gti','')+': '+process_state


        #exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()

#### extract_lc
def extract_lc(directory,binning_list=[1],bands='3-12',HR='6-10/3-6',overwrite=True,day_mode='both',
               skip_merge=True,thread=None,parallel=False):
    
    '''
    Wrapper for nicerl3-lc, with added matplotlib plotting of requested lightcurves and HRs
        
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/nicerl3-lc/

    Processes a directory using the nicerl3-lc script for every band asked, then creates plots for every lc/HR requested
    
    options:
        -binning: list of binning of LCs to extract, in seconds
        
        -bands: bands for each lightcurve to be created. The numbers should be in keV, separated by "-", and different lightcurves by ","
                ex: to create two lightcurves for, the 1-3 and 4-12 band, use '1-3,4-12'
                
        -hr: bands to be used for the HR plot creation. A single plot is possible for now. Creates its own lightcurve bands if necessary
        
        -overwrite: overwrite products or not

        -day_mode:      (day, night, both) use day, night or both products
                        (both gti and events are day/night specific)

        -skip_merge:
            do not extract the lightcurve of merge products
            (since they are typically big and redundant with
                        individual orbit lightcurves)

    Note: can produce no output without error if no gti in the event file

    -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections
    '''

    io_log=open(directory+'/extract_lc.log','w+')

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8',logfile=io_log if parallel else None)
    
    print('\n\n\nCreating lightcurves products...')
    
    #defining the number of lightcurves to create
    
    #decomposing for each band asked
    lc_bands=([] if HR is None else ravel_ragged([elem.split('/') for elem in HR.split(',')]).tolist())+bands.split(',')
    
    lc_bands=np.unique(lc_bands)[::-1]

    #storing the ids for the HR bands
    if HR is not None:
        id_band_num_HR=np.argwhere(HR.split('/')[0]==lc_bands)[0][0]
        id_band_den_HR=np.argwhere(HR.split('/')[1]==lc_bands)[0][0]
    
    set_var(bashproc)

        
    with (no_op_context() if parallel else StdoutTee(directory+'/extract_lc.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_lc.log',buff=1,file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read=sys.stdout

        #checking if gti files exist in the folder
        gti_files= np.array([elem for elem in glob.glob(os.path.join(directory,'xti/**') , recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        assert day_mode in ['both','day', 'night'], 'Error: day_mode not among the authorized values (day,night,both)'

        if skip_merge:
            gti_files=[elem for elem in gti_files if 'MRG_' not in elem]

        if day_mode!='both':

            gti_files=np.array([elem for elem in gti_files if ('_gti_D' in elem if day_mode=='day' else\
                                                               '_gti_N' in elem if day_mode=='night' else 0)])

        gti_files.sort()
        def extract_single_lc(binning,gtifile=None):

            '''
            wrapper for individual gti or full obsid spectral computations

            extracts the lightcruves in all bands with a single binning

            Note: automatically recognize if in day or night mode from the beginning of the gti keyword
            assumes night mode if no gti is provided
            '''

            gti_str='' if gtifile is None else ' gtifile='+gtifile

            if gtifile is not None:
                print('Creating lightcurve products with gti file '+gtifile)
            else:
                print('Creating lightcurve products products from the whole observation...')

            extract_mode='night'
            extract_suffix=''
            if gtifile is not None:
                if 'D' in gtifile.split('_')[-1]:
                    extract_mode='day'
                    print('Using day mode event file...\n')
                    extract_suffix='_day'
                elif 'N' in gtifile.split('_')[-1]:
                    extract_mode='night'
                    print('Using night mode event file...\n')
                else:
                    print('Error: cannot recognize day mode from gti file')
                    raise ValueError

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            time_zero_arr=np.array([None]*len(lc_bands))

            data_lc_arr=np.array([None]*len(lc_bands))

            #storing the lightcurve
            for i_lc,indiv_band in enumerate(lc_bands):

                old_files_lc=[elem for elem in glob.glob(os.path.join(directory,'xti/**/*'),recursive=True) if elem.endswith('.lc') and 'bin' not in elem]

                for elem_file in old_files_lc:
                    os.remove(elem_file)

                pi_band='-'.join((np.array(indiv_band.split('-')).astype(float)*100).astype(int).astype(str).tolist())

                bashproc.sendline('nicerl3-lc '+directory+' pirange='+pi_band+' timebin='+str(binning)+' '+
                                  ' clobber='+('YES' if overwrite else 'FALSE')+gti_str+
                                  ((" clfile='$CLDIR/ni$OBSID_0mpu7_cl_day.evt'  suffix=" + extract_suffix)
                                   if extract_mode == 'day' else ''))

                #updated for heasoft 6.35
                process_state=bashproc.expect(['DONE','ERROR: could not recompute normalized RATE/ERROR',
                                               'has EXPOSURE=0','Task aborting due','Task nicerl3-lc','Killed'],timeout=None)

                #raising an error to stop the process if the command has crashed for some reason
                if process_state>2:
                    if process_state==4:
                        return 'RAM Overload'

                    with open(directory+'/extract_lc.log') as file:
                        lines=file.readlines()

                    return lines[-1].replace('\n','')

                #this one is for empty lc, which is normal for day gtis of night orbits and vice-versa
                if process_state in [1,2]:

                    #emptying the buffer updated for heasoft 6.35
                    bashproc.expect(['Task nicerl3-lc','Task aborting due to zero EXPOSURE'])

                    #skipping the computation
                    return 'skip'

                #storing some info from the mkf file to get a similar starting time than with the flare lightcurves
                file_mkf = (glob.glob(os.path.join(directory, '**/**.mkf'), recursive=True) + glob.glob(
                    os.path.join(directory, '**/**.mkf.gz'), recursive=True))[0]
                with fits.open(file_mkf) as fits_mkf:
                    # from https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time_resolved_spec/
                    # this value is offset by the mjd_ref value

                    # note that there's an offset of few seconds between this value and the actual 1st column of
                    # the time vector, for some reason ???

                    # note that using leapinit can create a "Dumping CFITSIO error stack", see here:
                    # https: // heasarc.gsfc.nasa.gov / docs / nicer / analysis_threads / common - errors /
                    # so we don't consider the leapinit
                    # start_obs_s=fits_mkf[1].header['TSTART']+fits_mkf[1].header['TIMEZERO']-fits_mkf[1].header['LEAPINIT']

                    start_obs_s = fits_mkf[1].header['TSTART'] + fits_mkf[1].header['TIMEZERO']

                file_lc=[elem for elem in glob.glob(os.path.join(directory,'xti/**/*'),recursive=True) if elem.endswith('.lc') and elem.split('/')[-1].startswith('ni')][0]

                #storing the data of the lc
                with fits.open(file_lc) as fits_lc:

                    #time zero of the lc file (different from the time zeros of the gti files)
                    time_zero=Time(fits_lc[1].header['MJDREFI']+fits_lc[1].header['MJDREFF'],format='mjd')

                    #no leapinit here since we didn't use it to create the gtis
                    #time_zero+=TimeDelta((fits_lc[1].header['TIMEZERO']-fits_lc[1].header['LEAPINIT']),format='sec')

                    #using the same timezero value as the mkf file instead
                    time_zero += TimeDelta((start_obs_s),
                                           format='sec')

                    #and offsetting the data array to match this
                    delta_mkf_lc=fits_lc[1].header['TIMEZERO']-start_obs_s
                    fits_lc[1].data['TIME']+=delta_mkf_lc

                    #storing the shifted lightcurve
                    data_lc_arr[i_lc]=fits_lc[1].data

                    time_zero_arr[i_lc]=str(time_zero.to_datetime())

                    #saving the lc in a different file
                    fits_lc.writeto(file_lc.replace('.lc',gti_suffix+'_'+indiv_band+'_bin_'+str(binning)+'.lc')\
                                    .replace('mpu7_sr','').replace('ni'+directory.replace('/',''),directory.replace('/','')).replace('event_cl/','').replace('_day',''),overwrite=True)

                #removing the direct products
                new_files_lc = [elem for elem in glob.glob(os.path.join(directory, 'xti/**/*'), recursive=True) if
                                '.lc' in elem and 'bin' not in elem]

                #adding a condition to avoid issues if they were remaining temp files that god deleted along the way
                for elem_file in new_files_lc:
                    if os.path.isfile(elem_file):
                        os.remove(elem_file)

                #and plotting it
                fig_lc,ax_lc=plt.subplots(1,figsize=(10,8))

                plt.errorbar(data_lc_arr[i_lc]['TIME'],data_lc_arr[i_lc]['RATE'],xerr=float(binning)/2,yerr=data_lc_arr[i_lc]['ERROR'],ls='-',lw=1,color='grey',ecolor='blue')

                plt.suptitle('NICER lightcurve for observation '+directory+gti_suffix+' in the '+indiv_band+' keV band')

                plt.xlabel('Time (s) after '+time_zero_arr[i_lc])
                plt.ylabel('RATE (counts/s)')

                plt.tight_layout()
                plt.savefig('./'+directory+'/'+directory+gti_suffix+'_lc_'+indiv_band+'_bin_'+str(binning)+'.png')
                plt.close()

            if HR is not None:
                if time_zero_arr[id_band_num_HR]!=time_zero_arr[id_band_den_HR]:
                    print('NICER_datared error: both lightcurve for the HR have different zero values')
                    raise ValueError

                #creating the HR plot
                fig_hr,ax_hr=plt.subplots(1,figsize=(10,8))

                hr_vals=data_lc_arr[id_band_num_HR]['RATE']/data_lc_arr[id_band_den_HR]['RATE']

                hr_err=hr_vals*(((data_lc_arr[id_band_num_HR]['ERROR']/data_lc_arr[id_band_num_HR]['RATE'])**2+
                                (data_lc_arr[id_band_den_HR]['ERROR']/data_lc_arr[id_band_den_HR]['RATE'])**2)**(1/2))

                plt.errorbar(data_lc_arr[id_band_num_HR]['TIME'],hr_vals,xerr=float(binning)/2,yerr=hr_err,ls='-',lw=1,color='grey',ecolor='blue')

                plt.suptitle('NICER HR evolution for observation '+directory+gti_suffix+' in the '+HR+' keV band')

                plt.xlabel('Time (s) after '+time_zero_arr[id_band_num_HR])
                plt.ylabel('Hardness Ratio ('+HR+' keV)')

                plt.tight_layout()
                plt.savefig('./'+directory+'/'+directory+gti_suffix+'_hr_'+'_'.join(HR.split('/'))+'_bin_'+str(binning)+'.png')
                plt.close()

        if len(gti_files)==0:
            print('no gti files detected. Computing lightcurve products from the entire obsid...')

        else:
            print(str(len(gti_files))+' gti files detected. Computing lightcurve products from individual gtis...')

        for elem_gti in gti_files:

            for elem_binning in binning_list:
                process_state=extract_single_lc(elem_binning,gtifile=elem_gti)

                #stopping the loop in case of crash
                if process_state not in [None,'skip']:

                    #exiting the bashproc
                    bashproc.sendline('exit')
                    if thread is not None:
                        thread.set()

                    #raising an error to stop the process if the command has crashed for some reason
                    return 'GTI '+elem_gti.split('_gti_')[1].replace('.gti','')+': '+process_state

        #exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()

def extract_spectrum(directory,thread=None):
    
    '''
    
    DEPRECATED
    
    Extracts spectra using xselect commands
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/spectrum-creation/
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nSpectrum creation...')
    
    set_var(bashproc)
        
    #finding the event file
    evt_path=[elem for elem in glob.glob('**',recursive=True) if directory in elem and 'cl.evt' in elem]
    
    #stopping the process if there is no processed event file
    if len(evt_path)==0:
        raise ValueError
        
    evt_name=evt_path[0].split('/')[-1]
    
    if os.path.isfile(directory+'/extract_spectrum.log'):
        os.system('rm '+directory+'/extract_spectrum.log')
        
    if os.path.isfile(directory+'/'+directory+'_sp.pha'):
        os.remove(directory+'/'+directory+'_sp.pha')
        
    with StdoutTee(directory+'/extract_spectrum.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_spectrum.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout
        
        #starting xselect
        bashproc.sendline('xselect')
        bashproc.expect('session name',timeout=None)
        
        #it is necessary to give a name to the temp files created
        bashproc.sendline('sp')
        
        #setting the directory
        bashproc.sendline('set datadir '+directory+'/xti/event_cl')
        
        #reading the event
        bashproc.sendline('read event '+evt_name)
        
        #which in turn causes xselect to reset the mission
        bashproc.expect('mission',timeout=None)
        bashproc.sendline('yes')
        
        #setting the PI
        bashproc.sendline('set phaname PI')
        bashproc.expect('min and max',timeout=None)
        
        #extracting the Spectrum
        bashproc.sendline('extract Spectrum')
        bashproc.expect('Extension',timeout=None)
        
        #and saving it
        bashproc.sendline('save spectrum '+directory+'/'+directory+'_sp.pha')
        bashproc.expect('Wrote spectrum',timeout=None)
        
        #leaving xselect
        bashproc.sendline('exit')
        bashproc.sendline('no')
        
        #and exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()
def extract_background(directory,model,thread=None):
    
    '''
    
    DEPRECATED
    
    Extracts NICER background using associated commands and models
    
    model:
        -3C50 (needs to be installed before)
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/background/
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nBackground creation...')
    
    set_var(bashproc)

    if os.path.isfile(directory+'/extract_background.log'):
        os.system('rm '+directory+'/extract_background.log')
        
    with StdoutTee(directory+'/extract_background.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_background.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout
        
        if model=='3c50':
            bashproc.sendline('nibackgen3C50 rootdir=./ obsid='+directory+' bkgidxdir='+alias_3C50+' bkglibdir='+alias_3C50+' gainepoch=2019'+
                              ' clobber=YES')
            
            bg_state=bashproc.expect(['written the PHA data','terminating with status','no background estimate'],timeout=None)
            
            if bg_state!=0:
                print('Error while creating the BackGround')
            else:
                bashproc.expect('written the PHA data',timeout=None)
                bashproc.expect('written the PHA data',timeout=None)
            
            #waiting a bit for deletion of the temp files
            time.sleep(1)
            
            #moving the files (as they are created in the current directory, which needs to be above the obsid directory
            bashproc.sendline('mv nibackgen3C50* '+directory)
            
            #and renaming them
            bashproc.sendline('cd '+directory)
            bashproc.sendline('mv nibackgen3C50_bkg.pi '+directory+'_bkg.pi')

            #we don't need to use these currently so we don't rename them
            # bashproc.sendline('mv nibackgen3C50_bkg_ngt.pi '+directory+'_bkg_ngt.pi')
            # bashproc.sendline('mv nibackgen3C50_tot.pi '+directory+'_bkg_tot.pi')
            
            #advancing the display and logging
            bashproc.sendline('echo done')
            bashproc.expect('done')
        
        bashproc.sendline('exit')
        
        if thread is not None:
            thread.set()
def extract_response(directory,thread=None):
    
    '''
    
    DEPRECATED
    
    Extracts NICER rmf and arf using associated heasoft commands
    
    The source position is taken from the header of the spectrum fits file
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/arf-rmf/
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nResponse computation...')
    
    set_var(bashproc)
    
    if os.path.isfile(directory+'/extract_response.log'):
        os.system('rm '+directory+'/extract_response.log')
        
    with StdoutTee(directory+'/extract_response.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_response.log',buff=1,file_filters=[_remove_control_chars]):
            
        bashproc.logfile_read=sys.stdout
        
        #this time it's much easier to go inside the directory to have simpler directory structures
        os.chdir(directory)
        bashproc.sendline('cd '+directory)
        
        sp_path=directory+'_sp.pha'
        
        #fetching the object coordinates from the spectrum file
        with fits.open(sp_path) as hdul:
            obj_ra,obj_dec=hdul[0].header['RA_OBJ'],hdul[0].header['DEC_OBJ']
            
        #fetching the position of the mkf and event files
        subfiles=glob.glob('**',recursive=True)
        mkf_path=[elem for elem in subfiles if elem.endswith('.mkf')][0]
        evt_path=[elem for elem in subfiles if elem.endswith('_cl.evt')][0]
    
        
        #nicer arf command
        bashproc.sendline('nicerarf '+sp_path+' '+str(obj_ra)+' '+str(obj_dec)+' '+mkf_path+' '+evt_path+' '+sp_path.replace('_sp.pha','.arf')+
                          ' outwtfile='+sp_path.replace('_sp.pha','_wt.lis')+' clobber=YES')
        
        #the 'terminating with status' message is only shown when something goes wrong
        bashproc.expect(['DONE','terminating with status'],timeout=None)
        
        #nicer rmf command
        bashproc.sendline('nicerrmf '+sp_path+' '+mkf_path+' '+sp_path.replace('_sp.pha','.rmf')+
                          ' detlist=@'+sp_path.replace('_sp.pha','_wt.lis')+' clobber=YES')
    
        bashproc.expect(['DONE','terminating with status'],timeout=None)
        
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()
def regroup_spectral(directory,group='opt',thread=None,parallel=False):
    
    '''
    Regroups NICER spectrum from an obsid directory using ftgrouppha
    
    mode:
        -opt: follows the Kastra and al. 2016 binning
        
    Currently only accepts input from extract_all_spectral

    -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections
    '''

    io_log=open(directory+'/regroup_spectral.log','w+')

    currdir = os.getcwd()

    print('\n\n\nRegrouping spectrum...')

    #deleting previously existing grouped spectra to avoid problems when testing their existence
    if os.path.isfile(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha')):
        os.remove(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha'))

    with (no_op_context() if parallel else StdoutTee(directory+'/regroup_spectral.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/regroup_spectral.log',buff=1,file_filters=[_remove_control_chars])):

        # there seems to be an issue with too many groupings in one console so we recreate it every time

        #checking if gti files exist in the folder
        gti_files= np.array([elem for elem in glob.glob(os.path.join(directory ,'xti/**/*'), recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        gti_files.sort()

        def regroup_single_spectral(spawn,gtifile=None):

            if gtifile is not None:
                print('Regrouping spectral products with gti file '+gtifile)
            else:
                print('Regrouping spectral products')

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            #raising an error to stop the process if the command has crashed for some reason
            if not os.path.isfile(directory+'/'+directory+gti_suffix+'_sr.pha'):
                return 'Source spectrum missing'

            allfiles=glob.glob(os.path.join(directory,'xti/**'),recursive=True)

            #print for saving in the log file since it doesn't show clearly otherwise
            print('ftgrouppha infile='+directory+'/'+directory+gti_suffix+'_sr.pha'+' outfile='+directory+'/'+directory+gti_suffix+'_sp_grp_'+group+
            '.pha grouptype='+group+' respfile='+directory+'/'+directory+gti_suffix+'.rmf')

            spawn.sendline('ftgrouppha infile='+directory+'/'+directory+gti_suffix+'_sr.pha'+' outfile='+directory+'/'+directory+gti_suffix+'_sp_grp_'+group+
            '.pha grouptype='+group+' respfile='+directory+'/'+directory+gti_suffix+'.rmf')

            time.sleep(2)

            if not os.path.isfile(os.path.join(currdir,directory+'/'+directory+gti_suffix+'_sp_grp_'+group+'.pha')):
                print('Waiting for creation of file '+os.path.join(currdir,
                      directory+'/'+directory+gti_suffix+'_sp_grp_'+group+'.pha'))
                time.sleep(5)
                spawn.sendline('echo done')
                process_state = spawn.expect(['done', 'terminating with status -1'], timeout=60)

                assert process_state==0, 'Issue when regrouping'
            else:
                spawn.sendline('echo done')
                spawn.expect('done')

            time.sleep(1)

            #updating the grouped file header with the correct file names
            with fits.open(directory+'/'+directory+gti_suffix+'_sp_grp_'+group+'.pha',mode='update') as hdul:
                hdul[1].header['RESPFILE']=directory+gti_suffix+'.rmf'
                hdul[1].header['ANCRFILE']=directory+gti_suffix+'.arf'
                #saving changes
                hdul.flush()
            
        if len(gti_files)==0:
            print('no gti files detected. Regrouping spectral products from the entire obsid...')

        else:
            print(str(len(gti_files))+' gti files detected. Regrouping spectral products from individual gtis...')

        for elem_gti in gti_files:

            if not os.path.isfile(elem_gti.replace('/xti','').replace('_gti_','-').replace('.gti','_sr.pha')):
                print('\nNo spectrum created for gti '+elem_gti+'. Continuing...\n')
                continue

            bashproc = pexpect.spawn("/bin/bash", encoding='utf-8',logfile=io_log if parallel else None)

            set_var(bashproc)

            if not parallel:
                bashproc.logfile_read = sys.stdout

            process_state=regroup_single_spectral(bashproc,elem_gti)

            #stopping the loop in case of crash
            if process_state is not None:

                #exiting the bashproc
                bashproc.sendline('exit')
                if thread is not None:
                    thread.set()
                #raising an error to stop the process if the command has crashed for some reason
                return 'GTI '+elem_gti.split('_gti_')[1].replace('.gti','')+': '+process_state

            #exiting the bashproc
            bashproc.sendline('exit')
        if thread is not None:
            thread.set()
def batch_mover(directory,thread=None,parallel=False):
    
    '''
    copies all spectral products in a directory to a bigbatch directory above the obsid directory to prepare for spectrum analysis

        -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections
    '''

    io_log=open(directory+'/batch_mover.log','w+')

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8',logfile=io_log if parallel else None)

    if not parallel:
        bashproc.logfile_read=sys.stdout
    
    print('\n\n\nCopying spectral products to a merging directory...')
    
    set_var(bashproc)
    
    bashproc.sendline('mkdir -p bigbatch')
    
    bashproc.sendline('cd '+directory)

    #number of files to copy
    n_files_copy=len(glob.glob(os.path.join(directory,directory)+'**'))

    bashproc.sendline('cp --verbose '+directory+'* ../bigbatch'+' >batch_mover_list.log')

    #checking the log file
    copy_ok=False
    while not copy_ok:
        time.sleep(0.5)
        with open(os.path.join(directory,'batch_mover_list.log')) as f:
            n_lines=len(f.readlines())
        if n_lines==n_files_copy:
            copy_ok=1
        else:
            print('waiting for copy...')

    #reasonable waiting time to make sure files can be copied
    time.sleep(0.5)
    
    bashproc.sendline('exit')
    if thread is not None:
        thread.set()


def batch_mover_timing(directory,thread=None,parallel=False):

    '''
    copies all lc products in a directory to a lcbatch directory above the obsid directory to prepare for lc analysis

    -parallel: bool:tells the function it's running in a parallel configuration.
           Modifies the logging to avoid issues with redirections

    '''

    io_log=open(directory+'/batch_mover_timing.log','w+')

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8',logfile=io_log if parallel else None)

    if not parallel:
        bashproc.logfile_read = sys.stdout

    print('\n\n\nCopying lc products to a merging directory...')

    set_var(bashproc)

    bashproc.sendline('mkdir -p lcbatch')

    bashproc.sendline('cd ' + directory)

    #number of files to copy
    n_files_copy=len(glob.glob(os.path.join(directory,'xti','**.lc')))

    bashproc.sendline('cp --verbose ./xti/'+directory+'*.lc ../lcbatch' + ' >batch_mover_timing_list.log')

    #checking the log file
    copy_ok=False
    while not copy_ok:
        time.sleep(0.5)
        with open(os.path.join(directory,'batch_mover_timing_list.log')) as f:
            n_lines=len(f.readlines())
        if n_lines==n_files_copy:
            copy_ok=1
        else:
            print('waiting for copy...')

    # reasonable waiting time to make sure files can be copied
    time.sleep(0.5)

    bashproc.sendline('exit')
    if thread is not None:
        thread.set()

def clean_products(directory,clean_wait_value=2,thread=None):

    '''

    clean products in the xti/event_cl directory

    Useful to avoid bloating with how big these files are

    Note: we keep the cleaned event file and nimaketime because these are very useful when checking
    '''

    product_files=[elem for elem in glob.glob(os.path.join(directory,'xti/event_cl/**'),recursive=True)\
                   if not elem.endswith('/') and not elem.endswith('nimaketime.gti') and not elem.endswith('mpu7_cl.evt')\
                   and not elem.endswith('mpu7_cl_day.evt')]

    print('Cleaning '+str(len(product_files))+' elements in directory '+os.path.join(directory,'xti/event_cl/'))

    for elem_product in product_files:
        if not elem_product.endswith('/'):
            os.remove(elem_product)

    #reasonable waiting time to make sure big files can be deleted
    time.sleep(clean_wait_value)

    print('Cleaning complete.')

    if thread is not None:
        thread.set()

def update_bg():
    '''
    Function to clean bg issues in a previous version of the code
    '''

    bg_files=glob.glob('**_bg.py')
    for elem in bg_files:
        print('updating file '+elem)
        with open(elem) as f:
            f_lines=f.readlines()
        with open(elem,'w+') as f:
            for i_line,line in enumerate(f_lines):
                if line.startswith('nicer_srcrmf') and f_lines[i_line-1].startswith('nicer_srcrmf'):
                    continue
                if line.startswith('nicer_skyarf') and f_lines[i_line-1].startswith('nicer_skyarf'):
                    continue
                if line.startswith('nicer_diagrmf') and f_lines[i_line-1].startswith('nicer_diagrmf'):
                    continue
                f.write(line)


def clean_all(directory,clean_wait_value=2,thread=None):

    '''

    clean products in the xti/event_cl directory and main directory

    Useful to avoid bloating with how big these files are
    '''

    product_files=[elem for elem in glob.glob(os.path.join(directory,'xti/event_cl/**'),recursive=True)\
                   if not os.path.isdir(elem)]

    other_files=[elem for elem in glob.glob(os.path.join(directory,'**'),recursive=False)\
                   if not os.path.isdir(elem)]+[elem for elem in glob.glob(os.path.join(directory,'xti/**'),recursive=False)\
                   if not os.path.isdir(elem)]

    clean_files=product_files+other_files

    print('Cleaning ' + str(len(clean_files)) + ' elements in directory ' + directory)

    if len(clean_files)>0:

        for elem_product in clean_files:
            if not elem_product.endswith('/'):
                os.remove(elem_product)

        #reasonable waiting time to make sure big files can be deleted
        time.sleep(clean_wait_value)

        print('Cleaning complete.')

    if thread is not None:
        thread.set()
'''''''''''''''''''''
''''MAIN PROCESS'''''
'''''''''''''''''''''

#### MAIN PROCESS

#getting started by choosing the correct directory and storing it for future use
if not os.path.isdir(startdir):
    logging.error('Invalid directory, please start over.')
    sys.exit()
else:
    os.chdir(startdir)
    
startdir=os.getcwd()

#listing all of the subdirectories
subdirs=glob.glob('**/',recursive=True)
subdirs.sort()

# getting the last directory in the result (necessary if osbid folders are in subdirectories) (we take the last / of to avoid problems)

subdirs_obsid=[elem_dir for elem_dir in subdirs if len(elem_dir[:-1].split('/')[-1]) == 10\
                                                and elem_dir[:-1].split('/')[-1].isdigit()]


#summary header for the previously computed directories file
summary_folder_header='Subdirectory\tAnalysis state\n'

def startdir_state(action):
    #fetching the previously computed directories from the summary folder file
    try:
        with open('summary_folder_analysis_'+action+'.log') as summary_folder:
            launched_folders=summary_folder.readlines()
    
            #restricting to folders with completed analysis
            completed_folders=[elem.split('\t')[0] for elem in launched_folders if 'Done' in elem]
            launched_folders=[elem.split('\t')[0] for elem in launched_folders]
    except:
        launched_folders=[]
        completed_folders=[]
        
    return launched_folders,completed_folders

if load_functions:
    breakpoint()

def run_actions(obs_dir,action_list,parallel=False):

    '''
    Wrapper to run a list of actions in a single directory

    obs_dir: directory where to run the actions

    action_list: list of action strings

    parallel: boolean, will set all individual action parallel to avoid issues with logging if running in parallel

    '''

    # we insure each action will wait for the completion of the previous one by using threads
    process_obsdir_thread = threading.Event()
    extract_all_spectral_thread = threading.Event()
    extract_spectrum_thread = threading.Event()
    extract_response_thread = threading.Event()
    extract_background_thread = threading.Event()
    regroup_spectral_thread = threading.Event()
    batch_mover_thread = threading.Event()

    select_detector_thread = threading.Event()

    extract_lc_thread = threading.Event()
    clean_products_thread = threading.Event()
    clean_all_thread = threading.Event()

    create_gtis_thread = threading.Event()

    batch_mover_timing_thread = threading.Event()

    # process_obsdir_thread = None
    # extract_all_spectral_thread = None
    # extract_spectrum_thread = None
    # extract_response_thread = None
    # extract_background_thread = None
    # regroup_spectral_thread = None
    # batch_mover_thread = None
    #
    # select_detector_thread = None
    #
    # extract_lc_thread = None
    # clean_products_thread = None
    # clean_all_thread = None
    #
    # create_gtis_thread = None
    #
    # batch_mover_timing_thread = None

    started_folders, done_folders = startdir_state(args.action)

    # continue check
    if obs_dir in started_folders[:-2] and folder_cont:
        print('Directory ' + obs_dir + ' is not among the last two directories. Skipping...')
        return

    # directory overwrite check
    if obs_dir in done_folders and folder_over == False:
        print('Actions already computed for directory ' + obs_dir + '\nSkipping...')
        return

    # getting the last directory in the result (necessary if osbid folders are in subdirectories) (we take the last / of to avoid problems)
    dirname = obs_dir[:-1].split('/')[-1]


    print('\nFound obsid directory ' + obs_dir)

    above_obsdir = os.path.join(startdir, '/'.join(obs_dir[-1].split('/')[:-1]))

    os.chdir(above_obsdir)

    def action_loop(action_list,output,output_err):

        # for loop to be able to use different orders if needed
        for curr_action in action_list:

            # resetting the error string message
            output_err[0] = None
            output[0]= 'Running ' + curr_action

            if curr_action == '1':
                print(os.getcwd())

                process_obsdir(dirname, overwrite=overwrite_glob, keep_SAA=keep_SAA,
                               overshoot_limit=overshoot_limit,
                               undershoot_limit=undershoot_limit,
                               min_gti=min_gti, erodedilate=erodedilate, keep_lowmem=keep_lowmem,
                               br_earth_min=br_earth_min, day_mode=day_mode, thread=process_obsdir_thread,
                               parallel=parallel)
                process_obsdir_thread.wait()

            if curr_action == '2':
                select_detector(dirname, detectors=bad_detectors, thread=select_detector_thread,parallel=parallel)
                select_detector_thread.wait()

            if curr_action == 'gti':
                output_err[0] = create_gtis(dirname, split_arg=gti_split, band=gti_lc_band,
                                         flare_method=flare_method,
                                         int_split_band=int_split_band, int_split_bin=int_split_bin,
                                         flare_factor=flare_factor,
                                         clip_sigma=clip_sigma,
                                         peak_score_thresh=peak_score_thresh,
                                         gti_tool=gti_tool, erodedilate_overdyn=erodedilate_overdyn,
                                         overdyn_thresh=overdyn_thresh,
                                         overdyn_factor_low=overdyn_factor_low,
                                         overdyn_factor_high=overdyn_factor_high,
                                         underdyn_method=underdyn_method,
                                         underdyn_jump_width=underdyn_jump_width,underdyn_jump_factor=underdyn_jump_factor,
                                         hard_flare_segments=hard_flare_segments,hard_flare_sigma=hard_flare_sigma,
                                         erodedilate_hard_flare=erodedilate_hard_flare,
                                         hard_flare_min_duration=hard_flare_min_duration,
                                         HR_threshold=gti_HR_threshold,erodedilate_HR=erodedilate_HR,
                                         day_mode=day_mode, thread=create_gtis_thread,
                                         add_merge_gti=add_merge_gti,parallel=parallel)
                if type(output_err[0]) == str:
                    if catch_errors:
                        raise ValueError
                    else:
                        output[0] = output_err

                create_gtis_thread.wait()

            if curr_action == 'fs':
                output_err[0] = extract_all_spectral(dirname, bkgmodel=bgmodel,
                                                  language=bglanguage, overwrite=overwrite_glob,
                                                  relax_SAA_bg=relax_SAA_bg,
                                                  sp_systematics=sp_systematics, day_mode=day_mode,
                                                  thread=extract_all_spectral_thread,parallel=parallel)
                if type(output_err[0]) == str:
                    if catch_errors:
                        raise ValueError
                    else:
                        output[0] = output_err

                extract_all_spectral_thread.wait()

            if curr_action == 'l':
                output_err[0] = extract_lc(dirname, binning_list=lc_bin_list if 'intensity' \
                                                                             not in gti_split else [
                    int_split_bin],skip_merge=skip_merge_lc,
                                        bands=lc_bands_str, HR=hr_bands_str, overwrite=overwrite_glob,
                                        day_mode=day_mode, thread=extract_lc_thread,parallel=parallel)
                if type(output_err[0]) == str:
                    if catch_errors:
                        raise ValueError
                    else:
                        output[0] = output_err

                extract_lc_thread.wait()

            if curr_action == 's':
                extract_spectrum(dirname, thread=extract_spectrum_thread)
                extract_spectrum_thread.wait()

            if curr_action == 'b':
                extract_background(dirname, model=bgmodel, thread=extract_background_thread)
                extract_background_thread.wait()
            if curr_action == 'r':
                extract_response(dirname, thread=extract_response_thread)
                extract_response_thread.wait()

            if curr_action == 'g':
                output_err[0] = regroup_spectral(dirname, group=grouptype, thread=regroup_spectral_thread,parallel=parallel)
                if type(output_err[0]) == str:
                    if catch_errors:
                        raise ValueError
                    else:
                        output[0] = output_err

                regroup_spectral_thread.wait()

            if curr_action == 'm':
                batch_mover(dirname, thread=batch_mover_thread,parallel=parallel)
                batch_mover_thread.wait()

            if curr_action == 'ml':
                batch_mover_timing(dirname, thread=batch_mover_timing_thread,parallel=parallel)
                batch_mover_timing_thread.wait()

            if curr_action == 'c':
                clean_products(dirname, clean_wait_value=clean_wait_value, thread=clean_products_thread)
                clean_products_thread.wait()

            if curr_action == 'fc':
                clean_all(dirname, clean_wait_value=clean_wait_value, thread=clean_all_thread)
                clean_all_thread.wait()

            os.chdir(startdir)
        output[0]= 'Done'

    #just created for passing pointers and avoid issues with return and the try
    output_list=[None]
    output_err_list=[None]

    if catch_errors:

        try:
            action_loop(action_list,output=output_list,output_err=output_err_list)
            folder_state=output_list[0]

        except:
            # signaling unknown errors if they happened
            if 'Running' in output_list[0]:
                print('\nError while ' + output_list[0])
                folder_state = output_list[0].replace('Running', 'Aborted at') + (
                    '' if output_err_list[0] is None else ' --> ' + output_err_list[0])
            os.chdir(startdir)

    else:
        action_loop(action_list, output=output_list, output_err=output_err_list)
        folder_state=output_list[0]

    # adding the directory to the list of already computed directories
    file_edit('summary_folder_analysis_' + args.action + '.log', obs_dir,
              obs_dir + '\t' + folder_state + '\n', summary_folder_header)
    print( obs_dir + '\t' + folder_state + '\n', summary_folder_header)


if not local:
    #checking them in search for ODF directories

    if parallel!=1:

        res = Parallel(n_jobs=parallel)(
            delayed(run_actions)(
                obs_dir=elem_directory,
                action_list=action_list,parallel=True)

            for elem_directory in subdirs_obsid)

    else:

        if invert_subdirs:
            subdirs_obsid=subdirs_obsid[::-1]
        for directory in subdirs_obsid:
            run_actions(directory, action_list,parallel=False)


# if local:
#
#     #HEAVILY OUTDATED
#     assert True,'local mode currently outdated'
#     #taking of the merge action if local is set since there is no point to merge in local (the batch directory acts as merge)
#     action_list=[elem for elem in action_list if elem in ['m','ml']]
#
#     absdir=os.getcwd()
#
#     #just to avoid an error but not used since there is not merging in local
#     obsid=''
#
#     #for loop to be able to use different orders if needed
#     for curr_action in action_list:
#             if curr_action=='1':
#                 process_obsdir(absdir,overwrite=overwrite_glob,keep_SAA=keep_SAA,overshoot_limit=overshoot_limit,
#                                         undershoot_limit=undershoot_limit,
#                                min_gti=min_gti,erodedilate=erodedilate,keep_lowmem=keep_lowmem,
#                                br_earth_min=br_earth_min,day_mode=day_mode)
#                 process_obsdir_done.wait()
#             if curr_action=='2':
#                 select_detector(absdir,detectors=bad_detectors)
#                 select_detector_done.wait()
#
#             if curr_action=='gti':
#                 output_err = create_gtis(absdir, split_arg=gti_split, band=gti_lc_band,
#                                          flare_method=flare_method,
#                                         int_split_band=int_split_band,int_split_bin=int_split_bin,
#                                          flare_factor=flare_factor,
#                                          clip_sigma=clip_sigma,peak_score_thresh=peak_score_thresh,
#                                                    gti_tool=gti_tool,erodedilate_overdyn=erodedilate_overdyn,
#                                          day_mode=day_mode)
#                 create_gtis_done.wait()
#
#             if curr_action == 'fs':
#                 output_err = extract_all_spectral(absdir, bkgmodel=bgmodel, language=bglanguage,
#                                                   overwrite=overwrite_glob,
#                                                   relax_SAA_bg=relax_SAA_bg,
#                                                   sp_systematics=sp_systematics, day_mode=day_mode)
#                 if type(output_err) == str:
#                     raise ValueError
#                 extract_all_spectral_done.wait()
#
#             if curr_action == 'l':
#                 output_err = extract_lc(absdir, binning_list=lc_bin_list if 'intensity' \
#                                         not in gti_split else [int_split_bin],
#                                         bands=lc_bands_str, HR=hr_bands_str, overwrite=overwrite_glob,
#                                         day_mode=day_mode)
#                 if type(output_err) == str:
#                     raise ValueError
#                 extract_lc_done.wait()
#
#             if curr_action=='s':
#                 extract_spectrum(absdir)
#                 extract_spectrum_done.wait()
#             if curr_action=='b':
#                 extract_background(absdir,model=bgmodel)
#                 extract_background_done.wait()
#             if curr_action=='r':
#                 extract_response(absdir)
#                 extract_response_done.wait()
#             if curr_action=='g':
#                 regroup_spectral(absdir,group=grouptype)
#                 regroup_spectral_done.wait()
#             if curr_action=='m':
#                 batch_mover(absdir)
#                 batch_mover_done.wait()
#
#             if curr_action == 'ml':
#                 batch_mover_timing(absdir)
#                 batch_mover_timing_done.wait()
#
#             if curr_action == 'c':
#                 clean_products(absdir,clean_wait_value=clean_wait_value)
#                 clean_products_done.wait()
#
#             if curr_action == 'fc':
#                 clean_all(absdir,clean_wait_value=clean_wait_value)
#                 clean_all_done.wait()

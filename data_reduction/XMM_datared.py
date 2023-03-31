#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Thu Oct  7 16:11:23 2021
@author: Maxime Parra

Data reduction Script for XMM ODFs

Searches for all ODF type directories in the subdirectories and launches the process for each

list of possible actions : 
    
1. evt_build: IF not events are detected, builds the event lists for all cameras in specific subdirectories with em/epproc

2. filter_evt: Filters any detected event files with manual input or standard flare limits. 
   Note: Creates products with standardised names
   submodes : 2n : mode 'nolim', no flare cut (max fixed at 1000 counts/s)
              2a : mode 'auto', flare cuts according to src/bg 
              2std : mode 'std', flare cuts according to the std norms (given in argument)
              2snr : mode 'snr', flare cuts according to the best snr improvement (if at all)
3. extract_reg: Extracts regions for the source/bg, performs pile-up computations. 

sp. extract_sp: Extracts the spectrum with either manual region selection of automatic computation. 
               Also regroups the spectra after extraction.
               The products are copied into a "batch" directory

lc.extract_lc: Extracts lightcurves with either manual region selection or automatically from the previous extract_sp products.
              The products are copied into a "batch" directory

The actions can be listed in any order (although they might not work if you do not use a correct order)

ex: action='3,1,2' -> for each detected ODF directory, does extract_reg, then evt_build, then filter_evt

Secondary actions:

c. counts the Evt files in the entirety of the subdirectory structure.
   Prints some data on the number of Evts for each category, and also displays the paths of all "unfinished" Evts

m. merge all of the spectra of the batches directory in a single one

d. delete all of the previously created products in the subdirectories except the results of evt_build
    (i.e. anything in outputmos, outputpn, batch and bigbatch folders that is not directly created by epproc and emproc)

D. Same but also delete the epproc/emproc products

-------------------------------------
The data reduction is done through external (pexpect) calls of the bash commands

The log is both printed on terminal and copied into the log files of each main function

Other arguments to be added

CAREFUL : 

-cleaning the products (i.e. with -d) before relaunching overlapping/sequential actions in a directory is advised to avoid problems

-the overwrite option functions differently depending on if "merge" is in the actions or not.
If it is not, overwrite checks if the products are in the local directory of the exposure
If it is, overwrite checks if the products are in the global "mergedir" obsid

-Don't use directories with '_EMOS' or 'E_PN' in them
-the evt list which can be passed through with the -evtname argument must still contain _EMOS or _EPN
-The event list filtering caps at 12 keV, as such the spectrum will also be caped at 12 KeV max even though 
 the full channel interval is being used

-The algorithm's spectrum computation is tailored to point sources

-For crowded fields, it can be useful to set the bigger_fit argument to False to avoid a source mismatch, and to reduce the rad_crop value
-Elsewhere, keeping it on True allows better determination of overexposed PSFs
'''

#general imports
import os,sys
import subprocess
import pexpect
import argparse
import logging
import glob
import threading
import time
from tee import StdoutTee,StderrTee
import shutil
import warnings
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from matplotlib.collections import LineCollection

#astro imports
from astropy.io import fits
from astroquery.simbad import Simbad
from mpdaf.obj import sexa2deg,Image
from mpdaf.obj import WCS as mpdaf_WCS
from astropy.wcs import WCS as astroWCS
#from mpdaf.obj import deg2sexa

#image processing imports:
#mask to polygon conversion
from imantics import Mask

#point of inaccessibility
from polylabel import polylabel

#alphashape
from alphashape import alphashape

#polygon filling to mask
from rasterio.features import rasterize

#shape merging
from scipy.ndimage import binary_dilation

from general_tools import file_edit

#to visualise the alphashape :
# from descartes import PolygonPatch

#pileup import
# from line_detection import pileup_val,file_edit

#better errors : to test
# import pretty_errors
# pretty_errors.configure(
#     separator_character = '*',
#     filename_display    = pretty_errors.FILENAME_EXTENDED,
#     line_number_first   = True,
#     display_link        = True,
#     lines_before        = 5,
#     lines_after         = 2,
#     line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
#     code_color          = '  ' + pretty_errors.default_config.line_color,
#     truncate_code       = True,
#     display_locals      = True
# )

'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce XMM files.\n)')

#the basics
ap.add_argument("-dir", "--startdir", nargs='?', help="starting directory. Current by default", default='./', type=str)
ap.add_argument("-evtname",nargs='?',help='substring present in previously processed epproc and emproc event lists',
                default='Evts',type=str)


#global choices
ap.add_argument("-a","--action",nargs='?',help='Give which action(s) to proceed,separated by comas.'+
                '\n1.evt_build\n2.filter_evt\n3.extract_reg...',default='2std,3,l,s,m',type=str)

#std : '2n,3,l,s,m'

ap.add_argument("-c","--cameras",nargs='?',help='Cameras to reduce',default='all',type=str)
ap.add_argument("-e","--expmode",nargs=1,help='restrict the analysis to a single type of exposure (in caps)',default='all',type=str)
ap.add_argument("-l","--local",nargs=1,help='Launch actions directly in the current directory instead',
                default=False,type=bool)

#directory level overwrite (not active in local)
ap.add_argument('-folder_over',nargs=1,help='relaunch action through folders with completed analysis',default=False,type=bool)
ap.add_argument('-folder_cont',nargs=1,help='skip all but the last 2 directories in the summary folder file',default=False,type=bool)
    
#action specific overwrite
ap.add_argument("-over",nargs=1,help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder',default=True,type=bool)

#action specifics : 
'''event cleaning'''
ap.add_argument('-pnflare',nargs=1,help='pn flaring limit standard value',default=0.40,type=float)
ap.add_argument('-mosflare',nargs=1,help='mos flaring limit standard value',default=0.35,type=float)

ap.add_argument('-flareband',nargs=1,help='flare computation band',default='6.-10.',type=str)
#Should correspond to the most important energy band for subsequent science analysis. also used in the region computation

'''region computation'''

ap.add_argument("-mainfocus",nargs=1,help='only extracts spectra when the source is the main focus of the observation',
                default=False,type=bool)

ap.add_argument("-p", "--pileup_ctrl",nargs=1,
                help='mitigates pile-up (if there is any) with a progressive excision of the central region',
                default=True,type=str)
ap.add_argument('-p_tresh','--pileup_treshold',nargs=1,
                help='maximum acceptable pile-up value to stop the excision. Replaces other pile_up controls.'+
                ' "None" shuts off this method',default=0.05)
'''TIMING'''

ap.add_argument('-timing_check',nargs=1,help="For timing mode exposures, stops the computation if the distance between the identified analysis object and the observation's average is >30 arcsecs",default=True,type=bool)

'''IMAGING'''
ap.add_argument('-rad_crop',nargs=1,
                help='croppind radius around the theoretical source position before fit, in arcsecs',default=60,type=float)

ap.add_argument('-bigger_fit',nargs=1,help='allows to incease the crop window used before the gaussian fit for bright sources',
                default=True,type=bool)

ap.add_argument('-point_source',nargs=1,help="assume the source is point-like, I.E. fixes the gaussian's initial center to the brightest pixel",default=True,type=bool)
#helps to avoid the gaussian center shifting in case of diffuse emission

#if equal to crop, is set to rad_crop
ap.add_argument('-maxrad_source',nargs=1,help='maximum source radius for faint sources in units of PSF sigmas',default=6,type=float)

ap.add_argument('-pileup_max_ex',nargs=1,help='pileup maximum excision for imaging sources in units of PSF sigmas',default=3,type=float)
#only used when p_tresh is set to None

'''spectrum extraction''' 


'''lighctuve extraction'''
ap.add_argument('-bin',nargs=1,help='lightcurve binning in s',default=100,type=float)

'''merge'''
ap.add_argument("-mdir","--mergedir",nargs=1,help='directory name for the merging action',default='bigbatch',type=str)
ap.add_argument('-gm_action',nargs=1,help='action to fetch the summary from when using gm',
                default='2n,3,l,s,m',type=str)

'''
~~~~~~~~~~ thoughts ~~~~~~~~~~
To add:

-add choice for manual ds9 region input (3 choices in total)
-add imaging background for mos timing spectrum extraction (see SAS thread)

later: 
-add energy limits for arguments
-fix the savepng incomplete masking bug
-improve imaging background definition by using mpdaf (masked arrays) instead of masking manually, and a check to make sure the source and bg region do not overlap
'''

args=ap.parse_args()

evtname=args.evtname
startdir=args.startdir
action_list=args.action.split(',')
cameras_glob=args.cameras
local=args.local
expos_mode_glob=args.expmode
overwrite_glob=args.over
target_only=args.mainfocus
pileup_ctrl=args.pileup_ctrl
mergedir=args.mergedir
pnflare_lim=args.pnflare
mosflare_lim=args.mosflare
lc_bins=args.bin
flare_band=np.array(args.flareband.split('-')).astype(float)
bigger_fit=args.bigger_fit
rad_crop_arg=args.rad_crop
folder_over=args.folder_over
folder_cont=args.folder_cont
gm_action=args.gm_action
timing_check=args.timing_check
point_source=args.point_source
maxrad_source=args.maxrad_source
pileup_max_ex=args.pileup_max_ex
pileup_treshold=args.pileup_treshold

'''''''''''''''''
''''FUNCTIONS''''
'''''''''''''''''

#switching off matplotlib plot displays unless with plt.show()
plt.ioff()

camlist=['pn','mos1','mos2']

epproc_done=threading.Event()
emproc_done=threading.Event()
evt_build_done=threading.Event()
filter_evt_done=threading.Event()
extract_reg_done=threading.Event()
extract_sp_done=threading.Event()
extract_lc_done=threading.Event()

#function to remove (most) control chars
def _remove_control_chars(message):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)
            
def pileup_val(pileup_line):
    
    '''
    returns the maximal pileup value if there is pile-up, and 0 if there isn't
    '''
    
    #the errors are given by default with a 3 sigma confidence level
    
    pattern_s_val=float(pileup_line.split('s: ')[1].split('   ')[0].split(' ')[0])
    pattern_s_err=float(pileup_line.split('s: ')[1].split('   ')[0].split(' ')[2])
    pattern_d_val=float(pileup_line.split('d: ')[1].split('   ')[0].split(' ')[0])
    pattern_d_err=float(pileup_line.split('d: ')[1].split('   ')[0].split(' ')[2])
    
    #no pileup means the s and d pattern values are compatible with 1
    #however for d the value can be much lower for faint sources, as such we only test positives values for d
    max_pileup_s=max(max(pattern_s_val-pattern_s_err-1,0),max(1-pattern_s_val-pattern_s_err,0))
    max_pileup_d=max(pattern_d_val-pattern_d_err-1,0)
    
    return max(max_pileup_s,max_pileup_d)

def set_var(spawn,directory):
    
    '''
    Sets ODF and CCF environment variables
    '''
    spawn.sendline('heainit')
    spawn.sendline('sasinit')
    spawn.sendline('export SAS_ODF='+directory)
    if directory[-1]!='/':
        supp='/'
    else:
        supp=''
    spawn.sendline('export SAS_CCF='+directory+supp+'ccf.cif')
    spawn.expect('ccf.cif')

def evt_build(directory):

    '''
    Builds the event files in subdirectories if they do not exist yet
    '''

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    bashproc.sendline('cd '+directory)
    
    #setting up a logfile in parallel to terminal display :
    if os.path.isfile('evt_build.log'):
        os.system('rm evt_build.log')
    with StdoutTee('evt_build.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee('evt_build.log',buff=1,file_filters=[_remove_control_chars]):
            
        bashproc.logfile_read=sys.stdout
        
        #setting up the environment variables
        set_var(bashproc,directory)
        
        #searching recursively for the files in the directory
        filelist=glob.glob('**',recursive=True)
        
        # testing if there are actual event files in the directory
        if len([elem for elem in filelist if (elem.endswith('.ASC') or elem.endswith('.SAS') or elem.endswith('.FIT'))])==0:
            bashproc.sendline('\nexit')
            bashproc.expect(pexpect.EOF,timeout=None)
        
            evt_build_done.set()
            
            return 'empty directory'
        
        #here it is assumed that if there are no event list, odfingest and cifbuilds have not been run either.
        evt_list=[elem for elem in filelist if 'Evts' in elem or evtname in elem]
        build_MOS=False
        build_PN=False
        
        #simple ways of searching through the event list
        if '_EMOS' not in str(evt_list):
            print('\nNo MOS evt list detected.')
            build_MOS=True
        if '_EPN' not in str(evt_list):
            print('\nNo PN evt list detected.')
            build_PN=True
        
        #if both of the event lists are missing, it is assumed that nothing has been done on the files yet.
        if build_PN and build_MOS:
            print('\nRunning odfingest and cifbuild first : ')

            #here we lock the following commands by waiting for a string we know is only at the very end of the command
            #timeout=None is here to avoid stopping the process if it is very long.
            #pexpect.TIMEOUT might be unnecessary 
            bashproc.sendline('\ncifbuild')
            cifbuild_state=bashproc.expect(['does not appear to contain an ODF','] ended',pexpect.TIMEOUT],timeout=None)
            
            if cifbuild_state==0:
                bashproc.sendline('\nexit')
                bashproc.expect(pexpect.EOF,timeout=None)
                evt_build_done.set()
                return 'incomplete ODF in this directory'

            time.sleep(1)
            
            bashproc.sendline('\nodfingest')
            odfingest_state=bashproc.expect(['] ended','odfingest: error',pexpect.TIMEOUT],timeout=None)
            
            if odfingest_state==1:
                bashproc.sendline('\nexit')
                bashproc.expect(pexpect.EOF,timeout=None)
            
                evt_build_done.set()
                
                return 'error durring odfingest'
        
        if build_PN:
            print('\nRunning epproc in a dedicated subdirectory:')
            bashproc.sendline('\nmkdir -p outputpn')
            bashproc.sendline('\ncd outputpn')
            #epproc doesn't process burst mode event by default, thus the burst=yes option
            bashproc.sendline('\nepproc burst=yes')
            bashproc.expect(['epicproc',pexpect.TIMEOUT],timeout=None)
            bashproc.sendline('\ncd ..')

        if build_MOS:
            print('\nRunning emproc in a dedicated subdirectory:')
            bashproc.sendline('\nmkdir -p outputmos')
            bashproc.sendline('\ncd outputmos')
            #same here
            bashproc.sendline('\nemproc')
            bashproc.expect(['epicproc',pexpect.TIMEOUT],timeout=None)
            bashproc.sendline('\ncd ..')
    
        #closing the spawn
        bashproc.sendline('\nexit')
        bashproc.expect(pexpect.EOF,timeout=None)
    
        evt_build_done.set()
        
        return 'Event building complete.'
    
def count_evts(directory):
    '''
    counting Evts files in directories and listing errors
    '''
    
    import glob
    
    os.chdir(directory)
    
    lfiles=glob.glob('**',recursive=True)
    
    
    lfiles_evts=[elem for elem in lfiles if 'Evts' in elem]
    
    files_rep=[[[],[]],[[],[]],[[],[]]]
    
    files_subrep=[[],[],[]]
    files_rep_err=[[],[],[]]
    
    for elem in lfiles_evts:
        if 'EPN' in elem:
            files_subrep[0].append(elem)
            if 'Timing' in elem:
                files_rep[0][0].append(elem)
            elif 'Imaging' in elem:
                files_rep[0][1].append(elem)
            else:
                files_rep_err[0].append(elem)
                
        if 'EMOS1' in elem:
            files_subrep[1].append(elem)
            if 'Timing' in elem:
                files_rep[1][0].append(elem)
            elif 'Imaging' in elem:
                files_rep[1][1].append(elem)
            else:
                files_rep_err[1].append(elem)
                
        if 'EMOS2' in elem:
            files_subrep[2].append(elem)
            if 'Timing' in elem:
                files_rep[2][0].append(elem)
            elif 'Imaging' in elem:
                files_rep[2][1].append(elem)
            else:
                files_rep_err[2].append(elem)
                
    print('Number of total PN Events: '+str(len(files_subrep[0])))
    print('Number of total MOS1 Events: '+str(len(files_subrep[1])))
    print('Number of total MOS2 Events: '+str(len(files_subrep[2])))
    
    print('Number of PN Timing evts: '+str(len(files_rep[0][0])))
    print('Number of PN Imaging evts: '+str(len(files_rep[0][1])))
    print('Number of MOS1 Timing evts: '+str(len(files_rep[1][0])))
    print('Number of MOS1 Imaging evts: '+str(len(files_rep[1][1])))
    print('Number of MOS2 Timing evts: '+str(len(files_rep[2][0])))
    print('Number of MOS2 Imaging evts: '+str(len(files_rep[2][1])))
    
    if len(files_rep_err[0])==len(files_rep_err[1])==len(files_rep_err[2])==0:
        print('No unfinished Evts detected.')
    else:
        print('Unfinished Evts detected.')
        
        if len(files_rep_err[0])!=0:
            print(len(files_rep_err[0])+' for PN: ')
            print(files_rep_err[0])
        if len(files_rep_err[1])!=0:
            print(len(files_rep_err[1])+' for MOS1: ')
            print(files_rep_err[1])
        if len(files_rep_err[2])!=0:
            print(len(files_rep_err[2])+' for MOS2: ')
            print(files_rep_err[2])
    
def file_selector(directory,filetype,camera='all'):

    '''
    Searches for all of the files of a specific type (among the ones used in the data reduction), 
    and asks for input if more than one are detected.
    
    use "all" as input in camera to get the result for all 3 cameras (default value)
    
    Returns a single file + file path (not absolute) for each camera
    '''
    
    #getting the list of files in the directory
    flist=glob.glob('**',recursive=True)
    
    cameras=['pn','mos1','mos2']

    #list of accepted filetypes
    filetypes=['evt_raw','evt_clean']
    file_desc=['raw event files','clean event files']
    
    #getting the index of the file type for the keywords
    type_index=filetypes.index(filetype)
    
    #type keywords
    keyword_types=['Evts','clean']
    
    #camera keywords (3 for each type)
    #for unfiltered event (direct results of emproc and epproc)
    camword_evt=['_EPN','_EMOS1','_EMOS2']
    #for the other filetypes the camera keywords are just the camera names
    keyword_cams=[camword_evt,cameras]
    
    #cam_list is the list of cameras to use for the evt search
    if camera=='all':
        cam_list=cameras
    else:
        cam_list=[camera]
    
    result_list=[]
    for cam in cam_list:
        #getting the index of the camera for the camera keywords
        cam_index=cameras.index(cam)
        
        cutlist=[]
        #reducing the list for the correct file type and camera
        cutlist=[elem for elem in flist if (keyword_types[type_index] in elem and keyword_cams[type_index][cam_index] in elem)]
        
        if len(cutlist)==0:
            print('\nWarning : No '+file_desc[type_index]+' found for camera '+cam)
            camdir=['']
            camfile=['']
        else:
            print('\n'+str(len(cutlist))+' exposure(s) found for '+cam+' '+file_desc[type_index]+' :')
            print(np.array(cutlist).transpose())

            #loop on all the events
            camdir=[]
            camfile=[]
            for i in range(len(cutlist)):
                elem=cutlist[i]

                #Storing the name and directory of the event files
                #if the files are not found in the local folder, we need to separate the path and the file name
                if elem.rfind('/')!=-1:
                    #taking of the batch directory since we copy the event file and spectrum products inside
                    if elem[:elem.rfind('/')][elem[:elem.rfind('/')].rfind('/')+1:]!='batch':
                        camdir+=[elem[:elem.rfind('/')]]
                        camfile+=[elem[elem.rfind('/')+1:]]
                else:
                    camdir+=['./']
                    camfile+=[elem]
                    
        result_list+=[[camfile,camdir]]

    return result_list

def filter_evt(directory,mode='std',cams='all',expos_mode='all',overwrite=True):
    
    '''
    Filters existing event files following the standard procedure, with some small modifications for better visibility in 
    ds9 afterwards.
    
    For flares, depending on the manual argument, the process either uses 
    
    'std' mode : the standard rate limits given in argument (usuallyy :
        -0.35 for PN
        -0.4 for MOS
        
    'nolim' mode : no limit (limit fixed at 1000)
        -usually run at first to make a first computation of the source/bg
        
    'auto' mode : compares previously computed lightcurves of the source and bg region and substracts 'unusually high' background zone
        -requires both region computation and the associated lightcurves
        
    'snr' mode : cuts the time bins in order to maximize the snr of the previously computed source/bg flare lighcturves
        -requires both region computation and the associated lightcurves
        
    Some of the convoluted elements are just here to accept weird file name or positions
    
    Give standard names to the products, with added suffix_evt to identify the exposure and mode.
    Easier for inputting in the next step of the data reduction.
    '''
    
    #defining which mode are included with a variable instead of arguments in the functions
    if expos_mode=='all':
        expos_mode_filter='IMAGING TIMING BURST'
    else:
        expos_mode_filter=expos_mode

    def filter_single(spawn,file,filedir):
        
        '''
        Filters the event list (manually of not) for a single camera
        manual filtering can be skipped to standard values by pressing enter when asked for the values
        '''
        
        if file=='':
            print('\nNo evt to filter for this camera in this obsid directory.')
            return 'No evt to filter for this camera in this obsid directory.'
        
        #Obtaining the exposure mode of the observation and skipping if it isn't in the selected modes

        fulldir=os.path.join(directory,filedir)
        
        expos_mode_single=fits.open(fulldir+'/'+file)[0].header['DATAMODE']
        if expos_mode_single not in expos_mode_filter:
            return 'Exposure mode not among the selected ones'
        
        #we send this via the bash because its location might be different than the python console's location
        spawn.sendline('\ncurrdir=$(pwd)')
        spawn.sendline('\ncd '+filedir)
        
        #identifying the exposure number and mode to create the suffix
        suffix_evt=file.split('Evts')[0][-12:]
        
        if suffix_evt[0]=='_':
            suffix_evt=suffix_evt[1:]
        
        camera=fits.open(fulldir+'/'+file)[0].header['INSTRUME'][1:].swapcase()
        
        #expressions for the plot and filtering commands for later
        if camera=='pn':
            expression_plot='"#XMMEA_EP&&(PI>10000&&PI<12000)&&(PATTERN==0)"'
            expression_filter='"#XMMEA_EP && gti('+camera+'gti_'+suffix_evt+'.ds,TIME) && (PI>200&&PI<12000)"'
        elif camera in {'mos1','mos2'}:
            expression_plot='"#XMMEA_EM&&(PI>10000)&&(PATTERN==0)"'
            expression_filter='"#XMMEA_EM && gti('+camera+'gti_'+suffix_evt+'.ds,TIME) && (PI>200&&PI<12000)"'
         
        #creation of the high-energy rate table
        spawn.sendline('\nevselect table='+file+' withrateset=Y rateset=rate'+camera+'_'+suffix_evt+'.ds '+
                       ' maketimecolumn=Y timebinsize=100 makeratecolumn=Y  expression='+expression_plot)
        spawn.expect([pexpect.TIMEOUT,'selected'],timeout=None)
        
        mode_func=mode
        
        #manual input of the limit if chosen
        if mode_func=='manual':
            spawn.sendline('\ndsplot table=rate'+camera+'_'+suffix_evt+'.ds x=TIME y=RATE.ERROR &')
            spawn.expect('dstoplot')
            ratelim=input('\nRate limit for '+camera+'_'+suffix_evt+' ?\n\n')
            #shortcut to switch to auto if needed
            if ratelim=='':
                mode_func='auto'
                
        if mode_func=='std':
            if camera=='pn':
                ratelim=str(pnflare_lim)
            elif camera=='mos1' or camera=='mos2':
                ratelim=str(mosflare_lim)
        
        if mode_func in ['snr','auto']:
            
            #fetching the lightcurve and area ratio
            lc_src_flare_path=fulldir+'/'+camera+'_'+suffix_evt+'_auto_lc_src_flare.ds'
            lc_bg_flare_path=fulldir+'/'+camera+'_'+suffix_evt+'_auto_lc_bg_flare.ds'
            
            try:    
                lc_src_flare=fits.open(lc_src_flare_path)[1].data
                lc_bg_flare=fits.open(lc_bg_flare_path)[1].data
                bkg_ratio=fits.open(fulldir+'/'+camera+'_'+suffix_evt+'_auto_lc_src_broad_corr.ds')[1].header['BKGRATIO']
            except:
                print("Couldn't load the lightcurves. There must be a problem in the region or exposure.")
                
                spawn.sendline('\ncd $currdir')
                return "Couldn't load the lighcturves."
            
            #saving the previous region screen 
            shutil.copyfile(fulldir+'/'+camera+'_'+suffix_evt+'_auto_reg_screen.png',
                            fulldir+'/'+camera+'_'+suffix_evt+'_auto_reg_hist_screen.png')
            
        if mode_func=='snr' :
            
            '''
            Here we maximize the source/bg snr by testing how deleting each lightcurve bin affects the SNR
            we do this process iteratively, with a single deletion per step, to maximize the SNR gain
            '''
            
            #snr function definition to avoid overcomplicating lines
            def fsnr(gti_indexes):
                
                return lc_src_flare['RATE'][gti_indexes].sum() / ((lc_bg_flare['RATE'][gti_indexes].sum())/bkg_ratio)**(1/2)
                
            #setting up the loop and defining the gtis indexes
            snr_improve=True
            gtis=np.arange(len(lc_src_flare))
            
            snr_nocut=fsnr(gtis)
            
            #we make a loop because each snr improvement can lead to subsequent improvements
            while snr_improve:
                
                #this step's base SNR
                snr_precut=fsnr(gtis)
                
                #testing the effect of cutting each remaining index
                snr_arr=[]
                for i in range(len(gtis)):
                    curr_gtis=gtis[:i].tolist()+gtis[i+1:].tolist()
                    snr_arr.append((fsnr(curr_gtis)))

                #fetching the index whose deletion leads to the best SNR
                max_snr_id=np.argwhere(snr_arr==max(snr_arr))[0][0]
                
                #deleting it if it is better than the current SNR, else stopping the loop
                if snr_arr[max_snr_id]>snr_precut:
                    gtis=np.array(gtis[:max_snr_id].tolist()+gtis[max_snr_id+1:].tolist())
                else:
                    snr_improve=False
            
            print('Flagged '+str(round(100*(1-(len(gtis)/len(lc_src_flare))),2))+'% of the exposure.')
            print('\nSNR improvement '+str(snr_nocut)+'->'+str(snr_precut))
        
            #storing the new gti inside a fits file
            fits_gti=fits.open(lc_src_flare_path)
            
            #creating the boolean column
            gti_column=fits.ColDefs([fits.Column(name='IS_GTI', format='I',
                                    array=np.array([1 if i in gtis else 0 for i in range(len(lc_src_flare))]))])
            #replacing the hdu with a hdu containing it
            fits_gti[1]=fits.BinTableHDU.from_columns(gti_column+fits_gti[1].columns[2:])
            
            #saving it
            fits_gti.writeto(lc_src_flare_path.replace('lc_src_flare','gti_flare'),overwrite=True)
            
            #plotting a graph with the deletion highlighted
            plot_lc(lc_src_flare_path,lc_bg_flare_path,area_ratio=bkg_ratio,mode='lc_src_flare_hist_snr',save=True,close=True,hflare=True)
            
        if mode_func=='auto':
            '''
            Here we try to automatically detect flares from previously computed lightcurves (normally without gti cleaning).
            We compare the rate of the background and source high energy lightcurves to see if they are comparable
            The criteria is cts_bg>=0.5*cts_src-err(cts_src)
            We then compute the gti according to this criteria and extract the clean event list from that
            For that, we create a combined fits file with the rate columns from both the source and bg light curves
            '''

            #saving the src/background flare curves showing the upcoming cut, with a specific name to not have them overwritten
            plot_lc(lc_src_flare_path,lc_bg_flare_path,area_ratio=bkg_ratio,mode='lc_src_flare_hist_auto',save=True,close=True,hflare=True)
            
            #creating a combined fits file
            shutil.copyfile(lc_src_flare_path,lc_src_flare_path.replace('src','comb'))
            
            lc_comb_flare=fits.open(lc_src_flare_path.replace('src','comb'))
            
            #changing the source column (needs to be done on the hdul and not on the data attribute)
            lc_comb_flare[1].columns.change_name('RATE','RATE_SRC')
            lc_comb_flare[1].columns.change_name('ERROR','ERROR_SRC')
            
            lc_comb_flare.writeto(lc_src_flare_path.replace('src','comb'),overwrite=True)
            lc_comb_flare=fits.open(lc_src_flare_path.replace('src','comb'))
            
            #adding the cother columns
            comb_flare_cols=lc_comb_flare[1].columns[:-1]+lc_bg_flare.columns
            comb_flare_hdu=fits.BinTableHDU.from_columns(comb_flare_cols)
            
            #renaming the bg columns
            comb_flare_hdu.columns.change_name('RATE','RATE_BG')
            comb_flare_hdu.columns.change_name('ERROR','ERROR_BG')
            
            #multiplying by the area ratio
            comb_flare_hdu.data['RATE_BG']*=1/bkg_ratio
            comb_flare_hdu.data['ERROR_BG']*=1/bkg_ratio
            
            #saving the fits
            lc_comb_flare[1]=comb_flare_hdu
            lc_comb_flare.writeto(lc_src_flare_path.replace('src','comb'),overwrite=True)
        
        if mode_func in ['auto','nolim','snr']:
            #in auto mode, it's only here for the plot but we don't care since we don't plot it anyway
            ratelim=10000
                
        if mode_func not in ['manual','auto','std','nolim','snr']:
            print('Wrong mode for the filter_evt function. exiting...')
            sys.exit()

        #waiting for the fits file to be created before loading the fits to make the plot
        while not os.path.isfile(os.path.join(fulldir,'rate'+camera+'_'+suffix_evt+'.ds')):
            time.sleep(1)
            
        #checking the validity of the fits file
        try:
            fits.open(os.path.join(fulldir,'rate'+camera+'_'+suffix_evt+'.ds'))
        except:
            time.sleep(5)
        
        try:
            fits.open(os.path.join(fulldir,'rate'+camera+'_'+suffix_evt+'.ds'))
        except:
            print("\nCouldn't load the flare rate fits file. Skipping...")
            
            spawn.sendline('\ncd $currdir')
            return"Couldn't load the flare rate fits file."
            
        #here we only plot the rate cut if there is an interesting limit to show
        plot_lc(os.path.join(fulldir,'rate'+camera+'_'+suffix_evt+'.ds'),save=True,close=True,mode='rate_flare',
                plotcut=False if mode_func in ['nolim','auto','snr'] else True,flarelim=ratelim)
        
        #creation of the gti and actual filtering
        if mode_func=='auto':
            spawn.sendline('\ntabgtigen table='+lc_src_flare_path.replace('src','comb').split('/')[-1]+
                           ' expression="RATE_SRC-ERROR_SRC-(RATE_BG+ERROR_BG)>RATE_SRC/2" gtiset='+camera+'gti_'+suffix_evt+'.ds')
        elif mode_func=='snr':
            spawn.sendline('\ntabgtigen table='+lc_src_flare_path.replace('lc_src_flare','gti_flare')+' expression="IS_GTI==1"'+
                           ' gtiset='+camera+'gti_'+suffix_evt+'.ds')
        else:
            spawn.sendline('\ntabgtigen table=rate'+camera+'_'+suffix_evt+'.ds expression="RATE<='+str(ratelim)
                           +'" gtiset='+camera+'gti_'+suffix_evt+'.ds')
            
        spawn.expect([pexpect.TIMEOUT,'ended'],timeout=None)
        
        spawn.sendline('\nevselect table='+file+' withfilteredset=Y filteredset='+camera+'clean_'+suffix_evt+'.ds'+
                        ' destruct=Y keepfilteroutput=T  expression='+expression_filter)
        
        spawn.expect([pexpect.TIMEOUT,'ended'],timeout=None)
        
        if expos_mode_single=='IMAGING':
            #in imaging we simply rebin the event file
            expression_img='xcolumn=X ycolumn=Y ximagebinsize=80 yimagebinsize=80'
        elif expos_mode_single=='TIMING' or expos_mode=='BURST':
            #in timing, depending on the camera we create a plot of time against one of the dimensions
            if camera=='pn':
                expression_img='xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1'
            elif camera=='mos1' or camera=='mos2':
                expression_img='xcolumn=RAWX ycolumn=TIME ximagebinsize=1 yimagebinsize=10'
        
        #this one will only trigger if there is a problem in the output, so we add an actual timeout value to avoid blocking the script
        spawn.expect([pexpect.TIMEOUT,'ended'],timeout=30)
        
        spawn.sendline('\nevselect table='+camera+'clean_'+suffix_evt+'.ds imagebinning=binSize imageset='+camera+
                       'img_'+suffix_evt+'.ds withimageset=yes '+expression_img)
        
        #it seems going too fast messes with the outputs so we let some time for the spawn to recover at the end of every filtering
        spawn.expect([pexpect.TIMEOUT,'ended'],timeout=None)
        
        #waiting for the file to be created
        while not os.path.isfile(fulldir+'/'+camera+'img_'+suffix_evt+'.ds'):
            time.sleep(1)
        
        #copying the rate files
        spawn.sendline('\ncp *rate'+camera+'_'+suffix_evt+'* $currdir/batch')
        
        #small wait to make sure we have the time to copy the files
        time.sleep(1)
        
        spawn.sendline('\ncd $currdir')
        
        return 'Event filtering complete.'
    
    if cams=='all':
        camid_fevt=[0,1,2]
    else:
        camid_fevt=[]
        if 'pn' in cams:
            camid_fevt.append(0)
        if 'mos1'in cams:
            camid_fevt.append(1)
        if 'mos2' in cams:
            camid_fevt.append(2)
            
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nEvent filtering...')
    bashproc.sendline('cd '+directory)
    set_var(bashproc,directory)
    
    #summary file header
    if directory.endswith('/'):
        obsid=directory.split('/')[-2]
    else:
        obsid=directory.split('/')[-1]
    summary_header='Obsid\tFile identifier\tEvent filtering result\n'
   
    #recensing the event files for each camera
    #camfilelist shape : [[pnfiles,pndirs],[mos1files,mos1dirs],[mos2files,mos2dirs]]
    cam_filelist=file_selector(directory,'evt_raw')
    
    #creating the batch directory before anything else to avoid problems
    bashproc.sendline('mkdir -p batch')
    
    #filtering for the selected cameras
    #Nested loops to walk through both the chosen cameras and the detected files for each camera
    for i in camid_fevt:
        
        for j in range(len(cam_filelist[i][0])):
        
            raw_evtfile=cam_filelist[i][0][j]
            raw_evtdir=cam_filelist[i][1][j]
            
            #testing if the last file of the filtering process (the image) has been created in the detected event directory
            lastfile=raw_evtfile.split('Evts')[0][-12:]
            if lastfile!='':
                if lastfile[0]=='_':
                    lastfile=lastfile[1:]
            raw_evtid=camlist[i]+'_'+lastfile
            lastfile=camlist[i]+'img_'+lastfile+'.ds'

            if (overwrite or not os.path.isfile(directory+'/'+raw_evtdir+'/'+lastfile)) and raw_evtfile!='':
                
                #setting up a logfile in parallel to terminal display :
                if os.path.isfile(raw_evtdir+'/'+raw_evtid+'_filter_evt.log'):
                    os.system('rm '+raw_evtdir+'/'+raw_evtid+'_filter_evt.log')
                with StdoutTee(raw_evtdir+'/'+raw_evtid+'_filter_evt.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
                    StderrTee(raw_evtdir+'/'+raw_evtid+'_filter_evt.log',buff=1,file_filters=[_remove_control_chars]):
                                
                    bashproc.logfile_read=sys.stdout
                    print('\nFiltering events of exposure '+raw_evtfile+' for '+camlist[i]+' camera.')
                    
                    #main function
                    summary_line=filter_single(bashproc,raw_evtfile,raw_evtdir)
            
            else:
                if raw_evtfile=='':
                    print('\nNo evt to filter for camera '+camlist[i]+ ' in the obsid directory.')
                    
                    summary_line='No evt to filter for camera '+camlist[i]+ ' in the obsid directory.'
                    raw_evtid=camlist[i]
                else:
                    print('\nEvent filtering for the '+camlist[i]+' exposure '+raw_evtfile+
                          ' already done. Skipping...')
                    summary_line=''
            if summary_line!='':
                summary_content=obsid+'\t'+raw_evtid+'\t'+summary_line
                
                file_edit(os.path.join(directory,'batch','summary_filter_evt.log'),obsid+'\t'+raw_evtid,summary_content+'\n',
                          summary_header)

    print('\nFiltering of the current obsid directory events finished.')
    #closing the spawn
    bashproc.sendline('exit')
    
    filter_evt_done.set()

def disp_ds9(spawn,file,zoom='auto',scale='log',regfile='',screenfile='',give_pid=False,kill_last=''):
    
    '''
    Regfile is an input, screenfile is an output. Both can be paths
    If "screenfile" is set to a non empty str, we make a screenshot of the ds9 window in the given path
    This is done manually since the ds9 png saving command is bugged
    
    if give_pid is set to True, returns the pid of the newly created ds9 process
    '''
    
    if scale=='linear 99.5':
        scale='mode 99.5'
    elif ' ' in scale:
        scale=scale.split(' ')[0]+' mode '+scale.split(' '[1])
    
    #if automatic, we set a higher zoom for timing images since they usually only fill part of the screen by default
    if zoom=='auto':
        if 'Timing' in file and 'pn' in file:
            zoom=4
        else:
            zoom=1.67

    
    #region load command
    if regfile!='':
        regfile='-region '+regfile
    
    #parsing the open windows before and after the ds9 command to find the pid of the new ds9 window
    if screenfile!='' or give_pid:
        windows_before=subprocess.run(['wmctrl','-l'],stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        
    spawn.sendline('echo "Ph0t1n0s" | sudo -S ds9 -view buttons no -cmap Heat -geometry 1080x1080 -scale '+scale+' -mode region '+file+' -zoom '+str(zoom)+
                   ' '+regfile+' &')
    
    #the timeout limit could be increased for slower computers or heavy images
    spawn.expect(['password',pexpect.TIMEOUT],timeout=1)
    
    #second part of the windows parsing

    ds9_pid=0
    
    if screenfile!='' or give_pid:
        
        windows_after=subprocess.run(['wmctrl','-l'],stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        
        #since sometimes the ds9 window takes time to load, we loop until the window creation to be sure we can take
        #the screenshot
        delay=0
        while len(windows_after)==len(windows_before) and delay<=10:
            time.sleep(1)
            windows_after=subprocess.run(['wmctrl','-l'],stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
            delay+=1
        
        for elem in windows_after:
            if elem not in windows_before:
                ds9_pid=elem.split(' ')[0]
                print('\nIdentified the new ds9 window as process '+ds9_pid)
                
                if screenfile!='':
                    print('\nSaving screenshot...')
                    os.system('import -window '+ds9_pid+' '+screenfile)
    
    #we purposely do this at the very end
    if kill_last!='':
        print('\nClosing previous ds9 window...')

        os.system('wmctrl -ic '+kill_last)
    

    if give_pid:
        return ds9_pid
    
def reg_optimiser(mask):
    
    #for the shapely method :
    # from shapely.geometry import Polygon
    # from shapely.ops import polylabel as sh_polylabel
    # from shapely.validation import make_valid
    #Note: shapely.validation doesn't work with shapely 1.7.1, which is the standard version currently installed by conda/pip
    #manually install 1.8a3 so solve it

    '''
    Computes the biggest circle region existing inside of a mask using the polylabel algorithm
    See https://github.com/mapbox/polylabel
    
    This algorithms returns the point of inaccessibility, i.e. the point furthest from any frontier (hole or exterior) or a 
    polygon

    Note : the pylabel algorithm has been ported twice in python. 
    The shapely port is less powerful but can use wider input since it uses the polygon class of its own library in lieu of 
    a list of arrays. Still, it's more complex to use so we'll use the other port as long as it works.
    
    Process : 
    1. Transforming the mask into a set of polygons (shell and holes) with imantics
    2. swapping the array to have the exterior as the first element
    3. using polylabel
    
    Additional steps with shapely:
    2. Creating a valid polygon argument with shapely
    3. Computing the poi with the built-in polylabel of shapely
    4. computing the distance to get the maximal possible radius
    '''
    
    #creating a digestible input for the polygons function of imantics
    int_mask=mask.astype(int)
    
    #this function returns the set of polygons equivalent to the mask
    #the last polygon of the set should be the outer shell
    polygons=Mask(int_mask).polygons()
    
    #since we don't know the position of the outer shell (and there is sometimes no outer shell)
    #we'll consider the biggest polygon as the "main one". 
    #It's easily identifiable as the ones with the biggest number of points 
    #(since they are identified on pixel by pixel basis, there seems to be no "long" lines)
    
    shell_length=0
    for i in range(len(polygons.points)):
        if len(polygons.points[i])>shell_length:
            shell_id=i
            shell_length=len(polygons.points[i])
            
    #swapping the positions to have the shell as the first polygon in the array
    poly_args=polygons.points[:shell_id]+polygons.points[shell_id+1:]
    poly_args.insert(0,polygons.points[shell_id])
    
    coords=polylabel(poly_args,with_distance=True)

    #second method (was coded before the first so let's keep it just in case)
    '''
    #Creating a usable polygon with the shapely library
    CCD_shape=Polygon(shell=CCD_polygons.points[-1],holes=CCD_polygons.poings[:-1])
    
    #This polygon can be invalid due to intersections and as such the polylabel function might not work
    if CCD_shape.is_valid:
        CCD_shape_valid=CCD_shape
    else:
        #in this case, we make it valid, which outputs a list of geometries instead of the polygon.
        #Thus we skim through the resulting geometries until we get our "good" polygon
        CCD_shape_valid=make_valid(CCD_shape)
        while CCD_shape_valid.type!='Polygon':
            CCD_shape_valid=CCD_shape_valid[0]
    
    #Now the polylabel function can finally be used
    shape_poi=sh_polylabel(CCD_shape_valid)
    
    #the resulting position is in image coordinates
    bg_ctr=[shape_poi.xy[0][0],shape_poi.xy[1][0]]
    
    #to get the maximal radius of the bg circle, we compute the distance to the polygon
    bg_maxrad=CCD_shape.exterior.distance(shape_poi)
    
    #Just in case, we also compare it manually with the distance to the holes (might be redundant)
    for elem in CCD_shape.interiors:
        if elem.distance(shape_poi)<bg_maxrad:
            bg_maxrad=elem.distance(shape_poi)
            
    coords=(bg_ctr,bg_maxrad)
    '''
    return coords

def savepng(array,name,directory='./',astropy_wcs=None,mpdaf_wcs=None,title=None,imgtype=''):
    
    '''
    Separated since it might be of use elsewhere
    Note: astropy wcs do not work with mpdaf and mpdaf wcs do not work with matplotlib
    can mask the irrelevant parts of the image is the type is specified (ccd_crop or ccd_crop_mask)
    
    unfortunately the wcs projection doesn't work after masking the CCD
    Can probably be fixed by saving and reloading the fits wcs with astropy but it's not needed for now
    
    Note : there's something very weird going on with the masking as the ccd_crop mask always ends up being 
    wider than the ccd_crop one, with no apparent reason (same masking region in the variables but it still doesn't mask 
                                                          everything)
    To be fixed later
    '''
    
    if imgtype=='ccd_crop':
        def line_irrel(line):
            return len(np.unique(np.isnan(line)))==1 and np.isnan(line)[0]
    if imgtype=='ccd_crop_mask':
        def line_irrel(line):
            return len(np.unique(line))==1 and line[0]==0
    
    sep=''
    if directory[-1]!='/':
        sep='/'
    img=Image(data=array,wcs=mpdaf_wcs)
    
    #masking 
    if imgtype=='ccd_crop' or imgtype=='ccd_crop_mask':
    
        crop_boxes=[[0,np.size(array.T,0)-1],[0,np.size(array.T,1)-1]]
        
        #finding the first/last row/column for which the array contains relevant data
        for i in range(np.size(array,0)):
            if not line_irrel(array[i]) and crop_boxes[0][0]==0:
                crop_boxes[0][0]=i
            if not line_irrel(array[-i-1]) and crop_boxes[0][1]==np.size(array,0)-1:
                crop_boxes[0][1]=np.size(array,0)-1-i
                
        for j in range(np.size(array.T,0)):
            if not line_irrel(array.T[j]) and crop_boxes[1][0]==0:
                crop_boxes[1][0]=j
            if not line_irrel(array.T[-j-1]) and crop_boxes[1][1]==np.size(array.T,0)-1:
                crop_boxes[1][1]=np.size(array.T,0)-1-j

        #creating the correct arguments for the mask_region method
        widths=(crop_boxes[0][1]-crop_boxes[0][0],crop_boxes[1][1]-crop_boxes[1][0])
        center=(crop_boxes[0][0]+widths[0]/2,crop_boxes[1][0]+widths[1]/2)
        
        img.mask_region(center=center,radius=widths,unit_center=None,unit_radius=None,inside=False)
        img.crop()
        
        proj=None
    else:
        proj={'projection':astropy_wcs}
        
    fig,ax=plt.subplots(1,subplot_kw=proj,figsize=(12,10))
    fig.suptitle(title)
    img.plot(cmap='plasma',scale='log',colorbar='v')
    
    #this line is just here to avoid the spyder warning

    
    fig.tight_layout()
    fig.savefig(directory+sep+name+'.png')
    plt.close(fig)
    
def pileup_bool(pileup_line):
    
    '''
    returns True if there is pile-up
    
    Separated from the pileup computation in extract_sp for use in the line detection scripts
    '''
    
    #the errors are given by default with a 3 sigma confidence level
    
    pattern_s_val=float(pileup_line.split('s: ')[1].split('   ')[0].split(' ')[0])
    pattern_s_err=float(pileup_line.split('s: ')[1].split('   ')[0].split(' ')[2])
    pattern_d_val=float(pileup_line.split('d: ')[1].split('   ')[0].split(' ')[0])
    pattern_d_err=float(pileup_line.split('d: ')[1].split('   ')[0].split(' ')[2])
    
    #no pileup means the s and d pattern values are compatible with 1
    #we test only one side since s only gets lower and d only get higher
    
    #this will enable us to create exceptions when there is no detected events in the pileup computation
    #(which returns 0 values for everything)
    
    #in the rare case of dim sources with few photons sometimes there is no d value, but there can still be a source nonetheless
    if pattern_d_val==0:
        d_pileup=False
    else:
        d_pileup=abs(pattern_d_val-1)>pattern_d_err
        
    if pattern_s_val==0:
        return None    
    else:
        return abs(pattern_s_val-1)>pattern_s_err or d_pileup

def extract_reg(directory,mode='manual',cams='all',expos_mode='all',overwrite=True):
    
    '''
    Extracts the optimal source/bg regions for a given exposure

    As of now, only takes input formatted through the evt_filter function

    Only accepts circular regions (in manual mode)
    '''
    
    #defining which mode are included with a variable instead of arguments in the functions
    if expos_mode=='all':
        expos_mode_regex='IMAGING TIMING BURST'
    else:
        expos_mode_regex=expos_mode
        
    def interval_extract(list):
        
        '''
        From a list of numbers, outputs a list of the integer intervals contained inside it 
        '''
        
        list = sorted(set(list))
        range_start = previous_number = list[0]
      
        for number in list[1:]:
            if number == previous_number + 1:
                previous_number = number
            else:
                yield [range_start, previous_number]
                range_start = previous_number = number
        yield [range_start, previous_number]
    

    def extract_reg_single(spawn,file,filedir):
        
        if file=='':
            print('\nNo evt to extract spectrum from for this camera in the obsid directory.')
            return 'No evt to extract spectrum from for this camera in the obsid directory.'
        
        fulldir=directory+'/'+filedir

        #opening the fits file to extract some informations on the exposure
        fits_evtclean=fits.open(fulldir+'/'+file)
        expos_mode_single=fits_evtclean[0].header['DATAMODE']
        submode=fits_evtclean[0].header['SUBMODE']
        print('\nexpos mode:',expos_mode_single)
        camera=fits_evtclean[0].header['INSTRUME'][1:].swapcase()
        
        #quitting the extraction if the exposure mode is not in the selected ones
        if expos_mode_single not in expos_mode_regex:
            return 'Exposure mode not among the selected ones.'
        
        #identifying the exposure number and mode to create the suffix_evt
        #different method than in filter_evt because here we already know the structure of the file name
        suffix_evt=file.split('.')[0][file.find('_'):]
        
        spawn.sendline('\ncurrdir=$(pwd)')
        spawn.sendline('\ncd '+filedir)
        
        #the file given as argument should be the clean event and as such it won't give out anything good with ds9
        file_img=file.replace('clean','img')
        
        #copying the 'initial' evtclean file
        shutil.copyfile(fulldir+'/'+file,fulldir+'/'+camera+suffix_evt+'_'+mode+'_evt_save.ds')
        file_init=fulldir+'/'+camera+suffix_evt+'_'+mode+'_evt_save.ds'
        
        #opening the image file and saving it for verification purposes
        ds9_pid_sp_start=disp_ds9(spawn,file_img,screenfile=fulldir+'/'+camera+suffix_evt+'_auto_img_screen.png',give_pid=True)
        
        try:
            fits_img=fits.open(fulldir+'/'+file_img)
        except:
            print("\nCould not load the image fits file. There must be a problem with the exposure."+
                  "\nSkipping spectrum computation...")
            spawn.sendline('\ncd $currdir')
            return "Could not load the image fits file. There must be a problem with the exposure."
        
        if ds9_pid_sp_start==0:
            print("\nCould not load the image file with ds9. There must be a problem with the exposure."+
                  "\nSkipping spectrum computation...")
            spawn.sendline('\ncd $currdir')
            return "Could not load the image file with ds9. There must be a problem with the exposure."

        def source_catal(dirpath):  
            
            '''
            Tries to identify a Simbad object from either the directory structure or the source name in the file itself
            '''
            
            #splitting the directories and searching every name in Simbad
            dir_list=dirpath.split('/')[1:]
            
            #removing a few problematic names
            crash_sources=['M2','home','outputmos','BlackCAT','']
            #as well as obsid type names that can cause crashes
            for elem_dir in dir_list:
                if len(elem_dir)==10 and elem_dir.isdigit() or elem_dir in crash_sources:
                    dir_list.remove(elem_dir)
                    
            #Simbad.query_object gives a warning for a lot of folder names so we just skip them
            obj_list=None
            for elem_dir in dir_list:
                try:
                    with warnings.catch_warnings():
                        # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
                        # warnings.filterwarnings('ignore','.*Identifier not found.*',)
                        warnings.filterwarnings('ignore',category=UserWarning)
                        elem_obj=Simbad.query_object(elem_dir)
                        if elem_obj!=None:
                            obj_list=elem_obj
                except:
                    print('\nProblem during the Simbad query. This is the current directory list:')
                    print(dir_list)
                    spawn.sendline('\ncd $currdir')
                    return 'Problem during the Simbad query.'
                
            target_name=fits.open(dirpath+'/'+file)[0].header['OBJECT']
            try:
                with warnings.catch_warnings():
                    # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
                    # warnings.filterwarnings('ignore','.*Identifier not found.*',)
                    warnings.filterwarnings('ignore',category=UserWarning)
                    file_query=Simbad.query_object(target_name)
            except:
                print('\nProblem during the Simbad query. This is the current obj name:')
                print(dir_list)
                spawn.sendline('\ncd $currdir')
                return 'Problem during the Simbad query.'
                
            if obj_list is None:
                print("\nSimbad didn't recognize any object name in the directories."+
                      " Using the target of the observation instead...")
                obj_list=file_query

            if type(file_query)==type(None):
                print("\nSimbad didn't recognize the object name from the file header."+
                      " Using the name of the directory...")
                target_query=''
            else:
                target_query=file_query[0]['MAIN_ID']
                
            if type(obj_list)==type(file_query) and type(obj_list)==type(None):
                print("\nSimbad couldn't detect an object name. Skipping this observation...")
                spawn.sendline('\ncd $currdir')
                return "Simbad couldn't detect an object name."

            #if we have at least one detections, it is assumed the "last" find is the name of the object                
            obj_catal=obj_list[-1]

            print('\nValid name(s) detected. Object name assumed to be '+obj_catal['MAIN_ID'])

            if obj_catal['MAIN_ID']!=target_query and target_only:
                
                print('\nTarget only mode activated and the source studied is not the main focus of the observation.'+
                      '\nSkipping...')
                spawn.sendline('\ncd $currdir')
                return 'Target only mode activated and the source studied is not the main focus of the observation.'
            
            return obj_catal
            
        def spatial_expression(coords,type='XMM',mode=expos_mode_single,excision_radius=None,rawx_off=0.,timing_ds9type='src',timing_fits=None):
            
            '''
            In Timing:
                Returns the spatial expression for rectangular regions from list of included pixels. 
                Automatically computes single regions from neighboring pixels 
                The XMM expression is in RAWX coordinates, the ds9 expression in image (shifted by one pixel from RAWX)
            In Imaging : 
                Returns the spatial expression for circular region coordinates of type ([center1,center2],radius),
                or an annulus if the excision is set to a value.
                The expression itself will be in degrees
                type can be either XMM or ds9
            '''
            
            if mode=='TIMING' or mode=='BURST':
                
                '''
                transformation into intervals with the function above (found online)
                the weird call is here to switch the array of mono elements list into a proper list
                we also add 1 to switch back from pixel indexes to proper pixel positions in RAW
                (which starts at 1 instead of 0)
                Note: It's not strictly necessary for the source region now since we use a SNR test on a single region propagating 
                from the brightest pixel
                '''
                
                coords_intervals=list(interval_extract([elem+1 for elem in coords]))
                
                if type=='XMM':
                    
                    #we go back to RAWX from IMAGE coords so we substract the coordinates offset between RAWX and IMAGE
                    return '('+' || '.join(['(RAWX>='+str(elem[0]-rawx_off)+') && (RAWX<='+str(elem[1]-rawx_off)+')' 
                                                       for elem in coords_intervals])+')'
                
                if type=='ds9':
                        '''
                        Here, we use the fits argument to get the shape of the timing image
                        From it, we create a list of rectangular ds9 regions from each interval
                        +1 in the widths computation because we include the starting pixel
                        '''
                        
                        src_widths=[str(coord[1]+1-coord[0]) for coord in coords_intervals]
                        src_ctrs=[str(coord[0]+(coord[1]-coord[0])/2) for coord in coords_intervals]
                
                        #getting shape information from the fits timing image
                        reg_height=timing_fits[0].shape[0]
                        reg_height=str(reg_height)
                        reg_ctr=str(timing_fits[0].shape[0]/2)
                        
                        timing_str=''
    
                        if timing_ds9type=='pile-up':
                            text_add='# color=cyan width=1 \n'
                            
                        for i in range(len(src_widths)):
                            if timing_ds9type in ['src','bg']:
                                text_add=' # text={'+timing_ds9type+' '+str(i+1)+'}'
                            timing_str+='\nbox('+src_ctrs[i]+','+reg_ctr+','+src_widths[i]+','+reg_height+')'+text_add
    
                        return timing_str
                    
            if mode=='IMAGING':
                if type=='XMM':
                    if excision_radius==None:
                        return '((RA,DEC) in CIRCLE('+coords[0][0]+','+coords[0][1]+','+coords[1]+'))'
                    else:
                        return '((RA,DEC) in ANNULUS('+coords[0][0]+','+coords[0][1]+','+excision_radius+','+coords[1]+'))'
                if type=='ds9':
                    if excision_radius==None:
                        return 'circle('+coords[0][0]+','+coords[0][1]+','+coords[1]+'")'
                        
        #lightcurve computation function
        def make_lc_snr(table,suffix_lc,expression_region,binning):
            
            #standard part of the command expressions:
            if expos_mode_single in ['TIMING','BURST']:
                if camera=='pn':
                    expression_sp='(FLAG==0) && (PATTERN<=4) && '
                elif camera in {'mos1','mos2'}:
                    expression_sp='(FLAG==0) && (PATTERN<=0) && '
            elif expos_mode_single=='IMAGING':
                if camera=='pn':
                    expression_sp='(FLAG==0) && (PATTERN<=4) && '
                elif camera in {'mos1','mos2'}:
                    expression_sp='#XMMEA_EM && (PATTERN<=12) && '
                
            #the energy limits for the SNR lightcurve are the ones we will be interested in for our analysis, so here 
            elims_snr=' && (PI in ['+str(int(1000*flare_band[0]))+':'+str(int(1000*flare_band[1]))+'])'
            
            #complete expression
            expression_snr=expression_sp+expression_region+elims_snr
            
            lc_path=os.path.join(fulldir,camera+suffix_evt+prefix+'_lc_'+suffix_lc+'.ds')
            
            #removing any previously computed lightcurve if needed
            if os.path.isfile(lc_path):
                os.remove(lc_path)
                while os.path.isfile(lc_path):
                    time.sleep(1)
                    
            #sas command
            spawn.sendline('evselect table='+table+' energycolumn=PI '+
                            'expression="'+expression_snr+'" withrateset=yes rateset='+camera+suffix_evt+prefix+'_lc_'+suffix_lc+'.ds '+
                            'timebinsize='+str(binning)+' maketimecolumn=yes makeratecolumn=yes')
            #there are two occurences of this, one at the beginning and one at the end of the command 
            spawn.expect(['\[xmmsas_'],timeout=None)
            spawn.expect(['\[xmmsas_'],timeout=None)
            
            #waiting for the file to be created to continue
            while not os.path.isfile(lc_path):
                time.sleep(1)

            #waiting for the file to be readable to continue
            file_readable=False
            while file_readable==False:
                try:
                    fits.open(lc_path)
                    file_readable=True
                except:
                    time.sleep(1)
                    
            return os.path.join(fulldir,camera+suffix_evt+prefix+'_lc_'+suffix_lc+'.ds')

        def opti_single(data_lc_src,data_lc_bg,ratio):
            
            '''
            Here we maximize the snr of a source/bg lightcurve couple by testing how deleting each lightcurve bin affects the SNR
            we do this process iteratively, with a single deletion per step, to maximize the SNR gain
            
            ratio is the area ratio
            '''
            
            #snr function definition depending on the gti indexes to avoid overcomplicating lines
            def fsnr(gti_indexes):
                
                src_counts=data_lc_src['RATE'][gti_indexes].sum()
                
                bg_counts=data_lc_bg['RATE'][gti_indexes].sum()
                
                if str(src_counts)!='--':
                    #formula from https://xmm-tools.cosmos.esa.int/external/sas/current/doc/specgroup/node10.html
                    #1e-10 here to avoid wrong divisions
                    
                    #we also switch back to count instead of count rate by multiplying by the sqrt of the exposure
                    curr_binning=data_lc_src['TIME'][1]-data_lc_src['TIME'][0]
                    snr_local=(src_counts-bg_counts*ratio)/(src_counts+bg_counts*ratio**2+1e-10)**(1/2)\
                              *(len(gti_indexes)*curr_binning)**(1/2)
                    
                    return max(0,snr_local)
                else:
                    return 0
            
            #setting up the loop and defining the gtis indexes (we start with the whole lightcurve)
            snr_improve=True
            gtis=np.arange(len(data_lc_src))
            
            snr_nocut=fsnr(gtis)
            
            #if the region is empty there might be residual counts due to flares or such which might lead to a 0 SNR anyway
            if snr_nocut==0:
                print('\nEmpty region.')  
                return gtis,0
            #we make a loop because each snr improvement can lead to subsequent improvements
            while snr_improve:
                
                #this step's base SNR
                snr_precut=fsnr(gtis)
                
                #testing the effect of cutting each remaining index
                snr_arr=[]
                
                for i in range(len(gtis)):
                    curr_gtis=gtis[:i].tolist()+gtis[i+1:].tolist()
                    snr_arr.append((fsnr(curr_gtis)))                  
                
                #fetching the index whose deletion leads to the best SNR
                max_snr_id=np.argwhere(snr_arr==max(snr_arr))[0][0]
                    
                #deleting it if it is better than the current SNR, else stopping the loop
                if snr_arr[max_snr_id]>snr_precut:
                    gtis=np.array(gtis[:max_snr_id].tolist()+gtis[max_snr_id+1:].tolist())
                else:
                    snr_improve=False
            
            print('\nLightcurve SNR optimisation flagged '+str(round(100*(1-(len(gtis)/len(data_lc_src))),3))
                  +'% of the exposure.')
            print('\nSNR improvement: '+str(snr_nocut)+'->'+str(snr_precut))
            
            #outputing the optimised values
            return gtis,snr_precut
        
        def opti_events(gtis,path_best_lc_src,path_lc_bg,area_ratio,add_str):
            
            '''
            Saves the gtis and creates a plot of it, refilters the event list, and recreates an image according to the gti cut
            
            the lc needs to be the one corresponding to the best snr
            In order to avoid problems if pnclean is updated twice (after before and after the pile-up excision), 
            the pnclean file is copied to another name and the copied name serves as the base for the new clean event list
            creation
            '''
                
            #storing the new gti inside whatever lc fits file
            fits_gti=fits.open(path_best_lc_src)
            
            #creating the boolean column
            gti_column=fits.ColDefs([fits.Column(name='IS_GTI', format='I',
                                    array=np.array([1 if i in gtis else 0 for i in range(len(fits_gti[1].data))]))])
            #replacing the hdu with a hdu containing it
            fits_gti[1]=fits.BinTableHDU.from_columns(gti_column+fits_gti[1].columns[2:])
            fits_gti[1].name='IS_GTI'
            #saving it
            gti_path=path_best_lc_src[:-6].replace('lc_src_snr','gti_bool_snr')+add_str+'.ds'
            
            fits_gti.writeto(gti_path,overwrite=True)

            plot_lc(path_best_lc_src,path_lc_bg,
                    area_ratio=area_ratio,mode='lc_src_snr'+add_str,
                    save=True,close=True,hflare=True)

            #filtering the event list according to the gtis
            spawn.sendline('\ntabgtigen table='+gti_path.split('/')[-1]+\
                           ' expression="IS_GTI==1" gtiset='+camera+'gti'+suffix_evt+'_snr'+add_str+'.ds')
            
            spawn.expect([pexpect.TIMEOUT,'ended'],timeout=None)
            
            #expression filter for updating pnclean
            expression_filter='"gti('+camera+'gti'+suffix_evt+'_snr'+add_str+'.ds'+',TIME)"'
            
            #this should work with the evtclean file but might not with the first one since the clean event file is already 
            #filtered
            spawn.sendline('\nevselect table='+camera+suffix_evt+prefix+'_evt_save.ds'+' withfilteredset=Y filteredset='+file+
                            ' destruct=Y keepfilteroutput=T  expression='+expression_filter)
            
            spawn.expect([pexpect.TIMEOUT,'ended'],timeout=None)

            #recreating the image, this time with a different name
            if expos_mode_single in ['TIMING','BURST']:
                if camera=='pn':
                    expression_crea_img='xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1'
                elif camera=='mos1' or camera=='mos2':
                    expression_crea_img='xcolumn=RAWX ycolumn=TIME ximagebinsize=1 yimagebinsize=10'
            if expos_mode_single=='IMAGING':
                expression_crea_img='xcolumn=X ycolumn=Y ximagebinsize=80 yimagebinsize=80'
            
            spawn.sendline('\nevselect table='+file+' imagebinning=binSize imageset='+camera+
                           'img'+suffix_evt+'_snr'+add_str+'.ds withimageset=yes '+expression_crea_img)

            spawn.expect([pexpect.TIMEOUT,'ended'],timeout=None)

            path_img_clean=fulldir+'/'+camera+'img'+suffix_evt+'_snr'+add_str+'.ds'
            #waiting for the image file to be created to continue
            while not os.path.isfile(path_img_clean):
                time.sleep(1)

            #waiting for the file to be readable to continue
            file_readable=False
            while file_readable==False:
                try:
                    fits.open(path_img_clean)
                    file_readable=True
                except:
                    time.sleep(1)
            
        if expos_mode_single in ['TIMING','BURST']:
            
            if mode=='manual':

                print("\nManual mode for TIMING exposure:"+
                      "\nInput the image coordinates of the source and bg limits you've chosen."+
                      "\nNote : the values of the pixel you will input will be included in the region (i.e. 20 to 30 includes"+
                      "both pixels 20 and 30)")
                prefix=input("\nPrefix to be added in the files ?\n\n")
                
                print("\nSource :")
                src_coord1=input("\nFirst limit of the interval to be included (left)\n\n")
                src_coord2=input("\nSecond limit of the interval to be included (right)?\n\n")
                
                print("\nBackground :")
                bg_coord1=input("\nFirst limit of the interval to be included (left)\n\n")
                bg_coord2=input("\nSecond limit of the interval to be included (right)?\n\n")
                
                if prefix!='':
                    prefix='_'+prefix
                
                #summary variables. We add one level of lists to have a similar shape to the auto mode variables
                src_coords=[[src_coord1,src_coord2]]
                bg_coords=[[bg_coord1,bg_coord2]]
                
                '''
                In timing mode, we can only put rawx delimitations in the XMM selection expression, but we create box 
                ds9 regions for the sake of visualisation, which requires a bit of transformation
                If we want the boxes to be equivalent to the 1D selection expressions, the must span the entire Y axis 
                This can be done manually assuming a constant image size
                '''
                
                src_width1=(src_coord2-src_coord1)
                src_ctr1=src_coord1+src_width1/2
                
                bg_width1=(bg_coord2-bg_coord1)
                bg_ctr1=bg_coord1+bg_width1/2
                
                reg_name=camera+suffix_evt+prefix+'_reg.reg'
                with open(fulldir+'/'+reg_name,'w+') as regfile:
        
                    #standard ds9 format
                    regfile.write('# Region file format: DS9 version 4.1'+
                                  '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1'+
                                  ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+
                                  '\nimage'+
                                  '\nbox('+src_ctr1+',90,'+src_width1+',179) # text={manually selected source}'+
                                  '\nbox('+bg_ctr1+',90,'+bg_width1+',179) # text={manually selected background}')
                
                ds9_pid_sp_reg=disp_ds9(spawn,file_img,regfile=reg_name,screenfile=fulldir+'/'+camera+suffix_evt
                                        +prefix+'_reg_screen.png',give_pid=True,kill_last=ds9_pid_sp_start)

                
            elif mode=='auto':
                
                '''
                For the timing automode, we simply stack the 2D Timing image into a 1D histogram to get the photon distribution
                in function of the x coordinate. We can then extract suitable source and background limits.
                We arbitrarily choose to limit the source region to the 1 sigma upper end of the counts distribution
                In the same way, we fix the background region to the 2 sigma bottom end of the counts distribution
                '''
                
                print('\nAuto mode.')
                
                prefix='_auto'
                
                #But first, we check that the target of the timing observation is not too far from our own object.
                if timing_check:
                    obj_auto=source_catal(fulldir)
                    
                    #checking if the function returned an error message (folder movement done in the function)
                    if type(obj_auto)==str:
                        return obj_auto
                    
                    #if not, we compare the distance to the coordinates given in the file's header
                    obj_deg=sexa2deg([obj_auto['DEC'].replace(' ',':'),obj_auto['RA'].replace(' ',':')])[::-1]
                    
                    #to the theoretical pointing coordinates
                    target_obj_deg=fits_evtclean[0].header['RA_OBJ'],fits_evtclean[0].header['DEC_OBJ']
                    
                    #to the average coordinates
                    target_avg_deg=fits_evtclean[0].header['RA_PNT'],fits_evtclean[0].header['DEC_PNT']
                    
                    #computing the angular distance in arcsecs from both of those
                    dist_target_obj_catal=(np.sum(((obj_deg-target_obj_deg)*3600))**2)**(1/2)
                    dist_target_avg_catal=(np.sum(((obj_deg-target_avg_deg)*3600))**2)**(1/2)
                    
                    #most objects with correct pointings will have a very small obj to catal distance
                    #if they don't, the rage pointing to catal distance acts as a failsafe 
                    if dist_target_obj_catal>30 and dist_target_avg_catal>60:
                        print('\nTiming position check activated and the catalog position is too far from the target.\nSkipping...')
                        spawn.sendline('\ncd $currdir')
                        return 'Catalog position too far from the exposure target.'
                    
                #fetching any eventual discrepancy between the RAWX and IMAGE X axis
                try:
                    rawx_offset=fits_img[1].header['LTV1']
                except:
                    try:
                        rawx_offset=fits_img[0].header['LTV1']
                    except:
                        print('\nRAWX offset keyword missing in image fits file. Skipping region analysis...')
                        spawn.sendline('\ncd $currdir')
                        return 'RAWX offset keyword missing in image fits file.'
                    
                #stacking on one dimension
                distrib_timingimg=np.sum(fits_img[0].data,axis=0)
                
                #testing if the distribution is suspicious
                if distrib_timingimg.argmax() in [0,len(distrib_timingimg)-1]:
                    print('\nPeak value on the edge of the CCD. Exposure problematic. Skipping region analysis...')
                    spawn.sendline('\ncd $currdir')
                    return 'Peak value on the edge of the CCD.'
                    
                def opti_snr_timing(distrib_timing,excised=False,ds9_pid_prev=''):
                
                    '''                   
                    The SNR optimisation uses the following steps:
                        
                    0. Compute a bg region purely from the data distribution, compute the bg lightcurve in the energy 
                    band we're interested in
                    1. Compute a list of source regions centered on the brightest pixel and increasing of 1 pixel in width
                    (adding the brightest pixel not included in the region each time)

                    2. For each of the source region we're interested in':
                        a. Compute a source lightcurve in the energy band we're interested in
                        b. Compute the exposure slicing leading to the best SNR
                    
                    The region/exposure slicing with the best SNR is retained
                    
                    We then remake the event lists according to the new cut and plot graphs for everything
                    
                    We do not include the pile-up computation in the SNR optimisation because if we computed the pile-up for every region, 
                    it would be a lot harder to determine a limit to the excision without biasing towards bright or faint regions
                    '''

                    if excised:
                        add_str='_excised'
                    else:
                        add_str=''
                        
                    print('\nStarting snr optimisation process...')
                    
                    #ordering a copy to easily search the percentile
                    #Also, since a stack shouldn't have a zero value even if it is background, the 0 pixels can be removed
                    distrib_sorted=np.copy(distrib_timing)
                    distrib_sorted=distrib_sorted[distrib_sorted.nonzero()]
                    distrib_sorted.sort()
                    
                    #Background region definition using the non excised distribution (no need to change it)
                    distrib_initial_sorted=np.copy(distrib_timingimg[distrib_timingimg.nonzero()])
                    distrib_initial_sorted.sort()
                    distrib_id_bg=np.argwhere(distrib_timingimg<= distrib_initial_sorted[int(0.05*len(distrib_initial_sorted))])
                    distrib_id_bg=np.array([elem[0] for elem in distrib_id_bg])
                    
                    #since some images don't have borders but instead have a huge range of 0 pixels on the sides, we make sure
                    #to only get pixel ids with no 0 total (something that should not be possible in the normal image)
                    distrib_id_bg=np.intersect1d(distrib_id_bg,distrib_timingimg.nonzero()[0])
                    
                    #computing the bg expression and the unoptimised lightcurve
                    expression_bg=spatial_expression(distrib_id_bg,rawx_off=rawx_offset)

                    path_lc_bg_snr=make_lc_snr(file_init,'bg_snr',expression_bg,lc_bins)
                    
                    #and loading it
                    lc_bg_snr=fits.open(path_lc_bg_snr)[1].data
                    
                    #list of source pixel regions candidate, starting with the brightest pixel
                    distrib_id_snr=[[np.argwhere(distrib_timing==max(distrib_timing))[0][0]]]
                    
                    #copy of the distribution that will be progressively altered
                    distrib_copy=np.copy(distrib_timing)
                    #the -2 values correspond to pixels excised due to pile-up
                    distrib_copy=distrib_copy[distrib_copy!=-2]
                    
                    #putting the brightest pixel to -1
                    #This pixel will be used as a proxy of the source region's position in the distribution
                    distrib_copy[distrib_copy==max(distrib_copy)]=-1
                    
                    #we (hopefully) go further than the max snr value for plotting purposes
                    while len(distrib_id_snr[-1])<(len(distrib_timing[distrib_timing.nonzero()])/5):
                        
                        #fetching the current index of the source region pixel proxy
                        curr_src_pos=np.argwhere(distrib_copy==-1)[0][0]
                        
                        #computing the biggest value in the two pixels nearest to the current source region
                        if curr_src_pos==len(distrib_copy)-1:
                            bright_pixel=distrib_copy[curr_src_pos-1]
                        elif curr_src_pos==0:
                            bright_pixel=distrib_copy[curr_src_pos+1]
                        else:
                            bright_pixel=max(distrib_copy[curr_src_pos-1],distrib_copy[curr_src_pos+1])
                        
                        #addind the new source region candidate to the list
                        distrib_id_snr+=[np.sort(distrib_id_snr[-1]+\
                                    [np.argwhere(distrib_timing==bright_pixel)[0][0]]).tolist()]
                            
                        #cutting the modified pixel distribution of the which just got added to the region
                        distrib_copy=np.delete(distrib_copy,np.argwhere(distrib_copy==bright_pixel)[0][0])

                    'starting step 2.'                    
                    #array of source lightcurve paths
                    paths_lc_src_snr=np.array([None]*len(distrib_id_snr))
                    
                    #array of optimized gti events
                    gti_arr=np.array([None]*len(distrib_id_snr))
                    
                    #array of optimized snrs
                    src_snr_arr=np.zeros(len(distrib_id_snr))
                    
                    #loop for each source region candidate
                    for ind,curr_distrib in enumerate(distrib_id_snr):
                        
                        print('\nOptimising region '+str(ind+1))
                        #creating the expression
                        curr_expression_src=spatial_expression(curr_distrib,rawx_off=rawx_offset)
                        
                        #creating the lightcurve
                        paths_lc_src_snr[ind]=make_lc_snr(file_init,'src_snr_'+f"{ind+1:02}",curr_expression_src,lc_bins)
                        
                        #loading it
                        curr_lc_src_snr=fits.open(paths_lc_src_snr[ind])[1].data
                        
                        #skipping the computations if it is empty
                        if curr_lc_src_snr['RATE'].sum()==0:
                            src_snr_arr[ind]=0
                            continue
                        
                        #area ratio between the source and bg
                        area_norm=len(curr_distrib)/len(distrib_id_bg)
                        
                        #launching the optimisation for this source/bg combination
                        gti_arr[ind],src_snr_arr[ind]=opti_single(curr_lc_src_snr,lc_bg_snr,area_norm)
                        
                    if len(src_snr_arr.nonzero()[0])==0:
                        print('\nCould not compute SNR for a single region. Exiting...')
                        spawn.sendline('\ncd $currdir')
                        if excised==False:
                            return None,None,None,None,None
                        else:
                            return None,None,None,None
                    
                    #fetching the best source region, snr and gtis after optimisation
                    #if we by miracle have identical value we get the latest, because bigger regions will always be better for the 
                    #pile-up excision
                    
                    bestind=np.argwhere(src_snr_arr==max(src_snr_arr))[-1][0]
                    distrib_id_src=distrib_id_snr[bestind]
                    bestgti=gti_arr[bestind]
                    
                    #displaying the results
                    print('\n Best SNR after optimisation: '+str(max(src_snr_arr))
                          +'\n attained with a region width of '+str(len(distrib_id_src))+' pixels and while excluding '+
                          str(round(100*(1-(len(bestgti)/len(curr_lc_src_snr))),3))+'% of the exposure.')
                    
                    opti_events(bestgti,paths_lc_src_snr[bestind],path_lc_bg_snr,area_ratio=len(distrib_id_src)/len(distrib_id_bg),
                                add_str=add_str)
                    
                    '''Source fit plots'''
                        
                    #plotting both the crop/gaussian fit and the SNR evolution
                    fig_snr_src,ax_snr_src=plt.subplots(1,1,figsize=(12,10))
                    
                    if excised:
                        plot_title_str=''
                    else:
                        plot_title_str='excised '
                        
                    ax_snr_src.set_title("SNR evolution after optimising the SNR of "+plot_title_str+
                                            " source regions of increasing widths, centered on the brightest pixel")
                    ax_snr_src.set_xlabel('Region width (pixels)')
                    ax_snr_src.set_ylabel("Maximum SNR after gti optimisation, as computed in Heasoft's specgroup\n"+
                                             r"($S/N=(src-bg*area_{norm})/\sqrt{src+bg*area_{norm}^2}$)")
                    
                    #plotting the snr evolution
                    ax_snr_src.plot(range(1,len(src_snr_arr)+1),src_snr_arr,color='black')
                    
                    #plotting the points with a colormap for the amount of deleted exposure
                    gti_ratios=[100*len(gti_arr[i])/len(curr_lc_src_snr) if type(gti_arr[i])!=type(None) else 0\
                                for i in range(len(gti_arr))]
                    scatter_snr_src=ax_snr_src.scatter(range(1,len(src_snr_arr)+1),src_snr_arr,
                                                          label='optimized SNR',c=gti_ratios,cmap='plasma',\
                                                          vmin=min(gti_ratios)*0.9,vmax=100,zorder=1000)
                    plt.colorbar(scatter_snr_src,label='% of exposure retained after SNR filtering')
                    #highlighting the best radius
                    ax_snr_src.vlines(len(distrib_id_src),0,max(src_snr_arr),color='pink',linestyle='--',label='peak width',linewidth=2)
                    
                    #readjusting the plot to avoid useless space at negative SNR
                    range_snr=max(src_snr_arr)-min(src_snr_arr)
                    ax_snr_src.set_ylim(max(min(src_snr_arr)-range_snr*0.05,0),max(src_snr_arr)+range_snr*0.05)
        
                    plt.legend()
                    plt.tight_layout()
                    
                    plt.savefig(fulldir+'/'+camera+suffix_evt+prefix+'_opti'+add_str+'_screen.png')
                    plt.close()
                    
                    '''
                    Computing the final source expression
                    '''
                    
                    file_img_cleaned=fulldir+'/'+camera+'img'+suffix_evt+'_snr'+add_str+'.ds'
                    #fetching the new distribution for the pile-up computation
                    distrib_timing_cleaned=np.sum(fits.open(file_img_cleaned)[0].data,axis=0)

                    #creating a summary variable to be used in the pile-up
                    dist_summary=[distrib_timing_cleaned,distrib_id_src]
                    
                    expression_source=spatial_expression(distrib_id_src,rawx_off=rawx_offset)
    
                    '''
                    now we can create the equivalent ds9 region (we do this inside the function because in timing it's easier to revwrite
                                                                 the entire reg file than to edit it, unlike in imaging)
                    '''
    
                    reg_name=camera+suffix_evt+prefix+'_reg.reg'
                            
                    with open(fulldir+'/'+reg_name,'w+') as regfile:
            
                        #standard ds9 formatccd info xmm
                        regfile.write('# Region file format: DS9 version 4.1'+
                                      '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1'+
                                      ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+
                                      '\nimage')
                        regfile.write(spatial_expression(distrib_id_src,type='ds9',timing_fits=fits_img))
                        regfile.write(spatial_expression(distrib_id_bg,type='ds9',timing_fits=fits_img,
                                                         timing_ds9type='bg'))
                            
                    #When we perform the SNR optimisation after the pile-up computation, there's no need to rescreen and thus
                    #return a new ds9 pid
                    if excised==False:
                        ds9_pid=disp_ds9(spawn,file_img_cleaned,regfile=reg_name,screenfile=fulldir+'/'+camera+suffix_evt
                                                +prefix+'_reg'+add_str+'_screen.png',give_pid=True,kill_last=ds9_pid_prev)
                        
                        return expression_source,expression_bg,dist_summary,ds9_pid,max(src_snr_arr)
                    else:
                        return expression_source,expression_bg,dist_summary,max(src_snr_arr)
                
                #waiting for the ds9 window to be launched
                ds9launch=False
                while ds9launch==False:
                    try:
                        ds9_pid_sp_start=ds9_pid_sp_start
                        ds9launch=True
                    except:
                        time.sleep(1)
                    
                expression_source,expression_bg,distrib_summary,ds9_pid_sp_reg,best_SNR=\
                opti_snr_timing(distrib_timingimg,ds9_pid_prev=ds9_pid_sp_start)
                
                #testing if the process went through (directory change already made in the function error checks)
                if expression_source==None:
                    return 'Could not compute the SNR for a single region.'
        
        elif expos_mode_single=='IMAGING':
        
            if mode=='manual':
                print("\nManual mode for Imaging exposure:"+
                      "\nInput the fk5 coordinates of the source and bg circular regions you've chosen.")
                prefix=input("\nPrefix to be added in the files ?\n\n")
                
                print("\nSource :")
                src_center1=input("\nFirst coordinate of the center (degrees)?\n\n")
                src_center2=input("\nSecond coordinate of the center (degrees)?\n\n")
                src_radius=input("\nRadius of the circle (arcsec)?\n\n")
    
                print("\nBackground :")
                bg_center1=input("\nFirst coordinate of the center (degrees)?\n\n")
                bg_center2=input("\nSecond coordinate of the center (degrees)?\n\n")
                bg_radius=input("\nRadius of the circle (arcsec)?\n\n")
                
                if prefix!='':
                    prefix='_'+prefix
                
                #summary variables
                src_coords_man=[[src_center1,src_center2],src_radius]
                bg_coords_man=[[bg_center1,bg_center2],bg_radius]
                
                reg_name=camera+suffix_evt+prefix+'_reg.reg'
                with open(fulldir+'/'+reg_name,'w+') as regfile:
        
                    #standard ds9 format
                    regfile.write('# Region file format: DS9 version 4.1'+
                                  '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1'+
                                  ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+
                                  '\nfk5'+
                                  '\n'+spatial_expression(src_coords_man,type='ds9')+' # text={manually selected source}'+
                                  '\n'+spatial_expression(bg_coords_man,type='ds9')+' # text={manually selected background}')
                
                #getting back to actual degree values
                src_coords_man[1]=str(round(float(src_coords_man[1])/3600,8))
                bg_coords_man[1]=str(round(float(bg_coords_man[1])/3600,8))  
                
                ds9_pid_sp_reg=disp_ds9(spawn,file_img,regfile=reg_name,screenfile=fulldir+'/'+camera+suffix_evt
                                        +prefix+'_reg_screen.png',give_pid=True,kill_last=ds9_pid_sp_start)
                   
                #in the example threads they also add gti selection but it shouldn't matter if we directly use cleaned event files
                expression_source=spatial_expression(src_coords_man)
                expression_bg=spatial_expression(bg_coords_man)
            
            elif mode=='auto':
                
                print('\nAuto mode.')
                print('\nAutomatic search of the directory names in Simbad.')
                
                prefix='_auto'
                
                obj_auto=source_catal(fulldir)
                
                #checking if the function returned an error message (folder movement done in the function)
                if type(obj_auto)==str:
                    return obj_auto
                    
                #careful the output after the first line is in dec,ra not ra,dec
                obj_deg=sexa2deg([obj_auto['DEC'].replace(' ',':'),obj_auto['RA'].replace(' ',':')])
                obj_deg=[str(obj_deg[1]),str(obj_deg[0])]

                #loading the fits file with MPDAF has to be done after a preliminary fits load since the format isn't accepted
                src_mpdaf_WCS=mpdaf_WCS(fits_img[0].header)
                src_astro_WCS=astroWCS(fits_img[0].header)
                img_whole=Image(data=fits_img[0].data,wcs=src_mpdaf_WCS)

                #saving a screen of the image with the cropping zone around the catalog position highlighted
                catal_reg_name=camera+suffix_evt+prefix+'_catal_reg.reg'
                
                with open(fulldir+'/'+catal_reg_name,'w+') as regfile:
                    
                    #standard ds9 format
                    regfile.write('# Region file format: DS9 version 4.1'+
                                  '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1'+
                                  ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+
                                  '\nfk5'+
                                  '\n'+spatial_expression((obj_deg,str(rad_crop_arg)),type='ds9')
                                  +' # text={'+obj_auto['MAIN_ID']+' initial cropping zone}')
                
                ds9_pid_sp_start=disp_ds9(spawn,file_img,regfile=catal_reg_name,zoom=1.2,
                                        screenfile=fulldir+'/'+camera+suffix_evt+prefix+'_catal_reg_screen.png',give_pid=True,
                                        kill_last=ds9_pid_sp_start)
                
                #cropping the image to avoid zoom in the future plots
                #(size is double the radius since we crop at the edges of the previously cropped circle)
                rad_crop=rad_crop_arg
                
                try:
                    imgcrop_src=img_whole.copy().subimage(center=obj_deg[::-1],size=2*rad_crop)
                except:
                    print('\nCropping region entirely out of the image. Field of view issue. Skipping this exposure...')
                    spawn.sendline('\ncd $currdir')
                    return 'Cropping region entirely out of the image.'
                
                #masking the desired region
                imgcrop_src.mask_region(center=obj_deg[::-1],radius=rad_crop,inside=False)
                
                #testing if the resulting image is empty
                if len(imgcrop_src.data.nonzero()[0])==0:
                    print('\nEmpty image after cropping. Field of view Issue. Skipping this exposure...')
                    spawn.sendline('\ncd $currdir')
                    return 'Cropped image empty.'
                
                #plotting and saving imgcrop for verification purposes (if the above has computed, it should mean the crop is in the image)
                fig_catal_crop,ax_catal_crop=plt.subplots(1,1,subplot_kw={'projection':src_astro_WCS},figsize=(12,10))   
                ax_catal_crop.set_title('Cropped region around the theoretical source position')
                catal_plot=imgcrop_src.plot(cmap='plasma',scale='sqrt')
                plt.colorbar(catal_plot,location='bottom',fraction=0.046, pad=0.04)
                plt.savefig(fulldir+'/'+camera+suffix_evt+prefix+'_catal_crop_screen.png')
                plt.close()
                
                #testing if the resulting image contains a peak
                if imgcrop_src.peak()==None:
                    print('\nNo peak detected in cropped image. Skipping this exposure...')
                    spawn.sendline('\ncd $currdir')
                    return 'No peak in cropped image.'
                
                #fitting a gaussian on the source (which is assumed to be the brightest in the cropped region)
                #the only objective here is to get the center, the radius will be computed from the SNR
                print('\nExecuting gaussian fit...')
                
                if point_source:
                    source_center=(imgcrop_src.peak()['y'],imgcrop_src.peak()['x'])
                else:
                    source_center=None
                gfit=imgcrop_src.gauss_fit(center=source_center)
                gfit.print_param()
                #defining various bad flags on the gfit
                if np.isnan(gfit.fwhm[0]) or np.isnan(gfit.fwhm[1]) or np.isnan(gfit.err_peak) or gfit.peak<10 or gfit.err_peak>=gfit.peak:
                    
                    print('\nGaussian fit failed. Positioning or Field of View issue. Skipping these Evts...')
                    spawn.sendline('\ncd $currdir')
                    return 'Gaussian fit failed. Positioning or Field of View issue.'
                
                if not imgcrop_src.inside(gfit.center):
                    
                    print('\nGaussian fit centroid out of the crop zone. Wrong fit expected. Skipping these Evts...')
                    spawn.sendline('\ncd $currdir')
                    return 'Gaussian fit centroid further than '+str(rad_crop)+'" from catal source position.'
                
                
                #if the source is bright and wide, the gaussian is probably going to be able to fit it even if we increase the cropping region
                #And the fit might need a bigger image to compute correctly
                
                if max(gfit.peak,imgcrop_src.data.max())>1000 and bigger_fit and max(gfit.fwhm)>30:                    

                    rad_crop=2*rad_crop_arg
                    #new, bigger cropping the image to avoid zoom in the future plots
                    #(size is double the radius since we crop at the edges of the previously cropped circle)
                    imgcrop_src=img_whole.copy().subimage(center=obj_deg[::-1],size=2*rad_crop)
                    
                    #masking a bigger region
                    imgcrop_src.mask_region(center=obj_deg[::-1],radius=rad_crop,inside=False)
                    
                    #fitting a gaussian on the source (which is assumed to be the brightest in the cropped region)
                    print('\nExecuting gaussian fit...')
                    gfit=imgcrop_src.gauss_fit()
                    gfit.print_param()
                    
                    if np.isnan(gfit.fwhm[0]) or np.isnan(gfit.fwhm[1]):
                        
                        print('\nExtended (bright source) gaussian fit failed. Positioning or Field of View issue. Skipping these Evts...')
                        spawn.sendline('\ncd $currdir')
                        return 'Extended (bright source) gaussian fit failed. Positioning or Field of View issue.'
                
                    if not imgcrop_src.inside(gfit.center):
                        
                        print('\nGaussian fit centroid out of the crop zone. Wrong fit expected. Skipping these Evts...')
                        spawn.sendline('\ncd $currdir')
                        return 'Gaussian fit centroid further than '+str(2.5*rad_crop)+'" from catal source position.'

                #summary variable, first version which the background will be computed from, with a radius of 3sigmas of the gaussian fit
                src_coords=[[str(gfit.center[1]),str(gfit.center[0])],str(round(max(gfit.fwhm)*(3/2.355)/3600,8))]
                
                def opti_bg_imaging():
                    '''
                    Now, we evaluate the background from at least the entire CCD containing the gaussian center.
                    In order to do that, we detect the CCD with ecoordconv, which gives the CCD spanning a region or point.
                    If the result with the 3 sigma source region spans more than 4 CCDs, we consider the source as "big" and 
                    include all of the CCDs instead of just the central one
                    '''
    
                    print('\nComputing the source region CCD(s)...')
                    
                    spawn.sendline('\necoordconv imageset='+file_img+' srcexp="'+spatial_expression(src_coords)+'"')
                    spawn.expect('CCD')
                    #reading the correct line from the bash command output
                    CCD_line=spawn.readline()
                    CCD_ctr=CCD_line.split('CCD: ')[1].split("\r")[0]
    
                    #in this specific observing mode for MOS, there is a partial central mask and the exterior ones.
                    #As such, we consider thje others CCDs for the background instead
                    if submode=='PrimePartialW2':
                        CCD_numbers=['2','3','4','5','6','7']
                        CCD_expr=['CCDNR=='+elem+' || ' for elem in CCD_numbers]
                        CCD_expr[-1]=CCD_expr[-1].split(' ')[0]
                        CCD_expr=''.join(CCD_expr)
                    else:
                        #also reading the CCDs the region spans
                        CCD_cams=CCD_line.split('centred')[0].split(' ')         
                        CCD_numbers=[]
                        for elem in CCD_cams:
                            temp=elem
                            if temp.isdigit():
                                CCD_numbers.append(temp)
                        
                        print('\nThe detected source spans '+str(len(CCD_numbers))+' CCDs.')
                        if len(CCD_numbers)>4:
                            print('\nMore than 4 CCDs means that this must be a big source.')
                            print('\nCropping all of the source region CCDs.')     
                            CCD_expr=['CCDNR=='+elem+' || ' for elem in CCD_numbers]
                            CCD_expr[-1]=CCD_expr[-1].split(' ')[0]
                            CCD_expr=''.join(CCD_expr)
                        else:
                            print('\nCropping the CCD of the source centroid.')
                            CCD_expr='CCDNR=='+CCD_ctr
                    
                    CCD_ctr=CCD_line.split('CCD: ')[1].split("\r")[0]
                    
                    print('\nGenerating an image restricted to the CCD(s) of interest...')
                    spawn.sendline('\nevselect table='+file+' imagebinning=binSize imageset='+camera+'img'+suffix_evt
                                   +'_ccdcrop.ds '+'withimageset=yes xcolumn=X ycolumn=Y ximagebinsize=80 yimagebinsize=80 '+
                                   'expression="('+CCD_expr+')"')
                    
                    spawn.expect('selected')
                    
                    #Waiting for the file to be created before starting the next part (very ugly)
                    while not os.path.isfile(fulldir+'/'+camera+'img'+suffix_evt+'_ccdcrop.ds'):
                        time.sleep(1)
                    
                    '''
                    After cropping the image with the CCD index, we load it with fits in order to perform image manipulation on it
                    and compute how big the background region can be.
                    
                    The process is as follow :
                    1. Compute the alpha shape of the non-zero pixels (i.e. polygon that surrounds them) in the CCD with alphashape
                    2. Transform this polygon into a mask with rasterio 
                    3. delete the part outside of the ccd  (outlined by the polygon) by replacing the values outside with nans
                        ->This step is important because even in the base image, all of the pixels outside of the CCD have 0 cts/s
                          They have to be changed to avoid wrong sigma clipping
                    4. add sigma clipping to make sure the background region won't touch any source
                    5. transform the background array into a mask
                    6. compute the poi (see the reg_optimiser function)
                    '''
                    
                    CCD_fits=fits.open(fulldir+'/'+camera+'img'+suffix_evt+'_ccdcrop.ds')
                    CCD_data=CCD_fits[0].data
    
                    if len(CCD_data.nonzero()[0])==0:
                        print("\nEmpty bg image after CCD cropping. Skipping this observation...")
                        spawn.sendline('\ncd $currdir')
                        return "Empty bg image after CCD cropping."
                    
                    print('\nSaving the corresponding image...')
                    savepng(CCD_data,'vis_'+camera+suffix_evt+'_CCD_1_crop',astropy_wcs=src_astro_WCS,mpdaf_wcs=src_mpdaf_WCS,
                            directory=fulldir,title='Source image with CCDs cropped according to the region size and center')
                    
                    #listing non-zero pixels in the CCD
                    CCD_on=np.argwhere(CCD_data!=0)
                    
                    #transforming that into a digestible argument for the alphashape function
                    CCD_on=[tuple(elem) for elem in CCD_on]
                    
                    print('\nComputing the CCD mask...')
                    
                    '''
                    Computation of the alphashape. Alpha defines how tight the polygon is around the points cloud.
                    In most cases we use a conservative 0.1 value to avoid creating holes in our shape.
                    
                    However, in some specific MOS modes there's a huge hole in the middle (i.e. the missing CCD1).
                    In this case, we use a more aggressive alphashape and then dilation to get back an approximative CCD shape 
                    with the hole
                    
                    Since the alpha needed to get the hole varies, we do an iteration base on the value of the central pixel of
                    the alpha shape one translated in mask. If the hole was created, it should be set to false.
                    
                    The goal here is to avoid too much altering of the initial CCD shape, since a 'too' high alphashape parameter 
                    will considerably restrict the resulting polygon
                    '''
                    
                    if submode!='PrimePartialW2':
                        CCD_shape=alphashape(CCD_on,alpha=0.1)
                        
                        #converting the polygon to a mask
                        CCD_mask=rasterize([CCD_shape],out_shape=CCD_data.shape).T.astype(bool)
                    else:
                        #initalisaing the variables for the loop
                        hole_exists=False
                        alphashape_iter=0.1
                        
                        while hole_exists==False:
                            CCD_shape=alphashape(CCD_on,alpha=alphashape_iter)
                            
                            #converting the polygon to a mask
                            CCD_mask=rasterize([CCD_shape],out_shape=CCD_data.shape).T.astype(bool)
                            CCD_shape_ctr=np.array(CCD_shape.centroid.bounds[:2]).astype(int)
                            hole_exists=not CCD_mask[CCD_shape_ctr[0],CCD_shape_ctr[1]]
                            alphashape_iter+=0.1
                            
                        CCD_mask=binary_dilation(CCD_mask,iterations=10)
                        
                    print('\nSaving the corresponding image...')
                    savepng(CCD_mask,'vis_'+camera+suffix_evt+'_CCD_2_mask',astropy_wcs=src_astro_WCS,mpdaf_wcs=src_mpdaf_WCS,
                            directory=fulldir,title='Source image CCD(s) mask after clipping and filling of the holes')
                    
                    print('\nComputing the CCD masked image...')
                    #array which we will have the outside of the CCD masked with nans
                    CCD_data_cut=np.copy(CCD_data).astype(float)
                    #This other array stores the values inside the CCD, for an easy evaluation of the sigma limits
                    CCD_data_line=[]
                    for i in range(np.size(CCD_data,0)):
                        for j in range(np.size(CCD_data,1)):
                            if not CCD_mask[i][j]:
                                CCD_data_cut[i][j]=np.nan
                            else:
                                CCD_data_line.append(CCD_data_cut[i][j])
                                
                    print('\nSaving the corresponding image...')
                    savepng(CCD_data_cut,'vis_'+camera+suffix_evt+'_CCD_3_cut',astropy_wcs=src_astro_WCS,mpdaf_wcs=src_mpdaf_WCS,
                            directory=fulldir,title='Source image after CCD masking',imgtype='ccd_crop')
                    
                    #sigma cut, here at 0.95 (2 sigma) which seems to be a good compromise
                    CCD_data_line.sort()
                    
                    #for some extreme cases we have only 1 count/pixel max, in which case we don't want that
                    cut_sig=max(CCD_data_line[int(0.95*len(CCD_data_line))],1.)
                    
                    sigval='2'
                    perval='5'
                    
                    #sometimes for very bright sources there might be too much noise so we cut at 1 sigma instead
                    if cut_sig>20:
                        cut_sig=CCD_data_line[int(0.68*len(CCD_data_line))]
                        sigval='1'
                        perval='32'
    
                    print('\nComputing the CCD bg mask...')
                    #array which will contain the background mask
                    CCD_bg=np.copy(CCD_data_cut)
                    for i in range(np.size(CCD_data,0)):
                        for j in range(np.size(CCD_data,1)):
                            if CCD_data_cut[i][j]<=cut_sig:
                                CCD_bg[i][j]=1
                            else:
                                #masking the pixels above the treshold
                                CCD_bg[i][j]=0
                                
                    print('\nSaving the corresponding image...')
                    savepng(CCD_bg,'vis_'+camera+suffix_evt+'_CCD_4_bg',astropy_wcs=src_astro_WCS,mpdaf_wcs=src_mpdaf_WCS,
                            directory=fulldir,title='Source image background mask remaining after '+sigval+' sigma (top '+perval+
                            '% cts) counts removal',imgtype='ccd_crop_mask')
                    
                    bg_max_pix=reg_optimiser(CCD_bg)
                    
                    print('\nMaximal bg region coordinates in pixel units:')
                    print(bg_max_pix)
                    
                    '''
                    finally, we swap the region coordinates and radius to angular values, with ecoordconv.
                    '''
                    
                    spawn.sendline('\necoordconv imageset='+file_img+' coordtype=impix '+
                                   'x='+str(bg_max_pix[0][0])+' y='+str(bg_max_pix[0][1]))
                    spawn.expect('DEC')
                    bg_ctr_line=spawn.readline()
                    bg_max=[[bg_ctr_line.split(' ')[1],bg_ctr_line.split(' ')[2].split('\r')[0]],
                                   str(round(bg_max_pix[1]/900,8))]
                    return bg_max
                
                bg_coords_im=opti_bg_imaging()

                #returning the error message if there is one instead of the expected values (directory change done in function)
                if type(bg_coords_im)==str:
                    return bg_coords_im
                
                #defining the xmm selection expression
                expression_bg=spatial_expression(bg_coords_im)
                
                #creating a bg region cropped event lightcurve to compute the bg counts, from the first evtclean file
                path_lc_bg_snr=make_lc_snr(file_init,'bg_snr',expression_bg,lc_bins)

                #loading it
                lc_bg_snr=fits.open(path_lc_bg_snr)[1].data
                
                '''
                Source SNR optimisation
                '''

                def copy_gfit(ax1,ax2):
                    
                    '''
                    copies all linecollections (which are assumed to be from gfit) from ax1 to ax2
                    '''
                    
                    # -- Function to copy, or rather, transfer, attributes
                    #It copies the attributes given in attr_list (a sequence of the attributes names as
                    #  strings) from the object 'obj1' into the object 'obj2'
                    #It should work for any objects as long as the attributes are accessible by
                    # 'get_attribute' and 'set_attribute' methods.
                    def copy_attributes(obj2, obj1, attr_list):
                        for i_attribute  in attr_list:
                            getattr(obj2,'set_' + i_attribute)( getattr(obj1, 'get_' + i_attribute)() )

                    # #Returns a list of pairs (attribute string, attribute value) of the given 
                    # # attributes list 'attr_list' of the given object 'obj'                
                    # def get_attributes(obj, attr_list):
                    #     attr_val_list = []
                    #     for i_attribute  in attr_list:
                    #         i_val = getattr(obj, 'get_' + i_attribute)()
                    #         attr_val_list.append((i_attribute, i_val))
                    #     
                    #     return attr_val_list
                    
                    #Returns a list of the transferable attributes, that is the attributes having
                    # both a 'get' and 'set' method. But the methods in 'except_attributes' are not
                    # included
                    def list_transferable_attributes(obj, except_attributes):
                        obj_methods_list = dir(obj)
                    
                        obj_get_attr = []
                        obj_set_attr = []
                        obj_transf_attr =[]
                    
                        for name in obj_methods_list:
                            if len(name) > 4:
                                prefix = name[0:4]
                                if prefix == 'get_':
                                    obj_get_attr.append(name[4:])
                                elif prefix == 'set_':
                                    obj_set_attr.append(name[4:])
                    
                        for attribute in obj_set_attr:
                            if attribute in obj_get_attr and attribute not in except_attributes:
                                obj_transf_attr.append(attribute)
                    
                        return obj_transf_attr

                    except_attributes = ('transform', 'figure','capstyle','contains','joinstyle','paths')
                    
                    for children in ax1.get_children():
                        
                        #checking that the object is a line collection
                        if type(children)==mpl.collections.LineCollection:
                            
                            #obtaining the names of the transferable attributes
                            tr_attr=list_transferable_attributes(children, except_attributes)
                            
                            #creating an empty line collection
                            new_line=LineCollection([],[])
                            
                            copy_attributes(new_line,children,tr_attr)
                            ax2.add_collection(new_line)
                            
                    
                def opti_snr_imaging(excis_rad=0.):
            
                    '''                   
                    The SNR optimisation uses the following steps:
                        
                    0. Compute the bg lightcurve in the energy band we're interest in 
                    1. Compute a range of circular regions around the gfit center with increasing radius up to the size of the 
                    inital image crop
                    
                    for each:
                        a. Compute a source/bg lightcurve in the energy band we're interested in
                        b. Compute the exposure slicing leading to the best SNR
                    
                    The region/exposure slicing with the best SNR in the allowed range is retained
                    (range limited to a certain number of PSF sigmas for faint sources, not limited for bright -i.e. with gfit flux > 5000 ones)

                    
                    We then remake the event lists according to the new cut and plot graphs for everything
                    
                    We do not include the pile-up computation in the SNR optimisation because if we computed the pile-up for every region, 
                    it would be a lot harder to determine a limit to the excision without biasing towards bright or faint regions
                    '''
                    
                    if excis_rad!=0.:
                        add_str='_excised'
                    else:
                        add_str=''
                        
                    print('\nStarting snr optimisation process...')
                    
                    #defining a range of radius to create regions from
                    pixtoarc=imgcrop_src.get_step()[0]*3600

                    
                    #if the maximal excisable pile-up value is above 3/4 of the maximal allowed area of the source, 
                    #we directly increase it
                    
                    #else, we only increase it if the excision radius has been maxed out during the first iteration
                    
                    if pileup_treshold==None:
                        if (max(gfit.fwhm)*pileup_max_ex/2.355)**2>0.5*rad_crop_arg**2 or excis_rad==max(gfit.fwhm)*pileup_max_ex/2.355:
                            radlim_par=rad_crop
                        else:
                            radlim_par=rad_crop_arg
                    else:
                        if abs(excis_rad-(rad_crop_arg-1))<1e-3:
                            radlim_par=rad_crop
                        else:
                            radlim_par=rad_crop_arg

                    #We divide everything by two to have half pixel steps at least
                    rad_arr=np.arange(pixtoarc,radlim_par*2,pixtoarc)/2
                    if abs(rad_arr[-1]-float(radlim_par))>1e-2:
                        rad_arr=np.append(rad_arr,radlim_par)
                    
                    print('\nExploring radius range:')
                    print(rad_arr)
                    #array of source lightcurve paths
                    paths_lc_src_snr=np.array([None]*len(rad_arr))
                    
                    #array of optimized gti events
                    gti_arr=np.array([None]*len(rad_arr))
                    
                    #array of optimized snrs
                    src_snr_arr=np.zeros(len(rad_arr))
                    
                    for rad_ind,rad in enumerate(rad_arr):
                        
                        print('\nOptimising region '+str(rad_ind+1))
                        #defining an expression for evselect
                        if excis_rad==0.:
                            curr_expression_src=spatial_expression((src_coords[0],str(round(rad/3600,8))))
                        else:
                            curr_expression_src=spatial_expression((src_coords[0],str(round(rad/3600,8))),
                                                                   excision_radius=str(round(excis_rad/3600,8)))
                            
                        #creating a source region cropped event lightcurve to compute the counts, from the first evtclean file
                        paths_lc_src_snr[rad_ind]=make_lc_snr(file_init,'src_snr_'+f"{rad_ind+1:02}",curr_expression_src,lc_bins)
                        
                        #loading it
                        curr_lc_src_snr=fits.open(paths_lc_src_snr[rad_ind])[1].data
                        
                        #waiting a bit to make sure the file logging is up to date
                        time.sleep(1)
                        
                        #skipping the computations if it is empty

                        if curr_lc_src_snr['RATE'].sum()==0:
                            src_snr_arr[rad_ind]=0
                            continue

                        #area ratio between the source and bg
                        area_norm=(rad**2-excis_rad**2)/((float(bg_coords_im[1])*3600)**2)
                        
                        #launching the optimisation for this source/bg combination
                        gti_arr[rad_ind],src_snr_arr[rad_ind]=opti_single(curr_lc_src_snr,lc_bg_snr,area_norm)
                    
                    #defining the limit of the radius for faint sources
                    if gfit.flux<5000:
                        id_src_radlim=np.abs(rad_arr-max(gfit.fwhm)*maxrad_source/2.355).argmin()
                    else:
                        id_src_radlim=None
                        
                    #fetching the best SNR and its corresponding radius and exposure in the allowed radius range
                    #if by miracle we have identical value we get the latest, because bigger regions will always be better for the 
                    #pile-up excision                    
                    best_snr=max(src_snr_arr[:id_src_radlim])
                    bestrad_ind=np.argwhere(src_snr_arr==best_snr)[-1][0]
                    bestrad=round(rad_arr[bestrad_ind],3)
                    bestgti=gti_arr[bestrad_ind]

                    print('\n Best SNR in allowed range after optimisation: '+str(best_snr)
                          +'\n attained with a radius of '+str(bestrad)+'" and while excluding '+
                          str(round(100*(1-(len(bestgti)/len(curr_lc_src_snr))),3))+'% of the exposure.')
                    
                    #computing the new position of the gaussian fit center in imgcrop_src image pixels
                    #we use imgcrop_src here to avoid a different gaussian fit before and after the pile-up excision
                    gcenter_pix=imgcrop_src.gauss_fit(unit_center=None).center[::-1]
                    
                    opti_events(bestgti,paths_lc_src_snr[bestrad_ind],path_lc_bg_snr,area_ratio=(bestrad**2-excis_rad**2)\
                                /((float(bg_coords_im[1])*3600)**2),add_str=add_str)
                    
                    #loading the unexcised image for the new gauss fit 
                    #(this will not change no matter if this is the post-pileup excision run or not)
                    fits_img_nonexcised=fits.open(fulldir+'/'+camera+'img'+suffix_evt+'_snr.ds')
                    imgcrop_nonexcised=Image(data=fits_img_nonexcised[0].data,wcs=src_mpdaf_WCS)\
                        .subimage(center=obj_deg[::-1],size=2*rad_crop)
                    imgcrop_nonexcised.mask_region(center=obj_deg[::-1],radius=rad_crop,inside=False)
                    
                    #new gaussian fit
                    clean_gfit=imgcrop_nonexcised.gauss_fit()
                    clean_gfit.print_param()
                    
                    #new src coords (in the excision run only the bestrad value will differ)
                    clean_src_coords=[[str(clean_gfit.center[1]),str(clean_gfit.center[0])],
                                      str(round(bestrad/3600,8))]
                    gcenter_pix=imgcrop_nonexcised.gauss_fit(unit_center=None).center[::-1]
                    
                    '''Source fit plots'''

                    #reloading the current image
                    fits_img_cleaned=fits.open(fulldir+'/'+camera+'img'+suffix_evt+'_snr'+add_str+'.ds')
                    imgcrop_cleaned=Image(data=fits_img_cleaned[0].data,wcs=src_mpdaf_WCS).subimage(center=obj_deg[::-1],size=2*rad_crop)
                    imgcrop_cleaned.mask_region(center=obj_deg[::-1],radius=rad_crop,inside=False)
                    
                    #masking the excision radius in excision mode
                    if excis_rad!=0:
                        imgcrop_cleaned.mask_region(center=clean_gfit.center,radius=excis_rad,inside=True)
                        
                    #plotting both the crop/gaussian fit and the SNR evolution
                    ax_snr_src=np.array([None,None])
                    fig_snr_src,ax_snr_src=plt.subplots(1,2,subplot_kw={'projection':src_astro_WCS},figsize=(16,8))
                    
                    ax_snr_src[1].remove()
                    
                    #first plot is the source crop
                    ax_snr_src[0].set_title('Cropped region around the theoretical source position and gaussian fit')
                    
                    fig_snr_img=imgcrop_cleaned.plot(cmap='plasma',scale='sqrt')
                    cb=plt.colorbar(fig_snr_img,location='bottom',fraction=0.046, pad=0.04)
                    cb.set_label('Counts')
                    
                    #adding the gaussian fit
                    if excis_rad==0.:
                        imgcrop_cleaned.gauss_fit(plot=True)
                    else:
                        #since we can't get the same gfit on a different image, we copy it manually after recreating it with the 
                        #copy_gfit function                    
                        figtemp=plt.figure()
                        imgcrop_nonexcised.plot()
                        imgcrop_nonexcised.gauss_fit(plot=True)
                        copy_gfit(plt.gca(),ax_snr_src[0])

                    #adding a circle for the bestrad value
                    bestrad_circle=plt.Circle(gcenter_pix,radius=bestrad/pixtoarc,fill=False,color='pink',linestyle='--',linewidth=3,
                                              zorder=1000)
                    ax_snr_src[0].add_patch(bestrad_circle)
                    
                    #creating the snr evolution plot
                    ax_snr_src[1]=fig_snr_src.add_subplot(1,2,2)
                    if excis_rad!=0.:
                        plot_title_str='annular'
                        plot_title_str2=' outer'
                    else:
                        plot_title_str='circular'
                        plot_title_str2=''
                        
                    ax_snr_src[1].set_title("SNR evolution of "+plot_title_str+
                                            " regions centered on the gaussian's peak")
                    ax_snr_src[1].set_xlabel('Region'+plot_title_str2+' radius in arcsecs')
                    ax_snr_src[1].set_ylabel("Maximum SNR after gti optimisation, as computed in Heasoft's specgroup\n"+
                                             r"($S/N=(src-bg*area_{norm})/\sqrt{src+bg*area_{norm}^2}$)")
                    
                    #plotting the snr evolution
                    ax_snr_src[1].plot(rad_arr,src_snr_arr,color='black')
                    
                    #plotting the points with a colormap for the amount of deleted exposure
                    #here we assume all of the lightcurve have the same length and thus take an unspecific one to get their length
                    gti_ratios=np.array([100*len(gti_arr[i])/len(curr_lc_src_snr) if type(gti_arr[i])!=type(None) else 0\
                                         for i in range(len(gti_arr))])
                    
                    scatter_snr_src=ax_snr_src[1].scatter(rad_arr,src_snr_arr,label='optimized SNR',c=gti_ratios,cmap='plasma',\
                                                          vmin=min(gti_ratios)*0.9,vmax=100,zorder=1000)
                    plt.colorbar(scatter_snr_src,ax=ax_snr_src[1],label='% of exposure retained after SNR filtering')
                    
                    #highlighting the best radius
                    #the pink one is the one retained as the final radius value, the grey one is just there as information
                    ax_snr_src[1].vlines(bestrad,0,best_snr,color='pink',linestyle='--',
                                         label='peak SNR radius in allowed range',linewidth=2)
                    
                    #we do not bother showing the actual 6 sigma index unless it's used
                    if best_snr!=max(src_snr_arr):
                        ax_snr_src[1].vlines(rad_arr[id_src_radlim],0,src_snr_arr[id_src_radlim],
                                             color='grey',linestyle='--',
                                             label='point closest to \n'+r'$6\sigma$ radius for the PSF',linewidth=2)
                    #readjusting the plot to avoid useless space at negative SNR
                    range_snr=max(src_snr_arr)-min(src_snr_arr)
                    ax_snr_src[1].set_ylim(max(min(src_snr_arr)-range_snr*0.05,0),max(src_snr_arr)+range_snr*0.05)
                   
                    if excis_rad!=0.:
                        plt.close(figtemp)
                        
                    ax_snr_src[1].legend(loc=2)
                    
                    plt.tight_layout()
                    
                    plt.savefig(fulldir+'/'+camera+suffix_evt+prefix+'_opti'+add_str+'_screen.png')
                    plt.close()
                    
                    if excis_rad==0.:
                        return clean_src_coords,clean_gfit,max(src_snr_arr)
                    else:
                        return clean_src_coords,max(src_snr_arr)

                
                #first optimisation run without excising
                src_coords,gfit,best_SNR=opti_snr_imaging()
                        
                #standard XMM expression for imaging
                expression_source=spatial_expression(src_coords)
                
                '''
                For bright sources with the center pixels already masked, the pile-up computation is biased if we only consider a 
                circle region with the 'off' pixels included.
                Consequently, we search for such a feature in the source region and use an annulus instead of a circle if it exists
                '''
                
                source_intr_offrad=0
                
                #restricting the process to bright sources only
                if gfit.flux>5000:
                    
                    #loop for detecting central masked regions                            
                    center_is_off=True
                    while center_is_off:
                        
                        #creating a subimage at the very center of the source region (which will correspond to the excised region)
                        #with increasing size
                        try:
                            imgcrop_src_excis=imgcrop_src.subimage(center=gfit.center,size=source_intr_offrad+3)
                            
                            #testing if all the pixels in this center region are off
                            center_is_off=np.all(imgcrop_src_excis.data.data==0)
                            
                            #if they are, we increase the subimage size and start again
                            if center_is_off:
                                source_intr_offrad+=1
                        except:
                            #this can happen if the bigger image falls outside of the image boundaries. In this case we don't extend
                            center_is_off=False
                            pass

                    
                #the final value of source_intr_offrad (if it's not 0) has to be adjusted for the +3
                #since this is the first non-zero array, by excising it technically we lose a few pixels, but if the source is masked
                #there is surely an insane amount of pile-up anyway so it's not a problem
                if source_intr_offrad!=0:
                    source_intr_offrad+=3
                    
                    #replacing the circle expression with an annulus
                    expression_source=spatial_expression(src_coords,excision_radius=str(source_intr_offrad/3600))
                    
                    #and adding a visualiser to ds9, in green since it's not the pile-up yet
                    excis_reg_string_intr='\n-'+spatial_expression((src_coords[0],str(source_intr_offrad)),type='ds9')
                else:
                    excis_reg_string_intr=''
            
                '''
                Now we can put both of the regions in a ds9 file using the standard format
                '''
                
                reg_name=camera+suffix_evt+prefix+'_reg.reg'
                
                with open(fulldir+'/'+reg_name,'w+') as regfile:
                    
                    #standard ds9 format
                    regfile.write('# Region file format: DS9 version 4.1'+
                                  '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1'+
                                  ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+
                                  '\nfk5'+
                                  '\n'+spatial_expression((src_coords[0],str(round(float(src_coords[1])*3600,4))),type='ds9')
                                  +' # text={'+obj_auto['MAIN_ID']+'}'+
                                  '\n'+spatial_expression((bg_coords_im[0],str(round(float(bg_coords_im[1])*3600,4))),type='ds9')
                                  +' # text={automatic background}'+
                                  excis_reg_string_intr)
                
                ds9_pid_sp_reg=disp_ds9(spawn,file_img.replace('.ds','_snr.ds'),regfile=reg_name,
                                        screenfile=fulldir+'/'+camera+suffix_evt+prefix+'_reg_screen.png',give_pid=True,
                                        kill_last=ds9_pid_sp_start)
                    
        def pileup_util(expression_source,distrib_summary=None,src_coords=None,excise=True,max_ex=pileup_max_ex):
            
            '''
            Computes the pile-up for the event file and, if excise is set to true, recreate source regions with more and more
            excised radii up to the chosen condition is satisfied.
            Then, returns the final pile-up values

            IF there is no pile-up percentage limit setup (pileup_treshold set to None):
                -Stops excising when either the pileup is compatible with 0 or the last iteration didn't improve the pileup
                -For timing, stops at most at half of the source area. 
                -For imaging, stops at most at the 3 sigma radius of the PSF 
                
            In the other case:
                -Only stops excising when the given treshold is attained
                -No spatial limit to the excision besides the source size.

            for imaging, excises a linear range of circular regions up to the minimal value
            for timing, excises widths one pixel by one up to half the size of the source region
            '''
            
            def compute_pileup(exp_source):
                
                '''
                computes the pile-up from a spatial source expression and returns the resulting pile-up line
                if the computation went through
                '''
                
                #computing the filtered event list for the pileup
                
                #since file is been updated at each opti_snr_imaging() and timing() iteration, the computation will difer each time 
                #it changes
                spawn.sendline('\nevselect table='+file+' withfilteredset=yes filteredset='
                               +camera+suffix_evt+prefix+'_pileup.ds '+
                               'keepfilteroutput=yes expression="'+exp_source+'"')
                spawn.expect('selected')
                
                #computing the pileup
                spawn.sendline('\nepatplot set='+camera+suffix_evt+prefix+'_pileup.ds withqdp=yes '+
                               'plotfile='+camera+suffix_evt+prefix+'_pileup_plot.ps')
        
                epatplot_state=spawn.expect(['observed-to-model fractions:','epatplot: error','(emptyEvents)'],timeout=None)
                
                #we return an empty line if there was an error in the epatplot computation
                if epatplot_state==0:
                    spawn.readline()
                    return spawn.readline().split('\r')[0]
                else:
                    return ''
                
            is_pileup=True
            
            p_iter=0
            
            #if we have a defined pileup limit, the pileup computation has to continue even if one or two iterations do not manage to improve the 
            #pileup
            if pileup_treshold!=None:
                force_pileup=True
            else:
                force_pileup=False

            #the maximum number of iterations depends on the mode.
            
            #for imaging we compute the max iter from the size of the source
            if expos_mode_single=='IMAGING':
                if source_intr_offrad!=0:
                    pileup_start_rad=source_intr_offrad+5
                else:
                    pileup_start_rad=3
                
                #if there is no pile-up percentage limit, we use a PSF distribution limit for the maximal radius
                if pileup_treshold==None:
                    pileup_max_rad=int(max(gfit.fwhm)*pileup_max_ex/2.355)
                else:
                    #we can go up to a single arcsecond of annulus remaining
                    pileup_max_rad=round(float(src_coords[1])*3600)-1
                print('\n Maximal pileup-size set to '+str(pileup_max_rad))
                pileup_rad_space=np.arange(pileup_start_rad,pileup_max_rad+1)
                p_max=len(pileup_rad_space)
                
            elif expos_mode_single in ['TIMING','BURST']:
                if pileup_treshold==None:
                    p_max=int(len(distrib_summary[1])*1/2)
                else:
                    #we can go up to a single pixel remaining
                    p_max=int(len(distrib_summary[1])-1)

            reg_name=camera+suffix_evt+prefix+'_reg.reg'

            #defined here to avoid errors
            excis_reg_string_list=[]
            expression_source_list=[expression_source]
            excis_pixels_list=[]           
            pileup_line_list=[]
            pileup_val_list=[]
            
            #the loop is coded so that the first pileup computation happens no matter what 
            while is_pileup:
        
                if p_iter==0:
                    pileup_str=['','']
                else:
                    pileup_str=[' still',' once more']
                
                new_pileup_line=compute_pileup(expression_source_list[-1])
                
                if new_pileup_line=='':
                    print('\nNo signal detected in the pile-up computation. '
                          +'There is probably something wrong with the source region.')
                    print('\nSkipping spectrum computation for this exposure...')
                    spawn.sendline('\ncd $currdir')
                    #we use the pileup_lines argument for transmitting the error
                    return None,'No signal',None
                
                pileup_line_list+=[new_pileup_line]
                pileup_val_list+=[pileup_val(new_pileup_line)]
                
                #showing the procession of the pileup line list
                print(pileup_line_list)
                
                #first test before any condition modify this statement
                is_pileup=pileup_bool(pileup_line_list[-1])
                
                if is_pileup==None:
                    is_pileup=False
                
                #this step is here to force the pile_up computation to continue for imaging bright sources with incomplete masks 
                #when no pileup limit is given
                if expos_mode_single=='IMAGING' and gfit.peak>1000:
                    force_pileup=True

                #testing the pileup progression when no limit is given
                if p_iter!=0 and force_pileup==False:
                    
                    if pileup_val_list[-1]>=pileup_val_list[-2]:
                        print('\nThe excision is not improving the pile-up anymore.'+\
                              ' Recreating the previous step and stoppingthe process...')
                            
                        #deleting the last iteration
                        excis_reg_string_list=excis_reg_string_list[:-1]
                        expression_source_list=expression_source_list[:-1]
                        excis_pixels_list=excis_pixels_list[:-1]   
                        pileup_line_list=pileup_line_list[:-1]
                        pileup_val_list=pileup_val_list[:-1]
                            
                        is_pileup=False
                
                #testing the value itself when a treshold is given
                if pileup_treshold!=None:
                    if pileup_val_list[-1]<=pileup_treshold:
                        print('\nPile-up value below the given treshold. Stopping the pileup improvement process...')
                        is_pileup=False
                                
                if is_pileup and excise and p_iter<p_max:
                    print('\nPile-up'+pileup_str[0]+' detected. Attempting to improve the source region'+pileup_str[1]+'...')
                    
                    if expos_mode_single in ['TIMING','BURST']:
                        
                        '''
                        In timing mode, the source region is only a few pixel wide, so we take out 1 more pixel for each 
                        iteration
                        '''
                        
                        #here if the mode is manual we need to redo a few commands that weren't done before
                        if mode=='manual':
                            
                            #new distrib
                            fits_img_cleaned=fits.open(fits_img.filename().replace('.ds','_cleaned.ds'))
                            
                            #stacking on one dimension
                            distrib_timing_cleaned=np.sum(fits_img_cleaned[0].data,axis=0)
                            
                            #offset of 1 since those are originally ds9 axis values and not pixel positions
                            distrib_id_src=np.arange(src_coords[0][0]-1,src_coords[0][1]-1)
                    
                        else:
                            distrib_timing_cleaned=distrib_summary[0]
                            distrib_id_src=distrib_summary[1]

                            
                        #new iteration
                        excis_pixels_list+=[distrib_timing_cleaned.argsort()[-1-p_iter:]]
                        
                        #recreating the expression for the newer version of the pixel intervals, without the newly excised ones
                        distrib_id_excised=[elem for elem in distrib_id_src if elem not in excis_pixels_list[-1]]
                        
                        expression_source_list=[spatial_expression(distrib_id_excised,rawx_off=rawx_offset)]
                        excis_reg_string_list+=[spatial_expression(excis_pixels_list[-1],type='ds9',timing_ds9type='pile-up',
                                                            timing_fits=fits_img)]
                    
                    if expos_mode_single=='IMAGING':
                
                        '''
                        In imaging, we simply make an annulus with linear radius progression of 1 arcsec up to max_ex*the source area
                        '''

                        #we get back in degrees since the XMM selection expressions take degrees
                        expression_source_list+=[spatial_expression(src_coords,excision_radius=str(pileup_rad_space[p_iter]/3600))]
                        
                        #ds9 region string
                        excis_reg_string_list+=['\n-'+spatial_expression((src_coords[0],str(pileup_rad_space[p_iter])),type='ds9')\
                            +' # color=cyan \n']
                else:
                    is_pileup=False
                
                p_iter+=1
                
            #deleting the last pileup line if it was doubled 
            if len(pileup_line_list)>1 :
                if pileup_line_list[-1]==pileup_line_list[-2]:
                    pileup_line_list==pileup_line_list[:-1]

            #limiting the second round of computations to cases where there has been an excision
            if len(excis_reg_string_list)!=0:
                
                #doing a new SNR optimisation for the excised source region
                if expos_mode_single in ['TIMING','BURST']:
                    
                    #recreating a timing array with the pile-up pixel highlighted
                    #we recreate it from the INITIAL distribution and not the cleaned one, else the computation would be biased
                    distrib_timing_excised=np.copy(distrib_timingimg)
                    distrib_timing_excised[excis_pixels_list[-1]]=-2
                    
                    #launching the function (here the region edition is directly in the function)
                    expression_source,expression_bg,distrib_summary,best_SNR=opti_snr_timing(distrib_timing_excised,excised=True)
                    
                    #testing if the second round of optimisation converged
                    if expression_source==None:
                        #we use the pileup_lines argument for transmitting the error
                        return None,'After pile-up excision, could not compute SNR for a single region.',None
                    
                elif expos_mode_single=='IMAGING':
                    
                    #masking the excised part of a copy of the cropped image
                    rad_excision=float(expression_source_list[-1].split(',')[-2])*3600
                    
                    #launching the function
                    src_coords_excised,best_SNR=opti_snr_imaging(excis_rad=rad_excision)
                    
                    #replacing the source outer radius of the source expression with the new value
                    expression_source=spatial_expression(src_coords_excised,excision_radius=str(round(rad_excision/3600,8)))
                    
                    #replacing the source outer radius value in the ds9 region file
                    with open(fulldir+'/'+reg_name,'r') as regfile:
                        reg_lines=regfile.readlines()
                        
                    #the source region should be the fourth line in the file
                    reg_lines[3]=reg_lines[3].replace(reg_lines[3].split(',')[-1].split('"')[0],
                                                      str(round(float(src_coords_excised[1])*3600,4)))
                    
                    #rewritting the file with the new line
                    with open(fulldir+'/'+reg_name,'w+') as regfile:
                        for reg_line in reg_lines:
                            regfile.write(reg_line)
                

                
                # last pile-up computation with the new source expression
                pileup_line_list.append(compute_pileup(expression_source))
            
                #editing the ds9 region file to add the excised region
                with open(fulldir+'/'+reg_name,'r') as regfile:
                    reg_lines=regfile.readlines()
                
                #here we add the line as the very first region to have it on top of the source region(s)
                reg_lines.insert(3,excis_reg_string_list[-1])
                with open(fulldir+'/'+reg_name,'w+') as regfile:
                    for reg_line in reg_lines:
                        regfile.write(reg_line)
                    
                #and replacing the screen, with a log scale to aid visualisation 
                disp_ds9(spawn,file_img.replace('.ds','_snr_excised.ds'),regfile=reg_name,
                         screenfile=fulldir+'/'+camera+suffix_evt+prefix+'_reg_excised_screen.png',kill_last=ds9_pid_sp_reg)

            else:
                best_SNR=''
            return expression_source,np.array(pileup_line_list),best_SNR
            
        #fuck enclosing scope
        if mode=='manual' or expos_mode_single=='IMAGING':
            expression_source,pileup_lines,excised_SNR=pileup_util(expression_source,src_coords=src_coords,excise=pileup_ctrl)
        elif expos_mode_single in ['TIMING','BURST']:
            expression_source,pileup_lines,excised_SNR=pileup_util(expression_source,distrib_summary=distrib_summary,excise=pileup_ctrl)
        
        #overwriting the SNR if there was a new SNR computation
        if excised_SNR!='':
            best_SNR=excised_SNR
        
        #The cd command has already been done in the pileup_util function or in the opti_snr_timing function
        if type(pileup_lines)==str:
            return pileup_lines
        
        results_name=camera+suffix_evt+prefix+'_regex_results.txt'
        
        #storing the results of the function in a file for future use
        with open(fulldir+'/'+results_name,'w+') as results_file:
            line0='expression_source'+'\t'+expression_source+'\n'
            results_file.write(line0)
            
            line1='expression_bg'+'\t'+expression_bg+'\n'
            results_file.write(line1)
            
            line2='pileup_lines'+'\t'+','.join(pileup_lines)+'\n'
            results_file.write(line2)
            
            line3='Source SNR'+'\t'+str(best_SNR)+'\n'
            results_file.write(line3)
            
        print('\nRegion extraction complete')
        
        spawn.sendline('\ncd $currdir')
        
        return 'Region extraction complete.'
    
    if cams=='all':
        camid_regex=[0,1,2]
    else:
        camid_regex=[]
        if 'pn' in cams:
            camid_regex.append(0)
        if 'mos1'in cams:
            camid_regex.append(1)
        if 'mos2' in cams:
            camid_regex.append(2)
            
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nRegion extraction...')
    
    bashproc.sendline('cd '+directory)
    bashproc.sendline('mkdir -p batch')
    set_var(bashproc,directory)
       
    #recensing the cleaned event files available for each camera
    #camfilelist shape : [[pnfiles,pndirs],[mos1files,mos1dirs],[mos2files,mos2dirs]]
    clean_filelist=file_selector(directory,'evt_clean')
    
    #summary file header
    if directory.endswith('/'):
        obsid=directory.split('/')[-2]
    else:
        obsid=directory.split('/')[-1]
    summary_header='Obsid\tFile identifier\tRegion extraction result\n'
    
    #filtering for the selected cameras
    for i in camid_regex:
        
        for j in range(len(clean_filelist[i][0])):
            
            clean_evtfile=clean_filelist[i][0][j]
            clean_evtdir=clean_filelist[i][1][j]
            
            #testing if the last file of the spectrum extraction process has been created and moved into the batch or merging directory
            lastfile_auto=clean_evtfile.split('.')[0][clean_evtfile.find('_'):]
            lastfile_auto=camlist[i]+lastfile_auto+'_'+mode+'_regex_results.txt'
            
            if (mode=='manual' or overwrite \
            or 'm' not in action_list and not os.path.isfile(directory+'/'+clean_evtdir+'/'+lastfile_auto)\
            or 'm' in action_list and not os.path.isfile(startdir+'/'+mergedir+'/'+obsid+'_'+lastfile_auto))\
            and clean_evtfile!='':
                
                clean_evtid=clean_evtfile.split('.')[0].replace('clean','')
                
                #setting up a logfile in parallel to terminal display :
                
                if os.path.isfile(clean_evtdir+'/'+clean_evtid+'_extract_reg.log'):
                    os.system('rm '+clean_evtdir+'/'+clean_evtid+'_extract_reg.log')
                with StdoutTee(clean_evtdir+'/'+clean_evtid+'_extract_reg.log',
                               mode="a",buff=1,file_filters=[_remove_control_chars]),\
                    StderrTee(clean_evtdir+'/'+clean_evtid+'_extract_reg.log',buff=1,file_filters=[_remove_control_chars]):
                
                    bashproc.logfile_read=sys.stdout
                    print('\nCreating spectrum of '+camlist[i]+' exposure '+clean_evtfile)
                    
                    #launching the main extraction
                    summary_line=extract_reg_single(bashproc, clean_evtfile,clean_evtdir)

            else:
                if clean_evtfile=='':
                    print('\nNo evt to extract region from for camera '+camlist[i]+ ' in the obsid directory.')
                    
                    summary_line='No evt to extract region from for camera '+camlist[i]+ ' in the obsid directory.'
                    clean_evtid=camlist[i]
                else:
                    print('\nAuto mode region computation for the '+camlist[i]+' exposure '+clean_evtfile+
                          ' already done. Skipping...')
                    summary_line=''
                    
            if summary_line!='':
                summary_content=obsid+'\t'+clean_evtid+'\t'+summary_line
                file_edit(os.path.join(directory,'batch','summary_extract_reg.log'),obsid+'\t'+clean_evtid,summary_content+'\n',
                          summary_header)
            
    bashproc.sendline('\necho "Ph0t1n0s" |sudo -S pkill sudo')
    #this sometimes doesn't proc before the exit for whatever reason so we add a buffer just in case
    # bashproc.expect([pexpect.TIMEOUT],timeout=2)
        
    #closing the spawn
    bashproc.sendline('exit')
    
    print('\nRegion extraction of the current obsid directory events finished.')
        
    extract_reg_done.set()

def plot_lc(lc_src_path,lc_bg_path=None,area_ratio=1,save=True,close=True,mode='flare',plotcut=False,hflare=False,flarelim='std'):
    
    '''
    Plots the equivalent of the Grace dsplot window for lightcurves 
    In rate mode, also plots the current pn/mos automatic rate limits

    The "save" keyword saves the plot in rate_path's directory
    The "close" keyword closes the plot (after saving)
    
    plotcut allows plotting the rate limit (given by the std value if flarelim is set to std, else given by flarelim)
    
    hflare allows up highlighting the part of the bacgkround lightcurve that is categorized as flares in the filter_evt auto mode
    '''    
    if 'excised' in mode:
        add_str='_excised'
    else:
        add_str=''
    #we redefine everything to be able to use this anywhere
    
    #a few infos on the lc path
    lc_src_file=lc_src_path.split('/')[-1]
    
    if mode=='flare' or mode=='rate_flare':
        lc_id=lc_src_file.split('rate')[1].split('.')[0]
        lc_cam=lc_id.split('_')[0]
    elif 'lc' in mode:
        lc_id=lc_src_file.split('_lc')[0]
        lc_cam=lc_id.split('_')[0]
        
    #opening the fits file and getting a few others infos
    try:
        fits_lc_src=fits.open(lc_src_path)
    except:
        print('Could not read the fits source lightcurve file')
        return 0
    lc_obsid=fits_lc_src[0].header['OBS_ID']
    data_lc_src=fits_lc_src[1].data
    
    # #Fetching the SNR gti mask if the mask exists
    # if os.path.isfile(lc_src_path[:lc_src_path.rfind('lc')]+'gti_flare.ds') and ('cut' in mode or 'broad' in mode):
    #     gti_mask=~fits.open(lc_src_path[:lc_src_path.rfind('lc')]+'gti_flare.ds')[1].data['IS_GTI'].astype(bool)
    # else:
    #     gti_mask=np.repeat(False,len(fits_lc_src[1].data))
        
    #the actual figure
    fig_lc,ax_lc=plt.subplots(1,figsize=(14,10))
    
    if 'broad' not in mode:
        if 'rate' in mode:
            lc_ener_str='[10-12] keV' if lc_cam=='pn' else '>=10 keV'
        else:
            lc_ener_str='['+args.flareband+']'+' keV'
        if lc_cam=='pn':
            lc_autolim_val=mosflare_lim
        elif lc_cam=='mos1' or lc_cam=='mos2':
            lc_autolim_val=pnflare_lim
    else:
        lc_ener_str='[0.3-10] keV'
        
    if flarelim!='std':
        lc_autolim_val=flarelim
        
    #title definition
    prefix=''
    
    if 'flare' in mode:
        prefix+='Flares '
        
        if 'rate' in mode:
            prefix+='CCD '
            selection_str='XMMEA valid and PATTERN 0'
        else:
            selection_str='with reliable pattern/flags selected'
    elif 'snr' in mode:
        selection_str='XMMEA valid and PATTERN 0'
    else:
        selection_str='reliable pattern/flags selection and flares filtered'
        
    if 'corr' in mode:
        prefix+='Corrected '
            
    if mode!='flare':
        if mode.split('_')[1]=='src':
            prefix+='Source '
        
    if lc_bg_path!=None:
        prefix=prefix[:-1]
        prefix+='/Background '
        
    fig_lc.suptitle(prefix+'light curve with '+selection_str+' in the '+lc_ener_str+' band for expos '+lc_id+' of obsid '+lc_obsid)

    ax_lc.set_xlabel('Time after exposure start(ks)')
    ax_lc.set_ylabel('Count rate (count/s)')
    
    if lc_bg_path!=None:
        fits_lc_bg=fits.open(lc_bg_path)
        data_lc_bg=fits_lc_bg[1].data
        
        #adjusting the time to get readable values
        x_bg=(data_lc_src['TIME']-data_lc_src['TIME'][0])/1e3
    
        #converting the bg y array to a masked array to be able to erase the values switched off by the gti
        y_bg_ma=np.ma.array(data_lc_bg['RATE']*1/area_ratio)
        
        #shouldn't be needed now that we actually filter the gti
        #we only mask the gti if not in flare mode 
        # y_bg_ma.mask=gti_mask if ('flare' not in mode or 'cut' in mode) else False
        
        
        #plotting
        ax_lc.errorbar(x_bg,y_bg_ma,yerr=data_lc_bg['ERROR']*1/area_ratio,
                     label='Background Count rate\nrenormalized to the source area',color='grey')
        
        label_source_prefix='Source '
    else:
        label_source_prefix=''
        
    #adjusting the time to get readable values
    x_src=(data_lc_src['TIME']-data_lc_src['TIME'][0])/1e3
    
    #converting the source  y array to a masked array to be able to erase the values switched off by the gti
    y_src_ma=np.ma.array(data_lc_src['RATE'])
    
    # #we only mask the gti if not in flare mode 
    # y_src_ma.mask=gti_mask if ('flare' not in mode or 'cut' in mode) else False
        

    ax_lc.errorbar(x_src,y_src_ma,yerr=data_lc_src['ERROR'],label=label_source_prefix+'Count rate',color='blue')
        
    if 'flare' in mode and plotcut:
        ax_lc.axhline(y=lc_autolim_val,color='black',linestyle='dashed',label='standard limit for this camera')
    
    #highlighting the parts of the graph where the bg curve is detected as flare-like (see filter-evt)
    if hflare:
    
        #and the y lims we will fill between
        y_lims=ax_lc.get_ylim()
        
        if 'hist_auto' in mode:
            #storing the flare truth value of each bg point in th graph
            flare_bools=data_lc_src['RATE']-data_lc_src['ERROR']-1/area_ratio*(data_lc_bg['RATE']+data_lc_bg['ERROR'])\
                                  <data_lc_src['RATE']/2
            label_highlight='high background intervals'
            
        elif 'hist_snr' in mode:
            #fetching the boolean gti array
            flare_bools=~fits.open(lc_src_path.replace('lc_src','gti'))[1].data['IS_GTI'].astype(bool)
            label_highlight='suboptimal SNR intervals'
        elif 'snr' in mode:
            #fetching the boolean gti array
            flare_bools=~fits.open(lc_src_path[:-6].replace('lc_src_snr','gti_bool_snr')+'.ds')[1].data['IS_GTI'].astype(bool)
            label_highlight='suboptimal SNR intervals computed for this region'
        if 'excised' not in mode:
            #easy highlight
            ax_lc.fill_between(x_src,y_lims[0],y_lims[1],where=flare_bools,alpha=0.2,color='red',label=label_highlight)
            
        if 'snr_excised' in mode:
            #highlighting the changes compared to the unexcised zones
            flare_bools_cleaned=~fits.open(lc_src_path[:-6].replace('lc_src_snr','gti_bool_snr')+add_str+'.ds')\
                [1].data['IS_GTI'].astype(bool)
            added_flares=[flare_bools_cleaned[i]==1 and flare_bools_cleaned[i]!=flare_bools[i] for i in range(len(flare_bools_cleaned))]
            substracted_flares=[flare_bools_cleaned[i]==0 and flare_bools_cleaned[i]!=flare_bools[i]\
                                for i in range(len(flare_bools_cleaned))]
            ax_lc.fill_between(x_src,y_lims[0],y_lims[1],where=added_flares,
                               alpha=0.2,color='yellow',label='added suboptimal SNR intervals')
            ax_lc.fill_between(x_src,y_lims[0],y_lims[1],where=substracted_flares,
                               alpha=0.2,color='green',label='reinstated optimal SNR intervals')
        
    plt.legend()
    plt.tight_layout()
    if save:
        
        screen_path=lc_src_path.replace('.ds','.png')
        
        #different name for combined plots
        if lc_bg_path!=None:
            screen_path=screen_path.replace('src','comb')
        if 'cut' in mode:
            screen_path=screen_path.replace('.png','_cut.png')
        if 'hist_auto' in mode:
            screen_path=screen_path.replace('.png','_hist_auto.png')
        if 'hist_snr' in mode:
            screen_path=screen_path.replace('.png','_hist_snr.png')
        if 'src_snr' in mode:
            screen_path=screen_path[:-7]+add_str+'.png'

        screen_path=screen_path.replace('.png','_screen.png')
        
        plt.savefig(screen_path)
        
        if close:            
            plt.close(fig_lc)

def extract_lc(directory,mode='manual',cams='all',expos_mode='all',overwrite=True):
    
    '''
    Extracts the lightcurve of detected event files for one or several cameras and one of several modes
    
    The restriction happens through the 'cams' and 'expmode' keywords (strings containing all of the cams/modes names)

    Also computed raw event files lightcurves restricted to high energy (mimicking the flaring rate curve)
    
    As of now, only takes input formatted through the evt_filter function
    
    The automatic mode requires spectra created with the extract_sp function
    (because it copies the selection expression in the header)
    
    manual mode (beyond just reusing already existing regions with other prefixes) to be added
    
    '''
    
    def extract_lc_single(spawn,file,filedir):
        
        if file=='':
            print('\nNo evt to extract lightcurve from for this camera in the obsid directory.')
            return 'No evt to extract lightcurve from for this camera in the obsid directory.'
        
        fulldir=directory+'/'+filedir

        #opening the fits file to extract some informations on the exposure
        fits_evtclean=fits.open(fulldir+'/'+file)
        
        try:
            if fits_evtclean[1].header['ONTIME']==0:
                print('\nEmpty exposure.')
                return 'Empty exposure'
        except:
                print('\nEmpty exposure.')
                return 'Empty exposure'
            
        expos_mode_single=fits_evtclean[0].header['DATAMODE']
        # submode=fits_evtclean[0].header['SUBMODE']
        print('\nexpos mode:',expos_mode_single)
        camera=fits_evtclean[0].header['INSTRUME'][1:].swapcase()
        
        #quitting the extraction if the exposure mode is not in the selected ones
        if expos_mode_single not in expos_mode_lcex:
            return 'Exposure mode not among the selected ones.'
        
        #identifying the exposure number and mode to create the suffix_evt
        #different method than in filter_evt because here we already know the structure of the file name
        suffix_evt=file.split('.')[0][file.find('_'):]
        
        
        if mode=='auto':
            prefix='_auto'
        else:
            prefix=input("\nPrefix with region already saved?\n\n")
            
        #checking if the region extraction file exists
        regex_name=camera+suffix_evt+prefix+'_regex_results.txt'
        
        if not os.path.isfile(fulldir+'/'+regex_name):
            print('region extraction result file missing. Skipping...')
            return 'region extraction result file missing.'
        
        #extracting the information inside
        with open(fulldir+'/'+regex_name,'r') as regex_file:
            regex_lines=regex_file.readlines()
            
        spawn.sendline('\ncurrdir=$(pwd)')
        spawn.sendline('\ncd '+filedir)
        
        src_exp=regex_lines[0].split('\t')[1].replace('\n','')
        bg_exp=regex_lines[1].split('\t')[1].replace('\n','')
        
        #standard part of the command expressions:
        if expos_mode_single in ['TIMING','BURST']:
            if camera=='pn':
                expression_sp='(FLAG==0) && (PATTERN<=4) && '
            elif camera in {'mos1','mos2'}:
                expression_sp='(FLAG==0) && (PATTERN<=0) && '
        elif expos_mode_single=='IMAGING':
            if camera=='pn':
                expression_sp='(FLAG==0) && (PATTERN<=4) && '
            elif camera in {'mos1','mos2'}:
                expression_sp='#XMMEA_EM && (PATTERN<=12) && '
        
        #broad expressions
        src_exp_broad=expression_sp+src_exp
        bg_exp_broad=expression_sp+bg_exp
        
        #since we stopped using arbitrary limits we use the pattern/flag selection of the broad band spectrum
        # #changing the pattern selection for the flare lightcurves to get a similar selection to the rate curve made in filter_evt
        # exp_flare_cam='#XMMEA_EP' if camera=='pn' else '#XMMEA_EM'
        # src_exp_flare=exp_flare_cam+' && (PATTERN==0) && '+src_exp
        # bg_exp_flare=exp_flare_cam+' && (PATTERN==0) && '+bg_exp
        
        #since there is no energy selection in the arguments of the lightcurve, we need to add it to the expressions
        
        #for the broad band lightcurves
        elims_broad=' && (PI IN [300:10000])'
        
        #and for the flaring lightcurves
        elims_flare=' && (PI in ['+str(int(1000*flare_band[0]))+':'+str(int(1000*flare_band[1]))+'])'
            
        #lightcurve computation function
        def make_lc(table,suffix_lc,expression,binning,plot=True):
            
            #sas command
            spawn.sendline('\nevselect table='+table+' energycolumn=PI '+
                            'expression="'+expression+'" withrateset=yes rateset='+camera+suffix_evt+prefix+'_lc_'+suffix_lc+'.ds '+
                            'timebinsize='+str(binning)+' maketimecolumn=yes makeratecolumn=yes')
            #there are two occurences of this, one at the beginning and one at the end of the command 
            spawn.expect(['\[xmmsas_'],timeout=None)
            spawn.expect(['\[xmmsas_'],timeout=None)
            
            lc_path=os.path.join(fulldir,camera+suffix_evt+prefix+'_lc_'+suffix_lc+'.ds')
            #waiting for the file to be created to continue
            while not os.path.isfile(lc_path):
                time.sleep(1)

            #waiting for the file to be readable to continue
            file_readable=False
            while file_readable==False:
                try:
                    fits.open(lc_path)
                    file_readable=True
                except:
                    time.sleep(1)
            
            plot_mode='lc_'+suffix_lc
            
            if plot:    
                #we don't plot the rate cut since it won't make sense without being scaled to what's being plotted                
                plot_lc(lc_path,mode=plot_mode,save=True,close=True)
            else:
                return lc_path
        
        #broad band lightcurves for the source and bg
        #we do not plot them directly to plot both at the same time with the area ratio convinently computed during the correction task
        path_lc_src_broad=make_lc(file,'src_broad',src_exp_broad+elims_broad,binning=lc_bins,plot=False)
        path_lc_bg_broad=make_lc(file,'bg_broad',bg_exp_broad+elims_broad,binning=lc_bins,plot=False)
        
        #testing if the src region lightcurve is not empty without having to parse the sas messages
        if len(fits.open(path_lc_src_broad)[1].data['RATE'].nonzero()[0])==0:
            print('\nError: Empty broadband background lightcurve. Region definition/exposure Issue. Skipping...')
            
            spawn.sendline('\ncd $currdir')
            return'Empty broadband source lightcurve'
            
        #testing if the bg region lightcurve is not empty without having to parse the sas messages
        if len(fits.open(path_lc_bg_broad)[1].data['RATE'].nonzero()[0])==0:
            print('\nError: Empty broadband background lightcurve. Region definition/exposure Issue. Skipping...')
            
            spawn.sendline('\ncd $currdir')
            return'Empty broadband background lightcurve'
        
        #correcting the broad band lightcurve
        spawn.sendline('\nepiclccorr srctslist='+camera+suffix_evt+prefix+'_lc_src_broad.ds eventlist='+file+
                       ' outset='+camera+suffix_evt+prefix+'_lc_src_broad_corr.ds bkgtslist='+camera+suffix_evt+prefix+'_lc_bg_broad.ds'+
                       ' withbkgset=yes applyabsolutecorrections=yes')
        
        #we put several expects to see the progression
        spawn.expect('closing data set',timeout=None)

        lc_corr_state=spawn.expect(['epiclccorr-1.23.1','epiclccorr: error'],timeout=None)
    
        if lc_corr_state==1:
                print('\nError during corrected lightucurve computation. Skipping...')
                spawn.sendline('\ncd $currdir')
                return'Error during corrected lighctuve computation.'
                
        #it seems that sometimes the fits are created but are not readable yet when the next command runs, so we add a delay
        try:
            fits.open(os.path.join(fulldir,camera+suffix_evt+prefix+'_lc_src_broad_corr.ds'))
        except:
            time.sleep(5)
        #if they are still not readable after that, we can stop the run altogether
        try:
            fits.open(os.path.join(fulldir,camera+suffix_evt+prefix+'_lc_src_broad_corr.ds'))
        except:
            print('\nCould not load the corrected lightcurve fits file. Skipping...')
            
            spawn.sendline('\ncd $currdir')
            return'Could not load the corrected lightcurve fits file.'
        
        #if everything is ok, we can make the corrected plot
        plot_lc(os.path.join(fulldir,camera+suffix_evt+prefix+'_lc_src_broad_corr.ds'),mode='lc_src_broad_corr',save=True,close=True)
        
        #same check for the source and bg broad lightcurves
        try:
            fits.open(path_lc_src_broad)
            fits.open(path_lc_bg_broad)
        except:
            time.sleep(5)
        #if they are still not readable after that, we can stop the run altogether
        try:
            fits.open(path_lc_src_broad)
            fits.open(path_lc_bg_broad)
        except:
            print('\nCould not load at least one of the the broadband lightcurve fits file. Skipping...')
            
            spawn.sendline('\ncd $currdir')
            return'Could not load at least one of the broad band lightcurve fits file.'
            
        #if everything is fine, we plot the source and bg raw curves combined
        src_bg_ratio=fits.open(fulldir+'/'+camera+suffix_evt+prefix+'_lc_src_broad_corr.ds')[1].header['BKGRATIO']
        plot_lc(path_lc_src_broad,path_lc_bg_broad,area_ratio=src_bg_ratio,mode='lc_src_broad',save=True,close=True)
        
        '''
        flaring lightcurves for the source and bg
        for these, we use the non filtered event list instead of the filtered one (since the goal is to see the flares)
        To get it back, we retrieve the file name from the command copied in the header of the clean event list
        '''
        
        #it seems there are different syntaxes for the evselect header maybe due to timing or camera differences)
        #so we format the string a bit to be sure we have what we want
        raw_file=fits.open(fulldir+'/'+file)[0].header['XPROC0'].split('table=')[1].split(' ')[0].replace('.//','')
        raw_file=raw_file.replace('.//','').replace(':EVENTS','')

        #computing the lightcurve with this file and the corrected expressions
        #we don't care about doing a correction here
        path_lc_src_flare=make_lc(raw_file,'src_flare',src_exp_broad+elims_flare,binning=lc_bins,plot=False)
        path_lc_bg_flare=make_lc(raw_file,'bg_flare',bg_exp_broad+elims_flare,binning=lc_bins,plot=False)
        
        while not (os.path.isfile(path_lc_src_flare) and os.path.isfile(path_lc_bg_flare)):
            time.sleep(1)
        
        #it seems that sometimes the fits are created but are not readable yet when the next command runs, so we add a delay
        try:
            fits.open(path_lc_src_flare)
            fits.open(path_lc_bg_flare)
        except:
            time.sleep(5)
        
        #if they are still not readable after that, we can stop the run altogether
        try:
            fits.open(path_lc_src_flare)
            fits.open(path_lc_bg_flare)
        except:
            print('\nCould not load at least one of the flare lightcurve fits file. Skipping the end...')
            
            spawn.sendline('\ncd $currdir')
            return'Could not load at least one of the flare lightcurve fits file.'
            
        #plotting both without the gti cut for checking purposes
        plot_lc(path_lc_src_flare,path_lc_bg_flare,area_ratio=src_bg_ratio,mode='lc_src_flare',save=True,close=True)

        #cut plot for the summary file
        plot_lc(path_lc_src_flare,path_lc_bg_flare,area_ratio=src_bg_ratio,mode='lc_src_flare_cut',save=True,close=True)
        #copying the products
        spawn.sendline('\ncp *'+suffix_evt+prefix+'_lc* $currdir/batch')        

        #setting up a wait to ensure we copied everything:
        time.sleep(1)
            
        #copying the log file        
        spawn.sendline('\ncp *'+suffix_evt+'extract_lc.log $currdir/batch')
        
        print('Products copied to batch directory.')
        
        spawn.sendline('\ncd $currdir')
        
        if lc_corr_state==0:
            return 'Lighcturve extraction complete but missing the combined lc'
        else:            
            return 'Lightcurve extraction complete.'

        
    #defining which mode are included with a variable instead of arguments in the functions
    if expos_mode=='all':
        expos_mode_lcex='IMAGING TIMING BURST'
    else:
        expos_mode_lcex=expos_mode
        
    if cams=='all':
        camid_lcex=[0,1,2]
    else:
        camid_lcex=[]
        if 'pn' in cams:
            camid_lcex.append(0)
        if 'mos1'in cams:
            camid_lcex.append(1)
        if 'mos2' in cams:
            camid_lcex.append(2)
            
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nLightcurve extraction...')
    
    bashproc.sendline('cd '+directory)
    bashproc.sendline('mkdir -p batch')
    set_var(bashproc,directory)
       
    #recensing the cleaned event files available for each camera
    #camfilelist shape : [[pnfiles,pndirs],[mos1files,mos1dirs],[mos2files,mos2dirs]]
    clean_filelist=file_selector(directory,'evt_clean')
    
    #summary file header
    if directory.endswith('/'):
        obsid=directory.split('/')[-2]
    else:
        obsid=directory.split('/')[-1]
    summary_header='Obsid\tFile identifier\tLightcurve extraction result\n'
    
    #filtering for the selected cameras
    for i in camid_lcex:
        
        for j in range(len(clean_filelist[i][0])):
            
            clean_evtfile=clean_filelist[i][0][j]
            clean_evtdir=clean_filelist[i][1][j]
            
            #testing if the last file of the lightcurve extraction process has been created and moved into the batch or merging directory
            lastfile_auto=clean_evtfile.split('.')[0][clean_evtfile.find('_'):]
            lastfile_auto=camlist[i]+lastfile_auto+'_auto_lc_comb_flare_screen.png'
            
            if (mode=='manual' or overwrite \
            or 'm' not in action_list and not os.path.isfile(directory+'/'+clean_evtdir+'/'+lastfile_auto)\
            or 'm' in action_list and not os.path.isfile(startdir+'/'+mergedir+'/'+obsid+'_'+lastfile_auto))\
            and clean_evtfile!='':
                
                clean_evtid=clean_evtfile.split('.')[0].replace('clean','')

                #setting up a logfile in parallel to terminal display :
                if os.path.isfile(clean_evtdir+'/'+clean_evtid+'_extract_lc.log'):
                    os.system('rm '+clean_evtdir+'/'+clean_evtid+'_extract_lc.log')
                with StdoutTee(clean_evtdir+'/'+clean_evtid+'_extract_lc.log',
                               mode="a",buff=1,file_filters=[_remove_control_chars]),\
                    StderrTee(clean_evtdir+'/'+clean_evtid+'_extract_lc.log',buff=1,file_filters=[_remove_control_chars]):
                
                    bashproc.logfile_read=sys.stdout
                    print('\nCreating lightcurve of '+camlist[i]+' exposure '+clean_evtfile)
                    
                    #main function
                    summary_line=extract_lc_single(bashproc, clean_evtfile,clean_evtdir)

            else:
                if clean_evtfile=='':
                    print('\nNo evt to extract lightcurve from for camera '+camlist[i]+ ' in the obsid directory.')
                    
                    summary_line='No evt to extract lightcurve from for camera '+camlist[i]+ ' in the obsid directory.'
                    clean_evtid=camlist[i]
                else:
                    print('\nAuto mode lightcurve computation for the '+camlist[i]+' exposure '+clean_evtfile+
                          ' already done. Skipping...')
                    summary_line=''
            if summary_line!='':
                summary_content=obsid+'\t'+clean_evtid+'\t'+summary_line
                file_edit(os.path.join(directory,'batch','summary_extract_lc.log'),obsid+'\t'+clean_evtid,summary_content+'\n',
                          summary_header)
                
    #closing the spawn
    bashproc.sendline('exit')
    
    print('\nLightcurve extraction of the current obsid directory events finished.')
        
    extract_lc_done.set()
    
def extract_sp(directory,mode='manual',cams='all',expos_mode='all',overwrite=True):

    
    '''
    Extracts the spectrum of detected event files for one or several cameras and one of several modes
    
    The restriction happens through the 'cams' and 'expmode' keywords (strings containing all of the cams/modes names)

    As of now, only takes input formatted through the evt_filter and extract_region function

    Only accepts circular regions (in manual mode)
    '''
    
    #defining which mode are included with a variable instead of arguments in the functions
    if expos_mode=='all':
        expos_mode_spex='IMAGING TIMING BURST'
    else:
        expos_mode_spex=expos_mode
        
    def extract_sp_single(spawn,file,filedir):
        
        spawn.sendline('\ncurrdir=$(pwd)')
        spawn.sendline('\ncd '+filedir)
        
        
        if file=='':
            print('\nNo evt to extract spectrum from for this camera in the obsid directory.')
            return 'No evt to extract spectrum from for this camera in the obsid directory.'
        
        fulldir=directory+'/'+filedir

        #opening the fits file to extract some informations on the exposure
        fits_evtclean=fits.open(fulldir+'/'+file)
            
        expos_mode_single=fits_evtclean[0].header['DATAMODE']
        print('\nexpos mode:',expos_mode_single)
        
        camera=fits_evtclean[0].header['INSTRUME'][1:].swapcase()
        
        #quitting the extraction if the exposure mode is not in the selected ones
        if expos_mode_single not in expos_mode_spex:
            return 'Exposure mode not among the selected ones.'
        
        #identifying the exposure number and mode to create the suffix_evt
        #different method than in filter_evt because here we already know the structure of the file name
        suffix_evt=file.split('.')[0][file.find('_'):]
        
        if mode=='auto':
            prefix='_auto'
        else:
            prefix=input("\nPrefix with region already saved?\n\n")
        
        #testing if the ontime keyword is even there (can not be the case for empty exposures)
        try:
            exp_null=fits_evtclean[1].header['ONTIME']==0
        except:
            exp_null=1
            
        if exp_null==1:
            print('\nEmpty exposure.')
            
            #copying the products            
            spawn.sendline('\ncp *'+suffix_evt+prefix+'* $currdir/batch')        
    
            #copying the log file        
            spawn.sendline('\ncp *'+suffix_evt+'*.log $currdir/batch')
            
            spawn.sendline('\ncd $currdir')
            
            return 'Empty exposure.'
        
        regex_name=camera+suffix_evt+prefix+'_regex_results.txt'
        
        if not os.path.isfile(fulldir+'/'+regex_name):
            print('region extraction result file missing. Skipping...')
            
            #copying the products            
            spawn.sendline('\ncp *'+suffix_evt+prefix+'* $currdir/batch')        
    
            #copying the log file        
            spawn.sendline('\ncp *'+suffix_evt+'*.log $currdir/batch')
            
            spawn.sendline('\ncd $currdir')
        
            return 'region extraction result file missing.'
        
        with open(fulldir+'/'+regex_name,'r') as regex_file:
            regex_lines=regex_file.readlines()
        
        #the file given as argument should be the clean event 
        
        #we first see if the region file has already been created (mandatory)
        
        #standard part of the command expressions:
        if expos_mode_single in ['TIMING','BURST']:

            if camera=='pn':
                expression_sp='(FLAG==0) && (PATTERN<=4) && '
                specmax='20479'
            elif camera in {'mos1','mos2'}:
                expression_sp='(FLAG==0) && (PATTERN<=0) && '
                specmax='11999'
        
        elif expos_mode_single=='IMAGING':
            if camera=='pn':
                expression_sp='(FLAG==0) && (PATTERN<=4) && '
                specmax='20479'
            elif camera in {'mos1','mos2'}:
                expression_sp='#XMMEA_EM && (PATTERN<=12) && '
                specmax='11999'
                
        expression_source=regex_lines[0].split('\t')[1].replace('\n','')
        expression_bg=regex_lines[1].split('\t')[1].replace('\n','')
        pileup_lines=np.array(regex_lines[2].split('\t')[1].replace('\n','').split(','))
        
        #Source spectrum computation
        spawn.sendline('\nevselect table='+file+' withspectrumset=yes spectrumset='+camera+suffix_evt+prefix+'_sp_src.ds '+
                        'energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax='+specmax+' '+
                        'expression="'+expression_sp+expression_source+'"')
        spawn.expect('selected',timeout=None)
        
        #Background spectrum computation
        spawn.sendline('\nevselect table='+file+' withspectrumset=yes spectrumset='+camera+suffix_evt+prefix+'_sp_bg.ds '+
                        'energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax='+specmax+' '+
                        'expression="'+expression_sp+expression_bg+'"')
        spawn.expect('selected',timeout=None)
        
        #Backscale computations for the source and the background
        spawn.sendline('\nbackscale spectrumset='+camera+suffix_evt+prefix+'_sp_src.ds badpixlocation='+file)
        spawn.expect('backscale-1.6',timeout=None)
        
        spawn.sendline('\nbackscale spectrumset='+camera+suffix_evt+prefix+'_sp_bg.ds badpixlocation='+file)
        spawn.expect('backscale-1.6',timeout=None)
        
        #rmf computation
        spawn.sendline('\nrmfgen spectrumset='+camera+suffix_evt+prefix+'_sp_src.ds rmfset='+camera+suffix_evt+prefix+'.rmf')
        spawn.expect('Cleanup complete.',timeout=None)
        
        #arf computation
        spawn.sendline('\narfgen spectrumset='+camera+suffix_evt+prefix+'_sp_src.ds arfset='+camera+suffix_evt+prefix+'.arf '+
                        'withrmfset=yes rmfset='+camera+suffix_evt+prefix+'.rmf  badpixlocation='+file+' detmaptype=psf')
        spawn.expect('Closing arfset',timeout=None)
        
        #putting the pile-up values in the spectrum header:
        with fits.open(fulldir+'/'+camera+suffix_evt+prefix+'_sp_src.ds',mode='update') as fits_spectrum:
            fits_spectrum[0].header['PILE-UP']=','.join(pileup_lines)
            for pileup_iter in range(len(pileup_lines)):
                fits_spectrum[0].header['PILE-UP_'+str(pileup_iter)]=pileup_lines[pileup_iter]
        
        #quick specgroup function
        def spec_group(value):
            
            ####update for kaastra2016 opt binning
            '''
            wrapper for the command
            '''
            spawn.sendline('\nspecgroup spectrumset='+camera+suffix_evt+prefix+'_sp_src.ds mincounts='+str(value)
                            +' oversample=3 '+'rmfset='+camera+suffix_evt+prefix+'.rmf  arfset='+camera+suffix_evt
                            +prefix+'.arf '+'backgndset='+camera+suffix_evt+prefix+'_sp_bg.ds groupedset='+camera+
                            suffix_evt+prefix+'_sp_src_grp_'+str(value)+'.ds')
            spawn.expect('ended')
            
        #Specgrouping for a few values
        spec_group(20)
        spec_group(10)
        spec_group(5)

        print('\n'+file+' spectrum computation finished.')
        
        # print('\nReminder : pileup values (3 sigma) :\n')
        # print(pileup_lines.T)
        
        path_heavyfile=os.path.join(directory,'batch',camera+suffix_evt+prefix+'_pileup.ds')
        #removing the heaviest file first
        if os.path.isfile(path_heavyfile):
            os.remove(path_heavyfile)

        #copying the products            
        spawn.sendline('\ncp *'+suffix_evt+prefix+'* $currdir/batch')        

        time.sleep(5)
        
        #copying the log file        
        spawn.sendline('\ncp *'+suffix_evt+'*.log $currdir/batch')
        
        time.sleep(5)
        
        print('Products copied to batch directory.')
        
        spawn.sendline('\ncd $currdir')
        
        return 'Spectrum extraction complete.'
        
    if cams=='all':
        camid_spex=[0,1,2]
    else:
        camid_spex=[]
        if 'pn' in cams:
            camid_spex.append(0)
        if 'mos1'in cams:
            camid_spex.append(1)
        if 'mos2' in cams:
            camid_spex.append(2)
            
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nSpectrum extraction...')
    
    bashproc.sendline('cd '+directory)
    bashproc.sendline('mkdir -p batch')
    set_var(bashproc,directory)
       
    #recensing the cleaned event files available for each camera
    #camfilelist shape : [[pnfiles,pndirs],[mos1files,mos1dirs],[mos2files,mos2dirs]]
    clean_filelist=file_selector(directory,'evt_clean')
    
    #summary file header
    if directory.endswith('/'):
        obsid=directory.split('/')[-2]
    else:
        obsid=directory.split('/')[-1]
    summary_header='Obsid\tFile identifier\tSpectrum extraction result\n'
    
    #filtering for the selected cameras
    for i in camid_spex:
        
        for j in range(len(clean_filelist[i][0])):
            
            clean_evtfile=clean_filelist[i][0][j]
            clean_evtdir=clean_filelist[i][1][j]
            
            #testing if the last file of the spectrum extraction process has been created and moved into the batch or merging directory
            lastfile_auto=clean_evtfile.split('.')[0][clean_evtfile.find('_'):]
            lastfile_auto=camlist[i]+lastfile_auto+'_auto_sp_src_grp_5.ds'
            
            if (mode=='manual' or overwrite \
            or 'm' not in action_list and not os.path.isfile(directory+'/'+clean_evtdir+'/'+lastfile_auto)\
            or 'm' in action_list and not os.path.isfile(startdir+'/'+mergedir+'/'+obsid+'_'+lastfile_auto))\
            and clean_evtfile!='':
                
                clean_evtid=clean_evtfile.split('.')[0].replace('clean','')
                
                #setting up a logfile in parallel to terminal display :
                
                if os.path.isfile(clean_evtdir+'/'+clean_evtid+'_extract_sp.log'):
                    os.system('rm '+clean_evtdir+'/'+clean_evtid+'_extract_sp.log')
                with StdoutTee(clean_evtdir+'/'+clean_evtid+'_extract_sp.log',
                               mode="a",buff=1,file_filters=[_remove_control_chars]),\
                    StderrTee(clean_evtdir+'/'+clean_evtid+'_extract_sp.log',buff=1,file_filters=[_remove_control_chars]):
                
                    bashproc.logfile_read=sys.stdout
                    print('\nCreating spectrum of '+camlist[i]+' exposure '+clean_evtfile)
                    
                    #launching the main extraction
                    summary_line=extract_sp_single(bashproc, clean_evtfile,clean_evtdir)

            else:
                if clean_evtfile=='':
                    print('\nNo evt to extract spectrum from for camera '+camlist[i]+ ' in the obsid directory.')
                    
                    summary_line='No evt to extract spectrum from for camera '+camlist[i]+ ' in the obsid directory.'
                    clean_evtid=camlist[i]
                else:
                    print('\nAuto mode spectrum computation for the '+camlist[i]+' exposure '+clean_evtfile+
                          ' already done. Skipping...')
                    summary_line=''
                    
            if summary_line!='':
                summary_content=obsid+'\t'+clean_evtid+'\t'+summary_line
                file_edit(os.path.join(directory,'batch','summary_extract_sp.log'),obsid+'\t'+clean_evtid,summary_content+'\n',
                          summary_header)

    #closing the spawn
    bashproc.sendline('exit')
    
    print('\nSpectrum extraction of the current obsid directory events finished.')
        
    extract_sp_done.set()
            
def batch_mover(fin_dir):
    
    '''
    fin_dir must be an absolute path
    '''
    
    #shouldn't be needed but just in case
    os.chdir(obsdir)
    
    subdirs=glob.glob('**/',recursive=True)
    
    #creating the final directory if it doesn't exist yet
    os.system('mkdir -p '+fin_dir)
    
    for elem in subdirs:
        
        #restricting to batches
        if elem=='batch/':
            
            #listing the files inside the batch
            batchpath=os.path.join(obsdir,elem)
            batch_files=os.listdir(batchpath)

            #variables for the summary files
            summary_filt_lines_content=[]
            summary_reg_lines_content=[]
            summary_lc_lines_content=[]
            summary_sp_lines_content=[]
            
            #copying and renaming everything besides the lightcurves made to test the snr

            for product_name in [elem for elem in batch_files if elem[-16:-5]!='lc_src_snr_']:

                if 'summary' not in product_name:
                    #deleting before the copy to avoid potential conflicts with administrator privileges or whatever
                    if os.path.isfile(os.path.join(fin_dir,product_name)):
                        os.remove(os.path.join(fin_dir,product_name))
                
                    shutil.copy(batchpath+product_name,fin_dir)
                    os.replace(os.path.join(fin_dir,product_name),os.path.join(fin_dir,obsid+'_'+product_name))

                else:
                    #copying the information of each type of summary file
                    with open(os.path.join(batchpath,product_name)) as summary_file:
                        summary_lines=summary_file.readlines()
                        
                        if 'filter_evt' in product_name:
                            summary_filt_lines_content+=summary_lines[1:]
                            summary_filt_header=summary_lines[0]
                                
                        elif 'extract_reg' in product_name:
                            summary_reg_lines_content+=summary_lines[1:]
                            summary_reg_header=summary_lines[0]
                                
                        elif 'extract_lc' in product_name:
                            summary_lc_lines_content+=summary_lines[1:]
                            summary_lc_header=summary_lines[0]
                                
                        if 'extract_sp' in product_name:
                            summary_sp_lines_content+=summary_lines[1:]
                            summary_sp_header=summary_lines[0]
                    
            summary_filt_lines_id=[]
            summary_reg_lines_id=[]
            summary_lc_lines_id=[]
            summary_sp_lines_id=[]
            
            #in the global summary files the identifier must contain the obsids to avoid duplicates
            for content in summary_filt_lines_content:
                summary_filt_lines_id+=[content[:content.rfind('\t')]]
            
            for content in summary_reg_lines_content:
                summary_reg_lines_id+=[content[:content.rfind('\t')]]
                
            for content in summary_lc_lines_content:
                summary_lc_lines_id+=[content[:content.rfind('\t')]]
                
            for content in summary_sp_lines_content:
                summary_sp_lines_id+=[content[:content.rfind('\t')]]

            #saving the information in global summary files
            if summary_filt_lines_content!=[]:
                file_edit(os.path.join(fin_dir,'glob_summary_filter_evt.log'),line_id=summary_filt_lines_id,
                          line_data=summary_filt_lines_content,header=summary_filt_header)
                
            if summary_reg_lines_content!=[]:
                file_edit(os.path.join(fin_dir,'glob_summary_extract_reg.log'),line_id=summary_reg_lines_id,
                          line_data=summary_reg_lines_content,header=summary_reg_header)
                
            if summary_lc_lines_content!=[]:
                file_edit(os.path.join(fin_dir,'glob_summary_extract_lc.log'),line_id=summary_lc_lines_id,
                          line_data=summary_lc_lines_content,header=summary_lc_header)
                
            if summary_sp_lines_content!=[]:
                file_edit(os.path.join(fin_dir,'glob_summary_extract_sp.log'),line_id=summary_sp_lines_id,
                          line_data=summary_sp_lines_content,header=summary_sp_header)
                
    print('\nFiles successfully copied to '+fin_dir+' directory.')
    os.chdir(obsdir)
    
def product_delete(directory,del_evt=False):
    
    '''
    Delete reduction products. If del_evt is set to true, deletes everything. 
    If it is set to false, only delete products after the evt building (i.e. logs, clean evts, spectra, detection line outputs etc.)
    directory must be an absolute path
    '''
    
    #shouldn't be needed but just in case
    os.chdir(directory)
    
    subdirs=glob.glob('**/',recursive=True)
    
    for elem in subdirs:
        
        #restricting to output directories
        if elem.endswith('batch/'):
            
            #listing the files inside the batch
            os.chdir(elem)
            
            #deleting everything
            os.system('rm -rf *')
        
            #removing the current directory
            dirname=elem[:-1].split('/')[-1]
            os.chdir('..')
            
            os.system('rmdir '+dirname)

        if elem.endswith('outputmos/') or elem.endswith('outputpn/'):
            
            #listing the files inside the batch
            os.chdir(elem)
        
            obsid=os.getcwd().split('/')[-2]
            
            if del_evt:
                #deleting everything
                os.system('rm -rf *')
            
                #removing the current directory
                dirname=elem[:-1].split('/')[-1]
                os.chdir('..')
                
                os.system('rmdir '+dirname)
            else:
                for elemfile in glob.glob('**',recursive=True):
                    #the raw event files created by emproc and epproc all have the obsid in their name (while the products do not)
                    if not obsid in elemfile:
                        os.remove(elemfile)
                
            
        os.chdir(directory)
        
    #removing secondary products
    for elem in glob.glob('**',recursive=True):
        if elem.endswith('extract_sp.log') or elem.endswith('filter_evt.log')\
        or elem.endswith('evt_build.log') or elem.endswith('null'):
            os.remove(elem)
        #if we delete only the 2nd step products we keep the ccd.cif file
        if del_evt and elem.endswith('ccf.cif'):
            os.remove(elem)
                
    print('\nSuccessfully removed product files of '+directory+' directory.')
    

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

#taking off repeats with the outputmos, outputpn and batch directories
subdirs=[elem.replace('/outputpn','').replace('/outputmos','').replace('/batch','') for elem in subdirs]
subdirs=np.unique(subdirs)

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
    
started_folders,done_folders=startdir_state(args.action)
prev_started_folders,prev_done_folders=startdir_state(gm_action)
if local==False:
    #checking them in search for ODF directories
    for directory in subdirs:
        
        #continue check
        if directory in started_folders[:-2] and folder_cont:
            print('Directory '+directory+' is not among the last two directories. Skipping...')
            continue
        
        #directory overwrite check
        if directory in done_folders and folder_over==False:
            print('Actions already computed for directory '+directory+'\nSkipping...')
            continue
        
        #getting the last directory in the result (necessary if osbid folders are in subdirectories)
        dirname=directory[:-1]
        if dirname.find('/')!=-1:
            dirname=dirname[dirname.rfind('/')+1:]
        #checking if the directory has an obsid dir shape (10 numbers)
        #and not a general directory with both odf and pps subdirectories
        if len(dirname)==10 and dirname.isdigit() and "odf" not in os.listdir(directory):
            
            print('\nFound obsid directory '+dirname)
                        
            obsdir=os.path.join(startdir,directory)

            os.chdir(obsdir)
            obsid=obsdir[-11:-1]
            
            try:
            #for loop to be able to use different orders if needed
                for curr_action in action_list:
                    folder_state='Running '+curr_action
                    if curr_action=='1':
                        evt_state=evt_build(obsdir)
                        evt_build_done.wait()
                        if evt_state !='event building finished':
                            folder_state=evt_state
                            #breaking out of the try
                            raise NameError
                    if '2' in curr_action:
                        if 'std' in curr_action:
                            filter_mode='std'
                        if 'snr' in curr_action:
                            filter_mode='snr'
                        elif 'a' in curr_action:
                            filter_mode='auto'
                        elif 'n' in curr_action:
                            filter_mode='nolim'
                        filter_evt(obsdir,cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=True,mode=filter_mode)
                        filter_evt_done.wait()
                    if curr_action=='3':
                        extract_reg(obsdir,mode='auto',cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=overwrite_glob)
                        extract_reg_done.wait()
                    if curr_action=='l':
                        extract_lc(obsdir,mode='auto',cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=overwrite_glob)
                        extract_lc_done.wait()
                    if curr_action=='s':
                        extract_sp(obsdir,mode='auto',cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=overwrite_glob)
                        extract_sp_done.wait()
                    if curr_action=='m':
                        batch_mover(os.path.join(startdir,mergedir))
                    if curr_action=='gm':
                        #here we check if the current obsid directory is part of the finished directory for the gm_action actions
                        if directory in prev_done_folders:
                            batch_mover(os.path.join(startdir,mergedir))
                        else:
                            folder_state='Unfishined directory'
                            raise NameError
                    folder_state='Done'
            except:
                #signaling unknown errors if they happened
                if 'Running' in folder_state:
                    print('\nError while '+folder_state)
                    folder_state=folder_state.replace('Running','Aborted at')
                
            os.chdir(startdir)
            #adding the directory to the list of already computed directories
            file_edit('summary_folder_analysis_'+args.action+'.log',directory,directory+'\t'+folder_state+'\n',summary_folder_header)
                
else:
    #taking of the merge action if local is set since there is no point to merge in local (the batch directory acts as merge)
    action_list=[elem for elem in action_list if elem!='m']
    
    absdir=os.getcwd()
    
    #just to avoid an error but not used since there is not merging in local
    obsid=''
    
    #for loop to be able to use different orders if needed
    for curr_action in action_list:
            if curr_action=='1':
                evt_build(absdir)
                evt_build_done.wait()
            if '2' in curr_action:
                if 'std' in curr_action:
                    filter_mode='std'
                if 'snr' in curr_action:
                    filter_mode='snr'
                elif 'a' in curr_action:
                    filter_mode='auto'
                elif 'n' in curr_action:
                    filter_mode='nolim'
                filter_evt(absdir,cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=True,mode=filter_mode)
                filter_evt_done.wait()
            if curr_action=='3':
                extract_reg(absdir,mode='auto',cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=overwrite_glob)
                extract_reg_done.wait()
            if curr_action=='l':
                extract_lc(absdir,mode='auto',cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=overwrite_glob)
                extract_lc_done.wait()
            if curr_action=='s':
                extract_sp(absdir,mode='auto',cams=cameras_glob,expos_mode=expos_mode_glob,overwrite=overwrite_glob)
                extract_sp_done.wait()
                    
#out of the loop so we do the following for the entire directory structure and not just the detected obsid directories
if 'c' in action_list:
    count_evts(startdir)
if 'd' in action_list:
    product_delete(startdir,del_evt=False)
if 'D' in action_list:
    product_delete(startdir,del_evt=True)
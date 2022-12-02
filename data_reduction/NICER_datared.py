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

#astro imports
from astropy.io import fits

"""
Created on Thu Sep  1 23:18:16 2022

Data reduction Script for NICER Observations

Searches for all NICER Obs type directories in the subdirectories and launches the process for each

list of possible actions : 

1. process_obsdir: run the nicerl2 script to process an obsid folder

2. select_detector: removes specific detectors from the event file (not tested)

s. extract_spectrum: extract a pha spectrum from a process obsid folder using  Xselect

b. extract_background: extract a bg spectrum from a process obsid folder using a specific method

r. extract_response: extract a response from a processed obsid folder

g. group_spectra: group spectra using the optimized Kastra et al. binning

m.merge: merge all spectral products in the subdirectories to a bigbatch directory 

"""

'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce NICER files.\n)')

#the basics
ap.add_argument("-dir", "--startdir", nargs='?', help="starting directory. Current by default", default='./', type=str)
ap.add_argument("-l","--local",nargs=1,help='Launch actions directly in the current directory instead',
                default=False,type=bool)
ap.add_argument('-catch','--catch_errors',help='Catch errors while running the data reduction and continue',default=True,type=bool)

#global choices
ap.add_argument("-a","--action",nargs='?',help='Give which action(s) to proceed,separated by comas.'+
                '\n1.evt_build\n2.filter_evt\n3.extract_reg...',default='1,s,r,g,m',type=str)
ap.add_argument("-over",nargs=1,help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder',default=True,type=bool)


#directory level overwrite (not active in local)
ap.add_argument('-folder_over',nargs=1,help='relaunch action through folders with completed analysis',default=False,type=bool)
ap.add_argument('-folder_cont',nargs=1,help='skip all but the last 2 directories in the summary folder file',default=False,type=bool)
#note : we keep the previous 2 directories because bug or breaks can start actions on a directory following the initially stopped one

#action specific overwrite
ap.add_argument('-baddet','--bad_detectors',help='List detectors to exclude from the data reduction',default='-14,-34,-54',type=str)
ap.add_argument('-bg',"--bgmodel",help='Give the background model to use for the data reduction',default='3c50',type=str)
ap.add_argument('-gtype',"--grouptype",help='Give the group type to use in regroup_spectrum',default='opt',type=str)
    
ap.add_argument('-heasoft_init_alias',help="name of the heasoft initialisation script alias",default="heainit",type=str)
ap.add_argument('-caldbinit_init_alias',help="name of the caldbinit initialisation script alias",default="caldbinit",type=str)
ap.add_argument('-alias_3C50',help="bash alias for the 3C50 directory",default='$NICERBACK3C50',type=str)

args=ap.parse_args()
startdir=args.startdir
action_list=args.action.split(',')
local=args.local
folder_over=args.folder_over
folder_cont=args.folder_cont
overwrite_glob=args.over
catch_errors=args.catch_errors
bgmodel=args.bgmodel
grouptype=args.grouptype
bad_detectors=args.bad_detectors
heasoft_init_alias=args.heasoft_init_alias
caldbinit_init_alias=args.caldbinit_init_alias
alias_3C50=args.alias_3C50

'''''''''''''''''
''''FUNCTIONS''''
'''''''''''''''''

#we insure each action will wait for the completion of the previous one by using threads
process_obsdir_done=threading.Event()
extract_spectrum_done=threading.Event()
extract_response_done=threading.Event()
extract_background_done=threading.Event()
regroup_spectrum_done=threading.Event()
batch_mover_done=threading.Event()
select_detector_done=threading.Event()

def set_var(spawn):
    
    '''
    Sets starting environment variables for NICER data analysis
    '''
    spawn.sendline(heasoft_init_alias)
    spawn.sendline(caldbinit_init_alias)
    
#function to remove (most) control chars
def _remove_control_chars(message):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)

def file_edit(path,line_id,line_data,header):
    
    '''
    Edits (or create) the file given in the path and replaces/add the line(s) where the line_id str/LIST is with the line-content str/LIST.
    line_id should be included in line_content.
    Header is the first line of the file, with usually different informations.
    '''
    
    lines=[]
    if type(line_id)==str or type(line_id)==np.str_:
        line_identifier=[line_id]
    else:
        line_identifier=line_id
        
    if type(line_data)==str or type(line_data)==np.str_:
        line_content=[line_data]
    else:
        line_content=line_data
        
    if os.path.isfile(path):
        with open(path) as file:
            lines=file.readlines()
            
            #loop for all the lines to add
            for single_identifier,single_content in zip(line_identifier,line_content):
                line_exists=False
                if not single_content.endswith('\n'):
                    single_content+='\n'
                #loop for all the lines in the file
                for l,single_line in enumerate(lines):
                    if single_identifier in single_line:
                        lines[l]=single_content
                        line_exists=True
                if line_exists==False:
                    lines+=[single_content]
            
    else:
        #adding everything
        lines=line_content
    
    time.sleep(1)
    
    with open(path,'w+') as file:
        if lines[0]==header:
            file.writelines(lines)
        else:
            file.writelines([header]+lines)

def process_obsdir(directory,overwrite=True):
    
    '''
    Processes a directory using the nicerl2 script
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nEvent filtering...')
    
    set_var(bashproc)
        
    if os.path.isfile(directory+'/process_obsdir.log'):
        os.system('rm '+directory+'/process_obsdir.log')
        
    with StdoutTee(directory+'/process_obsdir.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/process_obsdir.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout
        bashproc.sendline('nicerl2 indir='+directory+' clobber='+('YES' if overwrite else 'FALSE'))
        
        process_state=bashproc.expect(['terminating with status','Event files written'],timeout=None)
        
        #exiting the bashproc
        bashproc.sendline('exit')
        process_obsdir_done.set()
        
        #raising an error to stop the process if the command has crashed for some reason
        if process_state==0:
            raise ValueError
            
def select_detector(directory,detectors='-14,-34,-54'):
    
    '''
    Removes specific detectors from the event file before continuing the analysis
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/fpmsel-using/
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\n Detector selection...')
    
    #finding the event file
    evt_path=[elem for elem in glob.glob('**',recursive=True) if directory in elem and 'cl.evt' in elem]
    
    #stopping the process if there is no processed event file
    if len(evt_path)==0:
        raise ValueError
        
    evt_name=evt_path[0].split('/')[-1]
    
    set_var(bashproc)

    if os.path.isfile(directory+'/select_detector.log'):
        os.system('rm '+directory+'/select_detector.log')
        
    with StdoutTee(directory+'/select_detector.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/select_detector.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout
        
        bashproc.sendline('nifpmsel '+evt_name+' '+evt_name.replace('.evt','_sel.evt')+'detlist=launch,'+detectors)
        
        select_state=bashproc.expect(['DONE','terminating with status'],timeout=None)
                
        #raising an error to stop the process if the command has crashed for some reason
        if select_state!=0:
            select_detector_done.set()
            raise ValueError
        
        #replacing the previous event file by the selected event file
        bashproc.sendline('mv '+evt_name.replace('.evt','_sel.evt')+' '+evt_name)
        
        #exiting the bashproc
        bashproc.sendline('exit')
        select_detector_done.set()
        
def extract_spectrum(directory):
    
    '''
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
        extract_spectrum_done.set()
    
def extract_background(directory,model):
    
    '''
    Extracts NICER background using associated commands and models
    
    model:
        -3C50 (needs to be installe before)
    
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
        
        extract_background_done.set()
        
def extract_response(directory):
    
    '''
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
        extract_response_done.set()
        
def regroup_spectrum(directory,group='opt'):
    
    '''
    Regroups NICER spectrum from an obsid directory using ftgrouppha
    
    mode:
        -opt: follows the Kastra and al. 2016 binning
    
    '''
        
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nRegrouping spectrum...')
    
    set_var(bashproc)
    
    currdir=os.getcwd()
    
    if os.path.isfile(directory+'/regroup_spectrum.log'):
        os.system('rm '+directory+'/regroup_spectrum.log')
        
    #deleting previously existing grouped spectra to avoid problems when testing their existence
    if os.path.isfile(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha')):
        os.remove(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha'))
            
    with StdoutTee(directory+'/regroup_spectrum.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/regroup_spectrum.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout
        
        bashproc.sendline('cd '+directory)
                    
        bashproc.sendline('ftgrouppha infile='+directory+'_sp.pha outfile='+directory+'_sp_grp_'+group+'.pha grouptype='+group+
                           ' respfile='+directory+'.rmf')
        
        while not os.path.isfile(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha')):
            time.sleep(1)
            
        bashproc.sendline('echo done')
        bashproc.expect('done')
        
        bashproc.sendline('exit')
        regroup_spectrum_done.set()

def batch_mover(directory):
    
    '''
    copies all spectral products in a directory to a bigbatch directory above the obsid directory to prepare for spectrum analysis
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    bashproc.logfile_read=sys.stdout
    
    print('\n\n\nCopying spectral products to a merging directory...')
    
    set_var(bashproc)
    
    bashproc.sendline('mkdir -p bigbatch')
    
    bashproc.sendline('cd '+directory)
    bashproc.sendline('cp -v '+directory+'* ../bigbatch')
    
    #reasonable waiting time to make sure files can be copied
    time.sleep(2)
    
    bashproc.expect('->')
    
    bashproc.sendline('exit')
    batch_mover_done.set()
    
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

if not local:
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
        
        #getting the last directory in the result (necessary if osbid folders are in subdirectories) (we take the last / of to avoid problems)
        dirname=directory[:-1].split('/')[-1]
            
        #checking if the directory has an obsid dir shape (10 numbers)
        #and not a general or subdirectory
        if len(dirname)==10 and dirname.isdigit() and "odf" not in os.listdir(directory):
            
            print('\nFound obsid directory '+dirname)
                        
            
            above_obsdir=os.path.join(startdir,'/'.join(directory[-1].split('/')[:-1]))

            os.chdir(above_obsdir)
            
            if catch_errors:
                try:
                #for loop to be able to use different orders if needed
                    for curr_action in action_list:
                        folder_state='Running '+curr_action
                        if curr_action=='1':
                            process_obsdir(dirname,overwrite=overwrite_glob)
                            process_obsdir_done.wait()
                        if curr_action=='2':
                            select_detector(dirname,detectors=bad_detectors)
                            select_detector_done.wait()
                        if curr_action=='s':
                            extract_spectrum(dirname)
                            extract_spectrum_done.wait()
                        if curr_action=='b':
                            extract_background(dirname,model=bgmodel)
                            extract_background_done.wait()
                        if curr_action=='r':
                            extract_response(dirname)
                            extract_response_done.wait()
                        if curr_action=='g':
                            regroup_spectrum(dirname,group=grouptype)
                            regroup_spectrum_done.wait()
                        if curr_action=='m':
                            batch_mover(dirname)
                            batch_mover_done.wait()

                        os.chdir(startdir)
                    folder_state='Done'

                except:
                    #signaling unknown errors if they happened
                    if 'Running' in folder_state:
                        print('\nError while '+folder_state)
                        folder_state=folder_state.replace('Running','Aborted at')
                    os.chdir(startdir)
            else:
                #for loop to be able to use different orders if needed
                for curr_action in action_list:
                    folder_state='Running '+curr_action
                    if curr_action=='1':
                        process_obsdir(dirname,overwrite=overwrite_glob)
                        process_obsdir_done.wait()
                    if curr_action=='2':
                        select_detector(dirname,detectors=bad_detectors)
                        select_detector_done.wait()
                    if curr_action=='s':
                        extract_spectrum(dirname)
                        extract_spectrum_done.wait()
                    if curr_action=='b':
                        extract_background(dirname,model=bgmodel)
                        extract_background_done.wait()
                    if curr_action=='r':
                        extract_response(dirname)
                        extract_response_done.wait()
                    if curr_action=='g':
                        regroup_spectrum(dirname,group=grouptype)
                        regroup_spectrum_done.wait()
                    if curr_action=='m':
                        batch_mover(dirname)
                        batch_mover_done.wait()
                        
                    os.chdir(startdir)
                folder_state='Done'
                        
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
                process_obsdir(absdir,overwrite=overwrite_glob)
                process_obsdir_done.wait()
            if curr_action=='2':
                select_detector(dirname,detectors=bad_detectors)
                select_detector_done.wait()
            if curr_action=='s':
                extract_spectrum(dirname)
                extract_spectrum_done.wait()
            if curr_action=='b':
                extract_background(dirname,model=bgmodel)
                extract_background_done.wait()
            if curr_action=='r':
                extract_response(dirname)
                extract_response_done.wait()
            if curr_action=='g':
                regroup_spectrum(dirname,group=grouptype)
                regroup_spectrum_done.wait()
            if curr_action=='m':
                batch_mover(dirname)
                batch_mover_done.wait()
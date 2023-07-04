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

from astropy.stats import sigma_clip
import matplotlib as mpl

#using agg because qtagg still generates backends with plt.ioff()
mpl.use('qtagg')

#mpl.use('qt5agg')

import matplotlib as mpl

import matplotlib.pyplot as plt
plt.ioff()

from matplotlib import pyplot as plt


from astropy.time import Time
from general_tools import file_edit,ravel_ragged,interval_extract

#astro imports
from astropy.io import fits

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

c. clean: clean event products in the observation's event_cl directory

DEPRECATED 
2. select_detector: removes specific detectors from the event file (not tested)

s. extract_spectrum: extract a pha spectrum from a process obsid folder using  Xselect

b. extract_background: extract a bg spectrum from a process obsid folder using a specific method

r. extract_response: extract a response from a processed obsid folder

 

"""

'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce NICER files.\n)')

#the basics
ap.add_argument("-dir", "--startdir", nargs='?', help="starting directory. Current by default", default='./', type=str)
ap.add_argument("-l","--local",nargs=1,help='Launch actions directly in the current directory instead',
                default=False,type=bool)
ap.add_argument('-catch','--catch_errors',help='Catch errors while running the data reduction and continue',default=False,type=bool)

#global choices
ap.add_argument("-a","--action",nargs='?',help='Give which action(s) to proceed,separated by comas.',
                default='1,gti,l,c,m',type=str)
ap.add_argument("-over",nargs=1,help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder',default=True,type=bool)

#directory level overwrite (not active in local)
ap.add_argument('-folder_over',nargs=1,help='relaunch action through folders with completed analysis',default=True,type=bool)
ap.add_argument('-folder_cont',nargs=1,help='skip all but the last 2 directories in the summary folder file',default=False,type=bool)
#note : we keep the previous 2 directories because bug or breaks can start actions on a directory following the initially stopped one

#action specific overwrite

#gti
ap.add_argument('-gti_split',nargs=1,help='GTI split method',default='orbit+clip',type=str)
ap.add_argument('-gti_lc_band',nargs=1,help='Band for the lightcurve used for GTI splitting',
                default='12-15',type=str)

#lightcurve
ap.add_argument('-lc_bin',nargs=1,help='Gives the binning of all lightcurces/HR evolutions (in s)',default=1,type=str)
#note: also defines the binning used for the gti definition

ap.add_argument('-lc_bands_str',nargs=1,help='Gives the list of bands to create lightcurves from',default='3-12',type=str)
ap.add_argument('-hr_bands_str',nargs=1,help='Gives the list of bands to create hrsfrom',default='6-10/3-6',type=str)


#spectra
ap.add_argument('-bg',"--bgmodel",help='Give the background model to use for the data reduction',default='scorpeon_script',type=str)
ap.add_argument('-bg_lang',"--bg_language",help='Gives the language output for the script generated to load spectral data into either PyXspec or Xspec',
                default='python',type=str)

ap.add_argument('-gtype',"--grouptype",help='Give the group type to use in regroup_spectral',default='opt',type=str)

#deprecated
ap.add_argument('-baddet','--bad_detectors',help='List detectors to exclude from the data reduction',default='-14,-34,-54',type=str)

    
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
bglanguage=args.bg_language

gti_split=args.gti_split
gti_lc_band=args.gti_lc_band

lc_bin=args.lc_bin
lc_bands_str=args.lc_bands_str
hr_bands_str=args.hr_bands_str

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
extract_all_spectral_done=threading.Event()
extract_spectrum_done=threading.Event()
extract_response_done=threading.Event()
extract_background_done=threading.Event()
regroup_spectral_done=threading.Event()
batch_mover_done=threading.Event()
select_detector_done=threading.Event()

extract_lc_done=threading.Event()
clean_products_done=threading.Event()

create_gtis_done=threading.Event()
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

####THIS IS DEPRECATED            
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
            bashproc.sendline('exit')
            select_detector_done.set()
            raise ValueError
        
        #replacing the previous event file by the selected event file
        bashproc.sendline('mv '+evt_name.replace('.evt','_sel.evt')+' '+evt_name)
        
        #exiting the bashproc
        bashproc.sendline('exit')
        select_detector_done.set()

def create_gtis(directory,split='orbit+clip',band='3-15',binning=1,overwrite=True):
    '''
    wrapper for a function to split nicer obsids into indivudal portions

    first creates a lightcurve with the chosen binning then uses it to define
    individual gtis

    modes (combinable:
        -orbit:split each obs into each individual nicer observation period
        -clip: isolates broad band flare/dip periods in each observation and creates individual
        note that all individual flare/dip periods in single orbits are grouped together
         gtis for them

    NOTE: requires sas and a sasinit alias to initialize it (to use tabgtigen)

    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    print('\n\n\nCreating gtis products...')

    set_var(bashproc)

    if os.path.isfile(directory + '/extract_gtis.log'):
        os.system('rm ' + directory + '/extract_gtis.log')


    pi_band = '-'.join((np.array(band.split('-')).astype(int) * 100).astype(str).tolist())

    #removing old lc files
    old_files_lc = [elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                    elem.endswith('.lc') and 'bin' not in elem]

    for elem_file in old_files_lc:
        os.remove(elem_file)

    #removing old gti files
    old_files_gti=[elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                   '_gti_' in elem]

    for elem_file_gti in old_files_gti:
        os.remove(elem_file_gti)

    if os.path.isfile(directory + '/create_gtis.log'):
        os.system('rm ' + directory + '/create_gtis.log')

    with StdoutTee(directory + '/create_gtis.log', mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(directory + '/create_gtis.log', buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        bashproc.sendline('nicerl3-lc ' + directory + ' pirange=' + pi_band + ' timebin=' + str(binning) + ' ' +
                          ' clobber=' + ('YES' if overwrite else 'FALSE'))

        process_state = bashproc.expect(['Task aborting due', 'DONE'], timeout=None)

        # raising an error to stop the process if the command has crashed for some reason
        if process_state == 0:
            with open(directory + '/extract_lc.log') as file:
                lines = file.readlines()

            bashproc.sendline('exit')
            extract_lc_done.set()
            return lines[-1].replace('\n', '')


        file_lc = [elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if elem.endswith('.lc')][0]

        # storing the data of the lc
        with fits.open(file_lc) as fits_lc:
            data_lc_arr = fits_lc[1].data

            #different format here to consider the leap second
            time_zero=fits_lc[1].header['TSTART']-fits_lc[1].header['LEAPINIT']

            #saving for titles later
            time_zero_str=Time(fits_lc[1].header['MJDREFI']+(fits_lc[1].header['TIMEZERO']-fits_lc[1].header['LEAPINIT'])/86400,format='mjd')

            time_zero_str=str(time_zero_str.to_datetime())

        # removing the direct products
        new_files_lc = [elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                        '.lc' in elem and 'bin' not in elem]

        times = data_lc_arr['TIME']

        id_gti_split=[-1]

        if 'orbit' in split:
            #adding gaps of more than 100s as cuts in the gtis
            for i in range(len(times) - 1):
                if times[i + 1] - times[i] > 100:
                    id_gti_split += [i]

        id_gti_orbit=[]
        if len(id_gti_split)==1:
            id_gti_orbit+=[range(len(times))]
        else:
            for id_split in range(len(id_gti_split)):

                #note:+1 at the end since we're using a range
                id_gti_orbit+=[list(range(id_gti_split[id_split]+1,(len(times)-1 if\
                    id_split==len(id_gti_split)-1 else id_gti_split[id_split+1])+1))]

        n_orbit=len(id_gti_orbit)

        if 'clip' in split:
            id_gti=[]
            id_flares=[]
            id_dips=[]
            for elem_gti_orbit in id_gti_orbit:
                clip_data = sigma_clip(data_lc_arr['RATE'][elem_gti_orbit], 3)

                clip_std=clip_data.std()
                clip_mean=clip_data.mean()

                #computing the gtis outside of the 3 sigma of the clipped distribution
                #even with uncertainties
                #note: inverted errors in the data on purpose
                elem_id_gti=[elem for elem in elem_gti_orbit if \
                             (data_lc_arr['RATE'][elem]+3*data_lc_arr['ERROR'][elem])>=clip_mean-3*clip_std\
                             and (data_lc_arr['RATE'][elem]-3*data_lc_arr['ERROR'][elem])<=clip_mean+3*clip_std]

                elem_id_flares=[elem for elem in elem_gti_orbit if \
                             (data_lc_arr['RATE'][elem]-3*data_lc_arr['ERROR'][elem])>clip_mean+3*clip_std]

                elem_id_dips=[elem for elem in elem_gti_orbit if \
                             (data_lc_arr['RATE'][elem]+3*data_lc_arr['ERROR'][elem])<clip_mean-3*clip_std]

                id_gti+=[elem_id_gti]
                id_flares+=[elem_id_flares]
                id_dips+=[elem_id_dips]

        else:
            id_gti=id_gti_orbit
            id_flares=[]
            id_dips=[]

        #creating global figure
        plt.figure(figsize=(15,8))
        plt.suptitle(
            'NICER global flaring lightcurve for observation ' + directory + ' in the ' + band + ' keV band')
        plt.xlabel('Time (s) after ' + time_zero_str)
        plt.ylabel('RATE (counts/s)')
        plt.tight_layout()
        plt.errorbar(data_lc_arr['TIME'], data_lc_arr['RATE'], xerr=binning,yerr=data_lc_arr['ERROR'],
                     ls='-',lw=1,color='grey',ecolor='blue')

        #creating figure (if no orbit cut is made, a single orbit will be performed)
        for id_orbit in range(n_orbit):

            #splitting the individual gti intervals in each orbit (even though they are stacked)
            #to avoid overlapping the colors

            for id_inter,list_inter in enumerate(list(interval_extract(id_gti[id_orbit]))):
                plt.axvspan(data_lc_arr['TIME'][min(list_inter)], data_lc_arr['TIME'][max(list_inter)], color='grey', alpha=0.2,label='standard gtis' if (id_inter==0 and id_orbit==0) else '')

            for id_inter,list_inter in enumerate(list(interval_extract(id_flares[id_orbit]))):
                plt.axvspan(data_lc_arr['TIME'][min(list_inter)], data_lc_arr['TIME'][max(list_inter)], color='green', alpha=0.2,label='flare gtis' if (id_inter==0 and id_orbit==0) else '')

            for id_inter,list_inter in enumerate(list(interval_extract(id_dips[id_orbit]))):
                plt.axvspan(data_lc_arr['TIME'][min(list_inter)], data_lc_arr['TIME'][max(list_inter)], color='red', alpha=0.2,label='dip gtis' if (id_inter==0 and id_orbit==0) else '')

        plt.legend()
        plt.savefig('./'+directory+'/'+directory+'_lc_'+band+'_bin_'+str(binning)+'_gtis.png')
        plt.close()

        #creating individual orbit figures
        for id_orbit in range(n_orbit):
            plt.figure(figsize=(15,8))

            plt.suptitle(
                'NICER flaring lightcurve for orbit '+('%3.f'%(id_orbit+1)).replace(' ','0')+' of observation ' + directory + ' in the ' + band + ' keV band')
            plt.xlabel('Time (s) after ' + time_zero_str)
            plt.ylabel('RATE (counts/s)')
            plt.tight_layout()

            plt.errorbar(data_lc_arr['TIME'][id_gti[id_orbit]], data_lc_arr['RATE'][id_gti[id_orbit]],xerr=binning,yerr=data_lc_arr['ERROR'][id_gti[id_orbit]],ls='-',lw=1,color='grey',ecolor='blue')

            for id_inter,list_inter in enumerate(list(interval_extract(id_gti[id_orbit]))):
                plt.axvspan(data_lc_arr['TIME'][min(list_inter)], data_lc_arr['TIME'][max(list_inter)], color='grey', alpha=0.2,label='standard gtis' if id_inter==0 else '')

            for id_inter,list_inter in enumerate(list(interval_extract(id_flares[id_orbit]))):
                plt.axvspan(data_lc_arr['TIME'][min(list_inter)], data_lc_arr['TIME'][max(list_inter)], color='green', alpha=0.2,label='flare gtis' if id_inter==0 else '')

            for id_inter,list_inter in enumerate(list(interval_extract(id_dips[id_orbit]))):
                plt.axvspan(data_lc_arr['TIME'][min(list_inter)], data_lc_arr['TIME'][max(list_inter)], color='red', alpha=0.2,label='dip gtis' if id_inter==0 else '')

            plt.legend()
            plt.savefig('./'+directory+'/'+directory+'-'+('%3.f'%(id_orbit+1)).replace(' ','0')+
                        '_lc_'+band+'_bin_'+str(binning)+'_gtis.png')
            plt.close()

        #creating the gti files for each part of the obsid
        bashproc.sendline('sasinit')

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

        def create_gti_files(id_gti,data_lc,suffix):

            if len(id_gti)>0:

                fits_gti=fits.open(file_lc)

                #creating a custom gti 'mask' file
                gti_column=fits.ColDefs([fits.Column(name='IS_GTI', format='I',
                                        array=np.array([1 if i in id_gti else 0 for i in range(len(data_lc))]))])

                #replacing the hdu with a hdu containing it
                fits_gti[1]=fits.BinTableHDU.from_columns(fits_gti[1].columns[:2]+gti_column)
                fits_gti[1].name='IS_GTI'

                lc_mask_path = os.path.join(file_lc[:file_lc.rfind('/')],directory+'_gti_mask_'+('%3.f'%(id_orbit+1)).replace(' ','0')+suffix)+'.fits'

                fits_gti.writeto(lc_mask_path, overwrite=True)

                #creating the orbit gti expression

                gti_path=os.path.join(file_lc[:file_lc.rfind('/')],directory+'_gti_'+
                ('%3.f'%(id_orbit+1)).replace(' ','0')+suffix)+'.gti'

                bashproc.sendline('tabgtigen table='+lc_mask_path+' expression="IS_GTI==1" gtiset='+gti_path)

                #waiting time to let time for the file to be created
                time.sleep(1)

                #opening and modifying the content of the header in the gti file for NICER

                with fits.open(gti_path,mode='update') as hdul:

                    #offsetting the table with the time zero of the observation
                    hdul[1].data['START']=hdul[1].data['START']+time_zero
                    hdul[1].data['STOP'] = hdul[1].data['STOP'] + time_zero

                    #and changing reference times
                    hdul[1].header['MJDREF']=56658+7.775925925925930E-04
                    #hdul[1].header['MJDREFI']=56658
                    #hdul[1].header['MJDREFF']=7.775925925925930E-04

                    hdul.flush()

        for id_orbit in range(n_orbit):

            create_gti_files(id_gti[id_orbit],data_lc_arr,'')
            create_gti_files(id_flares[id_orbit],data_lc_arr, 'F')
            create_gti_files(id_dips[id_orbit],data_lc_arr, 'D')

        #only deleting the lightcurve files after everything has been finished
        for elem_file in new_files_lc:
            os.remove(elem_file)

        #exiting the bashproc
        bashproc.sendline('exit')
        create_gtis_done.set()

#### extract_all_spectral
def extract_all_spectral(directory,bkgmodel='scorpeon_script',language='python',overwrite=True):
    
    '''
    Wrapper for nicerl3-spect, extracts spectra, creates bkg and rmfs

    if gti files created by create_gtis are present, instead of creating a full spectrum,
    creates individual spectral products for each gti
    
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/nicerl3-spect/

    Processes a directory using the nicerl3-spect script
    
    bgmodel options:
        -scorpeon_script: uses scorpeon in script mode to create a variable xspec-compatible bg model
        
        -scorpeon_file: uses scorpeon in file mode to produce a static background file
        
        -3c50: 3c50 model of Remillar et al., fetches data from the alias_3C50 argument
        
        -sw: Space weather model
        
    specific option for scorpeon_script:
        
        -language: if set to python, the generated scripts are made for Pyxspec instead of standard xspec
        
    Note: can produce no output without error if no gti in the event file
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nCreating spectral products...')
    
    set_var(bashproc)
        
    if os.path.isfile(directory+'/extract_all_spectral.log'):
        os.system('rm '+directory+'/extract_all_spectral.log')
        
    with StdoutTee(directory+'/extract_all_spectral.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_all_spectral.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout
        
        bkg_outlang_str=''
        
        if 'scorpeon' in bkgmodel:
            bkgmodel_str=bkgmodel.split('_')[0]
            bkgmodel_mode=bkgmodel.split('_')[1]
            
            #specific option for script mode
            if bkgmodel_mode=='script':
                if language=='python':
                    bkg_outlang_str='outlang=PYTHON'
                else:
                    print('NICER_datared_error: only python is implemented for the language output of scorpeon in script mode')
                    return 'NICER_datared_error: only python is implemented for the language output of scorpeon in script mode'
            
        else:
            bkgmodel_str=bkgmodel
            bkgmodel_mode='file'
            

        #checking if gti files exist in the folder
        gti_files= np.array([elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        gti_files.sort()

        def extract_single_spectral(gtifile=None):

            '''
            wrapper for individual gti or full obsid spectral computations
            '''

            gti_str='' if gtifile is None else ' gtifile='+gtifile

            if gtifile is not None:
                print('Creating spectral products with gti file '+gtifile)
            else:
                print('Creating spectral products')

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            bashproc.sendline('nicerl3-spect indir='+directory+' bkgmodeltype='+bkgmodel_str+' bkgformat='+bkgmodel_mode+' '+bkg_outlang_str+
                              ' clobber='+('YES' if overwrite else 'FALSE')+gti_str)

            process_state=bashproc.expect(['Task will exit with status','DONE'],timeout=None)

            #raising an error to stop the process if the command has crashed for some reason
            if process_state==0:
                with open(directory+'/extract_lc.log') as file:
                    lines=file.readlines()

                bashproc.sendline('exit')
                extract_all_spectral_done.set()
                return lines[-1].replace('\n','')

            allfiles=glob.glob(directory+'/xti/**',recursive=True)

            #fetching the path of the spectrum and rmf file (out of pre-compiled products
            spfile=[elem for elem in allfiles if '_sr.pha' in elem and '/products/' not in elem]

            if len(spfile)>1:
                print('NICER_datared_error: Several output spectra detected for single computation')
                raise ValueError
            elif len(spfile)==0:
                print('NICER_datared_error: No spectral file detected for single computation')
                raise ValueError
            else:
                spfile=spfile[0]

            #storing the full observation ID for later
            file_id=spfile.split('/')[-1][2:].replace('_sr.pha','')
            file_dir=spfile[:spfile.rfind('/')]
            file_suffix=file_id.split(directory)[-1]

            copyfile_suffixes=['_sr.pha','_bg.rmf','_bg.py','_sk.arf','.arf','.rmf']
            copyfile_list=['ni'+file_id+elem for elem in copyfile_suffixes]

            #copying the spectral products into the main directory
            for elem_file in copyfile_list:
                os.system('cp '+os.path.join(file_dir,elem_file)+' '+os.path.join(directory,elem_file))

            #renaming all the spectral products
            prod_files=glob.glob(directory+'/ni'+directory+'**',recursive=False)

            for elem in prod_files:
                os.system('mv '+elem+' '+elem.replace('ni','').replace(file_suffix,gti_suffix))

            #updating the file names in the bg load file
            with open(directory+'/'+directory+gti_suffix+'_bg.py') as old_bgload_file:
                old_bgload_lines=old_bgload_file.readlines()

            #removing the file
            os.remove(directory+'/'+directory+gti_suffix+'_bg.py')

            #and rewritting one with updated variables
            with open(directory+'/'+directory+gti_suffix+'_bg.py','w+') as new_bgload_file:
                for line in old_bgload_lines:
                    if line.startswith('nicer_srcrmf'):
                        new_bgload_file.writelines('nicer_srcrmf="'+directory+gti_suffix+'.rmf"\n')
                    elif line.startswith('nicer_skyarf'):
                        new_bgload_file.writelines('nicer_skyarf="'+directory+gti_suffix+'_sk.arf"\n')
                    elif line.startswith('nicer_diagrmf'):
                        new_bgload_file.writelines('nicer_diagrmf="'+directory+gti_suffix+'_bg.rmf"\n')
                    else:
                        new_bgload_file.writelines(line)

        if len(gti_files)==0:
            print('no gti files detected. Computing spectral products from the entire obsid...')

        else:
            print(str(len(gti_files))+' gti files detected. Computing spectral products from individual gtis...')

        for elem_gti in gti_files:
            extract_single_spectral(elem_gti)

        #exiting the bashproc
        bashproc.sendline('exit')
        extract_all_spectral_done.set()

#### extract_lc
def extract_lc(directory,binning=1,bands='3-15',HR='6-10/3-6',overwrite=True):
    
    '''
    Wrapper for nicerl3-lc, with added matplotlib plotting of requested lightcurves and HRs
        
    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/nicerl3-lc/

    Processes a directory using the nicerl3-lc script for every band asked, then creates plots for every lc/HR requested
    
    options:
        -binning: binning of the LC in seconds
        
        -bands: bands for each lightcurve to be created. The numbers should be in keV, separated by "-", and different lightcurves by ","
                ex: to create two lightcurves for, the 1-3 and 4-12 band, use '1-3,4-12'
                
        -hr: bands to be used for the HR plot creation. A single plot is possible for now. Creates its own lightcurve bands if necessary
        
        -overwrite: overwrite products or not
s
        
    Note: can produce no output without error if no gti in the event file
    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nCreating lightcurves products...')
    
    #defining the number of lightcurves to create
    
    #decomposing for each band asked
    lc_bands=bands.split(',')+([] if HR is None else ravel_ragged([elem.split('/') for elem in HR.split(',')]).tolist())
    
    lc_bands=np.unique(lc_bands)
    
    #storing the ids for the HR bands
    id_band_num_HR=np.argwhere(HR.split('/')[0]==lc_bands)[0][0]
    id_band_den_HR=np.argwhere(HR.split('/')[1]==lc_bands)[0][0]
    
    set_var(bashproc)

    if os.path.isfile(directory+'/extract_lc.log'):
        os.system('rm '+directory+'/extract_lc.log')
        
    with StdoutTee(directory+'/extract_lc.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_lc.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout

        #checking if gti files exist in the folder
        gti_files= np.array([elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        gti_files.sort()
        def extract_single_lc(gtifile=None):

            '''
            wrapper for individual gti or full obsid spectral computations
            '''

            gti_str='' if gtifile is None else ' gtifile='+gtifile

            if gtifile is not None:
                print('Creating spectral products with gti file '+gtifile)
            else:
                print('Creating spectral products')

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            time_zero_arr=np.array([None]*len(lc_bands))

            data_lc_arr=np.array([None]*len(lc_bands))

            #storing the
            for i_lc,indiv_band in enumerate(lc_bands):

                old_files_lc=[elem for elem in glob.glob(directory+'/xti/**/*',recursive=True) if elem.endswith('.lc') and 'bin' not in elem]

                for elem_file in old_files_lc:
                    os.remove(elem_file)

                pi_band='-'.join((np.array(indiv_band.split('-')).astype(int)*100).astype(str).tolist())

                bashproc.sendline('nicerl3-lc '+directory+' pirange='+pi_band+' timebin='+str(binning)+' '+
                                  ' clobber='+('YES' if overwrite else 'FALSE')+gti_str)

                process_state=bashproc.expect(['Task aborting due','DONE'],timeout=None)

                #raising an error to stop the process if the command has crashed for some reason
                if process_state==0:
                    with open(directory+'/extract_lc.log') as file:
                        lines=file.readlines()

                    return lines[-1].replace('\n','')

                file_lc=[elem for elem in glob.glob(directory+'/xti/**/*',recursive=True) if elem.endswith('.lc')][0]

                #storing the data of the lc
                with fits.open(file_lc) as fits_lc:
                    data_lc_arr[i_lc]=fits_lc[1].data

                    time_zero=Time(fits_lc[1].header['MJDREFI']+(fits_lc[1].header['TIMEZERO']-fits_lc[1].header['LEAPINIT'])/86400,format='mjd')

                    time_zero_arr[i_lc]=str(time_zero.to_datetime())

                    #saving the lc in a different file
                    fits_lc.writeto(file_lc.replace('.lc',gti_suffix+'_'+indiv_band+'_bin_'+str(binning)+'.dat'),overwrite=True)

                #removing the direct products
                new_files_lc = [elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                                '.lc' in elem and 'bin' not in elem]

                for elem_file in new_files_lc:
                    os.remove(elem_file)

                #and plotting it
                fig_lc,ax_lc=plt.subplots(1,figsize=(10,8))

                plt.errorbar(data_lc_arr[i_lc]['TIME'],data_lc_arr[i_lc]['RATE'],xerr=float(binning),yerr=data_lc_arr[i_lc]['ERROR'],ls='-',lw=1,color='grey',ecolor='blue')

                plt.suptitle('NICER lightcurve for observation '+directory+gti_suffix+' in the '+indiv_band+' keV band')

                plt.xlabel('Time (s) after '+time_zero_arr[i_lc])
                plt.ylabel('RATE (counts/s)')

                plt.tight_layout()
                plt.savefig('./'+directory+'/'+directory+gti_suffix+'_lc_'+indiv_band+'_bin_'+str(binning)+'.png')
                plt.close()

            if time_zero_arr[id_band_num_HR]!=time_zero_arr[id_band_den_HR]:
                print('NICER_datared error: both lightcurve for the HR have different zero values')
                raise ValueError

            #creating the HR plot
            fig_hr,ax_hr=plt.subplots(1,figsize=(10,8))

            hr_vals=data_lc_arr[id_band_num_HR]['RATE']/data_lc_arr[id_band_den_HR]['RATE']

            hr_err=hr_vals*(((data_lc_arr[id_band_num_HR]['ERROR']/data_lc_arr[id_band_num_HR]['RATE'])**2+
                            (data_lc_arr[id_band_den_HR]['ERROR']/data_lc_arr[id_band_den_HR]['RATE'])**2)**(1/2))

            plt.errorbar(data_lc_arr[id_band_num_HR]['TIME'],hr_vals,xerr=binning,yerr=hr_err,ls='-',lw=1,color='grey',ecolor='blue')

            plt.suptitle('NICER HR evolution for observation '+directory+gti_suffix+' in the '+HR+' keV band')

            plt.xlabel('Time (s) after '+time_zero_arr[id_band_num_HR])
            plt.ylabel('Hardness Ratio ('+HR+' keV)')

            plt.tight_layout()
            plt.savefig('./'+directory+'/'+directory+gti_suffix+'_hr_'+indiv_band+'_bin_'+str(binning)+'.png')
            plt.close()

        if len(gti_files)==0:
            print('no gti files detected. Computing lightcurve products from the entire obsid...')

        else:
            print(str(len(gti_files))+' gti files detected. Computing lightcurve products from individual gtis...')

        for elem_gti in gti_files:
            process_state=extract_single_lc(elem_gti)

            #stopping the loop in case of crash
            if process_state is not None:

                #exiting the bashproc
                bashproc.sendline('exit')
                extract_lc_done.set()

                #raising an error to stop the process if the command has crashed for some reason
                return 'GTI '+elem_gti.split('_gti_')[1].replace('.gti','')+': '+process_state

        #exiting the bashproc
        bashproc.sendline('exit')
        extract_lc_done.set()

def extract_spectrum(directory):
    
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
        extract_spectrum_done.set()
    
def extract_background(directory,model):
    
    '''
    
    DEPRECATED
    
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
        extract_response_done.set()
        
def regroup_spectral(directory,group='opt'):
    
    '''
    Regroups NICER spectrum from an obsid directory using ftgrouppha
    
    mode:
        -opt: follows the Kastra and al. 2016 binning
        
    Currently only accepts input from extract_all_spectral
    
    '''

    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nRegrouping spectrum...')
    
    set_var(bashproc)
    
    currdir=os.getcwd()
    
    if os.path.isfile(directory+'/regroup_spectral.log'):
        os.system('rm '+directory+'/regroup_spectral.log')
        
    #deleting previously existing grouped spectra to avoid problems when testing their existence
    if os.path.isfile(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha')):
        os.remove(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha'))
            
    with StdoutTee(directory+'/regroup_spectral.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/regroup_spectral.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout

        #checking if gti files exist in the folder
        gti_files= np.array([elem for elem in glob.glob(directory + '/xti/**/*', recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        gti_files.sort()

        def regroup_single_spectral(gtifile=None):

            if gtifile is not None:
                print('Regrouping spectral products with gti file '+gtifile)
            else:
                print('Regrouping spectral products')

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            #raising an error to stop the process if the command has crashed for some reason
            if not os.path.isfile(directory+'/'+directory+gti_suffix+'_sr.pha'):
                bashproc.sendline('exit')
                regroup_spectral_done.set()
                return 'Source spectrum missing'

            allfiles=glob.glob(directory+'/xti/**',recursive=True)

            #print for saving in the log file since it doesn't show clearly otherwise
            print('ftgrouppha infile='+directory+'/'+directory+gti_suffix+'_sr.pha'+' outfile='+directory+'/'+directory+gti_suffix+'_sp_grp_'+group+
            '.pha grouptype='+group+' respfile='+directory+'/'+directory+gti_suffix+'.rmf')

            bashproc.sendline('ftgrouppha infile='+directory+'/'+directory+gti_suffix+'_sr.pha'+' outfile='+directory+'/'+directory+gti_suffix+'_sp_grp_'+group+
            '.pha grouptype='+group+' respfile='+directory+'/'+directory+gti_suffix+'.rmf')

            time.sleep(1)

            while not os.path.isfile(os.path.join(currdir,directory+'/'+directory+gti_suffix+'_sp_grp_'+group+'.pha')):
                time.sleep(2)
                print('Waiting for creation of file '+os.path.join(currdir,directory+'/'+directory+gti_suffix+'_sp_grp_'+group+'.pha'))

            bashproc.sendline('echo done')
            bashproc.expect('done')

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
            process_state=regroup_single_spectral(elem_gti)

            #stopping the loop in case of crash
            if process_state is not None:

                #exiting the bashproc
                bashproc.sendline('exit')
                regroup_spectral_done.set()

                #raising an error to stop the process if the command has crashed for some reason
                return 'GTI '+elem_gti.split('_gti_')[1].replace('.gti','')+': '+process_state

        #exiting the bashproc
        bashproc.sendline('exit')
        regroup_spectral_done.set()

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
    
    bashproc.sendline('cp --verbose '+directory+'* ../bigbatch'+' >batch_mover.log')
    
    #reasonable waiting time to make sure files can be copied
    time.sleep(2)
    
    bashproc.sendline('exit')
    batch_mover_done.set()

def clean_products(directory):

    '''

    clean products in the xti/event_cl directory

    Useful to avoid bloating with how big these files are
    '''

    product_files=[elem for elem in glob.glob(directory+'/xti/event_cl/**',recursive=True)\
                   if not elem.endswith('/')]

    print('Cleaning '+str(len(product_files))+' elements in directory '+directory+'/xti/event_cl/')

    for elem_product in product_files:
        if not elem_product.endswith('/'):
            os.remove(elem_product)

    #reasonable waiting time to make sure big files can be deleted
    time.sleep(2)

    print('Cleaning complete.')

    clean_products_done.set()

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
                        
                        #resetting the error string message
                        output_err=None
                        folder_state='Running '+curr_action
                        
                        if curr_action=='1':
                            process_obsdir(dirname,overwrite=overwrite_glob)
                            process_obsdir_done.wait()
                        if curr_action=='2':
                            select_detector(dirname,detectors=bad_detectors)
                            select_detector_done.wait()

                        if curr_action=='gti':
                            output_err=create_gtis(dirname,split=gti_split,band=gti_lc_band,binning=lc_bin,
                                        overwrite=overwrite_glob)
                            if type(output_err)==str:
                                raise ValueError
                            create_gtis_done.wait()

                        if curr_action=='fs':
                            output_err=extract_all_spectral(dirname,bkgmodel=bgmodel,language=bglanguage,overwrite=overwrite_glob)
                            if type(output_err)==str:
                                raise ValueError
                            extract_all_spectral_done.wait()
                            
                        if curr_action=='l':
                            output_err=extract_lc(dirname,binning=lc_bin,bands=lc_bands_str,HR=hr_bands_str,overwrite=overwrite_glob)
                            if type(output_err)==str:
                                raise ValueError
                            extract_lc_done.wait()

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
                            output_err=regroup_spectral(dirname,group=grouptype)
                            if type(output_err)==str:
                                raise ValueError
                            regroup_spectral_done.wait()

                        if curr_action=='m':
                            batch_mover(dirname)
                            batch_mover_done.wait()

                        if curr_action=='c':
                            clean_products(dirname)
                            clean_products_done.wait()

                        os.chdir(startdir)
                    folder_state='Done'

                except:
                    #signaling unknown errors if they happened
                    if 'Running' in folder_state:
                        print('\nError while '+folder_state)
                        folder_state=folder_state.replace('Running','Aborted at')+('' if output_err is None else ' --> '+output_err)
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

                    if curr_action=='gti':
                        output_err=create_gtis(dirname,split=gti_split,band=gti_lc_band,binning=lc_bin,
                                    overwrite=overwrite_glob)
                        if type(output_err) == str:
                            folder_state=output_err
                        else:
                            pass
                        create_gtis_done.wait()

                    if curr_action == 'fs':
                        output_err = extract_all_spectral(dirname, bkgmodel=bgmodel, language=bglanguage,
                                                          overwrite=overwrite_glob)
                        if type(output_err) == str:
                            folder_state=output_err
                        else:
                            pass
                        extract_all_spectral_done.wait()

                    if curr_action == 'l':
                        output_err = extract_lc(dirname, binning=lc_bin, bands=lc_bands_str, HR=hr_bands_str,
                                                overwrite=overwrite_glob)
                        if type(output_err) == str:
                            folder_state=output_err
                        else:
                            pass
                        extract_lc_done.wait()
                            
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
                        output_err=regroup_spectral(dirname,group=grouptype)

                        if type(output_err) == str:
                            folder_state=output_err
                        else:
                            pass
                        regroup_spectral_done.wait()

                    if curr_action=='m':
                        batch_mover(dirname)
                        batch_mover_done.wait()

                    if curr_action=='c':
                        clean_products(dirname)
                        clean_products_done.wait()

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
                select_detector(absdir,detectors=bad_detectors)
                select_detector_done.wait()
            if curr_action=='s':
                extract_spectrum(absdir)
                extract_spectrum_done.wait()
            if curr_action=='b':
                extract_background(absdir,model=bgmodel)
                extract_background_done.wait()
            if curr_action=='r':
                extract_response(absdir)
                extract_response_done.wait()
            if curr_action=='g':
                regroup_spectral(absdir,group=grouptype)
                regroup_spectral_done.wait()
            if curr_action=='m':
                batch_mover(absdir)
                batch_mover_done.wait()
            if curr_action == 'c':
                clean_products(absdir)
                clean_products_done.wait()

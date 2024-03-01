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


from astropy.time import Time,TimeDelta

from general_tools import file_edit,ravel_ragged,interval_extract,MinorSymLogLocator,str_orbit

#astro imports
from astropy.io import fits


'''peak detection'''
from findpeaks import findpeaks
#mask to polygon conversion
from imantics import Mask
from shapely.geometry import Polygon,Point
#mask propagation for the peak detection
from scipy.ndimage import binary_dilation

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

#global choices
ap.add_argument("-a","--action",nargs='?',help='Give which action(s) to proceed,separated by comas.',
                default='c,1,gti,fs,l,g,m',type=str)
#default: 1,gti,fs,l,g,m,c

ap.add_argument("-over",nargs=1,help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder',default=True,type=bool)

#directory level overwrite (not active in local)
ap.add_argument('-folder_over',nargs=1,help='relaunch action through folders with completed analysis',default=False,type=bool)
ap.add_argument('-folder_cont',nargs=1,help='skip all but the last 2 directories in the summary folder file',default=False,type=bool)
#note : we keep the previous 2 directories because bug or breaks can start actions on a directory following the initially stopped one

#action specific overwrite

#1. process

#These arguments should be adjusted with lots of attention after looking at both the flare plots
#the summary of temporal filtering logged in process_obsdir, and the resulting spectra

#should only be done in very extreme cases
ap.add_argument('-keep_SAA',nargs=1,help='keep South Atlantic Anomaly (SAA) Periods',type=bool,default=True)

ap.add_argument('-overshoot_limit',nargs=1,help='overshoot event rate limit',type=float,default=30)

ap.add_argument('-undershoot_limit',nargs=1,help='undershoot event rate limit',type=float,default=500)

ap.add_argument('-min_gti',nargs=1,help='minimum gti size',type=float,default=5.0)

ap.add_argument('-erodedilate',nargs=1,help='Erodes increasingly more gtis around the excluded intervals',
                type=float,default=5.0)

#gti
#keyword for split: split_timeinsec
ap.add_argument('-gti_split',nargs=1,help='GTI split method',default='orbit+flare',type=str)
ap.add_argument('-flare_method',nargs=1,help='Flare extraction method(s)',default='clip+peak',type=str)

#note: not used currently
ap.add_argument('-gti_lc_band',nargs=1,help='Band for the lightcurve used for GTI splitting',
                default='12-15',type=str)

#lightcurve
ap.add_argument('-lc_bin',nargs=1,help='Gives the binning of all lightcurces/HR evolutions (in s)',default=1,type=str)
#note: also defines the binning used for the gti definition

ap.add_argument('-lc_bands_str',nargs=1,help='Gives the list of bands to create lightcurves from',default='3-10',type=str)
ap.add_argument('-hr_bands_str',nargs=1,help='Gives the list of bands to create hrsfrom',default='6-10/3-6',type=str)


#spectra
ap.add_argument('-bg',"--bgmodel",help='Give the background model to use for the data reduction',default='scorpeon_script',type=str)
ap.add_argument('-bg_lang',"--bg_language",
        help='Gives the language output for the script generated to load spectral data into either PyXspec or Xspec',
                default='python',type=str)

ap.add_argument('-gtype',"--grouptype",help='Give the group type to use in regroup_spectral',default='opt',type=str)

#deprecated
ap.add_argument('-baddet','--bad_detectors',help='List detectors to exclude from the data reduction',default='-14,-34,-54',type=str)

    
ap.add_argument('-heasoft_init_alias',help="name of the heasoft initialisation script alias",default="heainit",type=str)
ap.add_argument('-caldbinit_init_alias',help="name of the caldbinit initialisation script alias",default="caldbinit",type=str)
ap.add_argument('-alias_3C50',help="bash alias for the 3C50 directory",default='$NICERBACK3C50',type=str)

args=ap.parse_args()

load_functions=args.load_functions

startdir=args.startdir
action_list=args.action.split(',')
local=args.local
folder_over=args.folder_over
folder_cont=args.folder_cont
overwrite_glob=args.over
catch_errors=args.catch_errors
bgmodel=args.bgmodel
bglanguage=args.bg_language

keep_SAA=args.keep_SAA
overshoot_limit=args.overshoot_limit
undershoot_limit=args.undershoot_limit
min_gti=args.min_gti
erodedilate=args.erodedilate

gti_split=args.gti_split
gti_lc_band=args.gti_lc_band
flare_method=args.flare_method

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
clean_all_done=threading.Event()

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

def process_obsdir(directory,overwrite=True,keep_SAA=False,overshoot_limit=30.,undershoot_limit=500.,
                                            min_gti=5.0,erodedilate=5.0):
    
    '''
    Processes a directory using the nicerl2 script

    options (default values are the defaults of the function)

    see https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/nimaketime.html for details of these

    -keep_SAA: remove South Atlantic Anomaly filtering (default False).
                Should only be done in specific cases, see

    -overshoot/undershoot limit: limit above which to filter the events depending on these quantities

    -min_gti minimum gti

    -erodedilate: erodes gtis arode the excluded intervals

    '''
    
    bashproc=pexpect.spawn("/bin/bash",encoding='utf-8')
    
    print('\n\n\nEvent filtering...')
    
    set_var(bashproc)
        
    if os.path.isfile(directory+'/process_obsdir.log'):
        os.system('rm '+directory+'/process_obsdir.log')
        
    with StdoutTee(directory+'/process_obsdir.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/process_obsdir.log',buff=1,file_filters=[_remove_control_chars]):

        bashproc.logfile_read=sys.stdout

        bashproc.sendline('nicerl2 indir=' + directory +' clobber=' + ('YES' if overwrite else 'FALSE') +
                          ' nicersaafilt=' + ('NO' if keep_SAA else 'YES') +
                          ' overonly_range=0-%.1f'%overshoot_limit +
                          ' underonly_range=0-%.1f'%undershoot_limit +
                          ' mingti=%.1f' % min_gti +
                          ' erodedilate=%.1f' % erodedilate)

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

def create_gtis(directory,split='orbit+flare',band='3-15',binning=1,overwrite=True,clip_method='median',
                clip_sigma=2.,clip_band='8-12',flare_method='clip+peak',peak_score_thresh=2.):
    '''
    wrapper for a function to split nicer obsids into indivudal portions


    before:
    first creates a lightcurve with the chosen binning then uses it to define
    individual gtis

    now:
    computes the gti from the unusual parts of the 8-12keV lightcurve

    modes (combinable):
        -orbit:split each obs into each individual nicer observation period

        -clip: isolates background flare periods in each observation and creates individual
        note that all individual flare/dip periods in single orbits are grouped together

        -variability: isolates true flares/dips from the lightcurve already treated for the flares
        #should take the main lightcurve instead of the >20keV. To be implemented

        -split_X: on top of cutting splits and flares, splits each orbit in individual periods of X seconds for
                  time-resolved spectroscopy

    clip_method:
        -median or mean to clip from the mean or from the median

    clip_sigma:
        -sigma for which to apply clipping to

    clip_band:
        -which file or info from the mkf file to use to clip the flares
        currently implemented:
        -8-12keV
        -overshoot

    NOTE: requires sas and a sasinit alias to initialize it (to use tabgtigen)

    '''

    #ensuring a good obsid name even in local
    if directory=='./':
        obsid=os.getcwd().split('/')[-1]
    else:
        obsid=directory

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

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

    if os.path.isfile(os.path.join(directory + '/create_gtis.log')):
        os.system('rm ' + os.path.join(directory + '/create_gtis.log'))

    with StdoutTee(os.path.join(directory + '/create_gtis.log'), mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(os.path.join(directory + '/create_gtis.log'), buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        '''
        new method for the overshoots following https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/flares/
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

            counts_8_12=data_mkf['FPM_XRAY_PI_0800_1200']

            counts_overshoot=data_mkf['FPM_OVERONLY_COUNT']

            counts_undershoot = data_mkf['FPM_UNDERONLY_COUNT']

            cutoff_rigidity=data_mkf['COR_SAX']

        #fetching the standard gti file applied to the events
        #note that in some cases nimaketime is not created so we fetch the GTIs from the cleaned event file instead

        #file_nimaketime=glob.glob(os.path.join(directory,'**/**/nimaketime.gti'),recursive=True)[0]

        file_evt=glob.glob(os.path.join(directory,'**/**/**mpu7_cl.evt'),recursive=True)[0]

        #how this was done can be checked in process_obsdir, the details of the evolution of the gtis is written
        #see https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/nimaketime.html

        with fits.open(file_evt) as hdul:

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

        if 'split' in split:

            split_sampling=float([elem for elem in split.split('+') if 'split' in elem][0].split('_')[1])

            for i_orbit in range(n_orbit):
            
                # computing the non-gti intervals from nimaketime to begin with a non-broken interval
                id_nongti_nimkt = []
    
                for elem_gti in id_gti_orbit[i_orbit]:
                    # testing if the gti is in one of the gtis of the nimaketime
                    if not ((time_obs[elem_gti] >= gti_nimkt_arr.T[0]) & (time_obs[elem_gti] <= gti_nimkt_arr.T[1])).any():
                        # and storing if that's not the case
                        id_nongti_nimkt += [elem_gti]

                #computing the starting id, a bit convoluted but not to depend on the binning
                start_id=np.argwhere([elem not in id_nongti_nimkt for elem in id_gti_orbit[i_orbit]])\
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

        fig_flares,ax_flares=plt.subplots(1,figsize=(12,8))

        ax_flares.set_xlabel('Time (s) after ' + obs_start_str)
        ax_flares.set_ylabel('Count Rate (counts/s)')

        #we just want something above 0 here while keeping a log scale but allowing 0 counts
        ax_flares.set_yscale('symlog', linthresh=0.1, linscale=0.1)
        ax_flares.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.1))

        ax_rigidity=ax_flares.twinx()
        ax_rigidity.set_ylabel('Cutoff Rigidity (Gev/c)')

        for i_orbit in range(n_orbit):

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]],counts_035_8[id_gti_orbit[i_orbit]],
                               color='red',label='0.35-8 keV Count Rate' if i_orbit==0 else '')

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]],counts_8_12[id_gti_orbit[i_orbit]],
                               color='blue',label='8-12 keV Count Rate' if i_orbit==0 else '')

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_overshoot[id_gti_orbit[i_orbit]],
                               color='orange',label='Overshoot Rate (>20keV)' if i_orbit==0 else '')

            ax_rigidity.plot(time_obs[id_gti_orbit[i_orbit]],cutoff_rigidity[id_gti_orbit[i_orbit]],
                             color='green',label='Cutoff Rigidity' if i_orbit==0 else '')

        ax_rigidity.axhline(1.5, 0, 1, color='green', ls='--', label='Upper limit for risky regions')
        ax_flares.axhline(30, 0, 1, color='orange', ls='--', label='Default nicerl2 flare cut')

        ax_flares.legend(loc='upper left')
        ax_rigidity.legend(loc='upper right')
        ax_flares.set_ylim(0,ax_flares.get_ylim()[1])

        plt.tight_layout()

        plt.savefig(os.path.join(directory,obsid+'-global_flares.png'))

        #can be modified if needed

        if clip_band=='8-12':
            flare_lc=counts_8_12
        elif clip_band=='overshoot':
            flare_lc=counts_overshoot

        clip_lc=np.where(np.isnan(flare_lc),0,flare_lc)

        if 'flare' in split:
            id_gti=[]
            id_flares=[]
            id_dips=[]

            for i_orbit,elem_gti_orbit in enumerate(id_gti_orbit):

                elem_id_flares=[]

                if 'clip' in flare_method:
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
                                 (clip_lc[elem])>clip_base+clip_sigma*clip_std and clip_lc[elem]>=clip_base+np.log10(2)]


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

                    elem_id_flares+=peak_region_multi

                #ensuring no repeats
                elem_id_flares=np.unique(elem_id_flares).tolist()

                #defining the gtis after the flares have been defined
                elem_id_gti = [elem for elem in elem_gti_orbit if elem not in elem_id_flares]

                id_gti += [elem_id_gti]
                id_flares += [elem_id_flares]

                if 'split' in split:
                    for i_split in range(len(split_gti_arr[i_orbit])):
                        split_gti_arr[i_orbit][i_split]=[elem for elem in split_gti_arr[i_orbit][i_split]\
                                                     if elem not in id_flares[i_orbit]]


        else:
            id_gti=id_gti_orbit
            id_flares=[]
            id_dips=[]

        #creating individual orbit figures
        for i_orbit in range(n_orbit):

            fig_flares, ax_flares = plt.subplots(1, figsize=(12, 8))

            ax_flares.set_xlabel('Time (s) after ' + obs_start_str)
            ax_flares.set_ylabel('Count Rate (counts/s)')
            ax_rigidity = ax_flares.twinx()
            ax_rigidity.set_ylabel('Cutoff Rigidity (Gev/c)')

            # we just want something above 0 here while keeping a log scale but allowing 0 counts
            ax_flares.set_yscale('symlog', linthresh=0.1, linscale=0.1)
            ax_flares.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.1))

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_035_8[id_gti_orbit[i_orbit]],
                               color='red', label='0.35-8 keV Count Rate')

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_8_12[id_gti_orbit[i_orbit]],
                               color='blue', label='8-12 keV Count Rate')

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_overshoot[id_gti_orbit[i_orbit]],
                               color='orange', label='Overshoot Rate (>20keV)')

            ax_flares.errorbar(time_obs[id_gti_orbit[i_orbit]], counts_undershoot[id_gti_orbit[i_orbit]],
                               color='brown', label='Undershoot Rate')

            ax_rigidity.plot(time_obs[id_gti_orbit[i_orbit]], cutoff_rigidity[id_gti_orbit[i_orbit]],
                             color='green', label='Cutoff Rigidity')

            #flare and gti intervals
            for id_inter,list_inter in enumerate(list(interval_extract(id_gti[i_orbit]))):
                ax_rigidity.axvspan(time_obs[min(list_inter)], time_obs[max(list_inter)], color='grey', alpha=0.2,
                                    label='standard gtis' if id_inter==0 else '')

            for id_inter,list_inter in enumerate(list(interval_extract(id_flares[i_orbit]))):
                ax_rigidity.axvspan(time_obs[min(list_inter)], time_obs[max(list_inter)], color='blue', alpha=0.2,
                                    label='flare gtis' if id_inter==0 else '')

            #computing the non-gti intervals from nimaketime
            id_nongti_nimkt=[]

            for elem_gti in id_gti_orbit[i_orbit]:
                #testing if the gti is in one of the gtis of the nimaketime
                if not ((time_obs[elem_gti] >= gti_nimkt_arr.T[0]) & (time_obs[elem_gti] <= gti_nimkt_arr.T[1])).any():
                    #and storing if that's not the case
                    id_nongti_nimkt+=[elem_gti]

            #and plotting
            for id_inter,list_inter in enumerate(list(interval_extract(id_nongti_nimkt))):
                ax_rigidity.axvspan(time_obs[min(list_inter)], time_obs[max(list_inter)], color='red', alpha=0.1,
                                    label='std nimaketime excluded intervals' if id_inter==0 else '')


            #plotting the split gti intervals if in the right mode
            if 'split' in split:

                for id_inter, list_inter in enumerate(split_gti_arr[i_orbit]):

                    ax_rigidity.axvspan(time_obs[min(list_inter)], time_obs[max(list_inter)],
                                        color='green', alpha=0.1,
                                        label='split intervals '+
                                              ('(flare cuts included)' if 'flare' in split else '')\
                                              if id_inter == 0 else '')

            ax_rigidity.axhline(1.5, 0, 1, color='green', ls='--', label='Upper limit for risky regions')
            ax_flares.axhline(30, 0, 1, color='orange', ls='--', label='Default overshoot cut')
            ax_flares.axhline(500, 0, 1, color='brown', ls='--', label='Default undershoot cut')


            ax_flares.legend(loc='upper left')
            ax_rigidity.legend(loc='upper right')
            ax_flares.set_ylim(0, ax_flares.get_ylim()[1])
            plt.tight_layout()

            #note that str_orbit adds 1 to the counter
            plt.savefig(os.path.join(directory,obsid+'-'+str_orbit(i_orbit)+
                        '_flares.png'))
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

        def create_gti_files(id_gti,data_lc,orbit_prefix,suffix):

            '''
            creates a gti file from a list of indexes of times which will be picked in data_lc

            1.creates a copy of the mkf file with an additional column with a gti mask

            2.creates the gti file itself using sas's tabgtigen
            '''

            if len(id_gti)==0:
                return

            # Here we use the housekeeping file as the fits base for the gti mask file
            fits_gti=fits.open(file_mkf)

            #creating a custom gti 'mask' file
            gti_column=fits.ColDefs([fits.Column(name='IS_GTI', format='I',
                                    array=np.array([1 if i in id_gti else 0 for i in range(len(data_lc))]))])

            #replacing the hdu with a hdu containing it
            fits_gti[1]=fits.BinTableHDU.from_columns(fits_gti[1].columns[:2]+gti_column)
            fits_gti[1].name='IS_GTI'

            lc_mask_path = os.path.join(directory,'xti',obsid+'_gti_mask_'+orbit_prefix+suffix)+'.fits'

            if os.path.isfile(lc_mask_path):
                os.remove(lc_mask_path)

            fits_gti.writeto(lc_mask_path)

            #waiting for the file to be created
            while not os.path.isfile(lc_mask_path):
                time.sleep(0.1)

            #creating the orbit gti expression
            gti_path=os.path.join(directory,'xti',obsid+'_gti_'+orbit_prefix+suffix)+'.gti'

            print(gti_path)
            bashproc.sendline('tabgtigen table='+lc_mask_path+' expression="IS_GTI==1" gtiset='+gti_path)

            #this shouldn't take too long so we keep the timeout
            #two expects because there's one for the start and another for the end
            bashproc.expect('tabgtigen:- tabgtigen')
            bashproc.expect('tabgtigen:- tabgtigen')

            '''
            There is an issue with the way tabgtigen creates the exposure due to a lacking keyword
            To ensure things work correctly, we remake the contents of the file and keep the header
            '''

            #preparing the list of gtis to replace manually
            gti_intervals=np.array(list(interval_extract(id_gti))).T

            #opening and modifying the content of the header in the gti file for NICER
            with fits.open(gti_path,mode='update') as hdul:

                #for some reason we don't get the right values here so we recreate them
                # creating a custom gti 'mask' file

                #storing the current header
                prev_header=hdul[1].header

                #creating a START and a STOP column in "standard" GTI fashion
                #note: the 0.5 is there to allow the initial and final second bounds
                gti_column_start = fits.ColDefs([fits.Column(name='START', format='D',
                                                       array=np.array([time_obs[elem]+start_obs_s-0.5 for elem in gti_intervals[0]]))])
                gti_column_stop = fits.ColDefs([fits.Column(name='STOP', format='D',
                                                       array=np.array([time_obs[elem]+start_obs_s+0.5 for elem in gti_intervals[1]]))])

                #replacing the hdu
                hdul[1]= fits.BinTableHDU.from_columns(gti_column_start + gti_column_stop)

                #replacing the header
                hdul[1].header=prev_header

                #Changing the reference times
                hdul[1].header['MJDREF']=56658+7.775925925925930E-04

                # hdul[1].header['MJDREFI']=56658
                # hdul[1].header['MJDREFF']=7.775925925925930E-04

                #and the gti keywords
                hdul[1].header['ONTIME']=len(id_gti)
                hdul[1].header['TSTART']=hdul[1].data['START'][0]-start_obs_s
                hdul[1].header['TSTOP'] = hdul[1].data['STOP'][-1]-start_obs_s

                hdul.flush()

        for i_orbit in range(n_orbit):

            if 'split' in split:
                #create the gti files with a "S" keyword and keeping the orbit information in the name
                for i_split,split_gtis in enumerate(split_gti_arr[i_orbit]):
                    if len(split_gtis)>0:
                        create_gti_files(split_gtis,flare_lc,str_orbit(i_orbit),'S'+str_orbit(i_split))
            else:
                create_gti_files(id_gti[i_orbit],flare_lc,str_orbit(i_orbit),'')

            if len(id_flares[i_orbit])>0:
                create_gti_files(id_flares[i_orbit],flare_lc,str_orbit(i_orbit), 'F')

            # create_gti_files(id_dips[id_orbit],flare_lc,str_orbit(i_orbit), 'D')
        #
        # #only deleting the lightcurve files after everything has been finished
        # for elem_file in new_files_lc:
        #     os.remove(elem_file)

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
        gti_files= np.array([elem for elem in glob.glob(os.path.join(directory,'xti/**'), recursive=True) if
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

            process_state=bashproc.expect(['DONE','ERROR: could not find UFA file','Task aborting due to zero EXPOSURE'],timeout=None)

            #raising an error to stop the process if the command has crashed for some reason
            if process_state>2:
                with open(directory+'/extract_all_spectral.log') as file:
                    lines=file.readlines()

                bashproc.sendline('exit')
                extract_all_spectral_done.set()
                return lines[-1].replace('\n','')

            if process_state in [1,2]:

                #skipping the computation
                return 'skip'

            allfiles=glob.glob(os.path.join(directory,'xti/**'),recursive=True)

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
            process_state=extract_single_spectral(elem_gti)

            #stopping the loop in case of crash
            if process_state not in [None,'skip']:

                #exiting the bashproc
                bashproc.sendline('exit')
                extract_all_spectral_done.set()

                #raising an error to stop the process if the command has crashed for some reason
                return 'GTI '+elem_gti.split('_gti_')[1].replace('.gti','')+': '+process_state


        #exiting the bashproc
        bashproc.sendline('exit')
        extract_all_spectral_done.set()

#### extract_lc
def extract_lc(directory,binning=10,bands='3-12',HR='6-10/3-6',overwrite=True):
    
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
    lc_bands=([] if HR is None else ravel_ragged([elem.split('/') for elem in HR.split(',')]).tolist())+bands.split(',')
    
    lc_bands=np.unique(lc_bands)[::-1]

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
        gti_files= np.array([elem for elem in glob.glob(os.path.join(directory,'xti/**') , recursive=True) if
                        elem.endswith('.gti') and '_gti_' in elem and '_gti_mask_' not in elem])

        gti_files.sort()
        def extract_single_lc(gtifile=None):

            '''
            wrapper for individual gti or full obsid spectral computations
            '''

            gti_str='' if gtifile is None else ' gtifile='+gtifile

            if gtifile is not None:
                print('Creating lightcurve products with gti file '+gtifile)
            else:
                print('Creating lightcurve products products from the whole observation...')

            #suffix for naming products
            gti_suffix='' if gtifile is None else '-'+(gtifile[gtifile.rfind('/')+1:].split('_gti_')[1]).replace('.gti','')

            time_zero_arr=np.array([None]*len(lc_bands))

            data_lc_arr=np.array([None]*len(lc_bands))

            #storing the
            for i_lc,indiv_band in enumerate(lc_bands):

                old_files_lc=[elem for elem in glob.glob(os.path.join(directory,'xti/**/*'),recursive=True) if elem.endswith('.lc') and 'bin' not in elem]

                for elem_file in old_files_lc:
                    os.remove(elem_file)

                pi_band='-'.join((np.array(indiv_band.split('-')).astype(int)*100).astype(str).tolist())

                bashproc.sendline('nicerl3-lc '+directory+' pirange='+pi_band+' timebin='+str(binning)+' '+
                                  ' clobber='+('YES' if overwrite else 'FALSE')+gti_str)

                process_state=bashproc.expect(['DONE','ERROR: could not recompute normalized RATE/ERROR','Task aborting due','Task nicerl3-lc'],timeout=None)

                #raising an error to stop the process if the command has crashed for some reason
                if process_state>1:
                    with open(directory+'/extract_lc.log') as file:
                        lines=file.readlines()

                    return lines[-1].replace('\n','')

                if process_state==1:

                    #emptying the buffer
                    bashproc.expect('Task nicerl3-lc')

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

                file_lc=[elem for elem in glob.glob(os.path.join(directory,'xti/**/*'),recursive=True) if elem.endswith('.lc')][0]

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
                    fits_lc.writeto(file_lc.replace('.lc',gti_suffix+'_'+indiv_band+'_bin_'+str(binning)+'.dat'),overwrite=True)

                #removing the direct products
                new_files_lc = [elem for elem in glob.glob(os.path.join(directory, 'xti/**/*'), recursive=True) if
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
            if process_state not in [None,'skip']:

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

    currdir = os.getcwd()

    print('\n\n\nRegrouping spectrum...')
    

    if os.path.isfile(directory+'/regroup_spectral.log'):
        os.system('rm '+directory+'/regroup_spectral.log')
        
    #deleting previously existing grouped spectra to avoid problems when testing their existence
    if os.path.isfile(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha')):
        os.remove(os.path.join(currdir,directory,directory+'_sp_grp_'+group+'.pha'))

    with StdoutTee(directory+'/regroup_spectral.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/regroup_spectral.log',buff=1,file_filters=[_remove_control_chars]):

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
                process_state = spawn.expect(['done', 'terminating with status -1'], timeout=30)

                assert process_state==0, 'Issue when regrouping'
                return ''

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

            bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')
            set_var(bashproc)
            bashproc.logfile_read = sys.stdout

            process_state=regroup_single_spectral(bashproc,elem_gti)

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

    Note: we keep the cleaned event file and nimaketime because these are very useful when checking
    '''

    product_files=[elem for elem in glob.glob(os.path.join(directory,'xti/event_cl/**'),recursive=True)\
                   if not elem.endswith('/') and not elem.endswith('nimaketime.gti') and not elem.endswith('mpu7_cl.evt')]

    print('Cleaning '+str(len(product_files))+' elements in directory '+os.path.join(directory,'xti/event_cl/'))

    for elem_product in product_files:
        if not elem_product.endswith('/'):
            os.remove(elem_product)

    #reasonable waiting time to make sure big files can be deleted
    time.sleep(2)

    print('Cleaning complete.')

    clean_products_done.set()

def clean_all(directory):

    '''

    clean products in the xti/event_cl directory

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
        time.sleep(2)

        print('Cleaning complete.')

    clean_all_done.set()

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

if load_functions:
    breakpoint()


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
                            process_obsdir(dirname,overwrite=overwrite_glob,keep_SAA=keep_SAA,overshoot_limit=overshoot_limit,
                                                    undershoot_limit=undershoot_limit,min_gti=min_gti,erodedilate=erodedilate)
                            process_obsdir_done.wait()
                        if curr_action=='2':
                            select_detector(dirname,detectors=bad_detectors)
                            select_detector_done.wait()

                        if curr_action=='gti':
                            output_err=create_gtis(dirname,split=gti_split,band=gti_lc_band,binning=lc_bin,
                                        overwrite=overwrite_glob,flare_method=flare_method)
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

                        if curr_action=='fc':
                            clean_all(dirname)
                            clean_all_done.wait()

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
                        process_obsdir(dirname,overwrite=overwrite_glob,keep_SAA=keep_SAA,overshoot_limit=overshoot_limit,
                                                undershoot_limit=undershoot_limit,min_gti=min_gti,erodedilate=erodedilate)
                        process_obsdir_done.wait()
                    if curr_action=='2':
                        select_detector(dirname,detectors=bad_detectors)
                        select_detector_done.wait()

                    if curr_action=='gti':
                        output_err=create_gtis(dirname,split=gti_split,band=gti_lc_band,binning=lc_bin,
                                    overwrite=overwrite_glob,flare_method=flare_method)
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

                    if curr_action=='fc':
                        clean_all(dirname)
                        clean_all_done.wait()

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
                process_obsdir(absdir,overwrite=overwrite_glob,keep_SAA=keep_SAA,overshoot_limit=overshoot_limit,
                                        undershoot_limit=undershoot_limit,min_gti=min_gti,erodedilate=erodedilate)
                process_obsdir_done.wait()
            if curr_action=='2':
                select_detector(absdir,detectors=bad_detectors)
                select_detector_done.wait()

            if curr_action=='gti':
                output_err = create_gtis(absdir, split=gti_split, band=gti_lc_band, binning=lc_bin,
                                         overwrite=overwrite_glob,flare_method=flare_method)
                create_gtis_done.wait()

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

            if curr_action == 'fc':
                clean_all(absdir)
                clean_all_done.wait()

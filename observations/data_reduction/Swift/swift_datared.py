#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:32:37 2022

@author: parrama

notes on xrt pipeline for XRT:
it doesn't create an rmf because the default option is to take the standard rmfs in caldb
see https://www.swift.ac.uk/analysis/Gain_RMF_releases.html for the most up-to-date calibrations
see https://swift.gsfc.nasa.gov/analysis/xrt_swguide_v1_2.pdf for standard grade (PC 0-12, WT 0-2)
"""

import argparse
import shutil

import numpy as np
import pandas as pd
import os
import glob
import pexpect
import sys
import time
from scipy.stats import norm as scinorm
from tee import StdoutTee,StderrTee
import re
from general_tools import file_edit

##for batanalysis (see https://github.com/parsotat/BatAnalysis/blob/main/notebooks/trial_NGC2992.ipynb)
import glob
import os
import sys
import batanalysis as ba
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.io import fits
from pathlib import Path
#currently cloned from fork to allow modifs
#pip install git+https://github.com/Parrazyte/BatAnalysis
import swiftbat
import swiftbat.swutil as sbu
import pickle

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import warnings

from joblib import Parallel, delayed

#function to remove (most) control chars
def _remove_control_chars(message):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)


def source_catal(dirpath):
    '''
    Tries to identify a Simbad object from the directory structure
    '''

    # splitting the directories and searching every name in Simbad
    dir_list = dirpath.split('/')[1:]

    # removing a few problematic names
    crash_sources = ['M2', 'home', 'outputmos', 'BlackCAT', '']
    # as well as obsid type names that can cause crashes
    for elem_dir in dir_list:
        if len(elem_dir) == 10 and elem_dir.isdigit() or elem_dir in crash_sources:
            dir_list.remove(elem_dir)

    # Simbad.query_object gives a warning for a lot of folder names so we just skip them
    obj_list = []
    for elem_dir in dir_list:
        try:
            with warnings.catch_warnings():
                # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
                # warnings.filterwarnings('ignore','.*Identifier not found.*',)
                warnings.filterwarnings('ignore', category=UserWarning)
                elem_obj = Simbad.query_object(elem_dir)
                if type(elem_obj) != type(None):
                    obj_list +=[elem_obj]
        except:
            breakpoint()
            print('\nProblem during the Simbad query. This is the current directory list:')
            print(dir_list)
            return 'Problem during the Simbad query.'

    if len(obj_list)==0:
        print("\nSimbad didn't recognize any object name in the directories.")
        breakpoint()


    # if we have at least one detections, it is assumed the "last" find is the name of the object
    obj_catal = obj_list[-1]

    print('\nValid source(s) detected. Object name assumed to be ' + obj_catal['main_id'])

    return obj_catal

def untar_spectra():
    '''
    untars every file in the arborescence
    '''

    tarfiles=glob.glob('**/**.tar.gz',recursive=True)+glob.glob('**.tar.gz',recursive=True)
    currdir=os.getcwd()

    for elem_tar in tarfiles:
        print('untaring '+elem_tar)
        os.chdir(elem_tar[:elem_tar.rfind('/')+1])
        os.system('tar -zxvf '+elem_tar[elem_tar.rfind('/')+1:]+' --one-top-level')
        time.sleep(0.1)
        os.chdir(currdir)

def merge_swift_spectra_OT(overwrite=True):

    '''
    moves all swift spectral products with extensions from the online tool
    to a merge "bigbatch" directory
    '''
    
    allfiles=[elem for elem in glob.glob('**',recursive=True) if 'bigbatch' not in elem]
    specfiles=[elem for elem in allfiles if elem.endswith('source.pi') or elem.endswith('back.pi')\
               or elem.endswith('.rmf') or elem.endswith('.arf')]
    
    os.system('mkdir -p bigbatch')
    currdir=os.getcwd()
    
    for elemfile in specfiles:
        if not os.path.isfile(os.path.join(currdir,'bigbatch','xrt'+elemfile.split('/')[-1].replace('Obs_','')))\
                              or overwrite:
            shutil.copy(elemfile,os.path.join(currdir,'bigbatch','xrt'+elemfile.split('/')[-1].replace('Obs_','')))
            time.sleep(0.1)
        
    
def regroup_swift_spectra_OT(extension='source.pi',group='opt',skip_started=True):
    
    '''
    run above the bigbatch directory
    To be launched above all spectra with extensions from the online tool to regroup
    Note: for now, needs to be launched a bunch of times because things dont work every time for some reason
    '''

    def ft_group(file,grptype,heas_proc):
        
        '''wrapper for the command'''
        
        heas_proc.sendline('ftgrouppha infile='+file+' outfile='+file.replace('.','_grp_'+grptype+'.')+' grouptype='+grptype+
                           ' respfile='+file.replace('source','').replace(file[file.rfind('.'):],'.rmf'))
        
        heas_proc.sendline('echo done')
        heas_proc.expect('done')
        
    currdir=os.getcwd()
    assert ' ' not in currdir,'Issue: cannot have spaces in folder arborescence'
    allfiles=glob.glob('**',recursive=True)
    speclist=[elem for elem in allfiles if elem.endswith(extension) and 'bigbatch' in elem]
    
    speclist.sort()
    

    # if skip_started:
    #     pha2_spectra=[elem for elem in pha2_spectra if\
    #                 '/'.join(elem.split('/')[:-1])+('' if len(elem.split('/'))==1 else '/')+elem.split('/')[-1].split('_')[0]+'_heg_-1_grp_'+group+'.pha' not in allfiles or\
    #                 '/'.join(elem.split('/')[:-1])+('' if len(elem.split('/'))==1 else '/')+elem.split('/')[-1].split('_')[0]+'_heg_1_grp_'+group+'.pha' not in allfiles]
        
    for ind,specpath in enumerate(speclist):

        if skip_started and os.path.isfile(specpath.replace('.','_grp_'+group+'.')):
            print('\nSpectrum '+specpath+' already grouped')            
            continue

        # spawning heasoft terminal for Kastra grouping
        heas_proc = pexpect.spawn('/bin/bash', encoding='utf-8')
        heas_proc.logfile = sys.stdout
        heas_proc.sendline('\nheainit')

        heas_proc.sendline('cd ' + os.path.join(currdir, 'bigbatch'))

        specfile=specpath.split('/')[-1]

        #stat grouping
        if group is not None:
            
            if group=='opt':

                ft_group(specfile,grptype='opt',heas_proc=heas_proc)
                time.sleep(5)
        
        heas_proc.sendline('exit')

def convert_BAT_flux_spectra(observ_high_table_path,err_percent=90,e_low=15., e_high=50.):
    '''
    Takes an observ_high_table csv (obtainable in visual_line)
    then makes spectra using the flux values in 15-50keV
    the spectra are named after the obsid and created in the current directory

    e_low and e_high can be reduced to manage a more constraining value to help the fit
    '''

    csv=pd.read_csv(observ_high_table_path)

    #skipping two columns for the two headers
    csv_obsids=np.array(csv[csv.columns[2]][2:])

    csv_flux=np.array([csv['Flux_15-50'],csv['Flux_15-50.1'],csv['Flux_15-50.2']]).T[2:].astype(float)

    csv_BAT_expos=np.array(csv['BAT_expos_coded'][2:]).astype(float)

    # breakpoint()
    #making a text file to create the spectra

    # spawning heasoft terminal
    heas_proc = pexpect.spawn('/bin/bash', encoding='utf-8')
    heas_proc.logfile = sys.stdout
    heas_proc.sendline('\nheainit')

    for i_sp,obsid in enumerate(csv_obsids):

        #skipping cases where there is no BAT coverage
        if np.isnan(csv_BAT_expos[i_sp]):
            continue

        with open('temp_sp_base.txt','w+') as f:

            #important: need to have the full line with backspace for it to work
            f.write(str(e_low)+' '+str(e_high)+' '+str(csv_flux[i_sp][0])+' '+str(max(csv_flux[i_sp][1],csv_flux[i_sp][2])\
                                                          /scinorm.ppf((1 + err_percent/100) / 2))+'\n')


        #see https://heasarc.gsfc.nasa.gov/lheasoft/help/ftflx2xsp.html
        heas_proc = pexpect.spawn('/bin/bash', encoding='utf-8')
        heas_proc.logfile = sys.stdout
        heas_proc.sendline('\nheainit')

        heas_proc.sendline('ftflx2xsp temp_sp_base.txt '+obsid+'_BAT_regr_sp_'+str(e_low)+'_'+str(e_high)+'.pi '
                           +obsid+'_BAT_regr_'+str(e_low)+'_'+str(e_high)+'.rmf '+
                          'xunit=keV yunit="ergs/cm^2/s" clobber=yes')
        time.sleep(1)
        heas_proc.sendline('echo done')
        heas_proc.expect('done')
        time.sleep(1)
        heas_proc.sendline('exit')

        #modyfing the exposure of the file to match the exposure according to the daily BAT value
        with fits.open(obsid+'_BAT_regr_sp_'+str(e_low)+'_'+str(e_high)+'.pi',mode='update') as hdul:
            hdul[1].header['EXPOSURE']=csv_BAT_expos[i_sp]
            hdul.flush()

def fetch_BAT(object_name,date_start,date_stop,minexposure=1000,return_result=False,uksdc=False):

    '''
    wrapper around batanalysis to download some data

    see https://github.com/parsotat/BatAnalysis for installation

    -minexposure   # cm^2 after cos adjust

    -return_result
                return the downloaded obsids
    -uksdc: from the doc:
            "        uksdc : boolean
                    Set to True to download from the UK Swift SDC"
            Useful because some old data might not exist on heasarc.

    '''

    # ba.datadir(os.getcwd())

    logfile_name='./fetch_BAT_'+object_name+'_'+date_start+'_'+date_stop+'_minexp_'+str(minexposure)+'.log'

    if os.path.isfile(logfile_name):
        os.remove(logfile_name)

    with StdoutTee(logfile_name,
                   mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(logfile_name,buff=1,file_filters=[_remove_control_chars]):

        object_location = swiftbat.simbadlocation(object_name)
        object_batsource = swiftbat.source(ra=object_location[0], dec=object_location[1], name=object_name)

        # object_batsource = swiftbat.source(name=object_name)

        queryargs = dict(time=date_start+' .. '+date_stop, fields='All', resultmax=0)
        table_everything = ba.from_heasarc(**queryargs)

        exposures = np.array([object_batsource.exposure(ra=row['RA'], dec=row['DEC'], roll=row['ROLL_ANGLE'])[0] for row in
                              table_everything])
        table_exposed = table_everything[exposures > minexposure]
        print(
            f"Finding everything finds {len(table_everything)} observations, of which {len(table_exposed)} have more than {minexposure:0} cm^2 coded")


        if len(table_exposed)==0:
            return []

        print(table_exposed['OBSID'])

        result = ba.download_swiftdata(table_exposed,uksdc=uksdc)

    if return_result:
        return result

def DR_BAT(obsids='auto',noise_map_dir='environ',nprocs=1,clean_SNR=6,
           clean_expr='ALWAYS_CLEAN==T',custom_cat_path=None,reset_datadir=False):

    '''
    wrapper around batanalysis to reduce data in the current folder

    requires UNTARED noise maps in noise_map_dir

    see https://github.com/parsotat/BatAnalysis for installation

    -obsids: obsids to run the DR for. If set to auto, automatically detectes the obsid from numeric directories in the
             local folder

    -noise_map_dir: directory where the patttern maps are untarred
                    if set to 'environ', fetches the BAT_NOISE_MAP_DIR environment variable instead

    -nprocs: parallel number of procs
             if set to 1, doesn't parallelize
    -clean_SNR : argument for inpuct_dict
    -clean_expr : argument for input_dict

    -custom_cat_path: path of a custom catalog file created previously with create_custom_catalog
                        (allows to analyze sources which aren't in the current catalog)

    Note: if nothing gets out and no gti are recognized, it could be due to a lack of caldb initalization

    to tests things out if there is an issue and no GTIs are ever created:

    add in _call_batsurvey of BatSurvey in bat_survey.py
        from heasoftpy.swift import batsurvey
        input_dict['chatter']=5
        result=batsurvey(**input_dict)

    before the
            try:
            return hsp.batsurvey(**input_dict)²

    WARNING: one of the main isseus are too long directory names. Be careful about that
    '''

    if reset_datadir:
        ba.datadir(os.getcwd())

    if noise_map_dir=='environ':
        noise_map_dir_use=os.environ['BAT_NOISE_MAP_DIR']
    else:
        noise_map_dir_use=noise_map_dir

    if type(obsids)==str and obsids=='auto':
        obs_ids = [i.name for i in sorted(ba.datadir().glob("*")) if i.name.isnumeric()]
    else:
        obs_ids=obsids

    input_dict=dict(cleansnr=clean_SNR,cleanexpr=clean_expr)

    if custom_cat_path is not None:
        input_dict['incatalog']=custom_cat_path

    logfile_name='./DR_BAT.log'

    if os.path.isfile(logfile_name):
        os.system('rm '+logfile_name)

    with StdoutTee(logfile_name,
                   mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(logfile_name,buff=1,file_filters=[_remove_control_chars]):

        if nprocs==1:
            batsurvey_obs=[]
            for obsid in obs_ids:
                batsurvey_obs_indiv=ba.parallel._create_BatSurvey(obsid,input_dict=input_dict,
                                                     patt_noise_dir=noise_map_dir_use)
                if batsurvey_obs_indiv is not None:
                    batsurvey_obs+=[batsurvey_obs_indiv]
        else:
            batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids,input_dict=input_dict,
                                                     patt_noise_dir=noise_map_dir_use, nprocs=nprocs)

    return batsurvey_obs

def SA_BAT(survey_obs_list,object_name,ul_pl_index=2.5,recalc=False,nprocs=2,reset_datadir=False):

    '''
    Wrapper around batanalysis to perform spectral analysis of data in the current folder

    split from DR_BAT because each step can take very long

    survey_obs_list should be the output of DR_BAT

    '''

    if reset_datadir:
        ba.datadir(os.getcwd())

    logfile_name='./SA_BAT.log'

    if os.path.isfile(logfile_name):
        os.system('rm '+logfile_name)

    with StdoutTee(logfile_name,
                   mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(logfile_name,buff=1,file_filters=[_remove_control_chars]):

        sa_obs_list=ba.parallel.batspectrum_analysis(survey_obs_list, object_name, ul_pl_index=ul_pl_index,
                                                     recalc=recalc,nprocs=nprocs)

    return sa_obs_list

def merge_BAT_full(merge_ULs=False):
    '''
    Merges all spectral products in a bigbatch directory
    '''

    os.system('mkdir -p bigbatch')

    for elem_pha_dir in glob.glob('**/PHA_files/'):

        print('Copying files from directory '+elem_pha_dir)

        elem_sp_files=[elem for elem in glob.glob(elem_pha_dir+'**') if elem.split('.')[-1] in ['pha','rsp']]
        if not merge_ULs:
            elem_sp_files=[elem for elem in elem_sp_files if 'upperlim' not in elem]

        for elem_file in elem_sp_files:
            os.system('cp '+elem_file+' bigbatch')

        time.sleep(1)

# noise_map_dir=Path("/Users/tparsota/Documents/PATTERN_MAPS/")
# batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids, patt_noise_dir=noise_map_dir, nprocs=30)

def clean_events_BAT(source_dir):

    start_dir=os.getcwd()

    os.chdir(source_dir)

    logfile_name='./clean_events_BAT.log'

    if os.path.isfile(logfile_name):
        os.system('rm '+logfile_name)

    with StdoutTee(logfile_name,
                   mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(logfile_name,buff=1,file_filters=[_remove_control_chars]):

        print('Cleaning events in directory '+source_dir)

        # removing all of the heavy folders created during the DR and SA
        for elem_dir in glob.glob('**/'):
            print(elem_dir)
            if 'surveyresult' in elem_dir:
                print('cleaning directory ' + elem_dir)
                if 'mosaiced' in elem_dir:

                    # removing the total mosaic which we're not using
                    os.system('rm -rf ' + os.path.join(elem_dir, 'total_mosaic'))

                    # letting some time to clean the directory
                    time.sleep(5)

                    # removing the heavy files in the remaining mosaic dir
                    mosaic_dir = [elem for elem in glob.glob('mosaiced_surveyresults/**') if
                                  'mosaic' in elem.split('/')[-1]]

                    # removing only the files to let the PHA directory in peace
                    os.system('rm ' + os.path.join(mosaic_dir[0]) + '/*')

                    # letting some time to clean the directory
                    time.sleep(5)

                else:
                    os.system('rm -rf ' + elem_dir)

                    # letting some time to clean the directory
                    time.sleep(1)

    os.chdir(start_dir)

def inter_to_dir(date_start,date_stop):
    time_date_start=Time(date_start)
    time_date_stop=Time(date_stop)

    #testing whether the interval is 1 day
    if (time_date_stop-time_date_start).jd==1.0 and time_date_start.mjd==int(time_date_start.mjd) and time_date_stop.mjd==int(time_date_stop.mjd) :
        daily_inter=True
    else:
        daily_inter=False

    cycle_dir=date_start if daily_inter else '-'.join([date_start,date_stop])

    return cycle_dir

def integ_cycle_BAT(object_name,date_start,date_stop,minexposure=1000,noise_map_dir='environ',ul_pl_index=2.5,recalc=False,merge=True,custom_cat_coords=None,nprocs=1,uksdc=False,preclean_dir=False,):

    '''
    Performs a full data reduction download and cycle for a given object between date_start and date_stop

    only the start date is used if the date_stop-date_start interval is one day

    Also mosaics all of the images together for the given duration

    Most arguments are the basic arguments of the other functions

        -merge: merge the PHA files to a bigbatch at the end

        -custom_cat_coords: [ra,dec,l,b] in degrees array, or None
                            if not None, creates a new custom catalog for the current object name and passes it
                            to DR_BAT
                            (useful to analyze sources which aren't in the list of sources of the standard surveys)
    '''

    plt.ioff()

    init_dir=os.getcwd()

    cycle_dir=inter_to_dir(date_start,date_stop)

    if preclean_dir:
        os.system('rm -rf '+cycle_dir)
    os.system('mkdir -p '+cycle_dir)

    os.chdir(cycle_dir)

    ba.datadir(os.getcwd())

    logfile_name='./integ_cycle_BAT.log'

    if os.path.isfile(logfile_name):
        os.system('rm '+logfile_name)

    with StdoutTee(logfile_name,
                   mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(logfile_name,buff=1,file_filters=[_remove_control_chars]):

        fetched_pointings=fetch_BAT(object_name,date_start,date_stop,minexposure,return_result=True,uksdc=uksdc)

        if len(fetched_pointings)==0:
            os.chdir(init_dir)
            return 'No valid pointing downloaded'

        if custom_cat_coords is not None:

            #removing a potential previous catalog to avoid issues
            if os.path.isfile('./custom_catalog.cat'):
                os.system('rm custom_catalog.cat')

            custom_cat_posix = ba.create_custom_catalog(object_name,
                                                            custom_cat_coords[0],custom_cat_coords[1],custom_cat_coords[2],custom_cat_coords[3])

            if custom_cat_posix is None:
                os.chdir(init_dir)
                return 'Couldnt create custom catalog file'
            else:
                custom_cat_path=str(custom_cat_posix)
        else:
            custom_cat_path=None

        dr_pointings=DR_BAT(noise_map_dir=noise_map_dir,custom_cat_path=custom_cat_path,nprocs=nprocs)

        if len(dr_pointings)==0:
            os.chdir(init_dir)
            print('No valid pointings after data reduction')
            return 'No valid pointings after data reduction'

        outventory_file=ba.merge_outventory(dr_pointings)

        '''
        from batanalysis docs:
        the start_datetime value is automatically set to be the first BAT survey observation rounded to the nearest hole timedelta value (ie the floor function applied to the earliest BAT survey date to the start of that month in this case).
        '''

        #this one needs to be in daetime64
        time_delta=np.datetime64(date_stop)-np.datetime64(date_start)

        time_bins = ba.group_outventory(outventory_file, binning_timedelta=time_delta,
                                        start_datetime=Time(date_start),end_datetime=Time(date_stop))

        #procs=1 to avoid issue with parallelization over it
        #note that here we should get a single mosaic
        mosaic_list = ba.parallel.batmosaic_analysis(dr_pointings, outventory_file, time_bins,nprocs=nprocs,
                                                                   compute_total_mosaic=False,
                                                     catalog_file=custom_cat_path)

        if len(mosaic_list)==0:
            os.chdir(init_dir)
            print('No valid pointings after mosaic creation')
            return 'No valid pointings after mosaic creation'

        if len(mosaic_list)!=1:
            os.chdir(init_dir)
            print('Alert: more than one mosaic created. Check mosaic process')
            return "Alert: more than one mosaic created. Check mosaic process"

        #procs=1 to avoid issue with parallelization over it
        mosaic_list=ba.parallel.batspectrum_analysis(mosaic_list, object_name, ul_pl_index=ul_pl_index,
                                                     recalc=False,nprocs=nprocs)

        if len(mosaic_list)==0:
            os.chdir(init_dir)
            print('No valid mosaic after mosaic spectrum analysis')

            return 'No valid mosaic after mosaic spectrum analysis'

        pha_outputs=mosaic_list[0].pha_file_names_list

        if len(pha_outputs)==0:
            os.chdir(init_dir)
            print('No valid spectra after mosaic spectrum analysis')

            return 'No valid spectra after mosaic spectrum analysis'

        #copying spectral products to the bigbatch directory

        try:
            pha_dir=str(mosaic_list[0].pha_file_names_list[0])
        except:
            breakpoint()

        pha_dir=pha_dir[:pha_dir.rfind('/')]

        #removing the previously renamed spectral elements
        for elem in glob.glob(pha_dir+'/**'):
            if cycle_dir in elem.split('/')[-1]:
                os.system('rm '+elem)
        sp_products=[elem for elem in glob.glob(pha_dir+'/**') if elem.endswith('.pha') or elem.endswith('.rsp')]

        #testing if there was an non-ul file created
        if sum([elem.endswith('mosaic.pha') for elem in sp_products]):
            only_ul=False
        else:
            only_ul=True

        #renaming spectral files
        for elem in sp_products:

            if 'upperlim' in elem and not only_ul:
                    continue

            #renaming the file:
            if elem.endswith('upperlim.pha'):

                new_name=pha_dir+'/'+'BAT_'+cycle_dir+'_upperlim.pha'
                os.system('cp '+elem+' '+new_name)

                with fits.open(new_name,mode='update') as hdul:
                    hdul[1].header['RESPFILE']='BAT_'+cycle_dir+'.rmf'
                    hdul.flush()

            if elem.endswith('mosaic.pha'):

                new_name=pha_dir+'/'+'BAT_'+cycle_dir+'_mosaic.pha'
                os.system('cp '+elem+' '+new_name)

                with fits.open(new_name,mode='update') as hdul:
                    hdul[1].header['RESPFILE']='BAT_'+cycle_dir+'.rmf'
                    hdul.flush()

            if elem.endswith('.rsp'):
                os.system('cp '+elem+' '+pha_dir+'/'+'BAT_'+cycle_dir+'.rmf')

        if merge:

            os.system('mkdir -p ../bigbatch')

            #and copying them to the bigbatch directory
            sp_products_renamed=[elem for elem in glob.glob(pha_dir+'/**') if cycle_dir in elem.split('/')[-1]]

            for elem_renamed in sp_products_renamed:
                os.system('cp '+elem_renamed+' '+os.path.join(init_dir,'bigbatch'))


    os.chdir(init_dir)
    plt.ion()

    return 'Done'

def summary_state(summary_file):

    # fetching the previously computed directories from the summary folder file
    if not os.path.isfile(summary_file):
        return [],[]

    try:
        with open(summary_file) as summary_folder:
            launched_intervals = summary_folder.readlines()

            # restricting to intervals with completed analysis
            completed_intervals = [elem.split('\t')[0] for elem in launched_intervals if 'Done' in elem]
            launched_intervals = [elem.split('\t')[0] for elem in launched_intervals]
    except:
        launched_intervals = []
        completed_intervals = []

    return launched_intervals,completed_intervals

def full_cycle_BAT(object_name,increment_start,increment_stop,minexposure,noise_map_dir,ul_pl_index,recalc,
                   merge,custom_cat_coords,clean_events_dir=None,preclean_dir=False,
                   summary_file=None,header_name=None,summary_intervals_header=None,launched_intervals=None,nprocs=1,
                   uksdc=False):

    '''
    this function is mostly here to be parallelized

    clean_events_dir can be None or the directory to clean
    same for summary file

    launched_intervals is to test whether to skip if return_intervals is set to False

    WARNING: one of the main issues are too long directory names. Be careful about that

    '''

    if launched_intervals is not None and header_name in launched_intervals:
        print('interval ' + header_name + ' already computed. Skipping...')
        return

    interval_state = integ_cycle_BAT(object_name, increment_start, increment_stop, minexposure=minexposure,
                                     noise_map_dir=noise_map_dir, ul_pl_index=ul_pl_index, recalc=recalc, merge=merge,
                                     custom_cat_coords=custom_cat_coords,nprocs=nprocs,preclean_dir=preclean_dir,
                                     uksdc=uksdc)

    # cleaning the events if required
    if clean_events_dir is not None:
        clean_events_BAT(clean_events_dir)

    if summary_file is not None:
        # adding the directory to the list of already computed directories
        file_edit(summary_file, header_name, header_name + '\t' + interval_state + '\n',
                  summary_intervals_header)

def loop_cycle_BAT(object_name,input_days_file=None,interval_start=None,interval_stop=None,interval_delta='1',
                   interval_delta_unit='jd',minexposure=1000,noise_map_dir='environ',ul_pl_index=2.5,recalc=False,
                   merge=True,clean_events=True,preclean_dir=False,
                   rerun_started=True,rerun_completed=False,use_custom_cat=True,parallel=1,nprocs=1,uksdc=False):
    '''
    Bigger wrapper around integ_cycle_BAT

    loops differents integ_cycle_BAT by incrementing the intervals between interval_start and interval_stop
    with interval_delta astropy TimeDelta objects (of format interval_delta_unit)

    for now restricted to day dates

    if input_days_file is provided, use the list of days there instead of a full interval

    object_name:
        -a string which will be searched in Simbad,
        -'auto' -> fetches the closest directory in the current arborescence with a name in Simbad

    return_intervals:
        rerun or not intervals already logged in the summary_interval file

    use_custom_cat:
        passes source coordinates to integ_cycle_BAT, to create a custom catalog with the source position, which
        allows analysis of any source (and not just the ones pre-existing in the BAT surveys)

    parallel:
        parallelisation of the full_cycle_BAT function (currently doesn't work)
    nprocs:
        parallelization inside the integ_cycle_BAT function

    uksdc: force uk download to avoid issues with older obs

    WARNING: one of the main issues are too long directory names. Be careful about that

    '''


    if object_name!='auto':
        object_name_use=object_name
        object_simbad=Simbad.query_object(object_name)[0]
    else:
        object_simbad=source_catal(object_name)
        object_name_use=object_simbad['main_id'].replace(' ','')

    object_sky = SkyCoord(ra=object_simbad.columns['RA'], dec=object_simbad.columns['DEC'],
                          unit=(u.hourangle, u.deg))

    object_coords=object_sky.ra.value[0],object_sky.dec.value[0],object_sky.galactic.l.value[0],object_sky.galactic.b.value[0]

    # summary header for the previously computed directories file
    summary_intervals_header = 'Interval\tAnalysis state\n'

    assert input_days_file is not None or (interval_start is not None and interval_stop is not None),\
            'Error: lacking input date argument'

    if input_days_file is not None:

        with open(input_days_file) as f:
            day_lines = f.readlines()

        time_date_start=Time([elem.replace('\n', '') for elem in day_lines])
        time_date_stop=time_date_start+TimeDelta(1,format='jd')

        summary_file='summary_interval_analysis_' + object_name+\
                      '_'+'.'.join(input_days_file.split('.')[:-1])+'.log'

        launched_intervals,completed_intervals=summary_state(summary_file)

        increment_start_list=[elem.split(' ')[0] for elem in time_date_start.iso]
        increment_stop_list=[elem.split(' ')[0] for elem in time_date_stop.iso]

    else:

        time_date_start=Time(interval_start)
        time_date_stop=Time(interval_stop)

        delta=TimeDelta(interval_delta,format=interval_delta_unit)
        dates_increments = []

        summary_file='summary_interval_analysis_' + object_name+\
                      '_'+interval_start+'_'+interval_stop+'_'+interval_delta+'_'+interval_delta_unit+\
                      '.log'

        launched_intervals,completed_intervals=summary_state(summary_file)

        for i in np.arange(((time_date_stop - time_date_start) / delta)+1):
            dates_increments += [(time_date_start + i * delta).iso.split(' ')[0]]

        increment_start_list=dates_increments[:-1]
        increment_stop_list=dates_increments[1:]

    inter_dir_list=[inter_to_dir(elem_start,elem_stop) for elem_start,elem_stop in \
                    zip(increment_start_list,increment_stop_list)]

    header_name_list=['_'.join([elem_start,elem_stop]) for elem_start,elem_stop in \
                      zip(increment_start_list,increment_stop_list)]

    if parallel==1:

        for i_increment in range(len(increment_start_list)):

            increment_start=increment_start_list[i_increment]
            increment_stop=increment_stop_list[i_increment]

            header_name='_'.join([increment_start,increment_stop])

            inter_dir=inter_to_dir(increment_start,increment_stop)

            if not rerun_started and header_name in launched_intervals:
                print('interval '+header_name+' already started. Skipping...')
                continue

            if not rerun_completed and header_name in completed_intervals:
                print('interval '+header_name+' already completed. Skipping...')
                continue

            interval_state=integ_cycle_BAT(object_name,increment_start,increment_stop,minexposure=minexposure,noise_map_dir=noise_map_dir,ul_pl_index=ul_pl_index,recalc=recalc,merge=merge,
              custom_cat_coords=object_coords if use_custom_cat else None,nprocs=nprocs,uksdc=uksdc,
                                           preclean_dir=preclean_dir)


            #cleaning the events if required
            if clean_events:
                clean_events_BAT(inter_dir)

            # adding the directory to the list of already computed directories
            file_edit(summary_file, header_name, header_name + '\t' + interval_state + '\n',
                      summary_intervals_header)

    else:

        mask_use=[(elem not in launched_intervals if not rerun_started else 1) and \
                  (elem not in completed_intervals if not rerun_completed else 1)
                  for elem in header_name_list]

        res = Parallel(n_jobs=parallel)(
            delayed(full_cycle_BAT)(
                object_name=object_name,
                increment_start=elem_start,
                increment_stop=elem_stop,
                minexposure=minexposure,
                noise_map_dir=noise_map_dir,
                ul_pl_index=ul_pl_index,
                recalc=recalc,
                merge=merge,
                custom_cat_coords=object_coords if use_custom_cat else None,
                clean_events_dir=elem_inter_dir,
                summary_file=summary_file, header_name=elem_header_name,
                summary_intervals_header=summary_intervals_header,
                launched_intervals=launched_intervals if not rerun_started  else None,
                preclean_dir=preclean_dir,
                nprocs=1,
                uksdc=uksdc)
            for elem_start,elem_stop,elem_inter_dir,elem_header_name \
                    in zip(np.array(increment_start_list)[mask_use],np.array(increment_stop_list)[mask_use],
                           np.array(inter_dir_list)[mask_use],np.array(header_name_list)[mask_use]))


def DR_XRT_QL():
    '''
    Step1: xrtpipeline on the output of the ql (or normal obs) https://www.swift.ac.uk/analysis/xrt/xrtpipeline.php
    Step2: make region files
            -careful about pile-up in PC mode above 0.5 cts/s
            -in WT, take a 40pixel wide region around the source and a background far. Can be annoying with dead columns.

    Step3: extract SP with xselect
    (Step4:  extract expomap with xrtexpomap. CAN BE BYPASSED BY USING EXPOMAP FROM XRTPIPELINE)
            Need th event list, the pat.fits file, and the xhd.hk file
     If MJD-OBS is missing (e.g. from a uf file if no cl files are made in step 1),
            add it in the header of the hdul[1]

    Step5: extract ARF with xrtmkarf.
            -take manual rmfs from caldb (see below)
            -source pointing at -1 uses the coordinates of the source region
                for WT, use the real position of the source from simbad instead in degrees
                for PC, use the center of the circular region
            (If RMF not found in caldb, fetch the appropriate RMF manually in caldb
            (using reading the most up-to-date document in https://www.swift.ac.uk/analysis/Gain_RMF_releases.html
            and input it manually in the command with the rmffile argument)
    '''
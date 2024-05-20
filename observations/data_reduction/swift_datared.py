#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:32:37 2022

@author: parrama
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
import swiftbat
import swiftbat.swutil as sbu
import pickle


def merge_swift_spectra():

    '''
    moves all swift spectra to a merge "bigbatch" directory
    '''
    
    allfiles=glob.glob('**',recursive=True)
    specfiles=[elem for elem in allfiles if elem.endswith('source.pi') or elem.endswith('back.pi')\
               or elem.endswith('.rmf') or elem.endswith('.arf')]
    
    os.system('mkdir -p bigbatch')
    currdir=os.getcwd()
    
    for elemfile in specfiles:
        shutil.copy(elemfile,os.path.join(currdir,'bigbatch','sw'+elemfile.split('/')[-1].replace('Obs_','')))
        
    
def regroup_swift_spectra(extension='source.pi',group='opt',skip_started=True):
    
    '''To be launched above all spectra to regroup'''
    
    #spawning heasoft terminal for Kastra grouping
    heas_proc=pexpect.spawn('/bin/bash',encoding='utf-8')
    heas_proc.logfile=sys.stdout
    heas_proc.sendline('\nheainit')
    
    def ft_group(file,grptype):
        
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
    
    heas_proc.sendline('cd '+os.path.join(currdir,'bigbatch'))
    
    # if skip_started:
    #     pha2_spectra=[elem for elem in pha2_spectra if\
    #                 '/'.join(elem.split('/')[:-1])+('' if len(elem.split('/'))==1 else '/')+elem.split('/')[-1].split('_')[0]+'_heg_-1_grp_'+group+'.pha' not in allfiles or\
    #                 '/'.join(elem.split('/')[:-1])+('' if len(elem.split('/'))==1 else '/')+elem.split('/')[-1].split('_')[0]+'_heg_1_grp_'+group+'.pha' not in allfiles]
        
    for ind,specpath in enumerate(speclist):

        if skip_started and os.path.isfile(specpath.replace('.','_grp_'+group+'.')):
            print('\nSpectrum '+specpath+' already grouped')            
            continue
        
        specfile=specpath.split('/')[-1]

        #stat grouping
        if group is not None:
            
            if group=='opt':

                ft_group(specfile,grptype='opt')
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

def fetch_BAT(date_start='2021-09-20',date_stop='2021-09-21',object_name = '4U 1630-47',minexposure=1000):

    '''
    wrapper around batanalysis to download some data

    see https://github.com/parsotat/BatAnalysis for installation

    -minexposure   # cm^2 after cos adjust

    may need to add uksdc=True to download_swiftdata for old datasets
    '''


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

    result = ba.download_swiftdata(table_exposed)

def DR_BAT(noise_map_dir='/home/parrama/Soft/Swift-BAT/pattern_maps/',nprocs=2):

    '''
    wrapper around batanalysis to reduce data in the current folder

    requires UNTARED noise maps in noise_map_dir

    see https://github.com/parsotat/BatAnalysis for installation

    Note: if nothing gets out and no gti are recognized, it could be due to a lack of caldb initalization
    '''


    obs_ids = [i.name for i in sorted(ba.datadir().glob("*")) if i.name.isnumeric()]
    input_dict=dict(cleansnr=6,cleanexpr='ALWAYS_CLEAN==T')
    # input_dict=None
    batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids,input_dict=input_dict,
                                                 patt_noise_dir=noise_map_dir, nprocs=nprocs)

    return batsurvey_obs

def SA_BAT(survey_obs_list,object_name,ul_pl_index=2.5,recalc=True,nprocs=2):

    '''
    Wrapper around batanalysis to perform spectral analysis of data in the current folder

    split from DR_BAT because each step can take very long

    survey_obs_list should be the output of DR_BAT

    '''

    sa_obs_list=ba.parallel.batspectrum_analysis(survey_obs_list, object_name, ul_pl_index=ul_pl_index,
                                                 recalc=recalc,nprocs=nprocs)

    return sa_obs_list


# noise_map_dir=Path("/Users/tparsota/Documents/PATTERN_MAPS/")
# batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids, patt_noise_dir=noise_map_dir, nprocs=30)

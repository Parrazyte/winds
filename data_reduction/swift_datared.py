#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:32:37 2022

@author: parrama
"""

import argparse
import numpy as np
import pandas as pd
import os
import glob
import pexpect
import sys
import time

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
        os.rename(elemfile,os.path.join(currdir,'bigbatch',elemfile.split('/')[-1].replace('Obs_','')))
        
    
def regroup_swift_spectra(extension='source.pi',group='opt',skip_started=True):
    
    '''To be launched above all spectra to regroup'''
    
    #spawning heasoft spectra for Kastra grouping
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
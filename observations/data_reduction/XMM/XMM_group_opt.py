#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:42:43 2022

@author: parrama
"""

import pexpect
import sys
import glob
import os
import time
import numpy as np

def regroup_XMM_spectra(extension='sp_src.ds',group='opt',camera='mos1',skip_started=True):
    
    '''To be launched above the folders where to regroup'''
            
    def ft_group(file,grptype):
        
        '''wrapper for the command'''
        
        if 'min' in grptype:
            group_pref=grptype.split('_')[0]
            group_val=grptype.split('_')[1]
        else:
            group_pref=grptype
            group_val=''
        
        if group_val!='':
            minval_str=' groupscale='+group_val
        else:
            minval_str=''
        
        heas_proc.sendline('ftgrouppha infile='+file+' outfile='+file.replace('.','_grp_'+grptype+'.')+' grouptype='+group_pref+minval_str+
                           ' respfile='+file.replace(file[file.rfind('.'):],'.rmf').replace('_sp_src','')+' clobber=yes')
        
        heas_proc.sendline('echo done')
        heas_proc.expect('done')
        
    allfiles=glob.glob('**',recursive=True)
    
    XMM_spectra=[elem for elem in allfiles if elem.endswith(extension) and 'bigbatch' in elem and ('_'+camera+'_' in elem\
                 if camera!='all' else True)]
    
    XMM_dirs=os.getcwd()+'/'+np.array([elem[:elem.rfind('/')+1] for elem in XMM_spectra]).astype(object)

    for ind,specpath in enumerate(XMM_spectra):
        
        if os.path.isfile(specpath.replace('.ds','_grp_'+group+'.ds')) and skip_started:
            print(specpath+' already grouped. Skipping...\n')
            continue
        
        if 'CenX-2' in specpath:
            #skipping this folder
            continue
        
        #spawning heasoft spectra for Kastra grouping
        heas_proc=pexpect.spawn('/bin/bash',encoding='utf-8')
        heas_proc.logfile=sys.stdout
        heas_proc.sendline('\nheainit')
        
        specfile=specpath.split('/')[-1]
        #sending the ciao process in the right directory
        
        #stat grouping

        heas_proc.sendline('cd '+XMM_dirs[ind])
        
        if group is not None:

            ft_group(specfile,grptype=group)
            
            condition=os.path.isfile(specpath.replace('.ds','_grp_'+group+'.ds'))
            
            while not condition:
                
                time.sleep(1)
                
                condition=os.path.isfile(specpath.replace('.ds','_grp_'+group+'.ds'))
                    
        
        heas_proc.sendline('exit')

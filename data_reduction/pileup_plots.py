#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:46:39 2023

@author: parrama
"""

import glob

import os

from astropy.io import fits

import matplotlib.pyplot as plt

import numpy as np


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

#plotting the distribution of the pile-up values
#to launch in XXM/Sample


lineplots_recap=glob.glob('**/bigbatch/lineplots_opt/**_recap.pdf',recursive=True)

lineplots_recap=[elem.split('/')[-1] for elem in lineplots_recap]

spectra=glob.glob('**/bigbatch/**_sp_src.ds',recursive=True)

spectra=[elem for elem in spectra if '_pn' in elem and elem.split('/')[-1].replace('_sp_src.ds','_recap.pdf') in lineplots_recap]
                 
pileup_lines=[]

pileup_values=[]

pileup_values_cleaned=[]


for elem in spectra:
    try:
        with fits.open(elem) as hdul:
            pileup_lines+=[hdul[0].header['PILE-UP'].split(',')[-1]]
    except:
        breakpoint()
        print("tchou")
        
pileup_values=np.array([pileup_val(line) for line in pileup_lines])

plt.hist(pileup_values,bins=np.arange(0.,0.11,0.01),rwidth=0.8)

plt.xlabel('pileup %')

plt.ylabel('number of spectra')

spectra_above_005=np.array(spectra)[pileup_values>0.05]

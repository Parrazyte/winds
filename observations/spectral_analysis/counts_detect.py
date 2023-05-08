#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:55:19 2023

@author: parrama
"""

import os
import numpy as np
from tqdm import tqdm
from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,AllChains,Chain

#custom script with a few shorter xspec commands
from xspec_config_multisp import allmodel_data,model_data,model_load,addcomp,editmod,Pset,Pnull,rescale,reset,Plot_screen,store_plot,freeze,allfreeze,unfreeze,\
                         calc_error,delcomp,fitmod,fitcomp,calc_fit,plot_line_comps,\
                         xcolors_grp,comb_chi2map,plot_std_ener,coltour_chi2map,xPlot

#custom script with a some lines and fit utilities and variables
from fitting_tools import c_light,lines_std_names,lines_e_dict,ravel_ragged,n_absline,range_absline,model_list

import argparse

ap = argparse.ArgumentParser(description='Script to detect lines in XMM Spectra.\n)')
ap.add_argument("-line_cont_range",nargs=1,help='min and max energies of the line continuum broand band fit',default='4 10',type=str)
ap.add_argument('-instru',nargs=1,help='telescope to use for the test',default='Chandra',type=str)
ap.add_argument('-counts_min',nargs=1,help='minimum source counts in the source region in the line continuum range',default=5000,type=float)
ap.add_argument('-nfakes',nargs=1,help='number of simulations used. Limits the maximal significance tested to >1-1/nfakes',default=1e3,type=int)
args=ap.parse_args()

line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
instru=args.instru
counts_min=args.counts_min
nfakes=int(args.nfakes)

Xset.chatter=0

if instru=='XMM':
    #XMM test
    os.chdir('/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/XMM/Sample/GROJ1655-40/bigbatch')    
    Xset.restore('lineplots_opt/0155762501_pn_S001_Timing_auto_mod_autofit.xcm')
elif instru=='Chandra':
    #XMM test
    os.chdir('/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/Chandra/Sample/GROJ1655-40/bigbatch')    
    Xset.restore('lineplots_opt/5460_heg_-1_mod_autofit.xcm')
    
#deleting all lines
while 'gaussian' in AllModels(1).expression:
    delcomp('gaussian')
    

model_init=allmodel_data()

#freezing the model
allfreeze()

#computing the exposure needed to get 5000 counts in the band with this spectrum
exp_conv=sum([(np.array(AllData(i).rate)*AllData(i).exposure)[3] for i in range(1,AllData.nGroups+1)])/counts_min

exp_counts_min=AllData(1).exposure/exp_conv

fakeset=[FakeitSettings(response=AllData(i_grp).response.rmf,arf=AllData(i_grp).response.arf,background='',exposure=exp_counts_min,
                          fileName=AllData(i_grp).fileName) for i_grp in range(1,AllData.nGroups+1)]

EW_arr=np.zeros(nfakes)

print('Starting process')

#fake loop
with tqdm(total=nfakes) as pbar:
    for f_ind in range(nfakes):
        
        #reloading the high energy continuum
        mod_fake=model_load(model_init)
                    
        #replacing the current spectra with a fake with the same characteristics so this can be looped
        #applyStats is set to true but shouldn't matter for now since everything is frozen

        AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)

            
        #ensuring we are in the correct energy range
        AllData.notice('all')
        AllData.ignore('bad')
        AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')

        #adjusting the fit and storing the chi² 
        
        unfreeze()
        Fit.perform()
        
        '''
        Adding a FeKa26abs_line as a test
        '''
        
        #adding a narrow gaussian
        mod_fake=addcomp('FeKa26abs_'+('' if instru=='Chandra' else 'n')+'agaussian')
        
        #Fitting
        Fit.perform()
                
        gaussian_compnumber=int(np.argwhere(np.array(AllModels(1).componentNames)=='gaussian')[0][0]+1)
        
        #storing the Ew of the line
        AllModels.eqwidth(gaussian_compnumber)
        
        EW_arr[f_ind]=AllData(1).eqwidth[0]

        pbar.update(1)
        
sorted_EW_arr=abs(EW_arr)*1e3
sorted_EW_arr.sort()

if instru=='XMM':
    save_path='/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/XMM/'
elif instru=='Chandra':
    save_path='/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/Chandra/'
    
np.save(save_path+instru+'_EW_distrib_'+str(counts_min)+'.npy',sorted_EW_arr)

print('process_finished')
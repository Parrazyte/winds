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
                         calc_error,delcomp,calc_fit

#custom script with a some lines and fit utilities and variables
from fitting_tools import c_light,lines_std_names,lines_e_dict,n_absline,range_absline,model_list

import argparse

ap = argparse.ArgumentParser(description='Script to detect lines in  Spectra.\n)')
ap.add_argument("-line_cont_range",nargs=1,help='min and max energies of the line continuum broand band fit',default='4 10',type=str)
ap.add_argument('-instru',nargs=1,help='telescope to use for the test',default='NICER',type=str)
ap.add_argument('-counts_min',nargs=1,help='minimum source counts in the source region in the line continuum range',default=5000,type=float)
ap.add_argument('-nfakes',nargs=1,help='number of simulations used. Limits the maximal significance tested to >1-1/nfakes',default=1e2,type=int)
ap.add_argument('-test_SED',nargs=1,help='test effect of standard SED instead',default=True)

args=ap.parse_args()

line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
instru=args.instru
counts_min=args.counts_min
nfakes=int(args.nfakes)
test_SED=args.test_SED

Xset.chatter=0

if instru=='XMM':
    #XMM test
    os.chdir('/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/XMM/Sample/GROJ1655-40/bigbatch')    
    Xset.restore('lineplots_opt/0155762501_pn_S001_Timing_auto_mod_autofit.xcm')
elif instru=='Chandra':
    #XMM test
    os.chdir('/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/Chandra/Sample/GROJ1655-40/bigbatch')    
    Xset.restore('lineplots_opt/5460_heg_-1_mod_autofit.xcm')
elif instru=='NICER':
    #Chandra test
    os.chdir('/media/parrama/SSD/Observ/BHLMXB/NICER/Sample/4U1630-47/bigbatch')
    Xset.restore('lineplots_opt/1130010104_mod_broadband.xcm')


#deleting all lines
while 'gaussian' in AllModels(1).expression:
    delcomp('gaussian')
    

model_init=allmodel_data()

#freezing the model
allfreeze()

if test_SED:
    AllModels(1).diskbb.Tin.values=1.5
    AllModels(1).phabs.nH.values=5

    #this is the 2-10 flux for this SED and a diskbb normalization of 1
    conv_factor=4.863313283917692e-11

#computing the exposure needed to get 5000 counts in the band with this spectrum
if instru=='NICER':
    exp_counts=5000
else:
    exp_conv=sum([(np.array(AllData(i).rate)*AllData(i).exposure)[3] for i in range(1,AllData.nGroups+1)])/counts_min
    exp_counts=AllData(1).exposure/exp_conv


fakeset=[FakeitSettings(response=AllData(i_grp).response.rmf,arf=AllData(i_grp).response.arf,background='',
                        exposure=exp_counts,
                          fileName=AllData(i_grp).fileName) for i_grp in range(1,AllData.nGroups+1)]



print('Starting process')

val_norm=[None] if not test_SED else 1e-10/conv_factor*np.logspace(np.log10(5),np.log10(500),20)

EW_arr=np.zeros((len(val_norm),nfakes))

# fake loop
with tqdm(total=nfakes*len(val_norm)) as pbar:

    for i_norm,norm in enumerate(val_norm):

        if norm is not None:
            AllModels(1).diskbb.norm.values=norm

            model_init=allmodel_data()

        for f_ind in range(nfakes):

            #reloading the high energy continuum
            mod_fake=model_load(model_init)

            #replacing the current spectra with a fake with the same characteristics so this can be looped
            #applyStats is set to true but shouldn't matter for now since everything is frozen

            AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)


            #ensuring we are in the correct energy range
            AllData.notice('all')
            AllData.ignore('bad')
            if test_SED:
                AllData.ignore('**-2. 10.-**')
            else:
                AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')

            #adjusting the fit
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

            EW_arr[i_norm][f_ind]=AllData(1).eqwidth[0]

            pbar.update(1)

sorted_EW_arr=abs(EW_arr)*1e3

breakpoint()

sorted_EW_arr.sort()

if instru=='XMM':
    save_path='/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/XMM/'
elif instru=='Chandra':
    save_path='/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Observ/BHLMXB/Chandra/'
    
np.save(save_path+instru+'_EW_distrib_'+str(counts_min)+'.npy',sorted_EW_arr)

print('process_finished')
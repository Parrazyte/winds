#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:24:12 2023

@author: parrama
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from xspec import Xset,AllModels,Fit,Plot,AllData
from xspec_config_multisp import xPlot,reset,plot_std_ener,Pset,reset
from matplotlib.gridspec import GridSpec

reset()

mode='slabs'
#or 'slabs'

files=['lineplots_opt/22213_heg_-1_mod_autofit.xcm',
       'lineplots_opt/23435_heg_-1_mod_autofit.xcm',
       'lineplots_opt/24663_heg_-1_mod_autofit.xcm']

obsids=['22213','23435','24663']

grsdir='/media/parrama/SSD/Observ/BHLMXB/Chandra/Sample/GRS1915+105/bigbatch'

simdir='/media/parrama/SSD/Simu'

savedir='/home/parrama/Documents/Work/PhD/docs/papers/Wind review/'

fig=plt.figure(figsize=(15,10))

grid=GridSpec(2,3,figure=fig,hspace=0.,wspace=0.)

Plot.xLog=False

for i_col in range(3):
    
    ax_high=plt.subplot(grid[0,i_col])
    
    ax_low=plt.subplot(grid[1,i_col])
    
    #loading the files

    os.chdir(grsdir)
    
    Xset.restore(files[i_col])
        
    Fit.show()
    
    #rescaling the plot
    AllData.ignore('**-6. 7.5-**')
    
    #taking off the effective area in the counts plot
    Plot.area=True
    
    #saving the model of the first datagroup only
    Plot('ldata')
    
    mod_gauss=Plot.model()
    
    #resetting the model to avoid having 2 datagroup models
    AllModels.clear()
    
    ax_high.plot(Plot.x(1),mod_gauss,color='brown',label='gaussian model')

    #plotting in the ax
    xPlot('ldata',axes_input=ax_high,group_names=[obsids[i_col]+'_heg_-1',obsids[i_col]+'_heg_1'],
          hide_ticks=False,secondary_x=False,legend_position='lower left')
    
    plot_std_ener(ax_high,mode='')
    
    reset()

    os.chdir(grsdir)
    
    Xset.restore(files[i_col])
    
    
    AllModels.clear()
    os.chdir(simdir)
    
    AllModels+=('constant*phabs(mtable{4u1630_ezdisknthcomp.fits}*mtable{BH_Wind_lowxi.fits}*diskbb)')
    # AllModels+=('constant*phabs(gsmooth(mtable{4u1630_ezdisknthcomp.fits}*mtable{BH_Wind_lowxi.fits}*diskbb))')
    
    #setting and freezing the galactic nH
    AllModels(1)(2).values=5.3
    AllModels(1)(2).frozen=True
    
    #setting up the constant factors
    AllModels(2)(1).link=''
    AllModels(1)(1).frozen=True
    # AllModels(1)(1).values=1

    
    AllModels(1)(3).values=0.01
    # AllModels(1)(3).frozen=True
    
    Fit.perform()
    
    #getting errors on everything
    Fit.error('1-15')
    
    #saving the values of the full model
    valmod=[AllModels(1)(i).values[0] for i in range(1,11)]
    
    if os.path.isfile('slabs'+str(i_col)+'.xcm'):
        os.remove('slabs'+str(i_col)+'.xcm')
        
    Xset.save('slabs'+str(i_col)+'.xcm',info='a')
        
    Fit.show()
    
    AllModels.clear()
    
    #rescaling the plot
    AllData.ignore('**-6. 7.5-**')
    
    #taking off the effective area in the counts plot
    Plot.area=True
    
    #plotting in the ax
    xPlot('ldata',axes_input=ax_low,hide_ticks=False,secondary_x=False,group_names='nolabel',
          legend_position='upper right')
    
    #re-adding the first ionization zone's model
    AllModels+=('constant*phabs(mtable{4u1630_ezdisknthcomp.fits}*diskbb)')
    
    #loading before the break
    for i_par in range(1,6):
        AllModels(1)(i_par).values=valmod[i_par-1]
    
    #and after the break
    for i_par in range(9,11):
        AllModels(1)(i_par-3).values=valmod[i_par-1]
        
    Plot('ldata')
    
    mod_zone_1=Plot.model(1)
    
    AllModels.clear()
    
    #re-adding the second zone's model
    AllModels+=('constant*phabs(mtable{BH_Wind_lowxi.fits}*diskbb)')
    
    #loading before the break
    for i_par in range(1,3):
        AllModels(1)(i_par).values=valmod[i_par-1]
    
    #and after the break
    for i_par in range(6,11):
        AllModels(1)(i_par-3).values=valmod[i_par-1]
        
    Plot('ldata')
    
    mod_zone_2=Plot.model(1)
    
    ax_low.plot(Plot.x(1),mod_zone_1,color='royalblue',label=r'high $\xi$ photoionization zone')
    
    ax_low.plot(Plot.x(1),mod_zone_2,color='orange',label=r'low $\xi$ photoionization zone')
    
    e_lines_low=[6.544,6.586,6.587,6.497,6.506,6.629]
    
    e_lines_high=[6.7,6.668,6.97]
    
    for elem_e in e_lines_low:
        ax_low.axvline(x=elem_e,ymin=0,ymax=1,color='orange',ls='--',lw=0.75)
        
    for elem_e in e_lines_high:
        ax_low.axvline(x=elem_e,ymin=0,ymax=1,color='royalblue',ls='--',lw=0.75)
    
    
    plt.legend()
    
    # #getting a common y scale on the bottom
    # if i_col==0:
    #     ylims=ax_low.get_ylim()
    # else:
    #     ax_low.set_ylim(ylims)
        
    #turning off the lower x axis for the high ax
    ax_high.xaxis.set_visible(False)
    new_x_axis=ax_low.secondary_xaxis('top')
    new_x_axis.minorticks_on()
    plt.setp(new_x_axis.get_xticklabels(), visible=False)
    
    #and turning off the y axis after the first column
    if i_col!=0:
        #note: for the first we only turn of the title and ticks
        ax_high.yaxis.set_visible(False)
        
        ax_low.yaxis.set_visible(False)
        
    ax_high.set_ylabel(r'norm. counts s$^{-1}$ keV$^{-1}$')
    ax_low.set_ylabel(r'norm. counts s$^{-1}$ keV$^{-1}$')
    AllData.notice('3.-10.')
        
plt.tight_layout()
        
os.chdir(savedir)

plt.savefig('grs_obs.pdf')

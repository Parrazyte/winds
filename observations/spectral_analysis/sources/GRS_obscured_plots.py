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
from xspec_config_multisp import xPlot, reset, Pset, reset, catch_model_str, Plot_screen, calc_error, plot_std_ener
from matplotlib.gridspec import GridSpec

reset()

mode='slabs'
#or 'slabs'

files=['lineplots_opt/22213_heg_-1_mod_autofit.xcm',
       'lineplots_opt/23435_heg_-1_mod_autofit.xcm',
       'lineplots_opt/24663_heg_-1_mod_autofit.xcm']

obsids=['22213','23435','24663']

grsdir='/media/parrama/SSD/Observ/BHLMXB/Chandra/Sample/GRS1915+105/bigbatch'

simdir='/media/parrama/SSD/Simu/GRS_slab'

savedir='/home/parrama/Documents/Work/PhD/docs/papers/Wind review/'

with_bshift=True

with_redshift=True

fig=plt.figure(figsize=(15,10))

grid=GridSpec(2,3,figure=fig,hspace=0.,wspace=0.)

Plot.xLog=False

os.chdir(savedir)

Xset.openLog('grs_logs.txt')
logfile_read=open('grs_logs.txt',mode='r')

                 
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
    
    AllData.ignore('**-4.5 8.-**')
    
    AllModels.clear()
    os.chdir(simdir)
    
    if with_bshift and with_redshift:
    
        AllModels+=('constant*phabs*mtable{BH_Wind_lowxi.fits}*mtable{4u1630_ezdisknthcomp.fits}*diskbb')
    else:
        AllModels+=('constant*phabs*mtable{BH_Wind_lowxi.fits}*vashift*mtable{4u1630_ezdisknthcomp.fits}*diskbb')
    # AllModels+=('constant*phabs(gsmooth(mtable{4u1630_ezdisknthcomp.fits}*mtable{BH_Wind_lowxi.fits}*diskbb))')
    
    #setting and freezing the galactic nH
    AllModels(1)(2).values=5.3
    AllModels(1)(2).frozen=True
    
    #setting up the constant factors
    AllModels(2)(1).link=''
    AllModels(1)(1).frozen=True
    # AllModels(1)(1).values=1

    if with_bshift:
        
        if with_redshift:
            AllModels(1)(8).frozen=False
            AllModels(1)(8).values=[0,0.001,-0.01,-0.01,0.003,0.003]
            
        # if with_redshift:
        #     AllModels(1)(5).frozen=False
        #     AllModels(1)(5).values=[0,0.001,-0.01,-0.01,0.01,0.01]
            
        else:
            #unfreezing the vashift of the high xi component
            AllModels(1).vashift.Velocity.frozen=False
    
    Fit.perform()
    
    #getting errors on everything
    calc_error(logfile_read,'1-11')
    
    #saving the values of the full model
    #note: here it goes until parameter 10 or 11 with the range not taking the last parameter
    valmod=[AllModels(1)(i).values for i in range(1,11+(1 if with_bshift and not with_redshift else 0))]
    
    if os.path.isfile('slabs'+str(i_col)+('_with_bshift' if with_bshift else '')+('_redshift' if with_redshift else '')+'.xcm'):
        os.remove('slabs'+str(i_col)+('_with_bshift' if with_bshift else '')+('_redshift' if with_redshift else '')+'.xcm')
        
    
    Xset.save('slabs'+str(i_col)+('_with_bshift' if with_bshift else '')+('_redshift' if with_redshift else '')+'.xcm',info='a')
        
    catch_model_str(logfile_read,savepath='slabs'+str(i_col)+('_with_bshift' if with_bshift else '')+('_redshift' if with_redshift else '')+'.txt')
    
    Plot_screen('ldata,ratio,delchi','slabs'+str(i_col)+('_with_bshift' if with_bshift else '')+('_redshift' if with_redshift else '')+'_plot.png')

    Fit.show()
    
    AllModels.clear()
    
    #rescaling the plot
    AllData.ignore('**-6. 7.5-**')
    
    #taking off the effective area in the counts plot
    Plot.area=True
    
    #plotting in the ax
    xPlot('ldata',axes_input=ax_low,hide_ticks=False,secondary_x=False,group_names='nolabel',
          legend_position='upper right')
    
    #re-adding the high xi ionization zone's model
    
    if not with_bshift or with_redshift:
        AllModels+=('constant*phabs*mtable{4u1630_ezdisknthcomp.fits}*diskbb')
              
    else:
        AllModels+=('constant*phabs*vashift*mtable{4u1630_ezdisknthcomp.fits}*diskbb')

    #skipping 3 of the low ionization component
    pars_cut=valmod[:2]+valmod[5:]
    
    #loading
    
    for i_par in range(1,len(pars_cut)+1):
        AllModels(1)(i_par).values=pars_cut[i_par-1]


    Plot('ldata')
    
    mod_zone_high=Plot.model(1)
    
    AllModels.clear()
    
    #re-adding the low xi ionization zone's model

    AllModels+=('constant*phabs*mtable{BH_Wind_lowxi.fits}*diskbb')
              
    #skipping the high ionization component
    pars_cut=valmod[:5]+valmod[-2:]
    
    #loading 
    for i_par in range(1,len(pars_cut)+1):
        AllModels(1)(i_par).values=pars_cut[i_par-1]
    
    Plot('ldata')
    
    mod_zone_low=Plot.model(1)
    
    ax_low.plot(Plot.x(1),mod_zone_high,color='royalblue',label=r'high $\xi$ photoionization zone')
    
    ax_low.plot(Plot.x(1),mod_zone_low,color='orange',label=r'low $\xi$ photoionization zone')
    
    e_lines_low=[[6.629,6.586,6.587,6.676,6.662,6.544,6.7],
                 [6.676,6.662,6.629],
                 [6.676,6.662,6.629,6.586,6.587,6.7,6.544]]
    
    e_lines_high=[6.7,6.668,6.97]
    
    for elem_e in e_lines_low[i_col]:
        ax_low.axvline(x=elem_e,ymin=0,ymax=1,color='brown',ls='--',lw=0.75)
        
    for elem_e in e_lines_high:
        ax_low.axvline(x=elem_e,ymin=0,ymax=1,color='brown',ls='--',lw=0.75)
    
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

plt.savefig('grs_obs'+('_withbshift' if with_bshift else '')+('_redshift' if with_redshift else '')+'.pdf')

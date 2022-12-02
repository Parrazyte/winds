#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 06:54:27 2020

@author: petrucpi
"""
import os
from numpy import *
from astropy.io import ascii

#
#Fichier de Susmita
#
#The 1st coulmn is in keV
#The 2nd column is erg/s/cm2 ---> nuf(nu).
#I had used a normlisation constant 1e20, because CLOUDY needed a large number 
#(a weird bug that I have not figured out). So, you should divide 2nd Column of 
#Ep0.01p0.30_Angle15_StopDist1e6.sed by 1e20.
#The resulting "flux (2nd column)" would then be ~ 1e-16 erg/s/cm2. That is because I had 
#calculated for a 10 Msol Black hole placed at 100 Mpc. If you account for this and move 
#the black hole to about 10 Kpc, you get the ball park number of ~1e-8 erg/s/cm2 
#(the number that you are looking for). Voila!


#dir='191106_Spectra4Joern/AthenaNXrism/p0.10'
#file='Ep0.01p0.10_Angle20_StopDist1e6'
#tmp=genfromtxt(file+'.sed',skip_header=1)
#enemin=tmp[:,0]
#eneminsub=enemin[where(enemin > 1e-4)]
#flux=tmp[:,1]/1e20*1e8 #Normalisation values
#fluxsub=flux[where(enemin > 1e-4)]
#enemaxsub=roll(eneminsub,-1)
#enemeansub=0.5*(enemaxsub+eneminsub)
#l=len(eneminsub)
#
#spec=fluxsub*(enemaxsub-eneminsub)/enemeansub/(enemeansub*1.6e-9)
#fileout=file+'.txt'
#ascii.write([eneminsub[0:l-1],enemaxsub[0:l-1],spec[0:l-1]],'tmp.txt')
#os.system("sed '/col0/d' tmp.txt > "+file+".txt")
#
#command='ftflx2tab '+file+'.txt '+file+' '+file+'.fits clobber = yes'
#os.system(command)



#
#Fichier de Sudeb
#
#The 1st coulmn is in eV
#The 2nd column is 1.0e38erg/s/erg

#dir='201202_spectrafromSudeb/low_hard/ep_0_01_p_0_3_angle_30'
#dir='/Users/petrucpi/Boulot/SOUS/ANR/ANRCHAOS/WorkWind/AbsorptionEmissionline/Simulations/24032020/210712_spectrafromSudeb'
#file='Final_blueshifted.10E+07'

#dir='/Users/petrucpi/Boulot/COM/PAPIERS/ANRCHAOS/PAPIERWIND/WINDXRB2/210924_Vrsn16/AllThatPopNeeds4DoubletChecks/XstarSpectra'
#file='Ep0.01p0.3_StopDist1e6Rg_Angle15'

dir='/media/parrama/6f58c7c3-ba85-45e6-b8b8-a8f0d564ec15/Simu/'
file='Final_blueshifted.10E+06'
os.chdir(dir)

tmp=genfromtxt(file+'.dat',skip_header=1)
enemin=tmp[:,0]*1e-3 #keV
eneminsub=enemin[where(enemin > 1e-4)]
flux=tmp[:,1]*1e38/(4.*3.14*(10.*1e3*3e18)**2.) #erg/s/erg/cm2
fluxsub=flux[where(enemin > 1e-4)]
enemaxsub=roll(eneminsub,-1)
enemeansub=0.5*(enemaxsub+eneminsub)
l=len(eneminsub)

spec=fluxsub/(enemeansub*1.6e-9) #ph/s/erg/cm2
spec=spec*(enemaxsub-eneminsub)*1.6e-9 #ph/s/cm2
fileout=file+'.txt'
ascii.write([eneminsub[0:l-1],enemaxsub[0:l-1],spec[0:l-1]],'tmp.txt',overwrite=True)

os.system("sed '/col0/d' tmp.txt > "+file+".txt")

#%%
dir='/Users/petrucpi/Boulot/SOUS/ANR/ANRCHAOS/WorkWind/AbsorptionEmissionline/Simulations/24032020/210712_spectrafromKeigo'
os.chdir(dir)
file='test2'

enemin=tmp[:,0] #keV
eneminsub=enemin[where(enemin > 1e-4)]
enemaxsub=roll(eneminsub,-1)
enemeansub=0.5*(enemaxsub+eneminsub)
l=len(eneminsub)

flux=tmp[:,1] #keV/s/keV/cm2
fluxsub=flux[where(enemin > 1e-4)]

spec=fluxsub*(enemaxsub-eneminsub)/enemeansub #ph/s/cm2
ascii.write([eneminsub[0:l-1],enemaxsub[0:l-1],spec[0:l-1]],'tmpb.txt',overwrite=True)

os.system("sed '/col0/d' tmpb.txt > "+file+".txt")

#%%
command='ftflx2tab '+file+'.txt '+file+' '+file+'.fits clobber = yes'
os.system(command)

#xscale('log')
#yscale('log')
#plot(enemeansub,fluxsub)



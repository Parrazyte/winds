#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from xspec import AllModels,AllData, Model,Fit,Plot,Spectrum,Xset

from xspec_config_multisp import reset,xPlot,Pset,calc_fit
from astropy.io import fits

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

startdir='/media/parrama/SSD/Observ/Pola/CygX-1'
os.chdir(startdir)

reset()

Plot.background=True
Plot.xLog=True

def load_first():
    AllData.clear()
    AllModels.clear()
    os.chdir(os.path.join(startdir,'INTEGRAL'))

    #updating the headers because it wasn't done
    with fits.open('CygX-1_2651_sum_pha.fits',mode='update') as hdul:
        hdul[1].header['RESPFILE']=''
        hdul.flush()

    # sp_int=Spectrum('CygX-1_2651_sum_pha.fits')
    os.chdir('../NuSTAR/14-06')

    #updating the headers because it wasn't done
    with fits.open('nu80902318004A01_sr.grp',mode='update') as hdul:

        #if we want to remake a direct group load (but annoying to load in different data groups)
        # hdul[1].header['RESPFILE']='./nu80902318004A01_sr.rmf'
        # hdul[1].header['ANCRFILE'] = './nu80902318004A01_sr.arf'
        # hdul[1].header['BACKFILE'] = './nu80902318004A01_bk.pha'

        hdul[1].header['RESPFILE']=''
        hdul[1].header['ANCRFILE'] = ''
        hdul[1].header['BACKFILE'] = ''

        hdul.flush()
    #
    # sp_nua=Spectrum('nu80902318004A01_sr.grp')

    with fits.open('nu80902318004B01_sr.grp',mode='update') as hdul:

        #if we want to remake a direct group load (but annoying to load in different data groups)
        # hdul[1].header['RESPFILE'] = './nu80902318004B01_sr.rmf'
        # hdul[1].header['ANCRFILE'] = './nu80902318004B01_sr.arf'
        # hdul[1].header['BACKFILE'] = './nu80902318004B01_bk.pha'

        hdul[1].header['RESPFILE']=''
        hdul[1].header['ANCRFILE'] = ''
        hdul[1].header['BACKFILE'] = ''

        hdul.flush()

    # sp_nub=Spectrum('nu80902318004B01_sr.grp')
    os.chdir('../../')

    AllData('1:1 INTEGRAL/CygX-1_2651_sum_pha.fits 2:2 NuSTAR/14-06/nu80902318004A01_sr.grp '+
            '3:3 NuSTAR/14-06/nu80902318004B01_sr.grp')
    AllData(1).response='INTEGRAL/CygX-1_2651_sum_rbn_rmf.fits'

    AllData(2).response='NuSTAR/14-06/nu80902318004A01_sr.rmf'
    AllData(2).response.arf = 'NuSTAR/14-06/nu80902318004A01_sr.arf'
    AllData(2).background= 'NuSTAR/14-06/nu80902318004A01_bk.pha'

    AllData(3).response='NuSTAR/14-06/nu80902318004B01_sr.rmf'
    AllData(3).response.arf = 'NuSTAR/14-06/nu80902318004B01_sr.arf'
    AllData(3).background= 'NuSTAR/14-06/nu80902318004B01_bk.pha'

    AllData(1).ignore('**-30. 500.-**')
    AllData(1).ignore('bad')

    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 70.-**')

    AllData(3).ignore('bad')
    AllData(3).ignore('**-3. 70.-**')

def load_second():

    AllData.clear()
    AllModels.clear()

    os.chdir(os.path.join(startdir,'INTEGRAL'))


    # updating the headers because it wasn't done
    with fits.open('CygX-1_2653_sum_pha.fits', mode='update') as hdul:
        hdul[1].header['RESPFILE'] = ''
        hdul.flush()

    # sp_int=Spectrum('CygX-1_2651_sum_pha.fits')
    os.chdir('../NuSTAR/20-06')

    # updating the headers because it wasn't done
    with fits.open('nu80902318006A01_sr.grp', mode='update') as hdul:
        # if we want to remake a direct group load (but annoying to load in different data groups)
        # hdul[1].header['RESPFILE'] = './nu80902318006A01_sr.rmf'
        # hdul[1].header['ANCRFILE'] = './nu80902318006A01_sr.arf'
        # hdul[1].header['BACKFILE'] = './nu80902318006A01_bk.pha'

        hdul[1].header['RESPFILE'] = ''
        hdul[1].header['ANCRFILE'] = ''
        hdul[1].header['BACKFILE'] = ''

        hdul.flush()

    with fits.open('nu80902318006B01_sr.grp', mode='update') as hdul:
        # if we want to remake a direct group load (but annoying to load in different data groups)
        # hdul[1].header['RESPFILE'] = './nu80902318006B01_sr.rmf'
        # hdul[1].header['ANCRFILE'] = './nu80902318006B01_sr.arf'
        # hdul[1].header['BACKFILE'] = './nu80902318006B01_bk.pha'

        hdul[1].header['RESPFILE'] = ''
        hdul[1].header['ANCRFILE'] = ''
        hdul[1].header['BACKFILE'] = ''

        hdul.flush()

    # sp_nub=Spectrum('nu80902318004B01_sr.grp')
    os.chdir('../../')

    # taking off the grouping of the NICER spectrum to load it from out of the folder
    with fits.open('NICER/NICERbyObsID/20-06/nicer_obs2_TOTAL.grp', mode='update') as hdul:
        # if we want to remake a direct group load (but annoying to load in different data groups)
        # hdul[1].header['RESPFILE'] = './nicer_obs2_TOTAL.rmf'
        # hdul[1].header['ANCRFILE'] = './nicer_obs2_TOTAL.arf'
        # hdul[1].header['BACKFILE'] = './nicer_obs2_TOTAL.scorpeonbg'

        hdul[1].header['RESPFILE'] = ''
        hdul[1].header['ANCRFILE'] = ''
        hdul[1].header['BACKFILE'] = ''

        hdul.flush()


    AllData('1:1 INTEGRAL/CygX-1_2653_sum_pha.fits 2:2 NuSTAR/20-06/nu80902318006A01_sr.grp ' +
            '3:3 NuSTAR/20-06/nu80902318006B01_sr.grp 4:4 NICER/NICERbyObsID/20-06/nicer_obs2_TOTAL.grp')
    AllData(1).response = 'INTEGRAL/CygX-1_2651_sum_rbn_rmf.fits'

    AllData(2).response = 'NuSTAR/14-06/nu80902318004A01_sr.rmf'
    AllData(2).response.arf = 'NuSTAR/14-06/nu80902318004A01_sr.arf'
    AllData(2).background = 'NuSTAR/14-06/nu80902318004A01_bk.pha'

    AllData(3).response = 'NuSTAR/14-06/nu80902318004B01_sr.rmf'
    AllData(3).response.arf = 'NuSTAR/14-06/nu80902318004B01_sr.arf'
    AllData(3).background = 'NuSTAR/14-06/nu80902318004B01_bk.pha'

    AllData(4).response='NICER/NICERbyObsID/20-06/nicer_obs2_TOTAL.rmf'
    AllData(4).response.arf='NICER/NICERbyObsID/20-06//nicer_obs2_TOTAL.arf'
    AllData(4).background='NICER/NICERbyObsID/20-06/nicer_obs2_TOTAL.scorpeonbg'

    AllData.notice('all')

    AllData(1).ignore('**-30. 500.-**')
    AllData(1).ignore('bad')

    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 70.-**')

    AllData(3).ignore('bad')
    AllData(3).ignore('**-3. 70.-**')

    AllData(4).ignore('bad')
    AllData(4).ignore('**-2. 10.-**')


def make_std_mod(fit=False,cut_low_Nustar=True):

    AllData.notice('all')

    AllData(1).ignore('**-30. 500.-**')
    AllData(1).ignore('bad')

    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 70.-**')

    AllData(3).ignore('bad')
    AllData(3).ignore('**-3. 70.-**')

    if cut_low_Nustar:
        AllData(2).ignore('**-4.')
        AllData(3).ignore('**-4.')

    #if a NICER set is loaded
    if AllData.nGroups>=4:
        AllData(4).ignore('bad')
        AllData(4).ignore('**-2. 10.-**')

    test=Model('constant(diskbb+relxill+xillver)')

    AllModels(2)(1).link=''
    AllModels(3)(1).link=''

    if AllData.nGroups>=4:
        AllModels(4)(1).link=''

    #opening the constant factors
    AllModels(1)(1).frozen=True

    #unfreezing the reflection powerlaw index
    AllModels(1)(15).frozen = False

    #linking the relxill and xillver relevant parameters
    AllModels(1)(18).link='p12'
    AllModels(1)(20).link='p15'
    AllModels(1)(23).link='p8'


    if fit:
        calc_fit()


def make_model_high(fit=False):

    AllData.notice('all')

    AllData(1).ignore('**-30. 500.-**')
    AllData(1).ignore('bad')

    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 70.-**')

    AllData(3).ignore('bad')
    AllData(3).ignore('**-3. 70.-**')

    #if a NICER set is loaded
    if AllData.nGroups>=4:
        AllData(4).ignore('bad')
        AllData(4).ignore('**-2. 10.-**')

    AllData.ignore('**-10.')
    test=Model('constant(cutoffpl)')

    #opening the constant factors
    AllModels(1)(1).frozen=True

    AllModels(2)(1).link=''
    AllModels(3)(1).link=''

    if AllData.nGroups>=3:
        AllModels(4)(1).link=''

    if fit:
        calc_fit()

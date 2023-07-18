#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from xspec import AllModels,AllData, Model,Fit,Plot,Spectrum,Xset
import glob

from xspec_config_multisp import reset,xPlot,Pset,calc_fit
from astropy.io import fits

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

startdir='/media/parrama/SSD/Observ/Pola/CygX-1'
os.chdir(startdir)

reset()
Pset(xlog=True)
Plot.background=True


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

    AllData('1:1 NuSTAR/14-06/nu80902318004A01_sr.grp '+
            '2:2 NuSTAR/14-06/nu80902318004B01_sr.grp '+
            '3:3 INTEGRAL/CygX-1_2651_sum_pha.fits')

    AllData(1).response='NuSTAR/14-06/nu80902318004A01_sr.rmf'
    AllData(1).response.arf = 'NuSTAR/14-06/nu80902318004A01_sr.arf'
    AllData(1).background= 'NuSTAR/14-06/nu80902318004A01_bk.pha'
    AllData(1).ignore('bad')
    AllData(1).ignore('**-3. 70.-**')

    AllData(2).response='NuSTAR/14-06/nu80902318004B01_sr.rmf'
    AllData(2).response.arf = 'NuSTAR/14-06/nu80902318004B01_sr.arf'
    AllData(2).background= 'NuSTAR/14-06/nu80902318004B01_bk.pha'
    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 70.-**')

    AllData(3).response = 'INTEGRAL/CygX-1_2651_sum_rbn_rmf.fits'
    AllData(3).ignore('**-30. 500.-**')
    AllData(3).ignore('bad')


def load_second(NICER='obsid'):

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

    if NICER=='obsid':

        NICER_gti_grps = ['NICER/NICERbyObsID/20-06/nicer_obs2_TOTAL.grp']

    elif NICER == 'gtis':

        NICER_gti_grps = glob.glob('NICER/NICERbyGTI/20-06/**.grp')

    #and updating the fits files
    for elem_gti in NICER_gti_grps:
        # taking off the grouping of the NICER spectrum to load it from out of the folder
        with fits.open(elem_gti, mode='update') as hdul:
            # if we want to remake a direct group load (but annoying to load in different data groups)


            hdul[1].header['RESPFILE'] = ''
            hdul[1].header['ANCRFILE'] = ''
            hdul[1].header['BACKFILE'] = ''

            hdul.flush()

    datas=['NuSTAR/20-06/nu80902318006A01_sr.grp','NuSTAR/20-06/nu80902318006B01_sr.grp']

    datas+=NICER_gti_grps

    datas+=['INTEGRAL/CygX-1_2653_sum_pha.fits']

    AllData(' '.join([str(i+1)+':'+str(i+1)+' '+datas[i] for i in range(len(datas))]))

    AllData(1).response = datas[0].split('_')[0]+'_sr.rmf'
    AllData(1).response.arf = datas[0].split('_')[0]+'_sr.arf'
    AllData(1).background = datas[0].split('_')[0]+'_bk.pha'

    AllData(2).response = datas[1].split('_')[0]+'_sr.rmf'
    AllData(2).response.arf = datas[1].split('_')[0]+'_sr.arf'
    AllData(2).background = datas[1].split('_')[0]+'_bk.pha'


    AllData(1).ignore('**-3. 70.-**')
    AllData(2).ignore('**-3. 70.-**')

    for i_gti,elem_gti in enumerate(NICER_gti_grps):

        AllData(i_gti+3).response=datas[i_gti+2].split('.')[0]+'.rmf'
        AllData(i_gti+3).response.arf=datas[i_gti+2].split('.')[0]+'.arf'
        AllData(i_gti+3).background=datas[i_gti+2].split('.')[0]+'.scorpeonbg'

        AllData(i_gti+3).ignore('**-0.5 10.-**')

    #loading integral response
    AllData(3+len(NICER_gti_grps)).response = 'INTEGRAL/CygX-1_2651_sum_rbn_rmf.fits'
    AllData(3+len(NICER_gti_grps)).ignore('**-30. 500.-**')

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

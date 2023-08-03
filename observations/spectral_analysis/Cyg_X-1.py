#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from xspec import AllModels,AllData, Model,Fit,Plot,Spectrum,Xset
import glob
import time

from xspec_config_multisp import reset,xPlot,Pset,calc_fit, calc_error, addcomp,delcomp,\
    display_mod_errors,allmodel_data,calc_fit
from astropy.io import fits

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

startdir='/media/parrama/SSD/Observ/Pola/CygX-1'
os.chdir(startdir)

reset()
Pset(xlog=True)
Plot.background=True

AllModels.mdefine('mbknpo (max(E,B)-B)/abs(E-B+0.0000001)+(1-(max(E,B)-B)/abs(E-B+0.0000001))*(E/B)^I : mul')

def load_first(integral=True):
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
            ('3:3 INTEGRAL/CygX-1_2651_sum_pha.fits' if integral else ''))

    AllData(1).response='NuSTAR/14-06/nu80902318004A01_sr.rmf'
    AllData(1).response.arf = 'NuSTAR/14-06/nu80902318004A01_sr.arf'
    AllData(1).background= 'NuSTAR/14-06/nu80902318004A01_bk.pha'
    AllData(1).ignore('bad')
    AllData(1).ignore('**-3. 78.-**')

    AllData(2).response='NuSTAR/14-06/nu80902318004B01_sr.rmf'
    AllData(2).response.arf = 'NuSTAR/14-06/nu80902318004B01_sr.arf'
    AllData(2).background= 'NuSTAR/14-06/nu80902318004B01_bk.pha'
    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 78.-**')

    if integral:
        AllData(3).response = 'INTEGRAL/CygX-1_2651_sum_rbn_rmf.fits'
        AllData(3).ignore('**-30. 500.-**')
        AllData(3).ignore('bad')


def load_second(NICER='gtis', integral=True):

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

        NICER_gti_grps = np.array(glob.glob('NICER/NICERbyGTI/20-06/**.grp'))
        #sorting to match Ed's order
        NICER_gti_grps.sort()

        #reconverting in list to concatenate afterwards
        NICER_gti_grps=NICER_gti_grps.tolist()


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

    if integral:
        datas+=['INTEGRAL/CygX-1_2653_sum_pha.fits']

    AllData(' '.join([str(i+1)+':'+str(i+1)+' '+datas[i] for i in range(len(datas))]))

    AllData(1).response = datas[0].split('_')[0]+'_sr.rmf'
    AllData(1).response.arf = datas[0].split('_')[0]+'_sr.arf'
    AllData(1).background = datas[0].split('_')[0]+'_bk.pha'

    AllData(2).response = datas[1].split('_')[0]+'_sr.rmf'
    AllData(2).response.arf = datas[1].split('_')[0]+'_sr.arf'
    AllData(2).background = datas[1].split('_')[0]+'_bk.pha'

    #there seems to be an issue here so we add a 1s waiting time
    time.sleep(1)

    AllData(1).ignore('**-3. 78.-**')
    AllData(2).ignore('**-3. 78.-**')

    for i_gti,elem_gti in enumerate(NICER_gti_grps):

        AllData(i_gti+3).response=datas[i_gti+2].split('.')[0]+'.rmf'
        AllData(i_gti+3).response.arf=datas[i_gti+2].split('.')[0]+'.arf'
        AllData(i_gti+3).background=datas[i_gti+2].split('.')[0]+'.scorpeonbg'

        AllData(i_gti+3).ignore('**-0.5 10.-**')

    #loading integral response
    if integral:
        AllData(3+len(NICER_gti_grps)).response = 'INTEGRAL/CygX-1_2651_sum_rbn_rmf.fits'
        AllData(3+len(NICER_gti_grps)).ignore('**-30. 500.-**')

def load_nathan_ref_1(prefit=True,integral=False):
    load_first(integral=False)
    Xset.restore('full_reflection__2.xcm')

    # extending the energy range for simpl
    AllModels.setEnergies('0.05 1000 1000 log')

    if prefit:
        calc_fit(timeout=30)

    if integral:
        mod_init=allmodel_data()
        AllModels.clear()
        AllData.clear()

        load_first()
        mod_init.load()

        AllModels(3)(1).link=''
        AllModels(3)(1).frozen=False


def load_nathan_ref_2(prefit=True,integral=False,nolow=False):

    if not integral:
        load_second(NICER='gtis',integral=False)
        Xset.restore('full_reflection__3.xcm')

        #extending the energy range for simpl
        AllModels.setEnergies('0.05 1000 1000 log')

        if prefit:
            calc_fit(timeout=30)
    else:
        Xset.restore("ref_3_post_fit.xcm")

    if integral:
        mod_init=allmodel_data()

        AllModels.clear()
        AllData.clear()
        AllModels.systematic=0.

        load_second()

        mod_init.load()

        AllModels(16)(1).link=''
        AllModels(16)(1).frozen=False

        if nolow:
            AllModels(1)(8).frozen=True
            AllModels(1)(9).frozen = True
            AllModels(1)(10).frozen = True

            #ignoring the lower part of the NICER data
            for i in range(3,16):
                AllData(i).ignore('**-2.5')

            #fixing the low E gaussian values
            #the E and sigma are in the first datagroup and the norm in the first nicer
            #datagroup (third)
            AllModels(1)(43).frozen=True
            AllModels(1)(44).frozen = True
            AllModels(3)(45).frozen = True
            AllModels(1)(46).frozen = True
            AllModels(1)(47).frozen = True
            AllModels(3)(48).frozen = True

            #same for the first gabs, who's entirely from the nicer datagroup
            AllModels(3)(3).frozen=True
            AllModels(3)(4).frozen = True
            AllModels(3)(5).frozen=True

            #and the edge

            AllModels(3)(6).frozen = True
            AllModels(3)(7).frozen=True

            #approaching the normalization factor of some parameter else the fit crashes
            AllModels(16)(1).values = 4.95

            #putting the line normalizations to zero
            AllModels(3)(45).values=0
            AllModels(3)(48).values=0
            AllModels(3)(5).values=0
            AllModels(3)(7).values=0

def load_nathan_emp_1(integral=False):
    load_first(integral=False)
    Xset.restore('empirical__2.xcm')

    #this shouldn't be here so we're updating it
    AllModels.systematic=0
    Fit.perform()

    if integral:
        mod_init=allmodel_data()
        AllModels.clear()
        AllData.clear()
        AllModels.systematic=0.

        load_second()
        mod_init.load()


def load_nathan_emp_2(integral=False,nolow=False):

    load_second(NICER='gtis',integral=False)
    Xset.restore('empirical__3.xcm')

    if integral:
        mod_init=allmodel_data()
        AllModels.clear()
        AllData.clear()
        AllModels.systematic=0.

        load_second()
        mod_init.load()

        AllModels(16)(1).link=''
        AllModels(16)(1).frozen=False

        if nolow:
            AllModels(1)(3).frozen=True
            AllModels(1)(4).frozen = True
            AllModels(1)(5).frozen = True

            #ignoring the lower part of the NICER data
            for i in range(3,16):
                AllData(i).ignore('**-2.5')

def nathan_mod_refl():

    AllModels.clear()

    mod=Model('constant*mtable{nuMLIv1.mod}*gabs*edge*TBfeo*simpl(mbknpo(relxillCp + xillverCp) + diskbb) + gaussian + gaussian')


    # extending the energy range for simpl
    AllModels.setEnergies('0.05 1000 1000 log')

def nathan_mod_emp(integral=True,systematics=False,custom=False):

    mod=Model('constant*TBfeo(diskbb + smedge*nthComp + laor)')

    #freezing the first constant factor
    AllModels(1)(1).frozen=True

    #and unfreezing the second one
    AllModels(2)(1).link=''
    AllModels(2)(1).frozen=False


    #consrtaining the nH values (and freezing if not using NICER data (first obs)
    AllModels(1)(2).values = 0.7

    if AllData.nGroups<=3:
        AllModels(1)(2).frozen=True
    else:
        AllModels(1)(3).frozen = False

    AllModels(1)(4).values=0.5

    #resticting the diskbb Tin range
    AllModels(1)(6).values=[0.5,0.005,0.3,0.3,1.,1.]

    #fixing/restricting smedge parameters
    AllModels(1)(8).values=[8,0.01,7.,7.,9.,9.]
    AllModels(1)(11).values=7
    AllModels(1)(11).frozen=True

    #linking nthcomp parameters
    AllModels(1)(14).link='p6'
    AllModels(1)(15).values=1

    #setting up laor parameters
    AllModels(1)(18).values=6.67
    AllModels(1)(18).frozen=True
    AllModels(1)(19).frozen=False
    AllModels(1)(22).frozen=False

    last_group = AllData.nGroups

    #common norm factor for NICER groups if they are there
    if AllData.nGroups>3:
        for i_grp in range(3,last_group+(0 if integral else 1)):

            if i_grp==3:
                AllModels(i_grp)(1).link=''
                AllModels(i_grp)(1).frozen = False
            else:
                AllModels(i_grp)(1).link='p'+str(1+2*AllModels(1).nParameters)

        #putting systematics at 0.01 if for the second observation
        if systematics:
            AllModels.systematic=0.01

    #unfreezing the constant factor for the last data group (integral) if it's there
    if integral:
        AllModels(last_group)(1).link=''
        AllModels(last_group)(1).frozen=False

    #first fit without FPMA
    AllData(1).ignore('3.-7.')

    if custom:
        delcomp('smedge')
        #delcomp('laor')
        AllData(3).ignore('**-2.5')

        #freezing the nH values since it's mostly unconstrained with this energy restriction
        AllModels(1)(2).frozen=True
        AllModels(1)(3).frozen=True

    Fit.perform()

    #freezing the second constant factor
    AllModels(2)(1).frozen=True

    #re-noticing low-E FPMA
    AllData(1).notice('3.-7.')

    #adding the MLI component behind the constant factor
    addcomp('mtable{nuMLIv1.mod}',position=2)

    #freezing the second MLI to 1
    AllModels(2)(2).values=1
    AllModels(2)(2).frozen=True

    if AllData.nGroups>=2:
        for i_grp in range(2,last_group+1):
            #same for every other spectrum
            AllModels(i_grp)(2).values=1
            AllModels(i_grp)(2).frozen=True

    if custom:
        AllModels(2)(1).frozen=False

        Fit.perform()

        #re-adding the smedge with the correct parameters
        addcomp('smedge',position=5)
        AllModels(1)(9).values=[8.,0.,0.1,7.,7.,9.,9.]
        AllModels(1)(9).frozen=False
        AllModels(1)(12).values=7
        AllModels(1)(12).frozen=True


        Fit.perform()

def mod_nthcomp():

    test=Model("nthcomp")

    addcomp('glob_constant')

    Fit.perform()

def make_std_mod(fit=False,cut_low_Nustar=True):

    AllData.notice('all')

    AllData(1).ignore('**-30. 500.-**')
    AllData(1).ignore('bad')

    AllData(2).ignore('bad')
    AllData(2).ignore('**-3. 78.-**')

    AllData(3).ignore('bad')
    AllData(3).ignore('**-3. 78.-**')

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
    AllData(2).ignore('**-3. 78.-**')

    AllData(3).ignore('bad')
    AllData(3).ignore('**-3. 78.-**')

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

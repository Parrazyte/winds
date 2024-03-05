
import numpy as np

import time
import pexpect

from tqdm import tqdm
from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,Chain

#custom script with a few shorter xspec commands
from xspec_config_multisp import allmodel_data,model_load,addcomp,Pset,Pnull,rescale,reset,Plot_screen,store_plot,freeze,allfreeze,unfreeze,\
                         calc_error,delcomp,fitmod,calc_fit,xcolors_grp,xPlot,xscorpeon,catch_model_str,\
                         load_fitmod, ignore_data_indiv,par_degroup,xspec_globcomps

import matplotlib.pyplot as plt
import os,sys

def simu_xrism(mode='ew_lim',mod_path=None,rmf_path='Hp',arf_path='pointsource_GVclosed',expos=50,flux_range='1_100_20',
               line='FeKa26abs',line_v=[-3000,3000],line_w=[0,0.05],width_test_val=0.02,width_EW_resol=1):

    '''
    Computes XRISM simulations of line detection

    arguments:

        mode:
            -ew_lim: computes the ew limits at 1, 2, 3 sigma by fitting an additional absorption line in the spectrum
            -width_lim, computes the
        -mod_path:xspec_mod_path. if None, uses the currently loaded model instead

        -rmf_path/arf_path: rmf/arf to use for the fakes. must be an absolute path. available shortcuts:
            rmf_path:
                'Hp': rsl_Hp_5eV.rmf
                'Mp': rsl_Mp_6eV.rmf
                'Lp': rsl_Lp_18eV.rmf

            arf_path:
                'pointsource_GVclosed'      rsl_pointsource_GVclosed.arf
                'pointsource_off_GVclosed'  rsl_pointsource_off_GVclosed.arf

        -expos: exposure value in kiloseconds

        -flux_range:
            flux value interval to be parsed. The spectrum will be renormalized to have its values in the
            3-10keV band
            the interval is low_lim_high_lim_nsteps in log space, low and highlim in units of 1e-8 erg/cm²/s

        -line: name of the line to test (mainly for the energy)


        EW_lim mode:
            -line_v/line_w: min/max velocity and width intervals to allow when fitting the line

        width_lim mode:
            -the line is taken at 0 velocity and fitted

            -width_test_val: test width to witch to fetch the lowest EW
            -width_resol:resolution for when to stop when trying to find the limit for computing the width of the line


    '''

    AllData.clear()

    rmf_abv={'Hp':'/media/parrama/SSD/Observ/highres/XRISM_responses/rsl_Hp_5eV.rmf',
             'Mp':'/media/parrama/SSD/Observ/highres/XRISM_responses/rsl_Mp_6eV.rmf',
             'Lp':'/media/parrama/SSD/Observ/highres/XRISM_responses/rsl_Lp_18eV.rmf'}

    rmf_abv_list=list(rmf_abv.keys())

    arf_abv={'pointsource_GVclosed':'/media/parrama/SSD/Observ/highres/XRISM_responses/rsl_pointsource_GVclosed.arf',
            'pointsource_off_GVclosed':'/media/parrama/SSD/Observ/highres/XRISM_responses/rsl_pointsource_off_GVclosed.arf'}

    arf_abv_list=list(arf_abv.keys())

    if rmf_path in rmf_abv_list:
        rmf_path_use=rmf_abv[rmf_path]
    else:
        rmf_path_use=rmf_path

    if arf_path in arf_abv_list:
        arf_path_use=arf_abv[arf_path]
    else:
        arf_path_use=arf_path

    if mod_path is not None:
        AllModels.clear()
        Xset.restore(mod_path)
        freeze()

    #computing the 3-10 flux
    AllModels.calcFlux('3. 10.')
    flux_base=AllModels(1).flux

    flux_range_vals=np.array(flux_range.split('_')).astype(float)

    n_flux=int(flux_range_vals[2])
    flux_inter=np.logspace(np.log10(flux_range_vals[0]),np.log10(flux_range_vals[1]),n_flux)

    #adding a constant from the flux value to renormalize
    addcomp('glob_constant')

    #needs to be frozen for a single datagroup
    AllModels(1)(1).frozen=True

    mod_cont=allmodel_data()

    fakeset = FakeitSettings(response=rmf_path_use,arf=arf_path_use,exposure=expos*1000,
                              background='',
                              fileName='temp_sp.pi')

    #creating a bash process to run ftgrouppha on the fake spectra
    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')
    bashproc.sendline('heainit')

    print('Computing detectability')

    if mode=='ew_lim':

        print('Computing EW limits for the given flux range...')
        eqw_lim_arr=np.zeros((n_flux,3))

        with tqdm(total=n_flux) as pbar:
            for i_flux,elem_flux in enumerate(flux_inter):

                mod_cont.load()

                #freezing the parameters before faking
                freeze()

                AllModels(1)(1).values=flux_base/elem_flux

                #remove previously computed spectra
                if os.path.isfile('temp_sp.pi'):
                    os.remove('temp_sp.pi')

                if os.path.isfile('temp_sp_grp_opt.pi'):
                    os.remove('temp_sp_grp_opt.pi')

                AllData.fakeit(settings=fakeset, applyStats=True)

                #rebinning the spectrum before loading it
                bashproc.sendline('ftgrouppha infile=temp_sp.pi'+' outfile=temp_sp_grp_opt.pi grouptype= opt'+
                                  ' respfile='+rmf_path_use)

                #waiting for the spectrum to be created:
                while not os.path.isfile('temp_sp_grp_opt.pi'):
                    time.sleep(1)

                AllData.clear()

                AllData('1:1 temp_sp_grp_opt.pi')

                AllData.ignore('**-2. 10.-**')

                AllData(1).rmf.arf=arf_path_use

                #loading the continuum model and fitting
                mod_cont.load()
                AllModels(1)(1).values=flux_base/elem_flux

                calc_fit()

                #freezing the continuum
                freeze()

                #adding the line component
                comp_par,comp_num=addcomp(line+'_agaussian', position='lastinall',return_pos=True)

                #with appropriate parameter range
                AllModels(1)(comp_par[0]).values=[0.,10.,line_v[0],line_v[0],line_v[1],line_v[1]]
                AllModels(1)(comp_par[2]).values = [0., 0.01, line_w[0], line_w[0], line_w[1], line_w[1]]

                #fitting
                calc_fit()

                #and computing the eqwidth of the best fit in the interval

                AllModels.eqwidth(comp_num[-1], err=True, number=1000, level=68)

                eqw_lim_arr[i_flux][0]=-AllData(1).eqwidth[1]

                AllModels.eqwidth(comp_num[-1], err=True, number=1000, level=95)

                eqw_lim_arr[i_flux][0]=-AllData(1).eqwidth[1]

                AllModels.eqwidth(comp_num[-1], err=True, number=1000, level=99.7)

                eqw_lim_arr[i_flux][0]=-AllData(1).eqwidth[1]

                pbar.update()

        save_arr=np.concatenate((np.array([flux_inter]),eqw_lim_arr.T)).T

        header_elems=['mod_path '+str(mod_path),'rmf_path '+rmf_path,'arf_path '+arf_path,'expos '+str(expos)+' ks',
                      'flux_range logspace('+flux_range+') e-8 erg/s/cm² ',
                      'line '+line,'line_v '+str(line_v),'line_w '+str(line_w),
                      'columns: flux ew_limit at 1/2/3 sigma']

        np.savetxt('ew_lim_mod.txt',save_arr,header='\n'.join(header_elems))
        return flux_inter,eqw_lim_arr

    if mode=='width_lim':

        print('Computing width detectability for the given flux range...')
        width_lim_arr=np.zeros((n_flux,3))

        EW_init=100

        Ew_vals_queue=[]

        '''
        The method here is to parse the EW interval by multipling/diving by progressively lower factors until
        we find the right value (binary search)
        '''

        base_step=1/2

        step_1sig=base_step
        step_2sig=base_step
        step_3sig=base_step

        with tqdm(total=n_flux) as pbar:
            for i_flux,elem_flux in enumerate(flux_inter):

                #reloading the continuum model
                mod_cont.load()

                #adjusting the luminosity
                AllModels(1)(1).values = flux_base / elem_flux

                #adding the test line
                comp_par, comp_num = addcomp(line + '_agaussian', position='lastinall')

                # freezing the blueshift at 0
                AllModels(1)(comp_par[0]).frozen = True

                # and the width at the desired value
                AllModels(1)(comp_par[2]).values = width_test_val

                # faking a spectrum to get the eqwidth because xspec is garbage
                AllData.fakeit(noWrite=True)
                AllModels.eqwidth(comp_num[-1])

                # computing the normalization factor (with a negative so we can keep our EW in positive values
                norm_EW_factor = -AllModels(1)(comp_par[-1]).values[0]/AllData(1).eqwidth[0]

                #storing the model
                freeze()
                mod_width_base=allmodel_data()

                while (width_lim_arr[i_flux]==0).any():

                    #freezing the parameters before faking
                    freeze()

                    #remove previously computed spectra
                    if os.path.isfile('temp_sp.pi'):
                        os.remove('temp_sp.pi')

                    if os.path.isfile('temp_sp_grp_opt.pi'):
                        os.remove('temp_sp_grp_opt.pi')

                    #faking
                    AllData.fakeit(settings=fakeset, applyStats=True)

                    #rebinning the spectrum before loading it
                    bashproc.sendline('ftgrouppha infile=temp_sp.pi'+' outfile=temp_sp_grp_opt.pi grouptype= opt'+
                                      ' respfile='+rmf_path_use)

                    #waiting for the spectrum to be created:
                    while not os.path.isfile('temp_sp_grp_opt.pi'):
                        time.sleep(1)

                    AllData.clear()

                    AllData('1:1 temp_sp_grp_opt.pi')

                    AllData.ignore('**-2. 10.-**')

                    AllData(1).rmf.arf=arf_path_use

                    #loading the cont version of the width model (to avoid issues with parameters)
                    mod_cont.load()

                    # adjusting the luminosity
                    AllModels(1)(1).values = flux_base / elem_flux

                    #fitting
                    calc_fit()

                    #testing whether the line width is constrained

                    # computing the width with the current fit at a given sigma (check values of the delchi)
                    Fit.error('stop ,,0.1 max 100 9.00 ' + str(comp_par[-2]))

                    '''
                    we consider the width constrained if the lower bound is more than 1% of the base value
                    (since here we know the base value), because things tend to peg at very low values instead
                    of saying thart they're frozen at 0
                    '''
                    if AllModels(1)(comp_par[-2]).error[0]>=width_test_val/100:
                        step_1sig=False

                    #note: go down to the lowest value to which 1 sigma is constrained with a delta of width_resol, and then
                    #come back up, having each of the previous minimum values for which the 2 and 3 sigma weren't

                    #unfreezing



                pbar.update()

        # save_arr=np.concatenate((np.array([flux_inter]),eqw_lim_arr.T)).T
        #
        # header_elems=['mod_path '+str(mod_path),'rmf_path '+rmf_path,'arf_path '+arf_path,'expos '+str(expos)+' ks',
        #               'flux_range logspace('+flux_range+') e-8 erg/s/cm² ',
        #               'line '+line,'line_v '+str(line_v),'line_w '+str(line_w),
        #               'columns: flux ew_limit at 1/2/3 sigma']
        #
        # np.savetxt('ew_lim_mod.txt',save_arr,header='\n'.join(header_elems))
        # return flux_inter,eqw_lim_arr
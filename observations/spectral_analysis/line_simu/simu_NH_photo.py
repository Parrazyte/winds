import os,sys
import numpy as np

import time

#for dynamical argument fetching
import inspect

import pexpect
import matplotlib.pyplot as plt
from tqdm import tqdm
from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,Chain
from fitting_tools import sign_sigmas_delchi_1dof
#custom script with a few shorter xspec commands
from xspec_config_multisp import allmodel_data,model_load,addcomp,Pset,Pnull,rescale,reset,Plot_screen,store_plot,freeze,allfreeze,unfreeze,\
                         calc_error,delcomp,fitmod,calc_fit,xcolors_grp,xPlot,xscorpeon,catch_model_str,\
                         load_fitmod, ignore_data_indiv,par_degroup,xspec_globcomps,is_abs,lines_e_dict,calc_EW,set_ener

from general_tools import c_Km as c_0
from general_tools import file_edit

def simu_nh_photo(nh_photo_mode,
                  flux_inter,n_iter,
                  mod_cont,flux_base,
                  fakeset,fakestats,
                  regroup,rmf_path_use,arf_path_use,bkg_path_use,bashproc,
                  analysis_lowe,analysis_highe,
                  set_ener_data,set_ener_data_str,
                  mod_dict,photo_mod,photo_comp_pos,
                  photo_xi_range,photo_turb_range,photo_v_range,
                  logfile,to_error,
                  mod_path,expos,flux_range,flux_band,
                  n_cores,
                  photo_nsteppar_turb,
                  photo_nsteppar_v,
                  par_freeze_steppar
                  ):

    '''

    Wrapper for the part of line_simu specialized in NH upper limits estimates from photoionization models

    mode:
        -lim: computes 1/2/3 sigma NH upper limits for a photoionization model convolved to the source model

        -noise: computes photon noise 1/2/3 sigma of best fit noise NH for a photoionization model
            convolved to the source model. The NH value is taken to be the highest in a 2D steppar fit
            in v_turb/v space, within photo_turb_range (in log space) and photo_v_range (in lin space)

            photo_nsteppar_vturb/v:
            the number of steppar steps for each dimension
            note: to be provided in actual step numbers.
            The values actually given to steppar (which adds 1) are these -1

            par_freeze_steppar: array-like of integers

                Freeze the list of parameters before the computation is ran
                these parameters are assumed from the continuum, and as such should be given in the model enumeration
                before the photoionization model is added

                In noise mode, freezing parameters that tend to be pegged or unconstrained
                can considerably help improving the speed


        photo_xi_range:
            hot/warm+('_'+freeze)
            for the range of ionization parameter. if includes freeze, blocks the ionization parameter to the
            initial value.

        photo_turb_range/photo_v_range:
            the allowed velocity and turbulence parameter spaces for this mode


    '''

    print('Computing NH '+('limits' if nh_photo_mode=='lim' else 'photon noise' if nh_photo_mode=='noise' else '')+
          ' for photoionization within the given flux range...')
    
    n_flux=len(flux_inter)
    
    nh_lim_arr = np.zeros((n_flux, 3))

    instance_id=time.time()
    
    with (tqdm(total=n_flux * n_iter) as pbar):

        for i_flux, elem_flux in enumerate(flux_inter):

            #may get modified in noise mode
            n_iter_use=n_iter

            if nh_photo_mode=='lim':
                nh_val_distrib = np.repeat(None, 3 * n_iter).reshape(3, n_iter)

            elif nh_photo_mode=='noise':
                nh_val_distrib = np.repeat(None, n_iter)

                #testing whether a save file exists
                elem_flux_save_path='save_noise_flux_'+str(elem_flux).replace('.','p')+'.txt'

                #defining a header
                # fetching dynamicallly all the functio arguments to ensure we are doing the same computation
                # from https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
                sig, simu_locals = inspect.signature(simu_nh_photo), locals()
                par_list = [[param.name,simu_locals[param.name]] for param in sig.parameters.values()]
                save_header = str(par_list)


                if os.path.isfile(elem_flux_save_path):
                    with open(elem_flux_save_path) as f_save:
                        pre_save_lines=f_save.readlines()
                    assert pre_save_lines[0].replace('\n','')==save_header,\
                            'Error: previous save file has different arguments'

                    #removing a number of iterations equal to the number of saved results
                    n_iter_use=n_iter-(len(pre_save_lines)-1)
                    pbar.update(len(pre_save_lines)-1)

            for i_iter in range(n_iter_use):
                mod_cont.load()

                # freezing the parameters before faking
                freeze()

                AllModels(1)(1).values = elem_flux / flux_base

                # remove previously computed spectra
                for elem_set in fakeset:
                    if os.path.isfile(elem_set.fileName):
                        os.remove(elem_set.fileName)

                # faking the spectrum with the right parameters
                AllData.fakeit(nSpectra=len(fakeset), settings=fakeset, applyStats=fakestats)

                # rebinning the spectrum before loading it
                if regroup:
                    # using optsnmin puts some bins at weird wiggling ratios
                    # bashproc.sendline('ftgrouppha infile=temp_sp.pi'+' outfile=temp_sp_grp_opt.pi '+
                    #                   ' grouptype=optsnmin groupscale=3.0'+
                    #                   ' respfile='+rmf_path_use+' clobber=True')

                    # using opt puts some bins at 0 for some reason maybe bc the rmf has issues
                    group_str = 'ftgrouppha infile=temp_sp.pi outfile=temp_sp_grp_opt.pi ' + \
                                'grouptype=opt' + \
                                'respfile=' + rmf_path_use + ' clobber=True'
                    bashproc.sendline(group_str)

                    # waiting for the spectrum to be created:
                    while not os.path.isfile('temp_sp_grp_opt.pi'):
                        time.sleep(1)

                    AllData.clear()
                    AllData('1:1 temp_sp_grp_opt.pi')

                AllData.ignore('**-' + str(float(analysis_lowe)) + ' ' + str(float(analysis_highe)) + '-**')

                if set_ener_data:
                    set_ener(set_ener_data_str, xrism=True)

                for i_grp in range(1, AllData.nGroups + 1):
                    AllData(i_grp).response.arf = arf_path_use[i_grp - 1]

                # loading the continuum model and fitting
                mod_cont.load()
                AllModels(1)(1).values = elem_flux / flux_base

                calc_fit()

                if Fit.statistic / Fit.dof > 2:
                    print('Issue with fake continuum fitting.')
                    breakpoint()
                    pass

                XRISM_sp = AllData(1).fileinfo('TELESCOP') == 'XRISM'

                # adding the photoionization component
                comp_par, comp_num = addcomp(mod_dict[photo_mod][0] + '{' + mod_dict[photo_mod][1].split('/')[-1] + '}',
                                             position=photo_comp_pos, return_pos=True)

                # with appropriate parameter range
                if photo_mod == 'pion_abs_NS':

                    if photo_xi_range.split('_')[0] == 'warm':
                        AllModels(1)(comp_par[0]).values = [2., 0.02, 1., 1., 3., 3.]
                    elif photo_xi_range.split('_')[0] == 'hot':
                        AllModels(1)(comp_par[0]).values = [4., 0.02, 3., 3., 4.5, 4.5]

                    if '_' in photo_xi_range and photo_xi_range.split('_')[1] == 'freeze':
                        AllModels(1)(comp_par[0]).frozen = True

                    AllModels(1)(comp_par[1]).values = [1.0, 0.1, 1e-2, 1e-2, 10.0, 10.0]
                    AllModels(1)(comp_par[2]).values = [200.0, 2.0, 100, 100, 500, 500]

                    # equivalent to +/-1000km/s
                    AllModels(1)(comp_par[3]).values = [0.0, 1e-3, -0.00333, -0.00333, 0.00333, 0.00333]

                if photo_mod == 'pion_abs_canon_soft':
                    if photo_xi_range.split('_')[0] == 'cold':
                        AllModels(1)(comp_par[0]).values = [0., 0.02, 0., 0., 0.5, 0.5]
                    if photo_xi_range.split('_')[0] == 'warm':
                        AllModels(1)(comp_par[0]).values = [2., 0.02, 1., 1., 3., 3.]
                    elif photo_xi_range.split('_')[0] == 'hot':
                        AllModels(1)(comp_par[0]).values = [4., 0.02, 2.0, 3.5, 4.5, 4.5]

                    if '_' in photo_xi_range and photo_xi_range.split('_')[1] == 'freeze':
                        AllModels(1)(comp_par[0]).frozen = True

                    AllModels(1)(comp_par[1]).values = [1.0, 0.1, 1e-2, 1e-2, 10.0, 10.0]
                    AllModels(1)(comp_par[3]).values = [photo_turb_range[0], photo_turb_range[0] / 10,
                                                        AllModels(1)(comp_par[3]).values[2] if photo_turb_range[
                                                                                                   1] == 'min' else
                                                        photo_turb_range[1],
                                                        AllModels(1)(comp_par[3]).values[2] if photo_turb_range[
                                                                                                   1] == 'min' else
                                                        photo_turb_range[1],
                                                        AllModels(1)(comp_par[3]).values[-1] if photo_turb_range[
                                                                                                    1] == 'max' else
                                                        photo_turb_range[2],
                                                        AllModels(1)(comp_par[3]).values[-1] if photo_turb_range[
                                                                                                    1] == 'max' else
                                                        photo_turb_range[2],
                                                        ]

                    # equivalent to +/-1000km/s
                    AllModels(1)(comp_par[4]).values = [photo_v_range[0] / c_0, photo_v_range[0] / 10 / c_0,
                                                        photo_v_range[1] / c_0, photo_v_range[1] / c_0,
                                                        photo_v_range[2] / c_0, photo_v_range[2] / c_0]

                # fitting
                calc_fit()

                Fit.query = 'yes'
                # computing the error on the velocity shift parameter of the line to ensure we are not stuck
                calc_error(logfile, param=str(comp_par[3]), timeout=15, freeze_pegged=True)
                calc_fit()
                Fit.query = 'on'

                if nh_photo_mode=='lim':
                    print('Computing NH error at 1 sigma')

                    # computing the error on the column density of the absorber
                    err_1sig = calc_error(param=str(comp_par[1]), logfile=logfile,
                                          delchi_err=1., give_errors='bounds',
                                          timeout=to_error, indiv=False)

                    err_1sig_bounds = err_1sig[0][comp_par[1] - 1]
                    err_1sig_full = np.repeat(0., 2)
                    if err_1sig_bounds[0] == 0.:
                        err_1sig_full[0] = AllModels(1)(comp_par[1]).values[2]
                    else:
                        err_1sig_full[0] = err_1sig_bounds[0]
                    if err_1sig_bounds[1] == 0.:
                        err_1sig_full[1] = AllModels(1)(comp_par[1]).values[5]
                    else:
                        err_1sig_full[1] = err_1sig_bounds[1]

                    # DOESNT WORK VERY WELL
                    # err_1sig_rel = err_1sig[0][comp_par[1]]
                    #
                    # #storing no error if the value is unconstrained
                    # # (for that we're testing if it's close to the main value)
                    # err_1sig_full = np.array([-err_1sig_rel[0],err_1sig_rel[1]]) + AllModels(1)(comp_par[1]+1).values[0]

                    # #safeguards to correctly put the info for pegged values
                    # if (err_1sig_rel/AllModels(1)(comp_par[1]+1).values[0])[0]>0.9999 and \
                    #         (err_1sig_rel / AllModels(1)(comp_par[1] + 1).values[0])[0]<1.0001:
                    #     err_1sig_full[0]=AllModels(1)(comp_par[1] + 1).values[2]
                    #
                    # if (err_1sig_rel/AllModels(1)(comp_par[1]+1).values[0])[1]>0.9999 and \
                    #         (err_1sig_rel / AllModels(1)(comp_par[1] + 1).values[0])[1]<1.0001:
                    #     err_1sig_full[1]=AllModels(1)(comp_par[1] + 1).values[5]

                    nh_val_distrib[0][i_iter] = err_1sig_full[1]

                    print('Computing NH error at 2 sigma')
                    # computing the blueshift error of the line
                    err_2sig = calc_error(param=str(comp_par[1]), logfile=logfile,
                                          delchi_err=4., give_errors='bounds',
                                          timeout=to_error, indiv=False)

                    err_2sig_bounds = err_2sig[0][comp_par[1] - 1]
                    err_2sig_full = np.repeat(0., 2)
                    if err_2sig_bounds[0] == 0.:
                        err_2sig_full[0] = AllModels(1)(comp_par[1]).values[2]
                    else:
                        err_2sig_full[0] = err_2sig_bounds[0]
                    if err_2sig_bounds[1] == 0.:
                        err_2sig_full[1] = AllModels(1)(comp_par[1]).values[5]
                    else:
                        err_2sig_full[1] = err_2sig_bounds[1]

                    # err_2sig_rel = err_2sig[0][comp_par[1]]
                    #
                    # #storing no error if the value is unconstrained
                    # # (for that we're testing if it's close to the main value)
                    # err_2sig_full =np.array([-err_2sig_rel[0],err_2sig_rel[1]]) + AllModels(1)(comp_par[1]+1).values[0]
                    #
                    # #safeguards to correctly put the info for pegged values
                    # if (err_2sig_rel/AllModels(1)(comp_par[1]+1).values[0])[0]>0.99 and \
                    #         (err_2sig_rel / AllModels(1)(comp_par[1] + 1).values[0])[0]<1.01:
                    #     err_2sig_full[0]=AllModels(1)(comp_par[1] + 1).values[2]
                    #
                    # if (err_2sig_rel/AllModels(1)(comp_par[1]+1).values[0])[1]>0.99 and \
                    #         (err_2sig_rel / AllModels(1)(comp_par[1] + 1).values[0])[1]<1.01:
                    #     err_2sig_full[1]=AllModels(1)(comp_par[1] + 1).values[5]

                    nh_val_distrib[1][i_iter] = err_2sig_full[1]

                    print('Computing NH error at 3 sigma')
                    # computing the blueshift error of the line
                    err_3sig = calc_error(param=str(comp_par[1]), logfile=logfile,
                                          delchi_err=9., give_errors='bounds',
                                          timeout=to_error, indiv=False)

                    err_3sig_bounds = err_3sig[0][comp_par[1] - 1]
                    err_3sig_full = np.repeat(0., 2)
                    if err_3sig_bounds[0] == 0.:
                        err_3sig_full[0] = AllModels(1)(comp_par[1]).values[2]
                    else:
                        err_3sig_full[0] = err_3sig_bounds[0]
                    if err_3sig_bounds[1] == 0.:
                        err_3sig_full[1] = AllModels(1)(comp_par[1]).values[5]
                    else:
                        err_3sig_full[1] = err_3sig_bounds[1]
                    # err_3sig_rel = err_3sig[0][comp_par[1]]
                    #
                    # #storing no error if the value is unconstrained
                    # # (for that we're testing if it's close to the main value)
                    # err_3sig_full = np.array([-err_3sig_rel[0],err_3sig_rel[1]]) + AllModels(1)(comp_par[1]+1).values[0]
                    #
                    # #safeguards to correctly put the info for pegged values
                    # if (err_3sig_rel/AllModels(1)(comp_par[1]+1).values[0])[0]>0.9999 and \
                    #         (err_3sig_rel / AllModels(1)(comp_par[1] + 1).values[0])[0]<1.0001:
                    #     err_3sig_full[0]=AllModels(1)(comp_par[1] + 1).values[2]
                    #
                    # if (err_3sig_rel/AllModels(1)(comp_par[1]+1).values[0])[1]>0.9999 and \
                    #         (err_3sig_rel / AllModels(1)(comp_par[1] + 1).values[0])[1]<1.0001:
                    #     err_3sig_full[1]=AllModels(1)(comp_par[1] + 1).values[5]

                    nh_val_distrib[2][i_iter] = err_3sig_full[1]

                if nh_photo_mode=='noise':

                    assert photo_mod == 'pion_abs_canon_soft'

                    #freezing requested parameters
                    for par_conti in par_freeze_steppar:
                        par_conti_now=par_conti if par_conti<min(comp_par) else par_conti+len(comp_par)
                        AllModels(1)(par_conti_now).frozen=True

                    #setting up parallel computation
                    Fit.parallel.steppar=os.cpu_count()-n_cores if n_cores<0 else n_cores


                    #removing the logs for the steppar run to avoid issues
                    curr_chatter=Xset.chatter
                    curr_logChatter=Xset.logChatter
                    Xset.chatter=0
                    Xset.logChatter=0

                    #for now, we assume whether there is a density parameters from the number of parameters
                    #in the table. If 4, then there should be no density. If 5, then there should be a density
                    id_delta_photomod=len(comp_par)==4

                    #steppar command. we remove 1 for the number of steps in each dimension because steppar
                    #adds 1 step by default.
                    Fit.steppar(
                    'log '+str(comp_par[3-id_delta_photomod])
                           +' '+str(AllModels(1)(comp_par[3-id_delta_photomod]).values[2])
                           +' '+str(AllModels(1)(comp_par[3-id_delta_photomod]).values[-1])
                           +' '+str(photo_nsteppar_turb-1)
                    +'nolog ' + str(comp_par[4-id_delta_photomod])
                    + ' ' + str(AllModels(1)(comp_par[4 - id_delta_photomod]).values[2])
                    + ' ' + str(AllModels(1)(comp_par[4 - id_delta_photomod]).values[-1])
                    + ' ' + str(photo_nsteppar_v-1)
                    )

                    Xset.chatter=Xset.chatter
                    Xset.logChatter=Xset.logChatter

                    #storing the maximum value found for NH within the parameter space of the steppar
                    nh_val_distrib[i_iter]=max(Fit.stepparResults(comp_par[1]))

                    #storing in the save file

                    file_edit(elem_flux_save_path,str(instance_id),str(instance_id)+'\t'+nh_val_distrib[i_iter],
                              header=save_header)

                pbar.update()

            #at the end of the n_iter computations for a single flux point

            if nh_photo_mode=='lim':
                nh_val_distrib = np.array(nh_val_distrib, dtype=float)

                nh_val_distrib.sort()

                # storing the median of the distribution of the limits for this flux value
                nh_lim_arr[i_flux] = nh_val_distrib.T[n_iter // 2]

            if nh_photo_mode=='noise':

                #reloading the full values array from the save file in case the computation isn't full
                nh_val_distrib=np.loadtxt(elem_flux_save_path)
                nh_val_distrib.sort()
                nh_lim_arr[i_flux]=[nh_val_distrib[n_iter*0.68],
                                    nh_val_distrib[n_iter*0.95],
                                    nh_val_distrib[n_iter*0.997]]

    save_arr = np.concatenate((np.array([flux_inter]), nh_lim_arr.T)).T

    header_elems = ['mod_path ' + str(mod_path),
                    'rmf_path ' + str(rmf_path_use),
                    'arf_path ' + str(arf_path_use),
                    'bkg_path ' + str(bkg_path_use),
                    'expos ' + str(expos) + ' ks',
                    'Fake stats ' + str(fakestats),
                    'n_iter ' + str(n_iter),
                    'flux_range logspace(' + flux_range + ') (e-10 erg/s/cm²) in ' + str(
                        flux_band.replace('.', 'p').replace(' ', '_')) + '_keV',
                    'photo mod ' + photo_mod,
                    'photo xi range' + photo_xi_range,
                    'photo turb range' + str(photo_turb_range),
                    'photo v range' + str(photo_v_range),
                    'n cores'+str(n_cores),
                    'photo nsteppar turb'+str(photo_nsteppar_turb),
                    'photo nsteppar v'+str(photo_nsteppar_v),
                    'par freeze steppar'+str(par_freeze_steppar)+
                    'columns: flux | '
                    +('nh median limit at 1/2/3 sigma (1e22 cm^{-2})' if nh_photo_mode=='lim' else
                      '1/2/3 photon noise sigma max nh fitted value from seppar runs')]

    np.savetxt('photo_nh_'+nh_photo_mode+'_mod_' + mod_path[:mod_path.rfind('.')] +
               ('_regroup' if regroup else '') +
               ('_nostat' if not fakestats else '') +
               '_' + str(expos) + 'ks' +
               '_' + str(n_iter) + '_iter' +
               '_flux_' + flux_range +
               '_in_' + str(flux_band.replace('.', 'p').replace(' ', '_')) + '_keV' +
               '_mod_' + str(photo_mod) +
               '.txt',
               save_arr, header='\n'.join(header_elems))

    return save_arr
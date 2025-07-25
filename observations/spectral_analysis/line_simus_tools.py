import os,sys
import numpy as np

import time
import pexpect
import matplotlib.pyplot as plt
from tqdm import tqdm
from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,Chain
from fitting_tools import sign_sigmas_delchi_1dof
#custom script with a few shorter xspec commands
from xspec_config_multisp import allmodel_data,model_load,addcomp,Pset,Pnull,rescale,reset,Plot_screen,store_plot,freeze,allfreeze,unfreeze,\
                         calc_error,delcomp,fitmod,calc_fit,xcolors_grp,xPlot,xscorpeon,catch_model_str,\
                         load_fitmod, ignore_data_indiv,par_degroup,xspec_globcomps,is_abs,lines_e_dict,calc_EW,set_ener
reset()
Fit.query='yes'
Plot.xLog=False

import getpass
username=getpass.getuser()


rmf_abv = {'XRISM_Hp_AO2': '/media/'+username+'/crucial_SSD/Observ/BHLMXB/XRISM/Simu/rsl_Hp_L_2025.rmf',
            'XRISM_Hp_AO1': '/media/'+username+'/crucial_SSD/Observ/highres/XRISM_responses/rsl_Hp_5eV.rmf',
           'XRISM_Mp': '/media/'+username+'/crucial_SSD/Observ/highres/XRISM_responses/rsl_Mp_6eV.rmf',
           'XRISM_Lp': '/media/'+username+'/crucial_SSD/Observ/highres/XRISM_responses/rsl_Lp_18eV.rmf',
        'PN_Timing': '/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/resp_XMM/Timing/0670672901_pn_S003_Timing_auto.rmf',
        'heg_graded_-1':'/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/resp_Chandra/graded/13716_heg_-1.rmf',
         'heg_graded_1':'/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/resp_Chandra/graded/13716_heg_1.rmf'}

rmf_abv_list = list(rmf_abv.keys())

arf_abv = {'XRISM_pointsource_GVclosed_AO2':'/media/'+username+'/crucial_SSD/Observ/BHLMXB/XRISM/Simu/rsl_pntsrc_GVC_2025.arf',
           'XRISM_pointsource_GVclosed_AO1': '/media/'+username+'/crucial_SSD/Observ/highres/XRISM_responses/rsl_pointsource_GVclosed.arf',
           'XRISM_pointsource_off_GVclosed': '/media/'+username+'/crucial_SSD/Observ/highres/XRISM_responses/rsl_pointsource_off_GVclosed.arf',
           'PN_Timing': '/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/resp_XMM/Timing/0670672901_pn_S003_Timing_auto.arf',
           'heg_graded_-1': '/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/resp_Chandra/graded/13716_heg_-1.arf',
           'heg_graded_1': '/media/'+username+'/crucial_SSD/Observ/highres/linedet_compa/AO2/resp_Chandra/graded/13716_heg_1.arf'}


arf_abv_list = list(arf_abv.keys())


#line_simu('test_SED.xcm',mode='ew_lim',rmf_path='XRISM_Hp_AO2',
#                          arf_path='XRISM_pointsource_GVclosed_AO2',expos=50,
#                           line='FeKa26abs',
#                          chatter=0,n_iter=10)

#line_simu('test_SED.xcm',mode='bshift_err',rmf_path='XRISM_Hp_AO2',arf_path='XRISM_pointsource_GVclosed',chatter=0,n_iter=10)

#line_simu('test_SED.xcm',mode='bshift_err',rmf_path='PN_Timing',arf_path='PN_Timing',chatter=0,n_iter=100)
#line_simu('test_SED.xcm',mode='bshift_err',rmf_path=['heg_graded_-1','heg_graded_1'],arf_path=['heg_graded_-1','heg_graded_1'],
# ,n_iter=100)


def line_simu(outdir='./',mod_path=None,mode='ew_lim',rmf_path='XRISM_Hp_AO2',arf_path='XRISM_pointsource_GVclosed_AO2',
              expos=50,flux_range='1_100_10',chatter=1,
              regroup=False,fakestats=True,n_iter=10,set_ener_data=False,
              line='FeKa26abs',line_v=[-3000,3000],line_w=[0.005,0.005],
              width_test_val=0.005,width_EW_resol=0.05,width_EW_inter=[0.1,100],
              EW_bshift_lim=20,width_bshift_val=0.005,sampl_gabs_string='-3_2_500'):

    '''
    REGROUP IS OUTDATED

    Computes simulations of line detection with given rmf and paths

    Note that

    arguments:

        mode:
            -ew_lim: computes the ew limits at 1, 2, 3 sigma by fitting an additional absorption line in the spectrum
            -bshift_err: computes the bshift errors with the given response files for a series of luminosity
                        for a given line, with a given EW and width
            -width_lim, computes the
        -mod_path:xspec_mod_path. if None, uses the currently loaded model instead

        -rmf_path/arf_path: rmf/arf to use for the fakes. must be an absolute path. available shortcuts:
            rmf_path:
                'XRISM_Hp_AO2': rsl_Hp_5eV.rmf from AO2
                'XRISM_Hp_AO1': rsl_Hp_5eV.rmf from AO1
                'XRISM_Mp': rsl_Mp_6eV.rmf
                'XRISM_Lp': rsl_Lp_18eV.rmf
                'PN_Timing': 0670672901_pn_S003_Timing_auto.rmf
                'heg_graded_-1': 13716_heg_-1.rmf
                'heg_graded_1': 13716_heg_1.rmf

            arf_path:

                'XRISM_pointsource_GVclosed_AO2'        rsl_pointsource_GVclosed.arf from AO2
                'XRISM_pointsource_GVclosed_AO1'        rsl_pointsource_GVclosed.arf from AO1

                'XRISM_pointsource_off_GVclosed'    rsl_pointsource_off_GVclosed.arf
                'PN_Timing'                         0670672901_pn_S003_Timing_auto.arf
                'heg_graded_-1': 13716_heg_-1.arf
                'heg_graded_1': 13716_heg_1.arf

        Notes that paths that are too long may result in issues

        -expos: exposure value in kiloseconds

        -flux_range:
            flux value interval to be parsed. The spectrum will be renormalized to have its values in the
            3-10keV band
            the interval is low_lim_high_lim_nsteps in log space, low and highlim in units of 1e-10 erg/cm²/s

        -regroup: Regroup or not the spectra with ftgrouppha before analysis

        -fakestats: Use the applyStats options of fakeit to consider statistical fluctuations when faking
                    (otherwise a "perfect" spectrum with 0 chi² gets created)

        -niter: number of iterations to test for each flux

        -line: name of the line to test (mainly for the energy)

        -line_v/line_w: velocity and width range for the line.
                        If the start and end values are the same, freezes the parameters.
                        NOTE: currently there are issues when not freezing, would require a steppar
                        to explore the parameter ranges and avoid the fit getting stuck

        EW_lim mode:
            -line_v/line_w: min/max velocity and width intervals to allow when fitting the line

        width_lim mode:
            -the line is taken at 0 velocity and fitted

            -width_test_val: test width to witch to fetch the lowest EW in keV

            -width_EW_interval: EW interval in which to test for the lines

            -width_EW_resol: resolution for when to stop when trying to find the limit
                             for computing the width of the line, in RELATIVE units (so default value is 5% error)

        bshfit_err mode (for now only for absorption lines)
            -puts a gabs at a 0 velocity to avoid issues with unrealistic shapes for low widths
            -computes a single time the strength that gives the desired EW (EW_bshift_lim)
            -line created with width_bshift_lim
            -fits the line after creation, computes the 1/2/3 sigma errors on the blueshift

            sampl_gabs_string: np.logspace parameters for sampling the gabs strength to approximate the EW
    #TODO: add arf/rmf cutting
    #TODO: add low width_EW_interval threshold

    '''

    old_chatter=Xset.chatter
    Xset.chatter=chatter

    AllData.clear()

    if outdir is not None:
        os.system('mkdir -p '+outdir)
    if mod_path is not None:
        os.system('cp '+mod_path+' '+outdir)

    os.chdir(outdir)

    #here we assume that the elements are the same size

    rmf_path_use=[]
    arf_path_use=[]

    if type(rmf_path) in [list,np.ndarray,tuple]:
        rmf_path_list=rmf_path
        arf_path_list=arf_path
    else:
        rmf_path_list=[rmf_path]
        arf_path_list=[arf_path]

    for elem_rmf, elem_arf in zip(rmf_path_list,arf_path_list):
        if elem_rmf in rmf_abv_list:
            rmf_path_use+=[rmf_abv[elem_rmf]]
        else:
            rmf_path_use+=[elem_rmf]

        if elem_arf in arf_abv_list:
            arf_path_use+=[arf_abv[elem_arf]]
        else:
            arf_path_use+=[elem_arf]

    logfile_write=Xset.openLog('line_simu'+mode+'.log')
    #ensuring the log information gets in the correct place in the log file by forcing line to line buffering
    logfile_write.reconfigure(line_buffering=True)

    logfile=open(logfile_write.name,'r')

    if mod_path is not None:
        AllModels.clear()
        Xset.restore(mod_path)

    Plot.xLog=False
    Fit.statMethod = 'cstat'

    #computing the 3-10 flux
    AllModels.calcFlux('3. 10.')
    flux_base=AllModels(1).flux[0]

    flux_range_vals=np.array(flux_range.split('_')).astype(float)

    n_flux=int(flux_range_vals[2])
    flux_inter=np.logspace(np.log10(flux_range_vals[0]),np.log10(flux_range_vals[1]),n_flux)*1e-10

    #adding a constant from the flux value to renormalize
    addcomp('glob_constant')

    #needs to be frozen for a single datagroup
    AllModels(1)(1).frozen=True

    mod_cont=allmodel_data()

    fakeset = [FakeitSettings(response=elem_rmf,arf=elem_arf,exposure=expos*1000,
                              background='',
                              fileName='temp_sp_'+mode+'_'+str(i_grp+1)+'.pi') for i_grp,(elem_rmf,elem_arf) in\
                                enumerate(zip(rmf_path_use,arf_path_use))]

    #creating a bash process to run ftgrouppha on the fake spectra
    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')
    bashproc.sendline('heainit')

    line_E = lines_e_dict[line][0]

    if mode=='ew_lim':

        print('Computing EW limits for the given flux range...')
        ew_lim_arr=np.zeros((n_flux,3))

        with tqdm(total=n_flux*n_iter) as pbar:
            for i_flux,elem_flux in enumerate(flux_inter):

                ew_lim_distrib=np.repeat(None,3*n_iter).reshape(3,n_iter)

                for i_iter in range(n_iter):
                    mod_cont.load()

                    #freezing the parameters before faking
                    freeze()

                    AllModels(1)(1).values=elem_flux/flux_base

                    #remove previously computed spectra
                    if os.path.isfile('temp_sp.pi'):
                        os.remove('temp_sp.pi')

                    if os.path.isfile('temp_sp_grp_opt.pi'):
                        os.remove('temp_sp_grp_opt.pi')

                    #faking the spectrum with the right parameters
                    AllData.fakeit(nSpectra=len(fakeset), settings=fakeset, applyStats=fakestats)

                    #rebinning the spectrum before loading it
                    if regroup:
                        # using optsnmin puts some bins at weird wiggling ratios
                        # bashproc.sendline('ftgrouppha infile=temp_sp.pi'+' outfile=temp_sp_grp_opt.pi '+
                        #                   ' grouptype=optsnmin groupscale=3.0'+
                        #                   ' respfile='+rmf_path_use+' clobber=True')

                        # using opt puts some bins at 0 for some reason maybe bc the rmf has issues
                        group_str='ftgrouppha infile=temp_sp.pi outfile=temp_sp_grp_opt.pi '+\
                                          'grouptype=opt'+\
                                          'respfile='+rmf_path_use+' clobber=True'
                        bashproc.sendline(group_str)

                        #waiting for the spectrum to be created:
                        while not os.path.isfile('temp_sp_grp_opt.pi'):
                            time.sleep(1)

                        AllData.clear()
                        AllData('1:1 temp_sp_grp_opt.pi')

                    AllData.ignore('**-2. 10.-**')
                    AllData.ignore('bad')

                    if set_ener_data:
                        set_ener('thcomp', xrism=True)

                    for i_grp in range(1,AllData.nGroups+1):
                        AllData(i_grp).response.arf=arf_path_use[i_grp-1]


                    #loading the continuum model and fitting
                    mod_cont.load()
                    AllModels(1)(1).values=elem_flux/flux_base

                    calc_fit()

                    if Fit.statistic/Fit.dof>2:
                        print('Issue with fake continuum fitting.')
                        breakpoint()
                        pass

                    #freezing the continuum
                    freeze()

                    XRISM_sp=AllData(1).fileinfo('TELESCOP')=='XRISM'
                    if XRISM_sp:
                        #ignoring everything outside of the line energy to avoid issues because of too many bins
                        AllData.ignore('**-'+str(line_E-1)+' '+str(line_E+1)+'-**')

                    #adding the line component
                    comp_par,comp_num=addcomp(line+'_agaussian', position='lastinall',return_pos=True)

                    #with appropriate parameter range (and freezing if its a single value
                    AllModels(1)(comp_par[0]).values=[0.,0. if line_v[0]==line_v[1] else 10.,
                                                      line_v[0],line_v[0],line_v[1],line_v[1]]
                    AllModels(1)(comp_par[2]).values = [line_w[0], 0. if line_w[0]==line_w[1] else 0.01,
                                                        line_w[0], line_w[0], line_w[1], line_w[1]]


                    #fitting
                    calc_fit()

                    Fit.query='yes'
                    #computing the error on the velocity shift parameter of the line to ensure we are not stuck
                    calc_error(logfile,param=str(comp_par[0]),timeout=15,freeze_pegged=True)
                    calc_fit()
                    Fit.query='on'

                    #to be implemented if need be for parameter variations
                    # (we change the critical delta to avoid pyxspec getting stuck if the steppar finds a better fit)
                    # #storing the current fit delta
                    # curr_crit_delta=Fit.criticalDelta
                    #
                    # Fit.criticalDelta=1e10
                    #
                    # #doing a steppar on the velocity and the width
                    # Fit.steppar(str(comp_par[0])+' '+str(line_v[0])+' '+str(line_v[1])+'50 8 -1e-7 -1e-5 10')

                    #Fit.criticalDelta=curr_crit_delta

                    #and computing the eqwidth of the best fit in the interval
                    try:
                        AllModels.eqwidth(int(comp_num[-1]), err=True, number=1000, level=68.3)
                    except:
                        #this means both of the parameters are unconstrained and pegged
                        #thus here we fix the blueshift and rerun the eqwidth test
                        AllModels(1)(comp_par[0]).frozen=True
                        calc_fit()
                        AllModels.eqwidth(int(comp_num[-1]), err=True, number=1000, level=68.3)

                    ew_lim_distrib[0][i_iter]=-AllData(1).eqwidth[1]*1e3

                    AllModels.eqwidth(int(comp_num[-1]), err=True, number=1000, level=95)

                    ew_lim_distrib[1][i_iter]=-AllData(1).eqwidth[1]*1e3

                    AllModels.eqwidth(int(comp_num[-1]), err=True, number=1000, level=99.7)

                    ew_lim_distrib[2][i_iter]=-AllData(1).eqwidth[1]*1e3

                    pbar.update()

                ew_lim_distrib.sort()

                #storing the median of the distribution of the limits for this flux value
                ew_lim_arr[i_flux]=ew_lim_distrib.T[n_iter//2]

        save_arr=np.concatenate((np.array([flux_inter]),ew_lim_arr.T)).T

        header_elems=['mod_path '+str(mod_path),
                      'rmf_path '+str(rmf_path_use),'arf_path '+str(arf_path_use),
                      'expos '+str(expos)+' ks',
                      'Fake stats '+str(fakestats),
                      'n_iter '+str(n_iter),
                      'flux_range logspace('+flux_range+') (e-10 erg/s/cm²) ',
                      'line '+line,'line_v '+str(line_v)+' (km/s)','line_w '+str(line_w)+' (keV)',
                      'columns: flux | ew_limit at 1/2/3 sigma (eV)']

        np.savetxt('ew_lim_mod'+
                   ('_regroup' if regroup else '')+
                   ('_nostat' if not fakestats else '')+
                    '_'+str(expos)+'ks'+
                   '_'+str(n_iter)+'_iter'+
                   '_line_' + str(line) +
                   '_flux_' + flux_range +
                   '_width_'+str(line_w[0])+
                   '_'+str(line_w[1])+'.txt',save_arr,header='\n'.join(header_elems))

        Xset.chatter=old_chatter

        return save_arr

    if mode=='bshift_err':

        #n_iter to be implemented here

        print('Computing bshift errors for the given flux range...')
        bshift_err_arr=np.zeros((n_flux,3))

        '''
        The method is:
        1. Compute the strength of the gabs that will give the right EW \
         (will be independant of the continuum flux so can be used for all flux values)
        2. Loop on the flux values
        3. For each flux value, a given number of times (to average the fluctuations),
           fake the model with the right constant factor 
        4. Fit a continuum, fix it, add a gabs, fit it, compute the errors of the blueshift
        6. Store the median of the shift distribution for each flux values
        '''
        # reloading the continuum model
        mod_cont.load()

        # computing where to put the test line
        line_pos = 1 + sum([is_abs(comp) for comp in AllModels(1).componentNames]) \
                   + sum([comp in xspec_globcomps for comp in AllModels(1).componentNames])

        print('Will add the gabs line in position ' + str(line_pos))

        # adding the test line
        comp_par, comp_num = addcomp('gabs', position=line_pos, endmult=AllModels(1).componentNames[-1],return_pos=True)

        line_E = lines_e_dict[line][0]
        AllModels(1)(comp_par[0]).values = line_E

        # putting an even blueshift range
        max_bshift = max(abs(lines_e_dict[line][1]), lines_e_dict[line][2])
        eshift_range = [line_E * (1 - max_bshift / 299792.458), line_E * (1 + max_bshift / 299792.458)]

        # fixing the energy
        AllModels(1)(comp_par[0]).values = [line_E, line_E / 1000, eshift_range[0], eshift_range[0],
                                            eshift_range[1], eshift_range[1]]

        # and the width at the desired value
        AllModels(1)(comp_par[-2]).values = width_bshift_val

        # AllModels(1)(comp_par[2]).frozen=True

        # faking a spectrum to get the model values (to compute the EW because xspec is garbage
        AllData.fakeit(noWrite=True)

        # making a range of energies within 10 sigma of the line
        AllModels.setEnergies(str(line_E-10*width_bshift_val)+' '+str(line_E+10*width_bshift_val)+' 1000 lin')

        e_vals=AllModels(1).energies(1)

        e_vals_med=[(e_vals[i+1]+e_vals[i])/2 for i in range(len(e_vals)-1)]

        mod_line_list=[]

        n_vals_sampl_strength=100

        #sample of gaussian strength
        sampl_gabs_str_vals=np.array(sampl_gabs_string.split('_')).astype(float)
        sampl_gabs_strength=np.logspace(sampl_gabs_str_vals[0],sampl_gabs_str_vals[1],int(sampl_gabs_str_vals[2]))

        for elem_str in sampl_gabs_strength:
            AllModels(1)(comp_par[-1]).values=elem_str
            mod_line_list+=[AllModels(1).values(1)]

        mod_gabs_temp=allmodel_data()

        delcomp('gabs')

        mod_cont_vals=AllModels(1).values(1)

        indiv_EW=-np.array([calc_EW(e_vals_med,mod_cont_vals,mod_line_list[id]) for id in range(len(mod_line_list))])\
                 *1000

        min_delta_EW=min(abs(indiv_EW-EW_bshift_lim))

        assert min_delta_EW-EW_bshift_lim<1, 'Unsufficient gabs strength sampling'

        gabs_str_EW_select=sampl_gabs_strength[abs(indiv_EW-EW_bshift_lim).argmin()]

        print('Selecting gabs strength for an EW of %.3e'%(indiv_EW[abs(indiv_EW-EW_bshift_lim).argmin()])+' eV')
        #selecting the right value

        mod_gabs_temp.load()

        AllModels(1)(comp_par[-1]).values = gabs_str_EW_select

        freeze()

        #for checking purposes
        # plt.figure()
        # [plt.plot(mod_line_list[i]) for i in range(len(mod_line_list))]
        # plt.plot(mod_cont,ls='--')

        mod_gabs=allmodel_data()

        #resetting the energy range
        AllModels.setEnergies('reset')

        with tqdm(total=n_flux*n_iter) as pbar:
            for i_flux,elem_flux in enumerate(flux_inter):

                bshift_err_distrib=np.repeat(None,3*n_iter).reshape(3,n_iter)

                for i_iter in range(n_iter):

                    AllData.clear()

                    mod_gabs.load()

                    # adjusting the luminosity
                    AllModels(1)(1).values = elem_flux / flux_base

                    freeze()

                    #faking the spectrum with the right parameters
                    AllData.fakeit(nSpectra=len(fakeset), settings=fakeset, applyStats=fakestats)

                    AllData.ignore('**-2. 10.-**')

                    AllData.ignore('bad')

                    for i_grp in range(1, AllData.nGroups + 1):
                        AllData(i_grp).response.arf = arf_path_use[i_grp - 1]

                    # loading the continuum model and fitting
                    mod_cont.load()
                    AllModels(1)(1).values = elem_flux / flux_base

                    calc_fit()

                    # freezing the continuum
                    freeze()

                    XRISM_sp=AllData(1).fileinfo('TELESCOP')=='XRISM'
                    if XRISM_sp:
                        #ignoring everything outside of the line energy to avoid issues because of too many bins
                        AllData.ignore('**-'+str(line_E-1)+' '+str(line_E+1)+'-**')

                    # adding the test line
                    comp_par, comp_num = addcomp('gabs', position=line_pos,
                                                 endmult=AllModels(1).componentNames[-1],return_pos=True)

                    line_E = lines_e_dict[line][0]
                    AllModels(1)(comp_par[0]).values = line_E

                    # putting an even blueshift range
                    max_bshift = max(abs(lines_e_dict[line][1]), lines_e_dict[line][2])
                    eshift_range = [line_E * (1 - max_bshift / 299792.458), line_E * (1 + max_bshift / 299792.458)]

                    # giving the energy value
                    AllModels(1)(comp_par[0]).values = [line_E, line_E / 1000, eshift_range[0], eshift_range[0],
                                                        eshift_range[1], eshift_range[1]]

                    # and the width at the desired value
                    AllModels(1)(comp_par[2]).values = width_bshift_val

                    #fitting a first time
                    calc_fit()

                    #computing the error on the blueshift just to ensure we get the right value
                    calc_error(logfile,param=str(comp_par[0]))

                    # #trying to converge on the correct luminosity if the fit if it gets stuck
                    # if XRISM_sp:
                    #     #steppar on the width because that is the difficult parameter to compute
                    #     Fit.steppar('log '+str(comp_par[2])
                    #                 + ' ' + str(width_bshift_val/10)
                    #                 + ' ' + str(min(width_bshift_val*10,max_width_range)) + ' 20')
                    # else:
                    #     Fit.steppar(str(comp_par[0])+' '+str(bshift_min)+' '+str(bshift_max)+' 100')

                    #with XRISM, computing the errors while letting the fit improve is too long,
                    #so we block the fit improvement until the end of the error computation
                    if XRISM_sp:
                        curr_delta=Fit.criticalDelta
                        Fit.criticalDelta=1e9
                        Fit.query='no'

                    # #the first error computation will also help find the best fit
                    # print('Computing bshift error at 3 sigma')
                    #
                    # # computing the blueshift error of the line
                    # err_3sig = calc_error(param=str(comp_par[0]), logfile=logfile, delchi_err=9., give_errors=True,
                    #                       timeout=30 if XRISM_sp else 5,indiv=False)

                    if Fit.statistic / Fit.dof > 2:
                        print('Issue with fake continuum fitting.')
                        # breakpoint()
                        pass

                    # err_3sig_rel = err_3sig[0][comp_par[0]-1]

                    # if max(err_3sig_rel)==0:
                    #     bshift_err_distrib[2][i_iter]=0
                    # else:
                    #     # adding the main value because it can be false too
                    #     err_3sig_full = err_3sig_rel + AllModels(1)(comp_par[0]).values[0]
                    #
                    #     bshift_err_distrib[2][i_iter] = max(abs(err_3sig_full))

                    print('Computing bshift error at 1 sigma')


                    #computing the blueshift error of the line
                    err_1sig = calc_error(param=str(comp_par[0]), logfile=logfile,
                                          delchi_err=1., give_errors=True,
                                          timeout=30 if XRISM_sp else 5,indiv=False)

                    err_1sig_rel = err_1sig[0][comp_par[0]-1]

                    #storing no error if the value is unconstrained
                    # (for that we're testing if it's close to the main value)
                    if (abs(err_1sig_rel/AllModels(1).gabs.LineE.values[0])>0.9).any():
                        bshift_err_distrib[0][i_iter] = 0
                    else:
                        # adding the main value because it can be false too
                        err_1sig_full = err_1sig_rel + AllModels(1)(comp_par[0]).values[0]

                        bshift_err_distrib[0][i_iter] = max(abs(err_1sig_full-line_E))/line_E*299792.458

                    print('Computing bshift error at 2 sigma')

                    # computing the blueshift error of the line
                    err_2sig= calc_error(param=str(comp_par[0]), logfile=logfile,
                                         delchi_err=4., give_errors=True,
                                          timeout=30 if XRISM_sp else 5,indiv=False)

                    err_2sig_rel=err_2sig[0][comp_par[0]-1]

                    if (abs(err_2sig_rel/AllModels(1).gabs.LineE.values[0])>0.9).any():
                        bshift_err_distrib[1][i_iter] = 0
                    else:
                        # adding the main value because it can be false too
                        err_2sig_full = err_2sig_rel + AllModels(1)(comp_par[0]).values[0]

                        bshift_err_distrib[1][i_iter] = max(abs(err_2sig_full-line_E))/line_E*299792.458

                    #putting back the fit delta to its right value
                    if XRISM_sp:
                        Fit.criticalDelta=curr_delta
                        Fit.query='yes'

                    print('Computing bshift error at 3 sigma')

                    #computing the blueshift error of the line
                    err_3sig = calc_error(param=str(comp_par[0]), logfile=logfile,
                                          delchi_err=9., give_errors=True,
                                          timeout=30 if XRISM_sp else 5,indiv=False)

                    err_3sig_rel = err_3sig[0][comp_par[0]-1]

                    #storing no error if the value is unconstrained
                    if (abs(err_3sig_rel/AllModels(1).gabs.LineE.values[0])>0.9).any():
                        bshift_err_distrib[2][i_iter] = 0
                    else:
                        # adding the main value because it can be false too
                        err_3sig_full = err_3sig_rel + AllModels(1)(comp_par[0]).values[0]

                        bshift_err_distrib[2][i_iter] = max(abs(err_3sig_full-line_E))/line_E*299792.458

                    # if (bshift_err_distrib.T[i_iter]==0).any():
                    #     breakpoint()

                    pbar.update()

                bshift_err_distrib.sort()

                bshift_err_distrib_1sig_constr=bshift_err_distrib[0][bshift_err_distrib[0].nonzero()]
                bshift_err_distrib_2sig_constr = bshift_err_distrib[1][bshift_err_distrib[1].nonzero()]
                bshift_err_distrib_3sig_constr = bshift_err_distrib[2][bshift_err_distrib[2].nonzero()]

                #storing the median of the distribution of the limits for this flux value
                bshift_err_arr[i_flux][0]= np.median(bshift_err_distrib_1sig_constr)
                bshift_err_arr[i_flux][1]= np.median(bshift_err_distrib_2sig_constr)
                bshift_err_arr[i_flux][2]= np.median(bshift_err_distrib_3sig_constr)

                np.savetxt('bshift_err_arr.txt', bshift_err_arr)

        save_arr=np.concatenate((np.array([flux_inter]),bshift_err_arr.T)).T

        header_elems=['mod_path '+str(mod_path),
                      'rmf_path '+str(rmf_path_use),'arf_path '+str(arf_path_use),
                      'expos '+str(expos)+' ks',
                      'Fake stats '+str(fakestats),
                      'n_iter '+str(n_iter),
                      'flux_range logspace('+flux_range+') (e-10 erg/s/cm²) ',
                      'line '+line,'line_EW '+str(EW_bshift_lim)+' (eV)','line_w '+str(width_bshift_val)+' (keV)',
                      'columns: flux | bshift_limit at 1/2/3 sigma (eV)']

        np.savetxt('bshift_err_mod'+
                   ('_regroup' if regroup else '')+
                   ('_nostat' if not fakestats else '')+
                   '_' + str(expos) + 'ks' +
                   '_'+str(n_iter)+'_iter'+
                   '_flux_'+flux_range+
                   '_line_'+str(line)+
                   '_EW_'+str(EW_bshift_lim)+
                   '_width_'+str(width_bshift_val)+'.txt',save_arr,header='\n'.join(header_elems))

        Xset.chatter=old_chatter

        return save_arr

    if mode=='width_lim':

        #n_iter to be implemented here

        print('Computing width detectability for the given flux range...')
        width_lim_arr=np.zeros((n_flux,3))

        '''
        The method here is to parse the EW interval by multipling/diving by progressively lower factors until
        we find the right value (binary search). Here we do it in logspace because we don't know what to expect
        '''

        #starting at the max possible value allowed
        width_EW_init=width_EW_inter[1]

        #and starting with a middle step in log space
        base_step=(width_EW_inter[0]/width_EW_inter[1])**(1/2)

        with tqdm(total=n_flux) as pbar:
            for i_flux,elem_flux in enumerate(flux_inter):

                #reloading the continuum model
                mod_cont.load()

                #adjusting the luminosity
                AllModels(1)(1).values = elem_flux/flux_base

                #adding the test line
                comp_par, comp_num = addcomp(line + '_agaussian', position='lastinall')

                # freezing the blueshift at 0
                AllModels(1)(comp_par[0]).frozen = True

                # and the width at the desired value
                AllModels(1)(comp_par[2]).values = width_test_val

                AllModels(1)(comp_par[2]).frozen=True

                # faking a spectrum to get the eqwidth because xspec is garbage
                AllData.fakeit(noWrite=True)
                AllModels.eqwidth(int(comp_num[-1]))

                # computing the normalization factor (with a negative so we can keep our EW in positive values
                norm_EW_factor = -AllModels(1)(comp_par[-1]).values[0]/AllData(1).eqwidth[0]

                #storing the model
                freeze()
                mod_width_base=allmodel_data()

                #variable steps for each sigma
                step_1sig = base_step
                step_2sig = base_step
                step_3sig = base_step

                #final variable to assess whether a given sigma width has been stored for one flux value
                ok_constr_1sig=False
                ok_constr_2sig=False
                ok_constr_3sig=False

                EW_constr_1sig=[]
                EW_constr_2sig=[]
                EW_constr_3sig=[]

                EW_test_width=width_EW_init

                '''
                The loop is as follow:
                
                1. put the appropriate norm for EW_test
                2. Fake
                3. test if the width is constrained at 1/2/3 sigmas
                4. change EW test according to the 1 sigma constrain (2 if the 1sig is complete, 3 if the 2sig is)
                5. -if the current 'main' EW test gives the same result as the previous iteration, keep 
                    multiplying or dividing by the current step
                6. -otherwise, swap to multiply from division, divide the step by 2, and continue 
                
                widths not constrained for 100eV lines are considered unconstrained and store as such  
                    
                '''

                while (width_lim_arr[i_flux]==0).any():

                    #freezing the parameters before faking
                    freeze()

                    #chaning the normalization of the gaussian to match the EW width to test
                    AllModels(1)(comp_par[-1]).values=EW_test_width*norm_EW_factor

                    #remove previously computed spectra
                    if os.path.isfile('temp_sp.pi'):
                        os.remove('temp_sp.pi')

                    if os.path.isfile('temp_sp_grp_opt.pi'):
                        os.remove('temp_sp_grp_opt.pi')

                    #faking
                    AllData.fakeit(nSpectra=len(fakeset),settings=fakeset, applyStats=fakestats)

                    # #rebinning the spectrum before loading it
                    # bashproc.sendline('ftgrouppha infile=temp_sp.pi'+' outfile=temp_sp_grp_opt.pi grouptype= opt'+
                    #                   ' respfile='+rmf_path_use)
                    #
                    # #waiting for the spectrum to be created:
                    # while not os.path.isfile('temp_sp_grp_opt.pi'):
                    #     time.sleep(1)
                    #
                    # AllData.clear()
                    #
                    # AllData('1:1 temp_sp_grp_opt.pi')

                    AllData.ignore('**-2. 10.-**')
                    AllData.ignore('bad')

                    if set_ener_data:
                        set_ener('thcomp', xrism=True)

                    for i_grp in range(1, AllData.nGroups + 1):
                        AllData(i_grp).response.arf = arf_path_use[i_grp - 1]

                    #loading the cont version of the width model (to avoid issues with parameters)
                    mod_cont.load()

                    # adjusting the luminosity
                    AllModels(1)(1).values = flux_base / elem_flux

                    #fitting
                    calc_fit()

                    #testing whether the line width is constrained

                    # computing the width with the current fit at 1 sigma
                    Fit.error('stop ,,0.1 max 100 1. ' + str(comp_par[-2]))

                    '''
                    we consider the width constrained if the lower bound is more than 1% of the base value
                    (since here we know the base value), because things tend to peg at very low values instead
                    of saying thart they're frozen at 0
                    '''

                    if AllModels(1)(comp_par[-2]).error[0]<width_test_val/100:
                        EW_constr_1sig+=[-EW_test_width]
                        EW_constr_2sig+=[-EW_test_width]
                        EW_constr_3sig+=[-EW_test_width]
                    else:
                        EW_constr_1sig+=[EW_test_width]

                        #then at 2 sigma
                        Fit.error('stop ,,0.1 max 100 4. ' + str(comp_par[-2]))
                        if AllModels(1)(comp_par[-2]).error[0]<width_test_val/100:
                            EW_constr_2sig += [-EW_test_width]
                            EW_constr_3sig += [-EW_test_width]
                        else:
                            EW_constr_2sig += [EW_test_width]

                            # then at 3 sigma
                            Fit.error('stop ,,0.1 max 100 9. ' + str(comp_par[-2]))
                            if AllModels(1)(comp_par[-2]).error[0] < width_test_val / 100:
                                EW_constr_3sig += [-EW_test_width]
                            else:
                                EW_constr_3sig += [EW_test_width]

                    #limit tests if nothing is constrained at 100 eVs
                    if EW_constr_1sig[-1]==-width_EW_init:
                        width_lim_arr[i_flux][0]=abs(EW_constr_1sig)
                        ok_constr_1sig=True

                    if EW_constr_1sig[-1]==-width_EW_init:
                        width_lim_arr[i_flux][1] = abs(EW_constr_2sig)
                        ok_constr_2sig=True

                    if EW_constr_1sig[-1]==-width_EW_init:
                        width_lim_arr[i_flux][2] = abs(EW_constr_3sig)
                        ok_constr_3sig=True


                    if not ok_constr_1sig:
                        #testing for 1 sigma in priority

                        #going down for the first iteration after 100eVs
                        if len(EW_constr_1sig)==1:
                            EW_test_width*=step_1sig
                        else:
                            #since we store constrains and non-constrains width different signs,
                            #a change in constrain in the last two iterations will always be negative
                            constrain_change=EW_constr_1sig[-1]/EW_constr_1sig[-2]
                            constrain_fraction=abs((EW_constr_1sig[-1]-EW_constr_1sig[-2])/max(EW_constr_1sig[-2:]))

                            #if the constrain changed and the fraction delta is under the resolution, we're done
                            if constrain_change and constrain_fraction<width_EW_resol:
                                #note here that taking the max works both because we note non-constrain with negatives
                                #but also because the constrained one will always be higher
                                width_lim_arr[i_flux][0]=max(EW_constr_1sig[-2:])
                                ok_constr_1sig=True

                            else:
                                #taking the square root of the current step
                                # (to divide the remaining logarithmic interval by two)
                                step_1sig=step_1sig**(1/2)

                                if constrain_change:
                                    #changing the step direction
                                    step_1sig=1/step_1sig
                                    
                                EW_test_width*=step_1sig


                    if ok_constr_1sig and not ok_constr_2sig:
                        #testing for 2 sigma in priority

                        #going down for the first iteration after 100eVs
                        if len(EW_constr_2sig)==1:
                            EW_test_width*=step_2sig
                        else:
                            #checking if this is the first time we're moving according to this one
                            if step_2sig==base_step:
                                #computing the remaining log interval

                                #fetching the non constraints
                                EW_noconstr_2sig=np.array(EW_constr_2sig)[EW_constr_2sig < 0]

                                #and the constrains
                                EW_withconstr_2sig = np.array(EW_constr_2sig)[EW_constr_2sig > 0]
                                
                                #there should necessarily be non-constrains since we test for the limit
                                #of the least significant test first
                                if len(EW_noconstr_2sig)==0:
                                    breakpoint()
                                    print('This shouldnt be possible')

                                #putting the EW value on the constrained one
                                EW_test_width=min(EW_withconstr_2sig)
                                
                                #and computing the remaining half logspace factor
                                #min on the numerator here since they are in negative values
                                step_2sig=(-min(EW_noconstr_2sig)/\
                                          min(EW_withconstr_2sig))**(1/2)
                                                                
                            #otherwise we can proceed normally
                            else:
                                # since we store constrains and non-constrains width different signs,
                                # a change in constrain in the last two iterations will always be negative
                                constrain_change = EW_constr_2sig[-1] / EW_constr_2sig[-2]
                                constrain_fraction = abs(
                                    (EW_constr_2sig[-1] - EW_constr_2sig[-2]) / max(EW_constr_2sig[-2:]))

                                # if the constrain changed and the fraction delta is under the resolution, we're done
                                if constrain_change and constrain_fraction < width_EW_resol:
                                    # note here that taking the max works both because we note non-constrain with negatives
                                    # but also because the constrained one will always be higher
                                    width_lim_arr[i_flux][1] = max(EW_constr_2sig[-2:])
                                    ok_constr_2sig = True

                                else:
                                    # taking the square root of the current step
                                    # (to divide the remaining logarithmic interval by two)
                                    step_2sig = step_2sig ** (1 / 2)

                                    if constrain_change:
                                        # changing the step direction
                                        step_2sig = 1 / step_2sig

                                    EW_test_width*=step_2sig


                    if ok_constr_1sig and ok_constr_2sig and not ok_constr_3sig:
                        # testing for 3 sigma in priority

                        # going down for the first iteration after the first one
                        if len(EW_constr_3sig) == 1:
                            EW_test_width *= step_3sig
                        else:
                            # checking if this is the first time we're moving according to this one
                            if step_3sig == base_step:
                                # computing the remaining log interval

                                # fetching the non constraints
                                EW_noconstr_3sig = np.array(EW_constr_3sig)[EW_constr_3sig < 0]

                                # and the constrains
                                EW_withconstr_3sig = np.array(EW_constr_3sig)[EW_constr_3sig > 0]

                                # there should necessarily be non-constrains since we test for the limit
                                # of the least significant test first
                                if len(EW_noconstr_3sig) == 0:
                                    breakpoint()
                                    print('This shouldnt be possible')

                                # putting the EW value on the constrained one
                                EW_test_width = min(EW_withconstr_3sig)

                                # and computing the remaining half logspace factor
                                # min on the numerator here since they are in negative values
                                step_3sig = (-min(EW_noconstr_3sig) / \
                                             min(EW_withconstr_3sig)) ** (1 / 2)

                            # otherwise we can proceed normally
                            else:
                                # since we store constrains and non-constrains width different signs,
                                # a change in constrain in the last two iterations will always be negative
                                constrain_change = EW_constr_3sig[-1] / EW_constr_3sig[-2]
                                constrain_fraction = abs(
                                    (EW_constr_3sig[-1] - EW_constr_3sig[-2]) / max(EW_constr_3sig[-2:]))

                                # if the constrain changed and the fraction delta is under the resolution, we're done
                                if constrain_change and constrain_fraction < width_EW_resol:
                                    # note here that taking the max works both because we note non-constrain with negatives
                                    # but also because the constrained one will always be higher
                                    width_lim_arr[i_flux][2] = max(EW_constr_3sig[-2:])
                                    ok_constr_3sig = True

                                else:
                                    # taking the square root of the current step
                                    # (to divide the remaining logarithmic interval by two)
                                    step_3sig = step_3sig ** (1 / 2)

                                    if constrain_change:
                                        # changing the step direction
                                        step_3sig = 1 / step_3sig

                                    EW_test_width*=step_3sig


                pbar.update()

        save_arr=np.concatenate((np.array([flux_inter]),width_lim_arr.T)).T

        header_elems=['mod_path '+str(mod_path),'rmf_path '+str(rmf_path_use),'arf_path '+str(arf_path_use),
                      'expos '+str(expos)+' ks','Fake stats '+str(fakestats),
                      'flux_range logspace('+flux_range+') e-10 erg/s/cm² ',
                      'line '+line,
                      'tested width '+str(10**3*width_test_val)+' eVs',
                      'tested EW interval '+str(width_EW_inter),
                      'resolution fraction '+str(width_EW_resol),
                      'columns: flux ew_limit at 1/2/3 sigma']

        #to be updated
        np.savetxt('ew_lim_widthdet_mod'+('_regroup' if regroup else '')+('_nostat' if not fakestats else '')+
                   '_width_'+str(line_w[0])+'_'+str(line_w[1])+'.txt',save_arr,header='\n'.join(header_elems))
        return flux_inter,width_lim_arr
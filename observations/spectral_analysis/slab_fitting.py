import os,sys

import matplotlib.pyplot as plt
import glob
import argparse
import numpy as np

from xspec import AllModels,Fit,Xset,Plot,AllData
from xspec_config_multisp import calc_error,reset,load_mod,xscorpeon,addcomp,calc_fit,Plot_screen,rescale,\
                                 catch_model_str,store_fit,allmodel_data,freeze,delcomp

from linedet_utils import narrow_line_search,plot_line_search

ap = argparse.ArgumentParser(description='Script to fit photoionization models\n)')

#output directory
ap.add_argument("-outdir_base",nargs=1,help="name of output directory for line plots",
                default="slab_fits",type=str)

#two possible modes: dips to keep a single kT or 'tempvar' to allow only the temperature to vary
ap.add_argument('-mode',nargs=1,help='mode',default='tempvar',type=str)

#note that this must be a save of an allmodel_data() class, not an Xset save
ap.add_argument('-basemod_path',nargs=1,help='-base continuum model to load for each spectrum',
                default='test_mod_obj.mod')

#note that the recent change in this will make fake computations slower because they use the same grid
ap.add_argument("-line_search_e",nargs=1,
                help='min, max and step of the line energy search',default='4 10 0.02',type=str)

ap.add_argument("-line_search_norm",nargs=1,
                help='min, max and nsteps (for one sign)  of the line norm search (which operates in log scale)',
                default='0.01 10 500',type=str)

restrict_sp_list=["5665010401-002M002_sp_grp_opt.pha"]

ap.add_argument('-restrict',nargs=1,help='restrict to a given list of spectra',default=False)

args=ap.parse_args()

outdir_base=args.outdir_base

mode=args.mode

outdir=outdir_base+'_'+mode

basemod_path=args.basemod_path
line_search_e=np.array(args.line_search_e.split(' ')).astype(float)
line_search_norm=np.array(args.line_search_norm.split(' ')).astype(float)

restrict=args.restrict




os.system('mkdir -p '+outdir)

NICER_sp=np.array(glob.glob('**_sp_grp_opt.pha'))
NICER_sp.sort()

reset()
Plot.xLog=False

basemod=load_mod(basemod_path)

plt.ioff()

for elem_sp in NICER_sp:


    if restrict and elem_sp not in restrict_sp_list:
        print('\nRestrict mode activated and at least one spectrum not in the restrict array')

        continue

    epoch_observ=[elem_sp.split('_sp')[0]]

    epoch_orbit=int(epoch_observ[0].split('-')[1].split('M')[0])

    epoch_mancut=int(epoch_observ[0].split('-')[1].split('M')[1])

    if epoch_orbit<3:
        continue

    #skipping the non-dipping epochs
    if epoch_orbit>3:
        continue

    #resetting
    AllModels.clear()
    AllData.clear()

    '''Setting up a log file and testing the properties of each spectra'''

    curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log.log')

    #ensuring the log information gets in the correct place in the log file by forcing line to line buffering
    curr_logfile_write.reconfigure(line_buffering=True)

    curr_logfile=open(curr_logfile_write.name,'r')

    #loading the spectrum

    AllData('1:1 '+elem_sp)

    AllData.ignore('**-0.3 10.-**')

    # if epoch_orbit==3:
    #     # for models not in the dip
    #     xscorpeon.load('auto',frozen=True,fit_SAA_norm=epoch_mancut>1)
    #
    #     #fitting the main components without the iron line
    #     AllData.ignore('4.-8.')
    #     addcomp('cont_diskbb')
    #     addcomp('glob_TBabs')
    #
    #     calc_fit()
    #
    #     #adding the gaussian
    #     if epoch_mancut>1:
    #         addcomp('calNICERSiem_gaussian')
    #         calc_fit()
    #
    #     #noticing the rest
    #     AllData.notice('4.-8.')
    #
    #     #adding the broad gaussian and the cloudy model
    #     addcomp('gaussian',position='diskbb')
    #
    #     #adding the cloudy table
    #     addcomp('mtable{4u1630ezdisknthcomp.fits}',position='gaussian',endmult='diskbb')
    #     #and the cabs
    #     addcomp('cabs',position=2)
    #
    #     #link the cabs nH to the cloudy nH
    #     AllModels(1).cabs.nH.link='10.**(p'+str(AllModels(1).CLOUDY.logNH.index)+'-22.)'
    #
    #     #unfreezing the CLOUDY velocity shift
    #     AllModels(1).CLOUDY.z.frozen=False
    #     AllModels(1).CLOUDY.z.values=[0.,0.01, -0.05,-0.05,0.05,0.05]
    #
    #     #locking the ionization parameters to logical values
    #     AllModels(1).CLOUDY.logxi.values=[3.5,0.01,3.0,3.05,4.5,4.5]
    #
    #     #computing the fit again
    #     calc_fit()
    #
    #     mod_curr=allmodel_data()
    #
    #     #unfreezing the scorpeon model
    #     xscorpeon.load('auto',frozen=False,fit_SAA_norm=epoch_mancut>1,
    #                    scorpeon_save=mod_curr.scorpeon,load_save_freeze=False)
    #
    #     if epoch_mancut==1:
    #         AllData.notice('10.-12.')
    #
    #         #there is an issue with this parameter in this obs so we peg it
    #         AllModels(1,'nxb')(11).frozen=True
    #
    #     #and refitting
    #     calc_fit()
    #
    #     #refreezing the scorpeon model
    #     xscorpeon.freeze()
    #
    #     if epoch_mancut==1:
    #         AllData.ignore('8.-10.')
    #
    #         #refitting in the right band
    #         calc_fit()
    #
    #     #computing the errors
    #     err_fit=calc_error(curr_logfile,give_errors=True)[0]
    #
    #     #saving the fit
    #     store_fit('1zone',epoch_id=epoch_observ[0],outdir=outdir,logfile=curr_logfile)
    #
    #     #saving the errors
    #     np.savetxt(outdir + '/' + epoch_observ[0] + "_err_1zone.np",err_fit)
    #
    #     mod_1zone=allmodel_data()
    #
    #     #removing the CLOUDY model
    #     delcomp('CLOUDY')
    #     addcomp('cabs',position='diskbb')
    #     AllModels(1).cabs.nH.values=cabs_val[0]
    #
    #     mod_1zone_noabsline=allmodel_data()
    #
    #     #saving the screen of the cloudy-less version
    #     Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_observ[0]+'_screen_xspec_1zone_noabsline'))
    #
    #     #and a version both without absorption and without emission
    #     delcomp('gaussian')
    #     Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_observ[0]+'_screen_xspec_1zone_nolines'))
    #
    #     mod_1zone_nolines=allmodel_data()
    #
    #     Xset.chatter=1
    #     #making a blind search to look at the residuals
    #     chi_dict_1zone_nolines = narrow_line_search(mod_1zone_nolines, 'autofit',
    #                                           line_search_e=line_search_e, line_search_norm=line_search_norm,
    #                                           e_sat_low_indiv=[0.3],
    #                                           scorpeon_save=mod_1zone.scorpeon, data_fluxcont=mod_1zone_nolines)
    #
    #     plot_line_search(chi_dict_1zone_nolines, outdir, 'NICER', suffix='1zone_nolines', epoch_observ=epoch_observ)
    #
    #     Xset.chatter=10
    #
    # else:
    #     # for models in the dip

    #loading the model and scorpeon
    basemod.load()

    xscorpeon.load('auto',frozen=True)

    #fitting the nxb SAA
    AllModels(1,'nxb')(7).frozen=False

    #unfreezing the interstellar absorption
    AllModels(1).TBabs.nH.frozen=False

    #and the calibration gaussian norm and width
    AllModels(1).gaussian.norm.frozen=False
    AllModels(1).gaussian.Sigma.frozen=False

    if mode == 'tempvar':
        #unfreezing the temperature
        AllModels(1).diskbb.Tin.frozen=False
    #in dip mode we don't touch anything
    elif mode == 'dip':
        pass

    #adding the cloudy table
    addcomp('mtable{4u1630ezdisknthcomp.fits}',position='diskbb')
    #and the cabs
    addcomp('cabs',position=3)

    if epoch_orbit==3 and epoch_mancut==1:
        delcomp('gaussian')

    #link the cabs nH to the cloudy nH
    AllModels(1).cabs.nH.link='10.**(p'+str(AllModels(1).CLOUDY.logNH.index)+'-22.)'

    #unfreezing the CLOUDY velocity shift and giving it reasonable values
    AllModels(1).CLOUDY.z.frozen=False
    AllModels(1).CLOUDY.z.values=[0.,0.01, -0.05,-0.05,0.05,0.05]

    #Restricting the ionization parameter to reasonable values
    AllModels(1).CLOUDY.logxi.values=[3.5,0.01,3.0,3.05,4.5,4.5]

    #adding the emission line for the third obs
    if epoch_orbit==3:
        #adding the broad gaussian
        addcomp('gaussian',position='diskbb')

    #fitting
    calc_fit()

    mod=allmodel_data()

    #fitting with scorpeon
    xscorpeon.load('auto',frozen=False,scorpeon_save=mod.scorpeon,load_save_freeze=False,fit_SAA_norm=True)

    if epoch_orbit==3 and epoch_mancut == 1:
        #better constrains of the background like that here
        AllData.notice('10.-12.')

        # there is an issue with this parameter in this obs so we peg it
        AllModels(1, 'nxb')(11).frozen = True

    calc_fit()

    #refreezing scorpeon
    xscorpeon.freeze()

    #computing errors
    err_fit=calc_error(curr_logfile,give_errors=True)[0]

    #saving the fit
    store_fit('1zone',epoch_id=epoch_observ[0],outdir=outdir,logfile=curr_logfile)

    #and saving the errors
    np.savetxt(outdir + '/' + epoch_observ[0] + "_err_1zone.np",err_fit)

    freeze()

    mod_1zone=allmodel_data()

    #with this we keep the continuum decrease but remove the lines for a continuum base
    cabs_val=AllModels(1).cabs.nH.values[0]

    #removing the CLOUDY model
    delcomp('CLOUDY')
    addcomp('cabs',position='diskbb')
    AllModels(1).cabs.nH.values=cabs_val

    mod_1zone_noabsline=allmodel_data()

    #saving the screen of the cloudy-less version
    Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_observ[0]+'_screen_xspec_1zone_noabsline'))

    # and a version both without absorption and without emission
    if epoch_orbit==3:
        delcomp('gaussian')
        Plot_screen('ldata,ratio,delchi', os.path.join(outdir, epoch_observ[0] + '_screen_xspec_1zone_nolines'))

        mod_1zone_nolines = allmodel_data()

        Xset.chatter = 1
        # making a blind search to look at the residuals
        chi_dict_1zone_nolines = narrow_line_search(mod_1zone_nolines, 'autofit',
                                                    line_search_e=line_search_e, line_search_norm=line_search_norm,
                                                    e_sat_low_indiv=[0.3],
                                                    scorpeon_save=mod_1zone.scorpeon, data_fluxcont=mod_1zone_nolines)

        plot_line_search(chi_dict_1zone_nolines, outdir, 'NICER', suffix='1zone_nolines', epoch_observ=epoch_observ)

    Xset.chatter = 1

    #making a blind search to look at the residuals
    chi_dict_1zone_noabsline = narrow_line_search(mod_1zone_noabsline, 'autofit',
                                          line_search_e=line_search_e, line_search_norm=line_search_norm,
                                          e_sat_low_indiv=[0.3],
                                          scorpeon_save=mod_1zone.scorpeon, data_fluxcont=mod_1zone_noabsline)

    plot_line_search(chi_dict_1zone_noabsline, outdir, 'NICER', suffix='1zone_noabsline', epoch_observ=epoch_observ)

    chi_dict_1zone = narrow_line_search(mod_1zone, 'autofit',
                                          line_search_e=line_search_e, line_search_norm=line_search_norm,
                                          e_sat_low_indiv=[0.3],
                                          scorpeon_save=mod_1zone.scorpeon, data_fluxcont=mod_1zone_noabsline)

    plot_line_search(chi_dict_1zone, outdir, 'NICER', suffix='1zone', epoch_observ=epoch_observ)

    Xset.chatter=10

    #if needed for after
    mod_1zone.load()

plt.ion()

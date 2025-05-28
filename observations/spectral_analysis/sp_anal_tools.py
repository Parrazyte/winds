import pandas as pd

from xspec_config_multisp import *
import os
import glob
from astropy.time import Time,TimeDelta
from general_tools import file_edit

def sp_anal(obs_path,mod='powerlaw',baseload=False,model_load=False,obj='',scorpeon=True,overwrite=False,
            outdir='s_a',absval=None,set_ener_str=None,set_ener_xrism=False,line_ul='',freeze_cont_ul=True,
            bshift_range_ul=[-3000,3000],ul_level=99.7,n_ul_comp=101,ul_ener_range=[6.5,7.5]):

    plt.ioff()

    os.system('mkdir -p '+outdir)

    if os.path.isfile(outdir+"/infos_fit.txt"):
        lines_infos=pd.read_csv(os.path.join(outdir,'infos_fit.txt'),sep='\t')

        if not overwrite:

            if obs_path in lines_infos['Observ_file'].values:
                print(obs_path+' already analyzed')
                return

    Plot.xLog=False
    AllData.clear()
    AllModels.clear()

    if model_load:
        Xset.restore(obs_path)
        if set_ener_str is not None:
            set_ener(set_ener_str,xrism=set_ener_xrism)

        obs=AllData(1).fileName

    elif baseload:
        Xset.restore(obs_path)
        obs=AllData(1).fileName
        AllModels.clear()
    else:
        AllData('1:1 '+obs_path)
        obs=obs_path

    if not model_load:
        AllData.ignore('**-0.3 10.-**')
        #file infos

    with fits.open(obs) as hdul:
        telescope=hdul[1].header['TELESCOP']
        if telescope=='NICER':
            start_obs_s = hdul[1].header['TSTART'] + \
                          (hdul[1].header['TIMEZERO'] if telescope == 'NICER' else 0)
            # saving for titles later
            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

            obs_start=obs_start.isot

        else:
            try:
                obs_start = hdul[0].header['DATE-OBS']
            except:
                try:
                    obs_start = hdul[1].header['DATE-OBS']
                except:
                    obs_start = Time(hdul[1].header['MJDSTART'], format='mjd').isot

    logfile_write, logfile = xLog_rw(os.path.join(outdir,obs.split('.')[0]+'_xlog.log'))

    if not model_load:
        if telescope=='NICER' and scorpeon:
            xscorpeon.load('auto',frozen=True)

            if os.path.isfile(os.path.join(outdir,obs.split('.')[0]+'_baseload.xcm')):
                os.remove(os.path.join(outdir,obs.split('.')[0]+'_baseload.xcm'))

            Xset.save(os.path.join(outdir,obs.split('.')[0]+'_baseload.xcm'))

        if mod=='thcont':
            mod_list=['cont_diskbb', 'disk_thcomp', 'glob_TBfeo']
        elif mod=="powerlaw":
            mod_list=['powerlaw', 'glob_TBfeo']




        fitlines_strong=fitmod(mod_list,logfile,logfile_write,
                               absval=absval if absval is not None else
                                        10.7 if '4U1630-47' in obj else 0.25 if obj=='V4641Sgr' else None)

        fitlines_strong.add_allcomps(split_fit=False)

        if absval is not None:
            AllModels(1).TBfeo.nH.values=absval
            AllModels(1).TBfeo.nH.frozen=True

        # if obj=='4U1630-47' and abs:
        #     AllModels(1).TBfeo.nH.values=10.7
        #     AllModels(1).TBfeo.nH.frozen=True

        if obj=='4U1630-47_lock':
            # AllModels(1).TBfeo.nH.values=10.7
            # AllModels(1).TBfeo.nH.frozen=True
            AllModels(1).powerlaw.PhoIndex.values = 1.5
            AllModels(1).powerlaw.PhoIndex.frozen=True

        # if obj=='V4641Sgr':
        #     AllModels(1).TBfeo.nH.values=0.25
        #     AllModels(1).TBfeo.nH.frozen=True

        if obj=='17091':
            # AllModels(1).TBfeo.nH.values=1.537
            # AllModels(1).TBfeo.nH.frozen=True
            AllModels(1)(2).values=0.452
            AllModels(1)(3).values=2.33
        # AllModels(1)(2).frozen=False
        # AllModels(1)(3).frozen=False
        if mod=='thcont':
            AllModels(1).diskbb.Tin.values=[1.,0.01,0.4,0.4,2.,2.]

        Fit.perform()
        try:
            Fit.error('1-'+str(AllModels(1).nParameters))
        except:
            pass

        # if telescope=='NICER' and scorpeon:
        #     xscorpeon.load('auto',frozen=False,fit_SAA_norm=True)
        #
        # Fit.perform()
        # try:
        #     calc_error(logfile)
        # except:
        #     pass

    if os.path.isfile(os.path.join(outdir,obs.split('.')[0]+'_mod.xcm')):
        os.remove(os.path.join(outdir,obs.split('.')[0]+'_mod.xcm'))

    Xset.save(os.path.join(outdir,obs.split('.')[0]+'_mod.xcm'))

    #removing the calibation components and the interstellar absorption

    Plot_screen('ldata,ratio,delchi',os.path.join(outdir,obs.split('.')[0]+'_resid_screen.png'))

    # saving the model str
    catch_model_str(logfile, savepath=os.path.join(outdir,obs.split('.')[0]+'_mod.txt'))

    #computing the fluxes in different bands
    flux_arr=np.repeat(np.nan,9)
    for i in range(9):
        AllModels.calcFlux(str(float(i+1))+' '+str(float(i+2)))
        flux_arr[i]= AllData(1).flux[0]


    infos_store_path=os.path.join(outdir,'infos_fit_abs.txt')
    line_str = obs + \
               '\t' +str(obs_start)+\
               '\t' + telescope+\
               '\t%.4e' %flux_arr[0]+ \
               '\t%.4e' %flux_arr[1] + \
               '\t%.4e' %flux_arr[2] + \
               '\t%.4e' %flux_arr[3] + \
               '\t%.4e' %flux_arr[4] + \
               '\t%.4e' %flux_arr[5] + \
               '\t%.4e' %flux_arr[6] + \
               '\t%.4e' %flux_arr[7] + \
               '\t%.4e' %flux_arr[8] + \
                '\n'

    line_store_header = 'Observ_file\tt_start\ttelescope\t'+\
                        'flux_1-2\tflux_2-3\tflux_3-4\tflux_4-5\tflux_5-6\tflux_6-7\tflux_7-8\tflux_8-9\tflux_9-10\n'

    file_edit(path=infos_store_path, line_id=obs, line_data=line_str, header=line_store_header)

    mod_withabs=allmodel_data()

    if 'TBfeo' in AllModels(1).componentNames:
        delcomp('TBfeo')
    if 'TBabs' in AllModels(1).componentNames:
        delcomp('TBabs')

    #computing the fluxes in different bands
    flux_arr=np.repeat(np.nan,9)
    for i in range(9):
        AllModels.calcFlux(str(float(i+1))+' '+str(float(i+2)))
        flux_arr[i]= AllData(1).flux[0]


    infos_store_path=os.path.join(outdir,'infos_fit_deabs.txt')
    line_str = obs + \
               '\t' +str(obs_start)+\
               '\t' + telescope+\
               '\t%.4e' %flux_arr[0]+ \
               '\t%.4e' %flux_arr[1] + \
               '\t%.4e' %flux_arr[2] + \
               '\t%.4e' %flux_arr[3] + \
               '\t%.4e' %flux_arr[4] + \
               '\t%.4e' %flux_arr[5] + \
               '\t%.4e' %flux_arr[6] + \
               '\t%.4e' %flux_arr[7] + \
               '\t%.4e' %flux_arr[8] + \
                '\n'

    line_store_header = 'Observ_file\tt_start\ttelescope\t'+\
                        'flux_1-2\tflux_2-3\tflux_3-4\tflux_4-5\tflux_5-6\tflux_6-7\tflux_7-8\tflux_8-9\tflux_9-10\n'

    file_edit(path=infos_store_path, line_id=obs, line_data=line_str, header=line_store_header)

    if line_ul!='':

        mod_withabs.load()

        if freeze_cont_ul:
            allfreeze()

        if ul_ener_range is not None:
            AllData.ignore('**-'+str(ul_ener_range[0])+' '+str(ul_ener_range[1])+'-**')

        res_ul=np.array([None]*len(line_ul.split('+')))

        infos_store_path=os.path.join(outdir,'infos_line_deabs.txt')
        line_str = obs + \
                   '\t' +str(obs_start)+\
                   '\t' + telescope

        for i,elem_line in enumerate(line_ul.split('+')):

            elem_line_comp=fitcomp_line(elem_line,logfile=logfile,logfile_write=logfile_write)

            res_ul[i]=elem_line_comp.get_ew_ul(bshift_range=bshift_range_ul, line_width=5e-3, ul_level=ul_level,
                                               n_ul_comp=n_ul_comp)

            line_str+='\t%.4e' %res_ul[i]

        line_str+='\n'

        line_store_header = 'Observ_file\tt_start\ttelescope\t'+('\t'.join(line_ul.split('+')))+'\n'

        file_edit(path=infos_store_path, line_id=obs, line_data=line_str, header=line_store_header)

        mod_withabs.load()




def NICER_run_all_sp(sort=False,reverse=False,mod='thcont'):
    sp_list=glob.glob('**_grp_opt.pha')
    sp_list=[elem for elem in sp_list if 'MRG' not in elem and 'NF' not in elem]
    sp_list=np.array(sp_list)
    if sort:
        sp_list.sort()
    if reverse:
        sp_list=sp_list[::-1]

    for elem_sp in sp_list:
        sp_anal(elem_sp,mod=mod)

def swift_OT_run_all_sp(sort=False,reverse=False,mod='thcont',obj='V4641Sgr',outdir='s_a',absval=None,
                        overwrite_baseload=False):
    sp_list=glob.glob('**_grp_opt.pi')

    epoch_list=np.unique([elem.split('source_')[0][:-2] for elem in sp_list])

    epoch_list=np.array(epoch_list)
    if sort:
        epoch_list.sort()
    if reverse:
        epoch_list=epoch_list[::-1]

    for elem_epoch in epoch_list:

        if overwrite_baseload or not os.path.isfile(elem_epoch+'_baseload.xcm'):

            if os.path.isfile(elem_epoch+'_baseload.xcm'):
                os.remove(elem_epoch+'_baseload.xcm')

            elem_sp=[elem for elem in sp_list if elem.startswith(elem_epoch)]

            elem_sp.sort()
            two_sp=len(elem_sp)>1

            AllData('1:1 '+elem_sp[0]+('' if not two_sp else' 2:2 '+elem_sp[1]))
            AllData(1).response.arf=elem_sp[0].replace('source_grp_opt.pi','.arf')
            AllData(1).background=elem_sp[0].replace('source_grp_opt.pi','back.pi')

            if two_sp:
                AllData(2).response.arf = elem_sp[1].replace('source_grp_opt.pi', '.arf')
                AllData(2).background = elem_sp[1].replace('source_grp_opt.pi', 'back.pi')

            Xset.save(elem_epoch+'_baseload.xcm')
            AllData.clear()

        sp_anal(elem_epoch+'_baseload.xcm',mod=mod,baseload=True,obj=obj,outdir=outdir,absval=absval)


def fit_broader(epoch_id,add_gaussem=True,bat_interp_dir='/home/parrama/Documents/Observ/copy_SSD/Swift/BAT_interp',
                n_add=1,outdir='fit_broader',bat_emin=15.,bat_emax=50.,avg_BAT_norm=True):
    '''
    for quick refitting in broader bands

    bat_interp_dir is the directory where bat interpolation spectra (created with ftflx2xsp from the regression vals)
     are stored

    n_add is the number of times the spectrum is added (to give it more weight in the fit)
    '''

    # Plot.xLog=False

    plt.ioff()

    reset()
    AllModels.clear()
    AllData.clear()
    Plot.xLog=True
    Plot.xAxis='keV'

    Xset.restore('lineplots_opt/'+epoch_id+'_mod_broadband_post_auto.xcm')
    #adding the swift info
    groups=AllData.nGroups

    currdir=os.getcwd()
    os.chdir(bat_interp_dir)

    #loading n_add times the BAT spectrum of the observation if it exists
    if not os.path.isfile(epoch_id+'_BAT_regr_sp_'+str(bat_emin)+'_'+str(bat_emax)+'.pi'):
        print('No concurrent daily BAT regression spectrum available. Skipping this step...')
    else:
        add_str=' '.join([str(groups+i)+':'+str(groups+i)+' '+epoch_id+'_BAT_regr_sp_'
                          +str(bat_emin)+'_'+str(bat_emax)+'.pi' for i in range(1,n_add+1)])
        AllData(add_str)


    os.chdir(currdir)

    os.system('mkdir -p '+outdir)

    curr_logfile_write,curr_logfile=xLog_rw(os.path.join(outdir, epoch_id + '_broader.log'))

    AllModels.clear()
    AllModels.setEnergies('0.01 1000. 10000 log')

    print('Loading base model...')

    Xset.chatter=0

    addcomp('cont_diskbb')
    addcomp('glob_thcomp')
    AllModels(1).thcomp.kT_e.values=100
    AllModels(1).thcomp.kT_e.frozen=True
    AllModels(1).thcomp.Gamma_tau.values=[2.,0.01,1.5,1.5,3.5,3.5]
    addcomp('glob_TBabs')
    addcomp('glob_constant')

    #locking the constant factor(s) of the BAT elements
    for i in range(groups+1,AllData.nGroups+1):
        AllModels(i)(1).frozen=True

    AllData.notice('0.3-3.')

    xscorpeon.load('auto',frozen=True)

    Xset.chatter=10

    Fit.perform()

    addcomp('calNICERSiem_gaussian',position='lastin')

    if avg_BAT_norm and not add_gaussem:
        AllModels(i)(1).link = str(groups) + '.' + ' /(' + '+'.join([str(1 + AllModels(1).nParameters * id_grp) \
                                                                     for id_grp in range(groups)]) + ')'
    calc_fit()

    if add_gaussem:
        addcomp('FeKa0em_bgaussian', position='thcomp')

        #not that here the NICER edge is not removed for the BAT datagroup, but we don't care considering the energy
        addcomp('calNICER_edge')

        # locking the constant factor(s) of the BAT elements
        for i in range(groups + 1, AllData.nGroups + 1):

            if avg_BAT_norm:
                AllModels(i)(1).link = str(groups)+'.'+' /('+'+'.join([str(1+AllModels(1).nParameters*id_grp)\
                                                                         for id_grp in range(groups)])+')'

        calc_fit()


    xscorpeon.load('auto',frozen=False,extend_SAA_norm=True,fit_SAA_norm=True)

    calc_fit()
    xscorpeon.freeze()

    mod=allmodel_data()

    Plot.xLog=True
    xPlot('ldata,ratio,delchi')

    if os.path.isfile(os.path.join(outdir,epoch_id+'_mod_broader.xcm')):
        os.remove(os.path.join(outdir,epoch_id+'_mod_broader.xcm'))

    if os.path.isfile(os.path.join(outdir,epoch_id+'_mod_broader_mod.xcm')):
        os.remove(os.path.join(outdir,epoch_id+'_mod_broader_mod.xcm'))

    Xset.save(os.path.join(outdir,epoch_id+'_mod_broader.xcm'),info='a')

    Xset.save(os.path.join(outdir,epoch_id+'_mod_broader_mod.xcm'),info='m')

    Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_id+'_mod_broader_screen'))

    Plot.xLog=False
    Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_id+'_mod_broader_zoom_NICER_screen'),xlims=[0.3,10.0])
    Plot.xLog=True

    # saving the model str
    catch_model_str(curr_logfile, savepath=outdir + '/' + epoch_id + '_mod_broader.txt')

    # locking the constant factor(s) of the BAT elements
    for i in range(groups + 1, AllData.nGroups + 1):

        if avg_BAT_norm:
            AllModels(i)(1).link = ''

    #creating the SEDs
    save_broad_SED(path=os.path.join(outdir,epoch_id+'_mod_broader_SED.xcm'),
                   e_low=0.01,e_high=1000,nbins=2e3,retain_session=False,
                   remove_abs=True,remove_gaussian=True,remove_cal=True,
                   remove_scorpeon=True)

    plt.ion()

def fit_broader_BAT(epoch_list,
                    add_gaussem=True,
                    outdir='fit_broader'):
    '''

    for quick refitting in broader bands with BAT

    '''


    plt.ioff()

    reset()
    AllModels.clear()
    AllData.clear()
    Plot.xLog=True
    Plot.xAxis='keV'

    load_list(epoch_list)

    epoch_id='_'.join(shorten_epoch([elem.split('_sp')[0] for elem in epoch_list]))

    #adding the swift info
    groups=AllData.nGroups

    currdir=os.getcwd()

    os.system('mkdir -p '+outdir)

    curr_logfile_write,curr_logfile=xLog_rw(os.path.join(outdir, epoch_id + '_broader.log'))

    AllModels.clear()

    print('Loading base model...')

    Xset.chatter=0

    try:
        addcomp('cont_diskbb')
    except:
        breakpoint()
    addcomp('disk_thcomp')
    addcomp('glob_TBabs')
    addcomp('glob_constant')
    for i_grp in range(groups):
        if AllData(i_grp+1).fileinfo('TELESCOP')=='NICER':
            AllData(i_grp+1).ignore('**-0.3 10.-**')
        if i_grp!=0:
            AllModels(i_grp+1)(1).values=[1.,0.01,0.95,0.95,1.05,1.05]

    xscorpeon.load('auto',frozen=True)

    Xset.chatter=10

    calc_fit()

    addcomp('calNICERSiem_gaussian',position='lastin')

    calc_fit()

    if add_gaussem:
        addcomp('FeKa0em_bgaussian', position='thcomp')

        #not that here the NICER edge is not removed for the BAT datagroup, but we don't care considering the energy
        addcomp('calNICER_edge')

        calc_fit()


    xscorpeon.load('auto',frozen=False,extend_SAA_norm=True,fit_SAA_norm=True)

    calc_fit()
    xscorpeon.freeze()

    mod=allmodel_data()

    Plot.xLog=True
    xPlot('ldata,ratio,delchi')

    if os.path.isfile(os.path.join(outdir,epoch_id+'_mod_broader.xcm')):
        os.remove(os.path.join(outdir,epoch_id+'_mod_broader.xcm'))

    if os.path.isfile(os.path.join(outdir,epoch_id+'_mod_broader_mod.xcm')):
        os.remove(os.path.join(outdir,epoch_id+'_mod_broader_mod.xcm'))

    Xset.save(os.path.join(outdir,epoch_id+'_mod_broader.xcm'),info='a')

    Xset.save(os.path.join(outdir,epoch_id+'_mod_broader_mod.xcm'),info='m')

    Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_id+'_mod_broader_screen'))

    Plot.xLog=False
    Plot_screen('ldata,ratio,delchi',os.path.join(outdir,epoch_id+'_mod_broader_zoom_NICER_screen'),xlims=[0.3,10.0])
    Plot.xLog=True

    # saving the model str
    catch_model_str(curr_logfile, savepath=outdir + '/' + epoch_id + '_mod_broader.txt')

    #creating the SEDs
    save_broad_SED(path=os.path.join(outdir,epoch_id+'_mod_broader_SED.xcm'),
                   e_low=0.01,e_high=1000,nbins=2e3,retain_session=False,
                   remove_abs=True,remove_gaussian=True,remove_cal=True,
                   remove_scorpeon=True)

    plt.ion()

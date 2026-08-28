import time
import pexpect
import os,sys
import re
from astropy.io import fits
#for no_op_context
import contextlib
import numpy as np
from tee import StdoutTee, StderrTee
from tqdm import tqdm
from xspec import Xset,AllModels,AllData
from xspec_config_multisp import set_ener,save_mo_Ryota
import glob
import matplotlib.pyplot as plt

def _remove_control_chars(message):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)

def set_var_simput(spawn,heasoft_init_alias='heainit',sixte_init_alias='sixteinit'):
    '''
    Sets starting environment variables for data analysis
    '''
    if heasoft_init_alias is not None:
        spawn.sendline(heasoft_init_alias)

    if sixte_init_alias is not None:
        spawn.sendline(sixte_init_alias)

def SIXTE_run(outdir='./',

              XMLDIR='/home/parrazyte/Soft/SIXTE/instruments/new-athena-xifu/baseline',
              CONFIG='nofilt_defoc',


              #flux in units of 1e-11 erg/s/cm²
              flux_m11='auto',

              Emin=2.,
              Emax=10.,

              Elow=0.1,
              Eup=10.,
              #note: assumed linear unless another argument is added in part 1
              nbins=19800,

              #directory in which xspecfile will be loaded (must contain tables and equivalents)
              #absolute or relative from cwd
              xspec_dir='./',

              #should not contain calls to non-local tables
              xspec_file='',

              make_table_mod=False,
              table_mod_set_ener_str='',

              exposure=5000,
              min_countprod=0,
              fix_countprod=0,

              # time allocated for sixtesim spawn before timeout
              sixtesim_timeout=600,

              #for sixte_arfgen
              n_photons=100000,
              # only full.reg is implemented for now
              regfile='full.reg',
              #time allocated for arfgen spawn before timeout
              arfgen_timeout=600,

              #for extract_spec, should be a list
              grade_spec=[1,2,3,4,5,6],

              #for combined_spectrum, should be a list
              grade_mergespec=[1, 2, 3],
              heasoft_init_alias='heainit',
              sixte_init_alias='sixteinit',

              #clean arfs after finishing the run
              clean_arf=False,

              steps=[1,2,3,5,6],
              reload_seed='',
              parallel=False):

    '''
    Python wrapper for the SIXTE commands from xifu-extract-spectra

    Requires simput 3.5.0

    flux:
        number or 'auto' or iterable
        - if set to auto, will compute the xspec model flux in the E_min/E_max band and use that value
        - if set to a list, the code will run SIXTE once for every flux value, and add a suffix to the outdir every time

    min_countprod:
        used to ensure a dynamical minimum exposure per flux
        replaces exposure by min_countprod/elem_fluxm11 if this value is higher than exposure
        (if set to zero or a very low value, this will do nothing)

    fix_countprod:
        used to ensure a dynamical minimum exposure per flux
        if not set to zero, relaces the exposure by fix_countprod/elem_fuxm11

    '''

    assert regfile=='full.reg','Error: only full.reg region file for sixte_arfgen implemented for now'

    XMLFILE = XMLDIR + '/xifu_' + CONFIG + '.xml'

    currdir=os.getcwd()

    if type(flux_m11) in [list,tuple,np.ndarray]:
        flux_m11_use=flux_m11
    else:
        flux_m11_use=[flux_m11]

    with tqdm(total=len(flux_m11_use)) as pbar:

        for i_iter,elem_flux_m11 in enumerate(flux_m11_use):

            bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

            set_var_simput(bashproc,heasoft_init_alias,sixte_init_alias)

            elem_outdir=outdir+('' if type(flux_m11) not in [list,tuple,np.ndarray] else
                               '_'+(('%.3e'%(elem_flux_m11*1e-11)).replace('+','p').replace('-','m')))

            bashproc.sendline('cd '+currdir)

            elem_outdir_abs=os.path.join(currdir,elem_outdir)

            if not os.path.exists(elem_outdir_abs):
                os.makedirs(elem_outdir_abs)

            #ensuring exposure high enough for a minimum amount of counts
            exposure_use=max(exposure,min_countprod/elem_flux_m11)

            if fix_countprod!=0:
                exposure_use=fix_countprod/elem_flux_m11

            if reload_seed!='':
                time_str=reload_seed
            else:
                time_str=str(time.time()).replace('.','p')


            log_path=os.path.join(elem_outdir,'SIXTE_run_'+time_str+('_reload' if reload_seed else '')+'.log')

            os.system('mkdir -p '+elem_outdir)

            if os.path.isfile(log_path):
                os.system('rm ' + log_path)

            with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                             file_filters=[_remove_control_chars]), \
                  StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

                if not parallel:
                    bashproc.logfile_read = sys.stdout

                bashproc.sendline('cd '+os.path.join(currdir,elem_outdir))

                simput_file = 'simput_' + time_str + '.simput'

                if 1 in steps:

                    '''''''''''''''''''''''''''
                    from 01_build_simput.sh
                    '''''''''''''''''''''''''''

                    bashproc.sendline('cd '+os.path.join(currdir,xspec_dir))

                    flux_use = elem_flux_m11 * 1e-11

                    if make_table_mod or (type(elem_flux_m11)==str and elem_flux_m11=='auto'):
                        xspec_asciif=xspec_file.replace('.xcm','_ascii.dat')
                        os.chdir(xspec_dir)
                        Xset.restore(xspec_file)


                        if table_mod_set_ener_str!='':
                            set_ener(table_mod_set_ener_str,xrism=True)

                        if type(elem_flux_m11)==str and elem_flux_m11=='auto':
                            AllModels.calcFlux(str(float(Emin))+' '+str(float(Emax)))
                            flux_use=AllModels(1).flux[0]
                        else:
                            save_mo_Ryota(xspec_asciif)

                        AllModels.clear()
                        AllData.clear()
                        os.chdir(currdir)

                    bashproc.sendline('simputfile Simput='+simput_file+' Src_Name=first'
                                      +' RA=0.0 Dec=0.0 '
                                      +' srcFlux=%.3e'%(flux_use)+' Emin='+str(Emin)+' Emax='+str(Emax)
                                      +' Elow='+str(Elow)+' Eup='+str(Eup)+' NBins='+str(nbins)+
                                      (' XSPECFile='+xspec_file if not make_table_mod else ' ASCIIFile='+xspec_asciif))

                    #note: we expect 3 instances of this string in case of success
                    bashproc.expect('finished successfully!')
                    bashproc.expect('finished successfully!')
                    bashproc.expect('finished successfully!')

                    #moving back the simput to elem_outdir
                    bashproc.sendline('mv simput_'+time_str+'.simput '+os.path.join(currdir,elem_outdir))

                    #moving in elem_outdir
                    bashproc.sendline('cd '+os.path.join(currdir,elem_outdir))


                INFILE=simput_file
                EVFILE='evt_xifu_'+CONFIG+'_'+time_str+'.fits'

                if 2 in steps:

                    '''''''''''''''''''''''''''
                    from 02_run_sim.sh
                    '''''''''''''''''''''''''''

                    bashproc.sendline('sixtesim XMLFile='+XMLFILE+' Simput="'+INFILE+'" Exposure='+str(exposure_use)+' EvtFile='+EVFILE+
                                      ' doCrosstalk=none RA=0.0 Dec=0.0 clobber=y')

                    bashproc.expect('initialize ...')
                    bashproc.expect('start simulation...')
                    bashproc.expect('finished successfully!',timeout=sixtesim_timeout)


                if 3 in steps:

                    '''''''''''''''''''''''''''
                    from 03_gen_arf.sh
                    '''''''''''''''''''''''''''

                    #creating a "full.reg" file inside the directory
                    full_reg_str=['# Region file format: DS9 version 4.1\n',
                    'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n',
                    'fk5\n',
                    'circle(0.0000064,0.0000025,180.000")\n']
                    with open(os.path.join(elem_outdir,regfile), 'w+') as f:
                        f.writelines(full_reg_str)

                    bashproc.sendline('sixte_arfgen XMLFile='+XMLFILE+' Simput="'+INFILE+'" Exposure='+str(exposure_use)+
                                      ' RA=0.0 Dec=0.0 clobber=y RefRA=0 RefDec=0 sampling_factor=100'+
                                      ' n_photons='+str(n_photons)+' ARFCorr=arf_full.fits writePixARF=y regfile='+regfile)

                    bashproc.expect(['100%'],timeout=arfgen_timeout)
                    time.sleep(5)

                if 5 in steps:


                    '''''''''''''''''''''''''''
                    from 05_extract_spec.sh
                    '''''''''''''''''''''''''''
                    for GRADE in grade_spec:
                        print('---------------------Grade '+str(GRADE)+'---------------------')

                        bashproc.sendline('python3'
                        +' /home/parrazyte/Documents/Work/Scripts/Python/NewAthena/xifu-extract-spectra/grading_specext.py'
                        +' -XMLFile '+XMLFILE+' -Grade '+str(GRADE)+' -inARF arf_full.fits'
                        +' -evtFile evt_xifu_'+CONFIG+'_'+time_str+'.fits'
                        +' -outSpec xifu_grad'+str(GRADE)+'.pha -outARF arf_grad'+str(GRADE)+'.fits')

                        bashproc.expect('--- Cleanup ---')
                        time.sleep(0.1)

                if 6 in steps:

                    '''''''''''''''''''''''''''
                    from 06_combined_spectrum.sh
                    '''''''''''''''''''''''''''
                    for GRADE in grade_mergespec:

                        PHA=os.path.join(elem_outdir,'xifu_grad'+str(GRADE)+'.pha')

                        RMF=fits.open(PHA)[1].header['RESPFILE']
                        ARF=fits.open(PHA)[1].header['ANCRFILE']

                        # marfrmf seems to not handle long file names very well
                        # softlink the RMF and ARF into this directory temporarily
                        # source and destination paths
                        symlink_rmf= os.path.join(elem_outdir,'tmp_rmf_XXX.fits')
                        symlink_arf= os.path.join(elem_outdir,'tmp_arf_XXX.fits')

                        #removing them if there was an error before their deletion previously
                        if os.path.exists(symlink_rmf):
                            os.remove(symlink_rmf)
                        if os.path.exists(symlink_arf):
                            os.remove(symlink_arf)

                        os.symlink(RMF,symlink_rmf,)
                        os.symlink(ARF,symlink_arf)

                        bashproc.sendline('marfrmf rmfil=tmp_rmf_XXX.fits arfil=tmp_arf_XXX.fits outfil=mulrmf_grad'+str(GRADE)+'.rmf clobber=y')

                        bashproc.expect('RDARF1')

                        bashproc.sendline('rm tmp_arf_XXX.fits')
                        bashproc.sendline('rm tmp_rmf_XXX.fits')

                        bashproc.sendline('echo GRADE '+str(GRADE)+' mulrmf created')

                        bashproc.expect('created')


                    bashproc.sendline('echo mulrmf creation finished')
                    bashproc.expect('finished')

                    grade_mergespec_str=''.join(np.array(grade_mergespec).astype(str))
                    # add them up with weights of 1
                    bashproc.sendline('addrmf'
                                      +' list='+(','.join(('mulrmf_grad'+np.array(grade_mergespec).astype(str)+'.rmf').tolist()))
                                      +' weights='+(','.join(np.repeat('1',len(grade_mergespec)).tolist()))
                                      +' rmffile=grad_'+grade_mergespec_str+'.rmf')

                    for GADE in grade_mergespec:
                        bashproc.expect('successfully read')

                    bashproc.expect('successfully written')
                    # extract a new spectrum with all grades together
                    # NOTE: any sort of additional filtering (both when generating the ARF or the grade spectra)
                    # must also be done here!
                    EventFilter_key=' || '.join('GRADING=='+np.array(grade_mergespec).astype(str))
                    bashproc.sendline('makespec EvtFile='+EVFILE+' RSPPATH='+XMLDIR
                                     +' Spectrum=xifu_grad_'+grade_mergespec_str+'.pha clobber=y'
                                     +' EventFilter="'+EventFilter_key+'"')

                    bashproc.expect('calculate spectrum')

                    bashproc.expect('store spectrum')

                    with fits.open(os.path.join(elem_outdir,'xifu_grad_'+grade_mergespec_str+'.pha'),mode='update') as hdul:
                        # RMF contains ARF!
                        hdul[1].header['RESPFILE']='grad_'+grade_mergespec_str+'.rmf'
                        hdul[1].header['ANCRFILE']=''
                        hdul.flush()

                    bashproc.sendline('rm mulrmf_grad*.rmf')

                    bashproc.sendline('echo ending')
                    bashproc.expect('ending')
                    time.sleep(1)
                    bashproc.close()

                if clean_arf:
                    os.system('rm '+os.path.join(elem_outdir,'arf_*.fits'))

            pbar.update()


def flux_explo(save=False):

    '''
    used to summarize the branching ratio outputs of a flux exploration

    SIXTE_run(Emin=0.3,Emax=10.,
    xspec_dir='../SEDs',
    xspec_file='SED_soft_0p1Edd_2e22.xcm',
    exposure=100,steps=[1,2],
    flux_m11=np.logspace(1,4,301)*2.4,min_countprod=2e5)

    '''
    evt_list=glob.glob('**/evt_xifu_**.fits')

    grade_names=['VH',
                 'H',
                 'I',
                 'M',
                 'Lim',
                 'Low']
    evt_flux=[]
    evt_grades=[]
    evt_invalid=[]
    for elem_evt in evt_list:
        evt_flux+=[float(elem_evt.split('/')[0].split('_')[-1].replace('m','-'))]
        with fits.open(elem_evt) as hdul:
            elem_grades=[]
            for i in range(1,7):

                elem_grades+=[hdul[1].header['NGRAD'+str(i)]/hdul[1].header['EXPOSURE']]
            evt_grades+=[elem_grades]
            # evt_invalid+=[hdul[1].header['NINVALID']]

    evt_flux=np.array(evt_flux)
    evt_grades=np.array(evt_grades)
    evt_grades=evt_grades[evt_flux.argsort()].T
    evt_flux.sort()

    plt.figure(figsize=(10,8))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('flux (cgs)')
    plt.ylabel('counts')
    plt.title('evolution of count rate with flux for canonical soft SED')
    for i_grade in range(6):
        plt.plot(evt_flux,evt_grades[i_grade],label='GRADE '+str(i_grade+1))
    plt.plot(evt_flux, evt_grades[:3].sum(0), label='sum of GRADES 1-2-3',color='black',ls='--')
    # plt.plot(evt_flux, evt_invalid, label='Invalid',color='grey',ls=':')

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('counts_fluxevol.pdf')

    plt.figure(figsize=(10, 8))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('flux (cgs)')
    plt.ylabel('grade ratio (not counting null events)')
    plt.title('evolution of total counts with flux for canonical soft SED')

    for i_grade in range(6):
        plt.plot(evt_flux, evt_grades[i_grade]/(evt_grades.sum(0)), label='GRADE ' + str(i_grade+1))
    plt.plot(evt_flux, evt_grades[:3].sum(0)/(evt_grades.sum(0)), label='sum of GRADES 1-2-3', color='black', ls='--')

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('grade_fluxevol.pdf')



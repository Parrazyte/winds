

import os
import glob
from general_tools import shorten_epoch
from pdf_summary import pdf_summary

from ast import literal_eval
import sys
import numpy as np
from general_tools import file_edit
import subprocess
from joblib import Parallel, delayed
import pexpect

import time
def linedet_loop(epoch_list,arg_dict,arg_dict_path=None,parallel=1,heasoft_init_alias='heainit',
                 container_mode='python',container='default',job_id='default',
                 force_instance=False):
    '''

    epoch_list:
        list of epochs

    arg_dict:
        dictionnary created in line_detection_multisp

    arg_dict_path:
        path of the dump of arg_dict, to be used in instance runs

    parallel:
        number of cores used for parallelization. if set to 1, runs without instanciation,
         UNLESS force_instance is set to True

    heasoft init alias:
        bash heasoft initialisation command, necesarry for python container

    container_mode:
        python (standard)
        in this case container should be the path of the python executable

        singularity (for cluster)
        in this case container should be set to the name of the singularity heasoft instance
        OR default (so it gets fetched via the environment variable heasoft_singularity)


            Note: steps to use a heasoft singularity container in a cluster

            1. Install docker cluster image from the heasoft links

            2. convert docker to singularity
            first save the image from the create docker:
            ex:
            docker save -o ./mydocker.tar heasoft:v6.33.2

            3.then convert the tar to singularity (takes time)
            singularity build local_tar.sif docker-archive://local.tar

            4. run the singularity, install required python packages (here we cloned everything)

            5. when in the cluster, create a singularity instance and run a shell inside, use as desired (see below)

            Note: the backend plotting libraries often have conflicts with opencv, so it is important
            to force the 'agg' matplotlib backend to avoid issues with pyqt

    container:
        -default:
            if container_mode=singularity, uses the heasoft_singularity environment variable
            if container_mode=python, uses sys.executable
        -string: uses that string as the container path

    job_id: name to add to singularity containers to maintain separate containers if running different jobs
        -default:
            uses the 4 last directory elements + the outdir + the sat_glob
        -string:
        uses that strings
    '''


    compute_highflux_only=arg_dict['compute_highflux_only']
    spread_comput=arg_dict['spread_comput']
    outdir=arg_dict['outdir']
    write_aborted_pdf=arg_dict['write_aborted_pdf']
    sat_glob=arg_dict['sat_glob']

    if job_id=='default':
        job_id_use='_'.join(os.getcwd().split('/')[-4:]+[outdir,sat_glob])
    else:
        job_id_use=job_id

    singularity_instance_name = 'linedet_run_' + job_id_use

    #getting the current file path
    file_path=os.path.realpath(__file__)
    linedet_runner_path=file_path.replace('linedet_loop.py','linedet_runner.py')

    def linedet_subprocess(epoch_id,arg_dict_path):

        print('Starting line detection for epoch\n')
        print(epoch_list[epoch_id])

        io_log = open(outdir + '/'+epoch_list[epoch_id][0][:epoch_list[epoch_id][0].rfind('.')]+'_log_spawn.log', 'w+')

        bashproc = pexpect.spawn("/bin/bash", encoding='utf-8', logfile=io_log)

        if parallel==1:
            bashproc.logfile_read=sys.stdout

        #choosing the directory
        bashproc.sendline('cd '+os.getcwd())

        #note: for all instances except the first, we add a delay to ensure the first one will start first
        #and avoid conflicts with the creation of several instances at once
        if epoch_id!=0:
            time.sleep(2)

        if container_mode=='python':

            # default path for the container
            if container == 'default':
                container_use = sys.executable
            else:
                container_use = container

            if heasoft_init_alias != '':
                bashproc.sendline(heasoft_init_alias)

            #running the script with the arguments (in this case container is the path of the python executable)
            bashproc.sendline(container+' '+linedet_runner_path+
                              ' -epoch_id '+str(epoch_id)+
                              ' -arg_dict_path '+str(arg_dict_path))

            bashproc.expect(['linedet_runner complete'],timeout=None)

        elif container_mode=='singularity':
            # default path for the container
            if container == 'default':
                container_use = os.environ['heasoft_singularity']
            else:
                container_use = container

            # testing the whether already exists
            singul_list = str(subprocess.check_output("singularity instance list", shell=True)).split('\\n')

            singul_list_mask = [elem.startswith(singularity_instance_name) for elem in singul_list]

            if sum(singul_list_mask) == 0:
                # calling the docker with no mounts
                print("no running xstar singularity instance detected. Creating it...")

                #note: doing it via the spawn because it doesn't work with subprocess for some reason
                instance_create_line=' '.join(['singularity', 'instance', 'start', '--bind', os.getcwd() + ':/mnt',
                                                   container_use,singularity_instance_name])

                bashproc.sendline(instance_create_line)

                instance_create_code=bashproc.expect(['instance started successfully','FATAL'])

                if instance_create_code>0:
                    raise ValueError('Error: failed to start singularity instance')


            #starting a shell inside the instance
            bashproc.sendline('singularity shell instance://'+singularity_instance_name)

            #moving inside the right directory
            bashproc.sendline('cd /mnt')

            #and running the python script
            bashproc.sendline('python '+linedet_runner_path+
                              ' -epoch_id '+str(epoch_id)+
                              ' -arg_dict_path '+str(arg_dict_path))

            instance_run_code=bashproc.expect(['linedet_runner complete','(Pdb)','Aborted (core dumped)'],timeout=None)

            if instance_run_code ==1:
                breakpoint()

            if instance_run_code > 1:
                raise ValueError('Error while running singularity instance')

    #### line detections for exposure with a spectrum
    if parallel==1:
        for epoch_id,epoch_files in enumerate(epoch_list):
            if force_instance:
                linedet_subprocess(epoch_id,arg_dict_path)
            else:
                linedet_loop_single(epoch_id,arg_dict)

    else:
        res = Parallel(n_jobs=parallel)(
            delayed(linedet_subprocess)(
                epoch_id=epoch_id,
                arg_dict_path=arg_dict_path)

            for epoch_id in range(len(epoch_list)))

    #if necessary, killing the singularity instance now thart the process is finished
    if container_mode=='singularity' and parallel!=1 or force_instance:
        # stopping the instance at the end
        subprocess.call(['singularity', 'instance', 'stop', singularity_instance_name])

    assert not compute_highflux_only,'Stopping the computation here because no need to rebuild the summaries for this'

    #not creating the recap file in spread comput mode to avoid issues
    assert spread_comput==1, 'Stopping the computation here to avoid conflicts when making the summary'

    #loading the diagnostic messages after the analysis has been done
    if os.path.isfile(os.path.join(outdir,'summary_line_det.log')):
        with open(os.path.join(outdir,'summary_line_det.log')) as sumfile:
            glob_summary_linedet=sumfile.readlines()[1:]

    #creating summary files for the rest of the exposures
    lineplots_files=[elem.split('/')[1] for elem in glob.glob(outdir+'/*',recursive=True)]

    if sat_glob=='XMM':
        aborted_epochs=[epoch for epoch in epoch_list if not epoch[0].split('_sp')[0]+'_recap.pdf' in lineplots_files]

        aborted_files=[epoch for epoch in epoch_list if not epoch[0].split('_sp')[0]+'_recap.pdf' in lineplots_files]

    elif sat_glob in ['Chandra','Swift']:
        aborted_epochs=[[elem.replace('_grp_opt'+('.pi' if sat_glob=='Swift' else '.pha'),'') for elem in epoch]\
                        for epoch in epoch_list if not epoch[0].split('_grp_opt'+('.pi' if sat_glob=='Swift' else '.pha'))[0]+'_recap.pdf'\
                            in lineplots_files]

        aborted_files=[epoch for epoch in epoch_list if\
                       not epoch[0].split('_grp_opt'+('.pi' if sat_glob=='Swift' else '.pha'))[0]+'_recap.pdf'\
                            in lineplots_files]
    elif sat_glob in ['NICER','NuSTAR']:

        sp_suffix='_sp_grp_opt.pha' if sat_glob=='NICER' else '_sp_src_grp_opt.pha'
        #updated with shorten_epoch
        epoch_ids=[[elem.replace(sp_suffix,'') for elem in epoch] for epoch in epoch_list]

        aborted_epochs=[elem_epoch_id for elem_epoch_id in epoch_ids if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

        aborted_files=[elem_epoch for elem_epoch,elem_epoch_id in zip(epoch_list,epoch_ids) if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

    elif sat_glob=='Suzaku':
        #updated with shorten_epoch
        epoch_ids=[[elem.replace('_grp_opt.pha','').replace('_src_dtcor','').replace('_gti_event_spec_src','')\
                    for elem in epoch] for epoch in epoch_list]

        aborted_epochs=[elem_epoch_id for elem_epoch_id in epoch_ids if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

        aborted_files=[elem_epoch.tolist() for elem_epoch,elem_epoch_id in zip(epoch_list,epoch_ids) if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

    #for now should be fine because for now we won't multi on stuff that's too weak to be detected
    elif sat_glob=='multi':
        aborted_files=[]
        aborted_epochs=[]
        breakpoint()

    arg_dict['glob_summary_linedet']=glob_summary_linedet

    if write_aborted_pdf:
        for elem_epoch_files in aborted_files:

            #list conversion since we use epochs as arguments
            pdf_summary(elem_epoch_files,arg_dict=arg_dict)

            #not used for now
            # if sat_glob=='XMM':
            #     epoch_observ=[elem_file.split('_sp')[0] for elem_file in elem_epoch]
            # elif sat_glob in ['Chandra','Swift']:
            #     epoch_observ=[elem_file.split('_grp_opt')[0] for elem_file in elem_epoch]
            # elif sat_glob=='NICER':
            #     epoch_observ=[elem_file.split('_sp_grp_opt')[0] for elem_file in elem_epoch]
            #
            # elif sat_glob=='Suzaku':
            #     if megumi_files:
            #         epoch_observ=[elem_file.split('_src')[0].split('_gti')[0] for elem_file in elem_epoch]

    return aborted_epochs

def linedet_loop_single(epoch_id,arg_dict):

        #importing in the function so this code can be used without xspec (useful for instance computations)
        from line_detect import line_detect

        epoch_list=arg_dict['epoch_list']
        epoch_files=epoch_list[epoch_id]
        bad_flags=arg_dict['bad_flags']
        megumi_files=arg_dict['megumi_files']
        epoch_restrict=arg_dict['epoch_restrict']

        restrict=arg_dict['restrict']
        outdir = arg_dict['outdir']
        sat_glob = arg_dict['sat_glob']
        started_expos=arg_dict['started_expos']
        skip_started=arg_dict['skip_started']
        epoch_list_started=arg_dict['epoch_list_started']
        skip_complete=arg_dict['skip_complete']
        done_expos=arg_dict['done_expos']
        overwrite=arg_dict['overwrite']
        log_console=arg_dict['log_console']
        catch_errors=arg_dict['catch_errors']
        pdf_only=arg_dict['pdf_only']
        summary_header=arg_dict['summary_header']

        #bad spectrum prevention

        for i_sp,elem_sp in enumerate(epoch_files):
            if elem_sp in bad_flags:
                print('\nSpectrum previously set as bad. Skipping the spectrum...')
                epoch_files=epoch_files[:i_sp]+epoch_files[i_sp+1:]

        #we use the id of the first file as an identifier
        firstfile_id=epoch_files[0].split('_sp')[0]

        if sat_glob=='Suzaku' and megumi_files:
            file_ids = [elem.split('_spec')[0].split('_src')[0] for elem in epoch_files]
        else:
            file_ids=[elem.split('_sp')[0] for elem in epoch_files]

        short_epoch_id='_'.join(shorten_epoch(file_ids))

        if restrict and epoch_restrict!=['']:

            if sat_glob in ['NICER','Suzaku']:
                if short_epoch_id not in epoch_restrict:
                    print(short_epoch_id)
                    print('\nRestrict mode activated and at least one spectrum not in the restrict array')
                    return
            else:
                if len([elem_sp for elem_sp in epoch_files if elem_sp not in epoch_restrict])\
                        >max(len(epoch_files)-len(epoch_restrict),0):
                    print('\nRestrict mode activated and at least one spectrum not in the restrict array')
                    return

        #skip start check
        if sat_glob in ['Suzaku']:

            sp_epoch=[elem_sp.split('_spec')[0].split('_src')[0] for elem_sp in epoch_files]

            started_epochs=[literal_eval(elem.split(']')[0]+(']')) for elem in started_expos]

            if skip_started and sp_epoch in started_epochs:
                 print('\nSpectrum analysis already performed. Skipping...')
                 return

        elif sat_glob=='Swift':
            if skip_started and sum([[elem] in epoch_list_started for elem in epoch_files])>0:
                print('\nSpectrum analysis already performed. Skipping...')
                return

        elif sat_glob in ['Chandra','XMM']:

            if (skip_started and len([elem_sp for elem_sp in epoch_files[:1] if elem_sp.split('_sp')[0] not in started_expos])==0) or \
               (skip_complete and len([elem_sp for elem_sp in epoch_files[:1] if elem_sp.split('_sp')[0] not in done_expos])==0):

                print('\nSpectrum analysis already performed. Skipping...')
                return

        elif sat_glob in ['NICER','NuSTAR','multi']:

            if (skip_started and shorten_epoch(file_ids) in started_expos) or \
               (skip_complete and shorten_epoch(file_ids) in done_expos):

                print('\nSpectrum analysis already performed. Skipping...')
                return

        #overwrite check
        if not overwrite:
            pdf_name=os.path.join(outdir, short_epoch_id+'_recap.pdf')

            if os.path.isfile(pdf_name):
                print('\nLine detection already computed for this exposure (recap PDF exists). Skipping...')
                return

        #we don't use the error catcher/log file in restrict mode to avoid passing through bpoints
        if not restrict:

            if log_console:
                prev_stdout=sys.stdout
                log_text=open(outdir+'/'+epoch_files[0].split('_sp')[0]+'_terminal_log.log')

            if catch_errors:
                try:
                    #main function
                    summary_lines=line_detect(epoch_id,arg_dict)

                except:
                    summary_lines=['unknown error']
            else:

                summary_lines=line_detect(epoch_id,arg_dict)

            if log_console:
                sys.stdout=prev_stdout
                log_text.close()

            #0 is the default value for skipping overwriting the summary file
            #note: we don't overwrite the summary file in pdf_only for now
            if summary_lines is not None and not pdf_only:
                #creating the text of the summary line for this observation


                # if '_' in firstfile_id:
                #     obsid_id=firstfile_id[:firstfile_id.find('_')]
                #     file_id=firstfile_id[firstfile_id.find('_')+1:]
                # else:
                #     obsid_id=firstfile_id
                #     file_id=obsid_id

                if sat_glob=='Suzaku' and megumi_files:
                    epoch_files_suffix=np.unique([elem.split('_spec')[-1].split('_pin')[-1] for elem in epoch_files])
                    epoch_files_suffix=epoch_files_suffix[::-1].tolist()
                else:
                    epoch_files_suffix=np.unique([elem.split('_sp')[-1]for elem in epoch_files]).tolist()

                epoch_files_str=epoch_files_suffix

                if len(np.unique(summary_lines))==1:
                    summary_lines_use=summary_lines[0]
                else:
                    summary_lines_use=summary_lines.tolist()

                summary_content=str(shorten_epoch(file_ids))+'\t'+str(epoch_files_suffix)+'\t'+str(summary_lines_use)

                #adding it to the summary file
                file_edit(outdir+'/'+'summary_line_det.log',
                          str(shorten_epoch(file_ids))+'\t'+str(epoch_files_suffix),summary_content+'\n',summary_header)

        else:
            summary_lines=line_detect(epoch_id,arg_dict)


def make_linedet_parfile(parallel,outdir,cont_model,autofit_model='lines_narrow',
                        container='default',
                        satellite='multi',group_max_timedelta='day',
                        skip_started=True,catch_errors=False,
                        multi_focus='NICER',nfakes=1000):

    '''
    Inserts a parfile for a linedet computation
    '''

    param_dict={'parallel':parallel,
                'outdir':outdir,
                'cont_model':cont_model,
                'autofit_model':autofit_model,
                'container':container,
                'satellite':satellite,
                'group_max_timedelta':group_max_timedelta,
                'skip_started':skip_started,
                'catch_errors':catch_errors,
                'multi_focus':multi_focus,
                'nfakes':nfakes}

    parfile_name='./parfile'+'_outdir_'+outdir+'_satellite_'+satellite+'_cont_model_'+cont_model+'.par'
    #writing in the file
    with open(parfile_name,'w+')\
            as file:

        for param,val in zip(list(param_dict.keys()),list(param_dict.values())):
            file.write('\t'.join([param,str(val)])+'\n')

    return parfile_name

def make_linedet_script(startdir,cores,parfile_path,cpus=2,nodes=1,
                      walltime=300,mail="maxime.parra@univ-grenoble-alpes.fr"):

    '''
    Create standard oar script for ipag-calc computation

    core value should be set to the same value than the "parallel" number in make_linedet_parfile

    cpu/node value shouldn't be changed if in ipag-calc (all servers have two cpus,1 node)

    walltime is in hours

    to run:
        oarsub -p "host='ipag-calcX'" interrupt=0 $(pwd)/oar_script.sh
    ex:
        oarsub -p "host='ipag-calc2'" $(pwd)/oar_script.sh

    '''

    wall_h='%02.f'%(int(walltime))
    wall_m = '%02.f' % (int((walltime-int(walltime))*60))

    script_str=\
    "#OAR -l /nodes="+str(nodes)+"/cpu="+str(cpus)+"/core="+str(cores)+\
    ",walltime="+wall_h+":"+wall_m+":00\n"+\
    "#OAR --stdout "+startdir+".%jobid%.out\n"+\
    "#OAR --stderr "+startdir+".%jobid%.err\n"+\
    "#OAR --notify mail:"+mail+"\n"+\
    "shopt -s expand_aliases\n"+\
    "source /user/home/parrama/.bashrc\n"+\
    "\npyload_3.9"+\
    "\npyloadenv_linedet\n"+\
    "\ncd "+startdir+"\n"+\
    "\npython $linedet_script -parfile '"+parfile_path+"'"

    with open('./oar_script.sh','w+') as oar_file:
        oar_file.write(script_str)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

note: easy test on ipag_calc with:
 oarsub -I -p "host='ipag-calc2'"

note: currently the prefixes for job in oar scripts don't work
use full inline syntax like:

oarsub -p "host='ipag-calc3'" -l /nodes=1/core=2,walltime=256:00:00 ./oar_script.sh

'''
import os

import numpy as np
import pexpect
import glob

from solution_tools import sample_angle

compton_thick_thresh=1.5e24

# ! light speed in Km/s unit
c_Km = 2.99792e5
# ! light speed in cm/s unit
c_cgs = 2.99792e10
sigma_thomson_cgs = 6.6525e-25
c_SI = 2.99792e8
G_SI = 6.674e-11
Msol_SI = 1.98892e30
PI = 3.14159265
Km2m = 1000.0
m2cm = 100.0

def download_mantis(mantis_path,local_folder,load_folder=False):

    '''
    copies a path into a mantis folder

    if delete_previous is set to True, the path is supposed to be the one of the incident spectrum,
    and the previous incident spectrum is deleted if it is not a final box incident

    note: the mantis folder should be a relative path after mantis/userid/

    We can't directly write into mantis so we can't file_edit directly into it, instead we straight up replace the logpar file at every box

    all the paths should be relative and will be changed to absolute paths for the conversion
    '''

    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')

    # loading the file or directory
    irods_proc.sendline('iget '+(' -r' if load_folder else '')+ mantis_path + ' ' + os.path.join(os.getcwd(),local_folder))

def upload_mantis(path,mantis_folder,delete_previous=False):

    '''
    copies a path into a mantis folder

    if delete_previous is set to True, the path is supposed to be the one of the incident spectrum,
    and the previous incident spectrum is deleted if it is not a final box incident

    note: the mantis folder should be a relative path after mantis/userid/

    We can't directly write into mantis so we can't file_edit directly into it, instead we straight up replace the logpar file at every box

    all the paths should be relative and will be changed to absolute paths for the conversion
    '''

    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')

    # uploading the file
    irods_proc.sendline('iput ' + os.path.join(os.getcwd(), path) + ' ' + mantis_folder)

    #deleting the previous box spectrum if not in a final box
    if delete_previous:
        i_box=path.split('_')[-1].split('.')[0]
        previous_box_incident=os.path.join(os.getcwd(),path.replace(path.split('_')[-1],path.split('_')[-1].replace(i_box,str(int(i_box)-1))))
        irods_proc.sendline('-irm '+previous_box_incident+' '+mantis_folder)

def create_grid(grid_name, mhd_solutions_path,
                 SED,m_BH,rj,mdot='auto',xlum=None,
                 param_mode='split_angle',
                 h_over_r_vals=None,p_vals=None,mu_vals=None,angle_vals=None):

    '''

    Setups grid or cigrid computations by creating a proper folder tree in the current directory

    use exemple:
    create_grid('grid_groj','nathan_new_ep01.txt','5461_cont_deabs_fully_extended_xstar.txt',
                5.4,6,xlum=0.4167,angle_vals='range_30_80_4')

    grid should always begin with the world "grid"
    Main args:
        grid_name: name of grid aka folder tree where the arborescence will be created
        mhd_solutions_path: path where to find the mhd solutions to sample from

        param_mode:
                -all: computes the spectra for every single mhd solution in the file without decomposing
                -split_angle: considers all solutions but only in a given range of angles
                -decompose: combines the individual parameter space asked for every value, among the nearest available in the
                solution file (nearest for h_over_r_vals, 2-dimension nearest in log distance in p/mu and nearest again in angle)
                creates intermediary folders to reflect the dimensions

        coupled parameter arguments (should all be the same length or single to indicate one value to be repeated)
            SED:list of SED paths (will be copied to the grid directory and later to the save directory for use)
            mdot: mdot OBSERVED (natural units)
                    can be set to auto to fetch it automatically from xlum using the black hole mass
            xlum: Xray luminosities (1e38 erg/s in 1-1000 Ryd)
                    required ony if mdot is set to auto
            m_BH: black hole mass (M_sol)
            rj: JED-SAD transition radius. Double use:
                -acts as the starting point for the nH computation to get the angle thresholds
                -acts as the minimal ro for the xstar_wind computation


        decoupled parameter arguments for 'decompose' mode (one MHD solution for each combination):

            input possibles are either list/array like elements or str of type 'log/lin_valmin/valmax_nelements'
            (with valmin and valmax included) values should be in logspace for log type
            angle is the only variable which only accepts lin because it doesn't make sense otherwise

            h_over_r_vals: values for the aspect ratio
            p_vals: values for the ejection index
            mu_vals: values for the magnetisation
            angle_vals: values for the angle. LINEAR if interval
                        Note:
                        The max value will always be restricted to the maximal value possible to remain
                        non-compton thick in the solution. Putting 90 as the max is equivalent to fetching this

    '''

    '''
    creating the grid folder if necessary
    '''

    os.system('mkdir -p '+grid_name)

    '''
    decomposing the values if in interval form
    '''

    if param_mode=='decompose':
        #decomposing the required parameters
        if type(h_over_r_vals) in (list,np.ndarray):
            h_over_r_list=h_over_r_vals
        elif type(h_over_r_vals)==str:
            h_over_r_infos=h_over_r_vals.split('_')
            if h_over_r_infos[0]=='log':
                h_over_r_list=np.logspace(float(h_over_r_infos[1]),float(h_over_r_infos[2]),int(h_over_r_infos[3]))
            elif h_over_r_infos[0]=='lin':
                h_over_r_list = np.linspace(float(h_over_r_infos[1]),float(h_over_r_infos[2]),int(h_over_r_infos[3]))

        if type(p_vals) in (list,np.ndarray):
            p_list=p_vals
        elif type(p_vals)==str:
            p_infos=p_vals.split('_')
            if p_infos[0] == 'log':
                p_list = np.logspace(float(p_infos[1]), float(p_infos[2]),
                                            int(p_infos[3]))
            elif p_infos[0] == 'lin':
                p_list = np.linspace(float(p_infos[1]), float(p_infos[2]),
                                            int(p_infos[3]))

        if type(mu_vals) in (list,np.ndarray):
            mu_list=mu_vals
        elif type(mu_vals)==str:
            mu_infos=mu_vals.split('_')
            if mu_infos[0]=='log':
                mu_list=np.logspace(float(mu_infos[1]),float(mu_infos[2]),int(mu_infos[3]))
            elif mu_infos[0]=='lin':
                mu_list = np.linspace(float(mu_infos[1]),float(mu_infos[2]),int(mu_infos[3]))

    if param_mode in ['all','split_angle']:

        if type(angle_vals) in (list,np.ndarray):
            angle_list=angle_vals

        elif type(angle_vals)==str:
            angle_infos=angle_vals.split('_')
            if angle_infos[0]=='range':
                angle_list = np.arange(float(angle_infos[1]), float(angle_infos[2])+0.000001, float(angle_infos[3]))
            elif angle_infos[0]=='log':
                angle_list=np.logspace(float(angle_infos[1]),float(angle_infos[2]),int(angle_infos[3]))
            elif angle_infos[0]=='lin':
                angle_list = np.linspace(float(angle_infos[1]),float(angle_infos[2]),int(angle_infos[3]))


    #fetching the mdot from the luminosity if asked to

    '''
    converting the SED variables into lists
    '''

    SED_var_list=[SED,mdot,m_BH,rj,xlum]
    SED_list_var_list=[]

    #checking if any is already like that
    list_SED_vars=[elem for elem in SED_var_list if type(elem) in [np.ndarray,list,tuple]]

    #and deducting the desired lengths
    len_SED_vars=1 if len(list_SED_vars)==0 else len(list_SED_vars[0])

    #looping in all variables variables
    for SED_type_var in SED_var_list:
        if type(SED_type_var) not in [np.ndarray,list,tuple]:

            SED_list_var_list+=[np.repeat(SED_type_var,len_SED_vars)]
        else:
            assert  len(SED_type_var)==len_SED_vars,'Error: SED variables have different len'
            SED_list_var_list+=np.array(SED_type_var)

    #and outputing them in each variable name

    SED_arr,mdot_arr,m_BH_arr,rj_arr,xlum_arr=SED_list_var_list

    # #identifying the folder of the solution file
    # mhd_sol_dir = mhd_solutions_path[:mhd_solutions_path.rfind('/')]

    #creating the solution angle sampling for every solution

    SED_dirs=[]

    if param_mode=='split_angle':
        for elem_SED,elem_mdot,elem_m_BH,elem_rj,elem_xlum in zip(SED_arr,mdot_arr,m_BH_arr,rj_arr,xlum_arr):

            #the name of the grid is used as global directory
            #the name of the SED and other parameters will be used as the outdirs for the combination
            SED_dirs+=[os.path.join(grid_name,elem_SED.split('/')[-1][:elem_SED.split('/')[-1].rfind('.')]+\
                     '_mdot_'+str(elem_mdot)+('' if elem_mdot!='auto' else '_xlum_'+str(elem_xlum))+\
                                    '_m_bh_'+str(elem_m_BH)+'_rj_'+str(elem_rj))]

            #creating the grid in that folder
            #NOTE: should be updated for relativistic effects if going back to stop at compton=True

            sampl_grid_path=sample_angle(mhd_solutions_path,angle_list,elem_mdot,elem_m_BH,elem_rj,
                                         xlum=elem_xlum if elem_mdot=='auto' else None,
                                         outdir=SED_dirs[-1],
                                        return_file_path=True)

            sampl_grid_dir=sampl_grid_path[:sampl_grid_path.rfind('/')]

            #copying the SED in the folder
            os.system('cp '+elem_SED+' '+SED_dirs[-1])

            solutions_grid = np.loadtxt(sampl_grid_path)

            with open(sampl_grid_path) as grid_file:
                solution_header=grid_file.readlines()[0]

            #decomposing the folder tree for each solution
            for solution in solutions_grid:

                #note: no decomposition for the turbulence as of now
                dir_sol=os.path.join('eps_%.02f'%solution[0],
                                     'n_island_'+str(int(solution[1])),
                                     'p_%.04f'%solution[2]+'_mu_%.04f'%solution[3],
                                     'angle_'+str(round((solution[8]),1)))

                #creating the directory
                os.system('mkdir -p '+os.path.join(sampl_grid_dir,dir_sol))

                #making a 2-d array to save the solution in columns
                sol_save=np.array([solution])
                #saving the single line with the header in the newly created folder
                np.savetxt(os.path.join(sampl_grid_dir,dir_sol,dir_sol.replace('/','_')+'.txt'),sol_save,
                           header=solution_header[1:].replace('\n',''))


def create_oar_script(grid_folder,parfile,cores,cpus=2,nodes=1,
                      walltime=72,mail="maxime.parra@univ-grenoble-alpes.fr"):

    '''
    Create standard oar script for non-cigrid computations

    cpu value shouldn't be changed if in ipag-calc (all servers have two cpus)

    walltime is in hours

    parfile should be a relative path inside grid_folder


    '''

    #parfile_path=os.path.join(grid_folder,parfile)
    parfile_path=parfile

    wall_h='%02.f'%(int(walltime))
    wall_m = '%02.f' % (int((walltime-int(walltime))*60))

    script_str=\
    "#OAR -l /nodes="+str(nodes)+"/cpu="+str(cpus)+"/core="+str(cores)+\
    ",walltime="+wall_h+":"+wall_m+":00\n"+\
    "#OAR --stdout grid_folder.%jobid%.out\n"+\
    "#OAR --stderr grid_dolder.%jobid%.err\n"+\
    "#OAR --notify mail:"+mail+"\n"+\
    "shopt -s expand_aliases\n"+\
    "source /user/home/parrama/.bashrc\n"+\
    "\npyload"+\
    "\npyloadenv\n"+\
    "\npython $wind_runner -parfile "+parfile_path

    with open(os.path.join(grid_folder,'oar_script.sh'),'w+') as oar_file:
        oar_file.write(script_str)


def create_grid_parfile(grid_folder,save_grid_dir,sim_grid_dir,xlum,dr_r,
                        mode='server_standalone_default',v_resol=85.7,stop_d=1e6,progress_file='default'):

    '''
    Inserts a parfile inside an already existing grid folder structure

    exemple:
    create_grid_parfile('grid_groj_newrelat','/user/home/parrama/workdir/simu/xstar/grid/save',
    '/user/home/parrama/workdir/simu/xstar/grid/sim',xlum=0.4167,dr_r=0.05,mode='server_singularity_default')


    the list of parameters is the list of arguments of oar_wrapper

    save_grid_dir and sim_grid_dir are global folders, to which the gird_folder arborescence will be added

    mode: changes the computation behavior: 'type_xstaruse_xstaruseid'

        type:
        -server: uses a standard save folder with normal copying commands
        -cigrid: uses mantis commands for the save folder. save_grid_dir is expected to be a mantis absolute path


        xstar use:
        -standalone: uses an xstar version directly installed within an heasoft folder.
        -docker: uses an xstar version installed in a docker. Uses smart copying to avoid necessiting permissions
                (besides the one to run the docker)
        -singularity: uses an xstar version installed in a singularity container.
        -charliecloud: uses an xstar version installed in a charliecloud environment.

        xstar use id:
        in standalone: the path of the heasoft version to use. 'default' uses the standard version installed on the computer
        in docker/charliecloud: the identifier of the container (not the image) to launch
        in singularity: the path of the container to launch. 'default' uses the path put within the xstar_singularity
                        environment variable

    progress_file: location of the file where the progresses of the xstar computation are stored
                    if set to default, saves into a file named grid_progress.log in the main grid directory

    note: there is a line in oar_wrapper for different sed extensions. Might need to be added here in the future

    the default resolution value of 85.7 matches a XRISM resolution of 6eV with an oversampling of 3
    '''

    sol_files = glob.glob(grid_folder+'/**/eps**.txt', recursive=True)

    sol_files.sort()

    #creating an array with the argument elements for each solution

    parameters=np.array([None]*len(sol_files))

    for i_sol,elem_sol_file in enumerate(sol_files):

        sol_rel_dir=elem_sol_file[:elem_sol_file.rfind('/')]

        #fetching the parameters in the SED
        sed_file_pars=sol_rel_dir.split('/')[1][sol_rel_dir.split('/')[1].rfind('_mdot')+1:]

        sol_mdot=sed_file_pars.split('_')[1]

        #defining the outdir for xstar inside the computation directory with the grid's folder structure
        sol_outdir_comput=os.path.join(sim_grid_dir,elem_sol_file[:elem_sol_file.rfind('/')])

        sol_m_bh=sed_file_pars.split('m_bh_')[1].split('_')[0]

        #using rj here
        sol_ro=sed_file_pars.split('_rj_')[1].split('_')[0]

        #direct names of the xstar_wind function arguments.
        parameters[i_sol]={'solution_rel_dir':sol_rel_dir,
                           'save_grid_dir':save_grid_dir,
                           'comput_grid_dir':sim_grid_dir,

                           'mdot_obs':sol_mdot,
                           'xlum':str(xlum),
                           'm_BH':sol_m_bh,

                           'ro_init':sol_ro,
                           'dr_r':str(dr_r),
                           'stop_d_input':str(stop_d),
                           'v_resol': str(v_resol),
                           'mode':mode,
                           'progress_file':progress_file,
                           }

    #writing in the file
    with open(os.path.join(grid_folder,'parfile_xlum_'+str(xlum)+'_dr_r_'+str(dr_r)+'_v_resol_'+str(v_resol)+'_stop_d%.1e'%(stop_d)+'.par'),'w+')\
            as file:

        #writing the header
        file.write('#'+'\t'.join(list(parameters[0].keys()))+'\n')

        #writing the main lines
        file.writelines(['\t'.join(list(elem_dict.values()))+'\n' for elem_dict in parameters])


def setup_cigrid(mantis_grid_dir, silenus_grid_dir,cores,ex_time,priority):

    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')



    #zipping the files and then copying them

    # # loop in the SEDs:
    # for i_SED, (elem_SED, elem_mdot, elem_xlum) in enumerate(zip(SED_list, mdot_list, xlum_list)):
    #
    #     # creating a mantis subdirectory
    #     mantis_SED_dir = os.path.join(mantis_grid_dir, 'SED_' + str(i_SED + 1))
    #
    #     irods_proc.sendline('imkdir ' + mantis_SED_dir)
    #
    #     # copying the SED to mantis
    #     irods_proc.sendline('iput ' + elem_SED + ' ' + mantis_SED_dir)
    #
    #     if param_mode == 'all':
    #         # read the global sol file
    #
    #         # main parameter loop
    #         for param_comb, mhd_line in zip(param, mhd_lines):
    #             param_comb_dirname = "ha"
    #
    #             # creating the directory
    #             irocs_proc.sendline('imkdir ' + os.path.join(mantis_SED_dir, param_comb_dirname))
    #
    #             # creating a file for the individual solution (will be moved, defaulted to the folder where
    #             # the mhd solution is)
    #
    #             file_indiv = os.path.join(mhd_sol_dir, param_comb)
    #             with open(file_indiv, 'w+') as f_sol:
    #                 f_sol.write(mhd_line)
    #
    #             # creating the directory in mantis
    #             mantis_sol_dir = os.path.join(mantis_SED_dir, param_comb)
    #             irods_proc.sendline('imkdir ' + mantis_sol_dir)
    #
    #             # putting the solution file inside
    #             irods_proc.sendline('iput ' + os.path.join(mantis))
    #
    #
    #
    #     elif param_mode == 'decompose':
    #         for i_h_r, elem_h_r in enumerate(h_over_r_list):
    #             for i_p, elem_p in enumerate(p_list):
    #                 for i_mu, elem_mu in enumerate(mu_list):
    #                     for i_angle, elem_angle in enumerate(angle_list):
    #                         # implement nearest fetch in the solution file and folder decomposition
    #                         pass


    # jdl_path=os.path.join(mhd_sol_dir,'xstar_grid_'+param_mode+'_'+
    #                       (mhd_solutions_path.split('/')[-1] if param_mode=='all' else n_sol+'.jdl'))
    #
    # template='''
    #         {
    #           "name": "'''+'xstar_grid_'+param_mode+'_'+\
    #                       (mhd_solutions_path.split('/')[-1] if param_mode=='all' else n_sol)+'''",
    #           "resources": "/core='''+str(cores)+'''",
    #           "exec_file": "{HOME}/povray/start.bash",
    #           "exec_directory": "{HOME}/povray",
    #           "param_file": "{HOME}/povray_params.txt",
    #           "test_mode": "true",
    #           "type": "best-effort",
    #           "prologue": [
    #             "set -e",
    #             "source /applis/site/guix-start.sh",
    #             "cd {HOME}",
    #             "imkdir -p povray_results/{CAMPAIGN_ID}",
    #             "secure_iget -r -f povray",
    #             "chmod u+x {HOME}/povray/start.bash"
    #           ],
    #           "clusters": {
    #             "luke": {
    #               "project": "formation-ced-calcul",
    #               "walltime": "00:15:00"
    #             },
    #             "dahu": {
    #               "project": "formation-ced-calcul",
    #               "walltime": "00:10:00"
    #             },
    #             "bigfoot": {
    #               "project": "formation-ced-calcul",
    #               "walltime": "00:10:00",
    #               "properties": "gpumodel!='T4'"
    #             }
    #           }
    #         }
    #         '''

    # with open(jdl_path) as file:
    #     pass


    '''
    copying the arborescence to mantis
    '''


    # #creating the global mantis directory
    # irods_proc.sendline('imkdir '+mantis_grid_dir)



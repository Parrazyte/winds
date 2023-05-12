#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from astropy.io import ascii

import numpy as np
import pexpect
import time
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


def setup_cigrid(mantis_grid_dir,silenus_grid_dir, mhd_solution_path,param_mode,cores,ex_time,priority,
                 dr_r,v_resol,ro_init,stop_d,
                 SED_list,mdot_list,xlum_list,
                 h_over_r_vals=None,p_vals=None,mu_vals=None,angle_vals=None,
                 local=False):

    '''

    Setups Cigrid computations by creating a proper mantis folder tree, and associated launching script and parameter file

    Args:
        mantis_grid_dir:
        local_grid_dir:
        cores:
        ex_time:
        priority:
        local:creates the mantis directories, script and parameter files locally using the mantis_grid_dir instead (used for testing)

    Parameter arguments:
    global arguments (a single per run)
        dr_r
        v_resol
        ro_init
        stop_d
        mode:
            -all: computes the spectra for every single mhd solution in the file without decomposing
            -decompose: combines the individual parameter space asked for every value, among the nearest available in the
            solution file (nearest for h_over_r_vals, 2-dimension nearest in log distance in p/mu and nearest again in angle)
            creates intermediary folders to reflect the dimensions

    coupled parameter arguments (should all be the same length)
        SED_list:list of SED paths (will be copied to mantis_grid_dir for use)
        mdot_list: list of mdots
        xlum_list: list of xray luminosities



    decoupled parameter arguments for 'decompose' mode (one MHD solution for each combination)

    input possibles are either list/array like elements or str of type 'log/lin_valmin/valmax_nelements'
    (with valmin and valmax included). values should be in logspace for log type

        h_over_r_vals: values for the aspect ratio
        p_vals: values for the ejection index
        mu_vals: values for the magnetisation
        angle_vals: values for the angle
    '''

    if param_mode=='decompose':
        #decomposing the required parameters
        if type(h_over_r_vals) in (list,np.ndarray):
            h_over_r_list=h_over_r_vals
        elif type(h_over_r_vals)==str:
            h_over_r_infos=h_over_r_vals.split
            if h_over_r_infos[0]=='log':
                h_over_r_list=np.logspace(h_over_r_infos[1],h_over_r_infos[2],h_over_r_infos[3])
            elif h_over_r_infos[0]=='lin':
                h_over_r_list = np.linspace(h_over_r_infos[1], h_over_r_infos[2], h_over_r_infos[3])
                
        if type(p_vals) in (list,np.ndarray):
            p_list=p_vals
        elif type(p_vals)==str:
            p_infos=p_vals.split
            if p_infos[0]=='log':
                p_list=np.logspace(p_infos[1],p_infos[2],p_infos[3])
            elif p_infos[0]=='lin':
                p_list = np.linspace(p_infos[1], p_infos[2], p_infos[3])
                
        if type(mu_vals) in (list,np.ndarray):
            mu_list=mu_vals
        elif type(mu_vals)==str:
            mu_infos=mu_vals.split
            if mu_infos[0]=='log':
                mu_list=np.logspace(mu_infos[1],mu_infos[2],mu_infos[3])
            elif mu_infos[0]=='lin':
                mu_list = np.linspace(mu_infos[1], mu_infos[2], mu_infos[3])
                
        if type(angle_vals) in (list,np.ndarray):
            angle_list=angle_vals
        elif type(angle_vals)==str:
            angle_infos=angle_vals.split
            if angle_infos[0]=='log':
                angle_list=np.logspace(angle_infos[1],angle_infos[2],angle_infos[3])
            elif angle_infos[0]=='lin':
                angle_list = np.linspace(angle_infos[1], angle_infos[2], angle_infos[3])
        
    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')

    #creating the global mantis directory
    irods_proc.sendline('imkdir '+mantis_grid_dir)


    #reading the mhd solution file
    with open(mhd_solution_path) as glob_mhd_file:
        mhd_lines=glob_mhd_file.readlines()

    if param_mode=='all':
        n_sol=mhd_lines.readlines()
    else:
        n_sol=len(h_over_r_list)*len(p_list)*len(mu_list)*len(angle_list)

    #putting the scripts inside
    mhd_sol_dir = mhd_solution_path[:mhd_solution_path.rfind('/')]

    jdl_path=os.path.join(mhd_sol_dir,'xstar_grid_'+param_mode+'_'+
                          (mhd_solution_path.split('/')[-1] if param_mode=='all' else n_sol+'.jdl'))

    template='''
            {
              "name": "'''+'xstar_grid_'+param_mode+'_'+\
                          (mhd_solution_path.split('/')[-1] if param_mode=='all' else n_sol)+'''",
              "resources": "/core='''+str(cores)+'''",
              "exec_file": "{HOME}/povray/start.bash",
              "exec_directory": "{HOME}/povray",
              "param_file": "{HOME}/povray_params.txt",
              "test_mode": "true",
              "type": "best-effort",
              "prologue": [
                "set -e",
                "source /applis/site/guix-start.sh",
                "cd {HOME}",
                "imkdir -p povray_results/{CAMPAIGN_ID}",
                "secure_iget -r -f povray",
                "chmod u+x {HOME}/povray/start.bash"
              ],
              "clusters": {
                "luke": {
                  "project": "formation-ced-calcul",
                  "walltime": "00:15:00"
                },
                "dahu": {
                  "project": "formation-ced-calcul",
                  "walltime": "00:10:00"
                },
                "bigfoot": {
                  "project": "formation-ced-calcul",
                  "walltime": "00:10:00",
                  "properties": "gpumodel!='T4'"
                }
              }
            }
            '''

    # with open(jdl_path) as file:
    #
    # #loop in the SEDs:
    # for i_SED,(elem_SED,elem_mdot,elem_xlum) in enumerate(zip(SED_list,mdot_list,xlum_list)):
    #
    #     #creating a mantis subdirectory
    #     mantis_SED_dir=os.path.join(mantis_grid_dir,'SED_'+str(i_SED+1))
    #
    #     irods_proc.sendline('imkdir '+mantis_SED_dir)
    #
    #     #copying the SED to mantis
    #     irods_proc.sendline('iput '+elem_SED+' '+mantis_SED_dir)
    #
    #     if param_mode=='all':
    #         #read the global sol file
    #
    #         #main parameter loop
    #         for param_comb,mhd_line in zip(param,mhd_lines):
    #
    #             param_comb_dirname="ha"
    #
    #             #creating the directory
    #             irocs_proc.sendline('imkdir '+os.path.join(mantis_SED_dir,param_comb_dirname))
    #
    #             #creating a file for the individual solution (will be moved, defaulted to the folder where
    #             # the mhd solution is)
    #
    #             file_indiv=os.path.join(mhd_sol_dir,param_comb)
    #             with open(file_indiv,'w+') as f_sol:
    #                 f_sol.write(mhd_line)
    #
    #             #creating the directory in mantis
    #             mantis_sol_dir=os.path.join(mantis_SED_dir,param_comb)
    #             irods_proc.sendline('imkdir '+mantis_sol_dir)
    #
    #             #putting the solution file inside
    #             irods_proc.sendline('iput '+os.path.join(mantis))
    #
    #
    #
    #     elif param_mode=='decompose':
    #         for i_h_r,elem_h_r in enumerate(h_over_r_list):
    #             for i_p, elem_p in enumerate(p_list):
    #                 for i_mu, elem_mu in enumerate(mu_list):
    #                     for i_angle, elem_angle in enumerate(angle_list):
    #
    #                         #implement nearest fetch in the solution file and folder decomposition
    #                         pass





#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from astropy.io import ascii

from simul_tools import xstar_wind


import numpy as np
import pexpect
import time
def load_mantis(mantis_path,local_folder,load_folder=False):

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

def setup_cigrid(mantis_grid_dir,local_grid_dir,cores,ex_time,priority,local=False,
                 dr_r,v_resol,ro_init,stop_d,
                 SED_list,mdot_list,xlum_list,
                 h_over_r_vals,p_vals,mu_vals,angle_vals,):

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

    coupled parameter arguments (should all be the same length)
        SED_list:list of SED paths (will be copied to mantis_grid_dir for use)
        mdot_list:
        xlum_list
        
    decoupled parameter arguments (one MHD solution for each combination)
        h_over_r_vals:
        p_vals:
        mu_vals:
        angle_vals:
    '''

def cigri_wrapper(mantis_dir,SED_mantis_path,solution_mantis_path,p_mhd,mdot_obs,stop_d_input,xlum,outdir,
                  h_over_r=0.1, ro_init, dr_r, v_resol, m_BH):

    '''

    wrapper for cigri grid xstar computations

    Args:
        mantis_dir: mantis directory, where the file are saved. will be created if necessary
        Note: the first parameter is used to define grid process names so it must be different for each process, hence why we put the directory, which are different for every solution

    outdir should be an absolute path in this case
    '''

    os.system('mkdir -p'+outdir)

    #copying all the initial files and the content of the mantis directory to the outdir
    load_mantis(mantis_dir,outdir,load_folder=True)
    load_mantis(SED_mantis_path,outdir)
    load_mantis(solution_mantis_path,outdir)

    #it's easier to go directly in the outdir here
    os.chdir(outdir)

    xstar_wind(solution,p_mhd,mdot_obs,stop_d_input,SED_path,xlum,outdir='./',h_over_r=h_over_r,ro_init=ro_init,dr_r=dr_r,v_resol=v_resol,m_Bh=m_BH,comput_mode='gricad',mantis_folder=mantis_dir)



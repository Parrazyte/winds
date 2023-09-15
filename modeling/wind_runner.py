#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from simul_tools import oar_wrapper
from multiprocessing import Pool
import subprocess

'''
Wrapper for a single/multiple solution computations in an oar environment. 

Can be called in a bash script created from create_oar_script, with the parameters from a parameter_file 
created with create_grid_parfile

Can also use the "parfile" argument to read the parfile and decompose all other arguments internally
In this case, the multiprocessing is done internally through a pool loop of the main oar_wrapper function

'''

ap = argparse.ArgumentParser(description='Wrapper for a single wind solution run in a oar environment.\n)')

#parfile mode (empty string means not using this mode)
ap.add_argument('-parfile',nargs=1,help="parfile to use instead of single solution",default='',
                type=str)

#paths
ap.add_argument("-solution_rel_dir",nargs=1,help="solution relative directory inside the grid structure",type=str,default='')

ap.add_argument("-save_grid_dir",nargs=1,help="save grid directory, where all the files will be saved",type=str,default='')

ap.add_argument("-comput_grid_dir",nargs=1,help="computation grid directory, where all the xstar computations will be run",
                                        type=str,default='')


#solution physical parameters

#note: using str here in case mdot_obs is set to auto
#the conversion is done inside oar_wrapper
ap.add_argument("-mdot_obs",nargs=1,help="SED parameter for xstar",type=str,default='')

ap.add_argument("-xlum",nargs=1,help="SED parameter for xstar",type=float,default=0)

ap.add_argument("-m_BH",nargs=1,help="SED parameter for xstar",type=float,default=0)

#computational parameters

ap.add_argument("-ro_init",nargs=1,help="box parameter for xstar",type=float,default=0)

ap.add_argument("-dr_r",nargs=1,help="box parameter for xstar",type=float,default=0)

ap.add_argument("-stop_d_input",nargs=1,help="box parameter for xstar",type=float,default=0)

ap.add_argument("-v_resol",nargs=1,help="box parameter for xstar",type=float,default=0)

ap.add_argument("-mode",nargs=1,help="computation mode for the save",type=str,default='')

ap.add_argument("-progress_file",nargs=1,help='global progress file where to store the box evolution',type=str,default='')

ap.add_argument("-save_inter_sp",nargs=1,help='save intermediary transmitted rest frame spectra',type=bool,default=True)

args=ap.parse_args()

parfile_path=args.parfile

solution_rel_dir=args.solution_rel_dir
save_grid_dir=args.save_grid_dir
comput_grid_dir=args.comput_grid_dir

mdot_obs=args.mdot_obs
xlum=args.xlum
m_BH=args.m_BH

ro_init=args.ro_init
dr_r=args.dr_r
stop_d_input=args.stop_d_input
stop_d_input=args.stop_d_input
v_resol=args.v_resol


mode=args.mode
progress_file=args.progress_file
save_inter_sp=args.save_inter_sp

if parfile_path!='':
    #loading the file as an array
    param_arr=np.loadtxt(parfile_path[0],dtype=str).T

    #converting in object to modify the types of the variables inside
    param_arr=param_arr.astype(object)
    #note: ununsed as of now since pool.starmap directly takes an iterable
    # #and decomposing the arguments (with some type conversions to get the floats whenever needed)
    # solution_rel_dir_arr,save_grid_dir_arr,comput_grid_dir_arr=param_arr.T[:4]
    # mdot_obs_arr,xlum_arr,m_BH_arr,ro_init_arr,dr_r_arr,stop_d_input_arr,v_resol_arr=param_arr.T[3:10].astype(float)
    # mode_arr=param_arr.T[10]

    #converting some arguments in floats before retransposing back
    for i_par in range(4,10):
        param_arr[i_par]=param_arr[i_par].astype(float)

    param_arr=param_arr.T

    #pool loop
    with Pool() as pool:
        pool.starmap(oar_wrapper,param_arr)

else:
    oar_wrapper(solution_rel_dir=solution_rel_dir,save_grid_dir=save_grid_dir,comput_grid_dir=comput_grid_dir,
            mdot_obs=mdot_obs,xlum=xlum,m_BH=m_BH,
            ro_init=ro_init,dr_r=dr_r,stop_d_input=stop_d_input,v_resol=v_resol,
            mode=mode,progress_file=progress_file,save_inter_sp=save_inter_sp)




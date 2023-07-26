#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from simul_tools import oar_wrapper

'''
Wrapper for a single solution computation in an oar environment. 
Will be called in a bash script created from setup_grid, with the parameters from a parameter_file 
created with create_grid_parfile
'''

ap = argparse.ArgumentParser(description='Wrapper for a single wind solution run in a oar environment.\n)')

ap.add_argument("-solution_rel_dir",nargs=1,help="solution relative directory inside the grid structure",type=str)

ap.add_argument("-save_grid_dir",nargs=1,help="save grid directory, where all the files will be saved",type=str)

ap.add_argument("-comput_grid_dir",nargs=1,help="computation grid directory, where all the xstar computations will be run",
                                        type=str)


#solution physical parameters
ap.add_argument("-mdot_obs",nargs=1,help="SED parameter for xstar",type=float)

ap.add_argument("-xlum",nargs=1,help="SED parameter for xstar",type=float)

ap.add_argument("-m_BH",nargs=1,help="SED parameter for xstar",type=float)

#computational parameters

ap.add_argument("-ro_init",nargs=1,help="box parameter for xstar",type=float)

ap.add_argument("-dr_r",nargs=1,help="box parameter for xstar",type=float)

ap.add_argument("-stop_d_input",nargs=1,help="box parameter for xstar",type=float)

ap.add_argument("-v_resol",nargs=1,help="box parameter for xstar",type=float)

ap.add_argument("-mode",nargs=1,help="computation mode for the save",type=str)

args=ap.parse_args()

solution_rel_dir=args.solution_rel_dir
save_grid_dir=args.save_grid_dir
comput_grid_dir=args.comput_grid_dir

mdot_obs=args.mdot_obs
xlum=args.xlum
m_BH=args.m_BH

ro_init=args.ro_init
dr_r=args.dr_r
stop_d_input=args.stop_d_input
v_resol=args.v_resol
mode=args.mode

oar_wrapper(solution_rel_dir=solution_rel_dir,save_grid_dir=save_grid_dir,comput_grid_dir=comput_grid_dir,
            mdot_obs=mdot_obs,xlum=xlum,m_BH=m_BH,
            ro_init=ro_init,dr_r=dr_r,stop_d_input=stop_d_input,v_resol=v_resol,
            mode=mode)
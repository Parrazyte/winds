#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:22:22 2023

@author: parrama
"""

from simul_tools import xstar_wind
import os

ep=0.1
p=0.103
mu=0.067

####This should be changed
mdot=0.111


####rechange p if necessaryn and m to 8

dict_sol={}

test_list=[3.836508e-01,  1.278251e+00, 2.100000e+01,  1.369094e+00,  4.065250e-01,  1.455833e-01,  1.048574e+00,  1.503721e-01,  3.953974e-01,
           -5.069130e-01,  4.084034e-01,  1.393493e+00,  1.278200e-02]

os.chdir('/media/parrama/SSD/Simu/MHD/xstar')

dict_sol['z_A']=test_list[0]
dict_sol['r_A']=test_list[1]
dict_sol['angle']=test_list[2]
dict_sol['func_Rsph_by_ro']=test_list[3]

dict_sol['rho_mhd']=test_list[4]

dict_sol['vel_r']=test_list[5]
dict_sol['vel_phi']=test_list[6]
dict_sol['vel_z']=test_list[7]

dict_sol['func_B_r']=test_list[8]
dict_sol['func_B_phi']=test_list[9]
dict_sol['func_B_z']=test_list[10]

dict_sol['func_Tdyn']=test_list[11]
dict_sol['func_Tmhd']=test_list[12]

#note: unabsorbed flux in 0.3-10 keV, with all gaussians taken off (extended from the post autofit broadband model)
#extension made with AllModels.setEnergies("extend","low,0.3,100 log")
flux_GROJ=3.401e-8

dist_factor=1.22546712210745e+45

lum_GROJ=flux_GROJ*dist_factor/1e38

stop_d=1e6

# SED=os.path.join('incident_xstar_HS.dat')

# SED=os.path.join('test_xstar.dat')

SED_GROJ='5461_cont_deabs_extended_xstar.txt'

xstar_wind(dict_sol,p,mdot, stop_d, SED_GROJ, lum_GROJ,outdir='groj_dr_01_new_test',dr_r=0.1,v_resol=5000,chatter=1,m_BH=5.4)
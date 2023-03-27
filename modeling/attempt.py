#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:22:22 2023

@author: parrama
"""

from simul_tools import xstar_wind
import os

ep=0.1
p=0.1
mu=0.067
angle=25
dict_sol={}

test_list=[3.836508e-01,  1.278251e+00, 2.100000e+01,  1.369094e+00,  4.065250e-01,  1.455833e-01,  1.048574e+00,  1.503721e-01,  3.953974e-01,
           -5.069130e-01,  4.084034e-01,  1.393493e+00,  1.278200e-02]

os.chdir('xsol')

dict_sol['func_zbyr']=test_list[0]
dict_sol['func_rcyl_by_ro']=test_list[1]
dict_sol['func_angle']=test_list[2]
dict_sol['func_Rsph_by_ro']=test_list[3]
dict_sol['func_density_MHD']=test_list[4]
dict_sol['func_vel_r']=test_list[5]
dict_sol['func_vel_phi']=test_list[6]
dict_sol['func_vel_z']=test_list[7]
dict_sol['func_B_r']=test_list[8]
dict_sol['func_B_phi']=test_list[9]
dict_sol['func_B_z']=test_list[10]
dict_sol['func_Tdyn']=test_list[11]
dict_sol['func_Tmhd']=test_list[12]

stop_d=1e5

SED='./incident_xstar_HS.dat'

xstar_wind(ep, p, mu, dict_sol, stop_d, SED, 1)
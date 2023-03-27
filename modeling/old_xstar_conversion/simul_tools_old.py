#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:49:00 2022

@author: parrama
"""

import os
import sys
from astropy.io import ascii

import numpy as np
import pexpect
import time

from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,AllChains,Chain

from xstarsub_dummy import ener,ispecg,ispecgg,ispcg2,x116n5

#adding some libraries 
os.environ['LD_LIBRARY_PATH']+=os.pathsep+'/home/parrama/Soft/Heasoft/heasoft-6.29/x86_64-pc-linux-gnu-libc2.31/lib'

h_cgs = 6.624e-27
eV2erg = 1.6e-12
erg2eV = 1.0/eV2erg
Ryd2eV = 13.5864


def update_xstar_fortran_libs(path='./'):
    '''
    Fetches the xstar libraries in the current heasoft version and recreates the f2py libraries from them
    included: ener,ispecg,ispecgg,ispcg2
    
    '''
    
    
    
def xstar_wind(ep, p_mhd, mu, angle,dict_solution, ncn2=63599, chatter=0):
    
    
    '''
    Python wrapper for the xstar computation
    
    dict_solution is a dictionnary with all the arguments of a JED-SAD solution
    
    need to run f2py -c -m xstar xstarsub_versionum.f90 to get the xstar library (which doesn't work right now for some reason)

    eventual calls to fortran with arrays as arguments will need to use inverted arrays:
        the indexation is not the same as in python so using order='F' when creating arrays is important
    
    the fourth line of the xstar subroutine includes the param file variables, so the param file should be in the same folder
    
    xstar also uses atdb.fits & coheat.dat
    
    '''
    
    '''
    #### Physical constants
    '''
    
    def xstar_function(epi,ncn2,lpri,lunlog,xlum,enlum,zremsz,xpx,xpxcol,zeta,nbox,r_or_f,nbox_restart,vobsx,vturb_x):
        
        '''
        wrapper around the xstar function itself. Since it's too reliant on 
        '''
    #! light speed in Km/s unit
    c_Km = 2.99792e5 
    #! light speed in cm/s unit
    c_cgs = 2.99792e10 
    sigma_thomson_cgs = 6.6525e-25
    c_SI = 2.99792e8 
    G_SI = 6.674e-11
    Msol_SI = 1.98892e30
    PI = 3.14159265
    Km2m = 1000.0
    m2cm = 100.0
    
    
    #chatter value, 0 for not print, 1 for printing
    if chatter>=10:
        lpri=1
    else:
        lpri=0
        
    lunlog=6 
    
    #! in units of 1.0e38 erg/s !* computed by integrating over HighSoft_keV.dat and multiplied by D^2
    #!*4*Pi multiplication comes as xstar assumes spherical cloud.
    xlum = 0.0786*4.0*3.14 
    
    #! Accretion rate in Eddington units
    mdot_obs = 0.111 
    
    # ! From formula (rg/2.0*r_in), Equation (12) of Chakravorty et al. 2016. Here r_in is 6.0*r_g. eta_rad is assumed to be 1.0.
    eta_s = (1.0/12.0)
    
    mdot_Mhd = mdot_obs/eta_s
    
    #!* This value is used to match Keigo's normalization
    #!mdot_norm=4.7130834*2.48e15 
    
    logMbh = 1.00
    Mass_SI = (10.0**logMbh)*Msol_SI
    Rs_SI = 2.0*G_SI*Mass_SI/(c_SI*c_SI)
    
    #!* Gravitational radius
    Rg_SI = 0.5*Rs_SI
    Rg_cgs = Rg_SI*m2cm

    func_zbyr=dict_solution['func_zbyr']
    func_rcyl_by_ro=dict_solution['func_rcyl_by_ro']
    func_angle=dict_solution['func_angle']
    func_Rsph_by_ro=dict_solution['func_Rsph_by_ro']
    func_density_MHD=dict_solution['func_density_MHD']
    func_vel_r=dict_solution['func_vel_r']
    func_vel_phi=dict_solution['func_vel_phi']
    func_vel_z=dict_solution['func_vel_z']
    func_B_r=dict_solution['func_B_r']
    func_B_phi=dict_solution['func_B_phi']
    func_B_z=dict_solution['func_B_z']
    func_Tdyn=dict_solution['func_Tdyn']
    func_Tmhd=dict_solution['func_Tmhd']
    
    #### variable definition
    
    #one of the intrinsic xstar parameters (listed in the param file), the maximal number of points in the grids
    #used here to give the maximal size of the arrays
    ncn=99999
    
    nbox_stop=np.zeros(10,int)
    
    eptmp,zrtmp,zremsz=np.zeros((ncn,3))
    
    eptmp_shifted,zrtmp_shifted=np.zeros((ncn,2))
    
    #no need to create epi,xlum and enlum because they are outputs or defined elsewhere
    
    xpxcoll,xpxl,zetal,vobsl,vrel,del_E=np.zeros((1000,6))
    
    logxi_last,vobs_last,robyRg_last,stop_d,vrel_last,del_E_final,del_E_bs=np.zeros((10,7))

    vobs_start,robyRg_start,Rsph_cgs_start,density_cgs_start,logxi_start=np.zeros((1000,5))
    
    vobs_mid,robyRg_mid,Rsph_cgs_mid,density_cgs_mid,logxi_mid=np.zeros((1000,5))
    
    vobs_stop,robyRg_stop,Rsph_cgs_stop,density_cgs_stop,logxi_stop,NhOfBox=np.zeros((1000,6))

    xpxl_last,xpxcoll_last,zetal_last,vobsl_last=np.zeros((10,4))
    
    Rsph_cgs_last=np.zeros(10)
    
    vturb_in=np.zeros(1000)
    
    logxi_input=np.zeros(1000)
    
    #### building the stop distances
    
    #!last stop_distance
    stop_dl = 1.0e6
    
    stop_d[0] = 1.0e5
    i=0
    
    while stop_d[i]<=stop_dl:
        stop_d[i+1]=stop_d[i]*10
    
    stop_d_index=i
    
    #!The ionizing luminosity only is used as xlum to normalize the spectra
    L_xi_Source = xlum*1.0e38 

    # !* Reading functions of self-similar solutions
        
    if chatter>=5:
        print('func_zbyr=',func_zbyr)
        print('func_rcyl_by_ro=',func_rcyl_by_ro)
        print('func_angle=',func_angle)
        print('func_Rsph_by_ro=',func_Rsph_by_ro)
        print('func_density_MHD=',func_density_MHD)
        print('func_vel_r=',func_vel_r)
        print('func_vel_phi=',func_vel_phi)
        print('func_vel_z=',func_vel_z)
        print('func_B_r=',func_B_r)
        print('func_B_phi=',func_B_phi)
        print('func_B_z=',func_B_z)
        print('func_Tdyn=',func_Tdyn)
        print('func_Tmhd=',func_Tmhd)
        

    #### opening and reseting the files (this should be done on the fly to avoid issues)
    #445
    filobj_mhd=open("./MhdSol_new_cold_low_mag_ep"+str(ep)+"p"+str(p_mhd)+"mu"+str(mu)+"_angle_"+str(angle)+".dat")
    
    #446
    fileobj_box_details=open("./box_details_stop_dist"+str(stop_dl)+".log")
    
    #447
    fileobj_box_ascii=open("./box_Ascii_stop_dist_for_xstar"+str(stop_dl)+".dat")
    
    #448
    fileobj_box_ascii_last=open("last_box_Ascii_for_xstar"+str(stop_dl)+".dat")
    
    #449
    fileobj_box_ascii_stop_dis=open("./box_Ascii_stop_dist"+str(stop_dl)+".dat")
    
    fileobj_box_ascii_stop_dis.write('#Rsph_cgs_mid(nbox) log10(density_cgs_mid(nbox)) log10(NhOfBox(nbox)) logxi_mid(nbox) '+
                                     ' vobs_mid(nbox)\n')
    
    #450
    fileobj_box_ascii_last=open("last_box_Ascii"+str(stop_dl)+".dat")
        
    '''
    !* This following 'while' loop is used to find the first suitable value of 
    !* ro_by_Rg where logxi becomes less than some predefined suitable value. 
    !* Above than that, absorption does not contribute much 
    '''
    
    ro_by_Rg = 6.0
    DelFactorRo = 1.0001
    
    while ro_by_Rg <= 1.1e7 :
        
        #!* distance from black hole in cylindrical co-ordinates-the radial distance
        rcyl_SI = ro_by_Rg*Rg_SI*func_rcyl_by_ro  
        
        #!* distance from black hole in spherical co-ordinates
        Rsph_SI = rcyl_SI*(1.0+(func_zbyr*func_zbyr))**(1/2)  
        
        density_cgs = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))

        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*PI/180.0))+(vel_z_cgs*np.sin(func_angle*PI/180.0)))

        logxi = np.log10(L_xi_Source/(density_cgs*m2cm*Rsph_SI*m2cm*Rsph_SI))
        
        #!* Here we change the location of the first box as logxi is calculated for xstar to provide correct flux from luminosity. 

        if logxi <= 6.0: 
            print("starting anchoring radius ro_by_Rg=",ro_by_Rg)

            break
        else:
            #! A distance command : Step increase in the anchoring radius of the magnetic streamline
            ro_by_Rg = ro_by_Rg*DelFactorRo 

    #!* After getting the starting value of ro_by_Rg from the above 'while' loop, fixing the values for 1st box.
    Rsph_cgs_1st = Rsph_SI*m2cm
    vobs_1st = vel_obs_cgs/(Km2m*m2cm)
    logxi_1st = logxi
    robyRg_1st = ro_by_Rg

    fileobj_box_details.write('Rsph_cgs_1st='+str(Rsph_cgs_1st)+',logxi_1st='+str(logxi_1st)+',vobs_1st='+str(vobs_1st)
                              +',ro/Rg_1st='+str(robyRg_1st)+'\n')
    
    #!* Fixing the parameters' values for stop distance

    for i in range(stop_d_index):

        # ! This is ro/Rg.. Position of streamline
        ro_by_Rg = stop_d[i] 
        
        rcyl_SI = ro_by_Rg*Rg_SI*func_rcyl_by_ro
        Rsph_SI = rcyl_SI*np.sqrt(1.0+(func_zbyr*func_zbyr))
        
        density_cgs = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*PI/180.0))+(vel_z_cgs*np.sin(func_angle*PI/180.0)))
        
        logxi = np.log10(L_xi_Source/(density_cgs*m2cm*Rsph_SI*m2cm*Rsph_SI))
        
        Rsph_cgs_last[i]= Rsph_SI*m2cm
        vobs_last[i]= vel_obs_cgs/(Km2m*m2cm)
        logxi_last[i]= logxi
        robyRg_last[i]= ro_by_Rg

        #shouldn't this be outside the loop if it just mark the last parameters ?
        fileobj_box_details.write('Rsph_cgs_last='+str(Rsph_cgs_last[i])+',logxi_last='+str(logxi_last[i])
                               +',vobs_last='+str(vobs_last[i])+",ro/Rg_last="+str(robyRg_last[i])+'\n')


    #!* Building the boxes based on Delr
    rad_res= 0.115
    delr_by_r= rad_res
    Rsph_cgs_end= Rsph_cgs_1st
    nbox = 1
    i = 1
    
    #while(vobs_end>=vobs_last(stop_d_index))    
    while Rsph_cgs_end<Rsph_cgs_last[stop_d_index]:


        Rsph_cgs_stop[nbox]= Rsph_cgs_end*((2.0+delr_by_r)/(2.0-delr_by_r))

        if Rsph_cgs_last[i]<Rsph_cgs_stop[nbox]:
            delr_by_r = 2.0*(Rsph_cgs_last[i]-Rsph_cgs_stop(nbox-1))/(Rsph_cgs_last[i]+Rsph_cgs_stop(nbox-1))
        
        
        Rsph_cgs_start[nbox]= Rsph_cgs_end
        Rsph_SI= Rsph_cgs_start[nbox]/m2cm
        rcyl_SI = Rsph_SI/(np.sqrt(1.0+(func_zbyr*func_zbyr)))
        robyRg_start[nbox] = rcyl_SI/(Rg_SI*func_rcyl_by_ro)
        
        density_cgs_start[nbox] = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*PI/180.0))+(vel_z_cgs*np.sin(func_angle*PI/180.0)))
        
        vobs_start[nbox] = vel_obs_cgs/(Km2m*m2cm)
        
        logxi_start[nbox] = np.log10(L_xi_Source/(density_cgs_start[nbox]*m2cm*Rsph_SI*m2cm*Rsph_SI))
        
        #!* Recording quantities for the end point of the box*/
        
        Rsph_cgs_stop[nbox]= Rsph_cgs_start[nbox]*((2.0+delr_by_r)/(2.0-delr_by_r))
        
        #!*The above expression comes from R(i+1)-R[i]=(delr_by_r)*((R[i]+R(i+1))/2.0) 
        
        Rsph_SI= (Rsph_cgs_stop[nbox]/m2cm)
        rcyl_SI = Rsph_SI/np.sqrt(1.0+(func_zbyr*func_zbyr))
        
        robyRg_stop[nbox] = rcyl_SI/(Rg_SI*func_rcyl_by_ro)
        
        density_cgs_stop[nbox] = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*PI/180.0))+(vel_z_cgs*np.sin(func_angle*PI/180.0)))
        
        vobs_stop[nbox] = vel_obs_cgs/(Km2m*m2cm)
        
        logxi_stop[nbox] = np.log10(L_xi_Source/(density_cgs_stop[nbox]*Rsph_cgs_stop[nbox]*Rsph_cgs_stop[nbox]))
        
        #!* Recording quantities for the mid point of the box
        
        Rsph_cgs_mid[nbox]= (Rsph_cgs_start[nbox]+Rsph_cgs_stop[nbox])/2.0
        
        Rsph_SI= (Rsph_cgs_mid[nbox]/m2cm)
        rcyl_SI = Rsph_SI/np.sqrt(1.0+(func_zbyr*func_zbyr))
        
        robyRg_mid[nbox] = rcyl_SI/(Rg_SI*func_rcyl_by_ro)
        
        density_cgs_mid[nbox] = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*PI/180.0))+(vel_z_cgs*np.sin(func_angle*PI/180.0)))
        
        vobs_mid[nbox] = vel_obs_cgs/(Km2m*m2cm)
        
        logxi_mid[nbox] = np.log10(L_xi_Source/(density_cgs_mid[nbox]*Rsph_cgs_mid[nbox]*Rsph_cgs_mid[nbox]))
        
        logxi_input[nbox] = np.log10(L_xi_Source/(density_cgs_mid[nbox]*Rsph_cgs_start[nbox]*Rsph_cgs_start[nbox]))
        
        #!* Calculate Nh for the box
        
        NhOfBox[nbox] = density_cgs_mid[nbox]*(Rsph_cgs_stop[nbox]-Rsph_cgs_start[nbox])
        
        #!* Print the quantities in the log checking
        
        #! This step is for storing data for the last box
        if delr_by_r!=rad_res: 
            nbox_stop[i] = nbox-1
            fileobj_box_details.write('last box information\n')
            fileobj_box_details.write('stop_dist='+str(stop_d[i])+'\n')
        
        fileobj_box_details.write('Box no. is='+str(nbox)+'\n')
        fileobj_box_details.write('robyRg_start='+str(robyRg_start[nbox])+'\nvobs_start in km/s='+str(vobs_start[nbox])
                                 +'\nRsph_start in cm='+str(Rsph_cgs_start[nbox])+'\nlognH_start (in /cc)='
                                 +str(np.log10(density_cgs_start[nbox]))+"\nlogxi_start="+str(logxi_start[nbox])+'\n')
        
        # !* log(4*Pi) is subtracted to print the correct logxi.
        # !* We have multiplied xlum i.e. luminosity by 4*Pi to make the estimation of flux correctly.
        # !* xi value also we are providing from ASCII file to xstar wrong to estimate the distance correctly.
         
        fileobj_box_details.write('robyRg_mid='+str(robyRg_mid[nbox])+'\nvobs_mid in km/s='+str(vobs_mid[nbox])+
                                  '\nRsph_mid in cm='+str(Rsph_cgs_mid[nbox])+'\nlognH_mid (in /cc)='
                                  +str(np.log10(density_cgs_mid[nbox]))+'\nlogxi_mid='+str(logxi_mid[nbox])+'\n')
                                                   
        fileobj_box_details.write('robyRg_stop='+str(robyRg_stop[nbox])+'\nvobs_stop in km/s='+str(vobs_stop[nbox])+
                                  '\nRsph_stop in cm='+str(Rsph_cgs_stop[nbox])+'\nlognH_stop (in /cc)='
                                  +str(np.log10(density_cgs_stop[nbox]))+'\nlogxi_stop='+str(logxi_stop[nbox])+'\n')
        
        fileobj_box_details.write('Parameters to go to xstar are:\n')
        fileobj_box_details.write('Gas slab of lognH='+str(np.log10(density_cgs_mid[nbox]))+'\nlogNH='+str(np.log10(NhOfBox[nbox]))
                                  +'\nof logxi='+str(logxi_mid[nbox])+'\nis travelling at a velocity of vobs in Km/s='
                                  +str(vobs_mid[nbox])+'\n')
        
        # !* xi value we are providing from ASCII file to xstar wrong to estimate the distance correctly.
        # !* It is wrong because luminosity is calculated wrongly by multiplying 4.0*Pi 
        # !* This loop is to prepare the ASCII file which will be input of xstar
        
        if (delr_by_r==rad_res):
            #!* calculated suitably fron density_mid and Rsph_start
            #!* logxi_mid is changed to logxi_input
            fileobj_box_ascii.write(str(np.log10(density_cgs_mid[nbox]))+','+str(np.log10(NhOfBox[nbox]))+','
                                   +str(logxi_input[nbox])+','+str(vobs_mid[nbox])+'\n')
        else:
            #!* calculated suitably fron density_mid and Rsph_start
            #!* logxi_mid is changed to logxi_input
            fileobj_box_ascii_last.write(str(np.log10(density_cgs_mid[nbox]))+','+str(np.log10(NhOfBox[nbox]))+','
                                   +str(logxi_input[nbox])+','+str(vobs_mid[nbox])+'\n')
        
        #!* This loop is to prepare the ASCII file where xi value is the actual physical value
        
        if delr_by_r==rad_res:
            fileobj_box_ascii_stop_dis.write(str(Rsph_cgs_mid[nbox])+','+str(np.log10(density_cgs_mid[nbox]))+','+str(np.log10(NhOfBox[nbox]))+','+str(logxi_mid[nbox])+','+str(vobs_mid[nbox]))
        else:
            fileobj_box_ascii_last.write(str(Rsph_cgs_mid[nbox])+','+str(np.log10(density_cgs_mid[nbox]))+','+str(np.log10(NhOfBox[nbox]))+','+str(logxi_mid[nbox])+','+str(vobs_mid[nbox]))

        
        if chatter>=10:
            print(str(nbox)+','+str(Rsph_cgs_last[i])+','+str(Rsph_cgs_stop[nbox])+','
                 +str(delr_by_r)+','+str(Rsph_cgs_last[stop_d_index]))
        
        #!* Readjust the loop parameters
        
        if delr_by_r!=rad_res and i!=stop_d_index:
            print('I am after delr is changed'+str(delr_by_r))
            
            #!* maintaining the regular box no.
            nbox= nbox-1 
            i= i+1
            delr_by_r = rad_res
        
        Rsph_cgs_end = Rsph_cgs_stop[nbox]
        
        nbox = nbox + 1

    #!*write(*,*)Rsph_cgs_end-Rsph_cgs_last(stop_d_index)
    
    nbox_index = nbox_stop[stop_d_index]

    for i in range(stop_d_index):
        fileobj_box_details.write('stop_d(i)='+str(stop_d[i])+'\n')
        print("stop_d(i)="+stop_d[i])
        fileobj_box_details.write('nbox_stop='+str(nbox_stop[i])+'\n')
        print("nbox_stop="+nbox_stop[i])

    print("No. of boxes required="+str(nbox_index))
    
    #445
    filobj_mhd.close()
    
    #446
    fileobj_box_details.close()
    
    #447
    fileobj_box_ascii.close()
    
    #448
    fileobj_box_ascii_last.close()
    
    #449
    fileobj_box_ascii_stop_dis.close()
    
    #450
    fileobj_box_ascii_last.close()
    
    # !*******************************************************************
    # !* Information of physical variables of all the boxes is evaluated and stored in file "box_Ascii_stop_dist_for_xstar" for xstar 
    # input and "box_Ascii_stop_dist" for actual physical variables.
    # !********************************************************************

    epi=ener(ncn2) 
      
    #no need to initialize the elements of zremsz at zero since its created as so

    #211
    with open('./box_Ascii_stop_dist_for_xstar'+str(stop_dl)+'.dat') as fileobj_box_ascii_stop_dist:
        box_stop_dist_list=fileobj_box_ascii_stop_dist.readlines()
    
    #! xpx, xpxcol are given in log value. zeta is logxi. vobs in Km/s
    for nbox in range(nbox_index):
        xpxl[nbox],xpxcoll[nbox],zetal[nbox],vobsl[nbox]=np.array(box_stop_dist_list[nbox].replace('\n','').split(',')).astype(float)
    
    #212
    with open('./last_box_Ascii_stop_dist_for_xstar'+str(stop_dl)+'.dat') as fileobj_box_ascii_stop_dist_last:
        box_stop_dist_list_last=fileobj_box_ascii_stop_dist_last.readlines()
    
    for i in range(stop_d_index):
        xpxl_last[i],xpxcoll_last[i],zetal_last[i],vobsl_last[i]=\
            np.array(box_stop_dist_list_last[i].replace('\n','').split(',')).astype(float)
            
    nbox_restart= 1 
    c_n = 1
    r_or_f= 0

    #!* This file is to write different variables estimated from xstar.
    
    #678
    with open('./temp_ion_fraction_details.dat') as fileobj_ion_fraction:
        fileobj_ion_fraction.write('#starting_calculation_from_nbox='+str(nbox_restart)+'\n')

    #681
    with open('./xstar_output_details.dat') as fileobj_xstar_output:
        fileobj_xstar_output.write('#starting_calculation_from_nbox=',str(nbox_restart)+'\n')
        
    #!nbox_restart should be changed to 1 for fresh calculation
    for nbox in range(nbox_restart,nbox_index):
        
        #adding the variable as a condition for a skip later
        pass_writing=False

        print('nbox ='+str(nbox))
        
        #! Doppler shifting of spectra depending on relative velocity

        if nbox>1:
            vrel[nbox] = vobsl[nbox]-vobsl[nbox-1]
            vturb_in[nbox] = vobsl[nbox-1]-vobsl[nbox]
        else:
            vrel[nbox] = vobsl[nbox]
            vturb_in[nbox] = vobsl[nbox]
        
        del_E[nbox]= np.sqrt((1-vrel[nbox]/c_Km)/(1+vrel[nbox]/c_Km))
        #!del_E(nbox) = 1.00

        #! Reading input spectra from file: Initial spectra/output from last box 

        if nbox<2:
            
            #! Read incident spectra
            #111
            with open('incident_xstar_HS.dat') as fileobj_incident_spectra:
                incident_spectra_lines=fileobj_incident_spectra.readlines()
            
            Nengrid=int(incident_spectra_lines[0])
            
            #! eptmp, zrtmp becomes the input spectra.
            for i in range(Nengrid):
                eptmp[i],zrtmp[i]=incident_spectra_lines[i+1]
            
            #! ispecg transfers the incident spectra zrtmp to zremsz in epi(ncn2) grid
            zremsz=ispecg(eptmp,zrtmp,Nengrid,epi,ncn2,xlum,lpri)

            #! Normalize the spectrum
            #### still don't understand the lun11 here
            zremsz=ispecgg(xlum,epi,ncn2,zremsz,lpri,lunlog) 

            with open('./Incident_spectra.dat') as fileobj_inc_spectra:
                for i in range(ncn2):    
                    fileobj_inc_spectra.write(str(epi[i])+','+str(zremsz[i])+'\n')
                    

            #! calculate photon numbers
            enlum=ispcg2(zremsz,epi,ncn2,lpri,lunlog)            

            for i in range(1,ncn2):
                epi[i] = epi[i]*del_E[nbox]
                zremsz[i] = zremsz[i]*del_E[nbox]
                #!*zremsz[i] = zremsz(i)*1.00000

        else:

            #!* This portion is to compute the spectra if computation is stopped somewhere in between

            if nbox==nbox_restart and nbox>1:
                
                with open('./shifted_input'+str(nbox)+'.dat') as fileobj_shifted_box:
                    shifted_box_lines=fileobj_shifted_box.readlines()

                for i in range(ncn2):
                    epi[i],zremsz[i]=np.array(shifted_box_lines[i].split(',')[2:]).astype(float)
                
                pass_writing=True
                
                pass
            
            if pass_writing:
                pass
            
            #! Read output spectra from previous box, file named as varying_spectra.dat

            with open('./varying_spectra.dat') as fileobj_varying_spectra:
                varying_spectra_lines=fileobj_varying_spectra.readlines()
                
            for i in range(ncn2):
                eptmp[i],zrtmp[i]=np.array(varying_spectra_lines[i].split(',')).astype(float)
                
                    
                if r_or_f == 0:
                    #! for the final box if r_or_f == 1

                    eptmp_shifted[i] = eptmp[i]*del_E[nbox]
                    zrtmp_shifted[i] = zrtmp[i]*del_E[nbox]
                    #!*zrtmp_shifted(i) = zrtmp(i)*1.00000

                else:
                    #! for the final box

                    eptmp_shifted[i] = eptmp[i]*del_E_final[c_n]
                    zrtmp_shifted[i] = zrtmp[i]*del_E_final[c_n]
                    #!*zrtmp_shifted[i] = zrtmp[i]*1.00000

            #!* Mapping to epi grid after doppler shifting

            #l is an index to avoid searching through the whole energy grid at each point. Since we're increasing the energy, 
            #for each subsequent point we only need to search to increasing parts of the grid
            
            l = 0
                
            for i in range(ncn2):
                
                for j in range(l,ncn2):
                    
                    if epi[i]>=eptmp_shifted[j] and epi[i]<eptmp_shifted[j+1]:

                        zrtmp[i] = zrtmp_shifted[j]+((zrtmp_shifted[j+1]-zrtmp_shifted[j])\
                                                     /(eptmp_shifted[j+1]-eptmp_shifted[j]))*(epi[i]-eptmp_shifted[j])

                        l = j

                        break

                #adding an l=i condition to skip this in the case of the test above
                if l!=j and (epi[i]<eptmp_shifted[0] or epi[i]>eptmp_shifted[ncn2]):

                    zrtmp[i] = 0

                #!* zremsz becomes the incident spectrum for the next box
                zremsz[i] = zrtmp[i] 


        #skipped if the computation has stopped in the middle
        
        if not pass_writing:
            #!**Writing the shifted spectra in a file as it is input for next box 
    
            if r_or_f == 0:
                filename_shifted_input='./shifted_input'+str(nbox)+'.dat'
            else:
                filename_shifted_input='./shifted_input_final'+str(c_n)+'.dat'
            
            with open(filename_shifted_input) as fileobj_shifted_input:
                fileobj_shifted_input.write('#energy(eV)  transmitted(1.0e38erg/s/erg)')
                for i in range(ncn2):            
                    fileobj_shifted_input.write(str(epi[i])+','+str(zremsz[i]))

        
        # !***************************************************
        # !  Calling xstar to calculate spectra

        

        #!* This portion is to put the physical parameters of the box correctly

        if r_or_f==0:
           xpx = 10.0**(xpxl[nbox])
           xpxcol = 10.0**(xpxcoll[nbox])
           zeta = zetal[nbox]
           vobsx = vobsl[nbox]
           vturb_x = vturb_in[nbox]
        else:
           xpx = 10.0**(xpxl_last[c_n])
           xpxcol = 10.0**(xpxcoll_last[c_n])
           zeta = zetal_last[c_n]
           vobsx = vobsl_last[c_n]
           vturb_x = vobsl[nbox_stop[c_n]]-vobsl_last[c_n]

        '''
        The lines of sight considered should already be compton thin, the whole line of sight has to be compton thick and this is 
        checked directly from jonathan's solution
        The following test is just a sanity check
        '''
        
        if (xpxcol>1.5e24):
            print('Thomson depth of the cloud becomes unity')
            
            break

        zremsz=x116n5(epi,ncn2,lpri,lunlog,xlum,enlum,xpx,xpxcol,zeta,nbox,r_or_f,nbox_restart,vobsx,vturb_x)
        
        ####!* Computing spectra and blueshift for the final box depending on stop_dist.

        if r_or_f !=1 and nbox==nbox_stop[c_n]:
            
            #!regular or final, 1 indicate final
            r_or_f= 1 

            vrel_last[c_n] = vobsl_last[c_n]-vobsl[nbox]

            del_E_final[c_n] = np.sqrt((1-vrel_last[c_n]/c_Km)/(1+vrel_last[c_n]/c_Km))
            
            #!del_E_final[c_n] = 1.00
            
            #here in the fortran code we go back to the beginning of the code (where opening ./varying_spectra.dat)
            #we can't continue the loop because we are now computing the final box, which is not a standard "n+1" box
            #instead here we repeat the commands that should be run again
            
            #! Read output the new spectra from previous box which xstar has modified, file named as varying_spectra.dat

            with open('./varying_spectra.dat') as fileobj_varying_spectra:
                varying_spectra_lines=fileobj_varying_spectra.readlines()
                
            for i in range(ncn2):
                eptmp[i],zrtmp[i]=np.array(varying_spectra_lines[i].split(',')).astype(float)
                
                if r_or_f == 0:
                    #! for the final box if r_or_f == 1

                    eptmp_shifted[i] = eptmp[i]*del_E[nbox]
                    zrtmp_shifted[i] = zrtmp[i]*del_E[nbox]
                    #!*zrtmp_shifted(i) = zrtmp(i)*1.00000

                else:
                    #! for the final box

                    eptmp_shifted[i] = eptmp[i]*del_E_final[c_n]
                    zrtmp_shifted[i] = zrtmp[i]*del_E_final[c_n]
                    #!*zrtmp_shifted[i] = zrtmp[i]*1.00000 

            #!* Mapping to epi grid after doppler shifting

            #l is an index to avoid searching through the whole energy grid at each point. Since we're increasing the energy, 
            #for each subsequent point we only need to search to increasing parts of the grid
            
            l = 0
                
            for i in range(ncn2):
                
                for j in range(l,ncn2):
                    
                    if epi[i]>=eptmp_shifted[j] and epi[i]<eptmp_shifted[j+1]:

                        zrtmp[i] = zrtmp_shifted[j]+((zrtmp_shifted[j+1]-zrtmp_shifted[j])\
                                                     /(eptmp_shifted[j+1]-eptmp_shifted[j]))*(epi[i]-eptmp_shifted[j])

                        l = j

                        break

                #adding an l=i condition to skip this in the case of the test above
                if l!=j and (epi[i]<eptmp_shifted[0] or epi[i]>eptmp_shifted[ncn2]):

                    zrtmp[i] = 0

                #!* zremsz becomes the incident spectrum for the next box
                zremsz[i] = zrtmp[i] 


            #no need for tests on the final box and restart since we know we're at the final box here
            
            filename_shifted_input='./shifted_input_final'+str(c_n)+'.dat'
                
            with open(filename_shifted_input) as fileobj_shifted_input:
                fileobj_shifted_input.write('#energy(eV)  transmitted(1.0e38erg/s/erg)')
                for i in range(ncn2):            
                    fileobj_shifted_input.write(str(epi[i])+','+str(zremsz[i]))
    
            xpx = 10.0**(xpxl_last[c_n])
            xpxcol = 10.0**(xpxcoll_last[c_n])
            zeta = zetal_last[c_n]
            vobsx = vobsl_last[c_n]
            vturb_x = vobsl[nbox_stop[c_n]]-vobsl_last[c_n]
    
            '''
            The lines of sight considered should already be compton thin, the whole line of sight has to be compton thick and this is 
            checked directly from jonathan's solution
            The following test is just a sanity check
            '''
            
            if (xpxcol>1.5e24):
                print('Thomson depth of the cloud becomes unity')
                
                break
    
            zremsz=x116n5(epi,ncn2,lpri,lunlog,xlum,enlum,xpx,xpxcol,zeta,nbox,r_or_f,nbox_restart,vobsx,vturb_x)
        
        #! Blueshift for the final box
        del_E_bs[c_n] = np.sqrt((1+vobsl_last[c_n]/c_Km)/(1-vobsl_last[c_n]/c_Km))

        with open('./output_spectra_final'+str(nbox)+'.dat') as fileobj_final_output_spectra:

            final_output_spectra_lines=fileobj_final_output_spectra.readlines()
        
        for ll in range(ncn2):

            #note: keeping only columns 0 and 2 (out of 5 columns) as these are the one we're interested in here
            eptmp[ll],zrtmp[ll]=np.array(final_output_spectra_lines[ll+1].split(',')).astype(float)[0,2]
            eptmp[ll] = eptmp[ll]*del_E_bs[c_n]
            zrtmp[ll] = zrtmp[ll]*del_E_bs[c_n]
            #!*zrtmp[ll] = zrtmp[ll]*1.00000
            
        with open('./Final_blueshifted'+str(stop_d[c_n])+'.dat') as fileobj_final_blueshift:


            fileobj_final_blueshift.write('#energy(eV)  transmitted(1.0e38erg/s/erg)\n')

            for ll in range(ncn2):
                fileobj_final_blueshift.write(str(eptmp[ll])+','+str(zrtmp[ll]))


        r_or_f= 0

        c_n= c_n+1
        
        #! if loop for the last box calculation is completed

    
def freeze(model=None,modclass=AllModels,unfreeze=False,parlist=None):
    
    '''
    freezes/unfreezes an entire model or a part of it 
    if no model is given in argument, freezes the first existing models
    (parlist must be the list of the parameter numbers)
    '''
    if model is not None:
        xspec_mod=model
    else:
        xspec_mod=modclass(1)
    
    if parlist is None:
        for par_index in range(1,xspec_mod.nParameters+1):
            xspec_mod(par_index).frozen=int(not(unfreeze))
            
    else:
        for par_index in parlist:
            xspec_mod(par_index).frozen=int(not(unfreeze))

def nuLnu_to_xstar(path,M_BH,display=False):
    
    '''
    Converts a nuLnu spectral distribution file to something suitable as an xstar input
    if display is set to True, displays information on the spectral distribution
    
    adapted from Sudeb's script. 
    '''
    
    sed=np.loadtxt(path)
    
    L_Edd = 1.26e38*M_BH  # where M_BH is in units of Solar mass

    #x and y axes are in units of nu(Hz) and Lnu(erg/s/Hz). We have to convert to nu(eV) and Lnu(erg/s/erg).
    
    E_eV, Lnu_per_erg = np.zeros(np.size(sed[:,0])), np.zeros(np.size(sed[:,0]))

    #The following step converts in required units of xstar
    for j in range(np.size(sed[:,0])):
        E_eV[j] = sed[j,0]*h_cgs*erg2eV
        Lnu_per_erg[j] = sed[j,1]/h_cgs
    
    L = 0          # L gives total luminosity
    L_keV = 0      # L_keV gives luminosity in energy range between 1 to 100 keV
    
    #Following for loop calculates the unnormalized luminosity in 1-100 keV
    for i in range(np.size(sed[:,0])-1):
        L = L+((Lnu_per_erg[i+1]+Lnu_per_erg[i])/2.0)*(E_eV[i+1]-E_eV[i])*eV2erg
        if (1.0e3<E_eV[i]<1.0e5):
            L_keV = L_keV+((Lnu_per_erg[i+1]+Lnu_per_erg[i])/2.0)*(E_eV[i+1]-E_eV[i])*eV2erg
    
    norm_factor = 0.13*L_Edd/L_keV #Normalization factor is calculated to match the L_keV to 0.13 L_Edd
    
    with open('incident_xstar_55334.dat','w+') as f1:
    
        f1.write("%d\n" % (np.size(sed[:,0])))
        
        for j in range(np.size(sed[:,0])):
            Lnu_per_erg[j] = Lnu_per_erg[j]*norm_factor #L_nu is normalized such that L_keV equals to 0.13 L_Edd
            f1.write("%e  %e\n" % (E_eV[j], Lnu_per_erg[j]))
    
    if display:

        L = 0          # L gives total luminosity
        L_keV = 0      # L_keV gives luminosity in energy range between 1 to 100 keV
        L_Ryd = 0      # L_Ryd gives luminosity in energy range between 1 to 1000 Ryd

        for i in range(np.size(sed[:,0])-1):
            L = L+((Lnu_per_erg[i+1]+Lnu_per_erg[i])/2.0)*(E_eV[i+1]-E_eV[i])*eV2erg
            if (1.0e3<E_eV[i]<1.0e5):
                L_keV = L_keV+((Lnu_per_erg[i+1]+Lnu_per_erg[i])/2.0)*(E_eV[i+1]-E_eV[i])*eV2erg
            if (1.0<E_eV[i]/Ryd2eV<1.0e3):
                L_Ryd = L_Ryd+((Lnu_per_erg[i+1]+Lnu_per_erg[i])/2.0)*(E_eV[i+1]-E_eV[i])*eV2erg
                
        print(L, L_keV, L_Ryd)
        print(L_keV/L_Edd)

def xstar_to_table(path,outname=None):
    
    '''
    Converts an xstar output spectra to a model table usable by xspec (atable)
    adapted from pop's script
    '''

    currdir=os.getcwd()
    
    filedir='' if '/' not in path else path[:path.rfind('/')]

    file=path[path.rfind('/')+1:]
    
    filedir=path.replace(file,'')
    filename=path[:path.rfind('.')]
    
    tmp=np.genfromtxt(path,skip_header=1)
    enemin=tmp[:,0]*1e-3 #keV
    eneminsub=enemin[np.where(enemin > 1e-4)]
    flux=tmp[:,1]*1e38/(4.*3.14*(10.*1e3*3e18)**2.) #erg/s/erg/cm2
    fluxsub=flux[np.where(enemin > 1e-4)]
    enemaxsub=np.roll(eneminsub,-1)
    enemeansub=0.5*(enemaxsub+eneminsub)
    l=len(eneminsub)

    spec=fluxsub/(enemeansub*1.6e-9) #ph/s/erg/cm2
    spec=spec*(enemaxsub-eneminsub)*1.6e-9 #ph/s/cm2
    
    ascii.write([eneminsub[0:l-1],enemaxsub[0:l-1],spec[0:l-1]],filedir+'tmp.txt',overwrite=True)
    
    os.system("sed '/col0/d' "+filedir+"tmp.txt > "+filename+".txt")
    
    #spawning a bash process to produce the table from the txt modified SED
    heaproc=pexpect.spawn('/bin/bash',encoding='utf-8')
    heaproc.logfile=sys.stdout
    heaproc.sendline('heainit')
    heaproc.sendline('cd '+currdir)
    
    heaproc.sendline('ftflx2tab '+filename+'.txt '+filename+' '+filename+'.fits clobber = yes')
    
    #waiting for the file to be created before closing the bashproc
    while not os.path.isfile(filename+'.fits'):
        time.sleep(1)
        
    heaproc.sendline('exit')
    
    if outname is not None:
        os.system('mv '+filedir+filename+'.fits '+filedir+outname)
    
def create_fake_xstar(table,rmf,arf,exposure,nofile=True,reset=True):
    
    if reset:
        AllModels.clear()
        AllData.clear()
        
    #loading the model table in an xspec model
    tablemod=Model('atable{'+table+'}')

    #freezing the model to avoid probkems in case the table has free parameters
    freeze()
        
    #creating the fakeit settings
    fakeset=FakeitSettings(response=rmf,arf=arf,exposure=exposure)
    
    #launching the fakeit
    AllData.fakeit(settings=fakeset,applyStats=True,noWrite=False)
    
    
    
    
    
    
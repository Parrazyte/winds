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

from tqdm import tqdm

from xspec import AllModels,AllData,Model,FakeitSettings,Plot

import pyxstar as px

# #adding some libraries 
# os.environ['LD_LIBRARY_PATH']+=os.pathsep+'/home/parrama/Soft/Heasoft/heasoft-6.31.1/x86_64-pc-linux-gnu-libc2.31/lib'

h_cgs = 6.624e-27
eV2erg = 1.6e-12
erg2eV = 1.0/eV2erg
Ryd2eV = 13.5864

def file_edit(path,line_id,line_data,header):
    
    '''
    Edits (or create) the file given in the path and replaces/add the line(s) where the line_id str/LIST is with
    the line-content str/LIST.
    
    line_id should be included in line_content.
    
    Header is the first line of the file, with usually different informations.
    '''
    
    lines=[]
    if type(line_id)==str or type(line_id)==np.str_:
        line_identifier=[line_id]
    else:
        line_identifier=line_id
        
    if type(line_data)==str or type(line_data)==np.str_:
        line_content=[line_data]
    else:
        line_content=line_data
        
    if os.path.isfile(path):
        with open(path) as file:
            lines=file.readlines()
            
            #loop for all the lines to add
            for single_identifier,single_content in zip(line_identifier,line_content):
                line_exists=False
                if not single_content.endswith('\n'):
                    single_content+='\n'
                #loop for all the lines in the file
                for l,single_line in enumerate(lines):
                    if single_identifier in single_line:
                        lines[l]=single_content
                        line_exists=True
                if line_exists==False:
                    lines+=[single_content]
            
    else:
        #adding everything
        lines=line_content

    with open(path,'w+') as file:
        if lines[0]==header:
            file.writelines(lines)
        else:
            file.writelines([header]+lines)
            
def xstar_wind(ep, p_mhd, mu, angle,dict_solution,stop_d, SED_path, xlum, h_over_r=0.1, rad_res=0.115, ncn2=63599, chatter=0):
    
    
    '''
    Python wrapper for the xstar computation of a single solution
    
    Required parameters:
        ep,p_mhd,mu and angle are the main WED parameters of the solution
        
        dict_solution is a dictionnary with all the arguments of a JED-SAD solution

        stop_d is a single (or list of) stop distances in units of Rg
    
        SED is the path of the incident spectra (suitable for xstar input)
        Note: The normalization doesn't matter

        xlum is the bolometric luminosity of the spectrum in units of 1e38ergs/s
        Used internally by xstar to renormalize the given SED
        
        #### Note: should be updated with the speed of the material and potential significant absorption?
        

    Secondary options:
    
        h_over_r is the aspect ratio of the disk
        
        rad_res is the radial revolution
        ####weird use, should be converted to a variable which is used in a more straightforward manner
        ####+ why is there a distance jump between the boxes, are they not directly adjacent? 
    
        chatter gives out the number of infos displayed during the computation
    
    
    
    Notes on the python conversion:
        -since array numbers starts at 0, we use "index" box numbers (starting at 0) and adapt all of the consequences,
        but still print the correct box number (+1)

    ####SHOULD BE UPDATED TO ADD THE JED SAD N(R) DEPENDANCY
    '''
    
    '''
    #### Physical constants
    '''
    
    def shift_tr_spectra(bshift,path):
        
        '''
        shifts the current transmitted spectra from the xstar output files and stores it (in a xstar-accepted manner)
        in the "path" file
        '''
        
        #loading the continuum spectrum of the previous box
        prev_box_sp=px.ContSpectra()
        
        eptmp=prev_box_sp.energy()
        zrtmp=prev_box_sp.transmitted()

        eptmp_shifted = eptmp*bshift
        
        #multiplying the spectrum is not useful unless it's relativistic but just in case
        zrtmp_shifted = zrtmp*bshift

        #should not need to remap the spectrum since it will be done internally by xstar            

        shifted_input_arr=np.array([eptmp_shifted,zrtmp_shifted]).T
        
        #!**Writing the shifted spectra in a file as it is input for next box 
        np.savetxt(path,shifted_input_arr,header=str(len(eptmp)),delimiter='  ')
            
    def write_xstar_infos(nbox,vobsx,path):
        
        '''
        loads the main elements and details with abundances form the current xstar output in a file
        
        the file_edit function uses the box number + step number as line identifier
        
        jkp is the variable name of the step number
        
        '''
        
        # file_header='#nbox(1) jkp(2) rad_start(3) rad_end(4) '+\
        #  'delta_r(5) xpxcol/xpx(6) zeta_start(7) zeta_end(8) zeta_avg(9) xpx(10) vobsx(11) xpxcol(12) temp_box(13) O8(14) O7(15) '+\
        #  'Ne10(16) Ne9(17) Na11(18) Na10(19) Mg12(20) Mg11(21) Al13(22) Al12(23) Si14(24) Si13(25) S16(26) S15(27) Ar18(28) Ar17(29)'+\
        #  'Ca20(30) Ca19(31) Fe26(32) Fe25(33) Nh_O8(34) Nh_O7(35) Nh_Ne10(36) Nh_Ne9(37) Nh_Na11(38) Nh_Na10(39) Nh_Mg12(40)'+\
        #  'Nh_Mg11(41) Nh_Al13(42) Nh_Al12(43) Nh_Si14(44) Nh_Si13(45) Nh_S16(46) Nh_S15(47) Nh_Ar18(48) Nh_Ar17(49) Nh_Ca20(50) '+\
        #  'Nh_Ca19(51) Nh_Fe26(52) Nh_Fe25(53)\n'

        file_header='#nbox(1)\tjkp(2)\trad(3)\t'+\
        'delta_r(4)\txpxcol/xpx(5)\tzeta(6)\txpx(7)\tvobsx(8)\txpxcol(9)\ttemp_box(10)\tO8(11)\tO7(12)\t'+\
        'Ne10(13)\tNe9(14)\tNa11(15)\tNa10(16)\tMg12(17)\tMg11(18)\tAl13(19)\tAl12(20)\tSi14(21)\tSi13(22)\t'+\
        'S16(23)\tS15(24)\tAr18(25)\tAr17(26)\tCa20(27)\tCa19(28)\tFe26(29)\tFe25(30)\tNh_O8(31)\tNh_O7(32)\t'+\
        'Nh_Ne10(33)\tNh_Ne9(34)\tNh_Na11(35)\tNh_Na10(36)\tNh_Mg12(37)\tNh_Mg11(38)\tNh_Al13(39)\tNh_Al12(40)\t'+\
        'Nh_Si14(41)\tNh_Si13(42)\tNh_S16(43)\tNh_S15(44)\tNh_Ar18(45)\tNh_Ar17(46)\tNh_Ca20(47)\t'+\
        'Nh_Ca19(48)\tNh_Fe26(49)\tNh_Fe25(50)\n'

        #not really necessary ATM        
        # file_header_main='#nbox(1)\tjkp(2)\tr(3)\tdelta_r(4)\txpxcol/xpx(5)\tzeta(6)\txpx(7)\tvobsx(8)\txpxcol(9)\t'+\
        #                  'temp_box(10)\nO8(11)\tO7(12)\tNe10(13)\tNe9(14)\tNa11(15)\tNa10(16)\tMg12(17)\tMg11(18)\t'+\
        #                  'Al13(19)\tAl12(20)\tSi14(21)\tSi13(22)\tS16(23)\tS15(24)\tAr18(25)\tAr17(26)\tCa20(27)\t'+\
        #                  'Ca19(28)\tFe26(29)\tFe25(30)\n'
         
        n_steps=len(px.Abundances('o_iii'))

        for i_step in range(n_steps):
            
            plasma_pars=px.PlasmaParameters()
            
            ####Q: rad star, rad_end, rad_pos ????
            
            ####Q: same question for zeta_end etc.
            
            ####Q: should I use x_e for xpx and n_p for xpxcol ?
            
            #main infos
            
            main_infos=[nbox,n_steps+1,
                        plasma_pars.radius[i_step],plasma_pars.delta_r[i_step],plasma_pars.n_p[i_step]/plasma_pars.x_e[i_step],
                        plasma_pars.ion_parameter[i_step],plasma_pars.x_e[i_step],vobsx,plasma_pars.n_p[i_step],
                        plasma_pars.temperature[i_step]*1e4]
             
            #detailed abundances 

            ion_infos=[px.Abundances('O8')[0],
                       px.Abundances('O7')[0],
                       px.Abundances('Ne10')[0],
                       px.Abundances('Ne9')[0],
                       px.Abundances('Na11')[0],
                       px.Abundances('Na10')[0],
                       px.Abundances('Mg12')[0],
                       px.Abundances('Mg11')[0],
                       px.Abundances('Al13')[0],
                       px.Abundances('Al12')[0],
                       px.Abundances('Si14')[0],
                       px.Abundances('Si13')[0],
                       px.Abundances('S16')[0],
                       px.Abundances('S15')[0],
                       px.Abundances('Ar18')[0],
                       px.Abundances('Ar17')[0],
                       px.Abundances('Ca20')[0],
                       px.Abundances('Ca19')[0],
                       px.Abundances('Fe26')[0],
                       px.Abundances('Fe25')[0]]
            
            #detailed column densities for the second file
            
            col_infos=[px.Columns('O8')[0],
                       px.Columns('O7')[0],
                       px.Columns('Ne10')[0],
                       px.Columns('Ne9')[0],
                       px.Columns('Na11')[0],
                       px.Columns('Na10')[0],
                       px.Columns('Mg12')[0],
                       px.Columns('Mg11')[0],
                       px.Columns('Al13')[0],
                       px.Columns('Al12')[0],
                       px.Columns('Si14')[0],
                       px.Columns('Si13')[0],
                       px.Columns('S16')[0],
                       px.Columns('S15')[0],
                       px.Columns('Ar18')[0],
                       px.Columns('Ar17')[0],
                       px.Columns('Ca20')[0],
                       px.Columns('Ca19')[0],
                       px.Columns('Fe26')[0],
                       px.Columns('Fe25')[0]]
            
            file_edit(path=path,line_id='\t'.join(main_infos[:2]),line_data='\t'.join(main_infos+ion_infos+col_infos),header=file_header)
        
        
    def xstar_function(spectrum_file,lum,t_guess,n,nh,xi,nbox,vturb_x):
        
        '''
        wrapper around the xstar function itself with explicit calls to the parameters we're changing
        
        lum -> rlrad38
        
        xpx is the density (density)

        -xpxcol is the column density (column)
        
        -zeta is the xi parameter (rlogxi)
        
        -nbox, final_box,nbox_restart are yours, vobx is a sudep parameter that is not an argument but  
        
         vturb_x is vturb_i
        
        -niter>nlimd
        
        npass=number of computations (going outwards then inwards) to converge
        should always stay odd if increased to remain the one taken outwards
        '''
        
        #copying the parameter dictionnaries to avoid issues if overwriting the defaults
        xpar=px.par.copy()
        xhpar=px.hpar.copy()
        

        #changing the parameters varying for the xstar solution
        xhpar['ncn2']=ncn2
        
        #changing the parameters varying from the function
        
        xhpar['spectrum']='file'
        xhpar['spectrum_file']=spectrum_file
        xhpar['rlrad38']=xlum
        xpar['temperature']=t_guess
        xpar['density']=n
        xpar['column']=nh
        xpar['rloxi']=xi
        
        xhpar['vturbi']=vturb_x
        
        px.run_xstar(xpar,xhpar)
        
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
    
    #ncn=99999
    
    nbox_stop=np.zeros(len(stop_d),int)
    
    #no need to create epi,xlum and enlum because they are outputs or defined elsewhere
    
    xpxcoll,xpxl,zetal,vobsl,vrel,del_E=np.zeros((len(stop_d),6))
    
    logxi_last,vobs_last,robyRg_last,stop_d,vrel_last,del_E_final,del_E_bs=np.zeros((len(stop_d),7))

    xpxl_last,xpxcoll_last,zetal_last,vobsl_last=np.zeros((10,4))
    
    vturb_in=np.zeros(1000)
    
    logxi_input=np.zeros(1000)
        
    Rsph_cgs_last=np.zeros(len(stop_d))
    
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
    
    stop_dl=stop_d[-1]
    #446
    fileobj_box_details=open("./box_details_stop_dist"+str(stop_d)+".log")
    
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
    
    for i in range(len(stop_d)):

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
    delr_by_r= rad_res
    Rsph_cgs_end= Rsph_cgs_1st
    nbox = 0
    i = 0
    
    #### This is very ugly and should be changed to the proper number of boxes, computed before this loop
    
    vobs_start,robyRg_start,Rsph_cgs_start,density_cgs_start,logxi_start=np.zeros((1000,5))
    
    vobs_mid,robyRg_mid,Rsph_cgs_mid,density_cgs_mid,logxi_mid=np.zeros((1000,5))
    
    vobs_stop,robyRg_stop,Rsph_cgs_stop,density_cgs_stop,logxi_stop,NhOfBox=np.zeros((1000,6))
    
    while Rsph_cgs_end<Rsph_cgs_last[len(stop_d)-1]:


        Rsph_cgs_stop[nbox]= Rsph_cgs_end*((2.0+delr_by_r)/(2.0-delr_by_r))

        if Rsph_cgs_last[i]<Rsph_cgs_stop[nbox]:
            delr_by_r = 2.0*(Rsph_cgs_last[i]-Rsph_cgs_stop[nbox-1])/(Rsph_cgs_last[i]+Rsph_cgs_stop[nbox-1])
        
        
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
        
        fileobj_box_details.write('Box no. is='+str(nbox+1)+'\n')
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
                 +str(delr_by_r)+','+str(Rsph_cgs_last[-1]))
        
        #!* Readjust the loop parameters
        
        if delr_by_r!=rad_res and i!=len(stop_d)-1:
            print('I am after delr is changed'+str(delr_by_r))
            
            #!* maintaining the regular box no.
            nbox= nbox-1 
            i= i+1
            delr_by_r = rad_res
        
        Rsph_cgs_end = Rsph_cgs_stop[nbox]
        
        nbox = nbox + 1

    #!*write(*,*)Rsph_cgs_end-Rsph_cgs_last(stop_d_index)
    
    nbox_index = nbox_stop[len(stop_d)-1]

    for i in range(len(stop_d)):
        fileobj_box_details.write('stop_d(i)='+str(stop_d[i])+'\n')
        print("stop_d(i)="+stop_d[i])
        fileobj_box_details.write('nbox_stop='+str(nbox_stop[i])+'\n')
        print("nbox_stop="+nbox_stop[i])

    print("No. of boxes required="+str(nbox_index))
    
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

    #211
    with open('./box_Ascii_stop_dist_for_xstar'+str(stop_dl)+'.dat') as fileobj_box_ascii_stop_dist:
        box_stop_dist_list=fileobj_box_ascii_stop_dist.readlines()
    
    #! xpx, xpxcol are given in log value. zeta is logxi. vobs in Km/s

    for nbox in range(nbox_index):
        xpxl[nbox],xpxcoll[nbox],zetal[nbox],vobsl[nbox]=np.array(box_stop_dist_list[nbox].replace('\n','').split(',')).astype(float)
    
    #212
    with open('./last_box_Ascii_stop_dist_for_xstar'+str(stop_dl)+'.dat') as fileobj_box_ascii_stop_dist_last:
        box_stop_dist_list_last=fileobj_box_ascii_stop_dist_last.readlines()
    
    for i in range(len(stop_d)):
        xpxl_last[i],xpxcoll_last[i],zetal_last[i],vobsl_last[i]=\
            np.array(box_stop_dist_list_last[i].replace('\n','').split(',')).astype(float)
            
    nbox_restart= 1 
    c_n = 1
    final_box= False

    #!* This file is to write different variables estimated from xstar.
    
    #678
    with open('./temp_ion_fraction_details.dat') as fileobj_ion_fraction:
        fileobj_ion_fraction.write('#starting_calculation_from_nbox='+str(nbox_restart)+'\n')

    #681
    with open('./xstar_output_details.dat') as fileobj_xstar_output:
        fileobj_xstar_output.write('#starting_calculation_from_nbox=',str(nbox_restart)+'\n')
    
    '''
    t is the temperature of the plasma. Starts at the default value of 400, but will be updated starting on the second box
    with the temperature of the previous box as a "guess"
    '''
    
    tp=400
    
    ####main loop
    
    #!nbox_restart should be changed to 1 for fresh calculation
    for nbox in tqdm(range(nbox_restart,nbox_index)):
        
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
            
            # Nengrid=int(incident_spectra_lines[0])            

            xstar_input_SED=SED_path

        else:
                
            #reloading the iteration from the previous xstar run
            px.LoadFiles()
            
            #loading the temperature of the previous box
            plasma_par=px.PlasmaParameters()
            
            #retrieving the plasma temperature of the last step
            tp=plasma_par.temperature[-1]
            
            #!**Writing the shifted spectra in a file as it is input for next box 
            filename_shifted_input='./shifted_input'+str(nbox)+'.dat'
            
            #shifting the spectra and storing it in the xstar input file nae
            shift_tr_spectra(del_E[nbox],filename_shifted_input)
            
        xpx = 10.0**(xpxl[nbox])
        xpxcol = 10.0**(xpxcoll[nbox])
        zeta = zetal[nbox]
        vobsx = vobsl[nbox]
        vturb_x = vturb_in[nbox]
           
        '''
        The lines of sight considered should already be compton thin, the whole line of sight has to be compton thick and this is 
        checked directly from jonathan's solution
        The following test is just a sanity check
        '''
        
        if (xpxcol>1.5e24):
            print('Thomson depth of the cloud becomes unity')
 
            break

        xstar_function(filename_shifted_input,xpx,xpxcol,tp,zeta,nbox,final_box,nbox_restart,vobsx,vturb_x)
        
        
        #writing the infos
        px.LoadFiles()
        write_xstar_infos(nbox,vobsx,'xstar_output_details.dat')
        
        ####!* Computing spectra and blueshift for the final box depending on stop_dist.

        if nbox==nbox_stop[c_n]:
            
            #!regular or final, 1 indicate final
            final_box=True

            vrel_last[c_n] = vobsl_last[c_n]-vobsl[nbox]

            del_E_final[c_n] = np.sqrt((1-vrel_last[c_n]/c_Km)/(1+vrel_last[c_n]/c_Km))
            
            #!del_E_final[c_n] = 1.00
            
            #here in the fortran code we go back to the beginning of the code (where opening ./varying_spectra.dat)
            #we can't continue the loop because we are now computing the final box, which is not a standard "n+1" box
            #instead here we repeat the commands that should be run again
            
            #! Read output the new spectra from the box

            px.LoadFiles()
            
            #loading the temperature of the previous box
            plasma_par=px.PlasmaParameters()
            
            #retrieving the plasma temperature of the last step
            tp=plasma_par.temperature[-1]
            
            filename_shifted_input='./shifted_input_final'+str(c_n)+'.dat'
            
            #shifting the spectra in a different file name
            shift_tr_spectra(del_E_final[c_n],filename_shifted_input)
    
            xpx = 10.0**(xpxl_last[c_n])
            xpxcol = 10.0**(xpxcoll_last[c_n])
            zeta = zetal_last[c_n]
            vobsx = vobsl_last[c_n]
            vturb_x = vobsl[nbox_stop[c_n]]-vobsl_last[c_n]
    
            
            if (xpxcol>1.5e24):
                print('Thomson depth of the cloud becomes unity')
                
                break
    
            xstar_function(filename_shifted_input,xpx,xpxcol,tp,zeta,nbox,final_box,nbox_restart,vobsx,vturb_x)
        
            px.LoadFiles()
            filename_shifted_input='./Final_blueshifted'+str(stop_d[c_n])+'.dat'
            
            shift_tr_spectra(del_E_bs[c_n],filename_shifted_input)   
    
            final_box= 0
    
            c_n= c_n+1
            
            #writing the infos with the final iteration
            write_xstar_infos(nbox,vobsx,'xstar_final_output_details.dat')
            
            #! if loop for the last box calculation is completed

def nuLnu_to_xstar(path,renorm=False,Edd_ratio=1,M_BH=8,display=False):
    
    '''
    Converts a nuLnu spectral distribution file to something suitable as an xstar input
    if display is set to True, displays information on the spectral distribution
    
    adapted from Sudeb's script. 
    
    NOTE: currently we don't renormalize the spectrum to actual luminosity by default since it's not needed anymore:
          at this since xstar now renormalizes internally
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
    
    if renorm:
        #Normalization factor is calculated to reach the desired Eddington ratio for a given mass
        norm_factor = Edd_ratio*L_Edd/L_keV  
    else:
        norm_factor=1
    
    with open(path.replace(path[path.rfind('.'):],'_xstar'+path[path.rfind('.'):]),'w+') as f1:
    
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
    
    
def model_to_nuLnu(path):
    
    #store the current model's nuLnu in a file through
    
    curr_xaxis=Plot.xAxis
    
    Plot.xAxis="Hz"
    
    Plot('eeuf')
    
    x_arr=Plot.x()
    
    y_arr=Plot.y()
    
    save_arr=np.array([x_arr,y_arr]).T
    
    np.savetxt(path,save_arr,header='nu (Hz) Lnu (erg/s/Hz)',delimiter=' ')
    
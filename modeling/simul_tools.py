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


#trapezoid integration
from scipy.integrate import trapezoid
from general_tools import file_edit

from tqdm import tqdm

from xspec import AllModels,AllData,Model,FakeitSettings,Plot

import pyxstar as px

# #adding some libraries 
# os.environ['LD_LIBRARY_PATH']+=os.pathsep+'/home/parrama/Soft/Heasoft/heasoft-6.31.1/x86_64-pc-linux-gnu-libc2.31/lib'

h_cgs = 6.624e-27
eV2erg = 1.6e-12
erg2eV = 1.0/eV2erg
Ryd2eV = 13.5864

def xstar_wind(dict_solution,p_mhd,mdot_obs,stop_d_input, SED_path, xlum,outdir="xsol",nbox_restart=1,h_over_r=0.1, ro_init=1e3,rad_res=0.115, v_resol=100, m_BH=8,chatter=0):
    
    
    '''
    Python wrapper for the xstar computation of a single solution
    
    Required parameters:
        p_mhd and mdot_obs are the main parameters outside of the JED-SAD solution
        
        dict_solution is a dictionnary with all the arguments of a JED-SAD solution

        stop_d is a single (or list of) stop distances in units of Rg
    
        SED is the path of the incident spectra (suitable for xstar input)
        Note: The normalization doesn't matter

        xlum is the bolometric luminosity of the spectrum in units of 1e38ergs/s
        Used internally by xstar to renormalize the given SED
        
        The luminosity value is manually updated after each computation to consider the decrease due to absorption
        
        #### Note: should be updated with the speed of the material and potential significant absorption?
        

    Secondary options:
    
        h_over_r is the aspect ratio of the disk
        
        ro_init gives the initial value of ro_by_rg
        
        rad_res is the radial revolution
        ####weird use, should be converted to a variable which is used in a more straightforward manner
        ####+ why is there a distance jump between the boxes, are they not directly adjacent? 
    
        vsol gives the desired radial resolution desired, converted into a number of continuum bins used for the computation
             the computing time inversely scales with vsol
             
             100km/s gives 38678 bins
        
        chatter gives out the number of infos displayed during the computation
    
    
    
    Notes on the python conversion:
        -since array numbers starts at 0, we use "index" box numbers (starting at 0) and adapt all of the consequences,
        but still print the correct box number (+1)

    ####SHOULD BE UPDATED TO ADD THE JED SAD N(R) DEPENDANCY
    
    ####SHOULD BE UPDATED TO CONSIDER THE LOGXI DEPENDANCY WHEN L CHANGES
    '''
    
    '''
    #### Physical constants
    '''
    
    def shift_tr_spectra(bshift,path):
        
        '''
        shifts the current transmitted spectra from the xstar output files and stores it (in a xstar-accepted manner)
        in the "path" file
        
        Also returns the updated luminosity value by computing the ratio of the initial and current xstar spectra,
        and multiplies it by the initial xlum value
        
        Works with easy integration of the xstar file because the unit is ergs/s/cm²/erg, aka ergs*cst after integration
        Works independantly of the normalisation of the spectra   
        '''
        
        #loading the continuum spectrum of the previous box
        prev_box_sp=px.ContSpectra()
        
        eptmp=np.array(prev_box_sp.energy)
        zrtmp=np.array(prev_box_sp.transmitted)

        eptmp_shifted = eptmp*bshift
        
        #multiplying the spectrum is not useful unless it's relativistic but just in case
        zrtmp_shifted = zrtmp*bshift

        #should not need to remap the spectrum since it will be done internally by xstar            

        shifted_input_arr=np.array([eptmp_shifted,zrtmp_shifted]).T
        
        #!**Writing the shifted spectra in a file as it is input for next box 
        np.savetxt(path,shifted_input_arr,header=str(len(eptmp)),delimiter='  ',comments='')
        
        #the xstar output files has x axis in units of eV and y axis in units of 1e38erg/s/erg
        #so we need to integrate and the x axis must be renormalized to ergs (so with the conversion factor below)
        
        xlum_output=trapezoid(zrtmp_shifted,x=eptmp_shifted*1.6021773E-12)
        
        return xlum_output
    
            
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
            
            #main infos
            main_infos=np.array([nbox,i_step+1,
                        plasma_pars.radius[i_step],plasma_pars.delta_r[i_step],plasma_pars.n_p[i_step]/plasma_pars.x_e[i_step],
                        plasma_pars.ion_parameter[i_step],plasma_pars.x_e[i_step],vobsx,plasma_pars.n_p[i_step],
                        plasma_pars.temperature[i_step]*1e4]).astype(str).tolist()
            
            #detail for clarity
            main_infos[0]=str(int(float(main_infos[0])))
            main_infos[1]=str(int(float(main_infos[1])))
            
            #detailed abundances 

            ion_infos=np.array([px.Abundances('o_viii')[0],
                       px.Abundances('o_vii')[0],
                       px.Abundances('ne_x')[0],
                       px.Abundances('ne_ix')[0],
                       px.Abundances('na_xi')[0],
                       px.Abundances('na_x')[0],
                       px.Abundances('mg_xii')[0],
                       px.Abundances('mg_xi')[0],
                       px.Abundances('al_xiii')[0],
                       px.Abundances('al_xii')[0],
                       px.Abundances('si_xiv')[0],
                       px.Abundances('si_xiii')[0],
                       px.Abundances('s_xvi')[0],
                       px.Abundances('s_xv')[0],
                       px.Abundances('ar_xviii')[0],
                       px.Abundances('ar_xvii')[0],
                       px.Abundances('ca_xx')[0],
                       px.Abundances('ca_xix')[0],
                       px.Abundances('fe_xxvi')[0],
                       px.Abundances('fe_xxv')[0]]).astype(str).tolist()
            
            #detailed column densities for the second file
            
            col_infos=np.array([px.Columns('o_viii'),
                       px.Columns('o_vii'),
                       px.Columns('ne_x'),
                       px.Columns('ne_ix'),
                       px.Columns('na_xi'),
                       px.Columns('na_x'),
                       px.Columns('mg_xii'),
                       px.Columns('mg_xi'),
                       px.Columns('al_xiii'),
                       px.Columns('al_xii'),
                       px.Columns('si_xiv'),
                       px.Columns('si_xiii'),
                       px.Columns('s_xvi'),
                       px.Columns('s_xv'),
                       px.Columns('ar_xviii'),
                       px.Columns('ar_xvii'),
                       px.Columns('ca_xx'),
                       px.Columns('ca_xix'),
                       px.Columns('fe_xxvi'),
                       px.Columns('fe_xxv')]).astype(str).tolist()
            
            file_edit(path=path,line_id='\t'.join(main_infos[:2]),line_data='\t'.join(main_infos+ion_infos+col_infos),header=file_header)
         
    def xstar_func(spectrum_file,lum,t_guess,n,nh,xi,vturb_x,nbox,nbox_final,nsteps=1,niter=100,lcpres=0,path_logpars=None):
        
        '''
        wrapper around the xstar function itself with explicit calls to the parameters routinely being changed in the computation
        
        if path_logpars is not None, saves the list of modifiable parameters into a file
        
        lum -> rlrad38

        
        -lcpres determines if the pressure is constant. In this case t stays constant. Should be kept at 0
        
        xpx is the density (density)

        -xpxcol is the column density (column)
        
        -zeta is the xi parameter (rlogxi)
        
        -nbox, final_box,nbox_restart are yours, vobx is a sudep parameter that is not an argument but  
        
         vturb_x is vturb_i
        
        -niter>nlimd
        
        hpars
        
        -nsteps: number of (radial?) decompositions of the box. Currently bugged in the standalone version so should be kept at 1 and only make the
                 number of boxes vary
        
        -niter: determines the number of iterations to find the thermal equilibrium.
                Needs to be changed because the default PyXstar value is 0 (no equilibrium iteration found)
                Default value at 100 (max value) to get the most attemps at finding the equilibrium
        
        
        npass=number of computations (going outwards then inwards) to converge
        should always stay odd if increased to remain the one taken outwards
        
        
        '''
        
        #copying the parameter dictionnaries to avoid issues if overwriting the defaults
        xpar=px.par.copy()
        xhpar=px.hpar.copy()
        

        #changing the parameters varying for the xstar solution
        xhpar['ncn2']=nbins
        
        #changing the parameters varying from the function
        xpar['spectrum']='file'
        xpar['spectrum_file']=spectrum_file
        xpar['rlrad38']=lum
        xpar['temperature']=t_guess
        xpar['density']=n
        xpar['column']=nh
        xpar['rlogxi']=xi
        
        #making sure this remains at 0 if the default values get played with
        xpar['lcpres']=0
        
        #secondary parameters
        xhpar['nsteps']=nsteps
        
        #and putting this to hopefully not 0
        xhpar['niter']=niter
        
        xhpar['vturbi']=vturb_x
        
        if path_logpars is not None:
            parlog_str='\t'.join([str(nbox),str(nbox_final),spectrum_file,'%.3e'%lum,'%.3e'%t_guess,'%.3e'%n,'%.3e'%nh,'%.3e'%xi,'%.3e'%vturb_x+'\n'])
            
            parlog_header=['#v_resol= '+str(v_resol)+' km/s | nbins= '+str(nbins)+'\n',
                           '#nsteps= '+str(nsteps)+'\tniter= '+str(niter)+'\n',
                           '#(1)nbox\t(2)nbox_final\t(3)spectrum\t(4)lum\t(5)t_guess\t(6)n\t(7)nh\t(8)logxi\t(9)vturb_x\n']
            
            file_edit(path_logpars,'\t'.join([str(nbox),str(nbox_final),spectrum_file]),parlog_str,parlog_header)

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
    
    #making sure the stop variable is an iterable
    if type(stop_d_input) not in [list, np.ndarray]:
        stop_d=[stop_d_input]
    else:
        stop_d=stop_d_input
    
    
    #chatter value, 0 for not print, 1 for printing
    if chatter>=10:
        lpri=1
    else:
        lpri=0
    
    '''
    computing the number of bins to be used from the desired radial resolution
    see xstar manual p.34 for the formula (here we use log version instead of a very small power to avoid losing precision with numerical computations)
    delta_E/E is delta_V/c
    (note: mistake in the formula, its 0.49999 instead of 0.49)
    '''
    
    nbins=int(np.ceil(np.log(4*10**5)/np.log(1+v_resol/299792.458)))
    
    if chatter>=1:
        print('Number of bins for selected velocity resolution: '+str(nbins)+'\n')
    
    # ! From formula (rg/2.0*r_in), Equation (12) of Chakravorty et al. 2016. Here r_in is 6.0*r_g. eta_rad is assumed to be 1.0.
    eta_s = (1.0/12.0)
    
    mdot_Mhd = mdot_obs/eta_s
    
    #!* This value is used to match Keigo's normalization
    #!mdot_norm=4.7130834*2.48e15 

    m_BH_SI = m_BH*Msol_SI
    Rs_SI = 2.0*G_SI*m_BH_SI/(c_SI*c_SI)
    
    #!* Gravitational radius
    Rg_SI = 0.5*Rs_SI
    Rg_cgs = Rg_SI*m2cm


    #Self-similar functions f1-f10
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
    
    nbox_stop=np.zeros(len(stop_d),dtype=int)
    
    #no need to create epi,xlum and enlum because they are outputs or defined elsewhere
    
    logxi_last,vobs_last,robyRg_last,vrel_last,del_E_final,del_E_bs,xpxl_last,xpxcoll_last,zetal_last,vobsl_last=np.zeros((10,len(stop_d),))
    
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
        
    os.system('mkdir -p '+outdir)
    
    #copying the xstar input file inside the directory
    
    os.system('cp '+SED_path+' '+outdir+'/')
    
    #446
    fileobj_box_details=open('./'+outdir+"/box_details_stop_dist_%.1e"%stop_dl+".log",'w+')
    
    #447
    fileobj_box_ascii_xstar=open('./'+outdir+"/box_Ascii_stop_dist_for_xstar_%.1e"%stop_dl+".dat",'w+')
    
    #448
    fileobj_box_ascii_xstar_last=open('./'+outdir+"/last_box_Ascii_for_xstar_%.1e"%stop_dl+".dat",'w+')
    
    #449
    fileobj_box_ascii_stop_dis=open('./'+outdir+"/box_Ascii_stop_dist_%.1e"%stop_dl+".dat",'w+')
    
    fileobj_box_ascii_stop_dis.write('#Rsph_cgs_mid(nbox) log10(density_cgs_mid(nbox)) log10(NhOfBox(nbox)) logxi_mid(nbox) '+
                                     ' vobs_mid(nbox)\n')
    
    #450
    fileobj_box_ascii_last=open('./'+outdir+"/last_box_Ascii_%.1e"%stop_dl+".dat",'w+')
        
    '''
    !* This following 'while' loop is used to find the first suitable value of 
    !* ro_by_Rg where logxi becomes less than some predefined suitable value. 
    !* Above than that, absorption does not contribute much 
    '''
    
    ro_by_Rg = ro_init
    DelFactorRo = 1.0001
    
    while ro_by_Rg <= 1.1e7 :
        
        #!* distance from black hole in cylindrical co-ordinates-the radial distance
        rcyl_SI = ro_by_Rg*Rg_SI*func_rcyl_by_ro  
        
        #!* distance from black hole in spherical co-ordinates
        Rsph_SI = rcyl_SI*(1.0+(func_zbyr*func_zbyr))**(1/2)  
        
        density_cgs = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))

        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*np.pi/180.0))+(vel_z_cgs*np.sin(func_angle*np.pi/180.0)))

        logxi = np.log10(L_xi_Source/(density_cgs*m2cm*Rsph_SI*m2cm*Rsph_SI))
        
        #!* Here we change the location of the first box as logxi is calculated for xstar to provide correct flux from luminosity. 

        if logxi <= 6.0: 
            print("starting anchoring radius ro_by_Rg=",ro_by_Rg)

            break
        else:
            #! A distance command : Step increase in the anchoring radius of the magnetic streamline
            ro_by_Rg = ro_by_Rg*DelFactorRo 

    breakpoint()
    
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
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*np.pi/180.0))+(vel_z_cgs*np.sin(func_angle*np.pi/180.0)))
        
        logxi = np.log10(L_xi_Source/(density_cgs*m2cm*Rsph_SI*m2cm*Rsph_SI))
        
        Rsph_cgs_last[i]= Rsph_SI*m2cm
        vobs_last[i]= vel_obs_cgs/(Km2m*m2cm)
        logxi_last[i]= logxi
        robyRg_last[i]= ro_by_Rg

        fileobj_box_details.write('Rsph_cgs_last='+str(Rsph_cgs_last[i])+',logxi_last='+str(logxi_last[i])
                               +',vobs_last='+str(vobs_last[i])+",ro/Rg_last="+str(robyRg_last[i])+'\n')


    #!* Building the boxes based on Delr
    delr_by_r= rad_res
    Rsph_cgs_end= Rsph_cgs_1st
    i_box = 0
    i_last_box=0
    
    #### This is very ugly and should be changed to the proper number of boxes, computed before this loop
    vobs_start,robyRg_start,Rsph_cgs_start,density_cgs_start,logxi_start=np.zeros((5,1000))
    vobs_mid,robyRg_mid,Rsph_cgs_mid,density_cgs_mid,logxi_mid=np.zeros((5,1000))
    vobs_stop,robyRg_stop,Rsph_cgs_stop,density_cgs_stop,logxi_stop,NhOfBox=np.zeros((6,1000))
    
    while Rsph_cgs_end<Rsph_cgs_last[len(stop_d)-1]:

        Rsph_cgs_stop[i_box]= Rsph_cgs_end*((2.0+delr_by_r)/(2.0-delr_by_r))

        if Rsph_cgs_last[i_last_box]<Rsph_cgs_stop[i_box]:
            delr_by_r = 2.0*(Rsph_cgs_last[i_last_box]-Rsph_cgs_stop[i_box-1])/(Rsph_cgs_last[i_last_box]+Rsph_cgs_stop[i_box-1])
            ####SHOULD BE CHECKED FOR SEVERAL LAST BOXES
            
        Rsph_cgs_start[i_box]= Rsph_cgs_end
        Rsph_SI= Rsph_cgs_start[i_box]/m2cm
        rcyl_SI = Rsph_SI/(np.sqrt(1.0+(func_zbyr*func_zbyr)))
        robyRg_start[i_box] = rcyl_SI/(Rg_SI*func_rcyl_by_ro)
        
        density_cgs_start[i_box] = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        
        #this should be updated to switch to delta v over v (and then we get r_cyl)
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*np.pi/180.0))+(vel_z_cgs*np.sin(func_angle*np.pi/180.0)))
        
        vobs_start[i_box] = vel_obs_cgs/(Km2m*m2cm)
        
        logxi_start[i_box] = np.log10(L_xi_Source/(density_cgs_start[i_box]*m2cm*Rsph_SI*m2cm*Rsph_SI))
        
        #!* Recording quantities for the end point of the box*/
        
        Rsph_cgs_stop[i_box]= Rsph_cgs_start[i_box]*((2.0+delr_by_r)/(2.0-delr_by_r))
        
        #!*The above expression comes from R(i+1)-R[i]=(delr_by_r)*((R[i]+R(i+1))/2.0) 
        
        Rsph_SI= (Rsph_cgs_stop[i_box]/m2cm)
        rcyl_SI = Rsph_SI/np.sqrt(1.0+(func_zbyr*func_zbyr))
        
        robyRg_stop[i_box] = rcyl_SI/(Rg_SI*func_rcyl_by_ro)
        
        density_cgs_stop[i_box] = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*np.pi/180.0))+(vel_z_cgs*np.sin(func_angle*np.pi/180.0)))
        
        vobs_stop[i_box] = vel_obs_cgs/(Km2m*m2cm)
        
        logxi_stop[i_box] = np.log10(L_xi_Source/(density_cgs_stop[i_box]*Rsph_cgs_stop[i_box]*Rsph_cgs_stop[i_box]))
        
        #!* Recording quantities for the mid point of the box
        
        Rsph_cgs_mid[i_box]= (Rsph_cgs_start[i_box]+Rsph_cgs_stop[i_box])/2.0
        
        Rsph_SI= (Rsph_cgs_mid[i_box]/m2cm)
        rcyl_SI = Rsph_SI/np.sqrt(1.0+(func_zbyr*func_zbyr))
        
        robyRg_mid[i_box] = rcyl_SI/(Rg_SI*func_rcyl_by_ro)
        
        density_cgs_mid[i_box] = (mdot_Mhd/(sigma_thomson_cgs*Rg_cgs))*func_density_MHD*((rcyl_SI/Rg_SI)**(p_mhd-1.5))
        
        vel_r_cgs = c_cgs*func_vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
        vel_z_cgs = c_cgs*func_vel_z*((rcyl_SI/Rg_SI)**(-0.5))
        vel_obs_cgs = ((vel_r_cgs*np.cos(func_angle*np.pi/180.0))+(vel_z_cgs*np.sin(func_angle*np.pi/180.0)))
        
        vobs_mid[i_box] = vel_obs_cgs/(Km2m*m2cm)
        
        logxi_mid[i_box] = np.log10(L_xi_Source/(density_cgs_mid[i_box]*Rsph_cgs_mid[i_box]*Rsph_cgs_mid[i_box]))
        
        logxi_input[i_box] = np.log10(L_xi_Source/(density_cgs_mid[i_box]*Rsph_cgs_start[i_box]*Rsph_cgs_start[i_box]))
        #!* Calculate Nh for the box
        
        NhOfBox[i_box] = density_cgs_mid[i_box]*(Rsph_cgs_stop[i_box]-Rsph_cgs_start[i_box])
        
        #!* Print the quantities in the log checking
        
        #! This step is for storing data for the last box
        if delr_by_r!=rad_res: 
            
            #adding a +1 here to switch from an i_box to the actual box number
            nbox_stop[i] = i_box+1-1
            fileobj_box_details.write('\n-----------------------------\n')
            fileobj_box_details.write('\nLast box information\n')
            fileobj_box_details.write('stop_dist='+str(stop_d[i])+'\n')
        
        fileobj_box_details.write('\n-----------------------------\n')
        fileobj_box_details.write('Box n°'+str(i_box+1)+'\n')
        fileobj_box_details.write('robyRg_start='+str(robyRg_start[i_box])+'\nvobs_start in km/s='+str(vobs_start[i_box])
                                 +'\nRsph_start in cm='+str(Rsph_cgs_start[i_box])+'\nlognH_start (in /cc)='
                                 +str(np.log10(density_cgs_start[i_box]))+"\nlogxi_start="+str(logxi_start[i_box])+'\n')
        
        # !* log(4*Pi) is subtracted to print the correct logxi.
        # !* We have multiplied xlum i.e. luminosity by 4*Pi to make the estimation of flux correctly.
        # !* xi value also we are providing from ASCII file to xstar wrong to estimate the distance correctly.
         
        fileobj_box_details.write('robyRg_mid='+str(robyRg_mid[i_box])+'\nvobs_mid in km/s='+str(vobs_mid[i_box])+
                                  '\nRsph_mid in cm='+str(Rsph_cgs_mid[i_box])+'\nlognH_mid (in /cc)='
                                  +str(np.log10(density_cgs_mid[i_box]))+'\nlogxi_mid='+str(logxi_mid[i_box])+'\n')
                                                   
        fileobj_box_details.write('robyRg_stop='+str(robyRg_stop[i_box])+'\nvobs_stop in km/s='+str(vobs_stop[i_box])+
                                  '\nRsph_stop in cm='+str(Rsph_cgs_stop[i_box])+'\nlognH_stop (in /cc)='
                                  +str(np.log10(density_cgs_stop[i_box]))+'\nlogxi_stop='+str(logxi_stop[i_box])+'\n')
        
        fileobj_box_details.write('Parameters to go to xstar are:\n')
        fileobj_box_details.write('Gas slab of lognH='+str(np.log10(density_cgs_mid[i_box]))+'\nlogNH='+str(np.log10(NhOfBox[i_box]))
                                  +'\nof logxi='+str(logxi_mid[i_box])+'\nis travelling at a velocity of vobs in Km/s='
                                  +str(vobs_mid[i_box])+'\n')
        
        # !* xi value we are providing from ASCII file to xstar wrong to estimate the distance correctly.
        # !* It is wrong because luminosity is calculated wrongly by multiplying 4.0*Pi 
        # !* This loop is to prepare the ASCII file which will be input of xstar
        
        if (delr_by_r==rad_res):
            #!* calculated suitably fron density_mid and Rsph_start
            #!* logxi_mid is changed to logxi_input
            fileobj_box_ascii_xstar.write(str(np.log10(density_cgs_mid[i_box]))+'\t'+str(np.log10(NhOfBox[i_box]))+'\t'
                                   +str(logxi_input[i_box])+'\t'+str(vobs_mid[i_box])+'\n')
        else:
            #!* calculated suitably fron density_mid and Rsph_start
            #!* logxi_mid is changed to logxi_input
            fileobj_box_ascii_xstar_last.write(str(np.log10(density_cgs_mid[i_box]))+'\t'+str(np.log10(NhOfBox[i_box]))+'\t'
                                   +str(logxi_input[i_box])+'\t'+str(vobs_mid[i_box])+'\n')
        
        #!* This loop is to prepare the ASCII file where xi value is the actual physical value
        
        if delr_by_r==rad_res:
            fileobj_box_ascii_stop_dis.write(str(Rsph_cgs_mid[i_box])+'\t'+str(np.log10(density_cgs_mid[i_box]))+'\t'+str(np.log10(NhOfBox[i_box]))+'\t'+str(logxi_mid[i_box])+'\t'+str(vobs_mid[i_box])+'\n')
        else:
            fileobj_box_ascii_last.write(str(Rsph_cgs_mid[i_box])+'\t'+str(np.log10(density_cgs_mid[i_box]))+'\t'+str(np.log10(NhOfBox[i_box]))+'\t'+str(logxi_mid[i_box])+'\t'+str(vobs_mid[i_box])+'\n')

        
        if chatter>=10:
            print(str(i_box+1)+'\t'+str(Rsph_cgs_last[i])+'\t'+str(Rsph_cgs_stop[i_box])+'\t'
                 +str(delr_by_r)+'\t'+str(Rsph_cgs_last[-1]))
        
        #!* Readjust the loop parameters
        
        if delr_by_r!=rad_res and i_last_box!=len(stop_d)-1:
            print('I am after delr is changed'+str(delr_by_r))
            
            #!* maintaining the regular box no.
            i_box= i_box-1 
            i_last_box= i_last_box+1
            delr_by_r = rad_res
        
        Rsph_cgs_end = Rsph_cgs_stop[i_box]
        
        i_box = i_box + 1
        
    #### This is very ugly and should be changed to the proper number of boxes, computed before this loop
    vobs_start=vobs_start[:i_box]
    robyRg_start=robyRg_start[:i_box]
    Rsph_cgs_start=Rsph_cgs_start[:i_box]
    density_cgs_start=density_cgs_start[:i_box]
    logxi_start=logxi_start[:i_box]
    vobs_mid=vobs_mid[:i_box]
    robyRg_mid=robyRg_mid[:i_box]
    Rsph_cgs_mid=Rsph_cgs_mid[:i_box]
    density_cgs_mid=density_cgs_mid[:i_box]
    logxi_mid=logxi_mid[:i_box]
    vobs_stop=vobs_stop[:i_box]
    robyRg_stop=robyRg_stop[:i_box]
    Rsph_cgs_stop=Rsph_cgs_stop[:i_box]
    density_cgs_stop=density_cgs_stop[:i_box]
    logxi_stop=logxi_stop[:i_box]
    NhOfBox=NhOfBox[:i_box]

    #!*write(*,*)Rsph_cgs_end-Rsph_cgs_last(stop_d_index)
    
    nbox_index = nbox_stop[len(stop_d)-1]

    for i in range(len(stop_d)):
        fileobj_box_details.write('stop_d(i)='+str(stop_d[i])+'\n')
        print("stop_d(i)="+str(stop_d[i]))
        fileobj_box_details.write('nbox_stop='+str(nbox_stop[i])+'\n')
        print("nbox_stop="+str(nbox_stop[i]))

    print("No. of boxes required="+str(nbox_index))
    
    #446
    fileobj_box_details.close()
    
    #447
    fileobj_box_ascii_xstar.close()
    
    #448
    fileobj_box_ascii_xstar_last.close()
    
    #449
    fileobj_box_ascii_stop_dis.close()
    
    #450
    fileobj_box_ascii_last.close()
    
    # !*******************************************************************
    # !* Information of physical variables of all the boxes is evaluated and stored in file "box_Ascii_stop_dist_for_xstar" for xstar 
    # input and "box_Ascii_stop_dist" for actual physical variables.
    # !********************************************************************

    #211
    with open('./'+outdir+'/box_Ascii_stop_dist_for_xstar_%.1e'%stop_dl+'.dat','r') as fileobj_box_ascii_stop_dist:
        box_stop_dist_list=fileobj_box_ascii_stop_dist.readlines()
    
    #! xpx, xpxcol are given in log value. zeta is logxi. vobs in Km/s
    
    xpxcoll,xpxl,zetal,vobsl,vrel,del_E=np.zeros((6,nbox_index))

    for i_box in range(nbox_index):
        xpxl[i_box],xpxcoll[i_box],zetal[i_box],vobsl[i_box]=np.array(box_stop_dist_list[i_box].replace('\n','').split('\t')).astype(float)
    
    #212
    with open('./'+outdir+'/last_box_Ascii_for_xstar_%.1e'%stop_dl+'.dat','r') as fileobj_box_ascii_stop_dist_last:
        box_stop_dist_list_last=fileobj_box_ascii_stop_dist_last.readlines()
    
    for i in range(len(stop_d)):
        xpxl_last[i],xpxcoll_last[i],zetal_last[i],vobsl_last[i]=\
            np.array(box_stop_dist_list_last[i].replace('\n','').split('\t')).astype(float)

    #!* This file is to write different variables estimated from xstar.
    
    # #678
    # with open('./'+outdir+'/temp_ion_fraction_details.dat','w+') as fileobj_ion_fraction:
    #     fileobj_ion_fraction.write('#starting_calculation_from_nbox='+str(nbox_restart)+'\n')

    # #681
    # with open('./'+outdir+'/xstar_output_details.dat','w+') as fileobj_xstar_output:
    #     fileobj_xstar_output.write('#starting_calculation_from_nbox='+str(nbox_restart)+'\n')
    
    '''
    t is the temperature of the plasma. Starts at the default value of 400, but will be updated starting on the second box
    with the temperature of the previous box as a "guess"
    '''
    
    tp=400
    
    #current index of the list of "final" boxes for which we compute a final spectrum
    i_box_final = 0
    
    
    #effective luminosity value which will be modified
    xlum_eff=xlum
    
    #defining the path for the file where we log the evolution of the xstar parameters
    #note: no outdir here because this is done inside the xstar_func function so in outdir already
    path_log_xpars='./xstar_pars.log'
    
    ####main loop
    
    #!nbox_restart should be changed to 1 for fresh calculation
    
    #using i_box because it's an index here, not the actual box number (shifted by 1)
    for i_box in tqdm(range(nbox_restart-1,nbox_index)):
        
        #! Doppler shifting of spectra depending on relative velocity

        if i_box>0:
            vrel[i_box] = vobsl[i_box]-vobsl[i_box-1]
            vturb_in[i_box] = vobsl[i_box-1]-vobsl[i_box]
        else:
            vrel[i_box] = vobsl[i_box]
            vturb_in[i_box] = vobsl[i_box]
        
        del_E[i_box]= np.sqrt((1-vrel[i_box]/c_Km)/(1+vrel[i_box]/c_Km))
        #!del_E(i_box) = 1.00

        #! Reading input spectra from file: Initial spectra/output from last box 

        if i_box<1:
            
            # Nengrid=int(incident_spectra_lines[0])            

            xstar_input=SED_path

        else:
                
            #reloading the iteration from the previous xstar run
            px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                         file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')
            
            #loading the temperature of the previous box
            plasma_par=px.PlasmaParameters()
            
            #retrieving the plasma temperature of the last step
            tp=plasma_par.temperature[-1]
            
            xstar_input='./shifted_input'+str(i_box+1)+'.dat'
            
            #!**Writing the shifted spectra in a file as it is input for next box 
            xstar_input_save='./'+outdir+'/shifted_input'+str(i_box+1)+'.dat'
            
            #shifting the spectra and storing it in the xstar input file name
            xlum_eff=shift_tr_spectra(del_E[i_box],xstar_input_save)
            
        xpx = 10.0**(xpxl[i_box])
        xpxcol = 10.0**(xpxcoll[i_box])
        
        #correcting the ionization parameter for the evolution in luminosity
        zeta = zetal[i_box]*(xlum_eff/xlum)
        
        vobsx = vobsl[i_box]
        vturb_x = vturb_in[i_box]
        
        '''
        The lines of sight considered should already be compton thin, the whole line of sight has to be compton thick and this is 
        checked directly from jonathan's solution
        The following test is just a sanity check
        '''
        
        if (xpxcol>1.5e24):
            print('Thomson depth of the cloud becomes unity')
 
            break

        #### main xstar call
        
        currdir=os.getcwd()
        os.chdir(outdir)
        os.system('rm -f xout_*.fits')
        
        xstar_func(xstar_input,xlum_eff,tp,xpx,xpxcol,zeta,vturb_x,nbox=i_box+1,nbox_final=i_box_final,path_logpars=path_log_xpars)
        
        os.chdir(currdir)
        
        if i_box>0:
            #moving and renaming the log file
            os.system('mv ./'+outdir+'/xout_step.log'+' ./'+outdir+'/xout_step_box_'+str(i_box+1)+'.log')
            
        ####SDKFJSDKLDFJSKJLF where does this log file go ?
        
        #writing the infos
        px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                     file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')
        
        write_xstar_infos(i_box+1,vobsx,'./'+outdir+'/xstar_output_details.dat')
        
        ####!* Computing spectra and blueshift for the final box depending on stop_dist.

        if i_box+1==nbox_stop[i_box_final]:

            vrel_last[i_box_final] = vobsl_last[i_box_final]-vobsl[i_box]

            #blueshifts for the final box
            del_E_final[i_box_final] = np.sqrt((1-vrel_last[i_box_final]/c_Km)/(1+vrel_last[i_box_final]/c_Km))
            
            del_E_bs[i_box_final] = np.sqrt((1+vobsl_last[i_box_final]/c_Km)/(1-vobsl_last[i_box_final]/c_Km))
     
            #!del_E_final[i_box_final] = 1.00
            
            #here in the fortran code we go back to the beginning of the code (where opening ./varying_spectra.dat)
            #we can't continue the loop because we are now computing the final box, which is not a standard "n+1" box
            #instead here we repeat the commands that should be run again
            
            #! Read output the new spectra from the box
            px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                         file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')
            
            #loading the temperature of the previous box
            plasma_par=px.PlasmaParameters()
            
            #retrieving the plasma temperature of the last step
            tp=plasma_par.temperature[-1]
            
            xstar_input_save='./'+outdir+'/shifted_input_final_'+str(i_box_final+1)+'.dat'
            
            xstar_input='./shifted_input_final_'+str(i_box_final+1)+'.dat'
            
            #shifting the spectra in a different file name
            xlum_final=shift_tr_spectra(del_E_final[i_box_final],xstar_input_save)
    
            xpx = 10.0**(xpxl_last[i_box_final])
            xpxcol = 10.0**(xpxcoll_last[i_box_final])
            
            zeta = zetal_last[i_box_final]*(xlum_final/xlum)
            
            vobsx = vobsl_last[i_box_final]
            vturb_x = vobsl[nbox_stop[i_box_final]-1]-vobsl_last[i_box_final]
    
            
            if (xpxcol>1.5e24):
                print('Thomson depth of the cloud becomes unity')
                
                break
    
            currdir=os.getcwd()
            os.chdir(outdir)
            os.system('rm -f xout_*.fits')
            
            #using xlum_final here to avoid overwriting xlum_eff if using more than a single stop distance
            xstar_func(xstar_input,xlum_final,tp,xpx,xpxcol,zeta,vturb_x,nbox=i_box+1,nbox_final=i_box_final+1,path_logpars=path_log_xpars)
            
            os.chdir(currdir)
            
            #moving and renaming the log file
            os.system('mv '+outdir+'/xout_step.log'+' '+outdir+'/xout_step_final_'+str(i_box_final+1)+'.log')
            
            px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                         file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')
            
            xstar_input='./'+outdir+'/final_blueshifted_%.1e'%stop_d[i_box_final]+'.dat'
            
            
            xlum_eff=shift_tr_spectra(del_E_bs[i_box_final],xstar_input)   
    
            #switching to the next final box to be computed    
            i_box_final= i_box_final+1
            
            #writing the infos with the final iteration
            write_xstar_infos(i_box+1,vobsx,'./'+outdir+'/xstar_output_details_final.dat')
            
            #removing the standard xstar output to gain space
            os.system('rm -f '+outdir+'/xout_*.fits')
            

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

def xstar_to_table(path,outpath='xspecmod.fits',modname=None):
    
    '''
    Converts an xstar output spectra to a model table usable by xspec (atable)
    adapted from pop's script
    
    if modname is None, uses the file name in outpath as model name
    '''

    currdir=os.getcwd()
    
    filedir='' if '/' not in path else path[:path.rfind('/')]
    
    outdir='' if '/' not in outpath else outpath[:outpath.rfind('/')]
    outfile=outpath[outpath.rfind('/')+1:]
    
    
    outprefix=outfile[:outfile.rfind('.')]
    
    file=path[path.rfind('/')+1:]
    
    filedir=path.replace(file,'')
    
    #filename=path[:path.rfind('.')]
    
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
    
    ascii.write([eneminsub[0:l-1],enemaxsub[0:l-1],spec[0:l-1]],outprefix+'_tmp.txt',overwrite=True)
    
    os.system("sed '/col0/d' "+outprefix+"_tmp.txt > "+outprefix+".txt")
        
    #spawning a bash process to produce the table from the txt modified SED
    heaproc=pexpect.spawn('/bin/bash',encoding='utf-8')
    heaproc.logfile=sys.stdout
    heaproc.sendline('heainit')
    heaproc.sendline('cd '+currdir)
    
    #here, outfile is the name of the model
    heaproc.sendline('ftflx2tab '+outprefix+'.txt '+outprefix+' '+outfile+' clobber = yes')
    
    #waiting for the file to be created before closing the bashproc
    while not os.path.isfile(os.path.join(currdir,filedir,outfile)):
        time.sleep(1)
        
    #removing the temp product file
    heaproc.sendline('rm '+outprefix+'_tmp.txt')
    heaproc.sendline('rm '+outprefix+'.txt')
    heaproc.sendline('exit')
    
    if outdir!='':
        os.system('mv '+os.path.join(currdir,filedir,outfile)+' '+outpath)
        

    
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

def create_fake_xstar(table,rmf,arf,exposure,nofile=True,reset=True,prefix=""):
    
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
    AllData.fakeit(settings=fakeset,applyStats=True,noWrite=nofile,filePrefix=prefix)
    
    
def model_to_nuLnu(path):
    
    #store the current model's nuLnu in a file through
    
    curr_xaxis=Plot.xAxis
    
    Plot.xAxis="Hz"
    
    Plot('eeuf')
    
    x_arr=Plot.x()
    
    y_arr=Plot.y()
    
    save_arr=np.array([x_arr,y_arr]).T
    
    np.savetxt(path,save_arr,header='nu(Hz) Lnu(erg/s/Hz)',delimiter=' ')
    
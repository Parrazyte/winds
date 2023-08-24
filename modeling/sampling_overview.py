#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:11:18 2023

@author: parrama
"""

def ang2kev(x):

    '''note : same thing on the other side due to the inverse
    
    also same thing for mAngtoeV'''

    return 12.398/x

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time

#don't know where the bug comes from tbf
# try:
#     st.set_page_config(page_icon=":hole:",layout='wide')
# except:
#     pass

c_l=299792.458

with st.sidebar:
    st.header('Sampling')
    
    dr_r=st.number_input(r'$\Delta r/r$',value=5e-2,format='%.3e')
    
    
    val_oversampling=st.number_input(r'Resolution oversampling factor',value=3,format='%.3e')
    
    
    val_ener=st.number_input(r'Energy at which to take the $\delta E/E$ (keV)',value=7,format='%.3e') 
        
    radio_instr_type=st.radio('Spectral resolution using',('microcalorimeter (default XRISM)','gratings (default HETG)'))
    
    if radio_instr_type=='gratings (default HETG)':
        val_resol=st.number_input(r'Gratings resolution $\delta\lambda$ (mA)',value=12,format='%.3e')
            
        dv_v=val_resol*1e-3/ang2kev(val_ener)
    
    else:
        de=st.number_input(r'microcalorimeter resolution $\delta E$ (eV)',value=6,format='%.3e')
        
        dv_v=de*1e-3/val_ener
    
    
    st.latex('\delta V/V=%.2e'%dv_v)
    st.latex('v_{resol}=%.2e'%(dv_v*c_l)+'\;\mathrm{km/s}')
    st.latex('v_{max}^{sol}=%.2e'%(dv_v*c_l/val_oversampling)+'\;\mathrm{km/s}')
    
    val_rin=st.number_input(r'Starting radius (Rg)',value=1e3,min_value=1.,format='%.3e')
    
    val_rout=st.number_input(r'End radius (Rg)',value=1e6,min_value=val_rin,format='%.3e')


    st.header('Physical inputs')
    
    with st.expander('JED-SAD MHD solution'):
    
        #func_density_MHD
        rho_mhd=st.number_input('MHD density',value=4.065250e-01,format='%.3e')
        
        #mhd pressure
        p_mhd=st.number_input('MHD pressure',value=0.1,format='%.3e')        
        
        #func_rcyl_by_ro
        r_A=st.number_input('Alven(?) radius (in units of r0)',value=1.278251,format='%.3e')

        #func_zbyr
        z_A=st.number_input('Alven(?) altitude in units of Alven radius',value=3.836508e-01,format='%.3e')
    
        
        vel_r=st.number_input('Radial velocity',value= 1.455833e-01,format='%.3e')
        
        vel_z=st.number_input('Z axis velocity',value=1.503721e-01,format='%.3e')
        
        angle=st.number_input('angle (°)',value=21,format='%.3e')
        
        
    xlum=st.number_input(r'Luminosity (1e38erg/s)',value=3,format='%.3e')
    
    mdot_obs=st.number_input(r'Observed $\dot m$',value=0.111,min_value=1e-10,format='%.3e')
    
    m_BH=st.number_input(r'Black Hole Mass ($M_\odot$)',value=8.,min_value=1e-10,format='%.3e')

    
mdot_mhd=mdot_obs*12
    
#! light speed in Km/s unit
c_Km = 2.99792458e5 
#! light speed in cm/s unit
c_cgs = 2.99792458e10 
sigma_thomson_cgs = 6.6524587158e-25
c_SI = 2.99792458e8 
G_SI = 6.674e-11
Msol_SI = 1.98892e30
Km2m = 1000.0
m2cm = 100.0

m_BH_SI = m_BH*Msol_SI
Rs_SI = 2.0*G_SI*m_BH_SI/(c_SI*c_SI)

#!* Gravitational radius
Rg_SI = 0.5*Rs_SI
Rg_cgs = Rg_SI*m2cm
L_xi_Source=xlum*1e38

vturb_max=dv_v*c_l/val_oversampling

ro_by_Rg = val_rin
DelFactorRo = 1.0001

while ro_by_Rg <= 1.1e7 :
    
    #!* distance from black hole in cylindrical co-ordinates-the radial distance
    rcyl_SI = ro_by_Rg*Rg_SI*r_A
    
    #!* distance from black hole in spherical co-ordinates
    Rsph_SI = rcyl_SI*(1.0+(z_A*z_A))**(1/2)  
    
    density_cgs = (mdot_mhd/(sigma_thomson_cgs*Rg_cgs))*rho_mhd*((rcyl_SI/Rg_SI)**(p_mhd-1.5))

    vel_r_cgs = c_cgs*vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
    vel_z_cgs = c_cgs*vel_z*((rcyl_SI/Rg_SI)**(-0.5))
    vel_obs_cgs = ((vel_r_cgs*np.cos(angle*np.pi/180.0))+(vel_z_cgs*np.sin(angle*np.pi/180.0)))

    # breakpoint()
    
    logxi = np.log10(L_xi_Source/(density_cgs*m2cm*Rsph_SI*m2cm*Rsph_SI))
    
    #!* Here we change the location of the first box as logxi is calculated for xstar to provide correct flux from luminosity. 

    if logxi <= 6.0: 
        # print("starting anchoring radius ro_by_Rg=",ro_by_Rg)

        # breakpoint()
        
        break
    else:
        #! A distance command : Step increase in the anchoring radius of the magnetic streamline
        ro_by_Rg = ro_by_Rg*DelFactorRo 

# breakpoint()

#!* After getting the starting value of ro_by_Rg from the above 'while' loop, fixing the values for 1st box.
Rsph_cgs_1st = Rsph_SI*m2cm
vobs_1st = vel_obs_cgs/(Km2m*m2cm)
logxi_1st = logxi
robyRg_1st = ro_by_Rg


# ! This is ro/Rg.. Position of streamline
ro_by_Rg = val_rout

rcyl_SI = ro_by_Rg*Rg_SI*r_A
Rsph_SI = rcyl_SI*np.sqrt(1.0+(z_A*z_A))

density_cgs = (mdot_mhd/(sigma_thomson_cgs*Rg_cgs))*rho_mhd*((rcyl_SI/Rg_SI)**(p_mhd-1.5))

vel_r_cgs = c_cgs*vel_r*((rcyl_SI/Rg_SI)**(-0.5)) 
vel_z_cgs = c_cgs*vel_z*((rcyl_SI/Rg_SI)**(-0.5))
vel_obs_cgs = ((vel_r_cgs*np.cos(angle*np.pi/180.0))+(vel_z_cgs*np.sin(angle*np.pi/180.0)))

logxi = np.log10(L_xi_Source/(density_cgs*m2cm*Rsph_SI*m2cm*Rsph_SI))

Rsph_cgs_last= Rsph_SI*m2cm
vobs_last= vel_obs_cgs/(Km2m*m2cm)
logxi_last= logxi
robyRg_last= ro_by_Rg
        
#!* Building the boxes based on Delr
delr_by_r= dr_r
Rsph_cgs_end= Rsph_cgs_1st
i_box = 0
i_last_box=0

#### This is very ugly and should be changed to the proper number of boxes, computed before this loop
vobs_start,robyRg_start,Rsph_cgs_start,density_cgs_start,logxi_start=np.zeros((5,int(1e5)))
vobs_mid,robyRg_mid,Rsph_cgs_mid,density_cgs_mid,logxi_mid=np.zeros((5,int(1e5)))
vobs_stop,robyRg_stop,Rsph_cgs_stop,density_cgs_stop,logxi_stop,NhOfBox=np.zeros((6,int(1e5)))

logxi_input=np.zeros(int(1e5))
    
#the loop builds the boxes iteration by iteration
#to implement the delta v threshold in an easy way, whenever the speed is too high, 
#we pass the loop iteration before updating i_box with a new delta_r suitable for the corresponding resolution

        
def func_density(x):
    return (mdot_mhd/(sigma_thomson_cgs*Rg_cgs))*rho_mhd*(x**(p_mhd-1.5))

def func_logxi(x):
    return np.log10(L_xi_Source/(func_density(x)*(x*Rg_cgs)**2))  

def func_vel_obs(x):
    return (c_cgs*vel_r*((x)**(-0.5))*np.cos(angle*np.pi/180.0))+(c_cgs*vel_z*((x)**(-0.5))*np.sin(angle*np.pi/180.0))
    
def func_nh_int(x,x0=300):
    return (mdot_mhd/(sigma_thomson_cgs*Rg_cgs))*rho_mhd/(p_mhd-0.5)*(x**(p_mhd-0.5)-x0**(p_mhd-0.5))*Rg_cgs

#defining the constant to get back to cylindric radius computations in which are made all of the MHD value computations
cyl_cst=(np.sqrt(1.0+(z_A*z_A)))

#defining a specific fonction which inverts the dr computation
def func_max_dr(x_start,vmax):
    
    '''
    computes the maximal dr for a given radius and velocity which would end up with bulk velocity (hence why we use only vel_r and vel_z)
    delta of vmax
    
    note: vmax should be in cgs, x_start in R_g
    '''

    #more complete version in simul_tools

    rad_angle=angle*np.pi/180
    
    x_end=1/((x_start)**(-1/2)-vmax/(c_cgs*(vel_r*np.cos(rad_angle)+vel_z*np.sin(rad_angle))))**2
    
    return x_end/x_start
           
nbox_rad=0
nbox_v=0

while Rsph_cgs_end<Rsph_cgs_last:

    #computing the dr/r of the velocity threshold used. We now use end_box-start_box instead of the midpoint of different boxes
    max_dr_res=func_max_dr(Rsph_cgs_end/(Rg_cgs*cyl_cst),vturb_max*1e5)
    
    #and ensuring we're sampling well enough
    dr_factor=min((2.0+delr_by_r)/(2.0-delr_by_r),max_dr_res)

    #last box flag
    if Rsph_cgs_last<Rsph_cgs_stop[i_box-1]*dr_factor:
        dr_factor= Rsph_cgs_last/Rsph_cgs_end
    
    if dr_factor==(2.0+delr_by_r)/(2.0-delr_by_r):
        nbox_rad+=1
    elif dr_factor==max_dr_res:
        nbox_v+=1
        
    Rsph_cgs_start[i_box]= Rsph_cgs_end
    density_cgs_start[i_box] = func_density(Rsph_cgs_start[i_box]/(Rg_cgs*cyl_cst))
    logxi_start[i_box] = func_logxi(Rsph_cgs_start[i_box]/(Rg_cgs*cyl_cst))
    vobs_start[i_box]=func_vel_obs(Rsph_cgs_start[i_box]/(Rg_cgs*cyl_cst))/(Km2m*m2cm)
    robyRg_start[i_box] =Rsph_cgs_start[i_box]/(cyl_cst*Rg_cgs*r_A)

    Rsph_cgs_stop[i_box]= Rsph_cgs_end*dr_factor
    density_cgs_stop[i_box] = func_density(Rsph_cgs_stop[i_box]/(Rg_cgs*cyl_cst))
    logxi_stop[i_box] = func_logxi(Rsph_cgs_stop[i_box]/(Rg_cgs*cyl_cst))
    vobs_stop[i_box]=func_vel_obs(Rsph_cgs_stop[i_box]/(Rg_cgs*cyl_cst))/(Km2m*m2cm)
    robyRg_stop[i_box] = Rsph_cgs_stop[i_box]/(cyl_cst*Rg_cgs*r_A)
     
    Rsph_cgs_mid[i_box]= (Rsph_cgs_start[i_box]+Rsph_cgs_stop[i_box])/2.0
    density_cgs_mid[i_box] = func_density(Rsph_cgs_mid[i_box]/(Rg_cgs*cyl_cst))
    logxi_mid[i_box] = func_logxi(Rsph_cgs_mid[i_box]/(Rg_cgs*cyl_cst))
    vobs_mid[i_box]=func_vel_obs(Rsph_cgs_mid[i_box]/(Rg_cgs*cyl_cst))/(Km2m*m2cm)
    robyRg_mid[i_box] = Rsph_cgs_mid[i_box]/(cyl_cst*Rg_cgs*r_A)
    
    #!* Recording quantities for the end point of the box*/
    
    #this one is computed as this because this is only used as a starting value for Xstar's logxi computation, BUT also to retrieve the radius.
    #Thus using R_start allows the box to correctly retrieve Rstart.
    logxi_input[i_box] = np.log10(L_xi_Source/(density_cgs_mid[i_box]*Rsph_cgs_start[i_box]*Rsph_cgs_start[i_box]))
    
    #!* Calculate Nh for the box
    
    NhOfBox[i_box] = density_cgs_mid[i_box]*(Rsph_cgs_stop[i_box]-Rsph_cgs_start[i_box])
    
    #!* Print the quantities in the log checking
    
    #! This step is for storing data for the last box
    if delr_by_r!=dr_r: 
        
        #adding a +1 here to switch from an i_box to the actual box number
        nbox_stop = i_box+1-1
        
    #!* Readjust the loop parameters
    
    
    if delr_by_r!=dr_r and i_last_box!=0 :
        st.text('I am after delr is changed'+str(delr_by_r))
        
        #!* maintaining the regular box no.
        i_box= i_box-1 
        i_last_box= i_last_box+1
        delr_by_r = dr_r
    
    Rsph_cgs_end = Rsph_cgs_stop[i_box]
    
    i_box = i_box + 1
        
#### something is wrong, not computing the last box

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

nbox=i_box

nbins=int(np.ceil(np.log(4*10**5)/np.log(1+dv_v/val_oversampling)))

tab_sampling, tab_time = st.tabs(["Box Sampling", "Computing time"])

with tab_sampling:

    
    st.header(r'Using $r_0=%.2e'%robyRg_1st+' Rg$ as first anchoring radius | $r_{S,0}=%.2e'%(Rsph_cgs_1st/Rg_cgs)+' Rg$')

    st.header(r'Boxes: $n_{tot}='+str(nbox)+'$ | $n_{rad}='+str(nbox_rad)+'$ | $n_v='+str(nbox_v)+'$')
    
    st.header(r'total number of bins with over of 3: $n_{bins}='+str(nbins)+'$')
    
    col_1, col_2= st.columns(2)
    
    with col_1:
        fig_dbox,ax_dbox=plt.subplots(1)
        
        plt.suptitle('Evolution of the box spherical radiuses')
        
        plt.xlabel('R (Rg)')
        plt.ylabel('Box number')
        plt.xscale('log')
        plt.axvline(x=Rsph_cgs_1st/Rg_cgs,ls='--',color='grey',label=r'$R_{sph}$ at 1st anch. radius')
        plt.axvline(x=Rsph_cgs_stop[-1]/Rg_cgs,ls='-.',color='grey',label=r'$R_{sph}$ at last anch. radius')
        plt.errorbar(Rsph_cgs_mid/Rg_cgs,range(1,nbox+1),xerr=np.array((Rsph_cgs_mid-Rsph_cgs_start,Rsph_cgs_stop-Rsph_cgs_mid))/Rg_cgs,ls='')
        plt.legend()
        try:
            st.pyplot(fig_dbox)
        except:
            st.stop()
            
    with col_2:
        fig_rdr,ax_rdr=plt.subplots(1)
        
        plt.suptitle('Evolution of the box radial resolution')
        
        plt.xlabel('R (Rg)')
        plt.ylabel(r'$\Delta r/r$')
        plt.xscale('log')
        plt.errorbar(Rsph_cgs_mid/Rg_cgs,(Rsph_cgs_stop-Rsph_cgs_start)/(Rsph_cgs_mid),xerr=np.array((Rsph_cgs_mid-Rsph_cgs_start,Rsph_cgs_stop-Rsph_cgs_mid))/Rg_cgs,ls='')
        
        try:
            st.pyplot(fig_rdr)
        except:
            st.stop()
    
    
        
    #### Plotting the quantities
    
    col_fig0,col_fig1, col_fig2,col_fig3= st.columns(4)
    
    with col_fig0:
        fig_dens_thresh,ax_dens_thresh=plt.subplots(1)

        #threshold for significant difference in xi from absorption before the start of the computation
        #different of compton thick
        nh_thresh=-np.log(1/1.105)/sigma_thomson_cgs
        
        plt.suptitle(r'Evolution of the column density at $R_{start}$'+'\n'+r'nh$_{\Delta log(\xi)=0.1}=%.2e'%nh_thresh+'$')
        
        plt.xlabel(r'start of the wind (Rg)')
        plt.ylabel(r'$nh$ (cgs)')
        plt.yscale('log')
        plt.xscale('log')
        
        plt.axvline(x=Rsph_cgs_1st/Rg_cgs,ls='--',color='grey',label=r'$R_{sph}$ at 1st anch. radius')
        
        plt.axhline(y=nh_thresh,ls=':',color='red',label=r'nh$_{\Delta log(\xi)=0.1}$')
        # plt.axvline(x=Rsph_cgs_stop[-1]/Rg_cgs,ls='-.',color='grey',label=r'$R_{sph}$ at last anch. radius')
        
        plt.plot(np.logspace(1,np.log10(Rsph_cgs_start[0]/Rg_cgs)),func_nh_int(Rsph_cgs_start[0],np.logspace(1,np.log10(Rsph_cgs_start[0]/Rg_cgs))))
        plt.legend()
        try:
            st.pyplot(fig_dens_thresh)
        except:
            st.stop()
                    
        fig_dens_thresh,ax_dens_thresh=plt.subplots(1)

        plt.suptitle('Evolution of the column density of the boxes')
        
        plt.xlabel(r'$n_{box}$')
        plt.ylabel(r'$nh_{box}$ (cgs)')
        plt.yscale('log')
        plt.errorbar(range(1,nbox+1),NhOfBox,xerr=0.5,ls='')
        plt.legend()
        try:
            st.pyplot(fig_dens_thresh)
        except:
            st.stop()
            
    with col_fig1:
        fig_density,ax_density=plt.subplots(1)
        
        plt.suptitle('Evolution of the density of the solution')
        
        plt.xlabel('R (Rg)')
        plt.ylabel(r'$\rho_{cgs}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.axvline(x=Rsph_cgs_1st/Rg_cgs,ls='--',color='grey',label=r'$R_{sph}$ at 1st anch. radius')
        plt.axvline(x=Rsph_cgs_stop[-1]/Rg_cgs,ls='-.',color='grey',label=r'$R_{sph}$ at last anch. radius')
        plt.plot(np.logspace(1,7),func_density(np.logspace(1,7)))
        plt.legend()
        try:
            st.pyplot(fig_density)
        except:
            st.stop()
        
        fig_density,ax_density=plt.subplots(1)
        
        plt.suptitle('Evolution of the density of the boxes')
        
        plt.xlabel(r'$n_{box}$')
        plt.ylabel(r'$\rho_{cgs}$')
        plt.yscale('log')
        plt.errorbar(range(1,nbox+1),density_cgs_mid,xerr=0.5,ls='')
        plt.legend()
        try:
            st.pyplot(fig_density)
        except:
            st.stop()
    
    with col_fig2:
        fig_logxi,ax_logxi=plt.subplots(1)
        
        plt.suptitle(r'Evolution of the $\log(\xi)$ of the solution')
        
        plt.xlabel('R (Rg)')
        plt.ylabel(r'$\log(\xi)$')
        plt.xscale('log')
        plt.axvline(x=Rsph_cgs_1st/Rg_cgs,ls='--',color='grey',label=r'$R_{sph}$ at 1st anch. radius')
        plt.axvline(x=Rsph_cgs_stop[-1]/Rg_cgs,ls='-.',color='grey',label=r'$R_{sph}$ at last anch. radius')
        plt.plot(np.logspace(1,7),func_logxi(np.logspace(1,7)))
        plt.legend()
        try:
            st.pyplot(fig_logxi)
        except:
            st.stop()
        
        fig_logxi,ax_logxi=plt.subplots(1)
        
        plt.suptitle(r'Evolution of the $\log(\xi)$ of the boxes')
        
        plt.xlabel(r'$n_{box}$')
        plt.ylabel(r'$\log(\xi)$')
        plt.yscale('log')
        plt.errorbar(range(1,nbox+1),logxi_mid,xerr=0.5,ls='')
        plt.legend()
        try:
            st.pyplot(fig_logxi)
        except:
            st.stop()
            
    with col_fig3:
        fig_vel_obs,ax_vel_obs=plt.subplots(1)
        
        plt.suptitle('Evolution of the bulk velocity of the solution')
        
        plt.xlabel('R (Rg)')
        plt.ylabel(r'$v_{obs}$ (km/s)')
        plt.xscale('log')
        plt.yscale('log')
        plt.axvline(x=Rsph_cgs_1st/Rg_cgs,ls='--',color='grey',label=r'$R_{sph}$ at 1st anch. radius')
        plt.axvline(x=Rsph_cgs_stop[-1]/Rg_cgs,ls='-.',color='grey',label=r'$R_{sph}$ at last anch. radius')
        plt.plot(np.logspace(1,7),func_vel_obs(np.logspace(1,7))/1e5)
        plt.legend()
        try:
            st.pyplot(fig_vel_obs)
        except:
            st.stop()
        
        fig_vturb,ax_vturb=plt.subplots(1)
        
        plt.suptitle(r'Evolution of $\Delta v_{bulk}$ of the boxes')
        
        plt.xlabel(r'$n_{box}$')
        plt.ylabel(r'$v_{turb}$ (km/s)')
        plt.yscale('log')
        plt.errorbar(range(1,nbox+1),(vobs_start-vobs_stop),xerr=0.5,ls='')
        plt.axhline(y=vturb_max,ls='--',color='red',label=r'maximum allowed with current resolution')
        plt.legend()
        
        try:
            st.pyplot(fig_vturb)
        except:
            st.stop()
        

nbins_test=15475
nbox_test=39

#in hours
time_test=2.5

with tab_time:
    
    st.header('Parameter sampling')
    
    def_val=3
    
    n_rout=st.number_input(r'number of values for $R_{end}$',value=def_val,step=1,min_value=1)
    
    st.markdown('''Note: Additional $R_{end}$ values are considered differently, as if adding 1 box to the computing time.''')
    st.markdown('''This works only if using the biggest $R_{end}$ in the box options''')
    
    
    n_rin=st.number_input(r'number of values for $R_{start}$',value=def_val,step=1,min_value=1)
    
    n_p=st.number_input(r'number of values for $p$ (ejection index)',value=def_val,step=1,min_value=1)
    
    n_mu=st.number_input(r'number of values for $\mu$ (magnetization)',value=def_val,step=1,min_value=1)
    
    n_angle=st.number_input(r'number of values for $\theta$ (angle)',value=def_val,step=1,min_value=1)
    
    n_SED=st.number_input(r'number of SEDs',value=def_val,step=1,min_value=1)
    
    n_lum=st.number_input(r'number of luminosities/mdot/SEDs normalization',value=def_val,step=1,min_value=1)
    
    st.header('Solution computing time evolution')
    
    pl_index_nbins=st.number_input(r'Box computation time powerlaw index for $n_{bins}$',value=0.5,format='%.1e')
    pl_index_nbox=st.number_input(r'Box computation time powerlaw index for $n_{box}$',value=1.,format='%.1e')

    time_sim=(nbins/nbins_test)**pl_index_nbins*((nbox+n_rout-1)/nbox_test)**pl_index_nbox*time_test
    
    st.header(r'Single solution computation time: $t_{sim}\sim%.2f'%time_sim+'\mathrm{h} $')
    

    
    st.header('Table computing time')
    
    n_parall=st.number_input(r'Number of cores used:',value=6,step=1)
    
    table_time=time_sim*n_rin*n_p*n_mu*n_angle*n_SED*n_lum/n_parall
    
    st.header(r'Table computation time: $t_{table}\sim%.2f'%table_time+'\mathrm{h} $ (%.1f'%(table_time/24)+' j)')
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:21:46 2022

@author: parrama
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

#distance factor for the flux conversion later on
def dist_factor(d):
    
    '''
    d in kpc
    '''
    
    return 4*np.pi*(d*1e3*3.086e18)**2

#L_Edd unit factor
def Edd_factor(d,m):
    
    '''m in solar masses'''
    
    return dist_factor(d)/(1.26e38*m)

d_range=np.logspace(0,np.log10(20),200)

m_range=np.logspace(0,np.log10(20),200)

Edd_mesh=np.array([Edd_factor(d_range,m_range[i]) for i in range(len(m_range))])

Lum=1.9782914437268429e-10

#Suzaku
# Lum=2.5810032434461534e-13

L_edd_mesh=Lum*Edd_mesh

plt.figure(figsize=(8,6))

# plt.xscale('log')
plt.yscale('log')


cmesh=plt.pcolormesh(d_range,m_range,L_edd_mesh,norm=col.LogNorm())

# for i in range(3):
    
c_ls=['dotted','dashdot','dashed','solid']
c_labels=['1e-5','1e-4','1e-3','1e-2']
    
contours_Ledd=plt.contour(d_range,m_range,L_edd_mesh,levels=[1e-5,1e-4,1e-3,1e-2],colors='black',
                              linewidths=2,linestyles=c_ls)


for i in range(1,4):
    #there is an issue in the current matplotlib version with contour labels crashing the legend so we use proxies instead        
    plt.plot([], [], ls=c_ls[i], label=c_labels[i],color='black',lw=2)
   
    # contours_Ledd.collections[i].set_label(c_labels[i])
    
plt.xlabel('distance (kpc)')
plt.ylabel(r'mass ($M_{\odot}$)')

plt.colorbar(cmesh,label=r'Luminosity ($L_X/L_{Edd}$)')

plt.legend()

plt.title("Evolution of IGRJ17451's luminosity for the XMM exposure")


#%% lightcurves

from stingray import Lightcurve

lc_xi0=Lightcurve.read('lc_0.lc','ogip')
lc_xi1=Lightcurve.read('lc.evt','ogip')
lc_xi0bg=Lightcurve.read('lc_0bg.lc','ogip')
lc_xi1bg=Lightcurve.read('lc_bg.lc','ogip')



lc_net_xi0=lc_xi0-lc_xi0bg
lc_net_xi1=lc_xi1-lc_xi1bg

fig_lc=plt.figure()

plt.xlabel('time (s)')
plt.ylabel('counts')

plt.title('IGRJ 17451-3022 Suzaku net lightcurves')

plt.plot(lc_net_xi0.time-lc_net_xi0.time[0],lc_net_xi0.counts,label='xi0')
plt.plot(lc_net_xi1.time-lc_net_xi0.time[0],lc_net_xi1.counts,label='xi1')

plt.legend()

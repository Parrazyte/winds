#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:51:19 2022

@author: parrama
"""

import numpy as np

#few constants in km/s
c_light=299792.458

#table of delchis necessary for the 99% significance of a number of d.o.f. added in a model (1 to 10 here)
#(from https://fr.wikipedia.org/wiki/Loi_du_%CF%87%C2%B2)
sign_delchis_table=[6.63,9.21,11.34,13.28,15.09,16.81,18.48,20.09,21.67,23.21]

ftest_threshold=0.01

def ravel_ragged(array):
    
    '''ravels a 2/3d array/list even with ragged nested sequences'''

    #leaving if the array is 0d
    
    if type(array) not in [np.ndarray,list] or len(array)==0:
        return array
    #or 1d
    if type(array[0]) not in [np.ndarray,list]:
        return array

    #testing if the array is 3d
    if type(array[0][0]) in [np.ndarray,list]:
        return np.array([array[i][j][k] for i in range(len(array)) for j in range(len(array[i])) for k in range(len(array[i][j]))])
    else:
        return np.array([array[i][j] for i in range(len(array)) for j in range(len(array[i]))])
    
####Line informations
'''Line informations'''

lines_std={
                    'FeKaem':r'Fe K$\alpha$',
                    'FeKbem':r'Fe K$\beta$',
                    'FeDiazem':'Fe Diaz',
                  'FeKa25abs':r'FeXXV K$\alpha$',
                  'FeKa26abs':r'FeXXVI K$\alpha$',
                  'NiKa27abs':r'NiXXVII K$\alpha$',
                  'FeKb25abs':r'FeXXV K$\beta$',
                  'FeKb26abs':r'FeXXVI K$\beta$',
                  'FeKg26abs':r'FeXXVI K$\gamma$',
                  #these 4 are put at the end so they are not used in fakes computation etc since the computation will stop 
                  #at 3+the number of abslines
                  'FeKa25em':r'FeXXV K$\alpha$',
                  'FeKa26em':r'FeXXVI K$\alpha$',
                  #this are the narrow equivalent to FeKa and FeKb, but they are used as much more "physical" lines and as such 
                  #have restrained energy ranges compared to their broad counterparts
                  'FeKa0em':r'Fe K$\alpha$',
                  'FeKb0em':r'Fe K$\beta$'}

lines_std_names=list(lines_std.keys())

#Note : some line positions are explicitely assumed in some parts of the code (notably the first 3 being the broad emission lines)

#number of absorption lines in the current model list
n_absline=len([elem for elem in lines_std_names if 'abs' in elem])

range_absline=range(n_absline)

lines_e_dict={
                    'FeKaem':[6.4],
                    'FeKbem':[7.06],
                    'FeDiazem':[7.0],
                  'FeKa25abs':[6.7,-5000,10000],
                  'FeKa26abs':[6.97,-5000,10000],
                  'NiKa27abs':[7.8,-5000,3000],
                  'FeKb25abs':[7.89,-5000,10000],
                  'FeKb26abs':[8.25,-5000,10000],
                  'FeKg26abs':[8.7,-5000,10000],
                  #these 4 are put at the end so they are not used in fakes computation etc since the computation will stop 
                  #at 3+the number of abslines
                  'FeKa25em':[6.7,-3000,3000],
                  'FeKa26em':[6.97,-3000,3000],
                  #this are the narrow equivalent to FeKa and FeKb, but they are used as much more "physical" lines and as such 
                  #have restrained energy ranges compared to their broad counterparts
                  'FeKa0em':[6.4,-3000,3000],
                  'FeKb0em':[7.06,-3000,3000]}

# higher bshift range for IGRJ17091-3624
# lines_e_dict={
#                     'FeKaem':[6.4],
#                     'FeKbem':[7.06],
#                     'FeDiazem':[7.0],
#                   'FeKa25abs':[6.7,-5000,15000],
#                   'FeKa26abs':[6.97,-5000,15000],
#                   'NiKa27abs':[7.8,-5000,3000],
#                   'FeKb25abs':[7.89,-5000,15000],
#                   'FeKb26abs':[8.25,-5000,15000],
#                   'FeKg26abs':[8.7,-5000,15000]}

#restrict to Miller 2006 values for H1743-322 11048
# lines_e_dict={
#                     'FeKaem':[6.4],
#                     'FeKbem':[7.06],
#                     'FeDiazem':[7.0],
#                   'FeKa25abs':[6.7,-480,480],
#                   'FeKa26abs':[6.97,0,840],
#                   'NiKa27abs':[7.8,-5000,3000],
#                   'FeKb25abs':[7.89,-5000,10000],
#                   'FeKb26abs':[8.25,-5000,10000],
#                   'FeKg26abs':[8.7,-5000,10000]}


#forcing broad gaussians to avoid degeneracies with the absorption line
#resolved widths for abs lines (when using HETG)
#this value is not used if the line is specified as narrow
lines_w_dict={
                'FeKaem':[0.2,0.1,1.5],
                'FeKbem':[0.2,0.1,1.5],
                'FeDiazem':[0.2,0.1,1.5],
                  'FeKa25abs':[1e-3,0.,0.05],
                  'FeKa26abs':[1e-3,0.,0.05],
                  'NiKa27abs':[1e-3,0.,0.05],
                  'FeKb25abs':[1e-3,0.,0.05],
                  'FeKb26abs':[1e-3,0.,0.05],
                  'FeKg26abs':[1e-3,0.,0.05],
                  'FeKa0em':[1e-3,0.,0.05],
                  'FeKb0em':[1e-3,0.,0.05],
                  'FeKa25em':[1e-3,0.,0.05],
                  'FeKb26em':[1e-3,0.,0.05],
                  }

#link groups to tie line energies together
link_groups=np.array([['FeKa25abs','FeKb25abs'],['FeKa26abs','FeKb26abs','FeKg26abs'],['NiKa27abs']],dtype=object)
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

#same with 99.7% (3 sigma)
sign_3sig_delchi_stable=[8.81]

def_ftest_threshold=0.01

def_ftest_leeway=0.02

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
                  'FeKb25abs':[7.88,-5000,10000],
                  'FeKb26abs':[8.25,-5000,10000],
                  'FeKg26abs':[8.7,-5000,10000],
                  #these 4 are put at the end so they are not used in fakes computation etc since the computation will stop 
                  #at 3+the number of abslines
                  'FeKa25em':[6.7,-5000,10000],
                  'FeKa26em':[6.97,-5000,10000],
                  #this are the narrow equivalent to FeKa and FeKb, but they are used as much more "physical" lines and as such 
                  #have restrained energy ranges compared to their broad counterparts
                   'FeKa0em':[6.4,-10000,10000],
                   'FeKb0em':[7.06,-10000,10000]}
    
#+ line resolved

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
#resolved widths for narrow-ish abs/em lines
#this value is not used if the line is specified as narrow
lines_w_dict={
                'FeKaem':[0.3,0.2,0.5],
                'FeKbem':[0.3,0.2,0.5],
                'FeDiazem':[0.2,0.1,1.5],
                #note: starting with 1e-3 can make the fit gets stuck at 0 width (doesn't get out even with errors) although higher width values
                #are improving
                
                  'FeKa25abs':[1e-2,0.,0.05],
                  'FeKa26abs':[1e-2,0.,0.05],
                  'NiKa27abs':[1e-2,0.,0.05],
                  'FeKb25abs':[1e-2,0.,0.05],
                  'FeKb26abs':[1e-2,0.,0.05],
                  'FeKg26abs':[1e-2,0.,0.05],
                  #these are emission lines and thus we let some have bigger widths even without being broad
                  'FeKa0em':[1e-2,0.,0.05],
                  'FeKb0em':[1e-2,0.,0.05],
                  'FeKa25em':[1e-2,0.,0.05],
                  'FeKa26em':[1e-2,0.,0.05],
                  }

lines_broad_w_dict={
                  'FeKa0em':[0.3,0.2,0.7],
                  'FeKb0em':[0.3,0.2,0.7],
                  'FeKa25em':[0.3,0.2,0.7],
                  'FeKa26em':[0.3,0.2,0.7],
                  }

#for V404/V4641
# lines_broad_w_dict={
#                   'FeKa0em':[0.3,0.2,1.],
#                   'FeKb0em':[0.3,0.2,1.],
#                   'FeKa25em':[0.3,0.2,1.],
#                   'FeKa26em':[0.3,0.2,1.],
#                   }

# lines_broad_w_dict={
#                   'FeKa0em':[0.3,0.2,1.5],
#                   'FeKb0em':[0.3,0.2,1.5],
#                   'FeKa25em':[0.3,0.2,1.5],
#                   'FeKa26em':[0.3,0.2,1.5],
#                   }

#link groups to tie line energies together
link_groups=np.array([['FeKa25abs','FeKb25abs'],['FeKa26abs','FeKb26abs','FeKg26abs'],['NiKa27abs']],dtype=object)

def ang2kev(x):

    '''note : same thing on the other side due to the inverse
    
    also same thing for mAngtoeV'''

    return 12.398/x

def model_list(model_id='lines',give_groups=False):
    
    '''
    wrapper for the fitmod class with a bunch of models
        
    Model types:
        -lines : add high energy lines to a continuum.
                Available components (to be updated):
                    -2 gaussian emission lines at roughly >6.4/7.06 keV (narrow or not)
                    -6 narrow absorption lines (see addcomp)
                    -the diskbb and powerlaw to try to add them if they're needed
                    
                at every step, tries adding every remaining line in the continuum, and choose the best one out of all
                only considers each addition if the improved chi² is below a difference depending of the number of 
                free parameters in the added component
                
                interact comps lists the components susceptibles of interacting together during the fit 
                The other components of each group will be the only ones unfrozen when computing errors of a component in the group
                
        -lines_diaz:
                Almost the same thing but with a single emission component with [6.-8.keV] as energy interval
             
        -cont: create a (potentially absorbed) continuum with a diskbb and/or powerlaw
                available components:
                    -global phabs
                    -powerlaw
                    -diskbb
    '''
    
    if model_id=='lines_resolved':
        avail_comps=['FeKa0em_bgaussian','FeKb0em_bgaussian','FeKa25em_gaussian','FeKa26em_gaussian',
                      'FeKa25abs_agaussian','FeKa26abs_agaussian','NiKa27abs_agaussian',
                      'FeKb25abs_agaussian','FeKb26abs_agaussian','FeKg26abs_agaussian']
        
    if model_id=='lines_narrow':
        avail_comps=['FeKa0em_bgaussian','FeKb0em_bgaussian','FeKa25em_gaussian','FeKa26em_gaussian',
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
    if model_id=='lines_laor':
        '''
        subset of lines_narrow for spectra with huge emission in continuum. No emission line is allowed here
        '''
        avail_comps=['FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
    if model_id=='lines_emiwind':
        
        '''
        Model to test residuals with emission lines only in neutral wind emitters (V404 Cyg & V4641 Sgr)
        
        Contains :
            -1 broad neutral emission component
            -4 "narrow-ish" emission line components
            -6 "narrow-ish" absorption line components (can be disabled with no_abslines in the main script
        '''

        avail_comps=['FeKa0em_bgaussian','FeKa0em_gaussian','FeKb0em_gaussian','FeKa25em_gaussian','FeKa26em_gaussian',
                     'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                                   'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
    if model_id=='cont_laor':

        '''
        subset of continuum for spectra with huge emission
        '''

        avail_comps=['glob_constant','glob_phabs','cont_diskbb','cont_powerlaw','FeKa_laor']
        
        interact_groups=avail_comps
    
    if model_id=='cont':
        
        avail_comps=['glob_constant','glob_phabs','cont_diskbb','cont_powerlaw']
        
        interact_groups=avail_comps      

    if model_id=='cont_noabs':
        
        avail_comps=['glob_constant','cont_diskbb','cont_powerlaw']
        
        interact_groups=avail_comps   
    
    if model_id=='lines':
        
        '''
        Notes : no need to include continuum componennts here anymore since we now use the continuum fitmod as argument in the lines fitmod
        '''
        
        # avail_comps=['FeKaem_gaussian','FeKa26abs_nagaussian']
        avail_comps=['FeKaem_gaussian','FeKbem_gaussian',
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
        

    if model_id=='lines_resolved_noem':
        avail_comps=['FeKa25abs_agaussian','FeKa26abs_agaussian','NiKa27abs_agaussian',
                      'FeKb25abs_agaussian','FeKb26abs_agaussian','FeKg26abs_agaussian']
        
    if model_id=='lines_old':
        
        # avail_comps=['FeKaem_gaussian','FeKa26abs_nagaussian']
        avail_comps=['cont_diskbb','cont_powerlaw','FeKaem_gaussian','FeKbem_gaussian',
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
    
    if model_id=='lines_resolved_old':
        avail_comps=['cont_diskbb','cont_powerlaw','FeKaem_gaussian','FeKbem_gaussian',
                      'FeKa25abs_agaussian','FeKa26abs_agaussian','NiKa27abs_agaussian',
                      'FeKb25abs_agaussian','FeKb26abs_agaussian','FeKg26abs_agaussian']
        
        
    elif model_id=='lines_ns':
        # avail_comps=['FeKaem_gaussian','FeKa26abs_nagaussian']
        avail_comps=['cont_diskbb','cont_powerlaw','cont_bb','FeKaem_gaussian','FeKbem_gaussian',
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
    elif model_id=='lines_ns_noem':
        avail_comps=['cont_diskbb','cont_powerlaw','cont_bb','FeKaem_gaussian','FeKbem_gaussian',''
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
                
    if model_id=='lines_diaz':
        avail_comps=['FeDiaz_gaussian',
                     'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
            
    if model_id=='cont_bkn':
        avail_comps=['bknpow','glob_phabs']
        
    if give_groups:
        return avail_comps,interact_groups
    else:
        return avail_comps
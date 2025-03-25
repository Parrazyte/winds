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

#1,2,3 sigmas delchi for a single d.d.f.
# (computed with http://courses.atlas.illinois.edu/spring2016/STAT/STAT200/pchisq.html)
sign_sigmas_delchi_1dof=[1.,4.,9.]

def_ftest_threshold=0.01

def_ftest_leeway=0.02
    
####Line informations
'''Line informations'''

lines_std={         #don't change the first 8, there are explicit calls in the code
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
                  'FeKb0em':r'Fe K$\beta$',
                  'calNICERSiem':r'Nicer Cal',

                    'NeKa10abs': r'NeX K$\alpha$',

                    'SiKa14abs': r'SiXIV K$\alpha$',
                  'SKa16abs':r'SXVI K$\alpha$',
                  'CaKa20abs':r'CaXX K$\alpha$',
                  # 'CrKa23abs':r'CrXXIII K$\alpha$',
                  #resolved lines
                'FeKa25Zabs':r'FeXXV K$\alpha$ (z)',
                'FeKa25Yabs': r'FeXXV K$\alpha$ (y)',
                'FeKa25Xabs': r'FeXXV K$\alpha$ (x)',
                'FeKa25Wabs': r'FeXXV K$\alpha$ (w)',

                'FeKb25p1abs': r'FeXXV K$\beta$ (P$^{1/2}$)',
                'FeKb25p3abs': r'FeXXV K$\beta$ (P$^{3/2}$)',

                'FeKa26p1abs': r'FeXXVI K$\alpha$ (P$^{1/2}$)',
                'FeKa26p3abs': r'FeXXVI K$\alpha$ (P$^{3/2}$)',

                'FeKb26p1abs': r'FeXXVI K$\beta$ (P$^{1/2}$)',
                'FeKb26p3abs': r'FeXXVI K$\beta$ (P$^{3/2}$)',

                'FeKg26p1abs': r'FeXXVI K$\gamma$ (P$^{1/2}$)',
                'FeKg26p3abs': r'FeXXVI K$\gamma$ (P$^{3/2}$)',

                'NiKa27Zabs':r'NiXXVII K$\alpha$ (z)',
                'NiKa27Yabs': r'NiXXVII K$\alpha$ (y)',
                'NiKa27Xabs': r'NiXXVII K$\alpha$ (x)',
                'NiKa27Wabs': r'NiXXVII K$\alpha$ (w)',

                'CrKa23Zabs': r'CrXXIII K$\alpha$ (z)',
                'CrKa23Yabs': r'CrXXIII K$\alpha$ (y)',
                'CrKa23Xabs': r'CrXXIII K$\alpha$ (x)',
                'CrKa23Wabs': r'CrXXIII K$\alpha$ (w)',

}

lines_std_names=list(lines_std.keys())

#Note : some line positions are explicitely assumed in some parts of the code (notably the first 3 being the broad emission lines)

#number of absorption lines in the current model list
#for now restricting to the iron lines to avoid issues when using visual_line

n_absline=6

#too complicated nowadays
# n_absline=len([elem for elem in lines_std_names if 'abs' in elem and 'Si' not in elem])

range_absline=range(n_absline)

lines_e_dict={
                    'FeKaem':[6.404],
                    'FeKbem':[7.06],
                    'FeDiazem':[7.0],
                  'FeKa25abs':[6.7,-5000,10000],
                  'FeKa26abs':[6.97,-5000,10000],
                  #uncertainty of 5eV
                  'NiKa27abs':[7.793,-5000,3000],
                  'FeKb25abs':[7.88,-5000,10000],
                  'FeKb26abs':[8.25,-5000,10000],
                  'FeKg26abs':[8.7,-5000,10000],
                  #these 4 are put at the end so they are not used in fakes computation etc since the computation will stop 
                  #at 3+the number of abslines
                  'FeKa25em':[6.7,-5000,10000],
                  'FeKa26em':[6.97,-5000,10000],
                  #this are the narrow equivalent to FeKa and FeKb, but they are used as much more "physical" lines and as such 
                  #have restrained energy ranges compared to their broad counterparts

                    #see https://iopscience.iop.org/article/10.3847/0004-637X/818/2/164/pdf for precise edge
                    #and neutral K energies if needed
                    #this one is the average of the 6.391 (1/3) and 6.404 (2/3)
                   'FeKa0em':[6.400,-10000,10000],
                   'FeKb0em':[7.06,-10000,10000],
                   'calNICERSiem':[1.74],

                  'NeKa10abs':[1.02180,-3000,3000],
                  #energy from http://www.atomdb.org/Webguide/transition_information.php?lower=1s&upper=2p&z0=14&z1=13
                  'SiKa14abs':[2.005494,-3000,3000],
                  'SKa16abs':[2.6215,-3000,3000],
                  'CaKa20abs':[4.10505,-3000,3000],

                  #
                  #'CaKb20abs': [around 4.85, -3000, 3000],

    # resolved lines (from NIST 2023)
                #these have uncertainties of 1.4 meV

                #Z is the forbidden line F
                'FeKa25Zabs': [6.6363,-3000,3000],

                #Y and X are the intercombination (I) lines. Summed for diagnostics
                'FeKa25Yabs':  [6.6676,-3000,3000],
                'FeKa25Xabs':  [6.6823,-3000,3000],

                #W is the resonance line R
                'FeKa25Wabs':  [6.7004,-3000,3000],

                #these have uncertainties of 4-6 meV
                'FeKb25p1abs':  [7.872,-3000,3000],
                'FeKb25p3abs':  [7.881,-3000,3000],

                'FeKa26p1abs':  [6.9520,-3000,3000],
                'FeKa26p3abs':  [6.9732,-3000,3000],

                'FeKb26p1abs':  [8.2464,-3000,3000],
                'FeKb26p3abs':  [8.2527,-3000,3000],

                'FeKg26p1abs':  [8.6986,-3000,3000],
                'FeKg26p3abs':  [8.7012,-3000,3000],

                #this one has uncertainty of 1.3 meV
                'NiKa27Zabs': [7.7316, -3000, 3000],
                'NiKa27Yabs': [7.7657, -3000, 3000],
                'NiKa27Xabs': [7.7864, -3000, 3000],
                'NiKa27Wabs': [7.8056, -3000, 3000],

                # precision 1.2meV
                'CrKa23Zabs': [5.6269, -3000, 3000],
                'CrKa23Yabs': [5.6548, -3000, 3000],
                'CrKa23Xabs': [5.6651, -3000, 3000],
                'CrKa23Wabs': [5.6821, -3000, 3000],

}


#note: if find a line at 2.47 keV, careful about confusion with an interstellar gas S2 3p line at 2.47 keV
#for galactic sources
# see https://ui.adsabs.harvard.edu/abs/2009ApJ...695..888U/abstract p. 892 (3.2)

#+ line resolved

# higher bshift range for IGRJ17091-3624
# lines_e_dict={
#                     'FeKaem':[6.404],
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
#                     'FeKaem':[6.404],
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
                  'calNICERSiem':[0.1,0.05,0.2],
                  'SiKa14abs': [1e-2,0.,0.05],

                'NeKa10abs': [1e-2,0.,0.05],
                'SKa16abs': [1e-2,0.,0.05],
                'CaKa20abs': [1e-2,0.,0.05],
                'CrKa23abs': [1e-2, 0., 0.05],

    # resolved lines

            'FeKa25Zabs': [1e-3,0.,0.05],
            'FeKa25Yabs': [1e-3,0.,0.05],
            'FeKa25Xabs': [1e-3,0.,0.05],
            'FeKa25Wabs': [1e-3,0.,0.05],

            'FeKb25p1abs': [1e-3,0.,0.05],
            'FeKb25p3abs': [1e-3,0.,0.05],

            'FeKa26p1abs': [1e-3,0.,0.05],
            'FeKa26p3abs': [1e-3,0.,0.05],

            'FeKb26p1abs': [1e-3,0.,0.05],
            'FeKb26p3abs': [1e-3,0.,0.05],

            'FeKg26p1abs': [1e-3,0.,0.05],
            'FeKg26p3abs': [1e-3,0.,0.05],

            'NiKa27Zabs': [1e-3, 0., 0.05],
            'NiKa27Yabs': [1e-3, 0., 0.05],
            'NiKa27Xabs': [1e-3, 0., 0.05],
            'NiKa27Wabs': [1e-3, 0., 0.05],

            'CrKa23Zabs': [1e-3, 0., 0.05],
            'CrKa23Yabs': [1e-3, 0., 0.05],
            'CrKa23Xabs': [1e-3, 0., 0.05],
            'CrKa23Wabs': [1e-3, 0., 0.05],

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

def model_list(model_id='lines',give_groups=False,sat_list=[]):
    
    '''
    wrapper for the fitmod class with a bunch of models

    if sat_list is a list and a _var model is selected,
     will be used to add calibration components depending on the instruments in the list

    Model types:
        -lines : add high energy lines to a continuum.
                Available components (to be updated):
                    -2 gaussian emission lines at roughly >6.404/7.06 keV (narrow or not)
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

    interact_groups=None

    if model_id=='lines_XRISM_V4641_2024':
        avail_comps=['FeKa25Yabs_gaussian','FeKa25Wabs_gaussian',
                     'FeKa26p1abs_gaussian','FeKa26p3abs_gaussian',
                     'SKa16abs_gaussian','CaKa20abs_gaussian']

    if model_id=='lines_resolved':
        avail_comps=['FeKa0em_bgaussian','FeKb0em_bgaussian','FeKa25em_gaussian','FeKa26em_gaussian',
                      'FeKa25abs_agaussian','FeKa26abs_agaussian','NiKa27abs_agaussian',
                      'FeKb25abs_agaussian','FeKb26abs_agaussian','FeKg26abs_agaussian']

    if model_id=='lines_em':
        avail_comps=['FeKa0em_bgaussian','FeKb0em_bgaussian','FeKa25em_gaussian','FeKa26em_gaussian']

    if model_id=='lines_em_V4641Sgr':
        avail_comps=['FeKa0em_gaussian','FeKa25em_gaussian','FeKa26em_gaussian']

    if model_id=='lines_narrow':
        avail_comps=['FeKa0em_bgaussian','FeKb0em_bgaussian','FeKa25em_gaussian','FeKa26em_gaussian',
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']

    #for quicker autofit tests
    if model_id=='single_line_narrow':
        avail_comps=['FeKa26abs_nagaussian']


    if model_id=='lines_narrow_noem':
        avail_comps=['FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
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
            -6 "narrow-ish" absorption line components (can be disabled with no_abslines in the main script)
        '''

        avail_comps=['FeKa0em_bgaussian','FeKa0em_gaussian','FeKb0em_gaussian','FeKa25em_gaussian','FeKa26em_gaussian',
                     'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                                   'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']


    if model_id=='thcont_var':
        avail_comps = ['cont_diskbb', 'disk_thcomp', 'glob_TBabs']

        if "Suzaku" in sat_list:
            #note: this one still needs to be tested with multi sats
            avail_comps+=['Suzaku_crabcorr']
        else:
            avail_comps+=['glob_constant']

        if 'NICER' in sat_list:
            avail_comps+=['calNICERSiem_gaussian','calNICER_edge']

        if 'NuSTAR' in sat_list:
            avail_comps+=['calNuSTAR_edge']

        interact_groups=None

    if model_id == 'thcont_NICER':
        '''
        subset of continuum with thcomp for better broad band work + NICER calibration components
        '''

        avail_comps = ['cont_diskbb', 'disk_thcomp', 'glob_TBabs', 'glob_constant',
                       'calNICERSiem_gaussian','calNICER_edge']

        interact_groups = None

    if model_id=='cont_laor':

        '''
        subset of continuum for spectra with huge emission
        '''

        avail_comps=['glob_constant','glob_phabs','cont_diskbb','cont_powerlaw','FeKa_laor']
        
        interact_groups=avail_comps

    if model_id=='cont_detailed':

        '''
        subset of continuum with NICER and NuSTAR calibration components
        
        NOTE: the order here matters because we force the use of all these components in this order
        '''

        avail_comps = ['cont_diskbb', 'glob_TBabs','cont_powerlaw', 'glob_constant',
                       'calNICERSiem_gaussian','calNICER_edge','calNuSTAR_edge']

        interact_groups = None

    if model_id=='cont_NICER':
        '''
        subset of continuum with NICER calibration components
        '''

        avail_comps = ['cont_diskbb', 'glob_TBabs', 'cont_powerlaw', 'glob_constant',
                       'calNICERSiem_gaussian','calNICER_edge']

        interact_groups = None

    if model_id == 'cont_NuSTAR':
        '''
        subset of continuum with NuSTAR calibration components
        '''

        avail_comps = ['cont_powerlaw','cont_diskbb','glob_TBabs', 'glob_constant',
                       'calNuSTAR_edge']

        interact_groups = None

    if model_id=='nthcont':

        '''
        subset of continuum with nthcomp for better broad band work + NuSTAR calibration components
        '''

        avail_comps = ['cont_diskbb','disk_nthcomp','glob_TBabs', 'glob_constant']

        interact_groups = None

    if model_id=='nthcont_Suzaku':

        '''
        subset of continuum with nthcomp for better broad band work + Suzaku crabcorr replacing the constant
        note that the Suzaku_crabcorr component is specifically set to let only the FS CCDs delta gamma free
        '''

        avail_comps = ['cont_diskbb','disk_nthcomp','glob_TBabs', 'Suzaku_crabcorr']

        interact_groups = None

    if model_id=='nthcont_NuSTAR':

        '''
        subset of continuum with nthcomp for better broad band work + NuSTAR calibration components
        '''

        avail_comps = ['cont_diskbb','disk_nthcomp','glob_TBabs', 'glob_constant',
                       'calNuSTAR_edge']

        interact_groups = None

    if model_id=='nthcont_NICER':

        '''
        subset of continuum with nthcomp for better broad band work + NICER calibration components
        '''

        avail_comps = ['cont_diskbb','disk_nthcomp','glob_TBabs', 'glob_constant',
                       'calNICERSiem_gaussian','calNICER_edge']

        interact_groups = None

    if model_id=='nthcont_detailed':

        '''
        subset of continuum with nthcomp for better broad band work + NuSTAR/NICER calibration components
        '''

        avail_comps = ['cont_diskbb','disk_nthcomp','glob_TBabs', 'glob_constant',
                       'calNICERSiem_gaussian','calNuSTAR_edge','calNICER_edge']

        interact_groups = None

    if model_id=='cont':
        
        avail_comps=['cont_diskbb','cont_powerlaw','glob_constant','glob_phabs',]
        
        interact_groups=avail_comps      

    if model_id=='cont_noabs':
        
        avail_comps=['glob_constant','cont_diskbb','cont_powerlaw']
        
        interact_groups=avail_comps   
    
    if model_id=='lines':
        
        '''
        Notes : no need to include continuum components here anymore since we now use the continuum fitmod as argument in the lines fitmod
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
        
        
    if model_id=='lines_ns':
        # avail_comps=['FeKaem_gaussian','FeKa26abs_nagaussian']
        avail_comps=['cont_diskbb','cont_powerlaw','cont_bb','FeKaem_gaussian','FeKbem_gaussian',
                      'FeKa25abs_nagaussian','FeKa26abs_nagaussian','NiKa27abs_nagaussian',
                      'FeKb25abs_nagaussian','FeKb26abs_nagaussian','FeKg26abs_nagaussian']
        
    if model_id=='lines_ns_noem':
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


def line_e_ranges_fullarg(sat,sat_glob,diff_bands_NuSTAR_NICER,low_E_NICER,line_cont_ig_arg,
                  suzaku_xis_ignore,suzaku_pin_range,suzaku_xis_range,e_min_NuSTAR,e_max_XRT,
                  det=None):
    '''
    Determines the energy range allowed, as well as the ignore energies for a given satellite

    DO NOT USE INTS else it will be taken as channels instead of energies
    ignore_bands are bands that will be ignored on top of the rest, in ALL bands


    '''
    ignore_bands = None

    if sat == 'NuSTAR':
        e_sat_low = e_min_NuSTAR if (sat_glob == 'multi' and diff_bands_NuSTAR_NICER) else 4.
        e_sat_high = 79.

    if sat.upper()=='SWIFT':
        if det is not None and det.upper()=='BAT':
            e_sat_low=14.
            e_sat_high=195.
        else:
            e_sat_low=0.3
            if sat_glob=='multi':
                e_sat_high=e_max_XRT
            else:
                e_sat_high=10.
    if sat.upper()=='INTEGRAL':
        if det=='IBIS':
            e_sat_low=30
            #no high ignore
            e_sat_high=1000
        if det=='JMX1':
            #no low ignore
            e_sat_low=0.1
            e_sat_high=20

    if sat.upper() in ['XMM', 'NICER']:

        if sat == 'NICER':
            e_sat_low = 0.3 if (sat_glob == 'multi' and diff_bands_NuSTAR_NICER) else low_E_NICER
        else:
            e_sat_low = 0.3

        if sat.upper() in ['XMM']:
            if sat == 'XMM':
                e_sat_low = 2.

            e_sat_high = 10.
        else:
            if sat == 'NICER':
                e_sat_high = 10.
            else:
                e_sat_high = 10.

    elif sat == 'Suzaku':

        if det == None:
            e_sat_low = 1.9
            e_sat_high = 40.

            ignore_bands = suzaku_xis_ignore
        else:

            assert det in ['PIN', 'XIS'], 'Detector argument necessary to choose energy ranges for Suzaku'

            e_sat_low = suzaku_xis_range[0] if det == 'XIS' else suzaku_pin_range[0]
            e_sat_high = suzaku_xis_range[1] if det == 'XIS' else suzaku_pin_range[1]

            # note: we don't care about ignoring these with pin since pin doesn't go that low
            ignore_bands = suzaku_xis_ignore

    elif sat.upper() == 'CHANDRA':
        e_sat_low = 1.5
        e_sat_high = 10.

    '''
    computing the line ignore values, which we cap from the lower and upper bound of the global energy ranges to avoid issues 
    we also avoid getting upper bounds lower than the lower bounds because xspec reads it in reverse and still ignores the band you want to keep
    ####should eventually be expanded to include the energies of each band as for the lower bound they are higher and we could have the opposite issue with re-noticing low energies
    '''

    line_cont_ig = ''

    if line_cont_ig_arg == 'iron':

        if sat.upper() in ['XMM', 'CHANDRA', 'NICER', 'SWIFT', 'SUZAKU', 'NUSTAR','INTEGRAL']:

            if e_sat_low > 6:
                # not ignoring this band for high-E only e.g. high-E only NuSTAR, BAT, INTEGRAL spectra
                pass
            else:
                if e_sat_high > 6.5:

                    line_cont_ig += '6.5-' + str(min(7.1, e_sat_high))

                    if e_sat_high > 7.7:
                        line_cont_ig += ',7.7-' + str(min(8.3, e_sat_high))
                else:
                    #an empty string will work with xspec but cause issues so better put a None (which will crash)
                    line_cont_ig = None

        else:
            line_cont_ig = '6.-8.'

    return e_sat_low, e_sat_high, ignore_bands, line_cont_ig

def file_to_obs(file, sat,megumi_files):
    if sat == 'Suzaku':
        if megumi_files:
            return file.split('_src')[0].split('_gti')[0]
    elif sat in ['XMM', 'NuSTAR']:
        return file.split('_sp')[0]
    elif sat.upper() in ['CHANDRA', 'SWIFT']:
        return file.split('_grp_opt')[0]
    elif sat in ['NICER']:
        return file.split('_sp_grp_opt')[0]
    elif sat.upper()=='INTEGRAL':
        return file.split('_sum')[0]
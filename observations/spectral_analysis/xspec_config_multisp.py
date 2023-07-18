#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:01:03 2021

@author: parrama
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.io import fits

from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,AllChains,Chain

from fitting_tools import sign_delchis_table,lines_std,lines_e_dict,lines_w_dict,lines_broad_w_dict,\
        link_groups,lines_std_names,def_ftest_threshold,def_ftest_leeway,ang2kev

from general_tools import ravel_ragged


from contextlib import redirect_stdout
import subprocess
import os
import numpy as np
import re
from tqdm import tqdm
import time
import multiprocessing
import pandas as pd
from copy import copy
import dill

from matplotlib.ticker import Locator


from matplotlib.gridspec import GridSpec

def getoverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) \
                                   or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) \
                                    or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))
    
model_dir='/home/parrama/Soft/Xspec/Models'

#custom model loads
AllModels.lmod('relxill',dirPath=model_dir+'/relxill')

#example of model loading
# AllModels.initpackage('tbnew_mod',"lmodel_tbnew.dat",dirPath=model_dir+'/tbnew')
#### should be updated for the new custom model addition in xspec 12.13 and pyxspec 2.1.1
# #this line still has to be used every time. The model name in xspec is NOT tbnew_mod though
# AllModels.lmod('tbnew_mod',dirPath=model_dir+'/tbnew')

#list of multiplicative/convolution models (as of version 12.12 + with tbnew added)
xspec_multmods=\
'''
     SSS_ice    constant     ismdust      smedge      vphabs     zpcfabs
       TBabs     cyclabs    log10con    spexpcut        wabs      zphabs
       TBfeo        dust    logconst      spline      wndabs     zredden
       TBgas        edge       lyman      swind1        xion     zsmdust
     TBgrain      expabs       notch      tbnew       xscat     zvarabs
       TBpcf      expfac  olivineabs   tbnew_feo      zTBabs     zvfeabs
       TBrel        gabs      pcfabs   tbnew_gas      zbabs     zvphabs
    TBvarabs      heilin       phabs   tbnew_pcf       zdust       zwabs
      absori    highecut       plabs   tbnew_rel       zedge     zwndabs
     acisabs       hrefl        pwab       uvred    zhighect      zxipab
        cabs      ismabs      redden      varabs        zigm      zxipcf
       cflux    ireflect      kyconv     reflect      thcomp     xilconv
      clumin      kdblur     lsmooth     rfxconv     vashift     zashift
      cpflux     kdblur2     partcov     rgsxsrc     vmshift     zmshift
     gsmooth    kerrconv      rdblur       simpl     mtable{4u1630_best.fits}
'''.split()
    
xcolors_grp=['black','red','lightgreen','blue','cyan','purple','yellow']

#not used anymore now that we take the info directly from the log file

def screen_term(screenfile,wind_type='Spyder',kill=False):
    
    '''
    The spyder terminal should be the the window opened just before the Spyder window since they are both launched together, so
    this is the method we use to recognize it (assuming there is only 1 Spyder window opened)
    
    Sometimes it's not true, careful
    '''
    
    windows_current=subprocess.run(['wmctrl','-l'],stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    
    for i in range(1,len(windows_current)):
        elem=windows_current[i]
        elempre=windows_current[i-1]
        if wind_type in elem.split('    ')[-1]:
            if wind_type=='Spyder':
                terminal_wid=elempre.split(' ')[0]
            elif wind_type=='Grace':
                terminal_wid=elem.split(' ')[0]
            os.system('import -window '+terminal_wid+' '+screenfile)
            
            if kill==True:
                os.system('xkill -id '+terminal_wid)

def catch_model_str(logfile,savepath=None):
    
    '''
    catches the current model's paremeters and fit (AllModels(1).show + Fit.show) from the current logfile
    
    If savepath is not None, saves the str to the given path
    '''
    
    #saving the previous chatter state and ensuring the log chatter is correctly set
    prev_logchatter=Xset.logChatter
    
    Xset.logChatter=10

    #flushing the readline to get to the current point
    logfile.readlines()
    
    #Displaying the elements we're interested in once again
    AllModels.show()
    Fit.show()
    
    #catching them
    mod_str=logfile.readlines()
    
    #and writing them into a txt
    if savepath is not None:
        with open(savepath,'w') as mod_str_file:
            mod_str_file.writelines(mod_str)
        
    #switching back to the initial logchatter value
    Xset.logChatter=prev_logchatter
    
    return mod_str


'''
Model modification and utility commands.
Most of these are created because once a model is unloaded in PyXspec, everything but the component/parameter names is lost
'''


class allmodel_data:
    
    '''
    class for storing the information of all the current models (customs, and each group)
    
    stores scorpeon nicer bkg model differently
    '''

    def __init__(self,modclass=AllModels):
        
        mod_list=['default' if elem=='' else elem for elem in modclass.sources.values() if elem not in ['nxb','sky']]
        
        mod_keys=[elem for elem in modclass.sources.keys() if elem not in [98,99]]
        
        self.mod_list=mod_list
        self.mod_keys=mod_keys
        #storing standard models
        for elem_mod in self.mod_list:

            mod_data_arr=np.array([None]*max(1,AllData.nGroups))
            
            for i_grp in range(max(1,AllData.nGroups)):
                
                #testing if the model exists for this group
                try:
                    AllModels(i_grp+1,modName='' if elem_mod=='default' else elem_mod)
                except:
                    continue
                
                #saving the model if it exsits
                mod_data_arr[i_grp]=model_data(AllModels(i_grp+1,modName='' if elem_mod=='default' else elem_mod))

            setattr(self,'default' if elem_mod=='' else elem_mod,mod_data_arr)
            
        #testing if there is a nicer bkg model loaded
        if 'nxb' in modclass.sources.values() and 'sky' in modclass.sources.values():
            
            self.scorpeon=scorpeon_data()
        else:
            self.scorpeon=None
            
    def load(self):
                                
        for elem_mod,elem_key in zip(self.mod_list,self.mod_keys):
            model_load(getattr(self,elem_mod),mod_name='' if elem_mod=='default' else elem_mod,mod_number=elem_key)

        xchatter=Xset.chatter
        xlogchatter=Xset.logChatter
        
        #doing this silently to avoid surcharching the screen and log files        
        Xset.chatter=0
        Xset.logChatter=0
        
        if 'scorpeon' in dir(self):
            xscorpeon.load(scorpeon_save=self.scorpeon)

        Xset.chatter=xchatter
        Xset.logChatter=xlogchatter
        
mod_sky=None
mod_nxb=None


class model_data:
    
    '''
    Class for xspec model data to allow adding/deleting components more easily
    Also contains a copy of the component/parameter arborescence to allow for an easier access
    '''
    
    def __init__(self,xspec_model):
        
        self.expression=xspec_model.expression
        self.npars=xspec_model.nParameters
        self.comps=xspec_model.componentNames
        
        values=np.zeros((self.npars,6))
        links=np.array([None]*self.npars)
        frozen=np.zeros(self.npars).astype(bool)
        for  i in range(1,self.npars+1):
            values[i-1]=xspec_model(i).values
            links[i-1]=xspec_model(i).link.replace('= p','')
            frozen[i-1]=xspec_model(i).frozen
        self.values=values
        self.links=links
        self.frozen=frozen
    
        for elem_comp in self.comps:
            setattr(self,elem_comp,component_data(getattr(xspec_model, elem_comp)))
    
    def update(self):
        
        '''
        update order of the parameters according to a new expression with component positions swapped
        
        needs to be updated for multiple components with same name (for which the name changes with different expressions)
        '''
        
        new_comp_order=re.split(r'\W+',self.expression.replace(' ','').replace(')',''))
        
        id_par=0
        for elem_comp in new_comp_order:
            for elem_par in getattr(self,elem_comp).pars:
                self.values[id_par]=getattr(getattr(self,elem_comp),elem_par).values
                self.links[id_par]=getattr(getattr(self,elem_comp),elem_par).link.replace('= p','')
                self.frozen[id_par]=getattr(getattr(self,elem_comp),elem_par).frozen
                id_par+=1

class scorpeon_manager:
    
    '''
    Global class to manage scorpeon model backgrounds in xspec. A single instance of this is created, called xscorpeon
    '''
    
    def __init__(self):
        self.bgload_paths=None
        
    def load(self,bgload_paths=None,scorpeon_save=None,frozen=False):
        
        '''
        reloads the nicer bg model(s) from the stored path(s), and scorpeon save(s) if any
        
        can be called without bgload_paths after the first load to reset/reload bg models with or without a bg
        datagroups with no models should be left as empty in the bgload_paths array
        
        as of now, loads up to a single model per datagroup
        '''
        
        #updating the bgload paths if an argument if prodivded
        if bgload_paths is not None:
            #converting the input into an array like for easier manipulation
            if type(bgload_paths) not in [list,np.ndarray,tuple]:
                self.bgload_paths=[bgload_paths]
            else:
                self.bgload_paths=bgload_paths
        
        if self.bgload_paths is not None:
            #making sure the file actually exists
            assert np.array([elem is None or os.path.isfile(str(elem)) for elem in self.bgload_paths]).all(), 'One or more scorpeon load file path does not exist'
            
            #loading all of the models
            for i_bg,bg_path in enumerate(self.bgload_paths):
    
                if bg_path is not None:
                    nicer_bkgspect=i_bg+1
                    exec(open(bg_path).read())
        
        #freezing parameters if the current data doesn't cover large enough ranges 
        #(see https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/scorpeon-xspec/)
        
        for i_grp in range(AllData.nGroups):
            
            #assuming 1 spectrum per group here
            curr_grp_sp=AllData(i_grp+1)
            
            if curr_grp_sp.energies[0][0]>0.5:
                
                try:
                    mod_nxb=AllModels(i_grp+1,modName='nxb')
                    
                    #freezing noise_norm
                    mod_nxb(11).frozen=True
                except:
                    pass
                    
                try:
                    mod_sky=AllModels(i_grp+1,modName='sky')
                    #freezing gal_nh
                    mod_sky(1).frozen=True
                    
                    #freezing lhb_em
                    mod_sky(8).frozen=True
        
                    if curr_grp_sp.energies[0][0]>1.:
                        #freezing halo_em
                        mod_sky(6).frozen=True
                except:
                    pass
            
        #loading all of the saves
        if scorpeon_save is not None:
            scorpeon_save.load()
            
        #freezing the model if asked to
        if frozen:
            
            for i_grp in range(AllData.nGroups):
                
                try:
                    mod_nxb=AllModels(i_grp+1,modName='nxb')
                    
                    for i_par in range(mod_nxb.nParameters):
        
                        mod_nxb(i_par+1).frozen=True
                except:
                    pass
                
                try:
                    mod_sky=AllModels(i_grp+1,modName='sky')
                    
                    for i_par in range(mod_sky.nParameters):
        
                        mod_sky(i_par+1).frozen=True
                except:
                    pass
                    

class scorpeon_data:
            
    def __init__(self,modclass=AllModels):
        
        self.nxb_save_list=[]
        self.sky_save_list=[]

        self.all_frozen=True

        for i_grp in range(AllData.nGroups):
            
            try:
                self.nxb_save_list+=[scorpeon_group_save(modclass(i_grp+1,modName='nxb'))]

                #adding a check for every loaded datagroup model being completely frozen
                self.all_frozen=self.all_frozen and np.array(self.nxb_save_list[-1].frozen).all()

            except:
                self.nxb_save_list+=[None]
            
            try:
                self.sky_save_list+=[scorpeon_group_save(modclass(i_grp+1,modName='sky'))]

                #adding a check for every loaded datagroup model being completely frozen
                self.all_frozen=self.all_frozen and np.array(self.sky_save_list[-1].frozen).all()

            except:
                self.sky_save_list+=[None]


    def load(self):
        
        for i_grp in range(len(self.nxb_save_list)):
            
            nxb_save=self.nxb_save_list[i_grp]
            sky_save=self.sky_save_list[i_grp]
            
            if nxb_save is not None:
                mod_nxb=AllModels(i_grp+1,modName='nxb')
                
                for i_par in range(mod_nxb.nParameters):

                    mod_nxb(i_par+1).values=nxb_save.values[i_par]                        

                    mod_nxb(i_par+1).link=nxb_save.link[i_par]

                    mod_nxb(i_par+1).frozen=nxb_save.frozen[i_par]

            if sky_save is not None:
                mod_sky=AllModels(i_grp+1,modName='sky')
                
                for i_par in range(mod_sky.nParameters):
                    
                    mod_sky(i_par+1).values=sky_save.values[i_par]                       
                    mod_sky(i_par+1).link=sky_save.link[i_par]
                    mod_sky(i_par+1).frozen=sky_save.frozen[i_par]
                    
class scorpeon_group_save:
    
    '''
    simple scorpeon save for a single datagroup
    '''
    
    def __init__(self,model):
        
        self.values=[model(i_par+1).values for i_par in range(model.nParameters)]
    
        self.link=[model(i_par+1).link.replace('= p','') for i_par in range(model.nParameters)]
            
        self.frozen=[model(i_par+1).frozen for i_par in range(model.nParameters)]
            
xscorpeon=scorpeon_manager()

class component_data:
    
    '''
    copies the most often used information of the xspec component class
    '''
    
    def __init__(self,xspec_component):
        self.pars=xspec_component.parameterNames
        for elem_par in self.pars:
            setattr(self,elem_par,parameter_data(getattr(xspec_component,elem_par)))

class parameter_data:
    
    '''
    copies the most often used information of the xspec parameter class
    '''
    
    def __init__(self,xspec_parameter):
        self.values=xspec_parameter.values
        self.link=xspec_parameter.link.replace('= p','')
        self.frozen=xspec_parameter.frozen     
    
def par_degroup(parnumber,mod_name=''):
    
    '''
    computes the group and parameter index in that group of a given parameter given in all groups
    '''
    
    i_grp=1+(max(0,parnumber-1)//AllModels(1,modName=mod_name).nParameters)
    
    id_par=1+(max(0,parnumber-1)%AllModels(1,modName=mod_name).nParameters)
    
    return i_grp,id_par
    
def reset():
    
    '''Clears everything and gets back to standard fitting and abundance parameters'''
    
    AllChains.clear()
    AllData.clear()
    AllModels.clear()
    
    #reseting the xscorpeon bg paths
    xscorpeon.bgload_paths=None
    
    Xset.abund='wilm'
    Fit.nIterations=1000
    Plot.xAxis='keV'
    Plot.add=True

def model_load(model_saves,mod_name='',mod_number=1,gap_par=None,in_add=False,modclass=AllModels):

    '''
    loads a mod_data class into the active xspec model class or all model_data into all the current data groups
    if model_save is a list, loads all model_saves into the model data groups of the AllModels() data class
    
    can be used to lad custom models through model_name and mod_number. The default values update the "standard" xspec model
    
    gap par introduces a gap in the parameter loading. Used for loading with new expressions including new components 
    in the middle of the model.
    gap_par must be an interval string, i.e. '1-4'

    '''
    #lowering the chatter to avoid thousands of lines of load messages
    prev_chatter=Xset.chatter
    prev_logchatter=Xset.logChatter
    
    Xset.chatter=0
    Xset.logChatter=0
    
    if gap_par is not None:
        gap_start=int(gap_par.split('-')[0])
        gap_end=int(gap_par.split('-')[1])

    if type(model_saves)==allmodel_data and len(model_saves.mod_list)==1:
        model_saves_eff=model_saves.default
    else:
        model_saves_eff=model_saves

    if not (type(model_saves_eff)==list or type(model_saves_eff)==np.ndarray):
        model_saves_arr=np.array([model_saves_eff])
    else:
        model_saves_arr=model_saves_eff
                
    multi_groups=False
    #using the first save to create most of the elements that are common to all data groups
    if type(model_saves_arr)==list or type(model_saves_arr)==np.ndarray:
        if len(model_saves_arr)>1:
            multi_groups=True
        first_save=model_saves_eff[0]
    else:
        first_save=model_saves_eff
        
    #creating a model with the new model expression
    modclass+=(first_save.expression,mod_name,mod_number)

        
    # 
    # for i_grp in range(len(model_saves_arr)):
    #     AllModels(i_grp+1,mod_name,mod_number).untie()
        
    for i_grp,save_grp in enumerate(model_saves_arr):
        
        xspec_mod_grp=AllModels(i_grp+1,mod_name)
        
        #untying all the models at first to avoid interval problems with links to data groups that are not loaded yet
        xspec_mod_grp.untie()
        
        #creating a dictionnary of all the parameters values to load them in a single command
        parload_grp={}
        for i_par in range(1,xspec_mod_grp.nParameters+1):
            
            if gap_par is None:
                #the dictionnary argument must be a string with spaces, so we directly convert the array in string
                #and take off the brackets
                parload_grp[i_par]=str(save_grp.values[i_par-1])[1:-1]
            else:

                #everything is normal before the gap start
                if i_par<gap_start:
                    parload_grp[i_par]=str(save_grp.values[i_par-1])[1:-1]
                #after the gap the values are simply shifted
                elif i_par>gap_end:
                    
                    #we add 1 to the gap length since the lower bound is included in the gap                    
                    parload_grp[i_par]=str(save_grp.values[i_par-1-(gap_end-gap_start+1)])[1:-1]
                    
        xspec_mod_grp.setPars(parload_grp)

    #we load directly all the values dictionnaries before the rest to avoid problems with links if the intervals don't follow
    for i_grp,save_grp in enumerate(model_saves_arr):
        
        xspec_mod_grp=AllModels(i_grp+1,mod_name)
        
        #loading links and freeze states
        for i_par in range(1,xspec_mod_grp.nParameters+1):

            if gap_par is None:
                #we parse the link string to get the parameter number back
                xspec_mod_grp(i_par).link=save_grp.links[i_par-1].replace('= p','')
                
                #the if is just there to avoid useless prints        
                if xspec_mod_grp(i_par).frozen!=save_grp.frozen[i_par-1]:
                    xspec_mod_grp(i_par).frozen=save_grp.frozen[i_par-1]
            else:

                if i_par<gap_start:
                    i_shifted=i_par
                elif i_par>gap_end:
                    i_shifted=i_par-(gap_end-gap_start+1)
                    
                if i_par<gap_start or i_par>gap_end:
                    #the if loop is just here to avoid displaying more messages than necessary
                    if xspec_mod_grp(i_par).frozen!=save_grp.frozen[i_shifted-1]:
                        xspec_mod_grp(i_par).frozen=save_grp.frozen[i_shifted-1]
                    
                    '''
                    all links to elements after the first data groups need to be shifted by at least the size of the gap
                    times the data group number of the link pointer
                    
                    in each group we also need to shift once more links pointing to after the gap start in that group
                    '''
                    
                    if save_grp.links[i_shifted-1]!='':
                        #the shift by one and 0 minimum is here to allow all the last parameters of each groups to stay in the group below
                        link_pointer=str(int(save_grp.links[i_shifted-1].replace('= p',''))+(gap_end-gap_start+1)*\
                                         (max(int(save_grp.links[i_shifted-1].replace('= p',''))-1,0)//first_save.npars))
                        
                        #here we test if there were links to after the gap in a single group
                        #The second test is here for the last parameter of the group
                        if int(save_grp.links[i_shifted-1].replace('= p',''))%first_save.npars>=gap_start or \
                            (int(save_grp.links[i_shifted-1].replace('= p',''))%first_save.npars==0 and first_save.npars>=gap_start):                        
                            link_pointer=str(int(link_pointer)+(gap_end-gap_start+1))
                    else:
                        link_pointer=save_grp.links[i_shifted-1]
                            
                    #for links xspec doesn't display messages when so we don't care
                    xspec_mod_grp(i_par).link=link_pointer
                    
    #returning to the previous chatter value
    Xset.chatter=prev_chatter
    Xset.logChatter=prev_logchatter
    
    #if the load happens during addcomp, we don't show the new models since it will be done at the end of the addcomp anyway
    if not in_add:
        #showing the current models
        AllModels.show()
        
        #and the fit
        Fit.show()
    
    if multi_groups:
        return np.array([AllModels(i+1,mod_name) for i in range(AllData.nGroups)])
    else:
        return AllModels(1,mod_name)
    
def editmod(new_expression,model=None,modclass=AllModels,pointers=None):
    
    '''
    DEPRECATED, should be updated (also with new allmodel_data() format) if used, to specify which model to change
    
    Changes the expression of a model. the pointers string gives the component equivalence in case of ambiguity
    
    if a specific model is provided, uspdates this one. Else, updates all models in the AllModels class 
    (assumed to be from different data groups)
    
    The changes are implemented in increasing order of difficulty
    1. Nothing has changed in the components/orders, just the expression itself needs to be updated
    2. Same number and components, just swaps
    3. components added/deleted
    '''
    
    if model is not None:
        xspec_mod=model
    else:
        xspec_mod=modclass(1)
    
    model_saves=allmodel_data().default
    model_saves[0].expression=new_expression
    
    
    #testing if the parameters need to be swapped by comparing the components order
    new_expression_split=re.split(r'\W+',new_expression.replace(' ','').replace(')',''))
    old_expression_split=re.split(r'\W+',xspec_mod.expression.replace(' ','').replace(')',''))
    if new_expression_split==old_expression_split:
        return model_load(model_saves)
    
    #this should be updated for models with multiple components of the same type
    if str(np.sort(new_expression_split))==str(np.sort(old_expression_split)):
        for save_grp in model_saves:
            save_grp.update()
            
        return model_load(model_saves)
    
def numbered_expression(expression=None):

    '''
    Return an edit model xspec expression with each component's naming in the current xspec model
    by default, uses the current model expression
    '''
    
    if expression is None:
        string=AllModels(1).expression
    else:
        string=expression
        
    xspec_str=[]
    
    #variable defining the index of the start of the current word if any
    comp_str=-1
    for i in range(len(string)):
        if string[i] in '()+-* ':
            #adding the current punctuation if the last element was also a ponctuation
            if comp_str==-1:
                xspec_str+=[string[i]]
            #in the other case, adding the previous word which started from the index
            else:
                xspec_str+=[string[comp_str:i],string[i]]
            
            #and resetting the word count
            comp_str=-1
        else:
            #starting the word count at the current index
            if comp_str==-1:
                comp_str=i
            #storing the word if we arrived at the last char of the string
            if  i==len(string)-1:
             xspec_str+=[string[comp_str:i+1]]
    
    i=0
    #adding numbers by replacing by the actual model component names
    for j,elem in enumerate(xspec_str):
        if elem in '()+-* ':
            continue
        
        #needs a proper call to AllModels(1) here to be able to use it whenever we want
        xspec_str[j]=AllModels(1).componentNames[i]
        i+=1
            
    return ''.join(xspec_str)
    
def addcomp(compname,position='last',endmult=None,return_pos=False,modclass=AllModels,included_list=None,values=None,links=None,frozen=None):
    
    '''
    changes the default model to add a new component, by saving the old model parameters and thn loading a new model with the new 
    component
    
    for multiple data groups, does not unlink the parameters of the added components
    
    For additive components, if values is set, it must be an array with one element for each parameter of the component.
    Each element of the array can be 'id' to keep the standard value, a single value to replace the parameter default value, 
    or an array of 6 values which themselves can be 'id' to keep the default or values
    Same for links and frozen
    
    the "position" keyword refers to the position of the added component:
        -last: last component in the model
        -lastin: last component, but inside all the last multiplication(s)
        -*xspec component name (with underscore numbering if its not the first)/associated number: BEFORE this component
        the numbering is considered actual component numbering (starting at 1) for positive numbers, 
        and array numberings (negatives up to 0) for negative numbers
        
    the "endmult" keyword refers to the position of the end parenthesis for multiplicative components:
        -None: only include up to the next additive component 
        -all/-1: ends at the end of the model
        -*xspec component name/associated number: AFTER this component
        
        
    custom components (work together):
        
    Continuum:
        -glob_+ multiplicative component: abs with position=first and endfactor=all
        
        -glob_constant: global constant factor, frozen at 1 for the first data group and free for all others
            
        -cont_+component: added inside a first/second component absorption if any
        
    Lines :
        use (Linetype)-(n/a/na/...)gaussian
        
        'n' -> narrow (frozen Sigma=0) gaussian
        'a'-> absorption (locked <0 norm) gaussian
        'g'-> broad (forced high width values) gaussian
        
        named lines (to be included as prefix):
        
        for 'named' lines with the ionisation number, adds a vashift component with [X,Yk] in blueshift and a gaussian with energy frozen at the line 
        baseline energy level (see constants at the start of the code for up to date parameter definitions)
        
        'FeKa0' ->  neutral Fe emission at 6.40
        'FeKb0' ->  neutral Fe emission at 7.06
        'FeKa25' -> FeXXV absorption at 6.70keV.                                                
        'FeKa26' -> FeXXVI absorption at 6.97keV
        
        'NiKa27' -> NiXXVII absorption at 7.80 keV -> currently locked at 3000km/s max (i.e. ~7.88keV max) to avoid overlap with FeKb2
        'FeKb25' -> FeXXV absorption at 7.89keV
        'FeKb26' -> FeXXVI absorption at 8.25keV
        'FeKg26' -> FeXXVI absorption at 8.70 keV
        
            (old models) 
        'FeKa'->neutral Fe emission at 6.40
        'FeKb'->neutral Fe emission at 7.06
        
        both of these have loose energy constraints instead of vashift ranges
        
        'FeDiaz' ->unphysical Fe emission line, replacing both FeKa and FeKb. Line energy only constrained to a range in [6.-8.] keV
        
        if an includedlist of fitcomps is given in argument, will attempt to link energies in forward order for the same complexes
        
    Relativistic line:
        -FeKa_laor -> empirical laor line to account for huge em lines in some spectra of hig inclined binaries
        energy between 6.4 and 7.06
        inclination between 50 and 90
        index free
        Rin and Rout frozen
        
    all compnames must stay 'attribute'-valid else we won't be able to call them explicitely in fitcomps
    '''
        
    '''Component initialisation'''
    
    #component identifier booleans
    multipl=False
    
    start_position=position
    end_multipl=endmult
    
    gaussian_type=None
    narrow_gaussian=False
    abs_gaussian=False
    added_link_group=None
    
    #splitting custom parts of the component
    if '_' in compname:
        comp_custom=compname.split('_')[0]
        comp_split=compname.split('_')[1]
    else:
        comp_custom=None
        comp_split=compname
        
    if comp_split in xspec_multmods:
        multipl=True
    #dichotomy between custom models
    
    #global keyword for multiplicative components
    if multipl and comp_custom=='glob':
        start_position=1
        end_multipl=-1
        #staying inside the constant factor if there is one
        if AllModels(1).componentNames[0]=='constant':
            start_position+=1
            
    #continuum keyword
    if comp_custom=='cont':
        start_position=1
        try:
            #staying inside the constant factor if there is one
            if AllModels(1).componentNames[0]=='constant':
                start_position+=1
            #maintaining inside the absorption component if there is one
            if 'abs' in AllModels(1).componentNames[0 if AllModels(1).componentNames[0]!='constant' else 1]:
                start_position+=1
        except:
            pass
                
    #testing for lines
    if 'gaussian' in comp_split:
        
        line=True
        
        gaussian_type=comp_custom
        
        #restricting to the letter prefix
        comp_split_prefix=comp_split.replace('gaussian','')
        
        #identifying the shape of the line
        narrow_gaussian='n' in comp_split_prefix
        broad_gaussian='b' in comp_split_prefix
        abs_gaussian='a' in comp_split_prefix

        #updating the comp split which will be implemented without the letter keywords
        comp_split='gaussian'

        #restricting vashift use to named lines
        if gaussian_type is not None and sum([char.isdigit() for char in gaussian_type])>0:
            
            comp_split='vashift*gaussian'
            named_line=True
            
            #### link groups off for now
            # #and link groups to absorption lines (stays None for emission lines)
            # if gaussian_type.endswith('abs'):
            #     #(there can only be one since they do not share any element)
            #     added_link_group=[elem for elem in link_groups if gaussian_type in elem][0]
        else:
            named_line=False
    else:
        line=False
        
    '''component creation'''
    
    #checking if the current model is empty
    try:
        AllModels(1)

        is_model=True
    except:
        is_model=False
        xspec_model=Model(comp_split)

    if is_model:
        #saving the current models
        model_saves=allmodel_data().default
        
        #getting the xspec expression of the current model as well as the list of components
        num_expr=numbered_expression()
        
        #replacing a * by parenthesis for single constant*additive models to have an easier time later
        
        xcomps=AllModels(1).componentNames
        
        #determining where to place the new component
        if type(start_position)==int:
            #converting to actual component positions instead of array positions
            if start_position>0:
                xcomp_start=xcomps[start_position-1]
            else:
                 xcomp_start=xcomps[start_position]
        else:
            if start_position=='first':
                xcomp_start=xcomps[0]
            elif start_position in ['last','lastin']:
                #can't use an actual component here since we place the new component before it
                xcomp_start=-1
            else:
                xcomp_start=start_position
            
        #determining where to place the parenthesis for multiplicative components
        if multipl:
            if end_multipl is None:
                xcomp_end=xcomp_start
            else:
                if type(end_multipl)==int:
                    if end_multipl>0:
                        xcomp_end=xcomps[end_multipl-1]
                    else:
                        xcomp_end=xcomps[end_multipl]
                else:
                    if end_multipl=='all':
                        xcomp_end=xcomps[-1]
                    else:
                        xcomp_end=end_multipl

        #inserting the component
        if xcomp_start==-1:

            #if we are inserting our component inside of a constant with a single additivecomponent, we must replace the * by parenthesis
            if len(AllModels(1).componentNames)>1:
                if num_expr[num_expr.find(AllModels(1).componentNames[1])-1]=='*':
                    num_expr=num_expr.replace('*'+AllModels(1).componentNames[1],'('+AllModels(1).componentNames[1]+')',1)

            #at the very end of the model but inside parenthesis
            if position=='lastin':
                
                #counting the number of parenthesis at the end
                count_par=0
                if is_model:
                    if AllModels(1).componentNames[0]=='constant':
                            count_par+=1
                #adding inside them if there were any
                if count_par!=0:
                    new_expr=num_expr[:-count_par]+'+'+comp_split+num_expr[-count_par:]
                else:
                    new_expr=num_expr+'+'+comp_split
            else:
                #at the very end of the model
                new_expr=num_expr+'+'+comp_split
        else:
            
            #if we are inserting our component inside of a single multiplicative component, we must replace the * by parenthesis
            if num_expr[num_expr.find(xcomp_start)-1]=='*':
                num_expr=num_expr.replace('*'+xcomp_start,'('+xcomp_start+')',1)
                
            #inserting at the desired position                
            new_expr=num_expr.replace(xcomp_start,comp_split+('+' if not multipl else '(')+xcomp_start,1)
            
        #introducing the end parenthesis
        if multipl:
            new_expr=new_expr.replace(xcomp_end,xcomp_end+')',1)
            
        #returning the expression to its xspec readable equivalent (without numbering)
        for elemcomp in AllModels(1).componentNames:
            if '_' in elemcomp:
                #the 1 is important here to avoid replacing twice incompletely some of the first named components
                new_expr=new_expr.replace(elemcomp,elemcomp[:elemcomp.rfind('_')],1)
        
        #updating the save
        model_saves[0].expression=new_expr 
        
        #computing the gap with the new component(s)
        old_ncomps=len(AllModels(1).componentNames)
        
        if xcomp_start!=-1:
                
            #We compute the start of the gap from the 'old' version of the model
            gap_start=getattr(getattr(AllModels(1),xcomp_start),getattr(AllModels(1),xcomp_start).parameterNames[0]).index
        
            xcomp_start_n=np.argwhere(np.array(AllModels(1).componentNames)==xcomp_start)[0][0]
            #we compute the end gap as the parameter before the first parameter of the starting comp in the newer version of the model
            #We use the component number instead of its name to avoid problems
            
            try:
                xspec_model=Model(new_expr)
            except:
                print(new_expr)
                breakpoint()
                print(new_expr)
            added_ncomps=len(xspec_model.componentNames)-old_ncomps
            shifted_xcomp_start=xspec_model.componentNames[xcomp_start_n+added_ncomps]
            
            gap_end=getattr(getattr(xspec_model,shifted_xcomp_start),getattr(xspec_model,shifted_xcomp_start).parameterNames[0]).index-1
            
            #storing the position of the last component added for return if asked
            #the actual component is the one after but we're in array indices here so it already corresponds to the xspec indice of the one before
            added_comps_numbers=np.arange(xcomp_start_n+1,xcomp_start_n+added_ncomps+1).astype(int)
        else:
            gap_start=AllModels(1).nParameters+1

            try:
                xspec_model=Model(new_expr)
            except:
                
                '''
                Here to help debugging because there'll be issues with weird namings at some point
                '''
                print(new_expr)
                breakpoint()
                print(new_expr)
                
            gap_end=xspec_model.nParameters
            added_comps_numbers=np.arange(old_ncomps+1,len(AllModels(1).componentNames)+1).astype(int)
            
        gap_str=str(gap_start)+'-'+str(gap_end)
        
        model_load(model_saves,gap_par=gap_str,in_add=True)
        #we need to recreate the variable name because the model load has overriden it
        xspec_model=AllModels(1)
        
    else:
        xspec_model=Model(comp_split)
        gap_start=1
        gap_end=AllModels(1).nParameters
        added_comps_numbers=np.arange(1,len(AllModels(1).componentNames)+1).astype(int)
        
    '''continuum specifics'''
    
    #restricting the continuum powerlaw's photon index to physical values
    if compname=='cont_powerlaw':
        xspec_model(gap_end-1).values=[1.0, 0.01, 1.0, 1.0, 3.0, 3.0]
        
    #restricting the curvature bb/diskbb's kt to physical values
    if compname=='cont_diskbb':
        xspec_model(gap_end-1).values=[1.0, 0.01, 0.5, 0.5, 3.0, 3.0]
    #restricting the curvature bb's kt to physical values
    if compname=='cont_bb':
        xspec_model(gap_end-1).values=[1.0, 0.01, 0.1, 0.1, 4.0, 4.0]
    
    '''gaussian specifics (additive only)'''
    
    #this only works for non continuum components but we can assume gaussian lines will never be continuum components
        
    #switching the norm values of the gaussian
    if line:
        if abs_gaussian:
            
            xspec_model(gap_end).values=[-1e-4,1e-7,-5e-2,-5e-2,0,0]
            
            # #### ON: disabling FeKa25 
            # if gaussian_type=='FeKa25abs':
            #     xspec_model(gap_end).values=[-1e-7,1e-7,-5e-2,-5e-2,0,0]
            #     xspec_model(gap_end).frozen=True
            
            #### ON: disabling NiKa27 to avoid degeneracies
            if gaussian_type=='NiKa27abs':
                xspec_model(gap_end).values=[-1e-7,1e-7,-5e-2,-5e-2,0,0]
                xspec_model(gap_end).frozen=True
                
            # #### ON: disabling absorption lines
            # xspec_model(gap_end).values=[-1e-7,1e-7,-5e-2,-5e-2,0,0]
            # xspec_model(gap_end).frozen=True
                
        else:
            #stronger normalisations allowed for the emission lines
            xspec_model(gap_end).values=[1e-3,1e-6,0,0,1,1]
            
            # #### ON: disabling emission lines
            # #blocking emission lines if needed
            # xspec_model(gap_end).values=[1e-7,1e-7,5e-8,-5e-8,1e-6,1e-6]
            # xspec_model(gap_end).frozen=True
            
    #switching the width values of the lines
    if narrow_gaussian:
        xspec_model(gap_end-1).values=[0]+xspec_model(gap_end-1).values[1:]
        xspec_model(gap_end-1).frozen=True
        
    #changing more infos for specific lines
    if gaussian_type is not None:
        
        #selecting the corresponding energy
        ener_line=lines_e_dict[gaussian_type][0]

        #selecting the energy for non broad (named) lines
        if named_line:
            xspec_model(gap_end-2).values=[ener_line]+xspec_model(gap_end-2).values[1:]
        
        else:
            #restricting energies for emission lines
            if gaussian_type in ['FeDiaz']:
                xspec_model(gap_end-2).values=[ener_line,ener_line/100,6.0,6.0,8.0,8.0]
            else:                
                #outputing the line values (delta of 1/100 of the line ener, no redshift, +0.4keV blueshift max)
                xspec_model(gap_end-2).values=[ener_line,ener_line/100,ener_line-0.2,ener_line-0.2,ener_line+0.2,ener_line+0.2]


        #resticting the with of broad lines
        if broad_gaussian:
            
            #widths changes        
            width_line=lines_broad_w_dict[gaussian_type]
        
            #restricting widths of absorption lines and narrow emission lines
            xspec_model(gap_end-1).values=[width_line[0],
                                           (1e-3),
                                           width_line[1],width_line[1],
                                           width_line[2],width_line[2]]
        #and non-0 width lines
        elif not narrow_gaussian:
            
            #widths changes        
            width_line=lines_w_dict[gaussian_type]
        
            #restricting widths of absorption lines and narrow emission lines
            xspec_model(gap_end-1).values=[width_line[0],
                                           (1e-3),
                                           width_line[1],width_line[1],
                                           width_line[2],width_line[2]]
            
        #for named physical lines we freeze the gaussian energy and use the vashift instead
        if named_line:
            
            #freezing the energy
            xspec_model(gap_end-2).frozen=1
            
            #unfreezing the vashift
            xspec_model(gap_end-3).frozen=0
            
            #and forcing a specific range of blueshift/redshift depending on the line
            
            #note : we differenciate absorption an narrow emission through the 'em' in the lines energy dictionnary
            xspec_model(gap_end-3).values=[0,lines_e_dict[gaussian_type][2]/1e3,
                                           -lines_e_dict[gaussian_type][2],
                                           -lines_e_dict[gaussian_type][2],
                                           -lines_e_dict[gaussian_type][1],
                                           -lines_e_dict[gaussian_type][1]]

    '''laor specifics'''
    
    if comp_split=='laor':
        #selecting a broad energy range
        if 'FeKa' in comp_custom:
            xspec_model(gap_end-5).values=[6.6,0.01,6.4,6.4,7.06,7.06]
        
        #unfreezing some parameters
        #the index
        xspec_model(gap_end-4).frozen=False
        #the inclination
        xspec_model(gap_end-1).frozen=False
        
        #high-inclination constraints
        xspec_model(gap_end-1).values=[70,1,50,50,90,90]
        
    '''
    linking the vashifts from the same group IN FORWARD ORDER ONLY
    In order to do that, we parse the existing components to see if there are already existing components from the same link group. If so, 
    we link the new component's vashift to the vashift of the first found
    '''
    if included_list is not None and added_link_group is not None:
        for comp in [elemcomp for elemcomp in [elem for elem in included_list if elem is not None] if '_' in elemcomp.compname]:

            #we only link the component in a forward order 

            if comp.compname.split('_')[0] in added_link_group:
                if np.argwhere(np.array(added_link_group)\
                ==gaussian_type)[0][0]>np.argwhere(np.array(added_link_group)==comp.compname.split('_')[0])[0][0]:
                    #if we detect a component from the same group, its first parameter should be its vashift
                    xspec_model(gap_end-3).link=str(comp.parlist[0])
                    break

    '''
    Updating the values, links and frozen states if they were provided
    '''
    if values is not None:
        if type(values)!=np.ndarray:
            values_in=np.array(values)
        for j in range(1,len(values_in)+1):
            #not doing anything if the value is set to 'id'
            if values_in[j-1]!='id':
                def_values=xspec_model(gap_start+j).values
                
                #replacing everything or part of the values if the full values array is provided
                if len(values_in[j-1])==6:
                    new_values=np.where(values_in=='id',def_values,values_in)
                    xspec_model(gap_start+j).values=new_values
                
                #replacing only the current value if a single value is assigned
                elif len(values_in[j-1]==1):
                    def_values[0]=values_in[0]
                    xspec_model(gap_start+j).values=def_values
    
    #replacing default links if needed
    if links is not None:
        for j in range(1,len(links)+1):
            #not doing anything if the value is set to 'id'
            if links[j-1]!='id':
                xspec_model(gap_start+j).link=links[j-1]

    #replacing default freeze states if needed
    if frozen is not None:
        for j in range(1,len(frozen)+1):
            #not doing anything if the value is set to 'id'
            if frozen[j-1]!='id':
                xspec_model(gap_start+j).frozen=frozen[j-1]
               
    '''
    updating the value ranges of the new parameters for the other data groups to match the first one's
    This is necessary since model_load unlinks all components at the start and loading the save relinks them
    this means that the 'new' parameters end up unlinked, but this is on purpose to avoid linking before having updated the 
    value ranges (which we do here)
    
    Note that this is still not perfect as no matter what, values are reset to their initial ineterval when being unlinked
    '''
    
    for i_grp in range(2,AllData.nGroups+1):
        xspec_model_grp=AllModels(i_grp)
        for i_par in range(gap_start,gap_end+1):
            
            xspec_model_grp(i_par).values=AllModels(1)(i_par).values
            
            #updating the values break the link so we need to relink them afterwards
            xspec_model_grp(i_par).link=str(i_par)
            
    
    #creating the variable corresponding to the list of parameters
    return_pars=np.arange(gap_start,gap_end+1).astype(int).tolist()
    
    '''
    global constant factor specifics
    '''
    if compname=='glob_constant':
        for i_grp in range(1,AllData.nGroups+1):
            #setting the first data group to a fixed 1 value
            if i_grp==1:
                AllModels(i_grp)(gap_start).values=[1]+ AllModels(i_grp)(1).values[1:]
                AllModels(i_grp)(gap_start).frozen=True
            #unlinking the rest
            else:
                AllModels(i_grp)(gap_start).link=''
                AllModels(i_grp)(gap_start).frozen=False
                
        return_pars+=[1+AllModels(1).nParameters*i_grp for i_grp in range(1,AllData.nGroups)]
        
    AllModels.show()
        
    if return_pos:
        return return_pars,added_comps_numbers
    else:
        return AllModels(1)

def delcomp(compname,modclass=AllModels,give_ndel=False):
    
    '''
    changes the model to delete a component by saving the old model parameters and replacing it with the new one
    If values is set, it must be an array with one element for each parameter of the component.
    
    if multiple components have the same name, use the xspec name of the component you want to delete
    
    give_ndel returns the number of components that had to be deleted. Important for automatic fitting processes
    '''
    
    first_mod=modclass(1)
    
    model_saves=allmodel_data().default
    
    #deleting the space to avoid problems
    old_exp=model_saves[0].expression.replace(' ','')
    
    #separating the model expression when removing the component Name
    #this works even with multiple component with the same names because if it is the first one, find will get it first
    #(wouldn't work with split)
    
    
    #the easiest way to fetch the position of the component to delete is to transform the model expression according to xspec namings
    xspec_expr=numbered_expression(old_exp)          
    
    new_exp_bef=xspec_expr[:xspec_expr.find(compname)].replace(' ','')
    new_exp_aft=xspec_expr[xspec_expr.find(compname)+len(compname):].replace(' ','')
    
    #adding spaces to avoid indexation errors in case of empty expressions
    if new_exp_bef=='':
        new_exp_bef+=' '
    if new_exp_aft=='':
        new_exp_aft+=' '
        
    #getting the component name number
    #thankfully xspec edits component names to never have several components with the same name so we can always take the first index

    id_delcomp=np.where(np.array(first_mod.componentNames)==compname)[0][0]

        
    #replacing 
    #list of deleted component ids, will be updated accordingly if others need to be removed
    id_delcomp_list=[id_delcomp]
    
    #loop to edit the expression depending on possible problems until it becomes "mathematically" correct
    exp_ok=False
    
    delplus=False
    
    while not exp_ok:  
        
        #empty check
        if new_exp_bef==new_exp_aft==' ':
            print('Last component of the model has been deleted. Returning an empty string instead...')
            #return ''
            
        #testing if the component was the only one at the right of a convolutive/multiplicative component with parenthesis
        #or was a right product of a multiplicative component
        
        if (new_exp_bef[-1]=='(' and new_exp_aft[0]==')') or new_exp_bef[-1]=='*':
            print('Last component in a multiplicative/convolutive component group removed. Removing the associated component...')
            
            #identifying the name of the previous component
            #In case of multiple deleted components, we use the last element in the list of deleted components 
            del_asscomp=first_mod.componentNames[id_delcomp_list[-1]-1]
            
            id_delcomp_list+=[id_delcomp_list[-1]-1]
            
            #removing the right parenthesis if it exists
            new_exp_aft=new_exp_aft[1:] if new_exp_aft[0]==')' and new_exp_bef[new_exp_bef.find(del_asscomp)]=='(' else new_exp_aft
            
            #this removes both the component and the left parenthesis/multiplication
            new_exp_bef=new_exp_bef[:new_exp_bef.find(del_asscomp)]
            
            #re-adding space if needed
            if new_exp_bef=='':
                new_exp_bef+=' '
            
            if len(new_exp_aft)==0:
                new_exp_aft+=' '
        
        #removing an additive sign before if it existed
        if new_exp_bef[-1]=='+':
            new_exp_bef=new_exp_bef[:-1]
            delplus=True
            
        #removing an additive or multiplicative sign after if it existed
        if (new_exp_aft[0]=='+' and not delplus) or new_exp_aft[0]=='*':
            new_exp_aft=new_exp_aft[1:]
            if len(new_exp_aft)==0:
                new_exp_aft+=' '
            delplus=True
            
        #since each deletion can trigger other arithmetic issues, getting out of the loop is locked behind passing all three error 
        #tests at once in the same run (after the previous modifications)
        
        if not((new_exp_bef[-1]=='(' and new_exp_aft[0]==')') or new_exp_bef[-1]=='*' \
                or ((new_exp_bef[-1]=='+' or new_exp_bef[-1]=='-') and \
                (new_exp_aft[0]=='+' or new_exp_aft[0]=='-' or new_exp_aft[0]=='*'))):
            exp_ok=True
            
    #it is simpler to code the syntax change to delete everything but that leaves a case where the component was in the middle of a 
    #tripe addition. Thus, we add a '+' if the remaining exp_bef and exp_aft both end/begin with a letter/number 
    #(but models don't start with a number)
    if (new_exp_bef[-1].isalpha() or new_exp_bef[-1].isdigit()) and new_exp_aft[0].isalpha():
        add_str='+'
    else:
        add_str=''
        
    new_exp_temp=(new_exp_bef+add_str+new_exp_aft).replace(' ','')
    
    #deleting the remaining component numbers 
    new_exp_full=''
    i=0
    was_under=False
    
    #parsing the model expression
    for i,char in enumerate(new_exp_temp):
        #logging the presence of an underscore
        if char=='_':
            was_under=True
            #not storing the char if the next char is a digit
            if new_exp_temp[i+1].isdigit():
                continue
            else:
                new_exp_full+=char
                
        #if the current char is among a series of numbers after an underscore we keep skipping
        elif char.isdigit():
            if was_under:
                continue
        #finally, in any other case we reset the underscore count and add the char to the expression
        else:
            was_under=False
            new_exp_full+=char
            
    #Any remaining unnecessary parenthesis will be removed from the expression during the model creation
    
    #now we need to shift the attributions in the model_data class accordingly
    
    #we start by computing the number of parameters which aren't used anymore 
    skippar_n=np.sum([len(getattr(first_mod,first_mod.componentNames[elem]).parameterNames) for elem in id_delcomp_list])

    #then the starting number of these parameters (computed as the parameter following the sum of the number of parameters in the components
    #before the first deleted one
    #no +1 because these are list indexs at not actual parameter indexes (which start at 1)
    skippar_start=int(np.sum([len(getattr(first_mod,first_mod.componentNames[elem]).parameterNames)\
                              for elem in range(min(id_delcomp_list))]))
    
    #and we shift the values accordingly (needs to be done in two lines to avoid indexation problems)
    for grp_id,mod_data_grp in enumerate(model_saves):
        mod_data_grp.values[skippar_start:-skippar_n]=mod_data_grp.values[skippar_start+skippar_n:]
        mod_data_grp.values=mod_data_grp.values[:-skippar_n]
    
        #shifting/deleting link values accordingly
        for par_id,link in enumerate(mod_data_grp.links):
            
            #skipping empty links
            if link=='':
                continue
            
            '''
            deleting all links that points to deleted parameters
            the ecludian division allows the test to work for the deleted parameters of all data groups
            the goal of the concatenate here is to add the zero to consider when the link points to the last parameter of the group if this 
            parameter is among the deleted one
            '''
            
            if int(link)%AllModels(1).nParameters in np.concatenate((np.arange(skippar_start+1,skippar_start+skippar_n+1),
                                                    np.array([0]) if skippar_start+skippar_n==AllModels(1).nParameters else np.array([]))).astype(int):
                print('\nParameter '+str(grp_id*AllModels(1).nParameters+par_id+1)+
                      ' was linked to one of the deleted components parameters. Deleting link.')
                
                
                #Deleting links resets the values boundaries so we save and replace the values to make non-default bounds remain saved
                par_linked_vals=AllModels(grp_id+1)(par_id).values

                mod_data_grp.links[par_id]=''
                
                AllModels(grp_id+1)(par_id).values=par_linked_vals
                
                continue
                
            #shifting the link value if it points to a parameter originally after the deleted components
            #the 0 test accounts for the very last parameter, which will always need to be shifted if it wasn't in the deleted comps
            elif int(link)%AllModels(1).nParameters>=skippar_start+skippar_n or int(link)%AllModels(1).nParameters==0:
                mod_data_grp.links[par_id]=str(int(mod_data_grp.links[par_id])-skippar_n)
            
        mod_data_grp.links[skippar_start:-skippar_n]=mod_data_grp.links[skippar_start+skippar_n:]
        mod_data_grp.links=mod_data_grp.links[:-skippar_n]
            
        mod_data_grp.frozen[skippar_start:-skippar_n]=mod_data_grp.frozen[skippar_start+skippar_n:]
        mod_data_grp.frozen=mod_data_grp.frozen[:-skippar_n]
        
    #now we can finally recreate the model
    model_saves[0].expression=new_exp_full
    

    new_models=model_load(model_saves)

        
        
    if give_ndel:
        return len(id_delcomp_list)
    else:
        return new_models

def freeze(model=None,modclass=AllModels,unfreeze=False,parlist=None):
    
    '''
    freezes/unfreezes an entire model or a part of it 
    if no model is given in argument, freezes the first existing models
    (parlist must be the list of the parameter numbers)
    '''
    
    #lowering the chatter to avoid thousands of lines of load messages
    prev_chatter=Xset.chatter
    prev_logchatter=Xset.logChatter
    
    Xset.chatter=0
    Xset.logChatter=0
    
    if model is not None:
        xspec_mod=model
    else:
        xspec_mod=modclass(1)
    
    if parlist is None:
        for par_index in range(1,xspec_mod.nParameters+1):
            xspec_mod(int(par_index)).frozen=not(unfreeze)
            
    else:
        if type(parlist) in [np.int_,int]:
            parlist_use=[parlist]
        else:
            parlist_use=parlist
            
        for par_index in parlist_use:
            xspec_mod(int(par_index)).frozen=not(unfreeze)
    
    Xset.chatter=prev_chatter
    Xset.logChatter=prev_logchatter
    
def allfreeze():
    
    '''
    freeze all models of all datagroups
    '''
    for i_grp in range(AllData.nGroups):
        freeze(AllModels(i_grp+1))
            
def unfreeze(model=None,modclass=AllModels,parlist=None):
    
    '''
    just here to avoid calling arguments in freeze
    '''

    freeze(model=model,modclass=modclass,unfreeze=True,parlist=parlist)
    
def parse_xlog(log_lines,goal='lastmodel',no_display=False,replace_frozen=False,
               freeze_pegged=False):
    
    '''
    Parses the Xspec log file to search for a specific information. 
    Useful when xspec functions break before updating the model/Fit.
    
    To avoid losing precision since the parameters are only displayed up to a certain precision, 
    if replace_frozen is set to False, only non-frozen parameters are replaced
    
    if freeze_pegged is set to true, detects all parameters which were pegged to a value during:
        -error computations for explicit
        -the model display for implicit
        
    and freezes them
    (useful to avoid breaking the MC chain/EW when the fit is insensitive to some parameters)
    Also freezes parameter frozen implicitely (delta at negative values) during the fit
    
    When freeze pegged is set to True, infos on parameters FROM THE MAIN MODEL ONLY are returned 

    In orer to find the last models, we:
        -explore in reverse order the given xspec log lines
        -catch the last occurence of the models
        -crop the lines and split by model and group
        -select only the lines with variation (i.e. '+/-' at the end of the line) to avoid frozen and linked parameters
        -parse them for the value and replace the corresponding parameter number with the new values
        
    Note: it is assumed that the current models start with a default (no name) model
    
    To find the last errors, we parse the log lines for the last errors for each component.
    If there was a new model fit, we only parse the lines after the last improved fit 
    (to avoid fetching errors from previous fits if some errors didn't compute in the last fit)
     
    Note: These methods will need to be updated for more than 1 model per datagroups
    '''
    
    found_model=False
    
    par_peg=[]
    #searching the last 'Model' line
    
    #note: this can also apply to few other displays
    model_start_line='========================================================================\n'
    
    for i_startline in range(len(log_lines)):
        if model_start_line in log_lines[::-1][i_startline]:
            
            #we add other conditions to make sure its the start of a model display
            if 'Model' in log_lines[::-1][i_startline-1]:
                '''
                and the first model display
                for this we find the first line with Source numbe, and if its not 1, we keep searching
                '''
                i_line_source=i_startline-1
                while 'Source No.:' not in log_lines[::-1][i_line_source]:
                    i_line_source-=1
                
                #testing if its effectively the first model
                if 'Source No.: 1   Active/' in log_lines[::-1][i_line_source]:
                    found_model=True
                    break
                else:
                    continue
    
    #Exiting if we didn't find anything
    if not found_model:
        
        #if the goal is the last errors we do not need to display this message
        if goal!='lasterrors':
            print('\nCould not find a new model in the given logfile lines.')
        
        if goal=='lastmodel':
            return 0
        log_lines_lastfit=log_lines
    else:
        #note:here we stop at line "above' the detected i_start to catch the first model call
        log_lines_lastfit=log_lines[len(log_lines)-(i_startline+1):]
    
    if goal=='lasterrors':
        error_pars=np.zeros((AllModels(1).nParameters*AllData.nGroups,2))
        
        #resticting to after the last fit
        for line in log_lines_lastfit[::-1]:
            
            if freeze_pegged:
                #parsing the lines to search for all the Pegged parameters
                if line.startswith(' Parameter') and 'pegged' in line:
                    
                    #storing the parameter if it is not already in the list of stored pegged parameters
                    if line.split()[1] not in par_peg:
                        par_peg+=[line.split()[1]]
                        
            #error lines start with 5 blank spaces
            if line.startswith('    '):
                
                #testing which parameter errors the current line gives
                for parnum in range(1,AllModels(1).nParameters*AllData.nGroups+1):
                    
                    #the -1 and 0 floor are here to to keep correct identification for the last parameter of the group
                    i_grp,parnum_ingrp=par_degroup(parnum)
                        
                    #the lines with parameters before 10 begin with 5 spaces, the rest begins with 4 (we assume <100 parameters)
                    if parnum<10:
                        add_str=' '
                    else:
                        add_str=''
                    #we add a redundancy test to only use the last error computation in case of unexpected stuff hapening
                    if line.startswith('    '+add_str+str(parnum)) and (error_pars[parnum-1]==np.array([0,0])).all():
                        
                        #if the errors come from chain runs, the format is different (the error values are not always displayed) so 
                        #we need to parse them differently
                
                        if len(line.split())==3:
                            
                            #storing the errors for the MC version
                            #for now this is fine since the errors are calculated right after the MC fit is done
                            #might need to be adjusted if they are calculated later at some point
                            error_pars[parnum-1]=abs(np.array([AllModels(i_grp)(parnum_ingrp).values[0]-float(line.split()[1]),\
                                                               float(line.split()[2])-AllModels(i_grp)(parnum_ingrp).values[0]]))
                                
                        elif len(line.split())==4:
                            #storing the errors
                            error_pars[parnum-1]=np.array(line.split('(')[1].split(')')[0].split(',')).astype(float)
                            
                            #keeping the negative error positive for consistency
                            error_pars[parnum-1][0]=abs(error_pars[parnum-1][0])
        
        if freeze_pegged:
            
            #this is jus
            par_peg_infos=[]
            
            for parameter in par_peg:
                
                print('\nPegged parameter ('+str(parameter)+') detected. Freezing it...')

                #note: parameter for custom models have the name of the models beforer them in this instance
                if ':' in parameter:
                    
                    par_mod=parameter.split(':')[0]
                else:
                    par_mod=''
                    
                par_val=parameter.split(':')[-1]
                
                par_grp,par_number=par_degroup(int(par_val),mod_name=par_mod)
                
                ####Note: as of now we don't return pegged parameters for custom model to simplify things
                if par_mod=='':
                    par_peg_infos+=[[par_grp,par_number,par_mod]]
                
                AllModels(par_grp,modName=par_mod)(par_number).frozen=1
                
            AllModels.show()
            
        #reshaping the error results array to get in correct shape for each data group
        error_pars=error_pars.reshape(AllData.nGroups,AllModels(1).nParameters,2)
    
    if goal=='lastmodel' and not found_model:
        print('\nCould not find the end of the model. Stopping model load.')
        return 0
    

    '''
    In order to consider cases with several models, we split model lines between each detected model and then load them
    progressively
    '''   

    #computing the line numbers with model calls to split things evenly
    model_start_ids=np.argwhere([np.array(log_lines_lastfit)==model_start_line]).T[1]
    
    #same thing for model ends
    model_stop_line='________________________________________________________________________\n'
    model_stop_ids=np.argwhere([np.array(log_lines_lastfit)==model_stop_line]).T[1]
    
    #and storing the lines for each model separately
    model_lines_split=[]
    
    #this will work even if there are other bars after the end of the model calls because the zip will cut the enumeration
    #at the number of indexes of the model starts
    for start_id,stop_id in zip(model_start_ids,model_stop_ids):
        model_lines_split+=[log_lines_lastfit[start_id+1:stop_id]]

    for model_lines in model_lines_split:
        
        #displaying the model
        print('\nFound new model:'+model_lines[0].split('Model')[1].split('Source')[0])
        if not no_display:
            for line in model_lines:
                #printing the lines without adding a line break
                print(line[:-1])
        
        #identifying the model name to call it properly later
        if ':' not in model_lines[0].split('<1>')[0]:
            model_name=''
        else:
            model_name=model_lines[0].split(':')[0].split('Model ')[1]
            
        #splitting the lines for each group, by first identifying the starts
        if AllData.nGroups>1:
            grp_lines_ids=np.argwhere([np.array(model_lines)==elem for elem in model_lines if 'Data group' in elem]).T[1]
        else:
            grp_lines_ids=[0]
        
        grp_lines_split=[]

        if AllData.nGroups==1:
            #single data group doesn't require splitting, but we make sure to get the first line correctly to compute the npars
            start_line=np.argwhere([np.array(model_lines)==' par  comp\n'])[0][0]
            grp_lines_split+=[model_lines[start_line+1:]]
        else:
            for id_line_id,line_id in enumerate(grp_lines_ids):
                
                #the last datagroup ends at the last line
                if id_line_id==len(grp_lines_ids)-1:
                    grp_lines_split+=[model_lines[line_id:]]
                else:
                    #otherwise we take between the current and next datagroup line
                    grp_lines_split+=[model_lines[line_id:grp_lines_ids[id_line_id+1]]]
        
        #computing the number of parameters for this model
        test_len_lines=grp_lines_split[0]
        if '__________________________________' in test_len_lines[-1]:
            test_len_lines=test_len_lines[:-1]
        if 'Data group' in test_len_lines[0]:
            test_len_lines=test_len_lines[1:]
            
        npars=len(test_len_lines)
        
        #loading each data group
        for group_lines in grp_lines_split:
            
            #fetching the group number
            if AllData.nGroups==1:
                i_grp=1
            else:
                i_grp=int(group_lines[0].split('Data group: ')[1].replace('\n',''))
                
            #fetching the lines with variations
            var_lines=[line for line in group_lines if '+/-' in line]
            
            for line in var_lines:
                i_par=int(line.split()[0])%npars if int(line.split()[0])%npars!=0 else npars
                
                #note: there can be problems with paramater with very precise bounds from custom models,
                #so we adapt these bounds if necessary
                
                par_values=AllModels(i_grp,modName=model_name)(i_par).values
                
                if float(line.split()[-3])<par_values[2]:
                    par_values[2]=float(line.split()[-3])
                if float(line.split()[-3])>par_values[5]:
                    par_values[5]=float(line.split()[-3])
                    
                AllModels(i_grp,modName=model_name)(i_par).values=[float(line.split()[-3])]+par_values[1:]
                
                #freezing lines frozen during the fitting operation if the option is selected
                if freeze_pegged and line.endswith('+/-  -1.00000     \n'):
                    AllModels(i_grp,modName=model_name)(i_par).frozen=True
                    
            #also replacing frozen values if it is asked
            if replace_frozen:
                frozen_lines=[line for line in group_lines if line.endswith('frozen\n')]
                for line in frozen_lines:
                    i_par=int(line.split()[0])%npars if int(line.split()[0])%npars!=0 else npars
                    AllModels(i_grp,modName=model_name)(i_par).values=float(line.split()[-2])
                        
    
    if goal=='lasterrors':
        if freeze_pegged:
            return par_peg_infos
        else:
            return error_pars
        
    else:
        return 1

def calc_fit(timeout=30,logfile=None,iterations=None,delchi_tresh=0.1,nonew=False,noprint=False):
    
    '''
    Computes the fit with loops of fitting with query=no, while the fit is improving, until the timeout
    
    delchi_tresh is the treshold to keep iterations going, 
    
    nonew restricts the method to a single iteration
    
    if logfile is not None, computes the fit as a multiprocessing function to enable to stop it if a single run gets stuck
    '''
    
    #updating the logfile if it exists:
    if logfile is not None:
        logfile.readlines()
    
    #storing the previous query state
    old_query_state=Fit.query
    
    Fit.query='no'
    
    #replacing the iterations if the argument is passed
    if iterations is not None:
        old_iterations=Fit.nIterations
        Fit.nIterations=iterations
    
    #chi evolution variable
    chi2=Fit.statistic
    
    #fit improvement boolean    
    fit_improve=True    
    
    #initial time
    curr_time=time.time()
    
    def fit_func():
        try:
            #fitting with the constraints
            Fit.perform()
        except:
            
            #break in this case would only happen if all parameters end up pegged to a limit, in which case we won't seed any more improvmeent in the fit anyway
            pass
        
    
    while fit_improve:
        
        if logfile is not None:
            #creating the process
            p_fit=multiprocessing.Process(target=fit_func,name='fit_calc')
        
            #launching it
            p_fit.start()
            
            #stopping for 'timeout' seconds
            p_fit.join(timeout)
                        
            if p_fit.is_alive():
                
                #terminating the error if it still didn't end
                print('\nThe fit computation time depassed the allowed limit. Stopping the process...')
                p_fit.terminate()
            
            #logging the messages printed during the error computation
            log_lines=logfile.readlines()
    
            #searching for the model
            parse_xlog(log_lines,goal='lastmodel',no_display=True)        
        else:
            try:
                #fitting with the constraints
                Fit.perform()
            except:
                
                #break in this case would only happen if all parameters end up pegged to a limit, in which case we won't seed any more improvmeent in the fit anyway
                pass
        
        #we consider the fit improved if the last fit lowered the chi2 by at least 0.1
        if chi2-Fit.statistic>delchi_tresh:
            #storing the new chi²
            chi2=Fit.statistic
            
            #breaking if we go beyond the timeout
            if time.time()-curr_time>120:
                print('\nThe fit computation time depassed the allowed limit. Stopping the process...')
                break
        else:
            if not noprint:
                print("\nLast fit iteration didn't improve the chi2 significantly.Stopping the process...")
            fit_improve=False
        
    #changing back the fit parameters
    Fit.query=old_query_state
    if iterations is not None:
        Fit.nIterations=old_iterations
   

def calc_error(logfile,maxredchi=1e6,param='all',timeout=60,delchi_thresh=0.1,indiv=True,give_errors=False,freeze_pegged=False,test=False):
    
    '''
    Computes the fit errors in a multiprocessing environment to enable stopping the process after a specific time
    param can either be an error parameters string or 'all' to compute the errors for all the parameters
    
    timeout is the time one Fit.error is allowed to run before the process considers it's stuck and stops
    (The value taken in indiv mode is a quarter of timeout)
    
    if indiv is set to true, computes errors one parameter by one, with the process starting over each time a new model is found
    (similarly to calc_fit) as long as the delchi difference is superior to the threshold
    '''
    #testing if the model is currently fitted
    logfile.readlines()
    Fit.show()
    fitshow_lines=logfile.readlines()
        
    #redoing a fit if it needs to be done
    if ' Current data and model not fit yet.\n' in fitshow_lines:
        #re-computing the fit without the multiprocessing to be able to launch the errors
        calc_fit()
    
    print('\nComputing errors for up to '+str(timeout/4 if indiv else timeout)+' seconds'+(' per parameter.' if indiv else ''))
    
    if param=='all':
        glob_string_par='1-'+str(AllModels(1).nParameters*AllData.nGroups)
    else:
        glob_string_par=param    
    
    if indiv:
        #allowing computations of error for single parameters (thus with no - in the string)
        if len(glob_string_par)<=2:
            error_strlist=[glob_string_par]
        else:
            error_strlist=np.arange(int(glob_string_par.split('-')[0]),int(glob_string_par.split('-')[1])+1).astype(str).tolist()
    else:
        error_strlist=[glob_string_par]
    
    #defining the error function
    def error_func(string_par):
        try:
            #the fit returns an error when there is a new fit found so we pass the potential errors.
            Fit.error('max '+str(float(maxredchi))+' '+string_par)
        except:
            pass

    #storing the previous log chatter state and making sure the Xset logfile can correctly log new models
    curr_logchatter=Xset.logChatter
    Xset.logChatter=10
    
    #Flushing the readline
    logfile.readlines()

    #creating the test variable for a new best fit
    is_newmodel=True
    
    log_lines=[]
    
    par_peg_ids=[]
        
    #loop on the parameter str list, enclosed in a break to reset it when a new model is found (for indiv mode)
    while is_newmodel:
        
        #flushing the lines
        logfile.readlines()
        
        #initialising it at the beginning of the loop before computing the errors
        is_newmodel=False
        
        base_chi=Fit.statistic
        
        #resetting the error variable
        error_pars=np.zeros((AllData.nGroups,AllModels(1).nParameters,2))


        for error_str in error_strlist:
            
            #creating the process
            p_error=multiprocessing.Process(target=error_func,name='error_calc',args=(error_str,))
        
            
            #launching it
            p_error.start()
            
            #stopping for 'timeout' seconds
            p_error.join(timeout)

            '''
            note: in some cases with a peg or other weird situations, the error run will 
            get stuck and as such go up tot the whole length while finding the same marginal chi-2
            that won't get considered in the fit. The issue is that when piped the model changes
            are not loaded when resetting to it will attempt the same error search over an over until
            the limit.
            '''

            if p_error.is_alive():
                
                #terminating the error if it still didn't end
                print('\nThe error computation time depassed the allowed limit. Stopping the process...')
                p_error.terminate()
            
            #logging the messages printed during the error computation
            log_lines+=[logfile.readlines()]
                
            print('\nError computation finished.')
            
            if freeze_pegged:
                curr_par_peg_ids=parse_xlog(log_lines[-1],goal='lasterrors',freeze_pegged=freeze_pegged)
                #adding the pegged parameters to the list of pegged parameters at each error computation
                if curr_par_peg_ids!=[]:
                    par_peg_ids+=curr_par_peg_ids
                
            elif give_errors:
                new_errors=parse_xlog(log_lines[-1],goal='lasterrors')
                
                #in indiv mode we only update the value of the parameter for which the error was just computed
                if indiv:

                    error_pars[(int(error_str)-1)//AllModels(1).nParameters][(int(error_str)-1)%AllModels(1).nParameters]=\
                        new_errors[(int(error_str)-1)//AllModels(1).nParameters][(int(error_str)-1)%AllModels(1).nParameters]

                
            print('\nParsing the logs to see if a new minimum was found...')            
            
            #searching for the model
            is_newmodel=parse_xlog(log_lines[-1],goal='lastmodel',no_display=True)

            if is_newmodel:
                #recreating a valid fit
                calc_fit(logfile=logfile,nonew=True)            
                
                print('\nResulting model after loading...\n')
                #displaying the new model without changing the console chatter state
                curr_chatter=Xset.chatter
                Xset.chatter=10
                AllModels.show()
                Fit.show()
                Xset.chatter=curr_chatter

                if indiv and base_chi-Fit.statistic>delchi_thresh:
                    break
                else:
                    #when not in indiv mode or if the delchi isn't big enough, 
                    #there's no need for other iterations
                    is_newmodel=False
                    
    
    #changing back the Xset
    Xset.logChatter=curr_logchatter
        
    if freeze_pegged:
        return par_peg_ids
    if give_errors:
        return error_pars
            
'''Automatic fit'''

def load_fitmod(path):
    with open(path,'rb') as file:
        fitmod_save=dill.load(file)
        
    return fitmod_save
        
class fitmod:
    '''
    class used for a list of fitting components which will be used to fit in order
    complist must be a list/array of addcomp valid compnames
    idlist is either none or a list of component descriptions which will be passed to each's "identifier" method
    
    Starts with the current model as continuum (or from scratch if there's no model)
                                                
    If absval is not set to none, forces a specific value for the absorption

    '''

    def __init__(self,complist,logfile,logfile_write,absval=None,interact_groups=None,idlist=None,prev_fitmod=None):
            
        #defining empty variables
        self.name_complist=complist
        self.includedlist=[]
        self.logfile=logfile
        self.logfile_write=logfile_write
        
        self.idlist=idlist if idlist!=None else np.array([None]*len(complist))
        self.interact_groups=interact_groups
        self.complist=[]
        self.fixed_abs=absval
        self.progressive_delchis=[]
        
        #attempting to identify already existing elements in the current model
        
        self.cont_complist=[]
        # self.cont_compnames=[]
        self.name_cont_complist=[]
        
        try:
            AllModels(1)
            is_model=True
                
        except:
            print('\nNo model currently loaded. Starting from empty model...')   
            is_model=False
        
        if is_model:
            self.cont_save=allmodel_data()
            print('\nUsing current loaded model as continuum.')
            self.cont_pars=[i for i in range(1,AllModels(1).nParameters+1)]
            
            # print('\nStoring currently frozen parameters as locked.')
            # self.cont_unlockedpars=[i for i in self.cont_pars if not AllModels(1)(i).frozen]
            
            self.cont_xcompnames=AllModels(1).componentNames
            
            '''
            ####Loading the continuum
            We first try to load components from a given previous fitmod.
            Else, all existing xspec components are classed as continuum components
            '''
            
            if prev_fitmod is not None:
                    
                    for component in [elem for elem in prev_fitmod.complist if elem is not None]:
                        setattr(self,component.compname,component)
                        self.cont_complist+=[component]
                        self.name_cont_complist+=[component.compname]
                        self.complist+=[component]
                        
                        #switching the component to continuum
                        component.continuum=True
                        
                    #copying the includedlist to keep the order
                    self.includedlist+=prev_fitmod.includedlist
            else:
                
                #directly converting the existing xspec components in the various arrays
                for compname in self.cont_xcompnames:
                    
                    setattr(self,'cont_'+compname,fitcomp('cont_'+compname,self.logfile,self.logfile_write,continuum=True))
                    self.cont_complist+=[getattr(self,'cont_'+compname)]
                    self.includedlist+=[getattr(self,'cont_'+compname)]
                    self.name_cont_complist+=['cont_'+compname]
                    self.complist+=[getattr(self,'cont_'+compname)]
                    
                    
        #linking an attribute to each individual fitcomp and adding them to a list for convenience
          
        for i in range(len(self.name_complist)):
            

            #adding the continuum components not already in the model to the list of components to be added if there aren't in the 
            #current continuum
            #here we assume there are no numbered continuum components (should be editing for more complex continuums)
            #note : we keep this code part but it is not used anymore as long as we keep parenthood when creating new fitmods
            if self.name_complist[i].startswith('cont_'):
                
                #components already considered in the autofit continuum list should not be here twice
                if self.name_complist[i] not in self.name_cont_complist:
                    
                    setattr(self,self.name_complist[i],fitcomp(self.name_complist[i],self.logfile,self.logfile_write,self.idlist[i]))
                    
                    self.complist+=[getattr(self,self.name_complist[i])]
            else:
                setattr(self,self.name_complist[i],fitcomp(self.name_complist[i],self.logfile,self.logfile_write,self.idlist[i]))
                
                self.complist+=[getattr(self,self.name_complist[i])]

        self.errors=None
    
    def print_xlog(self,string):
        
        '''
        prints and logs info in the xspec log file, and flushed to ensure the logs are printed before the next xspec print
        '''
        print(string)
        self.logfile_write.write(time.asctime()+'\n')
        self.logfile_write.write(string)
        #adding a line for lisibility
        self.logfile_write.write('\n')
        self.logfile_write.flush()
    
    def update_fitcomps(self):
        
        '''
        updates fitcomp informations:
            -notably the current xcomp, xcompnam, compnumber, parlist, and unlocked_pars of each currently included component
            -the xspec names of the continuum components
            -the logfile of each component according to the currently used logfile
            
        Needed because deleting components or multiplicative (i.e. not from the right) addcomps will shift everything so this function
        should be called afterwards 
        
        This is done by directly parsing the currently included list of fitcomps (which should always be up to date), 
        in which each component's index will correspond to its corresponding xspec component index. 
        This way, we skip any need of understanding how xspec named or renamed anything
        '''

        for i_comp,comp in enumerate(self.includedlist):
            
            #skipping multiple component placeholders
            if comp is None:
                continue
            
            try:
                comp.xcompnames=[AllModels(1).componentNames[i] for i in range(i_comp-len(comp.xcompnames)+1,i_comp+1)]                
            except:
                raise ValueError
                
            comp.xcomps=[getattr(AllModels(1),comp.xcompnames[i]) for i in range(len(comp.xcompnames))]
            
            #xspec numbering here so needs to be shifted
            comp.compnumbers=np.arange(i_comp-len(comp.xcompnames)+2,i_comp+2).astype(int).tolist()

            #we directly define the parlist from the first parameter of the first component and the last of the last component
            #to ensure consistency with multi-components
            
            #note:for now we don't consider the parameter list in subsequent dagroups except for glob_constant
            first_par=getattr(comp.xcomps[0],comp.xcomps[0].parameterNames[0]).index
            last_par=getattr(comp.xcomps[-1],comp.xcomps[-1].parameterNames[-1]).index
            comp.parlist=np.arange(first_par,last_par+1).astype(int).tolist()
                       
            #testing if the component is a global_constant
            if 'constant' in comp.compname and comp.parlist[0]==1 and AllModels(1)(1).values[0]==1:
                comp.parlist+=[1+AllModels(1).nParameters*i_grp for i_grp in range(1,AllData.nGroups)]
                
            comp.unlocked_pars=[i for i in comp.parlist if (not AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen and\
                                                                  AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='')]

            
        #we also update the list of component xcomps (no multi-components here so w can take the first element in xcompnames safely)
        self.cont_xcompnames=[self.cont_complist[i].xcompnames[0] if self.cont_complist[i].included else ''\
                              for i in range(len(self.cont_complist))]
            
        #updating the logfile
        for comp in [elem for elem in self.complist if elem is not None]:
            comp.logfile=self.logfile
            comp.logfile_write=self.logfile_write

    def list_comps_errlocked(self):

        '''
        computes a list of error locked components that will be frozen before the error computation
        
        defined in a separate function to allow more flexibility later
        '''
        
        comps_errlocked=[]
        
        #refreezing the abs components before the error computation
        for allcomp in [elem for elem in self.includedlist if elem is not None]:

            #adding non fixed absorption components
            if allcomp.included and allcomp.absorption and self.fixed_abs is None:
                    comps_errlocked+=[allcomp]
                        
        return comps_errlocked
    
    def test_addcomp(self,chain=False,lock_lines=False,no_abslines=False,nomixline=True,split_fit=True,
                     ftest_threshold=def_ftest_threshold,ftest_leeway=def_ftest_leeway):
        
        '''
        Tests each of the component in the available list which are not yet in the model for how significant their addition is
        Adds the most significant out of all of them
        returns 1 as a break value to stop the loop process whenever there are no more (significant if needed) components to be added 
        (and thus there's no need to recheck for component deletion)
         
    
        if nomixline is set to True,
        the emission/absorption line components of the same type are mutually exclusive (aka including one deletes the other)
         
        the addition test uses a slightly lower threshold than theftest_threshold to allow the model to get into a better fit states in 
        which components become significant ftest wise
        '''

        #list of currently NOT included components (doesn't take into account continuum components not in the model list)
        curr_exclist=[elem for elem in self.complist if elem not in self.includedlist]
        
        if len(curr_exclist)==0:
            return True
        
        self.print_xlog('\nlog:Current available components:\n')
        self.print_xlog(str([curr_exclist[i].compname for i in range(len(curr_exclist))]))
        
        #array for the improvement of each component which can be added
        component_ftest=np.zeros(len(curr_exclist))
            
        for i_excomp,component in enumerate(curr_exclist):
                
            #avoiding starting the model with a multiplicative component
            if component.multipl and len(self.includedlist)==0:
                continue
            
            #skipping adding lines in lines locked mode
            if component.line and lock_lines:
                continue
            
            #skipping adding absorption lines in no abslines mode
            if component.named_absline and no_abslines:
                continue
            
            #excluding addition of named components with pendants when not allowing mixing between emission and absorption of the same line
            if component.named_line and nomixline:

                #fetching the name of the pendant
                if component.named_absline:
                    line_pendant_prefix=component.compname.split('_')[0].replace('abs','em')
                else:
                    line_pendant_prefix=component.compname.split('_')[0].replace('em','abs')

                #fetching the line pendant component                
                line_pendant_name_list=[elem for elem in self.name_complist if line_pendant_prefix in elem]
                
                if len(line_pendant_name_list)!=0:
                    if len(line_pendant_name_list)!=1:
                        #should not happen
                        breakpoint()
                        
                    line_pendant_name=line_pendant_name_list[0]
                    
                    line_pendant_comp=getattr(self,line_pendant_name)
                    
                    #skipping the component addition if the pendant is already included
                    if line_pendant_comp.included:
                        continue
                
            
            #not adding continuum components already in the model
            self.print_xlog('\nlog:Testing significance of '+component.compname+' component.')                    
            
            #storing the chi2/dof before adding the component
            init_chi=Fit.statistic
            init_dof=Fit.dof
            
            #copy of the includedlist for rollback after testing the component significance
            prev_includedlist=copy(self.includedlist)
            
            self.includedlist=component.addtomod(fixed_vals=[self.fixed_abs] if component.absorption\
                                                 and self.fixed_abs is not None else None,
                                                 incl_list=self.includedlist)
            
            #updating the fitcomps before anything else
            self.update_fitcomps()
                
            #fitting the component only
            component.fit(split_fit=split_fit)
    
            #fetching the interactive components from the list if it is provided                
            if self.interact_groups is not None:
                #searching the component in the intervals
                for group in self.interact_groups:
                    if component.compname in group:
                        comp_group=np.array(group)[np.array(group)!=component.compname]
                        break
                    
            '''
            unfreezing the components previously added and fitting once more every time
            we unfreeze them in reverse model order (i.e. ~increasing significance)
            '''
            if split_fit:
                #list of components not in the same group as the currently tested component
                comps_errlocked=self.list_comps_errlocked()
                                
                for i_comp,added_comp in enumerate(self.includedlist[::-1]):
                    
                    #skipping placeholders
                    if added_comp is None:
                        continue
                    
                    #if we arrived at the first (last) component of the initial continuum
                    if added_comp in self.cont_complist:
                        self.print_xlog('\nlog:Unfreezing the continuum')
                        
                        #we unfreeze all the remaining components (all from the continuum) and test them for absorption components
                        #which we error_lock
                        for cont_comp in self.includedlist[::-1][i_comp:]:
                            #here we do check for placeholders directly in the test since i_comp can factor in some placeholders
                            if cont_comp is not None and (not cont_comp.line or not lock_lines):
                                cont_comp.unfreeze()
        
                            #unfreezing constant factors for all but the first datagroup
                            if AllModels(1).componentNames[0]=='constant' and AllData.nGroups>1:
                                for i_grp in range(2,AllData.nGroups+1):
                                    AllModels(i_grp)(1).frozen=False
                                    
                    elif not added_comp.line or not lock_lines:
                        self.print_xlog('\nlog:Unfreezing '+added_comp.compname+' component.')
                        added_comp.unfreeze()
                
                    '''
                    Note: In rare cases, the fit can get stuck and increase the fit value when decreasing components (probably a bug)
                    To avoid this, we store the model and the chi² value at each unfreeze and reload the model at the end of each unfreezing
                    if the fit value has not improved
                    '''
                    
                    #storing the model and fit
                    pre_unfreeze_mod=allmodel_data()
                    pre_unfreeze_fit=Fit.statistic
                    
                    #fitting with the new freeze state
                    calc_fit(logfile=self.logfile if chain else None)
                    
                    #this is only for non continuum components
                    if self.interact_groups is not None:
                        
                        if added_comp.compname not in comp_group:
                            self.print_xlog('\nlog:Component '+added_comp.compname+' not in group of component '+component.compname+'.\n'+
                                  'adding it to the error-locked list before interaction to avoid bogging the fit down...')
                            comps_errlocked+=[added_comp]
                        else:
                            self.print_xlog('\nlog:Component '+added_comp.compname+' in the same group as component '+component.compname+'.\n')
                            
                    #freezing the error locked components before the error computation
                    for elem_comp in comps_errlocked:
                        elem_comp.freeze()
                    
                    #if we reached the continuum step, we compute the errors starting at parameter 1 (i.e. all parameters)
                    if added_comp.continuum:
                        error_str='1-'+str(AllModels(1).nParameters*AllData.nGroups)
                        
                    #in the other case, we start at the current component's first parameter index
                    else:
                        error_str=str(added_comp.parlist[0])+'-'+str(AllModels(1).nParameters*AllData.nGroups)
                        
                    calc_error(self.logfile,param=error_str,indiv=True)
                
                    #unfreezing everything back
                    for elem_comp in comps_errlocked:
                        elem_comp.unfreeze()
                    
                    #reloading the previous model iteration (aka not doing the fit with this combination of unfrozen components)
                    if Fit.statistic>pre_unfreeze_fit:
                        self.print_xlog('\nlog:Unfreezing '+added_comp.compname+' component worsens the fit.'+\
                                        '\n Restoring previous iteration for next component.')
                        pre_unfreeze_mod.load()
                        
                    #we can stop the loop after the continuum unfreezing iteration
                    if added_comp in self.cont_complist:
                        break
                
            new_chi=Fit.statistic
            new_dof=Fit.dof
            #storing the final fit in the component's save
            component.fitted_mod=allmodel_data()
            
            self.print_xlog('\nlog:Chi2 before adding the component:'+str(init_chi))
            self.print_xlog('\nlog:Chi2 after adding the component:'+str(new_chi))
            #testing the 99% significance of the comp with the associated number of d.o.f. added 
            #i.e. the number of parameters - the number of parameters we keep frozen
            self.print_xlog('\nlog:Delta chi2 for this component: '+str(init_chi-new_chi))
            
            #we always accept the component when there is no model
            #-1 because our table starts at index 0 for 1 d.o.f.
            
            if component.mandatory:
                self.print_xlog('\nlog:Mandatory component')
                
                #custom value out of standard ftest bounds
                component_ftest[i_excomp]=-3
            elif component.absorption and self.fixed_abs is not None:
                self.print_xlog('\nlog:Fixed absorption value provided. Assuming mandatory component')
                #note: we use a different chi² value to avoid conflicts when selecting one. They will simply be chosen
                #one after the other
                component_ftest[i_excomp]=-2
            else:
                #at this stage there's no question of unlinking energies for absorption lines so we can use n_unlocked_pars_base
                    
                if init_chi!=0:
                    ftest_val=Fit.ftest(new_chi,new_dof,init_chi,init_dof)
                    
                    ftest_condition=ftest_val<ftest_threshold+ftest_leeway
                    
                    delchi_condition=init_chi-new_chi>sign_delchis_table[max(0,init_dof-new_dof-1)]
                else:
                    #storing the chi² for direct comparison instead 
                    ftest_val=new_chi
                    
                    ftest_condition=True
                    
                    delchi_condition=True
                
                if ftest_condition and ftest_val>=0 and delchi_condition:

                    self.print_xlog('\nlog:The '+component.compname+' component is statistically significant.')
                        
                    #replacing strictly 0 values by a custom non zero value to avoid issues with the nonzero command later down
                    component_ftest[i_excomp]=ftest_val if ftest_val!=0 else -1
                    
                else:
                
                    self.print_xlog('\nlog:The '+component.compname+' component is not statistically significant at this stage.')


            #deleting the component (we only add the best one each time)
            component.delfrommod()
            
            #Note:here we do not actually delete the component but instead rollback to an early model state, so we do not need
            #to call update_fitcomps but in a more complex situation it would be necessary
            
            #letting some time for the component to be deleted
            time.sleep(1)
    
            #restoring the previous includedlist
            self.includedlist=prev_includedlist
        
        #filling the first remaining free element of the delchi evolution array with the current delchis array
        self.progressive_delchis+=[component_ftest]
        
        if len(component_ftest.nonzero()[0])==0:
            self.print_xlog('\nlog:No significant component remaining. Stopping fit process...')
            
            #updating the fitcomps before exiting the function
            self.update_fitcomps()
        
            return True
        
        #when there is no model the chi² are logged so the minimum works instead
        
        custom_ftest_mask=[elem in [-1,-2,-3] for elem in component_ftest]

        #fetching the minimum component in the custom ftest values if they are some
        if sum(custom_ftest_mask)!=0:
            bestcomp_in_custom_id=component_ftest[custom_ftest_mask].argmin()
            bestcomp=curr_exclist[np.argwhere(np.array(custom_ftest_mask))[0][bestcomp_in_custom_id]]
        else:
            bestcomp=np.array(curr_exclist)[component_ftest==min(component_ftest[component_ftest.nonzero()])][0]
        
        self.print_xlog('\nlog:The most significant component is '+bestcomp.compname+'. Adding it to the current model...')

        #re-adding it with its fit already loaded
        self.includedlist=bestcomp.addtomod(fixed_vals=[self.fixed_abs] if component.absorption\
                                            and self.fixed_abs is not None else None,
                                            incl_list=self.includedlist,fitted=True)
            
        #updating the fitcomps before anything else
        self.update_fitcomps()
        
        return False
                
    def test_delcomp(self,chain=False,lock_lines=False,in_add=False,ftest_threshold=def_ftest_threshold,ftest_leeway=def_ftest_leeway):

        '''
        Testing the effect of manually deleting any of the currently included components with the new configuration
        
        We do not cover the last added component since we just added it
        while in theory it's possible that its addition could allow to find a new minimum allowing the deletion of another component 
        which in turn allows to find a new minimum without this one, it seems very unlikely
        
        this step is skipped for lines in locked lines mode
        
        in_add differentiates between the use while components are being added and the second use at the end 
        (in which we test all components, and test unlinking the lines)
        also, in in_add mode, we add the same leeway than in test_addcomp to allow to get to the best fit position
        
        in the second test_delcomp out of the addition loop, this margin is not used anymore and thus the end components are still 
        significant at the right threshold
        '''
            
        if in_add:
            list_comp_test=self.includedlist[:-1]
        else:
            list_comp_test=self.includedlist
            
        for component in list_comp_test[::-1]:
            
            #skipping placeholders
            if component is None:
                continue
            
            if not component.included:
                continue
            
            if component.mandatory:
                continue
            
            #skipping deletion of fixed absorption
            if component.absorption and self.fixed_abs is not None:
                continue
            
            #skipping deletion of lines in locked lines mode
            if lock_lines and component.line:
                continue     
            
            #stopping the process if only one additive component remains
            if len([comp for comp in [elem for elem in self.includedlist if elem is not None] if not comp.multipl])==1:
                break
            
            #now we need to test for the unlinking of vashifts for the significance
            if component.named_line:
                n_unlocked_pars_with_unlink=2-(1 if AllModels(1)(component.parlist[0]).link!='' else 0)
            else:
                n_unlocked_pars_with_unlink=component.n_unlocked_pars_base
            
            self.print_xlog('\nlog:Testing the effect of deleting component '+component.compname)
                
            new_chi=Fit.statistic
            new_dof=Fit.dof
            
            #storing the current model iteration
            new_bestmod=allmodel_data()
    
            #here we don't use an enumerate in the loop itself since the positions can be modified when components are deleted
            i_comp=np.argwhere(np.array(self.includedlist)==component)[0][0]
            
            #storing the previous includedlist to come back to it at the end of the loop iteration
            prev_includedlist=self.includedlist
            

            #deleting the component and storing how many components were deleted in the process
            n_delcomp=component.delfrommod(rollback=False)

                
            #updating the current includedlist to delete as many components as what was deleted in xspec
            self.includedlist=self.includedlist[:i_comp+1-n_delcomp]+self.includedlist[i_comp+1:]
            #this time we need to update
            self.update_fitcomps()
                
            #refitting and recomputing the errors with everything free
            calc_fit(logfile=self.logfile if chain else None)
            
            del_chi=Fit.statistic
            
            #restricting the test to components which are not 'very' significant
            #we fix the limit to 10 times the delchi for the significance threshold with their corresponding number of parameters
            ftest_val=Fit.ftest(new_chi,new_dof,del_chi,new_dof+n_unlocked_pars_with_unlink)
            
            if ftest_val<ftest_threshold/100 and ftest_val>0:
                self.print_xlog('\nlog:Very significant component detected. Skipping deletion test.')
                new_bestmod.load()
                
                #updating the includedlist
                self.includedlist=prev_includedlist
                self.update_fitcomps()
                continue
                
            #we need this variable again
            comps_errlocked=self.list_comps_errlocked()
                    
            for comp_locked in comps_errlocked:
                comp_locked.freeze()
                
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)
                
            for comp_locked in comps_errlocked:
                comp_locked.unfreeze()
                
            #when we test for delcomp at the end of the procedure, we adding a new test:
            #to try to unlink before assessing the significance of the component
            if in_add or lock_lines:
                npars_unlinked=0
            else:
                npars_unlinked=self.test_unlink_lines(chain=chain)
                
            
            del_chi=Fit.statistic
            
            ftest_val=Fit.ftest(new_chi,new_dof,del_chi,new_dof+component.n_unlocked_pars_base+npars_unlinked)
            
            #second test keeps the sign_delchis_table to avoid too 'low' values of deltachi for very good fits
            
            if Fit.statistic!=0 and ftest_val>ftest_threshold+(ftest_leeway if in_add else 0) or ftest_val<0 and\
                del_chi-new_chi<sign_delchis_table[min(0,component.n_unlocked_pars_base+npars_unlinked-1)]:
                
                self.print_xlog('\nlog:Component '+component.compname+' is not significant anymore. Deleting it from the fit...')
                
                #stricto sensu we should delete them in reverse order of fit relevancy but it shouldn't really matter
                new_chi=Fit.statistic
                
                #these are not done yet for the component since we delete without rollback
                component.reset_attributes()
                
                #storing the new model iteration
                new_bestmod=allmodel_data()
                
            else:
                #we just rollback to the fit before deletion and keep exploring
                #note: the fitcomp parameters are not deleted
                
                self.print_xlog('\nlog:Component '+component.compname+' is still significant.')
                new_bestmod.load()
                
                #updating the includedlist
                self.includedlist=prev_includedlist
                
                self.update_fitcomps()
                           
    def test_unlink_lines(self,chain=False,lock_lines=False,ftest_threshold=def_ftest_threshold):
        
        '''
        tests the significance of unlinking the lines if they are linked
        
        Uses ftest (which might not be ok) as a discriminant
        
        Note: currently not used, instead the blueshift of the lines are tested at 3 sigma to see if complex are compatible
        '''
            
        n_unlinked=0
        
        for comp_unlink in [elem for elem in self.includedlist if elem is not None]:
            
            #restricting to named absorption line components
            if not comp_unlink.named_absline:
                continue
            
            #resticting to those who are linked
            if AllModels(par_degroup(comp_unlink.parlist[0])[0])(par_degroup(comp_unlink.parlist[0])[1]).link=='':
                continue
            
            #saving the model before making any change
            new_bestmod=allmodel_data()
            
            #saving the initial fit statistic for comparison
            base_chi_unlink=Fit.statistic
            base_dof_unlink=Fit.dof
            
            self.print_xlog('\nlog:Testing significance of unlinking blueshift of component '+comp_unlink.compname)
            
            #saving values to avoid issues with bounds
            par_unlink_values=AllModels(par_degroup(comp_unlink.parlist[0])[0])(par_degroup(comp_unlink.parlist[0])[1]).values
            
            #unlinking and restoring the values
            AllModels(par_degroup(comp_unlink.parlist[0])[0])(par_degroup(comp_unlink.parlist[0])[1]).link=''
            AllModels(par_degroup(comp_unlink.parlist[0])[0])(par_degroup(comp_unlink.parlist[0])[1]).values=par_unlink_values
            
            #adding the new d.o.f. to the thawed parameters of the component
            comp_unlink.unlocked_pars+=[par_degroup(comp_unlink.parlist[0])[1]]
                                                    
            #first fit while freezing all other components
            for other_comp_unlink in [elem for elem in self.includedlist if elem is not None]:
                if other_comp_unlink==comp_unlink:
                    continue
                other_comp_unlink.freeze()
                
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)
            
            #unfreezing the other ones        
            for other_comp_unlink in [elem for elem in self.includedlist if elem is not None]:
                if other_comp_unlink==comp_unlink or (other_comp_unlink.line and lock_lines) :
                    continue
                other_comp_unlink.unfreeze()

            calc_fit(logfile=self.logfile if chain else None)
            
            comps_errlocked=self.list_comps_errlocked()
            
            for comp_locked in comps_errlocked:
                comp_locked.freeze()
                
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)
                
            for comp_locked in comps_errlocked:
                comp_locked.unfreeze()
            
            new_chi_unlink=Fit.statistic
            new_dof_unlink=Fit.dof
            
            #testing the significance of the new version of the model (here we added 1 d.o.f.)
            ftest_val=Fit.ftest(new_chi_unlink,new_dof_unlink,base_chi_unlink,base_dof_unlink)
                
            if ftest_val<ftest_threshold and ftest_val>0 and base_chi_unlink-new_chi_unlink>sign_delchis_table[0]:
                self.print_xlog('\nlog:Freeing blueshift of component '+comp_unlink.compname+' is statistically significant.')
                n_unlinked+=1
            else:
                self.print_xlog('\nlog:Freeing blueshift of component '+comp_unlink.compname+' is not significant. Reverting to linked model.')
                new_bestmod.load()
                
                #deleting the d.o.f.
                comp_unlink.unlocked_pars=comp_unlink.unlocked_pars[:-1]
        
        return n_unlinked
        
    def test_mix_lines(self,chain=False):
            
        '''
        Testing the effect of manually replacing emission/absorption components of a given complex by their counterpart when both
        are in the model
        
        separated from the addcomp process because it can't be compared with adding other components
        
        this step is skipped for lines in locked lines mode
        
        '''
        
        for line_component in [comp for comp in self.includedlist if comp is not None and comp.named_line]:
            
            #### should be changed to try to include the line even if the absorption line is not included
            if not line_component.included:
                continue

            #fetching the name of the pendant
            if line_component.named_absline:
                line_pendant_prefix=line_component.compname.split('_')[0].replace('abs','em')
            else:
                line_pendant_prefix=line_component.compname.split('_')[0].replace('em','abs')

            
            #fetching the line pendant component                
            line_pendant_name_list=[elem for elem in self.name_complist if line_pendant_prefix in elem]
            
            #and testing if the line pendant is not part of the model list
            if len(line_pendant_name_list)==0:
                continue
                    
            line_pendant_name=line_pendant_name_list[0]
            
            line_pendant=getattr(self,line_pendant_name)
            
            #no ftest here because we can't compare directly like this
            curr_chi=Fit.statistic

            curr_mod=allmodel_data()
            curr_includedlist=self.includedlist


            #deleting the current line component
            delcomp(line_component.xcompnames[-1])

                
            #manually des-including the component to keep its parameters and everything
            #since we also need to delete the None one, we first fetch the position of the component
            line_comp_id=np.argwhere(np.array(self.includedlist)==line_component)[0][0]
            
            #deleting two components to also delete the None one
            self.includedlist=self.includedlist[:line_comp_id-1]+self.includedlist[line_comp_id+1:]
            
            #updating the fitcomps
            self.update_fitcomps()
            
            #adding the other component
            self.includedlist=line_pendant.addtomod(incl_list=self.includedlist)
            
            #updating the fitcomps before anything else
            self.update_fitcomps()
                
            self.print_xlog('\nlog:Fitting the new component by itself...')

            #fitting the component only
            line_pendant.fit()

            #unfreezing the rest of the components
            for comp in [elem for elem in self.includedlist if elem is not None]:
                comp.unfreeze()

            #Fitting with the whole model free
            calc_fit(logfile=self.logfile if chain else None)
            
            comps_errlocked=self.list_comps_errlocked()
            #freezing the error locked components before the error computation
            for elem_comp in comps_errlocked:
                elem_comp.freeze()

            #computing the errors
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)

            #re-unlocking error components
            for elem_comp in comps_errlocked:
                elem_comp.unfreeze()

            #testing if the new fit situation is better
            #could be improved but need more mathematical justification
            invert_line=Fit.statistic<curr_chi
            
            if invert_line:
                #we can now properly delete the other componet info
                line_component.reset_attributes()
                
            else:

                #deleting the inverted line
                line_pendant.delfrommod(rollback=False)

                    
                line_pendant.reset_attributes()
                
                #re-instating the initial fitmod iteration with the other line included and the corresponding model
                self.includedlist=curr_includedlist
                curr_mod.load()

                #and updating the fitcomps
                self.update_fitcomps()
                
    def idtocomp(self,par_peg_ids):
        
        '''
        transforms a list of group/id parameter couple into a list of component/index of the parameter in the list of that comp
        
        allows to keep track of parameters even after modifying components
        
        #here we assume custom models are not gonna be modified and thus return a different element for them (None)
        '''
        
        
        includedcomps=np.array([comp for comp in self.includedlist if comp is not None])
        
        parlist_included=np.array([elem.parlist for elem in includedcomps],dtype=object)
            
        #here the 1 index is here because both the group and the parameter number are stored
        mask_comp_pegged=[[elem[1] in comp_parlist for comp_parlist in parlist_included] for elem in par_peg_ids]
        
        id_comp_pegged=[np.argwhere(elem)[0][0] for elem in mask_comp_pegged]
            
        par_peg_comps=[[includedcomps[id_comp_pegged[i_peg]],
                        np.argwhere(np.array(parlist_included[id_comp_pegged[i_peg]])==par_peg_ids[i_peg][1])[0][0]]\
                       for i_peg,mask in enumerate(mask_comp_pegged)]
        
        return par_peg_comps
    
    def global_fit(self,chain=False,directory=None,observ_id=None,freeze_final_pegged=True,no_abslines=False,lock_lines=False,nomixline=True,split_fit=True,
                   ftest_threshold=def_ftest_threshold,ftest_leeway=def_ftest_leeway):
        
        '''
        Fits components progressively in order of fit significance when being added 1 by 1
        
        Components are added by default at the end of the model and inside global multiplicative components,
        except when keywords like glob or cont change this.
        
        When adding components that adds multiple components to the model, the parameter list of the component is changed to take into 
        account all the parameter added from all the components.
        
        every iteration starts by fitting the new component on its own, then unfreezing other added components up to the 
        continuum
        
        if chain is set to True, will load a chain from the last fit (in the directory+observ_id+identifier if given)
        in order to maximize the precision of the error computations
        '''

        #the main condition for stopping the fitting process is having added every possible component
        #of course we need to take off the placeholders to compare the lists
        while [elem for elem in self.includedlist if elem is not None]!=self.complist:
            
            comp_finished=self.test_addcomp(chain=chain,lock_lines=lock_lines,no_abslines=no_abslines,nomixline=nomixline,split_fit=split_fit,
                                            ftest_threshold=ftest_threshold,ftest_leeway=ftest_leeway)
                      
            if comp_finished:
                break
            
            #we only do this for >2 components since the first one is necessarily significant and the second one too if it was just added
            if len(self.includedlist)>2:
                self.test_delcomp(chain,lock_lines,in_add=True,ftest_threshold=ftest_threshold,ftest_leeway=ftest_leeway)

            #testing for line inversion when available
            if not lock_lines and nomixline:
                self.test_mix_lines(chain)

        '''Checking if unlinking the energies of each absorption line is statistically significant'''
                            
        #storing the chi value at this stage
        chi_pre_unlink=Fit.statistic
        dof_pre_unlink=Fit.dof
        
        if not lock_lines:
            self.test_unlink_lines(chain=chain,ftest_threshold=ftest_threshold)        

        try:
            #testing if freezing the pegged parameters improves the fit
            par_peg_ids=calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),freeze_pegged=freeze_final_pegged,indiv=True,
                                   test='FeKa26abs_agaussian' in self.name_complist and round(AllData(1).energies[0][0])==3)

        except:
            breakpoint()

        #computing the component position of the frozen parameter to allow to unfreeze them later even with modified component positions
        par_peg_comps=self.idtocomp(par_peg_ids)
        
        if len(par_peg_ids)!=0:
            
            #new fit with the new frozen state
            calc_fit(logfile=self.logfile if chain else None)        
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)

        '''
        last run of component deletion with the new minimum
        '''
        
        #this test is not skipped even if the fit has not been modified in the unlinking/pegging etc because it uses a stricter threshold
        self.test_delcomp(chain,lock_lines,in_add=False,ftest_threshold=ftest_threshold,ftest_leeway=ftest_leeway)
        
        # currently disabled
        # if not (abs(Fit.statistic-chi_pre_unlink)<0.1 and Fit.dof==dof_pre_unlink):
        #     if len([elem for elem in [comp for comp in self.includedlist if comp is not None] if not elem.multipl])>=2:
        # else:
        #     self.print_xlog('\nNo modification after main component test loop. Skipping last component deletion test.')
        
        ####testing unpegging pegged parameters
        '''
        testing if unpegging pegged parameters 1 by 1 improves the latest version of the fit
        '''
        
        if freeze_final_pegged and len(par_peg_ids)!=0:

                self.print_xlog('testing if unpegging parameters improves the latest version of the fit')
                for i_par_peg in range(len(par_peg_ids)):
                    
                    pegged_comp=par_peg_comps[i_par_peg][0]
                    #testing if the parameter is still included
                    if not pegged_comp.included:
                        continue
                    
                    #defining the current pegged_par index with the new configuration                    
                    pegged_par_index=par_peg_comps[i_par_peg][0].parlist[par_peg_comps[i_par_peg][1]]

                    #unfreezing the parameter
                    AllModels(par_peg_ids[i_par_peg][0])(pegged_par_index).frozen=False
                        
                    #computing the parameter position in all groups values    
                    par_peg_allgrp=(par_peg_ids[i_par_peg][0]-1)*AllModels(1).nParameters+par_peg_ids[i_par_peg][1]
                    
                    #no need for indiv mode here since we compute the error for a single parameter
                    calc_error(self.logfile,param=str(par_peg_allgrp))
                    
                    #re-freezing the parameter
                    AllModels(par_peg_ids[i_par_peg][0])(pegged_par_index).frozen=False
        
        ####chain
        
        '''
        Chain creation
        
        creating the MC chain for the model
        defaut Markov : algorithm = gw, bur=0, filetype=fits, length=100,
        proposal=gaussian fit, Rand=False, Rescale=None, Temperature=1., Walkers=10
        definition of chain parameters
        
        '''
        
        if chain:
            
            #new fit + error computation to get everything working for the MC (since freezing parameters will have broken the fit)
            calc_fit(logfile=self.logfile if chain else None)               
                    
            #note: we need to test again for pegging parameters as this can keep the MC chain from working
            par_peg_ids+=calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),freeze_pegged=freeze_final_pegged,indiv=True)
                        
            try:
                #new fit in case there were things to peg in the previous iteration
                Fit.perform() 
            except:
                
                #this can happen in rare cases so we re-open the parameters and do one more round of deletion
                if len(par_peg_ids)!=0:

                    for par_peg_grp,par_peg_id in par_peg_ids:
                        AllModels(par_peg_grp)(par_peg_id).frozen=False
                    
                
                self.test_delcomp(chain,lock_lines,ftest_threshold=ftest_threshold,ftest_leeway=ftest_leeway)
                
                Fit.perform()
                
            #updating the fitcomps in case a sneaky model freeze happens due to a parameter being frozen automatically by xspec 
            #(which wouldn't be considered in the current comp.unlocked_pars)
            
            self.update_fitcomps()
            
            #computing the number of free parameters (need to be done after a fit) as a baseline for the MC
            n_free_pars=sum([len(comp.unlocked_pars) for comp in [elem for elem in self.includedlist if elem is not None]])+\
                        (2*max(0,AllData.nGroups-1) if AllData.nGroups>1 and AllModels(1).componentNames[0]=='constant' else 0)
                        
            #computing the markov chain here since we will need it later anyway, it will allow better error definitions
            AllChains.defLength=4000*n_free_pars
            AllChains.defBurn=2000*n_free_pars
            AllChains.defWalkers=2*n_free_pars
            
            #ensuring we recreate the chains
            if os.path.exists(directory+'/'+observ_id+'_chain_autofit.fits'):
                os.remove(directory+'/'+observ_id+'_chain_autofit.fits')
                
            self.print_xlog('\nlog:Creating Markov Chain from the fit...')
            
            try:
                ####Note: should be replaced by emcee eventually
                Chain(directory+'/'+observ_id+'_chain_autofit.fits')
            except:
                breakpoint()
                
                
            #longer error computation with the MC (and no indiv needed here)
            self.errors=calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),give_errors=True,timeout=120)
        
        else:
            self.errors=calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),give_errors=True,indiv=True)
        #updating the fitcomps to have up to date links to the xspec components (since they break at each model_load)
        self.update_fitcomps()
        
        self.print_xlog('\nlog:Progressive fit complete.')
        
    def get_usedpars_vals(self):
        
        '''
        Gives the values and associated errors for all the final parameters of the model, 
        as well as parameters names formed as 'component name in addcomp'.'parameter name'
        Here we assume 1 group with parameters studied per spectrum
        '''

        #reorganising the errors
        par_errors=np.array([[[None]*3]*AllModels(1).nParameters]*AllData.nGroups)
        par_names=np.array([[None]*AllModels(1).nParameters]*AllData.nGroups)
        for i_grp in range(1,AllData.nGroups+1):
        
            for i_par in range(1,len(par_errors[0])+1):
                #returning specific strings if the component is linked
                if AllModels(i_grp)(i_par).link!='':
                    par_errors[i_grp-1][i_par-1]=AllModels(i_grp)(i_par).values[0],'linked','linked'
                else:
                    par_errors[i_grp-1][i_par-1]=AllModels(i_grp)(i_par).values[0],self.errors[i_grp-1][i_par-1][0],self.errors[i_grp-1][i_par-1][1]
            
        #fetching all corresponding parameter names with the custom naming of the fitcomps
        for comp_xname in AllModels(1).componentNames:
            comp=getattr(AllModels(1),comp_xname)
            for parname in comp.parameterNames:
                par=getattr(comp,parname)
                i_par=par.index
                #trying to identify a corresponding fitcomp component with a specific name
                for i_grp in range(1,AllData.nGroups+1):
                    data_prefix='_'.join(AllData(i_grp).fileName.split('_')[1:3])+'_'
                    for fcomp in [elem for elem in self.includedlist if elem is not None]:
                        if i_par in fcomp.parlist:
                            par_names[i_grp-1][i_par-1]=data_prefix+fcomp.compname+'.'+parname
                            break
                    #if that didn't lead to anything, we simply use the xspec name of the component
                    if par_names[i_grp-1][i_par-1] is None:
                        par_names[i_grp-1][i_par-1]=data_prefix+comp_xname+'.'+parname
                    
        return par_errors,par_names
        
    
    def get_absline_width(self):
        
        abs_lines=[elem for elem in self.name_complist if 'agaussian' in elem]
        
        #using 6 lines as default when no line is included
        if len(abs_lines)==0:
            n_abslines=6
        else:
            n_abslines=len(abs_lines)
        
        abslines_width=np.zeros((n_abslines,3))
        
        #and returning an empty width array
        if len(abs_lines)==0:
            return abslines_width
        
        for i_line,line in enumerate(abs_lines):
            
            fitcomp_line=getattr(self,line)
                        
            print('Computing width of component '+fitcomp_line.compname)
            
            #computing the line width at 3 sigma
            abslines_width[i_line]=fitcomp_line.get_width()
        
        return abslines_width
    
    def get_absline_info(self,par_draw,percentile=90):
        
        '''
        Computes and returns various informations on the absorption lines:
            -line flux, 
            -equivalent width with errors, computed from the chain if a chain is loaded
            -blueshift with errors from the currently stored self.errors (i.e. from a chain if a chain was used to compute them) 
            -energies and associated errors from the blueshift values
        '''
        
        mod_data_init=allmodel_data()
        
        abs_lines=[elem for elem in self.name_complist if 'agaussian' in elem]
        
        #using 6 abslines as default when there are no abslines used to go faster
        if len(abs_lines)==0:
            n_abslines=6
        else:
            n_abslines=len(abs_lines)

        abslines_flux=np.zeros((n_abslines,3))
        abslines_eqw=np.zeros((n_abslines,3))
        abslines_bshift=np.array([[None]*3]*n_abslines)
        abslines_bshift_distinct=np.array([None]*n_abslines)
        abslines_delchi=np.zeros(n_abslines)
        
        if len(abs_lines)==0:
            return abslines_flux,abslines_eqw,abslines_bshift,abslines_delchi,abslines_bshift_distinct
        
        base_chi=Fit.statistic
        
        Fit.perform()
        
        #computing how distinct the bshift distribution of the lines are in the same complex
        for i_line,line in enumerate(abs_lines):
            
            #no meaning for lines in Ka
            if i_line<3:
                continue
            if i_line in [3,4]:
                line_shift=3
            else:
                line_shift=5
                
            #fetching the current line fitcomp
            current_line_comp=[elem for elem in self.complist if lines_std_names[3+i_line] in elem.compname][0]

            #and the associated Ka line fitcomp
            Ka_line_comp=[elem for elem in self.complist if lines_std_names[3+i_line-line_shift] in elem.compname][0]

            #skipping computation if both lines are not included
            if not (current_line_comp.included and Ka_line_comp.included):
                continue
            
            #skipping the computation if either of the parameter is frozen (aka pegged aka unconstrained)
            if not current_line_comp.parlist[0] in current_line_comp.unlocked_pars and\
                    Ka_line_comp.parlist[0] in Ka_line_comp.unlocked_pars:
                continue
            
            #computing the bshift 3sigma distribution of the line
            Fit.error('max 100 8.81 '+str(current_line_comp.parlist[0]))
            
            #fetching the resulting interval
            line_bshift_inter=AllModels(1)(current_line_comp.parlist[0]).error[:2]
            
            #same thing with the Ka line
            Fit.error('max 100 8.81 '+str(Ka_line_comp.parlist[0]))
            
            #fetching the resulting interval
            Ka_bshift_inter=AllModels(1)(Ka_line_comp.parlist[0]).error[:2]
            
            #computing if the intervals intersect
            abslines_bshift_distinct[i_line]=getoverlap(line_bshift_inter,Ka_bshift_inter)==0
        
        for i_line,line in enumerate(abs_lines):
            
            fitcomp_line=getattr(self,line)
                        
            #storing the eqw (or the upper limit if there is no line)
            abslines_eqw[i_line]=fitcomp_line.get_eqwidth()
            
        #second loop since we modify the fit here and that affects the eqw computation
        for i_line,line in enumerate(abs_lines):
            
            fitcomp_line=getattr(self,line)
            
            #skipping if the line is not in the final model
            if not fitcomp_line.included:
                continue
            
            #fetching the xspec vashift parameter number of this line
            vashift_parid=fitcomp_line.parlist[0]
            
            #if the parameter is linked to something we note it instead of fetching the uncertainty:
            if AllModels(1)(vashift_parid).link!='':
                #identifying the parameter it is linked to
                vashift_parlink=int(AllModels(1)(vashift_parid).link.replace('= p',''))
                #fetching the name of the component associated to this parameter
                for comp in [elem for elem in self.includedlist if elem is not None]:
                    if vashift_parlink in comp.parlist:
                        linked_compname=comp.compname.split('_')[0]
                        break
                abslines_bshift[i_line]=[-AllModels(1)(vashift_parid).values[0],linked_compname,linked_compname]
            else:
                #creating a proper array with the blueshift of the line and its uncertainties
                #note: since we want the blueshift we need to swap the value of the vashift AND swap the + error with the - error
                abslines_bshift[i_line]=\
                    [-AllModels(1)(vashift_parid).values[0],self.errors[0][vashift_parid-1][1],self.errors[0][vashift_parid-1][0]]
            
            
            #getting the delchi of the line (we assume that the normalisation is the last parameter of the line component)
            
            #deleting the component
            delcomp(fitcomp_line.xcompnames[-1])
        
            #fitting and computing errors
            calc_fit(logfile=self.logfile)
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)
            
            #storing the delta chi
            abslines_delchi[i_line]=Fit.statistic-base_chi
            
            #reloading the model
            mod_data_init.load()
            
            '''
            Computing the individual flux of each absline
            
            For this, we delete every other component in the model then vary each parameter of the line for 1000 iterations
            from the MC computation, and draw the 90% flux interval from that
            '''
            
            #loop on all the components to delete them, in reverse order to avoid having to update them 
            for comp in [elem for elem in self.includedlist if elem is not None][::-1]:
                #skipping the current component
                if comp is fitcomp_line or comp.mandatory or (comp.absorption and comp.xcompnames[0] not in AllModels(1).componentNames):
                    continue
                
                comp.delfrommod(rollback=False)
                           
            #loop on the parameters        
            flux_line_dist=np.zeros(len(par_draw))
            for i_sim in range(len(par_draw)):
                #setting the parameters of the line according to the drawpar
                for i_parcomp in range(len(fitcomp_line.parlist)):
                    #we only draw from the first data group, hence the [0] index in par_draw
                    AllModels(1)(i_parcomp+1).values=[par_draw[i_sim][0][fitcomp_line.parlist[i_parcomp]-1]]+AllModels(1)(i_parcomp+1).values[1:]

                #computing the flux of the line 
                #(should be negligbly affected by the energy limits of instruments since we don't have lines close to these anyway)
                AllModels.calcFlux('0.3 10')
                
                #and storing it in the distribution
                #note: won't work with several datagroups
                flux_line_dist[i_sim]=AllData(1).flux[0]
                
            flux_line_dist=abs(flux_line_dist)
            flux_line_dist.sort()

            #we don't do that anymore to avoid main values outside of MC simulations in extreme cases
            # #storing the main value
            # abslines_flux[i_line][0]=AllModels.calcFlux('0.3 10')
            
            #drawing the quantile values from the distribution (instead of the main value) to get the errors
            abslines_flux[i_line][0]=flux_line_dist[int(len(par_draw)/2)]
            abslines_flux[i_line][1]=abslines_flux[i_line][0]-flux_line_dist[int((1-percentile/100)/2*len(par_draw))]
            abslines_flux[i_line][2]=flux_line_dist[int((1-(1-percentile/100)/2)*len(par_draw))]-abslines_flux[i_line][0]

                
            #reloading the model
            mod_data_init.load()
            self.update_fitcomps()
                
        #recreating a valid fit for the error computation
        calc_fit()
        
        return abslines_flux,abslines_eqw,abslines_bshift,abslines_delchi,abslines_bshift_distinct
    

    def get_eqwidth_uls(self,bool_sign,lines_bshift,sign_widths,pre_delete=False):
        
        '''
        Computes the eqwdith upper limit of each non significant line, with other parameters free except the other absorption lines
        '''
        
        abslines=[comp.compname for comp in [elem for elem in self.complist if elem is not None] if 'agaussian' in comp.compname]
    
        abslines_eqw_ul=np.zeros(len(abslines))
    
        for i_line,line in enumerate(abslines):
            
            #skipping significant lines
            if bool_sign[i_line]:
                continue
            else:
                self.print_xlog('Computing upper limit of component '+line)
                
            fitcomp_line=getattr(self,line)

            '''
            Here, we identify if lines in the same link group, or at least a Ka line, have been significantly detected,
            and fix the blueshift range to the same blueshift as these other lines if they exist.
            Otherwise, computes the upper limit from the entire available blueshift range
            '''
                        
            #fetching the Ka and Kb component names
            Ka25line_comp=[getattr(self,elem) for elem in abslines if elem.split('_')[0]=='FeKa25abs'][0]
            Ka26line_comp=[getattr(self,elem) for elem in abslines if elem.split('_')[0]=='FeKa26abs'][0]
                
            if Ka25line_comp.significant or Ka26line_comp.significant :
                #mask to identify which group the comp belongs to
                comp_link_group_mask=[fitcomp_line.compname.split('_')[0] in group for group in link_groups]
                
                #identifying if this component is included
                comp_Ka_group_prefix=link_groups[comp_link_group_mask][0][0]
                comp_Ka_id=np.argwhere(np.array(lines_std_names)==comp_Ka_group_prefix)[0][0]-3
                
                comp_Ka_group=getattr(self,[elem for elem in abslines if comp_Ka_group_prefix in elem][0])
                
                #setting the blueshift possible range to its value if it is included
                if comp_Ka_group.significant:
                    self.print_xlog('Using blueshift of component '+comp_Ka_group.compname+' in same link group.\n')
                    bshift_range=lines_bshift[comp_Ka_id]
                    bshift_range=[bshift_range[0]-bshift_range[1],bshift_range[0]+bshift_range[2]]
                    width_val=sign_widths[comp_Ka_id]
                else:   
                    #settling for the other line otherwise
                    other_comp_Ka=[elem for elem in [Ka25line_comp,Ka26line_comp] if elem is not comp_Ka_group and elem.significant][0]
                    
                    self.print_xlog('Using blueshift of component '+other_comp_Ka.compname+'\n')
                    
                    other_comp_Ka_id=np.argwhere(np.array(lines_std_names)==other_comp_Ka.compname.split('_')[0])[0][0]-3
                    
                    #and using its own blueshift
                    bshift_range=lines_bshift[other_comp_Ka_id]
                    bshift_range=[bshift_range[0]-bshift_range[1],bshift_range[0]+bshift_range[2]]
                    width_val=sign_widths[other_comp_Ka_id]
            else:
                self.print_xlog('Using full available blueshift range.\n')
                #standard range
                bshift_range=lines_e_dict[fitcomp_line.compname.split('_')[0]][1:]
                width_val=0
                
            #computing the upper limit with the given bshift range
            abslines_eqw_ul[i_line]=fitcomp_line.get_eqwidth_ul(bshift_range,width_val,pre_delete=pre_delete)
            
        return abslines_eqw_ul
    
    def save(self):
        
        '''
        saves the current model configuration
        '''
        
        self.save=allmodel_data()
        
    def dump(self,path=None):
        
        '''
        dump the current fitmod class into a file at the path location
        '''
        
        #before the dump we take off the logfile to avoid issues when reloading
        
        logfile_write=self.logfile_write
        logfile=self.logfile
        
        self.logfile_write=None
        self.logfile=None
        
        #updating the fitcomps to avoid resetting the logfile when loading
        self.update_fitcomps()
        
        with open(path,'wb') as file:
            dill.dump(self,file)
            
        #reloading
        self.logfile_write=logfile_write
        self.logfile=logfile
        self.update_fitcomps()
    
    def reload(self):
        
        '''
        Reloads the model save and updates itself to be ready for use
        '''
        
        self.save.load()
        
        self.update_fitcomps()
        
class fitcomp:
    
    '''
    class used for a singular component added in fitmod
    Stores various parameters for the automatic fit.
    compname must be a component name understandable by addcomp
    name can be any string and will be used to identify the component
    
    Warning: delfrommod with rollback set to True will reload the save made before the component was added for the last time !
    '''
    
    def __init__(self,compname,logfile=None,logfile_write=None,identifier=None,continuum=False):
        
        #component addcomp name
        self.compname=compname
        
        #component string identifier
        self.name=identifier
        
        #boolean for the component being included or not in the current model
        self.included=False        
    
        #storing the logfile variable
        self.logfile=logfile
        
        self.logfile_write=logfile_write
        
        #save of the model post component added + fit
        self.fitted_mod=None
        
        #various parameters used when the component is added to the model
        self.parlist=None
        self.unlocked_pars=None
        self.unlocked_pars_base_mask=None
        self.n_unlocked_pars_base=0
        self.compnumbers=None
        self.xcompnames=None
        self.new_includedlist=None
        self.continuum=continuum
        self.significant=False
        
        if '_' in compname:
            comp_prefix=compname.split('_')[0]
            comp_split=compname.split('_')[-1]
        else:
            comp_prefix=''
            comp_split=compname
            
        #defining if the component is an absorption component:
        if 'abs' in comp_split or 'tbnew' in comp_split or 'tbnew_gas' in comp_split:
            self.absorption=True
        else:
            self.absorption=False
            
        # or 'laor' in comp_split
        if 'gaussian' in comp_split:
            self.line=True
        else:
            self.line=False
            
        #defining if the component is a named line (i.e. there will be a vashift before)
        if comp_prefix in [elem for elem in lines_std_names if sum([char.isdigit() for char in elem])>0]:
            self.named_line=True
            
            if 'abs' in comp_prefix:
                self.named_absline=True
            else:
                self.named_absline=False
        else:
            self.named_line=False
            self.named_absline=False
            
        if 'constant' in comp_split:
            self.mandatory=True
        else:
            self.mandatory=False
            
        if comp_split in xspec_multmods:
           self.multipl=True
        else:
            self.multipl=False
            
        #parameters for continuum fitcomps
        if continuum:
            self.included=True
            self.xcomps=[getattr(AllModels(1),comp_split)]
            self.xcompnames=[comp_split]
            
            #continuum fitcomps are always mono-component so we can take the first element of xcompname here
            self.compnumbers=np.argwhere(np.array(AllModels(1).componentNames)==self.xcompnames[0])[0].tolist()
                                                
            self.parlist=[getattr(self.xcomps[0],self.xcomps[0].parameterNames[i]).index for i in range(len(self.xcomps[0].parameterNames))]
            
            #testing if the component is a global_constant
            if 'constant' in self.compname and self.parlist[0]==1 and AllModels(1)(1).values[0]==1:
                self.parlist+=[1+AllModels(1).nParameters*i_grp for i_grp in range(1,AllData.nGroups)]
                
            self.unlocked_pars=[i for i in self.parlist if (not (AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen) and\
                                                            AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='')]
            
    def print_xlog(self,string):
        
        '''
        prints and logs info in the xspec log file, and flushed to ensure the logs are printed before the next xspec print
        '''
        
        print(string)
        self.logfile_write.write(time.asctime()+'\n')
        self.logfile_write.write(string)
        self.logfile_write.flush()
        
    def addtomod(self,incl_list,fixed_vals=None,fitted=False):
        
        '''
        add the component to the model and store more fit specific parameters
        also updates the model save before doing so
        
        if fitted is set to true, loads a previous fit from fitted mod
        
        if an array of fixed_vals is provided, replaces the parameters in the component by the fixed values and 
        freezes them
        '''
        
        if self.included:
            self.print_xlog('\nlog:Fit component already added to model. Fit components can only be added once.')
        else:
            try:
                #saving the current model
                self.init_model=allmodel_data()
            
                self.old_mod_npars=AllModels(1).nParameters
            except:
                self.old_mod_npars=0
                
            #defining the list of parameter numbers associated to this/these component(s) and its/their component number(s)
            self.parlist,self.compnumbers=addcomp(self.compname,position='lastin',included_list=incl_list,return_pos=True)
                
            #fixing parameters if values are provided
            if fixed_vals is not None:
                for i_par,elem_par in enumerate(self.parlist):
                    AllModels(par_degroup(elem_par)[0])(par_degroup(elem_par)[1]).values=[fixed_vals[i_par]]+\
                    AllModels(par_degroup(elem_par)[0])(par_degroup(elem_par)[1]).values[1:]
                    AllModels(par_degroup(elem_par)[0])(par_degroup(elem_par)[1]).frozen=True
                    
            #note: the compnumbers are defined in xspec indexes, which start at 1

            #computing the unlocked parameters, i.e. the parameters not frozen at the creation of the component
            self.unlocked_pars=[i for i in self.parlist if (not AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen and\
                                                            AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='')]
            
            #computing a mask of base unlocked parameters to rethaw pegged parameters during the second round of autofit
            self.unlocked_pars_base_mask=[(not AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen and\
                                                            AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='') for i in self.parlist]
            
            if self.compname=='glob_constant' and AllData.nGroups>1:
                self.unlocked_pars+=[1+AllModels(1).nParameters*i_grp for i_grp in range(1,AllData.nGroups)]
                
            #this parameter is creating for the sole purpose of significance testing, to avoid using unlocked pars which is modified when
            #parameters are pegged (something which shouldn't affect the significance testing are pegged parameters were still variable initially)
            #with this test, the value is overwritten only the first time the component is added
            if self.n_unlocked_pars_base==0:
                self.n_unlocked_pars_base=len(self.unlocked_pars)
                
            #switching the included state
            self.included=True
            
            #updating the includedlist with placeholders and the main comp (and shifting back compnumbers to array indexes)
            new_included_list=incl_list[:self.compnumbers[0]-1]+[None]*(len(self.compnumbers)-1)+[self]+\
                               incl_list[self.compnumbers[0]-1:]
            
            self.xcompnames=[AllModels(1).componentNames[i_comp-1] for i_comp in self.compnumbers]
            
            self.xcomps=[getattr(AllModels(1),self.xcompnames[i]) for i in range(len(self.xcompnames))]
            
            if fitted:
                if self.fitted_mod is None:
                    self.print_xlog('\nlog:No fit loaded, cannot load previous fit iteration.')
                else:
                    self.fitted_mod.load()

            return new_included_list
    
    def reset_attributes(self):
        
        '''resets all fitcomp base attributes to their default value'''
        
        self.parlist=None
        self.unlocked_pars=None
        self.compnumbers=None
        self.xcompnames=None
        self.xcomps=None
        self.included=False
        self.init_model=None
        self.significant=False
        self.unlocked_pars_base_mask=None
        
    def delfrommod(self,rollback=True):
        
        '''
        rollback : load the model save before the component was saved. Doesn't necessicate to update_fitcomps afterwards
        '''
        if not self.included:
            self.print_xlog('\nlog:Fit component not yet added to model.')
            return AllModels(1)
        
        if not rollback:
                
            ndel=delcomp(self.xcompnames[-1],give_ndel=True)
            
            return ndel 
        
        else:
            
            #resetting the model instead of loading if the model was previously empty
            if self.old_mod_npars==0:

                prev_scorp=scorpeon_data()

                AllModels.clear()
                
                #reloading the NICER bg if any
                # if the scorpeon model is entirely frozen, we assume it is not being frozen and thus
                # restore the iteration after cleaning the model
                xscorpeon.load(scorpeon_save=prev_scorp if prev_scorp.all_frozen else None)

            else:
                self.init_model.load()

            self.reset_attributes()            
    
    def freeze(self):
        
        '''
        freezing the parameters of this component
        '''

    
        if not self.included:
            self.print_xlog('\nlog:Cannot freeze not included components.')
            return        
        
        #lowering the chatter to avoid thousands of lines of load messages
        prev_chatter=Xset.chatter
        prev_logchatter=Xset.logChatter
        
        Xset.chatter=0
        Xset.logChatter=0
        
        for par in self.parlist:
            AllModels(par_degroup(par)[0])(par_degroup(par)[1]).frozen=True
            
        Xset.chatter=prev_chatter
        Xset.logChatter=prev_logchatter
        
    def unfreeze(self):
        '''
        unfreezing the unlocked parameters of this component
        '''
        if not self.included:
            self.print_xlog('\nlog:Cannot freeze not included components.')
            return
        
        #lowering the chatter to avoid thousands of lines of load messages
        prev_chatter=Xset.chatter
        prev_logchatter=Xset.logChatter
        
        Xset.chatter=0
        Xset.logChatter=0
        
        for par in self.unlocked_pars:
            AllModels(par_degroup(par)[0])(par_degroup(par)[1]).frozen=False

        Xset.chatter=prev_chatter
        Xset.logChatter=prev_logchatter


            
    def fit(self,split_fit=True):
        
        '''
        Fit + errors of this component
        
        if split_fit is set to True, freezes all other components then computes the fit and error with this component unlocked only
        
        '''
        
        if not self.included:
            self.print_xlog('\nlog:Cannot fit non-included components.')
            return
        
        if split_fit:
            
            self.print_xlog('\nlog:Fitting the new component by itself...')

            #freezing everything
            allfreeze()
                
            #unfreezing this component
            self.unfreeze()
            
        else:
            self.print_xlog('\nlog:Fitting the new component with the whole model...')

        #Fitting
        calc_fit(logfile=self.logfile)

        if split_fit:
            #computing errors for the component parameters only
            calc_error(logfile=self.logfile,param=str(self.parlist[0])+'-'+str(self.parlist[-1]),indiv=True)
        else:
            #computing errors for all models
            calc_error(logfile=self.logfile,param='1-'+str(AllData.nGroups*AllModels(1).nParameters),indiv=True)
            
        self.fitted_mod=allmodel_data()

    def get_width(self):
        
        if not self.included:
            self.print_xlog('\nlog:Component not included. Returning 0 values')
            return np.zeros(3)
        
        width_par=AllModels(1)(self.parlist[-2])
        
        #returning 0 if the parameter is unconstrained (and thus has been frozen)
        if width_par.frozen:
            return np.array([0,0,0])
        
        self.logfile.readlines()

        #computing the width with the current fit
        Fit.error('stop ,,0.1 max 100 9.00 '+str(self.parlist[-2]))
        
        #storing the error lines
        log_lines=self.logfile.readlines()
        
        #testing if the parameter is pegged to 0 at 3 sigma
        if  '***Warning: Parameter pegged at hard limit: 0\n' in log_lines:
            return np.array([0,0,0])
        
        return np.array([width_par.values[0],width_par.values[0]-width_par.error[0],width_par.error[1]-width_par.values[0]])
                         
            
    def get_eqwidth(self):
        
        '''
        Note : we currently only compute the eqwidth of the first data group
        '''
        
        if not self.included:
            self.print_xlog('\nlog:Component not included. Returning 0 values')
            return np.zeros(3)
        
        #computing the eqwidth without errors first
        AllModels.eqwidth(self.compnumbers[-1])
        
        eqwidth_noerr=np.array(AllData(1).eqwidth)*1e3
        
        try:
            #eqwidth at 90%
            AllModels.eqwidth(self.compnumbers[-1],err=True,number=1000,level=90)
            
            #conversion in eV from keV (same e)
            eqwidth=np.array(AllData(1).eqwidth)*1e3
        
            #testing if the MC computation led to bounds out of the initial value :
            if eqwidth[1]<=eqwidth[0]<=eqwidth[2]:
                #getting the actual uncertainties and not the quantile values
                eqwidth_arr=np.array([abs(eqwidth[0]),eqwidth[0]-eqwidth[1],eqwidth[2]-eqwidth[0]])
            else:
                #if the bounds are outside we take the median of the bounds as the main value instead
                eqwidth_arr=\
                    np.array([abs(eqwidth[1]+eqwidth[2])/2,abs(eqwidth[2]-eqwidth[1])/2,abs(eqwidth[2]-eqwidth[1])/2])
        except:
            eqwidth_arr=eqwidth_noerr
            
        return eqwidth_arr
    
    def get_eqwidth_ul(self,bshift_range,line_width=0,pre_delete=False):
        
        '''
        Note : we compute the eqw ul from the first data group
        Also tests the upper limit for included components by removing them first
        '''
        
        if self.compname.split('_')[0]=='NiKa27abs':
            return 0
        
        curr_model=allmodel_data()
                
        distrib_eqw=[]
        
        prev_chatter=Xset.chatter
        prev_logChatter=Xset.logChatter
        
        Xset.chatter=5
        Xset.logChatter=5
        
        if type(bshift_range) not in (list,np.ndarray):
            #skipping interval computations when a precise value for the EQW is provided
            bshift_space=bshift_range
        else:
            bshift_space=np.linspace(-bshift_range[1],-bshift_range[0],101)
            
                
        with tqdm(total=len(bshift_space)) as pbar:
            
            #loop on a sampling in the line blueshift range
            for vshift in  bshift_space:
                            
                #restoring the initial model
                curr_model.load()
                
                #deleting the component if needed and the model is not in a no-abs line version
                if self.included and not pre_delete:
                    delcomp(self.compname)

                    
                #adding an equivalent component, without providing the included group because we don't want to link it
                addcomp(self.compname,position='lastin')
                
                npars=AllModels(1).nParameters
                
                #### forcing unfrozen lines even if the line is manually disabled in addcomp (unless its Nickel)
                if 'Nika' not in self.compname:
                    AllModels(1)(npars).values=[-1e-4,1e-7,-5e-2,-5e-2,0,0]
                    AllModels(1)(npars).frozen=False
                                    
                #skipping the computation if the line is above the maximum energy
                if AllModels(1)(npars-2).values[0]>AllData(1).energies[-1][1]:
                    self.print_xlog('Line above maximum energy range. Skipping computation...')
                    distrib_eqw+=[0]
                    break
                
                #freezing the width value
                AllModels(1)(npars-1).values=[line_width]+AllModels(1)(npars-1).values[1:]
                AllModels(1)(npars-1).frozen=1
                
                #freezing the blueshift
                AllModels(1)(npars-3).frozen=1
                
                #putting the vshift value
                AllModels(1)(npars-3).values=[vshift]+[vshift/1e3,bshift_space[0],bshift_space[0],bshift_space[-1],bshift_space[-1]]

                #reading and freezing the pegged parameters to keep the EW computation from crashing
                self.logfile.readlines()
                AllModels.show()

                #fitting
                calc_fit(logfile=self.logfile)
                    
                Xset.logChatter=10
                
                # #reading and freezing the pegged parameters to keep the EW computation from crashing
                # self.logfile.readlines()
                # AllModels.show()

                model_lines=self.logfile.readlines()
                parse_xlog(model_lines,freeze_pegged=True,no_display=True)

                #last attempt at fit computation with query at no
                calc_fit()

                try:
                    #computing the EQW with errors
                    AllModels.eqwidth(len(AllModels(1).componentNames),err=True,number=1000,level=99.7)
                    #adding the eqw value to the distribution
                    distrib_eqw+=[abs(AllData(1).eqwidth[1])]
                except:
                    self.print_xlog('Issue during EW computation')
                    distrib_eqw+=[0]
                
                Xset.logChatter=5
                
                pbar.update()

        Xset.chatter=prev_chatter
        Xset.logChatter=prev_logChatter
        
        #reloading the previous model iteration
        curr_model.load()
            
        return max(distrib_eqw)*1e3


####Plot commands
'''Plot commands'''
        
def Pset(window='/xs',xlog=False,ylog=False):

    '''sets a normal plot window, resets the commands and store the associated window id'''    

    if window!=None:
        windows_before=subprocess.run(['wmctrl','-l'],stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        Plot.device=window
        windows_after=subprocess.run(['wmctrl','-l'],stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    
    Plot.xAxis="keV"
    
    if xlog==True:
        Plot.xLog=True
    else:
        Plot.xLog=False
    
    if ylog==True:
        Plot.yLog=True
    else:
        Plot.yLog=False
        
    with redirect_stdout(None):
        Plot.commands=()
        Plot.addCommand('rescale')
        
    if window!=None:
        for elem in windows_after:
            if elem not in windows_before and 'PGPLOT Window ' in elem:
                xspec_windid=elem.split(' ')[0]
                print("\nIdentified this console's xspec window as "+xspec_windid)
                return xspec_windid
        
def Pnull(xspec_windid=''):

    '''Also closes the window'''
    
    Plot.device='null'

    os.system('wmctrl -ic '+xspec_windid)
    
def rescale(auto=False,autoxspec=False,xmin=None,xmax=None,ymin=None,ymax=None,reset=True):
    
    '''
    auto mode only works with a single spectrum for now
    
    To be improved to add x auto rescaling in Log mode
    '''
    if reset==True:
        Plot.commands=()
    if auto==False:
        Plot.addCommand('rescale y '+str(ymin)+' '+str(ymax))
        Plot.addCommand('rescale x '+str(xmin)+' '+str(xmax))
    else:
        Plot_add_state=Plot.add
        
        #rescaling without the components to avoid going haywire with cflux
        Plot.add=False
        
        Plotdata=[]
        Plot('data')
        
        #storing all of the spectra
        for i_sp in range(AllData.nSpectra):
            Plotdata+=Plot.y(i_sp+1)
        
        try:
            #adding as many backgrounds as there are loaded 
            Plot('background')
            for i_sp in range(AllData.nSpectra):
                Plotdata+=Plot.y(i_sp+1)
        except:
            pass
        
        Plotdata=np.array(Plotdata)
        
        #taking of zero values
        Plotdata=Plotdata[Plotdata!=0]
        
        #rescaling

        Plot.addCommand('rescale y '+str(np.min(Plotdata)*0.9)+' '+str(np.max(Plotdata)*1.1))

        Plot.add=Plot_add_state

    
class plot_save:
    
    '''
    Class saving properties of an xspec plot for later use
    
    currently accepted datatypes: (l)data,ratio,delchi
    '''
    def __init__(self,datastr):
        
        #plotting with xspec to update the Plot class
        Plot(datastr)
        
        #storing various informations about the plot        
        self.add=Plot.add
        self.addbg=Plot.background
        self.labels=Plot.labels()
        self.xLog=Plot.xLog
        self.yLog=Plot.yLog
        self.nGroups=AllData.nGroups
        
        #storing the datatype and switching to log mode if needed
        if datastr.startswith('l'):
            self.yLog=True
            self.datatype=datastr[1:]
        else:
            self.datatype=datastr
            
        #adding the main data
        self.x=np.array([None]*max(1,AllData.nGroups))
        self.xErr=np.array([None]*max(1,AllData.nGroups))
        self.y=np.array([None]*max(1,AllData.nGroups))
        self.yErr=np.array([None]*max(1,AllData.nGroups))
        
        #some of these may exist or not depending on the plot type
        for i_grp in range(1,max(1,AllData.nGroups)+1):
            try:
                self.x[i_grp-1]=np.array(Plot.x(i_grp))
            except:
                pass
            try:
                self.xErr[i_grp-1]=np.array(Plot.xErr(i_grp))
            except:
                pass
            try:
                self.y[i_grp-1]=np.array(Plot.y(i_grp))
            except:
                pass
            try:
                self.yErr[i_grp-1]=np.array(Plot.yErr(i_grp))
            except:
                pass

        #and the model
        #testing if there is a model loadded
        self.ismod=len(AllModels.sources)>0

        self.model=np.array([None]*(max(1,AllData.nGroups)))
        
        if self.ismod:         
            for i_grp in range(1,max(1,AllData.nGroups)+1):
                try:
                    self.model[i_grp-1]=np.array(Plot.model(i_grp))
                except:
                    pass
            
        #adding components if in data mode
        if self.datatype in ['data'] or 'model' in self.datatype and self.add and self.ismod:
                        
            self.addcomps=np.array([None]*(max(1,AllData.nGroups)))
            
            for i_grp in range(1,max(1,AllData.nGroups)+1):
        
                print('\nTesting number of additive components in the model...the following error is normal')
                
                n_addcomps=1
                while 1:
                    try:
                        Plot.addComp(addCompNum=n_addcomps+1)
                        n_addcomps+=1
                    except:
                        break
                    
                group_addcomps=np.array([None]*n_addcomps)
                
                #the addCompNum start at 1 and count only the additive components
                #we use up to all the component to avoid needing to know how many components there are
    
                for i_comp in range(n_addcomps):
                    try:
                        group_addcomps[i_comp]=np.array(Plot.addComp(addCompNum=i_comp+1,plotGroup=i_grp))
                    except:
                        if AllData.nGroups>0:
                            #can happen with only one component
                            Plot('ldata')
                            try:
                                group_addcomps[i_comp]=np.array(Plot.model())
                            except:
                                pass
                self.addcomps[i_grp-1]=group_addcomps
                
            #storing the addcomp names for labeling during the plots
            #### Here we assume the same addcomp names for all groups
            self.addcompnames=[]
            
            list_models=list(AllModels.sources.values())
            #adding default model component names if there is a default model
            if '' in list_models:
                
                #there are addcomps for the default models only if it has more than one additive components
                #we thus need to compare the total number of addcomps with the other model addcomps
                if len(self.addcomps[0])>(2 if 'nxb' in list_models else 0) + (2 if 'sky' in list_models else 0):
                    for elemcomp in AllModels(1).componentNames:
                        
                        #restricting to additive components
                        if elemcomp.split('_')[0] not in xspec_multmods:
                            self.addcompnames+=[elemcomp]    
                    
            #adding NICER background component Names
            if 'nxb' in list_models:
                self.addcompnames+=AllModels(1,'sky').componentNames
                
            if 'sky' in list_models:
                self.addcompnames+=AllModels(1,'nxb').componentNames
        else:
            self.addcomps=None
            self.addcompnames=[]

        self.background_x = np.array([None] * max(1, AllData.nGroups))
        self.background_xErr = np.array([None] * max(1, AllData.nGroups))
        self.background_y = np.array([None] * max(1, AllData.nGroups))
        self.background_yErr = np.array([None] * max(1, AllData.nGroups))

        # adding elements relevant to the dataplot
        if self.datatype == 'data':

            # the background (which necessitates a separate plotting to get the error)
            if self.addbg:

                try:
                    Plot('background')
                except:
                    pass

                # some of these may exist or not depending on the plot type
                for i_grp in range(1, max(1, AllData.nGroups) + 1):

                    #skipping if the background is empty
                    if len(np.nonzero(Plot.y(i_grp))[0])==0:
                        continue

                    self.background_x[i_grp - 1] = np.array(Plot.x(i_grp))

                    self.background_xErr[i_grp - 1] = np.array(Plot.xErr(i_grp))

                    self.background_y[i_grp - 1] = np.array(Plot.y(i_grp))

                    self.background_yErr[i_grp - 1] = np.array(Plot.yErr(i_grp))


def plot_saver(datatypes):

    '''wrapper to store multiple plots at once'''
 
    datatype_split=datatypes.split(',')
    plot_save_arr=np.array([None]*len(datatype_split))
    for i_plt in range(len(plot_save_arr)):
        plot_save_arr[i_plt]=plot_save(datatype_split[i_plt])

    return plot_save_arr

h_keV=6.626*10**(-34)/(1.602*10**(-16))

def EW_ang2keV(x,e_line):
    
    '''
    also works for mangtoeV
    '''
    
    l_line=h_keV*3*10**18/e_line
    
    return x*(h_keV*3*10**18)/l_line**2


def xPlot(types,axes_input=None,plot_saves_input=None,plot_arg=None,includedlist=None,group_names='auto',hide_ticks=True,
          secondary_x=True,legend_position=None,xlims=None,ylims=None,label_bg=False):
    
    '''
    Replot xspec plots using matplotib. Accepts custom types:
        -line_ratio:        standard ratio plot with iron line positions highlighted and a legend for a 2dchimap plot below
        
        #to be added if needed
        -2Dchimap:          2d chi² color + contour map
        -absorb_ratio:      modified ratio plot with absorption lines in the ratio and higlighted + absorption line positions highlighted
        
    plot_arg is an array argument with the values necessary for custom plots. each element should be None if the associated plot doesn't need plot arguments
    
    xlims and ylims should be tuples of values
    
    If plot_saves is not None, uses its array elements as inputs to create the plots. 
    Else, creates a range of plot_saves using plot_saver from the asked set of plot types
    
    if axes is not None, uses the axes as baselines to create the plots. Else, return an array of axes with the plots
    
    if includedlist is not None, replace xspec component names with fitcomp names
    
    group_names give str group names to each AllData group instead of a standard naming
    
    if group_names is set to "nolabel", doesn't show the groups
    
    hide ticks hide axis ticks for multiple plots and subplots
    
    secondary_x enables plotting an Angstrom top axis or not
    
    legend position forces a specific legend position
    '''
    
    if axes_input is None:
        fig=plt.figure(figsize=(10,8))
        grid=GridSpec(len(types.split(',')),1,figure=fig,hspace=0.)
        axes=[plt.subplot(elem) for elem in grid]
        
    else:
        axes=[axes_input] if type(axes_input) is not list else axes_input
        
    if plot_saves_input is None:
        plot_saves=plot_saver(','.join([elem if '_' not in elem else elem.split('_')[1] for elem in types.split(',')]))
    else:
        plot_saves=plot_saves_input

    #getting the group names into a list like
    if type(group_names) in (list,tuple,np.ndarray):
        group_names_list=group_names
    else:
        group_names_list=[group_names]
        
    # def plot_init(ax,plot_save):
        

            
            #plotting each model if needed
            
    #translation of standard xspec plot functions
    # def xratioplot(ax,plot_save):
                
        
    types_split=types.split(',')
    
    for i_ax,plot_type in enumerate(types_split):
        
        curr_ax=axes[i_ax]
        curr_save=plot_saves[i_ax]
        
        #plotting the title for the first axe
        if i_ax==0:

            if axes_input is None:
                curr_ax.set_title(curr_save.labels[-1])
            
            if secondary_x:
                #putting a wavelength copy of the x axis at the top           
                curr_ax_second=curr_ax.secondary_xaxis('top',functions=(ang2kev,ang2kev))
                curr_ax_second.set_xlabel('Angstrom')
                curr_ax_second.minorticks_on()
            
        #hiding the ticks values for the lower x axis if it's not in the last plot or if we're in a provided subplot
        if hide_ticks and (i_ax!=len(types_split)-1 or axes_input is not None):
            
            plt.setp(curr_ax.get_xticklabels(), visible=False)
            
            # # removing the last major tick label 
            # yticks = curr_ax.yaxis.get_major_ticks()
            # yticks[-1].label1.set_visible(False)

        curr_ax.minorticks_on()
            
        '''
        standard commands repeated for all plots
        '''
        
        curr_ax.set_xlabel(curr_save.labels[0])
        curr_ax.set_ylabel(curr_save.labels[1])
        if curr_save.xLog:
            curr_ax.set_xscale('log')
        if curr_save.yLog:
            curr_ax.set_yscale('log')
        
        #this needs to be performed independantly of how many groups there are
        if xlims is not None:
            curr_ax.set_xlim(xlims[0],xlims[1])
        if ylims is not None:
            curr_ax.set_ylim(ylims[0],ylims[1])

        for id_grp in range(max(1,curr_save.nGroups)):
        
            if group_names=='auto':
                #auto naming the group from header infos

                try:
                    with fits.open(AllData(id_grp+1).fileName) as hdul:

                        try:
                            grp_tel=hdul[1].header['TELESCOP']
                        except:
                            grp_tel=''

                        try:
                            grp_instru=hdul[1].header['INSTRUME']
                        except:
                            grp_instru=''
                        try:
                            grp_obsid=hdul[1].header['OBS_ID']
                        except:
                            try:
                                grp_obsid=hdul[1].header['OGID']
                            except:
                                grp_obsid=''
                except:
                    grp_tel=''
                    grp_instru=''
                    grp_obsid=''

                grp_name=' '.join([elem for elem in [grp_tel,grp_obsid,grp_instru] if len(elem)>0])
            else:
                grp_name='' if group_names=='nolabel' else\
                    ('group '+str(id_grp+1) if curr_save.nGroups>1 else '') if group_names is None else group_names_list[id_grp]

            if curr_save.y[id_grp] is not None:
                #plotting each data group
                curr_ax.errorbar(curr_save.x[id_grp],curr_save.y[id_grp],xerr=curr_save.xErr[id_grp],yerr=curr_save.yErr[id_grp],
                                 color=xcolors_grp[id_grp],linestyle='None',elinewidth=0.75,
                                 label='' if group_names=='nolabel' else grp_name)
            
            #plotting models
            if 'ratio' not in plot_type and curr_save.model[id_grp] is not None:
                curr_ax.plot(curr_save.x[id_grp],curr_save.model[id_grp],color=xcolors_grp[id_grp],alpha=0.5,
                             label='' if group_names=='nolabel' else '')

                    
            if 'data' in plot_type:
                
                #plotting backgrounds
                #empty backgrounds are stored as 0 values everywhere so we test for that to avoid plotting them for nothing
                if curr_save.addbg and curr_save.background_y[id_grp] is not None:
                    curr_ax.errorbar(curr_save.background_x[id_grp],curr_save.background_y[id_grp],xerr=curr_save.background_xErr[id_grp],
                                     yerr=curr_save.background_yErr[id_grp],color=xcolors_grp[id_grp],linestyle='None',elinewidth=0.5,
                                     marker='x',mew=0.5,label='' if not label_bg or group_names=='nolabel' else grp_name+' background')
        
        #### locking the axe limits
        '''
        locking the y axis limits requires considering the uncertainties (which are not considered for rescaling) 
        without perturbing the previous limits if they were beyond
        we don't bother doing that for the background
        '''

        if xlims is None:
            curr_ax.set_xlim(round(min(ravel_ragged(curr_save.x-curr_save.xErr)),2),round(max(ravel_ragged(curr_save.x+curr_save.xErr)),2))

        if ylims is None:
            curr_ax.set_ylim(min(curr_ax.get_ylim()[0],round(min(ravel_ragged(curr_save.y-curr_save.yErr)),4)),
                          max(curr_ax.get_ylim()[1],round(max(ravel_ragged(curr_save.y+curr_save.yErr)),4)))

            
        #plotting the components after locking axis limits to avoid huge rescaling
        if 'data' in plot_type and curr_save.add and curr_save.ismod:
            
            #assigning colors to the components (here we assumen the colormap used is not cyclic)
            norm_colors_addcomp=mpl.colors.Normalize(vmin=0,vmax=len(curr_save.addcomps[id_grp]))
            
            colors_addcomp=mpl.cm.ScalarMappable(norm=norm_colors_addcomp,cmap=mpl.cm.plasma)
            
            ls_types=['dotted','dashed','dashdot']

            list_models=list(AllModels.sources.values())

            #using fitcomp labels if possible            
            if includedlist is not None:
                label_comps=[comp if type(comp)==str else comp.compname for comp in [elem for elem in includedlist if elem is not None and not elem.multipl]]
                
                #adding NICER background component Names
                if 'nxb' in list_models:
                    label_comps+=AllModels(1,'sky').componentNames
                    
                if 'sky' in list_models:
                    label_comps+=AllModels(1,'nxb').componentNames
                
            else:
                label_comps=curr_save.addcompnames
                
            for id_grp in range(curr_save.nGroups):
                for i_comp in range(len(curr_save.addcomps[id_grp])):

                    try:
                        curr_ax.plot(curr_save.x[id_grp],curr_save.addcomps[id_grp][i_comp],color=colors_addcomp.to_rgba(i_comp),
                                 label=label_comps[i_comp],linestyle=ls_types[i_comp%3],linewidth=1)
                    except:
                        try:
                            curr_ax.plot(curr_save.x[0],curr_save.addcomps[id_grp][i_comp],color=colors_addcomp.to_rgba(i_comp),
                                     label=label_comps[i_comp],linestyle=ls_types[i_comp%3],linewidth=1)
                        except:
                            breakpoint()
                            print("check if other x work")

        #ratio line
        if 'ratio' in plot_type:
            curr_ax.axhline(y=1,xmin=0,xmax=1,color='green')
        if 'delchi' in plot_type:
            curr_ax.axhline(y=0,xmin=0,xmax=1,color='green')
                
        #plotting the legend for the first axe    
        curr_ax.legend(loc=legend_position)
    
    if axes_input is None:
        fig.tight_layout()
        return axes
    else:
        return None,None

def Plot_screen(datatype,path,mode='matplotlib',xspec_windid=None,includedlist=None):
    
    '''Saves a specific xspec plot, either through matplotlib or through direct plotting through xspec's interface'''
    
    if not path.endswith('.png') or path.endswith('.svg') or path.endswith('.pdf'):
        path_use=path+('.pdf' if mode=='matplotlib' else '.png')
    
    if mode=='matplotlib':
        xPlot(datatype,includedlist=includedlist)
        if not path.endswith('.png') or path.endswith('.svg') or path.endswith('.pdf'):
            path_use=path+('.pdf')
            plt.savefig(path_use)
            path_use=path+('.svg')
            plt.savefig(path_use)
            path_use=path+('.png')
            plt.savefig(path_use)
        else:
            path_use=path
            plt.savefig(path_use)
            
        time.sleep(0.1)
        plt.close()
        
    if mode=='native':
        Plot(datatype)
        os.system('import -window '+xspec_windid+' '+path_use)

def store_plot(datatype='data',comps=False):
    
    '''
    if comps=False, returns a 2*2 array containing the plot first curve's' x,xerr, y and y err
    
    if comps=True, returns an array containing the main model AND component values for each data group
    '''
    prev_plot_add_state=Plot.add
    
    Plot.add=True
    
    Plot(datatype)

    #storing the main plot values    
    plot_values=np.array([[[None,None]]*2]*AllData.nGroups)
    for i_grp in range(1,AllData.nGroups+1):
        plot_values[i_grp-1][0][0]=np.array(Plot.x(i_grp))
        plot_values[i_grp-1][0][1]=np.array(Plot.xErr(i_grp))
        plot_values[i_grp-1][1][0]=np.array(Plot.y(i_grp))
        plot_values[i_grp-1][1][1]=np.array(Plot.yErr(i_grp))
        
    if comps:
        
        print('\nTesting number of additive components in the model...the following error is normal')
        
        n_addcomps=1
        while 1:
            try:
                Plot.addComp(addCompNum=n_addcomps+1)
                n_addcomps+=1
            except:
                break
        
                
        mod_values=np.array([[None]*(n_addcomps+2)]*AllData.nGroups)
        '''
        for each data group, plot_values contains:
            -the x value for the model with the current datatype
            -the y value for the model with the current datatype
            -the y values for all the model components
        '''
        
        for i_grp in range(1,AllData.nGroups+1):

            mod_values[i_grp-1][0]=np.array(Plot.x(i_grp))
            mod_values[i_grp-1][1]=np.array(Plot.model(i_grp))

            #the addCompNum start at 1 and count only the additive components
            #we use up to all the component to avoid needing to know how many components there are

            for i_comp in range(n_addcomps):
                try:
                    mod_values[i_grp-1][2+i_comp]=np.array(Plot.addComp(addCompNum=i_comp+1,plotGroup=i_grp))
                except:
                    #can happen with only one component
                    Plot('ldata')
                    mod_values[i_grp-1][2+i_comp]=np.array(Plot.model())
                
        Plot.add=prev_plot_add_state
        return (plot_values,mod_values)
    Plot.add=prev_plot_add_state
    return plot_values


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:01:03 2021

@author: parrama
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


from astropy.io import fits
from scipy.integrate import trapezoid

'''
This script can be imported just for load_fitmod in the streamlit pipelines
In this case we don't need xspec and rewriting a copy of everything would be a mess 
so we test whether we're in this case and otherwise just skip the function and variable runs that would crash
we still make empty variables wih the name of all the xspec functions that are used to avoid issue
'''

try:
    from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,AllChains,Chain
    streamlit_mode=False
except:
    streamlit_mode=True
    AllModels, AllData, Fit, Spectrum, Model, Plot, Xset, AllChains, Chain=[None]*9


from fitting_tools import sign_delchis_table, lines_e_dict, lines_w_dict, lines_broad_w_dict, \
    link_groups, lines_std_names, def_ftest_threshold, def_ftest_leeway, lines_std

from general_tools import ravel_ragged,get_overlap,shorten_epoch,ang2kev


from contextlib import redirect_stdout
import subprocess
import os
import numpy as np
import re
from tqdm import tqdm
import time
import multiprocessing
from copy import copy
import dill

from matplotlib.gridspec import GridSpec

if 'xspec_models_dir' in os.environ:
    #example '/home/parrama/Soft/Xspec/Models'
    model_dir=os.environ['xspec_models_dir']
else:
    model_dir="/home/parrazyte/Soft/Xspec/Models"

# custom model loads
if not streamlit_mode and model_dir!=None:
    AllModels.lmod('relxill',dirPath=model_dir+'/relxill/2.6')

    # AllModels.lmod('fullkerr',dirPath=model_dir+'/fullkerr')
    # #swiftJ1658 dust scattering halo model from Jin2019
    #
    # AllModels.lmod('ismabs', dirPath=model_dir + '/ismabs.v1.2')
    # Xset.addModelString('ISMABSROOT', os.path.join(model_dir,'ismabs.v1.2/'))
    #
    #
    # AllModels.lmod('dscor', dirPath=model_dir + '/dscor')
    # #which needs an absolute path to a directory
    # Xset.addModelString('DSCOR_MODEL_DIR', os.path.join(model_dir,'dscor/mtables_swf1658m42'))

    # instype is the
    # type
    # of
    # instrument:
    # 0: XMM - Newton / pn
    # 1: XMM - Newton / MOS1
    # 2: XMM - Newton / MOS2
    # 3: Chandra / ACIS
    # 4: Swift / XRT
    # 5: NuSTAR / FPMA
    # 6: NuSTAR / FPMB
    pass

AllModels.mdefine('crabcorr (1./E^dGamma)crabcorrNorm : mul')

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
     gsmooth    kerrconv      rdblur       simpl    crabcorr     dscor
'''.split()

def is_abs(comp_split):
    '''
    Rule for defining standard absorption components in the model logic
    '''

    return ('abs' in comp_split and comp_split!='gabs') or 'TB' in comp_split\
                or 'tbnew' in comp_split or 'tbnew_gas' in comp_split

xspec_globcomps=\
'''
constant    crabcorr
'''.split()


#hopefully there's enough there
xcolors_grp=['black','red','limegreen','blue','cyan','purple','yellow',
             'black','red','limegreen','blue','cyan','purple','yellow',
             'black','red','limegreen','blue','cyan','purple','yellow',
             'black','red','limegreen','blue','cyan','purple','yellow',
             'black','red','limegreen','blue','cyan','purple','yellow',
             'black','red','limegreen','blue','cyan','purple','yellow',
             'black','red','limegreen','blue','cyan','purple','yellow']

xscat_pos_dict={'4U1630':0.9}


def rebinv_xrism(grp_number=1,sigma=2,max_bins=5000):
    Plot.setRebin(sigma, max_bins, grp_number)

def ignore_data_indiv(e_low_groups,e_high_groups,reset=False,sat_low_groups=None,sat_high_groups=None,
                      glob_ignore_bands=None):

    if reset:
        AllData.notice('all')
        AllData.ignore('bad')

    if type(e_low_groups) in [float,np.float64] and type(e_high_groups) in [float,np.float64]:
        e_low_groups_indiv=np.repeat(e_low_groups,AllData.nGroups)
        e_high_groups_indiv=np.repeat(e_high_groups,AllData.nGroups)
    else:
        e_low_groups_indiv=e_low_groups
        e_high_groups_indiv=e_high_groups
    for i_grp,(e_low,e_high) in enumerate(zip(e_low_groups_indiv,e_high_groups_indiv)):
            AllData(i_grp+1).ignore('**-'+str(e_low)+' '+str(e_high)+'-**')

    if sat_low_groups is not None and sat_high_groups is not None:
        #note: this will do nothing if e_sat_low indiv and e_sat_high_indiv are kept at their default values
        #because this will always be the same or bigger than the bands interval
        for i_grp, (e_sat_low, e_sat_high) in enumerate(zip(sat_low_groups, sat_high_groups)):
            AllData(i_grp + 1).ignore('**-' + str(e_sat_low) + ' ' + str(e_sat_high) + '-**')

    if glob_ignore_bands is not None:
        for elem_ignore_bands in glob_ignore_bands:
            if elem_ignore_bands is not None:
                for ignore_band in elem_ignore_bands:
                    AllData.ignore(ignore_band)

    if sat_high_groups is not None:
        max_e=max([min(elem_high,elem_sat_high) for (elem_high,elem_sat_high) in\
                  zip(e_high_groups_indiv,sat_high_groups)])
    else:
        max_e=max(e_high_groups)

    if max_e>12:
        Plot.xLog=True
    else:
        Plot.xLog=False

def ignore_indiv_ig(line_cont_ig_indiv):
    '''
    Wrapper around a list of line_cont_ig to avoid future mistakes forgetting to not ignore empty ignores
    '''


    for i_sp in range(AllData.nGroups):
        if line_cont_ig_indiv[i_sp] != None:
            AllData(i_sp + 1).ignore(line_cont_ig_indiv[i_sp])

def notice_indiv_ig(line_cont_ig_indiv):
    '''
    Wrapper around a list of line_cont_ig to avoid future mistakes forgetting to not ignore empty ignores
    '''


    for i_sp in range(AllData.nGroups):
        if line_cont_ig_indiv[i_sp] != None:
            AllData(i_sp + 1).notice(line_cont_ig_indiv[i_sp])


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

def load_list(iterable):
    AllData(' '.join([str(i+1)+':'+str(i+1)+' '+elem for i,elem in enumerate(iterable)]))

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

    def load(self,verbose=False,load_scorpeon=True):

        for elem_mod,elem_key in zip(self.mod_list,self.mod_keys):
            model_load(getattr(self,elem_mod),mod_name='' if elem_mod=='default' else elem_mod,mod_number=elem_key,
                       verbose=verbose)

        xchatter=Xset.chatter
        xlogchatter=Xset.logChatter

        #doing this silently to avoid surcharching the screen and log files
        Xset.chatter=10 if verbose else 0
        Xset.logChatter=10 if verbose else 0

        if 'scorpeon' in dir(self) and load_scorpeon:

            is_scorpeon=False

            for i_grp in range(AllData.nGroups):
                try:
                    AllModels(i_grp+1,'nxb')
                    is_scorpeon=True
                except:
                    pass

            if not is_scorpeon:
                print('No scorpeon model detected but save exists. Reloading in auto first...')

            xscorpeon.load('auto' if is_scorpeon else None,scorpeon_save=self.scorpeon)

        Xset.chatter=xchatter
        Xset.logChatter=xlogchatter

    def store(self,path):

        with open(path, 'wb+') as dump_file:
            dill.dump(self, file=dump_file)


def load_mod(path,load_xspec=True):

    with open(path, 'rb') as dump_file:
        allmod_data= dill.load(dump_file)

        if load_xspec:
            allmod_data.load()

        return allmod_data


mod_sky=None
mod_nxb=None

def obs_loader(grp_str,i_grp=1,mode='swift_mela',save_baseload=True):
    if mode=='swift_mela':
        grp_str_use=[]
        if os.path.isfile(grp_str+'wtsource.pi'):
            grp_str_use+=[grp_str+'wt']
        if os.path.isfile(grp_str+'pcsource.pi'):
            grp_str_use+=[grp_str+'pc']
        for id_obs,obs_str in enumerate(grp_str_use):
            AllData(str(i_grp+id_obs)+":"+str(i_grp+id_obs)+' '+obs_str+'source.pi')
            AllData(i_grp+id_obs).response=obs_str+'.rmf'
            AllData(i_grp+id_obs).response.arf=obs_str+'.arf'
            AllData(i_grp+id_obs).background=obs_str+'back.pi'

    if save_baseload:
        Xset.save('baseload.xcm')

def set_ener(mode='thcomp',xrism=False):
    if mode=='thcomp':

        if xrism:
            AllModels.setEnergies('0.01 0.1 1000 log, 10. 19800 lin, 1000. 1000 log')
        else:
            AllModels.setEnergies('0.01 1000. 10000 log')


def make_ls(low_e,high_e):
    Xset.restore('xrism_save_stronglines.xcm')
    Plot.setRebin(2,5000,1)
    freeze()
    mod_cont=allmodel_data()

    from linedet_utils import narrow_line_search,plot_line_search

    Xset.chatter=1
    narrow_out_val=narrow_line_search(mod_cont,'mod_cont',[1.5,1.5],
                                      [low_e,high_e,5e-3],line_cont_range=[low_e,high_e])
    with open('narrow_out_'+str(low_e)+'_'+str(high_e)+'.dill','wb+') as f:
        dill.dump(narrow_out_val,f)

    plot_line_search(narrow_out_val, './', 'XRISM', suffix=str(low_e)+'_'+str(high_e),
                     save=True, epoch_observ=['first_fit'], format='pdf')

    return narrow_out_val

class model_data:

    '''
    Class for xspec model data to allow adding/deleting components more easily
    Also contains a copy of the component/parameter arborescence to allow for an easier access
    '''

    def __init__(self,xspec_model):

        self.expression=xspec_model.expression
        self.npars=xspec_model.nParameters
        self.comps=xspec_model.componentNames

        #note: we do this because some models have length 1 values for choices (ex: xscat grain type)
        #in which case we can't use a regular npars*6 type array
        values=np.array([None]*self.npars)

        links=np.array([None]*self.npars)
        frozen=np.zeros(self.npars).astype(bool)

        for  i in range(1,self.npars+1):

            #keeping length 1
            values[i-1]=np.array(xspec_model(i).values)

            #only for normal parameters
            if len(values[i-1])>1:
                # safeguard against linked parameters with values out of their bounds
                if values[i-1][0] < values[i-1][2]:
                    values[i-1][2] = values[i-1][0]
                if values[i-1][0] > values[i-1][5]:
                    values[i-1][5] = values[i-1][0]

            links[i-1]=xspec_model(i).link.replace('= ','')
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

                #safeguard against linked parameters with values out of their bounds
                if self.values[0]<self.values[2]:
                    self.values[2]=self.values[0]
                if self.values[0]>self.values[5]:
                    self.values[5]=self.values[0]

                self.links[id_par]=getattr(getattr(self,elem_comp),elem_par).link.replace('=','')
                self.frozen[id_par]=getattr(getattr(self,elem_comp),elem_par).frozen
                id_par+=1

class scorpeon_manager:

    '''
    Global class to manage scorpeon model backgrounds in xspec. A single instance of this is created, called xscorpeon
    '''

    def __init__(self):
        self.bgload_paths=None

    def load(self,bgload_paths=None,scorpeon_save=None,load_save_freeze=True,frozen=False,extend_SAA_norm=True,
             fit_SAA_norm=False,fit_prel_norm=True):

        '''
        reloads the nicer bg model(s) from the stored path(s), and scorpeon save(s) if any

        can be called without bgload_paths after the first load to reset/reload bg models with or without a bg
        datagroups with no models should be left as empty in the bgload_paths array

        as of now, loads up to a single model per datagroup

        extend_SAA norm:
            -extends the maximal values of the nxb_SAA_norm parameter which for now are by default kept way too low
            for SAA passages

        fit_SAA_norm:
            -unfreezes this parameter which is usually frozen to allow to fit it

        fit_prel_norm:
            -unfreezes this parameter which is usually frozen to allow to fit it
            Don't know why it's currently frozen by default when the spectrum is created.
        '''

        #updating the bgload paths if an argument if prodivded
        if bgload_paths is not None:
            #auto loading with normal names
            if type(bgload_paths)==str and bgload_paths=='auto':
                bgloads_auto=np.array([None]*AllData.nGroups)
                for id_grp in range(AllData.nGroups):
                    #note that we don't use fileinfo anywhere because it crashes with i.e. integral spectra
                    with fits.open(AllData(id_grp+1).fileName) as hdul:
                        if 'TELESCOP' in hdul[1].header and hdul[1].header['TELESCOP']=='NICER':
                            bgloads_auto[id_grp]=AllData(id_grp+1).fileName.replace('_sp_grp_opt.pha','_bg.py')\
                                                                  .replace('_sr.pha','_bg.py')
                self.bgload_paths=bgloads_auto

            # converting the input into an array like for easier manipulation
            elif type(bgload_paths) not in [list,np.ndarray,tuple]:
                self.bgload_paths=[bgload_paths]
            else:
                self.bgload_paths=bgload_paths

        if self.bgload_paths is not None:
            #making sure the file actually exists
            if not np.array([elem is None or os.path.isfile(str(elem)) for elem in self.bgload_paths]).all():
                print('One or more scorpeon load file path does not exist')
                raise ValueError


            #loading all of the models
            for i_bg,bg_path in enumerate(self.bgload_paths):

                if bg_path is not None:
                    nicer_bkgspect=i_bg+1
                    exec(open(bg_path).read())

        #freezing parameters if the current data doesn't cover large enough ranges
        #(see https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/scorpeon-xspec/)

        for i_grp in range(AllData.nGroups):

            if self.bgload_paths is None or self.bgload_paths[i_grp] is None:
                continue

            #assuming 1 spectrum per group here
            curr_grp_sp=AllData(i_grp+1)

            try:
                mod_nxb = AllModels(i_grp + 1, modName='nxb')

                # extending the range for the SAA_norm parameter (note: might need to go above that for obs
                # with really high overshoots, but then careful about the spectrum...)

                #identifying the nxb saa norm parameter
                nxb_par_saa_list=[i_par for i_par in range(1,mod_nxb.nParameters+1) if 'saa_norm' in mod_nxb(i_par).name]
                assert len(nxb_par_saa_list)==1,'Issue in saa_norm parameter identification'
                nxb_par_saa_id=nxb_par_saa_list[0]

                #identifying the nxb prel_norm parameter
                nxb_par_prel_list=[i_par for i_par in range(1,mod_nxb.nParameters+1) if 'prel_norm' in mod_nxb(i_par).name]
                assert len(nxb_par_prel_list)==1,'Issue in saa_norm parameter identification'
                nxb_par_prel_id=nxb_par_prel_list[0]

                #here there's several parameters with the "nom" parameter name so we need to use the component
                comp_nxb_noise=getattr(mod_nxb,[elem for elem in mod_nxb.componentNames if '_noise' in elem][0])
                noisenorm_par_id=comp_nxb_noise.norm.index

                #note: should be useless now
                if extend_SAA_norm:
                    mod_nxb(nxb_par_saa_id).values = mod_nxb(7).values[:4] + [6000, 6000]

                if fit_SAA_norm:
                    mod_nxb(nxb_par_saa_id).frozen = False

                if fit_prel_norm:
                    mod_nxb(nxb_par_prel_id).frozen = False

                if curr_grp_sp.energies[0][0]>0.5:

                    #freezing noise_norm
                    mod_nxb(noisenorm_par_id).frozen=True

            except:
                pass

            try:
                mod_sky=AllModels(i_grp+1,modName='sky')

                gal_nh_par_id_list=[i_par for i_par in range(1,mod_sky.nParameters+1) if 'gal_nh' in mod_sky(i_par).name]
                assert len(gal_nh_par_id_list) == 1, 'Issue in gal_nh_par_id parameter identification'
                gal_nh_par_id=gal_nh_par_id_list[0]

                halo_em_par_id_list=[i_par for i_par in range(1,mod_sky.nParameters+1) if 'halo_em' in mod_sky(i_par).name]
                assert len(halo_em_par_id_list) == 1, 'Issue in halo_em_par_id parameter identification'
                halo_em_par_id=halo_em_par_id_list[0]

                lhb_em_par_id_list=[i_par for i_par in range(1,mod_sky.nParameters+1) if 'lhb_em' in mod_sky(i_par).name]
                assert len(lhb_em_par_id_list) == 1, 'Issue in lhb_em_par_id parameter identification'
                lhb_em_par_id=lhb_em_par_id_list[0]

                if curr_grp_sp.energies[0][0]>0.5:

                    #freezing gal_nh
                    mod_sky(gal_nh_par_id).frozen=True

                    #freezing lhb_em
                    mod_sky( lhb_em_par_id).frozen=True

                if curr_grp_sp.energies[0][0]>1.:
                    #freezing halo_em
                    mod_sky(halo_em_par_id).frozen=True
            except:
                pass

        self.nxb_frozen_states=np.repeat(None,AllData.nGroups)
        self.sky_frozen_states=np.repeat(None,AllData.nGroups)

        #saving the initial freeze states if the saves have been reloaded
        if self.bgload_paths is not None:

            for i_grp in range(AllData.nGroups):

                if self.bgload_paths[i_grp] is None:
                    continue

                try:
                    mod_nxb=AllModels(i_grp+1,modName='nxb')

                    nxb_frozen_states_grp=np.repeat(None,mod_nxb.nParameters)

                    for i_par in range(mod_nxb.nParameters):

                        nxb_frozen_states_grp[i_par]=mod_nxb(i_par+1).frozen

                    self.nxb_frozen_states[i_grp]=nxb_frozen_states_grp
                except:
                    pass

                try:
                    mod_sky=AllModels(i_grp+1,modName='sky')

                    sky_frozen_states_grp=np.repeat(None,mod_sky.nParameters)

                    for i_par in range(mod_sky.nParameters):

                        sky_frozen_states_grp[i_par]=mod_sky(i_par+1).frozen

                    self.sky_frozen_states[i_grp]=sky_frozen_states_grp

                except:
                    pass

        #loading all of the saves
        if scorpeon_save is not None:
            try:
                scorpeon_save.load(load_frozen=load_save_freeze)
            except:
                breakpoint()

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


    def clear(self):
        mod_save=allmodel_data()
        AllModels.clear()
        mod_save.load(load_scorpeon=False)

    def freeze(self):

        #freezing the model if asked to
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

    def unfreeze(self):
        # self.bgload_paths=None
        # self.load(frozen=False)

        #unfreezing the model if asked to, using the freeze states
        for i_grp in range(AllData.nGroups):

            try:
                mod_nxb=AllModels(i_grp+1,modName='nxb')

                for i_par in range(mod_nxb.nParameters):

                    mod_nxb(i_par+1).frozen=self.nxb_frozen_states[i_grp][i_par]
            except:
                pass

            try:
                mod_sky=AllModels(i_grp+1,modName='sky')

                for i_par in range(mod_sky.nParameters):

                    mod_sky(i_par+1).frozen=self.sky_frozen_states[i_grp][i_par]
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


    def load(self,load_frozen=False):

        '''
        if load_frozen is set to False, will check the current value range to avoid changing the current
        frozen states
        '''

        for i_grp in range(len(self.nxb_save_list)):

            nxb_save=self.nxb_save_list[i_grp]
            sky_save=self.sky_save_list[i_grp]

            if nxb_save is not None:
                mod_nxb=AllModels(i_grp+1,modName='nxb')

                for i_par in range(mod_nxb.nParameters):

                    curr_par_frozen=mod_nxb(i_par+1).frozen

                    mod_nxb(i_par+1).values=nxb_save.values[i_par]

                    mod_nxb(i_par+1).link=nxb_save.link[i_par]

                    mod_nxb(i_par+1).frozen=False if not load_frozen and not curr_par_frozen else nxb_save.frozen[i_par]

            if sky_save is not None:
                mod_sky=AllModels(i_grp+1,modName='sky')

                for i_par in range(mod_sky.nParameters):

                    curr_par_frozen=mod_sky(i_par+1).frozen

                    mod_sky(i_par+1).values=sky_save.values[i_par]
                    mod_sky(i_par+1).link=sky_save.link[i_par]
                    mod_sky(i_par+1).frozen=sky_save.frozen[i_par]

                    mod_sky(i_par+1).frozen=False if not load_frozen and not curr_par_frozen else sky_save.frozen[i_par]


class scorpeon_group_save:

    '''
    simple scorpeon save for a single datagroup
    '''

    def __init__(self,model):

        self.values=[model(i_par+1).values for i_par in range(model.nParameters)]

        self.link=[model(i_par+1).link.replace('=','') for i_par in range(model.nParameters)]

        self.frozen=[model(i_par+1).frozen for i_par in range(model.nParameters)]

if not streamlit_mode:
    xscorpeon=scorpeon_manager()

def save_broad_SED(path=None,e_low=0.1,e_high=100,nbins=1e3,retain_session=False,
                   remove_abs=True,remove_gaussian=True,remove_cal=True,
                   remove_scorpeon=True):

    '''
    Saves a version of the model with nbins log bins betwen e_low an e_high keV
    either to path or to a returned array

    three columns file to match stefano's code

    -retain_session:    save the Xset session before modifying the model and energies array and reloads it once
                        the model has been saved

    -remove_abs:        remove the abs/xabs components before saving the model

    -remove_gaussian:   remove the gaussian components in the model before saving the SED

    -remove_cal:        remove the calibration edge components in the model before saving the SED
    '''

    #saving the session and modifying the model if asked to
    if retain_session:
        if os.path.isfile('make_broad_mod_save.xcm'):
            os.remove('make_broad_mod_save.xcm')

        Xset.save('make_broad_mod_save.xcm')

    if remove_abs:
        for elem_abs_comp in ['phabs','TBabs','TBfeo','xabs','xscat']:
            while elem_abs_comp in AllModels(1).componentNames:
                delcomp(elem_abs_comp)
                time.sleep(1)

    if remove_gaussian:
        for elem_gaussian in ['gaussian','gabs']:
            while elem_gaussian in AllModels(1).componentNames:
                delcomp(elem_gaussian)
                time.sleep(1)

    if remove_cal:
        #removing the edge calibration components
        for elem_edge in ['edge','smedge']:
            while elem_edge in AllModels(1).componentNames:
                delcomp(elem_edge)
                time.sleep(1)
        #removing the line calibration components
        for elem_comp in AllModels(1).componentNames:
            if 'gaussian' in elem_comp:
                if getattr(AllModels(1),elem_comp).LineE.values[0] in [1.74]:
                    delcomp(elem_comp)

    if remove_scorpeon:
        xscorpeon.clear()

    cleaned_expression=AllModels(1).expression

    if nbins is not None:
        AllModels.setEnergies(str(e_low)+' '+str(e_high)+' '+str(int(nbins))+" log")

    #computing and storing the broadband flux
    AllModels.calcFlux(str(e_low)+' '+str(e_high))
    broad_flux=AllData(1).flux[0]

    Plot.xAxis='hz'

    Plot('emo')
    f_nu=Plot.model()
    nu=Plot.x()
    nu_err=Plot.xErr()

    save_arr=np.array([nu,nu_err,f_nu]).T

    if retain_session:
        Xset.restore('make_broad_mod_save.xcm')
        os.remove('make_broad_mod_save.xcm')

    data_groups_str='' if AllData.nGroups=='0' else 'Loaded epoch ids : '+' '.join(shorten_epoch('auto'))
    if path is not None:
        np.savetxt(path, save_arr, header='save of '+cleaned_expression+' with '+str(int(nbins))+
                                          ' log bins in the [%.2e'%e_low+',%.2e'%e_high+'] keV band'+
                                          ' | %.2e'%e_low+'-%.2e'%e_high+' flux: %.3e'%broad_flux+' ergs/cm²/s'
                                          '\n'+data_groups_str+'\nnu (Hz) \tnuErr (Hz) \tLnu (erg/s/Hz)',
                   delimiter=' ')
    else:
        return save_arr

def compute_snr(sp_source,emin,emax,sp_back=None,rmf_source=None,e_step=None,mode='keV'):

    '''
    Careful, will need to rescale the bakcscale for the Resolve spectra to consider the branching ratios

    mode can be keV or channel
    In channel mode, emin and emax need to be integers ,and e_step must be big enough so that it doesn't create floats

    6.4-7 is 12600-14000 in XRISM channels

    '''

    AllData.clear()
    AllModels.clear()

    AllData('1:1 '+sp_source)
    if rmf_source is not None:
        AllData(1).response=rmf_source

    if sp_back is not None:
        AllData(1).background=sp_back

    Plot.xAxis=mode

    if e_step is None:
        AllData.ignore('**-'+str(emin)+' '+str(emax)+'-**')

        SNR=np.sqrt(AllData(1).rate[0] * AllData(1).exposure)
        return SNR
    else:
        e_range=np.arange(emin,emax+e_step/2,e_step,dtype=int if mode=='channel' else float)
        SNR_arr=np.repeat(np.nan,len(e_range)-1)

        for i in range(len(e_range)-1):
            AllData.notice('**')
            AllData.ignore('**-'+str(e_range[i])+' '+str(e_range[i+1])+'-**')
            SNR_arr[i]=np.sqrt(AllData(1).rate[0] * AllData(1).exposure)

        return SNR_arr

def compa_SNR(sp_source_list,sp_back_list,
              emin,emax,e_step,
              rmf_source_list=None,mode='keV',
              save_suffix='',):
    '''
    Makes a SNR comparison for a list of spectra and/or backgrounds, and stores it in an array that is saved
    '''

    if rmf_source_list is None:
        rmf_source_list_use=np.repeat([None,len(sp_source_list)])
    else:
        rmf_source_list_use=rmf_source_list

    SNR_arr_list=np.array([None]*len(sp_source_list))

    for i_src,(elem_sp_source,elem_sp_back,elem_rmf_source) in enumerate(zip(sp_source_list,sp_back_list,rmf_source_list_use)):
        SNR_arr_list[i_src]=compute_snr(elem_sp_source,elem_sp_back,emin=emin,emax=emax,e_step=e_step,
                                        rmf_source=elem_rmf_source,
                                        mode=mode)

    with open('compa_SNR_dump'+('_' if save_suffix!='' else '')+save_suffix+'.dill','wb+') as f:
        dill.dump(SNR_arr_list,f)

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
        self.link=xspec_parameter.link.replace('=','')
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

    if Xset.version[1]=='12.15.1':
        print('Xspec 12.15.1 detected. Setting APECROOT and NEIAPECROOT to 3.1.3 for apec and bvvrnei')
        Xset.addModelString('NEIAPECROOT', '3.1.3')
        Xset.addModelString('APECROOT', '3.1.3')
        Xset.addModelString('APECTHERMAL', 'YES')
    Xset.abund='wilm'
    Fit.nIterations=1000
    Plot.xAxis='keV'
    Plot.add=True
    Fit.statMethod='cstat'
    Fit.query='no'
    Plot.xLog=False

def xLog_rw(path):

    '''
    opeans path as a new Xset Log file, reconfigures for line buffering to avoid delays when printing there
    and opens it independantly for reading
    returns both the w+ and the r io objects
    '''

    logfile_write = Xset.openLog(path)

    # ensuring the log information gets in the correct place in the log file by forcing line to line buffering
    logfile_write.reconfigure(line_buffering=True)

    logfile_read=open(logfile_write.name, 'r')

    return logfile_write,logfile_read

def model_load(model_saves,mod_name='',mod_number=1,gap_par=None,in_add=False,table_dict=None,modclass=AllModels,
               verbose=False):

    '''
    loads a mod_data class into the active xspec model class or all model_data into all the current data groups
    if model_save is a list, loads all model_saves into the model data groups of the AllModels() data class

    can be used to lad custom models through mod_name and mod_number. The default values update the "standard" xspec model


    gap par:    introduces a gap in the parameter loading. Used for loading with new expressions
                including new components in the middle of the model.
                gap_par must be an interval string, i.e. '1-4'

    in_add:     controls a display of the model and the fit from after the change in the model

    table_dict: inherited argument when using delcomp, to allow to load custom fit table models correctly

    '''
    #lowering the chatter to avoid thousands of lines of load messages
    prev_chatter=Xset.chatter
    prev_logchatter=Xset.logChatter

    Xset.chatter=10 if verbose else 0
    Xset.logChatter=10 if verbose else 0

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
    xModel(first_save.expression,table_dict=table_dict,mod_name=mod_name,mod_number=mod_number)

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
                xspec_mod_grp(i_par).link=save_grp.links[i_par-1].replace('=','')

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
                    #NOTE: as of now, only works with link expressions using a single parameter

                    if save_grp.links[i_shifted-1]!='':

                        #fetching the link parameter inside the link expression
                        save_grp_link_par_str=[elem for elem in split_str_expr(save_grp.links[i_shifted-1])
                                               if 'p' in elem]

                        assert len(save_grp_link_par_str)==1,'link with several parameters detected.'+\
                                                        'Custom functions update required'

                        save_grp_link_par_str=save_grp_link_par_str[0]

                        #getting something that can be transformed back to an int
                        save_grp_link_par=save_grp_link_par_str.replace('=','').replace(mod_name+':','').replace('p','')

                        #the shift by one and 0 minimum is here to allow all the last parameters of each groups to stay in the group below
                        link_pointer=str(int(save_grp_link_par)+(gap_end-gap_start+1)*\
                                         (max(int(save_grp_link_par)-1,0)//first_save.npars))

                        #here we test if there were links to after the gap in a single group
                        #The second test is here for the last parameter of the group
                        if int(save_grp_link_par)%first_save.npars>=gap_start or \
                            (int(save_grp_link_par)%first_save.npars==0 and first_save.npars>=gap_start):
                            link_pointer=str(int(link_pointer)+(gap_end-gap_start+1))

                        #replacing the pointer by the real expression with the new parameter
                        link_pointer=save_grp.links[i_shifted-1].replace(save_grp_link_par_str,'p'+link_pointer)

                        link_pointer=(mod_name+':' if mod_name!='' else '')+link_pointer
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

def split_str_expr(string):

    xspec_str=[]

    #variable defining the index of the start of the current word if any
    comp_str=-1

    for i in range(len(string)):
        if string[i] in '()+-* ':
            # adding the current punctuation if the last element was also a ponctuation
            if comp_str == -1:
                xspec_str += [string[i]]
            # in the other case, adding the previous word which started from the index
            else:
                xspec_str += [string[comp_str:i], string[i]]

            # and resetting the word count
            comp_str = -1
        else:
            # starting the word count at the current index
            if comp_str == -1:
                comp_str = i
            # storing the word if we arrived at the last char of the string
            if i == len(string) - 1:
                xspec_str += [string[comp_str:i + 1]]

    return xspec_str
def numbered_expression(expression=None,mult_conv='auto',mod_name=''):

    '''
    Return an edited model xspec expression with each component's naming in the current xspec model
    by default, uses the current model expression

    the mult_conv determines whether the * should be replaced by parenthesis
    (to know when to add new components as parts of multiplicative components)
        options:
            -auto (default): only replaces the * for global xspec_globcomps and absorption components
            -full:  replaces everything
            -False: doesn't replace

    Also returns a dictionnary containing all custom table components and their xspec names

    Note that for double * multiplications of the type X*A*B(C+D), this will give issues but this shouldn't happen

    '''

    if expression is None:
        string=AllModels(1,mod_name).expression
    else:
        string=expression

    dict_tables={}

    xspec_str=split_str_expr(string)

    i=0
    j=0

    #this will be used to store parenthesis
    add_par=[]

    #we use a while here since xspec_str can be modified
    while j<len(xspec_str):

        elem=xspec_str[j]

        if elem=='*':
            if mult_conv=='full' or (mult_conv=='auto' and \
                                     (xspec_str[j-1].split('_')[0] in xspec_globcomps or \
                                   is_abs(xspec_str[j-1].split('_')[0]))):
                xspec_str[j]='('
                add_par+=[')']
                count_lpar=0
                j+=1
                continue
            else:
                j+=1
                continue

        if elem=='(':
            if len(add_par)>0:
                count_lpar+=1

            j+=1
            continue

        if elem in ')+- ':

            #adding an ending parenthesis if necessary
            if len(add_par)!=0:
                #if we are following a * transformed into ( who englobes another (, we need to wait for the end
                #parenthesis (or several of them potentially)
                if count_lpar!=0:
                    if elem == ')':
                        count_lpar-=1
                else:
                    xspec_str.insert(j,add_par[-1])
                    add_par=add_par[:-1]

            j+=1
            continue

        #testing if the component is a table
        if '{' in xspec_str[j]:
            if xspec_str[j] not in dict_tables.values():
                #adding the table
                dict_tables[AllModels(1,mod_name).componentNames[i]]=xspec_str[j]

        #needs a proper call to AllModels(1) here to be able to use it whenever we want
        xspec_str[j]=AllModels(1,mod_name).componentNames[i]

        i+=1
        j+=1

    #adding any last parenthesis
    xspec_str+=add_par

    return ''.join(xspec_str),dict_tables

def calc_EW(x_vals,y_cont,y_line,broad_vs_resol=True):

    '''
    Needs the plot to be in energies
    return values in keV

    broad_vs_vresol: the line doesn't take a negligible (<100 bins) part of the energies
    '''

    assert Plot.xAxis=='keV', 'Issue: needs xAxis to be set to keV'

    #converting inputs to arrays to avoid issues with selection
    x_vals_use=np.array(x_vals)
    y_cont_use=np.array(y_cont)
    y_line_use=np.array(y_line)

    line_int=trapezoid(y_line_use-y_cont_use,x_vals_use)

    line_type='em' if line_int>0 else 'abs'

    if line_type=='abs':
        line_center_id=(y_line_use/y_cont_use).argmin()
    elif line_type=='em':
        line_center_id = (y_line_use/ y_cont_use).argmax()

    delta_id_max=min(line_center_id,len(x_vals_use)-line_center_id)

    #since the continuum isn't constant we compute the integral to get the values for different EWs
    cont_int=[trapezoid(y_cont_use[line_center_id-(i+1)//2:line_center_id+(i//2)],
                        x_vals_use[line_center_id-(i+1)//2:line_center_id+(i//2)])\
              for i in range(min(100,delta_id_max*2) if not broad_vs_resol else delta_id_max*2)]

    eners=[x_vals_use[line_center_id-(i+1)//2:line_center_id+(i//2)]\
              for i in range(min(100,delta_id_max*2) if not broad_vs_resol else delta_id_max*2)]


    #computing where the line integral matches the continuum integral

    line_w_id=abs(abs(line_int)-cont_int).argmin()

    line_w=(-1 if line_type=='abs' else 1) * (x_vals_use[line_center_id+(line_w_id//2)-1]-\
                                             x_vals_use[line_center_id-(line_w_id+1)//2])

    return round(line_w,6)

'''
In order to allow the use of custom atable & mtable components in the fitting architecture,
we need to replace the mtable model names (which appear when using AllModels.componentNames) with their fits table

in order to do so, we scan for potential mtables in the current model at each addcomp and then load the model with
a custom function that replaces the expression with the fits
'''

def xModel(expression,table_dict=None,modclass=AllModels,mod_name='',mod_number=1,return_mod=False):

    '''
    replace the elements in expression with names stored as custom xspec table model names in table_dict, and replaces
    accordingly to be able to load the model
    '''

    #variable expression that will be modified recursively
    expr_load=expression

    if table_dict is not None:

        table_keys = table_dict.keys()
        for table_comp in list(table_keys):

            #quick hack to avoid replacing the name inside of the mtable if it matches the component name
            #note: won't work if several tables have similar name matching a single component
            expr_load = expr_load.replace(table_dict[table_comp], 'XXXXXXXXXXXXXXX')

            #replacing the table component in the new model expression
            expr_load=expr_load.replace(table_comp,table_dict[table_comp])


            expr_load = expr_load.replace('XXXXXXXXXXXXXXX',table_dict[table_comp])

    #deleting eventual remains of numbered components
    for i in range(100):
        if '}_'+str(i) in expr_load:
            expr_load.replace('}_'+str(i),'}')

    modclass+=(expr_load,mod_name,mod_number)

    if return_mod:
        return AllModels(1,mod_name)

def addcomp(compname,position='last',endmult=None,return_pos=False,mod_name='',
            included_list=None,values=None,links=None,frozen=None,mult_conv_rule='auto'):

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

        -lastinall:last component, but inside all global multiplications
        Note: very useful when adding new components in a model with multiple global multiplicative components
        (e.g. absorption, calibration constants, etc.)

        -lastin: last component, but inside the last multiplication
        Note: very useful when adding new calibration components which we only want to be multiplied by
        a global calibration component

        -*xspec component name (with underscore numbering if its not the first)/associated number: BEFORE this component
        the numbering is considered actual component numbering (starting at 1) for positive numbers,
        and array numberings (negatives up to 0) for negative numbers

    the "endmult" keyword refers to the position of the end parenthesis for multiplicative components:
        -None: only include up to the next additive component
        -all/-1: ends at the end of the model
        -*xspec component name/associated number: AFTER this component

    mult_conv_rule: determines whether to replace the components with a single multiplication
    with parenthesis (which will integrate the new component inside even if it's added after that component)
    or not.
    Default value is at 'auto' which only does that for global absorption and calibration components
    Can be set to True or False to do things more manually if needed

    custom components (work together):

    Continuum:
        -glob_+ multiplicative component: abs with position=first and endfactor=all

        -glob_constant: global constant factor, frozen at 1 for the first data group and free for all others

        -glob_crabcorr: global constant factor+ multiple powerlawin normalization. All free except for first datagroup

        -Suzaku_crabcorr: crabcorr variation where only the Front-Illuminated CCD (xis0_xis3)
                          have a free to vary deltagamma

        -cont_+component: added inside absorption and edge components
                            NOTE: assumes that all currently existing absorption and edge components are at the
                            beginning of the model

        -disk_nthcomp: Continuum nthcomp with Te fixed to the disk's

        -disk_thcomp: thcomp multiplicating the disk, with frozen cutoff at 100 keV



        For the following (physical) models:
            The search for physical values is performed by parsing if
             objname is in the inclination, mass or distance tables

        -objname_kerrbb:
            kerrbb with i, M, D fixed at physical values if they exist.

            Also for anything with a prefix including kerrbb,
            -the spectral hardening factor is left free between 1.4 and 1.9
            -normalization: frozen to 1 if all three physical parameters are fixed

        -objname_relxillCp
            relxill model with i, M, D fixed at physical values if they exist.

            Also for anything with a prefix including relxillCp
            -the normalization fraction refl_frac is set to -1 to only include the reflected spectrum
            -Rbr is fixed at 18 Rg
            -Index 1 and Index 2 are left free
            -Rin is set at -1 (R_ISCO) and Rout is set at 1000 Rg

    Dust scattering
        -globOBJNAME_xscat: scattering component linked with existing absorption if any
                            regions extraction sizes extracted automatically
                            position fixed to a specific value if the object name is in the xscat_pos_dict dictionnary
    Lines :
        use (Linetype)-(n/a/na/...)gaussian

        'n' -> narrow (frozen Sigma=0) gaussian
        'a'-> absorption (locked <0 norm) gaussian
        'b'-> broad (forced high width values) gaussian

        named lines (to be included as prefix):

        for 'named' lines with the ionisation number, adds a vashift component with [X,Yk] in blueshift and a gaussian with energy frozen at the line
        baseline energy level (see constants at the start of the code for up to date parameter definitions)

        'FeKa1' ->  neutral Fe emission at 6.40
        'FeKb1' ->  neutral Fe emission at 7.06
        'FeKa25' -> FeXXV absorption at 6.70keV.
        'FeKa26' -> FeXXVI absorption at 6.97keV

        'NiKa27' -> NiXXVII absorption at 7.80 keV -> currently locked at 3000km/s max (i.e. ~7.88keV max) to avoid overlap with FeKb2
        'FeKb25' -> FeXXV absorption at 7.89keV
        'FeKb26' -> FeXXVI absorption at 8.25keV
        'FeKg26' -> FeXXVI absorption at 8.70 keV

        lineAem/abs_gaussian/lorentz
        ->adds a linked complex with all the subcomponents of the line linked together in width and velocity

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

    Calibration:
        calNICERSiem_gaussian -> NICER Si absorption line at 1.74keV (see 10.1093/mnras/stac2038 for an exemple)
        calNICER_edge -> NICER 2.42keV absorption edge. ONLY APPLIED TO NICER datagroups
        calNuSTAR_edge -> NuSTAR 9.51keV absorption edge. ONLY APPLIED TO NuSTAR datagroups

        calibration edge components are only applied to their respective telescope's datagroups.
        The remainder have MaxTau fixed at 0

    all compnames must stay 'attribute'-valid else we won't be able to call them explicitely in fitcomps
    '''

    '''Component initialisation'''

    #component identifier booleans
    multipl=False

    start_position=position
    end_multipl=endmult

    line_type=None
    line_set=False
    narrow_line=False
    abs_line=False
    added_link_group=None

    #defining the model source number according to its name
    if len(AllModels.sources.keys())==0:
        source_n=1
    else:
        source_n=int(np.array(list(AllModels.sources.keys()))[(np.array(list(AllModels.sources.values())) == mod_name).tolist()][0])

    #splitting custom parts of the component
    if '_' in compname:
        comp_custom=compname.split('_')[0]
        comp_split=compname.split('_')[1]
    else:
        comp_custom=None
        comp_split=compname

    if comp_split in xspec_multmods or comp_split.startswith('mtable'):
        multipl=True
    #dichotomy between custom models

    #global keyword for multiplicative components
    if multipl and comp_custom is not None\
            and ('glob' in comp_custom or (comp_split=='crabcorr' and comp_custom in ['Suzaku'])):
        start_position=1
        end_multipl=-1
        #staying inside the constant factor if there is one
        if AllModels(1,mod_name).componentNames[0] in xspec_globcomps:
            start_position+=1

        main_compnames = AllModels(1,mod_name).componentNames

        #maiting inside the edge components (assumed to be from calibration if there are some
        if comp_split not in xspec_globcomps:
            start_position+=sum(['edge' in elem for elem in main_compnames])

    if comp_custom is not None:
        if 'cal' in comp_custom:

            #note: gaussian calibration components are placed in lastin at the very end of the model
            if comp_split!='gaussian':
                start_position=1
                if multipl:
                    end_multipl=-1
                #staying inside the constant factor if there is one
                if AllModels(1,mod_name).componentNames[0] in xspec_globcomps:
                    start_position+=1

    if 'thcomp' in comp_split:
        #increasing the energies
        AllModels.setEnergies('0.01 1000.0 5000 log')

    #continuum type components
    if comp_custom=='cont' or comp_split.lower() in ['nthcomp','thcomp']:
        start_position=1
        try:

            main_compnames = AllModels(1,mod_name).componentNames

            #staying inside the constant factor if there is one
            if AllModels(1,mod_name).componentNames[0] in xspec_globcomps:
                start_position+=1

            #maintaining inside the absorption component if there are some
            start_position+=sum([elem.startswith('TB') or ('abs' in elem and elem!='gabs') or ('scat' in elem)\
                                 for elem in main_compnames])

            #maiting inside the edge components if there are some
            start_position+=sum(['edge' in elem for elem in main_compnames])

        except:
            pass

    #testing for lines
    if 'gaussian' in comp_split or 'lorentz' in comp_split:

        line=True

        line_type=comp_custom

        #restricting to the letter prefix
        comp_split_prefix=comp_split.replace('gaussian','').replace('lorentz','')

        #identifying the shape of the line
        narrow_line='n' in comp_split_prefix
        broad_line='b' in comp_split_prefix
        abs_line='a' in comp_split_prefix

        #updating the comp split which will be implemented without the letter keywords
        comp_split='gaussian' if 'gaussian' in comp_split else 'lorentz'

        #restricting vashift use to named lines
        if line_type is not None and sum([char.isdigit() for char in line_type])>0:

            #identifying line sets
            if line_type.replace('em','').replace('abs','').endswith('A'):
                line_set=True
                #identifying the number of line candidates

                #e.g. FeKa25
                line_set_complex=line_type.split('A')[0]

                #e.g. abs
                line_set_type=line_type.split('A')[-1]

                line_set_comps=[elem for elem in list(lines_e_dict.keys()) if
                line_set_complex in elem and line_set_type in elem
                                #here to only use resolved components
                                 and len(elem.split(line_set_complex)[1].split(line_set_type)[0]) > 0]

                comp_split='vashift*('+'+'.join([comp_split]*len(line_set_comps))+')'
            else:
                comp_split='vashift*'+comp_split

            named_line=True

                #### link groups off for now
                # #and link groups to absorption lines (stays None for emission lines)
                # if line_type.endswith('abs'):
                #     #(there can only be one since they do not share any element)
                #     added_link_group=[elem for elem in link_groups if line_type in elem][0]
        else:
            named_line=False
    else:
        line=False

    if compname=='c_zxipcf':
        comp_split='cabs*zxipcf'

    '''component creation'''

    #checking if the current model is empty
    try:
        AllModels(1,mod_name)

        is_model=True
    except:
        is_model=False

        #this one doesn't need to be loaded in a specific way with tables since here there is no model to begin with
        xspec_model=Model(comp_split)

    if is_model:
        #saving the current models
        model_saves=getattr(allmodel_data(),'default' if mod_name=='' else mod_name)

        #getting the xspec expression of the current model as well as the list of components
        num_expr,table_dict=numbered_expression(mult_conv=mult_conv_rule,mod_name=mod_name)

        #replacing a * by parenthesis for single constant*additive models to have an easier time later

        xcomps=AllModels(1,mod_name).componentNames

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
            elif start_position in ['last','lastin','lastinall']:
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
            if len(AllModels(1,mod_name).componentNames)>1:
                if num_expr[num_expr.find(AllModels(1,mod_name).componentNames[1])-1]=='*':
                    num_expr=num_expr.replace('*'+AllModels(1,mod_name).componentNames[1],'('+AllModels(1,mod_name).componentNames[1]+')',1)

            #at the very end of the model but inside parentheses
            if position in ['lastin','lastinall']:

                #counting the number of parenthesis at the end of the current model
                count_par_final=0
                if is_model:
                    for char in num_expr[::-1]:
                        if char==')':
                            count_par_final+=1
                        else:
                            break

                if position=='lastin':
                    #capping at one for the single lastin
                    count_par_final=min(count_par_final,1)

                #adding the componet inside the desired number of parentheses
                if count_par_final!=0:
                    new_expr=num_expr[:-count_par_final]+'+'+comp_split+num_expr[-count_par_final:]
                else:
                    new_expr=num_expr+'+'+comp_split

                pass
            else:
                #at the very end of the model
                new_expr=num_expr+'+'+comp_split
            pass
        else:

            #if we are inserting our component inside of a single multiplicative component, we must replace the * by parenthesis
            if num_expr[num_expr.find(xcomp_start)-1]=='*':
                num_expr=num_expr.replace('*'+xcomp_start,'('+xcomp_start+')',1)

            #inserting at the desired position
            new_expr=num_expr.replace(xcomp_start,comp_split+('+' if not multipl else '(')+xcomp_start,1)

        #introducing the end parenthesis
        if multipl:
            try:
                #note: the 1 is important here to avoid replacing in the wrong component
                '''
                here the issue is that we can replace fits table name if they have the name of their model 
                in their fits name
                Thus, we temporarily replace the comp_split element by a string that'll never be changed, 
                then reset it again
                '''
                new_expr=new_expr.replace(comp_split,'XXXXXXXXXXXXXXX')
                new_expr=new_expr.replace(xcomp_end,xcomp_end+')',1)
                new_expr=new_expr.replace('XXXXXXXXXXXXXXX',comp_split)
            except:
                breakpoint()

        #returning the expression to its xspec readable equivalent (without numbering)
        for elemcomp in AllModels(1,mod_name).componentNames:

            #here the second condition is to avoid supressing custom model names who have no numebering
            if '_' in elemcomp and elemcomp[elemcomp.rfind('_')+1:].isdigit():
                #the 1 is important here to avoid replacing twice incompletely some of the first named components
                new_expr=new_expr.replace(elemcomp,elemcomp[:elemcomp.rfind('_')],1)

        #updating the save
        model_saves[0].expression=new_expr

        #computing the gap with the new component(s)
        old_ncomps=len(AllModels(1,mod_name).componentNames)

        if xcomp_start!=-1:

            #We compute the start of the gap from the 'old' version of the model
            gap_start=getattr(getattr(AllModels(1,mod_name),xcomp_start),getattr(AllModels(1,mod_name),xcomp_start).parameterNames[0]).index

            xcomp_start_n=np.argwhere(np.array(AllModels(1,mod_name).componentNames)==xcomp_start)[0][0]
            #we compute the end gap as the parameter before the first parameter of the starting comp in the newer version of the model
            #We use the component number instead of its name to avoid problems

            try:
                xspec_model=xModel(new_expr,table_dict,return_mod=True,mod_number=source_n,mod_name=mod_name)
            except:
                print(new_expr)
                breakpoint()
                xspec_modelxspec_model = xModel(new_expr, table_dict, return_mod=True,mod_number=source_n,mod_name=mod_name)
                print(new_expr)

            added_ncomps=len(xspec_model.componentNames)-old_ncomps
            shifted_xcomp_start=xspec_model.componentNames[xcomp_start_n+added_ncomps]

            gap_end=getattr(getattr(xspec_model,shifted_xcomp_start),getattr(xspec_model,shifted_xcomp_start).parameterNames[0]).index-1

            #storing the position of the last component added for return if asked
            #the actual component is the one after but we're in array indices here so it already corresponds to the xspec indice of the one before
            added_comps_numbers=np.arange(xcomp_start_n+1,xcomp_start_n+added_ncomps+1).astype(int)
        else:
            gap_start=AllModels(1,mod_name).nParameters+1

            try:
                xspec_model=xModel(new_expr,table_dict,return_mod=True,mod_number=source_n,mod_name=mod_name)
            except:

                '''
                Here to help debugging because there'll be issues with weird namings at some point
                '''
                print(new_expr)
                breakpoint()
                print(new_expr)

            gap_end=xspec_model.nParameters
            added_comps_numbers=np.arange(old_ncomps+1,len(AllModels(1,mod_name).componentNames)+1).astype(int)

        gap_str=str(gap_start)+'-'+str(gap_end)

        model_load(model_saves,mod_name=mod_name,mod_number=source_n,
                   gap_par=gap_str,in_add=True,table_dict=table_dict)

        #we need to recreate the variable name because the model load has overriden it
        xspec_model=AllModels(1,mod_name)

    else:
        xspec_model=Model(comp_split,sourceNum=source_n,modName=mod_name)
        gap_start=1
        gap_end=AllModels(1,mod_name).nParameters
        added_comps_numbers=np.arange(1,len(AllModels(1,mod_name).componentNames)+1).astype(int)

    '''continuum specifics'''

    #restricting the continuum powerlaw's photon index to physical values
    if compname=='cont_powerlaw':
        xspec_model(gap_end-1).values=[1.0, 0.01, 1.0, 1.0, 3.5, 3.5]

    #restricting the curvature bb/diskbb's kt to physical values
    if compname=='cont_diskbb':
        xspec_model(gap_end-1).values=[1.0, 0.01, 0.5, 0.5, 3.0, 3.0]

    #restricting the curvature bb's kt to physical values
    if compname=='cont_bb':
        xspec_model(gap_end-1).values=[1.0, 0.01, 0.1, 0.1, 4.0, 4.0]

    if compname=='c_zxipcf':
        #linking the cabs absorption to the zxipcf absorption
        xspec_model(gap_end-4).link=str(gap_end-3)

    '''line specifics (additive only)'''

    #this only works for non continuum components but we can assume gaussian lines will never be continuum components

    #switching the norm values of the gaussian
    if line and not line_set:
        if abs_line:

            xspec_model(gap_end).values=[-1e-4,1e-7,-5e-2,-5e-2,0,0]

            # #### ON: disabling FeKa25
            # if line_type=='FeKa25abs':
            #     xspec_model(gap_end).values=[-1e-7,1e-7,-5e-2,-5e-2,0,0]
            #     xspec_model(gap_end).frozen=True

            #### ON: disabling NiKa27 to avoid degeneracies
            if line_type=='NiKa27abs':
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
    if narrow_line:
        xspec_model(gap_end-1).values=[0]+xspec_model(gap_end-1).values[1:]
        xspec_model(gap_end-1).frozen=True

    #changing more infos for specific lines
    if line_type is not None:

        if line_set:
            #selecting the entire set of lines
            line_comps_use=line_set_comps
        else:
            line_comps_use=[line_type]

        n_line_comps=len(line_comps_use)

        #in reverse to add them in forward order
        for i_line_type,elem_line_type in enumerate(line_comps_use[::-1]):

            #adjusting the norm if in a set since its more practical to do it here

            if line_set:

                if abs_line:
                    xspec_model(gap_end-3*i_line_type).values = [-1e-4, 1e-7, -5e-2, -5e-2, 0, 0]
                else:
                    # stronger normalisations allowed for the emission lines
                    xspec_model(gap_end-3*i_line_type).values = [1e-3, 1e-6, 0, 0, 1, 1]

                if narrow_line:
                    xspec_model(gap_end - 1-3*i_line_type).values = [0] + \
                                                                    xspec_model(gap_end - 1-3*i_line_type).values[1:]
                    xspec_model(gap_end - 1-3*i_line_type).frozen = True


            #selecting the corresponding energy
            ener_line=lines_e_dict[elem_line_type][0]

            #selecting the energy for non broad (named) lines
            if named_line:
                xspec_model(gap_end-2-3*i_line_type).values=[ener_line]+\
                                                            xspec_model(gap_end-2-3*i_line_type).values[1:]

            else:
                #restricting energies for emission lines
                if elem_line_type in ['FeDiaz']:
                    xspec_model(gap_end-2-3*i_line_type).values=[ener_line,ener_line/100,6.0,6.0,8.0,8.0]
                else:
                    if 'cal' in elem_line_type:
                        xspec_model(gap_end - 2-3*i_line_type).values = [ener_line, ener_line / 100,
                                                                         ener_line, ener_line,
                                                           ener_line,
                                                           ener_line]

                        #freezing the energy
                        xspec_model(gap_end - 2-3*i_line_type).frozen=True
                    else:
                        #outputing the line values (delta of 1/100 of the line ener, no redshift, +0.4keV blueshift max)
                        xspec_model(gap_end-2-3*i_line_type).values=[ener_line,ener_line/100,
                                                                     ener_line-0.2,ener_line-0.2,
                                                                     ener_line+0.2,ener_line+0.2]


            #resticting the width of broad lines
            if broad_line:

                #widths changes
                width_line=lines_broad_w_dict[elem_line_type]

                #restricting widths of absorption lines and narrow emission lines
                xspec_model(gap_end-1-3*i_line_type).values=[width_line[0],
                                               (1e-3),
                                               width_line[1],width_line[1],
                                               width_line[2],width_line[2]]
            #and non-0 width lines
            elif not narrow_line:

                #widths changes
                width_line=lines_w_dict[elem_line_type]

                #restricting widths of absorption lines and narrow emission lines
                xspec_model(gap_end-1-3*i_line_type).values=[width_line[0],
                                               (1e-3),
                                               width_line[1],width_line[1],
                                               width_line[2],width_line[2]]

            #linking the widths if in a set
            if line_set and i_line_type!=n_line_comps-1:
                xspec_model(gap_end - 1 - 3 * i_line_type).link=str(gap_end - 1 - 3 * (n_line_comps-1))
            #for named physical lines we freeze the gaussian energy and use the vashift instead
            if named_line:

                #freezing the energy
                xspec_model(gap_end-2-3*i_line_type).frozen=1

                #unfreezing the vashift for the very first component in the list
                if i_line_type==len(line_comps_use)-1:
                    xspec_model(gap_end-3-3*i_line_type).frozen=0

                    #and forcing a specific range of blueshift/redshift depending on the line

                    #note : we differenciate absorption an narrow emission through the 'em' in the lines energy dictionnary
                    xspec_model(gap_end-3-3*i_line_type).values=[0,lines_e_dict[elem_line_type][2]/1e3,
                                                   -lines_e_dict[elem_line_type][2],
                                                   -lines_e_dict[elem_line_type][2],
                                                   -lines_e_dict[elem_line_type][1],
                                                   -lines_e_dict[elem_line_type][1]]

    '''laor specifics'''

    if comp_split=='laor' and comp_custom is not None:
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

    '''nthcomp specifics'''

    if comp_split.lower()=='nthcomp' and comp_custom is not None:
        if comp_custom=='disk':
            xspec_model(gap_end-4).frozen=True
            xspec_model(gap_end - 3).frozen = False

            disk_comp=[elem for elem in xspec_model.componentNames if 'diskbb' in elem][0]
            disk_par_id=str(getattr(xspec_model,disk_comp).Tin.index)
            xspec_model(gap_end-3).link=disk_par_id
            xspec_model(gap_end-2).values=1

        xspec_model(gap_end-5).values=[1.7,0.017,1.001,1.001,3.5,3.5]

    if comp_split=='thcomp' and comp_custom is not None:
        if comp_custom=='disk':
            xspec_model(gap_end-2).values=100.
            xspec_model(gap_end-2).frozen=True
            xspec_model(gap_end-3).values=[1.7, 0.017, 1.001, 1.001, 3.5,3.5]


    if 'abs' in comp_split and comp_split!='gabs' and comp_custom is not None and 'glob' in comp_custom:
        xspec_model(gap_end).values=[1.,0.01,0.,0.,100,100]

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
                ==line_type)[0][0]>np.argwhere(np.array(added_link_group)==comp.compname.split('_')[0])[0][0]:
                    #if we detect a component from the same group, its first parameter should be its vashift
                    xspec_model(gap_end-3).link='p'+str(comp.parlist[0])
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
    
    Note that this is still not perfect as no matter what, values are reset to their initial interval when being unlinked
    '''

    for i_grp in range(2,AllData.nGroups+1):
        xspec_model_grp=AllModels(i_grp,modName=mod_name)
        for i_par in range(gap_start,gap_end+1):

            xspec_model_grp(i_par).values=AllModels(1,mod_name)(i_par).values

            #updating the values break the link so we need to relink them afterwards
            xspec_model_grp(i_par).link=(mod_name+':' if mod_name!='' else '')+str(i_par)


    #creating the variable corresponding to the list of parameters
    return_pars=np.arange(gap_start,gap_end+1).astype(int).tolist()

    '''
    global constant factor specifics
    '''
    if compname=='glob_constant':
        for i_grp in range(1,AllData.nGroups+1):
            #setting the first data group to a fixed 1 value
            if i_grp==1:
                AllModels(i_grp,modName=mod_name)(gap_start).values=[1]+ AllModels(i_grp,modName=mod_name)(gap_start).values[1:]
                AllModels(i_grp,modName=mod_name)(gap_start).frozen=True
            #unlinking the rest
            else:
                AllModels(i_grp,modName=mod_name)(gap_start).link=''
                AllModels(i_grp,modName=mod_name)(gap_start).frozen=False
                AllModels(i_grp,modName=mod_name)(gap_start).values=[1]+ [0.01, 0.7,0.7,1.3,1.3]


        return_pars+=[gap_start+AllModels(1,mod_name).nParameters*i_grp for i_grp in range(1,AllData.nGroups)]

    if comp_split=='crabcorr' and comp_custom is not None:
        for i_grp in range(1,AllData.nGroups+1):
            #setting the first data group to a fixed 1 value
            if i_grp==1:
                AllModels(i_grp,modName=mod_name)(gap_start).values=[0]+ [0.01, -0.15,-0.15,0.15,0.15]
                AllModels(i_grp,modName=mod_name)(gap_start).frozen=True

                #note that we edit the values range to avoid negatives values that would make the fit go wild
                AllModels(i_grp,modName=mod_name)(gap_start+1).values=[1]+ [0.01, 0.85,0.85,1.15,1.15]
                AllModels(i_grp,modName=mod_name)(gap_start+1).frozen=True
            #unlinking the rest
            else:
                AllModels(i_grp,modName=mod_name)(gap_start).link=''
                AllModels(i_grp,modName=mod_name)(gap_start).frozen=False
                AllModels(i_grp,modName=mod_name)(gap_start).values=[0]+ [0.01, -0.15,-0.15,0.15,0.15]

                AllModels(i_grp,modName=mod_name)(gap_start+1).link=''
                AllModels(i_grp,modName=mod_name)(gap_start+1).frozen=False
                AllModels(i_grp,modName=mod_name)(gap_start + 1).values = [1] + [0.01, 0.7,0.7,1.3,1.3]
            if i_grp!=AllData.nGroups:
                return_pars+=[gap_start+AllModels(1,mod_name).nParameters*i_grp,gap_start+1+AllModels(gap_start).nParameters*i_grp]

        if comp_custom=='Suzaku':
            #freezing the gamma of every parameter except for the front illuminated one
            for i_grp in range(1, AllData.nGroups + 1):
                if 'xis_0' not in AllData(i_grp).fileName and 'xis_2' not in AllData(i_grp).fileName and\
                        'xis3' not in AllData(i_grp).fileName:
                    AllModels(i_grp,modName=mod_name)(gap_start).frozen=True

    '''
    From here onwards we need to test the telescopes of the datagroups
    '''

    group_sp=[]
    for i_grp in range(1, AllData.nGroups + 1):
        if "xis0" in AllData(i_grp).fileName:
            with fits.open(AllData(i_grp).fileName) as hdul:
                if hdul[1].header['TELESCOP']!='SUZAKU':
                    # replacing the merged fits file to not loose the headers
                    group_sp += [AllData(i_grp).fileName.replace('xis0_xis2_xis3', 'xis1') \
                        .replace('xis0_xis3', 'xis1')]
        else:
            group_sp += [AllData(i_grp).fileName]

    '''
    Calibration specifics
    Done after the relink since they affect the linking between components
    '''

    if comp_custom is not None:
        if 'cal' in comp_custom:
            #for the edges
            if comp_custom=='calNICER':
                xspec_model(gap_end-1).values=2.42
                xspec_model(gap_end-1).frozen=True
                xspec_model(gap_end).values=1e-2

                first_group_use=True
                #using the edge only for NICER datagroups, otherwise freezing the normalization at 0
                for i_grp in range(1,AllData.nGroups+1):
                    #skipping non-local files
                    if not os.path.isfile(group_sp[i_grp-1]):
                        continue
                    with fits.open(group_sp[i_grp-1]) as hdul:
                        if 'TELESCOP' not in hdul[1].header or hdul[1].header['TELESCOP']!='NICER':
                            AllModels(i_grp,modName=mod_name)(gap_end).link=''
                            AllModels(i_grp,modName=mod_name)(gap_end).values=0
                            AllModels(i_grp,modName=mod_name)(gap_end).frozen=True

                        else:
                            if first_group_use:
                                #allowing the normalization to vary freely
                                AllModels(i_grp,modName=mod_name)(gap_end).values = 1
                                AllModels(i_grp,modName=mod_name)(gap_end).link=''
                                AllModels(i_grp,modName=mod_name)(gap_end).frozen = False
                                par_tolink=(i_grp-1)*AllModels(1,mod_name).nParameters+gap_end
                                first_group_use=False

                                #adding to the list of variable parameters
                                return_pars += [par_tolink]

                            else:
                                AllModels(i_grp,modName=mod_name)(gap_end).link=str(par_tolink)

            if comp_custom=='calNuSTAR':
                xspec_model(gap_end-1).values=9.51
                xspec_model(gap_end-1).frozen=True

                first_group_use=True
                #using the edge only for NuSTAR datagroups, otherwise freezing the normalization at 0
                for i_grp in range(1,AllData.nGroups+1):
                    with fits.open(group_sp[i_grp-1]) as hdul:
                        if 'TELESCOP' not in hdul[1].header or hdul[1].header['TELESCOP']!='NuSTAR':
                            AllModels(i_grp,modName=mod_name)(gap_end).values=0
                            AllModels(i_grp,modName=mod_name)(gap_end).frozen=True
                        else:
                            if first_group_use:
                                #allowing the normalization to vary freely
                                AllModels(i_grp,modName=mod_name)(gap_end).values=1
                                AllModels(i_grp,modName=mod_name)(gap_end).link=''
                                AllModels(i_grp,modName=mod_name)(gap_end).frozen = False
                                par_tolink=(i_grp-1)*AllModels(1,mod_name).nParameters+gap_end
                                first_group_use=False

                                #adding to the list of variable parameters
                                return_pars += [par_tolink]

                            else:
                                AllModels(i_grp,modName=mod_name)(gap_end).link=str(par_tolink)

    '''Dust Scattering Halo model (xscat) specifics'''

    if comp_split == 'xscat':

        #linking the nH to the first absorption component found in the model
        for comp in AllModels(1,mod_name).componentNames:
            if is_abs(comp.split('_')[0]):
                abs_comp=getattr(AllModels(1,mod_name),comp)
                abs_comp_firstpar=getattr(abs_comp,abs_comp.parameterNames[0])
                AllModels(1,mod_name)(gap_end-3).link=str(abs_comp_firstpar.index)

        # only doing presets if added with a prefix
        if comp_custom is not None:
            for i_grp in range(1, AllData.nGroups + 1):
                with fits.open(group_sp[i_grp-1]) as hdul:
                    if 'TELESCOP' in hdul[1].header and hdul[1].header['TELESCOP'] == 'NICER':
                        AllModels(i_grp,modName=mod_name)(gap_end-1).values=180

                    #testing if the spectrum has an extraction region part in its fits
                    if len(hdul)>=4 and 'REG' in hdul[3].name:

                        print('region info detected in '+group_sp[i_grp-1])

                        #if yes, we fix the extraction radius to the one in the file using the infos stored
                        first_reg_lastrad=hdul[3].data.R[0][-1]
                        pix_to_arcsec=abs(hdul[3].columns['X'].coord_inc*3600)

                        print('freezing Rext of group '+str(i_grp)+' at')
                        print(str(round(first_reg_lastrad*pix_to_arcsec)))
                        AllModels(i_grp,modName=mod_name)(gap_end-1).values=round(first_reg_lastrad*pix_to_arcsec)

            #fixing the position if an object is specified
            if comp_custom.replace('glob','') in list(xscat_pos_dict.keys()):
                AllModels(1,mod_name)(gap_end-2).values=xscat_pos_dict[comp_custom.replace('glob','')]
                AllModels(1,mod_name)(gap_end-2).frozen=True


    '''ISM model specifics'''

    if comp_split== 'ismabs':
        if comp_custom is not None and 'link' in comp_custom:

            #removing the redshift
            AllModels(1,mod_name)(gap_end).values=0.
            AllModels(1,mod_name)(gap_end).frozen=True

            '''
            The default neutral column densities in ISMABS are all from the equivalent of the TBabs values 
            for NH=1e21
            We ca 
            #linking all the individual densities to the H density so that they can scale together
            '''

            for id_par in np.arange(3,31,3)-1:

                AllModels(1,mod_name)(int(gap_start+id_par)).link=str(AllModels(1,mod_name)(int(gap_start+id_par)).values[0])+'*p1/0.1'


    '''Physical model specifics'''

    if comp_split== 'relxillCp':
        if comp_custom  is not None:

            from visual_line_tools import incl_dyn_dict, incl_jet_dict, incl_misc_dict

            #fixing the inclination, preferentially to dynamical values, else to jet, else to misc
            if comp_custom in list(incl_dyn_dict.keys()):
                AllModels(1,mod_name)(gap_end-14).values=incl_dyn_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end-14).frozen = True

            elif comp_custom in list(incl_jet_dict.keys()):
                AllModels(1,mod_name)(gap_end-14).values = incl_jet_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end-14).frozen = True

            elif comp_custom in list(incl_misc_dict.keys()):
                AllModels(1,mod_name)(gap_end-14).values = incl_misc_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end-14).frozen = True

            #freezing the reflection fraction to -1 to only get the reflection component
            AllModels(1,mod_name)(gap_end-1).values=-1
            AllModels(1,mod_name)(gap_end - 1).frozen=True

            #free indexes but with restrained values
            AllModels(1,mod_name)(gap_end-8).values=[3.,0.1,1.,1.,5.,5.]
            AllModels(1,mod_name)(gap_end-9).values=[3.,0.1,1.,1.,5.,5.]


            #fixing Rin=R_ISCO, Rout=1000, Rbr=18Rg
            AllModels(1,mod_name)(gap_end-10).values=18
            AllModels(1,mod_name)(gap_end-11).values=1e3
            AllModels(1,mod_name)(gap_end-12).values=-1

    if comp_split == 'kerrbb':
        if comp_custom is not None:

            # allowing some wiggle room for the spectral hardening
            AllModels(1,mod_name)(gap_end - 3).values = [1.7, 0.1, 1.4, 1.4, 1.9, 1.9]

            #fixing the max spin to 0.998 on both sides to avoid issues when linking with relxill
            AllModels(1,mod_name)(gap_end-8).values=[0.0, 0.01, -0.998, -0.998, 0.998, 0.998]

            from visual_line_tools import incl_dyn_dict, incl_jet_dict, incl_misc_dict, dist_dict, mass_dict

            #fixing the inclination, preferentially to dynamical values, else to jet, else to misc
            if comp_custom in list(incl_dyn_dict.keys()):
                AllModels(1,mod_name)(gap_end-7).values=incl_dyn_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end - 7).frozen = True

            elif comp_custom in list(incl_jet_dict.keys()):
                AllModels(1,mod_name)(gap_end - 7).values = incl_jet_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end - 7).frozen = True

            elif comp_custom in list(incl_misc_dict.keys()):
                AllModels(1,mod_name)(gap_end - 7).values = incl_misc_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end - 7).frozen = True

            #mass
            if comp_custom in list(mass_dict.keys()):
                AllModels(1,mod_name)(gap_end - 6).values = mass_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end - 6).frozen = True


            #distance
            if comp_custom in list(dist_dict.keys()):
                AllModels(1,mod_name)(gap_end - 4).values = dist_dict[comp_custom][0]
                AllModels(1,mod_name)(gap_end - 4).frozen = True

            #freezing the norm if all 3 physical parameters are frozen
            if AllModels(1,mod_name)(gap_end-7).frozen and AllModels(1,mod_name)(gap_end-6).frozen and AllModels(1,mod_name)(gap_end-4).frozen:
                AllModels(1,mod_name)(gap_end).frozen=True

    AllModels.show()

    if return_pos:
        return return_pars,added_comps_numbers
    else:
        return AllModels(1,mod_name)

def delcomp(compname,mod_name='',give_ndel=False):

    '''
    changes the model to delete a component by saving the old model parameters and replacing it with the new one
    If values is set, it must be an array with one element for each parameter of the component.

    if multiple components have the same name, use the xspec name of the component you want to delete

    give_ndel returns the number of components that had to be deleted. Important for automatic fitting processes

    Note: Deleting atable and mult_table requires using the component names instead of mtable{*.fits}
    '''

    first_mod=AllModels(1,mod_name)

    model_saves = getattr(allmodel_data(), 'default' if mod_name == '' else mod_name)

    #deleting the space to avoid problems
    old_exp=model_saves[0].expression.replace(' ','')

    #separating the model expression when removing the component Name
    #this works even with multiple component with the same names because if it is the first one, find will get it first
    #(wouldn't work with split)


    #the easiest way to fetch the position of the component to delete is to transform the model expression according to xspec namings
    #note: no need to replace the * when deleting
    xspec_expr,table_dict=numbered_expression(old_exp,mult_conv=False,mod_name=mod_name)

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
            print('Last component in a multiplicative/convolutive component group removed.')

            #identifying the name of the previous component
            #In case of multiple deleted components, we use the last element in the list of deleted components
            del_asscomp=first_mod.componentNames[id_delcomp_list[-1]-1]

            print('Removing the associated component: '+del_asscomp)

            id_delcomp_list+=[id_delcomp_list[-1]-1]

            #removing the right parenthesis if it exists
            new_exp_aft=new_exp_aft[1:] if new_exp_aft[0]==')' and\
                        new_exp_bef[new_exp_bef.find(del_asscomp)+len(del_asscomp)]=='(' else new_exp_aft

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

            #separating different elements within the link here
            #tbd

            '''
            deleting all links that points to deleted parameters
            the euclidian division allows the test to work for the deleted parameters of all data groups
            the goal of the concatenate here is to add the zero to consider when the link points to the last parameter of the group if this parameter is among the deleted ones
            '''

            # fetching the link parameter inside the link expression
            link_par_str = [elem for elem in split_str_expr(link) if 'p' in elem]

            assert len(link_par_str) == 1, 'link with several parameters detected.' + \
                                                'Custom functions update required'

            link_par_str = link_par_str[0]

            # getting something that can be transformed back to an int
            link_par = link_par_str.split(':')[-1].replace('=', '').replace('p', '')

            #computing the deleted parameter ids in the first datagroup
            deleted_par_ids_single=np.concatenate((np.arange(skippar_start+1,skippar_start+skippar_n+1),
                                                    np.array([0]) if skippar_start+skippar_n==AllModels(1,mod_name).nParameters\
                                                    else np.array([]))).astype(int)

            if int(link_par)%AllModels(1,mod_name).nParameters in deleted_par_ids_single:
                print('\nParameter '+str(grp_id*AllModels(1,mod_name).nParameters+par_id+1)+
                      ' was linked to one of the deleted components parameters. Deleting link.')

                #Deleting links resets the values boundaries so we save and replace the values to make non-default bounds remain saved
                par_linked_vals=AllModels(grp_id+1,modName=mod_name)(par_id+1).values

                mod_data_grp.links[par_id]=''
                AllModels(grp_id+1,modName=mod_name)(par_id+1).values=par_linked_vals
                continue

            #shifting the link value if it points to a parameter originally after the deleted components
            #the 0 test accounts for the very last parameter, which will always need to be shifted
            # if it wasn't in the deleted comps
            elif int(link_par)>=skippar_start+skippar_n or int(link_par)%AllModels(1,mod_name).nParameters==0:

                #this value is the numer of times skippar_n parameters are deleted before the link_par parameter
                link_shift_factor=1+(int(link_par)-(skippar_start+skippar_n))//AllModels(1,mod_name).nParameters

                #new link position, correctly accounting for additional skips in subsequent datagroups
                new_link_par=str(int(link_par)-skippar_n*(link_shift_factor))

                mod_data_grp.links[par_id]=link.replace(link_par_str,link_par_str.replace(link_par,new_link_par))


        mod_data_grp.links[skippar_start:-skippar_n]=mod_data_grp.links[skippar_start+skippar_n:]
        mod_data_grp.links=mod_data_grp.links[:-skippar_n]

        mod_data_grp.frozen[skippar_start:-skippar_n]=mod_data_grp.frozen[skippar_start+skippar_n:]
        mod_data_grp.frozen=mod_data_grp.frozen[:-skippar_n]

    #now we can finally recreate the model
    model_saves[0].expression=new_exp_full

    new_models=model_load(model_saves,mod_name=mod_name,table_dict=table_dict)

    if give_ndel:
        return len(id_delcomp_list)
    else:
        return new_models


def par_error(group=1,par=1,n_round=3,latex=False,mult=1,man_val_arr=[],
              model_name=''):

    '''
    returns an array with the error of the chosen parameter

    still imperfect

    n_round chooses the rounding of the returned elements
    if set to auto, uses 1e-3 times the first digit as a rounding reference

    latex:
        if True, returns a formated string for latex

    mult:
        multiplies the quantities by a given amount before truncating and returning them

    if man_val_arr is not an empty array, uses that instead of fetching a parameter error
    '''


    if len(man_val_arr)!=0:
        val_arr=[man_val_arr[0],man_val_arr[0]-man_val_arr[1],man_val_arr[2]-man_val_arr[0]]
    else:
        val_arr=np.array([AllModels(group,model_name)(par).values[0],
              0 if AllModels(group,model_name)(par).error[0]==0 else (AllModels(group,model_name)(par).values[0] - AllModels(group,model_name)(par).error[0]),
                      0 if AllModels(group,model_name)(par).error[1] == 0 else
                      AllModels(group,model_name)(par).error[1] - AllModels(group,model_name)(par).values[0]])

    if mult!=1:
        val_arr*=mult

    if n_round is not None:

        result_val=np.repeat(np.nan,3)
        result_val[0]=('%.'+str(n_round)+'e')%(val_arr[0])
        result_val[1]=('%.'+str(n_round)+'e')%(result_val[0]-float(('%.'+str(n_round)+'e')%(val_arr[0]-val_arr[1])))
        #result_val[2]=('%.'+str(n_round)+'e')%(float(('%.'+str(n_round)+'e')%(val_arr[0]+val_arr[2]))-result_val[0])

        upper_decade_change=int(np.floor(np.log10(abs(val_arr[0] + val_arr[2]))) - np.floor(np.log10(abs(val_arr[0]))))

        result_val[2]=('%.'+str(n_round+upper_decade_change)+'e')%(float(('%.'+str(n_round+upper_decade_change)+'e')
                                                     %(val_arr[0]+val_arr[2]))-result_val[0])

    else:
        result_val=val_arr

    if latex:

        result_val_latex='$'+str(result_val[0])+(r"\pm"+' '+str(result_val[1]) if result_val[1]==result_val[2] else
                                        '_{-'+str(result_val[1])+'}^{+'+str(result_val[2])+'}')+'$'

        return result_val_latex
    else:
        return result_val

def display_mod_errors(n_round=2):
    '''
    displays the errors of every parameter in the model

    if n_round is set to auto, rounds automatically to 1e-3 * the first non zero
    digit of each parameter
    '''
    for i_grp in range(1,AllData.nGroups+1):
        #fetching the name of the component and parameters
        comp_par=[]
        for elem_comp in AllModels(i_grp).componentNames:
            comp_par+=(elem_comp+'.'+np.array(getattr(AllModels(i_grp),elem_comp).parameterNames,dtype=object)).tolist()

        n_max_char=max([len(elem) for elem in comp_par])

        for i_par in range(1,AllModels(i_grp).nParameters+1):
            par_error_arr=par_error(i_grp,i_par,n_round=n_round)

            '''
            here we align each part of the display according to different alignment and lengths to get a good display, 
            see https://www.geeksforgeeks.org/string-alignment-in-python-f-string/ 
            '''

            par_value_str=('%.'+str(n_round)+'e')%(par_error_arr[0])
            print_str=f"{i_grp :>3}  "+\
                      f"{i_par :>3}  "+\
                      f"{comp_par[i_par - 1]:<{n_max_char}}  "+\
                      f"{par_value_str:>{n_round+7}} "
                        #('%.' + str(n_round) + 'e') % (par_error_arr[0])

            if AllModels(i_grp)(i_par).link!='':
                print_str+='\t'+AllModels(i_grp)(i_par).link
            elif AllModels(i_grp)(i_par).frozen:
                print_str+='\tfrozen'
            else:
                print_str+='\t-'+('%.'+str(n_round)+'e')%(par_error_arr[1])+\
                           '\t+'+('%.'+str(n_round)+'e')%(par_error_arr[2])
            print(print_str)

        print('')

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

    for mod_name in list(AllModels.sources.values()):
        for i_grp in range(AllData.nGroups):
            freeze(AllModels(i_grp+1,mod_name))



def unfreeze(model=None,modclass=AllModels,parlist=None):

    '''
    just here to avoid calling arguments in freeze
    '''

    freeze(model=model,modclass=modclass,unfreeze=True,parlist=parlist)

def unlink(parlist=None,model=None,modclass=AllModels):

    if model is not None:
        xspec_mod=model
    else:
        xspec_mod=modclass(1)

    try:
        iterator = iter(parlist)
        parlist_use=parlist
    except:
        parlist_use=[parlist]

    for elem_par in parlist_use:

        xspec_mod(elem_par).link=''

def parse_xlog(log_lines,goal='lastmodel',no_display=False,replace_frozen=False,
               freeze_pegged=False,return_pars=False,error_bounds=False):

    '''
    Parses the Xspec log file to search for a specific information.
    Useful when xspec functions break before updating the model/Fit.

    rerun_pars:     if goal is 'lastmodel', instead of reloading the model, returns the
                    set of main values of all parameters (independant of what is currently loaded)

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

    error_bounds: for lasterrors, if True, retrieve parameter bounds within errors instead of relative error values

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

                # if line.startswith(' Due to zero model norms') and 'fit parameters are temporarily frozen' in line:
                #     frozen_pars=line.split(':')[1].replace('\n','')
                #     for elem_par_frozen in frozen_pars.split():
                #         if elem_par_frozen not in par_peg:
                #             par_peg+=[elem_par_frozen]

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

                            if error_bounds:
                                error_pars[parnum-1]=np.array(line.split()[1:3],dtype=float)
                            else:
                                #storing the errors
                                error_pars[parnum-1]=np.array(line.split('(')[1].split(')')[0].split(',')).astype(float)

                                #keeping the negative error positive for consistency
                                error_pars[parnum-1][0]=abs(error_pars[parnum-1][0])

        if freeze_pegged:

            #this is just here to be returned to know what has been pegged
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

            #in this case we will return an information because we need to know if something has been changed
            # in the error computations in some cases
            if len(par_peg)>0:
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

    models_par_list=[]

    #this will work even if there are other bars after the end of the model calls because the zip will cut the enumeration
    #at the number of indexes of the model starts
    for start_id,stop_id in zip(model_start_ids,model_stop_ids):
        model_lines_split+=[log_lines_lastfit[start_id+1:stop_id]]

    for model_lines in model_lines_split:

        #displaying the model
        if not return_pars:
            print('\nFound new model:'+model_lines[0].split('Model')[1].split('Source')[0])
        if not no_display and not return_pars:
            for line in model_lines:
                #printing the lines without adding a line break
                print(line[:-1])

        #identifying the model name to call it properly later
        if ':' not in model_lines[0].split('<1>')[0]:
            mod_name=''
        else:
            mod_name=model_lines[0].split(':')[0].split('Model ')[1]

        #splitting the lines for each group, by first identifying the starts
        if sum(['Data group' in elem for elem in model_lines])>0:
            multi_grp=True
        else:
            multi_grp=False

        if multi_grp:
            grp_lines_ids=np.argwhere([np.array(model_lines)==elem for elem in model_lines if 'Data group' in elem]).T[1]
        else:
            grp_lines_ids=[0]

        grp_lines_split=[]

        if not multi_grp:
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

        curr_mod_parlist=[]

        #loading each data group
        for group_lines in grp_lines_split:

            #fetching the group number
            if not multi_grp:
                i_grp=1
            else:
                i_grp=int(group_lines[0].split('Data group: ')[1].replace('\n',''))


            #fetching the lines with variations
            var_lines=[line for line in group_lines if '+/-' in line]

            if return_pars:
                mod_lines=[line for line in group_lines if '+/-' in line or '= p' in line or 'frozen' in line]

                #adding all of the main values to the current model list
                curr_mod_parlist+=[float(line.split()[-3] if '+/-' in line or '= p' in line else line.split()[-2])\
                                   for line in mod_lines]

                continue

            for line in var_lines:
                i_par=int(line.split()[0])%npars if int(line.split()[0])%npars!=0 else npars

                #note: there can be problems with paramater with very precise bounds from custom models,
                #so we adapt these bounds if necessary

                try:
                    par_values=AllModels(i_grp,modName=mod_name)(i_par).values
                except:
                    breakpoint()

                if float(line.split()[-3])<par_values[2]:
                    par_values[2]=float(line.split()[-3])
                if float(line.split()[-3])>par_values[5]:
                    par_values[5]=float(line.split()[-3])

                AllModels(i_grp,modName=mod_name)(i_par).values=[float(line.split()[-3])]+par_values[1:]

                #freezing lines frozen during the fitting operation if the option is selected
                if freeze_pegged and line.endswith('+/-  -1.00000     \n'):
                    AllModels(i_grp,modName=mod_name)(i_par).frozen=True

            #also replacing frozen values if it is asked
            if replace_frozen:
                frozen_lines=[line for line in group_lines if line.endswith('frozen\n')]
                for line in frozen_lines:
                    i_par=int(line.split()[0])%npars if int(line.split()[0])%npars!=0 else npars
                    AllModels(i_grp,modName=mod_name)(i_par).values=float(line.split()[-2])

        models_par_list+=[curr_mod_parlist]

    if goal=='lastmodel' and return_pars:
        return models_par_list

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
                print("\nLast fit iteration didn't improve the Stat significantly.Stopping the process...")
            fit_improve=False

    #changing back the fit parameters
    Fit.query=old_query_state
    if iterations is not None:
        Fit.nIterations=old_iterations


def calc_error(logfile,maxredchi=1e6,param='all',delchi_err='',timeout=60,delchi_thresh=0.1,indiv=True,
               give_errors=False,
               freeze_pegged=False):

    '''

    give_errors:
        False or True or "bounds"
        if set to bounds, retrieve the parameter ranges instead of errors(much more useful to identify pegged values)

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
        if type(glob_string_par)==str and '-' not in glob_string_par:
            error_strlist=[glob_string_par]
        else:
            print(glob_string_par)
            error_strlist=np.arange(int(glob_string_par.split('-')[0]),int(glob_string_par.split('-')[1])+1).astype(str).tolist()

    else:
        error_strlist=[glob_string_par]

    #defining the error function
    def error_func(string_par):
        try:
            #the fit returns an error when there is a new fit found so we pass the potential errors.
            Fit.error('max '+str(float(maxredchi))+' '+('' if delchi_err=='' else str(float(delchi_err)))+' '+string_par)
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

            #skipping useless parameters in indiv mode
            if indiv:
                par_group,par_id=par_degroup(int(error_str))
                if AllModels(par_group)(par_id).link!='' or AllModels(par_group)(par_id).frozen:
                    continue

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

            # if 'A valid fit is first required in order to run error command.\n' in log_lines:
            #     breakpoint()
            #     pass

            print('\nError computation finished.')

            if freeze_pegged:
                curr_par_peg_ids=parse_xlog(log_lines[-1],goal='lasterrors',freeze_pegged=freeze_pegged)

                #adding the pegged parameters to the list of pegged parameters at each error computation
                if curr_par_peg_ids!=[]:
                    par_peg_ids+=curr_par_peg_ids

                #adding the new lines since since if something has been pegged we will have a new model displayed
                #which needs to be loaded to save this peg
                log_lines[-1]+=logfile.readlines()

            elif give_errors!=False:


                new_errors=parse_xlog(log_lines[-1],goal='lasterrors',error_bounds=give_errors=='bounds')

                #in indiv mode we only update the value of the parameter for which the error was just computed
                if indiv:

                    par_group,par_number=par_degroup(int(error_str))
                    error_pars[par_group-1][par_number-1]=new_errors[par_group-1][par_number-1]
                else:
                    error_pars=new_errors

            print('\nParsing the logs to see if a new minimum was found...')

            #searching for the model
            is_newmodel=parse_xlog(log_lines[-1],goal='lastmodel',no_display=True)

            if is_newmodel:
                #recreating a valid fit
                calc_fit(logfile=logfile,nonew=True)

                if Xset.chatter>5:

                    print('\nResulting model after loading...\n')
                    #displaying the new model without changing the console chatter state
                    curr_chatter=Xset.chatter
                    Xset.chatter=10
                    AllModels.show()
                    Fit.show()
                    Xset.chatter=curr_chatter
                    #reading the newly created lines we don't care about
                    logfile.readlines()

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

    fixed_gamma can be a value to freeze Gamma/Gamma_tau for the nthcomp/thcomp component

    thcomp_frac_frozen will freeze the thcomp component to 0 contribution and 3.5 gamma

    mandatory abs determines whether the absorption components will be forced to be mandatory
    '''

    def __init__(self,complist,logfile,logfile_write,absval=None,interact_groups=None,idlist=None,prev_fitmod=None,
                 fixed_gamma=None,sat_list=None,thcomp_frac_frozen=False,mandatory_abs=False):

        #defining empty variables
        self.name_complist=complist
        self.includedlist=[]
        self.logfile=logfile
        self.logfile_write=logfile_write
        self.sat_list=sat_list
        self.idlist=idlist if idlist!=None else np.array([None]*len(complist))
        self.interact_groups=interact_groups
        self.complist=[]
        self.fixed_abs=absval
        self.fixed_gamma=fixed_gamma
        self.thcomp_frac_frozen=thcomp_frac_frozen
        self.mandatory_abs=mandatory_abs

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

                    setattr(self,'cont_'+compname,make_fitcomp('cont_'+compname,self.logfile,self.logfile_write,
                                                          continuum=True,fitcomp_names=self.name_complist,
                                                          fitmod=self))
                    self.cont_complist+=[getattr(self,'cont_'+compname)]
                    self.includedlist+=[getattr(self,'cont_'+compname)]
                    self.name_cont_complist+=['cont_'+compname]
                    self.complist+=[getattr(self,'cont_'+compname)]


        #linking an attribute to each individual fitcomp and adding them to a list for convenience

        for i in range(len(self.name_complist)):


            #adding the continuum components not already in the model to the list of components to be added if there aren't in the
            #current continuum
            #here we assume there are no numbered continuum components (should be edited for more complex continuums)
            #note : we keep this code part but it is not used anymore as long as we keep parenthood when creating new fitmods
            if self.name_complist[i].startswith('cont_'):

                #components already considered in the autofit continuum list should not be here twice
                if self.name_complist[i] not in self.name_cont_complist:

                    setattr(self,self.name_complist[i],make_fitcomp(self.name_complist[i],self.logfile,self.logfile_write,
                                                               self.idlist[i],fitcomp_names=self.name_complist,
                                                               fitmod=self))

                    self.complist+=[getattr(self,self.name_complist[i])]
            else:
                setattr(self,self.name_complist[i],make_fitcomp(self.name_complist[i],
                                                           self.logfile,self.logfile_write,
                                                           self.idlist[i],
                                                           fitcomp_names=self.name_complist,
                                                           fitmod=self))
                new_comp=getattr(self,self.name_complist[i])
                if new_comp.absorption and mandatory_abs:
                    new_comp.mandatory=True

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

            #note:for now we don't consider the parameter list in subsequent datagroups except for glob_constant
            first_par=getattr(comp.xcomps[0],comp.xcomps[0].parameterNames[0]).index
            last_par=getattr(comp.xcomps[-1],comp.xcomps[-1].parameterNames[-1]).index
            comp.parlist=np.arange(first_par,last_par+1).astype(int).tolist()

            #adding some parameters for components which affect other datagroups
            if ('cal' in comp.compname and 'edge' in comp.compname)\
                    or np.any([elem in comp.compname for elem in xspec_globcomps]) :
                comp.parlist=ravel_ragged([np.array(comp.parlist)+AllModels(1).nParameters*i_grp\
                                           for i_grp in range(AllData.nGroups)]).tolist()

            comp.unlocked_pars=[i for i in comp.parlist if (not AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen
                                                        and AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='')]


        #we also update the list of component xcomps (no multi-components here so w can take the first element in xcompnames safely)
        self.cont_xcompnames=[self.cont_complist[i].xcompnames[0] if self.cont_complist[i].included else ''\
                              for i in range(len(self.cont_complist))]

        #this should be done for all components and not just the included ones
        for comp in self.complist:
            # safeguard to avoid issues for continuum complists imported in the autofit
            comp.fitmod = self

            # updating the logfile
            comp.logfile = self.logfile
            comp.logfile_write = self.logfile_write

        #making our life easier for when we don't need to know about the secondary components
        self.includedlist_main=[elem for elem in self.includedlist if elem is not None]

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

    def add_allcomps(self, chain=False, lock_lines=False, no_abslines=False, nomixline=True, split_fit=True,
                     fit_scorpeon=False,fit_SAA_norm=False,no_fit=False):

        '''
        Adds all components from the excl_list in the order of the list

        if nomixline is set to True,
        the emission/absorption line components of the same type are mutually exclusive (aka including one deletes the other)

        '''

        # list of currently NOT included components (doesn't take into account continuum components not in the model list)
        curr_exclist = [elem for elem in self.complist if elem not in self.includedlist]

        if len(curr_exclist) == 0:
            return True

        self.print_xlog('\nlog:Adding all available components:\n')
        self.print_xlog(str([curr_exclist[i].compname for i in range(len(curr_exclist))]))

        for i_excomp, component in enumerate(curr_exclist):

            # avoiding starting the model with a multiplicative component
            assert not (component.multipl and len(self.includedlist) == 0),\
                'First component of the mod list is multipl. Order change required.'


            # skipping adding lines in lines locked mode
            if component.line and lock_lines:
                continue

            # skipping adding absorption lines in no abslines mode
            if component.named_absline and no_abslines:
                continue

            #skipping adding calibration components before a first fit to avoid pegged issues
            if component.calibration:
                continue

            # excluding addition of named components with pendants when not allowing mixing between emission and absorption of the same line
            if component.named_line and nomixline:

                # fetching the name of the pendant
                if component.named_absline:
                    line_pendant_prefix = component.compname.split('_')[0].replace('abs', 'em')
                else:
                    line_pendant_prefix = component.compname.split('_')[0].replace('em', 'abs')

                # fetching the line pendant component
                line_pendant_name_list = [elem for elem in self.name_complist if line_pendant_prefix in elem]

                if len(line_pendant_name_list) != 0:
                    if len(line_pendant_name_list) != 1:
                        # should not happen
                        breakpoint()

                    line_pendant_name = line_pendant_name_list[0]

                    line_pendant_comp = getattr(self, line_pendant_name)

                    # skipping the component addition if the pendant is already included
                    if line_pendant_comp.included:
                        continue

            #there can be issues when adding crabcorr with datagroups who have very different normalization so
            #in this case we add a glob_constant first, log the values and then add the crabcorr with the
            #normalization values of the glob_constant
            if 'crabcorr' in component.compname:
                addcomp('glob_constant')
                calc_fit()
                norm_vals=[AllModels(i_grp+1)(1).values[0] for i_grp in range(AllData.nGroups)]
                delcomp('constant')

            fixed_vals=None

            if component.absorption:
                fixed_vals=[self.fixed_abs]
            elif "nthcomp" in component.compname or "thcomp" in component.compname:
                fixed_vals=[self.fixed_gamma]

            if component.compname=="disk_thcomp" and self.thcomp_frac_frozen:
                #fixing the gamma and the covering fraction
                fixed_vals=[3.5,None,0.,None]

            self.includedlist = component.addtomod(fixed_vals=fixed_vals,incl_list=self.includedlist)

            # updating the fitcomps before anything else
            self.update_fitcomps()

            if 'crabcorr' in component.compname:
                #pre-adjusting the crabcorr normalization values
                for i_grp in range(AllData.nGroups):
                    AllModels(i_grp+1)(2).values=norm_vals[i_grp]

            #for now we skip the individual component fittings to avoid settling into a local minima before all components
            #are now (can be an issue). Could add an option for this
            if i_excomp!=len(curr_exclist)-1:
                continue

            # fitting the component (no split fit here, not needed)
            component.fit(split_fit=False)

            # fetching the interactive components from the list if it is provided
            if self.interact_groups is not None:
                # searching the component in the intervals
                for group in self.interact_groups:
                    if component.compname in group:
                        comp_group = np.array(group)[np.array(group) != component.compname]
                        break

            '''
            unfreezing the components previously added and fitting once more every time
            we unfreeze them in reverse model order (i.e. ~increasing significance)
            '''

            if split_fit:
                # list of components not in the same group as the currently tested component
                comps_errlocked = self.list_comps_errlocked()

                for i_comp, added_comp in enumerate(self.includedlist[::-1]):

                    # skipping placeholders
                    if added_comp is None:
                        continue

                    # if we arrived at the first (last) component of the initial continuum
                    if added_comp in self.cont_complist:
                        self.print_xlog('\nlog:Unfreezing the continuum')

                        # we unfreeze all the remaining components (all from the continuum) and test them for absorption components
                        # which we error_lock
                        for cont_comp in self.includedlist[::-1][i_comp:]:
                            # here we do check for placeholders directly in the test since i_comp can factor in some placeholders
                            if cont_comp is not None and (not cont_comp.line or not lock_lines):
                                cont_comp.unfreeze()

                            # unfreezing constant factors for all but the first datagroup
                            if AllModels(1).componentNames[0] in xspec_globcomps and AllData.nGroups > 1:
                                if AllModels(1).componentNames[0]=='constant':
                                    for i_grp in range(2, AllData.nGroups + 1):
                                        AllModels(i_grp)(1).frozen = False

                                elif AllModels(1).componentNames[0]=='crabcorr':
                                    for i_grp in range(2, AllData.nGroups + 1):
                                        AllModels(i_grp)(2).frozen = False

                    elif not added_comp.line or not lock_lines:
                        self.print_xlog('\nlog:Unfreezing ' + added_comp.compname + ' component.')
                        added_comp.unfreeze()

                    '''
                    Note: In rare cases, the fit can get stuck and increase the fit value when decreasing components (probably a bug)
                    To avoid this, we store the model and the chi² value at each unfreeze and reload the model at the end of each unfreezing
                    if the fit value has not improved
                    '''

                    # storing the model and fit
                    pre_unfreeze_mod = allmodel_data()
                    pre_unfreeze_fit = Fit.statistic

                    # fitting with the new freeze state
                    calc_fit(logfile=self.logfile if chain else None)

                    # this is only for non continuum components
                    if self.interact_groups is not None:

                        if added_comp.compname not in comp_group:
                            self.print_xlog(
                                '\nlog:Component ' + added_comp.compname + ' not in group of component ' + component.compname + '.\n' +
                                'adding it to the error-locked list before interaction to avoid bogging the fit down...')
                            comps_errlocked += [added_comp]
                        else:
                            self.print_xlog(
                                '\nlog:Component ' + added_comp.compname + ' in the same group as component ' + component.compname + '.\n')

                    # freezing the error locked components before the error computation
                    for elem_comp in comps_errlocked:
                        elem_comp.freeze()

                    # if we reached the continuum step, we compute the errors starting at parameter 1 (i.e. all parameters)
                    if added_comp.continuum:
                        error_str = '1-' + str(AllModels(1).nParameters * AllData.nGroups)

                    # in the other case, we start at the current component's first parameter index
                    else:
                        error_str = str(added_comp.parlist[0]) + '-' + str(AllModels(1).nParameters * AllData.nGroups)

                    calc_error(self.logfile, param=error_str, indiv=True)

                    # unfreezing everything back
                    for elem_comp in comps_errlocked:
                        elem_comp.unfreeze()

                    # reloading the previous model iteration (aka not doing the fit with this combination of unfrozen components)
                    if Fit.statistic > pre_unfreeze_fit:
                        self.print_xlog('\nlog:Unfreezing ' + added_comp.compname + ' component worsens the fit.' + \
                                        '\n Restoring previous iteration for next component.')
                        pre_unfreeze_mod.load()

                    # we can stop the loop after the continuum unfreezing iteration
                    if added_comp in self.cont_complist:
                        break

            # storing the final fit in the component's save
            component.fitted_mod = allmodel_data()

        self.update_fitcomps()

        if no_fit:
            return


        calc_fit()

        #fitting the scorpeon model if asked to
        if fit_scorpeon and 'NICER' in self.sat_list:
            # unfreezing the scorpeon model by resetting it
            xscorpeon.load('auto',frozen=False,extend_SAA_norm=True,fit_SAA_norm=fit_SAA_norm)

            # computing a fit with the scorpeon model on
            calc_fit()

        #adding calibration components
        for i_excomp,component in enumerate(curr_exclist):

            if not component.calibration:
                continue

            #only adding the calibration components if their relevant energies is inside what is currently noticed
            Plot('ldata')
            ener_bounds=[Plot.x()[0],Plot.x()[-1]]

            #here we add a margin of 1 keV on each side to avoid having components at the very beginning/end
            # of the model, which would be hard to constrain
            if not component.cal_e-1>=ener_bounds[0] and component.cal_e+1<=ener_bounds[1]:
                continue

            fixed_vals=None

            if component.absorption:
                fixed_vals=[self.fixed_abs]
            elif "nthcomp" in component.compname or "thcomp" in component.compname:
                fixed_vals=[self.fixed_gamma]
            if component.compname=="disk_thcomp" and self.thcomp_frac_frozen:
                #fixing the gamma and the covering fraction
                fixed_vals=[3.5,None,0.,None]

            self.includedlist = component.addtomod(fixed_vals=fixed_vals,incl_list=self.includedlist)
            self.update_fitcomps()

            component.fit(split_fit=False,compute_errors=False,fit_to=120)

            component.fitted_mod=allmodel_data()

        #refreezing the scorpeon model
        xscorpeon.freeze()
        self.update_fitcomps()


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

            #only adding the calibration components if their relevant energies is inside what is currently noticed
            if component.calibration:
                Plot('ldata')
                ener_bounds=[Plot.x()[0],Plot.x()[-1]]
                if not component.cal_e>=ener_bounds[0] and component.cal_e<=ener_bounds[1]:
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

            try:
                #we do this to avoid issues when having a starting bg model such as scorpeon which gives values\
                # of chi and dof
                AllModels(1)
                init_chi=Fit.statistic
                init_dof=Fit.dof
            except:
                init_chi=0
                init_dof=0

            #copy of the includedlist for rollback after testing the component significance
            prev_includedlist=copy(self.includedlist)

            fixed_vals = None

            if component.absorption:
                fixed_vals=[self.fixed_abs]
            elif "nthcomp" in component.compname or "thcomp" in component.compname:
                fixed_vals=[self.fixed_gamma]
            if component.compname=="disk_thcomp" and self.thcomp_frac_frozen:
                #fixing the gamma and the covering fraction
                fixed_vals=[3.5,None,0.,None]

            self.includedlist=component.addtomod(fixed_vals=fixed_vals,incl_list=self.includedlist)

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

                            # unfreezing constant factors for all but the first datagroup
                            if AllModels(1).componentNames[0] in xspec_globcomps and AllData.nGroups > 1:
                                if AllModels(1).componentNames[0] == 'constant':
                                    for i_grp in range(2, AllData.nGroups + 1):
                                        AllModels(i_grp)(1).frozen = False

                                elif AllModels(1).componentNames[0] == 'crabcorr':
                                    for i_grp in range(2, AllData.nGroups + 1):
                                        AllModels(i_grp)(2).frozen = False
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

            self.print_xlog('\nlog:Stat before adding the component:'+str(init_chi))
            self.print_xlog('\nlog:Stat after adding the component:'+str(new_chi))
            #testing the 99% significance of the comp with the associated number of d.o.f. added
            #i.e. the number of parameters - the number of parameters we keep frozen
            self.print_xlog('\nlog:Delta Stat for this component: '+str(init_chi-new_chi))

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
            bestcomp=np.array(curr_exclist)[custom_ftest_mask][bestcomp_in_custom_id]

            #previous version that didn't work
            #bestcomp=curr_exclist[np.argwhere(np.array(custom_ftest_mask))[0][bestcomp_in_custom_id]]

        else:
            bestcomp=np.array(curr_exclist)[component_ftest==min(component_ftest[component_ftest.nonzero()])][0]

        self.print_xlog('\nlog:The most significant component is '+bestcomp.compname+'. Adding it to the current model...')

        #re-adding it with its fit already loaded
        self.includedlist=bestcomp.addtomod(fixed_vals=[self.fixed_abs] if component.absorption else \
                                                              [self.fixed_gamma] if 'nthcomp' in component.compname.lower()\
                                                              else None,incl_list=self.includedlist,fitted=True)

        #updating the fitcomps before anything else
        self.update_fitcomps()

        return False

    def remove_comp(self,component):

        # here we don't use an enumerate in the loop itself since the positions can be modified when components are deleted
        i_comp = np.argwhere(np.array(self.includedlist) == component)[0][0]

        # deleting the component and storing how many components were deleted in the process
        n_delcomp = component.delfrommod(rollback=False)

        # updating the current includedlist to delete as many components as what was deleted in xspec
        self.includedlist = self.includedlist[:i_comp + 1 - n_delcomp] + self.includedlist[i_comp + 1:]
        # this time we need to update
        self.update_fitcomps()

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

            #storing the previous includedlist to come back to it at the end of the loop iteration
            prev_includedlist=self.includedlist

            self.remove_comp(component)

            #refitting and recomputing the errors with everything free
            calc_fit(logfile=self.logfile if chain else None)

            del_chi=Fit.statistic

            #restricting the test to components which are not 'very' significant
            #we fix the limit to 10 times the delchi for the significance threshold with their corresponding number of parameters
            try:
                ftest_val=Fit.ftest(new_chi,new_dof,del_chi,new_dof+n_unlocked_pars_with_unlink)
            except:
                breakpoint()

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

    def idtocomp(self,par_ids):

        '''
        transforms a list of group/id parameter couple into a list of component/index of the parameter in the list of that comp

        allows to keep track of parameters even after modifying components

        #here we assume custom models are not gonna be modified and thus return a different element for them (None)

        NOTE: when used with par_peg_ids, par_ids is a 2 dimensionnal, len 3 list with the group, parameter and mod id
        In this case, we re-convert into the actual parlist because the parlist of each fitcomp is directly
        the full parameter number

        NOTE: will skip all parameters not in the main model (e.g. scorpeon bg)
        as they are not linked to a specific comp
        '''

        #if
        if np.ndim(par_ids)==2:

            mod_npars=AllModels(1).nParameters

            parlist=[mod_npars*(group_val-1)+par_val for group_val,par_val,mod_name in par_ids if mod_name=='']
        else:
            parlist=par_ids

        includedcomps=np.array([comp for comp in self.includedlist if comp is not None])

        parlist_included=np.array([elem.parlist for elem in includedcomps],dtype=object)

        #finding the corresponding component in each case
        mask_comp_match=[[par_number in comp_parlist for comp_parlist in parlist_included]\
                          for par_number in parlist]

        #their id
        id_comp_match=[np.argwhere(elem)[0][0] for elem in mask_comp_match]

        parlist_comps=[[includedcomps[id_comp_match[i_par]],
                        np.argwhere(np.array(parlist_included[id_comp_match[i_par]])==parlist[i_par])[0][0]]\
                       for i_par,mask in enumerate(mask_comp_match)]

        return parlist_comps

    def global_fit(self,chain=False,directory=None,observ_id=None,freeze_final_pegged=True,no_abslines=False,lock_lines=False,nomixline=True,split_fit=True,
                   ftest_threshold=def_ftest_threshold,ftest_leeway=def_ftest_leeway,
                   method='opt',fit_scorpeon=False,fit_SAA_norm=False):

        '''
        Fits components in diffrerent manners:
         -'force_all'   : forces the inclusion of all available component in the order of the list
         -'force_add'   : forces the addition of all available component in the order of the list, then
                          allows to delete them if they are not significant
         -'opt'         : progressively in order of fit significance when being added 1 by 1

        -fit_scorpeon: fits the scorpeon model when NICER is part of the loaded satellites (with sat_list)
            in force_all and force_all modes,will fit before adding calibration components
            in opt mode (normally done after continuum fits) will fit it AFTER the component addition test

        in opt mode, the scorpeon fit

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

        if method in ['force_all','force_add']:
            self.add_allcomps(chain=chain,lock_lines=lock_lines,no_abslines=no_abslines,nomixline=nomixline,
                              split_fit=False,fit_scorpeon=fit_scorpeon,fit_SAA_norm=fit_SAA_norm)
        elif method=='opt':
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

            # fitting the scorpeon model if asked to
            if fit_scorpeon and 'NICER' in self.sat_list:

                curr_mod=allmodel_data()

                curr_scorpeon=curr_mod.scorpeon

                # unfreezing the scorpeon model by resetting it (here considering previous saves but with the freeze
                # state of the first auto load)
                xscorpeon.load('auto',frozen=False, scorpeon_save=curr_scorpeon,extend_SAA_norm=True,
                               fit_SAA_norm=fit_SAA_norm,
                               load_save_freeze=False)

                calc_fit()

        '''Checking if unlinking the energies of each absorption line is statistically significant'''

        #storing the chi value at this stage
        chi_pre_unlink=Fit.statistic
        dof_pre_unlink=Fit.dof

        if not lock_lines:
            self.test_unlink_lines(chain=chain,ftest_threshold=ftest_threshold)

        #resetting the edges in case they were pegged at 0 with a previous fit (noticeable with very low absorption values)
        #for now only considers one single absorption value per edge (aka linked values between datagroups)
        for elem_comp in [elem for elem in self.includedlist if elem is not None]:
            if not 'edge' in elem_comp.compname:
                continue

            if len(elem_comp.unlocked_pars)>0:
                #the first unlocked can be for different datagroups so we do it like that
                if AllModels(par_degroup(elem_comp.unlocked_pars[0])[0])(par_degroup(elem_comp.unlocked_pars[0])[1]).values[0]<1e-4:
                    #1e-2 is too high but should allow to refit correctly
                    AllModels(par_degroup(elem_comp.unlocked_pars[0])[0])(par_degroup(elem_comp.unlocked_pars[0])[1]).values=1e-2

        #new fit with the updated edges
        calc_fit(logfile=self.logfile if chain else None)

        #testing if freezing the pegged parameters improves the fit
        par_peg_ids=calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),
                               freeze_pegged=freeze_final_pegged,indiv=True)

        #computing the component position of the frozen parameter to allow to unfreeze them later even with modified component positions
        par_peg_comps=self.idtocomp(par_peg_ids)

        if len(par_peg_ids)!=0:

            #new fit with the new frozen state
            calc_fit(logfile=self.logfile if chain else None)
            calc_error(self.logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),indiv=True)

        if method in ['opt','force_add']:
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

                    #defining the current pegged_par total index with the new configuration
                    pegged_par_index=par_peg_comps[i_par_peg][0].parlist[par_peg_comps[i_par_peg][1]]

                    #note that here we are not directly re-using the par_peg_id values because the position
                    #of the parameters could have been modified since with the new round of deletion
                    pegged_par_group,pegged_par_number=par_degroup(pegged_par_index)

                    #unfreezing the parameter
                    try:

                        AllModels(pegged_par_group)\
                            (pegged_par_number).frozen=False

                    except:
                        print('THIS SHOULDNT HAPPEN')
                        Xset.chatter=10
                        AllModels.show()
                        print('Saving to test_bp_'+str(time.time())+'.xcm')
                        Xset.save('test_bp_'+str(time.time())+'.xcm')

                        save={"par_peg_ids":par_peg_ids,
                              "i_par_peg":i_par_peg,
                              "pegged_par_index":pegged_par_index,
                              "mod1_npars":AllModels(1).nParameters}
                        with open('bp_dump_'+str(time.time())+'.dill','wb') as f:
                            dill.dump(save,f)

                        breakpoint()

                    # #computing the parameter position in all groups values
                    # par_peg_allgrp=(par_peg_ids[i_par_peg][0]-1)*AllModels(1).nParameters+par_peg_ids[i_par_peg][1]

                        #no need for indiv mode here since we compute the error for a single parameter
                    calc_error(self.logfile,param=str(pegged_par_index),freeze_pegged=True)

                    #re-freezing the parameter
                    #AllModels(par_peg_ids[i_par_peg][0])(par_peg_ids[i_par_peg][1]).frozen=False
                    #we don't do this currently since there is a freeze peg just after in the chain
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

                if method=='opt':
                    self.test_delcomp(chain,lock_lines,ftest_threshold=ftest_threshold,ftest_leeway=ftest_leeway)

                Fit.perform()

            #updating the fitcomps in case a sneaky model freeze happens due to a parameter being frozen automatically by xspec
            #(which wouldn't be considered in the current comp.unlocked_pars)

            self.update_fitcomps()

            #computing the number of free parameters (need to be done after a fit) as a baseline for the MC
            n_free_pars=sum([len(comp.unlocked_pars) for comp in [elem for elem in self.includedlist if elem is not None]])

            #computing the markov chain here since we will need it later anyway, it will allow better error definitions
            # AllChains.defLength=4000*n_free_pars
            # AllChains.defBurn=2000*n_free_pars
            # AllChains.defWalkers=2*n_free_pars

            AllChains.defLength=10000
            AllChains.defBurn=5000
            AllChains.defWalkers=2*n_free_pars

            #ensuring we recreate the chains
            if os.path.exists(directory+'/'+observ_id+'_chain_autofit.fits'):
                os.remove(directory+'/'+observ_id+'_chain_autofit.fits')

            self.print_xlog('\nlog:Creating Markov Chain from the fit with '+str(n_free_pars)+' free parameters...')

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

        self.print_xlog('\nlog:Global fit complete.')

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


    def get_absline_width(self,err_sigma=3):

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
            abslines_width[i_line]=fitcomp_line.get_width(err_sigma=err_sigma)

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
        abslines_ew=np.zeros((n_abslines,3))
        abslines_bshift=np.array([[None]*3]*n_abslines)
        abslines_bshift_distinct=np.array([None]*n_abslines)
        abslines_delchi=np.zeros(n_abslines)

        if len(abs_lines)==0:
            return abslines_flux,abslines_ew,abslines_bshift,abslines_delchi,abslines_bshift_distinct

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
            abslines_bshift_distinct[i_line]=get_overlap(line_bshift_inter,Ka_bshift_inter)==0

        for i_line,line in enumerate(abs_lines):

            fitcomp_line=getattr(self,line)

            #storing the ew (or the upper limit if there is no line)
            abslines_ew[i_line]=fitcomp_line.get_ew()

        #second loop since we modify the fit here and that affects the ew computation
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
                vashift_parlink=int(AllModels(1)(vashift_parid).link.replace('=',''))
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
                    [-AllModels(1)(vashift_parid).values[0],
                     self.errors[0][vashift_parid-1][1],
                     self.errors[0][vashift_parid-1][0]]


            #getting the delchi of the line (we assume that the normalisation is the last parameter of the line component)

            #deleting the component
            delcomp(fitcomp_line.xcompnames[-1])

            #fitting and computing errors
            #note: might need to be re-adjusted back to with a logfile if the fit gets stuck here
            calc_fit()

            #calc_fit(logfile=self.logfile)

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
            for comp in self.includedlist_main[::-1]:

                #skipping the current component, already removed absorptions
                if comp is fitcomp_line or (comp.absorption and comp.xcompnames[0] not in AllModels(1).componentNames):
                    continue

                #skipping multiplicative non-calibration components, which will be deleted anyway with the additive ones
                if not comp.calibration and comp.multipl and not comp.absorption\
                        and not comp.xcompnames[0] in xspec_globcomps:
                    continue

                print(comp.compname)

                comp.delfrommod(rollback=False)
                #fix this

            #loop on the parameters
            flux_line_dist=np.zeros(len(par_draw))
            for i_sim in range(len(par_draw)):
                #setting the parameters of the line according to the drawpar
                for id_parcomp,parcomp in enumerate(fitcomp_line.parlist):

                    #we only draw from the first data group BECAUSE WE TAKE THE FLUX FOR THE FIRST DG,
                    # hence the [0] index in par_draw
                    AllModels(1)(id_parcomp+1).values=[par_draw[i_sim][0][parcomp-1]]+AllModels(1)(id_parcomp+1).values[1:]


                #computing the flux of the line
                #(should be negligbly affected by the energy limits of instruments since we don't have lines close to these anyway)
                '''
                note that this could be an issue for lower energy lines if a high-energy telescope is loaded first
                there could be a need to extend its rmf
                '''
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

        return abslines_flux,abslines_ew,abslines_bshift,abslines_delchi,abslines_bshift_distinct

    def get_emline_info(self, par_draw, percentile=90):

        '''

        TO BE TESTED
        Computes and returns various informations on the absorption lines:
            -line flux,
            -equivalent width with errors, computed from the chain if a chain is loaded
            -blueshift with errors from the currently stored self.errors (i.e. from a chain if a chain was used to compute them)
            -energies and associated errors from the blueshift values
        '''

        mod_data_init = allmodel_data()

        em_lines = [elem for elem in self.name_complist if '_gaussian' in elem]

        # using 6 emlines as default when there are no emlines used to go faster
        if len(em_lines) == 0:
            n_emlines = 6
        else:
            n_emlines = len(em_lines)

        emlines_flux = np.zeros((n_emlines, 3))
        emlines_ew = np.zeros((n_emlines, 3))
        emlines_bshift = np.array([[None] * 3] * n_emlines)
        emlines_delchi = np.zeros(n_emlines)

        if len(em_lines) == 0:
            return emlines_flux, emlines_ew, emlines_bshift, emlines_delchi

        base_chi = Fit.statistic

        Fit.perform()


        for i_line, line in enumerate(em_lines):
            fitcomp_line = getattr(self, line)

            # storing the ew (or the upper limit if there is no line)
            emlines_ew[i_line] = fitcomp_line.get_ew()

        # second loop since we modify the fit here and that affects the ew computation
        for i_line, line in enumerate(em_lines):

            fitcomp_line = getattr(self, line)

            # skipping if the line is not in the final model
            if not fitcomp_line.included:
                continue

            # fetching the xspec vashift parameter number of this line
            vashift_parid = fitcomp_line.parlist[0]

            # if the parameter is linked to something we note it instead of fetching the uncertainty:
            if AllModels(1)(vashift_parid).link != '':
                # identifying the parameter it is linked to
                vashift_parlink = int(AllModels(1)(vashift_parid).link.replace('=', ''))
                # fetching the name of the component associated to this parameter
                for comp in [elem for elem in self.includedlist if elem is not None]:
                    if vashift_parlink in comp.parlist:
                        linked_compname = comp.compname.split('_')[0]
                        break
                emlines_bshift[i_line] = [-AllModels(1)(vashift_parid).values[0], linked_compname, linked_compname]
            else:
                # creating a proper array with the blueshift of the line and its uncertainties
                # note: since we want the blueshift we need to swap the value of the vashift AND swap the + error with the - error
                emlines_bshift[i_line] = \
                    [-AllModels(1)(vashift_parid).values[0],
                     self.errors[0][vashift_parid - 1][1],
                     self.errors[0][vashift_parid - 1][0]]

            # getting the delchi of the line (we assume that the normalisation is the last parameter of the line component)

            # deleting the component
            delcomp(fitcomp_line.xcompnames[-1])

            # fitting and computing errors
            # note: might need to be re-adjusted back to with a logfile if the fit gets stuck here
            calc_fit()

            # calc_fit(logfile=self.logfile)

            calc_error(self.logfile, param='1-' + str(AllModels(1).nParameters * AllData.nGroups), indiv=True)

            # storing the delta chi
            emlines_delchi[i_line] = Fit.statistic - base_chi

            # reloading the model
            mod_data_init.load()

            '''
            Computing the individual flux of each emline

            For this, we delete every other component in the model then vary each parameter of the line for 1000 iterations
            from the MC computation, and draw the 90% flux interval from that
            '''

            # loop on all the components to delete them, in reverse order to avoid having to update them
            for comp in self.includedlist_main[::-1]:

                # skipping the current component, already removed emorptions
                if comp is fitcomp_line or (comp.emorption and comp.xcompnames[0] not in AllModels(1).componentNames):
                    continue

                # skipping multiplicative non-calibration components, which will be deleted anyway with the additive ones
                if not comp.calibration and comp.multipl and not comp.emorption \
                        and not comp.xcompnames[0] in xspec_globcomps:
                    continue

                print(comp.compname)

                comp.delfrommod(rollback=False)
                # fix this

            # loop on the parameters
            flux_line_dist = np.zeros(len(par_draw))
            for i_sim in range(len(par_draw)):
                # setting the parameters of the line according to the drawpar
                for id_parcomp, parcomp in enumerate(fitcomp_line.parlist):
                    # we only draw from the first data group BECAUSE WE TAKE THE FLUX FOR THE FIRST DG,
                    # hence the [0] index in par_draw
                    AllModels(1)(id_parcomp + 1).values = [par_draw[i_sim][0][parcomp - 1]] + AllModels(1)(
                        id_parcomp + 1).values[1:]

                # computing the flux of the line
                # (should be negligbly affected by the energy limits of instruments since we don't have lines close to these anyway)
                '''
                note that this could be an issue for lower energy lines if a high-energy telescope is loaded first
                there could be a need to extend its rmf
                '''
                AllModels.calcFlux('0.3 10')

                # and storing it in the distribution
                # note: won't work with several datagroups
                flux_line_dist[i_sim] = AllData(1).flux[0]

            flux_line_dist = abs(flux_line_dist)
            flux_line_dist.sort()

            # we don't do that anymore to avoid main values outside of MC simulations in extreme cases
            # #storing the main value
            # emlines_flux[i_line][0]=AllModels.calcFlux('0.3 10')

            # drawing the quantile values from the distribution (instead of the main value) to get the errors
            emlines_flux[i_line][0] = flux_line_dist[int(len(par_draw) / 2)]
            emlines_flux[i_line][1] = emlines_flux[i_line][0] - flux_line_dist[
                int((1 - percentile / 100) / 2 * len(par_draw))]
            emlines_flux[i_line][2] = flux_line_dist[int((1 - (1 - percentile / 100) / 2) * len(par_draw))] - \
                                       emlines_flux[i_line][0]

            # reloading the model
            mod_data_init.load()
            self.update_fitcomps()
        # recreating a valid fit for the error computation
        calc_fit()

        return emlines_flux, emlines_ew, emlines_bshift, emlines_delchi

    def get_ew_uls(self, bool_sign, lines_bshift, sign_widths, pre_delete=False):

        '''
        Computes the ewdith upper limit of each non significant line, with other parameters free except the other absorption lines
        '''

        abslines=[comp.compname for comp in [elem for elem in self.complist if elem is not None] if 'agaussian' in comp.compname]

        abslines_ew_ul=np.zeros(len(abslines))

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
            abslines_ew_ul[i_line]=fitcomp_line.get_ew_ul(bshift_range,width_val,pre_delete=pre_delete)

        return abslines_ew_ul

    def save_mod(self):

        '''
        saves the current model configuration and the individual components one by one
        '''

        #saving the whole model
        self.save=allmodel_data()

        #saving each of the included components
        for incl_comp in [elem for elem in self.includedlist if elem is not None]:
            incl_comp.save_comp()


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
        # self.update_fitcomps()

        #updating the logfile
        for comp in [elem for elem in self.complist if elem is not None]:
            comp.logfile=self.logfile
            comp.logfile_write=self.logfile_write

        self.save_mod()

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

    def merge(self,prev_fitmod,add_notincl=True,load_frozen='auto',load_links=False,
                               load_relative_links=True,load_valrange=False):

        '''
        Merges with a previous fitmodel by replacing the saves of all included components also included
        in the previous model with their equivalent's save

        add_notincl: also includes the previous components if they are not included in the current model
                     (as long as they are part of complist)

        load_frozen/links/valragne: options on how to loach each fitcomp's save
        changing from the default can make things not work for automatic reloading
        '''

        prev_inclist=[elem for elem in prev_fitmod.includedlist if elem is not None]

        incl_compnames=[comp.compname for comp in [elem for elem in self.includedlist if elem is not None]]

        for elem_comp in prev_inclist:

            #skipping if the component is not a part of the new fitmod's components
            if elem_comp.compname not in self.name_complist+self.name_cont_complist:
                continue

            #including the component if asked to
            if elem_comp.compname not in incl_compnames \
                    and elem_comp.compname in self.name_complist+self.name_cont_complist:

                if not add_notincl:
                    continue

                self.includedlist = getattr(self,elem_comp.compname).addtomod(incl_list=self.includedlist)

                # updating the fitcomps before anything else
                self.update_fitcomps()

            #replacing the component saves by the previous model's save
            new_comp=getattr(self,elem_comp.compname)

            #we will only load the frozen states of calibation and glob components because we want to unpeg
            #things for the rest if some stuff has been pegged
            load_frozen_indiv=load_frozen==True or \
                              (load_frozen=='auto' and\
                               (elem_comp.calibration or elem_comp.compname.split('_')[-1] in xspec_globcomps))

            #here we need to use relative links to avoid breaking things
            new_comp.reload(elem_comp.save,load_frozen=load_frozen_indiv,load_links=load_links,
                            load_relative_links=load_relative_links,load_valrange=load_valrange)

def make_fitcomp(compname,logfile=None,logfile_write=None,identifier=None,continuum=False,
                 fitcomp_names=None,fitmod=None):

    '''
    Creates a given type of fitcomp depending on what the starting parameters are
    '''

    if '_' in compname:
        comp_prefix = compname.split('_')[0]
        comp_split = compname.split('_')[-1]
    else:
        comp_prefix = ''
        comp_split = compname

    if 'gaussian' in comp_split and not 'cal' in comp_split:
        return fitcomp_line(compname,logfile=logfile,logfile_write=logfile_write,identifier=identifier,
                           continuum=continuum,fitcomp_names=fitcomp_names,fitmod=fitmod)
    else:
        return fitcomp(compname,logfile=logfile,logfile_write=logfile_write,identifier=identifier,
                       continuum=continuum,fitcomp_names=fitcomp_names,fitmod=fitmod)

class fitcomp:

    '''
    class used for a singular component added in fitmod
    Stores various parameters for automatic fitting purposes.
    compname must be a component name understandable by addcomp

    Warning: delfrommod with rollback set to True will reload the save made before
             the component was added for the last time !
    '''

    def __init__(self,compname,logfile=None,logfile_write=None,identifier=None,continuum=False,
                 fitcomp_names=None,fitmod=None):

        #associated fitmod
        self.fitmod=fitmod
        #component addcomp name
        self.compname=compname

        #component string identifier
        self.name=identifier

        #boolean for the component being included or not in the current model
        self.included=False

        #storing the logfile variable
        self.logfile=logfile

        self.logfile_write=logfile_write

        self.fitcomp_names=fitcomp_names

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
        if is_abs(comp_split):
            self.absorption=True
        else:
            self.absorption=False

        if 'cal' in comp_prefix:
            self.calibration=True

            #storing the energies of the components to see if they should be tested for
            if comp_prefix=='calNICER':
                self.cal_e=2.42
            elif comp_prefix=='calNuSTAR':
                self.cal_e=9.51
            elif comp_prefix=='calNICERSiem':
                self.cal_e=1.74
        else:
            self.calibration=False

        # or 'laor' in comp_split
        if 'gaussian' in comp_split and not self.calibration:
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

        self.mandatory=False

        if np.any([elem in comp_split for elem in xspec_globcomps]) or self.calibration:
            self.mandatory=True

        #ensuring we won't lose the diskbb if there is a linked nthcomp or thcomp in the other proposed components
        if 'diskbb' in comp_split and self.fitcomp_names is not None:
            if 'disk_nthcomp' in self.fitcomp_names or 'disk_thcomp' in self.fitcomp_names:
                self.mandatory=True

        #also ensuring the thcomp stays mandatory
        if self.compname=='disk_thcomp':
            self.mandatory=True

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

            #adding some parameters for components which affect other datagroups
            if ('cal' in self.compname and 'edge' in self.compname)\
                    or np.any([elem in self.compname for elem in xspec_globcomps]) :
                self.parlist=ravel_ragged([np.array(self.parlist)+AllModels(1).nParameters*i_grp\
                                           for i_grp in range(AllData.nGroups)]).tolist()

            self.unlocked_pars=[i for i in self.parlist if (not (AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen)\
                                                        and AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='')]

    def print_xlog(self,string):

        '''
        prints and logs info in the associated xspec log file,
        flushed to ensure the logs are printed before the next xspec print
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
            #additive calibration components are added with lastin to not consider all the multiplicative calibration components


            self.parlist,self.compnumbers=addcomp(self.compname,
                                    position=('last' if AllModels(1).componentNames[0] not in xspec_globcomps else\
                                              'lastin') if self.calibration else 'lastinall',
                                                  included_list=incl_list,return_pos=True)

            #fixing parameters if values are provided
            if fixed_vals is not None:
                for i_fixedval,elem_fixedval in enumerate(fixed_vals):

                    #skipping placeholders
                    if elem_fixedval is None:
                        continue

                    #fixing the parameters
                    valpar=self.parlist[i_fixedval]
                    AllModels(par_degroup(valpar)[0])(par_degroup(valpar)[1]).values=[elem_fixedval]+\
                    AllModels(par_degroup(valpar)[0])(par_degroup(valpar)[1]).values[1:]
                    AllModels(par_degroup(valpar)[0])(par_degroup(valpar)[1]).frozen=True

            #note: the compnumbers are defined in xspec indexes, which start at 1

            #computing the unlocked parameters, i.e. the parameters not frozen at the creation of the component
            self.unlocked_pars=[i for i in self.parlist if (not AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen and\
                                                            AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='')]


            #computing a mask of base unlocked parameters to rethaw pegged parameters during the second round of autofit
            self.unlocked_pars_base_mask=[(not AllModels(par_degroup(i)[0])(par_degroup(i)[1]).frozen and\
                                                            AllModels(par_degroup(i)[0])(par_degroup(i)[1]).link=='') for i in self.parlist]

            #this parameter is created for the sole purpose of significance testing, to avoid using unlocked pars which is modified when
            #parameters are pegged (something which shouldn't affect the significance testing are pegged parameters were still variable initially)
            #with this test, the value is overwritten only the first time the component is added
            if self.n_unlocked_pars_base==0:
                #to avoid issues with ftests for fully fixed components
                self.n_unlocked_pars_base=min(len(self.unlocked_pars),1)

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
        rollback : load the model save before the component was saved.
        Doesn't requires update_fitcomps afterwards
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
                xscorpeon.load('auto',scorpeon_save=prev_scorp if prev_scorp.all_frozen else None)

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
        unfreezes the unlocked parameters of this component
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

    def fit(self,split_fit=True,compute_errors=True,fit_to=30):

        '''
        Fit + errors of this component

        if split_fit is set to True, freezes all other components
                                     then computes the fit and error with this component unlocked only

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
        calc_fit(logfile=self.logfile,timeout=fit_to)

        if compute_errors:
            if split_fit:
                #computing errors for the component parameters only
                calc_error(logfile=self.logfile,param=str(self.parlist[0])+'-'+str(self.parlist[-1]),indiv=True)
            else:
                #computing errors for all models
                calc_error(logfile=self.logfile,param='1-'+str(AllData.nGroups*AllModels(1).nParameters),indiv=True)

        self.fitted_mod=allmodel_data()

    def save_comp(self):
        '''
        saves all of the parameters values, links and frozen states in an array

        no direct parameter attribution to avoid issues when reload
        '''

        # note: we do this because some models have length 1 values for choices (ex: xscat grain type)
        # in which case we can't use a regular npars*6 type array

        #note: this number is the number of non-standard parameters in the comp
        #it doesn't count normally linked parameters after the first datagroup
        comp_npars=len(self.parlist)

        values = np.array([None] * comp_npars)
        links = np.array([None] * comp_npars)
        frozen = np.zeros(comp_npars).astype(bool)
        relative_links=np.array([None]*comp_npars)

        for id_par,i_par in enumerate(self.parlist):

            elem_par=AllModels(par_degroup(i_par)[0])(par_degroup(i_par)[1])

            # keeping length 1
            values[id_par] = np.array(elem_par.values)

            # only for normal parameters
            if len(values[id_par]) > 1:
                # safeguard against linked parameters with values out of their bounds
                if values[id_par][0] < values[id_par][2]:
                    values[id_par][2] = values[id_par][0]
                if values[id_par][0] > values[id_par][5]:
                    values[id_par][5] = values[id_par][0]

            frozen[id_par] = elem_par.frozen
            links[id_par] = elem_par.link.replace('= ', '')

            #for now all links should refer to parameters inside the parlists so this should be fine
            if links[id_par]!='':
                relative_links[id_par]=self.fitmod.idtocomp([int(links[id_par].replace('p',''))])[0]

        self.values = values
        self.frozen = frozen
        self.links = links
        self.relative_links=relative_links

        self.save=[self.values,self.frozen,self.links,self.relative_links]

    def reload(self,save=None,load_frozen=False,load_relative_links=True,load_links=False,load_valrange=False):

        '''
        reload the component values according to an internal or provided save

        can choose to load the whole value range, frozen and link states or not to give more flexibility and
        avoid issues when reloading a component from a previous model

        relative_links is a special case where you can reload using the relative position of parameters in the
        parlist of component.
        '''

        assert save is not None or self.values is not None, 'Cannot reload component without saved values or giving a save'

        if save is None:
            reload_values=self.values
            reload_links=self.links
            reload_frozen=self.frozen
            reload_relative_links=self.relative_links
        else:
            reload_values=save[0]
            reload_frozen=save[1]
            reload_links=save[2]
            reload_relative_links=save[3]

        for id_par,i_par in enumerate(self.parlist):

            elem_par=AllModels(par_degroup(i_par)[0])(par_degroup(i_par)[1])

            #changing the values overrides the freeze and link state so it's important to reload the ones
            #of the current model if we're not loading them

            #first we untie the parameter to allow replacing the values correctly
            init_par_frozen=elem_par.frozen
            init_par_link=elem_par.link.replace('= ','')

            elem_par.frozen=False
            elem_par.link=''

            #reloading the main value or the whole value array
            # (forcing it for global comps to avoid issues with the range of constant factors)

            if load_valrange or self.compname.split('_')[-1] in xspec_globcomps:
                elem_par.values=reload_values[id_par]
            else:
                elem_par.values=[reload_values[id_par][0]]+elem_par.values[1:]

            #reloading the frozen states
            if load_frozen:
                elem_par.frozen=reload_frozen[id_par]
            else:
                elem_par.frozen=init_par_frozen

            '''
            reloading the link states
            this is the most risky part when reloading components from previous models so it has more options
            '''

            if load_relative_links:

                if reload_relative_links[id_par] is not None:
                    '''
                    here we transform the (previous fitcomp,corresponding fitmod parlist index)
                    into the new parameter position of the new fitcomp
                    We make sure to fetch the new component by fetching the new component using
                    the previous component's compname
                    note that here we assume that both have the same number of parameters, which should stay true
                    '''
                    new_relative_link=getattr(self.fitmod,reload_relative_links[id_par][0].compname).\
                                     parlist[reload_relative_links[id_par][1]]
                    elem_par.link = str(new_relative_link)

            elif load_links:
                elem_par.link=reload_links[id_par]
            else:
                elem_par.link=init_par_link


class fitcomp_line(fitcomp):

    '''
    Child class of fitcomp to add additional methods for line components
    '''

    def __init__(self, compname, logfile=None, logfile_write=None, identifier=None,
                 continuum=False,fitcomp_names=None, fitmod=None):
        super().__init__(compname,logfile=logfile,logfile_write=logfile_write,identifier=identifier,
                       continuum=continuum,fitcomp_names=fitcomp_names,fitmod=fitmod)
    def get_width(self,err_sigma=3):

        '''
        computes errors to see if the width of the current component (assumed to be gaussian)
        is not compatible with 0 at 3 sigma
        If it isn't, returns its values, otherwise returns an array of 0
        '''

        if not self.included:
            self.print_xlog('\nlog:Component not included. Returning 0 values')
            return np.zeros(3)

        width_par=AllModels(1)(self.parlist[-2])

        #returning 0 if the parameter is unconstrained (and thus has been frozen)
        if width_par.frozen:
            return np.array([0,0,0])

        self.logfile.readlines()

        #computing the width with the current fit at n sigmas (aka a delchi of the square of that for 1 dof, see
        # the sign_sigmas_delchi_1dof arr)
        Fit.error('stop ,,0.1 max 100 '+str(err_sigma**2)+' '+str(self.parlist[-2]))

        #storing the error lines
        log_lines=self.logfile.readlines()

        #testing if the parameter is pegged to 0 at 3 sigma
        if  '***Warning: Parameter pegged at hard limit: 0\n' in log_lines:
            return np.array([0,0,0])

        return np.array([width_par.values[0],width_par.values[0]-width_par.error[0],width_par.error[1]-width_par.values[0]])

    def get_ew(self, err_percent=90):

        '''
        Note : we currently only compute the ewidth of the first data group
        '''

        if not self.included:
            self.print_xlog('\nlog:Component not included. Returning 0 values')
            return np.zeros(3)

        #computing the eqwidth without errors first
        AllModels.eqwidth(self.compnumbers[-1])

        ew_noerr=np.array(AllData(1).eqwidth)*1e3

        try:
            #ew at 90%
            AllModels.eqwidth(self.compnumbers[-1],err=True,number=1000,level=err_percent)

            #conversion in eV from keV (same e)
            ew=np.array(AllData(1).eqwidth)*1e3

            #testing if the MC computation led to bounds out of the initial value :
            if ew[1]<=ew[0]<=ew[2]:
                #getting the actual uncertainties and not the quantile values
                ew_arr=np.array([abs(ew[0]),ew[0]-ew[1],ew[2]-ew[0]])
            else:
                #if the bounds are outside we take the median of the bounds as the main value instead
                ew_arr=\
                    np.array([abs(ew[1]+ew[2])/2,abs(ew[2]-ew[1])/2,abs(ew[2]-ew[1])/2])
        except:
            ew_arr=ew_noerr

        return ew_arr

    def get_ew_ul(self,bshift_range,line_width=0,pre_delete=False,ul_level=99.7,n_ul_comp=101):

        '''
        Note : we compute the ew ul from the first data group
        Also tests the upper limit for included components by removing them first
        '''

        if self.compname.split('_')[0]=='NiKa27abs':
            return 0

        curr_model=allmodel_data()

        distrib_ew=[]

        prev_chatter=Xset.chatter
        prev_logChatter=Xset.logChatter

        if Xset.chatter>5:
            Xset.chatter=5
        Xset.logChatter=5

        if type(bshift_range) not in (list,np.ndarray):
            #skipping interval computations when a precise value for the space is provided
            bshift_space=bshift_range
        else:
            bshift_space=np.linspace(-bshift_range[1],-bshift_range[0],n_ul_comp)


        with tqdm(total=len(bshift_space)) as pbar:

            #loop on a sampling in the line blueshift range
            for vshift in  bshift_space:

                #restoring the initial model
                curr_model.load()

                #deleting the component if needed and the model is not in a no-abs line version
                if self.included and not pre_delete:
                    delcomp(self.compname)


                #adding an equivalent component, without providing the included group because we don't want to link it
                addcomp(self.compname,position='lastinall')

                npars=AllModels(1).nParameters

                #### forcing unfrozen lines even if the line is manually disabled in addcomp (unless its Nickel)
                if 'Nika' not in self.compname:
                    AllModels(1)(npars).values=[-1e-4,1e-7,-5e-2,-5e-2,0,0]
                    AllModels(1)(npars).frozen=False

                #skipping the computation if the line is above the maximum energy
                if AllModels(1)(npars-2).values[0]>AllData(1).energies[-1][1]:
                    self.print_xlog('Line above maximum energy range. Skipping computation...')
                    distrib_ew+=[0]
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
                    AllModels.eqwidth(len(AllModels(1).componentNames),err=True,number=1000,level=ul_level)
                    #adding the ew value to the distribution
                    distrib_ew+=[abs(AllData(1).eqwidth[1])]
                except:
                    self.print_xlog('Issue during EW computation')
                    distrib_ew+=[0]

                Xset.logChatter=5

                pbar.update()

        Xset.chatter=prev_chatter
        Xset.logChatter=prev_logChatter

        #reloading the previous model iteration
        curr_model.load()

        return max(distrib_ew)*1e3


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

            for i_grp in range(1,AllData.nGroups+1):
                addcomp_names_grp=[]
                #adding default model component names if there is a default model
                for mod in list_models:

                    # if mod=='':
                    #
                    #     mod_expression,mod_tables_dict=numbered_expression(mod_name=mod)
                    #
                    #     #there are addcomps for each  models only if it has more than one additive components
                    #     #we thus need to compare the total number of addcomps with the other model addcomps
                    #     #this is only for nicer and can break if there are addcomps in other nxb type models
                    #
                    #     if len(self.addcomps[0])>(2 if 'nxb' in list_models else 0) + (2 if 'sky' in list_models else 0):
                    #
                    #         for elemcomp in AllModels(1).componentNames:
                    #
                    #             #restricting to additive tables for table models
                    #             if elemcomp in list(mod_tables_dict.keys()):
                    #                 if 'atable{' in mod_tables_dict[elemcomp]:
                    #                     self.addcompnames += [elemcomp]
                    #             else:
                    #                 #restricting to additive components
                    #                 if elemcomp.split('_')[0] not in xspec_multmods:
                    #                     self.addcompnames+=[elemcomp]


                    #adding NICER background component Names
                    if mod=='nxb':
                        try:
                            addcomp_names_grp+=AllModels(i_grp,'sky').componentNames
                            break
                        except:
                            pass

                    elif mod=='sky':
                        try:
                            addcomp_names_grp += AllModels(i_grp, 'nxb').componentNames
                            break
                        except:
                            pass
                    else:

                        indiv_mod_addcompnames=[]

                        # mod_compnames = AllModels(i, mod).componentNames
                        mod_expression, mod_tables_dict = numbered_expression(mod_name=mod)

                        for elemcomp in AllModels(i_grp,mod).componentNames:

                            try:

                                # restricting to additive tables for table models
                                if elemcomp in list(mod_tables_dict.keys()):
                                    if 'atable{' in mod_tables_dict[elemcomp]:
                                        indiv_mod_addcompnames += [mod + '_' + elemcomp]
                                else:
                                    # restricting to additive components
                                    if elemcomp.split('_')[0] not in xspec_multmods:
                                        indiv_mod_addcompnames += [mod + '_' + elemcomp]
                            except:
                                pass

                        #only keeping the component if there is only one addcomp in the model
                        if len(indiv_mod_addcompnames)>1 or AllData.nGroups>1:
                            addcomp_names_grp+=indiv_mod_addcompnames
                        else:
                            #testing if this is
                            pass

                self.addcompnames+=[addcomp_names_grp]
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


def xPlot(types,axes_input=None,plot_saves_input=None,plot_arg=None,includedlist=None,group_names='auto',
          hide_ticks=True,
          secondary_x=True,legend_position=None,xlims=None,ylims=None,label_bg=False,
          mult_factors=None,label_indivcomps=False,
          no_name_data='auto',force_ylog_ratio=False,legend_ncols=None,
          data_colors=None,model_colors=None,model_ls=None,addcomp_colors=None,addcomp_ls=None,
          data_alpha=1,auto_figsize=(10,8),auto_panel_hratio=None,
          addcomp_source_cmaps=['YlGn','YlOrRd','PuBu','RdPu'],
          legend_sources=False,label_sources='auto',label_sources_cval='auto',legend_sources_loc="best",
          legend_sources_bbox=None,
          legend_addcomp_groups=False,
          skip_main_legend=False,
          addcomp_rebin=None,
          elinewidth_data=0.75):

    '''
    Replot xspec plots using matplotib. Accepts custom types:
        -line_ratio:        standard ratio plot with iron line positions highlighted and a legend for a 2dchimap plot below

        #to be added if needed
        -2Dchimap:          2d chi² color + contour map
        -absorb_ratio:      modified ratio plot with absorption lines in the ratio and higlighted + absorption line positions highlighted


    plot_arg is an array argument with the values necessary for custom plots. each element should be None if the associated plot doesn't need plot arguments

    specifics for standard plots:
        eemo: set to xLog=True and yLog=True

    xlims and ylims should be tuples of values
        can be made for individual subplots plot types independantly if set to 2-dimensionnal tuples
        in this case the null value is [None,None]

    mult_factors:
        if not None, a list of multiplicative factors for each datagroups,
        to offset their values for visual purposes

    If plot_saves is not None, uses its array elements as inputs to create the plots.
    Else, creates a range of plot_saves using plot_saver from the asked set of plot types

    if axes is not None, uses the axes as baselines to create the plots. Else, return an array of axes with the plots

    if includedlist is not None, replace xspec component names with fitcomp names

    group_names give str group names to each AllData group instead of a standard naming

    if group_names is set to "nolabel", doesn't show the groups

    hide ticks hide axis ticks for multiple plots and subplots

    secondary_x enables plotting an Angstrom top axis or not

    legend position forces a specific legend position

    no_name_data: -if set to auto, removes the data label for plots with more than 1 panel
                    and rearranges the position when there are many datagroups

    legend_ncols: overwrite the automatic number of legend columns

    data_colors: if not None 'data', should be an AllData.nGroups size iterable.
                 overwrite the default xspec colors for the data
    model_colors: if not None or 'data', should be an AllData.nGroups size iterable.
                 overwrite the default xspec colors for the models
                if set to 'data', copies the data_colors
    model_ls: if not None:
                if 'group', cycles through different ls for each datagroups
    addcomp_colors: if not None or 'data', should be an AllData.nGroups size iterable.
                    overwrite the default xspec colors for the models.
                    if set to 'data', copies the data_colors
                    if set to 'source', uses a different series of colormaps for each source using
                        addcomp_source_cmaps, following the source number

    addcomp_source_cmaps: colormaps to use for the different sources. if start with color_,
                            uses the next part of the string as a constant color

    addcomp_ls: if not None:
                if 'group', cycles through different ls for each datagroups. Matches model_ls

    data_alpha: alpha value for the ratio/delchi plot errorbars

    legend_sources:
                add an individual legends for different sources in addcomp.
                Should be used only with addcomp_colors='source'

    label_sources: 'auto' or string iter of len(AllModels.sources)
        if no 'auto', replace the default xspec source names by the strings

    label_sources_cval: 'auto' or float iter of len(AllModels.sources) in [0;1]
        if not 'auto', manual values of the value of the colormap of each source

    legend_sources_loc: the loc of the additional legend

    legend_addcomp_groups:
        Fuse the datagroup legend with the ls for the different groups used for addcomps.
        Should be used only with model_ls='group'

    skip_main_legend:
        does not plot the main legend

    addcomp_rebin: None or [float]*len(AllData.nGroups)
        if not None, do a separate plotting to get the addcomp components without lowering the resolution
        useful for cases where the components are visually rebinned with the source

    '''

    def combo_legend(ax):
        #taken from https://andrewpwheeler.com/2022/09/16/legends-in-python/
        handler, labeler = ax.get_legend_handles_labels()
        hd = []
        labli = list(set(labeler))

        for lab in labli:
            comb = [h for h, l in zip(handler, labeler) if l == lab]
            hd.append(tuple(comb))

        return hd, labli

    # ls_types=['dotted','dashed','dashdot']

    # linestyle_tuple = [
    #     ('loosely dotted', (0, (1, 10))),
    #     ('dotted', (0, (1, 1))),
    #     ('densely dotted', (0, (1, 1))),
    #     ('long dash with offset', (5, (10, 3))),
    #     ('loosely dashed', (0, (5, 10))),
    #     ('dashed', (0, (5, 5))),
    #     ('densely dashed', (0, (5, 1))),
    #
    #     ('loosely dashdotted', (0, (3, 10, 1, 10))),
    #     ('dashdotted', (0, (3, 5, 1, 5))),
    #     ('densely dashdotted', (0, (3, 1, 1, 1))),
    #
    #     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    #     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    #     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    ls_types = [(0, (3, 1, 1, 1)), (0, (5, 1)), (5, (10, 3)), (0, (1, 1)), \
                (0, (3, 10, 1, 10)), (0, (5, 10)), (0, (1, 10))]
    ls_types_group = ls_types

    if axes_input is None:
        fig=plt.figure(figsize=auto_figsize)
        grid=GridSpec(len(types.split(',')),1,figure=fig,hspace=0.,height_ratios=auto_panel_hratio)
        axes=[plt.subplot(elem) for elem in grid]

    else:
        axes=[axes_input] if type(axes_input) not in [list,np.ndarray] else axes_input

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

    if xlims is not None:
        if np.ndim(xlims)==1:
            xlims_use=np.array([xlims for i in range(len(types_split))])
        else:
            xlims_use=xlims
    else:
        xlims_use=None

    if ylims is not None:
        if np.ndim(ylims)==1:
            ylims_use=np.array([ylims for i in range(len(types_split))])
        else:
            ylims_use=ylims
    else:
        ylims_use=None

    for i_ax,plot_type in enumerate(types_split):

        curr_ax=axes[i_ax]
        curr_save=plot_saves[i_ax]

        #plotting the title for the first axe
        if i_ax==0:

            if axes_input is None:
                curr_ax.set_title(curr_save.labels[-1])

            if secondary_x and Plot.xAxis in ['keV','angstrom']:
                #putting a wavelength copy of the x axis at the top
                curr_ax_second=curr_ax.secondary_xaxis('top',functions=(ang2kev,ang2kev))
                curr_ax_second.set_xlabel(r' Wavelength $(\AA)$' if Plot.xAxis=='keV' else 'keV')
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
        if curr_save.xLog or plot_type.endswith('mo') :
            curr_ax.set_xscale('log')
        if curr_save.yLog or (plot_type=='ratio' and force_ylog_ratio) or plot_type.endswith('mo') :
            curr_ax.set_yscale('log')

        #this needs to be performed independantly of how many groups there are
        if xlims_use is not None and xlims_use[i_ax][0] is not None:
            curr_ax.set_xlim(xlims_use[i_ax][0],xlims_use[i_ax][1])
        if ylims_use is not None and ylims_use[i_ax][0] is not None:
            curr_ax.set_ylim(ylims_use[i_ax][0],ylims_use[i_ax][1])

        for id_grp in range(max(1,curr_save.nGroups)):

            if mult_factors is not None:
                mult_factor_grp=mult_factors[id_grp]
            else:
                mult_factor_grp=1

            if group_names=='auto':
                #auto naming the group from header infos
                try:
                    AllData(1)
                    no_sp=False
                except:
                    no_sp=True

                if not no_sp and 'sp' in AllData(id_grp+1).fileName:
                    grp_name=AllData(id_grp+1).fileName.split('_sp')[0]
                else:
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

                            if grp_tel=='INTEGRAL':
                                grp_obsid=AllData(id_grp+1).fileName.split('_sum_pha')[0].split('_')[-1]
                            else:
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

                grp_name=group_names[id_grp]

                # Plot.setGroup('2')
                #needs to be implemented
                # grp_name=
                # grp_name='' if group_names=='nolabel' else\
                #     ('group '+str(id_grp+1) if curr_save.nGroups>1 else '') if group_names is None else group_names_list[id_grp]

            if curr_save.y[id_grp] is not None:
                #plotting each data group

                curr_ax.errorbar(curr_save.x[id_grp],curr_save.y[id_grp]*(1 if 'data' not in plot_type else mult_factor_grp),
                                 xerr=curr_save.xErr[id_grp],
                                 yerr=curr_save.yErr[id_grp].clip(0)*(1 if 'data' not in plot_type else mult_factor_grp),
                                 color=xcolors_grp[id_grp] if data_colors is None else data_colors[id_grp],
                                 linestyle='None',
                                 elinewidth=elinewidth_data,alpha=data_alpha,
                                 label='' if legend_addcomp_groups else\
                            ('' if group_names=='nolabel' or (no_name_data=='auto' and i_ax!=len(types_split)-1) else grp_name))

                #adding an additional silent background line to show the datagroup ls if the legend is requested
                if legend_addcomp_groups:

                    #remakeing the errorbar legend with a bigger lw to make it more distinguishable
                    curr_ax.errorbar([],[],xerr=0,yerr=0,color=xcolors_grp[id_grp] if data_colors is None else data_colors[id_grp],
                                     linestyle='None',
                                     elinewidth=1.5,alpha=1.0,
                                     label='' if group_names=='nolabel' or (no_name_data=='auto' and i_ax!=len(types_split)-1) else grp_name)

                    #will later be fused by combo_legend
                    curr_ax.plot([],[],color='gray',ls=ls_types_group[id_grp],lw=1.5,alpha=0.5,
                                 label='' if group_names=='nolabel' or (no_name_data=='auto' and i_ax!=len(types_split)-1) else grp_name)

            #plotting models
            if 'ratio' not in plot_type and curr_save.model[id_grp] is not None:

                if model_colors is not None:
                    if model_colors == 'data':
                        model_color_grp=data_colors[id_grp]
                    else:
                        model_color_grp=model_colors[id_grp]
                else:
                    model_color_grp=None

                if model_ls is not None:
                    if model_ls == 'group':
                        model_ls_group=ls_types_group[id_grp]
                else:
                    model_ls_group='-'

                curr_ax.plot(curr_save.x[id_grp],curr_save.model[id_grp]*(1 if 'data' not in plot_type else mult_factor_grp),
                             color=xcolors_grp[id_grp] if model_color_grp is None else model_color_grp,
                             ls=model_ls_group,alpha=0.5,
                             label='' if group_names=='nolabel' else '')


            if 'data' in plot_type:

                #plotting backgrounds
                #empty backgrounds are stored as 0 values everywhere so we test for that to avoid plotting them for nothing
                if curr_save.addbg and curr_save.background_y[id_grp] is not None:
                    curr_ax.errorbar(curr_save.background_x[id_grp],
                                     curr_save.background_y[id_grp]*mult_factor_grp,
                                     xerr=curr_save.background_xErr[id_grp],
                                     yerr=curr_save.background_yErr[id_grp]*mult_factor_grp,
                                     color=xcolors_grp[id_grp] if data_colors is None else data_colors[id_grp],
                                     linestyle='None',elinewidth=0.5,alpha=data_alpha,
                                     marker='x',mew=0.5,label='' if not label_bg or group_names=='nolabel' else grp_name+' background')

        #### locking the axe limits
        '''
        locking the y axis limits requires considering the uncertainties (which are not considered for rescaling) 
        without perturbing the previous limits if they were beyond
        we don't bother doing that for the background
        '''

        if xlims_use is None or xlims_use[i_ax][0] is None:
            curr_ax.set_xlim(round(min(ravel_ragged(curr_save.x-curr_save.xErr)),2),
                             round(max(ravel_ragged(curr_save.x+curr_save.xErr)),2))

        #with condition to avoid rescaling if there is no data (e.g. for eemo)
        if (ylims_use is None or ylims_use[i_ax][0] is None) and len(np.argwhere(ravel_ragged(curr_save.y)!=None))!=0:
            curr_ax.set_ylim(min(curr_ax.get_ylim()[0],round(min(ravel_ragged(curr_save.y-curr_save.yErr)),4)),
                          max(curr_ax.get_ylim()[1],round(max(ravel_ragged(curr_save.y+curr_save.yErr)),4)))

        comp_source_list=[]
        source_names=[]
        source_ncomps=[]
        source_cmaps=[]

        #plotting the components after locking axis limits to avoid huge rescaling
        if 'data' in plot_type and curr_save.add and curr_save.ismod:

            #storing a non-rebinned version of the components to plot it if asked
            if addcomp_rebin is not None:
                for id_grp in range(curr_save.nGroups):
                    rebinv_xrism(id_grp+1,addcomp_rebin[id_grp])

                curr_save_addcomp=plot_saver(plot_type)[0]

            #assigning colors to the components (here we assumen the colormap used is not cyclic)
            norm_colors_addcomp=mpl.colors.Normalize(vmin=0,vmax=len(curr_save.addcomps[id_grp]))

            colors_addcomp=mpl.cm.ScalarMappable(norm=norm_colors_addcomp,cmap=mpl.cm.plasma)

            list_models=list(AllModels.sources.values())

            #using fitcomp labels if possible
            if includedlist is not None:
                label_comps=[comp if type(comp)==str else comp.compname for comp in [elem for elem in includedlist\
                                                                            if elem is not None and not elem.multipl]]

                #adding NICER background component Names
                if 'nxb' in list_models:
                    label_comps+=AllModels(1,'sky').componentNames

                if 'sky' in list_models:
                    label_comps+=AllModels(1,'nxb').componentNames

            else:
                label_comps=ravel_ragged(curr_save.addcompnames)

            for id_grp in range(curr_save.nGroups):

                if addcomp_colors == 'source':

                    comp_source_list_grp=[elem.split('_')[0] for elem in curr_save.addcompnames[id_grp]]
                    comp_source_list += [comp_source_list_grp]

                    source_names_grp=[]
                    for elem in comp_source_list_grp:
                        if elem not in source_names_grp:
                            source_names_grp+=[elem]
                    source_names += [np.array(source_names_grp)]

                    source_ncomps += [np.array([sum(np.array(comp_source_list[id_grp]) == elem) for elem in source_names[id_grp]])]
                    source_cmaps += [[(addcomp_source_cmaps[i].split('_')[-1] if addcomp_source_cmaps[i].startswith('color_')\
                                      else getattr(plt.cm, addcomp_source_cmaps[i])) for i in range(len(addcomp_source_cmaps))]]

                if mult_factors is not None:
                    mult_factor_grp = mult_factors[id_grp]
                else:
                    mult_factor_grp = 1

                #common attribution for each non-source type of colors
                if addcomp_colors is not None:
                    if addcomp_colors == 'data':
                        addcomp_color_grp=data_colors[id_grp]
                    elif type(addcomp_colors)==str and addcomp_colors!='source':
                        addcomp_color_grp=addcomp_colors
                    else:
                        addcomp_color_grp=addcomp_colors[id_grp]
                else:
                    addcomp_color_grp=None

                #loop through addcomps
                for i_comp in range(len(curr_save.addcomps[id_grp])):

                    if addcomp_colors=='source':

                        i_source=np.argwhere(curr_save.addcompnames[id_grp][i_comp].split('_')[0]==source_names[id_grp])[0][0]

                        #value cycling through the colormap
                        color_val=(0.8-0.7*(i_comp-sum(source_ncomps[id_grp][:i_source]))/\
                                                               source_ncomps[id_grp][i_source])
                        comp_color=source_cmaps[id_grp][i_source] if type(source_cmaps[id_grp][i_source])==str else\
                                    source_cmaps[id_grp][i_source](color_val)

                    else:
                        comp_color=colors_addcomp.to_rgba(i_comp) if  addcomp_colors not in [None,'data']\
                                         else  addcomp_color_grp

                    if addcomp_rebin is not None:
                        curr_save_use_addcomp=curr_save_addcomp
                    else:
                        curr_save_use_addcomp=curr_save
                    try:
                        curr_ax.plot(curr_save_use_addcomp.x[id_grp],curr_save_use_addcomp.addcomps[id_grp][i_comp]*mult_factor_grp,
                                     color=comp_color,
                                 label='' if not label_indivcomps else label_comps[i_comp] if id_grp==0 else '',
                                     linestyle=ls_types_group[id_grp] if addcomp_ls=='group'\
                                         else ls_types[(i_comp)%len(ls_types)],
                                     linewidth=1)
                    except:
                        try:
                            curr_ax.plot(curr_save_use_addcomp.x[0],curr_save_use_addcomp.addcomps[id_grp][i_comp]*mult_factor_grp,
                                         color=comp_color,
                                     label='' if not label_indivcomps else label_comps[i_comp] if id_grp==0 else '',
                                         linestyle=ls_types_group[id_grp] if addcomp_ls == 'group' \
                                             else ls_types[(i_comp) % len(ls_types)],
                                         linewidth=1)
                        except:
                            breakpoint()
                            print("check if other x work")

        #ratio line
        if 'ratio' in plot_type:
            curr_ax.axhline(y=1,xmin=0,xmax=1,color='green')
        if 'delchi' in plot_type:
            curr_ax.axhline(y=0,xmin=0,xmax=1,color='green')

        #plotting the legend in horizontal and below the main result if necessary

        #getting the main legend combo handles and labels
        main_leg_hd,main_leg_labli=combo_legend(curr_ax)
        #note: the default handlelength for items is 2, we double it if we're combining items to show the dash pattern

        if not skip_main_legend:
            if AllData.nGroups>=5 and no_name_data=='auto':
                if i_ax==0:
                    curr_ax.legend(main_leg_hd,main_leg_labli,loc=legend_position,
                                   ncols=3+np.ceil(AllData.nGroups/15) if legend_ncols==None else legend_ncols,
                                   handlelength=4 if legend_addcomp_groups else None)

                if i_ax==len(types_split)-1:
                    bbox_yval=max(-0.3-0.2*np.ceil(AllData.nGroups/2),-0.5)
                    curr_ax.legend(main_leg_hd,main_leg_labli,loc='lower center',
                                   bbox_to_anchor=(0.5,bbox_yval),
                                   ncols=3+np.ceil(AllData.nGroups/15) if legend_ncols==None else legend_ncols,
                                   handlelength=4 if legend_addcomp_groups else None)


            else:
                curr_ax.legend(main_leg_hd,main_leg_labli,loc=legend_position,
                               handlelength=4 if legend_addcomp_groups else None)

        #addcomp source legends if requested
        if 'data' in plot_type and curr_save.add and curr_save.ismod and legend_sources and addcomp_colors=='source':

            ax_legend_sources=curr_ax.twinx()
            ax_legend_sources.yaxis.set_visible(False)

            #for now we only do this for the first datagroup to avoid too much of a mess
            for i_source in range(len(source_names[0])):

                ax_legend_sources.plot([],[],
                             color=source_cmaps[0][i_source] if type(source_cmaps[0][i_source])==str else\
                                 source_cmaps[0][i_source]\
                                     (0.8 if type(label_sources_cval)==str and label_sources_cval=='auto'\
                                    else label_sources_cval[i_source]),
                             label=(source_names[0][i_source] if type(label_sources)==str and label_sources=='auto'\
                                    else label_sources[i_source]),
                             linestyle='-',
                             linewidth=1)

            ax_legend_sources.legend(loc=legend_sources_loc,title='sources',bbox_to_anchor=legend_sources_bbox)

    if axes_input is None:
        fig.tight_layout()
        return axes
    else:
        return None,None

def plot_comp_ratio(cont_addcomps,other_addcomps,ener_low,ener_high,
                   plot_type='eemo',
                    ylims=None,
                   xcm=None,
                   other_addcomps_labels=None,other_addcomps_colors=None,figsize=(10,5),
                    plot_transi=True,minor_locator=10,ylabel_prefix='',manual_bbox=False):

    '''
    Makes a ratio plot of individual components compared to the sum of others related to a specific weight

    plot_type (eemo or other xspec type where addcomp is expressed)
        will be used to get the addcomp values

    xcm : path or None
        will load a model or take the current one

    comp_addcomps: iterable of ints
        the list of addcomps ids (in xspec, so with indexes starting at 1)
        to be summed to create the continuum, to which
        all thethe "other_addcomps" will individually be normalized

    other_addcomps: iterable of either ints or iterables of ints
        the list of the addcomps ids (in xspec, so with indexes starting at 1)
        which will appear individually as the ratio on top of comp_addcomps
        note: can be combined by passing iterables within the list
        In this case, will sum all the addcomps in that iterbale for a single displayed component
        if some elements are None, will not plot anything but keeps the label spot for additional lines in the legend

    ener_low/ener_high: x axis limits of the plot

    other_addcomps_names/colors: None or iterable of same len as other_addcomps
        labels and colors for the additional components

    figsize:
        python plot figure size

    plot_transi: plot individual transitions with plt_std_ener (in grey)

    minor_locator: if not False, the sampling of the x axis minor locator

    ylabel_prefix: optional string to add before y label

    '''

    if xcm is not None:
        Xset.restore(xcm)

    Plot(plot_type)
    cont_addcomps_sum = np.array([Plot.addComp(addCompNum=i_comp, plotGroup=1) for i_comp in cont_addcomps]).sum(0)

    other_addcomps_vals=[]
    for indiv_cont_id in other_addcomps:
        if indiv_cont_id is None:
            other_addcomps_vals+=[None]
            continue

        if type(indiv_cont_id)==int:
            indiv_cont_id_list=[indiv_cont_id]
        else:
            indiv_cont_id_list=indiv_cont_id
        other_addcomps_vals+=[np.array([Plot.addComp(addCompNum=i_comp, plotGroup=1) for i_comp in indiv_cont_id_list]).sum(0)]

    other_addcomps_ratio=[None if elem is None else np.nan_to_num(elem/cont_addcomps_sum) for elem in other_addcomps_vals]

    plt.figure(figsize=figsize)
    ax_comp=plt.gca()

    plt.xlim(ener_low,ener_high)
    plt.xlabel('Energy '+Plot.xAxis)
    plt.ylabel(ylabel_prefix+(' ' if ylabel_prefix!='' else '')+'ratio to continuum')

    if other_addcomps_labels is None:
        other_addcomps_labels_use=np.repeat('',len(other_addcomps))
    else:
        other_addcomps_labels_use=other_addcomps_labels

    if other_addcomps_colors is None:
        other_addcomps_colors_use=np.repeat(None,len(other_addcomps))
    else:
        other_addcomps_colors_use=other_addcomps_colors

    for elem_ratio,elem_color,elem_label in \
     zip(other_addcomps_ratio,other_addcomps_colors_use,other_addcomps_labels_use):
        plt.plot([] if elem_ratio is None else Plot.x(1),
                  [] if elem_ratio is None else 1+elem_ratio,
                 color=elem_color,label=elem_label,lw=0 if elem_color==None else None)

    if ylims is not None:
        plt.ylim(ylims)

    if plot_transi:
        plot_std_ener(ax_comp,plot_indiv_transi=True,force_side='none',plot_em=True,squished_mode=False,color='grey')

    if minor_locator is not False:
        ax_comp.xaxis.set_minor_locator(AutoMinorLocator(minor_locator))

    plt.tight_layout()

    plt.tight_layout()


    if manual_bbox is not False:
        #plt.legend(loc='upper left', bbox_to_anchor=(0.045, 1), )
        plt.legend(loc='upper left', bbox_to_anchor=manual_bbox, )

    else:
        plt.legend()


def store_fit(mode, epoch_id, outdir, logfile, fitmod=None):
    '''
    plots and saves various informations about a fit
    '''

    # Since the automatic rescaling goes haywire when using the add command, we manually rescale (with our own custom command)
    rescale(auto=True)

    Plot_screen("ldata,ratio,delchi", outdir + '/' + epoch_id + '_screen_xspec_' + mode,
                includedlist=None if fitmod is None else fitmod.includedlist)

    # saving the model str
    catch_model_str(logfile, savepath=outdir + '/' + epoch_id + '_mod_' + mode + '.txt')

    if os.path.isfile(outdir + '/' + epoch_id + '_mod_' + mode + '.xcm'):
        os.remove(outdir + '/' + epoch_id + '_mod_' + mode + '.xcm')

    # storing the current configuration and model
    Xset.save(outdir + '/' + epoch_id + '_mod_' + mode + '.xcm', info='a')

def Plot_screen(datatype,path,mode='matplotlib',xspec_windid=None,includedlist=None,xlims=None,ylims=None):

    '''Saves a specific xspec plot, either through matplotlib or through direct plotting through xspec's interface'''

    if not path.endswith('.png') or path.endswith('.svg') or path.endswith('.pdf'):
        path_use=path+('.pdf' if mode=='matplotlib' else '.png')

    if mode=='matplotlib':
        xPlot(datatype,includedlist=includedlist,xlims=xlims,ylims=ylims)


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
    plot_values=np.array([[[None,None]]*2]*max(AllData.nGroups,1))
    for i_grp in range(1,max(2,AllData.nGroups+1)):
        plot_values[i_grp-1][0][0]=np.array(Plot.x(i_grp))
        plot_values[i_grp-1][0][1]=np.array(Plot.xErr(i_grp))

        if 'mo' in datatype:
            plot_values[i_grp-1][1][0]=np.array(Plot.model(i_grp))
        else:
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


        mod_values=np.array([[None]*(n_addcomps+2)]*max(AllData.nGroups,1))
        '''
        for each data group, plot_values contains:
            -the x value for the model with the current datatype
            -the y value for the model with the current datatype
            -the y values for all the model components
        '''

        for i_grp in range(1,max(2,AllData.nGroups+1)):

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


def plot_std_ener(ax_ratio, ax_contour=None, plot_em=False, mode='ratio',exclude_last=False,plot_indiv_transi=False,
                  squished_mode=False,force_side='none',alpha_line=1.,
                  noname=False,noline=False,color='default'):
    '''
    Plots the current absorption (and emission if asked) standard lines in the current axis
    also used in the autofit plots further down

    -plot_indiv_transi:
            -True/False: plots only/none of the individual transitions for instead of the averaged energies
            -prio_resolved: plots the resolved transitions when available, otherwise the non-resolved
            -only[X+Y+...]: only plots the resolved lines for X, Y, ...
    -squished mode: makes the absorption text slightly higher to avoid it going lower than the plot

    -force_side: shows all lines either in emission if 'em' or in absorption if 'abs'
    '''
    # since for the first plot the 1. ratio is not necessarily centered,
    # we need to fetch the absolute position of the y=1.0 line in graph height fraction
    # pos_ctr_ratio = 0.5 if mode=='chimap' else\
    #                 (1 - ax_ratio.get_ylim()[0]) / (ax_ratio.get_ylim()[1] - ax_ratio.get_ylim()[0])

    if ax_ratio.get_yscale() == 'log':
        pos_ctr_ratio = np.log10(1 / ax_ratio.get_ylim()[0]) / (np.log10(ax_ratio.get_ylim()[1]/ax_ratio.get_ylim()[0]))
    elif ax_ratio.get_yscale() == 'linear':
        pos_ctr_ratio = (1 - ax_ratio.get_ylim()[0]) / (ax_ratio.get_ylim()[1] - ax_ratio.get_ylim()[0])

    lines_names = np.array(lines_std_names)


    lines_abs_pos = ['abs' in elem for elem in lines_names]
    lines_em_pos = ['em' in elem for elem in lines_names]

    indiv_lines_nonresolved_std=np.unique([' '.join(elem.split(' ')[:2]) if len(elem.split(' '))<=2
                                    else '' for elem in lines_std.values()])
    indiv_lines_resolved_std=np.unique([' '.join(elem.split(' ')[:2]) if len(elem.split(' '))>2
                                    else '' for elem in lines_std.values()])

    indiv_lines_both_std=[elem for elem in indiv_lines_resolved_std if elem in indiv_lines_nonresolved_std]

    indiv_lines_both=[elem for elem in lines_names if lines_std[elem] in indiv_lines_both_std]

    #removing or restricting to resolved lines for the emission
    lines_resolved_mask=['(' in lines_std[elem] and ')' in lines_std[elem] for elem in lines_names]
    lines_resolved=lines_names[lines_resolved_mask]

    if type(plot_indiv_transi)==str and 'only' in plot_indiv_transi:
        plot_indiv_transi_lines=plot_indiv_transi.split('only')[1].split('+')
        lines_resolved_restrict=[elem for elem in lines_resolved if
                        sum([elem.startswith(subelem) for subelem in plot_indiv_transi_lines])>0]

    for i_line, line in enumerate(lines_names):

        if lines_e_dict[line][0] < ax_ratio.get_xlim()[0] or lines_e_dict[line][0] > ax_ratio.get_xlim()[1]:
            continue

        # skipping redundant indexes
        if line in ['FeKa25em','FeKa26em','FeKaem','FeKbem','FeKa1em','FeKb1em','calNICERSiem','FeDiazem']:
            continue

        # skipping Nika27, FeKa25em, FeKa26em:
        if i_line in [5,9,10]:
            continue

        # skipping display if emission lines are not asked
        if 'em' in line and not plot_em:
            continue

        if plot_indiv_transi=='prio_resolved':
            if  line in indiv_lines_both and not ('(' in lines_std[line] and ')' in lines_std[line]):
                continue
        elif type(plot_indiv_transi)==str and plot_indiv_transi.startswith('only'):
            if line in lines_resolved and line not in lines_resolved_restrict:
                continue
        else:
            if not 'em' in line and (line not in lines_resolved and plot_indiv_transi==True \
                                 or line in lines_resolved and plot_indiv_transi==False):
                continue



        # booleans for dichotomy in the plot arguments
        abs_bool = 'abs' in line and 'abs' in force_side
        em_bool = not abs_bool or 'em' in force_side

        if not noline:
            # plotting the lines on the two parts of the graphs
            ax_ratio.axvline(x=lines_e_dict[line][0],
                             ymin=ax_ratio.get_ylim()[0] if mode=='misc' else (0 if mode not in ['ratio','chimap'] else pos_ctr_ratio if not abs_bool else 0.),
                             ymax=ax_ratio.get_ylim()[1] if mode=='misc' else (1 if mode not in ['ratio','chimap'] else pos_ctr_ratio if not em_bool else 1.),
                             color=color if color!='default' else 'blue' if em_bool else 'brown',
                             linestyle='dashed', linewidth=1.5,zorder=-1,alpha=alpha_line)
            if ax_contour is not None:
                ax_contour.axvline(x=lines_e_dict[line][0], ymin=0.5 if em_bool else 0, ymax=1 if em_bool else 0.5,
                                   color=color if color!='default' else 'blue' if em_bool else 'brown', linestyle='dashed', linewidth=0.5)

        # small left horizontal shift to help the Nika27 display
        txt_hshift = 0.1 if 'Ni' in line else 0.006 if 'Si' in line else 0
        txt_line=lines_std[line]

        line_x_text=lines_e_dict[line][0] - txt_hshift

        if color=="grey":
            y_line_pos=(i_line+1)%2
        if not noname:

            add_height_squished=0
            if plot_indiv_transi:
                line_full_name=lines_std[line]

                #ensuring the line is an individual transition
                if '(' in line_full_name and ')' in line_full_name:

                    #additional tester to move line_x_tex if none of the conditions below are true
                    bool_shift=True

                    #shifting it to the left if is the first one of a complex
                    if i_line<len(lines_names)-1 and lines_std[lines_names[i_line+1]].split('(')[0]\
                                                   ==lines_std[lines_names[i_line]].split('(')[0]:

                        #line_x_text=lines_e_dict[line][0]-0.01*len(lines_std[line])/12 is good for 6.3-7.1
                        line_x_text=lines_e_dict[line][0]-0.01*len(lines_std[line])/12-\
                                    (0.015 if 'P' in lines_std[lines_names[i_line]].split('(')[1] else 0)

                        #avoiding overlap with the NiKa27 complex display
                        if line=='FeKb25p1abs':
                            txt_line = '(' + lines_std[line].split('(')[1]
                            line_x_text+=0.04
                        bool_shift=False
                    #removing everything except the complex name otherwise
                    if i_line>0 and lines_std[lines_names[i_line]].split('(')[0]==\
                                    lines_std[lines_names[i_line-1]].split('(')[0]:
                        line_x_text=lines_e_dict[line][0]+0.008+\
                        (0.01 if 'P' in lines_std[lines_names[i_line]].split('(')[1] else 0)

                        if line!='FeKb25p3abs':
                            txt_line = '(' + lines_std[line].split('(')[1]
                        else:
                            line_x_text+=0.03
                        bool_shift=False
                        add_height_squished=0.01 if  'P' in lines_std[lines_names[i_line]].split('(')[1] else 0

                    if bool_shift:
                        line_x_text+=0.04

                    if  color=='grey':
                        if line.startswith('FeKa25') and line.endswith('abs'):

                            y_line_pos=(y_line_pos+1)%2
                            line_transi_letter=line.replace('FeKa25','').replace('abs','')
                            if line_transi_letter=='Z':
                                line_x_text-=0.03
                            if line_transi_letter=='Y':
                                line_x_text-=0.01
                            if line_transi_letter=='X':
                                line_x_text-=0.005
                            if line_transi_letter=='W':
                                line_x_text-=0.002

                        if line=='FeKb1p1em':
                            y_line_pos=(y_line_pos+1)%2
                            line_x_text-=0.02
                else:
                    line_x_text+=0.04
            else:
                line_x_text += 0.04

            # but the legend on the top part only
            ax_ratio.text(x=line_x_text,
                          y=1.1 + 0.08*(y_line_pos)%2 if color=='grey' else 0.96 if not abs_bool else (0.06+(0.02+add_height_squished if squished_mode else 0)
                                                  if i_line % 2 == 1 else 0.12+(0.01 if squished_mode else 0)),
                          s=txt_line,
                          color=color if color!='default' else 'blue' if em_bool else 'brown',
                          transform=ax_ratio.get_xaxis_transform(), ha='center',
                          fontsize=None,
                          va='top')

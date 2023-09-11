#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Sep 12 15:49:00 2022

@author: parrama

simulation tools, separated from the rest to have few imports


"""

import os,sys,io
import numpy as np
import time
import glob
import subprocess

#trapezoid integration
from scipy.integrate import trapezoid
from scipy.interpolate import griddata


sys.path.extend(['/home/parrama/Documents/Work/PhD/Scripts/Python/general'])
from general_tools import file_edit

from tqdm import tqdm


from solution_tools import func_density_sol,func_nh_sol,func_r_boost_sol,func_density_relat_sol,\
                           func_vel_sol,func_logxi_sol,func_E_deboost_sol,func_lum_deboost_sol

from grid_tools import upload_mantis,download_mantis

sys.path.extend(['/home/parrama/Documents/Work/PhD/Scripts/Python/modeling/PyXstar'])
import pyxstar as px

# #adding some libraries 
# os.environ['LD_LIBRARY_PATH']+=os.pathsep+'/home/parrama/Soft/Heasoft/heasoft-6.31.1/x86_64-pc-linux-gnu-libc2.31/lib'

h_cgs = 6.624e-27
eV2erg = 1.6021773E-12
erg2eV = 1.0/eV2erg
Ryd2eV = 13.605693

compton_thick_thresh=1.5e24

# ! light speed in Km/s unit
c_Km = 2.99792e5
# ! light speed in cm/s unit
c_cgs = 2.99792e10
sigma_thomson_cgs = 6.6525e-25
c_SI = 2.99792e8
G_SI = 6.674e-11
Msol_SI = 1.98892e30
PI = 3.14159265
Km2m = 1000.0
m2cm = 100.0



# def produce_df(data, rows, columns, row_names=None, column_names=None, row_index=None, col_index=None):
#     """
#     rows is a list of lists that will be used to build a MultiIndex
#     columns is a list of lists that will be used to build a MultiIndex
#
#     Note:
#     replaces row_index and col_index by the values provided instead of building them if asked so
#     """
#
#     if row_index is None:
#         row_index_build = pd.MultiIndex.from_product(rows, names=row_names)
#     else:
#         row_index_build = row_index
#
#     if col_index is None:
#         col_index_build = pd.MultiIndex.from_product(columns, names=column_names)
#     else:
#         col_index_build = col_index
#
#     return pd.DataFrame(data, index=row_index_build, columns=col_index_build)

# def df_mhd_solution(solutions_path):
#
#     '''
#
#     TO BE IMPLEMENTED MAYBE, UNUSED FOR NOW BECAUSE FUCK 5 DIMENSIONNAL RAGGED OBJECTS
#     Returns a multi dimensionnal dataframe containing all the solutions inside the folder type
#     the solutions folders should all be of the type 'eps_'+epsvalue
#     inside, subfolders with each n, and inside each solution file
#     '''
#
#     #extracting the solution directories
#     eps_dir_list=[elem for elem in glob.glob(os.path.join(solutions_path,'*/'))\
#                if elem[:-1].split('/')[-1].startswith('eps_')]
#
#     #indexing the array by hand since it is irregular
#     glob_sol_array=np.array([None]*len(eps_dir_list))
#
#     glob_sol_parameters=np.array([None]*len(eps_dir_list))
#     for i_eps,eps_dir in enumerate(eps_dir_list):
#
#         #fetching subdirectories for each N (island number)
#         n_dir_list=glob.glob(os.path.join(eps_dir,'*/'))
#         elem_eps_array=np.array([None]*len(n_dir_list))
#         elem_eps_parameters=np.array([None]*len(n_dir_list))
#
#         #ordering the islands
#         n_dir_list.sort()
#
#         for i_n,n_dir in enumerate(n_dir_list):
#
#             #fetching files (each solution)
#             sol_list=glob.glob(os.path.join(n_dir,'**'))
#
#             elem_sol_array=np.array([None]*len(sol_list))
#             elem_sol_parameters=np.array([None]*len(sol_list))
#
#             for i_sol,sol in enumerate(sol_list):
#                 #fetching the full data
#                 elem_sol_array[i_sol] = np.loadtxt(os.path.join(sol), skiprows=14)
#
#                 #and the header
#                 with open(sol) as sol_file:
#                     sol_lines_header=sol_file.readlines()[1:11]
#
#
#                 elem_sol_parameters[i_sol]=np.array([elem.split()[-1] for elem in sol_lines_header]).astype(float)
#
#             #trying to re-index regularly the arrays whenever possible
#             elem_eps_array[i_n]=np.array([elem for elem in elem_sol_array],dtype='object')
#             try:
#                 elem_eps_array[i_n]=elem_eps_array[i_n].astype(float)
#             except:
#                 pass
#
#             elem_eps_parameters[i_n]=np.array([elem for elem in elem_sol_parameters],dtype='object')
#             try:
#                 elem_eps_parameters[i_n]=elem_eps_parameters[i_n].astype(float)
#             except:
#                 pass
#
#         glob_sol_array[i_eps]=np.array([elem for elem in elem_eps_array],dtype='object')
#         try:
#             glob_sol_array[i_eps] = glob_sol_array[i_eps].astype(float)
#         except:
#             pass
#
#         glob_sol_parameters[i_eps]=np.array([elem for elem in elem_eps_parameters],dtype='object')
#         try:
#             glob_sol_parameters[i_eps] = glob_sol_parameters[i_eps].astype(float)
#         except:
#             pass
#
#     glob_sol_array=np.array([elem for elem in glob_sol_array],dtype='object')
#     try:
#         glob_sol_array = glob_sol_array.astype(float)
#     except:
#         pass
#
#     glob_sol_parameters=np.array([elem for elem in glob_sol_parameters],dtype='object')
#     try:
#         glob_sol_parameters = glob_sol_parameters.astype(float)
#     except:
#         pass
#
#
#     eps_values=[float(elem.split('/')[-2].split('_')[1].replace('0','0.')) for elem in eps_dir_list]
#
#     #assuming regularity for now (aka all eps_values dirs have the same n_dirs)
#
#     n_values=[int(elem.split('/')[-2].split('n')[1]) for elem in n_dir_list]
#
#
#     #sol_lines=np.array(len(sol_files))



def oar_wrapper(solution_rel_dir,save_grid_dir,sim_grid_dir,
                  mdot_obs,xlum,m_BH,
                  ro_init, dr_r,stop_d_input,v_resol,
                  mode='server_standalone_default',
                  progress_file='',
                  sol_file='auto',
                  SED_file='auto_.dat'):

    '''
    wrapper for grid xstar computations

    Note: the first parameter is used to define grid process names so it must be different for each process,
          hence why we put the solution grid directory, which are different for every solution

          Assuming the naming conventions allow to have a shorter parameter file

    Args:
        solution_rel_dir: solution relative directory inside the grid structure
                          (will be added to both save_dir and simdir)

        save_grid_dir: save grid directory, where all the files will be saved

        sim_grid_dir: computation grid directory, where all the xstar computations will be run

        mdot_obs,m_BH,xlum: SED parameters for xstar
        ro_init,dr_r,v_resol,stop_d_input: box parameters for xstar

        mode: changes the computation behavior: 'type_xstaruse_xstaruseid'

            type:
            -cigrid: uses mantis as a save folder. save_grid_dir is expected to be a mantis absolute path
            -server: uses a standard save folder with normal copying commands

            xstar use:
            -standalone: uses an xstar version directly installed within an heasoft folder.
            -docker: uses an xstar version installed in a docker. Uses smart copying to avoid necessiting permissions
                    (besides the one to run the docker)
            -charliecloud: uses an xstar version installed in a charliecloud environment.

            xstarid:
            in standalone: the path of the heasoft version to use. 'default' uses the standard version installed on the computer
            in docker/charliecloud: the identifier of the container (not the image) to launch

        sol_file: naming of the sol file inside solution_rel_dir. If set to 'auto',
                 assumes the solution file name from the directory structure
        SED_file: naming of the SED file inside solution_rel_dir. If set to 'auto_fileextension',
                assumes the SED file name from the directory structure, using the extension fileextension
    '''

    if sol_file=='auto':
        solution_name=('/').join(solution_rel_dir[solution_rel_dir.rfind('rj'):].split('/')[1:]).replace('/','_')+'.txt'
        solution_rel_path=os.path.join(solution_rel_dir,solution_name)
    else:
        solution_rel_path=sol_file

    if SED_file.startswith('auto'):
        SED_extension=SED_file.split('_')[1]
        SED_name=solution_rel_dir[:solution_rel_dir.rfind('mdot')].split('/')[-1][:-1]+SED_extension

        #cutting the directory just above the epsilon directories, which are the first directories below the SED
        SED_rel_dir=('/').join(solution_rel_dir[:solution_rel_dir.rfind('eps')].split('/')[:-1])
        SED_rel_path=os.path.join(SED_rel_dir,SED_name)
    else:
        SED_rel_path=SED_file

    simdir=os.path.join(sim_grid_dir,solution_rel_dir)

    save_dir=os.path.join(save_grid_dir,solution_rel_dir)

    os.system('mkdir -p '+simdir)

    #splitting the mode informations
    comput_mode=mode.split('_')[0]
    xstar_mode=mode.split('_')[1]

    #this syntax avoids issues if there are _ in the path/identifier
    xstar_loc=mode[mode.find(xstar_mode)+len(xstar_mode)+1:]

    #copying all the initial files and the content of the mantis directory to the simdir
    if comput_mode=='cigrid':

        #downloading the elements in the save directory
        ####THIS SHOULD BE CHANGED TO ONLY DOWNLOAD THE LAST spectrum
        download_mantis(save_dir,simdir,load_folder=True)

        #and the SED
        download_mantis(os.path.join(save_grid_dir,SED_rel_path),simdir)

        #this shouldn't be needed
        #download_mantis(os.path.join(mantis_grid_dir, solution_rel_path), simdir)

    elif comput_mode=='server':

        sp_saves=glob.glob(save_dir+'/sp_**')

        #copying everything but the saves from the save_dir to the sim_dir
        for elem_file in [elem for elem in glob.glob(save_dir+'/**') if elem not in sp_saves]:
            os.system('cp '+os.path.join(save_dir,elem_file)+' '+simdir)

        #copying the last non-final incident and rest spectra to restart the computation if necessary
        sp_saves_rest=np.array([elem for elem in sp_saves if '_tr_rest_' in elem and '_final_' not in elem])
        if len(sp_saves_rest)>0:
            sp_saves.sort()
            os.system('cp '+os.path.join(save_dir,sp_saves_rest[-1])+' '+simdir)

            #we also copy the spectrum of the n-1 box to avoid issues when restarting from the last box
            if len(sp_saves_rest)>1:
                os.system('cp ' + os.path.join(save_dir, sp_saves_rest[-2]) + ' ' + simdir)

        sp_saves_incid = np.array([elem for elem in sp_saves if '_incid_' in elem and '_final_' not in elem])
        if len(sp_saves_incid) > 0:
            sp_saves.sort()
            os.system('cp ' + os.path.join(save_dir, sp_saves_incid[-1]) + ' ' + simdir)

        #copying the SED file
        os.system('cp '+os.path.join(save_grid_dir,SED_rel_path)+' '+simdir)


    #extracting the name for the function call below since we're going in simdir
    SED_name=SED_rel_path.split('/')[-1]
    solution_name=solution_rel_path.split('/')[-1]

    #creating the path of the progress file if asked to
    if progress_file=='auto':
        if comput_mode=='server':
            #in server mode, it's fine to put the logs in the save folder because we can access it easily
            progress_file_path=os.path.join(save_grid_dir,solution_rel_dir.split('/')[0],'grid_progress.log')
        elif comput_mode=='cigrid':
            progress_file_path = os.path.join(sim_grid_dir,solution_rel_dir.split('/')[0],'grid_progress.log')
    else:
        progress_file_path=None if progress_file=='' else progress_file

    #it's easier to go directly in the simdir here
    os.chdir(simdir)

    if mdot_obs.isdigit():
        mdot_obs_use=float(mdot_obs)
    else:
        mdot_obs_use=mdot_obs

    xstar_wind(solution_name,SED_path=SED_name,xlum=xlum,mdot_obs=mdot_obs_use,outdir='./',
               m_BH=m_BH,
               ro_init=ro_init,dr_r=dr_r,stop_d_input=stop_d_input,v_resol=v_resol,
               comput_mode=comput_mode,
               xstar_mode=xstar_mode,
               save_folder=save_dir,
               xstar_loc=xstar_loc,
               progress_file=progress_file_path)


def xstar_func(spectrum_file,lum,t_guess,n,nh,xi,vturb_x,nbins,nsteps=1,niter=100,lcpres=0,
               path_logpars=None,
               comput_mode='local',xstar_mode='standalone',xstar_loc='default',
               dict_box=None,save_folder='',no_write=False,extract_transmitted=False):
    
    '''
    wrapper around the xstar function itself with explicit calls to the parameters routinely being changed in the computation

    non-direct arguments:
        dict-box: information for the box number
        comput_mode/xstar_mode/xstar_loc: identical to the comput modes of the other functions
    
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
    
    #if the current box is the restarting box, we simply restart the computation with the parameter file already existing
    #however we also recreate the parfile for the final computation, because it shouldn't retake an existing parfile

    if dict_box is not None:
        nbox=dict_box['nbox']
        i_box_final=dict_box['i_box_final']
        dr_r_eff_list=dict_box['dr_r_eff_list']
        v_resol=dict_box['v_resol']
        xlum=dict_box['xlum']
        lum_corr_factor=dict_box['xlum_corr']
    else:
        nbox=''
        i_box_final=''
        xlum=0


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
    xpar['lcpres']=lcpres
    
    #secondary parameters
    xhpar['nsteps']=nsteps
    
    #and putting this to hopefully not 0
    xhpar['niter']=niter

    #turbulent speed
    xhpar['vturbi']=vturb_x

    xhpar['lprint']=0

    #turning on or off the writing of the output files if necessary
    if no_write:
        #no writing at all
        xhpar['lwrite']=-1
    else:
        #default fits writing without too much details
        xhpar['lwrite']=0

    parlog_header=['#xlum_init = %.3e'%(xlum)+' *1e38 erg/s | v_resol = '+str(v_resol)+' km/s | nbins = '+str(nbins)+'\n',
                   '#nsteps = '+str(nsteps)+'\tniter = '+str(niter)+'\n',
                   '#Remember logxi is shifted to give xstar the correct luminosity input and the density at the half-box radius\n',
                   '############################################################################################################\n',
                   '#nbox\tnbox_final\tspectrum\tlum\tlum_corr_factor\tt_guess\tn\tnh\tlogxi\tvturb_x\tdr_r\tt_run\n']

    # we don't save the gaz frame spectra to avoid storing too much data
    # first save before the xstar run
    if comput_mode in ['server', 'cigrid']:
        if comput_mode == 'cigrid':
            upload_mantis(spectrum_file, save_folder, delete_previous=True)
        elif comput_mode == 'server':
            os.system('cp ' + spectrum_file + ' ' + save_folder)

    if path_logpars is not None:
        parlog_str='\t'.join([str(nbox),str(i_box_final),spectrum_file,'%.6e'%lum,'%.6e'%lum_corr_factor,
                              '%.6e'%t_guess,'%.6e'%n,'%.6e'%nh,'%.6e'%xi,'%.6e'%vturb_x,
                              '%.6e'%dr_r_eff_list[nbox-1]])+'\n'
        
        file_edit(path_logpars,'\t'.join([str(nbox),str(i_box_final),spectrum_file]),parlog_str,parlog_header)


    if xstar_mode=='standalone':

        #using xstar_loc for the path of the headas folder where to run xstar
        px.run_xstar(xpar, xhpar, headas_folder=xstar_loc)

    else:

        #in order to ensure we're not gonna mix the xstar runs, we make a global identifier with the name
        #of the grid and the solution
        identifier_str=os.getcwd()[os.getcwd().find('grid'):]
        identifier_str=identifier_str.replace('/','_')

        #using xstar_loc for the name of the xstar container to create an image of

        if xstar_mode=='docker' and xstar_loc=='default':
            px.container_run_xstar(xpar, xhpar,mode=xstar_mode,dentifier=identifier_str)
        else:
            px.container_run_xstar(xpar, xhpar, mode=xstar_mode, container=xstar_loc,identifier=identifier_str)

    #storing the lines of the xstar log file
    with open('xout_step.log') as xlog:
        xlog_lines = xlog.readlines()

    #re-editing the file to add elapsed time
    if path_logpars is not None:

        xrun_time=str(round(float(xlog_lines[-1].split()[-1])))
        parlog_str=parlog_str.replace('\n','\t'+xrun_time+'\n')
        
        file_edit(path_logpars,'\t'.join([str(nbox),str(i_box_final),spectrum_file]),parlog_str,parlog_header)

    #compacting the current xstar log file to a global log file
    with open('./xout_log_global.log',mode='a') as xlog_global:
        xlog_global.write('\n\n************************\n BOX:'+str(nbox)+'   FINAL_BOX:'+str(i_box_final)+'\n\n')
        xlog_global.writelines(xlog_lines)

    #deleting the current log file
    os.remove('xout_step.log')

    #extracting the output spectrum if asked to
    if extract_transmitted:
        px.LoadFiles()
        out_sp=px.ContSpectra()
        out_arr= np.array([out_sp.energy,out_sp.transmitted]).T

        # !**Writing the shifted spectra in a file as it is input for next box
        np.savetxt('./xout_transmitted_'+str(nbox)+'_'+str(i_box_final)+'.txt', out_arr,
                   header=str(len(out_arr)), delimiter='  ', comments='')

    # second save for the modified logpar and the log file
    if comput_mode in ['server','cigrid']:

        if comput_mode=='cigrid':
            upload_mantis(path_logpars, save_folder)
            upload_mantis('./xout_log_global.log',save_folder)
        elif comput_mode=='server':
            os.system('cp '+path_logpars+' '+save_folder)

def xstar_wind(solution,SED_path,xlum,outdir,
               mdot_obs='auto',p_mhd_input=None,m_BH=8,
               ro_init=6.,dr_r=0.05,stop_d_input=1e6,v_resol=85.7,
               chatter=0,reload=True,
               comput_mode='local',xstar_mode='standalone',xstar_loc='default',
               save_folder='',
               force_ro_init=False,no_turb=False,cap_dr_resol=True,no_write=False,
               grid_type="standard",custom_grid_headas=None,progress_file=None):
    
    
    '''
    Python wrapper for the xstar computation of a single solution
    
    The box size is computed dynamically to "satisfy" two criteria, a maximal dr/r and a velocity resolution
    The velocity resolution should always be taken with a reasonable oversampling factor (at least 1/3) compared to the
    instrumental resolution


    Required parameters:

        solution is either a file path or a dictionnary with all the arguments of a JED-SAD solution

        p_mhd and mdot_obs are the main parameters outside of the JED-SAD solution

        if mdot_obs is set to auto, assumes the mdot value as a fraction of the Eddington luminosity
        with the provided Black Hole Mass

        p_mhd is generally given in the solution, but if it's not (in local),
         it can be directly inputted through a parameter
        in server/cigrid mode, p_mhd is directly in the solution file and as such


        stop_d is a single (or list of) stop distances in units of Rg
    
        SED is the path of the incident spectra (suitable for xstar input)
        Note: The normalization doesn't matter

        xlum is the bolometric luminosity of the spectrum in units of 1e38ergs/s between 1-1000 Ryd (aka 13.6eV-13.6keV)
        Used internally by xstar to renormalize the given SED
        
        The luminosity value is manually updated after each computation to consider the decrease due to absorption
        
        #### Note: should be updated with the speed of the material and potential significant absorption?
        

    Secondary options:
    
        h_over_r is the aspect ratio of the disk (unused for now)
        
        ro_init gives the initial value of ro_by_rg
        
        dr_r is the standard radial shift between to boxes, provided it doesn't give a turbulent velocity higher than v_resol
    
        v_resol gives the desired spectral resolution desired AND a threshold of turbulent velocity for the box size
                        (the spectral resolution is converted into a number of continuum bins used for writing the xstar spectra)
             the computing time inversely scales with vsol
             
             85.7 gives 45154 bins and corresponds to a microcalorimeter at a deltaE of 6eV (XRISM's resolution) with an oversampling of 3
                                                                                             
        chatter gives out the number of infos displayed during the computation
    
        reload determines if the computation automatically detects if it has been stopped at a given box and restarts from it, or instead
        relaunches entirely the computation

        force_ro_init: ignores the logxi<6 criteria and starts the boxes at ro_init

        no_turb: no turbulent speed in the computation

        cap_dr_resol: use (or not) the v_resol criteria as another cap on the dr on top of the dr/r given

        grid_type: assumes either a standard or custom grid
                    standard uses the standard xstar grid definition
                    (main grid for 0.1eV-400keV, coarser grid above up to 1MeV)

                    custom uses custom functions with two sandwiched grids
                    The main sampled one is only between 0.1keV and 10keV, and the coarser one covers the rest
                    (aka 0.1eV-0.1keV and 10keV-1MeV)

                    it is assumed that a custom grid_type will also have an xstar_loc pointing to a folder with
                    the corresponding grid type setup

    comput_mode:
        -local:
            standard behavior, computes everything in the outdir directory

        -server/cigrid:

            setups for grid computation on servers.
            The main difference is the way to run xstar and how saves are implemented

            the grid folder need to start by "grid" to get a recognizable identifier for the xstar_id and the progress_id

            cigrid (using Cigri on Dahu & Bigfoot)
                still in progress

                uses charliecloud instead of docker for launching xstar
                to be implemented for Luke:
                    -fetches from Mantis the current last input spectrum and parameter files when starting a job
                    -saves to Mantis the current input spectrum before each xstar run
                
                -saves each final transmitted spectrum, par log file, global xout log, (obtained through compacting of each individual xout log),
                and box data files to either  at the "save_folder" location.
                -(with "clean" option) cleans all the individual spectra at the end of the task to gain space

            server:
                same behavior but uses a save_dir in a normal arborescence


    xstar_mode:
    -standalone: uses an xstar version directly installed within an heasoft folder.
    -docker: uses an xstar version installed in a docker. Uses smart copying to avoid necessiting permissions
            (besides the one to run the docker)
    -charliecloud: uses an xstar version installed in a charliecloud environment.

    xstar_loc:
    in standalone: the path of the heasoft version to use. 'default' uses the standard version installed on the computer

    in docker/charliecloud: the identifier of the container (not the image) to launch
                            if default, merges the directory structure from the grid folder (which should start with grid)

    progress_file:
        -global log file for grid computation where the tqdm of all grid files are listed
        Useful to see how big computations are progressing
        if progress_file is not set to None, the tqdm display is redirected and doesn't appear in stderr

    Notes on the python conversion:
        -since array numbers starts at 0, we use "index" box numbers (starting at 0) and adapt all of the consequences,
        but still print the correct box number (+1)


    ####SHOULD BE UPDATED TO ADD THE JED SAD N(R) if necessary

    '''

    def shift_transmitted_sp_gaz_rest(psi, path, incident_path,xlum_eff):

        '''
        shifts the current xstar transmitted spectrum (in the gaz frame) to the rest frame

        This requires knowledge of both the incident spectrum (rest frame) AND the transmitted spectrum (gaz frame)

        Uses formulae from Luminari2020

        Starts from an xstar output files and stores it
        (in a xstar-accepted manner) in the "path" file

        Works with easy integration of the xstar file because the unit is ergs/s/cm²/erg,
        aka ergs*cst after integration

        the xstar output files has x axis in units of eV and y axis in units of 1e38erg/s/erg
        so we need to integrate and the x axis must be renormalized to ergs (so with the conversion factor below)
        the renormalization considers 1-1000 Rydbergs only, so we ensure to only consider thart part of the spectrum

        '''

        # loading the transmitted spectrum of the current box
        prev_box_sp = px.ContSpectra()

        eptmp_trans = np.array(prev_box_sp.energy)
        zrtmp_trans = np.array(prev_box_sp.transmitted)

        # loading the energy and spectrum from the input spectrum file (which should be in xstar form)
        eptmp_incid, zrtmp_incid = np.loadtxt(incident_path, skiprows=1).T

        '''
        the end point is the rest frame, so in energies this is the incident (rest frame) non-shifted grid
        the issue is that for the first computation the starting grid is not xstar-like
        Thus we interpolate the incident spectrum on the xstar grid before adding it
        
        NOTE: This means we "lose" all of the SED beyond the xstar grid bounds, 
              and the luminosity evolves accordingly. This is expected to be negligible
              
        '''
        #this is the xstar output grid (always the same)
        eptmp_relat = eptmp_trans

        #length condition avoids error when comparing elements in arrays of different lengths
        if len(eptmp_incid)!=len(eptmp_trans) or not np.all(eptmp_incid==eptmp_trans):
            zrtmp_incid_interp=10**griddata(np.log10(eptmp_incid),np.log10(zrtmp_incid),np.log10(eptmp_relat),
                                            method='linear',fill_value=-10)

            #renormalizing the incident flux in case the starting spectrum isn't normalized
            zrtmp_incid_interp=zrtmp_incid_interp*xlum_eff/trapezoid(zrtmp_incid_interp,x=eptmp_relat * eV2erg)
        else:
            zrtmp_incid_interp=zrtmp_incid

        '''
        true relativistic expression of the transmitted spectrum

        the *1/psi expression in the Luminari2020 is misleading
        It has to be used to doppler shift the frequencies of the transmitted spectra before adding it back to the rest
        this means we need to interpolate once more
        '''

        #ignoring the divided by 0 warning because this is just the result of using log on 0 values
        # (which can be there for bins with no flux)
        with np.errstate(divide='ignore'):
            zrtmp_trans_rest=10**griddata(np.log10(eptmp_relat/psi),np.log10(zrtmp_trans),np.log10(eptmp_relat),
                                                method='linear',fill_value=-10)

        zrtmp_relat= zrtmp_incid_interp*(1-psi**3)+zrtmp_trans_rest

        # should not need to remap the spectrum since it will be done internally by xstar if necessary
        shifted_input_arr = np.array([eptmp_relat, zrtmp_relat]).T

        # !**Writing the shifted spectra in a file as it is input for next box
        np.savetxt(path, shifted_input_arr, header=str(len(eptmp_relat)), delimiter='  ', comments='')

        #new value of the luminosity (rest frame)
        xlum_bol= trapezoid(zrtmp_relat,x=eptmp_relat * eV2erg)

        return xlum_bol

    def shift_incident_sp_rest_gaz(psi, path, incident_path):

        '''
        shifts the current incident spectrum (in the rest frame) to the gaz frame

        Uses formulae from Luminari2020

        Starts from xstar output type files and stores the result (in a xstar-accepted manner) in the "path" file

        Works with easy integration of the xstar file because the unit is ergs/s/cm²/erg,
        aka ergs*cst after integration
        Works independantly of the normalisation of the spectra

        For the first box, we don't load the xstar file but the initial spectrum instead,
        in which case we load from the "origin" path file
        '''

        # loading the energy and spectrum from the incident spectrum file (which should be in xstar form)
        eptmp, zrtmp = np.loadtxt(incident_path, skiprows=1).T

        #converting the energy and blueshift

        eptmp_shifted = eptmp * psi

        # multiplying the spectrum is not useful because it's normalized anyway but this way the files are correct
        zrtmp_shifted = zrtmp * psi**3

        # should not need to remap the spectrum since it will be done internally by xstar
        shifted_input_arr = np.array([eptmp_shifted, zrtmp_shifted]).T

        # !**Writing the shifted spectra in a file
        np.savetxt(path, shifted_input_arr, header=str(len(eptmp)), delimiter='  ', comments='')

        '''
        computing the ratio between the total luminosity and the luminosity in the xstar range
        
        Xstar renormalizes the spectrum according to its luminosity in 1-1000 Rydberg
        The Luminosity we use everywhere is the bolometric luminosity
        
        Thus, we need to re-normalize the luminosity (and the ionization parameter)
        given as an input to xstar according to the ratio
        between the 1-1000Ryd luminosity and the bolometric luminosity
        
        (note that this doesn't need the normalization to be correct as this is a relative computation)
        
        Note: we add an interpolation in case the initial spectrum or xstar isn't 
        '''

        eptmp_shifted_highres=np.logspace(np.log10(eptmp_shifted[0]),np.log10(eptmp_shifted[-1]),int(1e6))

        zrtmp_shifted_interp =10**(griddata(np.log10(eptmp_shifted),np.log10(zrtmp_shifted),
                                            np.log10(eptmp_shifted_highres), method='linear',fill_value=-10))

        energy_mask = (eptmp_shifted_highres / Ryd2eV > 1) & (eptmp_shifted_highres / Ryd2eV < 1000)

        #note: both absolute values of these luminosities can be false is the spectrum isn't correctly normalized,
        #but this doesn't affect the ratio of luminosities
        xlum_xstar_range = trapezoid(zrtmp_shifted_interp[energy_mask],
                                     x=eptmp_shifted_highres[energy_mask] * eV2erg)

        xlum_bol= trapezoid(zrtmp_shifted_interp,x=eptmp_shifted_highres * eV2erg)

        return xlum_xstar_range/xlum_bol

    def write_xstar_infos(nbox, vobsx, path):

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

        file_header = '#nbox(1)\tjkp(2)\trad(3)\t' + \
                      'delta_r(4)\txpxcol/xpx(5)\tzeta(6)\txpx(7)\tvobsx(8)\txpxcol(9)\ttemp_box(10)\tO8(11)\tO7(12)\t' + \
                      'Ne10(13)\tNe9(14)\tNa11(15)\tNa10(16)\tMg12(17)\tMg11(18)\tAl13(19)\tAl12(20)\tSi14(21)\tSi13(22)\t' + \
                      'S16(23)\tS15(24)\tAr18(25)\tAr17(26)\tCa20(27)\tCa19(28)\tFe26(29)\tFe25(30)\tNh_O8(31)\tNh_O7(32)\t' + \
                      'Nh_Ne10(33)\tNh_Ne9(34)\tNh_Na11(35)\tNh_Na10(36)\tNh_Mg12(37)\tNh_Mg11(38)\tNh_Al13(39)\tNh_Al12(40)\t' + \
                      'Nh_Si14(41)\tNh_Si13(42)\tNh_S16(43)\tNh_S15(44)\tNh_Ar18(45)\tNh_Ar17(46)\tNh_Ca20(47)\t' + \
                      'Nh_Ca19(48)\tNh_Fe26(49)\tNh_Fe25(50)\n'

        # not really necessary ATM
        # file_header_main='#nbox(1)\tjkp(2)\tr(3)\tdelta_r(4)\txpxcol/xpx(5)\tzeta(6)\txpx(7)\tvobsx(8)\txpxcol(9)\t'+\
        #                  'temp_box(10)\nO8(11)\tO7(12)\tNe10(13)\tNe9(14)\tNa11(15)\tNa10(16)\tMg12(17)\tMg11(18)\t'+\
        #                  'Al13(19)\tAl12(20)\tSi14(21)\tSi13(22)\tS16(23)\tS15(24)\tAr18(25)\tAr17(26)\tCa20(27)\t'+\
        #                  'Ca19(28)\tFe26(29)\tFe25(30)\n'

        n_steps = len(px.Abundances('o_iii'))

        for i_step in range(n_steps):
            plasma_pars = px.PlasmaParameters()

            # main infos
            try:
                main_infos = np.array([nbox, i_step + 1,
                                       plasma_pars.radius[i_step],
                                       plasma_pars.delta_r[i_step],
                                       px.Columns('h')[0]/ plasma_pars.n_p[i_step],
                                       plasma_pars.ion_parameter[i_step],
                                       plasma_pars.n_p[i_step], vobsx,
                                       px.Columns('h')[0],
                                       plasma_pars.temperature[i_step] * 1e4]).astype(str).tolist()
            except:
                breakpoint()

            # detail for clarity
            main_infos[0] = str(int(float(main_infos[0])))
            main_infos[1] = str(int(float(main_infos[1])))

            # detailed abundances

            ion_infos = np.array([px.Abundances('o_viii')[0],
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

            # detailed column densities for the second file

            col_infos = np.array([px.Columns('o_viii'),
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

            file_edit(path=path, line_id='\t'.join(main_infos[:2]),
                      line_data='\t'.join(main_infos + ion_infos + col_infos) + '\n', header=file_header)
            time.sleep(1)
    def clean_xstar_container(xstar_id,xstar_mode):

        #cleaning the xstar containers in case of issue

        if xstar_mode=='docker':
            docker_list = str(subprocess.check_output("docker ps", shell=True)).split('\\n')
            docker_xstar_list = [elem.split()[-1] for elem in docker_list\
                                 if elem.split()[-1].startswith('xstar_'+xstar_id)]

            #should only be a single one
            for elem_docker in docker_xstar_list:
                subprocess.call(['docker', 'container', 'rm', '--force', elem_docker])

        elif xstar_mode=='singularity':
            singul_list = str(subprocess.check_output("singularity instance list", shell=True)).split('\\n')
            singul_xstar_list = [elem.split()[0] for elem in singul_list\
                                if elem.startswith('xstar_' + xstar_id)]

            #should only be a single one
            for elem_instance in singul_xstar_list:
                print('deleting singularity instance '+elem_instance+'\n')
                subprocess.call(['singularity', 'instance', 'stop', elem_instance])

    if outdir=='./':
        xstar_dir=os.getcwd()
    else:
        xstar_dir=os.path.join(os.getcwd(),outdir)

    xstar_identifier = xstar_dir[xstar_dir.find('grid'):]
    xstar_identifier = xstar_identifier.replace('/', '_')

    print('Using xstar container id '+xstar_identifier)
    #cleaning previous xstar runs before starting the computation
    clean_xstar_container(xstar_identifier,xstar_mode=xstar_mode)

    #making sure the stop variable is an iterable
    if type(stop_d_input) not in [list, np.ndarray]:
        stop_d=[stop_d_input]
    else:
        stop_d=stop_d_input

    #deprecated because the solution path is not in the save atm
    #using the solution directory for the mantis save if asked to to gain space in the function call
    # if save_folder=="=sol":
    #     save_folder_use=solution[:solution.rfind('/')]
    # else:
    #     save_folder_use=save_folder

    save_folder_use=save_folder


    #chatter value, 0 for not print, 1 for printing
    if chatter>=10:
        lpri=1
    else:
        lpri=0
    
    '''
    computing the number of bins to be used from the desired radial resolution
    see xstar manual p.34 for the main formula 
    (here we use log version instead of a very small power to avoid losing precision with numerical computations)
    delta_E/E is delta_V/c
    
    (note: several mistakes in the formula so we adapt)
    
    IMPORTANT: the formula is not entirely correct, the standard ener has a double log grid, 
    with a main grid up to 500keV and another one with 50 times less bins up to 1 MeV
    this needs to be accounted for in the computation
    
    in custom grid mode, we use a different grid formation with a coarse grid (50 times less bins) 
    in 0.1eV-0.1keV and 10keV-1Mev, and the main grid for 0.1keV-10keV

    '''

    if grid_type=="standard":
        #note: the 1/0.98 factor here is here to reflect the addition of a 1/50 nbins grid for the higher interval
        #which needs to be accounted for
        nbins=max(999,int(np.ceil(np.log(4*10**6)/np.log(1+v_resol/299792.458))/0.98))

    elif grid_type=="custom":
        #here, each part of the sandwiching coarse grid has a 1/50 sampling, so we need to divide by 0.96 instead
        nbins=max(999,int(np.ceil(np.log(1e2)/np.log(1+v_resol/299792.458))/0.96))


    if chatter>=1:
        print('Number of bins for selected velocity resolution: '+str(nbins)+'\n')
        if nbins==999:
            print('(Minimum value accepted by xstar)\n')

    '''
    #### Physical constants
    '''

    # ! From formula (rg/2.0*r_in), Equation (12) of Chakravorty et al. 2016. Here r_in is 6.0*r_g. eta_rad is assumed to be 1.0.
    eta_s = (1.0/12.0)

    if mdot_obs=='auto':
        #notes: Lum_Edd = 1.26e38 for a BH of 1 solar Mass)
        mdot_mhd=xlum/(1.26*m_BH)/eta_s
    else:
        mdot_mhd = mdot_obs/eta_s

    m_BH_SI = m_BH*Msol_SI
    Rs_SI = 2.0*G_SI*m_BH_SI/(c_SI*c_SI)
    
    #!* Gravitational radius
    Rg_SI = 0.5*Rs_SI
    Rg_cgs = Rg_SI*m2cm

    if comput_mode in ['server','cigrid']:
        #loading the solution file
        parlist = np.loadtxt(solution)

        p_mhd=round(parlist[2],5)

        z_over_r, angle, r_cyl_r0, rho_mhd, vel_r, vel_phi, vel_z = [round(elem,5) for elem in parlist[7:14]]

    elif comput_mode=='local':

        p_mhd=p_mhd_input
        if type(solution)==dict:
            #Self-similar functions f1-f10
            z_over_r=solution['z_over_r']

            #this is given in units of r_0
            r_cyl_r0=solution['r_cyl_r0']

            #line of sight angle (0 is edge on)
            angle=solution['angle']

            func_Rsph_by_ro=solution['func_Rsph_by_ro']

            #mhd density
            rho_mhd=solution['rho_mhd']

            #radial, phi and z axis velocity at the alfven point
            vel_r=solution['vel_r']
            vel_phi=solution['vel_phi']
            vel_z=solution['vel_z']

            func_B_r=solution['func_B_r']
            func_B_phi=solution['func_B_phi']
            func_B_z=solution['func_B_z']
            func_Tdyn=solution['func_Tdyn']
            func_Tmhd=solution['func_Tmhd']
        else:
            ####deprecated
            pass
            #loading the solution file instead
            # z_over_r,r_cyl,angle,func_Rsph_by_ro,rho_mhd,vel_r,vel_phi,vel_z,func_B_r,func_B_phi,func_B_z,func_Tdyn,func_Tmhd=\
            # np.loadtxt(solution)

    #### variable definition
    
    #one of the intrinsic xstar parameters (listed in the param file), the maximal number of points in the grids
    #used here to give the maximal size of the arrays
    
    #ncn=99999
    
    nbox_stop=np.zeros(len(stop_d),dtype=int)
    
    #no need to create epi,xlum and enlum because they are outputs or defined elsewhere
    
    logxi_last,vobs_last,robyRg_last,vrel_last,del_E_final,del_E_bs,xpxl_last,xpxcoll_last,zetal_last,vobsl_last,\
        psi_box_last=np.zeros((11,len(stop_d)))

    Rsph_cgs_last=np.zeros(len(stop_d))
    
    #!The ionizing luminosity only is used as xlum to normalize the spectra
    L_xi_Source = xlum*1.0e38 

    # !* Reading functions of self-similar solutions
        
    if chatter>=5:
        print('z_over_r=',z_over_r)
        print('r_cyl_r0=',r_cyl_r0)
        print('angle=',angle)
        print('func_Rsph_by_ro=',func_Rsph_by_ro)
        print('rho_mhd=',rho_mhd)
        print('vel_r=',vel_r)
        print('vel_phi=',vel_phi)
        print('vel_z=',vel_z)
        print('func_B_r=',func_B_r)
        print('func_B_phi=',func_B_phi)
        print('func_B_z=',func_B_z)
        print('func_Tdyn=',func_Tdyn)
        print('func_Tmhd=',func_Tmhd)
        

    #### opening and reseting the files (this should be done on the fly to avoid issues)
    
    stop_dl=stop_d[-1]
        
    os.system('mkdir -p '+outdir)
    
    #copying the xstar input file inside the directory if we're not already there
    if outdir!='./':
        os.system('cp '+SED_path+' '+outdir)
    
    #446
    fileobj_box_details=open('./'+outdir+"/box_details_stop_dist_%.1e"%stop_dl+".log",'w+')
    
    #447
    fileobj_box_ascii_xstar=open('./'+outdir+"/box_Ascii_stop_dist_for_xstar_%.1e"%stop_dl+".dat",'w+')

    fileobj_box_ascii_xstar.write('#Xstar parameters for the non-final boxes. The logxi is only theoretical since  '+
                                    ' it assumes negligible absorption\n')
    fileobj_box_ascii_xstar.write('#n_box log10(density_cgs_mid(nbox)) log10(NhOfBox(nbox)) logxi_mid(nbox) '+
                                     ' vobs_mid(nbox) psi_box(nbox)\n')
    #448
    fileobj_box_ascii_xstar_last=open('./'+outdir+"/last_box_Ascii_for_xstar_%.1e"%stop_dl+".dat",'w+')

    fileobj_box_ascii_xstar_last.write('#Xstar parameters for the final boxes. The logxi is only theoretical since  '+
                                    ' it assumes negligible absorption\n')
    fileobj_box_ascii_xstar_last.write('#n_box_last log10(density_cgs_mid(nbox)) log10(NhOfBox(nbox)) logxi_mid(nbox) '+
                                     ' vobs_mid(nbox) psi_box(nbox)\n')

    #449
    fileobj_box_ascii_stop_dis=open('./'+outdir+"/box_Ascii_stop_dist_%.1e"%stop_dl+".dat",'w+')
    
    fileobj_box_ascii_stop_dis.write('#n_box Rsph_cgs_mid(nbox) log10(density_cgs_mid(nbox)) log10(NhOfBox(nbox)) logxi_mid(nbox) '+
                                     ' vobs_mid(nbox)\n')
    
    #450
    fileobj_box_ascii_last=open('./'+outdir+"/last_box_Ascii_%.1e"%stop_dl+".dat",'w+')

    fileobj_box_ascii_last.write('#n_box_last Rsph_cgs_mid(nbox) log10(density_cgs_mid(nbox)) log10(NhOfBox(nbox)) logxi_mid(nbox) '+
                                     ' vobs_mid(nbox)\n')

    '''
    !* This following 'while' loop is used to find the first suitable value of 
    !* ro_by_Rg where logxi becomes less than some predefined suitable value. 
    !* Above than that, absorption does not contribute much 
    '''
    
    #defining the constant to get back to cylindric radius computations in which are made all of the MHD value computations
    cyl_cst=np.sqrt(1.0+(z_over_r*z_over_r))
    
    #note: the self-similar functions are normalized for disk plane radiuses so we need to convert to r_cyl
    #the output is still for a given r_sph
    
    #all r_sph here should be given in units of Schwarzschild radii

    #note: here, the rcyl corresponds to the r/r0 because the r_cyl constant is given in units of r_0 directly
    
    def func_density(r_sph):
        return func_density_sol(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,m_BH)

    def func_density_relat(r_sph):
        return func_density_relat_sol(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,vel_r,vel_phi,vel_z,m_BH)

    def func_vel_r(r_sph):
        return func_vel_sol('r',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)

    def func_vel_phi(r_sph):
        return func_vel_sol('phi',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)

    def func_vel_z(r_sph):
        return func_vel_sol('z',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)
        
    def func_vel_obs(r_sph):
        return func_vel_sol('obs',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)
    def func_E_deboost(r_sph):
        return func_E_deboost_sol(r_sph, z_over_r, vel_r, vel_phi, vel_z, m_BH)
    def func_r_boost(r_sph):
        return func_r_boost_sol(r_sph, z_over_r, vel_r, vel_phi, vel_z, m_BH)

    #in this one, the distance appears directly so it should be the spherical one
    def func_logxi(r_sph,lum=L_xi_Source):

        #note: here we add a second argument to allow to recompute the luminosity from xstar's output
        return func_logxi_sol(r_sph,z_over_r,lum,rho_mhd,p_mhd,mdot_mhd,vel_r,vel_phi,vel_z,m_BH)
    
    def func_logxi_xstar(r_sph_start,r_sph_mid,density_relat_mid,lum):
        '''
        Modified value of a "box" logxi value to give as an xstar input

        Uses the density at the middle of the box but the radius at the start of the box,
        except for the relativistic corrections on the luminosiy and radius, which are computed at the box midpoint

        Computed like this because this is used as a starting value for Xstar's logxi computation,
        AND to retrieve the starting radius. Thus using R_start allows the box to correctly retrieve R_start,
        despite using the correct density (aka the one at the midpoint), since the luminosity is also given
        independantly

        '''

        return np.log10(lum * func_E_deboost(r_sph_mid)**4 \
                 / (density_relat_mid * (r_sph_start * Rg_cgs * func_r_boost(r_sph_start)) ** 2))


    ro_by_Rg = ro_init
    DelFactorRo = 1.0001

    if not force_ro_init:
        while ro_by_Rg <= 1.1e7 :

            #!* distance from black hole in cylindrical co-ordinates-the radial distance
            rcyl_SI = ro_by_Rg*Rg_SI*r_cyl_r0

            #!* distance from black hole in spherical co-ordinates
            Rsph_SI = rcyl_SI*cyl_cst

            Rsph_Rg=Rsph_SI/Rg_SI

            vel_obs_cgs = func_vel_obs(Rsph_Rg)

            #Note that this considers the deboosting effect but not any absorption
            logxi = func_logxi(Rsph_Rg)

            #!* Here we change the location of the first box as logxi is calculated for xstar to provide correct flux from luminosity.

            if logxi <= 6.0:
                print("starting anchoring radius ro_by_Rg=",ro_by_Rg)

                break
            else:
                #! A distance command : Step increase in the anchoring radius of the magnetic streamline
                ro_by_Rg = ro_by_Rg*DelFactorRo

    else:
        # building the standard infos
        rcyl_SI = ro_by_Rg * Rg_SI * r_cyl_r0
        Rsph_SI = rcyl_SI * cyl_cst
        Rsph_Rg = Rsph_SI / Rg_SI
        vel_obs_cgs = func_vel_obs(Rsph_Rg)
        logxi = func_logxi(Rsph_Rg)

    #!* After getting the starting value of ro_by_Rg from the above 'while' loop, fixing the values for 1st box.
    Rsph_cgs_1st = Rsph_SI*m2cm
    vobs_1st = vel_obs_cgs/(Km2m*m2cm)
    logxi_1st = logxi
    robyRg_1st = ro_by_Rg
    
    fileobj_box_details.write('Rsph_cgs_1st='+str(Rsph_cgs_1st)+',logxi_1st='+str(logxi_1st)+',vobs_1st='+str(vobs_1st)
                              +',ro/Rg_1st='+str(robyRg_1st)+'\n')
    
    #!* Fixing the parameters' values for stop distance
    
    for i_stop,ro_stop in enumerate(stop_d):

        Rsph_stop_Rg=ro_stop*r_cyl_r0*cyl_cst
        
        Rsph_cgs_last[i_stop]= Rsph_stop_Rg*Rg_cgs
        vobs_last[i_stop]= func_vel_obs(Rsph_stop_Rg)/(Km2m*m2cm)
        logxi_last[i_stop]= func_logxi(Rsph_stop_Rg)
        robyRg_last[i_stop]= ro_stop

        fileobj_box_details.write('Rsph_cgs_last='+str(Rsph_cgs_last[i_stop])+',logxi_last='+str(logxi_last[i_stop])
                               +',vobs_last='+str(vobs_last[i_stop])+",ro/Rg_last="+str(robyRg_last[i_stop])+'\n')

    
    #### This is very ugly and should be changed to the proper number of boxes, computed before this loop
    vobs_start,robyRg_start,Rsph_cgs_start,density_relat_cgs_start,logxi_start=np.zeros((5,int(1e5)))
    vobs_mid,robyRg_mid,Rsph_cgs_mid,density_relat_cgs_mid,logxi_mid=np.zeros((5,int(1e5)))
    vobs_stop,robyRg_stop,Rsph_cgs_stop,density_relat_cgs_stop,logxi_stop,NhOfBox=np.zeros((6,int(1e5)))
    logxi_input=np.zeros(int(1e5))
    psi_box=np.zeros(int(1e5))

    #defining a specific fonction which inverts the dr computation
    def func_max_dr(rsph_start,vmax):
        
        '''

        Note: not relativistic

        computes the maximal dr for a given radius and velocity which would end up with bulk velocity
        (hence why we use only vel_r and vel_z) delta of vmax
        
        note: vmax should be in cgs, x_start in R_g
        
        Here we compute the rcyl ratios but the Rsph ratio is identical, 
        and corresponds to the velocity limit for two boxes distant of the Rsph ratio
        
        this inversion loses meaning if vmax is bigger than the initial speed
        (which translates into rcyl_end becoming smaller than rcyl_start)

        In that case, the threshold won't ever be activated because the box delta
        cannot exceed the box initial speed
        In this case, all the speeds will be good, so we return 0 as a flag value
        '''
        
        #imported from sampling_overview
        
        rad_angle=angle*np.pi/180
        
        cst_v=c_cgs*(vel_r*np.cos(rad_angle)+vel_z*np.sin(rad_angle))

        rcyl_start=rsph_start/cyl_cst
        
        rcyl_end=1/((rcyl_start)**(-1/2)-vmax/cst_v)**2

        if rcyl_end<rcyl_start:
            return 0
        else:        
            return rcyl_end/rcyl_start
               
    #!* Building the boxes based on Delr
    Rsph_cgs_end= Rsph_cgs_1st
    i_box = 0
    i_box_stop=0

    nbox_rad=0
    nbox_v=0

    dr_r_eff_list=[]

    #note: we add a rounding in the test condition here to avoid an issue with np precision
    while np.round(Rsph_cgs_end/Rsph_cgs_last[len(stop_d)-1],8)<1.:
    
        #computing the dr/r of the velocity threshold used. We now use end_box-start_box instead of the midpoint of different boxes
        max_dr_res=0 if not cap_dr_resol else func_max_dr(Rsph_cgs_end/Rg_cgs,v_resol*1e5)
        
        if max_dr_res==0:
            dr_factor=(2.0+dr_r)/(2.0-dr_r)
        else:
            #and ensuring we're sampling well enough
            dr_factor=min((2.0+dr_r)/(2.0-dr_r),max_dr_res)
    
        #last box flag
        if Rsph_cgs_last[i_box_stop]<Rsph_cgs_stop[i_box-1]*dr_factor:
            
            final_box_flag=True
            dr_factor= Rsph_cgs_last[i_box_stop]/Rsph_cgs_end
        else:
            final_box_flag=False
            
        #logging the final dr_r value to add in a log file later on

        dr_r_eff_list+=[round(2*(dr_factor-1)/(1+dr_factor),4)]
            
        if dr_factor==(2.0+dr_r)/(2.0-dr_r):
            nbox_rad+=1
            box_type='radial limit'
        elif dr_factor==max_dr_res:
            nbox_v+=1
            box_type='spectral limit'
        else:
            box_type='r_out limit'
            
        Rsph_cgs_start[i_box]= Rsph_cgs_end

        '''
        All densities aer computed with relativistic corrections because they need to be correct to match
        the logxi, which needs to be relativistically corrected 
        
        The nH value on the other hand, is in the rest frame, because instead of relativistically correcting it
        (aka complicated changes on cross-sections), we use the rest/gas/rest spectrum changes
        Thus nH thus needs to use a non relativistic density
        '''

        density_relat_cgs_start[i_box] = func_density_relat(Rsph_cgs_start[i_box]/Rg_cgs)
        logxi_start[i_box] = func_logxi(Rsph_cgs_start[i_box]/Rg_cgs)
        vobs_start[i_box]=func_vel_obs(Rsph_cgs_start[i_box]/Rg_cgs)/(Km2m*m2cm)
              
        robyRg_start[i_box] =Rsph_cgs_start[i_box]/(Rg_cgs*cyl_cst*r_cyl_r0)
    
        Rsph_cgs_stop[i_box]= Rsph_cgs_end*dr_factor
        density_relat_cgs_stop[i_box] = func_density_relat(Rsph_cgs_stop[i_box]/Rg_cgs)
        logxi_stop[i_box] = func_logxi(Rsph_cgs_stop[i_box]/Rg_cgs)
        vobs_stop[i_box]=func_vel_obs(Rsph_cgs_stop[i_box]/Rg_cgs)/(Km2m*m2cm)
        robyRg_stop[i_box] = Rsph_cgs_stop[i_box]/(Rg_cgs*cyl_cst*r_cyl_r0)
         
        Rsph_cgs_mid[i_box]= (Rsph_cgs_start[i_box]+Rsph_cgs_stop[i_box])/2.0
        density_relat_cgs_stop[i_box] = func_density_relat(Rsph_cgs_mid[i_box]/Rg_cgs)
        logxi_mid[i_box] = func_logxi(Rsph_cgs_mid[i_box]/Rg_cgs)
        vobs_mid[i_box]=func_vel_obs(Rsph_cgs_mid[i_box]/Rg_cgs)/(Km2m*m2cm)
        
        robyRg_mid[i_box] = Rsph_cgs_mid[i_box]/(Rg_cgs*cyl_cst*r_cyl_r0)

        #relativistic energy correction for the box
        psi_box[i_box] = func_E_deboost(Rsph_cgs_mid[i_box]/(Rg_cgs))

        #!* Recording quantities for the end point of the box*/

        #different computation for the theoretical xstar logxi
        logxi_input[i_box]=func_logxi_xstar(r_sph_start=Rsph_cgs_start[i_box]/Rg_cgs,
                                            r_sph_mid=Rsph_cgs_mid[i_box]/Rg_cgs,
                                            density_relat_mid=density_relat_cgs_mid[i_box],
                                            lum=L_xi_Source)

        #computing the non-relativistic nH for the box
        NhOfBox[i_box] = func_density(Rsph_cgs_mid[i_box]/Rg_cgs)*(Rsph_cgs_stop[i_box]-Rsph_cgs_start[i_box])
        
        #!* Print the quantities in the log checking
        
        #! This step is for storing data for the last box
        if final_box_flag: 
            
            '''
            this is the number of the box so it should be +1, but if we are currently computing the final box, 
            the last "standard" box is the box before that, so -1 to get back to the previous box
            (this index is the number of the box after which to add a final box computation, NOT the total number of boxes)
            '''
            
            nbox_stop[i_box_stop] = i_box
            fileobj_box_details.write('\n-----------------------------\n')
            fileobj_box_details.write('\nLast box information\n')
            fileobj_box_details.write('stop_dist='+str(stop_d[i_box_stop])+'\n')
        
        fileobj_box_details.write('\n-----------------------------\n')
        fileobj_box_details.write('Box n°'+str(i_box+1)+'\n')
        fileobj_box_details.write('Box dimension criteria: '+box_type+'\nBox dr/r:'+str(dr_r_eff_list[i_box])+'\n\n')
        fileobj_box_details.write('robyRg_start='+str(robyRg_start[i_box])+'\nvobs_start in km/s='+str(vobs_start[i_box])
                                 +'\nRsph_start in cm='+str(Rsph_cgs_start[i_box])+'\nlog n(H)_start_relat (in /cc)='
                                 +str(np.log10(density_relat_cgs_start[i_box]))+"\nlogxi_start="+str(logxi_start[i_box])+'\n\n')

        # !* log(4*Pi) is subtracted to print the correct logxi.
        # !* We have multiplied xlum i.e. luminosity by 4*Pi to make the estimation of flux correctly.
        # !* xi value also we are providing from ASCII file to xstar wrong to estimate the distance correctly.
         
        fileobj_box_details.write('robyRg_mid='+str(robyRg_mid[i_box])+'\nvobs_mid in km/s='+str(vobs_mid[i_box])+
                                  '\nRsph_mid in cm='+str(Rsph_cgs_mid[i_box])+'\nlog n(H)_mid_relat (in /cc)='
                                  +str(np.log10(density_relat_cgs_mid[i_box]))+'\nlogxi_mid='+str(logxi_mid[i_box])+'\n\n')
                                                   
        fileobj_box_details.write('robyRg_stop='+str(robyRg_stop[i_box])+'\nvobs_stop in km/s='+str(vobs_stop[i_box])+
                                  '\nRsph_stop in cm='+str(Rsph_cgs_stop[i_box])+'\nlog n(H)_stop_relat (in /cc)='
                                  +str(np.log10(density_relat_cgs_stop[i_box]))+'\nlogxi_stop='+str(logxi_stop[i_box])+'\n\n')
        
        fileobj_box_details.write('Parameters to go to xstar are:\n')
        fileobj_box_details.write('Gas slab of log n(H)_relat='+str(np.log10(density_relat_cgs_mid[i_box]))+'\nlogNH='+str(np.log10(NhOfBox[i_box]))
                                  +'\nof logxi='+str(logxi_mid[i_box])+'\nis travelling at a velocity of vobs in Km/s='
                                  +str(vobs_mid[i_box])+'\n\n')
        
        # !* xi value we are providing from ASCII file to xstar wrong to estimate the distance correctly.
        # !* It is wrong because luminosity is calculated wrongly by multiplying 4.0*Pi 
        # !* This loop is to prepare the ASCII file which will be input of xstar

        if final_box_flag:

            #!* calculated suitably from density_mid and Rsph_start
            #!* logxi_mid is changed to logxi_input (to have the correct value with xstar balance)
            fileobj_box_ascii_xstar_last.write('%03i\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n'\
                                               %(i_box_stop+1,np.log10(density_relat_cgs_mid[i_box]),np.log10(NhOfBox[i_box]),
                                                logxi_input[i_box],vobs_mid[i_box],psi_box[i_box]))

            #!* This loop is to prepare the ASCII file where xi value is the actual physical value
            fileobj_box_ascii_last.write('%03i\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n'\
                                               %(i_box+1-i_box_stop,Rsph_cgs_mid[i_box],
                                                 np.log10(density_relat_cgs_mid[i_box]),
                                                 np.log10(NhOfBox[i_box]),
                                                logxi_mid[i_box],vobs_mid[i_box]))



        else:
            #!* calculated suitably fron density_mid and Rsph_start
            #!* logxi_mid is changed to logxi_input
            fileobj_box_ascii_xstar.write('%03i\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n'\
                                               %(i_box+1-i_box_stop,np.log10(density_relat_cgs_mid[i_box]),
                                                 np.log10(NhOfBox[i_box]),
                                                logxi_input[i_box],vobs_mid[i_box],psi_box[i_box]))

            #!* This loop is to prepare the ASCII file where xi value is the actual physical value
            fileobj_box_ascii_stop_dis.write('%03i\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n'\
                                               %(i_box+1-i_box_stop,Rsph_cgs_mid[i_box],
                                                 np.log10(density_relat_cgs_mid[i_box]),
                                                 np.log10(NhOfBox[i_box]),
                                                logxi_mid[i_box],vobs_mid[i_box]))

        
        if chatter>=10:
            print(str(i_box_stop+1)+'\t'+str(Rsph_cgs_last[i_box])+'\t'+str(Rsph_cgs_stop[i_box])+'\t'
                 +str(dr_factor)+'\t'+str(Rsph_cgs_last[-1]))
        
        #!* Readjusting the loop parameters to continue if not at the last stop distance
        
        if final_box_flag and i_box_stop!=len(stop_d)-1:

            #!* maintaining the regular box no.
            i_box= i_box-1 
            i_box_stop= i_box_stop+1
            dr_factor = dr_r
        
            final_box_flag=False
            
        Rsph_cgs_end = Rsph_cgs_stop[i_box]
        
        i_box = i_box + 1
        
    #### This is very ugly and should be changed to the proper number of boxes, computed before this loop
    vobs_start=vobs_start[:i_box]
    robyRg_start=robyRg_start[:i_box]
    Rsph_cgs_start=Rsph_cgs_start[:i_box]
    density_cgs_start=density_relat_cgs_start[:i_box]
    logxi_start=logxi_start[:i_box]
    vobs_mid=vobs_mid[:i_box]
    robyRg_mid=robyRg_mid[:i_box]
    Rsph_cgs_mid=Rsph_cgs_mid[:i_box]
    density_cgs_mid=density_relat_cgs_mid[:i_box]
    logxi_mid=logxi_mid[:i_box]
    vobs_stop=vobs_stop[:i_box]
    robyRg_stop=robyRg_stop[:i_box]
    Rsph_cgs_stop=Rsph_cgs_stop[:i_box]
    density_cgs_stop=density_relat_cgs_stop[:i_box]
    logxi_stop=logxi_stop[:i_box]
    NhOfBox=NhOfBox[:i_box]
    psi_box=psi_box[:i_box]

    #total number of boxes is the nbox of the last stop distance
    nbox_std = nbox_stop[len(stop_d)-1]

    for i in range(len(stop_d)):
        fileobj_box_details.write('stop_d(i)='+str(stop_d[i])+'\n')
        print("stop_d(i)="+str(stop_d[i]))
        fileobj_box_details.write('nbox_stop='+str(nbox_stop[i])+'\n')
        print("nbox_stop="+str(nbox_stop[i]))

    print("No. of boxes required="+str(nbox_std)+' + '+str(len(stop_d))+' final')
    
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
    # !* Information of physical variables of all the boxes is evaluated and stored in file
    # "box_Ascii_stop_dist_for_xstar" for xstar
    # input and "box_Ascii_stop_dist" for actual physical variables.
    # !********************************************************************

    #211
    with open('./'+outdir+'/box_Ascii_stop_dist_for_xstar_%.1e'%stop_dl+'.dat','r') as fileobj_box_ascii_stop_dist:
        box_stop_dist_list=fileobj_box_ascii_stop_dist.readlines()[2:]
    
    #! xpx, xpxcol are given in log value. zeta is logxi. vobs in Km/s
    
    xpxcoll,xpxl,zetal,vobsl,vrel,del_E,psi_box_std=np.zeros((7,nbox_std))

    for i_box in range(nbox_std):
        #note: we skip the box number info
        xpxl[i_box],xpxcoll[i_box],zetal[i_box],vobsl[i_box],psi_box_std[i_box]=\
            np.array(box_stop_dist_list[i_box].replace('\n','').split('\t'))[1:].astype(float)
    
    #212
    with open('./'+outdir+'/last_box_Ascii_for_xstar_%.1e'%stop_dl+'.dat','r') as fileobj_box_ascii_stop_dist_last:
        box_stop_dist_list_last=fileobj_box_ascii_stop_dist_last.readlines()[2:]

    if len(box_stop_dist_list_last)==0:

        #this means there's no final box because there's only one box.
        #currently we stop the xstar computation
        #should be adjusted to have a single final box instead (this is a bug)
        ####TODO: get rid of this with proper single box computing

        # copying the box files to the save directory
        if comput_mode == 'server':
            box_files = glob.glob(outdir + '/*box*')
            for elem_box_file in box_files:
                os.system('cp ' + elem_box_file + ' ' + save_folder_use)
        # cleaning the sim directory
        # removing the contents of the sim directory to gain space
        if comput_mode == 'server':
            os.system('rm -f ' + outdir + '/*')

            os.system('rm -f ' + save_folder_use + '/sp_incid*')

        return

    for i in range(len(stop_d)):
        #note: we skip the box number info
        xpxl_last[i],xpxcoll_last[i],zetal_last[i],vobsl_last[i],psi_box_last[i]=\
            np.array(box_stop_dist_list_last[i].replace('\n','').split('\t'))[1:].astype(float)

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
    

    nbox_restart=1

    #default value to test if there will be a restart
    xstar_input_restart=None
    

    dict_box={'dr_r_eff_list':dr_r_eff_list,
              'v_resol':v_resol,
              'i_box_final':i_box_final,
              'xlum':xlum}

    # computing the velocities and del_E factors for the whole grid

    # Changed vturb to each box's own delta to get more consistent result
    vturb_in = vobs_start - vobs_stop

    # #computing the initial redshift from the speed at the starting box (point of view of the central SED)
    # vrel[0]=vobsl[0]
    # #and the subsequent blueshifts from the progressive decelerration (point of view of the central SED)
    # vrel[1:]=vobsl[1:]-vobsl[:-1]

    if reload:
        #searching xstar_pars for existing boxes
        if os.path.isfile('./'+outdir+'/xstar_pars.log'):
            with open('./'+outdir+'/xstar_pars.log') as xstar_par_file:
                par_lines=xstar_par_file.readlines()[3:]
                

            i_last_line=1
            
            line_restart=par_lines[-i_last_line]
            
            #ensuring to take the last line not from a final box
            while 'final' in line_restart:
                i_last_line+=1
                line_restart=par_lines[-i_last_line]
            
            if int(line_restart.split('\t')[0])!=1:

                print('unfinished computation detected. Restarting from last computed box...\n')
                    
                #fetching the information about the box
                nbox_restart=int(line_restart.split('\t')[0])
                
                i_box_final=int(line_restart.split('\t')[1])
                
                xstar_input_restart=line_restart.split('\t')[2]
                
                xlum_restart=float(line_restart.split('\t')[3])

                dict_box['xlum_corr']=float(line_restart.split('\t')[4])
                
                t_restart=float(line_restart.split('\t')[5])
                
                n_restart=float(line_restart.split('\t')[6])
                
                nh_restart=float(line_restart.split('\t')[7])
                
                logxi_restart=float(line_restart.split('\t')[8])
                
                vturb_x_restart=float(line_restart.split('\t')[9])
                
                print("Restarting from box "+str(nbox_restart)+"\n")


    #creating an io if a progress file is given to redirect the tqdm messages
    if progress_file is not None:
        progress_io=io.StringIO()
    else:
        progress_io=None

    progress_header='#grid_identifier\tprogress\n'

    ####main loop

    # creating a progess_id to log the progress
    currdir = os.getcwd()
    os.chdir(outdir)
    progress_id = os.getcwd()[os.getcwd().find('grid'):]
    progress_id = progress_id.replace('/', '_')
    os.chdir(currdir)


    #using i_box because it's an index here, not the actual box number (shifted by 1)
    for i_box in tqdm(range(nbox_restart-1,nbox_std),file=progress_io,
                      initial=nbox_restart-1,total=nbox_std):

        #logging the tqdm values in a global progress file if asked to, using the same identifier as for xstar containers
        if progress_file is not None:

            #extracting the last progress bar iteration
            progress_val=progress_io.getvalue().split('\r')[-1]
            file_edit(progress_file,line_id=progress_id,line_data=progress_id+progress_val,header=progress_header)

        # resetting the global log file for the first computation
        if i_box == 0 and i_box_final == 0:
            if os.path.isfile('./'+outdir+'/xout_log_global.log'):
                os.remove('./'+outdir+'/xout_log_global.log')

        dict_box['nbox']=i_box+1

        #! Doppler shifting of spectra depending on relative velocity
        
        
        # if i_box>0:
        #     vrel[i_box] = vobsl[i_box]-vobsl[i_box-1]
        #     vturb_in[i_box] = vobsl[i_box-1]-vobsl[i_box]
        # else:
        #     vrel[i_box] = vobsl[i_box]
        #     vturb_in[i_box] = vobsl[i_box]

        #!del_E(i_box) = 1.00

        #! Reading input spectra from file: Initial spectrum/last box rest frame computation

        if i_box<1:
            
            # Nengrid=int(incident_spectra_lines[0])            

            shift_input=SED_path

        else:
                
            shift_input=os.path.join(outdir,'sp_tr_rest_%03i'%(i_box)+'.dat')

        #here because xstar is launched in outdir
        xstar_input_save=os.path.join(outdir,'sp_incid_gaz_%03i'%(i_box+1)+'.dat')
        xstar_input='./sp_incid_gaz_%03i'%(i_box+1)+'.dat'

        vobsx = vobsl[i_box]

        #not doing this when restarting because there's no need
        if not (i_box+1==nbox_restart and xstar_input_restart is not None):

            if i_box+1!=nbox_restart:
                #! Read output from the box
                px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                             file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')

                #note that if the code crashed between the end of a computation and the beginning
                #of the next one, but in this case the computation will restart on the previous box

                # loading the temperature of the previous box
                plasma_par = px.PlasmaParameters()

                # retrieving the plasma temperature of the last step
                tp=plasma_par.temperature[-1]

            #shifting the spectrum and storing it in the xstar input file name
            lum_corr_factor=shift_incident_sp_rest_gaz(psi_box_std[i_box],xstar_input_save,incident_path=shift_input)

            #this info is important so we log it in the box dictionnary to write it in the xstar file
            dict_box['xlum_corr']=lum_corr_factor

            #reminder: this is relativistic to match the relativistic logxi
            xpx = 10.0**(xpxl[i_box])

            #this isn't because we change the spectrum instead
            xpxcol = 10.0**(xpxcoll[i_box])

            #correcting the ionization parameter for the ratio between the xstar computed luminosity and the input
            # luminosity

            ####TODO: THIS SHOULD BE CHECKED
            #note: no psi_box_std here because the zetal already considers psi_box
            zeta_corr = zetal[i_box]+np.log10(xlum_eff/xlum*lum_corr_factor)

            xlum_corr=xlum_eff*psi_box_std[i_box]**4*lum_corr_factor

            if no_turb:
                vturb_x=0
            else:
                vturb_x = vturb_in[i_box]

            '''
            The lines of sight considered should already be compton thin, 
            the whole line of sight has to be compton thick and this is 
            checked directly from jonathan's solution
            The following test is just a sanity check
            '''
            ####TODO: Change this
            if (xpxcol>1.5e24):
                print('Thomson depth of the cloud becomes unity')

                break

        #### main xstar call
        
        currdir=os.getcwd()
        os.chdir(outdir)
        os.system('rm -f xout_*.fits')
        
        #directly loading the restart parameters if restarting at this box
        if i_box+1==nbox_restart and xstar_input_restart is not None:
            xstar_func(xstar_input_restart,xlum_restart,t_restart,n_restart,nh_restart,logxi_restart,
                       0 if no_turb else vturb_x_restart,nbins=nbins,
                       path_logpars=path_log_xpars,dict_box=dict_box,
                       comput_mode=comput_mode,
                       save_folder=save_folder_use,
                       no_write=no_write,
                       xstar_mode=xstar_mode,
                       xstar_loc=xstar_loc)
        else:
            xstar_func(xstar_input,xlum_corr,tp,xpx,xpxcol,zeta_corr,vturb_x,nbins=nbins,
                       path_logpars=path_log_xpars,dict_box=dict_box,
                       comput_mode=comput_mode,
                       save_folder=save_folder_use,
                       no_write=no_write,
                       xstar_mode=xstar_mode,
                       xstar_loc=xstar_loc)
        
        os.chdir(currdir)
        
        #writing the infos
        px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                     file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')
        
        write_xstar_infos(i_box+1,vobsx,'./'+outdir+'/xstar_output_details.dat')

        shift_output='./'+outdir+'/sp_tr_rest_%03i'%(i_box+1)+'.dat'

        #writing the rest frame output and storing the new value of the luminosity
        #note: the xlum_eff value in argument here is the bolometric luminosity of the incident spectrum
        xlum_eff=shift_transmitted_sp_gaz_rest(psi_box_std[i_box],shift_output,shift_input,xlum_eff)

        if comput_mode in ['server','cigrid']:
            if comput_mode=='cigrid':
                upload_mantis('./'+outdir+'/xstar_output_details.dat',save_folder_use)
                upload_mantis('./'+outdir+'/sp_tr_rest_%03i'%(i_box+1)+'.dat', save_folder_use)
            elif comput_mode=='server':
                os.system('cp '+'./'+outdir+'/xstar_output_details.dat'+' '+save_folder_use)
                os.system('cp ' + './'+outdir+'/sp_tr_rest_%03i'%(i_box+1)+'.dat' + ' ' + save_folder_use)

        ####!* Computing spectra and blueshift for the final box depending on stop_dist.

        if i_box+1==nbox_stop[i_box_final]:

            vrel_last[i_box_final] = vobsl_last[i_box_final]-vobsl[i_box]

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

            # here because xstar is launched in outdir
            xstar_input_save = './' + outdir + '/sp_incid_gaz_final_%03i' % (i_box_final+1) + '.dat'
            xstar_input ='./sp_incid_gaz_final_%03i' % (i_box_final+1) + '.dat'
            
            #shifting the spectra in a different file name
            #note: we use shift output here because we want the rest frame of the computation we just did,
            # not the one before
            lum_corr_factor=shift_incident_sp_rest_gaz(psi_box_last[i_box_final],xstar_input_save,shift_output)

            #reminder: this is relativistic to match the relativistic logxi
            xpx = 10.0**(xpxl_last[i_box_final])

            #this isn't because we change the spectrum instead
            xpxcol = 10.0**(xpxcoll_last[i_box_final])

            ####TODO: THIS SHOULD BE CHECKED
            zeta_corr_final = zetal_last[i_box_final]+np.log10(xlum_eff/xlum*lum_corr_factor)

            xlum_corr_final=xlum_eff*psi_box_last[i_box_final]**4*lum_corr_factor
            
            vobsx = vobsl_last[i_box_final]

            if no_turb:
                vturb_x=0
            else:
                vturb_x = vobsl[nbox_stop[i_box_final]-1]-vobsl_last[i_box_final]
    

            ####TODO: evolve this
            if (xpxcol>1.5e24):
                print('Thomson depth of the cloud becomes unity')
                
                break
    
            currdir=os.getcwd()
            os.chdir(outdir)
            os.system('rm -f xout_*.fits')
            
            dict_box['i_box_final']+=1
            #using xlum_final here to avoid overwriting xlum_eff if using more than a single stop distance
            xstar_func(xstar_input,xlum_corr_final,tp,xpx,xpxcol,zeta_corr_final,vturb_x,nbins=nbins,
                       path_logpars=path_log_xpars,dict_box=dict_box,
                       comput_mode=comput_mode,
                       save_folder=save_folder_use,
                       no_write=no_write,
                       xstar_mode=xstar_mode,
                       xstar_loc=xstar_loc)
            
            os.chdir(currdir)
            
            px.LoadFiles(file1='./'+outdir+'/xout_abund1.fits',file2='./'+outdir+'/xout_lines1.fits',
                         file3='./'+outdir+'/xout_rrc1.fits',file4='./'+outdir+'/xout_spect1.fits')

            shift_output_last= './' + outdir + '/sp_tr_rest_final_%03i' % (i_box_final + 1) + '.dat'

            #same here using shift output as the incident rest frame of the last computation, aka the endpoint of
            #the last regular box

            #note: we don't use this luminosity to overwrite xlum_eff since the final boxes are independant
            xlum_final=shift_transmitted_sp_gaz_rest(psi_box_last[i_box_final],shift_output_last,
                                                     incident_path=shift_output,xlum_eff=xlum_eff)
    
            #switching to the next final box to be computed    
            i_box_final= i_box_final+1
            
            #writing the infos with the final iteration
            write_xstar_infos(i_box+1,vobsx,'./'+outdir+'/xstar_output_details_final.dat')

            if comput_mode in['server','cigrid']:

                if comput_mode=='cigrid':
                    upload_mantis(shift_output_last,save_folder_use)
                    upload_mantis('./' + outdir + '/xstar_output_details_final.dat', save_folder_use)
                elif comput_mode=='server':
                    os.system('cp '+shift_output_last+' '+save_folder_use)
                    os.system('cp ./' + outdir + '/xstar_output_details_final.dat '+save_folder_use)

            #removing the standard xstar output to gain space
            os.system('rm -f '+outdir+'/xout_*.fits')

    #logging the final tqdm values in a global progress file if asked to, using the same identifier as for xstar containers
    if progress_file is not None:

        #extracting the last progress bar iteration
        progress_val=progress_io.getvalue().split('\r')[-1]
        file_edit(progress_file,line_id=progress_id,line_data=progress_id+progress_val,header=progress_header)

            
    #cleaning the xstar docker before ending the computation
    clean_xstar_container(xstar_identifier,xstar_mode=xstar_mode)

    #copying the box files to the save directory
    if comput_mode=='server':
        box_files=glob.glob(outdir+'/*box*')
        for elem_box_file in box_files:
            os.system('cp '+elem_box_file+' '+save_folder_use)

    #cleaning the sim directory
    #removing the contents of the sim directory to gain space
    if comput_mode=='server':
        os.system('rm -f '+outdir+'/*')

        #also removing the incident spectra except for the last one to be able to relaunch the last computation
        list_incid_spectra=glob.glob(save_folder_use+'/sp_incid_*')
        last_incid_spectra=[elem for elem in list_incid_spectra if '_final_' not in elem][-1]

        print(last_incid_spectra)

        print(list_incid_spectra)

        for elem_incid in [elem for elem in list_incid_spectra if elem!=last_incid_spectra]:
            os.system('rm -f '+save_folder_use+'/'+elem_incid)

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
            Lnu_per_erg[j] = Lnu_per_erg[j]*norm_factor
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

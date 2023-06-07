#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from astropy.io import ascii

import numpy as np
import pexpect
import glob

import time

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

def interp_yaxis(x_value,x_axis,y_axis,round=True):
    '''
    interpolates linearly the y values of the 2 closest x_axis values to x_value

    if round is set to True, returns directly a close value from x_axis if there's one
    (to avoid issues in equalities with numpy)
    '''

    #returning directly the y point to avoid issues in precision when the x_value is in the sample
    if x_value in x_axis:
        return y_axis[x_axis==x_value]

    #fetching the closest points
    min_mask=np.argmin(abs(x_axis-x_value))

    if round:
        if abs(x_value-x_axis[min_mask]<1e-6):
            return y_axis[min_mask]

    if x_value<x_axis[min_mask]:
        min_mask=[min_mask-1,min_mask]
    else:
        min_mask=[min_mask,min_mask+1]
    closest_x=x_axis[min_mask]
    closest_y=y_axis[min_mask]

    #ordering them correctly for the x axis
    closest_y = closest_y[closest_x.argsort()]
    closest_x.sort()

    # #interpolating (assuming linear here)
    # coeff= (closest_y[1]-closest_y[0])  /(closest_x[1]-closest_x[0])
    # ord_orig=closest_y[0]-coeff*closest_x[0]
    #y_value=coeff*x_value+ord_orig

    #directly giving it is easier
    y_value=closest_y[0]+(closest_y[1]-closest_y[0])*(x_value-closest_x[0])/(closest_x[1]-closest_x[0])

    return y_value

def print_log(elem,logfile_io):

    '''
    prints and logs at once
    '''
    print(elem)

    if logfile_io is not None:
        logfile_io.write(str(elem)+('\n' if not str(elem).endswith('\n') else ''))


def download_mantis(mantis_path,local_folder,load_folder=False):

    '''
    copies a path into a mantis folder

    if delete_previous is set to True, the path is supposed to be the one of the incident spectrum,
    and the previous incident spectrum is deleted if it is not a final box incident

    note: the mantis folder should be a relative path after mantis/userid/

    We can't directly write into mantis so we can't file_edit directly into it, instead we straight up replace the logpar file at every box

    all the paths should be relative and will be changed to absolute paths for the conversion
    '''

    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')

    # loading the file or directory
    irods_proc.sendline('iget '+(' -r' if load_folder else '')+ mantis_path + ' ' + os.path.join(os.getcwd(),local_folder))

def upload_mantis(path,mantis_folder,delete_previous=False):

    '''
    copies a path into a mantis folder

    if delete_previous is set to True, the path is supposed to be the one of the incident spectrum,
    and the previous incident spectrum is deleted if it is not a final box incident

    note: the mantis folder should be a relative path after mantis/userid/

    We can't directly write into mantis so we can't file_edit directly into it, instead we straight up replace the logpar file at every box

    all the paths should be relative and will be changed to absolute paths for the conversion
    '''

    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')

    # uploading the file
    irods_proc.sendline('iput ' + os.path.join(os.getcwd(), path) + ' ' + mantis_folder)

    #deleting the previous box spectrum if not in a final box
    if delete_previous:
        i_box=path.split('_')[-1].split('.')[0]
        previous_box_incident=os.path.join(os.getcwd(),path.replace(path.split('_')[-1],path.split('_')[-1].replace(i_box,str(int(i_box)-1))))
        irods_proc.sendline('-irm '+previous_box_incident+' '+mantis_folder)

def load_solutions(solutions,mode='file',split_sol=False,split_par=False):

    if mode=='file':
        solutions_arr = np.loadtxt(solutions)
    elif mode=='array':
        solutions_arr=solutions

    if not split_sol and not split_par:
        return solutions_arr

    solutions_ids = solutions_arr.T[:7].T

    solutions_ids_unique = np.unique(solutions_ids, axis=0)

    if split_sol and not split_par:

        # splitting the array by solution by fetching the occurences of the first 7 indexes
        split_sol_mask = np.array([(solutions_ids == elem).all(axis=1) for elem in solutions_ids_unique])

        # split per solution (increasing epsilon, then n_island, then p,...)
        solutions_split_arr = np.array([solutions_arr[elem_mask] for elem_mask in split_sol_mask], dtype=object)

        return solutions_split_arr

    if split_sol and split_par:

        eps_vals=np.unique(solutions_ids.T[0])

        solutions_split_arr=np.array([None]*len(eps_vals))

        for id_eps,elem_eps in enumerate(eps_vals):

            split_eps_mask = solutions_ids.T[0] == elem_eps

            solutions_split_eps = solutions_arr[split_eps_mask]

            n_vals=np.unique(solutions_split_eps.T[1])

            elem_split_eps = np.array([None] * len(n_vals))

            for id_n,elem_n in enumerate(n_vals):

                split_n_mask=solutions_split_eps.T[1]==elem_n

                solutions_split_n=solutions_split_eps[split_n_mask]

                p_mu_vals=np.unique(solutions_split_n.T[2:4].T,axis=0)

                split_p_mu_mask=[(solutions_split_n.T[2:4].T == elem).all(axis=1) for elem in p_mu_vals]

                #here we can directly make the element
                elem_split_eps[id_n]=np.array([solutions_split_n[elem_mask] for elem_mask in split_p_mu_mask],dtype=object)

            solutions_split_arr[id_eps]=elem_split_eps

        return solutions_split_arr


def sample_angle(solutions_path, angle_values, mdot_obs, m_BH, r_j=6., eta_mhd=1 / 12, outdir=None,
                 return_file_path=False,mode='file',return_compton_angle=False):
    '''
    split the solution grid for a range of angles up to the compton-thick point of each solution

    solutions_path: solutions path of the syntax of a load_solutions output

    outdir: output directory where the log file and solution file will be written

    angle_values: direct python list-type object containing the values of the angle to be sampled
                  (angle_max=90 is equivalent to not using an angle_max limit)
    mode:
        file: standard working mode
              logs the sampling of the angles for each solution and creates a solution file in outdir

        array: no log file
               returns directly the solution array instead of writing it in a file

                if return_compton_angle is set to True, returns a second array with the compton thick
                angle of each solution

    the default value of r_j is a massive particule's isco in Rg for non-spinning BH


    '''

    solutions_path_ext = '.' + solutions_path.split('/')[-1].split('.')[-1]


    if mode=='file':
        solutions_log_path = solutions_path.replace( \
            solutions_path_ext,
            '_angle_sampl_mdot_' + str(mdot_obs) + '_m_bh_' + str(m_BH) + '_rj_' + str(r_j) + '_log' + solutions_path_ext)

        solutions_mod_path = solutions_path.replace( \
            solutions_path_ext,
            '_angle_sampl_mdot_' + str(mdot_obs) + '_m_bh_' + str(m_BH) + '_rj_' + str(r_j) + solutions_path_ext)

        # adding the outdir into it
        solutions_log_path = os.path.join('/'.join(solutions_log_path.split('/')[:-1]), outdir,
                                          solutions_log_path.split('/')[-1])

        # adding the outdir into it
        solutions_mod_path = os.path.join('/'.join(solutions_mod_path.split('/')[:-1]), outdir,
                                          solutions_mod_path.split('/')[-1])

        os.system('mkdir -p '+outdir)
        solutions_log_io = open(solutions_log_path, 'w+')
    else:
        solutions_log_io=None

    solutions_split_arr=load_solutions(solutions_path,split_sol=True)

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm

    mdot_mhd = mdot_obs / eta_mhd

    solutions_sample = []

    n_angles = 0

    ang_compton_list=[]

    # working solution by solution
    for solutions_split in solutions_split_arr:
        def column_density_full(p_mhd, rho_mhd):
            '''
            computes the column density at infinity starting at Rg (aka the integral of the density starting at this value)

            Here we assume a SAD starting at r_j

            here the previous Rg_cgs factor at the denominator has been cancelled with the dx
            '''

            return mdot_mhd / (sigma_thomson_cgs) * rho_mhd * (r_j ** (p_mhd - 0.5) / (0.5 - p_mhd))

        def ang_compton_thick(p_mhd,rho_mhd,angles):
            #fetching the rho value giving exactly the compton thickness threshold
            rho_compton=compton_thick_thresh*sigma_thomson_cgs/(mdot_mhd*(r_j ** (p_mhd - 0.5) / (0.5 - p_mhd)))

            angle_compton=interp_yaxis(rho_compton,rho_mhd,angles)

            return angle_compton
        # retrieving p
        p_mhd_sol = solutions_split[0][2]

        # and the varying rho
        rho_mhd_sol = solutions_split.T[10]

        angle_sol=solutions_split.T[8]

        # computing the column densities
        col_dens_sol = column_density_full(p_mhd_sol, rho_mhd_sol)

        #the compton thick threshold

        ang_compton=ang_compton_thick(p_mhd_sol,rho_mhd_sol,angle_sol)

        ang_compton_list+=[ang_compton]

        # and the first angle value below compton-thickness
        sol_angle_thick = angle_sol[col_dens_sol < compton_thick_thresh][0]

        print_log('\n\n***************', solutions_log_io)
        print_log('Solution:\n' +
                  'epsilon=' + str(solutions_split[0][0]) + '\nn_island=' + str(solutions_split[0][1]) +
                  '\np=' + str(solutions_split[0][2]) + '\nmu=' + str(solutions_split[0][3]), solutions_log_io)

        print_log('\nCompton thick threshold at theta~'+str(ang_compton),solutions_log_io)
        print_log('\nFirst solution below comp thick at theta=' + str(sol_angle_thick), solutions_log_io)

        angle_values_nonthick=angle_values[angle_values<sol_angle_thick]

        # using it to determine how many angles will be probed (and addin a small delta to ensure the last value
        # is taken if it is a full one

        ####add log space option

        print_log('Angle sampling:', solutions_log_io)
        print_log(angle_values_nonthick, solutions_log_io)

        n_angles += len(angle_values_nonthick)

        # restricting to unique indexes to avoid repeating solutions
        id_sol_sample = np.unique([abs(angle_sol - elem_angle).argmin() for elem_angle in angle_values_nonthick])

        print_log('Angles of solutions selected:', solutions_log_io)
        print_log(angle_sol[id_sol_sample], solutions_log_io)

        # and fetching the corresponding closest solutions
        solutions_sample += solutions_split[id_sol_sample].tolist()

    solutions_sample = np.array(solutions_sample)

    print_log('\ntot angles init:' + str(n_angles), solutions_log_io)
    print_log('tot solutions:' + str(len(solutions_sample)), solutions_log_io)

    header_arr='#epsilon\tn_island\tp_xi\tmu\tchi_m\talpha_m\tPm\tz_over_r\ttheta\tr_cyl/r0\trho_mhd\tu_r\tu_phi\tu_z'+\
               '\tT_MHD\tB_r\tB_phi\tB_z\tT_dyn\ty_id\ty_SM\ty_A'
    # saving the global file

    if mode=='file':
        np.savetxt(solutions_mod_path, solutions_sample, delimiter='\t',
                   header=header_arr)

        if return_file_path:
            return solutions_mod_path
    elif mode=='array':
        if return_compton_angle:
            return solutions_sample,np.array(ang_compton_list)
        else:
            return solutions_sample


def create_grid(grid_name, mhd_solutions_path,
                 SED,mdot,xlum,m_BH,rj,
                 param_mode='split_angle',
                 h_over_r_vals=None,p_vals=None,mu_vals=None,angle_vals=None):

    '''

    Setups Cigrid computations by creating a proper mantis folder tree, and associated launching script and parameter file

    Args:

        grid_name: name of grid and of the folder tree which will be copied to mantis
        cores:
        ex_time:
        priority:
        local:creates the mantis directories, script and parameter files locally using the mantis_grid_dir instead (used for testing)

    Parameter arguments:
    global arguments (a single per run)
        dr_r
        v_resol
        ro_init
        stop_d
        mode:
            -all: computes the spectra for every single mhd solution in the file without decomposing
            -split_angle: considers all solutions but only in a given range of angles
            -decompose: combines the individual parameter space asked for every value, among the nearest available in the
            solution file (nearest for h_over_r_vals, 2-dimension nearest in log distance in p/mu and nearest again in angle)
            creates intermediary folders to reflect the dimensions

    coupled parameter arguments (should all be the same length or single to indicate one value to be repeated)
        SED:list of SED paths (will be copied to mantis_grid_dir for use)
        mdot: mdot OBSERVED (natural units)
        xlum: Xray luminosities (1e38 erg/s in 1-1000 Ryd)
        m_BH: black hole mass (M_sol)
        rj: JED-SAD transition radius
            (which acts as the starting point for the nH computation to get the angle thresholds)


    decoupled parameter arguments for 'decompose' mode (one MHD solution for each combination)

    input possibles are either list/array like elements or str of type 'log/lin_valmin/valmax_nelements'
    (with valmin and valmax included) values should be in logspace for log type
    angle is the only variable which only accepts lin because it doesn't make sense otherwise

        h_over_r_vals: values for the aspect ratio
        p_vals: values for the ejection index
        mu_vals: values for the magnetisation
        angle_vals: values for the angle. LINEAR if interval
                    Note:
                    The max value will always be restricted to the maximal value possible to remain
                    non-compton thick in the solution. Putting 90 as the max is equivalent to fetching this

    '''

    '''
    decomposing the values if in interval form
    '''

    if param_mode=='decompose':
        #decomposing the required parameters
        if type(h_over_r_vals) in (list,np.ndarray):
            h_over_r_list=h_over_r_vals
        elif type(h_over_r_vals)==str:
            h_over_r_infos=h_over_r_vals.split('_')
            if h_over_r_infos[0]=='log':
                h_over_r_list=np.logspace(float(h_over_r_infos[1]),float(h_over_r_infos[2]),int(h_over_r_infos[3]))
            elif h_over_r_infos[0]=='lin':
                h_over_r_list = np.linspace(float(h_over_r_infos[1]),float(h_over_r_infos[2]),int(h_over_r_infos[3]))
                
        if type(p_vals) in (list,np.ndarray):
            p_list=p_vals
        elif type(p_vals)==str:
            p_infos=p_vals.split('_')
            if p_infos[0] == 'log':
                p_list = np.logspace(float(p_infos[1]), float(p_infos[2]),
                                            int(p_infos[3]))
            elif p_infos[0] == 'lin':
                p_list = np.linspace(float(p_infos[1]), float(p_infos[2]),
                                            int(p_infos[3]))
                
        if type(mu_vals) in (list,np.ndarray):
            mu_list=mu_vals
        elif type(mu_vals)==str:
            mu_infos=mu_vals.split('_')
            if mu_infos[0]=='log':
                mu_list=np.logspace(float(mu_infos[1]),float(mu_infos[2]),int(mu_infos[3]))
            elif mu_infos[0]=='lin':
                mu_list = np.linspace(float(mu_infos[1]),float(mu_infos[2]),int(mu_infos[3]))

    if param_mode in ['all','split_angle']:

        if type(angle_vals) in (list,np.ndarray):
            angle_list=angle_vals

        elif type(angle_vals)==str:
            angle_infos=angle_vals.split('_')
            if angle_infos[0]=='range':
                angle_list = np.arange(float(angle_infos[1]), float(angle_infos[2])+0.000001, float(angle_infos[3]))
            elif angle_infos[0]=='log':
                angle_list=np.logspace(float(angle_infos[1]),float(angle_infos[2]),int(angle_infos[3]))
            elif angle_infos[0]=='lin':
                angle_list = np.linspace(float(angle_infos[1]),float(angle_infos[2]),int(angle_infos[3]))


    '''
    converting the SED variables into lists
    '''

    #future variables which will be used in the code
    SED_list=None
    mdot_list=None
    xlum_list=None
    m_BH_list=None
    rj_list=None


    SED_var_list=[SED,mdot,xlum,m_BH,rj]
    SED_list_var_list=[]

    #checking if any is already like that
    list_SED_vars=[elem for elem in SED_var_list if type(elem) in [np.ndarray,list,tuple]]

    #and deducting the desired lengths
    len_SED_vars=1 if len(list_SED_vars)==0 else len(list_SED_vars[0])

    #looping in all variables variables
    for SED_type_var in SED_var_list:
        if type(SED_type_var) not in [np.ndarray,list,tuple]:

            SED_list_var_list+=[np.repeat(SED_type_var,len_SED_vars)]
        else:
            assert  len(SED_type_var)==len_SED_vars,'Error: SED variables have different len'
            SED_list_var_list+=SED_type_var

    #and outputing them in each variable name

    SED_list,mdot_list,xlum_list,m_BH_list,rj_list=SED_list_var_list

    #converting the mhd solution files
    mhd_sol_arr=np.loadtxt(mhd_solutions_path)

    #identifying the folder of the solution file
    mhd_sol_dir = mhd_solutions_path[:mhd_solutions_path.rfind('/')]

    #converting the solutions depending on the parameter_mode

    #creating the solution angle sampling for every solution

    SED_dirs=[]
    # sampl_grid_paths=[]

    if param_mode=='split_angle':
        for elem_SED,elem_mdot,elem_m_BH,elem_rj in zip(SED_list,mdot_list,m_BH_list,rj_list):

            #the name of the grid is used as global directory
            #the name of the SED and other parameters will be used as the outdirs for the combination
            SED_dirs+=[os.path.join(grid_name,elem_SED.split('/')[-1][:elem_SED.split('/')[-1].rfind('.')]+\
                     '_mdot_'+str(elem_mdot)+'_m_bh_'+str(elem_m_BH)+'_rj_'+str(elem_rj))]

            #creating the grid in that folder
            sampl_grid_path=sample_angle(mhd_solutions_path,angle_list,elem_mdot,elem_m_BH,elem_rj,outdir=SED_dirs[-1],
                                        return_file_path=True)

            sampl_grid_dir=sampl_grid_path[:sampl_grid_path.rfind('/')]

            #copying the SED in the folder
            os.system('cp '+elem_SED+' '+SED_dirs[-1])

            solutions_grid = np.loadtxt(sampl_grid_path)

            with open(sampl_grid_path) as grid_file:
                solution_header=grid_file.readlines()[0]

            #decomposing the folder tree for each solution
            for solution in solutions_grid:

                #note: no decomposition for the turbulence as of now
                dir_sol=os.path.join('eps_'+str(solution[0]),
                                     'n_island_'+str(int(solution[1])),
                                     'p_'+str(solution[2])+'_mu_'+str(solution[3]),
                                     'angle_'+str(round((solution[8]),1)))

                #creating the directory
                os.system('mkdir -p '+os.path.join(sampl_grid_dir,dir_sol))

                #making a 2-d array to save the solution in columns
                sol_save=np.array([solution])
                #saving the single line with the header in the newly created folder
                np.savetxt(os.path.join(sampl_grid_dir,dir_sol,dir_sol.replace('/','_')+'.txt'),sol_save,
                           header=solution_header[1:].replace('\n',''))

    #creating the parameter file above everything

def create_grid_parfile(grid_folder,mantis_grid_dir,dr_r,v_resol,ro_init,stop_d):

    '''
    Inserts a parfile inside an already existing grid folder structure

    the list of parameters is the list of arguments of cigri_wrapper
    '''

    sol_files = glob.glob('grid/**/eps**.txt', recursive=True)

    #creating an array with the argument elements for each solution

    parameters=np.array([None]*len(sol_files))

    for i_sol,elem_sol_file in enumerate(sol_files):

        #creating the mantis dir
        sol_file_dir=elem_sol_file[:elem_sol_file.rfind('/')]
        mantis_dir=os.path.join(mantis_grid_dir,sol_file_dir)

        #fetching the SED id from the directory names
        sed_file_id=sol_file_dir.split('/')[1][:sol_file_dir.split('/')[1].rfind('_mdot')]

        #Convoluted expression to retrieve the extension of the SED
        sed_file=glob.glob(os.path.join(sol_file_dir.split('/')[0],sol_file_dir.split('/')[1],sed_file_id+'**'))[0]
        SED_mantis_path=os.path.join(mantis_grid_dir,sed_file)

        #the solution file
        solution_mantis_path=os.path.join(mantis_grid_dir,elem_sol_file)






def setup_cigrid(mantis_grid_dir, silenus_grid_dir,cores,ex_time,priority):

    #creating the spawn
    irods_proc=pexpect.spawn('./bin/bash',encoding='utf-8')

    #needs to be done when running a new terminal
    irods_proc.sendline('source /applis/site/nix.sh')



    #zipping the files and then copying them

    # # loop in the SEDs:
    # for i_SED, (elem_SED, elem_mdot, elem_xlum) in enumerate(zip(SED_list, mdot_list, xlum_list)):
    #
    #     # creating a mantis subdirectory
    #     mantis_SED_dir = os.path.join(mantis_grid_dir, 'SED_' + str(i_SED + 1))
    #
    #     irods_proc.sendline('imkdir ' + mantis_SED_dir)
    #
    #     # copying the SED to mantis
    #     irods_proc.sendline('iput ' + elem_SED + ' ' + mantis_SED_dir)
    #
    #     if param_mode == 'all':
    #         # read the global sol file
    #
    #         # main parameter loop
    #         for param_comb, mhd_line in zip(param, mhd_lines):
    #             param_comb_dirname = "ha"
    #
    #             # creating the directory
    #             irocs_proc.sendline('imkdir ' + os.path.join(mantis_SED_dir, param_comb_dirname))
    #
    #             # creating a file for the individual solution (will be moved, defaulted to the folder where
    #             # the mhd solution is)
    #
    #             file_indiv = os.path.join(mhd_sol_dir, param_comb)
    #             with open(file_indiv, 'w+') as f_sol:
    #                 f_sol.write(mhd_line)
    #
    #             # creating the directory in mantis
    #             mantis_sol_dir = os.path.join(mantis_SED_dir, param_comb)
    #             irods_proc.sendline('imkdir ' + mantis_sol_dir)
    #
    #             # putting the solution file inside
    #             irods_proc.sendline('iput ' + os.path.join(mantis))
    #
    #
    #
    #     elif param_mode == 'decompose':
    #         for i_h_r, elem_h_r in enumerate(h_over_r_list):
    #             for i_p, elem_p in enumerate(p_list):
    #                 for i_mu, elem_mu in enumerate(mu_list):
    #                     for i_angle, elem_angle in enumerate(angle_list):
    #                         # implement nearest fetch in the solution file and folder decomposition
    #                         pass


    # jdl_path=os.path.join(mhd_sol_dir,'xstar_grid_'+param_mode+'_'+
    #                       (mhd_solutions_path.split('/')[-1] if param_mode=='all' else n_sol+'.jdl'))
    #
    # template='''
    #         {
    #           "name": "'''+'xstar_grid_'+param_mode+'_'+\
    #                       (mhd_solutions_path.split('/')[-1] if param_mode=='all' else n_sol)+'''",
    #           "resources": "/core='''+str(cores)+'''",
    #           "exec_file": "{HOME}/povray/start.bash",
    #           "exec_directory": "{HOME}/povray",
    #           "param_file": "{HOME}/povray_params.txt",
    #           "test_mode": "true",
    #           "type": "best-effort",
    #           "prologue": [
    #             "set -e",
    #             "source /applis/site/guix-start.sh",
    #             "cd {HOME}",
    #             "imkdir -p povray_results/{CAMPAIGN_ID}",
    #             "secure_iget -r -f povray",
    #             "chmod u+x {HOME}/povray/start.bash"
    #           ],
    #           "clusters": {
    #             "luke": {
    #               "project": "formation-ced-calcul",
    #               "walltime": "00:15:00"
    #             },
    #             "dahu": {
    #               "project": "formation-ced-calcul",
    #               "walltime": "00:10:00"
    #             },
    #             "bigfoot": {
    #               "project": "formation-ced-calcul",
    #               "walltime": "00:10:00",
    #               "properties": "gpumodel!='T4'"
    #             }
    #           }
    #         }
    #         '''

    # with open(jdl_path) as file:
    #     pass


    '''
    copying the arborescence to mantis
    '''


    # #creating the global mantis directory
    # irods_proc.sendline('imkdir '+mantis_grid_dir)



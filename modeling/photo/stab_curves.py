
import numpy as np
from simul_tools import xstar_func

def make_stab_curves(SED_path,xlum=1,n=1e12,v_turb=500,logNH=23.5, logxi_range=[0,6,0.03],
                     abund='solar',suffix='',solver='xstar',solver_mode='singularity',
                     force_kill_instance=False):

    '''

    Computes the stability curve by solving the thermal equilibrium  in a

    Args:
        SED_path: initial SD path used for illumination

        xlum: x-ray luminosity in units of 1e38 erg/s/cm^2

        n: density in cm^-3

        v_turb: turbulent velocity in km/s

        logNH: logarithm of the column density in cm^-2

        logxi_range: range of logxi to parse to build the stability curve

        abund: change abundances. Solar by default (for now nothing else is implemented)

        suffix: additional suffix for the output file

        solver: Radiative transfer code used to compute the stability curve

        force_kill_instance: kills and recreates the xstar instance at each iteration
    '''


    log_file=SED_path[:SED_path.rfind('.')]+'_xlog.txt'

    logxi_range=np.arange(logxi_range[0],logxi_range[1],logxi_range[2])

    t_guess=400

    for i_logxi,elem_logxi in enumerate(logxi_range):


        if solver=='xstar':

            #note: here we overwrite the guess temperature with the output temperature of the previous iteration
            #to increase the speed of the computations
            t_guess=xstar_func(SED_path, xlum, t_guess=t_guess, n=n, nh=10**(logNH), logxi=elem_logxi,vturb_x=v_turb,nbins=1000,
                   xstar_mode=solver_mode,path_logpars=log_file,
                       kill_container=i_logxi==len(logxi_range)-1 or force_kill_instance,
                       no_write=True,instance_identifier='auto',id_comput=str(i_logxi+1),return_temp=True)



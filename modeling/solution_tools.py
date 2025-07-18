import os,sys
import numpy as np
import glob

#rough way of testing if online or not
online=os.getcwd().startswith('/mount/src')
project_dir='/'.join(__file__.split('/')[:-3])

#to be tested online
sys.path.append(os.path.join(project_dir,'general/'))


from general_tools import print_log

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


def interp_yaxis(x_value,x_axis,y_axis,log_y=False,log_x=False,round=True):
    '''
    interpolates linearly or logarithmically the y values of the 2 closest x_axis values to x_value

    if round is set to True, returns directly a close value from x_axis if there's one
    (to avoid issues in equalities with numpy)
    '''

    #returning directly the y point to avoid issues in precision when the x_value is in the sample
    if x_value in x_axis:
        return y_axis[x_axis==x_value]

    #fetching the closest points
    min_mask=np.argmin(abs(x_axis-x_value))

    if round:
        if abs(x_value-x_axis[min_mask])<1e-6:
            return y_axis[min_mask]

    #note: we're indexing on the inverse side because the angles are decreasing
    if x_value<x_axis[min_mask]:
        min_mask=[min_mask+1,min_mask]
    else:
        min_mask=[min_mask,min_mask-1]

    closest_x=x_axis[min_mask]
    closest_y=y_axis[min_mask]

    #ordering them correctly for the x axis
    closest_y = closest_y[closest_x.argsort()]
    closest_x.sort()

    if log_y:

        assert (closest_y>0).all() or (closest_y<0).all(),'Sign change in log mode'

        sign_y=abs(closest_y[0])/closest_y[0]

        closest_y=np.log10(abs(closest_y))
    if log_x:
        assert (closest_x > 0).all(), 'Negative x in log mode'

        closest_x=np.log10(closest_x)

    # #interpolating (assuming linear here)
    coeff= (closest_y[1]-closest_y[0])  /(closest_x[1]-closest_x[0])
    ord_orig=closest_y[0]-coeff*closest_x[0]
    y_value=coeff*x_value+ord_orig

    if log_y:
        y_value=10**(y_value)*sign_y

    #directly giving it is easier
    # y_value=closest_y[0]+(closest_y[1]-closest_y[0])*(x_value-closest_x[0])/(closest_x[1]-closest_x[0])

    return y_value

def func_density_sol(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,m_BH):

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm

    cyl_cst=np.sqrt(1.0+(z_over_r*z_over_r))

    r_cyl = r_sph / cyl_cst

    return (mdot_mhd / (sigma_thomson_cgs * Rg_cgs)) * rho_mhd * (r_cyl ** (p_mhd - 1.5))

def func_density_relat_sol(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,vel_r,vel_phi,vel_z,m_BH):

    '''
    Relativistically corrected density, with expression from the PhD of Alfredo Luminari

    The factor is quite straightforward:
    you can always decompose the volume in one dimension parallel to the gas speed angle,
    which is dilated by a factor gamma, and two others with no dilatation. Then inversion because it's a density

    '''
    v_gas=func_vel_sol('tot',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)

    gamma_gas=1/np.sqrt(1-(v_gas/c_cgs)**2)

    return func_density_sol(r_sph, z_over_r, rho_mhd, p_mhd, mdot_mhd, m_BH) / gamma_gas

def func_temp_mhd(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,m_BH):

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm

    cyl_cst=np.sqrt(1.0+(z_over_r*z_over_r))
    r_cyl = r_sph / cyl_cst

    return (mdot_mhd / (sigma_thomson_cgs * Rg_cgs)) * rho_mhd * (r_cyl ** (p_mhd - 1.5))

def func_vel_sol(coordinate,r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH):

    '''

    returns various information about the relativistic speed of the gas

    The (special) relativistic computation requires the 3 components, so requires vel_r, vel_phi and vel_z
    '''

    cyl_cst = np.sqrt(1.0 + (z_over_r * z_over_r))
    r_cyl = r_sph / cyl_cst

    try:

        #nonrelat
        u_r_nr=c_cgs * vel_r * ((r_cyl) ** (-0.5))
    except:
        breakpoint()

    u_phi_nr = c_cgs * vel_phi * ((r_cyl) ** (-0.5))
    u_z_nr = c_cgs * vel_z * ((r_cyl) ** (-0.5))

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm

    u_nonrelat=np.sqrt(u_r_nr**2+u_phi_nr**2+u_z_nr**2)

    #done by hand
    u_relat=u_nonrelat/np.sqrt(1+u_nonrelat**2/c_cgs**2)

    gamma=1/np.sqrt(1-(u_relat/c_cgs)**2)

    if coordinate=='r':
        return u_r_nr*u_nonrelat/u_relat
    elif coordinate=='phi':
        return u_phi_nr *u_nonrelat/u_relat
    elif coordinate=='z':
        return u_z_nr *u_nonrelat/u_relat
    if coordinate=='obs':
        angle_rad=np.arctan(z_over_r)
        return (u_r_nr*np.cos(angle_rad)+u_z_nr*np.sin(angle_rad))*u_nonrelat/u_relat

    if coordinate=='tot':
        return u_relat

    if coordinate=='angle':
        # scalar product between the gas and the line of sight
        # pov of the gas so the gas speed is inverted, very important for the aberration
        # (which has coords (1,z_over_r,0) when putting r at 1
        scal_gas_los=-u_r_nr*u_nonrelat/u_relat*1/cyl_cst-u_z_nr*u_nonrelat/u_relat*z_over_r/cyl_cst
        # scal_gas_los=-u_r_nr*u_relat/u_nonrelat*1/cyl_cst-u_z_nr*u_relat/u_nonrelat*z_over_r/cyl_cst
        cos_angle=scal_gas_los/u_relat
        ####TODO: is any of thisis correct ? this needs to be fixed

        # relativistic aberration on the angle

        #note: if want to keep the same value with opposite angle convention,
        # also need to invert u_relat, it's an algebric value
        cos_angle_relat=(cos_angle-u_relat/c_cgs)/(1-u_relat/c_cgs*cos_angle)

        #returning a value in degrees
        # print('cos:')
        # print(cos_angle_relat)
        # print('angle')
        # print(np.arccos(cos_angle_relat)*180/np.pi)
        val=np.arccos(cos_angle_relat)*180/np.pi
        if np.isnan(val).all():
            pass
        return np.arccos(cos_angle_relat)*180/np.pi

def E_deboost_arg(v_gas,angle=180):

    '''
    Formula from the Luminari20 for the change in Energies in a more general case
    v_gas in cgs

    angle_gas_los is from the pov of the gas, so 180° is outwards motion (default)

    the angle computation considers the relatvistic aberration
    '''


    gamma_gas=1/np.sqrt(1-(v_gas/c_cgs)**2)
    beta_gas=v_gas/c_cgs

    psi=1/(gamma_gas*(1-beta_gas*np.cos(angle*np.pi/180)))
    if np.isnan(psi).all():
        pass

    return psi

def func_E_deboost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH):

    '''
    Formula from the Luminari20 for the change in Energies

    the angle computation considers the relatvistic aberration
    '''

    v_gas=func_vel_sol('tot',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)

    angle_gas_los=func_vel_sol('angle',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)

    gamma_gas=1/np.sqrt(1-(v_gas/c_cgs)**2)
    beta_gas=v_gas/c_cgs

    psi=1/(gamma_gas*(1-beta_gas*np.cos(angle_gas_los*np.pi/180)))
    if np.isnan(psi).all():
        pass

    return psi

def func_lum_deboost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH):

    '''
    Formula from the Luminari20 for the global deboost in luminosities

    the angle computation considers the relatvistic aberration
    '''

    #here if need to test things on func_E without breaking lum_deboos
    # v_gas=func_vel_sol('tot',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)
    #
    # angle_gas_los=func_vel_sol('angle',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)
    #
    # gamma_gas=1/np.sqrt(1-(v_gas/c_cgs)**2)
    # beta_gas=v_gas/c_cgs
    #
    # psi=1/(gamma_gas*(1-beta_gas*np.cos(angle_gas_los*np.pi/180)))
    #
    # return psi**4

    return func_E_deboost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)**4

def func_r_boost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH):
    
    '''
    Relativistic correction on the length from the gas pov
    
    use for the logxi computation and also to allow the correct radius

    From the PhD of Alfredo Luminari p.77 BUT inverted since the angle convention we use here is the opposite
    '''
    
    v_gas=func_vel_sol('tot',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)
    
    angle_gas_los=func_vel_sol('angle',r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)

    gamma_gas=1/np.sqrt(1-(v_gas/c_cgs)**2)
    
    return 1+(1- gamma_gas)*np.cos(angle_gas_los*np.pi/180)
    
# in this one, the distance appears directly so it should be the spherical one
def func_logxi_sol(r_sph,z_over_r,L_xi_Source,rho_mhd,p_mhd,mdot_mhd,vel_r,vel_phi,vel_z,m_BH,trig=False):

    '''

    NOTE: this is computed in the GAZ FRAME, hence all the relativistic effects

    in this one, the distance appears directly so it should be the spherical one

    Also considers the speed of the material at this distance, which deboosts the luminosity and contracts the length
    (see function above)
    '''

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm
    plop=L_xi_Source*func_lum_deboost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)\
                    / (func_density_relat_sol(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,vel_r,vel_phi,vel_z,m_BH) *\
                       (r_sph * Rg_cgs * func_r_boost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)) ** 2)
    if trig:
        breakpoint()
    return np.log10(L_xi_Source*func_lum_deboost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)\
                    / (func_density_relat_sol(r_sph,z_over_r,rho_mhd,p_mhd,mdot_mhd,vel_r,vel_phi,vel_z,m_BH) *\
                       (r_sph * Rg_cgs * func_r_boost_sol(r_sph,z_over_r,vel_r,vel_phi,vel_z,m_BH)) ** 2))

def func_nh_sol(r_sph,r_sph_0,z_over_r,rho_mhd,p_mhd,mdot_mhd,m_BH):

    '''
    here r_sph_0 is converted to non-spherical, so for a rj of 6 r_sph_0 should use 6*cyl_cst
    '''

    m_BH_SI = m_BH * Msol_SI
    Rs_SI = 2.0 * G_SI * m_BH_SI / (c_SI * c_SI)

    # !* Gravitational radius
    Rg_SI = 0.5 * Rs_SI
    Rg_cgs = Rg_SI * m2cm

    cyl_cst = np.sqrt(1.0 + (z_over_r * z_over_r))
    r_cyl=r_sph/cyl_cst
    r_cyl_0=r_sph_0/cyl_cst

    #adding a max condition to avoid issues with rounding for r_sph=rsph_0
    return np.array([max(0,elem) for elem in (mdot_mhd/(sigma_thomson_cgs*Rg_cgs))*rho_mhd/(p_mhd-0.5)*(r_cyl**(p_mhd-0.5)-r_cyl_0**(p_mhd-0.5))*Rg_cgs])


def load_solutions(solutions,mode='file',split_sol=False,split_par=False):

    '''
    loads the solutions from either a file or an array and splits them depending on the arguments passed

    returns an array
    '''

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



def sample_angle(solutions_path, angle_values, mdot_obs, m_BH, r_j=6., eta_mhd=1 / 12, xlum=None,
                 outdir=None,
                 return_file_path=False,mode='file',return_compton_angle=False,silent=True,
                 stop_at_compton=False):
    '''



    split the solution grid for a range of angles, up to the compton-thick point of each solution
        if stop_at_compton is set to True, otherwise to the given limit

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
        solutions_log_path = solutions_path.split('/')[-1].replace( \
            solutions_path_ext,
            '_angle_sampl_mdot_' + str(mdot_obs) +('' if mdot_obs != 'auto' else '_xlum_' + str(xlum)) +\
            '_m_bh_' + str(m_BH) + '_rj_' + str(r_j) + '_log' + solutions_path_ext)

        solutions_mod_path = solutions_path.split('/')[-1].replace( \
            solutions_path_ext,
            '_angle_sampl_mdot_' + str(mdot_obs) +('' if mdot_obs != 'auto' else '_xlum_' + str(xlum)) +\
            '_m_bh_' + str(m_BH) + '_rj_' + str(r_j) + solutions_path_ext)

        # adding the outdir into it
        solutions_log_path = os.path.join(outdir,solutions_log_path)

        # adding the outdir into it
        solutions_mod_path = os.path.join(outdir,solutions_mod_path)

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

    if mdot_obs=='auto':
        mdot_mhd=xlum/(1.26*m_BH)*2/eta_mhd
    else:
        mdot_mhd = mdot_obs*2/eta_mhd


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

            #no relativistic correction if we consider this in rest frame (0 speed at infinity)
            #this is a good approximation if we assume the gas won't become compton thick before getting to low speeds

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

        print_log('\n\n***************', solutions_log_io,silent)
        print_log('Solution:\n' +
                  'epsilon=' + str(solutions_split[0][0]) + '\nn_island=' + str(solutions_split[0][1]) +
                  '\np=' + str(solutions_split[0][2]) + '\nmu=' + str(solutions_split[0][3]), solutions_log_io,silent)

        print_log('\nCompton thick threshold at theta~'+str(ang_compton),solutions_log_io,silent)
        print_log('\nFirst solution below comp thick at theta=' + str(sol_angle_thick), solutions_log_io,silent)

        angle_values_nonthick=angle_values[angle_values<sol_angle_thick]

        if stop_at_compton:
            angle_values_select=angle_values_nonthick
        else:
            angle_values_select=angle_values

        # using it to determine how many angles will be probed (and adding a small delta to ensure the last value
        # is taken if it is a full one

        ####add log space option

        print_log('Angle sampling:', solutions_log_io,silent)
        print_log(angle_values_select, solutions_log_io,silent)

        n_angles += len(angle_values_select)

        # restricting to unique indexes to avoid repeating solutions
        id_sol_sample = np.unique([abs(angle_sol - elem_angle).argmin() for elem_angle in angle_values_select])

        print_log('Angles of solutions selected:', solutions_log_io,silent)

        print_log(angle_sol[id_sol_sample][::-1], solutions_log_io,silent)

        # and fetching the corresponding closest solutions
        solutions_sample += solutions_split[id_sol_sample].tolist()

    solutions_sample = np.array(solutions_sample)

    print_log('\ntot angles init:' + str(n_angles), solutions_log_io,silent)
    print_log('tot solutions:' + str(len(solutions_sample)), solutions_log_io,silent)

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


def merge_mhd_solution(solutions_path,tree='new'):

    '''
    Merges all the solutions inside the given folder into a unique csv with all variables directly
    the solutions folders should all be of the type 'eps_'+epsvalue
    inside, subfolders with each n, and inside each solution file
    '''

    startdir=os.getcwd()

    #extracting the solution directories
    os.chdir(solutions_path)

    sol_list=glob.glob('**/solN**.dat',recursive=True)

    global_sol_arr=[]

    for sol in sol_list:

        #list with the epsilon and n
        first_cols=[float(sol.split('/')[0].split('_')[-1].split('ep')[-1].replace('0','0.')),float((sol.split('/')[1].replace('n','')))]

        #reading the header of the file to get the second part of the solution elements
        with open(sol) as sol_file:
            sol_head = sol_file.readlines()[1:17]

        sol_lines_header = (sol_head[:8] + sol_head[10:13]) if tree=='new' else sol_head[:10]

        sol_parameters = np.array([elem.split()[-1] for elem in sol_lines_header]).astype(float).tolist()

        #list with all the relevant solution parameters (0 is xi/p, 2 is mu, 4 5 6 are chi_m/alpha_m/Pm)
        main_sol_pars=first_cols+[sol_parameters[0]]+[sol_parameters[2]]+sol_parameters[4:7]

        secondary_sol_pars=sol_parameters[7:10]

        #extracting the mhd solution lines
        sol_lines=np.loadtxt(sol,skiprows=17 if tree=='new' else 14)

        #merging them with the solution parameters
        merged_sol_line=[main_sol_pars+elem.tolist()+secondary_sol_pars for elem in sol_lines]

        global_sol_arr+=merged_sol_line

    global_sol_arr=np.array(global_sol_arr)

    header_arr='#epsilon\tn_island\tp_xi\tmu\tchi_m\talpha_m\tPm\tz_over_r\ttheta\tr_cyl/r0\trho_mhd\tu_r\tu_phi\tu_z'+\
               '\tT_MHD\tB_r\tB_phi\tB_z\tT_dyn\ty_id\ty_SM\ty_A'

    #saving the global file
    np.savetxt(os.getcwd().split('/')[-1]+'.txt', global_sol_arr, delimiter='\t',
               header=header_arr)

    os.chdir(startdir)
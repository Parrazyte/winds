import numpy as np
from general_tools import gamma_relat
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

def delta_relat(beta,theta):

    '''
    theta in degrees, aka the apparent doppler shift compared to the true velocity
    of the emitter and the inclination angle

    basic but see e.g. A1 in https://iopscience.iop.org/article/10.1086/133630
    '''

    tr=theta/180*np.pi

    return 1/(gamma_relat(beta)*(1-beta*np.cos(tr)))

def beta_from_delta(delta,theta):
    '''
    delta is the non-fractional doppler factor (see above), so >1 for blueshifts

    theta in degrees is the angle relative to the line of sight

    this can be used to derive the jet lorentz factor from doppler effect
    '''

    tr=theta*np.pi/180

    # r1=(delta**2*np.cos(tr)+np.sqrt(delta**2*(1-np.cos(tr)**2)+1))/\
    #    (1+delta**2*np.cos(tr)**2)
    #
    # r2=(delta**2*np.cos(tr)-np.sqrt(delta**2*(1-np.cos(tr)**2)+1))/\
    #    (1+delta**2*np.cos(tr)**2)

    #from wolfram alpha bc I'm garbage
    r1=(2*delta**2*np.cos(tr)+np.sqrt(2)*np.sqrt(delta**2*np.cos(2*tr)-delta**2+2))/ \
       (2*(1+delta**2*np.cos(tr)**2))

    r2=(2*delta**2*np.cos(tr)-np.sqrt(2)*np.sqrt(delta**2*np.cos(2*tr)-delta**2+2))/ \
       (2*(1+delta**2*np.cos(tr)**2))

    return r1,r2

def plot_beta_from_delta(delta):

    plt.figure()
    theta_sampl=np.arange(0, 90, 0.1)
    beta_res=np.array([beta_from_delta(delta,elem) for elem in theta_sampl]).T


    plt.plot(theta_sampl,beta_res[0])
    plt.plot(theta_sampl,beta_res[1])

def gamma_from_delta(delta,theta):

    '''
    for practicity
    '''

    b1,b2 = beta_from_delta(delta,theta)

    return gamma_relat(b1),gamma_relat(b2)

def min_pars_y(beta_app_forward):

    '''
    determines the minimum true lorentz factor from the forward facing jet
    depending on the inclination and the corresponding inclination angle
    check 10.1146/annurev.astro.37.1.409 p.11 for details about the apparent proper motions
    we then take the maximum possible beta_forward (so minmum beta in comparison), which happens for theta=arcos beta

    '''

    beta_min=beta_app_forward/np.sqrt(beta_app_forward**2+1)
    gamma_min=gamma_relat(beta_min)
    theta_beta_min=np.arccos(beta_min)*180/np.pi

    return beta_min,gamma_min,theta_beta_min

def find_roots(x, y):
    '''
    from https://stackoverflow.com/questions/46909373/how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0/46911822#46911822
    '''
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)

def par_space_gamma_theta(delta,beta_app_forward,plot_mode='both',figsize=(8,8)):
    '''
    plots the available parameter space for the two main equations dictating the blueshift
    of a relativistically accelerated source of emission (e.g. from x-ray spectroscopy)
    and the apparent velocity derived from proper motion (e.g. from radio)

    the y axis ca be both in beta mode or in delta mode
    '''

    #range theta (large sampling to help interpolation)
    theta_sampl=np.array(np.arange(0,80,0.001).tolist()+np.arange(80,90,0.00001).tolist())

    beta_sampl=np.linspace(0.,0.999,1000)

    #verified on wolfram
    theta_func_beta=np.arccos((delta-1/gamma_relat(beta_sampl))/(delta*beta_sampl))*180/np.pi

    beta_func_theta_bshift_fast=(2*delta**2*np.cos(theta_sampl*np.pi/180)+\
                np.sqrt(2)*np.sqrt(2-delta**2+delta**2*np.cos(2*theta_sampl*np.pi/180)))\
                    /(2*(1+delta**2*np.cos(theta_sampl*np.pi/180)**2))

    beta_func_theta_bshift_slow=(2*delta**2*np.cos(theta_sampl*np.pi/180)-\
                np.sqrt(2)*np.sqrt(2-delta**2+delta**2*np.cos(2*theta_sampl*np.pi/180)))\
                    /(2*(1+delta**2*np.cos(theta_sampl*np.pi/180)**2))
    #
    beta_func_theta_motion=beta_app_forward/(np.sin(theta_sampl*np.pi/180)+beta_app_forward*np.cos(theta_sampl*np.pi/180))

    roots_fast = find_roots(theta_sampl, beta_func_theta_motion - beta_func_theta_bshift_fast)
    roots_fast=roots_fast[~np.isnan(roots_fast)]

    roots_slow = find_roots(theta_sampl, beta_func_theta_motion - beta_func_theta_bshift_slow)
    roots_slow=roots_slow[~np.isnan(roots_slow)]

    #for the blueshift
    if plot_mode in ['beta','both']:

        plt.figure(figsize=figsize,layout='constrained')
        plt.xlabel(r'$\theta$ (°)')
        plt.ylabel(r'$\beta$')
        plt.xlim(0., 90.)

        #from blueshift equation
        # plt.plot(theta_func_beta,beta_sampl,
        #          color='blue',label='from blueshift')

        plt.plot(theta_sampl,beta_func_theta_bshift_fast,
                 color='blue',label='from blueshift fast')

        plt.plot(theta_sampl,beta_func_theta_bshift_slow,
                 color='orange',label='from blueshift slow')

        #from proper motion equation
        plt.plot(theta_sampl,beta_func_theta_motion,
                 color='green',
                 label='from proper motion')

        plt.ylim(0,1)

        plt.plot(roots_fast, np.interp(roots_fast, theta_sampl,beta_func_theta_bshift_fast),
                 marker="o", ls="", ms=4, color="violet")

        plt.plot(roots_slow, np.interp(roots_slow, theta_sampl,beta_func_theta_bshift_slow),
                 marker="o", ls="", ms=4, color="violet")

        plt.legend()

    if plot_mode in ['delta','both']:
        plt.figure(figsize=figsize,layout='constrained')
        plt.yscale('log')

        plt.xlabel(r'$\theta$ (°)')
        plt.ylabel(r'$\gamma$')
        plt.xlim(0., 90.)

        # from blueshift equation
        # plt.plot(theta_func_beta,beta_sampl,
        #          color='blue',label='from blueshift')

        gamma_func_theta_bshift_fast= gamma_relat(beta_func_theta_bshift_fast)
        gamma_func_theta_bshift_slow= gamma_relat(beta_func_theta_bshift_slow)

        plt.plot(theta_sampl,gamma_func_theta_bshift_fast,
                 color='blue', label='from blueshift fast')

        plt.plot(theta_sampl,gamma_func_theta_bshift_slow,
                 color='orange', label='from blueshift slow')

        gamma_func_theta_motion= gamma_relat(beta_func_theta_motion)

        # from proper motion equation
        plt.plot(theta_sampl, gamma_func_theta_motion,
                 color='green',
                 label='from proper motion')

        plt.plot(roots_fast, np.interp(roots_fast, theta_sampl,gamma_func_theta_bshift_fast), marker="o", ls="", ms=4, color="C1")
        plt.plot(roots_slow, np.interp(roots_slow, theta_sampl,gamma_func_theta_bshift_slow), marker="o", ls="", ms=4, color="C1")


        plt.ylim(1, plt.ylim()[1])
        plt.legend()

def sol_space_delta_vapp(figsize=(8,8)):

    delta_sampl=1+np.logspace(-3,1,300)
    vapp_sampl=np.logspace(-1,1,100)

    theta_sampl=np.array(np.arange(0,80,0.001).tolist()+np.arange(80,90,0.00001).tolist())
    beta_sampl=np.linspace(0.,0.999,1000)

    roots_theta=np.zeros((len(delta_sampl),len(vapp_sampl)))
    roots_beta=np.zeros((len(delta_sampl),len(vapp_sampl)))

    with tqdm(total=len(delta_sampl)) as pbar:
        for i_sampl,elem_delta in enumerate(delta_sampl):
            for j_sampl,elem_vapp in enumerate(vapp_sampl):
                beta_func_theta_bshift_fast = (2 * elem_delta ** 2 * np.cos(theta_sampl * np.pi / 180) + \
                                          np.sqrt(2) * np.sqrt(
                            2 - elem_delta ** 2 + elem_delta ** 2 * np.cos(2 * theta_sampl * np.pi / 180))) \
                                         / (2 * (1 + elem_delta ** 2 * np.cos(theta_sampl * np.pi / 180) ** 2))
                beta_func_theta_bshift_slow = (2 * elem_delta ** 2 * np.cos(theta_sampl * np.pi / 180) - \
                                          np.sqrt(2) * np.sqrt(
                            2 - elem_delta ** 2 + elem_delta ** 2 * np.cos(2 * theta_sampl * np.pi / 180))) \
                                         / (2 * (1 + elem_delta ** 2 * np.cos(theta_sampl * np.pi / 180) ** 2))
                #
                beta_func_theta_motion = elem_vapp / (
                            np.sin(theta_sampl * np.pi / 180) + elem_vapp * np.cos(theta_sampl * np.pi / 180))

                roots_fast = find_roots(theta_sampl, beta_func_theta_motion - beta_func_theta_bshift_fast)
                roots_fast = roots_fast[~np.isnan(roots_fast)]
                roots_fast= [elem for elem in roots_fast if elem!=0]

                roots_slow = find_roots(theta_sampl, beta_func_theta_motion - beta_func_theta_bshift_slow)
                roots_slow = roots_slow[~np.isnan(roots_slow)]
                roots_slow= [elem for elem in roots_slow if elem!=0]

                if len(roots_fast)+len(roots_slow)!=1:
                    breakpoint()
                if len(roots_fast)==1:
                    roots_theta[i_sampl][j_sampl]=roots_fast[0]
                    roots_beta[i_sampl][j_sampl]=np.interp(roots_fast, theta_sampl,beta_func_theta_bshift_fast)[0]
                else:
                    roots_theta[i_sampl][j_sampl]=roots_slow[0]
                    roots_beta[i_sampl][j_sampl]=np.interp(roots_slow, theta_sampl,beta_func_theta_bshift_slow)[0]
            pbar.update()

    dir="/home/parrazyte/Documents/Work/PostDoc/docs/docs_XRISM/MAXI J1744-294/paper/jets"
    np.save(os.path.join(dir,'roots_theta_'+str(int(1000*time.time()))),roots_theta)
    np.save(os.path.join(dir,'roots_beta_'+str(int(1000*time.time()))),roots_beta)

    #the delta-1 is here
    X,Y = np.meshgrid(delta_sampl-1,vapp_sampl)

    #making the theta figure
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\delta_{frac}$')
    plt.ylabel(r'$\beta_{app}$')

    img_theta=plt.pcolormesh(X,Y, roots_theta.T, vmin=0,vmax=90, cmap='plasma')
    colorbar = plt.colorbar(img_theta,spacing='proportional',ticks=np.arange(0,91,10))
    colorbar.set_label(r'$\theta')
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\delta_{frac}$')
    plt.ylabel(r'$\beta_{app}$')
    img_beta=plt.pcolormesh(X,Y, roots_beta.T, cmap='plasma',norm='log',vmin=8e-3,vmax=1)

    colorbar = plt.colorbar(img_beta,spacing='proportional')
    colorbar.set_label(r'$\beta_{intr}')

    #making the beta figure


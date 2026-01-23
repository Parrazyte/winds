import numpy as np
from general_tools import gamma_relat
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import glob

from bipolar import hotcold
cm_bipolar = hotcold(neutral=0)


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

def plot_beta_app(figsize=(8,8)):
    '''
    to visualize both front and back facing jets
    note that typically we take theta in [0,90] and beta positive or negative but here it's simpler
    to plot [0,180] and beta positive
    '''
    beta_sampl=(-np.logspace(-2,-1e-3,200)).tolist()+np.logspace(-2,-1e-3,200).tolist()
    beta_sampl.sort()
    theta_sampl=np.arange(0,90)

    beta_app=np.zeros((len(theta_sampl),len(beta_sampl)))
    for i_t, t in enumerate(theta_sampl):
        for i_b,b in enumerate(beta_sampl):
            beta_app[i_t][i_b]=b*np.sin(t*np.pi/180)/(1-b*np.cos(t*np.pi/180))

    fig,ax=plt.subplots(figsize=figsize)
    plt.yscale('log')
    plt.xlim(0,90)
    plt.yscale('symlog', linthresh=0.001, linscale=0.1)
    # plt.ylim(1e-2,1)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\beta$')
    X,Y = np.meshgrid(theta_sampl,beta_sampl)
    img_gamma=plt.pcolormesh(X,Y,beta_app.T,norm='linear',cmap=cm_bipolar.reversed())
    colorbar = plt.colorbar(img_gamma)
    colorbar.set_label(r'$\beta_{app}$')


def get_beta_roots(delta,beta_app_forward,theta_sampl,output='all'):

    '''
    computes the available parameter space for the two main equations dictating the blueshift
    of a relativistically accelerated source of emission (e.g. from x-ray spectroscopy)
    and the apparent velocity derived from proper motion (e.g. from radio)

    AND the intersection between the two curves
    there are two solutions for theta_bshift so we test both
    we interpolate the position of the intersection point if the sampling is not good enough

    note that here we disregard solutions for a backwards jet because they would be
    '''
    # verified on wolfram
    beta_func_theta_bshift_fast = (2 * delta ** 2 * np.cos(theta_sampl * np.pi / 180) + \
                                   np.sqrt(2) * np.sqrt(
                2 - delta ** 2 + delta ** 2 * np.cos(2 * theta_sampl * np.pi / 180))) \
                                  / (2 * (1 + delta ** 2 * np.cos(theta_sampl * np.pi / 180) ** 2))

    # verified on wolfram
    beta_func_theta_bshift_slow = (2 * delta ** 2 * np.cos(theta_sampl * np.pi / 180) - \
                                   np.sqrt(2) * np.sqrt(
                2 - delta ** 2 + delta ** 2 * np.cos(2 * theta_sampl * np.pi / 180))) \
                                  / (2 * (1 + delta ** 2 * np.cos(theta_sampl * np.pi / 180) ** 2))

    #note: this can give negative solutions but we don't want them since they would correspond to a backwards direction
    # verified on wolfram
    beta_func_theta_motion = beta_app_forward / (
                np.sin(theta_sampl * np.pi / 180) + beta_app_forward * np.cos(theta_sampl * np.pi / 180))

    #finding obvious roots with each beta_bshift solution
    roots_fast = find_roots(theta_sampl, beta_func_theta_motion - beta_func_theta_bshift_fast)
    roots_fast = roots_fast[~np.isnan(roots_fast)]
    roots_fast = [elem for elem in roots_fast if elem != 0]

    roots_slow = find_roots(theta_sampl, beta_func_theta_motion - beta_func_theta_bshift_slow)
    roots_slow = roots_slow[~np.isnan(roots_slow)]
    roots_slow = [elem for elem in roots_slow if elem != 0]

    #testing whether we should add an inbetween root
    last_ang_slow = theta_sampl[~np.isnan(beta_func_theta_bshift_slow)][-1]
    last_ang_fast = theta_sampl[~np.isnan(beta_func_theta_bshift_fast)][-1]

    if len(roots_fast) + len(roots_slow) == 0 and last_ang_slow == last_ang_fast:
        roots_slow += [last_ang_slow]

    return beta_func_theta_bshift_fast,beta_func_theta_bshift_slow,beta_func_theta_motion,roots_fast,roots_slow


def plot_par_space_gamma_theta(delta,beta_app_forward,plot_mode='beta',figsize=(8,8)):
    '''
    plots the available parameter space for the two main equations dictating the blueshift
    of a relativistically accelerated source of emission (e.g. from x-ray spectroscopy)
    and the apparent velocity derived from proper motion (e.g. from radio)

    the y axis ca be both in beta mode or in delta mode
    '''

    assert delta>1,'Are you sure you want a redshift?'
    #sampling the base parameters
    theta_sampl=np.arange(0,90,0.0001)
    beta_sampl=np.linspace(0.,0.999,1000)

    #verified on wolfram
    # theta_func_beta=np.arccos((delta-1/gamma_relat(beta_sampl))/(delta*beta_sampl))*180/np.pi

    beta_func_theta_bshift_fast, beta_func_theta_bshift_slow, beta_func_theta_motion, roots_fast, roots_slow=\
    get_beta_roots(delta,beta_app_forward,theta_sampl)

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

        plt.plot(roots_fast, np.interp(roots_fast, theta_sampl,
                                       beta_func_theta_motion),
                 marker="o", ls="", ms=4, color="violet")

        plt.plot(roots_slow, np.interp(roots_slow, theta_sampl,
                                       beta_func_theta_motion),
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

        plt.plot(roots_fast, np.interp(roots_fast, theta_sampl,gamma_func_theta_motion),
                 marker="o", ls="", ms=4, color="violet")
        plt.plot(roots_slow, np.interp(roots_slow, theta_sampl,gamma_func_theta_motion),
                 marker="o", ls="", ms=4, color="violet")


        plt.ylim(1, plt.ylim()[1])
        plt.legend()

def comput_sol_space_delta_vapp(delta_range=[-3,0,300],vapp_range=[-1,1,200]):

    delta_sampl=1+np.logspace(delta_range[0],delta_range[1],delta_range[2])
    vapp_sampl=np.logspace(vapp_range[0],vapp_range[1],vapp_range[2])

    #sampling the base parameters
    theta_sampl=np.arange(0,90,0.001)
    beta_sampl=np.linspace(0.,0.999,1000)

    roots_theta=np.zeros((len(delta_sampl),len(vapp_sampl)))
    roots_beta=np.zeros((len(delta_sampl),len(vapp_sampl)))

    with tqdm(total=len(delta_sampl)) as pbar:
        for i_sampl,elem_delta in enumerate(delta_sampl):
            for j_sampl,elem_vapp in enumerate(vapp_sampl):

                beta_func_theta_bshift_fast, beta_func_theta_bshift_slow, beta_func_theta_motion, roots_fast, roots_slow = \
                    get_beta_roots(elem_delta,elem_vapp, theta_sampl)

                gamma_func_theta_motion = gamma_relat(beta_func_theta_motion)

                if len(roots_fast)+len(roots_slow)!=1:
                    breakpoint()
                if len(roots_fast)==1:
                    roots_theta[i_sampl][j_sampl]=roots_fast[0]
                    roots_beta[i_sampl][j_sampl]=np.interp(roots_fast, theta_sampl,beta_func_theta_motion)[0]
                else:
                    roots_theta[i_sampl][j_sampl]=roots_slow[0]
                    roots_beta[i_sampl][j_sampl]=np.interp(roots_slow, theta_sampl,beta_func_theta_motion)[0]

            pbar.update()

    time_str=str(int(1000*time.time()))
    dir="/home/parrazyte/Documents/Work/PostDoc/docs/docs_XRISM/MAXI J1744-294/paper/jets"
    str_suffix='_delta_range_'+('_'.join(np.array(delta_range,dtype='str')))+\
               '_vapp_range_'+('_'.join(np.array(vapp_range, dtype='str')))
    np.save(os.path.join(dir,'roots_theta'+str_suffix+'_'+time_str),roots_theta)
    np.save(os.path.join(dir,'roots_beta'+str_suffix+'_'+time_str),roots_beta)


def func_beta(gamma):
    '''
    note: positive root
    '''
    return np.sqrt(1-(1/gamma**2))

def func_delta_from_delta_beta_app(delta,beta_app_forward):
    return (beta_app_forward**2+delta**2+1)/(2*delta)

def func_theta_from_delta_beta_app(delta,beta_app_forward):
    '''
    output in degrees
    '''
    return np.arctan(2*beta_app_forward/(beta_app_forward**2+delta**2-1))*180/np.pi

def plot_sol_space_delta_vapp(delta_range=[-3,0,300],vapp_range=[-1,1,200],figsize=(8,7),
                              dir="/home/parrazyte/Documents/Work/PostDoc/docs/docs_XRISM/MAXI J1744-294/paper/jets",
                              mode='analytical',
                              load_order='auto'):

    '''
    note that in the plot we show the fractional delta but we use the full (1+frac) delta in the computations above

    update: found the analytical formula in A6 and A7 of https://iopscience.iop.org/article/10.1086/133630
    no need to manually search for solutions

    mode:
        approx: loads solutions manually computed by comput_sol_space_delta_vapp
        analytical: uses the analytical formula
    '''



    #this is the factional delta
    delta_sampl=np.logspace(delta_range[0],delta_range[1],delta_range[2])

    vapp_sampl=np.logspace(vapp_range[0],vapp_range[1],vapp_range[2])

    #sampling the base parameters
    theta_sampl=np.arange(0,90,0.001)
    beta_sampl=np.linspace(0.,0.999,1000)

    #the delta-1 is here
    X,Y = np.meshgrid(delta_sampl,vapp_sampl)

    if mode=='approx':
        str_suffix='_delta_range_'+('_'.join(np.array(delta_range,dtype='str')))+\
                   '_vapp_range_'+('_'.join(np.array(vapp_range, dtype='str')))+'_'
        roots_theta_files=glob.glob(os.path.join(dir,'roots_theta'+str_suffix+'**'),recursive=True)
        roots_beta_files=glob.glob(os.path.join(dir,'roots_beta'+str_suffix+'**'),recursive=True)

        roots_theta_files.sort()
        roots_beta_files.sort()

        assert len(roots_theta_files)>0,'Error: no roots_theta files found for current parameter space'
        assert len(roots_beta_files)>0,'Error: no roots_beta files found for current parameter space'

        if load_order=='auto':
            print('loading '+roots_theta_files[-1])
            #loading the wo main arrays
            roots_theta=np.load(roots_theta_files[-1])

            print('loading '+roots_beta_files[-1])
            roots_beta=np.load(roots_beta_files[-1])
    else:
        roots_theta=func_theta_from_delta_beta_app(X+1,Y).T
        roots_beta=func_beta(func_delta_from_delta_beta_app(X+1,Y)).T

    #making the theta figure
    fig_theta,ax_theta=plt.subplots(figsize=figsize)
    ax_theta.set_aspect(3/2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\delta_{line}-1$')
    plt.ylabel(r'$\beta_{app}$')
    img_theta=plt.pcolormesh(X.T,Y.T,roots_theta, vmin=0,vmax=90, cmap='plasma')
    colorbar = plt.colorbar(img_theta,ticks=np.arange(0,91,10),fraction=0.05, pad=0.02)
    colorbar.set_label(r'$\theta$ (°)',rotation='horizontal')
    theta_contours_vals = np.arange(0,79,10)
    theta_contours_labels = np.arange(0,79,10).astype(str)+'°'
    theta_contours_ls = 'dashed'


    # beta_contours = ax_theta.contour(X.T, Y.T, roots_beta, levels=beta_contours_vals, colors='black', lw=1,
    #                                 linestyles=beta_contours_ls, label=beta_contours_labels)
    theta_contours = ax_theta.contour(X.T,Y.T,roots_theta, levels=theta_contours_vals, colors='black',
                                      linewidths=0.5,
                                      alpha=0.5,linestyles='-', label=theta_contours_labels)
    # beta_clabels = {}
    # for i,(l, s) in enumerate(zip(beta_contours.levels, beta_contours_labels)):
    #     beta_clabels[l] = s
    theta_clabels = {}
    for i,(l, s) in enumerate(zip(theta_contours.levels, theta_contours_labels)):
        theta_clabels[l] = s
    ax_theta.clabel(theta_contours, fontsize=10,fmt=theta_clabels)

    cax_theta=colorbar.ax

    #too complicated
    # #adding the high-i highlight
    # cs = ax_theta.contourf(X.T,Y.T,roots_theta,levels=[55,90],hatches=[''],extend='higher',
    #                   color='grey',lw=0.,alpha=0.1)
    # cax_theta.axhspan(55, 90, color='grey',alpha=0.3)

    theta_contour_highi = ax_theta.contour(X.T,Y.T,roots_theta, levels=[55], colors='white',lw=0.5,
                               linestyles='dashed')
    theta_highi_clabel = {}
    for i,(l, s) in enumerate(zip(theta_contour_highi.levels, [' 55° - absorption lines / dips '])):
        theta_highi_clabel[l] = s
    ax_theta.clabel(theta_contour_highi, fontsize=10,fmt=theta_highi_clabel)
    cax_theta.axhline(55,color='white',ls='dashed')

    eb_theta_abs=cax_theta.errorbar(0.5,55,5,0.,ecolor='white',lolims=True)
    eb_theta_abs[-1][1].set_linestyle('--')

    theta_contour_highi = ax_theta.contour(X.T,Y.T,roots_theta, levels=[80], colors='black',lw=0.5,
                               linestyles='dashed')
    theta_highi_clabel = {}
    for i,(l, s) in enumerate(zip(theta_contour_highi.levels, ['80° - eclipses'])):
        theta_highi_clabel[l] = s
    ax_theta.clabel(theta_contour_highi, fontsize=10,fmt=theta_highi_clabel)

    cax_theta.axhline(80,color='black',ls='dashed')
    eb_theta_ecl=cax_theta.errorbar(0.5,80,5,0.,ecolor='black',lolims=True)
    eb_theta_ecl[-1][1].set_linestyle('--')

    #adding SS433
    theta_contour_SS433 = ax_theta.contour(X.T, Y.T, roots_beta, levels=[0.26], colors='cyan', lw=1,
                                    linestyles='--')
    theta_clabel_SS433 = {}
    for i,(l, s) in enumerate(zip(theta_contour_SS433.levels,[r'SS433 $\beta_{intr}$'])):
        theta_clabel_SS433[l] = s

    ax_theta.clabel(theta_contour_SS433, fontsize=10,fmt=theta_clabel_SS433,manual=[(0.03,0.265)])

    #adding the lines for the two main detected blue components
    ax_theta.axvline(0.00689983,color='green',ls='dashed',lw=1,label=r'MAXI J1744 CIE')
    ax_theta.axvline(0.0204404,color='green',ls='dashed',lw=1)

    plt.legend()
    plt.savefig(os.path.join(dir,'jet_pars_theta_'+mode+'.pdf'))
    plt.savefig(os.path.join(dir,'jet_pars_theta_'+mode+'.png'),dpi=300)
    # #hatch is not super beautiful
    # # cs = ax_theta.contourf(X.T,Y.T,roots_theta,levels=[0,55,90],hatches=['/',''],
    # #                   color='none',linew=0.2,extend='both',alpha=0.)
    # # #manually changing the hatch color because it seems we can't do it wi
    # thin the function
    # # cs._hatch_color=(256,256,256, 0.2)
    # # breakpoint()
    #
    # ax_theta.clabel(beta_contours, fontsize=10,fmt=beta_clabels)
    plt.subplots_adjust(left=0.06)

    # making the theta figure
    fig_beta, ax_beta = plt.subplots(figsize=figsize)
    ax_beta.set_aspect(3/2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\delta_{line}-1$')
    plt.ylabel(r'$\beta_{app}$')
    img_beta = plt.pcolormesh(X.T, Y.T, roots_beta,norm='linear', cmap='plasma',vmin=0,vmax=1)
    colorbar = plt.colorbar(img_beta,fraction=0.05, pad=0.02)
    colorbar.set_label(r'$\beta_{intr}$',rotation='horizontal')
    beta_contours_vals = (np.arange(10)/10).tolist()+[0.99,0.999]
    beta_contours_labels = r'$\beta_{intr}=$'+np.array((np.arange(10)/10).tolist()+[0.99,0.999]).astype(str)
    beta_contours_ls = 'dashed'

    # theta_contours = ax_beta.contour(X.T,Y.T,roots_theta, levels=theta_contours_vals, colors='black',lw=1,
    #                            linestyles=theta_contours_ls, label=theta_contours_labels)
    beta_contours = ax_beta.contour(X.T, Y.T, roots_beta, levels=beta_contours_vals, colors='black', linewidths=0.5,
                                    alpha=0.5,linestyles='-', label=beta_contours_labels)

    # theta_clabels = {}
    # for i,(l, s) in enumerate(zip(theta_contours.levels, theta_contours_labels)):
    #     theta_clabels[l] = s
    beta_clabels = {}
    for i,(l, s) in enumerate(zip(beta_contours.levels, beta_contours_labels)):
        beta_clabels[l] = s

    # ax_beta.clabel(theta_contours, fontsize=10,fmt=theta_clabels)
    ax_beta.clabel(beta_contours, fontsize=10, fmt=beta_clabels,
                   manual=[(0.0017,0.107),
                           (0.03+0.017,0.197),
                           (0.03+0.017,0.304),
                           (0.0025,0.428),
                           (0.0025,0.548),
                           (0.0025, 0.701),
                           (0.0025, 0.9),
                           (0.0025, 1.15),
                           (0.002, 1.6),
                           (0.0025, 3.5),
                           (0.0025, 6.56),
                           ])

    #adding SS433
    beta_contour_SS433 = ax_beta.contour(X.T, Y.T, roots_beta, levels=[0.26], colors='cyan', lw=1,
                                    linestyles='--')
    beta_clabel_SS433 = {}
    for i,(l, s) in enumerate(zip(beta_contour_SS433.levels,[r'SS433 $\beta_{intr}$'])):
        beta_clabel_SS433[l] = s
    ax_beta.clabel(beta_contour_SS433, fontsize=10,fmt=beta_clabel_SS433,manual=[(0.002,0.265)])

    #adding the inclination zones
    beta_contour_highi = ax_beta.contour(X.T,Y.T,roots_theta, levels=[55], colors='white',lw=0.5,
                               linestyles='dashed')
    beta_highi_clabel = {}
    for i,(l, s) in enumerate(zip(beta_contour_highi.levels, [' 55° - absorption lines / dips '])):
        beta_highi_clabel[l] = s
    ax_beta.clabel(beta_contour_highi, fontsize=10,fmt=beta_highi_clabel,manual=[(0.01,1.4)])

    beta_contour_highi = ax_beta.contour(X.T,Y.T,roots_theta, levels=[80], colors='black',lw=0.5,
                               linestyles='dashed')
    beta_highi_clabel = {}
    for i,(l, s) in enumerate(zip(beta_contour_highi.levels, ['80° - eclipses'])):
        beta_highi_clabel[l] = s
    ax_beta.clabel(beta_contour_highi, fontsize=10,fmt=beta_highi_clabel)

    #adding the lines for the two main detected blue components
    ax_beta.axvline(0.00689983,color='green',ls='dashed',lw=1,label=r'MAXI J1744 CIE')
    ax_beta.axvline(0.0204404,color='green',ls='dashed',lw=1)

    plt.legend()
    plt.savefig(os.path.join(dir,'jet_pars_beta_'+mode+'.pdf'))
    plt.savefig(os.path.join(dir,'jet_pars_beta_'+mode+'.png'),dpi=300)

    # fig,ax=plt.subplots(figsize=figsize)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'$\delta_{frac}$')
    # plt.ylabel(r'$\beta_{app}$')
    # img_beta=plt.pcolormesh(X,X, roots_beta.T, cmap='plasma',norm='linear',vmin=0,vmax=1)
    # colorbar = plt.colorbar(img_beta)
    # colorbar.set_label(r'$\beta_{intr}$')

    # fig,ax=plt.subplots(figsize=figsize)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'$\delta_{frac}$')
    # plt.ylabel(r'$\beta_{app}$')
    # img_gamma=plt.pcolormesh(X,Y, gamma_relat(roots_beta.T), cmap='plasma',norm='log',vmin=1)
    # colorbar = plt.colorbar(img_gamma)
    # colorbar.set_label(r'$\gamma_{intr}$')

    #making the beta figure

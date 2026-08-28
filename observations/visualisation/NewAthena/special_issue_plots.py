import os
from xspec_config_multisp import *
from xspec import Xset,AllModels
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def model_compa_highE(save=None):
    #for mhd

    currdir=os.getcwd()
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto')

    Xset.restore('baseload.xcm')
    set_ener('thcomp',xrism=True)

    fig,ax=plt.subplots(figsize=(10,8))
    plt.axhline(1,color='gray',lw=0.5,alpha=0.5,ls='--')
    plt.axhline(0.1,color='gray',lw=0.5,alpha=0.5,ls='--')
    plt.axhline(0.01,color='gray',lw=0.5,alpha=0.5,ls='--')
    plt.axhline(0.001,color='gray',lw=0.5,alpha=0.5,ls='--')
    AllModels(1)(2).values=30.
    AllModels(1)(4).values=1.
    AllModels(1)(8).link='p4'
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')
    AllModels(1)(2).values=50.
    AllModels(1)(4).values=0.1
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')
    AllModels(1)(2).values=70.
    AllModels(1)(4).values=0.01
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')

    #since we cannot really go below 80 degrees for Ryota's setup
    AllModels(1)(2).values=80
    AllModels(1)(4).values=0.001
    AllModels(1)(8).link=''
    AllModels(1)(4).values=0.
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')
    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    ax.tick_params(labelbottom=True)

    plt.tight_layout()

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_total.xcm')
    set_ener('thcomp',xrism=True)
    # fig,ax=plt.subplots(figsize=(10,8))
    # plt.axhline(1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.01,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.001,color='gray',lw=0.5,alpha=0.5,ls='--')
    #basically 30 degrees
    AllModels(1)(1).values=0.85
    AllModels(1)(3).values=1.
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')
    AllModels(1)(1).values=np.cos(np.pi/180*50)
    AllModels(1)(3).values=0.1
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')
    AllModels(1)(1).values=np.cos(np.pi/180*70)
    AllModels(1)(3).values=0.01
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_emi.xcm')
    set_ener('thcomp',xrism=True)
    #basically 80 degrees
    AllModels(1)(1).values=0.16
    AllModels(1)(3).values=0.001
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')

    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()

    #if combining
    plt.gca().get_children()[-3].remove()
    ax.tick_params(labelbottom=True)
    plt.tight_layout()

    #replacing the legend
    plt.gca().get_children()[-2].remove()
    plt.plot([],[],color='red',label='thermal-radiative wind (no clumping)')
    plt.plot([],[],color='blue',label='cold MHD wind (no clumping)')
    plt.legend()
    plt.ylabel('residuals to continuum (shifted for clarity)')
    ax_right=ax.secondary_yaxis('right')
    ax_right.set_yticks([1e-3,1e-2,1e-1,1])
    ax_right.set_yticklabels(['80°\n(scatt. \nonly)','70°','50°','30°',])
    ax_right.set_ylabel('sightline (0°=face-on)')
    plt.ylim(1e-5,plt.ylim()[1])
    plt.tight_layout()

    ax.xaxis.set_minor_locator(MultipleLocator(0.02))

    os.chdir(currdir)
    if save is not None:
        plt.savefig(save)

def model_compa_lowE(save=None):

    currdir=os.getcwd()

    # for mhd
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto')

    Xset.restore('baseload.xcm')
    set_ener('thcomp', xrism=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.axhline(1, color='gray', lw=0.5, alpha=0.5, ls='--')
    plt.axhline(0.1, color='gray', lw=0.5, alpha=0.5, ls='--')
    plt.axhline(0.01, color='gray', lw=0.5, alpha=0.5, ls='--')
    plt.axhline(0.001, color='gray', lw=0.5, alpha=0.5, ls='--')
    AllModels(1)(2).values = 30.
    AllModels(1)(4).values = 1.
    AllModels(1)(8).link = 'p4'
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')
    AllModels(1)(2).values = 50.
    AllModels(1)(4).values = 0.1
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')
    AllModels(1)(2).values = 70.
    AllModels(1)(4).values = 0.01
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')

    # since we cannot really go below 80 degrees for Ryota's setup
    AllModels(1)(2).values = 80
    AllModels(1)(4).values = 0.001
    AllModels(1)(8).link = ''
    AllModels(1)(4).values = 0.
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')
    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()


    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_total.xcm')
    set_ener('thcomp', xrism=True)
    # fig,ax=plt.subplots(figsize=(10,8))
    # plt.axhline(1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.01,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.001,color='gray',lw=0.5,alpha=0.5,ls='--')
    # basically 30 degrees
    AllModels(1)(1).values = 0.85
    AllModels(1)(3).values = 1.
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')
    AllModels(1)(1).values = np.cos(np.pi / 180 * 50)
    AllModels(1)(3).values = 0.1
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')
    AllModels(1)(1).values = np.cos(np.pi / 180 * 70)
    AllModels(1)(3).values = 0.01
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_emi.xcm')
    set_ener('thcomp', xrism=True)
    # basically 80 degrees
    AllModels(1)(1).values = 0.16
    AllModels(1)(3).values = 0.001
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')

    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()

    # if combining
    plt.gca().get_children()[-3].remove()
    ax.tick_params(labelbottom=True)
    plt.tight_layout()

    # replacing the legend
    plt.gca().get_children()[-2].remove()
    plt.plot([], [], color='red', label='thermal-radiative wind (no clumping)')
    plt.plot([], [], color='blue', label='cold MHD wind (no clumping)')
    plt.legend()
    plt.ylabel('residuals to continuum (shifted for clarity)')
    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax_right.set_yticklabels(['80°\n(scatt. \nonly)', '70°', '50°', '30°', ])
    ax_right.set_ylabel('sightline (0°=face-on)')
    plt.ylim(3e-6, plt.ylim()[1])
    plt.tight_layout()

    ax.xaxis.set_minor_locator(MultipleLocator(0.02))

    os.chdir(currdir)
    if save is not None:
        plt.savefig(save)


def SED_soft_factors(edd_ratio=0.1,m_BH=8):

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/SED')

    Xset.restore('SED_soft_xspec_norenorm.xcm')

    set_ener('thcomp',xrism=False)

    AllModels.calcFlux("0.0136 13.6")

    flux_1_1000_Ryd = AllModels(1).flux[0]

    save_mo_Ryota('SED_soft_edd_'+str(edd_ratio)+'mbh_'+str(m_BH)+'.dat',
                  factor=edd_ratio*1.26e38*m_BH/flux_1_1000_Ryd)



def incl_NH_dep_HerX1():
    '''
    '''
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/Obs/Inclination')
    wind_arr = np.loadtxt('HerX1_XMM_Chandra_results.txt', skiprows=1).T
    #given by Peter

    #note: the uncertainties are with + first then - in this table, and - uncertainties are negative

    plt.figure()
    plt.errorbar(wind_arr[2], wind_arr[5], xerr=abs(wind_arr[3:5])[::-1], yerr=abs(wind_arr[6:8])[::-1], ls='')

    plt.xlabel('wind x position (cm)')
    plt.ylabel('wind y position (cm)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(6e9, 6.5e10)
    plt.ylim(1e9, 1.5e10)

    x_curves=np.logspace(8,12,300)
    [plt.plot(x_curves,x_curves*np.tan(i*np.pi/180),color='red',alpha=0.3) for i in range(1,89)]

    tan_theta=wind_arr[5]/wind_arr[2]
    tan_theta_bounds=[(wind_arr[5]+wind_arr[7])/(wind_arr[2]+wind_arr[3]),
                   (wind_arr[5]+wind_arr[6])/(wind_arr[2]+wind_arr[4])]
    
    theta=np.arctan(tan_theta)*180/np.pi
    theta_bounds=np.arctan(tan_theta_bounds)*180/np.pi
    theta_err=[theta-theta_bounds[0],theta_bounds[1]-theta]
    
    plt.figure()
    plt.errorbar(theta,wind_arr[8],xerr=theta_err,yerr=abs(wind_arr[9:11])[::-1],ls='')
    plt.xlabel('theta (°)')
    plt.ylabel(r'wind NH ($10^{22}$ cm$^-2$)')
    plt.yscale('log')

    pass

def compa_ion_par(logxi=[0,1,2,3,4],nh_22=[1,1,1,10,10],
                  np_14=None,v_rms=None,z=None,
                  cmap='plasma',set_ener_str='large_canon',
                  mtable='pionabsmtablecanonicallarge.fits',
                  xlims=[0.3,2.],
                  nh_cold=0.1,
                  ylims=[1e-15,1]):

    '''
    comparator for different sets of pion_abs solutions
    '''

    if np_14 is not None and type(np_14) not in (list,tuple,np.ndarray):
        np_14_use=np.repeat(np_14,len(logxi))
    else:
        np_14_use=np_14

    if v_rms is not None and type(v_rms) not in (list,tuple,np.ndarray):
        v_rms_use=np.repeat(v_rms,len(logxi))
    else:
        v_rms_use=v_rms

    if z is not None and type(z) not in (list,tuple,np.ndarray):
        z_use=np.repeat(z,len(logxi))
    else:
        z_use=z

    if type(nh_cold) not in (list,tuple,np.ndarray):
        nh_cold_use=np.repeat(nh_cold,len(logxi))
    else:
        nh_cold_use=nh_cold



    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/Obs/ion_soft')

    plop=Model('TBabs(diskbb)')

    addcomp('glob_mtable{'+mtable+'}')

    getattr(AllModels(1),AllModels(1).componentNames[0]).z.values=0.

    fig, ax = plt.subplots(figsize=(10, 8))

    color_cmap = getattr(mpl.cm,cmap)
    c_norm= mpl.colors.Normalize(vmin=0,
                                         vmax=len(logxi))
    set_ener(set_ener_str,xrism=True)
    time.sleep(1)
    colors_func = mpl.cm.ScalarMappable(norm=c_norm, cmap=color_cmap)

    AllModels.show()

    for i_model,(elem_logxi,elem_nh,elem_nh_cold) in enumerate(zip(logxi,nh_22,nh_cold_use)):
        AllModels(1)(1).values=elem_logxi
        AllModels(1)(2).values=elem_nh
        AllModels(1)(6).values=elem_nh_cold

        if np_14 is not None:
            AllModels(1)(3).values=np_14_use[i_model]
        if v_rms is not None:
            AllModels(1)(4).values = v_rms_use[i_model]
        if z is not None:
            AllModels(1)(5).values = z_use[i_model]

        AllModels.show()

        indiv_color = colors_func.to_rgba(i_model)
        xPlot('eemo',axes_input=ax,model_colors=[indiv_color],
              group_names=['logxi_'+str(elem_logxi)+'_nh'+str(elem_nh)+
                           ('' if np_14 is None else '_np_%.2e'%(np_14_use[i_model]*1e14))+
                            ('' if v_rms is None else '_v_rms_'+str(v_rms_use[i_model]))+
                           ('' if z is None else '_z'+str(v_rms_use[i_model]))+
                                                 ('_nhcold_'+str(elem_nh_cold))])
        ax.tick_params(labelbottom=True)

    plt.xscale('linear')

    if xlims is not None:
        plt.xlim(xlims)

    if ylims is not None:
        plt.ylim(ylims)


#for density
#compa_ion_par(logxi=np.repeat(3,7),nh_22=np.repeat(1,7),
# np_14=[0.001001     , 0.00316228, 0.01      , 0.03162278, 0.1       ,0.31622777, 1.        ],
# xlims=[1.,1.1],ylims=[8e-4,4e-3])

# compa_ion_par(logxi=np.repeat(3.,7),nh_22=np.repeat(0.1,7),
# np_14=[0.001001     , 0.00316228, 0.01      , 0.03162278, 0.1       ,0.31622777, 1.        ],xlims=[1.,1.1],ylims=[8e-4,4e-3])

#testing intrinsic lines at logxi=0
#compa_ion_par(logxi=[0,0,5],nh_22=[0.1,0.1,0.1],nh_cold=[0.1,2.,2.])

'''
for merching chandra spectra
in CIAO
combine_spectra src_spectra="fake_highxi_src_50ks*" src_arf="*garf" src_rmf="*grmf" clob+ bkg_spectra=none verbose=5 method=avg
'''

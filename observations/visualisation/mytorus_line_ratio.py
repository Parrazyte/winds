from xspec_config_multisp import *
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
os.chdir('/home/parrazyte/Soft/Xspec/Models/mytorus')

mytorus_table='mytl_V000HLZnEp000_v01.fits'
#getting the inclination, gamma and NH bins for the model


def save_flux_vals(table_path='/home/parrazyte/Soft/Xspec/Models/mytorus/mytl_V000HLZnEp000_v01.fits'):
    '''
    function to save the Fe Ka/Kb line flux values from mytorus table
    '''

    mytorus_table=table_path

    with fits.open(mytorus_table) as hdul:
        nh_range=np.unique(hdul[1].data[0][-2])
        incl_range=np.unique(hdul[1].data[1][-2])
        #removing the 0 value for gamma
        gamma_range=np.unique(hdul[1].data[2][-2])[1:]

    #loading the model
    test=Model('atable{'+mytorus_table+'}')

    #setting the parameters range

    set_ener('thcomp',xrism=True)

    FeKa_flux=np.zeros((len(nh_range),len(incl_range),len(gamma_range)))
    FeKb_flux=np.zeros((len(nh_range),len(incl_range),len(gamma_range)))

    for i_nh,nh in enumerate(nh_range):
        #rounding to avoid errors
        AllModels(1)(1).values = nh.round(3)
        for i_incl,incl in enumerate(incl_range):
            #note: using a ceiling here because the numerical angle steps values
            #are weird because they're made from cosine, so we want to avoid rounding
            #to the wrong step
            AllModels(1)(2).values = np.ceil(incl*100)/100
            for i_gamma,gamma in enumerate(gamma_range):
                #rounding to avoid errors, avoiding 2.6 because the upper par range is
                #2.59999999999
                AllModels(1)(3).values =gamma.round(3)-(0 if gamma!=2.6 else 0.001)

                #computing the flux values in 2 ranges
                AllModels.calcFlux('6.35 6.45')
                FeKa_flux[i_nh][i_incl][i_gamma]=AllModels(1).flux[0]

                #computing the flux values in 2 ranges
                AllModels.calcFlux('7.01 7.11')
                FeKb_flux[i_nh][i_incl][i_gamma]=AllModels(1).flux[0]


    #saving the table
    np.save(os.path.join(os.path.dirname(table_path),'FeKa_flux.npy'),FeKa_flux)
    np.save(os.path.join(os.path.dirname(table_path),'FeKb_flux.npy'),FeKb_flux)


def plot_ratios(table_path='/home/parrazyte/Soft/Xspec/Models/mytorus/mytl_V000HLZnEp000_v01.fits',
                dir='/home/parrazyte/Soft/Xspec/Models/mytorus/'):

    mytorus_table=table_path

    with fits.open(mytorus_table) as hdul:
        nh_range=np.unique(hdul[1].data[0][-2])
        incl_range=np.unique(hdul[1].data[1][-2])
        #removing the 0 value for gamma
        gamma_range=np.unique(hdul[1].data[2][-2])[1:]


    #transposed in a Gamma,incl,Nh range for convenience
    FeKa_flux=np.load(os.path.join(dir,'FeKa_flux.npy')).transpose(2,1,0)
    FeKb_flux=np.load(os.path.join(dir,'FeKb_flux.npy')).transpose(2,1,0)

    plt.ioff()

    for i_gamma,gamma in enumerate(gamma_range):

        plt.figure(figsize=(3*1.25,3),layout='constrained')

        plt.title(r'$\Gamma=$'+str(gamma))

        plt.xscale('log')
        plt.xlabel(r'NH ($10^{24}$ cm$^{-2}$)')
        plt.ylabel(r'FeII K$\beta$/ K$\alpha$ line core flux ratio')
        colors_val=np.cos(incl_range*np.pi/180).round(2)
        colors=col = plt.cm.plasma(colors_val)
        plt.ylim(0.135,0.21)

        plots=[]
        for i_incl,incl in enumerate(incl_range):

            plots+=[plt.plot(nh_range,FeKb_flux[i_gamma][i_incl]/FeKa_flux[i_gamma][i_incl],
                     color=colors[i_incl])]

        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        sm.set_array([])
        cb=plt.colorbar(sm,ax=plt.gca(),ticks=colors_val,label=r'$\theta$ (°)')
        cb.set_ticklabels(incl_range.astype(int))

        # plt.tight_layout()
        plt.savefig(os.path.join(dir,'Fe_K_flux_ratio_gamma_'+str(gamma).replace('.','p')+'.pdf'))
        plt.close()



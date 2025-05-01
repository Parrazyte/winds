

from xspec_config_multisp import addcomp,delcomp
from xspec import AllModels,AllData,Xset
from visual_line_tools import corr_factor_lbat,bat_rate_to_gamma,convert_BAT_count_flux
import numpy as np

from tqdm import tqdm


def make_nthcomp_table():

    '''
    samples a range of nthcomp parameter sto create a 3 dimensional table with
    kT, gamma, and flux at norm 1 for each couple of parameters
    '''

    AllModels.clear()
    AllData.clear()

    # creating a correct nthcomp
    addcomp('diskbb')
    addcomp('disk_nthcomp')
    delcomp('diskbb')

    mod_nth = AllModels(1)

    '''
    Table of nthcomp flux values depending on kt, gamma and the normalization
    we tabulate only kt and gamma, and compute a flux value with a normalization of 1 
    The flux scales linearly with the normalization
    '''

    #range of T
    kt_space=np.arange(0.5,2.01,0.01)
    gamma_space=np.arange(1.,3.5,0.01)

    n_vals=len(kt_space)*len(gamma_space)
    array_vals=np.zeros((n_vals,3))

    Xset.chatter=0

    with tqdm(total=n_vals) as pbar:
        for i_kt,elem_kt in enumerate(kt_space):
            for i_gamma,elem_gamma in enumerate(gamma_space):

                mod_nth.nthComp.kT_bb.values = elem_kt
                mod_nth.nthComp.Gamma.values = elem_gamma
                mod_nth.nthComp.norm.values = 1

                AllModels.calcFlux("15 50")
                norm_flux = AllModels(1).flux[0]

                i_arrtot=i_gamma+i_kt*len(gamma_space)

                #rounding to avoid numpy issues
                array_vals[i_arrtot][0] = round(elem_kt,4)
                array_vals[i_arrtot][1] = round(elem_gamma,4)

                array_vals[i_arrtot][2] = norm_flux

                pbar.update()

    Xset.chatter=10

    np.savetxt('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/nthcomp_table.txt',
               array_vals,header='kt\tgamma\tflux at norm 1')

nthcomp_tab=np.loadtxt('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/nthcomp_table.txt')

Edd_factor_dict={
    "4U1630-47":7789529.26214016}
def get_nthcomp_pars(BAT_rate,kT,source_name='4U1630-47',nthcomp_table=nthcomp_tab):

    '''
    Reverses a nthcomp SED from a kT and BAT 15-50 keV rate.
    The rate is converted from gamma (lower limit at 2) from the rate/Gamma conversion
    It is also converted in flux from the rate/flux conversion (two steps with the webpimms conversion and its correction)

    Then fetches the nthcomp table for the best match and returns the corresponding parameters
    '''

    #here we do a back and forth multiplication by the eddington factor because we need the corr_factor
    #in Edd units, and then we need to retrive the flux after. This is not equal to not putting it bc the corr_factor
    #is not linear
    conv_flux=corr_factor_lbat(BAT_rate*convert_BAT_count_flux[source_name]*Edd_factor_dict[source_name])\
                /Edd_factor_dict[source_name]

    print('converted flux %.3e'%conv_flux)
    conv_gamma=round(bat_rate_to_gamma(BAT_rate),2)

    conv_flux_norm_1=nthcomp_tab.T[2][(nthcomp_tab.T[0]==kT) & (nthcomp_tab.T[1]==conv_gamma)]

    assert len(conv_flux_norm_1)==1,'Issue when fetching flux values for selected couple of parameters :'+\
                                    str(kT)+' '+str(conv_gamma)

    conv_norm=conv_flux/conv_flux_norm_1[0]

    return kT,conv_gamma,conv_norm


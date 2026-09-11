import matplotlib.pyplot as plt
from stingray import Lightcurve, Powerspectrum, Crossspectrum
from astropy.io import fits

def plot_PDS(lc,save=True):
    if save:
        plt.ioff()

    data_lc=fits.open(lc)[1].data
    sting_lc=Lightcurve(data_lc['TIME'],data_lc['RATE'],data_lc['ERROR'])
    sting_pow=Powerspectrum(sting_lc,norm='frac')

    plt.figure(layout='constrained',figsize=(8,6))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('norm (frac. rms)')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('PDS (unbinned)')
    plt.plot(sting_pow.freq,sting_pow.power)

    if save:
        plt.savefig(lc[:lc.rfind('.')]+'_pds_unbinned.pdf')
        plt.close()

    sting_pow_rebin=sting_pow.rebin_log(0.01)
    plt.figure(layout='constrained',figsize=(8,6))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('norm (frac. rms)')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('PDS (0.01 log rebinned)')
    plt.plot(sting_pow_rebin.freq,sting_pow_rebin.power)

    if save:
        plt.savefig(lc[:lc.rfind('.')] + '_pds_binned.pdf')
        plt.close()

def plot_cross_NuSTAR(lc,save=True):

    '''
    from timing_tools import plot_cross_NuSTAR

    for elem in glob.glob('**FPMA_lc_src_**0.002.lc'):
    plot_cross_NuSTAR(elem, save=True)


    '''
    if save:
        plt.ioff()

    data_lc=fits.open(lc)[1].data
    sting_lc_A=Lightcurve(data_lc['TIME'],data_lc['RATE'],data_lc['ERROR'])
    data_lc_fpmb=fits.open(lc.replace('FPMA','FPMB'))[1].data
    sting_lc_B=Lightcurve(data_lc_fpmb['TIME'],data_lc_fpmb['RATE'],data_lc_fpmb['ERROR'])

    sting_cross=Crossspectrum(sting_lc_A,sting_lc_B,norm='frac')

    plt.figure(layout='constrained',figsize=(8,6))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('norm (frac. rms)')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('Cross Spectrum (unbinned)')
    plt.plot(sting_cross.freq,sting_cross.power)

    if save:
        plt.savefig(lc[:lc.rfind('.')]+'_cspec_unbinned.pdf')
        plt.close()

    sting_cross_rebin=sting_cross.rebin_log(0.01)
    plt.figure(layout='constrained',figsize=(8,6))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('norm (frac. rms)')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('Cross Spectrum (0.01 log rebinned)')
    plt.plot(sting_cross_rebin.freq,sting_cross_rebin.power)

    if save:
        plt.savefig(lc[:lc.rfind('.')] + '_cspec_binned.pdf')
        plt.close()


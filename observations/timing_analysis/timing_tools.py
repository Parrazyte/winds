import matplotlib.pyplot as plt
from stingray import Lightcurve, Powerspectrum
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


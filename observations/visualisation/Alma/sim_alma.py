
import matplotlib.pyplot as plt
import numpy as np
'''
#python installation of the wheels with

pip3.10 install casaconfig==1.0.2,casatools==6.7.0.31,casatasks==6.7.0.31,casaplotms==2.6.2,casaviewer==2.3.2, casashell==6.7.0.31, casaplotserver==1.9.2, casatestutils==6.7.0.31, casatablebrowser==0.0.37, casalogger==1.0.21, casafeather==0.0.24, casampi==0.5.6

'''



'''
1. Create simulated lightcurve from QRM obs power spectrum


'''


alma_1s_default_sensitivity=0.4686425943451101

frequency_ranges={

    #10.1088/2041-8205/754/2/L23

    #from https://ui.adsabs.harvard.edu/abs/2022ApJ...937...33Y/abstract
    '4U1630-47':[15,19],

    #from https://dx.doi.org/10.3847/1538-4357/ad58d1
    'MAXIJ1348-630':[0.08,0.15],
}

rms_ranges={
    '4U1630-47':[0.11,0.165],
    'MAXIJ1348-630':[0.03,0.09],
}

plt.plot(np.arange(10))


#  Set simalma to default parameters
# default("simalma")
#
# # Our project name will be "m51", and all simulation products will be placed in a subdirectory "m51/"
# project="m51"
# overwrite=True
#
# # Model sky = H_alpha image of M51
# os.system('curl https://casaguides.nrao.edu/images/3/3f/M51ha.fits.txt -f -o M51ha.fits')
# skymodel="M51ha.fits"
#
# # Set model image parameters:
# indirection="J2000 23h59m59.96s -34d59m59.50s"
# incell="0.1arcsec"
# inbright="0.004"
# incenter="330.076GHz"
# inwidth="50MHz"
# antennalist=["alma.cycle5.3.cfg","aca.cycle5.cfg"]
# totaltime="1800s"
# tpnant = 2
# tptime="7200s"
# pwv=0.6
# mapsize="1arcmin"
# dryrun = False
# simalma()
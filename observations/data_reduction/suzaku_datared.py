#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:42:43 2022

@author: parrama
"""

import pexpect
import sys
import glob
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time,TimeDelta

def plot_suzaku_lc(camera='all'):

    plt.ioff()
    def plot_single_lc(lc_path):

        with fits.open(lc_path) as fits_lc:
            # time zero of the lc file (different from the time zeros of the gti files)
            time_zero = Time(fits_lc[1].header['MJDREFI'] + fits_lc[1].header['MJDREFF'], format='mjd') + \
                            TimeDelta(fits_lc[1].header['TIMEZERO'], format='sec')
            time_zero=time_zero.to_datetime()
            time_zero=str(time_zero)

            binning = fits_lc[1].header['TIMEDEL']

            obsid = fits_lc[1].header['OBS_ID']

            telescope = fits_lc[1].header['TELESCOP']

            instru = fits_lc[1].header['INSTRUME'] + ' (mode ' + fits_lc[1].header['DETNAM']+')'

            object=fits_lc[1].header['OBJECT']

            time = fits_lc[1].data['TIME']

            rate = fits_lc[1].data['RATE']

            rate_err = fits_lc[1].data['ERROR']

        indiv_band=lc_path.split('subbgd')[-1].split('kev')[0]

        if indiv_band[0]=='_':
            indiv_band=indiv_band[1:]

        indiv_band=indiv_band.replace('_','-')

        title = telescope+' '+instru+' lightcurve for observation ' + obsid + ' of '+object+' in the '+\
                indiv_band + ' keV band'

        lc_screen_path = elem_lc.replace('.lc', '.png')

        # and plotting it
        plt.subplots(1, figsize=(10, 8))

        plt.errorbar(time+0.5,rate, xerr=binning,yerr=rate_err, ls='-', lw=1, color='grey', ecolor='blue')

        plt.suptitle(title)

        plt.xlabel('Time (s) after ' + time_zero)
        plt.ylabel('RATE (counts/s)')

        plt.tight_layout()
        plt.savefig(lc_screen_path)
        plt.close()

    allfiles = glob.glob('**', recursive=True)

    suzaku_lc = [elem for elem in allfiles if
                   elem.endswith('.lc') and ( camera + '_' in elem if camera != 'all' else True)]

    for elem_lc in suzaku_lc:

        print("plotting lightcurve "+elem_lc+'...')
        plot_single_lc(elem_lc)
        print("plotting complete")

def regroup_Suzaku_spectra(extension='megumi', group='opt', camera='all',skip_started=True):

    '''To be launched above the folders where to regroup'''

    if extension=='megumi':
        extension_use='_spec_src.pha'

    def ft_group(file, grptype,file_type='std',directory=None):

        '''wrapper for the command'''

        if 'min' in grptype:
            group_pref = grptype.split('_')[0]
            group_val = grptype.split('_')[1]
        else:
            group_pref = grptype
            group_val = ''

        if group_val != '':
            minval_str = ' groupscale=' + group_val
        else:
            minval_str = ''

        if file_type=='pin':

            rsp_file=glob.glob(os.path.join(directory,'ae_hxd_*.rsp'),recursive=True)[0]

            ftgroup_str='ftgrouppha infile=' + file + ' outfile=' + file.replace('.','_grp_' + grptype + '.')+ \
                        ' grouptype=' + group_pref + minval_str + ' respfile=' + \
                        rsp_file + ' clobber=yes'
        else:
            ftgroup_str='ftgrouppha infile=' + file + ' outfile=' + file.replace('.','_grp_' + grptype + '.') +\
                    ' grouptype=' + group_pref + minval_str +' respfile=' +\
                    file.replace(file[file.rfind('.'):], '.rmf').replace('_sp_src','') + ' clobber=yes'

        if extension=='megumi':
            ftgroup_str=ftgroup_str.replace('_src.rmf','_rsp.rmf')

        heas_proc.sendline(ftgroup_str)

        heas_proc.sendline('echo done')
        heas_proc.expect('done')

    allfiles = glob.glob('**', recursive=True)

    suzaku_spectra = [elem for elem in allfiles if
                   elem.endswith(extension_use) and ( camera + '_' in elem if camera != 'all' else True)]

    suzaku_dirs = os.getcwd() + '/' + np.array([elem[:elem.rfind('/') + 1] for elem in suzaku_spectra]).astype(object)

    if extension=='megumi':
        suzaku_spectra_pin=[elem for elem in allfiles if
                   elem.endswith('src_dtcor.pha') and ( camera + '_' in elem if camera != 'all' else True)]

        suzaku_dirs_pin=os.getcwd() + '/' + np.array([elem[:elem.rfind('/') + 1] for elem in suzaku_spectra_pin]).astype(object)

        suzaku_spectra+=suzaku_spectra_pin

        suzaku_dirs=suzaku_dirs.tolist()+suzaku_dirs_pin.tolist()

    for ind, specpath in enumerate(suzaku_spectra):

        if os.path.isfile(specpath.replace('.pha', '_grp_' + group + '.pha')) and skip_started:
            print(specpath + ' already grouped. Skipping...\n')
            continue

        # spawning heasoft spectra for Kastra grouping
        heas_proc = pexpect.spawn('/bin/bash', encoding='utf-8')
        heas_proc.logfile = sys.stdout
        heas_proc.sendline('\nheainit')

        specfile = specpath.split('/')[-1]
        # sending the ciao process in the right directory

        # stat grouping

        heas_proc.sendline('cd ' + suzaku_dirs[ind])

        if group is not None:

            if 'pin_src' in specpath.split('/')[-1]:
                file_type='pin'
                directory='./'

            else:
                file_type='std'
                directory=None
            ft_group(specfile, grptype=group,file_type=file_type,directory=suzaku_dirs[ind])

            condition = os.path.isfile(specpath.replace('.pha', '_grp_' + group + '.pha'))

            while not condition:
                time.sleep(1)

                condition = os.path.isfile(specpath.replace('.pha', '_grp_' + group + '.pha'))

        heas_proc.sendline('exit')


def batch_mover():
    '''
    copies all  products in a directory to a bigbatch directory above the obsid directory to prepare for spectrum analysis
    Also adds obsids to each names for distinction

    Here for megumi's suzaku dr, we can copy everything
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    bashproc.logfile_read = sys.stdout
    bashproc.logfile_write = sys.stdout
    print('\n\n\nCopying spectral products to a merging directory...')

    currdir=os.getcwd()


    bashproc.sendline('mkdir -p bigbatch')

    topdirs=glob.glob('**/',recursive=False)
    topdirs_obsid=[elem for elem in topdirs if len(elem)==10 and elem[:-1].isdigit()]

    for elem_obsdir in topdirs_obsid:

        print("Merging elements of directory "+elem_obsdir)
        obsid=elem_obsdir.split('/')[0]

        obsid_files=[elem for elem in glob.glob(elem_obsdir+'**',recursive=True) if not os.path.isdir(elem)]

        # breakpoint()
        for elem_file in obsid_files:

            file_name=elem_file.split('/')[-1]
            #
            # if elem_obsdir!='400010060/':
            #     breakpoint()
            # + ' >'+elem_obsdir+'batch_mover.log' will overwrite the log for each file so not really useful here
            bashproc.sendline('cp --verbose ' + elem_file + ' bigbatch/')

            bashproc.sendline('mv bigbatch/'+file_name+' bigbatch/'+obsid+'_'+file_name)

            time.sleep(0.1)

            bashproc.expect('->')

    # reasonable waiting time to make sure files can be copied
    time.sleep(2)

    bashproc.sendline('exit')

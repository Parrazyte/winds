#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from astropy.io import ascii

import numpy as np
import pexpect
import time

from xspec import AllModels,AllData,Model,FakeitSettings,Plot

def xstar_to_table(path, outpath='xspecmod.fits', modname=None):
    '''
    Converts an xstar output spectra to a model table usable by xspec (atable)
    adapted from pop's script

    if modname is None, uses the file name in outpath as model name
    '''

    currdir = os.getcwd()

    filedir = '' if '/' not in path else path[:path.rfind('/')]

    outdir = '' if '/' not in outpath else outpath[:outpath.rfind('/')]
    outfile = outpath[outpath.rfind('/') + 1:]

    outprefix = outfile[:outfile.rfind('.')]

    file = path[path.rfind('/') + 1:]

    filedir = path.replace(file, '')

    # filename=path[:path.rfind('.')]

    tmp = np.genfromtxt(path, skip_header=1)
    enemin = tmp[:, 0] * 1e-3  # keV
    eneminsub = enemin[np.where(enemin > 1e-4)]
    flux = tmp[:, 1] * 1e38 / (4. * 3.14 * (10. * 1e3 * 3e18) ** 2.)  # erg/s/erg/cm2
    fluxsub = flux[np.where(enemin > 1e-4)]
    enemaxsub = np.roll(eneminsub, -1)
    enemeansub = 0.5 * (enemaxsub + eneminsub)
    l = len(eneminsub)

    spec = fluxsub / (enemeansub * 1.6e-9)  # ph/s/erg/cm2

    #this doesn't work
    # spec = spec * (enemaxsub - eneminsub) * 1.6e-9  # ph/s/cm2

    #note: this works, although I don't know why
    spec = spec *(enemaxsub - eneminsub)**(1/2) * 1.6e-9

    ascii.write([eneminsub[0:l - 1], enemaxsub[0:l - 1], spec[0:l - 1]], outprefix + '_tmp.txt', overwrite=True)

    os.system("sed '/col0/d' " + outprefix + "_tmp.txt > " + outprefix + ".txt")

    # spawning a bash process to produce the table from the txt modified SED
    heaproc = pexpect.spawn('/bin/bash', encoding='utf-8')
    heaproc.logfile = sys.stdout
    heaproc.sendline('heainit')
    heaproc.sendline('cd ' + currdir)

    # here, outfile is the name of the model
    heaproc.sendline('ftflx2tab ' + outprefix + '.txt ' + outprefix + ' ' + outfile + ' clobber = yes')

    # waiting for the file to be created before closing the bashproc
    while not os.path.isfile(os.path.join(currdir, filedir, outfile)):
        time.sleep(1)

    # removing the temp product file
    heaproc.sendline('rm ' + outprefix + '_tmp.txt')
    heaproc.sendline('rm ' + outprefix + '.txt')
    heaproc.sendline('exit')

    if outdir != '':
        os.system('mv ' + os.path.join(currdir, filedir, outfile) + ' ' + outpath)


def freeze(model=None, modclass=AllModels, unfreeze=False, parlist=None):
    '''
    freezes/unfreezes an entire model or a part of it
    if no model is given in argument, freezes the first existing models
    (parlist must be the list of the parameter numbers)
    '''
    if model is not None:
        xspec_mod = model
    else:
        xspec_mod = modclass(1)

    if parlist is None:
        for par_index in range(1, xspec_mod.nParameters + 1):
            xspec_mod(par_index).frozen = int(not (unfreeze))

    else:
        for par_index in parlist:
            xspec_mod(par_index).frozen = int(not (unfreeze))


def create_fake_xstar(table, rmf, arf, exposure, nofile=True, reset=True, prefix=""):
    if reset:
        AllModels.clear()
        AllData.clear()

    # loading the model table in an xspec model
    tablemod = Model('atable{' + table + '}')

    # freezing the model to avoid problems in case the table has free parameters
    freeze()

    # creating the fakeit settings
    fakeset = FakeitSettings(response=rmf, arf=arf, exposure=exposure)

    # launching the fakeit
    AllData.fakeit(settings=fakeset, applyStats=True, noWrite=nofile, filePrefix=prefix)


def model_to_nuLnu(path):
    # store the current model's nuLnu in a file through

    curr_xaxis = Plot.xAxis

    Plot.xAxis = "Hz"

    Plot('eeuf')

    x_arr = Plot.x()

    y_arr = Plot.model()

    save_arr = np.array([x_arr, y_arr]).T

    np.savetxt(path, save_arr, header='nu(Hz) Lnu(erg/s/Hz)', delimiter=' ')

def gridmodel(gridpath, interp='log'):
    '''
    Fetches all final spectra in the gridpath arborescence,
    and creates an xspec model using the
    '''


    final_sp_list=glob.glob('**/**sp_tr_rest_final**',recursive=True)

    '''
    here we convert the first part of the parameters, from the SED file description
    the first parameter is the SED
    the second is the mdot, converted from the xlum if needed through the Black Hole Mass
    the third is the black hole mass itself
    the fourth is the starting radius of the SAD rj
    '''
    param_list_1=[[elem.split('/')[0].split('mdot_')[0][:-1],
                          (elem.split('/')[0].split('mdot_')[1].split('_')[0] if \
                              elem.split('/')[0].split('mdot_')[1].split('_')[0]!='auto' else\
                          float(elem.split('/')[0].split('xlum_')[1].split('_')[0])/ \
                           (1.26*float(elem.split('/')[0].split('m_bh_')[1].split('_')[0]))),\
                          float(elem.split('/')[0].split('m_bh_')[1].split('_')[0]),
                          float(elem.split('/')[0].split('rj_')[1].split('_')[0])]\
                  for elem in final_sp_list]

    #here we convert the second part of the parameters, from the rest of the description
    param_list_2=[[float(elem.split('/')[1].split('eps_')[1]),
                   float(elem.split('/')[3].split('p_')[1].split('_')[0]),
                   float(elem.split('/')[3].split('mu_')[1].split('_')[0]),
                   float(elem.split('/')[4].split('angle_')[1])] for elem in final_sp_list]

    param_list=np.array([param_list_1[i_sol]+ param_list_2[i_sol] for i_sol in range(len(final_sp_list))],dtype=object).T

    #ensuring the mdot is in float, as it's still in string if we're using direct mdot values to avoid an error when
    #that's not the case
    param_list[1]=param_list[1].astype(float)

    #listing the parameters in a tuple

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from astropy.io import ascii

import numpy as np
import pexpect
import time

import pyxstar as px

from xspec import AllModels,AllData,Model,FakeitSettings,Plot

def xstar_out_to_table(out_prefix,folder='./',modname=None,xlum='auto',dist=10):

    '''
    Converts the transmitted, emitted, and incident spectrum to files and xspec atables.
    '''

    os.chdir(folder)

    px.LoadFiles()
    out_sp=px.ContSpectra()

    tr_arr= np.array([out_sp.energy,out_sp.transmitted]).T
    inc_arr= np.array([out_sp.energy,out_sp.incident]).T
    em_out_arr=np.array([out_sp.energy,out_sp.emit_outward]).T


    # !**Writing the spectra in files
    tr_txt_path='./xout_'+out_prefix+'_tr.txt'
    inc_txt_path='./xout_'+out_prefix+'_inc.txt'
    em_out_txt_path='./xout_'+out_prefix+'_em_out.txt'
    np.savetxt(tr_txt_path, tr_arr,
               header=str(len(tr_arr)), delimiter='  ', comments='')
    np.savetxt(inc_txt_path, inc_arr,
               header=str(len(tr_arr)), delimiter='  ', comments='')
    np.savetxt(em_out_txt_path, em_out_arr,
               header=str(len(tr_arr)), delimiter='  ', comments='')

    #converting to xspec tables
    xstar_to_table_pop(tr_txt_path,outpath=tr_txt_path.replace('.txt','_xspec.fits'),xlum=xlum,dist=dist)
    xstar_to_table_pop(inc_txt_path,outpath=inc_txt_path.replace('.txt','_xspec.fits'),xlum=xlum,dist=dist)
    xstar_to_table_pop(em_out_txt_path,outpath=em_out_txt_path.replace('.txt','_xspec.fits'),xlum=xlum,dist=dist)


def xstar_to_table_pop(path, outpath='xspecmod.fits', modname=None,xlum='auto',dist=10):
    '''
    Converts an xstar output spectra to a model table usable by xspec (atable)
    adapted from pop's script

    if modname is None, uses the file name in outpath as model name

    xlum (optional):
        bolometric luminosity of the output, for renormalization. IN UNITS OF 1e38 erg/s
        if set to auto, uses directly the file's luminosity sum
    dist (optional):
        distance of the source, for renormalization. In units of kpc

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
    flux = (xlum *1e38 if xlum!='auto' else tmp[:, 1] * 1e38) / (4. * np.pi * (dist * 1e3 * 3.086e18) ** 2.)  # erg/s/erg/cm2
    fluxsub = flux[np.where(enemin > 1e-4)]
    enemaxsub = np.roll(eneminsub, -1)
    enemeansub = 0.5 * (enemaxsub + eneminsub)
    l = len(eneminsub)

    spec = fluxsub / (enemeansub * 1.6e-9)  # ph/s/erg/cm2

    #this doesn't work
    spec = spec * (enemaxsub - eneminsub) * 1.6e-9  # ph/s/cm2

    #note: this works, although I don't know why
    # spec = spec *(enemaxsub - eneminsub)**(1/2) * 1.6e-9

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
    heaproc.sendline('rm ' './'+ outprefix + '_tmp.txt')
    heaproc.sendline('rm ' './'+ outprefix + '.txt')
    time.sleep(1)
    heaproc.sendline('exit')

    if outdir not in ['','./']:
        os.system('mv ' + os.path.join(currdir, filedir, outfile) + ' ' + outpath)

def xstar_to_table(path,outpath='auto',modname='auto',xlum='auto',dist=10):

    '''
    TO BE FINISHED IF NEED TO USE
    Version using the conversion formula from https://heasarc.gsfc.nasa.gov/xstar/docs/xstarmanual.pdf
    page 38

    modname: internal xspec name of the model
            if set to auto, uses the file (without the extension) name of outpath
    '''

    currdir = os.getcwd()

    outdir = '' if '/' not in outpath else outpath[:outpath.rfind('/')]

    if outpath=='auto':
        outpath_use=path[:path.rfind('.')]+'_xspec.fits'
    else:
        outpath_use=outpath

    outfile = outpath_use[outpath_use.rfind('/') + 1:]

    if modname=='auto':

        modname_use = outfile[:outfile.rfind('.')]

    else:
        modname_use=modname

    file = path[path.rfind('/') + 1:]

    filedir = path.replace(file, '')

    xstar_e,xstar_f=np.loadtxt(path,skiprows=1).T

    if xlum=='auto':
        xlum_use=1
    else:
        xlum_use=xlum
    #note: we don't interpolate to avoid adding supplementary uncertainties

    #computing the energy step from the average of the first 1000 bins to average the floating error fluctuations
    #note that this assumes a constant grid up to the 1000th point.

    assert xstar_e[1000]<4e5,'xstar grid significantly beyond the start of the coarse grid'

    delta_e=(xstar_e[1000]/xstar_e[0])**(1/1000)-1

    xspec_f=1e38/(4*np.pi*dist*(3.086e21)**2)*xstar_f*1e38/xlum_use*delta_e

    mask_maingrid=(xstar_e>1e-1) & (xstar_e<4e5)

    #note: could be better to interpolate to the middle of the bin
    arr_ftflx2tab=np.array([xstar_e[mask_maingrid][:-1],xstar_e[mask_maingrid][1:],xspec_f[mask_maingrid][:-1]]).T

    input_name='ftflx2tab_input_temp_'+str(time.time()).replace('.','')+'.txt'
    np.savetxt(input_name,arr_ftflx2tab)

    # spawning a bash process to produce the table from the txt modified SED
    heaproc = pexpect.spawn('/bin/bash', encoding='utf-8')
    heaproc.logfile = sys.stdout
    heaproc.sendline('heainit')
    heaproc.sendline('cd ' + currdir)

    # here, outfile is the name of the model
    heaproc.sendline('ftflx2tab ' + input_name +' '+ modname_use + ' ' + outfile + ' clobber = yes redshift=True')

    time.sleep(10)

    heaproc.sendline('exit')

    os.remove(input_name)

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

    Plot('emo')

    #note: it seems that plot eeuf will only show the data so will not show the rest of the model
    # if extended to higher/lower energies
    # Plot('eeuf')

    x_arr = Plot.x()

    y_arr = Plot.model()

    save_arr = np.array([x_arr, y_arr]).T

    np.savetxt(path, save_arr, header='nu(Hz) Lnu(erg/s/Hz)', delimiter=' ')

def gridmodel(gridpath, interp='log'):
    '''
    Fetches all final spectra in the gridpath arborescence,
    and creates an xspec model .

    Note: need non physical parameter sampling due to uneven solution parameter space.
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

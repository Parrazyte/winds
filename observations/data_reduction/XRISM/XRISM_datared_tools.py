
# general imports
import os, sys
import subprocess
import pexpect
import argparse
import logging
import glob
import threading
import time
from tee import StdoutTee, StderrTee
import shutil
import warnings
import re

#for no_op_context
import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

# note: might need to install opencv-python-headless to avoid dependencies issues with mpl

# import matplotlib.cm as cm
from matplotlib.collections import LineCollection

#using agg because qtagg still generates backends with plt.ioff()
# mpl.use('agg')
# plt.ioff()

# astro imports
from astropy.time import Time,TimeDelta
from astropy.io import fits
from astroquery.simbad import Simbad
from mpdaf.obj import sexa2deg,deg2sexa,Image
from mpdaf.obj import WCS as mpdaf_WCS
from astropy.wcs import WCS as astroWCS

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib.patches import Polygon


# from mpdaf.obj import deg2sexa

# image processing imports:
# mask to polygon conversion
from imantics import Mask

# point of inaccessibility
from polylabel import polylabel

# alphashape
from alphashape import alphashape

# polygon filling to mask
from rasterio.features import rasterize

# shape merging
from scipy.ndimage import binary_dilation

from general_tools import file_edit, ravel_ragged,MinorSymLogLocator,interval_extract,str_orbit,source_catal,plot_lc


'''
The energy scale reports for each pixel are available publicly for all observations at 
https://heasarc.gsfc.nasa.gov/FTP/xrism/postlaunch/software/v001/

the pdf summaries have URLs like
https://heasarc.gsfc.nasa.gov/FTP/xrism/postlaunch/gainreports/2/201011010_resolve_energy_scale_report.pdf

One of the important things to check is if the evolution of the temperature of the calibration PIXEL
(Fig. 7) matches that of the pixels (Fig. 6)
Also a good way to identify weirdly behaving pixels (ex. hot pixel 27)
see e.g. https://xrism-c2c.atlassian.net/wiki/spaces/XRISMPV/pages/174293062/4U1630-472
'''



@contextlib.contextmanager
def no_op_context():
    yield

#function to remove (most) control chars
# to keep the same loops with or without the Tees in several functions
def _remove_control_chars(message):
    ansi_escape =re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)

def kev_to_PI(e_val,instru='resolve'):

    '''
    conversion from examples in
    https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/quickstart/xrism_quick_start_guide_v2p3_240918a.pdf

    '''
    if instru=='resolve':
        return round(e_val*2000)
    elif instru=='xtend':
        return round(e_val*1000/6)

def set_var(spawn,heasoft_init_alias='heainit',caldb_init_alias='heainit'):
    '''
    Sets starting environment variables for data analysis
    '''
    if heasoft_init_alias is not None:
        spawn.sendline(heasoft_init_alias)

    if caldb_init_alias is not None:
        spawn.sendline(caldb_init_alias)

#bruteforcing is easier
coord_pixel_conv_dict={
    '0':[4,3],
    '1': [6, 3],
    '2': [5, 3],
    '3': [6,2],
    '4': [5, 2],
    '5': [6, 1],
    '6': [5,1],
    '7': [4,2],
    '8': [4,1],
    '9': [1, 3],
    '10': [2, 3],
    '11': [1,2],
    '12': [1,1],
    '13': [2,2],
    '14': [2,1],
    '15': [3,2],
    '16': [3,1],
    '17': [3, 3],
    '18': [3,4],
    '19': [1,4],
    '20': [2,4],
    '21': [1,5],
    '22': [2,5],
    '23': [1,6],
    '24': [2, 6],
    '25': [3,5],
    '26': [3,6],
    '27': [6,4],
    '28': [5,4],
    '29': [6,5],
    '30': [6,6],
    '31': [5,5],
    '32': [5,6],
    '33': [4,5],
    '34': [4,6],
    '35': [4,4]}

#note that here the x and y axis are inverted and transposed compared to the dictionnary to match the images in the POG
coord_pixel_conv_arr=np.array([[23, 24, 26, 34, 32, 30],
       [21, 22, 25, 33, 31, 29],
       [19, 20, 18, 35, 28, 27],
       [9, 10, 17, 0, 2, 1],
       [11, 13, 15, 7, 4, 3],
       [12, 14, 16, 8, 6, 5]])


def compa_SNR_shiftpix(shift='h1',pixel_excl=[14,13,11,9,19,21,23]):

    list_pix=np.arange(36)

    list_pix_include=[elem for elem in list_pix if elem not in pixel_excl]

def rsl_npixel_to_coord(number):
    '''
    returns the resolve image/detector coordinates of a given pixel number.
     See e.g. https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html
    '''

    return coord_pixel_conv_dict[str(number)]

def renorm_pix_backscale(file,mult_factor,suffix=''):
    '''

    if the branching ratio of the source is X and the background from another obs is Y
    to rescale we multiply by Y/X
    ex: source has 0.5, bkg has 1. The bkg is twice too high, so the backscale needs to be x2 to lower its effect
    (see https://heasarc.gsfc.nasa.gov/docs/asca/abc_backscal.html)
    '''

    file_cp=shutil.copy(file,file.replace(file[file.rfind('.'):],'_backscale_renorm'+file[file.rfind('.')]))


def repro_dir(directory='auto',repro_suffix='repro',overwrite=True,
               heasoft_init_alias='heainit',caldb_init_alias='caldbinit',parallel=False):

    '''

    Reprocesses directory using xapipeline then copies the initial directory to a directory_repro version,
    where all the files created by xapipeline replace the initial files.
    Logs the value of heasoft and caldb used for comparisons, since currently the headers are not properly updated

    Important to do if the data was downloaded before a new heasoft/CALDB version got released

    requires CALDB environent variable to be set

    '''

    def copy_checker(spawn,init_path_spawn,end_path_spawn,init_path,path_spawn,logfile_prefix):

        # number of files to copy
        n_files_copy = len(glob.glob(os.path.join(init_path, "**"),recursive=True))

        logfile_name=logfile_prefix+'_cp_list'+str(time.time()).replace('.','p')+'.log'

        # copying the individual xrism obs subdirs
        spawn.sendline('cp -rv '+init_path_spawn+' '+end_path_spawn+' >'+logfile_name)

        # checking the log file
        copy_ok = False
        while not copy_ok:
            time.sleep(0.5)
            with open(os.path.join(path_spawn,logfile_name)) as f:
                n_lines = len(f.readlines())
            if n_lines == n_files_copy:
                copy_ok = 1
            else:
                print('waiting for copy...')
                pass
        # reasonable waiting time to make sure files can be copied
        time.sleep(0.5)


    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc,heasoft_init_alias,caldb_init_alias)

    if directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    bashproc.sendline('cd '+os.getcwd())

    time_str=str(time.time()).replace('.','p')
    if os.path.isfile(directory_use + '/repro_dir'+time_str+'.log'):
        os.system('rm ' + directory_use + '/repro_dir'+time_str+'.log')

    with (no_op_context() if parallel else StdoutTee(directory_use + '/repro_dir'+time_str+'.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory_use + '/repro_dir'+time_str+'.log', buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        # testing whether there is already a reprocessed directory
        repro_dir=str(directory_use)+'_'+str(repro_suffix)

        repro_version_match=False
        # getting the version of heasoft and the calibration used for the reprocessing
        bashproc.sendline('fversion >'+os.path.join(directory_use,'heasoft_version.txt'))

        bashproc.sendline('echo $CALDB >'+os.path.join(directory_use,'caldb_path.txt'))


        time.sleep(1.)
        with open(os.path.join(directory_use,'heasoft_version.txt')) as f:
            heasoft_ver=f.readlines()[0].replace('\n','').split('_')[1]

        with open(os.path.join(directory_use,'caldb_path.txt')) as f:
            caldb_path=f.readlines()[0].replace('\n','')

        os.system('rm '+os.path.join(directory_use,'heasoft_version.txt'))

        # heasoft_ver = os.environ['HEADAS']

        caldb_ver = 'gen' + os.path.realpath(os.path.join(caldb_path,
                                                          'data/xrism/gen/caldb.indx')).split('indx')[-1]
        caldb_ver += '_xtd' + os.path.realpath(os.path.join(caldb_path,
                                                            'data/xrism/xtend/caldb.indx')).split('indx')[-1]
        caldb_ver += '_res' + os.path.realpath(os.path.join(caldb_path,
                                                            'data/xrism/resolve/caldb.indx')).split('indx')[-1]

        if os.path.isdir(repro_dir):

            #and comparing to existing logs
            repro_ver_file=os.path.join(repro_dir, 'repro_ver.txt')
            if os.path.isfile(repro_ver_file):

                with open(repro_ver_file) as f:
                    repro_lines=f.readlines()

                repro_version_match=repro_lines[0].replace('\n','')==heasoft_ver \
                                    and repro_lines[1].replace('\n','')==caldb_ver
        else:
             #reading a file from one of the files in the initial directory
            with fits.open(glob.glob(os.path.join(directory_use,'auxil','**'))[0]) as hdul:

                if 'SOFTVER' not in hdul[1].header:
                    init_version_heasoft='notfound'
                else:
                    init_version_heasoft=hdul[1].header['SOFTVER'].split('_')[2]

                if 'CALDBVER' not in hdul[1].header:
                    init_version_caldb='notfound'
                else:
                    init_version_caldb=hdul[1].header['CALDBVER']

            repro_version_match=init_version_heasoft==heasoft_ver and init_version_caldb==caldb_ver

        if not repro_version_match or overwrite:

            bashproc.sendline('mkdir -p '+repro_dir)
            bashproc.sendline('cd '+repro_dir)

            bashproc.sendline('echo valid')
            bashproc.expect('valid')
            bashproc.sendline('rm -rf resolve repro_logs log  xtend  auxil')
            time.sleep(5)

            bashproc.sendline('xapipeline'+
                          ' indir= ../'+str(directory_use)+
                          ' outdir= ./'+
                          ' steminputs=xa'+str(directory_use)+
                          ' STEMOUTPUTS=DEFAULT'+
                          ' instrument=ALL'+
                          ' verify_input=no'
                          ' entry_stage=1'+
                          ' exit_stage=2'+
                          ' clobber=YES')

            #note that the exit code is a bit annoying to parse so we're using sevral lines to isolate it
            bashproc.expect('Total warnings/errors',timeout=None)
            bashproc.expect('Running xapipeline')
            bashproc.expect('Exit with no errors')
            time.sleep(1)
            bashproc.sendline('echo valid')

            bashproc.expect('valid')

            #moving the logfiles to a log folder
            bashproc.sendline('mkdir -p repro_logs')
            bashproc.sendline('mv xapipeline**.log repro_logs')

            copy_checker(bashproc,'../'+directory_use+'/resolve','./',
                         init_path=directory_use+'/resolve',path_spawn=repro_dir,
                         logfile_prefix='resolve')

            copy_checker(bashproc,'../'+directory_use+'/xtend','./',
                         init_path=directory_use+'/xtend',path_spawn=repro_dir,
                         logfile_prefix='xtend')

            copy_checker(bashproc,'../'+directory_use+'/log','./',
                         init_path=directory_use+'/log',path_spawn=repro_dir,
                         logfile_prefix='log')

            copy_checker(bashproc,'../'+directory_use+'/auxil','./',
                         init_path=directory_use+'/auxil',path_spawn=repro_dir,
                         logfile_prefix='auxil')

            #replacing each file for which xaapipeline created a new version
            #listing the files in one of the subdirs. We need 2 / to ensure we're in one
            file_list=\
                [elem for elem in np.unique(glob.glob(os.path.join(repro_dir,'**'),
                                                      recursive=True))
                 if 'repro_logs' not in elem and elem.count('/')>1]

            file_move_list=[elem.split('/')[-1] for elem in
                                    glob.glob(os.path.join(repro_dir,'**'))
                                              if 'xa' in elem]

            #reading all the lines to reset the buffer
            try:

                bashproc.timeout=1
                bashproc.readlines()
            except:
                bashproc.timeout=30



            for elem_file_repro in file_move_list:
                file_arbo_pos=[elem for elem in file_list if elem_file_repro in elem]
                if len(file_arbo_pos)>1:
                    breakpoint()
                elif len(file_arbo_pos)==0:
                    continue

                #cutting the first dir since that will be the repro obsid
                file_arbo_dir='/'.join(file_arbo_pos[0].split('/')[:-1])
                file_arbo_spawn_dir='/'.join(file_arbo_pos[0].split('/')[1:-1])
                file_arbo_spawn_path='/'.join(file_arbo_pos[0].split('/')[1:])

                bashproc.sendline('rm '+file_arbo_spawn_path)
                bashproc.sendline('mv '+elem_file_repro+' '+file_arbo_spawn_dir)
                while not os.path.isfile(os.path.join(file_arbo_dir,elem_file_repro)):
                    time.sleep(0.5)
                    print('Wating for file move...')
                    if not os.path.isfile(os.path.join(file_arbo_dir,elem_file_repro)):
                        time.sleep(2.5)
                        if not os.path.isfile(os.path.join(file_arbo_dir, elem_file_repro)):
                            breakpoint()
                            pass
                time.sleep(0.5)

                bashproc.sendline('echo valid')
                bashproc.expect('valid')

                print('Replaced file '+elem_file_repro)

            #logging the version of heasoft and the calibration used for the reprocessing
            heasoft_ver=os.environ['HEADAS']

            caldb_ver='gen_'+os.path.realpath(os.path.join(caldb_path,
                                                           'data/xrism/gen/caldb.indx')).split('indx')[-1]
            caldb_ver+='_res_'+os.path.realpath(os.path.join(caldb_path,
                                                           'data/xrism/resolve/caldb.indx')).split('indx')[-1]
            caldb_ver+='_xtd_'+os.path.realpath(os.path.join(caldb_path,
                                                           'data/xrism/xtend/caldb.indx')).split('indx')[-1]

            with open(os.path.join(repro_dir,'repro_ver.txt'),'w+') as f:
                f.write(heasoft_ver+'\n')
                f.write(caldb_ver+'\n')

        else:
            print('overwrite not selected and repro versions matching. Skipping xapipeline...')

        bashproc.sendline('exit')

def init_anal(directory='auto_repro',anal_dir_suffix='',resolve_filters='open',xtd_config='all',gz=False,
              repro_suffix='repro'):
    '''

    Copies main event lists and other useful files to an analysis directory

    resolve_filters:
        all or X+Y+Z
            names:
                undefined
                open
                Al
                ND
                Be
                Fe55

    xtd_config:
        all or X+Y+Z
            names:
                all_FW
                1-2_1/8
                1-2_FW_burst
                1-2_1/8_burst
                3-4_FW

    see https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/quickstart/xrism_quick_start_guide_v2p3_240918a.pdf
    for terminology of resolve and xtd file names
    '''

    resolve_filters_disp=np.array(['undefined','open','Al','ND','Be','Fe55'])
    resolve_filters_namecodes=np.array(['p0px0000','p0px1000','p0px2000','p0px3000','p0px4000','p0px5000'])

    xtd_config_disp=np.array(["all_FW","1-2_1/8","1-2_FW_burst","1-2_1/8_burst","3-4_FW"])
    xtd_config_namecodes=np.array(['p0300','p0311','p0312','p0313','p0320'])

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir=os.path.join(directory_use,'analysis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else ''))

    os.system('mkdir -p '+anal_dir)

    dirfiles=glob.glob(os.path.join(directory_use,'**'),recursive=True)

    resolve_evts=[elem for elem in dirfiles if elem.endswith('cl.evt'+('.gz' if gz else '')) and len(elem.split('/'))>2 and
                  elem.split('/')[-3]=='resolve']
    print('Found '+str(len(resolve_evts))+' resolve event files')
    print(resolve_evts)

    if resolve_filters!='all':
        print('Restricting to selected resolve filters...\n')
        resolve_evts_use=[]
        resolve_filters_list=resolve_filters.split('+')
        for elem_filter in resolve_filters_list:
            elem_namecode=resolve_filters_namecodes[resolve_filters_disp==elem_filter][0]
            resolve_evts_use+=[elem for elem in resolve_evts if elem_namecode in elem.split('/')[-1]]
    else:
        resolve_evts_use=resolve_evts


    xtd_evts=[elem for elem in dirfiles if elem.endswith('cl.evt'+('.gz' if gz else '')) and len(elem.split('/'))>2 and
                  elem.split('/')[-3]=='xtend']
    print('Found '+str(len(xtd_evts))+' xtend event files')
    print(xtd_evts)

    if xtd_config!='all':
        print('Restricting to selected xtend configurations...\n')
        xtd_evts_use=[]
        xtd_config_list=xtd_config.split('+')
        for elem_config in xtd_config_list:
            elem_namecode=xtd_config_namecodes[xtd_config_disp==elem_config][0]
            xtd_evts_use+=[elem for elem in xtd_evts if elem_namecode in elem.split('/')[-1]]
    else:
        xtd_evts_use=xtd_evts

    #copying the files to the analysis directory
    for elem in resolve_evts_use+xtd_evts_use:
        os.system('cp '+elem+' '+anal_dir)

    #untarring the files that were just copied in anal_dir
    os.system('gunzip '+os.path.join(anal_dir,'**'))

def resolve_RTS(directory='auto_repro',anal_dir_suffix='',heasoft_init_alias='heainit',caldb_init_alias='caldbinit',
                parallel=False,repro_suffix='repro'):

    '''
    Filters all available resolve event files in the analysis subdirectory of a directory for Rise-Time Screening
    Following https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/quickstart/xrism_quick_start_guide_v2p3_240918a.pdf
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc,heasoft_init_alias,caldb_init_alias)

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir=os.path.join(directory_use,'analysis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else ''))

    resolve_files=[elem for elem in glob.glob(os.path.join(anal_dir,'**')) if 'rsl_' in elem.split('/')[-1] and
                   elem.endswith('_cl.evt')]

    bashproc.sendline('cd '+os.path.join(os.getcwd(),anal_dir))

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str=str(time.time()).replace('.','p')

    log_path=os.path.join(log_dir,'resolve_RTS'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')+time_str+'.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        for indiv_file in resolve_files:

            # print('ftcopy infile="'+indiv_file.split('/')[-1]+'[EVENTS]'+
            #                 '[(PI>=600) && (((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))'+
            #                 '&&ITYPE<4)||(ITYPE==4))&&STATUS[4]==b0]" outfile='+
            #                  indiv_file.split('/')[-1].replace('_cl.evt','_cl_RTS.evt')+
            #                 ' copyall=yes clobber=yes history=yes')

            bashproc.sendline('ftcopy infile="'+indiv_file.split('/')[-1]+'[EVENTS]'+
                            '[(PI>=600) && (((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))'+
                            '&&ITYPE<4)||(ITYPE==4))&&STATUS[4]==b0]" outfile='+
                             indiv_file.split('/')[-1].replace('_cl.evt','_cl_RTS.evt')+
                            ' copyall=yes clobber=yes history=yes')

            while not os.path.isfile(os.path.join(anal_dir,indiv_file.split('/')[-1].replace('_cl.evt','_cl_RTS.evt'))):
                print('waiting for copy...')
                time.sleep(1)

            time.sleep(5)


            bashproc.sendline('echo valid')

            bashproc.expect('valid')


    bashproc.sendline('exit')

def compute_avg_BR_pixlist(branch_file='auto',pixel_str='branch_filter',branch_txt_file='auto',
                           excl_pixel_list=None,band=True):

    '''
    Computes the time-averaged, pixel averaged branching ratio for a combination of pixels

    the input can be either directly from a list of excluded pixel (excl_pixel_list) or a pixel_str in the style of
    the other commands of this script

    branch_file:
        if set to 'auto', searches for the first file in the current directories ending with '_brVpxcnt.fits'

    pixel_str:
        pixel filtering list for xrism
            if set to branch_filter, excludes the pixels listed in the **branch_filter.txt file of
            the observation, made by resolve_BR
            can be taken automatically if branch_txt_file is set to auto, otherwise manually
            otherwise, manual input:

            example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
            also accepts pixels to exclude, such as '-(10:14,28,32)'

    '''

    if branch_file=='auto':
        branch_file_use=glob.glob('**/**_brVpxcnt.fits',recursive=True)[0]
        print('Using branch file '+branch_file_use)
    else:
        branch_file_use=branch_file

    if type(pixel_str)==str and pixel_str.startswith('branch_filter'):
        # reading the branch filter file

        if branch_txt_file == 'auto':
            branch_txt_use = glob.glob('**_branch_filter.txt', recursive=True)[0]
            print('Using branch file ' + branch_txt_use)
        else:
            branch_txt_use = branch_txt_file

        with open(branch_txt_use) as branch_f:
            branch_lines = branch_f.readlines()
        branch_filter_line = [elem for elem in branch_lines if not elem.startswith('#')][0]
        # reformatting the string
        valid_pix_list = rsl_pixel_manip('-(' + branch_filter_line[1:-2] + ')',
                                            mode='pix_list',remove_cal_pxl_resolve=True)
    elif pixel_str is not None:
        valid_pix_list=rsl_pixel_manip(pixel_str,mode='pix_list',remove_cal_pxl_resolve=True)
    elif excl_pixel_list is not None:
        valid_pix_list=[elem for elem in np.arange(36) if elem not in excl_pixel_list and elem!=12]
    else:
        valid_pix_list=[elem for elem in np.arange(36) if elem!=12]

    with fits.open(branch_file_use) as hdul:
        branch_data=hdul[4 if band else 1].data

    branch_grade_names=['BRANCHHP','BRANCHMP','BRANCHMS','BRANCHLP','BRANCHLS']

    branch_avg_list=np.array([(branch_data['RATETOT'][valid_pix_list]/np.sum(branch_data['RATETOT'][valid_pix_list])\
                *branch_data[elem_name][valid_pix_list]).sum() for elem_name in branch_grade_names])


    return branch_avg_list

def plot_BR_band_compar(branch_file_num,branch_file_denom,excl_pixel=[],count_rate='sum',
                        save_path=None,mode='full'):

    '''
    computes the ratio between the BR of different files or bands. For now made for bands.
    count_rate (for the X axis positions):
        -sum: sums the total band count rates of both branch_files
        -num: uses only the numerator count rate
        -denom: uses only the denominator count rate
    '''

    with fits.open(branch_file_num) as branch_fits:
        branch_num_data_lsreal_efull = branch_fits[1].data
        branch_num_simu_lsreal_efull = branch_fits[3].data
        branch_num_data_lsreal_eband = branch_fits[4].data
        e_band_num=[elem for elem in branch_fits[1].header['HISTORY'] if 'eband' in elem][0].split(' ')[-1]

    # removing pixel 12 to avoid problems later with ordering
    branch_num_data_lsreal_efull = branch_num_data_lsreal_efull[[elem for elem in range(36) if elem != 12]]
    branch_num_simu_lsreal_efull = branch_num_simu_lsreal_efull[[elem for elem in range(36) if elem != 12]]
    branch_num_data_lsreal_eband= branch_num_data_lsreal_eband[[elem for elem in range(36) if elem != 12]]

    with fits.open(branch_file_denom) as branch_fits:
        branch_denom_data_lsreal_efull = branch_fits[1].data
        branch_denom_simu_lsreal_efull = branch_fits[3].data
        branch_denom_data_lsreal_eband = branch_fits[4].data
        e_band_denom=[elem for elem in branch_fits[1].header['HISTORY'] if 'eband' in elem][0].split(' ')[-1]

    # removing pixel 12 to avoid problems later with ordering
    branch_denom_data_lsreal_efull = branch_denom_data_lsreal_efull[[elem for elem in range(36) if elem != 12]]
    branch_denom_simu_lsreal_efull = branch_denom_simu_lsreal_efull[[elem for elem in range(36) if elem != 12]]
    branch_denom_data_lsreal_eband= branch_denom_data_lsreal_eband[[elem for elem in range(36) if elem != 12]]


    # making a plot with the information for the eband
    fig_branch_band_eband, ax_brand_band_eband = plt.subplots(figsize=(16, 10))


    ax_brand_band_eband.set_yscale('log' if mode=='full' else 'linear')
    ax_brand_band_eband.set_xscale('log')
    x_axis_str='[' + e_band_num + '] '+'+ ['+e_band_denom+']' if count_rate=='sum' \
                else '[' + e_band_num + ']' if count_rate=='num' else'['+e_band_denom+']' if count_rate=='denom' else ''

    ax_brand_band_eband.set_xlabel(r'Pixel count rate in the '+x_axis_str+' keV band (s$^{-1}$)')
    ax_brand_band_eband.set_ylabel('Ratio of observed branching ratios between the [ '
                                        +e_band_num+'] and ['+e_band_denom+'] keV bands')
    ax_brand_band_eband.set_title('Ratio of observed branching ratios between the  '
                                  +e_band_num+' and '+e_band_denom+' keV bands')

    if count_rate=='sum':
        branch_x_band_sum=branch_num_data_lsreal_eband['RATETOT']+branch_denom_data_lsreal_eband['RATETOT']
    elif count_rate=='num':
        branch_x_band_sum=branch_num_data_lsreal_eband['RATETOT']
    elif count_rate=='denom':
        branch_x_band_sum=branch_denom_data_lsreal_eband['RATETOT']

    plt.axhline(1,0,1,color='black')
    # showcasing the branch_band ratios

    if mode=='full':
        plt.plot(branch_x_band_sum,
                 branch_num_data_lsreal_eband['BRANCHHP']/branch_denom_data_lsreal_eband['BRANCHHP'],
                 ls='', marker='d',
                 color='green', label='Hp')

        plt.plot(branch_x_band_sum,
                 branch_num_data_lsreal_eband['BRANCHMP']/branch_denom_data_lsreal_eband['BRANCHMP'],
                 ls='', marker='d',
                 color='blue', label='Mp')
        plt.plot(branch_x_band_sum,
                 branch_num_data_lsreal_eband['BRANCHMS']/branch_denom_data_lsreal_eband['BRANCHMS'],
                 ls='', marker='d',
                 color='cyan', label='Ms')
        plt.plot(branch_x_band_sum,
                 branch_num_data_lsreal_eband['BRANCHLP']/branch_denom_data_lsreal_eband['BRANCHLP'],
                 ls='', marker='d',
                 color='orange', label='Lp')
        plt.plot(branch_x_band_sum,
                 branch_num_data_lsreal_eband['BRANCHLS']/branch_denom_data_lsreal_eband['BRANCHLS'],
                 ls='', marker='d',
                 color='red', label='Ls')
    elif mode=='Hp+Mp':

        plt.plot(branch_x_band_sum,
                 (branch_num_data_lsreal_eband['BRANCHHP']+branch_num_data_lsreal_eband['BRANCHMP'])/
                 (branch_denom_data_lsreal_eband['BRANCHHP']+branch_denom_data_lsreal_eband['BRANCHMP']),
                 ls='', marker='d',
                 color='teal', label='Hp+Mp')


    # creating a secondary axis to show the pixel positions
    ax_up = ax_brand_band_eband.secondary_xaxis('top')

    # removing the ticks
    ax_up.set_xticks([], minor=True)
    ax_up.set_xticks([], minor=True)

    # replacing them with the pixel ids
    pixel_order = branch_x_band_sum.argsort()

    # the random addition is here to avoid ticks overlapping if the count rate is the same
    # (which makes the labels disappear
    ax_up.set_xticks(branch_x_band_sum[pixel_order] + np.random.rand(35) * 1e-5)

    # and putting labels on differnet lines to avoid cluttering
    arr_pxl_names = branch_num_data_lsreal_eband['PIXEL'][pixel_order].astype(str)
    arr_pxl_shifted = [elem + ('\n' * (i % 5)) for i, elem in enumerate(arr_pxl_names)]

    ax_up.set_xticklabels(arr_pxl_shifted)
    # adjusting the color of the excluded pixel labels
    color_arr_label = np.where([int(elem) in excl_pixel for elem in arr_pxl_names], 'red', 'black')
    for xtick, color in zip(ax_up.get_xticklabels(), color_arr_label):
        xtick.set_color(color)

    ax_up.set_xlabel('Pixel number' + ('(excluded in red)' if len(excl_pixel) > 0 else ''))

    # adding vertical lines
    for pix_number, pix_rate in zip(branch_num_data_lsreal_eband['PIXEL'], branch_x_band_sum):
        plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                    zorder=-1)

    plt.legend(loc='best')
    plt.tight_layout()

    if save_path is not None:
        if save_path=='auto':
            plt.savefig('branch_ratio_save.pdf')
        else:
            plt.savefig(save_path)
        plt.close()

def plot_BR(branch_file, save_paths=None, excl_pixel=[],task='rslbratios',plot_hp_sim_curve_band=True):
    '''
    Wrapper around the branching ratios plotting function

    Different versions for the new and old tasks
    if the old versions with rslbranch:
    1 plot with the actual values

    in the new version with rslbratios, makes 3 plots:
    1 in the energy band with only the actual values
    2 in the full band: one with the values and the predictions, one with the ratios between the two
    if save_path is not None, will disable the gui and save before closing the plot

    excl_pixel can be an interable of excluded pixels to highlight
    '''


    if task=='rslbranch':
        with fits.open(branch_file) as branch_fits:
            branch_data = branch_fits[2].data

        # removing pixel 12 to avoid problems later with ordering
        branch_data = branch_data[[elem for elem in range(36) if elem != 12]]

        if save_paths is not None:
            plt.ioff()

        # making a plot with the information
        fig_branch, ax_branch = plt.subplots(figsize=(16, 10))
        ax_branch.set_yscale('log')
        ax_branch.set_xscale('log')
        ax_branch.set_ylim(1e-3, 1.1)
        ax_branch.set_xlabel(r'Pixel count rate in the full (0-30 keV) band (s$^{-1}$)')
        ax_branch.set_ylabel('Pixel branching ratios in the full (0-30 keV) band (s$^{-1}$)')

        # showcasing the branching ratios
        plt.plot(branch_data['RATETOT'],
                 branch_data['BRANCHHP'], ls='', marker='d',
                 color='green', label='Hp')
        plt.plot(branch_data['RATETOT'],
                 branch_data['BRANCHMP'], ls='', marker='d',
                 color='blue', label='Mp')
        plt.plot(branch_data['RATETOT'],
                 branch_data['BRANCHMS'], ls='', marker='d',
                 color='cyan', label='Ms')
        plt.plot(branch_data['RATETOT'],
                 branch_data['BRANCHLP'], ls='', marker='d',
                 color='orange', label='Lp')
        plt.plot(branch_data['RATETOT'],
                 branch_data['BRANCHLS'], ls='', marker='d',
                 color='red', label='Ls')

        # creating a secondary axis to show the pixel positions
        ax_up = ax_branch.secondary_xaxis('top')

        # removing the ticks
        ax_up.set_xticks([], minor=True)
        ax_up.set_xticks([], minor=True)

        # replacing them with the pixel ids
        pixel_order = branch_data['RATETOT'].argsort()

        # the random addition is here to avoid ticks overlapping if the count rate is the same
        # (which makes the labels disappear
        ax_up.set_xticks(branch_data['RATETOT'][pixel_order] + np.random.rand(35) * 1e-5)

        # and putting labels on differnet lines to avoid cluttering
        arr_pxl_names = branch_data['PIXEL'][pixel_order].astype(str)
        arr_pxl_shifted = [elem + ('\n' * (i % 5)) for i, elem in enumerate(arr_pxl_names)]

        ax_up.set_xticklabels(arr_pxl_shifted)
        # adjusting the color of the excluded pixel labels
        color_arr_label = np.where([int(elem) in excl_pixel for elem in arr_pxl_names], 'red', 'black')
        for xtick, color in zip(ax_up.get_xticklabels(), color_arr_label):
            xtick.set_color(color)

        ax_up.set_xlabel('Pixel number' +('(excluded in red)' if len(excl_pixel)>0 else ''))

        # adding vertical lines
        for pix_number, pix_rate in zip(branch_data['PIXEL'], branch_data['RATETOT']):
            plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                        zorder=-1)

        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_paths is not None:
            plt.savefig(save_paths[0])
            plt.close()
            plt.ion()

    elif task=='rslbratios':

        if save_paths is not None:
            plt.ioff()

        with fits.open(branch_file) as branch_fits:
            branch_data_lsreal_efull = branch_fits[1].data
            branch_simu_lsreal_efull = branch_fits[3].data
            branch_data_lsreal_eband = branch_fits[4].data

            branch_band=[elem for elem in branch_fits[4].header['history'] if 'eband' in elem][0].split(' = ')[1]


        # removing pixel 12 to avoid problems later with ordering
        branch_data_lsreal_efull = branch_data_lsreal_efull[[elem for elem in range(36) if elem != 12]]
        branch_simu_lsreal_efull = branch_simu_lsreal_efull[[elem for elem in range(36) if elem != 12]]
        branch_data_lsreal_eband= branch_data_lsreal_eband[[elem for elem in range(36) if elem != 12]]

        # making a plot with the information for the eband
        fig_branch_band_eband, ax_brand_band_eband = plt.subplots(figsize=(16, 10))
        ax_brand_band_eband.set_yscale('log')
        ax_brand_band_eband.set_xscale('log')
        ax_brand_band_eband.set_xlabel(r'Pixel count rate in the '+branch_band+' keV band (s$^{-1}$)')
        ax_brand_band_eband.set_ylabel('Pixel branching ratios in the '+branch_band+' keV band (s$^{-1}$)')
        ax_brand_band_eband.set_title('Observed and theoretical branching ratios in the '+branch_band+' keV band')

        # showcasing the branch_band ratios
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHHP'],
                 ls='', marker='d',
                 color='green', label='Hp')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHMP'],
                 ls='', marker='d',
                 color='blue', label='Mp')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHMS'],
                 ls='', marker='d',
                 color='cyan', label='Ms')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHLP'],
                 ls='', marker='d',
                 color='orange', label='Lp')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHLS'],
                 ls='', marker='d',
                 color='red', label='Ls')

        # creating a secondary axis to show the pixel positions
        ax_up = ax_brand_band_eband.secondary_xaxis('top')

        # removing the ticks
        ax_up.set_xticks([], minor=True)
        ax_up.set_xticks([], minor=True)

        # replacing them with the pixel ids
        pixel_order = branch_data_lsreal_eband['RATETOT'].argsort()

        # the random addition is here to avoid ticks overlapping if the count rate is the same
        # (which makes the labels disappear
        ax_up.set_xticks(branch_data_lsreal_eband['RATETOT'][pixel_order] + np.random.rand(35) * 1e-5)

        # and putting labels on differnet lines to avoid cluttering
        arr_pxl_names = branch_data_lsreal_eband['PIXEL'][pixel_order].astype(str)
        arr_pxl_shifted = [elem + ('\n' * (i % 5)) for i, elem in enumerate(arr_pxl_names)]

        ax_up.set_xticklabels(arr_pxl_shifted)
        # adjusting the color of the excluded pixel labels
        color_arr_label = np.where([int(elem) in excl_pixel for elem in arr_pxl_names], 'red', 'black')
        for xtick, color in zip(ax_up.get_xticklabels(), color_arr_label):
            xtick.set_color(color)

        ax_up.set_xlabel('Pixel number' +('(excluded in red)' if len(excl_pixel)>0 else ''))

        # adding vertical lines
        for pix_number, pix_rate in zip(branch_data_lsreal_eband['PIXEL'], branch_data_lsreal_eband['RATETOT']):
            plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                        zorder=-1)

        #adding the theoretical values since they shouldn't be energy dependant


        rate_pred_order=branch_simu_lsreal_efull['RATETOT'].argsort()

        if plot_hp_sim_curve_band:
            plt.plot(branch_data_lsreal_eband['RATETOT'][rate_pred_order],
                     branch_simu_lsreal_efull['BRANCHHP'][rate_pred_order],
                     ls='-', marker='',
                     color='green', label='')
            plt.plot([],[],
                 ls='-', marker='',
                 color='black', label='theoretical values '+('\n w.r.t. 0-30 keV \n pixel count rate'))

        # plt.axhline(1,0,1,ls='-', marker='',
        #              color='green', label='')

        plt.plot(branch_data_lsreal_eband['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHMP'][rate_pred_order],
                 ls='-', marker='',
                 color='blue', label='')

        plt.plot(branch_data_lsreal_eband['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHMS'][rate_pred_order],
                 ls='-', marker='',
                 color='cyan', label='')

        plt.plot(branch_data_lsreal_eband['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHLP'][rate_pred_order],
                 ls='-', marker='',
                 color='orange', label='')

        plt.plot(branch_data_lsreal_eband['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHLS'][rate_pred_order],
                 ls='-', marker='',
                 color='red', label='')

        #ax_brand_band_eband.set_ylim(ax_brand_band_eband.get_ylim()[0], 1.1)
        ax_brand_band_eband.set_ylim(5e-4, 1.1)

        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_paths is not None:
            plt.savefig(save_paths[0])
            plt.close()


        # making a plot with the information and the prediction for the full band
        fig_branch_band_full, ax_branch_band_full = plt.subplots(figsize=(16, 10))
        ax_branch_band_full.set_yscale('log')
        ax_branch_band_full.set_xscale('log')
        ax_branch_band_full.set_xlabel(r'Pixel count rate in the full (0-30 keV) band (s$^{-1}$)')
        ax_branch_band_full.set_ylabel('Pixel branching ratios in the full (0-30 keV) band (s$^{-1}$)')
        ax_branch_band_full.set_title('Observed and theoretical branching ratios in the full (0-30 keV) XRISM band')

        # showcasing the branch_banding ratios
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHHP'],
                 ls='', marker='d',
                 color='green', label='Hp')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHMP'],
                 ls='', marker='d',
                 color='blue', label='Mp')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHMS'],
                 ls='', marker='d',
                 color='cyan', label='Ms')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHLP'],
                 ls='', marker='d',
                 color='orange', label='Lp')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHLS'],
                 ls='', marker='d',
                 color='red', label='Ls')

        rate_pred_order=branch_simu_lsreal_efull['RATETOT'].argsort()

        #simulated values

        plt.plot(branch_simu_lsreal_efull['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHHP'][rate_pred_order],
                 ls='-', marker='',
                 color='green', label='')


        plt.plot(branch_simu_lsreal_efull['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHMP'][rate_pred_order],
                 ls='-', marker='',
                 color='blue', label='')

        plt.plot(branch_simu_lsreal_efull['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHMS'][rate_pred_order],
                 ls='-', marker='',
                 color='cyan', label='')

        plt.plot(branch_simu_lsreal_efull['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHLP'][rate_pred_order],
                 ls='-', marker='',
                 color='orange', label='')

        plt.plot(branch_simu_lsreal_efull['RATETOT'][rate_pred_order],
                 branch_simu_lsreal_efull['BRANCHLS'][rate_pred_order],
                 ls='-', marker='',
                 color='red', label='')

        # creating a secondary axis to show the pixel positions
        ax_up_band_full = ax_branch_band_full.secondary_xaxis('top')

        ax_up_band_full.set_xlabel('Pixel number' +('(excluded in red)' if len(excl_pixel)>0 else ''))


        # removing the ticks
        ax_up_band_full.set_xticks([], minor=True)
        ax_up_band_full.set_xticks([], minor=True)

        # replacing them with the pixel ids
        pixel_order = branch_data_lsreal_efull['RATETOT'].argsort()

        # the random addition is here to avoid ticks overlapping if the count rate is the same
        # (which makes the labels disappear
        ax_up_band_full.set_xticks(branch_data_lsreal_efull['RATETOT'][pixel_order] + np.random.rand(35) * 1e-5)

        # and putting labels on differnet lines to avoid cluttering
        arr_pxl_names = branch_data_lsreal_efull['PIXEL'][pixel_order].astype(str)
        arr_pxl_shifted = [elem + ('\n' * (i % 5)) for i, elem in enumerate(arr_pxl_names)]

        ax_up_band_full.set_xticklabels(arr_pxl_shifted)
        # adjusting the color of the excluded pixel labels
        color_arr_label = np.where([int(elem) in excl_pixel for elem in arr_pxl_names], 'red', 'black')
        for xtick, color in zip(ax_up_band_full.get_xticklabels(), color_arr_label):
            xtick.set_color(color)


        # adding vertical lines
        for pix_number, pix_rate in zip(branch_data_lsreal_efull['PIXEL'], branch_data_lsreal_efull['RATETOT']):
            plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                        zorder=-1)

        #upper-only ylim
        #ax_branch_band_full.set_ylim(ax_branch_band_full.get_ylim()[0], 1.1)
        ax_branch_band_full.set_ylim(5e-4, 1.1)

        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_paths is not None:
            plt.savefig(save_paths[1])
            plt.close()

        # making a plot with the data to prediction ratio for the full band
        fig_branch_full_ratio, ax_branch_full_ratio = plt.subplots(figsize=(16, 10))
        ax_branch_full_ratio.set_yscale('log')
        ax_branch_full_ratio.set_xscale('log')
        ax_branch_full_ratio.set_xlabel(r'Pixel count rate in the full (0-30 keV) band (s$^{-1}$)')
        ax_branch_full_ratio.set_ylabel('Ratio between observed and theoretical pixel branching ratios'
                                        ' in the full (0-30 keV) XRISM band')

        ax_branch_full_ratio.set_title('Ratio between observed and theoretical branching ratios '
                                      ' in the full (0-30 keV) XRISM band')

        # showcasing the branch_banding ratios
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHHP']/branch_simu_lsreal_efull['BRANCHHP'],
                 ls='', marker='d',
                 color='green', label='Hp')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHMP']/branch_simu_lsreal_efull['BRANCHMP'],
                 ls='', marker='d',
                 color='blue', label='Mp')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHMS']/branch_simu_lsreal_efull['BRANCHMS'],
                 ls='', marker='d',
                 color='cyan', label='Ms')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHLP']/branch_simu_lsreal_efull['BRANCHLP'],
                 ls='', marker='d',
                 color='orange', label='Lp')
        plt.plot(branch_data_lsreal_efull['RATETOT'],
                 branch_data_lsreal_efull['BRANCHLS']/branch_simu_lsreal_efull['BRANCHLS'],
                 ls='', marker='d',
                 color='red', label='Ls')

        # creating a secondary axis to show the pixel positions
        ax_up_full_ratio = ax_branch_full_ratio.secondary_xaxis('top')
        ax_up_full_ratio.set_xlabel('Pixel number' +('(excluded in red)' if len(excl_pixel)>0 else ''))

        # removing the ticks
        ax_up_full_ratio.set_xticks([], minor=True)
        ax_up_full_ratio.set_xticks([], minor=True)

        # replacing them with the pixel ids
        pixel_order = branch_data_lsreal_efull['RATETOT'].argsort()

        # the random addition is here to avoid ticks overlapping if the count rate is the same
        # (which makes the labels disappear
        ax_up_full_ratio.set_xticks(branch_data_lsreal_efull['RATETOT'][pixel_order] + np.random.rand(35) * 1e-5)

        # and putting labels on differnet lines to avoid cluttering
        arr_pxl_names = branch_data_lsreal_efull['PIXEL'][pixel_order].astype(str)
        arr_pxl_shifted = [elem + ('\n' * (i % 5)) for i, elem in enumerate(arr_pxl_names)]

        ax_up_full_ratio.set_xticklabels(arr_pxl_shifted)
        # adjusting the color of the excluded pixel labels
        color_arr_label = np.where([int(elem) in excl_pixel for elem in arr_pxl_names], 'red', 'black')
        for xtick, color in zip(ax_up_full_ratio.get_xticklabels(), color_arr_label):
            xtick.set_color(color)


        # adding vertical lines
        for pix_number, pix_rate in zip(branch_data_lsreal_efull['PIXEL'],
                                        branch_data_lsreal_efull['RATETOT']):
            plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                        zorder=-1)

        plt.legend(loc='upper right')
        plt.tight_layout()

        if save_paths is not None:
            plt.savefig(save_paths[2])
            plt.close()

        # making a plot with the data to prediction ratio for the restricted band
        fig_branch_band_ratio, ax_branch_band_ratio = plt.subplots(figsize=(16, 10))
        ax_branch_band_ratio.set_yscale('log')
        ax_branch_band_ratio.set_xscale('log')

        ax_branch_band_ratio.set_xlabel(r'Pixel count rate in the '+branch_band+' keV band (s$^{-1}$)')
        ax_branch_band_ratio.set_ylabel('Ratio between observed and theoretical pixel branching ratios in the '
                                        +branch_band+' keV band (s$^{-1}$)')
        ax_branch_band_ratio.set_title('Ratio between Observed and theoretical branching ratios in the '
                                        +branch_band+' keV band')

        # showcasing the branch_banding ratios
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHHP']/branch_simu_lsreal_efull['BRANCHHP'],
                 ls='', marker='d',
                 color='green', label='Hp')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHMP']/branch_simu_lsreal_efull['BRANCHMP'],
                 ls='', marker='d',
                 color='blue', label='Mp')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHMS']/branch_simu_lsreal_efull['BRANCHMS'],
                 ls='', marker='d',
                 color='cyan', label='Ms')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHLP']/branch_simu_lsreal_efull['BRANCHLP'],
                 ls='', marker='d',
                 color='orange', label='Lp')
        plt.plot(branch_data_lsreal_eband['RATETOT'],
                 branch_data_lsreal_eband['BRANCHLS']/branch_simu_lsreal_efull['BRANCHLS'],
                 ls='', marker='d',
                 color='red', label='Ls')

        # creating a secondary axis to show the pixel positions
        ax_up_band_ratio = ax_branch_band_ratio.secondary_xaxis('top')
        ax_up_band_ratio.set_xlabel('Pixel number' +('(excluded in red)' if len(excl_pixel)>0 else ''))

        # removing the ticks
        ax_up_band_ratio.set_xticks([], minor=True)
        ax_up_band_ratio.set_xticks([], minor=True)

        # replacing them with the pixel ids
        pixel_order = branch_data_lsreal_eband['RATETOT'].argsort()

        # the random addition is here to avoid ticks overlapping if the count rate is the same
        # (which makes the labels disappear
        ax_up_band_ratio.set_xticks(branch_data_lsreal_eband['RATETOT'][pixel_order] + np.random.rand(35) * 1e-5)

        # and putting labels on differnet lines to avoid cluttering
        arr_pxl_names = branch_data_lsreal_eband['PIXEL'][pixel_order].astype(str)
        arr_pxl_shifted = [elem + ('\n' * (i % 5)) for i, elem in enumerate(arr_pxl_names)]

        ax_up_band_ratio.set_xticklabels(arr_pxl_shifted)
        # adjusting the color of the excluded pixel labels
        color_arr_label = np.where([int(elem) in excl_pixel for elem in arr_pxl_names], 'red', 'black')
        for xtick, color in zip(ax_up_band_ratio.get_xticklabels(), color_arr_label):
            xtick.set_color(color)


        # adding vertical lines
        for pix_number, pix_rate in zip(branch_data_lsreal_eband['PIXEL'],
                                        branch_data_lsreal_eband['RATETOT']):
            plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                        zorder=-1)

        plt.legend(loc='upper right')
        plt.tight_layout()

        if save_paths is not None:
            plt.savefig(save_paths[3])
            plt.close()
            plt.ion()


def resolve_BR(directory='auto_repro', anal_dir_suffix='',
               use_raw_evt_rsl=False,
               task='rslbratios',
               lightcurves=False,
               emin=2,emax=12,
               remove_cal_pxl_resolve=False,
               pixel_filter_rule='ratio_LS_6+remove_27',
               heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
               parallel=False,repro_suffix='repro',plot_hp_sim_curve_band=True):
    '''
    Computes a file and plot with the branching ratio information for each resolve event file

    Note that this needs to be update to plot the right branching ratios

    Two commands exist:
        -rslbranch (see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslbranch.html)
        -rslbratios (see yourheasoftpath/help/rslbratios.html)

    -lghtcurve can be switched on or off for the rslbratios command, depending on whether the user wants
    to compute lighcurves of the evolution of each event grade averaged over all pixels

    pixel_filter_rule:
        if not None, uses the rules to determine pixels to EXCLUDE
        creates a txt file with the information of the pixels removed and the remaining pixels
        (which works as an input for later commands)
        different rules can be combined with +. use "and" for subrules

        example:clip_LS_0.02andcompa_LS>MS+remove_27


    '''

    emin_str=str(emin).replace('.','p')
    emax_str=str(emax).replace('.','p')

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str=str(time.time()).replace('.','p')

    log_path=os.path.join(log_dir,'resolve_BR'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')+time_str+'.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        for indiv_file in resolve_files:

            if task=='rslbranch':
                bashproc.sendline('rslbranch infile=' + indiv_file.split('/')[-1] +
                                  ' outfile=' + indiv_file.split('/')[-1].replace('.evt', '_branch.fits') +
                                  ' filetype=real' +
                                  ' pixfrac=NONE' +
                                  ' pixmask=NONE')

                bashproc.expect('Finished RSLBRANCH', timeout=60)

                time.sleep(1)

                # adding a filtering if it is requested
                if pixel_filter_rule is not None:

                    with fits.open(indiv_file.replace('.evt', '_branch.fits')) as branch_fits:
                        branch_data = branch_fits[2].data

                    mask_exclude = np.repeat(False, 36)

                    for elem_rule in pixel_filter_rule.split('+'):

                        mask_subexclude = np.repeat(True, 36)

                        for subelem_rule in elem_rule.split('_and'):

                            if subelem_rule.split('_')[0] == 'clip':
                                mask_subexclude = (mask_subexclude) & \
                                                  (branch_data['BRANCH' + subelem_rule.split('_')[1]] > float(
                                                      subelem_rule.split('_')[2]))

                            if subelem_rule.split('_')[0] == 'compa':
                                if '<' in subelem_rule.split('_')[1]:
                                    mask_subexclude = (mask_subexclude) & \
                                                      (branch_data['BRANCH' + subelem_rule.split('_')[1].split('<')[0]]
                                                       < branch_data['BRANCH' + subelem_rule.split('_')[1].split('<')[1]])
                                elif '>' in subelem_rule.split('_')[1]:
                                    mask_subexclude = (mask_subexclude) & \
                                                      (branch_data['BRANCH' + subelem_rule.split('_')[1].split('>')[0]]
                                                       > branch_data['BRANCH' + subelem_rule.split('_')[1].split('>')[1]])

                            if subelem_rule.split('_')[0] == 'remove':
                                submask_remove = [str(elem) in subelem_rule.split('_')[1].split(',')
                                                  for elem in range(36)]
                                mask_subexclude = (mask_subexclude) & (submask_remove)

                            if subelem_rule.split('_')[0] == 'only':
                                submask_only = [str(elem) not in subelem_rule.split('_')[1].split(',')
                                                  for elem in range(36)]
                                mask_subexclude = (mask_subexclude) &  (submask_only)

                        mask_exclude = (mask_exclude) | mask_subexclude

            elif task=='rslbratios':
                bashproc.sendline('mkdir -p branch')
                bashproc.sendline('cd branch')

                bashproc.sendline('rslbratios'
                                  +' infile=../'+indiv_file.split('/')[-1]
                                  +' filetype="cl"'
                                  +' outroot=branch'
                                  +' eband='+str(emin)+'-'+str(emax)
                                  +' lcurve='+str(lightcurves)+
                                  ' clobber=yes')

                bashproc.expect('Finished RSLBRATIOS',timeout=None)

                time.sleep(1)

                mask_exclude = np.repeat(False, 36)

                with fits.open('/'.join(indiv_file.split('/')[:-1]) +
                               '/branch/branch_' + emin_str + 'to' + emax_str + 'keV_brVpxcnt.fits') as branch_fits:
                    branch_data_real_full = branch_fits[1].data
                    branch_simu_real_full = branch_fits[3].data
                    branch_data_real_band = branch_fits[4].data

                    # for the pixel txt file
                    branch_data = branch_fits[1].data

                # adding a filtering if it is requested
                if pixel_filter_rule is not None:

                    for elem_rule in pixel_filter_rule.split('+'):

                        mask_subexclude = np.repeat(True, 36)

                        for subelem_rule in elem_rule.split('_and'):

                            branch_data_use=branch_data_real_full

                            if subelem_rule.split('_')[-1]=='band':

                                #using the band data to compute the filtering
                                branch_data_use=branch_data_real_band
                                assert 'ratio' not in subelem_rule,\
                                    'No simulated data available for band branching ratios'
                                print("chou")

                            if subelem_rule.split('_')[0] == 'clip':
                                mask_subexclude = (mask_subexclude) & \
                                                  (branch_data_use['BRANCH' + subelem_rule.split('_')[1]] > float(
                                                      subelem_rule.split('_')[2]))

                            if subelem_rule.split('_')[0] == 'ratio':
                                mask_subexclude = (mask_subexclude) & \
                                  (branch_data_real_full['BRANCH' + subelem_rule.split('_')[1]]\
                                   /branch_simu_real_full['BRANCH' + subelem_rule.split('_')[1]] \
                                   > float(subelem_rule.split('_')[2]))

                            if subelem_rule.split('_')[0] == 'compa':
                                if '<' in subelem_rule.split('_')[1]:
                                    mask_subexclude = (mask_subexclude) & \
                                                      (branch_data_use['BRANCH' + subelem_rule.split('_')[1].split('<')[0]]
                                                       < branch_data_use['BRANCH' + subelem_rule.split('_')[1].split('<')[1]])
                                elif '>' in subelem_rule.split('_')[1]:
                                    mask_subexclude = (mask_subexclude) & \
                                                      (branch_data_use['BRANCH' + subelem_rule.split('_')[1].split('>')[0]]
                                                       > branch_data_use['BRANCH' + subelem_rule.split('_')[1].split('>')[1]])

                            if subelem_rule.split('_')[0] == 'remove':
                                submask_remove = [str(elem) in subelem_rule.split('_')[1].split(',')
                                                  for elem in range(36)]
                                mask_subexclude = (mask_subexclude) & (submask_remove)

                            if subelem_rule.split('_')[0] == 'only':
                                submask_only = [str(elem) not in subelem_rule.split('_')[1].split(',')
                                                  for elem in range(36)]
                                mask_subexclude = (mask_subexclude) &  (submask_only)

                        mask_exclude = (mask_exclude) | mask_subexclude

            bashproc.sendline('echo valid')

            # this is normally fast so their should be no need for a long time-out
            bashproc.expect('valid')

            pixel_exclude_list = np.arange(36)[mask_exclude]
            # saving the output of the filtering in a file

            pixel_filter_file = indiv_file.replace('.evt', '_branch_filter.txt')

            pixel_avg_branch=compute_avg_BR_pixlist(
                branch_file='/'.join(indiv_file.split('/')[:-1])+'/branch/branch_'+emin_str+'to'+emax_str+'keV_brVpxcnt.fits',
                pixel_str=None if pixel_filter_rule is None else '-(' + str(pixel_exclude_list.tolist())[1:-2] + ')')

            with open(pixel_filter_file, 'w+') as f:
                f.write('#Filter applied: ' + str(pixel_filter_rule) + '\n')
                f.write('#Combined count rate of excluded pixels: %.3e' % (
                    branch_data['RATETOT'][mask_exclude].sum()) + '\n')
                f.write('#Combined count rate of all pixels: %.3e' % (branch_data['RATETOT'].sum()) + '\n')
                f.write('#Remaining count rate proportion: %.3e'
                        % (1 - branch_data['RATETOT'][mask_exclude].sum() / branch_data['RATETOT'].sum()) + '\n')

                f.write('#list of excluded pixels:\n')
                f.write(str(pixel_exclude_list.tolist()) + '\n')
                f.write('#pixel-averaged '+emin_str+'-'+emax_str+' keV branching ratios with current pixel selection for HP-MP-MS-LP-LS:\n')
                f.write(str(pixel_avg_branch.tolist()) + '\n')


            if task=='rslbranch':
                plot_BR(indiv_file.replace('.evt', '_branch.fits'),
                    save_paths=[indiv_file.replace('.evt', '_branch_screen.png')],
                    excl_pixel=pixel_exclude_list)

            elif task=='rslbratios':

                plot_BR('/'.join(indiv_file.split('/')[:-1])+'/branch/branch_'+emin_str+'to'+emax_str+'keV_brVpxcnt.fits',
                    save_paths=[indiv_file.replace('.evt', '_branch_screen_'+emin_str.replace('.','p')
                                                   +'-'+emax_str.replace('.','p')+'.png'),
                                indiv_file.replace('.evt', '_branch_screen_full.png'),
                                indiv_file.replace('.evt', '_branch_screen_full_ratio.png'),
                                indiv_file.replace('.evt', '_branch_screen_'+emin_str.replace('.','p')
                                                   +'-'+emax_str.replace('.','p')+'_ratio.png')],
                    excl_pixel=pixel_exclude_list,plot_hp_sim_curve_band=plot_hp_sim_curve_band)

                plot_BR('/'.join(indiv_file.split('/')[:-1])+'/branch/branch_'+emin_str+'to'+emax_str+'keV_brVpxcnt.fits',
                    save_paths=[indiv_file.replace('.evt', '_branch_screen_'+ emin_str.replace('.', 'p')
                                                   + '-' + emax_str.replace('.', 'p')+'.pdf'),
                                indiv_file.replace('.evt', '_branch_screen_full.pdf'),
                                indiv_file.replace('.evt', '_branch_screen_full_ratio.pdf'),
                                indiv_file.replace('.evt', '_branch_screen_' + emin_str.replace('.', 'p')
                                                   + '-' + emax_str.replace('.', 'p') + '_ratio.pdf')],
                    excl_pixel=pixel_exclude_list,plot_hp_sim_curve_band=plot_hp_sim_curve_band)

            bashproc.sendline('exit')

def resolve_counts_BR_Eevol(file,pixels='all',ebin=10,emin=0.,emax=12.):
    '''
    Largely adapted from Kai Matsunaga's code

        ebin in eV

        emin and emax in keV
    '''

    grade_colors={'Hp':'green','Mp':'blue','Ms':'cyan','Lp':'orange', 'Ls':'red'}

    fits_evt= fits.open(file)

    raw_energy=fits_evt[1].data['UPI']
    raw_energy_mask=(fits_evt[1].data['UPI'] >=emin*1000) & (fits_evt[1].data['UPI'] <emax*1000)

    if pixels!='all':
        raw_pixels_mask=fits_evt[1].data['PIXEL'] in rsl_pixel_manip(pixels,mode='pi_list')
    else:
        raw_pixels_mask=True

    raw_mask=raw_energy_mask & raw_pixels_mask

    evt_energy_bin = np.round(fits_evt[1].data['UPI'][raw_energy_mask] / ebin) * ebin
    evt_grade = fits_evt[1].data['TYPE'][raw_energy_mask]

    # pandas の DataFrame に変換
    df = pd.DataFrame({
        'energy': evt_energy_bin,
        'grade_type': evt_grade
    })

    # energyごとに grade_type の出現数を集計
    pivot = df.groupby(['energy', 'grade_type']).size().unstack(fill_value=0)

    fig_counts,ax_counts = plt.subplots(figsize=(12,6))

    # ax1 = fig.add_axes([0.05, 0.1, 0.42, 0.8])

    # 各 grade_type ごとに scatter プロット
    for grade in pivot.columns:
        ax_counts.scatter(pivot.index/1000, pivot[grade], label=grade, s=1 if grade=='Hp' else 2,color=grade_colors[grade])
        ax_counts.plot(pivot.index/1000, pivot[grade], label='', zorder=2,color=grade_colors[grade],lw=1,
                       alpha=0.3)

    # ax1.tick_params(axis="x", which='major', direction='in', length=5, width=1)
    # ax1.tick_params(axis="y", which='major', direction='in', length=5, width=1)
    # ax1.tick_params(axis="x", which='minor', direction='in', length=2, width=1)
    # ax1.tick_params(axis="y", which='minor', direction='in', length=2, width=1)

    # ax1.xaxis.set_ticks_position('both')
    # ax1.yaxis.set_ticks_position('both')

    ax_counts.set_xlabel('Energy (keV)')
    ax_counts.set_ylabel(str(ebin)+' eV binned Count rate')
    # ax1.set_xlim(1000, 12000)
    ax_counts.set_yscale('log')
    ax_counts.set_ylim(1, 1e4)
    ax_counts.set_xlim(emin,emax)

    ax_counts.set_xticks(np.arange(0, 12,0.2), minor=True)

    ax_counts.legend()
    ax_counts.legend(markerscale=5)

    # ax1.grid(True)
    # ax1.set_title('Spectra')

    ratio = pivot.div(pivot.sum(axis=1), axis=0)

    fig_branch,ax_branch = plt.subplots(figsize=(12,6))

    for grade in ratio.columns:
        ax_branch.scatter(ratio.index/1000, ratio[grade], label=grade, s=1 if grade=='Hp' else 2, zorder=2,color=grade_colors[grade])
        ax_branch.plot(ratio.index/1000, ratio[grade], label='', zorder=2,color=grade_colors[grade],lw=1,
                       alpha=0.3)

    ax_branch.set_xlabel('Energy (keV)')
    ax_branch.set_ylabel(str(ebin)+' eV binned Event Grade Ratio')
    # ax1.set_xlim(1000, 12000)
    ax_branch.set_yscale('log')
    ax_branch.set_xlim(emin,emax)
    ax_branch.set_xticks(np.arange(0, 12,0.2), minor=True)

    ax_branch.set_ylim(ax_branch.get_ylim()[0],1.1)
    ax_branch.axhline(1,color='grey',lw=1)
    ax_branch.legend()
    ax_branch.legend(markerscale=5)

    fig_counts.tight_layout()
    fig_branch.tight_layout()

    file_extension=file.split('/')[-1][file.split('/')[-1].rfind('.'):]
    fig_counts.savefig(file.replace(file_extension,'_grade_evol_'+
                                                  '_emin_'+str(emin).replace('.','p')+
                                                  '_emax_' + str(emax).replace('.', 'p') +
                                                  '_ebin_' + str(ebin).replace('.', 'p') +
                                                    '.pdf'))

    fig_branch.savefig(file.replace(file_extension,'_grade_ratio_evol'+
                                                  '_emin_'+str(emin).replace('.','p')+
                                                  '_emax_' + str(emax).replace('.', 'p') +
                                                  '_ebin_' + str(ebin).replace('.', 'p') +
                                                    '.pdf'))

    breakpoint()


def mpdaf_load_img(sky_img_path):
    # loading the IMG file with mpdaf
    with fits.open(sky_img_path) as hdul:
        try:

            img_data = hdul[0].data
            src_mpdaf_WCS = mpdaf_WCS(hdul[0].header)
            src_astro_WCS = astroWCS(hdul[0].header)

            img_obj_whole = Image(data=img_data, wcs=src_mpdaf_WCS)
        except:

            img_data = hdul[1].data
            src_mpdaf_WCS = mpdaf_WCS(hdul[1].header)
            src_astro_WCS = astroWCS(hdul[1].header)

            img_obj_whole = Image(data=img_data, wcs=src_mpdaf_WCS)

    return img_obj_whole,src_mpdaf_WCS,src_astro_WCS

def target_deg(source_name,target_coords=None):
    '''
    source_name will be passed into simbad

    target_coords can be a [str,str] (will then be converted from sexa to deg)
                         a [float,float] (in which case it remains unchanged)
    '''

    if target_coords is None:
        print('\nAuto mode.')
        print('\nAutomatic search of the directory names in Simbad.')

        obj_auto = Simbad.query_object(source_name)[0]

        # if the output is already in degree units
        if type(obj_auto['dec']) == np.float64 and type(obj_auto['ra']) == np.float64:
            obj_deg = [str(obj_auto['ra']), str(obj_auto['dec'])]
        else:
            # careful the output after the first grade is in dec,ra not ra,dec
            obj_deg = sexa2deg([float(obj_auto['dec']).replace(' ', ':'), float(obj_auto['ra']).replace(' ', ':')])[
                      ::-1]
            obj_deg = [str(obj_deg[0]), str(obj_deg[1])]
    else:
        if type(target_coords[0]) == str:
            obj_deg = sexa2deg([target_coords[1].replace(' ', ':'), target_coords[0].replace(' ', ':')])[::-1]
        else:
            obj_deg = target_coords

    return obj_deg

def mpdaf_plot_img(sky_img_path,rad_crop=[200,200],crop_coords=None,
                   target_name_list=[None],target_coords_list='target',
                   target_sizes_pix=[10],target_colors=['red'],
                   target_names=['auto'],target_names_offset=[1.1],
                   title='',save=False):

    '''
    Plot an mpdaf image in sky coordinates, with a given cropping if requested,
        and additional regions highlighting sources if requested.
        The crop is made centered on the position of the first source if crop_coords is None, otherwise to
        the coordinates given

        source_names/target_coords/target_sizes_pix: iterables of the same len
        source_names/target_coords are used in target_deg to get the position of the sources
        target_sizes_pix gives the source region

        target_cords_list:

            'target' to follow the targets

            note: by default target_sizes_pix is in pixlels, so will have to be converted accordingly
            reminder: xtend r_arsec=1.768*r_pixel
    '''

    img_obj_whole,src_mpdaf_WCS,src_astro_WCS=mpdaf_load_img(sky_img_path)

    if target_name_list[0] is None and (target_coords_list[0] is None or target_coords_list=='target'):
        obj_deg_list=[]
    else:
        obj_deg_list=np.array([target_deg(elem_source_name,elem_target_coords) for
                  (elem_source_name,elem_target_coords) in zip(target_name_list,target_coords_list)],dtype=float)

    if crop_coords is None:
        crop_center=obj_deg_list[0]
    elif type(crop_coords)==str and crop_coords=='auto':
        with fits.open(sky_img_path) as hdul:
            crop_center=[hdul[0].header['RA_PNT'],hdul[0].header['DEC_PNT']]
    else:
        crop_center=sexa2deg(crop_coords[::-1])[::-1]
    if len(obj_deg_list)!=0 or crop_coords!=None:
        try:
            imgcrop_src = img_obj_whole.copy().subimage(center=crop_center[::-1], size=rad_crop)
        except:
            print('\nCropping region entirely out of the image. Field of view issue....')
            return '\nCropping region entirely out of the image. Field of view issue....'
    else:
        imgcrop_src=img_obj_whole
    '''
    showing the bounds of the desired region
    no easy way to do it currently so we draw a circle manually after converting
    the angular coordinates to physical coordinates with the WCS
    note that the crop re-sizes the axes so we need to offset the position of the circle afterwards
    the "0,0" ends up at the bottom left of the graph 
    so we need to remove half a rad_crop in y and add half a rad_crop in x
    '''

    if save:
        plt.ioff()
    # plotting and saving imgcrop
    fig_catal_crop, ax_catal_crop = plt.subplots(1, 1, subplot_kw={'projection': src_astro_WCS},
                                                 figsize=(12, 10))

    circle_rad_pos=[]
    target_circles=[]

    coord_crop_eff=[imgcrop_src.wcs.naxis1,imgcrop_src.wcs.naxis2][::-1]
    coord_start_eff=imgcrop_src.wcs.get_start()[::-1]

    #the axis increment is actually modified afte resizing and doesnt' match the intiial values, so
    #we update it aswell
    axis_increment_eff=imgcrop_src.get_axis_increments()*(np.array(coord_crop_eff))/np.array(rad_crop)*2
    for i_target,(elem_ra_deg,elem_dec_deg) in enumerate(obj_deg_list):

        #kinda fucked up but I think it works and my brain can't figure out the simple formula
        #note that manual checking with ds9 showed that there was a 0.5 pixel offset on the y axis
        #so we correct that manually
        #note that the sub-pixel accuracy is so-so
        # circle_rad_pos+= [[ \
        #     ((rad_crop[0]/2) if i_target==0 else (rad_crop[0]  + (elem_ra_deg - obj_deg_list[0][0])*3600/2))/ \
        #                                                     (imgcrop_src.get_axis_increments()[0] * 3600)\
        #     %(rad_crop[0]/(-imgcrop_src.get_axis_increments()[1] * 3600)),
        #     (rad_crop[1] / 2 + (elem_dec_deg - obj_deg_list[0][1])*3600) / \
        #                                                     (-imgcrop_src.get_axis_increments()[1] * 3600)\
        #     %(rad_crop[1]/(-imgcrop_src.get_axis_increments()[1] * 3600))-0.5]]

        # circle_rad_pos+= [[ \
        #     ((rad_crop[0] / 2) if elem_ra_deg==crop_center[0] else (rad_crop[0]  + (elem_ra_deg - crop_center[0])*3600/2))/ \
        #                                                     (imgcrop_src.get_axis_increments()[0] * 3600)\
        #     %(rad_crop[0]/(imgcrop_src.get_axis_increments()[0] * 3600)),
        #     (rad_crop[1] / 2 + (elem_dec_deg - crop_center[1])*3600) / \
        #                                                     (-imgcrop_src.get_axis_increments()[1] * 3600)\
        #     %(rad_crop[1]/(-imgcrop_src.get_axis_increments()[1] * 3600))-0.5]]

        circle_rad_pos+= [[(coord_start_eff[0]-elem_ra_deg)/axis_increment_eff[0]-0.5,
            (rad_crop[1] / 2 + (elem_dec_deg - crop_center[1])*3600) / \
                                                            (-imgcrop_src.get_axis_increments()[1] * 3600)\
            %(rad_crop[1]/(-imgcrop_src.get_axis_increments()[1] * 3600))-0.5]]

        # circle_rad_pos+= [[ \
        #     rad_crop_eff[0]/2+((crop_center[0]-elem_ra_deg)*3600)/ \
        #                                                     (imgcrop_src.get_axis_increments()[0] * 3600)\
        #     %(rad_crop[0]/(imgcrop_src.get_axis_increments()[0] * 3600)),
        #     (rad_crop[1] / 2 + (elem_dec_deg - crop_center[1])*3600) / \
        #                                                     (-imgcrop_src.get_axis_increments()[1] * 3600)\
        #     %(rad_crop[1]/(-imgcrop_src.get_axis_increments()[1] * 3600))-0.5]]
        # if i_target==1:
        #
        #     middle=rad_crop_eff[0]/(imgcrop_src.get_axis_increments()[0] * 3600)
        #
        #     fac=1/(imgcrop_src.get_axis_increments()[0] * 3600)
        #     cc=crop_center[0]*3600
        #     raa=elem_ra_deg*3600
        #
        #     fv=(cc-raa)*fac
        #     breakpoint()
        #
        # breakpoint()

        target_circles+= [plt.Circle([circle_rad_pos[-1][0], circle_rad_pos[-1][1]], target_sizes_pix[i_target],
                            color=target_colors[i_target], zorder=1000, fill=False)]
        if target_names[i_target]!='':
            if target_names[i_target]=='auto':
                curr_target_name=target_name_list[i_target]
            else:
                curr_target_name=target_names[i_target]

            ax_catal_crop.text(circle_rad_pos[-1][0],circle_rad_pos[-1][1]+target_sizes_pix[i_target]*target_names_offset[i_target],
                               curr_target_name,
                     color=target_colors[i_target],horizontalalignment='center')

    # testing if the resulting image is empty
    if len(imgcrop_src.data.nonzero()[0]) == 0:
        print('Cropped image empty.')
        return 'Cropped image empty.'

    if title!='':
        ax_catal_crop.set_title(title)
    catal_plot = imgcrop_src.plot(cmap='plasma', scale='log')
    plt.colorbar(catal_plot, location='bottom', fraction=0.046, pad=0.04)
    for elem_circle in target_circles:
        ax_catal_crop.add_patch(elem_circle)

    #updating the colorbar
    ax_cb=ax_catal_crop.get_figure().get_children()[-1]
    ax_cb.set_xticks(np.logspace(0, np.log10(imgcrop_src.data.max()), 8))

    #adding a top xaxis label since the cmap is hiding the bottom one
    ax_catal_crop.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=True,
        labeltop=True,
        direction='out')

    if save:
        plt.savefig(sky_img_path[:sky_img_path.rfind('.')]+'_mpdaf_img.pdf')
        plt.close()
        plt.ion()

def xtend_SFP(directory='auto_repro',filtering='flat_top',
              #for flat_top
              base_config='313',threshold_mult=1.1,rad_psf=15,
              source_name='auto',
              target_coords=None,
              target_only=False,use_file_target=True,use_file_target_coords=False,
              logprob2=None,bgd_level=None,cellsize=None,n_division=None,grade='ALL',
              logprob1=10,
              anal_dir_suffix='',parallel=False,sudo_screen=False,
              heasoft_init_alias='heainit',caldb_init_alias='caldbinit',repro_suffix='repro'):
    '''

    #COULD BE IMPROVED BY USING A SPECIFIC ENERGY BAND

    MAXIJ1744 approximate coords: ['17:45:40.45','-29:00:46.6']
    MAXIJ1744 Chandra Atel Coords ['17:45:40.476', '-29:00:46.10']

    V4641Sgr Coordinates
    Run searchflixpix to clean the xtend data of flickering pixels.
    See https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/quickstart/xrism_quick_start_guide_v2p3_240918a.pdf

    filtering determine the filtering method
    -flat_top: computes a sky coordinates image of either each event file
                (OR just the configuration provided in 'base' if base is not set to 'indiv')
                With MPDAF, computes the brightest pixel in a 15" region around the source theoretical position

    for flat_top:
        base_config:
            -determines whether the flat_top threshold is determined for a single configuration and used everywhere
            or individually
            if set to one of the xtend configuration substrings
            (see https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/xrism_abc.pdf p.11), will use this one
            exclusively. if set to indiv, will be run independently for each configuration

        threshold: multiplicator for the value of the brightest pixel in the psf before the threshold is applied
                    T=threshold*max_psf
        rad_psf: radius where the brightest pixel is searched in arcseconds.
                 centered around the source position.
                 For lisibility the image is cropped at 4 times radcrop in the screenshot.

    -manual:
            runs searchflixpix manually with each logprob, thresh,...value for each event file

    -target coords:
        coordinates of the source for manual positioning.
        Sexagecimal assumed if str, otherwise decimal.
        If set to None, the script will search in the directory names or in the file target name,
         or in the file target coordinates, in that order
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc,heasoft_init_alias,caldb_init_alias)

    if sudo_screen:
        sudo_mdp_use = input('Give sudo mdp')
    else:
        sudo_mdp_use = ''

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir=os.path.join(directory_use,'analysis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else ''))

    xtend_files=[elem for elem in glob.glob(os.path.join(anal_dir,'**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl.evt')]

    bashproc.sendline('cd '+os.path.join(os.getcwd(),anal_dir))

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str=str(time.time()).replace('.','p')

    log_path=os.path.join(log_dir,'xtend_SFP'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')+time_str+'.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout


        if filtering == 'flat_top':

            threshold_arr=np.repeat(None,len(xtend_files))

            for i_elem,elem_evt_raw in enumerate(xtend_files):

                if base_config!='indiv':
                    if not 'xtd_p0'+base_config in elem_evt_raw:
                        continue

                sky_image_file=elem_evt_raw.replace('.evt','_img_sky.ds')
                #computing a SKY image of the source
                xsel_util(elem_evt_raw.split('/')[-1],elem_evt_raw.replace('.evt','_img_sky.ds').split('/')[-1],
                          mode='image',image_mode='SKY',directory=anal_dir,
                          spawn=bashproc,heasoft_init_alias=heasoft_init_alias,caldb_init_alias=caldb_init_alias,
                          sudo_screen=sudo_screen,sudo_mdp=sudo_mdp_use)
                if target_coords is None:
                    print('\nAuto mode.')
                    print('\nAutomatic search of the directory names in Simbad.')

                    prefix = '_auto'

                    if source_name=='auto':
                        # using the full directory structure here
                        obj_auto = source_catal(bashproc, './', elem_evt_raw,
                                                target_only=target_only,
                                                use_file_target=use_file_target, )

                    else:
                        obj_auto=Simbad.query_object(source_name)[0]
                    # checking if the function returned an error message (folder movement done in the function)
                    if type(obj_auto) == str:
                        if not use_file_target_coords:
                            assert type(obj_auto) != str,obj_auto
                        else:
                            obj_auto = {'main_id': main_source_name}
                            obj_deg = [main_source_ra, main_source_dec]
                    else:

                        #if the output is already in degree units
                        if type(obj_auto['dec'])==np.float64 and type(obj_auto['ra'])==np.float64:
                            obj_deg=[str(obj_auto['ra']),str(obj_auto['dec'])]
                        else:
                            # careful the output after the first line is in dec,ra not ra,dec
                            obj_deg = sexa2deg([float(obj_auto['dec']).replace(' ', ':'), float(obj_auto['ra']).replace(' ', ':')])[::-1]
                            obj_deg = [str(obj_deg[0]), str(obj_deg[1])]
                else:
                    if type(target_coords[0])==str:
                        obj_deg=sexa2deg([target_coords[1].replace(' ',':'),target_coords[0].replace(' ',':')])[::-1]
                    else:
                        obj_deg=target_coords

                # loading the IMG file with mpdaf
                with fits.open(sky_image_file) as hdul:
                    img_data = hdul[0].data
                    src_mpdaf_WCS = mpdaf_WCS(hdul[0].header)
                    src_astro_WCS = astroWCS(hdul[0].header)
                    main_source_name = hdul[0].header['object']
                    main_source_ra = hdul[0].header['RA_OBJ']
                    main_source_dec = hdul[0].header['DEC_OBJ']

                img_obj_whole = Image(data=img_data, wcs=src_mpdaf_WCS)

                rad_crop = 8*rad_psf

                # breakpoint()

                try:
                    imgcrop_src = img_obj_whole.copy().subimage(center=obj_deg[::-1], size=rad_crop)
                except:
                    print('\nCropping region entirely out of the image. Field of view issue. Skipping this exposure...')
                    continue

                '''
                showing the bounds of the desired region
                no easy way to do it currently so we draw a circle manually after converting
                the angular coordinates to physical coordinates with the WCS
                note that the crop re-sizes the axes so we need to offset the position of the circle afterwards
                the "0,0" ends up at the bottom left of the graph 
                so we need to remove half a rad_crop in y and add half a rad_crop in x
                '''

                circle_rad_shift = rad_crop /(imgcrop_src.get_axis_increments()[0] * 3600)

                circle_rad_pix= rad_psf /(imgcrop_src.get_axis_increments()[0] * 3600)


                circle_psf = plt.Circle([circle_rad_shift/2,circle_rad_shift/2], circle_rad_pix,
                                     color='g', zorder=1000, fill=False)

                # testing if the resulting image is empty
                if len(imgcrop_src.data.nonzero()[0]) == 0:
                    print('\nEmpty image after cropping. Field of view Issue. Skipping this exposure...')
                    bashproc.sendline('\ncd $currdir')
                    return 'Cropped image empty.'

                plt.ioff()
                # plotting and saving imgcrop for verification purposes (if the above has computed, it should mean the crop is in the image)
                fig_catal_crop, ax_catal_crop = plt.subplots(1, 1, subplot_kw={'projection': src_astro_WCS},
                                                             figsize=(12, 10))
                ax_catal_crop.set_title('Cropped region around the theoretical source position')
                catal_plot = imgcrop_src.plot(cmap='plasma', scale='log')
                plt.colorbar(catal_plot, location='bottom', fraction=0.046, pad=0.04)
                ax_catal_crop.add_patch(circle_psf)

                plt.savefig(elem_evt_raw.replace('.evt','_catal_crop_screen.png'))

                plt.close()
                plt.ion()
                #further cropping to the actual testing radius
                imgcrop_thresh = img_obj_whole.copy().subimage(center=obj_deg[::-1], size=2*rad_psf)

                source_peak=imgcrop_thresh.peak()
                # testing if the resulting image contains a peak
                assert source_peak is not None,'Error: no peak detected in the rad_psf region'

                if base_config=='indiv':
                    threshold_arr[i_elem]=source_peak['data']
                else:
                    threshold_arr=np.repeat(source_peak['data'],len(xtend_files))





        #second loop with the actual command
        for i_elem,elem_evt_raw in enumerate(xtend_files):
            if filtering=='flat_top':

                cellsize_use='0'
                logprob2_use='-30'
                n_division_use='1'
                bgd_level_use=str(round(threshold_arr[i_elem]*threshold_mult))
                grade_use='ALL'

            else:
                cellsize_use=str(cellsize)
                logprob2_use=str(logprob2)
                n_division_use=str(n_division)
                bgd_level_use=str(bgd_level)
                grade_use=str(grade)

            logprob1_use=str(logprob1)

            bashproc.sendline('punlearn searchflickpix')

            #first run for the flagpix
            bashproc.sendline('searchflickpix '+
                              ' infile='+elem_evt_raw.split('/')[-1]+
                              ' outfile='+elem_evt_raw.split('/')[-1].replace('_cl.evt','.fpix')+
                              ' cellsize='+cellsize_use+
                              ' logprob1='+logprob1_use+
                              ' logprob2='+logprob2_use+
                              ' iterate=no'+
                              ' n_division='+str(n_division_use)+
                              ' bthresh='+bgd_level_use+
                              ' xcol=DETX'+
                              ' ycol=DETY'+
                              ' grade='+str(grade_use)+
                              ' cleanimg=no'+
                              ' impfac=320'+
                              ' clobber=YES')

            bashproc.expect('Finished',timeout=None)

            time.sleep(1)

            bashproc.sendline('echo valid')

            bashproc.expect('valid')

            #second one for the updated event file
            bashproc.sendline('searchflickpix '+
                              ' infile='+elem_evt_raw.split('/')[-1]+
                              ' outfile='+elem_evt_raw.split('/')[-1].replace('_cl.evt','_cl_SFP.evt')+
                              ' cellsize='+cellsize_use+
                              ' logprob1='+logprob1_use+
                              ' logprob2='+logprob2_use+
                              ' iterate=no'+
                              ' n_division='+str(n_division_use)+
                              ' bthresh='+bgd_level_use+
                              ' xcol=DETX'+
                              ' ycol=DETY'+
                              ' grade='+str(grade_use)+
                              ' cleanimg=yes'+
                              ' impfac=320'+
                              ' clobber=YES')

            bashproc.expect('Finished',timeout=None)

            time.sleep(1)

            #adding the GTI of the non_SFP event to the SFP event

            bashproc.sendline('ftappend '+elem_evt_raw.split('/')[-1]+'+2 '+
                              elem_evt_raw.split('/')[-1].replace('_cl.evt','_cl_SFP.evt'))

            bashproc.sendline('echo valid')

            bashproc.expect('valid')

        bashproc.sendline('exit')


def rsl_pixel_manip(pixel_str,remove_cal_pxl_resolve=True,mode='default',region_path=None):

    '''
    Small function to get correct pixel strings from a list of pixels

    As of Heasoftv6.35.1 ,the PIXEL extraction command has a bug and doesn't accept a single pixel as the last element
    We thus convert all pixels to ranges for the time being

    example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
    also accepts pixels to exclude, such as '-(10:14,28,32)'
    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    mode:
        -default
        -rmf:
            uses the rmf way of noting the intervals, with - instead of : for ranges
        -pix_list:
            returns the list of valid pixel

    if make_region is not None, saves a region with all valid pixels in a ds9 type file.
    '''
    if pixel_str in ['all',None]:
        pixel_ok_list=np.arange(36)
    else:
        if pixel_str.startswith('-'):
            pixel_inter_remove_init = np.array(pixel_str.split('-(')[1].split(')')[0].split(','), dtype=str)
            pixel_remove_list = ravel_ragged([[int(elem) if ':' not in elem else
                                               np.arange(int(elem.split(':')[0]), int(elem.split(':')[1]) + 1) for elem in
                                               pixel_inter_remove_init]])
            pixel_ok_list = [elem for elem in np.arange(36) if elem not in pixel_remove_list]

        else:
            pixel_inter_include_init = np.array(pixel_str.split(','), dtype=str)
            pixel_ok_list = ravel_ragged([[int(elem) if ':' not in elem else
                                           np.arange(int(elem.split(':')[0]), int(elem.split(':')[1]) + 1) for elem in
                                           pixel_inter_include_init]])

    if remove_cal_pxl_resolve:
        pixel_ok_list = [elem for elem in pixel_ok_list if elem not in [12]]

    if mode=='pix_list':
        return np.array(pixel_ok_list)

    pixel_ok_inter = list(interval_extract(pixel_ok_list))


    #preventing the bug for xselect
    if mode=='rmf':
        pixel_ok_inter_str = [str(elem[0]) + ':' + str(elem[1]) if elem[0] != elem[1] else str(elem[0])
                          for elem in pixel_ok_inter]
    else:
        #to prevent the bug
        pixel_ok_inter_str = [str(elem[0]) + ':' + str(elem[1])
                          for elem in pixel_ok_inter]

    pixel_ok_str_use = ','.join(pixel_ok_inter_str)

    if mode=='rmf':
        pixel_ok_str_use=pixel_ok_str_use.replace(':','-')

    if region_path is not None:
        with open(region_path,'w+') as pixel_region_f:
                pixel_region_f.write('physical\n')
                for pixel in pixel_ok_list:
                    pixel_coords=rsl_npixel_to_coord(pixel)
                    pixel_region_f.write('+box('+str(pixel_coords[0])+','+\
                                         str(pixel_coords[1])+',1,1.00000000) # text={'+str(pixel)+'} \n')

    return pixel_ok_str_use

def xsel_util(evt_path,save_path,mode,directory='./',
              e_low=None, e_high=None,
              #for products
              region_str=None,
              grade_str=None,
              image_mode='DET',
              remove_cal_pxl_resolve=True,
              gti_file=None,
              #for images
              sudo_screen=False,
              sudo_mdp='',
              #for lc
              exposure=0.8,binning=128,
              spawn=None,
              heasoft_init_alias='heainit',caldb_init_alias='caldbinit'):
    '''
    Uses Xselect to create a XRISM product from a bash spawn

        e_low and e_high should be in keV

    xtend arcsec to pix conversion:
    n_acsec=1.767984120108072*n_pix

    mode='image','lc' or 'spectrum'

    products only arguments:
        region_str

        region_str should be
            -for xtend: the region path for an xtend file

            -for resolve:
                -the pixel filtering string (after =) for a resolve file
                OR
                -a list of pixels to ignore in the form -(A,B,C,...)

                if set to None, will be ignored

                see https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html for the fucked up pixel numbering
                the calibration (inactive) pixel is number 12

        remove_cal_resolve:
            always removes the pixel 12 no matter which str is given (or even if no region is given)

        grade_str:the event grade of pixels
                    (see https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html for the explanation
                    and https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/xrism_abc.pdf p.16
                    for the numbering)

                  ignored if set to None

    image arguments

    sudo screen and sudo mdp pass a sudo password to enable ds9 within pexpect if it doesn't work out of SUDO

    lc arguments:
        exposure and binning are arguments for

    save_path should be a relative path from the current directory of bashproc or an absolute path

    Reason for doing two steps for resolve spectra
    Q1: Using xselect, I cannot simultaneously execute "pixel selection" and "pha cutoff". What should I do?

A1: For Heasoft ver. 6.34, you need to execute the save command between "pixel selection" and "pha cutoff".

    The exposure behavior determines the lower thresold for the fraction of a bin outside of selected gtis to
    reject that bin. See https://heasarc.gsfc.nasa.gov/FTP/software/lheasoft/release/doc/xselect/Xselect_2.5.pdf
    p. 25

    '''

    evt_dir = './' if '/' not in evt_path else evt_path[:evt_path.rfind('/')]
    evt_file = evt_path[evt_path.rfind('/') + 1:]

    if 'xtd_' in evt_file:
        xtd_mode=True
    else:
        xtd_mode=False

    if e_low is not None:
        if e_low == 'auto':
            if xtd_mode:
                pha_low = str(kev_to_PI(0.3, instru='xtend'))
            else:
                pha_low = str(kev_to_PI(2, instru='resolve'))
        else:
            pha_low=str(kev_to_PI(e_low,instru='xtend' if xtd_mode else 'resolve'))
    else:
        pha_low=None
    pha_high = None if e_high is None else str(kev_to_PI(e_high,instru='xtend' if xtd_mode else 'resolve'))

    if spawn is None:
        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')
        spawn_use.logfile_read = sys.stdout
        set_var(spawn_use,heasoft_init_alias,caldb_init_alias)
        spawn_use.sendline('cd '+os.getcwd())
    else:
        spawn_use=spawn

    spawn_use.sendline('xselect')
    spawn_use.expect('XSELECT')

    # session name
    spawn_use.sendline('new')

    # reading events
    spawn_use.sendline('read events')

    #removing the saved session if need bes
    line_code=spawn_use.expect(['Use saved session?','XRISM'])
    if line_code==0:
        spawn_use.sendline('no')

        spawn_use.sendline('read events')


    spawn_use.expect('Event file dir')
    spawn_use.sendline(evt_dir)

    spawn_use.expect('Event file list')
    spawn_use.sendline(evt_file)

    # resetting mission
    line_code=spawn_use.expect(['Reset','Observation Catalogue'])
    if line_code==0:
        spawn_use.sendline('yes')
        spawn_use.expect(['Observation Catalogue'])

    time.sleep(1)
    spawn_use.sendline('set image '+image_mode)

    temp_evt_name=None

    #energy filtering
    if e_low is not None and e_high is not None:
        spawn_use.sendline('filter pha_cutoff ' + pha_low + ' ' + pha_high)

    if grade_str is not None:
        spawn_use.sendline('filter GRADE '+grade_str)

    if region_str is not None or (not xtd_mode and remove_cal_pxl_resolve):
        if xtd_mode:
            spawn_use.sendline('filter region '+region_str)
        else:

            if e_low is not None and e_high is not None:
                #saving the first filtering because currently xselect cannot handle both at once
                spawn_use.sendline('extract events')
                temp_evt_name=evt_path.replace('.evt',str(time.time()).replace('.','')+'.evt')
                spawn_use.sendline('save events '+temp_evt_name)

                #reloading the events directly from the newly saved event list
                spawn_use.sendline('yes')

                spawn_use.sendline('set image ' + image_mode)

            spawn_use.sendline('filter column "PIXEL='+rsl_pixel_manip(region_str,
                                remove_cal_pxl_resolve=remove_cal_pxl_resolve)+'"')

    if gti_file is not None:
        spawn_use.sendline('filter time file '+gti_file)

    if mode=='image':

        spawn_use.sendline('set xybinsize 1')
        spawn_use.sendline('extract image')
        spawn_use.expect('Image')

        # commands to save image
        spawn_use.sendline('save image '+save_path)

        # # can take time so increased timeout
        # spawn_use.expect('Give output file name', timeout=120)
        #
        # spawn_use.sendline(save_path)

        over_code = spawn_use.expect(['File already exists', 'Wrote image to '])

        if over_code == 0:
            spawn_use.sendline('yes')
            spawn_use.expect(['Wrote image to '])

    if mode=='lc':

        spawn_use.sendline('set binsize ' + str(binning))
        spawn_use.sendline('extract curve exposure='+str(exposure))

        spawn_use.sendline('save curve ' + save_path)

        over_code = spawn_use.expect(['File already exists', 'Wrote FITS light curve '])

        if over_code == 0:
            spawn_use.sendline('yes')
            spawn_use.expect(['Wrote FITS light curve '])

    if mode=='spectrum':

        spawn_use.sendline('extract spectrum')

        spawn_use.sendline('save spectrum ' + save_path)

        over_code = spawn_use.expect(['File already exists', 'Wrote spectrum '])

        if over_code == 0:
            spawn_use.sendline('yes')
            spawn_use.expect(['Wrote spectrum '])

    if mode=='event':

        spawn_use.sendline('extract event')

        spawn_use.sendline('save event ' + save_path)

        over_code = spawn_use.expect(['File already exists', 'Wrote '])

        if over_code == 0:
            spawn_use.sendline('yes')
            spawn_use.expect(['Wrote '])


    print('Letting some time to create the file...')
    # giving some time to create the file
    time.sleep(1)

    for i_sleep in range(20):
        if not os.path.isfile(os.path.join(directory, save_path)):
            print('File still not ready. Letting more time...')
            time.sleep(5)
            if not os.path.isfile(os.path.join(directory, save_path)):
                breakpoint()
                pass

    if not os.path.isfile(os.path.join(directory, save_path)):
        print('Issue with file check or file creation')
        breakpoint()

    if mode=='event':

        spawn_use.expect(['Use filtered events'])
        spawn_use.sendline('no')

    spawn_use.sendline('exit')
    spawn_use.sendline('no')

    #to ensure xselect is quitted properly
    spawn_use.sendline('echo valid')
    spawn_use.expect('valid')

    if mode=='image':
        # opening the image file and saving it for verification purposes
        disp_ds9(os.path.join(os.getcwd(),directory,save_path),
                 scale=('log' if xtd_mode else 'linear'),
                 screenfile=os.path.join(directory,save_path.replace('.ds', '_screen.png')),
                                give_pid=False,
                 sudo_mode=sudo_screen,sudo_mdp=sudo_mdp)

    if mode=='lc':

        plot_lc(save_path,binning,directory=directory,outdir='/'.join(save_path.split('/')[:-1]),
                            e_low=e_low,e_high=e_high,save=True)

    if temp_evt_name is not None:

        spawn_use.sendline('rm '+temp_evt_name)

def plot_temp_evol(directory='auto_repro',pixels='all',repro_suffix='repro',man_file='',save=True):

    '''
    plots the temperature evolution of individual pixels along an observation

    if man_file is an empty string, searches in the directory (or auto directory if set to auto/auto_repro)
    for the calibration file (in /resolve/event_uf/xaXXXXXXXXXrsl_000_fe55.ghf

    Similarly to Figure 6 of the energy scale reports (https://heasarc.gsfc.nasa.gov/FTP/xrism/postlaunch/gainreports/)
    pixels should be 'all' or a list of integers
    '''

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    if man_file!='':
        fe55_file=man_file
    else:
        fe55_file=[elem for elem in glob.glob(os.path.join(directory_use,'resolve','event_uf','**'))\
                   if 'fe55.ghf' in elem]
        assert len(fe55_file)>0,'Error: fe55 file not found in the '+str(directory_use)+' directory'
        fe55_file=fe55_file[0]

    with fits.open(fe55_file) as hdul:

        mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

        tstart_s=hdul[1].header['TSTART']
        obs_start = mjd_ref + TimeDelta(tstart_s, format='sec')

        cal_event_time=hdul[1].data['TIME']
        cal_event_pix=hdul[1].data['PIXEL']
        cal_event_temp=hdul[1].data['TEMP_FIT']*1000-50

        file_obsid=hdul[1].header['OBS_ID']

    fig_lc, ax_lc = plt.subplots(1, figsize=(16, 8),layout='constrained')

    plt.suptitle( ' Temperature evolution by pixel for observation '+file_obsid)

    plt.xlabel('Time (s) after ' + obs_start.isot)
    plt.ylabel('pixel fit temperature -50mK (K)')

    ls_list=['solid','dotted','dashed','dashdot']

    indiv_plots=[]
    for i_pix in range(36):
        if pixels!='all' and i_pix not in pixels:
            continue
        pixel_mask=cal_event_pix==i_pix
        indiv_plots+=[plt.plot(cal_event_time[pixel_mask]-tstart_s,cal_event_temp[pixel_mask],ls=ls_list[i_pix//9],
                 marker='.',
                 label=str(i_pix))]

    plt.legend(bbox_to_anchor=(0.5, 1.07),loc='center',ncol=len(indiv_plots)//3)

    plt.minorticks_on()

    if save:
        plt.savefig(os.path.join(directory_use if man_file=='' else '','temp_evol_pixels.pdf'))


def disp_ds9(file, zoom='auto', scale='log', regfile='', screenfile='', give_pid=False, close=True,
             kill_last='',spawn=None,
             sudo_mode=True,sudo_mdp=''):
    '''
    Regfile is an input, screenfile is an output. Both can be paths
    If "screenfile" is set to a non empty str, we make a screenshot of the ds9 window in the given path
    This is done manually since the ds9 png saving command is bugged

    if regfile is a list of strings, loads all regions successively instead

    if give_pid is set to True, returns the pid of the newly created ds9 process

    In some installations like mine ds9 struggles to start outside of sudo,
     so there is a sudo mode where a sudo command
    (with password) is used to launch and remove ds9

    ds9 window handling and screenshoting require the ubuntu packages wmctrl and imagemagick

    '''

    if scale == 'linear 99.5':
        scale_use = 'mode 99.5'
    elif ' ' in scale:
        scale_use = scale.split(' ')[0] + ' -scale mode ' + scale.split(' ')[1]
    else:
        scale_use=scale

    # if automatic, we set a higher zoom for timing images since they usually only fill part of the screen by default
    if zoom == 'auto':
        if 'rsl_' in file:
            zoom = 50

            pan_str=' -pan to 3.5 3.5 '
        else:
            zoom = 1.

            #different pan values to center the image depending on what mode is used
            pan_str=' -pan to 734 910' if 'xtd_p0313' in file else ' -pan to 1250 910' if 'xtd_p0320' in file else ''

            pan_str+=' -rotate 90 '
    # parsing the open windows before and after the ds9 command to find the pid of the new ds9 window
    if screenfile != '' or give_pid:
        windows_before = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

    if type(regfile)!=str:
        if len(regfile)==0:
            regfile_use_str=''
        else:
            regfile_use_str='-region '+(' -region '.join(regfile))
    else:
        regfile_use_str='' if regfile=='' else '-region '+regfile

    if spawn is None:
        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')
        spawn_use.logfile_read = sys.stdout
    else:
        spawn_use=spawn

    if sudo_mode:


        if sudo_mdp=='':
            sudo_mdp_use=input('Give sudo mdp')
        else:
            sudo_mdp_use=sudo_mdp

        spawn_use.sendline(
            'echo "'+sudo_mdp_use+'" | sudo -S ds9 -view buttons no -cmap Heat -geometry 1500x1500 -scale '
                       + scale_use + ' -mode region '
                       + file + ' -zoom ' + str(zoom) + ' ' + regfile_use_str + pan_str +' &')

        # the timeout limit could be increased for slower computers or heavy images
        spawn_use.expect(['password', pexpect.TIMEOUT], timeout=1)

    else:
        spawn_use.sendline('ds9 -view buttons no -cmap Heat -geometry 1080x1080 -scale ' + scale_use + ' -mode region '
                       + file + ' -zoom ' + str(zoom) + ' ' + regfile_use_str + pan_str +' &')

    spawn_use.sendline('echo valid')
    spawn_use.expect('valid')

    # second part of the windows parsing

    ds9_pid = 0

    if screenfile != '' or give_pid:

        windows_after = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

        # since sometimes the ds9 window takes time to load, we loop until the window creation to be sure we can take
        # the screenshot
        delay = 0
        while len(windows_after) == len(windows_before) and delay <= 10:
            time.sleep(2)
            #requires wmctrl
            windows_after = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
            delay += 1

        for elem in windows_after:
            if elem not in windows_before:
                ds9_pid = elem.split(' ')[0]
                print('\nIdentified the new ds9 window as process ' + ds9_pid)

                if screenfile != '':
                    print('\nSaving screenshot...')
                    #requires imagemagick
                    os.system('import -window ' + ds9_pid + ' ' + screenfile)

    if close:
        time.sleep(1)
        os.system('wmctrl -ic ' + str(ds9_pid))

    # we purposely do this at the very end
    if kill_last != '':
        print('\nClosing previous ds9 window...')

        os.system('wmctrl -ic ' + kill_last)

    if give_pid:
        return ds9_pid

def extract_img(directory='auto_repro',anal_dir_suffix='',
                instru='all',
                   use_raw_evt_xtd=False,use_raw_evt_rsl=False,
                   heasoft_init_alias='heainit',caldb_init_alias='caldbinit',
                   sudo_screen=False,
                   parallel=False,repro_suffix='repro',e_band=None):

    '''
    Extract images from event files in the analysis subdirectory of a directory

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    e_band: None or 'float_float'
    '''
    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc,heasoft_init_alias,caldb_init_alias)

    if sudo_screen:
        sudo_mdp_use = input('Give sudo mdp')
    else:
        sudo_mdp_use = ''

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl' + ('' if use_raw_evt_xtd else '_SFP') + '.evt')]

    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))


    time_str=str(time.time()).replace('.','p')

    log_path=os.path.join(log_dir,'extract_img'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')+time_str+'.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):


        if not parallel:
            bashproc.logfile_read=sys.stdout

        if len(resolve_files)!=0:
            # for removing the calibration sources
            os.system('cp $HEADAS/refdata/region_RSL_det.reg '+anal_dir)

        bashproc.sendline('cd '+os.path.join(os.getcwd(),anal_dir))

        for elem_evt in resolve_files+xtend_files:

            xsel_util(elem_evt.split('/')[-1],elem_evt.split('/')[-1].replace('.evt','_img'+
                                                                        ('' if e_band is None else str(e_band))+'.ds'),
                     mode='image',directory=anal_dir,sudo_screen=sudo_screen,sudo_mdp=sudo_mdp_use,
                      spawn=bashproc,
                      e_low=(None if e_band is None else float(e_band.split('_')[0])),
                      e_high=(None if e_band is None else float(e_band.split('_')[1])))


def create_gtis(directory='auto_repro',anal_dir_suffix='',
                split_arg='orbit',split_lc_file='file',split_lc_method=None,
                split_auto_bin=1,
                gti_tool='NICERDAS',
                heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                gti_subdir='gti',
                thread=None, parallel=False,repro_suffix='repro'):
    '''
    wrapper for a function to split xrism obsids into indivudal gti portions with different methods
    the default binning when split_lc_method is set is 1s

    products end by default in Obsdir/analysis+anal_dir_suffix/gti_subdir

    overwrite is always on here since we don't use a specific nicerdas task with the overwrite option

    split_arg:

        -orbit:split each obs into each individual nicer observation period. Generally, should always be enabled.
               GTIs naming: obsid-XXX chronologically for each split


        -manual: provides an interactive window to make individual gti splits. GTI naming: obsid-MXXX
            by default a single interval
        -manual_multi:
            acts as manual but allows for a set of successive intervals until the end of the observation is reached
        -manual_combi
            acts as manual but allows for a set of successive intervals that will be combined in a single GTI

    split_lc_file:
        -lightcurve used as the based for the lc splits and gti computation. Can be either a lightcurve path
        or an event path if split_lc_method is not set to None

    split_lc_method:
        if None, split_lc_file is taken as the path of the base lightcurve for the computation
        if an id:
            resolve_allpixel: computes a resolve lightcurve in 2-10 keV with all pixels except the calibration pixel
                             from the event file given in split_lc_file (assumed to be in the directory's analysis folder)

    split_auto_binning:
        if split_lc_method is an id, the binning of the lightcurve that will be used as a base for the gti cut

    gti_tool:
        software used for gti creation. For now only NICERDAS

    -parallel: bool:tells the function it's running in a parallel configuration.
               Modifies the logging to avoid issues with redirections

    '''

    def create_gti_files(id_gti, data_lc, orbit_prefix, suffix, file_base,
                         time_gtis, gti_tool='NICERDAS',outdir='',anal_dir='',
                         id_gti_multi=None):

        '''
        creates a gti file from a list of indexes of times which will be picked in data_lc

        Note: currently only works with xirsm lightcurves since it edits the nicer gti with
            the mjd_ref and mjd_reff of the input lightcurve

        1.creates a copy of a lightcurve_type file (file_base) with an additional column with a gti mask
                (typically this is a lightcurve file for flexible resolution)

        2.creates the gti file itself (in the Obsid/dir directory if dir is not '', otherwise where file_base is)

        gti_tool:
            -NICERDAS    uses nicerdas nigti tool


        if id_gti_multi is provided, id_gti is assumed to be the raveled array and id_gti_multi an array with
        an additional dimension with each required gti split (typically by orbit).
        '''

        if len(id_gti) == 0:
            return



        # Here we use the housekeeping file as the fits base for the gti mask file
        with fits.open(file_base) as fits_gti_base:

            mjd_refi_xrism =fits_gti_base[1].header['MJDREFI']
            mjd_reff_xrism=fits_gti_base[1].header['MJDREFF']


        input_dir=os.path.join(anal_dir,outdir)

        # creating the gti expression
        gti_path = os.path.join(input_dir,
                                (file_base[:file_base.rfind('.')]+
                                '_gti_' + orbit_prefix + suffix + '.gti').split('/')[-1])

        if outdir!='':

            os.system(' mkdir -p '+os.path.join(directory_use,input_dir))

        if id_gti_multi is None:
            # preparing the list of gtis to replace manually
            gti_intervals = np.array(list(interval_extract(id_gti))).T
        else:
            # doing the interval extract within eahc subarray to automatically get the splits
            gti_intervals_full = [list(interval_extract(id_gti_multi[i_orbit])) for i_orbit in
                                  range(len(id_gti_multi))]
            gti_intervals = [[], []]

            for i_orbit in range(len(id_gti_split)):
                gti_intervals[0] += np.array(gti_intervals_full[i_orbit]).T[0].tolist()
                gti_intervals[1] += np.array(gti_intervals_full[i_orbit]).T[1].tolist()

            gti_intervals = np.array(gti_intervals)

        delta_time_gtis = (time_gtis[1] - time_gtis[0]) / 2

        if gti_tool == 'NICERDAS':
            '''
            the task nigti doesn't accept ISOT formats with decimal seconds so we use NICER MET instead 
            (see https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/nigti.html)

            we still add a -0.5*delta and +0.5*delta on each side to avoid issues with losing the last bins of lightcurves
            '''

            file_base_name=file_base.split('/')[-1]
            # creating the gti expression
            gti_input_path = os.path.join(input_dir,
                        file_base_name[:file_base.rfind('.')] + '_gti_input_' + orbit_prefix + suffix + '.txt')

            with open(gti_input_path, 'w+') as f_input:
                f_input.writelines([str(start_obs_s + time_gtis[gti_intervals[0][i]] - delta_time_gtis) + ' ' +
                                    str(start_obs_s + time_gtis[gti_intervals[1][i]] + delta_time_gtis) + '\n' \
                                    for i in range(len(gti_intervals.T))])

            bashproc.sendline('nigti @' + gti_input_path + ' ' + gti_path + ' clobber=YES chatter=4')
            bashproc.expect('ngti=')


            #opening the file
            with fits.open(gti_path,mode='update') as fits_gti:
                fits_gti[1].header['MJDREFI']=mjd_refi_xrism
                fits_gti[1].header['MJDREFF']=mjd_reff_xrism
                fits_gti.flush()

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    time_str=str(time.time()).replace('.','p')

    io_log = open(directory_use + '/create_gtis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')+time_str+'.log', 'w+')

    # ensuring a good obsid name even in local
    if directory_use == './':
        obsid = os.getcwd().split('/')[-1]
    else:
        obsid = directory_use

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8', logfile=io_log if parallel else None)

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    print('\n\n\nCreating gtis products...')

    set_var(bashproc)

    # removing old gti files
    old_files_gti = [elem for elem in glob.glob(os.path.join(directory_use, 'analysis/**'), recursive=True) if
                     '_gti_' in elem]

    for elem_file_gti in old_files_gti:
        os.remove(elem_file_gti)

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))


    log_path=os.path.join(log_dir,'extract_gtis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')
                                        +'_'+gti_subdir+time_str+'.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):


        if not parallel:
            bashproc.logfile_read = sys.stdout

        '''
        new method for the flares following https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/flares/
        '''
        # should always be in auxil but we cover rank -2 directories this way. Also testing both gunzipped and non-gunzipped
        # (one is enough assuming they're the same)

        if split_lc_method is None:
            lc_input=os.path.join(anal_dir,split_lc_file)
        else:
            if split_lc_method=='resolve_allpixel':

                split_lc_for_gti=split_lc_file.replace(split_lc_file[split_lc_file.rfind('.'):],
                                                       '_auto_gti_input'+split_lc_file[split_lc_file.rfind('.'):])

                xsel_util(split_lc_file,save_path=split_lc_for_gti,
                          mode='lc',e_low=2,e_high=10,grade_str='0:0',binning=split_auto_bin)

                lc_input=split_lc_for_gti

        with fits.open(lc_input) as fits_input:

            data_input = fits_input[1].data

            # from https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time_resolved_spec/
            # this value is offset by the mjd_ref value

            # note that there's an offset of 8 seconds between this value and the actual 1st column of
            # the time vector, for some reason ???

            # note that using leapinit can create a "Dumping CFITSIO error stack", see here:
            # https: // heasarc.gsfc.nasa.gov / docs / nicer / analysis_threads / common - errors /
            # so we don't consider the leapinit
            # start_obs_s=fits_mkf[1].header['TSTART']+fits_mkf[1].header['TIMEZERO']-fits_mkf[1].header['LEAPINIT']

            start_obs_s = fits_input[1].header['TSTART']
            # saving for titles later
            mjd_ref = Time(fits_input[1].header['MJDREFI'] + fits_input[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

            obs_start_str = str(obs_start.to_datetime())

            binning_use=(data_input['TIME'][1]-data_input['TIME'][0]) if split_lc_method is None else split_auto_bin

            #in xrism it's normalized against TSTART (with a half bin delta)
            time_obs = data_input['TIME']+binning_use/2

            input_rate = data_input['RATE']

        print('Creating gtis from ' + lc_input + ' file data\n')


        if 'orbit' in split_arg:

            # adding gaps of more than 1000s as cuts in the gtis

            # first computing the gti where the jump happens
            id_gti_split = [-1]
            # adding gaps of more than 100s as cuts in the gtis
            for i in range(len(time_obs) - 1):
                if time_obs[i + 1] - time_obs[i] > 2000:
                    id_gti_split += [i]

                id_gti_orbit = []
                if len(id_gti_split) == 1:
                    id_gti_orbit += [range(len(time_obs))]
                else:
                    for id_split in range(len(id_gti_split)):
                        # note:+1 at the end since we're using a range
                        id_gti_orbit += [list(range(id_gti_split[id_split] + 1, (len(time_obs) - 1 if \
                                                                                     id_split == len(id_gti_split) - 1 else
                                                                                 id_gti_split[id_split + 1]) + 1))]

                n_orbit = len(id_gti_orbit)

                id_gti=id_gti_orbit

                for i_orbit, id_gti_orbit in enumerate(id_gti_orbit):

                    create_gti_files(id_gti[i_orbit], data_input, str_orbit(i_orbit), suffix='',file_base=lc_input,
                             time_gtis=time_obs, gti_tool=gti_tool,outdir=gti_subdir)


        if 'manual' in split_arg:

            split_gti_arr = []

            i_cut = 0
            split_keyword = 'MAN'

            if split_arg in ['manual_multi','manual_combi']:
                cut_times = []

                while len(cut_times)==0 or cut_times[-1][-1] < time_obs[-1]:

                    plot_lc_output=plot_lc(lc_input.replace(anal_dir,'.'),binning=binning_use,interact=True,
                                       #here to allow to start from the previous ending gti value for each subsequent
                                       #cut
                                       directory=anal_dir,
                                       interact_tstart=None if len(cut_times)==0 else cut_times[-1][-1]
                                       ,save=True,suffix=split_keyword+str_orbit(i_cut),
                                       outdir=gti_subdir)
                    cut_times += \
                        [plot_lc_output[:2]]

                    i_cut+=1

                    print('Added manual gti split interval at for t in ' + str(cut_times[-1]) + ' s')

                    if plot_lc_output[-1]==1:
                        break
            else:

                plot_lc_output =plot_lc(lc_input.replace(anal_dir,'.'),binning=binning_use,interact=True,
                                   #here to allow to start from the previous ending gti value for each subsequent
                                   #cut
                                   directory=anal_dir,
                                   interact_tstart=None,save=True,suffix=split_keyword+str_orbit(i_cut),
                                   outdir=gti_subdir)
                cut_times = \
                    [plot_lc_output[:2]]

            n_cuts = len(cut_times)

            cut_gtis_id=[]
            for elem_cut in cut_times:
                cut_gtis_id += [[abs(np.array([1e6 if elem <0 else elem for elem in time_obs-elem_cut[0]])).argmin(),
                        abs(np.array([1e6 if elem >0 else elem for elem in time_obs-elem_cut[1]])).argmin()]]

            #check that the rest works

            # note that the min(i_cut+1) offsets all but the first cut's gti starts by one, and the end is always
            # offset by one. So here we make the choice that the cut is part of the gti up to that cut for
            # gtis that follow one another
            split_gti_arr = np.array(
                [(0 if i_cut==0 else int(cut_gtis_id[i_cut][0]-cut_gtis_id[i_cut-1][1]==0))
                 + np.arange(cut_gtis_id[i_cut][0],
                             cut_gtis_id[i_cut][1] + max(1 - i_cut, 0))
                 for i_cut in range(n_cuts)], dtype=object)

            if split_arg=='manual_combi':
                # create the gti files with a "S" keyword and keeping the orbit information in the name
                i_split=0
                split_gti_combi=ravel_ragged(split_gti_arr)

                if len(split_gti_combi) > 0:
                    create_gti_files(split_gti_combi, data_input, split_keyword+'_COMBI' + str_orbit(i_split),
                                     suffix='',
                                     file_base=lc_input, anal_dir=anal_dir,
                                     time_gtis=time_obs, gti_tool=gti_tool, outdir=gti_subdir)
            else:
                # create the gti files with a "S" keyword and keeping the orbit information in the name
                for i_split, split_gtis in enumerate(split_gti_arr):
                    if len(split_gtis) > 0:
                        create_gti_files(split_gtis, data_input, split_keyword+str_orbit(i_split), suffix='',
                                         file_base=lc_input,anal_dir=anal_dir,
                                         time_gtis=time_obs, gti_tool=gti_tool,outdir=gti_subdir)


        # exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()


def extract_lc(directory='auto_repro', anal_dir_suffix='',lc_subdir='lc',
                   use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                    instru='all',
                   region_src_xtd='auto', region_bg_xtd='auto',
                   pixel_str_rsl='branch_filter',grade_str_rsl=None,
                   remove_cal_pxl_resolve=True,
                   gti=None,gti_subdir='gti',
                   band='1-3+3-6+6-10+0.3-10',
                   binning='128+1',exposure_rsl=0.6,
                   exposure_xtd=0.0,
                   heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                   parallel=False,repro_suffix='repro'):

    '''

    Extract lightcurves from event files in the analysis subdirectory of a directory

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    Instru determines whether the analysis is run on all or a single instrument

    region_src_xtd/region_bg_xtd:
        if set to auto, fetches source/background regions with the evt file name _src_reg.reg/_bg_reg.reg
        as the base, and only extracts products when corresponding files are found
        Regions are assumed to be in DET coordinates
        Will make a screenshot of the regions before saving the spectra

    pixel_str_xrism:
        pixel filtering list for xrism
            example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
            also accepts pixels to exclude, such as '-(10:14,28,32)'

            If different configurations, join sets with +
            if one element of the set is all, the keyword is ignored

            products are named according to their pixel combination if there is more than one pixel_str

            example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
            also accepts pixels to exclude, such as '-(10:14,28,32)'
    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    grade_str_rsl:the event grade of the pixels
                    (see https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html for the explanation
                    and https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/xrism_abc.pdf p.16
                    for the numbering)

                  ignored if set to None

    gti:
        apply a gti cut to the data
        if set to 'all', uses all the gti files in the gti subdir argument
        if set to None, does not use any GTI

    band:
        string for a set of bands in "A-B+C-D+..." style

    binning:
        string for a set of binnings in 'A+B+...' style
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    os.system('mkdir -p '+os.path.join(anal_dir,lc_subdir))


    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl' + ('' if use_raw_evt_xtd else '_SFP') + '.evt')]

    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]


    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))


    time_str=str(time.time()).replace('.','p')

    log_path=os.path.join(log_dir,'extract_lc'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')
                                   +'_'+lc_subdir+'_'+gti_subdir+time_str+'.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        if len(resolve_files) != 0:
            # for removing the calibration sources
            os.system('cp $HEADAS/refdata/region_RSL_det.reg ' + anal_dir)

        bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

        if gti is None:
            gti_str_arr=['']
            gti_files_arr=[None]
        else:
            if gti=='all':
                gti_files_arr=[elem.replace(anal_dir,'.')
                               for elem in glob.glob(os.path.join(anal_dir,gti_subdir)+'/**')
                               if elem.endswith('.gti')]
                gti_str_arr=[elem[elem.rfind('_gti_'):].split('.')[0] for elem in gti_files_arr]
            else:
                gti_files_arr=gti
                gti_str_arr='_'+gti.split('/')[-1]

        for elem_gti_str,elem_gti_file in zip(gti_str_arr,gti_files_arr):
            for elem_evt in resolve_files + xtend_files:

                xtd_mode='xtd_' in elem_evt

                if xtd_mode:
                    if region_src_xtd=='auto':
                        region_src_xtd_use = elem_evt.replace('.evt','_src_reg.reg')
                    else:
                        region_src_xtd_use=region_src_xtd

                    if region_bg_xtd=='auto':
                        region_bg_xtd_use = elem_evt.replace('.evt','_bg_reg.reg')
                    else:
                        region_bg_xtd_use=region_bg_xtd

                    for i_reg,elem_region in enumerate([region_src_xtd_use,region_bg_xtd_use]):

                        if elem_region is None:
                            print('Skipping '+('source' if i_reg==0 else 'bg')+' region for event file '+elem_evt)
                            continue

                        if not os.path.isfile(elem_region):
                            print('No matching region found. Skipping '+('source' if i_reg==0 else 'bg')+
                                  'event file '+elem_evt)
                            continue

                        reg_str=('_auto_src' if region_src_xtd=='auto' else region_src_xtd.split('.')[0]) if i_reg==0 else \
                                ('_auto_bg' if region_bg_xtd == 'auto' else region_bg_xtd.split('.')[0])

                        for elem_band in band.split('+'):
                            for elem_binning in binning.split('+'):
                                product_name=os.path.join(lc_subdir,
                                             elem_evt.split('/')[-1].replace('.evt',
                                             reg_str+
                                             '_lc_'+elem_band+'_'+elem_binning+'s'+
                                             elem_gti_str+'.lc'))

                                xsel_util(elem_evt.split('/')[-1],
                                          product_name,
                                          mode='lc',
                                          directory=anal_dir,
                                          region_str=elem_region.split('/')[-1],
                                          e_low=float(elem_band.split('-')[0]),
                                          e_high=float(elem_band.split('-')[1]),
                                          binning=elem_binning,
                                          exposure=exposure_xtd,
                                          spawn=bashproc,
                                          gti_file=elem_gti_file)

                else:

                    if pixel_str_rsl.startswith('branch_filter'):
                        # reading the branch filter file
                        with open(elem_evt.replace('.evt','_'+pixel_str_rsl+'.txt')) as branch_f:
                            branch_lines=branch_f.readlines()
                        branch_filter_line=[elem for elem in branch_lines if not elem.startswith('#')][0]
                        #reformatting the string
                        pixel_str_rsl_use='-('+branch_filter_line[1:-2]+')'
                    else:
                        pixel_str_rsl_use=pixel_str_rsl

                    for elem_pixel_str in pixel_str_rsl_use.split('+'):

                        reg_str = pixel_str_rsl if pixel_str_rsl.startswith('branch_filter') else elem_pixel_str

                        for elem_band in band.split('+'):
                            for elem_binning in binning.split('+'):
                                product_name = os.path.join(lc_subdir,
                                                            elem_evt.split('/')[-1].replace('.evt',
                                '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                                ('_withcal' if not remove_cal_pxl_resolve else '')+
                                ('' if grade_str_rsl is None else '_grade_' + grade_str_rsl.replace(':', 'to'))+
                                '_lc_' + elem_band + '_' + elem_binning + 's' +
                                elem_gti_str+'.lc'))

                                xsel_util(elem_evt.split('/')[-1],
                                          product_name,
                                          mode='lc',
                                          directory=anal_dir,
                                          region_str=elem_pixel_str if elem_pixel_str!='all' else None,
                                          grade_str=grade_str_rsl,
                                          remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                          e_low=float(elem_band.split('-')[0]),
                                          e_high=float(elem_band.split('-')[1]),
                                          binning=elem_binning,
                                          exposure=exposure_rsl,
                                          spawn=bashproc,
                                          gti_file=elem_gti_file)

def extract_sp(directory='auto_repro', anal_dir_suffix='',sp_subdir='sp',
                   use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                    instru='all',
                   region_src_xtd='auto', region_bg_xtd='auto',
                   pixel_str_rsl='branch_filter', grade_str_rsl='0:1',
                   remove_cal_pxl_resolve=True,
                   gti=None, gti_subdir='gti',
                   e_low_rsl=None,e_high_rsl=None,
                   e_low_xtd=None,e_high_xtd=None,
                   screen_reg=True,sudo_screen=False,
                   heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                   parallel=False,repro_suffix='repro',
                   image_mode='DET'):

    '''

    #pixels for 4U: '-(5,23,27,29,30)'

    Extract spectra from event files in the analysis subdirectory of a directory

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    Instru determines whether the analysis is run on all or a single instrument


    region_src_xtd/region_bg_xtd:
        if set to auto, fetches source/background regions with the evt file name _src_reg.reg/_bg_reg.reg
        as the base, and only extracts products when corresponding files are found
        manual region names are assumed to be in the anal_dir directory
        Regions are assumed to be in DET coordinates

        ds9 saves should be in physical

        Will make a screenshot of the regions before saving the spectra

    pixel_str_rsl:
        pixel filtering list for xrism
            if set to branch_filter, excludes the pixels listed in the branch_filter.txt file of
            the observation, made by resolve_BR
            otherwise, manual input:

            example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
            also accepts pixels to exclude, such as '-(10:14,28,32)'

            If different configurations, join sets with +
            if one element of the set is all, the keyword is ignored

            products are named according to their pixel combination if there is more than one pixel_str

    grade_str_rsl:the event grade of the pixels
                    (see https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html for the explanation
                    and https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/xrism_abc.pdf p.16
                    for the numbering)

                  ignored if set to None

    gti:
        apply a gti cut to the data
        if set to 'all', uses all the gti files in the gti subdir argument
        if set to None, does not use any GTI



    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    e_low/e_high: if not None, the energy bounds of the spectra
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    os.system('mkdir -p '+os.path.join(anal_dir,sp_subdir))

    if sudo_screen:
        sudo_mdp_use = input('Give sudo mdp')
    else:
        sudo_mdp_use = ''


    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl' + ('' if use_raw_evt_xtd else '_SFP') + '.evt')]

    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str = str(int(time.time()))

    log_path=os.path.join(log_dir,'extract_sp'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')
                      +'_'+sp_subdir+'_'+gti_subdir+ time_str + '.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        if len(resolve_files) != 0:
            # for removing the calibration sources
            os.system('cp $HEADAS/refdata/region_RSL_det.reg ' + anal_dir)

        bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

        if gti==None:
            gti_str_arr=['']
            gti_files_arr=[None]
        else:
            if gti=='all':
                gti_files_arr=[elem.replace(anal_dir,'.')
                               for elem in glob.glob(os.path.join(anal_dir,gti_subdir)+'/**')
                               if elem.endswith('.gti')]
                gti_str_arr=[elem[elem.rfind('_gti_'):].split('.')[0] for elem in gti_files_arr]
            else:
                gti_files_arr=gti
                gti_str_arr='_'+gti.split('/')[-1]

        for elem_gti_str,elem_gti_file in zip(gti_str_arr,gti_files_arr):

            for elem_evt in resolve_files + xtend_files:

                xtd_mode='xtd_' in elem_evt

                if xtd_mode:
                    if region_src_xtd=='auto':
                        region_src_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_src_reg.reg'))
                    elif region_src_xtd is None:
                        region_src_xtd_use=None
                    else:
                        region_src_xtd_use=os.path.join(os.getcwd(),anal_dir,region_src_xtd)

                    if region_bg_xtd=='auto':
                        region_bg_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_bg_reg.reg'))
                    elif region_bg_xtd is None:
                        region_bg_xtd_use=None
                    else:
                        region_bg_xtd_use=os.path.join(os.getcwd(),anal_dir,region_bg_xtd)

                    if (region_bg_xtd_use is not None and os.path.isfile(region_bg_xtd_use))\
                    or (region_src_xtd_use is not None and os.path.isfile(region_src_xtd_use)) and screen_reg:
                        disp_ds9(os.path.join(os.getcwd(),elem_evt.replace('.evt','_img.ds')),scale='log',
                             regfile=[elem for elem in [region_src_xtd_use,region_bg_xtd_use] if elem is not None],
                             sudo_mode=sudo_screen,sudo_mdp=sudo_mdp_use,
                             screenfile=elem_evt.replace('.evt','_screen_reg.png'))

                    for i_reg,elem_region in enumerate([region_src_xtd_use,region_bg_xtd_use]):

                        if elem_region is None:
                            print('Skipping '+('source' if i_reg==0 else 'bg')+' region for event file '+elem_evt)
                            continue

                        if not os.path.isfile(elem_region):
                            print('No matching region found. Skipping '+('source' if i_reg==0 else 'bg')+
                                  ' region for event file '+elem_evt)
                            continue

                        reg_str=('_auto_src' if region_src_xtd=='auto' else '_manual_src_'+region_src_xtd.split('.')[0]) if i_reg==0 else \
                                ('_auto_bg' if region_bg_xtd == 'auto' else '_manual_bg_'+region_bg_xtd.split('.')[0])

                        product_name = os.path.join(sp_subdir,
                                                    elem_evt.split('/')[-1].replace('.evt',reg_str + '_sp' +
                                ('_' + str(e_low_xtd) + '_' + str(e_high_xtd) if (e_low_xtd is not None and e_high_xtd is not None) else '')
                        +elem_gti_str +'.ds'))

                        xsel_util(elem_evt.split('/')[-1],
                                  product_name,
                                  mode='spectrum',
                                  directory=anal_dir,
                                  region_str=elem_region,
                                  e_low=e_low_xtd,
                                  e_high=e_high_xtd,
                                  spawn=bashproc,
                                  gti_file=elem_gti_file,image_mode=image_mode)


                else:

                    if pixel_str_rsl.startswith('branch_filter'):
                        # reading the branch filter file
                        with open(elem_evt.replace('.evt','_'+pixel_str_rsl+'.txt')) as branch_f:
                            branch_lines=branch_f.readlines()
                        branch_filter_line=[elem for elem in branch_lines if not elem.startswith('#')][0]
                        #reformatting the string
                        pixel_str_rsl_use='-('+branch_filter_line[1:-2]+')'
                    else:
                        pixel_str_rsl_use=pixel_str_rsl

                    for elem_pixel_str in pixel_str_rsl_use.split('+'):

                        reg_str = pixel_str_rsl if pixel_str_rsl.startswith('branch_filter') else elem_pixel_str


                        if screen_reg:
                            disp_ds9(os.path.join(os.getcwd(),elem_evt.replace('.evt','_img.ds')),scale='linear',
                                 regfile=os.path.join(anal_dir,'region_RSL_det.reg'),
                                 screenfile=elem_evt.replace('.evt', '_screen_reg.png'),
                                 sudo_mode=sudo_screen,sudo_mdp=sudo_mdp_use)

                        product_name = os.path.join(sp_subdir,elem_evt.split('/')[-1].replace('.evt',
                                '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                                ('_withcal' if not remove_cal_pxl_resolve else '')+
                            ('' if grade_str_rsl is None else '_grade_'+grade_str_rsl.replace(':','to'))+ '_sp' +
                            ('_' + str(e_low_rsl) + '_' + str(e_high_rsl) if (e_low_rsl is not None and e_high_rsl is not None) else '')
                                                +elem_gti_str +'.ds'))

                        xsel_util(elem_evt.split('/')[-1],
                                  product_name,
                                  mode='spectrum',
                                  directory=anal_dir,
                                  region_str=elem_pixel_str if elem_pixel_str!='all' else None,
                                  grade_str=grade_str_rsl,
                                  remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                  e_low=e_low_rsl,
                                  e_high=e_high_rsl,
                                  spawn=bashproc,
                                  gti_file=elem_gti_file,image_mode=image_mode)

                    if e_low_rsl is not None and e_high_rsl is not None:
                        #creating a copy of the spectrum with only the right bins to be compatible with the cut rmf and arf

                        cut_sp_path=os.path.join(anal_dir,product_name.replace('.ds','_cut.ds'))
                        os.system('cp '+os.path.join(anal_dir,product_name)+' '+
                                        cut_sp_path)
                        with fits.open(cut_sp_path,mode='update') as cut_sp_file:

                            pha_cutoff_low=kev_to_PI(e_low_rsl,instru='resolve')
                            pha_cutoff_high=kev_to_PI(e_high_rsl,instru='resolve')
                            cut_sp_file[1].data=cut_sp_file[1].data[pha_cutoff_low:pha_cutoff_high]
                            cut_sp_file[1].header['DETCHANS']=pha_cutoff_high-pha_cutoff_low
                            cut_sp_file[1].header['TLMIN1']=pha_cutoff_low
                            cut_sp_file[1].header['TLMAX1']=pha_cutoff_high-1
                            cut_sp_file[1].header['NAXIS2']=pha_cutoff_high-pha_cutoff_low

                            cut_sp_file.flush()

def xtd_mkrmf(infile,outfile,
              rmfparam=None,
              eminin=None,
              dein=None,
              nchanin=None,
              eminout=None,
              deout=None,
              nchanout=None,
              overwrite=True,
              spawn=None,
              heasoft_init_alias='heainit',caldb_init_alias='caldbinit'):

    '''
    simple wrapper around rsl_mkrmf to have arguments

    The chatter=2 is fixed at two to be able to get a line with a fixed information for pexpect

    the regmode argument is required but not used since regfile is set to None

    see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslmkrmf.html
    '''

    if spawn is None:

        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')

        set_var(spawn_use, heasoft_init_alias, caldb_init_alias)

        spawn_use.logfile_read = sys.stdout

        spawn_use.sendline('cd ' + os.getcwd())
    else:
        spawn_use = spawn

    spawn_use.sendline('punlearn xtdrmf')

    spawn_use.sendline('xtdrmf '+
                       ' infile='+infile+
                       ' outfile='+outfile+
                       ('' if rmfparam is None else ' rmfparam='+rmfparam)+
                       ('' if eminin is None else ' eminin='+str(eminin))+
                       ('' if dein is None else ' dein=' + str(dein)) +
                       ('' if nchanin is None else ' nchanin=' + str(nchanin)) +
                       ('' if eminout is None else ' eminout=' + str(eminout)) +
                       ('' if deout is None else ' deout=' + str(deout)) +
                       ('' if nchanout is None else ' nchanout=' + str(nchanout))+
                        ' chatter=2'+
                       (' clobber=YES' if overwrite else ''))

    spawn_use.expect('INFO: Finished calculating response matrix, closing RMF file',timeout=60)

    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

def rsl_mkrmf(whichrmf,infile,outfileroot,
              pixlist='0-35',
              resolist='0',
              eminin=0.,dein=0.5,nchanin=60000,
             useingrd=True,
                regionfile='NONE',
                spawn=None,
              overwrite=True,
              splitrmf=True,
              splitcomb=True,
              heasoft_init_alias='heainit',caldb_init_alias='caldbinit'):

    '''
    simple wrapper around rsl_mkrmf to have arguments

    see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslmkrmf.html

    The chatter=2 is fixed at two to be able to get a line with a fixed information for pexpect

    '''

    if spawn is None:

        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')

        set_var(spawn_use, heasoft_init_alias, caldb_init_alias)

        spawn_use.logfile_read = sys.stdout

        spawn_use.sendline('cd ' + os.getcwd())
    else:
        spawn_use=spawn

    spawn_use.sendline('punlearn rslmkrmf')

    spawn_use.sendline('rslmkrmf ' +
                      ' infile=' + infile +
    ' outfileroot=' + outfileroot +
    ' regionfile='+regionfile+
    ' regmode=DET'
    ' pixlist="' + pixlist+ '"' +
    ' whichrmf=' + whichrmf +
    ' resolist=' + resolist +
    ' eminin=' + str(eminin) +
    ' dein=' + str(dein) +
    ' nchanin=' + str(nchanin)+
    ' useingrd='+str(useingrd)+
    ' splitrmf='+("yes" if splitcomb else "no")+
    ' splitcomb='+("yes" if splitcomb else "no")+
    ' chatter=2 '+
                       (' clobber=YES' if overwrite else ''))

    spawn_use.expect('Finished',timeout=None)

    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

def extract_rmf(directory='auto_repro',instru='all',rmf_subdir='sp',
                #resolve options
                rmf_type_rsl='X',pixel_str_rsl='branch_filter',rsl_rmf_grade='0,1',
                split_rmf_rsl=True,
                comb_rmf_rsl=True,
                remove_cal_pxl_resolve=True,
                # resolve grid
                # eminin_rsl=300,dein_rsl=0.5,nchanin_rsl=23400,
                eminin_rsl=0, dein_rsl=0.5, nchanin_rsl=60000,
                useingrd_rsl=True,
                e_band_evt_rsl_rmf='2-12',
                #xtend grid
                eminin_xtd=200.,dein_xtd='"2,24"',nchanin_xtd='"5900,500"',
                eminout_xtd=0.,deout_xtd=6,nchanout_xtd=4096,
                #general event options and gti selection
                use_raw_evt_rsl=False,use_raw_evt_xtd=False,
                gti=None, gti_subdir='gti',
                #common arguments
                anal_dir_suffix='', heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                parallel=False,repro_suffix='repro'):

    '''

    Extract rmf from event files in the analysis subdirectory of a directory

    Instru determines whether the analysis is run on all or a single instrument

    To gain time, for now,  we consider a single Resolve RMF no matter the GTI

    emin_rsl: minimum energy of the resolve input grid
    dein_rsl: energy step of the resolve input grid
    nchanin_rsl: number of channels of the resolve input grid

    the default values are made to cover a 0.3-12 keV band with the default step of 0.5eV

    emin_xtd: minimum energy of the resolve input grid
    dein_xtd: energy step of the resolve input grid
    nchanin_xtd: number of channels of the resolve input grid

    eminout,deout,nchanout: the same but or the rmf ebounds extension
    for resolve, ignored if useingrd=yes

    e_band_evt_rsl_rmf: "X-Y" keV or None
        if not None, creates a custom event file with restricted energy band to use as input for the rmf creation
        useful to mitigate the influence of the LS events on the rmf normalization
        see https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/workshops/doc_feb25/1_6_SDC.pdf p. 10

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    the gti files are not used directly but determine which spectra are used

    pixel_str_rsl:
        pixel filtering list for xrism
            if set to branch_filter, excludes the pixels listed in the branch_filter.txt file of
            the observation, made by resolve_BR
            otherwise, manual input:

            example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
            also accepts pixels to exclude, such as '-(10:14,28,32)'

            If different configurations, join sets with +
            if one element of the set is all, the keyword is ignored

            products are named according to their pixel combination if there is more than one pixel_str

    rsl_rmf_grade:the event grade of the arf. Different syntax so no ":" in the string
                    (see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslmkrmf.html)
                    To use several grades, if the event file is made from several, use '0,1,...'  instead
                    note that the fil will be written using '_' to avoid issues

    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    e_low/e_high: if not None, the energy bounds of the spectra
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    #xtend input files are spectra, not event files
    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**'),recursive=True) if 'xtd_' in elem.split('/')[-1] and
                   '_sp' in elem and '_src_' in elem and '_grp_' not in elem and
                   (1 if use_raw_evt_xtd else '_SFP' in elem) and elem.endswith('.ds')]

    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str = str(int(time.time()))

    log_path=os.path.join(log_dir,'extract_rmf'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')
                      +'_'+rmf_subdir+'_'+gti_subdir+time_str + '.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):


        if not parallel:
            bashproc.logfile_read = sys.stdout

        bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

        if gti==None:
            gti_str_arr=['']
            gti_files_arr=[None]
        else:
            if gti=='all':
                gti_files_arr=[elem.replace(anal_dir,'.')
                               for elem in glob.glob(os.path.join(anal_dir,gti_subdir)+'/**')
                               if elem.endswith('.gti')]
                gti_str_arr=[elem[elem.rfind('_gti_'):].split('.')[0] for elem in gti_files_arr]
            else:
                gti_files_arr=gti
                gti_str_arr='_'+gti.split('/')[-1]

        for elem_gti_str,elem_gti_file in zip(gti_str_arr,gti_files_arr):

            for elem_sp in [elem for elem in xtend_files if elem_gti_str in elem]:

                        #here the elem_sp already selects them according to the current gti
                        product_name = elem_sp.replace(anal_dir,'.').replace('.ds','.rmf')

                        xtd_mkrmf(infile=elem_sp.replace(anal_dir,'.'),
                                  outfile=product_name,
                                  eminin=eminin_xtd,
                                  dein=dein_xtd,
                                  nchanin=nchanin_xtd,
                                  eminout=eminout_xtd,
                                  deout=deout_xtd,
                                  nchanout=nchanout_xtd,
                                  spawn=bashproc,
                                  heasoft_init_alias=heasoft_init_alias,
                                  caldb_init_alias=caldb_init_alias)


            for elem_evt in resolve_files:

                if pixel_str_rsl.startswith('branch_filter'):
                    # reading the branch filter file
                    with open(elem_evt.replace('.evt','_'+pixel_str_rsl+'.txt')) as branch_f:
                        branch_lines = branch_f.readlines()
                    branch_filter_line = [elem for elem in branch_lines if not elem.startswith('#')][0]
                    # reformatting the string
                    pixel_str_rsl_use = '-(' + branch_filter_line[1:-2] + ')'
                else:
                    pixel_str_rsl_use = pixel_str_rsl

                for elem_pixel_str in pixel_str_rsl_use.split('+'):

                    reg_str = pixel_str_rsl if pixel_str_rsl.startswith('branch_filter') else elem_pixel_str

                    product_root = os.path.join(rmf_subdir,elem_evt.replace(anal_dir,'.').replace('.evt',
                            '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                            ('_withcal' if not remove_cal_pxl_resolve else '')+
                        '_grade_'+rsl_rmf_grade.replace(',','and')+ '_rmf' +
                        ('_evtbase_'+e_band_evt_rsl_rmf.replace('-','_').replace('.','p') if e_band_evt_rsl_rmf is not None else '')+
                        ('_' + str(eminin_rsl).replace('.','')+ '_' + str(dein_rsl).replace('.','')+'_'+str(nchanin_rsl))
                                                                   +elem_gti_str))

                    infile_use=elem_evt.replace(anal_dir,'.')

                    if e_band_evt_rsl_rmf is not None:

                        #creating an event file for a restricted energy band as the rmf input
                        infile_use=elem_evt.replace(anal_dir,'.').replace('.evt',
                                                                            '_'+e_band_evt_rsl_rmf+'.evt')

                        xsel_util(elem_evt.split('/')[-1],
                                  save_path=infile_use.split('/')[-1],
                                    directory=anal_dir,region_str=elem_pixel_str,
                                              mode='event',
                                              e_low=float(e_band_evt_rsl_rmf.split('-')[0]),
                                              e_high=float(e_band_evt_rsl_rmf.split('-')[1]),
                                              spawn=bashproc)

                    rsl_mkrmf(infile=infile_use,
                              outfileroot=product_root,
                              pixlist=rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                      mode='rmf'),
                              whichrmf=rmf_type_rsl,
                              resolist=rsl_rmf_grade,
                              eminin=eminin_rsl,
                              dein=dein_rsl,
                              splitrmf=split_rmf_rsl,
                              splitcomb=comb_rmf_rsl,
                              nchanin=nchanin_rsl,
                              useingrd=useingrd_rsl,
                              spawn=bashproc,heasoft_init_alias=heasoft_init_alias,caldb_init_alias=caldb_init_alias)

def create_expo(anal_dir,instrument,evt_file,gti_file,directory='auto_repro',out_file='auto',
                spawn=None,heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                delta=20,numphi=1,repro_suffix='repro'):

    '''
    Wrapper to compute an exposure map for a given instrument
    Should be run ontop of the directory, which means that the evtfile and gtifile should be paths starting from
    the observation directory

    the default delta and numphi values are taken from
    e.g. https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/quickstart/xrism_quick_start_guide_v2p3_240918a.pdf
    or https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/xrism_abc.pdf
    and are not the function defaults

    if no gti cut is desired, gtifile should be the event file instead
    '''

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    files_dir=glob.glob(directory_use+'/**',recursive=True)

    #note: here the paths to the ehk, pixgti,...file must be adapted to come out of the analysis directory first

    anal_dir_out_prefix=('/').join(np.repeat(['..'],len(anal_dir.split('/'))).tolist())
    ehk_file=[os.path.join(anal_dir_out_prefix,elem) for elem in files_dir if '.ehk' in elem][0]

    #the gpg condition is to avoid the first file being crypted. we don't put an endswith('.gti') because
    #the file may or may not still be compressed
    if instrument=='resolve':
        badimg_file='NONE'

        pixgti_file=[os.path.join(anal_dir_out_prefix,elem) for elem in files_dir if '_exp.gti' in elem
                     and not elem.endswith('.gpg') and 'px'+evt_file.split('/')[-1].split('_')[1].split('px')[1] in elem][0]

    elif instrument=='xtend':
        #note: the second condition is there to ensure the badimg file corresponds
        # to the xtend configuration of the evt
        badimg_file=[os.path.join(anal_dir_out_prefix,elem) for elem in files_dir if '.bimg' in elem
                     and not elem.endswith('.gpg') and evt_file.split('/')[-1].split('_')[1] in elem][0]

        pixgti_file=[os.path.join(anal_dir_out_prefix,elem) for elem in files_dir if '.fpix' in elem
                     and not elem.endswith('.gpg') and evt_file.split('/')[-1].split('_')[1].replace('p','')
                     and 'event_uf' in elem][0]

    if gti_file is None:
        gti_file_use=evt_file
    else:
        gti_file_use=gti_file

    if out_file=='auto':
        out_file_path=gti_file.replace('.evt','.expo').replace('.gti','.expo')
    else:
        out_file_path=out_file

    if spawn is None:
        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')
        spawn_use.logfile_read = sys.stdout
        set_var(spawn_use,heasoft_init_alias,caldb_init_alias)
        spawn_use.sendline('cd '+os.getcwd())
        spawn_use.sendline('cd '+anal_dir)
    else:
        spawn_use=spawn

    spawn_use.sendline('punlearn xaexpmap')

    spawn_use.sendline('xaexpmap '+
                       ' ehkfile='+ehk_file+
                       ' gtifile='+gti_file_use+
                       ' instrume='+instrument+
                       ' badimgfile='+badimg_file+
                       ' pixgtifile='+pixgti_file+
                       ' outfile='+out_file_path+
                       ' outmaptype=EXPOSURE'+
                       ' delta='+str(delta)+
                       ' numphi='+str(numphi)+
                       ' clobber=YES'+
                       ' chatter=2')

    spawn_use.expect('Closed output FITS file',timeout=60)

    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

    return out_file_path

def create_arf(instrument,out_rtfile,source_ra,source_dec,emap_file,out_file,
               region_file,rmf_file,
               source_type='POINT',flatradius="3",image=None,
               e_low=0.3,e_high=12.0,
                numphoton=300000,minphoton=100,
                telescope='XRISM',
                spawn=None,heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
               e_low_image=0.0, e_high_image=0.0,
               reg_mode='DET'):

    '''
    Wrapper around the heasoft xaarfgen to compute an arf for a given instrument

    see https://heasarc.gsfc.nasa.gov/lheasoft/help/xaarfgen.html

    regmode is currently locked to DET because that's the only region type currently accepted

    source_ra and source_dec should be decimal degree values

    e_low_image and e_high_image are the energy bounds of the image created if source_type is set to IMAGE

    There is sometimes a bug with the previous functions that remove the gti column out of the RTS xrism file.
    If crash with the resolve event file when not using gtis, may need to rerun resolve_RTS.
    '''

    if spawn is None:
        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')
        spawn_use.logfile_read = sys.stdout
        set_var(spawn_use,heasoft_init_alias,caldb_init_alias)
        spawn_use.sendline('cd '+os.getcwd())
    else:
        spawn_use=spawn

    spawn_use.sendline('punlearn xaarfgen')

    spawn_use.sendline('xaarfgen '+
                       ' xrtevtfile='+out_rtfile+
                       ' source_ra='+str(source_ra)+''
                       ' source_dec='+str(source_dec)+''+
                       ' instrume='+instrument+
                       ' emapfile='+emap_file+
                       ' regmode='+reg_mode+
                       ' regionfile='+region_file+
                       ' sourcetype='+source_type+
                     (' flatradius='+str(flatradius) if source_type=='FLATCIRCLE' else '')+
                      ' imgfile='+ (image if source_type == 'IMAGE' else 'NONE') +
                       ' rmffile='+rmf_file+
                       ' erange="'+("NONE" if e_low is None else str(e_low))+' '+("NONE" if e_high is None else str(e_high))+' '+str(e_low_image)+' '+str(e_high_image)+'"'+
                       ' numphoton='+str(numphoton)+
                       ' minphoton='+str(minphoton)+
                       ' outfile='+out_file+
                       ' clobber=YES'+
                       ' telescop=XRISM'+
                       ' qefile=CALDB'+
                       ' contamifile=CALDB'+
                       ' gatevalvefile=CALDB'+
                       ' onaxisffile=CALDB'+
                       ' onaxiscfile=CALDB'+
                       ' mirrorfile=CALDB'+
                       ' obstructfile=CALDB'+
                       ' frontreffile=CALDB'+
                       ' backreffile=CALDB'+
                       ' pcolreffile=CALDB'+
                       ' scatterfile=CALDB'+
                       ' chatter=3')

    out_code=spawn_use.expect(['Error during subroutine finalize','xaxmaarfgen: Fraction of PSF inside Region'],
                     timeout=None)

    if out_code==0:
        print('Error during xaarfgen computation')
        raise ValueError

    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

def extract_arf(directory='auto_repro',anal_dir_suffix='',on_axis_check=None,arf_subdir='sp',
                source_coords='on-axis',
                source_name='auto',
                target_only=False,use_file_target=True,
                source_type='POINT',
                suffix='',
                flatradius=3,
                arf_image=None,
                instru='all',use_comb_rmf_rsl=True,
                use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                region_src_xtd='auto', region_bg_xtd='auto',
                reg_mode='DET',
                pixel_str_rsl='branch_filter', grade_str_rsl='0:1',
                remove_cal_pxl_resolve=True,
                gti=None, gti_subdir='gti',skip_gti_emap=True,
                # e_low_rsl=None, e_high_rsl=None,
                e_low_rsl=0.3, e_high_rsl=12.0,
                #default values in hte pipeline
                e_low_xtd=0.3, e_high_xtd=15.0,
                e_low_image=0.0, e_high_image=0.0,
                numphoton=300000,
                minphoton=100,
                heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                parallel=False,repro_suffix='repro'):

    '''

    center of cropped Sgr A east image (can be important for computations on Sag A east arf)
    ('17:45:43.1813', '-29:00:22.924')

    MAXI J1744-294:
    ('17:45:40.476', '-29:00:46.10')

    AXJ1745.6-2901:
    ['17:45:35.6400', '-29:01:33.888']

    Extract arf from event files in the analysis/arf_subdir of an observation directory

    the gtis used for computing the exposure maps and finding the rmf files

    Instru determines whether the analysis is run on all or a single instrument

    on_axis_check checks whether the region centroids match the source center (to be implemented)

    source_type: POINT or FLATCIRCLE
    for source_type=POINT
        source_coord:
            -on-axis:
                assumes the coordinates of the source (aka the pointing is close enough to being on-axis)
            -an array:
                takes the values provided manually.
                Converts string values (assumed as sexadecimal), take float values directly

        source_name (used if source_coord is set to on-axis):
            -auto: fetches on Simbad the source matching the name of the directory directly above the
                    obsid directory
            -anything else is given to simbad directly
    for source_type=FLARTCIRCLE
        flaradius: the flat radius of the circle in arcminutes
    for source_type=IMAGE
        arf_image: the image to be used to make the arf.
        e_low_image/e_high_image: lower and upper energy bounds of the image

    -reg_mode:
        DET for detector (standard) or RADEC for sky
    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts


    on axis check performs a check to compare the position of the image center to that of the source
    pixel_str_rsl:
        pixel filtering list for xrism
            if set to branch_filter, excludes the pixels listed in the branch_filter.txt file of
            the observation, made by resolve_BR
            otherwise, manual input:

            example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
            also accepts pixels to exclude, such as '-(10:14,28,32)'

            If different configurations, join sets with +
            if one element of the set is all, the keyword is ignored

            products are named according to their pixel combination if there is more than one pixel_str

    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    e_low/e_high: if not None, the energy bounds of the arf
    '''

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))


    os.system('mkdir -p '+os.path.join(anal_dir,arf_subdir))

    if arf_image is not None:
        os.system('cp '+arf_image+' '+anal_dir)

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl' + ('' if use_raw_evt_xtd else '_SFP') + '.evt')]

    #here using any event file should work
    if len(resolve_files+xtend_files)==0:
        print('No event file satisfy the requirements. Skipping...')
        return 0
    else:

        if type(source_coords)==str and source_coords=='on-axis':
            any_event=(resolve_files+xtend_files)[0]
            if source_name == 'auto':

                obj_auto = source_catal(bashproc, './', any_event,
                                        target_only=target_only,
                                        use_file_target=use_file_target)

            else:
                obj_auto = Simbad.query_object(source_name)[0]

            try:
                source_ra, source_dec = sexa2deg([obj_auto['DEC'], obj_auto['RA']])[::-1]
            except:
                source_ra, source_dec = obj_auto['ra'],obj_auto['dec']

        else:
            if type(source_coords[0])==str:

                obj_deg=sexa2deg([source_coords[1].replace(' ',':'),source_coords[0].replace(' ',':')])[::-1]
            else:
                obj_deg=source_coords
            source_ra,source_dec=obj_deg


    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]

    if os.path.isfile('~/pfiles/xaarfgen.par'):
        os.system('rm ~/pfiles/xaarfgen.par')
    if os.path.isfile('~/pfiles/xaxmaarfgen.par'):
        os.system('rm ~/pfiles/xaxmaarfgen.par')

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str = str(int(time.time()))

    log_path=os.path.join(log_dir,'extract_arf'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')
                      +'_'+arf_subdir+'_'+gti_subdir+time_str + '.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        if len(resolve_files) != 0:
            # for removing the calibration sources
            os.system('cp $HEADAS/refdata/region_RSL_det.reg ' + anal_dir)

        bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

        if gti==None:
            gti_str_arr=['']
            gti_files_arr=[None]
        else:
            if gti=='all':
                gti_files_arr=[elem.replace(anal_dir,'.')
                               for elem in glob.glob(os.path.join(anal_dir,gti_subdir)+'/**')
                               if elem.endswith('.gti')]
                gti_str_arr=[elem[elem.rfind('_gti_'):].split('.')[0] for elem in gti_files_arr]
            else:
                gti_files_arr=gti
                gti_str_arr='_'+gti.split('/')[-1]

        for elem_gti_str,elem_gti_file in zip(gti_str_arr,gti_files_arr):

            for elem_evt in resolve_files + xtend_files:

                xtd_mode='xtd_' in elem_evt

                #building the exposure map
                expo_path=create_expo(anal_dir,
                                      instrument='xtend' if xtd_mode else 'resolve',
                            evt_file=elem_evt.replace(anal_dir,'.'),
                            gti_file=elem_evt.replace(anal_dir,'.') if elem_gti_file is None or skip_gti_emap else
                                     elem_gti_file.replace(anal_dir,'.'))

                if xtd_mode:

                    if region_src_xtd=='auto':
                        region_src_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_src_reg.reg'))
                    elif region_src_xtd is None:
                        region_src_xtd_use=None
                    else:
                        region_src_xtd_use=os.path.join(os.getcwd(),anal_dir,region_src_xtd)

                    if region_bg_xtd=='auto':
                        region_bg_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_bg_reg.reg'))
                    elif region_bg_xtd is None:
                        region_bg_xtd_use=None
                    else:
                        region_bg_xtd_use=os.path.join(os.getcwd(),anal_dir,region_bg_xtd)

                    for i_reg,elem_region in enumerate([region_src_xtd_use,region_bg_xtd_use]):

                        if elem_region is None:
                            print('Skipping '+('source' if i_reg==0 else 'bg')+' region for event file '+elem_evt)
                            continue

                        if not os.path.isfile(elem_region):
                            print('No matching region found. Skipping '+('source' if i_reg==0 else 'bg')+
                                  ' region for event file '+elem_evt)
                            continue

                        reg_str=('_auto_src' if region_src_xtd=='auto' else region_src_xtd.split('.')[0]) if i_reg==0 else \
                                ('_auto_bg' if region_bg_xtd == 'auto' else region_bg_xtd.split('.')[0])

                        # fetching the matching rmf
                        rmf_list = \
                        [elem for elem in glob.glob(os.path.join(anal_dir,'**'), recursive=True) if elem.endswith('.rmf')
                         and ('xtd_') in elem and elem_evt.split('/')[-1].split('_')[1] in elem
                         and reg_str in elem and '/'+arf_subdir+'/' in elem]

                        if len(rmf_list)==0:
                            print('No rmf found for xtend event '+elem_evt)
                            print('Skipping arf computation...')
                            continue

                        #removing the rtfile to ensure we recreate it
                        if os.path.isfile(elem_evt.replace('.evt','_raytracing.evt')):
                            os.remove(elem_evt.replace('.evt','_raytracing.evt'))

                        if gti is not None:
                            print('should think about whether this is done properly')
                            breakpoint()

                        if len(rmf_list)>1:
                            print(rmf_list)
                            rmf_id=input('Give index of rmf to use')
                        else:
                            rmf_id=0
                        rmf_path=rmf_list[rmf_id]

                        out_name = rmf_path.replace('.rmf',
                                                    '_' + source_type
                                                    + ('_fr' + str(flatradius).replace('.', 'p')
                                                          if source_type == 'FLATCIRCLE' else '')
                                                    + ('_img_' + arf_image[:arf_image.rfind('.')].split('/')[-1]
                                                          if source_type == 'IMAGE' else '')
                                                    + ('_' + suffix if suffix != '' else '')
                                                                        + '.arf').replace(
                                                        anal_dir, '.').replace(',','and')

                        create_arf(instrument='xtend',
                                   out_rtfile=elem_evt.replace(anal_dir,'.').replace('.evt','_raytracing.evt'),
                                   #need to replace the ',' with '_' because it splits the file otherwise
                                   out_file=out_name,
                                   source_ra=source_ra,
                                   source_dec=source_dec,
                                   emap_file=os.path.join(os.getcwd(),anal_dir,expo_path.split('/')[-1]),
                                   region_file=elem_region,
                                   rmf_file=rmf_path.replace(anal_dir,'.'),
                                   source_type=source_type,
                                   flatradius=flatradius,
                                   e_low=e_low_xtd,e_high=e_high_xtd,
                                   numphoton=numphoton,
                                   minphoton=minphoton,
                                   spawn=bashproc,
                                   image=arf_image,
                                   e_low_image=e_low_image,
                                   e_high_image=e_high_image,
                                   reg_mode=reg_mode)

                else:

                    if pixel_str_rsl.startswith('branch_filter'):
                        # reading the branch filter file
                        with open(elem_evt.replace('.evt','_'+pixel_str_rsl+'.txt')) as branch_f:
                            branch_lines=branch_f.readlines()
                        branch_filter_line=[elem for elem in branch_lines if not elem.startswith('#')][0]
                        #reformatting the string
                        pixel_str_rsl_use='-('+branch_filter_line[1:-2]+')'
                    else:
                        pixel_str_rsl_use=pixel_str_rsl

                    for elem_pixel_str in pixel_str_rsl_use.split('+'):

                        reg_str = pixel_str_rsl if pixel_str_rsl.startswith('branch_filter') else pixel_str_rsl_use


                        product_root = elem_evt.split('/')[-1].replace('.evt',
                                '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                                ('_withcal' if not remove_cal_pxl_resolve else ''))

                        rmf_list=[elem for elem in glob.glob(os.path.join(anal_dir,'**'), recursive=True) if
                                  product_root in elem and elem.endswith('.rmf') and \
                                  ('_comb' in elem if use_comb_rmf_rsl else '_comb' not in elem and '_elc' not in elem)
                                  and '/'+arf_subdir+'/' in elem]
                        #for now we consider a single Resolve RMF no matter the GTI,

                        #getting the name
                        region_name=rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                    mode='rmf')

                        region_path=os.path.join(anal_dir,'resolve_pixels_'+region_name.replace(',','_')+'.reg')
                        #creating the file
                        rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                 region_path=region_path)

                        #removing the rtfile to ensure we recreate it
                        if os.path.isfile(elem_evt.replace('.evt','_raytracing.evt')):
                            os.remove(elem_evt.replace('.evt','_raytracing.evt'))

                        if len(rmf_list)>1:
                            for id_elem,elem in enumerate(rmf_list):
                                print('\n')
                                print(id_elem,elem)

                            rmf_id=input('Give index of rmf to use')
                        else:
                            rmf_id=0
                        rmf_path=rmf_list[int(rmf_id)]


                        out_name= rmf_path.replace('.rmf',
                                                    '_' + source_type
                                                   + ('_fr' + str(flatradius).replace('.', 'p')
                                                               if source_type == 'FLATCIRCLE' else '')
                                                   + ('_img_' + arf_image[:arf_image.rfind('.')].split('/')[-1]
                                                      if source_type == 'IMAGE' else '')
                                                   + ('_' + suffix if suffix != '' else '')
                                                   + '.arf').replace(
                                                        anal_dir, '.').replace(',','and')


                        create_arf(instrument='resolve',
                                   out_rtfile=elem_evt.replace(anal_dir,'.').replace('.evt','_raytracing.evt'),
                                   out_file=out_name,
                                   source_ra=source_ra,
                                   source_dec=source_dec,
                                   emap_file=os.path.join(os.getcwd(),anal_dir,expo_path),
                                   region_file=region_path.replace(anal_dir,'.'),
                                   rmf_file=rmf_path.replace(anal_dir,'.'),
                                   source_type=source_type,
                                   flatradius=flatradius,
                                   e_low=e_low_rsl, e_high=e_high_rsl,
                                   numphoton=numphoton,
                                   minphoton=minphoton,
                                   spawn=bashproc,
                                   image=arf_image,
                                   e_low_image=e_low_image,
                                   e_high_image=e_high_image,
                                   reg_mode=reg_mode)



def rsl_mkrsp(

              #specific arguments
              inevtfile,
              outfileroot,
              pixlist,
              includels=True,
              gfelo=2.0,
              gfehi=12.0,

              #arf arguments
              out_rtfile='', source_ra='', source_dec='', emap_file='',
              source_type='POINT', flatradius="3", image=None,
              e_low=0.3, e_high=12.0,
              numphoton=300000, minphoton=100,
              e_low_image=0.0, e_high_image=0.0,
              spawn=None, heasoft_init_alias='heainit', caldb_init_alias='caldbinit',

              #rmf arguments
              whichrmf='XL',
              resolist='0:1',
              eminin=0., dein=0.5, nchanin=60000,
              useingrd=True,
              regionfile='NONE',
              overwrite=True,
              splitrmf=True,
              splitcomb=True,
              ):

    '''
    Wrapper around the heasoft xaarfgen to compute an arf for a given instrument, available staring in Heasoft 6.36

    for details see file $HEADAS/hitomixrism/x86_64-pc-linux-gnu-libc2.41/help/rslmkrsp.html (not available online yet)

    '''

    '''
    simple wrapper around rsl_mkrmf to have arguments

    see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslmkrmf.html

    The chatter=2 is fixed at two to be able to get a line with a fixed information for pexpect

    '''

    if spawn is None:

        spawn_use = pexpect.spawn("/bin/bash", encoding='utf-8')

        set_var(spawn_use, heasoft_init_alias, caldb_init_alias)

        spawn_use.logfile_read = sys.stdout

        spawn_use.sendline('cd ' + os.getcwd())
    else:
        spawn_use=spawn

    spawn_use.sendline('punlearn rslmkrsp')

    plop='rslmkrsp' +\
                      ' inevtfile=' + inevtfile +\
    ' outfileroot=' + outfileroot +\
    ' includels=' +('yes' if includels else 'no')+\
    ' gfelo=' +str(gfelo)+\
    ' gfehi=' +str(gfehi)+\
                       ' splitrmf='+("yes" if splitrmf and whichrmf=='X' else "no")+\
    ' splitcomb='+("yes" if splitcomb and whichrmf=='X' else "no")+\
    ' resolist=' + resolist +\
    ' pixlist="' + pixlist+ '"' +\
    ' whichrmf=' + whichrmf +\
    ' eminin=' + str(eminin) +\
    ' dein=' + str(dein) +\
    ' nchanin=' + str(nchanin)+\
    ' useingrd='+str(useingrd)+\
                       ' xrtevtfile='+out_rtfile+\
                       ' source_ra='+str(source_ra)+''\
                       ' source_dec='+str(source_dec)+''+\
                       ' emapfile='+emap_file+\
                       ' sourcetype='+source_type+\
                     (' flatradius='+str(flatradius) if source_type=='FLATCIRCLE' else '')+\
                      ' imgfile='+ (image if source_type == 'IMAGE' else 'NONE') +\
                       ' erange="'+("NONE" if e_low is None else str(e_low))+' '+\
                                    ("NONE" if e_high is None else str(e_high))+\
                                    ' '+str(e_low_image)+' '+str(e_high_image)+'"'+\
                       ' numphoton='+str(numphoton)+\
                       ' minphoton='+str(minphoton)+\
                       ' regionfile=NONE'+\
    ' chatter=2 '+\
    ' qefile=CALDB'+\
    ' contamifile=CALDB'+\
    ' gatevalvefile=CALDB'+ \
     ' onaxisffile=CALDB' + \
    ' onaxiscfile=CALDB' +\
    ' mirrorfile=CALDB' +\
    ' obstructfile=CALDB' +\
    ' frontreffile=CALDB' +\
    ' backreffile=CALDB' +\
    ' pcolreffile=CALDB' +\
    ' scatterfile=CALDB'+\
    (' clobber=YES' if overwrite else '')

    spawn_use.sendline('rslmkrsp' +\
                      ' inevtfile=' + inevtfile +\
    ' outfileroot=' + outfileroot +
    ' includels=' +('yes' if includels else 'no')+
    ' gfelo=' +str(gfelo)+
    ' gfehi=' +str(gfehi)+
    #rmf
                       ' splitrmf=' + ("yes" if splitrmf and whichrmf == 'X' else "no") + \
                       ' splitcomb=' + ("yes" if splitcomb and whichrmf == 'X' else "no") + \
                       ' resolist="' + resolist +'"'+
    ' pixlist="' + pixlist+ '"' +
    ' whichrmf=' + whichrmf +
    ' eminin=' + str(eminin) +
    ' dein=' + str(dein) +
    ' nchanin=' + str(nchanin)+
    ' useingrd='+str(useingrd)+

                        #arf
                       ' xrtevtfile='+out_rtfile+
                       ' source_ra='+str(source_ra)+''
                       ' source_dec='+str(source_dec)+''+
                       ' emapfile='+emap_file+
                       ' sourcetype='+source_type+
                     (' flatradius='+str(flatradius) if source_type=='FLATCIRCLE' else '')+
                      ' imgfile='+ (image if source_type == 'IMAGE' else 'NONE') +
                       ' erange="'+("NONE" if e_low is None else str(e_low))+' '+
                                    ("NONE" if e_high is None else str(e_high))+
                                    ' '+str(e_low_image)+' '+str(e_high_image)+'"'+
                       ' numphoton='+str(numphoton)+
                       ' minphoton='+str(minphoton)+

                       ' regionfile=NONE'+\
    ' qefile=CALDB'+\
    ' contamifile=CALDB'+\
    ' gatevalvefile=CALDB'+ \
     ' onaxisffile=CALDB' + \
    ' onaxiscfile=CALDB' +\
    ' mirrorfile=CALDB' +\
    ' obstructfile=CALDB' +\
    ' frontreffile=CALDB' +\
    ' backreffile=CALDB' +\
    ' pcolreffile=CALDB' +\
    ' scatterfile=CALDB'+\

    ' chatter=3 '+
                       (' clobber=YES' if overwrite else ''))

    #note: the first one is at the start of the task, the second is at the end
    spawn_use.expect('Running rslmkrsp',timeout=None)
    spawn_use.expect('Running rslmkrsp',timeout=None)
    spawn_use.expect('Finished',timeout=None)


    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

def extract_rsp(directory='auto_repro',rsp_subdir='sp', anal_dir_suffix='',
                #rsp-specific arguments
                     grade_elow=2.0,
                     grade_ehigh=10.0,
                     include_ls=True,

                #resolve options
                rmf_type_rsl='X',pixel_str_rsl='branch_filter',grade='0,1',
                split_rmf_rsl=True,
                comb_rmf_rsl=True,
                remove_cal_pxl_resolve=True,
                # resolve grid
                # eminin_rsl=300,dein_rsl=0.5,nchanin_rsl=23400,
                eminin_rsl=0, dein_rsl=0.5, nchanin_rsl=60000,
                useingrd_rsl=True,

                #for now only works with this
                e_band_evt_rsl_rmf=None,


                #arf arguments
                     on_axis_check=None,source_coords='on-axis',
                     source_name='auto',
                     target_only=False, use_file_target=True,
                     source_type='POINT',
                     flatradius=3,
                     arf_image=None,
                    e_low_arf=0.3, e_high_arf=12.0,
                    e_low_image=0.0, e_high_image=0.0,
                     numphoton=300000,
                     minphoton=100,
                     skip_gti_emap=True,
                suffix='',
                 # general event options and gti selection
                 use_raw_evt_rsl=False, use_raw_evt_xtd=False,
                 gti=None, gti_subdir='gti',
                #common arguments
                heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                parallel=False,repro_suffix='repro',
                premade_expmap='',
                ):

    '''
    Combined task that produces both RMFs and ARFs for resolve with a better consideration for spatially variable
    grade fractions accross the FoV

    note that this task does not require creating regions for the arf, the pixel list is used directly

    gti consideration not yet implemented

    for now the arf file should be put as a direct file name and is assumed to be in the analysis directory

    premade_expmap:
        uses already computed expomap instead of remaking it. the path should be from the analysis directory
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)


    if os.path.isfile('~/pfiles/xaarfgen.par'):
        os.system('rm ~/pfiles/xaarfgen.par')
    if os.path.isfile('~/pfiles/xaxmaarfgen.par'):
        os.system('rm ~/pfiles/xaxmaarfgen.par')

    if directory=='auto_repro':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]+'_'+\
                      repro_suffix
    elif directory=='auto':
        #fetching the first obsid-like directory in the cwd
        directory_use=[elem[:-1] for elem in glob.glob('**/') if len(elem[:-1])==9 and elem[:-1].isdigit()][0]
    else:
        directory_use=directory

    anal_dir = os.path.join(directory_use, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    #here using any event file should work
    if len(resolve_files)==0:
        print('No event file satisfy the requirements. Skipping...')
        return 0
    else:

        if type(source_coords)==str and source_coords=='on-axis':
            any_event=(resolve_files)[0]
            if source_name == 'auto':

                obj_auto = source_catal(bashproc, './', any_event,
                                        target_only=target_only,
                                        use_file_target=use_file_target)

            else:
                obj_auto = Simbad.query_object(source_name)[0]

            try:
                source_ra, source_dec = sexa2deg([obj_auto['DEC'], obj_auto['RA']])[::-1]
            except:
                source_ra, source_dec = obj_auto['ra'],obj_auto['dec']

        else:
            if type(source_coords[0])==str:

                obj_deg=sexa2deg([source_coords[1].replace(' ',':'),source_coords[0].replace(' ',':')])[::-1]
            else:
                obj_deg=source_coords
            source_ra,source_dec=obj_deg

    log_dir=os.path.join(os.getcwd(),anal_dir,'log')
    os.system('mkdir -p '+os.path.join(os.getcwd(),anal_dir,'log'))

    time_str = str(int(time.time()))

    log_path=os.path.join(log_dir,'extract_rsp'+('_'+anal_dir_suffix if anal_dir_suffix!='' else '')
                      +'_'+rsp_subdir+'_'+gti_subdir+'_'+time_str + '.log')

    if os.path.isfile(log_path):
        os.system('rm ' + log_path)

    with (no_op_context() if parallel else StdoutTee(log_path, mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(log_path, buff=1, file_filters=[_remove_control_chars])):

        if len(resolve_files) != 0:
            # for removing the calibration sources
            os.system('cp $HEADAS/refdata/region_RSL_det.reg ' + anal_dir)

        if not parallel:
            bashproc.logfile_read = sys.stdout

        bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

        rsp_dir='rsp_'+time_str
        os.mkdir(os.path.join(anal_dir,rsp_dir))

        if gti==None:
            gti_str_arr=['']
            gti_files_arr=[None]
        else:
            if gti=='all':
                gti_files_arr=[elem.replace(anal_dir,'.')
                               for elem in glob.glob(os.path.join(anal_dir,gti_subdir)+'/**')
                               if elem.endswith('.gti')]
                gti_str_arr=[elem[elem.rfind('_gti_'):].split('.')[0] for elem in gti_files_arr]
            else:
                gti_files_arr=gti
                gti_str_arr='_'+gti.split('/')[-1]

        for elem_gti_str,elem_gti_file in zip(gti_str_arr,gti_files_arr):

            for elem_evt in resolve_files:

                gti_path_forexpo=elem_evt.replace(anal_dir,'.')\
                                    if elem_gti_file is None or skip_gti_emap else \
                                     elem_gti_file.replace(anal_dir,'.')

                #building the exposure map

                if premade_expmap!='':
                    expo_path=os.path.join('..',premade_expmap)
                else:
                    expo_path=create_expo(os.path.join(anal_dir,rsp_dir),
                                      instrument='resolve',
                            evt_file=os.path.join('..',elem_evt.replace(anal_dir,'.')),
                            gti_file=os.path.join('..',gti_path_forexpo),
                            out_file=gti_path_forexpo.replace('../','').\
                                    replace('.evt','.expo').replace('.gti','.expo'))

                if pixel_str_rsl.startswith('branch_filter'):
                    # reading the branch filter file
                    with open(elem_evt.replace('.evt','_'+pixel_str_rsl+'.txt')) as branch_f:
                        branch_lines = branch_f.readlines()
                    branch_filter_line = [elem for elem in branch_lines if not elem.startswith('#')][0]
                    # reformatting the string
                    pixel_str_rsl_use = '-(' + branch_filter_line[1:-2] + ')'
                else:
                    pixel_str_rsl_use = pixel_str_rsl

                for elem_pixel_str in pixel_str_rsl_use.split('+'):

                    reg_str = pixel_str_rsl if pixel_str_rsl.startswith('branch_filter') else elem_pixel_str

                    product_root = elem_evt.replace(anal_dir,'.').replace('.evt',
                            '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                            ('_withcal' if not remove_cal_pxl_resolve else '')+
                        '_grade_'+grade.replace(',','and')+ '_rmf' +
                        ('_evtbase_'+e_band_evt_rsl_rmf.replace('-','_').replace('.','p') if e_band_evt_rsl_rmf is not None else '')+
                        ('_' + str(eminin_rsl).replace('.','')+ '_' + str(dein_rsl).replace('.','')+'_'+str(nchanin_rsl)
                                                                   +elem_gti_str)+'_' + source_type
                                                    + ('_fr' + str(flatradius).replace('.', 'p')
                                                          if source_type == 'FLATCIRCLE' else '')
                                                    + ('_img_' + arf_image[:arf_image.rfind('.')]
                                                          if source_type == 'IMAGE' else '')
                                                    +('_'+suffix if suffix!='' else ''))

                    infile_use=os.path.join('..',elem_evt.replace(anal_dir,'.'))

                    if e_band_evt_rsl_rmf is not None:

                        #creating an event file for a restricted energy band as the rmf input
                        infile_use=elem_evt.replace(anal_dir,'.').replace('.evt',
                                                                            '_'+e_band_evt_rsl_rmf+'.evt')

                        xsel_util(elem_evt.split('/')[-1],
                                  save_path=infile_use.split('/')[-1],
                                    directory=anal_dir,region_str=elem_pixel_str,
                                              mode='event',
                                              e_low=float(e_band_evt_rsl_rmf.split('-')[0]),
                                              e_high=float(e_band_evt_rsl_rmf.split('-')[1]),
                                              spawn=bashproc)

                    #no need now that we do things in separate folders
                    # #removing the rtfile to ensure we recreate it
                    # if os.path.isfile(elem_evt.replace('.evt','_raytracing.evt')):
                    #     os.remove(elem_evt.replace('.evt','_raytracing.evt'))

                    bashproc.sendline('cd ' + rsp_dir)

                    rsl_mkrsp(inevtfile=infile_use,
                              outfileroot=product_root,
                              pixlist=rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                      mode='rmf'),
                              includels=include_ls,
                              gfelo=grade_elow,
                              gfehi=grade_ehigh,
                              whichrmf=rmf_type_rsl,
                              resolist=grade,
                              eminin=eminin_rsl,
                              dein=dein_rsl,
                              splitrmf=split_rmf_rsl,
                              splitcomb=comb_rmf_rsl,
                              nchanin=nchanin_rsl,
                              useingrd=useingrd_rsl,

                              spawn=bashproc,heasoft_init_alias=heasoft_init_alias,caldb_init_alias=caldb_init_alias,

                                   out_rtfile=elem_evt.replace(anal_dir,'.').replace('.evt','_raytracing.evt'),
                                   source_ra=source_ra,
                                   source_dec=source_dec,
                                   emap_file=expo_path,
                                   source_type=source_type,
                                   flatradius=flatradius,
                                   e_low=e_low_arf, e_high=e_high_arf,
                                   numphoton=numphoton,
                                   minphoton=minphoton,
                                   image=(None if arf_image is None else os.path.join('..',arf_image)),
                                   e_low_image=e_low_image,
                                   e_high_image=e_high_image)


def cut_xrism_rmf(xrism_rmf_path,e_low_new,e_high_new):

    #I think this doesn't work because of the internal links between different energies, should investigate

    pass

def cut_xrism_arf(xrism_arf_path,e_low_new,e_high_new):

    '''This should work'''

    cut_arf_path =xrism_arf_path.replace('.arf','_cut_'\
                                                        +str(e_low_new).replace('.','p')+'_'
                                                        +str(e_high_new).replace('.','p')+'.arf')
    cut_arf_path=shutil.copyfile(xrism_arf_path,cut_arf_path)
    with fits.open(cut_arf_path,mode='update') as cut_arf:
        cut_arf[1].data=cut_arf[1].data[(cut_arf[1].data['ENERG_LO'] >= e_low_new)\
                                        & (cut_arf[1].data['ENERG_HI'] <= e_high_new)]

        cut_arf[1].header['NAXIS2']=len(cut_arf[1].data)

        cut_arf.flush()


def loop_anal_pix(pixlist='all',rmf_type='L',repro_suffix = 'reprocorr',
                  grades='0:1'):
    '''
    grades:
        directly translated with a , for the rmf so will be wrong if using intervals and not one or two
    '''

    if pixlist=='all':
        pixlist_use=range(36)



    for i in pixlist_use:
        if i==12:
            pass

        anal_suffix='pixindiv_'+str(i)

        init_anal(directory='auto_repro', anal_dir_suffix=anal_suffix,
                  resolve_filters='open', xtd_config='all', gz=False,
                  repro_suffix=repro_suffix)

        resolve_RTS(directory='auto_repro', anal_dir_suffix=anal_suffix,
                    heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                    parallel=False, repro_suffix=repro_suffix)

        resolve_BR(directory='auto_repro', anal_dir_suffix=anal_suffix,
                   use_raw_evt_rsl=False,
                   task='rslbratios',
                   lightcurves=False,
                   emin=2, emax=12,
                   remove_cal_pxl_resolve=False,
                   pixel_filter_rule='only_'+str(i),
                   heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                   parallel=False, repro_suffix=repro_suffix, plot_hp_sim_curve_band=True)

        extract_sp(directory='auto_repro', anal_dir_suffix=anal_suffix, sp_subdir='sp',
                   use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                   instru='resolve',
                   region_src_xtd='auto', region_bg_xtd='auto',
                   pixel_str_rsl='branch_filter', grade_str_rsl=grades,
                   remove_cal_pxl_resolve=True,
                   gti=None, gti_subdir='gti',
                   e_low_rsl=None, e_high_rsl=None,
                   e_low_xtd=None, e_high_xtd=None,
                   screen_reg=True, sudo_screen=False,
                   heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                   parallel=False, repro_suffix=repro_suffix)


        extract_rmf(directory='auto_repro', instru='resolve', rmf_subdir='sp',
                        # resolve options
                        rmf_type_rsl=rmf_type, pixel_str_rsl='branch_filter',
                    rsl_rmf_grade=grades.replace(':',','),
                        split_rmf_rsl=True,
                        comb_rmf_rsl=True,
                        remove_cal_pxl_resolve=True,
                        # resolve grid
                        # eminin_rsl=300,dein_rsl=0.5,nchanin_rsl=23400,
                        eminin_rsl=0, dein_rsl=0.5, nchanin_rsl=60000,
                        useingrd_rsl=True,
                        e_band_evt_rsl_rmf='2-12',
                        # xtend grid
                        eminin_xtd=200., dein_xtd='"2,24"', nchanin_xtd='"5900,500"',
                        eminout_xtd=0., deout_xtd=6, nchanout_xtd=4096,
                        # general event options and gti selection
                        use_raw_evt_rsl=False, use_raw_evt_xtd=False,
                        gti=None, gti_subdir='gti',
                        # common arguments
                        anal_dir_suffix=anal_suffix, heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                        parallel=False, repro_suffix=repro_suffix)

        # extract_arf(directory='auto_repro',anal_dir_suffix=anal_suffix,on_axis_check=None,arf_subdir='sp',
        #             source_coords=('17:45:40.476', '-29:00:46.10'),
        #             source_name='MAXIJ1744-294',
        #             target_only=False,use_file_target=True,
        #             suffix='MAXIJ1744',
        #             source_type='POINT',
        #             flatradius=3,
        #             arf_image=None,
        #             instru='resolve',use_comb_rmf_rsl=rmf_type=='X',
        #             use_raw_evt_xtd=False, use_raw_evt_rsl=False,
        #             region_src_xtd='auto', region_bg_xtd='auto',
        #             pixel_str_rsl='branch_filter', grade_str_rsl=grades,
        #             remove_cal_pxl_resolve=True,
        #             gti=None, gti_subdir='gti',skip_gti_emap=True,
        #             # e_low_rsl=None, e_high_rsl=None,
        #             e_low_rsl=0.3, e_high_rsl=12.0,
        #             #default values in hte pipeline
        #             e_low_xtd=0.3, e_high_xtd=15.0,
        #             e_low_image=0.0, e_high_image=0.0,
        #             numphoton=300000,
        #             minphoton=100,
        #             heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
        #             parallel=False,repro_suffix=repro_suffix)

        extract_arf(directory='auto_repro',anal_dir_suffix=anal_suffix,on_axis_check=None,arf_subdir='sp',
                    source_coords=['17 45 35.6400', '-29 01 33.888'],
                    source_name='AXJ1745.6-2901',
                    target_only=False,use_file_target=True,
                    suffix='AXJ1745',
                    source_type='POINT',
                    flatradius=3,
                    arf_image=None,
                    instru='resolve',use_comb_rmf_rsl=rmf_type=='X',
                    use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                    region_src_xtd='auto', region_bg_xtd='auto',
                    pixel_str_rsl='branch_filter', grade_str_rsl=grades,
                    remove_cal_pxl_resolve=True,
                    gti=None, gti_subdir='gti',skip_gti_emap=True,
                    # e_low_rsl=None, e_high_rsl=None,
                    e_low_rsl=0.3, e_high_rsl=12.0,
                    #default values in hte pipeline
                    e_low_xtd=0.3, e_high_xtd=15.0,
                    e_low_image=0.0, e_high_image=0.0,
                    numphoton=300000,
                    minphoton=100,
                    heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                    parallel=False,repro_suffix=repro_suffix)

        extract_arf(directory='auto_repro',anal_dir_suffix=anal_suffix,on_axis_check=None,arf_subdir='sp',
                    source_coords=('17:45:43.1813', '-29:00:22.924'),
                    source_name='diffuse',
                    target_only=False,use_file_target=True,
                    suffix='diffuse',
                    source_type='FLATCIRCLE',
                    flatradius=3,
                    instru='resolve',use_comb_rmf_rsl=rmf_type=='X',
                    use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                    region_src_xtd='auto', region_bg_xtd='auto',
                    pixel_str_rsl='branch_filter', grade_str_rsl=grades,
                    remove_cal_pxl_resolve=True,
                    gti=None, gti_subdir='gti',skip_gti_emap=True,
                    # e_low_rsl=None, e_high_rsl=None,
                    e_low_rsl=0.3, e_high_rsl=12.0,
                    #default values in hte pipeline
                    e_low_xtd=0.3, e_high_xtd=15.0,
                    e_low_image=0., e_high_image=0.,
                    numphoton=300000,
                    minphoton=100,
                    heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                    parallel=False,repro_suffix=repro_suffix)

        extract_arf(directory='auto_repro',anal_dir_suffix=anal_suffix,on_axis_check=None,arf_subdir='sp',
                    source_coords=('17:45:43.1813', '-29:00:22.924'),
                    source_name='SgrAEast',
                    target_only=False,use_file_target=True,
                    suffix='SgrAEast',
                    source_type='IMAGE',
                    flatradius=3,
                    arf_image='/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1744-294/Hideki/chandra_SgrAeast_FeKa25.img',
                    instru='resolve',use_comb_rmf_rsl=rmf_type=='X',
                    use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                    region_src_xtd='auto', region_bg_xtd='auto',
                    pixel_str_rsl='branch_filter', grade_str_rsl=grades,
                    remove_cal_pxl_resolve=True,
                    gti=None, gti_subdir='gti',skip_gti_emap=True,
                    # e_low_rsl=None, e_high_rsl=None,
                    e_low_rsl=0.3, e_high_rsl=12.0,
                    #default values in hte pipeline
                    e_low_xtd=0.3, e_high_xtd=15.0,
                    e_low_image=6.6, e_high_image=6.8,
                    numphoton=300000,
                    minphoton=100,
                    heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                    parallel=False,repro_suffix=repro_suffix)




def xaxmaarfgen_looper(rtfile,rmf,expmap,weight_dir_suffix='',pixels='all',heasoft_init_alias='heainit', caldb_init_alias='caldbinit'):

    '''
    looper for psf weights

    files should be relative paths from the analysis directory
    '''

    pixel_reg_string = {
        0: "box(4,3,1,1,0)",
        1: "box(6,3,1,1,0)",
        2: "box(5,3,1,1,0)",
        3: "box(6,2,1,1,0)",
        4: "box(5,2,1,1,0)",
        5: "box(6,1,1,1,0)",
        6: "box(5,1,1,1,0)",
        7: "box(4,2,1,1,0)",
        8: "box(4,1,1,1,0)",
        9: "box(1,3,1,1,0)",
        10: "box(2,3,1,1,0)",
        11: "box(1,2,1,1,0)",
        13: "box(2,2,1,1,0)",
        14: "box(2,1,1,1,0)",
        15: "box(3,2,1,1,0)",
        16: "box(3,1,1,1,0)",
        17: "box(3,3,1,1,0)",
        18: "box(3,4,1,1,0)",
        19: "box(1,4,1,1,0)",
        20: "box(2,4,1,1,0)",
        21: "box(1,5,1,1,0)",
        22: "box(2,5,1,1,0)",
        23: "box(1,6,1,1,0)",
        24: "box(2,6,1,1,0)",
        25: "box(3,5,1,1,0)",
        26: "box(3,6,1,1,0)",
        27: "box(6,4,1,1,0)",
        28: "box(5,4,1,1,0)",
        29: "box(6,5,1,1,0)",
        30: "box(6,6,1,1,0)",
        31: "box(5,5,1,1,0)",
        32: "box(5,6,1,1,0)",
        33: "box(4,5,1,1,0)",
        34: "box(4,6,1,1,0)",
        35: "box(4,4,1,1,0)"
    }

    currdir=os.getcwd()

    if pixels=='all':
        pix_list=[elem for elem in np.arange(36) if elem!=12]

    for elem_pix in pix_list:

        pixdir='arf_weights_'+str(weight_dir_suffix)+'/pixel_'+str(elem_pix)
        os.system('mkdir -p '+pixdir)

        #creating the region file
        reg_pix_name='region_pix_'+str(elem_pix)+'_DET.reg'
        with open(os.path.join(pixdir,reg_pix_name),'w+') as f:
            f.write(pixel_reg_string[elem_pix])

        bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

        set_var(bashproc, heasoft_init_alias, caldb_init_alias)

        bashproc.sendline('cd '+os.path.join(currdir,pixdir))

        bashproc.logfile_read = sys.stdout

        logfile='xaxmaarfgen_arf_forweights_'+str(elem_pix)+'.log'
        if os.path.isfile(os.path.join(pixdir,logfile)):
            os.path.join(pixdir,logfile)

        outfile='arf_forweights_pix'+str(elem_pix)+'.arf'
        bashproc.sendline('xaxmaarfgen '+
         ' xrtevtfile=../../' + rtfile +
         ' instrume=resolve'+
         ' emapfile=../../' + expmap +
         ' rmffile=../../' + rmf +
         ' regionfile=' + reg_pix_name +
         ' outfile='+outfile+
         ' clobber=YES' +
         ' telescop=XRISM' +
         ' qefile=CALDB' +
         ' contamifile=CALDB' +
         ' gatevalvefile=CALDB' +
         ' onaxisffile=CALDB' +
         ' onaxiscfile=CALDB' +
        #the ! allows to remake the file at every instance
         ' logfile=xaxmaarfgen_arf_forweights_'+str(elem_pix)+'.log'+
         ' chatter=3')

        insufficient_str='Insufficient number of raytrace photons in detector region: check coordinates and region files for errors before increasing'
        out_code=bashproc.expect(['Buffering '+outfile,insufficient_str])

        if out_code==1:
            #note: this will become inappropriate if the energy range is changed
            bashproc.expect('xaxmaarfgen: INFO:     12')

        time.sleep(1)
        bashproc.sendline('exit')
        time.sleep(1)


def PSF_frac_plot(raytrace_files=[],weight_dirs=[],e_min=2,e_max=10):

    '''
    plot the psf fractions of different sources for each pixel in the array, following one or several runs of
    xaxmaarfgen_looper

    raytrace files:
        the list of base raytrace files used for xaxmaarfgen_looper runs,
    weight_dirs:
        the directories where each run was performed

    e_min/e_max:
        energy band in which to compute the PSF fractions
        Note:
            heavily quantized in the arfs, available values are:
            0.3 0.4 0.5 1 2 3 4 5 6 7 8 9 10 11 12

    example:
    PSF_frac_plot(raytrace_files=['rsp_MAXIJ1744/xa901002010rsl_p0px1000_cl_RTS_raytracing.evt',
'rsp_AXJ1745/xa901002010rsl_p0px1000_cl_RTS_raytracing.evt',
'rsp_diffuse/xa901002010rsl_p0px1000_cl_RTS_raytracing.evt',
'rsp_SgrAEast/xa901002010rsl_p0px1000_cl_RTS_raytracing.evt'],
weight_dirs=['arf_weights_MAXIJ1744','arf_weights_AXJ1745','arf_weights_diffuse','arf_weights_SgrAEast'])

    '''

    #storing the PSF fraction normalization

    raytrace_tots=np.zeros(len(raytrace_files))
    for i_file,elem_rt in enumerate(raytrace_files):
        with fits.open(elem_rt) as rt_fits:
            raytrace_tots[i_file]=len(rt_fits[1].data[(rt_fits[1].data['energy']>=e_min) \
                                                    & (rt_fits[1].data['energy']<=e_max)])


    #storing the Pixel photon numbers per energy

    raytrace_pix_full=np.repeat(None,len(weight_dirs))
    for i_weights in range(len(weight_dirs)):
        pix_matr={}
        arf_weight_logs = np.array(glob.glob(os.path.join(weight_dirs[i_weights],'**/xaxmaarfgen_arf_forweights**.log'), recursive=True))
        arf_weight_logs.sort()

        for elem_log in arf_weight_logs:

            pix_number=int(elem_log.split('/')[-1].split('_')[-1].replace('.log',''))

            with open(elem_log,'r') as log_f:
                log_lines_inv=log_f.readlines()[::-1]

                last_weight_disp_line=np.argwhere(np.array(log_lines_inv)==\
                                                  'xaxmaarfgen: INFO: ENERGY      PHOTONS PER ENERGY\n').T[0][0]
                lines_clean = log_lines_inv[last_weight_disp_line - 15:last_weight_disp_line]

                weights_arr_pix=np.array([elem.split()[2:]for elem in lines_clean[::-1]],dtype=float).T
                pix_matr[pix_number]=(weights_arr_pix[1][(weights_arr_pix[0]>=e_min)& (weights_arr_pix[0]<=e_max)].sum())/raytrace_tots[i_file]

        raytrace_pix_full[i_weights]=pix_matr
    '''
    This is adapted from chatGPT
    '''

    if len(raytrace_pix_full)==3:
        triangle_mode=True
        B,C,D=raytrace_pix_full
    else:
        triangle_mode=False
        A,B,C,D=raytrace_pix_full

    rows, cols = 6, 6

    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    # Define colormaps
    # cmaps = {
    #     'A': plt.cm.Blues,
    #     'B': plt.cm.Reds,
    #     'C': plt.cm.Purples,
    #     'D': plt.cm.Oranges
    # }

    cmaps = {
        'A': plt.cm.YlGn,
        'B': plt.cm.YlOrRd,
        'C': plt.cm.PuBu,
        'D': plt.cm.RdPu,
    }

    # Normalize each matrix separately
    # norms = {
    #     'A': colors.Normalize(vmin=0., vmax=max(A.values())),
    #     'B': colors.Normalize(vmin=0., vmax=max(B.values())),
    #     'C': colors.Normalize(vmin=0., vmax=max(C.values())),
    #     'D': colors.Normalize(vmin=0., vmax=max(D.values()))
    # }

    norms = {
        'A': None if triangle_mode else colors.LogNorm(vmin=1e-3, vmax=sum(A.values())),
        'B': colors.LogNorm(vmin=1e-3, vmax=sum(B.values())),
        'C': colors.LogNorm(vmin=1e-3, vmax=sum(C.values())),
        'D': colors.LogNorm(vmin=1e-3, vmax=sum(D.values()))
    }

    for i in range(rows):
        for j in range(cols):

            x, y = j, rows - i - 1

            pix_number_xrism=int(coord_pixel_conv_arr.T[::-1][i][j])

            if pix_number_xrism==12:
                continue

            center = (x + 0.5, y + 0.5)

            # Define triangles for each matrix

            if triangle_mode:
                equa_rad_val=1/(3+4*np.sqrt(3)/3)
                triangles = {
                    'B': [(x,y),(x, y+equa_rad_val), (x + 0.5, y + 0.5),(x + 1, y+equa_rad_val),(x+1,y)],  # Bottom
                    'C': [(x, y+equa_rad_val), (x + 0.5, y + 0.5), (x+0.5, y + 1),(x,y+1)],  # Left
                    'D': [(x + 1, y+equa_rad_val), (x + 0.5, y + 0.5), (x + 0.5, y + 1),(x+1,y+1)]  # Right
                }
            else:

                triangles = {
                    'A': [(x, y + 1), (x + 1, y + 1), center],  # Top
                    'B': [(x, y), (x + 1, y), center],  # Bottom
                    'C': [(x, y), (x, y + 1), center],  # Left
                    'D': [(x + 1, y), (x + 1, y + 1), center]  # Right
                }

            values = {'A': None if triangle_mode else A[pix_number_xrism], 'B': B[pix_number_xrism],
                      'C': C[pix_number_xrism], 'D': D[pix_number_xrism]}

            # Draw and label triangles
            for key in triangles:
                poly = Polygon(triangles[key], closed=True,
                               facecolor=cmaps[key](norms[key](values[key])),
                               edgecolor='black', linewidth=0.5)
                ax.add_patch(poly)

                # Text position - average of triangle points
                tx = np.mean([p[0] for p in triangles[key]])
                ty = np.mean([p[1] for p in triangles[key]])
                ax.text(tx, ty, f"{values[key]:.3f}", fontsize=8, ha='center', va='center', color='black', alpha=0.8)

            # Draw main cell border
            ax.add_patch(plt.Rectangle((x, y), 1, 1,
                                       fill=False, edgecolor='black', linewidth=2))

            # Add cell number
            ax.text(x + 0.5, y + 0.5, str(pix_number_xrism),
                    ha='center', va='center', fontsize=14, fontweight='bold', color='black', alpha=1.)

    fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9)

    # Add colorbars around the plot
    #note: here we force specific values of the ticks to show the total value contained in the array
    #(to which the colorbars are normalized

    if not triangle_mode:
        cb_A=fig.colorbar(plt.cm.ScalarMappable(norm=norms['A'], cmap=cmaps['A']),
                     ax=ax, orientation='horizontal', location='top', fraction=0.046, pad=-0.105,extend='min',
                          ticks=[1e-3, 1e-2, 1e-1, sum(A.values())],
                     label='MAXI J1744-294 ['+str(e_min)+'-'+str(e_max)+'] keV PSF fraction (top)')

        cb_A.set_ticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','%.3f'%sum(A.values())])

    cb_C=fig.colorbar(plt.cm.ScalarMappable(norm=norms['C'], cmap=cmaps['C']),
                 ax=ax, orientation='vertical', location='left',fraction=0.0495, pad=0.015,extend='min',
                      ticks=[1e-3, 1e-2, 1e-1, sum(C.values())],anchor=(1.0,0.48),
                 label="GCXE "+'['+str(e_min)+'-'+str(e_max)+'] keV PSF fraction (left)')

    cb_C.set_ticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','%.3f'%sum(C.values())])

    cb_B=fig.colorbar(plt.cm.ScalarMappable(norm=norms['B'], cmap=cmaps['B']),
                 ax=ax, orientation='horizontal', fraction=0.046, pad=-0.102 if triangle_mode else -0.097,extend='min',
                      ticks=[1e-3, 1e-2, 1e-1, sum(B.values())],
                 label='AX J1745.6-2901 ['+str(e_min)+'-'+str(e_max)+'] keV PSF fraction (bottom)')  # placed below

    cb_B.set_ticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','%.3f'%sum(B.values())])

    cb_D=fig.colorbar(plt.cm.ScalarMappable(norm=norms['D'], cmap=cmaps['D']),
                 ax=ax, orientation='vertical',  fraction=0.0495, pad=-0.01 if triangle_mode else -0.038,extend='min',
                      ticks=[1e-3, 1e-2, 1e-1, sum(D.values())],anchor=(1.0,0.48),
                 label='Sgr A East ['+str(e_min)+'-'+str(e_max)+'] keV PSF fraction (right)')

    cb_D.set_ticklabels([r'10$^{-3}$',r'10$^{-2}$',r'10$^{-1}$','%.3f'%sum(D.values())])

    plt.show()

    plt.savefig('PSF_weights_compa_'+('tri_' if triangle_mode else '')
                +str(e_min).replace('.','p')+'_'+str(e_max).replace('.','p')+'.pdf')



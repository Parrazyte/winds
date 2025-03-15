
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
from matplotlib.widgets import Slider,Button

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
from mpdaf.obj import sexa2deg,deg2sexa, Image
from mpdaf.obj import WCS as mpdaf_WCS
from astropy.wcs import WCS as astroWCS
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

from general_tools import file_edit, ravel_ragged,MinorSymLogLocator,interval_extract,str_orbit,source_catal


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

def rsl_npixel_to_coord(number):
    '''
    returns the resolve image/detector coordinates of a given pixel number.
     See e.g. https://heasarc.gsfc.nasa.gov/docs/xrism/proposals/POG/Resolve.html
    '''

    #bruteforcing is easier
    coord_dict={
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

    return coord_dict[str(number)]

def init_anal(directory,anal_dir_suffix='',resolve_filters='open',xtd_config='all'):
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

    anal_dir=os.path.join(directory,'analysis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else ''))

    os.system('mkdir -p '+anal_dir)

    dirfiles=glob.glob(os.path.join(directory,'**'),recursive=True)

    resolve_evts=[elem for elem in dirfiles if elem.endswith('cl.evt.gz') and len(elem.split('/'))>2 and
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


    xtd_evts=[elem for elem in dirfiles if elem.endswith('cl.evt.gz') and len(elem.split('/'))>2 and
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

def resolve_RTS(directory,anal_dir_suffix='',heasoft_init_alias='heainit',caldb_init_alias='caldbinit',
                parallel=False):

    '''
    Filters all available resolve event files in the analysis subdirectory of a directory for Rise-Time Screening
    Following https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/quickstart/xrism_quick_start_guide_v2p3_240918a.pdf
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc,heasoft_init_alias,caldb_init_alias)

    anal_dir=os.path.join(directory,'analysis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else ''))

    resolve_files=[elem for elem in glob.glob(os.path.join(anal_dir,'**')) if 'rsl_' in elem.split('/')[-1] and
                   elem.endswith('_cl.evt')]

    bashproc.sendline('cd '+os.path.join(os.getcwd(),anal_dir))

    if os.path.isfile(directory + '/resolve_RTS.log'):
        os.system('rm ' + directory + '/resolve_RTS.log')

    with (no_op_context() if parallel else StdoutTee(directory + '/resolve_RTS.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/resolve_RTS.log', buff=1, file_filters=[_remove_control_chars])):

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
                time.sleep(1)

            time.sleep(1)


            bashproc.sendline('echo valid')

            bashproc.expect('valid')


    bashproc.sendline('exit')


def plot_BR(branch_file, save_path=None, excl_pixel=[]):
    '''
    Wrapper around the branching ratios plotting function

    if save_path is not None, will disable the gui and save before closing the plot

    excl_pixel can be an interable of excluded pixels to highlight
    '''

    with fits.open(branch_file) as branch_fits:
        branch_data = branch_fits[2].data

    # removing pixel 12 to avoid problems later with ordering
    branch_data = branch_data[[elem for elem in range(36) if elem != 12]]

    if save_path is not None:
        plt.ioff()

    # making a plot with the information
    fig_branch, ax_branch = plt.subplots(figsize=(16, 10))
    ax_branch.set_yscale('log')
    ax_branch.set_xscale('log')
    ax_branch.set_ylim(1e-3, 1.1)
    ax_branch.set_xlabel(r'Pixel count rate (s$^{-1}$)')
    ax_branch.set_ylabel('Pixel branching ratios')

    # showcasing the branching ratios
    plt.plot(branch_data['RATETOT'], branch_data['BRANCHHP'], ls='', marker='d',
             color='green', label='Hp')
    plt.plot(branch_data['RATETOT'], branch_data['BRANCHMP'], ls='', marker='d',
             color='blue', label='Mp')
    plt.plot(branch_data['RATETOT'], branch_data['BRANCHMS'], ls='', marker='d',
             color='cyan', label='Ms')
    plt.plot(branch_data['RATETOT'], branch_data['BRANCHLP'], ls='', marker='d',
             color='orange', label='Lp')
    plt.plot(branch_data['RATETOT'], branch_data['BRANCHLS'], ls='', marker='d',
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

    ax_up.set_xlabel('Pixel number (excluded in red)')

    # adding vertical lines
    for pix_number, pix_rate in zip(branch_data['PIXEL'], branch_data['RATETOT']):
        plt.axvline(pix_rate, ls=':', color='red' if pix_number in excl_pixel else 'grey',
                    zorder=-1)

    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        plt.ion()


def resolve_BR(directory, anal_dir_suffix='',
               use_raw_evt_rsl=False,
               pixel_str_rsl='all',
               remove_cal_pxl_resolve=False,
               pixel_filter_rule='clip_LS_0.02andcompa_LS>MS+remove_27',
               heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
               parallel=False):
    '''
    Computes a file and plot with the branching ratio information for each resolve event file
    see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslbranch.html

    pixel_filter_rule:
        if not None, uses the rules to determine pixels to EXCLUDE
        creates a txt file with the information of the pixels removed and the remaining pixels
        (which works as an input for later commands)
        different rules can be combined with +.
        subrules with and

    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    bashproc.sendline('cd ' + os.path.join(os.getcwd(), anal_dir))

    if os.path.isfile(directory + '/resolve_BR.log'):
        os.system('rm ' + directory + '/resolve_BR.log')

    with (no_op_context() if parallel else StdoutTee(directory + '/resolve_BR.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/resolve_BR.log', buff=1, file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read = sys.stdout

        for indiv_file in resolve_files:

            bashproc.sendline('rslbranch infile=' + indiv_file.split('/')[-1] +
                              ' outfile=' + indiv_file.split('/')[-1].replace('.evt', '_branch.fits') +
                              ' filetype=real' +
                              ' pixfrac=NONE' +
                              ' pixmask=NONE')

            bashproc.expect('Finished RSLBRANCH', timeout=60)

            time.sleep(1)

            bashproc.sendline('echo valid')

            # this is normally fast so their should be no need for a long time-out
            bashproc.expect('valid')

            # adding a filtering if it is requested
            if pixel_filter_rule is not None:

                with fits.open(indiv_file.replace('.evt', '_branch.fits')) as branch_fits:
                    branch_data = branch_fits[2].data

                mask_exclude = np.repeat(False, 36)

                for elem_rule in pixel_filter_rule.split('+'):

                    mask_subexclude = np.repeat(True, 36)

                    for subelem_rule in elem_rule.split('and'):

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

                    mask_exclude = (mask_exclude) | mask_subexclude

            pixel_exclude_list = np.arange(36)[mask_exclude]
            # saving the output of the filtering in a file

            pixel_filter_file = indiv_file.replace('.evt', '_branch_filter.txt')

            with open(pixel_filter_file, 'w+') as f:
                f.write('#Filter applied: ' + str(pixel_filter_rule) + '\n')
                f.write('#Combined count rate of excluded pixels: %.3e' % (
                    branch_data['RATETOT'][mask_exclude].sum()) + '\n')
                f.write('#Combined count rate of all pixels: %.3e' % (branch_data['RATETOT'].sum()) + '\n')
                f.write('#Remaining count rate proportion: %.3e'
                        % (1 - branch_data['RATETOT'][mask_exclude].sum() / branch_data['RATETOT'].sum()) + '\n')

                f.write('#list of excluded pixels:\n')
                f.write(str(pixel_exclude_list.tolist()) + '\n')

            plot_BR(indiv_file.replace('.evt', '_branch.fits'),
                    save_path=indiv_file.replace('.evt', '_branch_screen.png'),
                    excl_pixel=pixel_exclude_list)

            bashproc.sendline('exit')


def xtend_SFP(directory,filtering='flat_top',
              #for flat_top
              base_config='313',threshold_mult=1.1,rad_psf=15,
              source_name='auto',
              target_coords=None,
              target_only=False,use_file_target=True,use_file_target_coords=False,
              logprob2=None,bgd_level=None,cellsize=None,n_division=None,grade='ALL',
              logprob1=10,
              anal_dir_suffix='',parallel=False,sudo_screen=True,
              heasoft_init_alias='heainit',caldb_init_alias='caldbinit'):
    '''

    MAXIJ1744 approximate coords: ['17:45:40.45','-29:00:46.6']

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


    anal_dir=os.path.join(directory,'analysis'+('_'+anal_dir_suffix if anal_dir_suffix!='' else ''))

    xtend_files=[elem for elem in glob.glob(os.path.join(anal_dir,'**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl.evt')]

    bashproc.sendline('cd '+os.path.join(os.getcwd(),anal_dir))

    if os.path.isfile(directory + '/xtend_SFP.log'):
        os.system('rm ' + directory + '/xtend_SFP.log')

    with (no_op_context() if parallel else StdoutTee(directory + '/xtend_SFP.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/xtend_SFP.log', buff=1, file_filters=[_remove_control_chars])):

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

                # loading the IMG file with mpdaf
                with fits.open(sky_image_file) as hdul:
                    img_data = hdul[0].data
                    src_mpdaf_WCS = mpdaf_WCS(hdul[0].header)
                    src_astro_WCS = astroWCS(hdul[0].header)
                    main_source_name = hdul[0].header['object']
                    main_source_ra = hdul[0].header['RA_OBJ']
                    main_source_dec = hdul[0].header['DEC_OBJ']

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
                            obj_auto = {'MAIN_ID': main_source_name}
                            obj_deg = [main_source_ra, main_source_dec]
                    else:
                        # careful the output after the first line is in dec,ra not ra,dec
                        obj_deg = sexa2deg([obj_auto['DEC'].replace(' ', ':'), obj_auto['RA'].replace(' ', ':')])[::-1]
                        obj_deg = [str(obj_deg[0]), str(obj_deg[1])]
                else:
                    if type(target_coords[0])==str:
                        obj_deg=sexa2deg([target_coords[1].replace(' ',':'),target_coords[0].replace(' ',':')])[::-1]
                    else:
                        obj_deg=target_coords

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

            bashproc.sendline('echo valid')

            bashproc.expect('valid')

        bashproc.sendline('exit')


def rsl_pixel_manip(pixel_str,remove_cal_pxl_resolve=True,mode='default',region_path=None):

    '''
    Small function to get correct pixel strings from a list of pixels

    example:for 'PIXEL=0:11,13:35', put '0:11,13:35'
    also accepts pixels to exclude, such as '-(10:14,28,32)'
    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    if mode is set to rmf, uses the rmf way of noting the intervals, with - instead of : for ranges

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
    pixel_ok_inter = interval_extract(pixel_ok_list)
    pixel_ok_inter_str = [str(elem[0]) + ':' + str(elem[1]) if elem[0] != elem[1] else str(elem[0])
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
              sudo_screen=True,
              sudo_mdp='',
              #for lc
              exposure=0.8,binning=128,
              spawn=None,
              heasoft_init_alias='heainit',caldb_init_alias='caldbinit'):
    '''
    Uses Xselect to create a XRISM product from a bash spawn

        e_low and e_high should be in keV

    mode='img','lc' or 'spectrum'

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
    spawn_use.sendline('')

    # reading events
    spawn_use.sendline('read events')

    #removing the saved session if need bes
    line_code=spawn_use.expect(['Use saved session?','SUZAKU'])
    if line_code==0:
        spawn_use.sendline('no')

        spawn_use.sendline('read events')


    spawn_use.expect('Event file dir')
    spawn_use.sendline(evt_dir)

    spawn_use.expect('Event file list')
    spawn_use.sendline(evt_file)

    # resetting mission
    spawn_use.expect('Reset')
    spawn_use.sendline('yes')

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

                spawn_use.sendline('set image DET')



            spawn_use.sendline('filter column "PIXEL='+rsl_pixel_manip(region_str,
                                remove_cal_pxl_resolve=remove_cal_pxl_resolve)+'"')

    if gti_file is not None:
        spawn_use.sendline('filter time file '+gti_file)

    if mode=='image':

        spawn_use.sendline('set xybinsize 1')
        spawn_use.sendline('extract image')
        spawn_use.expect('Image')

        # commands to save image
        spawn_use.sendline('save image')

        # can take time so increased timeout
        spawn_use.expect('Give output file name', timeout=120)

        spawn_use.sendline(save_path)

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

    print('Letting some time to create the file...')
    # giving some time to create the file
    time.sleep(1)

    for i_sleep in range(20):
        if not os.path.isfile(os.path.join(directory, save_path)):
            print('File still not ready. Letting more time...')
            time.sleep(5)

    if not os.path.isfile(os.path.join(directory, save_path)):
        print('Issue with file check or file creation')
        breakpoint()

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

        plot_lc_xrism(save_path,binning,directory=directory,e_low=e_low,e_high=e_high,save=True)

    if temp_evt_name is not None:

        spawn_use.sendline('rm '+temp_evt_name)

def plot_lc_xrism(lc_path,binning='auto',directory='./',e_low='',e_high='',
                  interact=False,
                  interact_tstart=None,
                  save=False,suffix='',outdir=''):

    '''

    Wrapper to plot xrism lightcurves with or without interactivity

    outdir: if set to None, saves the lightcure where lc_path is, otherwise in the outdir subdirectory
    '''

    with fits.open(os.path.join(directory, lc_path)) as fits_lc:
        data_lc_arr = fits_lc[1].data

        telescope = fits_lc[1].header['TELESCOP']
        instru = fits_lc[1].header['INSTRUME']

        time_zero = Time(fits_lc[1].header['MJDREFI'] + fits_lc[1].header['MJDREFF'], format='mjd')
        time_zero += TimeDelta(fits_lc[1].header['TIMEZERO'], format='sec')

    # and plotting it

    if save:
        plt.ioff()
    fig_lc, ax_lc = plt.subplots(1, figsize=(16, 8))

    if binning=='auto':
        binning_use=data_lc_arr['TIME'][1]-data_lc_arr['TIME'][0]
    else:
        binning_use=str(binning)

    plt.errorbar(data_lc_arr['TIME'], data_lc_arr['RATE'], xerr=float(binning_use) / 2,
                 yerr=data_lc_arr['ERROR'], ls='-', lw=1, color='grey', ecolor='blue')


    binning_str=str(binning_use)

    plt.suptitle(
        telescope + ' ' + instru + ' lightcurve for observation ' + lc_path.split('_lc')[0].split('_pixel')[0] +
        (' with pixel ' + lc_path.split('_pixel_')[-1].split('_')[0] if instru == 'RESOLVE' else
         ' with region ' + lc_path.split('_cl_')[-1].split('_lc')[0]) +
        ' in [' + str(e_low) + '-' + str(e_high) + '] keV with ' + binning_str + ' s binning')

    plt.xlabel('Time (s) after ' + time_zero.isot)
    plt.ylabel('RATE (counts/s)')

    plt.tight_layout()

    if interact:

        plt.subplots_adjust(bottom=0.2)

        ax_slider_start = fig_lc.add_axes([0.2, 0.1, 0.65, 0.03])

        ax_slider_end = fig_lc.add_axes([0.2, 0.05, 0.65, 0.03])


        #note: we add one binning unit to allow to go all the way and take the last bin
        slider_start = Slider(ax_slider_start, label='gti start (s)',
                      valmin=data_lc_arr['TIME'][0]+float(binning_use)/2 if interact_tstart is None else interact_tstart,
                      valmax=data_lc_arr['TIME'][-1]+float(binning_use), valstep=float(binning_use))

        slider_end = Slider(ax_slider_end, label='gti end (s)',
                      valmin=data_lc_arr['TIME'][0]+float(binning_use)/2,
                      valmax=data_lc_arr['TIME'][-1]+float(binning_use)/2, valstep=float(binning_use))
        def slider_update(val):

            for elem_child in ax_lc.get_children():
                if elem_child._label == 'current gti':
                    elem_child.remove()

            ax_lc.axvspan(slider_start.val, slider_end.val,0, 1, alpha=0.3, color='green', label='current gti')

            fig_lc.legend()

        slider_start.on_changed(slider_update)
        slider_end.on_changed(slider_update)

        ax_button = fig_lc.add_axes([0.9, 0.025, 0.08, 0.04])

        but = Button(ax=ax_button, label='Save GTI')

        def func_button(val):
            plt.close()
            print(slider_start.val)
            print(slider_end.val)

        plt.show()
        but.on_clicked(func_button)

        #will block the code as long as the button isn't pressed
        plt.show(block=True)

        if not save:
            return slider_start.val,slider_end.val

    if save:
        lc_path_extension=lc_path[lc_path.rfind('.'):]

        os.system('mkdir -p '+os.path.join(directory, outdir))

        fig_lc.savefig(os.path.join(directory, outdir, lc_path.replace(lc_path_extension,
                                                        ('_'+ suffix if suffix!='' else '')+'_screen.png')))
        plt.close()
        plt.ion()

        if interact:
            return slider_start.val,slider_end.val


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
            time.sleep(1)
            windows_after = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
            delay += 1

        for elem in windows_after:
            if elem not in windows_before:
                ds9_pid = elem.split(' ')[0]
                print('\nIdentified the new ds9 window as process ' + ds9_pid)

                if screenfile != '':
                    print('\nSaving screenshot...')
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

def extract_img(directory,anal_dir_suffix='',
                instru='all',
                   use_raw_evt_xtd=False,use_raw_evt_rsl=False,
                   heasoft_init_alias='heainit',caldb_init_alias='caldbinit',
                   sudo_screen=True,
                   parallel=False):

    '''
    Extract images from event files in the analysis subdirectory of a directory

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    '''
    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc,heasoft_init_alias,caldb_init_alias)

    if sudo_screen:
        sudo_mdp_use = input('Give sudo mdp')
    else:
        sudo_mdp_use = ''

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl' + ('' if use_raw_evt_xtd else '_SFP') + '.evt')]

    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]

    if os.path.isfile(directory + '/extract_img.log'):
        os.system('rm ' + directory + '/extract_img.log')

    with (no_op_context() if parallel else StdoutTee(directory+'/extract_img.log',mode="a",buff=1,file_filters=[_remove_control_chars]),\
        StderrTee(directory+'/extract_img.log',buff=1,file_filters=[_remove_control_chars])):

        if not parallel:
            bashproc.logfile_read=sys.stdout

        if len(resolve_files)!=0:
            # for removing the calibration sources
            os.system('cp $HEADAS/refdata/region_RSL_det.reg '+anal_dir)

        bashproc.sendline('cd '+os.path.join(os.getcwd(),anal_dir))

        for elem_evt in resolve_files+xtend_files:

            xsel_util(elem_evt.split('/')[-1],elem_evt.split('/')[-1].replace('.evt','_img.ds'),
                     mode='image',directory=anal_dir,sudo_screen=sudo_screen,sudo_mdp=sudo_mdp_use,
                      spawn=bashproc)


def create_gtis(directory, split_arg='orbit',split_lc_file='file',split_lc_method=None,
                split_auto_bin=1,
                gti_tool='NICERDAS',
                anal_dir_suffix='', heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                gti_subdir='gti',
                thread=None, parallel=False):
    '''
    wrapper for a function to split xrism obsids into indivudal gti portions with different methods
    the default binning when split_lc_method is set is 1s

    products end by default in Obsdir/analysis+anal_dir_suffix/gti_subdir

    overwrite is always on here since we don't use a specific nicerdas task with the overwrite option

    split modes:

        -orbit:split each obs into each individual nicer observation period. Generally, should always be enabled.
               GTIs naming: obsid-XXX chronologically for each split


        -manual: provides an interactive window to make individual gti splits. GTI naming: obsid-MXXX
            by default a single interval, otherwise use 'manual_multi':
                 acts as manual but allows for a set of intervals until the end of the obseration is reached

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
                         time_gtis, gti_tool='NICERDAS',outdir='',
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


        input_dir=os.path.join('./' if '/' not in file_base else file_base[:file_base.rfind('/')],outdir)

        # creating the gti expression
        gti_path = os.path.join(input_dir,
                                (file_base[:file_base.rfind('.')]+
                                '_gti_' + orbit_prefix + suffix + '.gti').split('/')[-1])

        if outdir!='':

            os.system(' mkdir -p '+os.path.join(directory,input_dir))

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

            # creating the gti expression
            gti_input_path = os.path.join(input_dir,
                                          (file_base[:file_base.rfind('.')] +
                                    '_gti_input_' + orbit_prefix + suffix + '.txt').split('/')[-1])

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

    io_log = open(directory + '/create_gtis.log', 'w+')

    # ensuring a good obsid name even in local
    if directory == './':
        obsid = os.getcwd().split('/')[-1]
    else:
        obsid = directory

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8', logfile=io_log if parallel else None)

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

    print('\n\n\nCreating gtis products...')

    set_var(bashproc)

    if os.path.isfile(os.path.join(directory + '/extract_gtis.log')):
        os.system('rm ' + os.path.join(directory + '/extract_gtis.log'))

    # removing old gti files
    old_files_gti = [elem for elem in glob.glob(os.path.join(directory, 'analysis/**'), recursive=True) if
                     '_gti_' in elem]

    for elem_file_gti in old_files_gti:
        os.remove(elem_file_gti)

    with (no_op_context() if parallel else StdoutTee(os.path.join(directory + '/create_gtis.log'), mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(os.path.join(directory + '/create_gtis.log'), buff=1, file_filters=[_remove_control_chars])):

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

            if split_arg=='manual_multi':
                cut_times = []

                while len(cut_times)==0 or cut_times[-1][-1] < time_obs[-1]:
                    cut_times += \
                        [plot_lc_xrism(lc_input.split('/')[-1],binning=binning_use,interact=True,
                                       #here to allow to start from the previous ending gti value for each subsequent
                                       #cut
                                       directory=anal_dir,
                                       interact_tstart=None if len(cut_times)==0 else cut_times[-1][-1]
                                       ,save=True,suffix=split_keyword+str_orbit(i_cut),
                                       outdir=gti_subdir)]

                    i_cut+=1

                    print('Added manual gti split interval at for t in ' + str(cut_times[-1]) + ' s')

            else:
                cut_times = \
                    [plot_lc_xrism(lc_input.split('/')[-1],binning=binning_use,interact=True,
                                   #here to allow to start from the previous ending gti value for each subsequent
                                   #cut
                                   directory=anal_dir,
                                   interact_tstart=None,save=True,suffix=split_keyword+str_orbit(i_cut),
                                   outdir=gti_subdir)]

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


            # create the gti files with a "S" keyword and keeping the orbit information in the name
            for i_split, split_gtis in enumerate(split_gti_arr):
                if len(split_gtis) > 0:
                    create_gti_files(split_gtis, data_input, split_keyword+str_orbit(i_split), suffix='',
                                     file_base=lc_input,
                                     time_gtis=time_obs, gti_tool=gti_tool,outdir=gti_subdir)


        # exiting the bashproc
        bashproc.sendline('exit')
        if thread is not None:
            thread.set()


def extract_lc(directory, anal_dir_suffix='',lc_subdir='lc',
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
                   parallel=False):

    '''

    Extract lightcurves from event files in the analysis subdirectory of a directory

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    Instru determines whether the analysis is run on all or a single instrument

    region_src_xtd/region_bg_xtd:
        if set to auto, fetches source/background regions with the evt file name _src_reg.reg/_bg_reg.reg
        as the base, and only extracts products when corresponding files are found
        Regions are assumed to be in DET coordinates

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

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

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

    if os.path.isfile(directory + '/extract_lc.log'):
        os.system('rm ' + directory + '/extract_lc.log')

    with (no_op_context() if parallel else StdoutTee(directory + '/extract_lc.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/extract_lc.log', buff=1, file_filters=[_remove_control_chars])):

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
                                          region_str=elem_region,
                                          e_low=float(elem_band.split('-')[0]),
                                          e_high=float(elem_band.split('-')[1]),
                                          binning=elem_binning,
                                          exposure=exposure_xtd,
                                          spawn=bashproc,
                                          gti_file=elem_gti_file)

                else:

                    if pixel_str_rsl == 'branch_filter':
                        # reading the branch filter file
                        with open(elem_evt.replace('.evt','_branch_filter.txt')) as branch_f:
                            branch_lines=branch_f.readlines()
                        branch_filter_line=[elem for elem in branch_lines if not elem.startswith('#')][0]
                        #reformatting the string
                        pixel_str_rsl_use='-('+branch_filter_line[1:-2]+')'
                    else:
                        pixel_str_rsl_use=pixel_str_rsl

                    for elem_pixel_str in pixel_str_rsl_use.split('+'):

                        reg_str = pixel_str_rsl if pixel_str_rsl=='branch_filter' else elem_pixel_str

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

def extract_sp(directory, anal_dir_suffix='',sp_subdir='sp',
                   use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                    instru='all',
                   region_src_xtd='auto', region_bg_xtd='auto',
                   pixel_str_rsl='branch_filter', grade_str_rsl='0:0',
                   remove_cal_pxl_resolve=True,
                   gti=None, gti_subdir='gti',
                   e_low_rsl=0.3,e_high_rsl=12.,
                   e_low_xtd=None,e_high_xtd=None,
                   sudo_screen=True,
                   heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                   parallel=False):

    '''

    #pixels for 4U: '-(5,23,27,29,30)'

    Extract spectra from event files in the analysis subdirectory of a directory

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    Instru determines whether the analysis is run on all or a single instrument


    region_src_xtd/region_bg_xtd:
        if set to auto, fetches source/background regions with the evt file name _src_reg.reg/_bg_reg.reg
        as the base, and only extracts products when corresponding files are found
        Regions are assumed to be in DET coordinates

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

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

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

    if os.path.isfile(directory + '/extract_sp.log'):
        os.system('rm ' + directory + '/extract_sp.log')

    with (no_op_context() if parallel else StdoutTee(directory + '/extract_sp.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/extract_sp.log', buff=1, file_filters=[_remove_control_chars])):

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
                    else:
                        region_src_xtd_use=region_src_xtd

                    if region_bg_xtd=='auto':
                        region_bg_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_bg_reg.reg'))
                    else:
                        region_bg_xtd_use=region_bg_xtd

                    if (region_bg_xtd_use is not None and os.path.isfile(region_bg_xtd_use))\
                    or (region_src_xtd_use is not None and os.path.isfile(region_src_xtd_use)):
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

                        reg_str=('_auto_src' if region_src_xtd=='auto' else region_src_xtd.split('.')[0]) if i_reg==0 else \
                                ('_auto_bg' if region_bg_xtd == 'auto' else region_bg_xtd.split('.')[0])

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
                                  gti_file=elem_gti_file)


                else:

                    if pixel_str_rsl == 'branch_filter':
                        # reading the branch filter file
                        with open(elem_evt.replace('.evt','_branch_filter.txt')) as branch_f:
                            branch_lines=branch_f.readlines()
                        branch_filter_line=[elem for elem in branch_lines if not elem.startswith('#')][0]
                        #reformatting the string
                        pixel_str_rsl_use='-('+branch_filter_line[1:-2]+')'
                    else:
                        pixel_str_rsl_use=pixel_str_rsl

                    for elem_pixel_str in pixel_str_rsl_use.split('+'):

                        reg_str = pixel_str_rsl if pixel_str_rsl=='branch_filter' else elem_pixel_str


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
                                  gti_file=elem_gti_file)

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
    ' chatter=2 '+
                       (' clobber=YES' if overwrite else ''))

    spawn_use.expect('Finished',timeout=None)

    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

def extract_rmf(directory,instru='all',rmf_subdir='sp',
                #resolve options
                rmf_type_rsl='X',pixel_str_rsl='branch_filter',rsl_arf_grade='0',
                remove_cal_pxl_resolve=True,
                # resolve grid
                eminin_rsl=300,dein_rsl=0.5,nchanin_rsl=23400,
                useingrd_rsl=True,
                #xtend grid
                eminin_xtd=200,dein_xtd='"2,24"',nchanin_xtd='"5900,500"',
                eminout_xtd=0.,deout_xtd=6,nchanout_xtd=4096,
                #general event options and gti selection
                use_raw_evt_rsl=False,use_raw_evt_xtd=False,
                gti=None, gti_subdir='gti',
                #common arguments
                anal_dir_suffix='', heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                parallel=False):

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

    rsl_arf_grade:the event grade of the arf. Different syntax so no ":" in the string
                    (see https://heasarc.gsfc.nasa.gov/lheasoft/help/rslmkrmf.html)

    no matter the selection of pixel_str_xrism, if remove_cal_px_resolve is set to True, pixel 12 (calibration pixel)
    will be removed

    e_low/e_high: if not None, the energy bounds of the spectra
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))

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

    if os.path.isfile(directory + '/extract_rmf.log'):
        os.system('rm ' + directory + '/extract_rmf.log')

    with (no_op_context() if parallel else StdoutTee(directory + '/extract_rmf.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/extract_rmf.log', buff=1, file_filters=[_remove_control_chars])):

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

                if pixel_str_rsl == 'branch_filter':
                    # reading the branch filter file
                    with open(elem_evt.replace('.evt', '_branch_filter.txt')) as branch_f:
                        branch_lines = branch_f.readlines()
                    branch_filter_line = [elem for elem in branch_lines if not elem.startswith('#')][0]
                    # reformatting the string
                    pixel_str_rsl_use = '-(' + branch_filter_line[1:-2] + ')'
                else:
                    pixel_str_rsl_use = pixel_str_rsl

                for elem_pixel_str in pixel_str_rsl_use.split('+'):

                    reg_str = pixel_str_rsl if pixel_str_rsl == 'branch_filter' else elem_pixel_str

                    product_root = os.path.join(rmf_subdir,elem_evt.replace(anal_dir,'.').replace('.evt',
                            '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                            ('_withcal' if not remove_cal_pxl_resolve else '')+
                        '_grade_'+rsl_arf_grade+ '_rmf' +
                        ('_' + str(eminin_rsl).replace('.','')+ '_' + str(dein_rsl).replace('.','')+'_'+str(nchanin_rsl))
                                                                   +elem_gti_str))

                    rsl_mkrmf(infile=elem_evt.replace(anal_dir,'.'),
                              outfileroot=product_root,
                              pixlist=rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                      mode='rmf'),
                              whichrmf=rmf_type_rsl,
                              resolist=rsl_arf_grade,
                              eminin=eminin_rsl,
                              dein=dein_rsl,
                              nchanin=nchanin_rsl,
                              useingrd=useingrd_rsl,
                              spawn=bashproc,heasoft_init_alias=heasoft_init_alias,caldb_init_alias=caldb_init_alias)

def create_expo(directory,anal_dir,instrument,evt_file,gti_file,out_file='auto',
                spawn=None,heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                delta=20,numphi=1):

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

    files_dir=glob.glob(directory+'/**',recursive=True)

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
                     and not elem.endswith('.gpg') and evt_file.split('/')[-1].split('_')[1] in elem][0]

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
                       ' gtifile='+gti_file_use.replace('_SFP','')+
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

def create_arf(directory,instrument,out_rtfile,source_ra,source_dec,emap_file,out_file,
               region_file,rmf_file,source_type='POINT',e_low=0.3,e_high=12.0,
                numphoton=300000,minphoton=100,
                telescope='XRISM',
                spawn=None,heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
               e_low_image=0.0, e_high_image=0.0):

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
                       ' regmode=DET'+
                       ' regionfile='+region_file+
                       ' sourcetype='+source_type+
                       ' rmffile='+rmf_file+
                       ' erange="'+str(e_low)+' '+str(e_high)+' '+str(e_low_image)+' '+str(e_high_image)+'"'+
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
                       ' imgfile=NONE')

    out_code=spawn_use.expect(['Error during subroutine finalize','xaxmaarfgen: Fraction of PSF inside Region'],
                     timeout=None)

    if out_code==0:
        print('Error during xaarfgen computation')
        raise ValueError

    time.sleep(1)

    spawn_use.sendline('echo valid')

    #this is normally fast so their should be no need for a long time-out
    spawn_use.expect('valid')

def extract_arf(directory,anal_dir_suffix='',on_axis_check=None,arf_subdir='sp',
                source_coords='on-axis',
                target_coords=None,
                source_name='auto',
                target_only=False,use_file_target=True,
                source_type='POINT',
                instru='all',
                use_raw_evt_xtd=False, use_raw_evt_rsl=False,
                region_src_xtd='auto', region_bg_xtd='auto',
                pixel_str_rsl='branch_filter', grade_str_rsl='0:0',
                remove_cal_pxl_resolve=True,
                gti=None, gti_subdir='gti',
                e_low_rsl=0.3, e_high_rsl=12.0,
                e_low_xtd=0.3, e_high_xtd=12.0,

                numphoton=300000,
                minphoton=100,
                heasoft_init_alias='heainit', caldb_init_alias='caldbinit',
                parallel=False):

    '''

    Extract arf from event files in the analysis/arf_subdir of an observation directory

    the gtis used for computing the exposure maps and finding the rmf files

    Instru determines whether the analysis is run on all or a single instrument

    on_axis_check checks whether the region centroids match the source center (to be implemented)

    source_coord:
        -on-axis:
            assumes the coordinates of the source (aka the pointing is close enough to being on-axis)
        -an array:
            takes the values provided manually

    source_name:
        -auto: fetches on Simbad the source matching the name of the directory directly above the
                obsid directory
        -anything else is given to simbad directly

    use_raw_evt_xtd/use_raw_evt_rsl determine if the images are created from raw or filtered evts

    on axis check performs a chekc to compare the position of the image center to that of the source
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

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    set_var(bashproc, heasoft_init_alias, caldb_init_alias)

    anal_dir = os.path.join(directory, 'analysis' + ('_' + anal_dir_suffix if anal_dir_suffix != '' else ''))


    os.system('mkdir -p '+os.path.join(anal_dir,arf_subdir))


    resolve_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'rsl_' in elem.split('/')[-1] and
                     elem.endswith('_cl' + ('' if use_raw_evt_rsl else '_RTS') + '.evt')]

    xtend_files = [elem for elem in glob.glob(os.path.join(anal_dir, '**')) if 'xtd_' in elem.split('/')[-1] and
                   elem.endswith('_cl' + ('' if use_raw_evt_xtd else '_SFP') + '.evt')]

    #here using any event file should work
    if len(resolve_files+xtend_files)==0:
        print('No event file satisfy the requirements. Skipping...')
        return 0
    else:

        if source_coords is None:
            any_event=(resolve_files+xtend_files)[0]
            if source_coords == 'on-axis':

                if source_name == 'auto':

                    obj_auto = source_catal(bashproc, './', any_event,
                                            target_only=target_only,
                                            use_file_target=use_file_target)

                else:
                    obj_auto = Simbad.query_object(source_name)[0]

            source_ra, source_dec = sexa2deg([obj_auto['DEC'], obj_auto['RA']])[::-1]

        else:
            if type(target_coords[0])==str:
                obj_deg=sexa2deg([target_coords[1].replace(' ',':'),target_coords[0].replace(' ',':')])[::-1]
            else:
                obj_deg=target_coords
            source_ra,source_dec=obj_deg


    if instru!='all':
        if instru=='xtend':
            resolve_files=[]
        elif instru=='resolve':
            xtend_files=[]

    if os.path.isfile(directory + '/extract_arf.log'):
        os.system('rm ' + directory + '/extract_arf.log')

    if os.path.isfile('~/pfiles/xaarfgen.par'):
        os.system('rm ~/pfiles/xaarfgen.par')
    if os.path.isfile('~/pfiles/xaxmaarfgen.par'):
        os.system('rm ~/pfiles/xaxmaarfgen.par')

    with (no_op_context() if parallel else StdoutTee(directory + '/extract_arf.log', mode="a", buff=1,
                                                     file_filters=[_remove_control_chars]), \
          StderrTee(directory + '/extract_arf.log', buff=1, file_filters=[_remove_control_chars])):

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
                expo_path=create_expo(directory,anal_dir,instrument='xtend' if xtd_mode else 'resolve',
                            evt_file=elem_evt.replace(anal_dir,'.'),
                            gti_file=elem_evt.replace(anal_dir,'.') if elem_gti_file is None else
                                     elem_gti_file.replace(anal_dir,'.'))

                if xtd_mode:

                    if region_src_xtd=='auto':
                        region_src_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_src_reg.reg'))
                    else:
                        region_src_xtd_use=region_src_xtd

                    if region_bg_xtd=='auto':
                        region_bg_xtd_use = os.path.join(os.getcwd(),elem_evt.replace('.evt','_bg_reg.reg'))
                    else:
                        region_bg_xtd_use=region_bg_xtd

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
                         and reg_str in elem]

                        if len(rmf_list)==0:
                            print('No rmf found for xtend event '+elem_evt)
                            print('Skipping arf computation...')
                            continue

                        #removing the rtfile to ensure we recreate it
                        if os.path.isfile(elem_evt.replace('.evt','_raytracing.evt')):
                            os.remove(elem_evt.replace('.evt','_raytracing.evt'))

                        rmf_path=rmf_list[0]
                        create_arf(directory,instrument='xtend',
                                   out_rtfile=elem_evt.replace(anal_dir,'.').replace('.evt','_raytracing.evt'),
                                   out_file=rmf_path.replace('.rmf','.arf').replace(anal_dir,'.'),
                                   source_ra=source_ra,
                                   source_dec=source_dec,
                                   emap_file=os.path.join(os.getcwd(),anal_dir,expo_path.split('/')[-1]),
                                   region_file=elem_region,
                                   rmf_file=rmf_path.replace(anal_dir,'.'),
                                   source_type=source_type,
                                   e_low=e_low_xtd,e_high=e_high_xtd,
                                   numphoton=numphoton,
                                   minphoton=minphoton,
                                   spawn=bashproc)

                else:

                    if pixel_str_rsl == 'branch_filter':
                        # reading the branch filter file
                        with open(elem_evt.replace('.evt','_branch_filter.txt')) as branch_f:
                            branch_lines=branch_f.readlines()
                        branch_filter_line=[elem for elem in branch_lines if not elem.startswith('#')][0]
                        #reformatting the string
                        pixel_str_rsl_use='-('+branch_filter_line[1:-2]+')'
                    else:
                        pixel_str_rsl_use=pixel_str_rsl

                    for elem_pixel_str in pixel_str_rsl_use.split('+'):

                        reg_str = pixel_str_rsl if pixel_str_rsl=='branch_filter' else pixel_str_rsl_use


                        product_root = elem_evt.split('/')[-1].replace('.evt',
                                '_pixel_'+reg_str.replace(':','to').replace(',','-').replace('-(','no').replace(')','')  +
                                ('_withcal' if not remove_cal_pxl_resolve else ''))

                        rmf_path=[elem for elem in glob.glob(os.path.join(anal_dir,'**'), recursive=True) if
                                  product_root in elem and elem.endswith('.rmf')][0]
                        #for now we consider a single Resolve RMF no matter the GTI,

                        #gettingthe name
                        region_name=rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                    mode='rmf')

                        region_path=os.path.join(anal_dir,'resolve_pixels_'+region_name.replace(',','_')+'.reg')
                        #creating the file
                        rsl_pixel_manip(elem_pixel_str,remove_cal_pxl_resolve=remove_cal_pxl_resolve,
                                                 region_path=region_path)

                        #removing the rtfile to ensure we recreate it
                        if os.path.isfile(elem_evt.replace('.evt','_raytracing.evt')):
                            os.remove(elem_evt.replace('.evt','_raytracing.evt'))

                        create_arf(directory, instrument='resolve',
                                   out_rtfile=elem_evt.replace(anal_dir,'.').replace('.evt','_raytracing.evt'),
                                   out_file=rmf_path.replace('.rmf', '.arf').replace(anal_dir,'.'),
                                   source_ra=source_ra,
                                   source_dec=source_dec,
                                   emap_file=os.path.join(os.getcwd(),anal_dir,expo_path),
                                   region_file=region_path.replace(anal_dir,'.'),
                                   rmf_file=rmf_path.replace(anal_dir,'.'),
                                   source_type=source_type,
                                   e_low=e_low_rsl, e_high=e_high_rsl,
                                   numphoton=numphoton,
                                   minphoton=minphoton,
                                   spawn=bashproc)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
In progress script for NuSTAR data reduction
'''

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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# note: might need to install opencv-python-headless to avoid dependancies issues with mpl

# import matplotlib.cm as cm
from matplotlib.collections import LineCollection

# astro imports
from astropy.time import Time
from astropy.io import fits
from astroquery.simbad import Simbad
from mpdaf.obj import sexa2deg, Image
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

from general_tools import file_edit, ravel_ragged

"""
Created on 09-11-2023

Data reduction Script for NuSTAR Observations

Searches for all NuSTAR Obs type directories in the subdirectories and launches the process for each

list of possible actions : 

1. process_obsdir: run the nupipeline script to process an obsid folder

gti. create_gtis: create custom gtis files to be used later for lightcurve and spectrum creation

fs. extract_all_spectral: runs the nicerl3-spect script to compute spectral products of an obsid folder (aka s,b,r at the same time)

l. extract_lightcurve: runs a set of nicerl3-lc scripts to compute a range of lightcurve and HR evolutions

g. group_spectra: group spectra using the optimized Kastra et al. binning

m.merge: merge all spectral products in the subdirectories to a bigbatch directory

c. clean_products: clean event products in the observation's event_cl directory

fc. clean_all: clean all files including standard products and products of this script from the directory

DEPRECATED 
2. select_detector: removes specific detectors from the event file (not tested)

s. extract_spectrum: extract a pha spectrum from a process obsid folder using  Xselect

b. extract_background: extract a bg spectrum from a process obsid folder using a specific method

r. extract_response: extract a response from a processed obsid folder



"""

# better errors : to test
# import pretty_errors
# pretty_errors.configure(
#     separator_character = '*',
#     filename_display    = pretty_errors.FILENAME_EXTENDED,
#     line_number_first   = True,
#     display_link        = True,
#     lines_before        = 5,
#     lines_after         = 2,
#     line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
#     code_color          = '  ' + pretty_errors.default_config.line_color,
#     truncate_code       = True,
#     display_locals      = True
# )


'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce NICER files.\n)')

# the basics

ap.add_argument('-load_functions', nargs=1, help="Load functions but don't launch anything", default=False, type=bool)

ap.add_argument("-dir", "--startdir", nargs='?', help="starting directory. Current by default", default='./', type=str)
ap.add_argument("-l", "--local", nargs=1, help='Launch actions directly in the current directory instead',
                default=False, type=bool)
ap.add_argument('-catch', '--catch_errors', help='Catch errors while running the data reduction and continue',
                default=True, type=bool)

# global choices
ap.add_argument("-a", "--action", nargs='?', help='Give which action(s) to proceed,separated by comas.',
                default='c', type=str)

# default: 1,gti,fs,l,g,m,c

ap.add_argument("-over", nargs=1, help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder', default=True, type=bool)

# directory level overwrite (not active in local)
ap.add_argument('-folder_over', nargs=1, help='relaunch action through folders with completed analysis', default=False,
                type=bool)
ap.add_argument('-folder_cont', nargs=1, help='skip all but the last 2 directories in the summary folder file',
                default=False, type=bool)
# note : we keep the previous 2 directories because bug or breaks can start actions on a directory following the initially stopped one

# action specific overwrite

ap.add_argument('-gtype', "--grouptype", help='Give the group type to use in regroup_spectral', default='opt', type=str)

ap.add_argument('-heasoft_init_alias', help="name of the heasoft initialisation script alias", default="heainit",
                type=str)
ap.add_argument('-caldbinit_init_alias', help="name of the caldbinit initialisation script alias", default="caldbinit",
                type=str)

# Should correspond to the most important energy band for subsequent science analysis. also used in the region computation

'''region computation'''

ap.add_argument('-use_file_coords', nargs=1,
                help='Allows to extract regions when Simbad doesnt recognize the name of the source', default=True)

ap.add_argument("-target_only", nargs=1,
                help='only extracts spectra when the source is the main focus of the observation',
                default=False, type=bool)

ap.add_argument('-image_band',nargs=1,help='band in which to extract the image for region computation',default='3_79',
                type=str)

ap.add_argument('-rad_crop', nargs=1,
                help='croppind radius around the theoretical source position before fit, in arcsecs', default=120,
                type=float)

ap.add_argument('-bigger_fit', nargs=1,
                help='allows to incease the crop window used before the gaussian fit for bright sources',
                default=True, type=bool)

ap.add_argument('-point_source', nargs=1,
                help="assume the source is point-like, I.E. fixes the gaussian's initial center to the brightest pixel",
                default=True, type=bool)
# helps to avoid the gaussian center shifting in case of diffuse emission

# if equal to crop, is set to rad_crop
ap.add_argument('-max_rad_source', nargs=1, help='maximum source radius for faint sources in units of PSF sigmas',
                default=5, type=float)

'''lightcurve'''
ap.add_argument('-lc_bin', nargs=1, help='Gives the binning of all lightcurces/HR evolutions (in s)', default=1,
                type=str)
# note: also defines the binning used for the gti definition

ap.add_argument('-lc_bands_str', nargs=1, help='Gives the list of bands to create lightcurves from', default='3-79',
                type=str)
ap.add_argument('-hr_bands_str', nargs=1, help='Gives the list of bands to create hrsfrom', default='10-50/3-10',
                type=str)

args = ap.parse_args()

load_functions=args.load_functions

startdir=args.startdir
action_list=args.action.split(',')
local=args.local
folder_over=args.folder_over
folder_cont=args.folder_cont
overwrite_glob=args.over
catch_errors=args.catch_errors
image_band=args.image_band

lc_bin=args.lc_bin
lc_bands_str=args.lc_bands_str
hr_bands_str=args.hr_bands_str

grouptype=args.grouptype
heasoft_init_alias=args.heasoft_init_alias
caldbinit_init_alias=args.caldbinit_init_alias

use_file_coords=args.use_file_coords
target_only=args.mainfocus
rad_crop=args.rad_crop
bigger_fit=args.bigger_fit
point_source=args.point_source
max_rad_source=args.max_rad_source



'''''''''''''''''
''''FUNCTIONS''''
'''''''''''''''''

# switching off matplotlib plot displays unless with plt.show()
plt.ioff()

camlist = ['FPM1', 'FPM2']

process_obsdir_done = threading.Event()
extract_reg_done=threading.Event()
extract_lc_done=threading.Event()
extract_sp_done=threading.Event()

# function to remove (most) control chars
def _remove_control_chars(message):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)

def kev_to_PI(e_val):

    '''
    conversion from https://heasarc.gsfc.nasa.gov/docs/nustar/nustar_faq.html#pi_to_energy
    '''
    return round(e_val-1.6)/0.04

def set_var(spawn):
    '''
    Sets starting environment variables for NICER data analysis
    '''
    spawn.sendline(heasoft_init_alias)
    spawn.sendline(caldbinit_init_alias)


def file_evt_selector(filetype, camera='all', bright=False):
    '''
    Searches for all of the files of a specific type (among the ones used in the data reduction),
    and asks for input if more than one are detected.

    use "all" as input in camera to get the result for all 3 cameras (default value)

    Returns a single file + file path (not absolute) for each camera

    If the keyword bright is set to True, requires the "bright" keyword in the last directory of the event files
    '''

    # getting the list of files in the directory (note that file_evt_selector is launched while in the directory)
    flist = glob.glob('**', recursive=True)

    cameras = ['FPMA', 'FPMB']

    # list of accepted filetypes
    filetypes = ['evt_clean','src_reg','bg_reg']
    file_desc = ['clean event files','source region','background region']

    # getting the index of the file type for the keywords
    type_index = filetypes.index(filetype)

    # type keywords
    keyword_types = ['_cl.evt','src_reg.reg','bg_reg.reg']

    # camera keywords (1 for each type)
    camword_evt = ['A01', 'B01']
    keyword_cams = [camword_evt]

    # cam_list is the list of cameras to use for the evt search
    if camera == 'all':
        cam_list = cameras
    else:
        cam_list = [camera]

    result_list = []

    for cam in cam_list:
        # getting the index of the camera for the camera keywords
        cam_index = cameras.index(cam)

        # computing the list for the correct file type and camera
        cutlist = [elem for elem in flist if keyword_cams[type_index][cam_index] in elem\
                   and elem.endswith(keyword_types[type_index])\
                   and (1 if not bright else 'bright' in elem.split('/')[-2])]

        if len(cutlist) == 0:
            print('\nWarning : No ' + file_desc[type_index] + ' found for camera ' + cam)
            camdir = ['']
            camfile = ['']
        else:
            print('\n' + str(len(cutlist)) + ' exposure(s) found for ' + cam + ' ' + file_desc[type_index] + ' :')
            print(np.array(cutlist).transpose())

            # loop on all the events
            camdir = []
            camfile = []
            for i in range(len(cutlist)):
                elem = cutlist[i]

                # Storing the name and directory of the event files
                # if the files are not found in the local folder, we need to separate the path and the file name
                if elem.rfind('/') != -1:
                    camdir += [elem[:elem.rfind('/')]]
                    camfile += [elem[elem.rfind('/') + 1:]]
                else:
                    camdir += ['./']
                    camfile += [elem]

        result_list += [[camfile, camdir]]

    return result_list

def process_obsdir(directory, overwrite=True, bright=False):
    '''
    Processes a directory using the nupipeline script

    if the count rate is above 100 in lightcurves later, the 'bright' mode adds this keyword:
    statusexpr="(STATUS==b0000xxx00xxxx000)&&(SHIELD==0)"

    if bright is set to 'noshield', uses only statusexpr="(STATUS==b0000xxx00xxxx000)"

    (see https://heasarc.gsfc.nasa.gov/docs/nustar/nustar_faq.html#bright )
    '''

    bright_str= ' statusexpr="(STATUS==b0000xxx00xxxx000)&&(SHIELD==0)"' if bright else\
                ' statusexpr="STATUS==b0000xxx00xxxx000"' if bright=='noshield' else ''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    print('\n\n\nEvent filtering...')

    set_var(bashproc)

    if os.path.isfile(directory + '/process_obsdir.log'):
        os.system('rm ' + directory + '/process_obsdir.log')

    with StdoutTee(directory + '/process_obsdir.log', mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(directory + '/process_obsdir.log', buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        #note: for NuSTAR, we go in the directory because the script creates a bunch of temp files which could
        #conflict if we run several iterations together

        bashproc.sendline('cd '+directory)
        #note: nupipeline requires explicit indir, steminputs and outdir arguments
        bashproc.sendline('nupipeline indir=' + "./" + ' steminputs=nu'+directory+
                          ' outdir=./out'+('_bright' if bright!=False else '')+
                          ' clobber=' + ('YES' if overwrite else 'FALSE')+bright_str)

        #will need to update this in case of updates
        process_state = bashproc.expect(['nupipeline_0.4.9: Exit'], timeout=None)

        # exiting the bashproc
        bashproc.sendline('exit')
        process_obsdir_done.set()

        # raising an error to stop the process if the command has crashed for some reason
        if process_state == 0:
            raise ValueError
def disp_ds9(spawn, file, zoom='auto', scale='log', regfile='', screenfile='', give_pid=False, kill_last=''):
    '''
    Regfile is an input, screenfile is an output. Both can be paths
    If "screenfile" is set to a non empty str, we make a screenshot of the ds9 window in the given path
    This is done manually since the ds9 png saving command is bugged

    if give_pid is set to True, returns the pid of the newly created ds9 process
    '''

    if scale == 'linear 99.5':
        scale = 'mode 99.5'
    elif ' ' in scale:
        scale = scale.split(' ')[0] + ' mode ' + scale.split(' '[1])

    # if automatic, we set a higher zoom for timing images since they usually only fill part of the screen by default
    if zoom == 'auto':
        if 'Timing' in file and 'pn' in file:
            zoom = 4
        else:
            zoom = 1.67

    # region load command
    if regfile != '':
        regfile = '-region ' + regfile

    # parsing the open windows before and after the ds9 command to find the pid of the new ds9 window
    if screenfile != '' or give_pid:
        windows_before = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

    spawn.sendline(
        'echo "Ph0t1n0s" | sudo -S ds9 -view buttons no -cmap Heat -geometry 1080x1080 -scale ' + scale + ' -mode region ' + file + ' -zoom ' + str(
            zoom) +
        ' ' + regfile + ' &')

    # the timeout limit could be increased for slower computers or heavy images
    spawn.expect(['password', pexpect.TIMEOUT], timeout=1)

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

    # we purposely do this at the very end
    if kill_last != '':
        print('\nClosing previous ds9 window...')

        os.system('wmctrl -ic ' + kill_last)

    if give_pid:
        return ds9_pid

def reg_optimiser(mask):
    # for the shapely method :
    # from shapely.geometry import Polygon
    # from shapely.ops import polylabel as sh_polylabel
    # from shapely.validation import make_valid
    # Note: shapely.validation doesn't work with shapely 1.7.1, which is the standard version currently installed by conda/pip
    # manually install 1.8a3 so solve it

    '''
    Computes the biggest circle region existing inside of a mask using the polylabel algorithm
    See https://github.com/mapbox/polylabel

    This algorithms returns the point of inaccessibility, i.e. the point furthest from any frontier (hole or exterior) or a
    polygon

    Note : the pylabel algorithm has been ported twice in python.
    The shapely port is less powerful but can use wider input since it uses the polygon class of its own library in lieu of
    a list of arrays. Still, it's more complex to use so we'll use the other port as long as it works.

    Process :
    1. Transforming the mask into a set of polygons (shell and holes) with imantics
    2. swapping the array to have the exterior as the first element
    3. using polylabel

    Additional steps with shapely:
    2. Creating a valid polygon argument with shapely
    3. Computing the poi with the built-in polylabel of shapely
    4. computing the distance to get the maximal possible radius
    '''

    # creating a digestible input for the polygons function of imantics
    int_mask = mask.astype(int)

    # this function returns the set of polygons equivalent to the mask
    # the last polygon of the set should be the outer shell
    polygons = Mask(int_mask).polygons()

    # since we don't know the position of the outer shell (and there is sometimes no outer shell)
    # we'll consider the biggest polygon as the "main one".
    # It's easily identifiable as the ones with the biggest number of points
    # (since they are identified on pixel by pixel basis, there seems to be no "long" lines)

    shell_length = 0
    for i in range(len(polygons.points)):
        if len(polygons.points[i]) > shell_length:
            shell_id = i
            shell_length = len(polygons.points[i])

    # swapping the positions to have the shell as the first polygon in the array
    poly_args = polygons.points[:shell_id] + polygons.points[shell_id + 1:]
    poly_args.insert(0, polygons.points[shell_id])

    coords = polylabel(poly_args, with_distance=True)

    # second method (was coded before the first so let's keep it just in case)
    '''
    #Creating a usable polygon with the shapely library
    CCD_shape=Polygon(shell=CCD_polygons.points[-1],holes=CCD_polygons.poings[:-1])

    #This polygon can be invalid due to intersections and as such the polylabel function might not work
    if CCD_shape.is_valid:
        CCD_shape_valid=CCD_shape
    else:
        #in this case, we make it valid, which outputs a list of geometries instead of the polygon.
        #Thus we skim through the resulting geometries until we get our "good" polygon
        CCD_shape_valid=make_valid(CCD_shape)
        while CCD_shape_valid.type!='Polygon':
            CCD_shape_valid=CCD_shape_valid[0]

    #Now the polylabel function can finally be used
    shape_poi=sh_polylabel(CCD_shape_valid)

    #the resulting position is in image coordinates
    bg_ctr=[shape_poi.xy[0][0],shape_poi.xy[1][0]]

    #to get the maximal radius of the bg circle, we compute the distance to the polygon
    bg_maxrad=CCD_shape.exterior.distance(shape_poi)

    #Just in case, we also compare it manually with the distance to the holes (might be redundant)
    for elem in CCD_shape.interiors:
        if elem.distance(shape_poi)<bg_maxrad:
            bg_maxrad=elem.distance(shape_poi)

    coords=(bg_ctr,bg_maxrad)
    '''
    return coords

def imgarr_to_png(array, name, directory='./', astropy_wcs=None, mpdaf_wcs=None, title=None, imgtype=''):
    '''
    Separated since it might be of use elsewhere
    Note: astropy wcs do not work with mpdaf and mpdaf wcs do not work with matplotlib
    can mask the irrelevant parts of the image is the type is specified (ccd_crop or ccd_crop_mask)

    unfortunately the wcs projection doesn't work after masking the CCD
    Can probably be fixed by saving and reloading the fits wcs with astropy but it's not needed for now

    Note : there's something very weird going on with the masking as the ccd_crop mask always ends up being
    wider than the ccd_crop one, with no apparent reason (same masking region in the variables but it still doesn't mask
                                                          everything)
    To be fixed later
    '''

    if imgtype == 'ccd_crop':
        def line_irrel(line):
            return len(np.unique(np.isnan(line))) == 1 and np.isnan(line)[0]
    if imgtype == 'ccd_crop_mask':
        def line_irrel(line):
            return len(np.unique(line)) == 1 and line[0] == 0

    sep = ''
    if directory[-1] != '/':
        sep = '/'
    img = Image(data=array, wcs=mpdaf_wcs)

    # masking
    if imgtype == 'ccd_crop' or imgtype == 'ccd_crop_mask':

        crop_boxes = [[0, np.size(array.T, 0) - 1], [0, np.size(array.T, 1) - 1]]

        # finding the first/last row/column for which the array contains relevant data
        for i in range(np.size(array, 0)):
            if not line_irrel(array[i]) and crop_boxes[0][0] == 0:
                crop_boxes[0][0] = i
            if not line_irrel(array[-i - 1]) and crop_boxes[0][1] == np.size(array, 0) - 1:
                crop_boxes[0][1] = np.size(array, 0) - 1 - i

        for j in range(np.size(array.T, 0)):
            if not line_irrel(array.T[j]) and crop_boxes[1][0] == 0:
                crop_boxes[1][0] = j
            if not line_irrel(array.T[-j - 1]) and crop_boxes[1][1] == np.size(array.T, 0) - 1:
                crop_boxes[1][1] = np.size(array.T, 0) - 1 - j

        # creating the correct arguments for the mask_region method
        widths = (crop_boxes[0][1] - crop_boxes[0][0], crop_boxes[1][1] - crop_boxes[1][0])
        center = (crop_boxes[0][0] + widths[0] / 2, crop_boxes[1][0] + widths[1] / 2)

        img.mask_region(center=center, radius=widths, unit_center=None, unit_radius=None, inside=False)
        img.crop()

        proj = None
    else:
        proj = {'projection': astropy_wcs}

    fig, ax = plt.subplots(1, subplot_kw=proj, figsize=(12, 10))
    fig.suptitle(title)
    img.plot(cmap='plasma', scale='log', colorbar='v')

    # this line is just here to avoid the spyder warning

    fig.tight_layout()
    fig.savefig(directory + sep + name + '.png')
    plt.close(fig)

def xsel_img(bashproc,evt_path,save_path,e_low,e_high):

    '''
    Uses Xselect to create a NuSTAR image from bashproc

    e_low and e_high should be in keV

    save_path should be a relative path from the current directory of bashproc or an absolute path
    '''

    evt_dir='./' if '/' not in evt_path else evt_path[:evt_path.rfind('/')]
    evt_file= evt_path[evt_path.rfind('/')+1:]

    pha_low=kev_to_PI(e_low)
    pha_high=kev_to_PI(e_high)
    bashproc.sendline('xselect')
    bashproc.readline('XSELECT')

    #session name
    bashproc.sendline('')

    #reading events
    bashproc.sendline('read events')

    bashproc.readline('Event file dir')
    bashproc.sendline(evt_dir)

    bashproc.readline('Event file list')
    bashproc.readline(evt_file)

    #resetting mission
    bashproc.readline('Reset')
    bashproc.sendline('yes')

    #commands to prepare image creation
    bashproc.sendline('filter pha_cutoff '+pha_low+' '+pha_high)
    bashproc.sendline('set xybinsize 1')
    bashproc.sendline('extract image')
    bashproc.expect('Image')

    #commands to save image
    bashproc.sendline('save image')
    bashproc.expect('Give output file name')

    bashproc.sendline(save_path)
    bashproc.expect('Wrote image to ')

    bashproc.sendline('exit')
    bashproc.sendline('no')


def spatial_expression(coords):
    '''
    Returns the spatial ds9 expression for circular region coordinates of type ([center1,center2],radius),
    The expression itself will be in degrees
    '''

    return 'circle(' + coords[0][0] + ',' + coords[0][1] + ',' + coords[1] + '")'

def ds9_to_reg(ds9_regfile):
    '''
    Returns the coordinates of the circular regions listed in the ds9 file
    '''

    with open(ds9_regfile) as file:
        ds9_lines=file.readlines()

    reg_coords=[]
    for line in ds9_lines:
        if line.startswith('circle'):
            indiv_coords=line.split('(')[1].split(')')[0].split(',')

            reg_coords+=[[indiv_coords[0],indiv_coords[1]],indiv_coords[2]]
    return reg_coords

def extract_reg(directory, cams='all', use_file_coords=False,
                overwrite=True,e_low_img=3,e_high_img=79,rad_crop=rad_crop,bright=False):
    '''
    Extracts the optimal source/bg regions for a given exposure

    As of now, only takes input formatted through the evt_filter function

    Only accepts circular regions (in manual mode)
    '''

    def extract_reg_single(spawn, file, filedir):

        '''
        Individual region extraction for a single exposure and event file

        Significantly simplified version of the XMM_datared equivalent, because here we don't need so many options
        (only imaging, single CCD, no timing SNR optimisation, no pile-up,...) so we can compute the SNR optimization
        internally

        '''

        '''
        required functions
        '''
        def source_catal(dirpath, use_file_coords=False):

            '''
            Tries to identify a Simbad object from either the directory structure or the source name in the file itself

            If use_file_coords is set to True, does not produce cancel the process when Simbad
            doesn't recognize the source and uses the file coordinates instead

            '''

            # splitting the directories and searching every name in Simbad
            dir_list = dirpath.split('/')[1:]

            # removing a few problematic names
            crash_sources = ['M2', 'home', 'outputmos', 'BlackCAT', '']
            # as well as obsid type names that can cause crashes
            for elem_dir in dir_list:
                if len(elem_dir) == 10 and elem_dir.isdigit() or elem_dir in crash_sources:
                    dir_list.remove(elem_dir)

            # Simbad.query_object gives a warning for a lot of folder names so we just skip them
            obj_list = None
            for elem_dir in dir_list:
                try:
                    with warnings.catch_warnings():
                        # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
                        # warnings.filterwarnings('ignore','.*Identifier not found.*',)
                        warnings.filterwarnings('ignore', category=UserWarning)
                        elem_obj = Simbad.query_object(elem_dir)
                        if type(elem_obj) != type(None):
                            obj_list = elem_obj
                except:
                    breakpoint()
                    print('\nProblem during the Simbad query. This is the current directory list:')
                    print(dir_list)
                    spawn.sendline('\ncd $currdir')
                    return 'Problem during the Simbad query.'

            target_name = fits.open(dirpath + '/' + file)[0].header['OBJECT']
            try:
                with warnings.catch_warnings():
                    # warnings.filterwarnings('ignore','.*No known catalog could be found.*',)
                    # warnings.filterwarnings('ignore','.*Identifier not found.*',)
                    warnings.filterwarnings('ignore', category=UserWarning)
                    file_query = Simbad.query_object(target_name)
            except:
                print('\nProblem during the Simbad query. This is the current obj name:')
                print(dir_list)
                spawn.sendline('\ncd $currdir')
                return 'Problem during the Simbad query.'

            if obj_list is None:
                print("\nSimbad didn't recognize any object name in the directories." +
                      " Using the target of the observation instead...")
                obj_list = file_query

            if type(file_query) == type(None):
                print("\nSimbad didn't recognize the object name from the file header." +
                      " Using the name of the directory...")
                target_query = ''
            else:
                target_query = file_query[0]['MAIN_ID']

            if type(obj_list) == type(file_query) and type(obj_list) == type(None):

                print("\nSimbad couldn't detect an object name.")
                if not use_file_coords:
                    print("\nSkipping this observation...")
                    spawn.sendline('\ncd $currdir')

                return "Simbad couldn't detect an object name."

            # if we have at least one detections, it is assumed the "last" find is the name of the object
            obj_catal = obj_list[-1]

            print('\nValid name(s) detected. Object name assumed to be ' + obj_catal['MAIN_ID'])

            if obj_catal['MAIN_ID'] != target_query and target_only:
                print('\nTarget only mode activated and the source studied is not the main focus of the observation.' +
                      '\nSkipping...')
                spawn.sendline('\ncd $currdir')
                return 'Target only mode activated and the source studied is not the main focus of the observation.'

            return obj_catal

        def opti_bg_imaging():
            '''
            Now, we evaluate the background from the image
            Currently using the entirety of it and not just one CCD because it's much more convenient for bright images

            W load it with fits in order to perform image manipulation on it
            and compute how big the background region can be.

            The process is as follow :
            1. Compute the alpha shape of the non-zero pixels (i.e. polygon that surrounds them) in the CCD with alphashape
            2. Transform this polygon into a mask with rasterio
            3. delete the part outside of the ccd  (outlined by the polygon) by replacing the values outside with nans
                ->This step is important because even in the base image, all of the pixels outside of the CCD have 0 cts/s
                  They have to be changed to avoid wrong sigma clipping
            4. add sigma clipping to make sure the background region won't touch any source
            5. transform the background array into a mask
            6. compute the poi (see the reg_optimiser function)
            '''

            #currently using the first image file created
            with fits.open(img_file) as hdul:
                CCD_data = hdul[0].data

            CCD_img_obj=Image(data=CCD_data,wcs=src_mpdaf_WCS)

            if len(CCD_data.nonzero()[0]) == 0:
                print("\nEmpty bg image after CCD cropping. Skipping this observation...")
                spawn.sendline('\ncd $currdir')
                return "Empty bg image after CCD cropping."

            print('\nSaving the corresponding image...')
            imgarr_to_png(CCD_data, file_id+'_vis_' + '_CCD_1_crop', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=fulldir, title='Source image with CCDs cropped according to the region size and center')

            # listing non-zero pixels in the CCD
            CCD_on = np.argwhere(CCD_data != 0)

            # transforming that into a digestible argument for the alphashape function
            CCD_on = [tuple(elem) for elem in CCD_on]

            print('\nComputing the CCD mask...')

            '''
            Computation of the alphashape. Alpha defines how tight the polygon is around the points cloud.
            In most cases we use a conservative 0.1 value to avoid creating holes in our shape.
            '''

            CCD_shape = alphashape(CCD_on, alpha=0.1)

            # converting the polygon to a mask
            CCD_mask = rasterize([CCD_shape], out_shape=CCD_data.shape).T.astype(bool)


            print('\nSaving the corresponding image...')
            imgarr_to_png(CCD_data, file_id+'vis_CCD_2_mask', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=fulldir, title='Source image CCD(s) mask after clipping and filling of the holes')

            print('\nComputing the CCD masked image...')
            # array which we will have the outside of the CCD masked with nans
            CCD_data_cut = np.copy(CCD_data).astype(float)
            # This other array stores the values inside the CCD, for an easy evaluation of the sigma limits
            CCD_data_line = []
            for i in range(np.size(CCD_data, 0)):
                for j in range(np.size(CCD_data, 1)):
                    if not CCD_mask[i][j]:
                        CCD_data_cut[i][j] = np.nan
                    else:
                        CCD_data_line.append(CCD_data_cut[i][j])

            print('\nSaving the corresponding image...')
            imgarr_to_png(CCD_data_cut, file_id+'vis_CCD_3_cut', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=fulldir, title='Source image after CCD masking', imgtype='ccd_crop')

            # sigma cut, here at 0.95 (2 sigma) which seems to be a good compromise
            CCD_data_line.sort()

            # for some extreme cases we have only 1 count/pixel max, in which case we don't want that
            cut_sig = max(CCD_data_line[int(0.95 * len(CCD_data_line))], 1.)

            sigval = '2'
            perval = '5'

            # sometimes for very bright sources there might be too much noise so we cut at 1 sigma instead
            if cut_sig > 20:
                cut_sig = CCD_data_line[int(0.68 * len(CCD_data_line))]
                sigval = '1'
                perval = '32'

            print('\nComputing the CCD bg mask...')
            # array which will contain the background mask
            CCD_bg = np.copy(CCD_data_cut)
            for i in range(np.size(CCD_data, 0)):
                for j in range(np.size(CCD_data, 1)):
                    if CCD_data_cut[i][j] <= cut_sig:
                        CCD_bg[i][j] = 1
                    else:
                        # masking the pixels above the treshold
                        CCD_bg[i][j] = 0

            print('\nSaving the corresponding image...')
            imgarr_to_png(CCD_bg, file_id+'vis_CCD_4_bg', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=fulldir,
                    title='Source image background mask remaining after ' + sigval + ' sigma (top ' + perval +
                          '% cts) counts removal', imgtype='ccd_crop_mask')

            bg_max_pix = reg_optimiser(CCD_bg)

            print('\nMaximal bg region coordinates in pixel units:')
            print(bg_max_pix)


            '''
            finally, we convert the region coordinates and radius to angular values, using mpdaf.
            '''

            bg_center_radec=CCD_img_obj.get_start()+CCD_img_obj.get_axis_increments()*bg_max_pix[0]

            bg_max = [bg_center_radec.to_str(),
                      str(round(bg_max_pix[1] * CCD_img_obj.get_axis_increments()[0]*3600, 4))]

            return bg_max

        '''
        MAIN BEHAVIOR
        '''

        if file == '':
            print('\nNo evt to extract spectrum from for this camera in the obsid directory.')
            return 'No evt to extract spectrum from for this camera in the obsid directory.'

        fulldir = directory + '/' + filedir

        #note: only the file here
        file_id=file.replace('_cl.evt','')

        img_file=file_id+'_img_'+str(e_low_img)+'_'+str(e_high_img)+'.ds'

        #creating an image to load with mpdaf for image analysis
        xsel_img(spawn,os.path.join(filedir,file),os.path.join(filedir,img_file),e_low=e_low_img,e_high=e_high_img)

        spawn.sendline('\ncurrdir=$(pwd)')
        spawn.sendline('\ncd '+filedir)

        #opening the image file and saving it for verification purposes
        ds9_pid_sp_start=disp_ds9(spawn,img_file,screenfile=fulldir+'/'+img_file.replace('.ds','_screen.png'),
                        give_pid=True)

        try:
            fits_img = fits.open(fulldir + '/' + img_file)
        except:
            print("\nCould not load the image fits file. There must be a problem with the exposure." +
                  "\nSkipping spectrum computation...")
            spawn.sendline('\ncd $currdir')
            return "Could not load the image fits file. There must be a problem with the exposure."

        if ds9_pid_sp_start == 0:
            print("\nCould not load the image file with ds9. There must be a problem with the exposure." +
                  "\nSkipping spectrum computation...")
            spawn.sendline('\ncd $currdir')
            return "Could not load the image file with ds9. There must be a problem with the exposure."

        #loading the IMG file with mpdaf
        with fits.open(img_file) as hdul:
            img_data=hdul[0].data
            src_mpdaf_WCS=mpdaf_WCS(hdul[0].header)
            src_astro_WCS=astroWCS(fits_img[0].header)
            main_source_name=hdul[0].header['object']
            main_source_ra=hdul[0].header['RA_OBJ']
            main_source_dec=hdul[0].header['RA_DEC']

        print('\nAuto mode.')
        print('\nAutomatic search of the directory names in Simbad.')

        prefix = '_auto'

        obj_auto = source_catal(fulldir, use_file_coords=use_file_coords)

        # checking if the function returned an error message (folder movement done in the function)
        if type(obj_auto) == str:
            if not use_file_coords:
                return obj_auto
            else:
                obj_auto = {'MAIN_ID': main_source_name}
                obj_deg = [main_source_ra,main_source_dec]
        else:
            # careful the output after the first line is in dec,ra not ra,dec
            obj_deg = sexa2deg([obj_auto['DEC'].replace(' ', ':'), obj_auto['RA'].replace(' ', ':')])
            obj_deg = [str(obj_deg[1]), str(obj_deg[0])]

        img_obj_whole=Image(data=img_data,wcs=src_mpdaf_WCS)

        # saving a screen of the image with the cropping zone around the catalog position highlighted
        reg_catal_name = file_id + prefix+ '_reg_catal.reg'

        with open(os.path.join(fulldir,reg_catal_name), 'w+') as regfile:
            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression((obj_deg, str(rad_crop)), type='ds9')
                          + ' # text={' + obj_auto['MAIN_ID'] + ' initial cropping zone}')

        ds9_pid_sp_start = disp_ds9(spawn, img_file, regfile=reg_catal_name, zoom=1.2,
                                    screenfile=fulldir + '/' +file_id + prefix + '_reg_catal_screen.png',
                                    give_pid=True,
                                    kill_last=ds9_pid_sp_start)

        rad_crop_use=rad_crop

        try:
            imgcrop_src = img_obj_whole.copy().subimage(center=obj_deg[::-1], size=2 * rad_crop)
        except:
            print('\nCropping region entirely out of the image. Field of view issue. Skipping this exposure...')
            spawn.sendline('\ncd $currdir')
            return 'Cropping region entirely out of the image.'

        # masking the desired region
        imgcrop_src.mask_region(center=obj_deg[::-1], radius=rad_crop, inside=False)

        # testing if the resulting image is empty
        if len(imgcrop_src.data.nonzero()[0]) == 0:
            print('\nEmpty image after cropping. Field of view Issue. Skipping this exposure...')
            spawn.sendline('\ncd $currdir')
            return 'Cropped image empty.'

        # plotting and saving imgcrop for verification purposes (if the above has computed, it should mean the crop is in the image)
        fig_catal_crop, ax_catal_crop = plt.subplots(1, 1, subplot_kw={'projection': src_astro_WCS}, figsize=(12, 10))
        ax_catal_crop.set_title('Cropped region around the theoretical source position')
        catal_plot = imgcrop_src.plot(cmap='plasma', scale='sqrt')
        plt.colorbar(catal_plot, location='bottom', fraction=0.046, pad=0.04)
        plt.savefig(fulldir + '/' + file_id + prefix + '_catal_crop_screen.png')
        plt.close()

        # testing if the resulting image contains a peak
        if imgcrop_src.peak() == None:
            print('\nNo peak detected in cropped image. Skipping this exposure...')
            spawn.sendline('\ncd $currdir')
            return 'No peak in cropped image.'

        # fitting a gaussian on the source (which is assumed to be the brightest in the cropped region)
        # the only objective here is to get the center, the radius will be computed from the SNR
        print('\nExecuting gaussian fit...')

        if point_source:
            source_center = (imgcrop_src.peak()['y'], imgcrop_src.peak()['x'])
        else:
            source_center = None
        gfit = imgcrop_src.gauss_fit(center=source_center)
        gfit.print_param()
        # defining various bad flags on the gfit
        if np.isnan(gfit.fwhm[0]) or np.isnan(gfit.fwhm[1]) or np.isnan(
                gfit.err_peak) or gfit.peak < 10 or gfit.err_peak >= gfit.peak:
            print('\nGaussian fit failed. Positioning or Field of View issue. Skipping these Evts...')
            spawn.sendline('\ncd $currdir')
            return 'Gaussian fit failed. Positioning or Field of View issue.'

        if not imgcrop_src.inside(gfit.center):
            print('\nGaussian fit centroid out of the crop zone. Wrong fit expected. Skipping these Evts...')
            spawn.sendline('\ncd $currdir')
            return 'Gaussian fit centroid further than ' + str(rad_crop) + '" from catal source position.'

        # if the source is bright and wide, the gaussian is probably going to be able to fit it even if we increase the cropping region
        # And the fit might need a bigger image to compute correctly

        if max(gfit.peak, imgcrop_src.data.max()) > 1000 and bigger_fit and max(gfit.fwhm) > 30:

            rad_crop_use = 2 * rad_crop
            # new, bigger cropping the image to avoid zoom in the future plots
            # (size is double the radius since we crop at the edges of the previously cropped circle)
            imgcrop_src = img_obj_whole.copy().subimage(center=obj_deg[::-1], size=2 * rad_crop)

            # masking a bigger region
            imgcrop_src.mask_region(center=obj_deg[::-1], radius=rad_crop, inside=False)

            # fitting a gaussian on the source (which is assumed to be the brightest in the cropped region)
            print('\nExecuting gaussian fit...')
            gfit = imgcrop_src.gauss_fit()
            gfit.print_param()

            if np.isnan(gfit.fwhm[0]) or np.isnan(gfit.fwhm[1]):
                print(
                    '\nExtended (bright source) gaussian fit failed. Positioning or Field of View issue. Skipping these Evts...')
                spawn.sendline('\ncd $currdir')
                return 'Extended (bright source) gaussian fit failed. Positioning or Field of View issue.'

            if not imgcrop_src.inside(gfit.center):
                print('\nGaussian fit centroid out of the crop zone. Wrong fit expected. Skipping these Evts...')
                spawn.sendline('\ncd $currdir')
                return 'Gaussian fit centroid further than ' + str(2.5 * rad_crop) + '" from catal source position.'

        bg_coords_im = opti_bg_imaging()

        # returning the error message if there is one instead of the expected values (directory change done in function)
        if type(bg_coords_im) == str:
            spawn.sendline('\ncd $currdir')
            return bg_coords_im

        '''                   
        The SNR optimisation is very simple here since we don't optimize the LC binning.
         
         We use the following steps:

        1. Compute a range of circular regions around the gfit center with increasing radius up to the size of the 
        inital image crop

        for each:
            a. Compute the number of source counts in this region
            b. Compute the SNR compared to the (constant) counts in the BG region

        The region with the best SNR in the allowed range is retained
        (range limited to a certain number of PSF sigmas for faint sources, 
        not limited for bright -i.e. with gfit flux > 5000 ones)
        '''

        #computing the encircled number of counts in the background region
        counts_bg=img_obj_whole.ee((bg_coords_im[0][::-1]),radius=float(bg_coords_im[1]))

        #computing the SNR for a range of radiuses

        rad_test_arr=np.arange(5,min(rad_crop,max(gfit.fwhm)*max_rad_source*2.355),2)

        snr_vals=np.repeat(0,len(rad_test_arr))

        for id_rad,rad in enumerate(rad_test_arr):

            counts_src=img_obj_whole.ee(gfit.center,radius=rad_test_arr)

            backscale=(rad/float(bg_coords_im[1]))**2
            snr_vals[id_rad] = (counts_src - counts_bg* backscale / (counts_src + counts_bg * backscale + 1e-10))\
                                ** (1 / 2)

        rad_max_snr=rad_test_arr[np.argmax(snr_vals)]

        # summary variable, first version which the background will be computed from, with a radius of 3sigmas of the gaussian fit
        src_coords = [[str(gfit.center[1]), str(gfit.center[0])], str(round(rad_max_snr,4))]

        '''
        Now we can put both of the regions in a ds9 file using the standard format, for visualisation
        '''

        reg_name = file_id+ prefix + '_reg.reg'

        with open(os.path.join(fulldir,reg_name), 'w+') as regfile:

            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression(src_coords)
                          + ' # text={' + obj_auto['MAIN_ID'] + '}' +
                          '\n' + spatial_expression(bg_coords_im)
                          + ' # text={automatic background}' )

        ds9_pid_sp_reg = disp_ds9(spawn, img_file, regfile=reg_name,
                                    screenfile=fulldir + '/' +file_id + prefix + 'reg_screen.png',
                                    give_pid=True,
                                    kill_last=ds9_pid_sp_start)

        '''
        and in individual region files for the extraction
        '''

        reg_src_name=file_id+ prefix + '_reg_src.reg'
        reg_bg_name=file_id+ prefix + '_reg_bg.reg'

        with open(os.path.join(fulldir,reg_src_name), 'w+') as regfile:

            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression(bg_coords_im)
                          + ' # text={automatic background}' )

        with open(os.path.join(fulldir,reg_bg_name), 'w+') as regfile:

            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression(src_coords)
                          + ' # text={' + obj_auto['MAIN_ID'] + '}')

        return 'Region extraction complete.'

    '''MAIN BEHAVIOR'''

    if cams == 'all':
        camid_list= [0, 1]
    else:
        camid_list = []
        if 'FPMA' in [elem.upper() for elem in cams]:
            camid_list.append(0)
        if 'FPMB' in [elem.upper() for elem in cams]:
            camid_list.append(1)

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    print('\n\n\nRegion extraction...')

    bashproc.sendline('cd ' + directory)
    set_var(bashproc, directory)

    # recensing the cleaned event files available for each camera
    # clean_filelist shape : [[FPMA_files,FPMA_dirs],[FPMB_files,FPMB_dirs]]
    clean_filelist = file_evt_selector('evt_clean',cameras=cams,bright=bright)

    # summary file header
    if directory.endswith('/'):
        obsid = directory.split('/')[-2]
    else:
        obsid = directory.split('/')[-1]

    summary_header = 'Obsid\tFile identifier\tRegion extraction result\n'

    bashproc.logfile_read = sys.stdout

    # filtering for the selected cameras
    for i_cam in camid_list:

        for i_exp in range(len(clean_filelist[i_cam][0])):

            clean_evtfile = clean_filelist[i_cam][0][i_exp]
            clean_evtdir = clean_filelist[i_cam][1][i_exp]

            '''
            testing if the last file of the process (the ds9 screen file with the regions)
            has been created
            '''

            lastfile_extract_reg = clean_evtfile.replace('_cl.evt','_reg_screen.png')

            if clean_evtfile=='':
                print('\nNo evt to extract region from for camera ' + camlist[i_cam] + ' in the obsid directory.')

                summary_line = 'No evt to extract region from for camera ' + camlist[i_cam] + ' in the obsid directory.'
                clean_evtid = camlist[i_cam]

            elif overwrite or not os.path.isfile(lastfile_extract_reg):
                clean_evtid = clean_evtfile.split('.')[0].replace('clean', '')

                # setting up a logfile in parallel to terminal display :

                if os.path.isfile(clean_evtdir + '/' + clean_evtid + '_extract_reg.log'):
                    os.system('rm ' + clean_evtdir + '/' + clean_evtid + '_extract_reg.log')
                with StdoutTee(clean_evtdir + '/' + clean_evtid + '_extract_reg.log',
                               mode="a", buff=1, file_filters=[_remove_control_chars]), \
                        StderrTee(clean_evtdir + '/' + clean_evtid + '_extract_reg.log', buff=1,
                                  file_filters=[_remove_control_chars]):

                    bashproc.logfile_read = sys.stdout
                    print('\nComputing region of ' + camlist[i_cam] + ' exposure ' + clean_evtfile)

                    # launching the main extraction
                    summary_line = extract_reg_single(bashproc, clean_evtfile, clean_evtdir)

            else:
                print('\nRegion computation for the ' + camlist[i_cam] + ' exposure ' + clean_evtfile +
                      ' already done. Skipping...')
                summary_line = ''

            if summary_line != '':
                summary_content = obsid + '\t' + clean_evtid + '\t' + summary_line
                file_edit(os.path.join(directory, 'summary_extract_reg.log'), obsid + '\t' + clean_evtid,
                          summary_content + '\n',
                          summary_header)

    bashproc.sendline('\necho "Ph0t1n0s" |sudo -S pkill sudo')
    # this sometimes doesn't proc before the exit for whatever reason so we add a buffer just in case
    # bashproc.expect([pexpect.TIMEOUT],timeout=2)

    # closing the spawn
    bashproc.sendline('exit')

    print('\nRegion extraction of the current obsid directory events finished.')

    extract_reg_done.set()

def extract_lc(directory,binning='1',lc_bands='3-79',hr_bands='10-50/3-10',cams='all',bright=False):

    '''
    Wrapper for a version of nuproducts to computes only lightcurves in the desired bands,
    with added matplotlib plotting of requested lightcurves and HRs

    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nustar/analysis/nustar_swguide.pdf 5.3D
    options:
        -binning: binning of the LC in seconds

        -bands: bands for each lightcurve to be created.
                The numbers should be in keV, separated by "-", and different lightcurves by ","
                ex: to create two lightcurves for, the 1-3 and 4-12 band, use '1-3,4-12'

        -hr: bands to be used for the HR plot creation.
             A single plot is possible for now. Creates its own lightcurve bands if necessary

        -overwrite: overwrite products or not

    NOTE THAT THE BACKSCALE CORRECTION IS APPLIED MANUALLY
    '''

    '''MAIN BEHAVIOR'''

    def extract_lc_single(spawn, directory, binning, instru, steminput, src_reg, bg_reg, e_low, e_high,bright=False,backscale=1):

        lc_src_name = steminput + '_' + instru + '_lc_src_' + e_low + '_' + e_high + '_bin_' + binning + '.lc'
        lc_bg_name = steminput + '_' + instru + '_lc_bg_' + e_low + '_' + e_high + '_bin_' + binning + '.lc'

        lc_src_path = os.path.join(directory,'products'+('_bright' if bright else ''), lc_src_name)
        lc_bg_path = os.path.join(directory,'products'+('_bright' if bright else ''), lc_bg_name)

        pi_low = str(kev_to_PI(float(e_low)))
        pi_high = str(kev_to_PI(float(e_high)))

        # building the lightcurve
        spawn.sendline('nuproducts indir' + directory + ' instrument=' + instru + ' steminputs=' + steminput +
                       ' lcfile=' + lc_src_name +  ' srcregionfile=' + src_reg +
                       ' bkglcfile=' + lc_bg_name + ' bkgregionfile=' + bg_reg +
                       ' pilow=' + pi_low + ' pihigh=' + pi_high + ' binsize=' + binning+
                       ' outdir=./products'+('_bright' if bright else '')+
                       ' phafile=NONE bkgphafile=NONE' + ' imagefile=NONE runmkarf=no runmkrmf=no ')

        ####TODO: check what's the standard message here
        spawn.expect('complete')

        # loading the data of both lc
        with fits.open(lc_src_path) as fits_lc:
            # time zero of the lc file (different from the time zeros of the gti files)
            time_zero = Time(fits_lc[1].header['MJDREFI'] + fits_lc[1].header['MJDREFF'], format='mjd')

            # and offsetting the data array to match this
            delta_lc_src = fits_lc[1].header['TIMEZERO']

            fits_lc[1].data['TIME'] += delta_lc_src

            # storing the shifted lightcurve
            data_lc_src = fits_lc[1].data

            time_zero_str = str(time_zero.to_datetime())

        with fits.open(lc_bg_path) as fits_lc:
            # and offsetting the data array to match this
            delta_lc_bg = fits_lc[1].header['TIMEZERO']

            fits_lc[1].data['TIME'] += delta_lc_bg

            # storing the shifted lightcurve
            data_lc_bg = fits_lc[1].data

        # plotting the source and bg lightcurves together

        fig_lc, ax_lc = plt.subplots(1, figsize=(10, 8))

        ax_lc.errorbar(data_lc_src['TIME'], data_lc_src['RATE'], xerr=float(binning),
                     yerr=data_lc_src['ERROR'], ls='-', lw=1, color='grey', ecolor='blue', label='raw source')

        ax_lc.errorbar(data_lc_bg['TIME'], data_lc_bg['RATE']*backscale, xerr=float(binning),
                     yerr=data_lc_bg['ERROR']*backscale, ls='-', lw=1, color='grey', ecolor='brown', label='scaled background')

        plt.suptitle('NuSTAR ' + instru + ' lightcurve for observation ' + steminput +
                     ' in the ' + e_low + '-' + e_high + ' keV band with ' + binning + ' s binning')

        ax_lc.set_xlabel('Time (s) after ' + time_zero_str)
        ax_lc.set_ylabel('RATE (counts/s)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(directory,'products'+('_bright' if bright else ''),
                                 steminput + '_' + instru + '_lc_screen_' + e_low + '_' + e_high + '_bin_' + binning + '.png'))
        plt.close()

        return 'Lightcurve creation complete',[time_zero_str,data_lc_src['TIME'],data_lc_src['RATE']-data_lc_bg['RATE']*backscale,
                                                data_lc_src['ERROR']+data_lc_bg['ERROR']*backscale]

    if cams == 'all':
        camid_list= [0, 1]
    else:
        camid_list = []
        if 'FPMA' in [elem.upper() for elem in cams]:
            camid_list.append(0)
        if 'FPMB' in [elem.upper() for elem in cams]:
            camid_list.append(1)

    # recensing the reg files available for each camera
    # clean_filelist shape : [[FPMA_files,FPMA_dirs],[FPMB_files,FPMB_dirs]]
    src_reg = file_evt_selector('src_reg',cameras=cams,bright=bright)
    bg_reg = file_evt_selector('src_reg',cameras=cams,bright=bright)

    if len(src_reg[0][0])!=1 or len(src_reg[1][0])!=1 or len(bg_reg[0][0])!=1 or len(bg_reg[1][0])!=1:

        print('Issue: missing/too many region files detected')
        print(src_reg)
        print(bg_reg)
        return 'Issue: missing/too many region files detected'

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    print('\n\n\nCreating lightcurves products...')

    # defining the number of lightcurves to create

    # decomposing for each band asked
    lc_bands = ([] if hr_bands is None else ravel_ragged([elem.split('/') for elem in hr_bands.split(',')]).tolist())\
               + lc_bands.split(',')

    lc_bands = np.unique(lc_bands)[::-1]

    # storing the ids for the HR bands
    id_band_num_HR = np.argwhere(hr_bands.split('/')[0] == lc_bands)[0][0]
    id_band_den_HR = np.argwhere(hr_bands.split('/')[1] == lc_bands)[0][0]

    summary_header='Obsid\tcamera\tenergy band\tLightcurve extraction result\n'

    set_var(bashproc)

    if os.path.isfile(directory + '/extract_lc.log'):
        os.system('rm ' + directory + '/extract_lc.log')

    with StdoutTee(directory + '/extract_lc.log', mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(directory + '/extract_lc.log', buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        # filtering for the selected cameras
        for i_cam in camid_list:

            if directory.endswith('/'):
                obsid = directory.split('/')[-2]
            else:
                obsid = directory.split('/')[-1]

            bashproc.logfile_read = sys.stdout
            print('\nComputing lightcurves of camera ' + camlist[i_cam])

            #fetching region files for this camera
            src_reg_indiv=src_reg[i_cam][0]
            bg_reg_indiv=bg_reg[i_cam][0]

            #computing the backscale
            src_reg_coords=ds9_to_reg(os.path.join(src_reg_indiv))
            bg_reg_coords=ds9_to_reg(os.path.join(bg_reg_indiv))

            backscale=(float(src_reg_coords[1])/float(bg_reg_coords[1]))**2

            # launching the main extraction

            evt_dir=os.path.join(directory,'out'+('' if not bright else '_bright'))
            lc_prods=[]
            for band in lc_bands:

                summary_line,lc_prods = extract_lc_single(bashproc,directory=evt_dir,binning=binning,instru=camlist[i_cam],
                                                 steminput='nu'+obsid,src_reg=src_reg_indiv,
                                                 bg_reg=bg_reg_indiv,e_low=band.split('-')[0],e_high=band.split('-')[1],
                                                 bright=bright,backscale=backscale)

                summary_content = obsid + '\t' + camlist[i_cam] +'\t'+ band + '\t' + summary_line
                file_edit(os.path.join(directory, 'summary_extract_lc.log'), obsid + '\t' + camlist[i_cam] +'\t'+ band ,
                          summary_content + '\n',
                          summary_header)


            assert lc_prods[id_band_den_HR][0]==lc_prods[id_band_num_HR][0], 'Differing timezero values between HR lightcurves'

            time_zero_HR=lc_prods[id_band_num_HR][0]

            #here we implicitely assume the time array is identical for both lightcurves and for source/bg
            time_HR=lc_prods[id_band_num_HR][1]

            rate_num_HR=lc_prods[id_band_num_HR][2]
            rate_err_num_HR=lc_prods[id_band_num_HR][3]

            rate_den_HR=lc_prods[id_band_den_HR][2]
            rate_err_den_HR=lc_prods[id_band_den_HR][3]

            fig_hr, ax_hr = plt.subplots(1, figsize=(10, 8))

            hr_vals = rate_num_HR /rate_den_HR

            hr_err = hr_vals * (((rate_err_num_HR / rate_num_HR) ** 2 +
                                 (rate_err_den_HR / rate_den_HR) ** 2) ** (
                                            1 / 2))

            plt.errorbar(time_HR, hr_vals, xerr=binning, yerr=hr_err, ls='-', lw=1,
                         color='grey', ecolor='blue')

            plt.suptitle('NuSTAR '+camlist[i_cam]+' net HR evolution for observation ' + obsid +' in the ' + hr_bands + ' keV band'+
                         'with '+binning+' s binning')

            plt.xlabel('Time (s) after ' + time_zero_HR)
            plt.ylabel('Hardness Ratio (' + hr_bands + ' keV)')

            plt.tight_layout()
            plt.savefig(os.path.join(directory,'products'+('_bright' if bright else ''),
                        'nu'+obsid + '_' + camlist[i_cam]+ '_hr_screen_'+hr_bands.replace('/','_')+'_bin_' + binning + '.png'))
            plt.close()

    extract_lc_done.set()

def extract_sp(directory,cams='all',e_low=None,e_high=None,bright=False):

    '''
    Wrapper for a version of nuproducts to computes only spectral products

    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nustar/analysis/nustar_swguide.pdf 5.3B
    options:
        -binning: binning of the LC in seconds

        -bands: bands for each lightcurve to be created.
                The numbers should be in keV, separated by "-", and different lightcurves by ","
                ex: to create two lightcurves for, the 1-3 and 4-12 band, use '1-3,4-12'

        -hr: bands to be used for the HR plot creation.
             A single plot is possible for now. Creates its own lightcurve bands if necessary

        -overwrite: overwrite products or not

    Note: can produce no output without error if no gti in the event file
    '''

    '''MAIN BEHAVIOR'''

    def extract_sp_single(spawn, directory, instru, steminput, src_reg, bg_reg, e_low=None, e_high=None,bright=False):

        if e_low!=None:
            pi_low = str(kev_to_PI(float(e_low)))

        if e_high!=None:
            pi_high = str(kev_to_PI(float(e_high)))

        # building the spectral products
        spawn.sendline('nuproducts indir' + directory + ' instrument=' + instru + ' steminputs=' + steminput +
                       ' srcregionfile=' + src_reg +' bkgregionfile=' + bg_reg +
                       +('' if e_low==None else' pilow=' + pi_low)+
                       +('' if e_high==None else' pihigh=' + pi_high)+
                       ' outdir=./products'+('_bright' if bright else '')+
                       ' lcfile=NONE bkglcfile=None imagefile=NONE')

        ####TODO: check what's the standard message here
        spawn.expect('complete')

        return 'Spectral products creation complete'

    if cams == 'all':
        camid_list= [0, 1]
    else:
        camid_list = []
        if 'FPMA' in [elem.upper() for elem in cams]:
            camid_list.append(0)
        if 'FPMB' in [elem.upper() for elem in cams]:
            camid_list.append(1)

    # recensing the reg files available for each camera
    # clean_filelist shape : [[FPMA_files,FPMA_dirs],[FPMB_files,FPMB_dirs]]
    src_reg = file_evt_selector('src_reg',cameras=cams,bright=bright)
    bg_reg = file_evt_selector('src_reg',cameras=cams,bright=bright)

    if len(src_reg[0][0]!=1) or len(src_reg[0][0]!=1):

        print('Issue: missing/too many region files detected')
        print(src_reg)
        print(bg_reg)
        return 'Issue: missing/too many region files detected'

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    print('\n\n\nCreating spectral products...')

    summary_header='Obsid\tcamera\tSpectrum extraction result\n'

    set_var(bashproc)

    if os.path.isfile(directory + '/extract_sp.log'):
        os.system('rm ' + directory + '/extract_sp.log')

    with StdoutTee(directory + '/extract_sp.log', mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(directory + '/extract_sp.log', buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        # filtering for the selected cameras
        for i_cam in camid_list:

            if directory.endswith('/'):
                obsid = directory.split('/')[-2]
            else:
                obsid = directory.split('/')[-1]

            bashproc.logfile_read = sys.stdout
            print('\nComputing lightcurves of camera ' + camlist[i_cam])

            #fetching region files for this camera
            src_reg_indiv=src_reg[i_cam]
            bg_reg_indiv=bg_reg[i_cam]

            # launching the main extraction

            evt_dir=os.path.join(directory,'out'+('' if not bright else '_bright'))

            summary_line,lc_prods = extract_sp_single(bashproc,directory=evt_dir,instru=camlist[i_cam],
                                             steminput='nu'+obsid,src_reg=src_reg_indiv,
                                             bg_reg=bg_reg_indiv,e_low=e_low,e_high=e_high,bright=bright)

            summary_content = obsid + '\t' + camlist[i_cam] +'\t'+ summary_line
            file_edit(os.path.join(directory, 'summary_extract_lc.log'), obsid + '\t' + camlist[i_cam],
                      summary_content + '\n',
                      summary_header)

    extract_sp_done.set()

'''''''''''''''''''''
''''MAIN PROCESS'''''
'''''''''''''''''''''

if load_functions:
    breakpoint()

#### MAIN PROCESS

#getting started by choosing the correct directory and storing it for future use
if not os.path.isdir(startdir):
    logging.error('Invalid directory, please start over.')
    sys.exit()
else:
    os.chdir(startdir)

startdir=os.getcwd()

#listing all of the subdirectories
subdirs=glob.glob('**/',recursive=True)
subdirs.sort()

#summary header for the previously computed directories file
summary_folder_header='Subdirectory\tAnalysis state\n'

def startdir_state(action):
    #fetching the previously computed directories from the summary folder file
    try:
        with open('summary_folder_analysis_'+action+'.log') as summary_folder:
            launched_folders=summary_folder.readlines()

            #restricting to folders with completed analysis
            completed_folders=[elem.split('\t')[0] for elem in launched_folders if 'Done' in elem]
            launched_folders=[elem.split('\t')[0] for elem in launched_folders]
    except:
        launched_folders=[]
        completed_folders=[]

    return launched_folders,completed_folders

if load_functions:
    breakpoint()


started_folders,done_folders=startdir_state(args.action)

if not local:
    #checking them in search for ODF directories
    for directory in subdirs:

        #continue check
        if directory in started_folders[:-2] and folder_cont:
            print('Directory '+directory+' is not among the last two directories. Skipping...')
            continue

        #directory overwrite check
        if directory in done_folders and folder_over==False:
            print('Actions already computed for directory '+directory+'\nSkipping...')
            continue

        #getting the last directory in the result (necessary if osbid folders are in subdirectories) (we take the last / of to avoid problems)
        dirname=directory[:-1].split('/')[-1]

        #checking if the directory has an obsid dir shape (10 numbers)
        #and not a general or subdirectory
        if len(dirname)==10 and dirname.isdigit() and "odf" not in os.listdir(directory):

            print('\nFound obsid directory '+dirname)


            above_obsdir=os.path.join(startdir,'/'.join(directory[-1].split('/')[:-1]))

            os.chdir(above_obsdir)

#             if catch_errors:
#                 try:
#                 #for loop to be able to use different orders if needed
#                     for curr_action in action_list:
#
#                         #resetting the error string message
#                         output_err=None
#                         folder_state='Running '+curr_action
#
#                         if curr_action=='1':
#                             process_obsdir(dirname,overwrite=overwrite_glob)
#                             process_obsdir_done.wait()
#
#
#                         if curr_action=='2':
#                             select_detector(dirname,detectors=bad_detectors)
#                             select_detector_done.wait()
#
#                         if curr_action=='gti':
#                             output_err=create_gtis(dirname,split=gti_split,band=gti_lc_band,binning=lc_bin,
#                                         overwrite=overwrite_glob,flare_method=flare_method)
#                             if type(output_err)==str:
#                                 raise ValueError
#                             create_gtis_done.wait()
#
#                         if curr_action=='fs':
#                             output_err=extract_all_spectral(dirname,bkgmodel=bgmodel,language=bglanguage,overwrite=overwrite_glob)
#                             if type(output_err)==str:
#                                 raise ValueError
#                             extract_all_spectral_done.wait()
#
#                         if curr_action=='l':
#                             output_err=extract_lc(dirname,binning=lc_bin,bands=lc_bands_str,HR=hr_bands_str,overwrite=overwrite_glob)
#                             if type(output_err)==str:
#                                 raise ValueError
#                             extract_lc_done.wait()
#
#                         if curr_action=='s':
#                             extract_spectrum(dirname)
#                             extract_spectrum_done.wait()
#
#                         if curr_action=='b':
#                             extract_background(dirname,model=bgmodel)
#                             extract_background_done.wait()
#                         if curr_action=='r':
#                             extract_response(dirname)
#                             extract_response_done.wait()
#
#                         if curr_action=='g':
#                             output_err=regroup_spectral(dirname,group=grouptype)
#                             if type(output_err)==str:
#                                 raise ValueError
#                             regroup_spectral_done.wait()
#
#                         if curr_action=='m':
#                             batch_mover(dirname)
#                             batch_mover_done.wait()
#
#                         if curr_action=='c':
#                             clean_products(dirname)
#                             clean_products_done.wait()
#
#                         if curr_action=='fc':
#                             clean_all(dirname)
#                             clean_all_done.wait()
#
#                         os.chdir(startdir)
#                     folder_state='Done'
#
#                 except:
#                     #signaling unknown errors if they happened
#                     if 'Running' in folder_state:
#                         print('\nError while '+folder_state)
#                         folder_state=folder_state.replace('Running','Aborted at')+('' if output_err is None else ' --> '+output_err)
#                     os.chdir(startdir)
#             else:
#                 #for loop to be able to use different orders if needed
#                 for curr_action in action_list:
#                     folder_state='Running '+curr_action
#                     if curr_action=='1':
#                         process_obsdir(dirname,overwrite=overwrite_glob)
#                         process_obsdir_done.wait()
#                     if curr_action=='2':
#                         select_detector(dirname,detectors=bad_detectors)
#                         select_detector_done.wait()
#
#                     if curr_action=='gti':
#                         output_err=create_gtis(dirname,split=gti_split,band=gti_lc_band,binning=lc_bin,
#                                     overwrite=overwrite_glob,flare_method=flare_method)
#                         if type(output_err) == str:
#                             folder_state=output_err
#                         else:
#                             pass
#                         create_gtis_done.wait()
#
#                     if curr_action == 'fs':
#                         output_err = extract_all_spectral(dirname, bkgmodel=bgmodel, language=bglanguage,
#                                                           overwrite=overwrite_glob)
#                         if type(output_err) == str:
#                             folder_state=output_err
#                         else:
#                             pass
#                         extract_all_spectral_done.wait()
#
#                     if curr_action == 'l':
#                         output_err = extract_lc(dirname, binning=lc_bin, bands=lc_bands_str, HR=hr_bands_str,
#                                                 overwrite=overwrite_glob)
#                         if type(output_err) == str:
#                             folder_state=output_err
#                         else:
#                             pass
#                         extract_lc_done.wait()
#
#                     if curr_action=='s':
#                         extract_spectrum(dirname)
#                         extract_spectrum_done.wait()
#                     if curr_action=='b':
#                         extract_background(dirname,model=bgmodel)
#                         extract_background_done.wait()
#                     if curr_action=='r':
#                         extract_response(dirname)
#                         extract_response_done.wait()
#
#                     if curr_action=='g':
#                         output_err=regroup_spectral(dirname,group=grouptype)
#
#                         if type(output_err) == str:
#                             folder_state=output_err
#                         else:
#                             pass
#                         regroup_spectral_done.wait()
#
#                     if curr_action=='m':
#                         batch_mover(dirname)
#                         batch_mover_done.wait()
#
#                     if curr_action=='c':
#                         clean_products(dirname)
#                         clean_products_done.wait()
#
#                     if curr_action=='fc':
#                         clean_all(dirname)
#                         clean_all_done.wait()
#
#                     os.chdir(startdir)
#                 folder_state='Done'
#
#             #adding the directory to the list of already computed directories
#             file_edit('summary_folder_analysis_'+args.action+'.log',directory,directory+'\t'+folder_state+'\n',summary_folder_header)
#
# else:
#     #taking of the merge action if local is set since there is no point to merge in local (the batch directory acts as merge)
#     action_list=[elem for elem in action_list if elem!='m']
#
#     absdir=os.getcwd()
#
#     #just to avoid an error but not used since there is not merging in local
#     obsid=''
#
#     #for loop to be able to use different orders if needed
#     for curr_action in action_list:
#             if curr_action=='1':
#                 process_obsdir(absdir,overwrite=overwrite_glob)
#                 process_obsdir_done.wait()
#             if curr_action=='2':
#                 select_detector(absdir,detectors=bad_detectors)
#                 select_detector_done.wait()
#
#             if curr_action=='gti':
#                 output_err = create_gtis(absdir, split=gti_split, band=gti_lc_band, binning=lc_bin,
#                                          overwrite=overwrite_glob,flare_method=flare_method)
#                 create_gtis_done.wait()
#
#             if curr_action=='s':
#                 extract_spectrum(absdir)
#                 extract_spectrum_done.wait()
#             if curr_action=='b':
#                 extract_background(absdir,model=bgmodel)
#                 extract_background_done.wait()
#             if curr_action=='r':
#                 extract_response(absdir)
#                 extract_response_done.wait()
#             if curr_action=='g':
#                 regroup_spectral(absdir,group=grouptype)
#                 regroup_spectral_done.wait()
#             if curr_action=='m':
#                 batch_mover(absdir)
#                 batch_mover_done.wait()
#             if curr_action == 'c':
#                 clean_products(absdir)
#                 clean_products_done.wait()
#
#             if curr_action == 'fc':
#                 clean_all(absdir)
#                 clean_all_done.wait()
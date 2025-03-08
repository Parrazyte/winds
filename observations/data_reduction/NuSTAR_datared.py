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
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# note: might need to install opencv-python-headless to avoid dependencies issues with mpl

# import matplotlib.cm as cm
from matplotlib.collections import LineCollection

#using agg because qtagg still generates backends with plt.ioff()
mpl.use('agg')
plt.ioff()

# astro imports
from astropy.time import Time,TimeDelta
from astropy.io import fits
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

from general_tools import file_edit, ravel_ragged,MinorSymLogLocator,interval_extract,str_orbit,source_catal

"""
Created on 09-11-2023

Data reduction Script for NuSTAR Observations

Searches for all NuSTAR Obs type directories in the subdirectories and launches the process for each

list of possible actions : 

build.      process_obsdir: run the nupipeline script to process an obsid folder.

reg.    extract_reg: compute an image and an optimized source/bg region automatically to maximize SNR, using the directory names or 
                  the event file header as a base

lc.     extract_lc: computes the lightcurves and HR ratio evolution of the source and bg regions of extract_reg in different bands.
                    Also flags for recomputation of the entire action list in bright mode if the source count rate is high enough

                    can also create gtis to cut the individual observations in orbits
                    
sp.     extract_sp: computes the spectral products of the source and bg regions of extract_reg in different bands.
                    can use the gtis created in lc
g.      regroup_spectral: regroups spectral products according to the requirements

m.      batch_mover: copies all products to a global directory to prepare for large scale analysis

fc.     clean_all:  deletes all products in out(bright), products(bright) and the log files in an obsid directory
"""

'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce NuSTAR files.\n)')

# the basics

ap.add_argument('-load_functions', nargs=1, help="Load functions but don't launch anything", default=False, type=bool)

ap.add_argument("-dir", "--startdir", nargs='?', help="starting directory. Current by default", default='./', type=str)
ap.add_argument("-l", "--local", nargs=1, help='Launch actions directly in the current directory instead',
                default=False, type=bool)
ap.add_argument('-catch', '--catch_errors', help='Catch errors while running the data reduction and continue',
                default=False, type=bool)

# global choices
ap.add_argument("-a", "--action", nargs='?', help='Give which action(s) to proceed,separated by comas.',
                default='build,reg,lc,sp,g,m', type=str)
# default: build,reg,lc,sp,g,m

ap.add_argument("-over", nargs=1, help='overwrite computed tasks (i.e. with products in the batch, or merge directory\
                if "m" is in the actions) in a folder', default=True, type=bool)

ap.add_argument('-cameras',nargs=1,help='which cameras to restrict the analysis to. "all" takes both FPMA and FPMB',
                default='all',type=str)

ap.add_argument('-bright_check',nargs=1,help='recompute the entire set of actions in bright mode if the source lightcurve'+
                                             'is above the standard count limits',default=False,type=bool)

ap.add_argument('-force_bright',help="Force bright mode for the tasks from the get go",default=False)

# directory level overwrite (not active in local)
ap.add_argument('-folder_over', nargs=1, help='relaunch action through folders with completed analysis', default=True,
                type=bool)
ap.add_argument('-folder_cont', nargs=1, help='skip all but the last 2 directories in the summary folder file',
                default=False, type=bool)
# note : we keep the previous 2 directories because bug or breaks can start actions on a directory following the initially stopped one

ap.add_argument('-heasoft_init_alias', help="name of the heasoft initialisation script alias", default="heainit",
                type=str)
ap.add_argument('-caldbinit_init_alias', help="name of the caldbinit initialisation script alias", default="caldbinit",
                type=str)


'''region computation'''

ap.add_argument('-use_file_target', nargs=1,
                help='Allows to extract regions when Simbad doesnt recognize the name of the source from'
                     'the directory structure', default=True)

ap.add_argument("-target_only", nargs=1,
                help='only extracts spectra when the source is the main focus of the observation',
                default=False, type=bool)

# Should correspond to the most important energy band for subsequent science analysis.
ap.add_argument('-image_band',nargs=1,help='band in which to extract the image for region computation',default='3-79',
                type=str)

ap.add_argument('-rad_crop', nargs=1,
                help='cropping radius around the theoretical source position before fit, in arcsecs', default=120,
                type=float)

# if equal to crop, is set to rad_crop
ap.add_argument('-max_rad_source', nargs=1, help='maximum source radius for faint sources in units of PSF sigmas',
                default=10, type=float)

#can be varied for crowded fields or stray lights
ap.add_argument('-bg_area_factor',nargs=1,
                help='gives the maximum radius of the background in units of rad_crop',default=2,type=float)

#note: value of 0 to deactivate
ap.add_argument('-bg_rm_src_sigmas',nargs=1,
                help='remove N sigmas of the source PSF sigmas in the background image before treating the bg image',
                default=10.,type=float)

ap.add_argument('-bg_distrib_cut',nargs=1,help='Distribution portion of the bg camera to remove',
                default=0.997,type=float)

ap.add_argument('-bigger_fit', nargs=1,
                help='allows to incease the crop window used before the gaussian fit for bright sources',
                default=True, type=bool)

ap.add_argument('-point_source', nargs=1,
                help="assume the source is point-like, I.E. fixes the gaussian's initial center to the brightest pixel",
                default=True, type=bool)
# helps to avoid the gaussian center shifting in case of diffuse emission

#if set to true, wil ask for sudo mdp at script launch
ap.add_argument('-sudo_mode',nargs=1,help='put to true if the ds9 installation needs to be run in sudo',
               default=False,type=bool)

'''lightcurve'''

#note: this binning will also be used to CREATE the gtis
ap.add_argument('-lc_bin_std', nargs=1, help='Gives the binning of all standard lightcurces/HR evolutions (in s)',
                default='10',type=str)

ap.add_argument('-lc_bin_gti', nargs=1, help='Gives the binning of all lightcurves used for gti cutting (in s)',
                default='1',type=str)

# note: also defines the binning used for the gti definition

ap.add_argument('-lc_bands_str', nargs=1, help='Gives the list of bands to create lightcurves from', default='3-79',
                type=str)
ap.add_argument('-hr_bands_str', nargs=1, help='Gives the list of bands to create hrs from', default='10-50/3-10',
                type=str)

#note: also makes the spectrum function create spectra uniquely from GTIs
ap.add_argument('-make_gti_orbit',nargs=1,help='cut individual observations per orbits with gtis',default=False,
                type=bool)

ap.add_argument('-gti_tool',nargs=1,help='tool to make gti files',default='NICERDAS',type=str)

ap.add_argument('-use_gtis',nargs=1,help='use already existing gti files to make lightcurves if available',
                default=True,type=bool)
'''spectra'''

ap.add_argument('-spectral_band',nargs=1,help='Energy band to compute the spectra in (format "x-y" in keV).'+
                                               'if set to None, no restriction are applied',
                default=None,type=str)

'''regroup'''

ap.add_argument('-gtype', "--grouptype", help='Group type to use in the regrouping function', default='opt', type=str)


args = ap.parse_args()

load_functions=args.load_functions

cameras_glob=args.cameras

startdir=args.startdir
action_list=args.action.split(',')
local=args.local
folder_over=args.folder_over
folder_cont=args.folder_cont
overwrite_glob=args.over
catch_errors=args.catch_errors
e_low_img,e_high_img=np.array(args.image_band.split('-'),dtype=float)

sudo_mode=args.sudo_mode

bright_check=args.bright_check
force_bright=args.force_bright

lc_bin_std=args.lc_bin_std
lc_bin_gti=args.lc_bin_gti

lc_bands_str=args.lc_bands_str
hr_bands_str=args.hr_bands_str
make_gti_orbit=args.make_gti_orbit
gti_tool=args.gti_tool
use_gtis=args.use_gtis

e_low_sp,e_high_sp=[None,None] if args.spectral_band==None else args.spectral_band.split('-')

grouptype=args.grouptype
heasoft_init_alias=args.heasoft_init_alias
caldbinit_init_alias=args.caldbinit_init_alias

use_file_target=args.use_file_target
target_only=args.target_only
rad_crop=args.rad_crop
bigger_fit=args.bigger_fit
point_source=args.point_source
max_rad_source=args.max_rad_source
bg_area_factor=args.bg_area_factor
bg_rm_src_sigmas=args.bg_rm_src_sigmas
bg_distrib_cut=args.bg_distrib_cut

if sudo_mode:
    sudo_mdp=input('Sudo mode activated. Enter sudo password for ds9')
else:
    sudo_mdp=''
'''''''''''''''''
''''FUNCTIONS''''
'''''''''''''''''

# switching off matplotlib plot displays unless with plt.show()
plt.ioff()

camlist = ['FPMA', 'FPMB']

process_obsdir_done = threading.Event()
extract_reg_done=threading.Event()
extract_lc_done=threading.Event()
extract_sp_done=threading.Event()
regroup_spectral_done=threading.Event()
batch_mover_done=threading.Event()
clean_all_done=threading.Event()


# function to remove (most) control chars
def _remove_control_chars(message):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', message)

def kev_to_PI(e_val):

    '''
    conversion from https://heasarc.gsfc.nasa.gov/docs/nustar/nustar_faq.html#pi_to_energy
    '''
    return round((e_val-1.6)/0.04)


def set_var(spawn):
    '''
    Sets starting environment variables for data analysis
    '''
    if heasoft_init_alias is not None:
        spawn.sendline(heasoft_init_alias)

    if caldbinit_init_alias is not None:
        spawn.sendline(caldbinit_init_alias)


def file_evt_selector(filetype, cameras='all', bright=False):
    '''
    Searches for all of the files of a specific type (among the ones used in the data reduction),
    and asks for input if more than one are detected.

    use "all" as input in camera to get the result for all 3 cameras (default value)

    Returns a single file + file path (not absolute) for each camera

    If the keyword bright is set to True, requires the "bright" keyword in the last directory of the event files
    '''

    # getting the list of files in the directory (note that file_evt_selector is launched while in the directory)
    flist = glob.glob(os.path.join(directory,'**'), recursive=True)

    cameras_avail = ['FPMA', 'FPMB']

    # list of accepted filetypes
    filetypes = ['evt_clean','src_reg','bg_reg']
    file_desc = ['clean event files','source region','background region']

    # getting the index of the file type for the keywords
    type_index = filetypes.index(filetype)

    # type keywords
    keyword_types = ['_cl.evt','reg_src.reg','reg_bg.reg']

    # camera keywords (always the same for NuSTAR)
    camword_evt = ['A01', 'B01']
    keyword_cams = camword_evt

    # cam_list is the list of cameras to use for the evt search
    if cameras == 'all':
        cam_list = cameras_avail
    else:
        cam_list = [cameras]

    result_list = []

    for cam in cam_list:
        # getting the index of the camera for the camera keywords
        cam_index = cam_list.index(cam)

        # computing the list for the correct file type and camera
        #with some safekeeps to ensure we don't pick out things we don't want
        cutlist = [elem for elem in flist if keyword_cams[cam_index] in elem\
                   and elem.endswith(keyword_types[type_index])\
                   and (1 if not bright else 'bright' in elem.split('/')[-2])\
                   and elem.split('/')[-1].startswith('nu')]

        if len(cutlist) == 0:
            print('\nWarning : No ' + file_desc[type_index] + ' found for camera ' + cam)
            camdir = ['']
            camfile = ['']
        else:
            print('\n' + str(len(cutlist)) + ' file found for ' + cam + ' ' + file_desc[type_index] + ' :')
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
        bashproc.sendline('nupipeline indir=' + "." + ' steminputs=nu'+directory+
                          ' outdir=./out'+('_bright' if bright!=False else '')+
                          ' clobber=' + ('YES' if overwrite else 'FALSE')+bright_str)

        #will need to update this in case of updates
        process_state = bashproc.expect(['nupipeline_0.4.9: Exit','ERROR: Pipeline exit with error'], timeout=None)

        # exiting the bashproc
        bashproc.sendline('exit')
        process_obsdir_done.set()

        # raising an error to stop the process if the command has crashed for some reason
        if process_state != 0:
            raise ValueError

def disp_ds9(spawn, file, zoom='auto', scale='log', regfile='', screenfile='', give_pid=False, kill_last='',
             sudo_mode=False,sudo_mdp=''):
    '''
    Regfile is an input, screenfile is an output. Both can be paths
    If "screenfile" is set to a non empty str, we make a screenshot of the ds9 window in the given path
    This is done manually since the ds9 png saving command is bugged

    if give_pid is set to True, returns the pid of the newly created ds9 process

    In some installations like mine ds9 struggles to start outside of sudo, so there is a sudo mode where a sudo command
    (with password) is used to launch and remove ds9

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

    if sudo_mode:
        spawn.sendline(
            'echo "'+sudo_mdp+'" | sudo -S ds9 -view buttons no -cmap Heat -geometry 1080x1080 -scale ' + scale + ' -mode region ' + file + ' -zoom ' + str(
                zoom) +
            ' ' + regfile + ' &')

        # the timeout limit could be increased for slower computers or heavy images
        spawn.expect(['password', pexpect.TIMEOUT], timeout=1)

    else:
        spawn.sendline('ds9 -view buttons no -cmap Heat -geometry 1080x1080 -scale ' + scale + ' -mode region '
                       + file + ' -zoom ' + str(zoom) + ' ' + regfile + ' &')



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

    #if need
    # plt.figure(figsize=(10,10))
    # plt.imshow(mask)
    # plt.savefig('mask.png')

    # this function returns the set of polygons equivalent to the mask
    # the last polygon of the set should be the outer shell
    polygons =Mask(int_mask).polygons()

    # since we don't know the position of the outer shell (and there is sometimes no outer shell)
    # we'll consider the biggest polygon as the "main one".
    # It's easily identifiable as the ones with the biggest number of points
    # (since they are identified on pixel by pixel basis, there seems to be no "long" lines)

    shell_length = 0

    #if need
    # #polygon figure
    # plt.figure(figsize=(10,10))
    # poly_plot_coords=[]
    # for i in range(len(polygons.points)):
    #     if len(polygons.points[i])>4:
    #         poly_plot_coords+=[Polygon(polygons.points[i]).exterior.xy]
    #         plt.plot(poly_plot_coords[-1][0],poly_plot_coords[-1][1],c='red')
    # plt.savefig('testpoly.png')


    for i in range(len(polygons.points)):

        if len(polygons.points[i]) > shell_length:
            shell_id = i
            shell_length = len(polygons.points[i])



    # swapping the positions to have the shell as the first polygon in the array
    poly_args = polygons.points[:shell_id] + polygons.points[shell_id + 1:]
    poly_args.insert(0, polygons.points[shell_id])

    test = [len(polygons.points[id]) for id in range(len(polygons.points))]

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

def xsel_img(bashproc,evt_path,save_path,e_low,e_high,directory):

    '''
    Uses Xselect to create a NuSTAR image from bashproc

    e_low and e_high should be in keV

    save_path should be a relative path from the current directory of bashproc or an absolute path
    '''

    evt_dir='./' if '/' not in evt_path else evt_path[:evt_path.rfind('/')]
    evt_file= evt_path[evt_path.rfind('/')+1:]

    pha_low=str(kev_to_PI(e_low))
    pha_high=str(kev_to_PI(e_high))

    bashproc.sendline('xselect')
    bashproc.expect('XSELECT')

    #session name
    bashproc.sendline('')

    #reading events
    bashproc.sendline('read events')

    bashproc.expect('Event file dir')
    bashproc.sendline(evt_dir)

    bashproc.expect('Event file list')
    bashproc.sendline(evt_file)

    #resetting mission
    bashproc.expect('Reset')
    bashproc.sendline('yes')

    #commands to prepare image creation
    bashproc.sendline('filter pha_cutoff '+pha_low+' '+pha_high)
    bashproc.sendline('set xybinsize 1')
    bashproc.sendline('extract image')
    bashproc.expect('Image')

    #commands to save image
    bashproc.sendline('save image')

    #can take time so increased timeout
    bashproc.expect('Give output file name',timeout=120)

    bashproc.sendline(save_path)

    over_code=bashproc.expect(['File already exists','Wrote image to '])

    if over_code==0:
        bashproc.sendline('yes')
        bashproc.expect(['Wrote image to '])

    print('Letting some time to create the file...')
    #giving some time to create the file
    time.sleep(5)

    for i_sleep in range(20):
        if not os.path.isfile(os.path.join(directory,save_path)):
            print('File still not ready. Letting more time...')
            time.sleep(5)

    if not os.path.isfile(os.path.join(directory,save_path)):
        print('Issue with file check or file creation')
        breakpoint()

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

def extract_reg(directory, cams='all', use_file_target=False,
                overwrite=True,e_low_img=3,e_high_img=79,rad_crop=120,bg_area_factor=2.,bg_rm_src_sigmas=10.,
                bg_distrib_cut=0.99,
                bright=False,sudo_mode=False,sudo_mdp=''):
    '''
    Extracts the optimal source/bg regions for a given exposure

    As of now, only takes input formatted through the evt_filter function

    Only accepts circular regions (in manual mode)

    bg_area_factor: gives the radius maximum background as rad_crop*bg_area_factor (if it can go that far)
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
            with fits.open(os.path.join(filedir,img_file)) as hdul:
                CCD_data = hdul[0].data

            CCD_img_obj=Image(data=CCD_data,wcs=src_mpdaf_WCS)

            if len(CCD_data.nonzero()[0]) == 0:
                print("\nEmpty bg image after CCD cropping. Skipping this observation...")
                spawn.sendline('\ncd $currdir')
                return "Empty bg image after CCD cropping."

            print('\nSaving the corresponding image...')
            imgarr_to_png(CCD_data, file_id+'_vis' + '_CCD_1_crop', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=filedir, title='Source image with CCDs cropped according to the region size and center')

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
            imgarr_to_png(CCD_data, file_id+'_vis_CCD_2_mask', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=filedir, title='Source image CCD(s) mask after clipping and filling of the holes')

            print('\nComputing the CCD masked image...')
            # array which we will have the outside of the CCD masked with nans
            CCD_data_cut = np.copy(CCD_data).astype(float)

            print('\nSaving the corresponding image...')
            imgarr_to_png(CCD_data_cut, file_id+'_vis_CCD_3_cut', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=filedir, title='Source image after CCD masking', imgtype='ccd_crop')


            #source cut if asked to
            if bg_rm_src_sigmas!=0:
                #masking the region through mpdaf
                #note: we don't put the units here because
                CCD_img_obj.mask_region(gfit.center, max(gfit.fwhm)/2.355*bg_rm_src_sigmas,
                                  inside=True, posangle=0.0)

                mpdaf_mask=CCD_img_obj.mask[::-1]
                #replacing the pixels in the data array
                CCD_data_cut[CCD_img_obj.mask]=np.nan

                print('\nSaving the source-cut image...')
                imgarr_to_png(CCD_data_cut, file_id+'_vis_CCD_3b_src_rm', astropy_wcs=src_astro_WCS,
                        mpdaf_wcs=src_mpdaf_WCS,
                        directory=filedir, title='Source image after Source region removal', imgtype='ccd_crop')

            # This other array stores the values inside the CCD, for an easy evaluation of the sigma limits
            CCD_data_line = []
            for i in range(np.size(CCD_data_cut, 0)):
                for j in range(np.size(CCD_data_cut, 1)):
                    if not CCD_mask[i][j]:
                        CCD_data_cut[i][j] = np.nan
                    else:
                        CCD_data_line.append(CCD_data_cut[i][j])

            # sigma cut, here at 0.95 (2 sigma) which seems to be a good compromise
            CCD_data_line.sort()

            # for some extreme cases we have only 1 count/pixel max, in which case we don't want that
            cut_sig = max(CCD_data_line[int(bg_distrib_cut * len(CCD_data_line))], 1.)

            # sigval = '2'
            # perval = '5'

            # # sometimes for very bright sources there might be too much noise so we cut at a different percentage
            # if cut_sig > 50 and bg_distrib_cut_bright!=0:
            #     cut_sig = CCD_data_line[int(bg_distrib_cut_bright * len(CCD_data_line))]
            #     # sigval = '1'
            #     # perval = '32'

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
            imgarr_to_png(CCD_bg, file_id+'_vis_CCD_4_bg', astropy_wcs=src_astro_WCS,
                    mpdaf_wcs=src_mpdaf_WCS,
                    directory=filedir,
                    title='Source image background mask remaining after %.2f'%(100*bg_distrib_cut)+
                          '% cts) counts removal', imgtype='ccd_crop_mask')

            bg_max_pix = reg_optimiser(CCD_bg)

            print('\nMaximal bg region coordinates in pixel units:')
            print(bg_max_pix)

            '''
            finally, we convert the region coordinates and radius to angular values,
            using mpdaf's wcs pix2sky method. No need to invert the coordinates because the polygon mask
            axis have the same axe origins (lower left) than mpdaf
            '''

            bg_center_mpdaf=np.array([bg_max_pix[0][1],bg_max_pix[0][0]])

            #first index because the result is wrapped in an array
            bg_center_radec=src_mpdaf_WCS.pix2sky(bg_center_mpdaf)[0]

            bg_max = [bg_center_radec.astype(str)[::-1],
                      str(round(bg_max_pix[1] * CCD_img_obj.get_axis_increments()[0]*3600, 4))]

            return bg_max

        '''
        MAIN BEHAVIOR
        '''

        if file == '':
            print('\nNo evt to extract spectrum from for this camera in the obsid directory.')
            return 'No evt to extract spectrum from for this camera in the obsid directory.'

        #different directory structure since the spawn is already in the directory
        spawndir = filedir.replace(directory,'.')

        #note: only the file here
        file_id=file.replace('_cl.evt','')

        img_file=file_id+'_img_'+str(e_low_img)+'_'+str(e_high_img)+'.ds'

        #creating an image to load with mpdaf for image analysis
        xsel_img(spawn,os.path.join(spawndir,file),os.path.join(spawndir,img_file),e_low=e_low_img,e_high=e_high_img,
                 directory=directory)

        spawn.sendline('\ncurrdir=$(pwd)')
        spawn.sendline('\ncd '+filedir)

        #waiting for the file to be created if it hasn't loaded yet
        try:
            fits_img = fits.open(os.path.join(filedir,img_file))
        except:

            time.sleep(5)

            try:
                fits_img = fits.open(os.path.join(filedir, img_file))
            except:

                #even longer waiting time if there's a very big file
                time.sleep(15)

        #opening the image file and saving it for verification purposes
        ds9_pid_sp_start=disp_ds9(spawn,os.path.join(spawndir,img_file),
                                  screenfile=os.path.join(filedir,img_file).replace('.ds','_screen.png'),
                                  give_pid=True,sudo_mode=sudo_mode,sudo_mdp=sudo_mdp)

        try:
            fits_img = fits.open(os.path.join(filedir,img_file))
            fits_img.close()
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
        with fits.open(os.path.join(filedir,img_file)) as hdul:
            img_data=hdul[0].data
            src_mpdaf_WCS=mpdaf_WCS(hdul[0].header)
            src_astro_WCS=astroWCS(hdul[0].header)
            main_source_name=hdul[0].header['object']
            main_source_ra=hdul[0].header['RA_OBJ']
            main_source_dec=hdul[0].header['DEC_OBJ']

        print('\nAuto mode.')
        print('\nAutomatic search of the directory names in Simbad.')

        prefix = '_auto'

        #using the full directory structure here
        obj_auto = source_catal(spawn, os.path.join(os.getcwd(),filedir), file,
                                target_only=target_only,
                                use_file_target=use_file_target, )

        # checking if the function returned an error message (folder movement done in the function)
        if type(obj_auto) == str:
            if not use_file_target:
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

        with open(os.path.join(filedir,reg_catal_name), 'w+') as regfile:
            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression((obj_deg, str(rad_crop)))
                          + ' # text={' + obj_auto['MAIN_ID'] + ' initial cropping zone}')

        ds9_pid_sp_start = disp_ds9(spawn, os.path.join(spawndir,img_file), regfile=os.path.join(spawndir,reg_catal_name),
                                    zoom=1.2,
                                    screenfile=filedir + '/' +file_id + prefix + '_reg_catal_screen.png',
                                    give_pid=True,
                                    kill_last=ds9_pid_sp_start,sudo_mode=sudo_mode,sudo_mdp=sudo_mdp)

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
        plt.savefig(filedir + '/' + file_id + prefix + '_catal_crop_screen.png')
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

        #capping the bg size to a given number of times radcrop
        bg_coords_im[1]=str(round(min(float(bg_coords_im[1]),bg_area_factor*rad_crop),4))

        #bg_coords_im[0]=bg_coords_im[0].tolist()

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

        '''
        Here I edited slightly the ee function in mpdaf.image because the current version does the conversion of
        rad to physical and then slices the image accordingly before computing the ee
        The issue is that it will try to slice with float index values because there's no conversion, 
        so I added floors and ceils and int conversion before imin/jmin and imax/jmax to allow it to work
        
        Also need to convert to tuples or the ee won't accept the center 
        '''

        counts_bg=img_obj_whole.ee(center=tuple(bg_coords_im[0].astype(float))[::-1],radius=float(bg_coords_im[1]))

        #computing the SNR for a range of radiuses

        rad_test_arr=np.arange(4,min(rad_crop,max(gfit.fwhm)*max_rad_source/2.355),2)

        snr_vals=np.repeat(0,len(rad_test_arr))

        for id_rad,rad in enumerate(rad_test_arr):

            counts_src=img_obj_whole.ee(center=tuple(gfit.center),radius=rad_test_arr[id_rad])

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

        with open(os.path.join(filedir,reg_name), 'w+') as regfile:

            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression(src_coords)
                          + ' # text={' + obj_auto['MAIN_ID'] + '}' +
                          '\n' + spatial_expression(bg_coords_im)
                          + ' # text={automatic background}' )

        ds9_pid_sp_reg = disp_ds9(spawn,os.path.join(spawndir,img_file), regfile=os.path.join(spawndir,reg_name),
                                    screenfile=filedir + '/' +file_id + prefix + '_reg_screen.png',
                                    give_pid=True,
                                    kill_last=ds9_pid_sp_start,sudo_mode=sudo_mode,sudo_mdp=sudo_mdp)

        '''
        and in individual region files for the extraction
        '''

        reg_src_name=file_id+ prefix + '_reg_src.reg'
        reg_bg_name=file_id+ prefix + '_reg_bg.reg'

        with open(os.path.join(filedir,reg_src_name), 'w+') as regfile:

            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression(src_coords)
                          + ' # text={' + obj_auto['MAIN_ID'] + '}')

        with open(os.path.join(filedir,reg_bg_name), 'w+') as regfile:

            # standard ds9 format
            regfile.write('# Region file format: DS9 version 4.1' +
                          '\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1' +
                          ' highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1' +
                          '\nfk5' +
                          '\n' + spatial_expression(bg_coords_im)
                          + ' # text={automatic background}')

        if sudo_mode:
            bashproc.sendline('\necho "' + sudo_mdp + '" |sudo -S pkill sudo')
        else:
            os.system('wmctrl -ic ' + str(ds9_pid_sp_reg))

        # this sometimes doesn't proc before the exit for whatever reason so we add a buffer just in case
        # bashproc.expect([pexpect.TIMEOUT],timeout=2)

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
    set_var(bashproc)

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

    # closing the spawn
    bashproc.sendline('exit')

    print('\nRegion extraction of the current obsid directory events finished.')

    extract_reg_done.set()




    def extract_lc_single(spawn, directory, binning, instru, steminput, src_reg, bg_reg, e_low, e_high,bright=False,backscale=1):

        lc_src_name = steminput + '_' + instru + '_lc_src_' + e_low + '_' + e_high + '_bin_' + binning + '.lc'
        lc_bg_name = steminput + '_' + instru + '_lc_bg_' + e_low + '_' + e_high + '_bin_' + binning + '.lc'

        #the spawn paths are different because in the spawn we cd in the obsid directory to avoid
        #putting the temp files everywhere
        lc_src_path = os.path.join(directory,'products'+('_bright' if bright else ''), lc_src_name)
        lc_bg_path = os.path.join(directory,'products'+('_bright' if bright else ''), lc_bg_name)

        pi_low = str(kev_to_PI(float(e_low)))
        pi_high = str(kev_to_PI(float(e_high)))

        # building the lightcurve
        spawn.sendline('nuproducts indir=./out'+('_bright' if bright else '')+
                       ' instrument=' + instru + ' steminputs=' + steminput +
                       ' lcfile=' + lc_src_name +  ' srcregionfile=' + src_reg +
                       ' bkgextract=yes'+
                       ' bkglcfile=' + lc_bg_name + ' bkgregionfile=' + bg_reg +
                       ' pilow=' + pi_low + ' pihigh=' + pi_high + ' binsize=' + binning+
                       ' outdir=./products'+('_bright' if bright else '')+
                       ' barycorr=no phafile=NONE bkgphafile=NONE' + ' imagefile=NONE runmkarf=no runmkrmf=no'+
                       ' clobber=YES cleanup=YES')

        ####TODO: check what's the standard message here
        err_code=spawn.expect(['nuproducts_0.3.3: Exit with success','nuproducts error'],timeout=None)

        if err_code!=0:
            return 'Nuproduct error','',''

        # loading the data of both lc
        with fits.open(lc_src_path) as fits_lc:
            # time zero of the lc file (different from the time zeros of the gti files)
            time_zero = Time(fits_lc[1].header['MJDREFI'] + fits_lc[1].header['MJDREFF'], format='mjd')

            # and offsetting the data array to match this
            delta_lc_src = fits_lc[1].header['TIMEZERO']

            # storing the lightcurve
            data_lc_src = fits_lc[1].data

            time_zero_str = str((time_zero+TimeDelta(delta_lc_src,format='sec')).to_datetime())

        if float(binning)>=1 and e_low=='3' and e_high=='79' and not bright:
            bright_flag=max(data_lc_src['RATE'])>100
        else:
            bright_flag=False

        with fits.open(lc_bg_path) as fits_lc:
            # and offsetting the data array to match this
            delta_lc_bg = fits_lc[1].header['TIMEZERO']

            fits_lc[1].data['TIME'] += delta_lc_bg -delta_lc_src

            # storing the shifted lightcurve
            data_lc_bg = fits_lc[1].data

        # plotting the source and bg lightcurves together

        fig_lc, ax_lc = plt.subplots(1, figsize=(10, 8))

        ax_lc.set_yscale('symlog', linthresh=0.1, linscale=0.1)
        ax_lc.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.1))

        ax_lc.errorbar(data_lc_src['TIME'], data_lc_src['RATE'], xerr=float(binning),
                     yerr=data_lc_src['ERROR'], ls='-', lw=1, color='grey', ecolor='blue', label='raw source')

        ax_lc.errorbar(data_lc_bg['TIME'], data_lc_bg['RATE']*backscale, xerr=float(binning),
                     yerr=data_lc_bg['ERROR']*backscale, ls='-', lw=1, color='grey', ecolor='brown', label='scaled background')

        ax_lc.axhline(100,0,1,color='red',ls='-',lw=1,label='bright obs threshold')

        plt.suptitle('NuSTAR ' + instru + ' lightcurve for observation ' + steminput +
                     ' in the ' + e_low + '-' + e_high + ' keV band with ' + binning + ' s binning')

        ax_lc.set_xlabel('Time (s) after ' + time_zero_str)
        ax_lc.set_ylabel('RATE (counts/s)')

        ax_lc.set_ylim(0,ax_lc.get_ylim()[1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(directory,'products'+('_bright' if bright else ''),
                                 steminput + '_' + instru + '_lc_screen_' + e_low + '_' + e_high + '_bin_' + binning + '.png'))
        plt.close()

        return 'Lightcurve creation complete',[time_zero_str,data_lc_src['TIME'],data_lc_src['RATE']-data_lc_bg['RATE']*backscale,
                                                data_lc_src['ERROR']+data_lc_bg['ERROR']*backscale],bright_flag

def extract_lc_single(spawn, directory, binning, instru, steminput, src_reg, bg_reg, e_low, e_high,bright=False,
                      backscale=1,gti_mode=False,gti=None,id_orbit=''):

    id_orbit_str='-'+str_orbit(id_orbit) if gti is not None else ''

    lc_src_name = steminput + id_orbit_str+ '_' + instru + '_lc_src_' + e_low + '_' + e_high + '_bin_' + binning + '.lc'
    lc_bg_name = steminput + id_orbit_str+ '_' + instru + '_lc_bg_' + e_low + '_' + e_high + '_bin_' + binning + '.lc'

    #the spawn paths are different because in the spawn we cd in the obsid directory to avoid
    #putting the temp files everywhere
    lc_src_path = os.path.join(directory,'products'+('_bright' if bright else ''), lc_src_name)
    lc_bg_path = os.path.join(directory,'products'+('_bright' if bright else ''), lc_bg_name)

    pi_low = str(kev_to_PI(float(e_low)))
    pi_high = str(kev_to_PI(float(e_high)))

    #removing the first directory to match the spawn cd
    gti_spawn='' if gti is None else '/'.join(gti.split('/')[1:])

    # building the lightcurve
    spawn.sendline('nuproducts indir=./out'+('_bright' if bright else '')+
                   ' instrument=' + instru + ' steminputs=' + steminput +
                   ' lcfile=' + lc_src_name +  ' srcregionfile=' + src_reg +
                   ' bkgextract=yes'+
                   ' bkglcfile=' + lc_bg_name + ' bkgregionfile=' + bg_reg +
                   ' pilow=' + pi_low + ' pihigh=' + pi_high + ' binsize=' + binning+
                   ' outdir=./products'+('_bright' if bright else '')+
                   ' barycorr=no phafile=NONE bkgphafile=NONE' + ' imagefile=NONE runmkarf=no runmkrmf=no'+
                   ' clobber=YES cleanup=YES'+(' usrgtifile=./'+gti_spawn if gti is not None else ''))

    ####TODO: check what's the standard message here
    err_code=spawn.expect(['nuproducts_0.3.3: Exit with success',"nuproducts_0.3.3: ERROR running 'nulivetime'",
                           '-------------------- nuproducts  error',
                           "nuproducts_0.3.3: Error: running 'lcurve'"],timeout=None)

    if err_code!=0:
        return 'Nuproduct error','',''

    # loading the data of both lc
    with fits.open(lc_src_path) as fits_lc:
        # time zero of the lc file (different from the time zeros of the gti files)
        time_zero = Time(fits_lc[1].header['MJDREFI'] + fits_lc[1].header['MJDREFF'], format='mjd')

        # and offsetting the data array to match this
        delta_lc_src = fits_lc[1].header['TIMEZERO']

        # storing the lightcurve
        data_lc_src = fits_lc[1].data

        time_zero_str = str((time_zero+TimeDelta(delta_lc_src,format='sec')).to_datetime())

    if float(binning)>=1 and e_low=='3' and e_high=='79' and not bright:
        bright_flag=max(data_lc_src['RATE'])>100
    else:
        bright_flag=False

    with fits.open(lc_bg_path) as fits_lc:
        # and offsetting the data array to match this
        delta_lc_bg = fits_lc[1].header['TIMEZERO']

        fits_lc[1].data['TIME'] += delta_lc_bg -delta_lc_src

        # storing the shifted lightcurve
        data_lc_bg = fits_lc[1].data

    # plotting the source and bg lightcurves together

    fig_lc, ax_lc = plt.subplots(1, figsize=(10, 8))

    if gti is None:
        ax_lc.set_yscale('symlog', linthresh=0.1, linscale=0.1)
        ax_lc.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=0.1))

        #plotting the background
        ax_lc.errorbar(data_lc_bg['TIME'], data_lc_bg['RATE']*backscale, xerr=float(binning),
                 yerr=data_lc_bg['ERROR']*backscale, ls='-', lw=1, color='grey', ecolor='brown', label='scaled background')


    ax_lc.errorbar(data_lc_src['TIME'], data_lc_src['RATE'], xerr=float(binning),
                 yerr=data_lc_src['ERROR'], ls='-', lw=1, color='grey', ecolor='blue', label='raw source')


    ax_lc.axhline(100,0,1,color='red',ls='-',lw=1,label='bright obs threshold')

    plt.suptitle('NuSTAR ' + instru + ' lightcurve for observation ' + steminput + id_orbit_str+
                 ' in the ' + e_low + '-' + e_high + ' keV band with ' + binning + ' s binning')

    ax_lc.set_xlabel('Time (s) after ' + time_zero_str)
    ax_lc.set_ylabel('RATE (counts/s)')

    ax_lc.set_ylim(0,ax_lc.get_ylim()[1])

    # finishing the figure
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'products' + ('_bright' if bright else ''),
                             steminput + id_orbit_str + '_' + instru + '_lc_screen_' + e_low + '_' + e_high + '_bin_' + binning + '.png'))

    if gti_mode:
        #returning the figure to add the gti intervals later, and the lc path
        return lc_src_path,fig_lc
    if not gti_mode:
        #closing the figure and returning the standard lc products
        plt.close()
        return 'Lightcurve creation complete',[time_zero_str,data_lc_src['TIME'],data_lc_src['RATE']-data_lc_bg['RATE']*backscale,
                                            data_lc_src['ERROR']+data_lc_bg['ERROR']*backscale],bright_flag

    else:
        return lc_src_path

def extract_lc(directory,binning='1',lc_bands_str='3-79',hr_bands='10-50/3-10',cams='all',bright=False,
               make_gtis=False,use_gtis=True,gti_binning='1',gti_tool='NICERDAS'):

    '''
    Wrapper for a version of nuproducts to computes only lightcurves in the desired bands,
    with added matplotlib plotting of requested lightcurves and HRs

    also flags an output if the source lightcurve in the 3-79 band goees above 100 cts/s for recomputing.
    ONLY FLAGS if bright mode is not already activated

    We follow the steps highlighted in https://heasarc.gsfc.nasa.gov/docs/nustar/analysis/nustar_swguide.pdf 5.3D
    options:
        -binning: binning of the LC in seconds

        -bands: bands for each lightcurve to be created.
                The numbers should be in keV, separated by "-", and different lightcurves by ","
                ex: to create two lightcurves for, the 1-3 and 4-12 band, use '1-3,4-12'

        -hr: bands to be used for the HR plot creation.
             A single plot is possible for now. Creates its own lightcurve bands if necessary

        -overwrite: overwrite products or not

        -make_gtis: Cuts the nustar observations by orbit

        -use_gtis: uses previously existing gtis if available

    NOTE THAT THE BACKSCALE CORRECTION IS APPLIED MANUALLY

    NOTE2: currently, the bright flag will be set to true for all individual orbits together
    Should be changed to individual returns for each observation period if we look at source varying over and
    under the treshold in the span of an observation
    '''

    '''MAIN BEHAVIOR'''

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
    bg_reg = file_evt_selector('bg_reg',cameras=cams,bright=bright)

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
               + lc_bands_str.split(',')

    lc_bands = np.unique(lc_bands)[::-1]

    # storing the ids for the HR bands
    id_band_num_HR = np.argwhere(hr_bands.split('/')[0] == lc_bands)[0][0]
    id_band_den_HR = np.argwhere(hr_bands.split('/')[1] == lc_bands)[0][0]

    summary_header='Obsid\tcamera\tenergy band\tLightcurve extraction result\n'

    set_var(bashproc)

    bright_flag_tot=False

    if os.path.isfile(directory + '/extract_lc.log'):
        os.system('rm ' + directory + '/extract_lc.log')

    with StdoutTee(directory + '/extract_lc.log', mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(directory + '/extract_lc.log', buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        #putting the bashproc in the directory to compartiment the temporary files of the commands
        bashproc.sendline('cd '+directory)

        # filtering for the selected cameras
        for i_cam in camid_list:

            if directory.endswith('/'):
                obsid = directory.split('/')[-2]
            else:
                obsid = directory.split('/')[-1]

            bashproc.logfile_read = sys.stdout
            print('\nComputing lightcurves of camera ' + camlist[i_cam])

            #fetching region files for this camera
            src_reg_indiv='/'.join(np.array(src_reg[i_cam]).T[0][::-1])
            bg_reg_indiv='/'.join(np.array(bg_reg[i_cam]).T[0][::-1])

            if src_reg_indiv=='/':
                print('Source region file missing')
                return 'Source region file missing'

            #computing the backscale
            src_reg_coords=ds9_to_reg(src_reg_indiv)
            bg_reg_coords=ds9_to_reg(bg_reg_indiv)

            #creating a path without the main directory for the spawn
            src_reg_indiv_spawn=src_reg_indiv.replace(directory,'.',1)
            bg_reg_indiv_spawn=bg_reg_indiv.replace(directory,'.',1)


            #the last character of the radius is the arcsec " so we remove it
            backscale=(float(src_reg_coords[1][:-1])/float(bg_reg_coords[1][:-1]))**2

            # launching the main extraction

            if make_gtis:
                lc_cut_path,lc_cut_fig=extract_lc_single(bashproc,directory=directory,
                                                binning=gti_binning,instru=camlist[i_cam],steminput='nu'+obsid,
                                                src_reg=src_reg_indiv_spawn,bg_reg=bg_reg_indiv_spawn,
                                                e_low=lc_bands[0].split('-')[0],e_high=lc_bands[0].split('-')[1],
                                                bright=bright,backscale=backscale,gti_mode=True)

                gti_list=create_gtis(bashproc,lc_cut_path,lc_cut_fig,gti_tool=gti_tool)
            else:
                if use_gtis:
                    # checking if gti files exist in the folder
                    gti_list = np.array([elem for elem in
                                          glob.glob(os.path.join(directory, 'products_bright' if bright else 'products',
                                                                 '**'),
                                                    recursive=True) \
                                          if elem.endswith('.gti') and '_gti_' in elem \
                                          and camlist[i_cam] in elem \
                                          and '_gti_mask_' not in elem and '_gti_input' not in elem])

                    gti_list.sort()

                    if len(gti_list)==0:
                        gti_list=[None]
                else:
                    gti_list=[None]

            for id_orbit,elem_gti in enumerate(gti_list):

                lc_prods=np.array([None]*len(lc_bands))

                for id_band,band in enumerate(lc_bands):

                    summary_line,lc_prods[id_band],bright_flag_single = extract_lc_single(bashproc,directory=directory,
                                                    binning=binning,instru=camlist[i_cam],steminput='nu'+obsid,
                                                    src_reg=src_reg_indiv_spawn,bg_reg=bg_reg_indiv_spawn,
                                                    e_low=band.split('-')[0],e_high=band.split('-')[1],
                                                    bright=bright,backscale=backscale,
                                                    gti=elem_gti,id_orbit=id_orbit)

                    #adding a flag to skip the computation of the HR if the lc computation crashed in at least
                    #one band
                    if type(lc_prods)==str:
                        no_HR_flag=1
                    else:
                        no_HR_flag=0

                    id_orbit_str = '-' + str_orbit(id_orbit) if elem_gti is not None else ''

                    summary_content = obsid + '\t' + camlist[i_cam] +'\t'+ band + '\t' + summary_line
                    file_edit(os.path.join(directory, 'summary_extract_lc.log'), obsid + id_orbit_str + '\t' + camlist[i_cam] +'\t'+ band ,
                              summary_content + '\n',
                              summary_header)

                    #updating the global bright flag if a flagged obs appears (note: the bright_flag is force to False when not in the 3-79 band)
                    bright_flag_tot=bright_flag_tot or bright_flag_single

                #potentially skipping the HR computation
                if no_HR_flag:
                   continue

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

                plt.errorbar(time_HR, hr_vals, xerr=float(binning), yerr=hr_err.clip(0), ls='-', lw=1,
                         color='grey', ecolor='blue')

                plt.suptitle('NuSTAR '+camlist[i_cam]+' net HR evolution for observation ' + obsid + id_orbit_str+' in the ' + hr_bands + ' keV band'+
                             'with '+binning+' s binning')

                plt.xlabel('Time (s) after ' + time_zero_HR)
                plt.ylabel('Hardness Ratio (' + hr_bands + ' keV)')

                plt.tight_layout()
                plt.savefig(os.path.join(directory,'products'+('_bright' if bright else ''),
                            'nu'+obsid + id_orbit_str+'_' + camlist[i_cam]+ '_hr_screen_'+hr_bands.replace('/','_')+'_bin_' + binning + '.png'))
                plt.close()

    extract_lc_done.set()

    return bright_flag_tot

def create_gtis(spawn,cut_lc,fig_cut_lc,gti_tool='NICERDAS'):

    '''
    wrapper for a function to create gti files from an individual lightcurve nicer obsids into indivudal portions

    before:
    first creates a lightcurve with the chosen binning then uses it to define
    individual gtis from orbits

    gtitool: -NICERDAS for NICER's nigti
             -SAS for XMM SAS tabgtigen
    '''

    with fits.open(cut_lc) as fits_mkf:

        data_cut_lc = fits_mkf[1].data

        start_obs_s = fits_mkf[1].header['TSTART']
        # saving for titles later
        mjd_ref = Time(fits_mkf[1].header['MJDREFI'] + fits_mkf[1].header['MJDREFF'], format='mjd')

        obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

        obs_start_str = str(obs_start.to_datetime())

        time_obs = data_cut_lc['TIME']

        # adding gaps of more than 100s as cuts in the gtis
        # useful in all case to avoid inbetweens in the plot even if we don't cut the gtis

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
        orbit_bounds=[list(interval_extract(elem))[0] for elem in id_gti_orbit]

        ax_cut_lc=fig_cut_lc.get_axes()[0]

        for i_orbit in range(len(id_gti_orbit)):
            ax_cut_lc.axvspan(time_obs[orbit_bounds[i_orbit][0]], time_obs[orbit_bounds[i_orbit][1]],
                              color='green', alpha=0.2,
                                label='standard gtis' if i_orbit == 0 else '')

        fig_cut_lc.legend()
        fig_cut_lc.savefig(cut_lc.split('_lc')[0]+'_lc_orbit_screen.png')

        if gti_tool=='SAS':
            # creating the gti files for each part of the obsid
            spawn.sendline('sasinit')

        def create_gti_files(id_orbit,id_gti, data_lc):

            if len(id_gti) > 0:

                fits_gti = fits.open(data_lc)

                time_gtis=fits_gti[1].data['TIME']

                # creating the orbit gti expression
                gti_path = cut_lc.split('_lc')[0]+'_gti_' +str_orbit(id_orbit) + '.gti'
                gti_spawn_path = '/'.join(gti_path.split('/')[1:])

                # preparing the list of gtis to replace manually
                gti_intervals = np.array(list(interval_extract(id_gti))).T

                delta_time_gtis = (time_gtis[1] - time_gtis[0]) / 2

                start_obs_s = fits_gti[1].header['TIMEZERO']

                # saving for titles later
                mjd_ref_nustar = Time(fits_gti[1].header['MJDREFI'] + fits_gti[1].header['MJDREFF'], format='mjd')

                #adding the delta with the NuSTAR mjdrefs
                mjd_ref_nicer= Time(56658 + 7.775925925925930E-04,format='mjd')


                obs_start = mjd_ref_nustar + TimeDelta(start_obs_s, format='sec') +(mjd_ref_nustar-mjd_ref_nicer)

                start_obs_s_nicer=start_obs_s+(mjd_ref_nustar-mjd_ref_nicer).to('s').to_value()

                start_obs_s_nicer=start_obs_s


                if gti_tool == 'NICERDAS':
                    '''
                    the task nigti doesn't accept ISOT formats with decimal seconds so we use NICER MET instead 
                    (see https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/nigti.html)

                    we still add a -0.5*delta and +0.5*delta on each side to avoid issues with losing the last bins of lightcurves
                    '''

                    gti_input_path =  cut_lc.split('_lc')[0]+'_gti_input_' + str_orbit(id_orbit) + '.txt'

                    gti_input_spawn_path='/'.join(gti_input_path.split('/')[1:])

                    with open(gti_input_path, 'w+') as f_input:
                        f_input.writelines([str(start_obs_s_nicer + time_gtis[gti_intervals[0][i]] - delta_time_gtis) + ' ' +
                                            str(start_obs_s_nicer + time_gtis[gti_intervals[1][i]] + delta_time_gtis) + '\n' \
                                            for i in range(len(gti_intervals.T))])

                    spawn.sendline('nigti @' + gti_input_spawn_path + ' ' + gti_spawn_path + ' clobber=YES chatter=4')
                    spawn.expect('ngti=')

                    #TODO: test to remove the GTI manual edition below

                elif gti_tool=='SAS':
                    # creating a custom gti 'mask' file
                    gti_column = fits.ColDefs([\
                        fits.Column(name='IS_GTI', format='I',
                                    array=np.array([1 if i in id_gti else 0 for i in range(len(data_lc))]))])

                    # replacing the hdu with a hdu containing it
                    fits_gti[1] = fits.BinTableHDU.from_columns(fits_gti[1].columns[:2] + gti_column)
                    fits_gti[1].name = 'IS_GTI'

                    lc_mask_path=cut_lc.split('_lc')[0]+'_gti_mask_' + str_orbit(id_orbit)+ '.fits'

                    if os.path.isfile(lc_mask_path):
                        os.remove(lc_mask_path)

                    fits_gti.writeto(lc_mask_path)

                    # waiting for the file to be created
                    while not os.path.isfile(lc_mask_path):
                        time.sleep(0.1)

                    lc_mask_spawn_path='/'.join(lc_mask_path.split('/')[1:])

                    spawn.sendline('tabgtigen table=' + lc_mask_spawn_path +
                                   ' expression="IS_GTI==1" gtiset=' + gti_spawn_path)

                    # this shouldn't take too long so we keep the timeout
                    # two expects because there's one for the start and another for the end
                    spawn.expect('tabgtigen:- tabgtigen')
                    spawn.expect('tabgtigen:- tabgtigen')

                    '''
                    There is an issue with the way tabgtigen creates the exposure due to a lacking keyword
                    To ensure things work correctly, we remake the contents of the file and keep the header
                    '''

                    # preparing the list of gtis to replace manually
                    gti_intervals = np.array(list(interval_extract(id_gti))).T

                    # opening and modifying the content of the header in the gti file for NICER
                    with fits.open(gti_path, mode='update') as hdul:

                        # for some reason we don't get the right values here so we recreate them
                        # creating a custom gti 'mask' file

                        # storing the current header
                        prev_header = hdul[1].header

                        # creating a START and a STOP column in "standard" GTI fashion
                        # note: the 0.5 is there to allow the initial and final second bounds
                        gti_column_start = fits.ColDefs([fits.Column(name='START', format='D',
                                                                     array=np.array(
                                                                         [time_obs[elem] + start_obs_s - 0.5 for elem in
                                                                          gti_intervals[0]]))])
                        gti_column_stop = fits.ColDefs([fits.Column(name='STOP', format='D',
                                                                    array=np.array(
                                                                        [time_obs[elem] + start_obs_s + 0.5 for elem in
                                                                         gti_intervals[1]]))])

                        # replacing the hdu
                        hdul[1] = fits.BinTableHDU.from_columns(gti_column_start + gti_column_stop)

                        # replacing the header
                        hdul[1].header = prev_header

                        # and the gti keywords
                        hdul[1].header['ONTIME'] = 2 * delta_time_gtis * len(id_gti)

                        hdul[1].header['TSTART'] = hdul[1].data['START'][0] - start_obs_s
                        hdul[1].header['TSTOP'] = hdul[1].data['STOP'][-1] - start_obs_s

                        #NuSTAR values
                        hdul[1].header['MJDREFI'] = 55197
                        hdul[1].header['MJDREFF']=0.00076601852
                        hdul.flush()

                return gti_path

        gti_path_list=[]

        for id_orbit in range(n_orbit):
            gti_path_output=create_gti_files(id_orbit,id_gti_orbit[id_orbit], cut_lc)
            if gti_path_output is not None:
                gti_path_list+=[gti_path_output]

        return gti_path_list

def extract_sp(directory,cams='all',e_low=None,e_high=None,bright=False,gti_mode=False):

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

    def extract_sp_single(spawn, directory, instru, steminput, src_reg, bg_reg, e_low=None, e_high=None,bright=False,
                          gti=None,id_orbit=''):

        id_orbit_str = '-' + str_orbit(id_orbit) if gti is not None else ''

        if e_low!=None:
            pi_low = str(kev_to_PI(float(e_low)))

        if e_high!=None:
            pi_high = str(kev_to_PI(float(e_high)))

        # removing the first directory to match the spawn cd
        gti_spawn = '' if gti is None else '/'.join(gti.split('/')[1:])

        cam_suffix='A01' if instru=='FPMA' else 'B01' if instru=='FPMB' else ''
        # building the spectral products
        spawn.sendline('nuproducts indir=./out'+('_bright' if bright else '')+
                       ' instrument=' + instru + ' steminputs=' + steminput +
                       ' stemout='+steminput+cam_suffix+id_orbit_str+
                       ' srcregionfile=' + src_reg +' bkgregionfile=' + bg_reg +
                       ('' if e_low==None else ' pilow=' + pi_low)+
                       ('' if e_high==None else ' pihigh=' + pi_high)+
                       ' outdir=./products'+('_bright' if bright else '')+
                       ' lcfile=NONE bkglcfile=None imagefile=NONE'+
                       ' clobber=yes'+(' usrgtifile='+gti_spawn if gti is not None else ''))

        ####TODO: check what's the standard message here
        err_code=spawn.expect(['nuproducts_0.3.3: Exit with success','nuproducts error'],timeout=None)

        if err_code!=0:
            return 'Nuproducts error'
        else:
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
    bg_reg = file_evt_selector('bg_reg',cameras=cams,bright=bright)

    if len(src_reg[0][0])!=1 or len(src_reg[0][0])!=1:

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

        bashproc.sendline('cd '+directory)

        # filtering for the selected cameras
        for i_cam in camid_list:

            if directory.endswith('/'):
                obsid = directory.split('/')[-2]
            else:
                obsid = directory.split('/')[-1]

            # checking if gti files exist in the folder
            gti_files = np.array([elem for elem in
                        glob.glob(os.path.join(directory,'products_bright' if bright else 'products','**'),
                                  recursive=True)\
                                  if elem.endswith('.gti') and '_gti_' in elem\
                                  and camlist[i_cam] in elem\
                                  and '_gti_mask_' not in elem])

            gti_files.sort()

            bashproc.logfile_read = sys.stdout
            print('\nComputing spectral products of camera ' + camlist[i_cam])

            # fetching region files for this camera
            src_reg_indiv = '/'.join(np.array(src_reg[i_cam]).T[0][::-1])
            bg_reg_indiv = '/'.join(np.array(bg_reg[i_cam]).T[0][::-1])

            if src_reg_indiv=='/':
                print('Source region file missing')
                return 'Source region file missing'

            # creating a path without the main directory for the spawn
            src_reg_indiv_spawn = src_reg_indiv.replace(directory, '.',1)
            bg_reg_indiv_spawn = bg_reg_indiv.replace(directory, '.',1)

            # launching the main extraction
            if gti_mode:
                for i_gti,elem_gti in enumerate(gti_files):

                    id_orbit_str = '-' + str_orbit(i_gti)

                    summary_line = extract_sp_single(bashproc,directory=directory,instru=camlist[i_cam],
                                                    steminput='nu'+obsid,
                                                    src_reg=src_reg_indiv_spawn,bg_reg=bg_reg_indiv_spawn,
                                                    e_low=e_low,e_high=e_high,bright=bright,
                                                     gti=elem_gti,id_orbit=i_gti)

                    summary_content = obsid + '\t' + camlist[i_cam] +'\t'+ summary_line
                    file_edit(os.path.join(directory, 'summary_extract_sp.log'), obsid +id_orbit_str+
                              '\t' + camlist[i_cam],
                              summary_content + '\n',
                              summary_header)

            else:
                summary_line = extract_sp_single(bashproc,directory=directory,instru=camlist[i_cam],
                                                steminput='nu'+obsid,
                                                src_reg=src_reg_indiv_spawn,bg_reg=bg_reg_indiv_spawn,
                                                e_low=e_low,e_high=e_high,bright=bright)

                summary_content = obsid + '\t' + camlist[i_cam] +'\t'+ summary_line
                file_edit(os.path.join(directory, 'summary_extract_lc.log'), obsid + '\t' + camlist[i_cam],
                          summary_content + '\n',
                          summary_header)

    extract_sp_done.set()


def regroup_spectral(directory, group='opt'):
    '''
    Regroups NuSTAR spectram from an obsid directory using ftgrouppha

    mode:
        -opt: follows the Kastra and al. 2016 binning

    note: will detect all bright and non-bright folders
    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    print('\n\n\nRegrouping spectra in directory '+directory+'...')

    set_var(bashproc)

    currdir = os.getcwd()

    def regroup_single_spectral(spfile, spfile_dir):

        # note: the spawn is already in spfile dir to group without issues with the directories

        print('Regrouping spectral file ' + spfile)

        spfile_group = spfile.replace('_sr.pha', '_sp_src_grp_' + group + '.pha')
        rmf_file = spfile.replace('.pha', '.rmf')

        # print for saving in the log file since it doesn't show clearly otherwise
        print('ftgrouppha infile=' + spfile + ' outfile=' + spfile_group +
              ' grouptype=' + group + ' respfile=' + rmf_file)

        bashproc.sendline('ftgrouppha infile=' + spfile + ' outfile=' + spfile_group +
                          ' grouptype=' + group + ' respfile=' + rmf_file)

        time.sleep(1)

        while not os.path.isfile(os.path.join(spfile_dir, spfile_group)):
            time.sleep(1)
            print('Waiting for creation of file ' + os.path.join(os.path.join(spfile_dir, spfile_group)))

        bashproc.sendline('echo done')

        bashproc.expect('done')

    if os.path.isfile(directory + '/regroup_spectral.log'):
        os.system('rm ' + directory + '/regroup_spectral.log')

    with StdoutTee(directory + '/regroup_spectral.log', mode="a", buff=1, file_filters=[_remove_control_chars]), \
            StderrTee(directory + '/regroup_spectral.log', buff=1, file_filters=[_remove_control_chars]):

        bashproc.logfile_read = sys.stdout

        # listing spectral files in the folder
        spfile_paths= np.array([elem for elem in glob.glob(os.path.join(directory,'**'), recursive=True) if
                              elem.endswith('_sr.pha')])

        spfile_paths.sort()

        if len(spfile_paths)==0:
            print('No spectral file detected.')
            return 'No spectral file detected.'
        else:
            print('\nFound spectral files: ')
            print(spfile_paths)

        for elem_path in spfile_paths:

            # deleting previously existing grouped spectra to avoid problems when testing their existence
            if os.path.isfile(os.path.join(directory,elem_path)):
                os.remove(os.path.join(directory,elem_path))

            elem_dir=elem_path[:elem_path.rfind('/')]
            elem_file=elem_path.split('/')[-1]

            bashproc.sendline('cd '+currdir)
            bashproc.sendline('cd '+elem_dir)

            process_state = regroup_single_spectral(elem_file,elem_dir)


            # stopping the loop in case of crash
            if process_state is not None:
                # exiting the bashproc
                bashproc.sendline('exit')
                regroup_spectral_done.set()

                # raising an error to stop the process if the command has crashed for some reason
                return 'spfile ' + elem_path + ': ' + process_state

        # exiting the bashproc
        bashproc.sendline('exit')
        regroup_spectral_done.set()


def batch_mover(directory,bright_check=True,force_bright=False):

    '''
    copies all spectral products in a directory to a bigbatch directory
    above the obsid directory to prepare for spectrum analysis

    copies the lc files independantly to a lcbatch folder

    can copy either the products_bright or the products folder depending on the bright keywords:
        -if force_bright is set to True, copies ONLY the products_bright folder
        -if bright_check is set to True and force_bright to False, copies products_bright if it's there else output
        -if both are set to False, copies ONLY the products folder


    '''

    bashproc = pexpect.spawn("/bin/bash", encoding='utf-8')

    bashproc.logfile_read = sys.stdout

    print('\n\n\nCopying products to a merging directory...')

    set_var(bashproc)

    os.system('mkdir -p bigbatch')

    os.system('mkdir -p lcbatch')


    time.sleep(1)

    bashproc.sendline('cd ' + directory)

    if force_bright or (bright_check and os.path.isdir(os.path.join(directory,'products_bright'))):
        print('Copying bright mode products')
        merge_dir='products_bright'
        dr_dir='out_bright'
    else:
        print('Copying standard products')
        merge_dir='products'
        dr_dir='out'

    copy_files=[elem for elem in glob.glob(os.path.join(directory,merge_dir,'**')) if elem.endswith('.png') or \
                elem.endswith('.pha') or elem.endswith('.rmf') or elem.endswith('.arf')]
    copy_files_lc=[elem for elem in glob.glob(os.path.join(directory,merge_dir,'**')) if elem.endswith('.lc')]

    #also adding the region crops
    copy_files+=[elem for elem in glob.glob(os.path.join(directory,dr_dir,'**')) if elem.endswith('.png')]


    for elem_file in copy_files:

        print('Copying file '+elem_file)

        print('cp --verbose ' + elem_file + ' bigbatch')
        os.system('cp --verbose ' + elem_file + ' bigbatch')

        while not os.path.isfile(os.path.join('bigbatch',elem_file.split('/')[-1])):
            print('waiting for file '+elem_file+' to copy...')
            time.sleep(1)

    for elem_file_lc in copy_files_lc:

        print('Copying lc file '+elem_file_lc)

        print('cp --verbose ' + elem_file_lc + ' lcbatch')
        os.system('cp --verbose ' + elem_file_lc + ' lcbatch')

        while not os.path.isfile(os.path.join('lcbatch',elem_file_lc.split('/')[-1])):
            print('waiting for file '+elem_file_lc+' to copy...')
            time.sleep(1)

    time.sleep(1)
    print('\nMerge complete')

    bashproc.sendline('exit')
    batch_mover_done.set()

def clean_all(directory):

    '''

    clean products in the products and out directories and the logs in the main  directory

    Useful to avoid bloating with how big these files are
    '''

    log_files=[elem for elem in glob.glob(os.path.join(directory,'**'),recursive=False)\
                   if (not os.path.isdir(elem) and elem.endswith('.log') and 'pipe.log' not in elem) ]

    product_files=[elem for elem in glob.glob(os.path.join(directory,'products**/**'),recursive=True) if not elem.endswith('/')]

    product_dirs=[elem for elem in glob.glob(os.path.join(directory,'products**/'),recursive=True)]

    out_files=[elem for elem in glob.glob(os.path.join(directory,'out**/**'),recursive=True) if not elem.endswith('/')]

    out_dirs=[elem for elem in glob.glob(os.path.join(directory,'out**/'),recursive=True)]

    clean_files=product_files+out_files+log_files

    clean_dirs=out_dirs+product_dirs

    print('Cleaning ' + str(len(clean_files)+len(clean_dirs)) + ' elements in directory ' + directory)

    if len(clean_files)>0 or len(clean_dirs)>0:

        for elem_product in clean_files:
            os.remove(elem_product)

        #removing directories
        try:
            for elem_dir in clean_dirs:
                os.system('rmdir '+elem_dir)
        except:
            breakpoint()

        #reasonable waiting time to make sure big files can be deleted
        time.sleep(10)

        print('Cleaning complete.')

    clean_all_done.set()
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
subdirs=glob.glob('**/',recursive=False)
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

started_folders,done_folders=startdir_state(args.action)

if not local:
    #checking them in search for ODF directories
    for directory in subdirs:

        #bright flag to relaunch the analysis for bright sources if needed and asked
        bright_flag_dir=False

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
        if len(dirname)==11 and dirname.isdigit() and "odf" not in os.listdir(directory):

            print('\nFound obsid directory '+dirname)


            above_obsdir=os.path.join(startdir,'/'.join(directory[-1].split('/')[:-1]))

            os.chdir(above_obsdir)

            # note: in this code we use a while loop with indexing to be able to reset the loop position if needed
            id_action = 0

            if catch_errors:
                try:
                    while id_action<len(action_list):
                        curr_action=action_list[id_action]

                        #resetting the error string message
                        output_err=None
                        folder_state='Running '+curr_action

                        if curr_action=='build':
                            process_obsdir(dirname,overwrite=overwrite_glob,bright=force_bright or bright_flag_dir)
                            process_obsdir_done.wait()

                        #note: the first actions are not performed with bright mode
                        if curr_action=='reg':
                            output_err=extract_reg(dirname,cams=cameras_glob,use_file_target=use_file_target,overwrite=overwrite_glob,
                                                   e_low_img=e_low_img,e_high_img=e_high_img,rad_crop=rad_crop,
                                                   bg_area_factor=bg_area_factor,
                                                   bg_rm_src_sigmas=bg_rm_src_sigmas,
                                                   bg_distrib_cut=bg_distrib_cut,
                                                   bright=force_bright or bright_flag_dir,
                                                   sudo_mode=sudo_mode,sudo_mdp=sudo_mdp)
                            if type(output_err)==str:
                                raise ValueError
                            extract_reg_done.wait()

                        if curr_action=='lc':
                            output_lc=extract_lc(dirname,binning=lc_bin_std,lc_bands_str=lc_bands_str,hr_bands=hr_bands_str,cams=cameras_glob,
                                                  bright=force_bright or bright_flag_dir,make_gtis=make_gti_orbit,
                                                 gti_tool=gti_tool,gti_binning=lc_bin_gti,use_gtis=use_gtis)

                            if type(output_lc)==str:
                                raise ValueError

                            elif output_lc:

                                #doing it this way to keep the bright flag to True on the second run when the output_lc
                                #bright flag is set to false to avoid infinite computations
                                bright_flag_dir=bright_flag_dir or output_lc

                                if bright_check and output_lc:

                                    print("bright obs detected. Restarting the computations in bright mode...")
                                    #resetting the position to the start of the actions to relaunch the computations now that the bright flag has
                                    #and ensuring we rebuild first
                                    if action_list[0]!='build':
                                        action_list=['build']+action_list

                                    if action_list[1]!='reg':
                                        action_list=[action_list[0]]+['reg']+action_list[1:]
                                    id_action=-1

                            extract_lc_done.wait()


                        if curr_action=='sp':
                            output_err=extract_sp(dirname,cams=cameras_glob,e_low=e_low_sp,e_high=e_high_sp,
                                                  bright=force_bright or bright_flag_dir,gti_mode=make_gti_orbit)

                            if type(output_err)==str:
                                raise ValueError
                            extract_sp_done.wait()

                        if curr_action=='g':
                            output_err=regroup_spectral(dirname,group=grouptype)
                            if type(output_err)==str:
                                raise ValueError
                            regroup_spectral_done.wait()

                        if curr_action=='m':
                            batch_mover(dirname,bright_check=bright_check,force_bright=force_bright)
                            batch_mover_done.wait()

                        if curr_action=='fc':
                            clean_all(dirname)
                            clean_all_done.wait()

                        os.chdir(startdir)

                        id_action+=1

                    folder_state='Done'

                except:
                    #signaling unknown errors if they happened
                    if 'Running' in folder_state:
                        print('\nError while '+folder_state)
                        folder_state=folder_state.replace('Running','Aborted at')+('' if output_err is None else ' --> '+output_err)
                    os.chdir(startdir)
            else:

                while id_action < len(action_list):

                    curr_action = action_list[id_action]

                    folder_state='Running '+curr_action

                    if curr_action == 'build':
                        process_obsdir(dirname, overwrite=overwrite_glob, bright=force_bright or bright_flag_dir)
                        process_obsdir_done.wait()

                    # note: the first actions are not performed with bright mode
                    if curr_action == 'reg':
                        output_err = extract_reg(dirname, cams=cameras_glob, use_file_target=use_file_target,
                                                 overwrite=overwrite_glob,
                                                 e_low_img=e_low_img, e_high_img=e_high_img, rad_crop=rad_crop,
                                                 bg_area_factor=bg_area_factor,
                                                 bg_rm_src_sigmas=bg_rm_src_sigmas,
                                                 bg_distrib_cut=bg_distrib_cut,
                                                 bright=force_bright or bright_flag_dir,
                                                 sudo_mode=sudo_mode, sudo_mdp=sudo_mdp)
                        if type(output_err) == str:
                            raise ValueError
                        extract_reg_done.wait()

                    if curr_action == 'lc':
                        output_lc = extract_lc(dirname, binning=lc_bin_std, lc_bands_str=lc_bands_str, hr_bands=hr_bands_str,
                                               cams=cameras_glob,
                                               bright=force_bright or bright_flag_dir,make_gtis=make_gti_orbit,
                                                 gti_tool=gti_tool,gti_binning=lc_bin_gti,use_gtis=use_gtis)

                        if type(output_lc) == str:
                            raise ValueError

                        elif output_lc:

                            #doing it this way to keep the bright flag to True on the second run when the output_lc
                            #bright flag is set to false to avoid infinite computations
                            bright_flag_dir=bright_flag_dir or output_lc

                            if bright_check and output_lc:

                                print("bright obsd detected. Restarting the computations in bright mode...")
                                # resetting the position to the start of the actions to relaunch the computations now that the bright flag has
                                # and ensuring we rebuild first
                                if action_list[0] != 'build':
                                    action_list = ['build'] + action_list

                                if action_list[1] != 'reg':
                                    action_list = [action_list[0]] + ['reg'] + action_list[1:]
                                id_action = -1

                        extract_lc_done.wait()

                    if curr_action == 'sp':
                        output_err = extract_sp(dirname, cams=cameras_glob, e_low=e_low_sp, e_high=e_high_sp,
                                                bright=force_bright or bright_flag_dir,gti_mode=make_gti_orbit)

                        if type(output_err) == str:
                            raise ValueError
                        extract_sp_done.wait()

                    if curr_action=='g':
                        output_err=regroup_spectral(dirname,group=grouptype)
                        if type(output_err)==str:
                            raise ValueError
                        regroup_spectral_done.wait()

                    if curr_action=='m':
                        batch_mover(dirname,bright_check=bright_check,force_bright=force_bright)
                        batch_mover_done.wait()

                    if curr_action=='fc':
                        clean_all(dirname)
                        clean_all_done.wait()

                    os.chdir(startdir)

                    id_action+=1

                folder_state='Done'

            #adding the directory to the list of already computed directories
            file_edit('summary_folder_analysis_'+args.action+'.log',directory,directory+'\t'+folder_state+'\n',summary_folder_header)

else:
    #taking of the merge action if local is set since there is no point to merge in local (the batch directory acts as merge)
    action_list=[elem for elem in action_list if elem!='m']

    id_action=0

    absdir=os.getcwd()

    while id_action < len(action_list):
        curr_action = action_list[id_action]

        folder_state = 'Running ' + curr_action

        if curr_action == 'build':
            process_obsdir(absdir, overwrite=overwrite_glob, bright=force_bright or bright_flag_dir)
            process_obsdir_done.wait()

        # note: the first actions are not performed with bright mode
        if curr_action == 'reg':
            output_err = extract_reg(absdir, cams=cameras_glob, use_file_target=use_file_target,
                                     overwrite=overwrite_glob,
                                     e_low_img=e_low_img, e_high_img=e_high_img, rad_crop=rad_crop,
                                     bg_area_factor=bg_area_factor,
                                     bg_rm_src_sigmas=bg_rm_src_sigmas,
                                     bg_distrib_cut=bg_distrib_cut,
                                     bright=force_bright or bright_flag_dir,
                                     sudo_mode=sudo_mode, sudo_mdp=sudo_mdp)
            if type(output_err) == str:
                raise ValueError
            extract_reg_done.wait()

        if curr_action == 'lc':
            output_lc = extract_lc(absdir, binning=lc_bin_std, lc_bands_str=lc_bands_str, hr_bands=hr_bands_str,
                                   cams=cameras_glob,
                                   bright=force_bright or bright_flag_dir,make_gtis=make_gti_orbit,
                                                 gti_tool=gti_tool,gti_binning=lc_bin_gti,use_gtis=use_gtis)

            if type(output_lc) == str:
                raise ValueError

            elif output_lc:

                # doing it this way to keep the bright flag to True on the second run when the output_lc
                # bright flag is set to false to avoid infinite computations
                bright_flag_dir = bright_flag_dir or output_lc

                if bright_check and output_lc == True:
                    print("bright obsd detected. Restarting the computations in bright mode...")
                    # resetting the position to the start of the actions to relaunch the computations now that the bright flag has
                    # and ensuring we rebuild first
                    if action_list[0] != 'build':
                        action_list = ['build'] + action_list

                    if action_list[1] != 'reg':
                        action_list = [action_list[0]] + ['reg'] + action_list[1:]
                    id_action = -1

            extract_lc_done.wait()

        if curr_action == 'sp':
            output_err = extract_sp(absdir, cams=cameras_glob, e_low=e_low_sp, e_high=e_high_sp,
                                    bright=force_bright or bright_flag_dir,gti_mode=make_gti_orbit)

            if type(output_err) == str:
                raise ValueError
            extract_sp_done.wait()

        if curr_action == 'g':
            output_err = regroup_spectral(absdir, group=grouptype)
            if type(output_err) == str:
                raise ValueError
            regroup_spectral_done.wait()

        if curr_action == 'm':
            batch_mover(absdir, bright_check=bright_check, force_bright=force_bright)
            batch_mover_done.wait()

        if curr_action == 'fc':
            clean_all(absdir)
            clean_all_done.wait()

        id_action+=1
import numpy as np
from matplotlib.ticker import Locator
import time
import os
import astropy.units as u
import random
import glob
import warnings
from astropy.io import fits
from astroquery.simbad import Simbad
import io
import zipfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time,TimeDelta
from mpdaf.obj import WCS as mpdaf_WCS
from astropy.wcs import WCS as astroWCS
from mpdaf.obj import sexa2deg,deg2sexa,Image

def mpdaf_load_img(sky_img_path,rotate=False):
    # loading the IMG file with mpdaf

    #NOTE: OUTDATED: normal astro WCS can be accessed as the wcs method within the mpdaf wcs. Should be updated
    with fits.open(sky_img_path) as hdul:
        try:

            if rotate:
                img_data=hdul[0].data[::-1].T
            else:
                img_data = hdul[0].data
            src_mpdaf_WCS = mpdaf_WCS(hdul[0].header)
            src_astro_WCS = astroWCS(hdul[0].header)

            img_obj_whole = Image(data=img_data, wcs=src_mpdaf_WCS)
        except:

            if rotate:
                img_data=hdul[1].data.T
            else:
                img_data = hdul[1].data
                src_mpdaf_WCS = mpdaf_WCS(hdul[1].header)
            src_astro_WCS = astroWCS(hdul[1].header)

            img_obj_whole = Image(data=img_data, wcs=src_mpdaf_WCS)

    return img_obj_whole,src_mpdaf_WCS,src_astro_WCS

def mpdaf_plot_img(sky_img_path, rad_crop=[200, 200], rad_crop_u='pixel', center_crop_u=None, crop_coords=None,
                   target_name_list=[None],

                   #careful: if iterable, then elements in str are assumed to be in sexa, elements in float in deg
                   target_coords_list='target',

                   target_sizes=[10],
                   target_size_u=['pixel'],
                   target_colors=['green'],
                   target_ls='auto',
                   target_disp_names=['auto'],
                   target_disp_names_offset=[1.1],
                   title='', save=False, rotate=False, img_scale='log'):
    '''
    Plot an mpdaf image in sky coordinates, with a given cropping if requested,
        and additional regions highlighting sources if requested.

        The crop is made centered on the position of the first source if crop_coords is None, otherwise to
        the coordinates given
        Note that the crop
        source_names/target_coords/target_sizes/target_sizes_u iterables of the same len
        source_names/target_coords are used in target_deg to get the position of the sources
        target_sizes gives the source region(s)

        target_cords_list:

            'target' to follow the targets

            note: by default target_sizes is in pixel, so will have to be converted accordingly

            other unit available is for now 'arcsec'


            reminder: xtend r_arsec=1.768*r_pixel

        target_size_u: iterable of len the number of targets, containing instances of 'pixel' or 'arcsec'

        target_disp_names:
            -str to use for each source (can be used independetly from name reconigtion or source coordinates)
        as of now target_coords_list should be in explicit radec decimal coordinates if not set to target

    #note: there is a small offset sometimes due to the approximation of the coordinates when cropping
    '''

    img_obj_whole, src_mpdaf_WCS, src_astro_WCS = mpdaf_load_img(sky_img_path, rotate=rotate, )

    if target_name_list[0] is None and (target_coords_list[0] is None or target_coords_list == 'target'):
        obj_deg_list = []
    else:
        obj_deg_list = np.array([target_deg(elem_source_name, elem_target_coords) for
                                 (elem_source_name, elem_target_coords) in zip(target_name_list, target_coords_list)],
                                dtype=float)

    if crop_coords is None:
        if len(obj_deg_list) != 0:
            crop_center = obj_deg_list[0]
    elif type(crop_coords) == str and crop_coords == 'auto':
        with fits.open(sky_img_path) as hdul:
            crop_center = [hdul[0].header['RA_PNT'], hdul[0].header['DEC_PNT']]
    else:
        if type(crop_coords[0]) == str:
            crop_center = sexa2deg(crop_coords[::-1])[::-1]
        else:
            crop_center = crop_coords

    if crop_coords != None:
        try:
            imgcrop_src = img_obj_whole.copy().subimage(center=crop_center[::-1], size=rad_crop,
                                                        unit_center=None if center_crop_u == 'pixel' else u.deg,
                                                        unit_size=None if rad_crop_u == 'pixel' else u.arcsec)
        except:
            print('\nCropping region entirely out of the image. Field of view issue....')
            return '\nCropping region entirely out of the image. Field of view issue....'
    else:
        imgcrop_src = img_obj_whole
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

    fig_catal_crop, ax_catal_crop = plt.subplots(1, 1, subplot_kw={'projection': imgcrop_src.wcs.wcs},
                                                 figsize=(12, 10))
    circle_rad_pos = []
    target_circles = []

    coord_crop_eff = [imgcrop_src.wcs.naxis1, imgcrop_src.wcs.naxis2][::-1]
    coord_start_eff = imgcrop_src.wcs.get_start()[::-1]

    # note that there are some small uncertainties remaining
    # the non-cropped version seems slightly worse than the cropped version when comparing to ds9

    if rad_crop is not None:

        axis_increment_eff = 1 / ((imgcrop_src.wcs.get_end() - imgcrop_src.wcs.get_start()) / rad_crop)[::-1]

        # old and outdated
        # axis_increment_eff=1/(np.array([(imgcrop_src.wcs.get_end()[i]-imgcrop_src.wcs.get_start()[i])/(2*rad_crop[i])
        #             for i in range(2)][::-1])/2)

    else:
        img_size = coord_crop_eff
        axis_increment_eff = 1 / ((imgcrop_src.wcs.get_end() - imgcrop_src.wcs.get_start()) / coord_crop_eff)[::-1]

    if len(obj_deg_list) > 0:
        target_sizes_pix = [
            target_sizes[i] if target_size_u[i] == 'pixel' else target_sizes[i] * abs(axis_increment_eff[1]) / 3600
            if target_size_u[i] == 'arcsec' else None for i in range(len(obj_deg_list))]

    for i_target, (elem_ra_deg, elem_dec_deg) in enumerate(obj_deg_list):

        circle_rad_pos += [[(elem_ra_deg - coord_start_eff[0]) * axis_increment_eff[0],
                            ((elem_dec_deg - coord_start_eff[1]) * axis_increment_eff[1]) + 0.5]]

        target_circles += [plt.Circle([circle_rad_pos[-1][0], circle_rad_pos[-1][1]], target_sizes_pix[i_target],
                                      color=target_colors[i_target],
                                      ls='-' if type(target_ls) == str else target_ls[i_target],
                                      zorder=1000, fill=False)]

        if target_disp_names[i_target] != '':
            if target_disp_names[i_target] == 'auto':
                curr_target_name = target_name_list[i_target]
            else:
                curr_target_name = target_disp_names[i_target]

            ax_catal_crop.text(circle_rad_pos[-1][0],
                               circle_rad_pos[-1][1] + target_sizes_pix[i_target] * target_disp_names_offset[i_target],
                               curr_target_name,
                               color=target_colors[i_target], horizontalalignment='center')

    # testing if the resulting image is empty
    if len(imgcrop_src.data.nonzero()[0]) == 0:
        print('Cropped image empty.')
        return 'Cropped image empty.'

    if title != '':
        ax_catal_crop.set_title(title)
    catal_plot = imgcrop_src.plot(cmap='plasma', scale=img_scale)
    plt.colorbar(catal_plot, location='bottom', fraction=0.046, pad=0.04)
    for elem_circle in target_circles:
        ax_catal_crop.add_patch(elem_circle)

    # updating the colorbar
    ax_cb = ax_catal_crop.get_figure().get_children()[-1]
    ax_cb.set_xticks(np.logspace(0, np.log10(imgcrop_src.data.max()), 8))

    # adding a top xaxis label since the cmap is sometimes hiding the bottom one
    ax_catal_crop.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=True,
        labeltop=True,
        direction='out')

    plt.tight_layout()

    if save is not False :
        save_str= sky_img_path[:sky_img_path.rfind('.')] + '_mpdaf_img.pdf' if save=='auto' else save
        plt.savefig(save_str)
        plt.close()
        plt.ion()

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
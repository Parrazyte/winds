#!/usr/bin/env python3

""" xtend_pileup_gen_plfraction.py

This is a python script to plot the radial profile of the image.
History:
2024-05-20 ; ver 1; S.Yamada
2024-05-21 ; ver 1.1; Y.Sakai, update pileup fraction image
2024-06-16 ; ver 2; S.Yamada, found a bug in pfrac thanks to Nobukawa-kun
2025:XX: Forked by M. PARRA, adding some QOL stuff

"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cr
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import convolve
import pandas as pd

plt.ioff()
params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# global variable for pileup estimateion
target_values = [0.03, 0.01]  # pileup fraction 3%, 1%
target_lss = ["-", "--"]  # pileup linestyle


class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def color_print(msg, color):
    print(f"{color}{msg}{ConsoleColors.ENDC}")


def calc_pileupfraction(r):
    # Convert input 'r' to a numpy array if it is not already
    r = np.asarray(r)  # Ensure 'r' is a numpy array for element-wise operations

    # Initialize an array of zeros with the same shape as 'r'
    result = np.zeros_like(r)  # To store the final pile-up fraction values

    # Create a mask for elements where 'r' > 0
    mask = (r > 0)  # Boolean mask to identify valid (positive) values in 'r'

    # Calculate the pile-up fraction for elements where 'r' > 0
    result[mask] = (1. - (1. + r[mask]) * np.exp(-r[mask])) / (1. - np.exp(-r[mask]))
    # Formula explanation:
    # (1 - (1 + r) * exp(-r)) / (1 - exp(-r)) calculates the pile-up fraction
    # Only applied to elements where 'r' > 0, using the mask

    # Return the array of pile-up fractions
    return result  # 'result' contains the computed values for all elements in 'r'


def generate_region_file(c_ra, c_dec, rin, rout, output_dir=".", unit='arcsec'):
    """
    Generate a .reg file for DS9.

    Parameters:
        c_ra (float): Central RA coordinate.
        c_dec (float): Central DEC coordinate.
        rin (int): Inner radius.
        rout (int): Outer radius.
        output_dir (str): Directory to save the output file.
    """
    # Automatically generate the filename
    filename = f"region_{int(rin)}_{int(rout)}.reg"
    filepath = os.path.join(output_dir, filename)

    if unit == 'pixel':
        pixel_to_arcsec = 1.767996

        rin_use = rin / pixel_to_arcsec

        rout_use = rout / pixel_to_arcsec

        coord_line = 'physical\n'
    else:
        rin_use = rin
        rout_use = rout
        coord_line = 'fk5\n'

    # Template for DS9 format
    if rin == 0:
        content = (
                "# Region file format: DS9 version 4.1\n"
                "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
                + coord_line +
                f"circle({c_ra},{c_dec},{rout_use}" + ('' if unit == 'pixel' else "\"") + ")\n"
        )
    else:
        content = (
                "# Region file format: DS9 version 4.1\n"
                "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
                + coord_line +
                f"annulus({c_ra},{c_dec},{rin_use}" + ('' if unit == 'pixel' else "\"") +
                "," + str(rout_use) + ('' if unit == 'pixel' else "\"") + ")\n"
        )

    # Write to file
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Generated: {filepath}")


def interpolate_centers(radial_centers, radial_pileupfraction, target_values=[0.03, 0.01]):
    # Initialize an empty list to store the interpolated center values
    result = []

    # Loop over each target value in the list of target pile-up fractions
    for target in target_values:
        # Default value for interpolation result; -1 indicates no valid interpolation found
        interp_center = -1

        # Iterate through the pile-up fraction array to find the interval containing the target
        for i in range(1, len(radial_pileupfraction)):
            # Check if the target lies between two consecutive pile-up fraction values
            if (radial_pileupfraction[i - 1] >= target >= radial_pileupfraction[i] or
                    radial_pileupfraction[i - 1] <= target <= radial_pileupfraction[i]):
                # Perform linear interpolation to find the radial center corresponding to the target
                interp_center = np.interp(
                    target,  # The target value to interpolate for
                    [radial_pileupfraction[i - 1], radial_pileupfraction[i]],  # y-values
                    [radial_centers[i - 1], radial_centers[i]]  # x-values
                )
                break  # Exit the loop once the target interval is found and interpolated

        # Append the interpolated center value to the result list
        result.append(interp_center)

    # Return the list of interpolated radial centers
    return result


class Fits:
    """Class for handling FITS files and plotting radial profiles."""

    def __init__(self, eventfile, debug=False):
        self.eventfilename = eventfile
        self.debug = debug

        if os.path.isfile(eventfile):
            self._initialize_fits_data(eventfile)
            self._print_initialization_info()
        else:
            print(f"ERROR: Can't find the fits file: {self.eventfilename}")
            quit()

    def _check_if_eventfile(self):
        """ check if the inputfile is image FITS, not event FITS"""
        try:
            hdu1 = fits.open(self.filename)[1]
            if hdu1.header["EXTNAME"] == "EVENTS":
                color_print(f"[ERROR] input file should be image fits. ", ConsoleColors.FAIL)
                color_print(f"You should start from xtend_pileup_check_quick.sh", ConsoleColors.FAIL)
                return
                False
        except:
            pass
        return True

    def _initialize_fits_data(self, eventfile):
        """Initialize FITS data and extract header information."""
        self.filename = eventfile
        if not self._check_if_eventfile():
            sys.exit()
        self.hdu = fits.open(self.filename)[0]
        self.wcs = WCS(self.hdu.header)
        self.cdelt = self.wcs.wcs.cdelt
        self.p2deg = np.abs(self.cdelt[0])  # [deg/pixel]
        self.p2arcsec = 3600.0 * np.abs(self.cdelt[0])  # [arcsec/pixel]
        self.dirname = f"{eventfile.rsplit('.', 1)[0]}"
        self.data = self.hdu.data
        self.dateobs = self.hdu.header["DATE-OBS"]
        self.obsid = self.hdu.header["OBS_ID"]
        self.object = self.hdu.header["OBJECT"]
        self.datamode = self.hdu.header["DATAMODE"]
        self.exposure = self.hdu.header["EXPOSURE"]
        self.timedel = self.hdu.header["TIMEDEL"]
        self.c_ra = self.hdu.header["RA_NOM"]
        self.c_dec = self.hdu.header["DEC_NOM"]

        color_print(
            f"[init] ... dateobs={self.dateobs}, obsid={self.obsid}, object={self.object}, datamode={self.datamode}",
            ConsoleColors.OKGREEN)
        color_print(f"[init] ... exposure={self.exposure}, timedel={self.timedel} ....", ConsoleColors.OKGREEN)

        self._process_data()

    def _process_data(self):
        """Process FITS data to handle NaNs and ensure positivity."""
        self.data[np.isnan(self.data)] = 0
        self.data = np.abs(self.data)

    def _print_initialization_info(self):
        """Print initialization info for debugging purposes."""
        color_print("[def _print_initialization_info]", ConsoleColors.OKBLUE)
        print(f"      self.p2deg      = {self.p2deg} [deg/pixel]")
        print(f"      self.p2arcsec   = {self.p2arcsec} [arcsec/pixel]")

        if self.debug:
            print("... in __init__ Class Fits()")
            print(f"filename = {self.filename}")
            print(f"hdu = {self.hdu}")
            print(f"data.shape = {self.data.shape}")
            self.wcs.printwcs()

    def plotwcs(self, x_center, y_center, vmin=1e-1, vmax=20, manual=False):

        """Plot the entire image with WCS."""
        color_print("\n[def plotwcs]", ConsoleColors.OKBLUE)
        if not manual:
            vmin = np.amin(self.hdu.data[self.hdu.data > 0]) + 0.1  # Avoid vmin being too close to 0 for LogNorm
            vmax = np.amax(self.hdu.data)
        color_print(f"     vmin = {vmin}, vmax = {vmax}", ConsoleColors.OKCYAN)

        plt.figure(figsize=(12, 8))
        plt.figtext(0.1, 0.95,
                    f"OBSID={self.obsid} {self.object} mode={self.datamode} exp={self.exposure} unit={self.p2arcsec}[arcsec/pixel] timedel={self.timedel}")
        plt.figtext(0.05, 0.025, f"fname={self.filename}")

        img = plt.imshow(self.data, origin='lower', cmap=plt.cm.CMRmap, norm=cr.LogNorm(vmin=vmin, vmax=vmax))
        plt.scatter(x_center, y_center, c="k", s=300, marker="x", alpha=0.3)

        cbar = plt.colorbar(img)
        cbar.set_label('Intensity (c/bin)', rotation=270, labelpad=15)

        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfigname = f"{self.dirname}_plotwcs.pdf"
        plt.savefig(outputfigname)
        color_print(f"..... {outputfigname} is created.", ConsoleColors.OKGREEN)

    def plotwcs_ps(self, x_center, y_center, search_radius=40, vmin=1e-1, vmax=20, manual=False):
        """Plot the enlarged image around the given center."""

        color_print("\n[def plotwcs_ps]", ConsoleColors.OKBLUE)
        if not manual:
            vmin = np.amin(self.hdu.data[self.hdu.data > 0]) + 0.1  # Avoid vmin being too close to 0 for LogNorm
            vmax = np.amax(self.hdu.data)
        color_print(f"     vmin = {vmin}, vmax = {vmax}", ConsoleColors.OKCYAN)

        pix_coords = np.array([[x_center, y_center]], dtype=float)
        world_coords = self.wcs.all_pix2world(pix_coords, 1)
        ra, dec = world_coords[0]

        self.ra = ra
        self.dec = dec

        print(f"x_center, y_center = {x_center}, {y_center}")
        print(f"ra, dec    = {ra}, {dec}")

        plt.figure(figsize=(12, 8))
        plt.figtext(0.1, 0.95,
                    f"OBSID={self.obsid} {self.object} mode={self.datamode} exp={self.exposure} unit={self.p2arcsec}[arcsec/pixel] timedel={self.timedel}")
        plt.figtext(0.05, 0.025, f"fname={self.filename}")

        ax = plt.subplot(111, projection=self.wcs)
        img = plt.imshow(self.data, origin='lower', cmap=plt.cm.CMRmap, norm=cr.LogNorm(vmin=vmin, vmax=vmax))
        ax.scatter(x_center, y_center, c="k", s=300, marker="x", alpha=0.3)
        print(
            f"self.hdu.data[x_center][y_center] = {self.hdu.data[x_center][y_center]} (x_center, y_center) = ({x_center}, {y_center})")

        ax.set_xlim(x_center - search_radius, x_center + search_radius)
        ax.set_ylim(y_center - search_radius, y_center + search_radius)
        cbar = plt.colorbar(img)
        cbar.set_label('Intensity (c/bin)', rotation=270, labelpad=15)

        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfigname = f"{self.dirname}_plotwcs_ps_{x_center}_ycenter{y_center}.pdf"
        plt.savefig(outputfigname)
        color_print(f"..... {outputfigname} is created.", ConsoleColors.OKGREEN)

    def mk_radialprofile(self, x_center, y_center, search_radius=40, ndiv=20, vmin=1e-1, vmax=20, manual=False,
                         cellsize=3 * 3,zoom_pix_size=50):
        """Create the radial profiles."""
        color_print("\n[def mk_radialprofile]", ConsoleColors.OKBLUE)
        if not manual:
            vmin = np.amin(self.hdu.data[self.hdu.data > 0]) + 0.1  # Avoid vmin being too close to 0 for LogNorm
            vmax = np.amax(self.hdu.data)
        color_print(f"     vmin = {vmin}, vmax = {vmax}", ConsoleColors.OKCYAN)

        pix_coords = np.array([[x_center, y_center]], dtype=float)
        world_coords = self.wcs.all_pix2world(pix_coords, 1)
        ra, dec = world_coords[0]

        self.ra = ra
        self.dec = dec
        print(f"x_center, y_center = {x_center}, {y_center}")
        print(f"ra, dec    = {ra}, {dec}")

        rc, rp = calc_radial_profile(self.data, x_center, y_center, search_radius=search_radius, ndiv=ndiv,
                                     exposure=self.exposure)
        radial_centers, radial_pileupfraction = calc_radial_pileupfraction(self.data, x_center, y_center, \
                                                                           exposure=self.exposure,
                                                                           search_radius=search_radius, ndiv=ndiv,
                                                                           cellsize=cellsize, timedel=self.timedel)
        print(f" radial_pileupfraction = {radial_pileupfraction}")
        pileupfraction_img = calc_2d_pileupfraction(self.data, exposure=self.exposure, cellsize=cellsize,
                                                    timedel=self.timedel)
        pileup_result = interpolate_centers(radial_centers, radial_pileupfraction, target_values=target_values)
        # Plot images and radial profiles
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.87, wspace=0.5, hspace=0.4)
        plt.figtext(0.1, 0.95,
                    f"OBSID={self.obsid} {self.object} mode={self.datamode} exp={self.exposure} unit={self.p2arcsec}[arcsec/pixel] timedel={self.timedel}")
        plt.figtext(0.1, 0.93,
                    f"Input center [deg] (x, y, ra, dec) = ({x_center:.4f}, {y_center:.4f}, {ra:.4f}, {dec:.4f}) cellsize for pileup = {cellsize}")
        plt.figtext(0.05, 0.025, f"fname={self.filename}")

        outputfigname = f"{self.dirname}_radialprofile_xcenter{x_center}_ycenter{y_center}.pdf"

        ax1 = fig.add_subplot(2, 4, 1)
        self._plot_image(ax1, self.data, "(1) SKY image", x_center, y_center, vmin, vmax)

        ax2 = fig.add_subplot(2, 4, 5, projection=self.wcs)
        self._plot_image(ax2, self.data, "(2) Ra Dec image (FK5)", x_center, y_center, vmin, vmax, wcs=self.wcs)

        #replace the other projections in the same way
        ax3 = fig.add_subplot(2, 4, 2)
        self._plot_image(ax3, self.data, "(3) SKY image", x_center, y_center, vmin, vmax, search_radius)

        ax4 = fig.add_subplot(2, 4, 6, projection=self.wcs)
        self._plot_image(ax4, self.data, "(4) Ra Dec image (FK5)", x_center, y_center, vmin, vmax, search_radius,
                         wcs=self.wcs)

        ax5 = fig.add_subplot(2, 4, 3)
        self._plot_radial_profile(ax5, rc, rp, "(5) Radial profile (pix)", 'radial distance (pixel)',
                                  'c s$^{-1}$ pixel$^{-2}$')

        ax6 = fig.add_subplot(2, 4, 7)
        self._plot_radial_profile(ax6, radial_centers, radial_pileupfraction, "(6) Pileup Fraction",
                                  'radial distance (pixel)', 'pileup fraction')

        for k, (_target, _r, _ls) in enumerate(zip(target_values, pileup_result, target_lss)):
            if _r > 0:
                color_print(f"***** possible pileup : pileup fraction = {_target * 100:.2f} % at {_r:.2f} pixel *****",
                            ConsoleColors.WARNING)
                ax6.axhline(y=_target, color='g', ls=_ls, alpha=0.5,
                            label=f"plfrac={_target * 100:.2f} % at {_r:.2f} pixel")
                plt.legend(numpoints=1, frameon=True)
                # Add the definition in the top-right corner using figtext
                plt.figtext(
                    0.70, 0.49 - k * 0.02,  # X, Y position in figure coordinates (1.0 is the edge of the figure)
                    f"pl. fraction = {_target * 100:.2f} % at {_r:.2f} pixel",  # Text content
                    ha="right",  # Align text to the right
                    va="top",  # Align text to the top
                    fontsize=9,  # Text size
                    color="red",  # Text color
                )
            else:
                plt.figtext(
                    0.70, 0.49 - k * 0.02,  # X, Y position in figure coordinates (1.0 is the edge of the figure)
                    f"pl. fraction = {_target * 100:.2f} % at {_r:.2f} pixel",  # Text content
                    ha="right",  # Align text to the right
                    va="top",  # Align text to the top
                    fontsize=9,  # Text size
                    color="blue",  # Text color
                )

        # Add the definition in the top-right corner using figtext
        plt.figtext(
            0.95, 0.95,  # X, Y position in figure coordinates (1.0 is the edge of the figure)
            "Green solid (---) Pileup fraction 3%\nGreen dashed (- -) Pileup fraction 1%",  # Text content
            ha="right",  # Align text to the right
            va="top",  # Align text to the top
            fontsize=12,  # Text size
            color="green",  # Text color
        )

        ax7 = fig.add_subplot(2, 4, 4)
        pileup_vmin, pileup_vmax = self._pileup_image_vmin_vmax(pileupfraction_img, x_center, y_center, search_radius)

        self._plot_image(ax7, pileupfraction_img, "(7) Pileup Fraction image", x_center, y_center, max(pileup_vmin,1e-3),
                         max(pileup_vmax,0.1), search_radius, label='pileup fraction',cb_extent='min')
        self._plot_contour(ax7, pileupfraction_img, x_center, y_center, search_radius, target_values, target_lss)

        ax8 = fig.add_subplot(2, 4, 8)

        zoom_in_range = zoom_pix_size


        pileup_vmin, pileup_vmax = self._pileup_image_vmin_vmax(pileupfraction_img, x_center, y_center, zoom_in_range)
        self._plot_image(ax8, pileupfraction_img, "(8) Pileup Fraction image (zoom in)", x_center, y_center,
                         max(pileup_vmin,1e-3), max(pileup_vmax,0.1), zoom_in_range*2, label='pileup fraction',cb_extent='min')
        self._plot_contour(ax8, pileupfraction_img, x_center, y_center, zoom_in_range, target_values, target_lss)


        fig_p1,ax_p1=plt.subplots(figsize=(6,6),layout='constrained')
        self._plot_image(ax_p1, self.data, "(1) SKY image", x_center, y_center, vmin, vmax)
        plt.savefig(outputfigname.replace('.pdf','_p1.pdf'))

        fig_p2=plt.figure(figsize=(6,6))
        ax_p2=fig_p2.add_subplot(1,1,1,projection=self.wcs)
        self._plot_image(ax_p2, self.data, "(2) Ra Dec image (FK5)", x_center, y_center, vmin, vmax, wcs=self.wcs)
        plt.savefig(outputfigname.replace('.pdf','_p2.pdf'))

        fig_p3,ax_p3=plt.subplots(figsize=(6,6),layout='constrained')
        self._plot_image(ax_p3, self.data, "(3) SKY image", x_center, y_center, vmin, vmax, search_radius)
        plt.savefig(outputfigname.replace('.pdf','_p3.pdf'))

        fig_p4=plt.figure(figsize=(6,6))
        ax_p4=fig_p4.add_subplot(1,1,1,projection=self.wcs)
        self._plot_image(ax_p4, self.data, "(4) Ra Dec image (FK5)", x_center, y_center, vmin, vmax, search_radius,
                         wcs=self.wcs)
        plt.savefig(outputfigname.replace('.pdf','_p4.pdf'))

        fig_p5,ax_p5=plt.subplots(figsize=(6,6),layout='constrained')
        self._plot_radial_profile(ax_p5, rc, rp, "(5) Radial profile (pix)", 'radial distance (pixel)',
                                  'c s$^{-1}$ pixel$^{-2}$')
        plt.savefig(outputfigname.replace('.pdf','_p5.pdf'))


        fig_p5_zoom,ax_p5_zoom=plt.subplots(figsize=(6,6),layout='constrained')
        self._plot_radial_profile(ax_p5_zoom, rc, rp, "(5) Radial profile (pix)", 'radial distance (pixel)',
                                  'c s$^{-1}$ pixel$^{-2}$')
        plt.xlim(0,50)
        plt.savefig(outputfigname.replace('.pdf','_p5_zoom.pdf'))

        fig_p6,ax_p6=plt.subplots(figsize=(6,6),layout='constrained')
        self._plot_radial_profile(ax_p6, radial_centers, radial_pileupfraction, "(6) Pileup Fraction",
                                  'radial distance (pixel)', 'pileup fraction')
        plt.savefig(outputfigname.replace('.pdf','_p6.pdf'))

        fig_p6_zoom,ax_p6_zoom=plt.subplots(figsize=(6,6),layout='constrained')
        self._plot_radial_profile(ax_p6_zoom, radial_centers, radial_pileupfraction, "(6) Pileup Fraction",
                                  'radial distance (pixel)', 'pileup fraction')
        plt.xlim(0,50)
        plt.savefig(outputfigname.replace('.pdf','_p6_zoom.pdf'))

        fig_p7,ax_p7=plt.subplots(figsize=(6,6),layout='constrained')
        pileup_vmin, pileup_vmax = self._pileup_image_vmin_vmax(pileupfraction_img, x_center, y_center, search_radius)

        # Add the definition in the top-right corner using figtext
        plt.figtext(
            0.95, 0.95,  # X, Y position in figure coordinates (1.0 is the edge of the figure)
            "Green solid (-) Pileup fraction 3%\nGreen dashed (- -) Pileup fraction 1%",  # Text content
            ha="right",  # Align text to the right
            va="top",  # Align text to the top
            fontsize=12,  # Text size
            color="green",  # Text color
        )
        self._plot_image(ax_p7, pileupfraction_img, "(7) Pileup Fraction image", x_center, y_center, max(pileup_vmin,1e-3),
                         max(pileup_vmax,0.1), search_radius, label='pileup fraction',cb_extent='min')
        self._plot_contour(ax_p7, pileupfraction_img, x_center, y_center, search_radius, target_values, target_lss)
        plt.savefig(outputfigname.replace('.pdf','_p7.pdf'))


        fig_p8,ax_p8=plt.subplots(figsize=(6,6),layout='constrained')
        pileup_vmin, pileup_vmax = self._pileup_image_vmin_vmax(pileupfraction_img, x_center, y_center, zoom_in_range)

        # Add the definition in the top-right corner using figtext
        plt.figtext(
            0.95, 0.95,  # X, Y position in figure coordinates (1.0 is the edge of the figure)
            "Green solid (-) Pileup fraction 3%\nGreen dashed (- -) Pileup fraction 1%",  # Text content
            ha="right",  # Align text to the right
            va="top",  # Align text to the top
            fontsize=12,  # Text size
            color="green",  # Text color
        )

        self._plot_image(ax_p8, pileupfraction_img, "(8) Pileup Fraction image (zoom in)", x_center, y_center,
                         max(pileup_vmin,1e-3), max(pileup_vmax,0.1), zoom_in_range*2, label='pileup fraction',cb_extent='min')
        self._plot_contour(ax_p8, pileupfraction_img, x_center, y_center, zoom_in_range, target_values, target_lss)
        plt.savefig(outputfigname.replace('.pdf','_p8.pdf'))

        fig.savefig(outputfigname)
        color_print(f"..... {outputfigname} is created.", ConsoleColors.OKGREEN)

        self._save_radial_profile(rc, rp, f"{self.dirname}_radialprofile_xcenter{x_center}_ycenter{y_center}.txt")
        self._save_radial_profile(radial_centers, radial_pileupfraction,
                                  f"{self.dirname}_pileupfraction_xcenter{x_center}_ycenter{y_center}.txt")
        self._save_pileup_oneline(pileup_result,
                                  f"{self.dirname}_pileupfraction_xcenter{x_center}_ycenter{y_center}.csv")

        return pileup_result

    def _pileup_image_vmin_vmax(self, data, x_center, y_center, search_radius=None, debug=False):
        if debug: print(f"data = {data}")
        if search_radius is None:
            data_copy = data.copy()
        else:
            data_copy = data[y_center - search_radius:y_center + search_radius + 1,
                        x_center - search_radius:x_center + search_radius + 1].copy()

        if debug: print(f"data_copy = {data_copy}")
        clean_data = data_copy[~np.isnan(data_copy)]
        if debug: print(f"clean_data = {clean_data}")
        finite_data = clean_data[np.isfinite(clean_data)]
        if debug: print(f"finite_data = {finite_data}")
        positive_data = finite_data[finite_data > 0]
        if debug: print(f"positive_data = {positive_data}")
        vmin = np.min(positive_data)
        vmax = np.max(positive_data)

        return vmin, vmax

    def _plot_image(self, ax, data, title, x_center, y_center, vmin, vmax, search_radius=None, wcs=None, label=None,
                    cb_extent='neither'):
        ax.set_title(title)
        im = ax.imshow(data, origin='lower', cmap=plt.cm.CMRmap, norm=cr.LogNorm(vmin=vmin, vmax=vmax), aspect='auto',
                       extent=[0, data.shape[1], 0, data.shape[0]])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,extend=cb_extent)
        cbar.set_label(label, rotation=270, labelpad=15)
        ax.scatter(x_center, y_center, c="k", s=300, marker="x", alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if search_radius is not None:
            ax.set_xlim(x_center - search_radius, x_center + search_radius)
            ax.set_ylim(y_center - search_radius, y_center + search_radius)
        if wcs is not None:
            ax.coords.grid(True, color='gray', ls='solid')
        ax.set_aspect('equal', adjustable='box')

    def _plot_contour(self, ax, data, x_center, y_center, search_radius=None, levels=None, linestyles=None):
        sorted_indices = np.argsort(levels)
        sorted_levels = np.array(levels)[sorted_indices]
        sorted_linestyles = np.array(linestyles)[sorted_indices]

        cs = ax.contour(data, levels=sorted_levels, colors='g', origin='lower', linestyles=sorted_linestyles,
                        extent=[0, data.shape[1], 0, data.shape[0]])

        if search_radius is not None:
            ax.set_xlim(x_center - search_radius, x_center + search_radius)
            ax.set_ylim(y_center - search_radius, y_center + search_radius)
        ax.set_aspect('equal', adjustable='box')

    def _plot_radial_profile(self, ax, rc, rp, title, xlabel, ylabel):
        ax.set_title(title)
        ax.errorbar(rc, rp, fmt="ko-", label="data")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(numpoints=1, frameon=True)

    def _save_radial_profile(self, rc, rp, outputcsv):
        with open(outputcsv, "w") as fout:
            for _rc, _rcsec, _rp in zip(rc, rc * self.p2arcsec, rp):
                fout.write(f"{_rc},{_rcsec},{_rp}\n")
        color_print(f"..... {outputcsv} is created.", ConsoleColors.OKGREEN)

    def _save_pileup_oneline(self, pileup_result, outputcsv, visualize=True):
        rough_rate = np.sum(self.data) / self.exposure
        with open(outputcsv, "w") as fout:
            fout.write(
                f"{self.obsid},{pileup_result[0]},{pileup_result[1]},{self.object},{self.datamode},{self.exposure},{self.timedel},{rough_rate}\n")
        color_print(f"..... {outputcsv} is created.", ConsoleColors.OKGREEN)
        if visualize:
            self._load_visualize(outputcsv)

    def _load_visualize(self, outputcsv):
        csvdata = pd.read_csv(outputcsv, header=None, names=[
            "obsid", "pileup_hi", "pileup_lo", "object", "datamode", "exposure", "timedel", "rough_rate"])
        print(f"CSV file '{outputcsv}' loaded successfully.")

        for index, row in csvdata.iterrows():
            obsid = row['obsid']
            pileup_hi = row['pileup_hi']
            pileup_lo = row['pileup_lo']
            exposure = row['exposure']
            rough_rate = row['rough_rate']
            timedel = row['timedel']
            rough_rate = row['rough_rate']

            # Create a bar plot for pileup rate and error
            plt.figure(figsize=(8, 6))
            # Add labels and title
            plt.plot(target_values, [pileup_hi, pileup_lo], "o")
            plt.xlabel("Pileup Fraction")
            plt.ylabel("Pileup Radius (pixel)")

            plt.title(f"Visualization of pileup estimation for OBSID: {obsid}")
            plt.legend()

            # Add text to the plot
            text_str = (f"Object: {self.object}\n"
                        f"OBSID: {self.obsid}\n"
                        f"filename: {self.filename}\n"
                        f"Datamode: {self.datamode}\n"
                        f"Exposure: {exposure:.2f}s\n"
                        f"Lastdel: {timedel:.6f}\n"
                        f"rough_rate: {rough_rate:.4f}\n"
                        f"----------------------------\n"
                        f"Pileup : {pileup_hi:.2f} pixel at pileup fraction = {target_values[0]} %\n"
                        f"Pileup : {pileup_lo:.2f} pixel at pileup fraction = {target_values[1]} %\n"
                        )

            plt.gcf().text(0.2, 0.5, text_str, fontsize=10, va='center', transform=plt.gcf().transFigure)

            # Save the plot as pdf
            output_path = outputcsv.replace(".csv", "_result.pdf")
            plt.savefig(output_path)
            print(f"Visualization saved: {output_path}")
            # Close the figure to free memory
            plt.close()


def calc_radial_pileupfraction(data, x_center, y_center, search_radius=10, ndiv=10, exposure=1e4, cellsize=3 * 3,
                               debug=False, timedel=4.0):
    """
    Calculate the radial pileup fraction around a given center.

    Parameters:
    data (2D array): The input data array.
    x_center (int): The x-coordinate of the center.
    y_center (int): The y-coordinate of the center.
    search_radius (int): The radius around the center to search for the profile. Default is 10.
    ndiv (int): The number of divisions in the radial profile. Default is 10.
    exposure (float): The exposure time to calculate pileup fraction. Default is 1e4.
    cellsize (float): The cell size for single phtoon. Default is 3x3, though need to check.
    debug (bool): If True, print debug information. Default is False.
    timedel (float): readout time for a given count rate

    Returns:
    tuple: The radial coordinates and the normalized radial profile.
    """
    transposed_data = data.T
    radial_bins = np.linspace(0, search_radius, ndiv)
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_profile = np.zeros(len(radial_bins) - 1)

    x_indices, y_indices = np.meshgrid(np.arange(2 * search_radius), np.arange(2 * search_radius), indexing='ij')
    x_indices = x_center - search_radius + x_indices
    y_indices = y_center - search_radius + y_indices

    distances = dist2d(x_indices, y_indices, x_center, y_center)
    values = transposed_data[x_indices, y_indices]

    for k in range(len(radial_bins) - 1):
        mask = (distances >= radial_bins[k]) & (distances < radial_bins[k + 1])
        radial_profile[k] = np.sum(values[mask])

    # Normalize the radial profile and calculate pileup fraction
    areas = np.pi * (radial_bins[1:] ** 2 - radial_bins[:-1] ** 2)
    # pile fraction is c/s/cellsize x timedel ~ counts/1 readout.
    counts_per_cellsize_timedel = radial_profile * cellsize * timedel / (areas * exposure)
    radial_pileupfraction = calc_pileupfraction(counts_per_cellsize_timedel)

    if debug:
        for m in range(len(radial_bins) - 1):
            print(m, radial_profile[m], radial_pileupfraction[m], radial_bins[m], radial_bins[m + 1], areas[m])

    return radial_centers, radial_pileupfraction


def calc_2d_pileupfraction(data, exposure=1e4, cellsize=3 * 3, debug=False, timedel=4.0):
    """
    Calculate the pileup from image information.

    Parameters:
    data (2D array): The input data array.
    exposure (float): The exposure time to calculate pileup fraction. Default is 1e4.
    cellsize (float): The cell size for single phtoon. Default is 3x3, though need to check.
    debug (bool): If True, print debug information. Default is False.
    timedel (float): readout time for a given count rate

    Returns:
    2D array: The pileup 2d image.
    """
    cellsize_kernel = np.ones((3, 3), dtype=data.dtype)
    cellsize_per_exposure = convolve(data, cellsize_kernel, mode='same') * timedel / exposure
    if not cellsize == 9:  # this is a phenomenological way to adjuted 3x3 convolution to cellsize.
        cellsize_per_exposure = cellsize_per_exposure * cellsize / 9

    pileupfraction_img = calc_pileupfraction(cellsize_per_exposure)

    if debug:
        print("vmin = ", np.min(pileupfraction_img), "vmax = ", np.max(pileupfraction_img), "mean = ",
              np.mean(pileupfraction_img))

    return pileupfraction_img


def calc_radial_profile(data, x_center, y_center, search_radius=10, ndiv=10, exposure=1e4, debug=False):
    """
    Calculate the radial profile of the data around a given center.

    Parameters:
    data (2D array): The input data array.
    x_center (int): The x-coordinate of the center.
    y_center (int): The y-coordinate of the center.
    search_radius (int): The radius around the center to search for the profile. Default is 10.
    ndiv (int): The number of divisions in the radial profile. Default is 10.
    debug (bool): If True, print debug information. Default is False.

    Returns:
    tuple: The radial coordinates and the normalized radial profile.
    """
    transposed_data = data.T
    radial_bins = np.linspace(0, search_radius, ndiv)
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_profile = np.zeros(len(radial_bins) - 1)

    x_indices, y_indices = np.meshgrid(np.arange(2 * search_radius), np.arange(2 * search_radius), indexing='ij')
    x_indices = x_center - search_radius + x_indices
    y_indices = y_center - search_radius + y_indices

    distances = dist2d(x_indices, y_indices, x_center, y_center)
    values = transposed_data[x_indices, y_indices]

    for k in range(len(radial_bins) - 1):
        mask = (distances >= radial_bins[k]) & (distances < radial_bins[k + 1])
        radial_profile[k] = np.sum(values[mask])

    # Normalize the radial profile
    areas = np.pi * (radial_bins[1:] ** 2 - radial_bins[:-1] ** 2)
    normalized_radial_profile = radial_profile / (areas * exposure)

    if debug:
        for m in range(len(radial_bins) - 1):
            print(m, radial_profile[m], normalized_radial_profile[m], radial_bins[m], radial_bins[m + 1], areas[m])

    return radial_centers, normalized_radial_profile


def dist2d(x, y, x_center, y_center):
    """Calculate the Euclidean distance between points (x, y) and (x_center, y_center)."""
    return np.hypot(x - x_center, y - y_center)


def calc_center(data, x_center, y_center, search_radius=10):
    """
    Calculate the simple peak (just for consistency check) and barycentric peak.

    Parameters:
    data (2D array): The input data array.
    x_center (int): The x-coordinate of the initial guess for the peak.
    y_center (int): The y-coordinate of the initial guess for the peak.
    search_radius (int): The radius around the initial guess to search for the peak. Default is 10.

    Returns:
    tuple: The coordinates of the barycentric peak (barycenter_x, barycenter_y) and the total counts (total_counts).
    """
    transposed_data = data.T
    x_indices, y_indices = np.meshgrid(np.arange(-search_radius, search_radius),
                                       np.arange(-search_radius, search_radius), indexing='ij')
    x_indices += x_center
    y_indices += y_center

    values = transposed_data[x_indices, y_indices]
    total_counts = np.sum(values)

    if total_counts > 0:
        barycenter_x = np.sum(values * x_indices) / total_counts
        barycenter_y = np.sum(values * y_indices) / total_counts
    else:
        print("[ERROR] in calc_center: Total counts are non-positive. Something went wrong.")
        sys.exit(1)

    peak_idx = np.unravel_index(np.argmax(values), values.shape)
    peak_x, peak_y = x_indices[peak_idx], y_indices[peak_idx]
    peak_value = values[peak_idx]

    # Print information about the simple peak
    color_print(f"..... in calc_center: [simple peak] x_max, y_max, z_max = {peak_x}, {peak_y}, {peak_value}",
                ConsoleColors.OKCYAN)

    # Print information about the barycenter
    color_print(
        f"..... in calc_center: [barycenter] barycenter_x = {barycenter_x:.2f}, barycenter_y = {barycenter_y:.2f}",
        ConsoleColors.OKCYAN)

    # Print the total counts in the search radius
    color_print(f"..... in calc_center: [total counts in search radius] total_counts = {total_counts}",
                ConsoleColors.OKCYAN)

    if peak_value < 0:
        print("[ERROR] in calc_center: Peak value not found. Something went wrong.")
        sys.exit(1)

    return barycenter_x, barycenter_y, total_counts


def main():
    """Start main loop"""
    curdir = os.getcwd()

    """Setting for options"""
    usage = '%(prog)s fits.img'
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument('-filename', help='Input FITS file, should be a 0.3-10keV image',
                        default='xa300044010xtd_p030000010_cl.img')


    parser.add_argument('-o', '--auto_updated_center', type=bool,
                        help='Flag to automatically calculate center', default=False)
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to show detailed information', default=False)
    parser.add_argument('-m', '--manual', action='store_true', help='Flag to use vmax, vmin', default=False)
    parser.add_argument('-x', '--x_center', type=int, help='x coordinate of center', default=1255)
    parser.add_argument('-y', '--y_center', type=int, help='y coordinate of center', default=1156)
    parser.add_argument('-c', '--cellsize', type=int, help='cell size for a single event',
                        default=1 + 4)  # this value need to be verified
    parser.add_argument('-r', '--rout', type=int, help='outer redius of the region', default=60)
    parser.add_argument('-a', '--vmax', type=float, help='VMAX', default=3e3)
    parser.add_argument('-i', '--vmin', type=float, help='VMIN', default=1)
    parser.add_argument('-s', '--search_radius', type=int, help='Search radius for data extraction', default=128)
    parser.add_argument('-n', '--numberOfDivision', type=int, help='Number of divisions for annulus', default=32)
    parser.add_argument("--output-dir", default="./regions", help="Directory to save region files.")
    parser.add_argument('--zoom_pix_size',help='size of the zoomed figures',default=45)

    args = parser.parse_args()

    print(f"---------------------------------------------\n>>>>> START  :  {__file__}")
    print(f"..... auto_updated_center {args.auto_updated_center}")
    print(f"..... x_center {args.x_center} y_center {args.y_center}")
    print(
        f"..... search_radius {args.search_radius} (bin of image) ndiv {args.numberOfDivision} cellsize {args.cellsize}")
    print(f"..... debug {args.debug} manual {args.manual}")
    print(f"..... vmax {args.vmax} vmin {args.vmin}")

    filename = args.filename

    print(f"\n>>>>> Read each file and do process \n")
    print(f"START : {filename}")
    eve = Fits(filename, args.debug)

    if args.auto_updated_center:
        color_print("[[update calc_center]]", ConsoleColors.OKBLUE)
        x_center, y_center, _ = calc_center(eve.data, args.x_center, args.y_center, search_radius=args.search_radius)
        x_center, y_center = int(x_center), int(y_center)
        color_print(
            f"(x_center, y_center) is updated from ({args.x_center}, {args.y_center}) to ({x_center}, {y_center}).",
            ConsoleColors.UNDERLINE)
    else:
        x_center, y_center = args.x_center, args.y_center

    eve.plotwcs(x_center, y_center, vmax=args.vmax, vmin=args.vmin, manual=args.manual)
    eve.plotwcs_ps(x_center, y_center, search_radius=args.search_radius, vmax=args.vmax, vmin=args.vmin,
                   manual=args.manual)
    pileup_result = eve.mk_radialprofile(x_center, y_center, search_radius=args.search_radius, \
                                         ndiv=args.numberOfDivision, vmax=args.vmax, vmin=args.vmin, \
                                         manual=args.manual, cellsize=args.cellsize,zoom_pix_size=args.zoom_pix_size)

    # Create the output directory
    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)

    # Generate files for each rin, rout pair
    for _target, _r in zip(target_values, pileup_result):
        if _r > 0:
            _r_radec = eve.p2arcsec * _r
            if _r_radec < args.rout:
                color_print(
                    f">>>>> create a region file at pileup fraction = {_target * 100:.2f} % at {_r:.2f} pixel (= {_r_radec:.2f} arcsec)",
                    ConsoleColors.WARNING)

                if args.auto_updated_center:
                    generate_region_file(eve.c_ra, eve.c_dec, _r_radec, args.rout, output_dir=output_directory)
                else:
                    generate_region_file(x_center, y_center, _r_radec, args.rout, output_dir=output_directory,
                                         unit='pixel')

            else:
                color_print(f">>>>> did not create a region file, due to rin > rout : {_r_radec:.2f} > {args.rout:.2f}",
                            ConsoleColors.FAIL)

    print(">>>>> finish \n")


if __name__ == '__main__':
    main()
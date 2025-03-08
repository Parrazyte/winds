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

from general_tools import file_edit, ravel_ragged,MinorSymLogLocator,interval_extract,str_orbit

'''~~~~~~~~~~ ARGUMENTS ~~~~~~~~~~'''

ap = argparse.ArgumentParser(description='Script to reduce XRISM observation directories.\n)')

ap.add_argument('-heasoft_init_alias', help="name of the heasoft initialisation script alias", default="heainit",
                type=str)
ap.add_argument('-caldbinit_init_alias', help="name of the caldbinit initialisation script alias", default="caldbinit",
                type=str)

args = ap.parse_args()

heasoft_init_alias=args.heasoft_init_alias
caldb_init_alias=args.caldb_init_alias


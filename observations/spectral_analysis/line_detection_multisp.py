#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:17:29 2021

Compute spectral analysis and attempts to detect absorption lines in the iron band
for all spectra in the current merge directory

When finished or if asked to, computes a global PDF merging the result of the line detection process for each exposure
with previous data reduction results if any using the HID of visual_line

If using multi_obj, it is assumed the lineplots directory is outdirf


Changelog (very lazily updated, better check github):

v1.4 in dev as of (12/23):
    -some changes to pdf summaries
    -added NuSTAR
    -added multi instrument automatic analysis

v 1.3 (11/23):
many changes unlogged since may:
    -Suzaku 'megumi files' mode implemented
    -daily epoch matching for NICER
    -NICER gti implementation
    -spread computations with many options
    -autofit reloading
    -upper limit only mode
    -new naming conventions for recap files and the summaries
    -HID graph now in function and similar to what is used in visual_line
    -individual data groups now can have individual energy ignore bands. Useful for Suzaku and to prepare for future
     multi-instrument computations
    -lot of smaller changes and fixes

V 1.2 (06/23):
    -changed PDF summary to use visual_line tools

V 1.1 (04/23):
    -added multi models to use NICER scorpeon background.

V 1.0(10:01:23):
    -strong line modeling implemented with laor
    -added options to manually select the ftest threshold and a ftest leeway (for initial comp additions) in the autofit
    -added the option to split (or not) the components when fitting (turning it off when using laor helps tremendously)
    -fixed issue when using an empty string as line_ig
    -fixed various issues

V X(26:12:22):
    -restricting cont_powerlaw gamma to [1,3] instead of [0,4]
    -restricting diskbb kt to at least somewhat constrained values
    -switching to Kaatra binning for all XMM spectra
    -switching to C-stat for analysis

V X(25:12:22):
    -force narrow lines for XMM since none of them are resolved to help ftest and speed up computations
    -testing raising low-E to 2keV due to huge residuals

V X(24:12:22):
    -added proper saving for distinct blueshift flag

V X (23:12:22):
    -fixed issue in change in includedlist in mixlines
    -changed order of component deletion to reverse significance to avoid issues with lines deleting in an order we don't want
    -changed parameter unfreezing before second autofit to also thaw line parameters pegged during the first autofit to allow
    -to refit (and repeg but only if needed) them
    -taken off line linking and insted added flag for lines with bshift distinct at 3 sigma in the same complex

V X (22:12:22):
    -new emission framework working
    -fixed EW computation issue for NiKa27
    -fixed general EW computation issue with pegged parameter
    -> fixed missing implicit peg in model display of 1 datagroup models
    -added freeze computation before the EW upper limit computation since the EW has the same issue than the chain
    - changed log file of fitlines to the correct one after switching xspec log to fakes
    -edited txt table display
    -some changes in mass values in visual_line_tools to be more precise
    -solved issues in the distance dichotomy in visual_line_tools

"""

#general imports
import os,sys
import glob
import argparse
import re as re
import time

import numpy as np

#matplotlib imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from matplotlib.gridspec import GridSpec

#other stuff
from ast import literal_eval
from copy import deepcopy

#powerful saves
import dill

#progress bar
from tqdm import tqdm

#pdf conversion with HTML parsin
#install with fpdf2 NOT FPDF otherwise HTML won't work
from fpdf import FPDF, HTMLMixin

class PDF(FPDF, HTMLMixin):
    pass

#pdf merging
from PyPDF2 import PdfMerger

'''Astro'''
#general astro importss
from astropy.io import fits
from astropy.time import Time,TimeDelta

#custom script with a few shorter xspec commands

#custom script with a some lines and fit utilities and variables
from fitting_tools import c_light,lines_std_names,lines_e_dict,n_absline,range_absline,model_list

from linedet_loop import linedet_loop

from general_tools import ravel_ragged,shorten_epoch,expand_epoch,get_overlap


'''peak detection'''


ap = argparse.ArgumentParser(description='Script to perform line detection in X-ray Spectra.\n)')

'''PARALLELIZATION OPTIONS'''

#1 for no parallelization and using the current python interpreter
ap.add_argument('-parallel',help='number of processors for parallel directories',
                default=1,type=int)

'''THE FOLLOWING ARGUMENTS ARE ONLY USED IF PARALLEL>1'''
#singularity or python (which will the run a specific python environment).
ap.add_argument('-container_mode',help='type of container to run pyxspec in',default='singularity',
                type=str)

#if python, should be the path of the python environment with the right libraries
#if set to default, choose sys.executable for python, and the HEASOFT_SINGULARITY environment variable for singularity
ap.add_argument('-container',help='path of the container to use',default='default',type=str)

#useful for debugging
ap.add_argument('-force_instance',help='force instantiation even in parallel is set to 1',default=True,type=bool)


#parfile mode (empty string means not using this mode)
ap.add_argument('-parfile',nargs=1,help="parfile to use instead of standard sets of arguments",
                default='parfile_outdir_lineplots_opt_test_satellite_multi_cont_model_nthcont_NICER.par',
                type=str)

'''GENERAL OPTIONS'''


ap.add_argument('-satellite',nargs=1,help='telescope to fetch spectra from',default='multi',type=str)

#used for NICER, NuSTAR and multi for now
#can also be set for "day" to finish the current day
ap.add_argument('-group_max_timedelta',nargs=1,
                help='maximum time delta for epoch/gti grouping in dd_hh_mm_ss_ms',default='day',type=str)

#00_00_00_10 for NICER TR
#00_00_15_00 for NuSTAR individual orbits
#01_00_00_00 for dailies
#00_08_00_00 for NuSTAR multi

ap.add_argument("-cameras",nargs=1,help='Cameras to use for spectral analysis',default='all',type=str)
ap.add_argument("-expmodes",nargs=1,help='restrict the analysis to a single type of exposure',default='all',type=str)
ap.add_argument("-grouping",nargs=1,help='specfile grouping for spectra in Kaastra type or cts/bin',default='opt',
                type=str)

ap.add_argument('-fitstat',nargs=1,help='fit statistic to be used for the spectral analysis',default='cstat',type=str)

ap.add_argument('-xspec_query',nargs=1,help='fit query command for xspec',default='no',type=str)
ap.add_argument("-prefix",nargs=1,help='restrict analysis to a specific prefix',default='auto',type=str)

####output directory
ap.add_argument("-outdir",nargs=1,help="name of output directory for line plots",
                default="lineplots_opt_parallel",type=str)

#overwrite
#global overwrite based on recap PDF
ap.add_argument('-overwrite',nargs=1,
            help='rerun individual computations even if individual recap PDF files already exists',
            default=True,type=bool)

#note : will skip exposures for which the exposure didn't compute or with logged errors
ap.add_argument('-skip_started',nargs=1,help='skip all exposures listed in the local summary_line_det file',
                default=False,type=bool)

ap.add_argument('-skip_complete',nargs=1,help='skip completed exposures listed in the local summary_line_det file',
                default=False,type=bool)

ap.add_argument('-save_epoch_list',nargs=1,help='Save epoch list in a file before starting the line detections',
                default=True,type=bool)

ap.add_argument('-see_search',nargs=1,help='plot every single iteration of the line search',default=False,type=bool)

ap.add_argument('-log_console',nargs=1,help='log console output instead of displaying it on the screen',default=False,
                type=bool)

ap.add_argument('-catch_errors',nargs=1,
                help='catch errors when the line detection process crashes and continue for other exposures',
                default=False,type=bool)

ap.add_argument('-launch_cpd',nargs=1,
                help='launch cpd /xs window to be able to plot elements through xspec native commands',default=False,
                 type=bool)

ap.add_argument('-xspec_window',nargs=1,help='xspec window id (auto tries to pick it automatically)',
                default='auto',type=str)
#note: bash command to see window ids: wmctrl -l


'''MODELS'''
#### Models and abslines lock
ap.add_argument('-cont_model',nargs=1,help='model list to use for the autofit computation',
                default='thcont_NICER',type=str)

#useful to gain time when the abs components can be constrained for sure
ap.add_argument('-mandatory_abs',nargs=1,help='Consider absorption component as mandatory',
                default=True,type=bool)

ap.add_argument('-autofit_model',nargs=1,help='model list to use for the autofit computation',
                default='lines_narrow',type=str)
#narrow or resolved mainly

ap.add_argument('-no_abslines',nargs=1,
                help='turn off absorption lines addition in the fit (still allows for UL computations)',
                default=False,type=str)

'''DIRECTORY SPECIFICS'''

ap.add_argument("-local",nargs=1,help='launch analysis in the current directory instead',default=True,type=bool)

ap.add_argument("-h_update",nargs=1,help='update the bg, rmf and arf file names in the grouped spectra headers',
                default=True,type=bool)

'''ANALYSIS RESTRICTION'''

#this will prevent new analysis and can help for merged folders
ap.add_argument('-rewind_epoch_list',nargs=1,
                help='only uses the epoch already existing in the summary file instead of scanning',
                default=False,type=bool)

ap.add_argument('-spread_comput',nargs=1,
                help='spread sources in N subsamples to poorly parallelize on different consoles',
                default=1,type=bool)

ap.add_argument('-reverse_spread',nargs=1,help='run the spread computation lists in reverse',default=False,type=bool)

#note: having both reverse spread and reverse epoch with spread_comput>1 will give you back the normal order
#note that reverse_epoch is done after the spread comput
ap.add_argument('-reverse_epoch',nargs=1,help='reverse epoch list order',default=False,type=bool)

#better when spread computations are not running
ap.add_argument('-skip_started_spread',nargs=1,
                help='skip already finished computations when splitting the exposures',
                default=True,type=bool)

#better when some spread computations are still running
ap.add_argument('-spread_overwrite',nargs=1,
                help='consider already finished computations when creating the spreads',
                default=False,type=bool)


#in this mode, the line detection function isn't wrapped in a try, and the summary isn't updasted
ap.add_argument('-restrict',nargs=1,
                help='restrict the computation to a number of predefined exposures',
                default=False,type=bool)

epoch_restrict=['1130010141-001-002-003-004-005-006-007']

'''
NICER no abslines: 4130010128-001_4130010129-001

'''
ap.add_argument('-force_epochs',nargs=1,help='force epochs to given set of spectra instead of auto matching',
                default=False,type=bool)

force_epochs_str=\
'''
['5665010401-002M003_sp_grp_opt.pha'];
'''
force_epochs_str_list=[literal_eval(elem.replace('\n','')) for elem in force_epochs_str.split(';')[:-1]]

ap.add_argument('-force_epochs_list',nargs=1,help='force epochs list',default=force_epochs_str_list)


ap.add_argument('-SNR_min',nargs=1,help='minimum source Signal to Noise Ratio',default=30,type=float)
#shouldn't be needed now that we have a counts min limit + sometimes false especially in timing when the bg is the source

ap.add_argument('-counts_min',nargs=1,
                help='minimum source counts in the source region in the line continuum range',default=5000,type=float)

ap.add_argument('-fit_lowSNR',nargs=1,
                help='fit the continuum of low quality data to get the HID values',default=False,type=str)

ap.add_argument('-counts_min_HID',nargs=1,
                help='minimum counts for HID fitting in broad band',default=200,type=float)

#deprecated
ap.add_argument('-skip_nongrating',nargs=1,
                help='skip non grating Chandra obs (used to reprocess with changes in the restrictions)',
                default=False,type=bool)

ap.add_argument('-skip_flares',nargs=1,help='skip flare GTIs',default=True,type=bool)

ap.add_argument('-write_pdf',nargs=1,help='overwrite finished pdf at the end of the line detection',
                default=True,type=bool)

#can be set to false to gain time when testing or if the aborted pdf were already done
ap.add_argument('-write_aborted_pdf',nargs=1,help='create aborted pdfs at the end of the computation',
                default=True,type=bool)

'''MODES'''

ap.add_argument('-load_epochs',nargs=1,help='prepare epochs then exit',default=False)

#options: "opt" (tests the significance of each components and add them accordingly)
# and "force_all" to force all components
ap.add_argument('-cont_fit_method',nargs=1,help='fit logic for the broadband fits',
                default='force_all')

ap.add_argument('-reload_autofit',nargs=1,
                help='Reload existing autofit save files to gain time if a computation has crashed',
                default=True,type=bool)

ap.add_argument('-reload_fakes',nargs=1,
                help='Reload fake delchi array file to skip the fake computation if possible',
                default=True,type=bool)

ap.add_argument('-pdf_only',nargs=1,
                help='Updates the pdf with already existing elements but skips the line detection entirely',
                default=False,type=bool)

#note: used mainly to recompute obs with bugged UL computations. Needs FINISHED computations firsthand, else
#use reload_autofit and reload_fakes
ap.add_argument('-line_ul_only',nargs=1,help='Reloads the autofit computations and re-computes the ULs',
                default=False,type=bool)

#requires reload_autofit to be set to true
ap.add_argument('-compute_highflux_only',help='Reloads the autofit computation and simply computes the high flux array',
                default=False,type=bool)

ap.add_argument('-hid_only',nargs=1,help='skip the line detection and directly plot the hid',
                default=False,type=bool)

#date or HR
ap.add_argument('-hid_sort_method',nargs=1,help='HID summary observation sorting',default='date',type=str)

ap.add_argument('-multi_obj',nargs=1,
                help='compute the hid for multiple obj directories inside the current directory',
                default=False)

ap.add_argument('-autofit',nargs=1,
                help='enable auto fit with lines if the peak search detected at least one absorption',
                default=True,type=bool)

#using the lines found during the procedure to re-estimate the fit parameters and HID
ap.add_argument('-refit_cont',nargs=1,
                help='After the autofit, refit the continuum without excluding the iron region',
                default=True)

ap.add_argument('-merge_cont',nargs=1,
                help='Reload the previous continuum before refitting the continuum aftrer autofit',
                default=True)
####split fit
ap.add_argument('-split_fit',nargs=1,
                help='Split fitting procedure between components instead of fitting the whole model directly',
                default=True)

#line significance assessment parameter
ap.add_argument('-assess_line',nargs=1,
                help='use fakeit simulations to estimate the significance of each absorption line',
                default=True,type=bool)

ap.add_argument('-assess_line_upper',nargs=1,help='compute upper limits of each absorption line',
                default=True,type=bool)


'''SPECTRUM PARAMETERS'''

#pile-up control
ap.add_argument("-plim","--pileup_lim",nargs=1,help='maximal pileup value',default=0.10,type=float)
ap.add_argument("-pmiss",nargs=1,help='include spectra with no pileup info',default=True,type=bool)

#note: these values are modified for higher energy instruments, like suzaku or NuSTAR
ap.add_argument("-hid_cont_range",nargs=1,
                help='min and max energies of the hid band fit',default='3 10',type=str)

#this skips the HID fitmod fit procedure entirely and replaces it by the broad band fit
#useful to get better broadband constrains for the HID
#also skips the computation of the chain for the broadband fit
ap.add_argument('-broad_HID_mode',nargs=1,
                help='reuses the broad band fit for the HID computations',default=True,type=str)

ap.add_argument("-line_cont_range",nargs=1,
                help='min and max energies of the line continuum broand band fit',default='4 10',type=str)

ap.add_argument('-force_ener_bounds',nargs=1,
                help='force the energy limits above instead of resetting to standard bounds for each epoch',
                default=False,type=bool)
#### line cont ig
ap.add_argument("-line_cont_ig_arg",nargs=1,
                help='min and max energies of the ignore zone in the line continuum broand band fit',
                default='iron',type=str)

#note that the recent change in this will make fake computations slower because they use the same grid
ap.add_argument("-line_search_e",nargs=1,
                help='min, max and step of the line energy search',default='4 10 0.02',type=str)

ap.add_argument("-line_search_norm",nargs=1,
                help='min, max and nsteps (for one sign)  of the line norm search (which operates in log scale)',
                default='0.01 10 500',type=str)

#skips fakes testing at high energy to gain time
ap.add_argument('-restrict_fakes',nargs=1,
                help='restrict range of fake computation to 8keV max',default=False,type=bool)

'''MULTI'''


ap.add_argument('-plot_epoch_overlap',nargs=1,help='plot overlap between different epochs',default=False)

#in this case other epochs from other instruments are matched against the obs of this one
#useful to center epoch matching on a specific instrument
#off value is False
ap.add_argument('-multi_focus',nargs=1,help='restricts epoch matching to having a specific telescope',
                default='NICER',type=str)

ap.add_argument('-add_mosaic_BAT',nargs=1,help='add mosaiced Swift-BAT survey spectra to the epoch creation',
                default=True,type=str)

ap.add_argument('-add_ungrouped_BAT',nargs=1,help='add ungrouped Swift-BAT spectra to the epoch creation',
                default=True,type=str)

ap.add_argument('-skip_single_instru',nargs=1,help='skip epochs with a single instrument',
                default=False,type=bool)

#for multi focus
ap.add_argument('-match_closest_NICER',nargs=1,help='only add the closest NICER obsid',default=False,type=bool)

#off value is False. ex: "NICER+NuSTAR"
ap.add_argument('-restrict_combination',nargs=1,help='restrict multi epochs to a specific satellite combination',
                default=False)

ap.add_argument('-single_obsid_NuSTAR',nargs=1,
                help='limit NuSTAR epoch grouping to single obsids',default=True,type=bool)

ap.add_argument('-diff_bands_NuSTAR_NICER',nargs=1,help='different energy bounds for multi NuSTAR/NICER combinations',
                default=True,type=bool)

ap.add_argument('-force_nosplit_fit_multi',nargs=1,help='force no split fit for multi satellites',default=False)

#note: the NuSTAR SNR filtering is included by default in multi

'''CHANDRA'''
#Chandra issues
ap.add_argument('-restrict_graded',nargs=1,
                help='restrict range of line analysis to 8keV max for old CC33_graded spectra',
                default=False,type=bool)

#### restrict order
ap.add_argument('-restrict_order',nargs=1,help='restrict HETG spectral analysis to the -1 order only',
                default=False,type=bool)

'''NICER'''
ap.add_argument('-NICER_bkg',nargs=1,help='NICER background type',default='scorpeon_mod',type=str)

ap.add_argument('-pre_reduced_NICER',nargs=1,
                help='change NICER data format to pre-reduced obsids',default=False,type=bool)

ap.add_argument('-NICER_lc_binning',nargs=1,help='NICER LC binning',default='1',type=str)

#in this case the continuum components require the NICER calibration components to get a decent fit
ap.add_argument('-low_E_NICER',nargs=1,help='NICER lower energy threshold for broadband fits',default=0.3,type=str)

# ap.add_argument('-extend_scorpeon_nxb_SAA',nargs=1,help='Extend scorpeon nxb_SAA range',default=True)

#ueseful when the SAA values/overshoots are high and thus the default parameter value is underestimated
#note that the nxb SAA norm is automatically extended when this is set to true because its default range used to
#be way too small. This has been fixed since
ap.add_argument('-fit_SAA_norm',nargs=1,help='unfreeze the nxb_saa_norm parameter to fit it',default=True)

'''NuSTAR'''

ap.add_argument('-freeze_nH',nargs=1,help='Freeze main absorption to a fiducial value',default=False,type=bool)
ap.add_argument('-freeze_nH_val',nargs=1,help='Frozen main absorption value (10^22 cm^-2)',default=14,type=bool)

#A value of 0 disables the SNR testing. Done for each 1keV band, independantly for each NuSTAR spectrum
ap.add_argument('-filter_NuSTAR_SNR',nargs=1,help='restrict the NuSTAR band to where the SNR is above a given value',
                default=3,type=float)

'''SUZAKU'''

ap.add_argument('-megumi_files',nargs=1,help='adapt suzaku file structure for megumi data reduction',
                default=True,type=bool)

ap.add_argument('-suzaku_xis_range',nargs=1,help='range of energies usable for suzaku xis',default='1.9 9.',type=str)
ap.add_argument('-suzaku_xis_ignore',nargs=1,help='range of energies to ignore for suzaku xis',default="['2.1-2.3','3.0-3.4']",type=str)

ap.add_argument('-suzaku_pin_range',nargs=1,help='range of energies usable for suzaku pin',default='12. 40.',type=str)

'''XMM'''

ap.add_argument("-skipbg_timing",nargs=1,help='do not use background for the -often contaminated- timing backgrounds',
                default=True,type=bool)
ap.add_argument('-max_bg_imaging',nargs=1,
                help='maximal imaging bg rate compared to standard bg values',default=100,type=float)

'''PEAK/MC DETECTION PARAMETERS'''

ap.add_argument('-peak_thresh',nargs=1,help='chi difference threshold for the peak detection',default=9.21,type=float)

ap.add_argument('-peak_clean',nargs=1,
                help='try to distinguish a width for every peak (experimental)',default=False,type=bool)

ap.add_argument('-nfakes',nargs=1,
                help='number of simulations used. Limits the maximal significance tested to >1-1/nfakes',
                default=100,type=int)

ap.add_argument('-sign_threshold',nargs=1,
                help='data significance used to start the upper limit procedure and estimate the detectability',
                default=0.997,type=float)

'''AUTOFIT PARAMETERS'''

ap.add_argument('-force_autofit',nargs=1,
                help='force autofit even when there are no abs peaks detected',default=True,type=bool)

ap.add_argument('-trig_interval',nargs=1,
                help='interval restriction for the absorption peak to trigger the autofit process',default='6.5 9.1',
                type=str)

ap.add_argument('-overlap_flag',nargs=1,
                help='overlap value to trigger the overlap flag, in absorption line area fraction',
                default=0.5,type=float)

#note: currently not used

'''VISUALISATION'''

ap.add_argument('-plot_mode',nargs=1,
                help='system used for the visualisation',default='matplotlib',type=str)

ap.add_argument('-paper_look',nargs=1,
                help='changes some visual elements for a more official look',default=True,type=bool)

'''GLOBAL PDF SUMMARY'''

ap.add_argument('-line_infos_pdf',nargs=1,help='write line infos in the global object pdf',default=True,type=bool)

'''Chatter'''
ap.add_argument('-xchatter',nargs=1,help='xspec main chatter value for the computations',default=1,type=int)

args=ap.parse_args()

'''
Notes:
-Only works for the auto observations (due to prefix naming) for now

-For the HID computation we fix the masses of all the objets at 8M_sol unless a good dynamical measurement exists

-Due to the way the number of steps is computed, we explore one less value for the positive side of the normalisation

-The norm_stepval argument is for a fixed flux band, and the value is scaled in the computation depending on the line energy step
'''

#attributing variables dynamically to avoid explicitely creating each variable

parallel=args.parallel
container_mode=args.container_mode
container=args.container
force_instance=args.force_instance
parfile=args.parfile

sat_glob=args.satellite
cameras=args.cameras
expmodes=args.expmodes
grouping=args.grouping
prefix=args.prefix
local=args.local
h_update=args.h_update
line_cont_range=np.array(args.line_cont_range.split(' ')).astype(float)
line_cont_ig_arg=args.line_cont_ig_arg
line_search_e=np.array(args.line_search_e.split(' ')).astype(float)
line_search_norm=np.array(args.line_search_norm.split(' ')).astype(float)
assess_line=args.assess_line
assess_line_upper=args.assess_line_upper

nfakes=args.nfakes
autofit=args.autofit
force_autofit=args.force_autofit
autofit_model=args.autofit_model
write_pdf=args.write_pdf
trig_interval=np.array(args.trig_interval.split(' ')).astype(float)
hid_cont_range=np.array(args.hid_cont_range.split(' ')).astype(float)
catch_errors=args.catch_errors
sign_threshold=args.sign_threshold
overlap_flag=args.overlap_flag
xspec_window=args.xspec_window
restrict_graded=args.restrict_graded
skip_nongrating=args.skip_nongrating
restrict_fakes=args.restrict_fakes
pre_reduced_NICER=args.pre_reduced_NICER
fit_lowSNR=args.fit_lowSNR
counts_min_HID=args.counts_min_HID

refit_cont=args.refit_cont
launch_cpd=args.launch_cpd
pdf_only=args.pdf_only
glob_summary_save_line_infos=args.line_infos_pdf
restrict_order=args.restrict_order
no_abslines=args.no_abslines
NICER_bkg=args.NICER_bkg
line_ul_only=args.line_ul_only
NICER_lc_binning=args.NICER_lc_binning
# extend_scorpeon_nxb_SAA=args.extend_scorpeon_nxb_SAA
fit_SAA_norm=args.fit_SAA_norm

reload_autofit=args.reload_autofit
reverse_spread=args.reverse_spread
spread_overwrite=args.spread_overwrite
force_ener_bounds=args.force_ener_bounds
write_aborted_pdf=args.write_aborted_pdf
hid_sort_method=args.hid_sort_method
reverse_epoch=args.reverse_epoch
reload_fakes=args.reload_fakes
broad_HID_mode=args.broad_HID_mode
freeze_nH=args.freeze_nH
freeze_nH_val=args.freeze_nH_val

mandatory_abs=args.mandatory_abs

add_ungrouped_BAT=args.add_ungrouped_BAT
add_mosaic_BAT=args.add_mosaic_BAT

cont_fit_method=args.cont_fit_method
xchatter=args.xchatter

merge_cont=args.merge_cont

rewind_epoch_list=args.rewind_epoch_list
force_epochs=args.force_epochs
force_epochs_list=args.force_epochs_list

megumi_files=args.megumi_files

suzaku_xis_range=np.array(args.suzaku_xis_range.split(' ')).astype(float)
suzaku_xis_ignore=literal_eval(args.suzaku_xis_ignore)

suzaku_pin_range=np.array(args.suzaku_pin_range.split(' ')).astype(float)

compute_highflux_only=args.compute_highflux_only

load_epochs=args.load_epochs

if compute_highflux_only:
    assert reload_autofit,'Reload autofit required for this mode'

save_epoch_list=args.save_epoch_list

low_E_NICER=args.low_E_NICER

skip_flares=args.skip_flares
spread_comput=args.spread_comput
skip_started_spread=args.skip_started_spread

filter_NuSTAR_SNR=args.filter_NuSTAR_SNR

diff_bands_NuSTAR_NICER=args.diff_bands_NuSTAR_NICER
multi_focus=args.multi_focus
group_max_timedelta=args.group_max_timedelta
single_obsid_NuSTAR=args.single_obsid_NuSTAR
restrict_combination=args.restrict_combination
match_closest_NICER=args.match_closest_NICER
plot_epoch_overlap=args.plot_epoch_overlap
skip_single_instru=args.skip_single_instru

xspec_query=args.xspec_query

outdir=args.outdir
pileup_lim=args.pileup_lim
pileup_missing=args.pmiss
skipbg_timing=args.skipbg_timing
peak_thresh=args.peak_thresh
see_search=args.see_search
peak_clean=args.peak_clean
overwrite=args.overwrite
hid_only=args.hid_only
restrict=args.restrict
multi_obj=args.multi_obj
skip_started=args.skip_started
skip_complete=args.skip_complete
SNR_min=args.SNR_min
counts_min=args.counts_min
max_bg_imaging=args.max_bg_imaging
paper_look=args.paper_look
plot_mode=args.plot_mode
log_console=args.log_console
fitstat=args.fitstat
cont_model=args.cont_model

force_nosplit_fit_multi=args.force_nosplit_fit_multi
split_fit=args.split_fit and not (sat_glob=='multi' and force_nosplit_fit_multi)

#replacing some arguments with those of the parameter file if it exists
if parfile!='':
    # loading the file as an array
    if type(parfile)==str:
        param_arr = np.loadtxt(parfile, dtype=str)
    else:
        param_arr = np.loadtxt(parfile[0], dtype=str)

    parallel=int(param_arr[0][1])
    outdir=param_arr[1][1]
    cont_model=param_arr[2][1]
    autofit_model=param_arr[3][1]
    container=param_arr[4][1]
    satellite=param_arr[5][1]
    groum_max_timedelta=param_arr[6][1]
    skip_started=bool(param_arr[7][1])
    catch_errors=bool(param_arr[8][1])
    multi_focus=param_arr[9][1]
    nfakes=int(param_arr[10][1])

'''utility functions'''

#switching off matplotlib plot displays unless with plt.show()
plt.ioff()
#fetching the line detection in specific directories
def folder_state(folderpath='./'):
    #fetching the previously computed directories from the summary folder file
    try:
        with open(os.path.join(folderpath,outdir,'summary_line_det.log')) as summary_expos:
            launched_expos=summary_expos.readlines()

            #creating variable for completed analysis only
            completed_expos=['_'.join(elem.split('\t')[:-1]) for elem in launched_expos if 'Line detection complete.' in elem]
            launched_expos=['_'.join(elem.split('\t')[:-1]) for elem in launched_expos]
            launched_expos=[elem for elem in launched_expos if not elem.startswith('Obsid')]
    except:
        launched_expos=[]
        completed_expos=[]

    return launched_expos,completed_expos


#for the current directory:
started_expos,done_expos=folder_state()

if sat_glob=='multi':
    started_expos = [[elem] if not elem.startswith('[') else literal_eval(elem.split(']')[0]+']') for elem in
                     started_expos]
    done_expos = [[elem] if not elem.startswith('[') else literal_eval(elem.split(']')[0]+']') for elem in
                     done_expos]
elif sat_glob in ['NICER','NuSTAR']:
    started_expos=[[elem.split('_')[0]] if not elem.startswith('[') else literal_eval(elem.split('_')[0]) for elem in started_expos]
    done_expos=[[elem.split('_')[0]] if not elem.startswith('[') else literal_eval(elem.split('_')[0]) for elem in done_expos]

#bad spectrum manually taken off
bad_flags=[]

'''initialisation'''

#readjusting the variables in lists
if sat_glob=='XMM':
    if cameras=='all':
        cameras=['pn','mos1','mos2']
    else:
        cameras=[cameras]
        if 'pn' in cameras[0]:
            cameras=cameras+['pn']
        if 'mos1' in cameras[0]:
            cameras=cameras+['mos1']
        if 'mos2' in cameras[0]:
            cameras=cameras+['mos2']
        cameras=cameras[1:]

elif sat_glob=='Chandra':
    cameras=['hetg']
elif sat_glob=='NICER':
    cameras=['xti']
elif sat_glob=='Suzaku':
    cameras=['XIS','PIN']
elif sat_glob=='Swift':
    cameras=['xrt']
elif sat_glob=='NuSTAR':
    cameras=['FMPA','FPMB']

if expmodes=='all':
    expmodes=['Imaging','Timing']
else:
    expmodes=[expmodes]
    if 'timing' in expmodes[0] or 'Timing' in expmodes[0]:
        expmodes=expmodes+['Timing']
    if 'imaging' in expmodes[0] or 'Imaging' in expmodes[0]:
        expmodes=expmodes+['Imaging']
    expmodes=expmodes[1:]

if not local:
    os.chdir('bigbatch')

# if launch_cpd:
#     #the weird constant error is just to avoid having an error detection in Spyder due to xspec_id not being created at this point
#     if 1==0:
#         xspec_id=1
#
#     try:
#         Pnull(xspec_id)
#     except:
#         Pnull()

spfile_list=[]

#listing the spectral files for the single object mode

#### File fetching

glob_summary_reg=[]
glob_summary_sp=[]

if multi_obj==False:

    #assuming the last top directory is the object name
    obj_name=os.getcwd().split('/')[-2]

    #path to the line results file
    line_store_path=os.path.join(os.getcwd(),outdir,'line_values_'+args.line_search_e.replace(' ','_')+'_'+
                                 args.line_search_norm.replace(' ','_')+'.txt')

    #path to the autofit file
    autofit_store_path=os.path.join(os.getcwd(),outdir,'autofit_values_'+args.line_search_e.replace(' ','_')+'_'+
                                 args.line_search_norm.replace(' ','_')+'.txt')

    if sat_glob=='XMM':
        for elem_cam in cameras:
            for elem_exp in expmodes:
                spfile_list=spfile_list+glob.glob('*'+elem_cam+'*'+elem_exp+'_'+prefix+'_sp_src_grp_'+grouping+'.*')
                #taking of modified spectra with background checked
                spfile_list=[elem for elem in spfile_list if 'bgtested' not in elem]
    elif sat_glob in ['Chandra','NICER','Suzaku','Swift','NuSTAR']:
        # if pre_reduced_NICER and sat_glob=='NICER':
        #     spfile_list=glob.glob('*.grp')
        # else:
        spfile_list=glob.glob('*_grp_'+grouping+('.pi' if sat_glob=='Swift' else '.pha') )
    elif sat_glob=='multi':
        #just combining both so we can get both XMM spectra and non-xmm spectra
        spfile_list = glob.glob('*_grp_' + grouping + '.pi')\
                     +glob.glob('*_grp_' + grouping + '.pha')\
                    +[elem for elem in \
                    glob.glob('*_'+prefix+'_sp_src_grp_'+grouping+'.*') if 'bgtested' not in elem]

        if add_mosaic_BAT:
            spfile_list += [elem for elem in glob.glob('BAT_*_mosaic.pha')]
        if add_ungrouped_BAT:
            spfile_list+=[elem for elem in  glob.glob('*_survey_point_*.pha')]

    #creating the output directory
    os.system('mkdir -p '+outdir)

    #listing the exposure ids in the bigbatch directory
    bigbatch_files=glob.glob('**')

    if sat_glob=='XMM':
        #tacking of 'spectra' allows to disregard the failed combined lightcurve computations of some obsids
        #as unique exposures compared to their spectra
        exposid_list=np.unique(['_'.join(elem.split('_')[:4]).replace('rate','').replace('.ds','')+'_auto' for elem in bigbatch_files\
                      if '/' not in elem and 'spectrum' not in elem and elem[:10].isdigit() and True in ['_'+elemcam+'_' in elem for elemcam in cameras]])
        #fetching the summary files for the data reduction steps
        with open('glob_summary_extract_reg.log') as sumfile:
            glob_summary_reg=sumfile.readlines()[1:]
        with open('glob_summary_extract_sp.log') as sumfile:
            glob_summary_sp=sumfile.readlines()[1:]

else:
    #switching off the spectral analysis
    hid_only=True

spfile_list=np.array(spfile_list)
spfile_list.sort()

#we create these variables in any case because the multi_obj plots require them
#note: we add half a step to get rid of rounding problems and have the correct steps
line_search_e_space=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2]/2,line_search_e[2])
#this one is here to avoid adding one point if incorrect roundings create problem
line_search_e_space=line_search_e_space[line_search_e_space<=line_search_e[1]]

norm_par_space=np.concatenate((-np.logspace(np.log10(line_search_norm[1]),np.log10(line_search_norm[0]),int(line_search_norm[2]/2)),np.array([0]),
                                np.logspace(np.log10(line_search_norm[0]),np.log10(line_search_norm[1]),int(line_search_norm[2]/2))))
norm_nsteps=len(norm_par_space)

'''''''''''''''''''''''''''''''''''''''
''''''''''''''Fitting Loop'''''''''''''
'''''''''''''''''''''''''''''''''''''''

#summary file header
summary_header='Obsid\tFile identifier\tSpectrum extraction result\n'

def file_to_obs(file,sat):
    if sat=='Suzaku':
        if megumi_files:
            return file.split('_src')[0].split('_gti')[0]
    elif sat in ['XMM','NuSTAR']:
        return file.split('_sp')[0]
    elif sat in ['Chandra','Swift','SWIFT']:
        return file.split('_grp_opt')[0]
    elif sat in ['NICER']:
        return file.split('_sp_grp_opt')[0]

'''
#### Epoch matching
'''

# if sat_glob=='XMM':
#     spfile_dates=np.array([[None,None]]*len(spfile_list))
#
#     #storing the dates of all the exposures
#     for file_index,elem_sp in enumerate(spfile_list):
#         with fits.open(elem_sp) as fits_spec:
#             spfile_dates[file_index][0]=Time(fits_spec[0].header['DATE-OBS'])
#             spfile_dates[file_index][1]=Time(fits_spec[0].header['DATE-END'])
#
#     def overlap_fraction(dates_1,dates_2):
#         duration_1=dates_1[1]-dates_1[0]
#         duration_2=dates_2[1]-dates_2[0]
#
#         max_overlap=max(0,min(dates_1[1],dates_2[1])-max(dates_1[0],dates_2[0]))
#
#         return max(max_overlap/duration_1,max_overlap/duration_2)
#
#     epoch_list=[]
#     #and matching them
#     while len(ravel_ragged(epoch_list))!=len(spfile_list):
#
#         elem_epoch=[]
#
#         #taking a new spectrum
#         curr_sp_id,curr_sp=[[i,spfile_list[i]] for i in range(len(spfile_list)) if spfile_list[i] not in ravel_ragged(epoch_list)][0]
#
#         #adding it to a new epoch
#         elem_epoch+=[curr_sp]
#
#         #testing all remaining spectrum for overlap
#         #we do this incrementally to test overlap between with all the spectra in the epoch
#         id_ep=0
#         while id_ep<len(elem_epoch):
#             curr_tested_epoch=elem_epoch[id_ep]
#             curr_tested_epoch_id=np.argwhere(spfile_list==curr_tested_epoch)[0][0]
#             for elem_sp_id,elem_sp in [[i,spfile_list[i]] for i in range(len(spfile_list)) if
#                                        (spfile_list[i] not in ravel_ragged(epoch_list) and spfile_list[i] not in elem_epoch)]:
#                 #fetching the index of each
#                 if overlap_fraction(spfile_dates[curr_tested_epoch_id],spfile_dates[elem_sp_id]).value>0.5:
#                     elem_epoch+=[elem_sp]
#             id_ep+=1
#
#         '''
#         ordering the epoch files with pn, mos1, mos2 (or any part of this)
#         '''
#         elem_epoch_sorted=[]
#         for cam in ['pn','mos1','mos2']:
#             for elem_sp in elem_epoch:
#                 if elem_sp.split('_')[1]==cam:
#                     elem_epoch_sorted+=[elem_sp]
#
#         epoch_list+=[elem_epoch_sorted]
#
# elif sat_glob=='Chandra':
#     epoch_list=[]
#     epoch_list_started=[]
#     obsid_list_chandra=np.unique([elem.split('_')[0] for elem in spfile_list])
#     for obsid in obsid_list_chandra:
#         epoch_list+=[[elem for elem in spfile_list if elem.startswith(obsid)]]
#
#     obsid_list_started_chandra=np.unique([elem.split('_')[0] for elem in started_expos[1:]])
#     for obsid in obsid_list_started_chandra.tolist():
#         epoch_list_started+=[[elem,elem.replace('-1','1')] for elem in started_expos if elem.startswith(obsid)]
#
# elif sat_glob=='NICER':
#     epoch_list=[]
#     tstart_list=[]
#     tstop_list=[]
#     for elem_file in spfile_list:
#
#         try:
#             with fits.open(elem_file) as hdul:
#
#                 start_obs_s = hdul[1].header['TSTART'] + hdul[1].header['TIMEZERO']
#                 stop_obs_s = hdul[1].header['TSTOP'] + hdul[1].header['TIMEZERO']
#
#                 # saving for titles later
#                 mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')
#
#                 obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')
#                 obs_stop = mjd_ref + TimeDelta(stop_obs_s, format='sec')
#
#
#         except:
#             print('Issue with fits opening for file:'+elem_file)
#             continue
#
#         #note: don't convert to jd, jd have 0 at 12:00 instead of 00:00
#         tstart_list+=[obs_start.mjd]
#         tstop_list+=[obs_stop.mjd]
#
#     epoch_id_list_ravel=[]
#     epoch_id_list=[]
#
#     with tqdm(total=len(tstart_list)) as pbar:
#         for id_elem,elem_tstart in enumerate(tstart_list):
#
#             #skipping computation for already grouped elements
#             if id_elem in epoch_id_list_ravel:
#                 continue
#
#             elem_delta=np.array([-get_overlap([elem_tstart,elem_tstop],[other_start,other_stop],distance=True) for other_start,other_stop in zip(tstart_list,tstop_list)])
#
#             #list of matchable epochs
#             # we automatically match epochs that have some time in common
#             #if they don't, the maximum gap is symmetrical for time values, and otherwise
#             #it is the distance to the beginning of the day where the obs started/the end of the day where the obs
#             # finishes
#             elem_epoch_id=np.array([id for id in range(len(tstart_list)) if\
#                                      id not in epoch_id_list_ravel and\
#                                     (elem_delta[id]<=0 or \
#                                      (elem_delta[id]<max_delta_bef and tstop_list[id]<elem_tstart) or \
#                                      (elem_delta[id]<max_delta_aft and elem_tstop<tstart_list[id])) ])
#
#             # max delta between gti starts in sec
#             if group_max_timedelta == 'day':
#                 max_delta_bef = TimeDelta(np.ceil(elem_tstop)-elem_tstop,format='jd')
#                 max_delta_aft = TimeDelta(elem_tstart-np.floor(elem_tstart),format='jd')
#             else:
#                 max_delta_bef = max_delta_aft = (TimeDelta(group_max_timedelta.split('_')[0], format='jd') + \
#                              TimeDelta(group_max_timedelta.split('_')[1], format='jd') / 24 + \
#                              TimeDelta(group_max_timedelta.split('_')[2], format='jd') / (24 * 60) + \
#                              TimeDelta(group_max_timedelta.split('_')[3], format='jd') / (24 * 3600) + \
#                              TimeDelta(group_max_timedelta.split('_')[4], format='jd') / (24 * 3600 * 1e3)).to_value(
#                     'jd')
#
#             #note that here,
#             elem_delta=np.array([-get_overlap([elem_tstart,elem_tstop],[other_start,other_stop],distance=True) for other_start,other_stop in zip(tstart_list,tstop_list)])
#
#             #list of matchable epochs
#             # we automatically match epochs that have some time in common
#             #if they don't, the maximum gap is symmetrical for time values, and otherwise
#             #it is the distance to the beginning of the day where the obs started/the end of the day where the obs
#             # finishes
#             elem_epoch_id=np.array([id for id in range(len(tstart_list)) if\
#                                      id not in epoch_id_list_ravel and\
#                                     (elem_delta[id]<=0 or \
#                                      (elem_delta[id]<max_delta_bef and tstop_list[id]<elem_tstart) or \
#                                      (elem_delta[id]<max_delta_aft and elem_tstop<tstart_list[id])) ])
#
#             epoch_id_list_ravel+=elem_epoch_id
#
#             if len(elem_epoch_id)>0:
#                 epoch_id_list+=[elem_epoch_id]
#
#             pbar.update(n=len(elem_epoch_id))
#
#     epoch_list=[spfile_list[elem] for elem in epoch_id_list]
#
#
#     #skipping flares if asked for
#     if skip_flares:
#         epoch_list=[[subelem for subelem in elem if "F_sp" not in subelem] for elem in epoch_list]
#         epoch_list=[elem for elem in epoch_list if len(elem)>0]
#
#     epoch_list=np.array(epoch_list,dtype=object)
#
#     #not needed atm
#     # def str_to_epoch(str_epoch):
#     #     str_epoch_list=[]
#     #     for elem_obsid_str in str_epoch:
#     #         if '-' not in elem_obsid_str:
#     #             str_epoch_list+=elem_obsid_str
#     #         else:
#     #             str_epoch_list+=[elem_obsid_str.split('-')[0]+elem_obsid_str.split('-')[i]\
#     #                              for i in range(1,len(elem_obsid_str.split('-')))]
#
#     epoch_list_started=started_expos
#     epoch_list_done=done_expos
#
# elif sat_glob in ['Suzaku','Swift']:
#     epoch_list=[]
#     epoch_list_started=[]
#     if sat_glob=='Swift':
#         obsid_list=np.unique([elem[:11] for elem in spfile_list])
#     else:
#         obsid_list=np.unique([elem.split('_')[0] for elem in spfile_list])
#
#     for obsid in obsid_list:
#
#         epoch_list+=[[elem for elem in spfile_list if elem.startswith(obsid+'_')]]
#
#     if sat_glob=='Swift':
#         obsid_list_started=np.unique([elem.split('_')[0][:11] for elem in started_expos[1:]])
#         for obsid in obsid_list_started.tolist():
#             epoch_list_started+=[[elem] for elem in started_expos if elem.startswith(obsid)]
#
#     if sat_glob=='Suzaku':
#         #reversing the order to have the FI xis first, then the BI xis, then pin instead of the opposite
#         epoch_list=[elem[::-1] for elem in epoch_list]
#
#         epoch_list_started=[literal_eval(elem.split(']')[0]+']') for elem in started_expos[1:]]
#
# elif sat_glob=='NuSTAR':
#
#     epoch_list = []
#
#     # skipping flares if asked for
#     if skip_flares:
#         spfile_list = np.array([elem for elem in spfile_list if "F_sp" not in elem])
#
#     tstart_list = np.array([None] * len(spfile_list))
#     det_list = np.array([None] * len(spfile_list))
#     tstop_list = np.array([None] * len(spfile_list))
#
#     for i_file, elem_file in enumerate(spfile_list):
#
#         # for Suzaku this won't work for meugmi's xis0_xis3 files bc their header has been replaced
#         # so we replace them by the xis1 to be able to load the exposure
#         elem_file_load = elem_file.replace('xis0_xis3', 'xis1')
#
#         with fits.open(elem_file_load) as hdul:
#             if 'TELESCOP' in hdul[1].header:
#                 det_list[i_file] = hdul[1].header['TELESCOP'].replace('SUZAKU', 'Suzaku')
#             else:
#                 # the only files without TELESCOP in the header should be the fused megumi_files suzaku sp
#                 assert megumi_files, 'Issue with detector handling'
#
#                 det_list[i_file] = 'Suzaku'
#
#             if 'TIMEZERO' in hdul[1].header:
#                 start_obs_s = hdul[1].header['TSTART'] + hdul[1].header['TIMEZERO']
#                 stop_obs_s = hdul[1].header['TSTOP'] + hdul[1].header['TIMEZERO']
#             else:
#                 start_obs_s = hdul[1].header['TSTART']
#                 stop_obs_s = hdul[1].header['TSTOP']
#
#             # saving for titles later
#             mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')
#
#             obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')
#             obs_stop = mjd_ref + TimeDelta(stop_obs_s, format='sec')
#
#         #note: don't convert to jd, jd have 0 at 12:00 instead of 00:00
#         tstart_list[i_file] = obs_start.mjd
#         tstop_list[i_file] = obs_stop.mjd
#
#     epoch_id_list_ravel = []
#     epoch_id_list = []
#
#     tstart_list_base = tstart_list
#     tstop_list_base = tstop_list
#     det_list_base = det_list
#     id_base = np.arange(len(spfile_list))
#
#     with tqdm(total=len(tstart_list)) as pbar:
#         for id_elem, (elem_tstart, elem_tstop, elem_det) in enumerate(
#                 zip(tstart_list_base, tstop_list_base, det_list_base)):
#
#             # skipping computation for already grouped elements
#             if id_base[id_elem] in epoch_id_list_ravel:
#                 continue
#
#             # max delta between gti starts in sec
#             if group_max_timedelta == 'day':
#                 TimeDelta(np.ceil(elem_tstart) - elem_tstart, format='jd')
#             else:
#                 max_delta = (TimeDelta(group_max_timedelta.split('_')[0], format='jd') + \
#                              TimeDelta(group_max_timedelta.split('_')[1], format='jd') / 24 + \
#                              TimeDelta(group_max_timedelta.split('_')[2], format='jd') / (24 * 60) + \
#                              TimeDelta(group_max_timedelta.split('_')[3], format='jd') / (24 * 3600) + \
#                              TimeDelta(group_max_timedelta.split('_')[4], format='jd') / (24 * 3600 * 1e3)).to_value(
#                     'jd')
#
#             elem_delta = np.array([-get_overlap([elem_tstart, elem_tstop], [other_start, other_stop], distance=True) for
#                                    other_start, other_stop in zip(tstart_list, tstop_list)])
#
#             # list of matchable epochs
#             elem_epoch_id = np.array([id for id in range(len(tstart_list)) if \
#                                       id not in epoch_id_list_ravel and elem_delta[id] < max_delta])
#
#             epoch_id_list_ravel += elem_epoch_id.tolist()
#
#             if len(elem_epoch_id) > 0:
#                 epoch_id_list += [elem_epoch_id]
#
#             pbar.update(n=len(elem_epoch_id))
#
#     epoch_list = [spfile_list[elem] for elem in epoch_id_list]
#
#     epoch_list = np.array(epoch_list, dtype=object)
#
#     epoch_list_started = started_expos
#     epoch_list_done = done_expos
#

#currently testing to apply the multi matching permanently
epoch_list=[]

#skipping flares if asked for
if skip_flares:
    spfile_list=np.array([elem for elem in spfile_list if "F_sp" not in elem])

tstart_list=np.array([None]*len(spfile_list))
det_list=np.array([None]*len(spfile_list))
tstop_list=np.array([None]*len(spfile_list))

file_ok_ids=[]
for i_file,elem_file in enumerate(spfile_list):

    #for Suzaku this won't work for meugmi's xis0_xis3 files bc their header has been replaced
    # so we replace them by the xis1 to be able to load the exposure
    elem_file_load=elem_file.replace('xis0_xis3','xis1')

    try:
        fits.open(elem_file_load)
    except:
        print('Issue with fits opening for file:'+elem_file)
        continue

    with fits.open(elem_file_load) as hdul:
        if 'TELESCOP' in hdul[1].header:
            det_list[i_file]=hdul[1].header['TELESCOP'].replace('SUZAKU','Suzaku')
        else:
            #the only files without TELESCOP in the header should be the fused megumi_files suzaku sp
            assert megumi_files, 'Issue with detector handling'

            det_list[i_file]='Suzaku'


        if 'TIMEZERO' in hdul[1].header:
            start_obs_s = hdul[1].header['TSTART'] + hdul[1].header['TIMEZERO']
            stop_obs_s= hdul[1].header['TSTOP'] + hdul[1].header['TIMEZERO']
        else:
            start_obs_s=hdul[1].header['TSTART']
            stop_obs_s=hdul[1].header['TSTOP']

        # saving for titles later
        if 'MJDREFI' not in hdul[1].header:
            if det_list[i_file]=='SWIFT' and hdul[1].header['INSTRUME']=='BAT':
                MJDREFI = 51910
                MJDREFF = 7.4287037E-4
            else:
                print('Error: Cannot file MJDREF information in spectrum header')
                breakpoint()
        else:
            MJDREFI=hdul[1].header['MJDREFI']
            MJDREFF=hdul[1].header['MJDREFF']
        mjd_ref = Time(MJDREFI+MJDREFF, format='mjd')

        obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')
        obs_stop=mjd_ref+TimeDelta(stop_obs_s,format='sec')

    #note: don't convert to jd, jd have 0 at 12:00 instead of 00:00
    tstart_list[i_file]=obs_start.mjd
    tstop_list[i_file]=obs_stop.mjd

    file_ok_ids += [i_file]

epoch_id_list_ravel=[]
epoch_id_list=[]

tstart_list=tstart_list[file_ok_ids]
tstop_list = tstop_list[file_ok_ids]
det_list = det_list[file_ok_ids]

if sat_glob=='multi' and multi_focus!=False:
    #restricting the match epochs to a specific satellite
    mask_multi_focus=[elem==multi_focus for elem in det_list[file_ok_ids]]
    tstart_list_base=tstart_list[mask_multi_focus]
    tstop_list_base=tstop_list[mask_multi_focus]
    det_list_base=det_list[mask_multi_focus]
    id_base=np.arange(len(spfile_list))[mask_multi_focus]
else:
    tstart_list_base=tstart_list
    tstop_list_base=tstop_list
    det_list_base=det_list
    id_base=np.arange(len(spfile_list))

with tqdm(total=len(tstart_list)) as pbar:
    for id_elem,(elem_tstart,elem_tstop,elem_det) in enumerate(zip(tstart_list_base,tstop_list_base,det_list_base)):

        #skipping computation for already grouped elements
        if id_base[id_elem] in epoch_id_list_ravel:
            continue

        # max delta between gti starts in sec
        if group_max_timedelta == 'day':
            max_delta_bef = TimeDelta(np.ceil(elem_tstop)-elem_tstop,format='jd')
            max_delta_aft = TimeDelta(elem_tstart-np.floor(elem_tstart),format='jd')
        else:
            max_delta_bef = max_delta_aft = (TimeDelta(group_max_timedelta.split('_')[0], format='jd') + \
                         TimeDelta(group_max_timedelta.split('_')[1], format='jd') / 24 + \
                         TimeDelta(group_max_timedelta.split('_')[2], format='jd') / (24 * 60) + \
                         TimeDelta(group_max_timedelta.split('_')[3], format='jd') / (24 * 3600) + \
                         TimeDelta(group_max_timedelta.split('_')[4], format='jd') / (24 * 3600 * 1e3)).to_value(
                'jd')

        elem_delta=np.array([-get_overlap([elem_tstart,elem_tstop],[other_start,other_stop],distance=True) for other_start,other_stop in zip(tstart_list,tstop_list)])

        #list of matchable epochs
        # we automatically match epochs that have some time in common
        #if they don't, the maximum gap is symmetrical for time values, and otherwise
        #it is the distance to the beginning of the day where the obs started/the end of the day where the obs
        # finishes
        elem_epoch_id=np.array([id for id in range(len(tstart_list)) if\
                                 id not in epoch_id_list_ravel and\
                                (elem_delta[id]<=0 or \
                                 (elem_delta[id]<max_delta_aft and tstop_list[id]<elem_tstart) or \
                                 (elem_delta[id]<max_delta_bef and elem_tstop<tstart_list[id])) ])

        #multi options
        if sat_glob=='multi':
            #restricting match to single NICER epoch if required
            if match_closest_NICER and len(elem_epoch_id)>0:
                match_det_NICER=elem_epoch_id[det_list[elem_epoch_id]=='NICER']
                #restricting to the observations with overlap AND the closest non-overlapping
                match_valid_NICER=elem_delta[match_det_NICER]

                if len(match_valid_NICER)>0:
                    #computing NICER obs with some overlap
                    mask_overlaps_NICER=match_valid_NICER<0

                    #and the closest non-overlapping one
                    if len(match_valid_NICER[match_valid_NICER>=0])>0:
                        id_closest_NICER=match_valid_NICER[match_valid_NICER>=0].argmin()

                        #merging both
                        match_det_NICER_restrict=match_det_NICER[mask_overlaps_NICER].tolist()+\
                                                 [match_det_NICER[id_closest_NICER]]
                    else:
                        match_det_NICER_restrict=match_det_NICER[mask_overlaps_NICER].tolist()

                    #and replacing the initial NICER matches by this in the elem_epoch_id array
                    elem_epoch_id=np.array([elem for elem in elem_epoch_id if elem not in match_det_NICER\
                                            or elem in match_det_NICER_restrict])

            if single_obsid_NuSTAR:
                elem_epoch_obsids_NuSTAR=np.unique([elem.split('_')[0].split('-')[0][:-3] for elem in\
                    spfile_list[elem_epoch_id] if elem.startswith('nu')])

                mask_obsid_restrict=[not elem.startswith('nu') or elem.startswith(elem_epoch_obsids_NuSTAR[0])
                                     for elem in spfile_list[elem_epoch_id]]

                elem_epoch_id=np.array([elem_epoch_id])[[mask_obsid_restrict]].tolist()

            if skip_single_instru and len(np.unique(det_list[elem_epoch_id]))==1:
                continue
        else:
            #restricting to the obs of that telescope
            epoch_det_mask=[elem.lower() == sat_glob.lower() for elem in det_list[elem_epoch_id]]
            elem_epoch_id = elem_epoch_id[epoch_det_mask].tolist()

        epoch_id_list_ravel+=elem_epoch_id

        if len(elem_epoch_id)>0:
            epoch_id_list+=[elem_epoch_id]

        pbar.update(n=len(elem_epoch_id))

epoch_list=[spfile_list[elem] for elem in epoch_id_list]

epoch_list=np.array(epoch_list,dtype=object)

if sat_glob=='multi' and restrict_combination:
    epoch_list=[epoch_list[id_epoch] for id_epoch in range(len(epoch_list)) if (np.unique(det_list[epoch_id_list[id_epoch]])==restrict_combination.split('+')).all()]

#saving a txt file the tstart and tstop of each epoch
epoch_tstart_arr=np.array([np.array([tstart_list[np.argwhere(spfile_list[file_ok_ids]==elem)[0][0]] for elem in epoch])\
                       for epoch in epoch_list],
             dtype=object)

epoch_tstop_arr=np.array([np.array([tstop_list[np.argwhere(spfile_list[file_ok_ids]==elem)[0][0]] for elem in epoch])\
                       for epoch in epoch_list],
             dtype=object)

epoch_tstart_glob=np.array([min(elem) for elem in epoch_tstart_arr])
epoch_tstop_glob=np.array([max(elem) for elem in epoch_tstop_arr])
epoch_bounds=np.array([epoch_tstart_glob,epoch_tstop_glob]).T

#doing it like this because Time fucks up multidimensionnal orders
epoch_bounds_time=np.array([Time(epoch_bounds.T[0],format='mjd').isot,Time(epoch_bounds.T[1],format='mjd').isot]).T

epoch_bounds_save=np.array([epoch_bounds_time.T[0],epoch_bounds_time.T[1],
                    epoch_bounds.T[0].astype(str),epoch_bounds.T[1].astype(str)]).T

with open(os.path.join(outdir,'epoch_bounds.txt'),'w+') as f:
    f.write('#isot_start\tisot_stop\tmjd_start\tmjd_stop\n')
    f.writelines(['\t'.join(elem)+'\n' for elem in epoch_bounds_save])

#for bat if needed
epoch_mean_days=np.array([elem.split('T')[0] for elem in Time(epoch_bounds.mean(1),format='mjd').isot])
with open(os.path.join(outdir,'epoch_mean_days.txt'),'w+') as f:
    f.writelines([elem+'\n' for elem in epoch_mean_days])

if plot_epoch_overlap:

    '''
    We plot both the initial telescope overlaps with colors for each instrument, 
    then the final epochs as they end up, cycling through colors for each epoch
    '''

    from visual_line_tools import telescope_colors
    import matplotlib.dates as mdates

    print('Plotting multi epoch overlaps...')

    for elem_epoch in epoch_list:

        fig_exp, ax_exp = plt.subplots(figsize=(17, 6))

        # precise format because we might need it
        date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')

        epoch_mask=np.array([np.argwhere(spfile_list==elem)[0][0] for elem in elem_epoch])


        tel_col_list = list(telescope_colors.keys())
        mask_tel = np.array([np.array(det_list[epoch_mask]) == elem for elem in tel_col_list])

        num_dates_start_epoch = mdates.date2num(Time(tstart_list.astype(float)[epoch_mask], format='mjd').datetime)
        num_dates_stop_epoch = mdates.date2num(Time(tstop_list.astype(float)[epoch_mask], format='mjd').datetime)

        #not needed now that we plot everything afterwards
        # #cylcing through each telescope and their respective epochs to get different colors
        # for i_det in range(len(tel_col_list)):
        #
        #     for i_exp in range(sum(mask_tel[i_det])):
        #         ax_exp.axvspan(xmin=num_dates_start[mask_tel[i_det]][i_exp],
        #                        xmax=num_dates_stop[mask_tel[i_det]][i_exp],
        #                        ymin=0, ymax=0.5, color=telescope_colors[tel_col_list[i_det]],
        #                        label=tel_col_list[i_det] if i_exp == 0 else '', alpha=0.2)

        #and doing the same with the remaining elements of epoch_list
        prop_cycle = plt.rcParams['axes.prop_cycle']
        mpl_cycle_colors = prop_cycle.by_key()['color']

        epoch_color = 'blue'
        # epoch_color=mpl_cycle_colors[i_epoch%len(mpl_cycle_colors)]

        for i_file,elem_file in enumerate(elem_epoch):

            num_date_start_file=num_dates_start_epoch[spfile_list[epoch_mask]==elem_file][0]
            num_date_stop_file=num_dates_stop_epoch[spfile_list[epoch_mask]==elem_file][0]

            ax_exp.axvspan(xmin=num_date_start_file,
                           xmax=num_date_stop_file,
                           ymin=0.5, ymax=1, color=epoch_color,
                           label='selected epochs' if i_file==0 else '', alpha=0.4)

        ax_exp.xaxis.set_major_formatter(date_format)
        for label in ax_exp.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='right')

        #locking the x axis with a one day margin around the epoch
        ax_exp.set_xlim(ax_exp.get_xlim()[0]-1, ax_exp.get_xlim()[1]+1)

        #plotting the rest of the exposures (cheap way to show them without having to sort them
        mask_tel = np.array([np.array(det_list) == elem for elem in tel_col_list])

        num_dates_start_all = mdates.date2num(Time(tstart_list.astype(float), format='mjd').datetime)
        num_dates_stop_all = mdates.date2num(Time(tstop_list.astype(float), format='mjd').datetime)

        #cylcing through each telescope and their respective epochs to get different colors

        tel_col_shown=[]

        for i_det in range(len(tel_col_list)):

            for i_exp in range(sum(mask_tel[i_det])):

                bar_in_plot=get_overlap([num_dates_start_all[mask_tel[i_det]][i_exp],num_dates_stop_all[mask_tel[i_det]][i_exp]],ax_exp.get_xlim())>0

                ax_exp.axvspan(xmin=num_dates_start_all[mask_tel[i_det]][i_exp],
                               xmax=num_dates_stop_all[mask_tel[i_det]][i_exp],
                               ymin=0, ymax=0.5, color=telescope_colors[tel_col_list[i_det]],
                               label=tel_col_list[i_det] if (bar_in_plot and tel_col_list[i_det] not in tel_col_shown) else '', alpha=0.4)

                if bar_in_plot:
                    tel_col_shown+=[tel_col_list[i_det]]

        plt.tight_layout()
        plt.legend()

        #short epoch id
        short_ep_str=shorten_epoch([elem_sp.split('_gti')[0].split('_sp')[0].split('src')[0] for elem_sp in elem_epoch])
        plt.savefig(os.path.join(outdir,'_'.join(short_ep_str)+'_epoch_matching.png'))
        plt.savefig(os.path.join(outdir,'_'.join(short_ep_str)+'_epoch_matching.pdf'))
        plt.close()

epoch_list_started=started_expos
epoch_list_done=done_expos

if save_epoch_list:
    with open(os.path.join(outdir,'epoch_list.txt'),'w+') as f:
        f.writelines([str(np.array(elem_epoch).tolist())+'\n'  for elem_epoch in epoch_list])

if force_epochs:
    epoch_list=force_epochs_list

if spread_comput!=1:

    epoch_list_save=epoch_list

    if skip_started_spread:
        epoch_list=np.array([elem_epoch for elem_epoch in epoch_list\
                    if shorten_epoch([elem_sp.split('_sp')[0]  for elem_sp in elem_epoch]) not in started_expos],
                            dtype=object)

    spread_epochs=np.array_split(epoch_list, spread_comput)

    files_spread=glob.glob(os.path.join(outdir,'spread_epoch_'+('rev_' if reverse_spread else '')+'*'),recursive=True)

    assert len(files_spread)<spread_comput,'Computation already split over desired amount of elements'

    for i in range(spread_comput):
        file_spread=os.path.join(outdir,'spread_epoch_'+('rev_' if reverse_spread else '')+str(i+1)+'.txt')
        if not os.path.isfile(file_spread):

            if not spread_overwrite:
                split_epoch=[epoch for epoch in spread_epochs[i] if shorten_epoch([elem.split('_sp')[0]\
                                                                              for elem in epoch]) not in started_expos]
            else:
                split_epoch=spread_epochs[i]

            with open(file_spread,'w+') as f:
                f.writelines([str(elem)+'\n' for elem in split_epoch[::(-1 if reverse_spread else 1)]])
            epoch_list=split_epoch[::(-1 if reverse_spread else 1)]
            break

if reverse_epoch:
    epoch_list=epoch_list[::-1]

# #just to be able to make a first summary earlier
# #first obs
# epoch_list=np.concatenate((epoch_list[:26],epoch_list[27:]))
#
# #second obs
# epoch_list=np.concatenate((epoch_list[:-51],epoch_list[-50:]))

#replacing epoch list by what's in the summary folder if asked to
if rewind_epoch_list:
    assert sat_glob in ['NICER','NuSTAR'], 'rewind epoch list not implemented with sats other than NICER'
    if sat_glob=='NICER':
        suffix_str='_sp_grp_opt.pha'
    elif sat_glob=='NuSTAR':
        suffix_str='_sp_src_grp_opt.pha'

        epoch_list=[[elem+suffix_str for elem in expand_epoch(started_expos[i])] for i in range(len(started_expos))]

if load_epochs:
    print('Epoch loading mode activated. Stopping here...')
    breakpoint()


#creating a HUGE argument dictionnary which will be saved in a file to make parralelizing simpler
arg_dict={}

arg_dict['NICER_bkg']=NICER_bkg
arg_dict['NICER_lc_binning']=NICER_lc_binning
arg_dict['SNR_min']=SNR_min
arg_dict['assess_line']=assess_line
arg_dict['assess_line_upper']=assess_line_upper
arg_dict['autofit']=autofit
arg_dict['autofit_model']=autofit_model
arg_dict['autofit_store_path']=autofit_store_path
arg_dict['bad_flags']=bad_flags
arg_dict['broad_HID_mode']=broad_HID_mode
arg_dict['cameras']=cameras
arg_dict['catch_errors']=catch_errors
arg_dict['compute_highflux_only']=compute_highflux_only
arg_dict['cont_fit_method']=cont_fit_method
arg_dict['cont_model']=cont_model
arg_dict['counts_min']=counts_min
arg_dict['counts_min_HID']=counts_min_HID
arg_dict['done_expos']=done_expos
arg_dict['epoch_list']=epoch_list
arg_dict['epoch_list_started']=epoch_list_started
arg_dict['epoch_restrict']=epoch_restrict
arg_dict['expmodes']=expmodes
arg_dict['filter_NuSTAR_SNR']=filter_NuSTAR_SNR
arg_dict['fit_SAA_norm']=fit_SAA_norm
arg_dict['fit_lowSNR']=fit_lowSNR
arg_dict['fitstat']=fitstat
arg_dict['force_autofit']=force_autofit
arg_dict['force_ener_bounds']=force_ener_bounds
arg_dict['freeze_nh']=freeze_nH
arg_dict['freeze_nH_val']=freeze_nH_val
arg_dict['glob_summary_reg']=glob_summary_reg
arg_dict['glob_summary_sp']=glob_summary_sp
arg_dict['h_update']=h_update
arg_dict['hid_cont_range']=hid_cont_range
arg_dict['line_cont_range']=line_cont_range
arg_dict['line_cont_range_arg']=args.line_cont_range
arg_dict['line_search_e']=line_search_e
arg_dict['line_search_e_arg']=args.line_search_e
arg_dict['line_search_norm']=line_search_norm
arg_dict['line_search_norm_arg']=args.line_search_norm
arg_dict['line_store_path']=line_store_path
arg_dict['line_ul_only']=line_ul_only
arg_dict['log_console']=log_console
arg_dict['mandatory_abs']=mandatory_abs
arg_dict['max_bg_imaging']=max_bg_imaging
arg_dict['megumi_files']=megumi_files
arg_dict['merge_cont']=merge_cont
arg_dict['nfakes']=nfakes
arg_dict['no_abslines']=no_abslines
arg_dict['obj_name']=obj_name
arg_dict['outdir']=outdir
arg_dict['overwrite']=overwrite
arg_dict['pdf_only']=pdf_only
arg_dict['peak_clean']=peak_clean
arg_dict['peak_thresh']=peak_thresh
arg_dict['pileup_lim']=pileup_lim
arg_dict['pileup_missing']=pileup_missing
arg_dict['pre_reduced_NICER']=pre_reduced_NICER
arg_dict['refit_cont']=refit_cont
arg_dict['reload_autofit']=reload_autofit
arg_dict['reload_fakes']=reload_fakes
arg_dict['restrict']=restrict
arg_dict['restrict_graded']=restrict_graded
arg_dict['restrict_order']=restrict_order
arg_dict['sat_glob']=sat_glob
arg_dict['sign_threshold']=sign_threshold
arg_dict['skip_complete']=skip_complete
arg_dict['skip_nongrating']=skip_nongrating
arg_dict['skip_started']=skip_started
arg_dict['skip_bg_timing']=skipbg_timing
arg_dict['split_fit']=split_fit
arg_dict['spread_comput']=spread_comput
arg_dict['started_expos']=started_expos
arg_dict['summary_header']=summary_header
arg_dict['trig_interval']=trig_interval
arg_dict['write_aborted_pdf']=write_aborted_pdf
arg_dict['write_pdf']=write_pdf
arg_dict['xchatter']=xchatter
arg_dict['epoch_list']=epoch_list
arg_dict['xspec_query']=xspec_query

arg_dict['diff_bands_NuSTAR_NICER']=diff_bands_NuSTAR_NICER
arg_dict['low_E_NICER']=low_E_NICER
arg_dict['suzaku_xis_ignore']=suzaku_xis_ignore
arg_dict['suzaku_xis_range'] =suzaku_xis_range
arg_dict['suzaku_pin_range'] =suzaku_pin_range
arg_dict['line_cont_ig_arg'] = line_cont_ig_arg

arg_dict_path=os.path.join(outdir,'arg_dict_dump.dill')

with open(os.path.join(outdir,'arg_dict_dump.dill'),'wb') as dump_file:
    dill.dump(arg_dict,dump_file)

if not hid_only:
    aborted_epochs=linedet_loop(epoch_list,arg_dict,arg_dict_path=arg_dict_path,parallel=parallel,
                                container_mode=container_mode,container=container,force_instance=force_instance)
else:
    aborted_epochs=[]

'''''''''''''''''''''''''''''''''''''''
''''''Hardness-Luminosity Diagrams''''''
'''''''''''''''''''''''''''''''''''''''

#stopping the program before the final pdfs to avoid problems with server side computation
# (which has issues with streamlit)
if os.getcwd().startswith('/user/'):
    sys.exit()

#importing some graph tools from the streamlit script
from visual_line_tools import load_catalogs,dist_mass,obj_values,abslines_values,values_manip,n_infos,telescope_list,hid_graph


'Distance and Mass determination'

catal_blackcat,catal_watchdog,catal_blackcat_obj,catal_watchdog_obj,catal_maxi_df,catal_maxi_simbad,\
    catal_bat_df,catal_bat_simbad=load_catalogs()

all_files=glob.glob('**',recursive=True)
lineval_id='line_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
lineval_files=[os.path.join(os.getcwd(),elem) for elem in all_files if outdir+'/' in elem and lineval_id in elem]

abslines_id='autofit_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
abslines_files=[os.path.join(os.getcwd(),elem) for elem in all_files if outdir+'/' in elem and abslines_id in elem]

if multi_obj:
    obj_list=[elem.split('/')[0] for elem in lineval_files]
else:
    obj_list=np.array([obj_name])

dict_linevis={
    'ctl_blackcat':catal_blackcat,
    'ctl_blackcat_obj':catal_blackcat_obj,
    'ctl_watchdog':catal_watchdog,
    'ctl_watchdog_obj':catal_watchdog_obj,
    'lineval_files':lineval_files,
    'obj_list':obj_list,
    'cameras':'all' if sat_glob=='multi' else cameras,
    'expmodes':expmodes,
    'multi_obj':multi_obj,
    'range_absline':range_absline,
    'n_infos':n_infos,
    'args_cam':args.cameras,
    'args_line_search_e':args.line_search_e,
    'args_line_search_norm':args.line_search_norm,
    'visual_line':False
    }

#getting the single parameters
dist_obj_list,mass_obj_list=dist_mass(dict_linevis)

#distance factor for the flux conversion later on
dist_factor=4*np.pi*(dist_obj_list*1e3*3.086e18)**2

#L_Edd unit factor
Edd_factor=dist_factor/(1.26e38*mass_obj_list)

#Reading the results files
observ_list,lineval_list,lum_list,date_list,instru_list,exptime_list,fitmod_broadband_list,epoch_obs_list,\
    flux_high_list=obj_values(lineval_files,Edd_factor,dict_linevis)

dict_linevis['lum_list']=lum_list
dict_linevis['exptime_list']=exptime_list
#the values here are for each observation
abslines_infos,autofit_infos=abslines_values(abslines_files,dict_linevis)

# getting all the variations we need

# getting all the variations we need
abslines_infos_perline, abslines_infos_perobj, abslines_plot, abslines_ener, \
    lum_plot, hid_plot, incl_plot, width_plot, nh_plot, kt_plot = values_manip(abslines_infos, dict_linevis,
                                                                                autofit_infos,
                                                                                lum_list)

'''
in this form, the order is:
    -each habsorption line
    -the number of sources
    -the number of obs for each source
    -the info (4 rows, eqw/bshift/delchi/sign)
    -it's uncertainty (3 rows, main value/neg uncert/pos uncert,useless for the delchi and sign)
'''

flag_noexp=0
#taking of the bad files points from the HiD
if multi_obj:

    #in multi object mode, we loop one more time for each object
    for i in range(len(observ_list)):

        bad_index=[]
        #check if the obsid identifiers of every index is in the bad flag list
        for j in range(len(observ_list[i])):
            if np.any(observ_list[i][j] in bad_flags):
                bad_index+=[j]

        #and delete the resulting indexes from the arrays
        observ_list[i]=np.delete(observ_list[i],bad_index)
        lineval_list[i]=np.delete(lineval_list[i],bad_index,axis=0)
        lum_list[i]=np.delete(lum_list[i],bad_index,axis=0)
        # link_list[i]=np.delete(link_list[i],bad_index)

#same process for a single object
else:
    bad_index=[]

    #checking if the observ list isn't empty before trying to delete anything
    if len(observ_list)!=0:
        for j in range(len(observ_list[0])):
            if np.any(observ_list[0][j] in bad_flags):
                bad_index+=[j]

        #and delete the resulting indexes from the arrays
        observ_list[0]=np.delete(observ_list[0],bad_index)
        lineval_list[0]=np.delete(lineval_list[0],bad_index,axis=0)
        lum_list[0]=np.delete(lum_list[0],bad_index,axis=0)
        # link_list[0]=np.delete(link_list[0],bad_index)

#checking if the obsid identifiers of every index is in the bad flag list or if there's just no file
if len(observ_list.ravel())==0:
    print('\nNo line detection to build HID graph.')
    flag_noexp=True
else:
    flag_noexp=False


# string of the colormap legend for the informations
radio_info_label = ['Velocity shift', r'$\Delta-C$', 'Equivalent width ratio']

# masking for restriction to single objects
mask_obj_select = np.repeat(True, len(obj_list))

# masking the objects depending on inclination
mask_inclin = True

# double mask taking into account both single/multiple display mode and the inclination

mask_obj_base = (mask_obj_select) & (mask_inclin)

#### Array restrictions

# time delta to add some leeway to the limits available and avoid directly cropping at the observations
delta_1y = TimeDelta(365, format='jd')
delta_1m = TimeDelta(30, format='jd')

slider_date = [Time(min(ravel_ragged(date_list[mask_obj_base]))).datetime,
               Time(max(ravel_ragged(date_list[mask_obj_base]))).datetime]

# creating a mask according to the sources with observations in the current date restriction
mask_obj_intime = True

# restricting mask_obj_base with the new base
mask_obj_base = mask_obj_base & mask_obj_intime

#only selecting the Ka complex for display
mask_lines=np.array([True,True,False,False,False,False])

mask_obs_intime_repeat = np.array([np.repeat(((np.array([Time(subelem) for subelem in elem]) >= Time(slider_date[0])) & \
                                              (np.array([Time(subelem) for subelem in elem]) <= Time(slider_date[1]))),
                                             sum(mask_lines)) for elem in date_list], dtype=bool)

# checking which sources have no detection in the current combination

if 1 or multi_obj:
    global_displayed_sign = np.array(
        [ravel_ragged(elem)[mask] for elem, mask in zip(abslines_plot[4][0][mask_lines].T, mask_obs_intime_repeat)],
        dtype=object)
else:
    global_displayed_sign = np.array(
        [ravel_ragged(elem)[mask.astype] for elem, mask in zip(np.transpose(abslines_plot[4][0][mask_lines].T,
                                                                     (1,2,0)), mask_obs_intime_repeat)],
        dtype=object)

# creating a mask from the ones with at least one detection
# (or at least one significant detections if we don't consider non significant detections)
mask_obj = mask_obj_base

mask_obj_withdet=np.array([(elem>=sign_threshold).any() for elem in global_displayed_sign])

#storing the number of objects with detections
n_obj_withdet=sum(mask_obj_withdet & mask_obj_base)

if len(mask_obj)==1 and np.ndim(hid_plot)==4:
    #this means there's a single object and thus the arrays are built directly
    #in this case the restrcited array are the same than the non-restricted one
    hid_plot_restrict=deepcopy(hid_plot)
    nh_plot_restrict=deepcopy(nh_plot)
    kt_plot_restruct=deepcopy(kt_plot)
else:
    hid_plot_restrict=hid_plot.T[mask_obj].T

    nh_plot_restrict = deepcopy(nh_plot)
    nh_plot_restrict = nh_plot_restrict.T[mask_obj].T

    kt_plot_restrict = deepcopy(kt_plot)
    kt_plot_restrict = kt_plot_restrict.T[mask_obj].T


incl_plot_restrict = incl_plot[mask_obj]

# creating variables with values instead of uncertainties for the inclination and nh colormaps

#no need to do this here
# incl_cmap = np.array([incl_plot.T[0], incl_plot.T[0] - incl_plot.T[1], incl_plot.T[0] + incl_plot.T[2]]).T
# incl_cmap_base = incl_cmap[mask_obj_base]
# incl_cmap_restrict = incl_cmap[mask_obj]

incl_cmap=None
incl_cmap_base=None
incl_cmap_restrict=None




radio_info_cmap='Time'
radio_cmap_i = 0

# colormap when not splitting detections
cmap_color_source = mpl.cm.hsv_r.copy()

cmap_color_source.set_bad(color='grey')
cyclic_cmap = True

# colormaps when splitting detections
cmap_color_det = mpl.cm.plasma.copy()
cmap_color_det.set_bad(color='grey')

cyclic_cmap_det = False

cmap_color_nondet = mpl.cm.viridis_r.copy()
cmap_color_nondet.set_bad(color='grey')

cyclic_cmap_nondet = False

# computing the extremal values of the whole sample/plotted sample to get coherent colormap normalisations, and creating the range of object colors

global_plotted_sign = ravel_ragged(abslines_plot[4][0]).astype(float)
global_plotted_data = ravel_ragged(abslines_plot[radio_cmap_i][0]).astype(float)

# objects colormap for common display
norm_colors_obj = mpl.colors.Normalize(vmin=0,
                                       vmax=max(0, len(abslines_infos_perobj) + (-1 if not cyclic_cmap else 0)))
colors_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_source)

norm_colors_det = mpl.colors.Normalize(vmin=0, vmax=max(0, n_obj_withdet + (-1 if not cyclic_cmap_det else 0) + (
    1 if n_obj_withdet == 0 else 0)))
colors_det = mpl.cm.ScalarMappable(norm=norm_colors_det, cmap=cmap_color_det)

norm_colors_nondet = mpl.colors.Normalize(vmin=0, vmax=max(0, len(abslines_infos_perobj) - n_obj_withdet + (
    -1 if not cyclic_cmap_nondet else 0)))
colors_nondet = mpl.cm.ScalarMappable(norm=norm_colors_nondet, cmap=cmap_color_nondet)

# the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
global_plotted_datetime = np.array([elem for elem in date_list for i in range(len(mask_lines))], dtype='object')

global_mask_intime = np.repeat(True, len(ravel_ragged(global_plotted_datetime)))

global_mask_intime_norepeat = np.repeat(True, len(ravel_ragged(date_list)))

# global_nondet_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])<=slider_sign) & (global_mask_intime)

if not multi_obj:
    global_det_mask =(global_plotted_sign > 0) & (global_mask_intime)

    global_sign_mask = (global_plotted_sign>=sign_threshold ) & (global_mask_intime)

    global_det_data = global_plotted_data[global_det_mask]

    # this second array is here to restrict the colorbar scalings to take into account significant detections only
    global_sign_data = global_plotted_data[global_sign_mask]

else:
    global_det_mask = (np.array([subelem for elem in global_plotted_sign for subelem in elem]) > 0) & (global_mask_intime)

    global_sign_mask = (np.array([subelem for elem in global_plotted_sign for subelem in elem]) >= sign_threshold) & (
        global_mask_intime)

    global_det_data = np.array([subelem for elem in global_plotted_data for subelem in elem])[global_det_mask]

    # this second array is here to restrict the colorbar scalings to take into account significant detections only
    global_sign_data = np.array([subelem for elem in global_plotted_data for subelem in elem])[global_sign_mask]

# same for the color-coded infos
cmap_info = mpl.cm.plasma_r.copy() if radio_info_cmap not in ['Time', 'nH', 'kT'] else mpl.cm.plasma.copy()

cmap_info.set_bad(color='grey')

# normalisation of the colormap
if radio_cmap_i == 1 or radio_info_cmap == 'EW ratio':
    gamma_colors = 1 if radio_cmap_i == 1 else 0.5
    cmap_norm_info = colors.PowerNorm(gamma=gamma_colors)

elif radio_info_cmap not in ['Inclination', 'Time', 'kT']:
    cmap_norm_info = colors.LogNorm()
else:
    # keeping a linear norm for the inclination
    cmap_norm_info = colors.Normalize()

#replacing some variable names to keep similar dictionnary items
slider_sign=sign_threshold
mask_lines_ul=mask_lines

#doesn't matter here
choice_telescope=None
bool_incl_inside=False
bool_noincl=True

#necessary items for the hid graph run
items_list=[
abslines_infos_perobj,
abslines_plot,nh_plot,kt_plot,hid_plot, incl_plot,
mask_obj, mask_obj_base, mask_lines, mask_lines_ul,
obj_list, date_list, instru_list, lum_list, choice_telescope, telescope_list,
bool_incl_inside, bool_noincl,
slider_date, slider_sign,
radio_info_cmap, radio_cmap_i,
cmap_color_source, cmap_color_det, cmap_color_nondet]

'''Diagram creation'''

if multi_obj:
    os.system('mkdir -p glob_batch')

fig_hid, ax_hid = plt.subplots(1, 1, figsize=(12,6))

#log x scale for an easier comparison with Ponti diagrams
ax_hid.set_xscale('log')
ax_hid.set_xlabel('Hardness Ratio ([6-10]/[3-6] keV bands)')
ax_hid.set_ylabel(r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units')
ax_hid.set_yscale('log')


items_str_list=['abslines_infos_perobj',
'abslines_plot','nh_plot','kt_plot','hid_plot','incl_plot',
'mask_obj','mask_obj_base', 'mask_lines', 'mask_lines_ul',
'obj_list', 'date_list', 'instru_list', 'lum_list', 'choice_telescope', 'telescope_list',
'bool_incl_inside', 'bool_noincl',
'slider_date', 'slider_sign',
'radio_info_cmap', 'radio_cmap_i',
'cmap_color_source', 'cmap_color_det', 'cmap_color_nondet']

for dict_key, dict_item in zip(items_str_list,items_list):
    dict_linevis[dict_key]=dict_item

#we don't use this one
dict_linevis['Tin_diskbb_plot_restrict']=None
dict_linevis['Tin_diskbb_plot']=None
dict_linevis['diago_color']=None
dict_linevis['custom_states_color']=None
dict_linevis['custom_outburst_color']=None
dict_linevis['custom_outburst_number']=None
dict_linevis['hr_high_plot_restrict']=None
dict_linevis['lum_high_1sig_plot_restrict']=None
dict_linevis['lum_high_sign_plot_restrict']=None
dict_linevis['hid_log_HR']=True
dict_linevis['flag_single_obj']=True
dict_linevis['display_minorticks']=False
dict_linevis['hatch_unstable']=False
dict_linevis['change_legend_position']=False

#individual plotting options for the graph that will create the PDF
display_single=not multi_obj
display_upper=True
display_evol_single=display_single
diago_color=None


hid_graph(ax_hid,dict_linevis,
          display_single=True,display_upper=True,display_evol_single=True,
          alpha_abs=0.5)

'''Global recap creation'''

#some naming variables for the files
save_dir='glob_batch' if multi_obj else outdir

if multi_obj:
    save_str_prefix=''
else:
    save_str_prefix=obj_list[0]+'_'

def save_pdf(fig):

    '''
    Saves a global PDF with, for each currently drawn point, a copy of the figure with the point highlighted, and the associated PDF

    Also adds a part for failed detections

    In order to do that, we fetch the current figure's hid axe, then identify all of the points and order them per increasing hardness.
    We then draw a one point plot with a circle marker (standard patch shapes wouldn't work with log log) around each point,
    save an image of the plot and add it, along with the point's associated pdf, to a global pdf

    The pdf is created in three parts :

    First, we create every single HID page.

    Second: We parse and fusion the sections with each recap PDF

    Third: we add the merge of all the unfinished pdfs
    '''

    #fetching the axe
    if fig is not None:
        ax_hid=plt.gca().get_figure().get_axes()[0]
    else:
        ax_hid=fig.get_axes()[0]
    #fetching the (up to 3) Path Collections of the scatter plots
    linecols_hid=[elem for elem in ax_hid.get_children()[:3] if type(elem)==mpl.collections.PathCollection]

    #stopping the process in there are no points currently drawn although there should be (i.e. the no exposure flag is not set to True)
    if linecols_hid==[] and not flag_noexp:
        print('\nNo points currently drawn in the figure. Cannot save pdf.')
        return None

    #creating the main pdf
    pdf=FPDF(orientation="landscape")
    pdf.add_page()

    pdf_path=save_dir+'/'+save_str_prefix+'HLD_cam_'+args.cameras+'_'+\
                args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'_'+hid_sort_method+'.pdf'

    #the main part of the summary creation is only gonna work if we have points in the graph
    if not flag_noexp:

        #rassembling all points
        #we concatenate all of the points array of the (up to 3) groups, and restrict uniques to avoid two points for absorption/emissions
        points_hid=np.unique(np.concatenate([linecols_hid[i].get_offsets().data for i in range(len(linecols_hid))]),axis=0)

        #NOTE: bad idea + doesn't work
        #re-organizing the points according to the date

        #sorting the displayed HID values according to how they are stored initially

        sort_hid_disp=np.array([np.argwhere(points_hid.T[1] == elem)[0][0] for elem in ravel_ragged(hid_plot[1][0])])

        if hid_sort_method=='date':
            hid_sort=ravel_ragged(date_list).argsort()
            points_hid=points_hid[sort_hid_disp][hid_sort]

        elif hid_sort_method=='hr':

            #default sorting
            pass
            # hid_sort=np.array(range(len(ravel_ragged(date_list))),dtype=int)



        #saving the current graph in a pdf format
        plt.savefig(save_dir+'/curr_hid.png')

        #before adding it to the pdf
        pdf.image(save_dir+'/curr_hid.png',x=1,w=280)

        #resticting other infos to if they are asked
        #note : these are very outdated compared to everything visual_line does nowadays so we don't plot these by default

        '''
        identification of the points
        '''

        #fetching the number of pages after all the graphs
        init_pages=pdf.page_no()

        #this is only for single object mode
        if not multi_obj:
            outline_circles=np.array([None]*len(points_hid))

            with tqdm(total=len(points_hid)) as pbar:

                #first loop for the creation of section pages
                for point_id,single_point in enumerate(points_hid):

                    #drawing the circle around the point (ms is the marker size, mew it edge width, mfc its face color, mec its edge color)
                    outline_circles[point_id]=ax_hid.plot(single_point[0],single_point[1],color='white',\
                                                          zorder=0,marker='o',ms=40,mew=2,mfc=None,mec='red')

                    #saving the figure to pdf
                    plt.savefig(save_dir+'/curr_hid_highlight_'+str(point_id)+'.png')

                    pdf.add_page()

                    #identifying the exposure id for this point
                    #note : this won't work in multi object mode
                    point_observ=observ_list[0][np.argwhere(lum_list[0].T[4][0]==single_point[1])[0][0]]

                    pdf.image(save_dir+'/curr_hid_highlight_'+str(point_id)+'.png',x=1,w=280)

                    outline_circles[point_id][0].remove()
                    pbar.update()

            for i in range(len(points_hid)):
                os.remove(save_dir+'/curr_hid_highlight_'+str(i)+'.png')

            os.remove(save_dir+'/curr_hid.png')

        #adding unfinished analysis section
        pdf.add_page()

        pdf.output(pdf_path)

        #listing the files in the save_dir
        save_dir_list=glob.glob(save_dir+'/*',recursive=True)

        #creating the merger pdf
        merger=PdfMerger()

        #adding the first page
        merger.append(pdf_path,pages=(0,init_pages))

        #stopping the pdf creation here for multi_obj mode
        if multi_obj:
            merger.write(save_dir+'/temp.pdf')
            merger.close()

            os.remove(pdf_path)
            os.rename(save_dir+'/temp.pdf',pdf_path)
            print('\nHLD summary PDF creation complete.')
            return

        bkm_completed=merger.add_outline_item('Completed Analysis',len(merger.pages))
        #second loop to insert the recaps
        for point_id,single_point in enumerate(points_hid):

            #once again fetching the exposure identier
            point_observ=observ_list[0][np.argwhere(lum_list[0].T[4][0]==single_point[1])[0][0]]

            #there should only be one element here
            try:
                point_recapfile=[elem for elem in save_dir_list if point_observ+'_recap.pdf' in elem][0]
            except:
                avail_recapfile=glob.glob(os.path.join(save_dir, point_observ + '**_recap.pdf'))

                if len(avail_recapfile)!=1:
                    breakpoint()

                assert len(avail_recapfile)==1,"Issue with finding individual pdf recap files"

                point_recapfile=avail_recapfile[0]

            if sat_glob=='NICER':
                point_observ=point_recapfile.split('/')[-1].split('_recap')[0]

            #adding the corresponding hid highlight page
            merger.add_outline_item(point_observ,len(merger.pages),parent=bkm_completed)
            merger.append(pdf_path,pages=(point_id+init_pages,point_id+init_pages+1))

            #adding the recap
            merger.append(point_recapfile)

            # #adding the aborted analysis section to the merger
            # merger.append(pdf_path,pages=(point_id+3,point_id+4))
    else:
        pdf.output(pdf_path)

        #creating the merger pdf
        merger=PdfMerger()
        # #adding the (empty) pdf in order to be able to add a bookmark
        # merger.append(pdf_path)

    bkm_aborted=merger.add_outline_item('Aborted Analysis',len(merger.pages))

    for elem_epoch in aborted_epochs:
        curr_pages=len(merger.pages)

        if sat_glob=='NICER':
            short_ep_id='_'.join(shorten_epoch(elem_epoch))

            merger.append(save_dir + '/' + short_ep_id+ '_aborted_recap.pdf')
            bkm_completed=merger.add_outline_item(short_ep_id,curr_pages,parent=bkm_aborted)
        else:
            merger.append(save_dir + '/' + elem_epoch[0].split('_sp')[0] + '_aborted_recap.pdf')
            bkm_completed=merger.add_outline_item(elem_epoch[0].split('_sp')[0],curr_pages,parent=bkm_aborted)

    #overwriting the pdf with the merger, but not directly to avoid conflicts
    merger.write(save_dir+'/temp.pdf')
    merger.close()

    os.remove(pdf_path)
    os.rename(save_dir+'/temp.pdf',pdf_path)

    print('\nHLD summary PDF creation complete.')

save_pdf(fig_hid)
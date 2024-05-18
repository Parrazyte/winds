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

#trapezoid integration
from scipy.integrate import trapezoid

from astropy.time import Time,TimeDelta

'''Astro'''
#general astro importss
from astropy.io import fits
from astropy.time import Time
from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,Chain
from xspec import AllChains

#custom script with a few shorter xspec commands
from xspec_config_multisp import allmodel_data,model_load,addcomp,Pset,Pnull,rescale,reset,Plot_screen,store_plot,freeze,allfreeze,unfreeze,\
                         calc_error,delcomp,fitmod,calc_fit,xcolors_grp,xPlot,xscorpeon,catch_model_str,\
                         load_fitmod, ignore_data_indiv,par_degroup,xspec_globcomps,store_fit

from linedet_utils import plot_line_comps,plot_line_search,plot_std_ener,coltour_chi2map,narrow_line_search,\
                            plot_line_ratio

#custom script with a some lines and fit utilities and variables
from fitting_tools import c_light,lines_std_names,lines_e_dict,n_absline,range_absline,model_list

from general_tools import file_edit,ravel_ragged,shorten_epoch,expand_epoch,get_overlap

# #importing the pileup evaluation function
# from XMM_datared import pileup_val
#defined here instead because it tends to launch datared functions when importing from it

#Catalogs and manipulation
from astroquery.vizier import Vizier

'''peak detection'''
from findpeaks import findpeaks
#mask to polygon conversion
from imantics import Mask
from shapely.geometry import Polygon,Point
#mask propagation for the peak detection
from scipy.ndimage import binary_dilation

ap = argparse.ArgumentParser(description='Script to perform line detection in X-ray Spectra.\n)')

'''GENERAL OPTIONS'''

ap.add_argument('-satellite',nargs=1,help='telescope to fetch spectra from',default='multi',type=str)

#used for NICER and multi for now
ap.add_argument('-group_max_timedelta',nargs=1,
                help='maximum time delta for epoch/gti grouping in dd_hh_mm_ss_ms',default='00_8_00_00_000',type=str)

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
                default="NuSTAR_epochs_8h",type=str)

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
                default='nthcont',type=str)

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
                help='minimum source counts in the source region in the line continuum range',default=2500,type=float)

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

ap.add_argument('-load_epochs',nargs=1,help='prepare epochs then exit',default=True)
#options: "opt" (tests the significance of each components and add them accordingly)
# and "force_all" to force all components
ap.add_argument('-cont_fit_method',nargs=1,help='fit logic for the broadband fits',
                default='force_add')

ap.add_argument('-reload_autofit',nargs=1,
                help='Reload existing autofit save files to gain time if a computation has crashed',
                default=False,type=bool)

ap.add_argument('-reload_fakes',nargs=1,
                help='Reload fake delchi array file to skip the fake computation if possible',
                default=False,type=bool)

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
ap.add_argument('-plot_multi_overlap',nargs=1,help='plot overlap between different epochs',default=True)

#in this case other epochs from other instruments are matched against the obs of this one
#useful to center epoch matching on a specific instrument
#off value is False
ap.add_argument('-multi_focus',nargs=1,help='restricts epoch matching to having a specific telescope',
                default='NuSTAR',type=str)

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

#note: Important to correctly assess the background of observations within the SAA
ap.add_argument('-extend_scorpeon_nxb_SAA',nargs=1,help='Extend scorpeon nxb_SAA range',default=True)

#ueseful when the SAA values/overshoots are high and thus the default parameter value is underestimated
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
                default=1000,type=int)

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
extend_scorpeon_nxb_SAA=args.extend_scorpeon_nxb_SAA
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
plot_multi_overlap=args.plot_multi_overlap
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

'''utility functions'''

#switching off matplotlib plot displays unless with plt.show()
plt.ioff()

def interval_extract(list):
    list = sorted(set(list))
    range_start = previous_number = list[0]

    for number in list[1:]:
        if number == previous_number + 1:
            previous_number = number
        else:
            yield [range_start, previous_number]
            range_start = previous_number = number
    yield [range_start, previous_number]

def pileup_val(pileup_line):

    '''
    returns the maximal pileup value if there is pile-up, and 0 if there isn't
    '''

    #the errors are given by default with a 3 sigma confidence level

    pattern_s_val=float(pileup_line.split('s: ')[1].split('   ')[0].split(' ')[0])
    pattern_s_err=float(pileup_line.split('s: ')[1].split('   ')[0].split(' ')[2])
    pattern_d_val=float(pileup_line.split('d: ')[1].split('   ')[0].split(' ')[0])
    pattern_d_err=float(pileup_line.split('d: ')[1].split('   ')[0].split(' ')[2])

    #no pileup means the s and d pattern values are compatible with 1
    #however for d the value can be much lower for faint sources, as such we only test positives values for d
    max_pileup_s=max(max(pattern_s_val-pattern_s_err-1,0),max(1-pattern_s_val-pattern_s_err,0))
    max_pileup_d=max(pattern_d_val-pattern_d_err-1,0)

    return max(max_pileup_s,max_pileup_d)

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

    if launch_cpd:
        #obtaining the xspec window id. It is important to call the variable xspec_id, since it is called by default in other functions
        #we set yLog as False to use ldata in the plot commands and avoid having log delchis
        xspec_id=Pset(xlog=True,ylog=False)
        if xspec_window!='auto':
            xspec_id=xspec_window
            Plot.xLog=True
    else:
        Pset(window=None,xlog=False,ylog=False)

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

#reducing the amount of data displayed in the terminal (doesn't affect the log file)
Xset.chatter=xchatter

#defining the standard number of fit iterations
Fit.nIterations=100

Fit.query=xspec_query

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

def pdf_summary(epoch_files,fit_ok=False,summary_epoch=None,e_sat_low_list=None,e_sat_high_list=None):

    #used to have specific energy limits for different instruments. can be modified later

    if sat_glob == 'multi':
        e_sat_low_indiv = np.repeat([None], len(epoch_files))
        e_sat_high_indiv = np.repeat([None], len(epoch_files))
        line_cont_ig_indiv = np.repeat([None], len(epoch_files))
        sat_indiv = np.repeat([None], len(epoch_files))
        for id_epoch, elem_file in enumerate(epoch_files):
            # fetching the instrument of the individual element
            # note that we replace the megumi xis0_xis3 files by the xis1 because the merged xis0_xis3 have no header
            #we also replace SUZAKU in caps by Suzaku to have a better time matching strings
            sat_indiv[id_epoch] = fits.open(elem_file.replace('xis0_xis3','xis1'))[1].header['TELESCOP']\
                .replace('SUZAKU','Suzaku')
            e_sat_low_indiv[id_epoch], e_sat_high_indiv[id_epoch], temp,line_cont_ig_indiv[id_epoch], = \
                line_e_ranges(sat_indiv[id_epoch])
    else:
        e_sat_low_val, e_sat_high_val, temp,line_cont_ig_val = line_e_ranges(sat_glob)
        e_sat_low_indiv = np.repeat(e_sat_low_val, len(epoch_files))
        e_sat_high_indiv = np.repeat(e_sat_high_val, len(epoch_files))
        line_cont_ig_indiv = np.repeat(line_cont_ig_val, len(epoch_files))
        sat_indiv=np.repeat(sat_glob,len(epoch_files))

    if e_sat_low_list is not None:
        e_sat_low_indiv=e_sat_low_list
    if e_sat_high_list is not None:
        e_sat_high_indiv=e_sat_high_list

    if sat_glob == 'multi':
        epoch_observ = [file_to_obs(elem_file, elem_telescope) for elem_file, elem_telescope in \
                        zip(epoch_files, sat_indiv)]
    else:
        epoch_observ = [file_to_obs(elem_file, sat_glob) for elem_file in epoch_files]

    '''PDF creation'''

    print('\nPreparing pdf summary for exposures ')
    print(epoch_observ)

    #fetching the SNRs
    epoch_SNR=[]
    for elem_observ in epoch_observ:
        if os.path.isfile(elem_observ+'_regex_results.txt'):
            with open(elem_observ+'_regex_results.txt','r') as regex_file:
                regex_lines=regex_file.readlines()
                epoch_SNR+=[float(regex_lines[3].split('\t')[1])]
        else:
            epoch_SNR+=['X']

    epoch_inf = [elem_observ.split('_') for elem_observ in epoch_observ]

    short_epoch_id= '_'.join(shorten_epoch([elem.split('_sp')[0] for elem in epoch_observ]))


    pdf=PDF(orientation="landscape")
    pdf.add_page()
    pdf.set_font('helvetica', 'B', max(10,16-(2*len(epoch_observ))//5))

    line_skip_len=max(6,10-2*len(epoch_observ)//5)

    pdf.cell(1,1,'Epoch name :'+short_epoch_id,align='C',center=True)
    pdf.ln(10)

    pdf.cell(1,1,'Spectra informations:\n',align='C',center=True)

    #line skip
    pdf.ln(10)

    if sat_glob=='XMM':
        #global flare lightcurve screen (computed in filter_evt)
        rate_name_list=[elem_observ.replace(elem_observ.split('_')[1],'rate'+elem_observ.split('_')[1]) for elem_observ in epoch_observ]

        rate_name_list=[rate[:rate.rfind('_')]+'_screen.png' for rate in rate_name_list]


    is_sp=[]
    is_cleanevt=[]
    filename_list=[]
    exposure_list=[]
    expmode_list=[]
    for i_obs,(elem_observ,elem_sat) in enumerate(zip(epoch_observ,sat_indiv)):

        #creating new pages regularly for many GTIs
        if i_obs%8==0 and i_obs!=0:
            pdf.add_page()

        if elem_sat=='XMM':
            if os.path.isfile(elem_observ+'_sp_src_grp_20.ds'):
                is_sp+=[True]
                is_cleanevt+=[True]
            elif os.path.isfile(elem_observ+'_evt_save.ds'):
                is_sp+=[False]
                is_cleanevt+=[True]
            else:
                is_sp+=[False]
                is_cleanevt+=[False]
        elif elem_sat in ['Chandra','NICER','Swift','SWIFT','NuSTAR']:
            is_sp+=[True]
            is_cleanevt+=[False]

            # if elem_sat=='NICER' and pre_reduced_NICER:
            #     filename_list+=[elem_observ]
            # else:

        elif elem_sat=='Suzaku':
            if megumi_files:
                is_sp += [True]


        with fits.open(epoch_files[i_obs]) as hdul:

            try:
                exposure_list+=[hdul[1].header['EXPOSURE']]
            except:
                try:
                    exposure_list+=[hdul[1].header['ONTIME']]
                except:
                    pass

            if elem_sat=='Chandra':
                epoch_grating=hdul[1].header['GRATING']
                expmode_list+=[hdul[1].header['DATAMODE']]
            else:
                expmode_list += [''] if (pre_reduced_NICER or 'DATAMODE' not in hdul[0].header.keys())\
                                else [hdul[0].header['DATAMODE']]

            if elem_sat in ['NICER','NuSTAR']:

                if elem_sat=='NICER' and pre_reduced_NICER:
                        pdf.cell(1,1,'Object: '+obj_name+' | Date: '+Time(hdul[1].header['MJDSTART'],format='mjd').isot+
                                 ' | Obsid: '+epoch_inf[i_obs][0],align='C',center=True)
                else:
                    start_obs_s = hdul[1].header['TSTART'] + (0 if elem_sat=='NuSTAR' else hdul[1].header['TIMEZERO'])
                    # saving for titles later
                    mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

                    obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

                    date_str=str(obs_start.isot)
                    pdf.cell(1,1,'Object: '+obj_name+' | Date: '+date_str+' | Obsid: '+epoch_inf[i_obs][0],
                          align='C',center=True)

            else:
                date_str=' ' if 'DATE-OBS' not in hdul[0].header.keys() else hdul[0].header['DATE-OBS'].split('T')[0]
                pdf.cell(1,1,'Object: '+obj_name+' | Date: '+date_str+' | Obsid: '+epoch_inf[i_obs][0],
                      align='C',center=True)

            pdf.ln(line_skip_len)

            if elem_sat=='XMM':
                pdf.cell(1,1,'exposure: '+epoch_inf[i_obs][2]+' | camera: '+epoch_inf[i_obs][1]+' | mode: '+epoch_inf[i_obs][3]+
                      ' | submode: '+hdul[0].header['SUBMODE']+' | clean exposure time: '+str(round(exposure_list[i_obs]))+
                      's',align='C',center=True)
            elif elem_sat=='Chandra':

                pdf.cell(1,1,'grating: '+epoch_grating+' | mode: '+expmode_list[0]+
                         ' clean exposure time: '+str(round(exposure_list[i_obs]))+'s',align='C',center=True)
            elif elem_sat in ['NICER','Suzaku','Swift','SWIFT','NuSTAR']:
                pdf.cell(1,1,'mode: '+expmode_list[0]+
                         ' clean exposure time: '+str(round(exposure_list[i_obs]))+'s',align='C',center=True)

            pdf.ln(2)

            #we only show the third line for XMM spectra with spectrum infos if there is an actual spectrum
            if epoch_SNR[i_obs]!='X':

                pdf.ln(8)

                grouping_str='SNR: '+str(round(epoch_SNR[i_obs],3))+' | Spectrum bin grouping: '+epoch_inf[i_obs][-1].split('.')[0]+' cts/bin | '
                try:
                    pileup_lines=fits.open(elem_observ+'_sp_src.ds')[0].header['PILE-UP'].split(',')
                    pdf.cell(1,1,grouping_str+'pile-up values:'+pileup_lines[-1][10:],align='C',center=True)
                except:
                    pdf.cell(1,1,grouping_str+'no pile-up values for this exposure',align='C',center=True)

                pdf.ln(2)
        pdf.ln(line_skip_len)

        #turned off for now
        # if flag_bg[i_obs]:
        #     pdf.cell(1,1,'FLAG : EMPTY BACKGROUND')


    '''Line detection infos'''
    if summary_epoch is None:

        #the replace avoid problems with using full chandra/NICER file names as epoch loggings in the summary files

        assert sat_glob!='multi', 'This is not ready yet'

        logged_epochs=[(elem.split('\t')[0] if sat_glob in ['NICER','Swift']\
                        else '_'.join([elem.split('\t')[0],elem.split('\t')[1].replace('_grp_opt','')\
                                      .replace('sp_grp_opt','').replace('.pha','').replace('.pi','')]))\
                       for elem in glob_summary_linedet]
        right_epoch_match=np.array(logged_epochs)==str(shorten_epoch(ravel_ragged(epoch_inf)))

        #storing the message
        result_epoch_val=np.array(glob_summary_linedet)[right_epoch_match][0].split('\t')[2]

        # #and repeating it for all the individual sp
        result_epoch=np.repeat(result_epoch_val,len(epoch_inf))

    else:
        result_epoch=summary_epoch

    def disp_broadband_data():

        '''
        display the different raw spectra in the epoch
        '''

        pdf.cell(1,1,'Broad band data ('+str(min(e_sat_low_indiv))+'-'+str(max(e_sat_high_indiv))+' keV)',align='C',center=True)
        sp_displayed=False

        if len(epoch_observ)==1 or sat_glob!='XMM':
            if os.path.isfile(outdir+'/'+epoch_observ[0]+'_screen_xspec_spectrum.png'):
                pdf.image(outdir+'/'+epoch_observ[0]+'_screen_xspec_spectrum.png',x=30,y=50,w=200)
                sp_displayed=True
        else:
            #displaying the spectra of all the exposures in the epochs that have one
            for i_obs in range(len(epoch_observ)):
                if os.path.isfile(outdir+'/'+epoch_observ[i_obs]+'_screen_xspec_spectrum.png'):
                    obs_cam=epoch_observ[i_obs].split('_')[1]
                    if obs_cam=='pn':
                        x_pos=0
                    elif obs_cam=='mos1':
                        x_pos=100
                    elif obs_cam=='mos2':
                        x_pos=200
                    else:
                        x_pos=50
                    pdf.image(outdir+'/'+epoch_observ[i_obs]+'_screen_xspec_spectrum.png',x=x_pos,y=50,
                              w=200 if obs_cam not in ['pn','mos1','mos2'] else 100)
                    sp_displayed=True
        if not sp_displayed:
            pdf.ln(30)
            pdf.cell(1,1,'No spectrum to display',align='C',center=True)

    if sum(is_sp)>0:

        pdf.cell(1,1,'Line detection summary:',align='C',center=True)
        pdf.ln(10)
        pdf.cell(1, 1, short_epoch_id + ': ' + result_epoch[0], align='C', center=True)

        # for i_elem,elem_result in enumerate(result_epoch):
        #     pdf.cell(1,1,epoch_observ[i_elem]+': '+elem_result,align='C',center=True)
        #     pdf.ln(10)

        pdf.add_page()

        if not fit_ok:
            #when there is no line analysis, we directly show the spectra
            pdf.ln(10)
            disp_broadband_data()

            if os.path.isfile(outdir+'/'+epoch_observ[0]+'_screen_xspec_broadband.png'):
                pdf.add_page()

        #fetching the order of the spectra in the multi-grp spectra (which we assume are ones which went through the linedet process)
        good_sp=[epoch_observ[i_obs] for i_obs in range(len(epoch_observ)) if list(result_epoch)[i_obs]=='Line detection complete.']

        def disp_multigrp(lines):
            def format_lines(lines):

                '''
                feat chatgpt, actually terrible I should have done this manually
                '''
                formatted_lines = []
                datagroup_number = 0
                in_datagroup = False

                for line in lines:
                    # Skip empty lines
                    if line.strip() == '' or line.strip()=='\n':
                        continue

                    # Check if the line contains "Data group"
                    data_group_match = re.match(r'\s*Data group:\s+(\d+)', line)
                    if data_group_match:
                        datagroup_number = int(data_group_match.group(1))
                        in_datagroup = True
                        formatted_lines.append('-' * 65+'\n')  # Add horizontal line
                        continue

                    #Putting a space before the first statistic line
                    if 'Fit statistic' in line:
                            formatted_lines.append('\n')

                    # Check if the line contains "Warning" and skip it
                    if "Warning: cstat statistic is only valid for Poisson data." in line:
                        continue
                    if "Source file is not Poisson" in line:
                        continue

                    # Add the Data group number as a vertical column
                    if in_datagroup and re.match(r'\s+\d+\s+\d+\s+', line):
                        line = f'{datagroup_number: >2}' + line

                    # Add the column title near the left of "par comp"
                    if "par  comp" in line:
                        line = ' Dg  ' + line

                    # Add the line to the formatted lines without line jumps
                    formatted_lines.append(line)

                return formatted_lines

            '''
            scans model lines for multiple data groups and displays only the first lines of each data group after the first
            (to show the constant factors)
            '''

            # removing the bg models

            model_sep_lines = np.argwhere(np.array([elem.startswith('_') for elem in lines])).T[0]
            if len(model_sep_lines) == 1:
                lines_nobg = lines
            else:
                lines_nobg = lines[:model_sep_lines[0]] + lines[model_sep_lines[-1] + 1:]

            lineid_grp_arr = np.argwhere(np.array(['Data group' in elem for elem in np.array(lines_nobg)])).T[0]

            # no need to do anything for less than two datagroups
            if len(lineid_grp_arr) < 2:
                return lines_nobg,1
            else:
                lines_cleaned = []
                # we display up to the second data group, then only non-standardly linked lines
                for i_grp, lineid_grp in enumerate(lineid_grp_arr):
                    if i_grp == 0:
                        i_begin = 0
                        i_end = lineid_grp_arr[i_grp + 1]
                        #adding the whole group
                        lines_cleaned += lines_nobg[i_begin:i_end]
                    else:

                        lines_cleaned += [lines_nobg[lineid_grp]]

                        #testing every line after that
                        i_begin = lineid_grp+1
                        #we can allow skipping the last line here because it will always be a fit display line
                        i_end = -1 if i_grp==len(lineid_grp_arr)-1 else lineid_grp_arr[i_grp + 1]

                        #computing the group size
                        grp_size=sum([False if len(elem.split())==0 else elem.split()[0].isdigit()\
                                      for elem in lines_nobg[i_begin:i_end]])

                        for i_line in range(i_begin,i_begin+grp_size):
                            elem_line=lines_nobg[i_line]
                            elem_line_status=elem_line.split()[-1]
                            elem_line_parnumber=int(elem_line.split()[0])

                            #comparing the link to the expected link value (aka degrouped parameter value)
                            elem_line_link_std=elem_line_status=='p'+str(1+(max(0,int(elem_line_parnumber-1)%grp_size)))

                            if not elem_line_link_std:
                                lines_cleaned+=[elem_line]

                # adding everything after the end of the model (besides the last line which is just a 'model not fit yet' line)
                lines_cleaned += lines_nobg[i_begin + lineid_grp_arr[1] - lineid_grp_arr[0]:-1]

                # formatting to change the position of the datagroup
                lines_formatted = format_lines(lines_cleaned)

                return lines_formatted,len(lineid_grp_arr)

        def display_fit(fit_type):

            if 'broadband' in fit_type:
                fit_title='broad band'
                fit_ener=str(min(e_sat_low_indiv))+'-'+str(max(e_sat_high_indiv))
            elif 'broadhid' in fit_type:
                fit_title='HID'
                fit_ener=str(min(e_sat_low_indiv))+'-'+str(max(e_sat_high_indiv)) if broad_HID_mode else '3.-10.'

            if 'linecont' in fit_type:
                #overwrites the broadband if
                fit_title='Line continuum'
                fit_ener=args.line_cont_range
            if 'autofit' in fit_type:
                fit_title='Autofit'
                fit_ener=args.line_cont_range
            if 'post_auto' in fit_type:
                fit_title+=' post autofit'
            if 'zoom' in fit_type:
                fit_title+=' zoom'

            image_id=outdir+'/'+epoch_observ[0]+'_screen_xspec_'+fit_type

            if os.path.isfile(image_id+'.png'):

                pdf.set_font('helvetica', 'B', 16)

                #selecting the image to be plotted
                image_path=image_id+'.png'
                pdf.ln(5)

                pdf.cell(1,-5,fit_title+' ('+fit_ener+' keV):',align='C',center=True)

                #no need currently since we have that inside the graph now
                # if fit_type=='broadband' and len(epoch_observ)>1 or sat_glob=='Chandra':
                #     #displaying the colors for the upcoming plots in the first fit displayed
                #     pdf.cell(1,10,'        '.join([xcolors_grp[i_good_sp]+': '+'_'.join(good_sp[i_good_sp].split('_')[1:3])\
                #                                    for i_good_sp in range(len(good_sp))]),align='C',center=True)

                pdf.image(image_path,x=0,y=50,w=150)

                #fetching the model unless in zoom mode where the model was displayed on the page before
                if 'zoom' not in fit_type:
                    #and getting the model lines from the saved file
                    with open(outdir+'/'+epoch_observ[0]+'_mod_'+fit_type+'.txt') as mod_txt:
                        fit_lines=mod_txt.readlines()

                        cleaned_lines,n_groups=disp_multigrp(fit_lines)

                        pdf.set_font('helvetica', 'B',min(8,11-len(cleaned_lines)//13))
                        pdf.multi_cell(150,2.6-(0.2*max(0,8-(min(8,11-len(cleaned_lines)//13)))),
                                       '\n'*max(0,int((30 if 'auto' in fit_type else 30)\
                                                              -1.5*len(cleaned_lines)**2/100))\
                                                          +''.join(cleaned_lines))

                    #in some cases due to the images between some fit displays there's no need to add a page
                    if 'linecont' not in fit_type:
                        pdf.add_page()
                    else:
                        #restrictingthe addition of a new page to very long models
                        if len(cleaned_lines)>35:
                            pdf.add_page()
                else:
                    pass

        display_fit('broadband_post_auto')
        display_fit('broadhid_post_auto')


        pdf.set_margins(0.5,0.5,0.5)

        display_fit('autofit')
        display_fit('autofit_zoom')

        pdf.set_margins(1.,1.,1.)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_autofit_components_plot_'+args.line_search_e.replace(' ','_')+'_'+\
                          args.line_search_norm.replace(' ','_')+'.png'):
            #combined autofit component plot
            pdf.image(outdir+'/'+epoch_observ[0]+'_autofit_components_plot_'+args.line_search_e.replace(' ','_')+'_'+\
                          args.line_search_norm.replace(' ','_')+'.png',x=1,w=280)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_autofit_line_comb_plot_'+args.line_search_e.replace(' ','_')+'_'+\
                          args.line_search_norm.replace(' ','_')+'.png'):
            #Combined plot
            pdf.image(outdir+'/'+epoch_observ[0]+'_autofit_line_comb_plot_'+args.line_search_e.replace(' ','_')+'_'+\
                      args.line_search_norm.replace(' ','_')+'.png',x=1,w=280)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_abslines_table.txt'):
            with open(outdir+'/'+epoch_observ[0]+'_abslines_table.txt','r') as table_file:
                table_html=table_file.readlines()
            pdf.add_page()
            #autofit absorption lines data table
            pdf.set_font('helvetica', 'B', 9)
            pdf.write_html(''.join(table_html))
            pdf.add_page()

        pdf.set_margins(0.5,0.5,0.5)

        display_fit('broadband')
        display_fit('broadhid')
        display_fit('broadband_linecont')

        pdf.set_margins(1.,1.,1.)

        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_cont_line_comb_plot_'+args.line_search_e.replace(' ','_')+'_'+\
                          args.line_search_norm.replace(' ','_')+'.png'):
            #Combined plot
            pdf.image(outdir+'/'+epoch_observ[0]+'_cont_line_comb_plot_'+args.line_search_e.replace(' ','_')+'_'+\
                      args.line_search_norm.replace(' ','_')+'.png',x=1,w=280)

            #not needed at the moment
            # pdf.image(outdir+'/'+exposid+'_line_cont_plot_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')\
            #           +'.png',x=1,w=280)
            # pdf.image(outdir+'/'+exposid+'_line_col_plot_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')\
            #           +'.png',x=1,w=280)


        if fit_ok:
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            disp_broadband_data()

    #displaying error messages for XMM epochs with no spectrum
    elif sat_glob=='XMM':

        #for the ones with no spectra, we have only one obs per epoch so not need to loop
        #displaying the reason the region computation failed if it did
        pdf.cell(1,1,'Region extraction summary:',align='C',center=True)
        pdf.ln(10)
        pdf.cell(1,1,[elem.split('\t')[2] for elem in glob_summary_reg \
                      if '_'.join([elem.split('\t')[0],elem.split('\t')[1]])==epoch_observ[0].replace('_auto','')][0],align='C',center=True)

        pdf.ln(10)

        #displaying the reason the spectrum computation failed
        pdf.cell(1,1,'Spectrum computation summary:',align='C',center=True)
        pdf.ln(10)
        pdf.cell(1,1,[elem.split('\t')[2] for elem in glob_summary_sp \
                      if '_'.join([elem.split('\t')[0],elem.split('\t')[1]])==epoch_observ[0].replace('_auto','')][0],align='C',center=True)


    shown_obsids_NICER=[]
    '''Data reduction displays'''

    if sat_glob=='multi':
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(1, 10, 'Epoch matching', align='C', center=True)
        pdf.ln(10)
        try:
            pdf.image(outdir +'/'+ short_epoch_id+'_multi_matching.pdf', x=20, y=30, w=250)
        except:
            pass

    for i_obs,(elem_epoch,elem_sat) in enumerate(zip(epoch_observ,sat_indiv)):

        if elem_sat=='NICER':

            #adding the global flares curve for each obsid
            elem_obsid=elem_epoch.split('-')[0]

            if elem_obsid not in shown_obsids_NICER:
                shown_obsids_NICER+=[elem_obsid]
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(1,10,'Orbits for obsid '+elem_obsid,align='C',center=True)
                pdf.ln(10)
                try:
                    pdf.image(elem_obsid +'-global_flares.png',x=20,y=30,w=250)
                except:
                    pass

            #and adding the individual GTI's flare and lightcurves
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1,10,'GTIS and lightcurves for gti '+elem_epoch,align='C',center=True)
            pdf.ln(10)

            #recognizing time-resolved spectra
            elem_orbit=elem_epoch.split('S')[0].split('M')[0].split('F')[0].split('I')[0]

            try:
                pdf.image(elem_orbit+ '_flares.png',x=2,y=70,w=140)

                pdf.image(elem_epoch + '_lc_3-10_bin_' + NICER_lc_binning + '.png', x=150, y=30, w=70)
                pdf.image(elem_epoch + '_hr_3-10_bin_' + NICER_lc_binning + '.png', x=220, y=30, w=70)

                pdf.image(elem_epoch + '_lc_3-6_bin_' + NICER_lc_binning + '.png', x=150, y=120, w=70)
                pdf.image(elem_epoch + '_lc_6-10_bin_' + NICER_lc_binning + '.png', x=220, y=120, w=70)

            except:
                pass

        if elem_sat=='NuSTAR':


            elem_epoch_nogti=elem_epoch.split('-')[0]

            if 'A0' in elem_epoch.split('_')[0]:
                nudet='FPMA'
            elif 'B0' in elem_epoch.split('_')[0]:
                nudet='FPMB'

            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1, 10, nudet+' Data reduction for observation ' + elem_epoch_nogti, align='C', center=True)
            pdf.ln(10)
            pdf.ln(10)
            pdf.cell(1, 30, 'Region definition                                               ' +
                     'BBG selection', align='C', center=True)


            pdf.image(elem_epoch_nogti + '_auto_reg_screen.png', x=2, y=50, w=140)

            pdf.image(elem_epoch_nogti + '_vis_CCD_1_crop.png', x=150, y=50, w=70)
            pdf.image(elem_epoch_nogti + '_vis_CCD_2_mask.png', x=220, y=50, w=70)
            pdf.image(elem_epoch_nogti + '_vis_CCD_3_cut.png', x=150, y=120, w=70)
            pdf.image(elem_epoch_nogti + '_vis_CCD_4_bg.png', x=220, y=120, w=70)

            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1, 10, nudet+' Obs Lightcurves for observation ' + elem_epoch, align='C', center=True)

            elem_epoch_obsid = elem_epoch.split('-')[0][:-3]

            pdf.image(elem_epoch_obsid + '_'+nudet+'_lc_screen_3_79_bin_100.png', x=2, y=50, w=140)
            pdf.image(elem_epoch_obsid + '_'+nudet+'_hr_screen_10-50_3-10_bin_100.png', x=150, y=50, w=70)
            pdf.image(elem_epoch_obsid + '_'+nudet+'_lc_screen_3_10_bin_100.png', x=220, y=50, w=70)
            pdf.image(elem_epoch_obsid + '_'+nudet+'_lc_screen_10_50_bin_100.png', x=220, y=120, w=70)

            #adding one page for individual orbits
            if '-' in elem_epoch:
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(1, 10, nudet+' GTI Lightcurves for observation ' + elem_epoch, align='C', center=True)

                elem_epoch_gti=elem_epoch.split('-')[1]

                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_lc_screen_3_79_bin_10.png', x=2, y=50, w=140)
                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_hr_screen_10-50_3-10_bin_10.png', x=150, y=50, w=70)
                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_lc_screen_3_10_bin_10.png', x=220, y=50, w=70)
                pdf.image(elem_epoch_obsid + '-' + elem_epoch_gti + '_'+nudet+'_lc_screen_10_50_bin_10.png', x=220, y=120, w=70)


        if elem_sat=='XMM':
                if is_sp[i_obs]:
                    pdf.add_page()
                    pdf.set_font('helvetica', 'B', 16)
                    pdf.cell(1,10,'Data reduction for observation '+elem_epoch,align='C',center=True)
                    pdf.ln(10)
                    pdf.cell(1,30,'Initial region definition                                        '+
                              'Post pile-up excision (if any) region definition',align='C',center=True)
                    pdf.image(elem_epoch+'_reg_screen.png',x=2,y=50,w=140)

                    if os.path.isfile(elem_epoch+'_reg_excised_screen.png'):
                        pdf.image(elem_epoch+'_reg_excised_screen.png',x=155,y=50,w=140)

                    if expmode_list[i_obs]=='IMAGING':
                        pdf.add_page()
                        pdf.image(elem_epoch+'_opti_screen.png',x=1,w=280)

                        #adding a page for the post-pileup computation if there is one
                        if os.path.isfile(elem_epoch+'_opti_excised_screen.png'):
                            pdf.add_page()
                            pdf.image(elem_epoch+'_opti_excised_screen.png',x=1,w=280)

                    elif expmode_list[i_obs]=='TIMING' or expmode_list[i_obs]=='BURST':
                        pdf.add_page()
                        pdf.cell(1,30,'SNR evolution for different source regions, first iteration',align='C',center=True)
                        pdf.image(elem_epoch+'_opti_screen.png',x=10,y=50,w=140)

                        #adding a page for the post-pileup computation if there is one
                        if os.path.isfile(elem_epoch+'_opti_excised_screen.png'):
                            pdf.image(elem_epoch+'_opti_excised_screen.png',x=150,y=50,w=140)

                elif is_cleanevt[i_obs]:
                    pdf.set_font('helvetica', 'B', 16)
                    if expmode_list[i_obs]=='IMAGING':
                        pdf.add_page()
                        pdf.cell(1,30,'Raw image                     '+'              position catalog cropping zone          '+
                                  '            cropped region zoom',align='C',center=True)
                        try:
                            pdf.image(elem_epoch+'_img_screen.png',x=1,y=70,w=90)
                            pdf.image(elem_epoch+'_catal_reg_screen.png',x=100,y=70,w=90)
                            pdf.image(elem_epoch+'_catal_crop_screen.png',x=190,y=65,w=120)
                        except:
                            pass
                    if expmode_list[i_obs]=='TIMING' or expmode_list[i_obs]=='BURST':
                        pdf.add_page()
                        pdf.cell(1,30,'Raw image',align='C',center=True)
                        try:
                            pdf.image(elem_epoch+'_img_screen.png',x=70,y=50,w=150)
                        except:
                            pass

                '''flare curves'''

                pdf.add_page()
                try:
                    #source/bg flare "first iteration" lightcurves (no flare gti cut) with flares zones highlighted
                    pdf.image(elem_observ+'_lc_comb_snr_screen.png',x=10,y=10,w=130)
                except:
                    pass
                #corresponding flare rate curve and rate limit
                try:
                    pdf.image(rate_name_list[i_obs],x=150,y=10,w=130)
                except:
                    pass

                #source/bg flare "second iteration" lightcurve
                try:
                     pdf.image(elem_observ+'_lc_comb_snr_excised_screen.png',x=10,y=105,w=130)
                except:
                    pass

                #broad band source/bg lightcurve
                try:
                    pdf.image(elem_observ+'_lc_comb_broad_screen.png',x=150,y=105,w=130)
                except:
                    pass

    #naming differently for aborted and unaborted analysis
    if not fit_ok:
        pdf.output(outdir + '/' + ('_'.join(shorten_epoch(epoch_observ))) + '_aborted_recap.pdf')
    else:
        pdf.output(outdir+'/'+('_'.join(shorten_epoch(epoch_observ)))+'_recap.pdf')

def line_e_ranges(sat,det=None):
    '''
    Determines the energy range allowed, as well as the ignore energies for a given satellite

    DO NOT USE INTS else it will be taken as channels instead of energies
    ignore_bands are bands that will be ignored on top of the rest, in ALL bands
    '''
    ignore_bands=None

    if sat=='NuSTAR':
        e_sat_low=8. if (sat_glob=='multi' and diff_bands_NuSTAR_NICER) else 4.
        e_sat_high=79.

    if sat.upper() in ['XMM', 'NICER', 'SWIFT']:

        if sat == 'NICER':
            e_sat_low = 0.3 if (sat_glob=='multi' and diff_bands_NuSTAR_NICER) else low_E_NICER
        else:
            e_sat_low = 0.3

        if sat.upper() in ['XMM', 'SWIFT']:
            if sat == 'XMM':
                e_sat_low = 2.

            e_sat_high = 10.
        else:
            if sat == 'NICER':
                e_sat_high = 10.
            else:
                e_sat_high = 10.

    elif sat == 'Suzaku':

        if det==None:
            e_sat_low=1.9
            e_sat_high=40.

            ignore_bands=suzaku_xis_ignore
        else:

            assert det in ['PIN','XIS'],'Detector argument necessary to choose energy ranges for Suzaku'

            e_sat_low =suzaku_xis_range[0] if det=='XIS' else suzaku_pin_range[0]
            e_sat_high = suzaku_xis_range[1] if det=='XIS' else suzaku_pin_range[1]

            # note: we don't care about ignoring these with pin since pin doesn't go that low
            ignore_bands = suzaku_xis_ignore

    elif sat == 'Chandra':
        e_sat_low = 1.5
        e_sat_high = 10.

    '''
    computing the line ignore values, which we cap from the lower and upper bound of the global energy ranges to avoid issues 
    we also avoid getting upper bounds lower than the lower bounds because xspec reads it in reverse and still ignores the band you want to keep
    ####should eventually be expanded to include the energies of each band as for the lower bound they are higher and we could have the opposite issue with re-noticing low energies
    '''

    if line_cont_ig_arg == 'iron':

        if sat in ['XMM', 'Chandra', 'NICER', 'Swift', 'SWIFT', 'Suzaku', 'NuSTAR']:

            line_cont_ig = ''

            if e_sat_low>6:
                #not ignoring this band for high-E only NuSTAR fits
                line_cont_ig=''
            else:
                if e_sat_high > 6.5:

                    line_cont_ig += '6.5-' + str(min(7.1, e_sat_high))

                    if e_sat_high > 7.7:
                        line_cont_ig += ',7.7-' + str(min(8.3, e_sat_high))
                else:
                    # failsafe in case the e_sat_high is too low, we ignore the very first channel of the spectrum
                    # which will be ignored anyway
                    line_cont_ig = str(1)

        else:
            line_cont_ig = '6.-8.'
    else:
        line_cont_ig = ''

    return e_sat_low,e_sat_high,ignore_bands,line_cont_ig

def reload_sp(baseload_path,keyword_skip=None,write_baseload=True,newbl_keyword='skip'):
    '''
    Reloads a baseload (with a list of spectra) but excluding files with a specific keyword in their names
    works by parsing the baseload and rewriting a new version without the corresponding line before loading them

    -baseload_path: base data loading file

    -keywod_skip: basic filter for the data file names to know which files to filter out

    -write_baseload: delete the baseload or not after loading the data

    -newbl_keyword: name for the new baseload file
    '''

    with open(baseload_path) as f_baseload:
        baseload_lines=f_baseload.readlines()

    new_baseload_lines=[]
    for i_line in range(len(baseload_lines)):
        if baseload_lines[i_line].startswith('data ') and keyword_skip in baseload_lines[i_line]:
            i_line+=1
            #skipping this line and all lines describing the sp below
            while baseload_lines[i_line].split()[0] in ['response','rmf','arf']:
                i_line+=1

        else:
            new_baseload_lines+=[baseload_lines[i_line]]

    #writing the lines in a new file
    new_baseload_path=baseload_path.replace('.xcm','_'+newbl_keyword+'.xcm')
    with open(new_baseload_path,'w+') as f_baseload_skip:
        f_baseload_skip.writelines(new_baseload_lines)

    Xset.restore(new_baseload_path)

    if not write_baseload:
        os.remove(new_baseload_path)

def line_detect(epoch_id):

    '''
    line detection for a single epoch
    we use the index as an argument to fill the chi array
    '''

    def html_table_maker():

        def strmaker(value_arr,is_overlap=False):

            '''
            wrapper for making a string of the line abs values

            set is_shift to true for energy/blueshift values, for which 0 values or low uncertainties equal to the value
            are sign of being pegged to the blueshift limit
            '''

            #the first case is for eqwidths and blueshifts (argument is an array with the uncertainties)
            if type(value_arr)==np.ndarray:
                #If the value array is entirely empty, it means the line is not detected and thus we put a different string
                if len(np.nonzero(value_arr)[0])==0:
                    newstr='/'
                else:
                    #maybe the is_shift test needs to be put back
                    if type(value_arr[1])==str:
                        #we do not show uncertainties for the linked parameters since it is just a repeat
                        newstr=str(round(value_arr[0],2))

                    else:
                        #to get a more logical display in cases where the error bounds are out of the interval, we test the
                        #sign of the errors to write the error range differently
                        str_minus=' -' if str(round(value_arr[1],1))[0]!='-' else' +'
                        str_plus=' +' if str(round(value_arr[1],1))[0]!='-' else' '
                        newstr=str(round(value_arr[0],1))+str_minus+str(abs(round(value_arr[1],1)))+\
                                str_plus+str(round(value_arr[2],1))

            #the second case is for the significance, delchis and the eqw upper limit, which are floats
            else:
                #same empty test except for overlap values which can go to zero
                if value_arr==0:
                    if not is_overlap:
                        newstr='/'
                    else:
                        return '0'
                else:
                    #the significance is always lower than 1
                    if value_arr<=1 and not is_overlap:
                        newstr=(str(round(100*value_arr,len(str(nfakes)))) if value_arr!=1 else '>'+str((1-1/nfakes)*100))+'%'
                    #and the delchis should always be higher than 1 else the component would have been deleted
                    else:
                        newstr=str(round(value_arr,2))

            return newstr

        #emission lines to be added
        '''
        </tr>
          <tr>
            <td>emission</td>
            <td>Fe Ka Neutral</td>
            <td>6.40</td>
            <td>no</td>
            <td>/</td>
            <td>'''+''''strmaker(abslines_ener[0])'''+'''</td>
            <td>'''+'''strmaker(abslines_bshift[0])'''+'''</td>
            <td>/</td>
          </tr>
          <tr>
            <td>emission</td>
            <td>Fe Kb Neutral</td>
            <td>7.06</td>
            <td>no</td>
            <td>/</td>
            <td>'''+'''strmaker(abslines_ener[1])'''+'''</td>
            <td>'''+'''strmaker(abslines_bshift[1])'''+'''</td>
            <td>/</td>
          </tr>
        '''

        html_summary=\
        '''
        <table>
        <thead>
          <tr>
            <th width="9%">Line</th>
            <th width="9%">rest energy</th>
            <th width="9%">em overlap</th>
            <th width="14%">line flux</th>
            <th width="14%">blueshift</th>
            <th width="8%">3sig. distinct</th>
            <th width="9%">width</th>
            <th width="9%">EW</th>
            <th width="9%">EW '''+str(sign_threshold)+''' UL</th>
            <th width="5%">MC sign.</th>
            <th width="5%">delstat</th>
          </tr>
        </thead>
        <tbody>
        <tr>
            <td></td>
            <td>keV</td>
            <td></td>
            <td>1e-12 erg/s/cm²</td>
            <td>km/s</td>
            <td></td>
            <td>eV (+-3sigma)</td>
            <td>eV</td>
            <td>eV</td>
            <td></td>
            <td></td>

        </tr>
        '''

        for i_line,line_name in enumerate([elem for elem in lines_e_dict.keys() if 'em' not in elem]):
            html_summary+='''
          <tr>
            <td>'''+line_name+'''</td>
            <td>'''+str(lines_e_dict[line_name][0])+'''</td>
            <td>'''+('/' if strmaker(abslines_sign[i_line])=='/' else\
                strmaker(abslines_em_overlap[i_line],is_overlap=True))+'''</td>
            <td>'''+strmaker(abslines_flux[i_line]*1e12)+'''</td>
            <td>'''+strmaker(abslines_bshift[i_line])+'''</td>
            <td>'''+str('/' if abslines_bshift_distinct[i_line] is None else abslines_bshift_distinct[i_line])+'''</td>
            <td>'''+strmaker(abslines_width[i_line]*1e3)+'''</td>
            <td>'''+strmaker(abslines_eqw[i_line])+'''</td>
            <td>'''+strmaker(abslines_eqw_upper[i_line])+'''</td>
            <td>'''+strmaker(abslines_sign[i_line])+'''</td>
            <td>'''+strmaker(abslines_delchi[i_line])+'''</td>
          </tr>
          '''
        html_summary+='''
        </tbody>
        </table>
        '''
        return html_summary

    epoch_files=epoch_list[epoch_id]

    #this variable will store the final message for each spectra
    epoch_result=np.array([None]*len(epoch_files))
    def fill_result(string,result_array=epoch_result):

        '''
        small wrapper to fill the non defined epoch results with a string
        '''

        #defining a copy of the result array to fill it
        result_arr=np.array(result_array)
        for i in range(len(result_arr)):
            if result_arr[i] is None:
                result_arr[i]=string

        return result_arr

    #deprecated
    obs_grating=False

    #Switching fit to C-stat
    Fit.statMethod=fitstat

    #this variable will serve for custom energy changes between different datagroups
    epoch_dets=[]

    #skipping observation if asked
    if sat_glob=='Chandra' and skip_nongrating and not obs_grating:
        return None

    #useful for later
    spec_inf=[elem_sp.split('_') for elem_sp in epoch_files]

    if sat_glob=='multi':
        sat_indiv_init = np.repeat([None], len(epoch_files))

        for id_epoch, elem_file in enumerate(epoch_files):
            # fetching the instrument of the individual element
            # note that we replace the megumi xis0_xis3 files by the xis1 because the merged xis0_xis3 have no header
            # we also replace SUZAKU in caps by Suzaku to have a better time matching strings
            sat_indiv_init[id_epoch] = fits.open(elem_file.replace('xis0_xis3', 'xis1'))[1].header['TELESCOP'] \
                .replace('SUZAKU', 'Suzaku')
    else:
        sat_indiv_init = np.repeat([sat_glob], len(epoch_files))


    #Step 0 is to readjust the response and bg file names if necessary (i.e. files got renamed)
    if h_update:
        for i_sp,(elem_sp,elem_sat) in enumerate(zip(epoch_files,sat_indiv_init)):

            file_obsid=elem_sp.split('_')[0]

            with fits.open(elem_sp,mode='update') as hdul:
                if  elem_sat=='XMM':
                    hdul[1].header['BACKFILE']=elem_sp.split('_sp')[0]+'_sp_bg.ds'
                    hdul[1].header['RESPFILE']=elem_sp.split('_sp')[0]+'.rmf'
                    hdul[1].header['ANCRFILE']=elem_sp.split('_sp')[0]+'.arf'
                    #saving changes
                    hdul.flush()

                if elem_sat=='NICER':
                    hdul[1].header['RESPFILE']=elem_sp.split('_sp')[0]+'.rmf'
                    hdul[1].header['ANCRFILE']=elem_sp.split('_sp')[0]+'.arf'
                    #saving changes
                    hdul.flush()

                if elem_sat=='Suzaku':
                    if megumi_files:
                        #update for pin files
                        if 'PIN' in hdul[1].header['DETNAM']:
                            epoch_dets+=['PIN']
                            #for megumi files we always update the header
                            pin_rspfile=glob.glob(file_obsid+'_ae_hxd_**.rsp')[0]
                            hdul[1].header['RESPFILE']=pin_rspfile
                            hdul[1].header['BACKFILE']=elem_sp.replace('src_dtcor_grp_opt','nxb_cxb')

                        elif 'XIS' in hdul[1].header['INSTRUME'] or '_xis' in  elem_sp:
                            epoch_dets+=['XIS']
                            hdul[1].header['RESPFILE']=elem_sp.replace('src_grp_opt.pha','rsp.rmf')
                            hdul[1].header['BACKFILE']=elem_sp.replace('src_grp_opt','bgd')
                else:
                    epoch_dets+=[None]

                #saving changes
                hdul.flush()

    '''Energy bands, ignores, and setup'''

    #used to have specific energy limits for different instruments. can be modified later

    e_sat_low_indiv_init = np.repeat([None], len(epoch_files))
    e_sat_high_indiv_init = np.repeat([None], len(epoch_files))
    ignore_bands_indiv_init = np.repeat([None], len(epoch_files))
    line_cont_ig_indiv_init = np.repeat([None], len(epoch_files))

    if sat_glob != 'multi':
        sat_indiv_init = np.repeat([sat_glob], len(epoch_files))

    for id_epoch, elem_file in enumerate(epoch_files):
        e_sat_low_indiv_init[id_epoch], e_sat_high_indiv_init[id_epoch], ignore_bands_indiv_init[id_epoch],\
            line_cont_ig_indiv_init[id_epoch] = line_e_ranges(sat_indiv_init[id_epoch],epoch_dets[id_epoch])

    if sat_glob == 'multi':
        epoch_observ = [file_to_obs(elem_file, elem_telescope) for elem_file, elem_telescope in \
                        zip(epoch_files, sat_indiv_init)]
    else:
        epoch_observ = [file_to_obs(elem_file, sat_glob) for elem_file in epoch_files]

    Xset.logChatter=10

    print('\nStarting line detection for files ')
    print(epoch_files)

    #reset the xspec config
    reset()

    '''Setting up a log file and testing the properties of each spectra'''

    curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log'+
                ('_pdf' if pdf_only else '_ul' if line_ul_only\
                else '_reload' if reload_autofit and os.path.isfile(outdir+'/'+epoch_observ[0]+'_fitmod_autofit.pkl')\
                else '')+'.log')
    #ensuring the log information gets in the correct place in the log file by forcing line to line buffering
    curr_logfile_write.reconfigure(line_buffering=True)

    curr_logfile=open(curr_logfile_write.name,'r')

    # curr_logfile_linedet=open(outdir+'/'+epoch_observ[0]+'_linedet_log.log','w+')

    def print_xlog(string,logfile_write=curr_logfile_write):

        '''
        prints and logs info in the xspec log file, and flushed to ensure the logs are printed before the next xspec print
        Different log file from the main xspec one to (hopefully) avoid issues
        '''
        print(string)
        logfile_write.write(time.asctime()+'\n')
        logfile_write.write(string)
        #adding a line for lisibility
        logfile_write.write('\n')
        logfile_write.flush()

    #list containing the epoch files rejected by the test
    epoch_files_good=[]
    sat_indiv_good=[]
    e_sat_low_indiv=[]
    e_sat_high_indiv=[]
    ignore_bands_indiv=[]
    line_cont_ig_indiv=[]
    epoch_dets_good=[]
    '''
    Pile-up and bg tests for XMM spectra
    '''
    for i_sp, (elem_sp,elem_sat) in enumerate(zip(epoch_files,sat_indiv_init)):

        if elem_sat=='XMM':

            AllData.clear()

            bg_off_flag=False
            try:
                curr_spec=Spectrum(elem_sp)
            except:
                try:
                    curr_spec=Spectrum(elem_sp,backfile=None)
                    print_xlog('\nLoaded the spectrum '+elem_sp+' with no background')
                    bg_off_flag=True
                except:
                    print_xlog("\nCouldn't load the spectrum "+elem_sp+"  with Xspec. Negative exposure time can cause this. Skipping the spectrum...")
                    epoch_result[i_sp]="Couldn't load the spectrum with Xspec."
                    continue

            #this is useful for the check plots
            Plot.background=True
            Plot.add=True

            #saving a screen of the spectrum for verification purposes
            AllData.ignore('**-'+str(e_sat_low_indiv_init[i_sp])+' '+str(e_sat_high_indiv_init[i_sp])+'-**')

            #checking if the spectrum is empty after ignoring outside of the broad interval
            if curr_spec.rate[0]==0:
                    print_xlog("\nSpectrum "+elem_sp+" empty in the ["+str(e_sat_low_indiv_init[i_sp])+"-"+str(e_sat_high_indiv_init[i_sp])+"] keV range. Skipping the spectrum...")
                    epoch_result[i_sp]="Spectrum empty in the ["+str(e_sat_low_indiv_init[i_sp])+"-"+str(e_sat_high_indiv_init[i_sp])+"]keV range."
                    continue
            else:
                Plot_screen("ldata",outdir+'/'+epoch_observ[i_sp]+"_screen_xspec_spectrum")
            AllData.notice('all')

            '''Various Checks'''

            #pile-up test
            #we use the ungrouped spectra for the headers since part of the header is often changed during the regrouping
            try :
                pileup_lines=fits.open(epoch_observ[i_sp]+'_sp_src.ds')[0].header['PILE-UP'].split(',')
                pileup_value=pileup_val(pileup_lines[-1])
                if pileup_value>pileup_lim:
                    print_xlog('\nPile-up value for spectrum '+elem_sp+' above the given limit of '+str(round(100*pileup_lim,1))+
                          '%. Skipping the spectrum...')
                    epoch_result[i_sp]='Pile-up above the given limit ('+str(round(100*pileup_lim,1))+'%).'
                    continue
                else:
                    print_xlog('\nPile-up value for spectrum '+elem_sp+' OK.')

            except:
                print_xlog('\nNo pile-up information available for spectrum '+elem_sp)
                pileup_value=-1
                if pileup_missing==False:
                    print_xlog('\nSkipping the spectrum...')
                    epoch_result[i_sp]='No pile-up info available'
                    continue

            '''Unloading Imaging backgrounds if they are too bright compared to blank field standards'''

            with fits.open(elem_sp) as hdul:
                curr_expmode=hdul[0].header['DATAMODE']
                curr_cam=hdul[1].header['INSTRUME'][1:].swapcase()

            if curr_expmode=='IMAGING' and bg_off_flag==False:

                AllData.ignore('**-'+str(e_sat_low_indiv_init[i_sp])+' '+str(e_sat_high_indiv_init[i_sp])+'.-**')
                #for that we must fetch the size of the background
                with open(epoch_observ[i_sp]+'_reg.reg') as regfile:
                    bg_rad=float(regfile.readlines()[-1].split('")')[0].split(',')[-1])

                #these values were computed by summing (and renormalizing) the graph rates in https://www.cosmos.esa.int/web/xmm-newton/bs-countrate
                #for standard patterns and thin filter opening
                if curr_cam=='pn':
                    bg_blank=1.02e-7
                elif curr_cam in ['mos1','mos2']:
                    bg_blank=1.08e-8

                #now we can compare to the standard rates
                if curr_spec.rate[1]/(bg_rad)**2>max_bg_imaging*bg_blank:
                    print_xlog('\nIntense background detected, probably contaminated by the source. Unloading it to avoid affecting the source...')
                    bg_off_flag=True

            '''unloading timing backgrounds'''

            if (curr_expmode in ['TIMING','BURST'] and skipbg_timing):
                bg_off_flag=True

            '''
            creating a duplicate spectrum file without the background so as to load the spectra natively without background 
            (the goal here is to avoid the crashes that sometimes happen when loading with a background)
            '''

            if os.path.isfile(elem_sp.replace('.ds','_bgtested.ds')):
                os.remove(elem_sp.replace('.ds','_bgtested.ds'))
            with fits.open(elem_sp) as hdul:
                if bg_off_flag==True:
                        hdul[1].header['BACKFILE']=''
                hdul.writeto(elem_sp.replace('.ds','_bgtested.ds'))

            #SNR limit (only tested when the bg is kept)
            if bg_off_flag==False:
                with open(epoch_observ[i_sp]+'_regex_results.txt','r') as regex_file:
                    regex_lines=regex_file.readlines()
                    curr_SNR=float(regex_lines[3].split('\t')[1])
                if curr_SNR<SNR_min:
                    print_xlog('\nSpectrum  '+elem_sp+' Signal to Noise Ratio below the limit. Skipping the spectrum ...')
                    epoch_result[i_sp]='Spectrum SNR below the limit ('+str(SNR_min)+')'
                    continue

            #saving the spectra if it passed all the test
            epoch_files_good+=[elem_sp]
            sat_indiv_good+=[elem_sat]
            e_sat_low_indiv+=[e_sat_low_indiv_init[i_sp]]
            e_sat_high_indiv+= [e_sat_high_indiv_init[i_sp]]
            ignore_bands_indiv+= [ignore_bands_indiv_init[i_sp]]
            line_cont_ig_indiv+= [line_cont_ig_indiv_init[i_sp]]
            epoch_dets_good+=[epoch_dets[i_sp]]
        else:
            epoch_files_good+=[elem_sp]
            sat_indiv_good+=[elem_sat]
            e_sat_low_indiv+=[e_sat_low_indiv_init[i_sp]]
            e_sat_high_indiv+= [e_sat_high_indiv_init[i_sp]]
            ignore_bands_indiv+= [ignore_bands_indiv_init[i_sp]]
            line_cont_ig_indiv+= [line_cont_ig_indiv_init[i_sp]]
            epoch_dets_good+=[epoch_dets[i_sp]]

        #testing if all spectra have been taken off
        if len(epoch_files_good)==0:
            return epoch_result


    epoch_files_good = np.array(epoch_files_good)
    sat_indiv_good = np.array(sat_indiv_good)
    e_sat_low_indiv = np.array(e_sat_low_indiv)
    e_sat_high_indiv = np.array(e_sat_high_indiv)
    ignore_bands_indiv = np.array(ignore_bands_indiv,dtype=object)
    line_cont_ig_indiv = np.array(line_cont_ig_indiv)
    epoch_dets_good = np.array(epoch_dets_good)

    '''
    Data load
    '''

    #creating the file load str in the xspec format
    data_load_str=''

    for i_sp,(elem_sp,elem_sat) in enumerate(zip(epoch_files_good,sat_indiv_good)):

        index_str=str(i_sp+1)+':'+str(i_sp+1)
        data_load_str+=index_str

        if elem_sat=='XMM':
            data_load_str+=' '+elem_sp.replace('.ds','_bgtested.ds')+' '
        if elem_sat=='Chandra':
            if 'heg_1' in elem_sp and restrict_order:
                data_load_str=data_load_str[:-len(index_str)]
                continue
            data_load_str+=' '+elem_sp+' '
        if elem_sat in ['NICER','Suzaku','Swift','SWIFT','NuSTAR']:
            data_load_str+=' '+elem_sp+' '

    AllData(data_load_str)

    #updating individual spectra bg, rmf arf if needed
    scorpeon_list_indiv=np.repeat(None,len(epoch_files_good))
    for i_sp,(elem_sp,elem_sat) in enumerate(zip(epoch_files_good,sat_indiv_good)):

        if elem_sat=='Chandra':
            if 'heg_1' in elem_sp and restrict_order:
                continue
            AllData(i_sp+1).response.arf=elem_sp.replace('_grp_opt.pha','.arf')

        if elem_sat =='NICER':
                if NICER_bkg=='scorpeon_mod':

                    #loading the background and storing the bg python path in xscorpeon
                    #note that here we assume that all files have a valid scorpeon background
                    scorpeon_list_indiv[i_sp]=elem_sp.replace('_sp_grp_opt.pha','_bg.py')
        if elem_sat in ['Swift','SWIFT']:

            AllData(i_sp+1).response.arf=epoch_observ[0].replace('source','')+'.arf'
            AllData(i_sp+1).background=epoch_observ[0].replace('source','')+('back.pi')

    if 'NICER' in sat_indiv_good:
        xscorpeon.load(scorpeon_list_indiv,frozen=True)

    '''resetting the energy bands if needed'''
    if not force_ener_bounds:
        hid_cont_range[1] = max(e_sat_high_indiv)

        #here we just test for the snr and create the line searc hspace, this will re-adjusted esat_high later
        line_cont_range[1] = min(np.array(args.line_cont_range.split(' ')).astype(float)[1],max(e_sat_high_indiv))

    # note: we add half a step to get rid of rounding problems and have the correct steps
    line_search_e_space = np.arange(line_search_e[0], line_search_e[1] + line_search_e[2] / 2, line_search_e[2])
    # this one is here to avoid adding one point if incorrect roundings create problem
    line_search_e_space = line_search_e_space[line_search_e_space <= line_search_e[1]]

    #outdated because now included in the line_e_ranges
    # for i_obs,elem_sat in enumerate(sat_indiv_good):
    #     #ad hoc way to only consider the suzaku detectors
    #     i_det_suzaku=0
    #     if elem_sat == 'Suzaku':
    #         e_sat_high_indiv[i_obs]=40. if epoch_dets[i_det_suzaku]=='PIN' else 9.
    #         e_sat_low_indiv[i_obs]=12. if epoch_dets[i_det_suzaku]=='PIN' else 1.9
    #         #test whether this works
    #         i_det_suzaku+=1

    #in preparation for the raw counts test
    if sat_glob=='Suzaku':
            line_cont_range[1]=9.

    if max(e_sat_high_indiv)>12:
        Plot.xLog=True
    else:
        Plot.xLog=False

    if 'NuSTAR' in sat_indiv_good or 'SWIFT' in sat_indiv_good:
        Plot.background = True

    #Putting the right energy bands and screening the spectra (note that we don't ignore the ignore bands here on purpose)
    ignore_data_indiv(e_sat_low_indiv,e_sat_high_indiv)

    Plot_screen("ldata",outdir+'/'+epoch_observ[0]+"_screen_xspec_spectrum")

    '''
    Testing the amount of raw source counts in the line detection range for all datagroups combined
    '''

    #### Testing if the data is above the count limit

    #for the line detection
    AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')
    glob_counts=0
    indiv_counts=[]

    bg_counts=0

    for id_grp,elem_sat in enumerate(sat_indiv_good):

        i_grp=id_grp+1
        #for NICER we subtract the rate from the background which at this point is the entire model ([3])
        if elem_sat=='NICER':
            indiv_counts+=[round((AllData(i_grp).rate[0]-AllData(i_grp).rate[3])*AllData(i_grp).exposure)]

            bg_counts+=AllData(i_grp).rate[3]*AllData(i_grp).exposure
        else:
            indiv_counts+=[round(AllData(i_grp).rate[0]*AllData(i_grp).exposure)]

            bg_counts+=(AllData(i_grp).rate[2]-AllData(i_grp).rate[0])*AllData(i_grp).exposure

        glob_counts+=indiv_counts[-1]

    '''
    SNR formula from https://xmm-tools.cosmos.esa.int/external/sas/current/doc/specgroup.pdf 4.3.2 (1)
    #note: here indiv_count is a net count, so a full source counts is indiv_counts+ bg and
    #source + bg is thus net + 2*bg
    '''

    SNR=(sum(indiv_counts))/np.sqrt(sum(indiv_counts)+2*bg_counts)

    if glob_counts<counts_min:
        flag_lowSNR_line=True
        if not fit_lowSNR:
            print_xlog('\nInsufficient net counts ('+str(round(glob_counts))+' < '+str(round(counts_min))+
                  ') in line detection range.')
            return fill_result('Insufficient net counts ('+str(round(glob_counts))+' < '+str(round(counts_min))+\
                               ') in line detection range.')

    elif SNR<SNR_min:
        if not fit_lowSNR:
            print_xlog('\nInsufficient SNR ('+str(round(SNR,1))+'<50) in line detection range.')
            return fill_result('Insufficient SNR ('+str(round(SNR,1))+'<50) in line detection range.')
    else:
        flag_lowSNR_line=False

    if not force_ener_bounds:
        #re-adjusting the line cont range to esathigh to allow fitting up to higher energies if needed
        line_cont_range[1] = min(np.array(args.line_cont_range.split(' ')).astype(float)[1], max(e_sat_high_indiv))

    #limiting to the HID energy range
    ignore_data_indiv(hid_cont_range[0],hid_cont_range[1],reset=True,sat_low_groups=e_sat_low_indiv,sat_high_groups=e_sat_high_indiv,glob_ignore_bands=ignore_bands_indiv)

    glob_counts=0
    indiv_counts=[]

    for i_grp in range(1,AllData.nGroups+1):
        #could add background here
        indiv_counts+=[round(AllData(i_grp).rate[2]*AllData(i_grp).exposure)]
        glob_counts+=indiv_counts[-1]
    if glob_counts<counts_min_HID:
        print_xlog('\nInsufficient counts ('+str(round(glob_counts))+' < '+str(round(counts_min_HID))+
              ') in HID detection range.')
        return fill_result('Insufficient counts ('+str(round(glob_counts))+' < '+str(round(counts_min_HID))+\
                           ') in HID detection range.')

    #testing the NuSTAR bands if asked to
    if filter_NuSTAR_SNR!=0.:
        for i_exp,elem_sat in enumerate(sat_indiv_init):
            SNR_low=False
            if elem_sat=='NuSTAR':
                #starting at 10keV (stops at 78)
                for val_keV in np.arange(10.,79.):
                    AllData(i_exp+1).notice('all')
                    AllData(i_exp+1).ignore('**-'+str(val_keV)+' '+str(val_keV+1)+'-**')
                    #SNR formula from https://xmm-tools.cosmos.esa.int/external/sas/current/doc/specgroup.pdf 4.3.2 (1)

                    SNR_val=np.sqrt(AllData(i_exp+1).exposure)*AllData(i_exp+1).rate[0]\
                            /np.sqrt(AllData(i_exp+1).rate[0]+2*(AllData(i_exp+1).rate[2]-AllData(i_exp+1).rate[0]))

                    #stopping the computation if the SNR goes below the limit
                    if SNR_val<filter_NuSTAR_SNR:
                        SNR_low=True
                        break

                #resetting the energy band of the exposure
                AllData(i_exp+1).notice('all')
                AllData(i_exp+1).ignore('**-'+str(e_sat_low_indiv[i_exp])+' '+str(e_sat_high_indiv[i_exp])+'-**')

                #storing the final value
                e_sat_high_indiv[i_exp]=79. if not SNR_low else val_keV

                print_xlog('High energy limit of exposure '+str(i_exp+1)+' fixed to '+str(e_sat_high_indiv[i_exp])+
                      ' keV with a SNR limit of '+str(filter_NuSTAR_SNR))

    '''re-resetting the energy bands if needed to adapt to the NuSTAR SNR cuts'''
    if not force_ener_bounds:
        hid_cont_range[1] = max(e_sat_high_indiv)

        #here we just test for the snr and create the line searc hspace, this will re-adjusted esat_high later
        line_cont_range[1] = min(np.array(args.line_cont_range.split(' ')).astype(float)[1],max(e_sat_high_indiv))

    if pdf_only:

        if catch_errors:

            try:

                pdf_summary(epoch_files,fit_ok=True,summary_epoch=fill_result('Line detection complete.'),
                            e_sat_low_list=e_sat_low_indiv,e_sat_high_list=e_sat_high_indiv)

                #closing the logfile for both access and Xspec
                curr_logfile.close()
                Xset.closeLog()

                return fill_result('Line detection complete.')
            except:
                return fill_result('Missing elements to compute PDF.')

        else:

            pdf_summary(epoch_files, fit_ok=True, summary_epoch=fill_result('Line detection complete.'),
                        e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

            # closing the logfile for both access and Xspec
            curr_logfile.close()
            Xset.closeLog()

            return fill_result('Line detection complete.')

    if line_ul_only:

        #loading the autofit model
        #note that this also restores the data ignore states
        Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm')

        #reloading the fitlines class
        fitlines=load_fitmod(outdir+'/'+epoch_observ[0]+'_fitmod_autofit.pkl')

        #updating fitcomps
        fitlines.update_fitcomps()

        #recreating the no abs version

        #deleting all absorption components (reversed so we don't have to update the fitcomps)
        for comp in [elem for elem in fitlines.includedlist if elem is not None][::-1]:
            if comp.named_absline:
                #note that with no rollback we do not update the values of the component so it has still its included status and everything else
                comp.delfrommod(rollback=False)

        #storing the no abs line 'continuum' model
        data_autofit_noabs=allmodel_data()

        #reloading previously computed information
        dict_linevis={'visual_line':False,
                      'cameras':'all' if sat_glob=='multi' else cameras,
                      'expmodes':expmodes}

        #the goal here is to avoid importing streamlit if possible
        from visual_line_tools import abslines_values

        precomp_absline_vals,precomp_autofit_vals=abslines_values(autofit_store_path,dict_linevis,
                                             obsid=epoch_observ[0])

        #selecting object 0 and obs 0 aka this obs
        precomp_absline_vals=precomp_absline_vals[0][0]
        precomp_autofit_vals=precomp_autofit_vals[0][0]

        abslines_eqw,abslines_bshift,abslines_delchi,abslines_flux,abslines_sign=precomp_absline_vals[:5]

        abslines_eqw_upper=np.zeros(len(range_absline))

        abslines_em_overlap,abslines_width,abslines_bshift_distinct=precomp_absline_vals[6:]

        autofit_parerrors,autofit_parnames=precomp_autofit_vals

        sign_widths_arr=np.array([elem[0] if elem[0]-elem[1]>1e-6 else 0 for elem in abslines_width])

        #fetching the ID of this observation
        # freezing the model to avoid it being affected by the missing absorption lines
        # note : it would be better to let it free when no absorption lines are there but we keep the same procedure for
        # consistency
        allfreeze()

        # computing a mask for significant lines
        mask_abslines_sign = abslines_sign >= sign_threshold

        # computing the upper limits for the non significant lines
        abslines_eqw_upper = fitlines.get_eqwidth_uls(mask_abslines_sign, abslines_bshift, sign_widths_arr,
                                                      pre_delete=True)

        # here will need to reload an accurate model before updating the fitcomps
        '''HTML TABLE FOR the pdf summary'''

        abslines_table_str = html_table_maker()

        with open(outdir + '/' + epoch_observ[0] + '_abslines_table.txt', 'w+') as abslines_table_file:
            abslines_table_file.write(abslines_table_str)

            Xset.logChatter = 10

        # storing line string
        autofit_store_str = epoch_observ[0] + '\t' + \
                            str(abslines_eqw.tolist()) + '\t' + str(abslines_bshift.tolist()) + '\t' + str(
            abslines_delchi.tolist()) + '\t' + \
                            str(abslines_flux.tolist()) + '\t' + str(abslines_sign.tolist()) + '\t' + str(
            abslines_eqw_upper.tolist()) + '\t' + \
                            str(abslines_em_overlap.tolist()) + '\t' + str(abslines_width.tolist()) + '\t' + str(
            abslines_bshift_distinct.tolist()) + '\t' + \
                            str(autofit_parerrors.tolist()) + '\t' + str(autofit_parnames.tolist()) + '\n'

        '''Storing the results'''

        autofit_store_header = 'Observ_id\tabslines_eqw\tabslines_bshift\tablines_delchi\tabslines_flux\t' + \
                               'abslines_sign\tabslines_eqw_upper\tabslines_em_overlap\tabslines_width\tabslines_bshift_distinct' + \
                               '\tautofit_parerrors\tautofit_parnames\n'

        file_edit(path=autofit_store_path, line_id=epoch_observ[0], line_data=autofit_store_str,
                  header=autofit_store_header)

        '''PDF creation'''

        if write_pdf:
            pdf_summary(epoch_files, fit_ok=True, summary_epoch=fill_result('Line detection complete.'),
                        e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

        # closing the logfile for both access and Xspec
        curr_logfile.close()
        Xset.closeLog()

        return fill_result('Line detection complete.')

    #creating the continuum model list
    comp_cont=model_list(cont_model)

    #taking off the constant factor if there's only one data group
    if AllData.nGroups==1:
        comp_cont=[elem for elem in comp_cont if elem!='glob_constant']

    #testing the presence of arf, rmf etc in the datasets
    isbg_grp=[]
    isrmf_grp=[]
    isarf_grp=[]
    for i_grp in range(1,AllData.nGroups+1):
        try:
            AllData(i_grp).background
            isbg_grp+=[True]
        except:
            isbg_grp+=[False]

        try:
            AllData(i_grp).response
            isrmf_grp+=[True]
        except:
            isrmf_grp+=[False]

        try:
            AllData(i_grp).response.arf
            isarf_grp+=[True]
        except:
            isarf_grp+=[False]


    abslines_table_str=None

    def curr_store_fit(mode='broadband',fitmod=None):
        store_fit(mode=mode, epoch_id=epoch_observ[0],outdir=outdir, logfile=curr_logfile, fitmod=fitmod)

    def hid_fit_infos(fitmodel, broad_absval, post_autofit=False):

        '''
        computes various informations about the fit
        '''

        if post_autofit:
            add_str = '_post_auto'
        else:
            add_str = ''

        if not broad_HID_mode:
            # freezing what needs to be to avoid problems with the Chain
            calc_error(curr_logfile, param='1-' + str(AllModels(1).nParameters * AllData.nGroups), timeout=60,
                       freeze_pegged=True, indiv=True)

        Fit.perform()

        fitmodel.update_fitcomps()

        # storing the flux and HR with the absorption to store the errors
        # We can only show one flux in the HID so we use the first one, which should be the most 'precise' with our order (pn first)

        #dont know why this causes an issue
        from xspec import AllChains

        AllChains.defLength = 10000
        AllChains.defBurn = 5000
        AllChains.defWalkers = 10

        # deleting the previous chain to avoid conflicts
        AllChains.clear()

        if os.path.exists(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits'):
            os.remove(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits')

        #we don't need to make a new chain if the broad fit has done it already
        if not broad_HID_mode:

            # Creating a chain to avoid problems when computing the errors
            Chain(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits')

        # trying to freeze pegged parameters again in case the very last fit created pegs
        calc_error(curr_logfile, param='1-' + str(AllModels(1).nParameters * AllData.nGroups), timeout=20,
                   freeze_pegged=True, indiv=True)

        Fit.perform()

        mod_hid=allmodel_data()

        fitmodel.update_fitcomps()

        if not broad_HID_mode:
            # Creating a chain to avoid problems when computing the errors
            Chain(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits')

        # computing and storing the flux for the full luminosity and two bands for the HR
        spflux_single = [None] * 5

        '''the first computation is ONLY to get the errors, the main values are overwritten below'''
        # we still only compute the flux of the first model even with NICER because the rest is BG

        AllModels.calcFlux(str(hid_cont_range[0]) + ' ' + str(hid_cont_range[1]) + " err 1000 90")

        spflux_single[0] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
                                               AllData(1).flux[0]
        AllModels.calcFlux("3. 6. err 1000 90")
        spflux_single[1] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
                                               AllData(1).flux[0]
        AllModels.calcFlux("6. 10. err 1000 90")
        spflux_single[2] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
                                               AllData(1).flux[0]
        AllModels.calcFlux("1. 3. err 1000 90")
        spflux_single[3] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
                                               AllData(1).flux[0]
        AllModels.calcFlux("3. 10. err 1000 90")
        spflux_single[4] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
                                               AllData(1).flux[0]

        #easier to save the fit here to reload it without having changed the energies
        if os.path.isfile(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm'):
            os.remove(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm')
        Xset.save(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm', info='a')

        if max(e_sat_high_indiv)>=20:
            #re-arranging the energies array to compute a high energy flux value
            AllModels.setEnergies('0.1 100. 1000 log')
            #need to remake a new fit after that
            calc_fit()
            AllModels.calcFlux("15. 50. err 1000 90")
            spflux_high = np.array([AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
                                                   AllData(1).flux[0]])
        else:
            spflux_high=None

        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm')

        spflux_single = np.array(spflux_single)

        AllChains.clear()

        mod_hid.load()

        for i_sp in range(len(epoch_files_good)):
            if line_cont_ig_indiv[i_sp] != '':
                try:
                    AllData(i_sp+1).notice(line_cont_ig_indiv[i_sp])
                except:
                    breakpoint()
                    pass
        curr_store_fit(mode='broadhid' + add_str, fitmod=fitmodel)

        # storing the fitmod class into a file
        fitmodel.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid' + add_str + '.pkl')

        # taking off the absorption (if it is in the final included components) before computing the flux
        abs_incl_comps = (np.array(fitmodel.complist)[[elem.absorption and elem.included for elem in \
                                                       [elem_comp for elem_comp in fitmodel.complist if
                                                        elem_comp is not None]]])
        if len(abs_incl_comps)!=0:
            main_abs_comp=abs_incl_comps[0]
            main_abs_comp.xcomps[0].nH.values=0
        else:
            main_abs_comp=None

        #removing the calibration components and absorption lines, reverse included order to not have to update the fitcomps
        for comp in [elem for elem in fitmodel.includedlist if elem is not None][::-1]:
            if comp.calibration or comp.named_absline:
                comp.delfrommod(rollback=False)

        #removing the absorption lines
        # and replacing the main values with the unabsorbed flux values
        # (conservative choice since the other uncertainties are necessarily higher)

        #this value is fully useless at this point
        AllModels.calcFlux(str(hid_cont_range[0]) + ' ' + str(hid_cont_range[1]))

        spflux_single[0][0] = AllData(1).flux[0]
        AllModels.calcFlux("3. 6.")
        spflux_single[1][0] = AllData(1).flux[0]
        AllModels.calcFlux("6. 10.")
        spflux_single[2][0] = AllData(1).flux[0]
        AllModels.calcFlux("1. 3.")
        spflux_single[3][0] = AllData(1).flux[0]
        AllModels.calcFlux("3. 10.")
        spflux_single[4][0] = AllData(1).flux[0]

        #re-arranging the energies array to compute a high energy flux value
        if max(e_sat_high_indiv)>=20:
            AllModels.setEnergies('0.1 100. 1000 log')
            AllModels.calcFlux("15. 50.")
            spflux_high[0]=AllData(1).flux[0]

        #and resetting the energies
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '.xcm')

        '''
        Note that the errors are significantly underestimated, should do a cflux*continuum and freeze one of the
        norms of the parts of the continuum
        issues are:
            -potential issues with cflux computation (as a convolution component could be imprecise)
            -what to put in cflux (do we consider e.g. emission lines ? might need to re-organize the components then)
            -cflux needs to be initialized quite precisely or fitted first independantly, 
                otherwise it'll block the fit due to too high statistics
            -remake a chian just for this?
        '''

        spflux_single = spflux_single.T

        mod_hid.load()

        fitmodel.update_fitcomps()

        return spflux_single,spflux_high

    #reload previously stored autofits to gain time if asked to
    if reload_autofit and os.path.isfile(outdir+'/'+epoch_observ[0]+'_fitmod_autofit.pkl'):

        print_xlog('completed autofit detected...Reloading computation.')
        #reloading the broad band fit and model and re-storing associed variables
        fitlines_broad=load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_broadband_post_auto.pkl')
        Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_broadband_post_auto.xcm')
        fitlines_broad.update_fitcomps()
        data_broad=allmodel_data()

        # re fixing the absorption parameter and storing the value to retain it
        # if the absorption gets taken off and tested again
        abs_incl_comps = (np.array(fitlines_broad.complist)[[elem.absorption and elem.included for elem in \
                                                       [elem_comp for elem_comp in fitlines_broad.complist if
                                                        elem_comp is not None]]])
        if len(abs_incl_comps)!=0:
            main_abs_comp=abs_incl_comps[0]
            broad_absval=main_abs_comp.xcomps[0].nH.values[0]
        else:
            broad_absval=0

        #reloading the hid band fit and model and re-storing associed variables
        fitlines_broad=load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_broadband_post_auto.pkl')
        Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_broadhid_post_auto.xcm')
        fitlines_broad.update_fitcomps()
        data_broad=allmodel_data()

        if os.path.isfile(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid_post_auto.pkl'):
            fitlines_hid=load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid_post_auto.pkl')
            Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_broadhid_post_auto.xcm')
            fitlines_hid.update_fitcomps()
            data_broadhid=allmodel_data()

        else:
            fitlines_hid=fitlines_broad

            assert not broad_HID_mode, "This failsafe is for older computation which shouldn't run broad_HID_mode"

            # refitting in hid band for the HID values
            ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                              sat_high_groups=e_sat_high_indiv,glob_ignore_bands=ignore_bands_indiv)

            # fitting the model to the new energy band first
            calc_fit(logfile=fitlines_hid.logfile)

            # autofit
            fitlines_hid.global_fit(chain=True, lock_lines=True, directory=outdir, observ_id=epoch_observ[0],
                                split_fit=split_fit)

            fitlines_hid.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid_post_auto.pkl')

        from xspec import AllChains

        AllChains.clear()

        main_spflux,main_spflux_high = hid_fit_infos(fitlines_hid, broad_absval, post_autofit=True)

        if max(e_sat_high_indiv)>=20:
            np.savetxt(outdir + '/' + epoch_observ[0] + '_main_spflux_high.txt',main_spflux_high)

        if compute_highflux_only:
            return fill_result('Line detection complete.')


        # restoring the linecont save
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadband_linecont.xcm')

        data_mod_high=allmodel_data()

        cont_abspeak,cont_peak_points,cont_peak_widths,cont_peak_delchis,cont_peak_eqws,chi_dict_init=\
            narrow_line_search(data_mod_high,'cont',line_search_e=line_search_e,line_search_norm=line_search_norm,
                               e_sat_low_indiv=e_sat_low_indiv,peak_thresh=peak_thresh,peak_clean=peak_clean,
                               line_cont_range=line_cont_range,trig_interval=trig_interval,
                               scorpeon_save=data_broad.scorpeon)

        with open(outdir+'/'+epoch_observ[0]+'_chi_dict_init.pkl','wb') as file:
            dill.dump(chi_dict_init,file)

        # re-ignoring PIN if needed
        if 'PIN' in epoch_dets_good:
            mask_nodeload = epoch_dets_good != 'PIN'
        else:
            mask_nodeload = np.repeat(True,AllData.nGroups)

        # reloading the continuum models to get the saves back and compute the continuum infos
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm')

        #loading the autofit model
        Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm')

        #reloading the fitlines class
        fitlines=load_fitmod(outdir+'/'+epoch_observ[0]+'_fitmod_autofit.pkl')

        #updating fitcomps
        fitlines.update_fitcomps()

        #reloading the fit
        calc_fit()

        #reloading the chain
        AllChains+=outdir+'/'+epoch_observ[0]+'_chain_autofit.fits'

        data_autofit=allmodel_data()

    else:

        '''Continuum fits'''
        def high_fit(broad_absval,broad_abscomp,broad_gamma_nthcomp):

            '''
            high energy fit and flux array computation
            '''

            print_xlog('\nComputing line continuum fit...')

            AllModels.clear()
            xscorpeon.load(scorpeon_save=data_broad.scorpeon,frozen=True)

            #limiting to the line search energy range
            ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                          sat_low_groups=e_sat_low_indiv[mask_nodeload],
                          sat_high_groups=e_sat_high_indiv[mask_nodeload],
                          glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

            #if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
            if not flag_lowSNR_line:
                for i_grp in range(len(epoch_files_good[mask_nodeload])):
                    #ignoring the line_cont_ig energy range for the fit to avoid contamination by lines
                    AllData(i_grp+1).ignore(line_cont_ig_indiv[i_grp])

            #comparing different continuum possibilities with a broken powerlaw or a combination of diskbb and powerlaw

            #creating the automatic fit class for the standard continuum

            # Xset.closeLog()

            # Xset.openLog('lineplots_opt_nth/test.log')
            # curr_logfile = Xset.log
            # curr_logfile_write = Xset.log
            # curr_logfile_write.reconfigure(line_buffering=True)
            # curr_logfile = open(curr_logfile_write.name, 'r')

            if broad_absval!=0:
                fitcont_high=fitmod(comp_cont,
                                    curr_logfile,curr_logfile_write,absval=broad_absval)
            else:

                #creating the fitcont without the absorption component if it didn't exist in the broad model
                fitcont_high=fitmod([elem for elem in comp_cont if elem!=broad_abscomp],
                                    curr_logfile,curr_logfile_write)

            #freezing the gamma of the nthcomp to avoid nonsenses:
            fitcont_high.fixed_gamma=broad_gamma_nthcomp

            # forcing the absorption component to be included for the broad band fit
            if sat_glob == 'NuSTAR' and freeze_nH:
                main_abscomp = (np.array(fitcont_high.complist)[[elem.absorption for elem in \
                                                                fitcont_high.complist]])[0]
                main_abscomp.mandatory = True

            fitcont_high.global_fit(split_fit=split_fit,method=cont_fit_method)

            # mod_fitcont=allmodel_data()

            chi2_cont=Fit.statistic
            # except:
            #     pass
            #     chi2_cont=0

            # AllModels.clear()
            # xscorpeon.load(scorpeon_save=data_broad.scorpeon,frozen=True)
            #not used currently
            # #with the broken powerlaw continuum
            # fitcont_high_bkn=fitmod(comp_cont_bkn,curr_logfile)

            # try:
            #     #fitting
            #     fitcont_high_bkn.global_fit(split_fit=split_fit))

            #     chi2_cont_bkn=Fit.statistic
            # except:
            #     pass
            chi2_cont_bkn=0

            if chi2_cont==0 and chi2_cont_bkn==0:

                print_xlog('\nProblem during line continuum fit. Skipping line detection for this exposure...')
                return ['\nProblem during line continuum fit. Skipping line detection for this exposure...']
            # else:
            #     if chi2_cont<chi2_cont_bkn:
            #         model_load(mod_fitcont)


            #renoticing the line energy range
            #note: for now this is fine but might need to be udpated later with telescopes with global ignore bands
            #matching part of this

            for i_sp in range(len(epoch_files_good[mask_nodeload])):
                if line_cont_ig_indiv[i_sp] != '':
                    AllData(i_sp+1).notice(line_cont_ig_indiv[i_sp])

            #saving the model data to reload it after the broad band fit if needed
            mod_high_dat=allmodel_data()

            #rescaling before the prints to avoid unecessary loggings in the screen
            rescale(auto=True)

            #screening the xspec plot and the model informations for future use
            Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+"_screen_xspec_broadband_linecont",
                        includedlist=fitcont_high.includedlist)

            #saving the model str
            catch_model_str(curr_logfile,savepath=outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.txt')

            #deleting the model file since Xset doesn't have a built-in overwrite argument and crashes when asking manual input
            if os.path.isfile(outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.xcm'):
                os.remove(outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.xcm')

            #storing the current configuration and model
            Xset.save(outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.xcm',info='a')

            #storing the class
            fitcont_high.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadband_linecont.pkl')

            return [mod_high_dat,fitcont_high]

        def broad_fit():

            '''Broad band fit to get the HR ratio and Luminosity'''

            #first broad band fit in e_sat_low-10 to see the spectral shape
            print_xlog('\nComputing broad band fit for visualisation purposes...')

            ignore_data_indiv(e_sat_low_indiv, e_sat_high_indiv, reset=True, glob_ignore_bands=ignore_bands_indiv)

            #if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
            if not flag_lowSNR_line:
                for i_sp in range(len(epoch_files_good)):
                    if line_cont_ig_indiv[i_sp] != '':
                        AllData(i_sp+1).ignore(line_cont_ig_indiv[i_sp])

            #forcing an absorption value if asked to
            if sat_glob=='NuSTAR' and freeze_nH:
                broad_absval=freeze_nH_val
            else:
                broad_absval=None

            #creating the automatic fit class for the standard continuum
            fitcont_broad=fitmod(comp_cont,curr_logfile,curr_logfile_write,absval=broad_absval,sat_list=sat_indiv_good)

            # forcing the absorption component to be included for the broad band fit
            if sat_glob == 'NuSTAR' and freeze_nH:
                main_abscomp = (np.array(fitcont_broad.complist)[[elem.absorption for elem in \
                                                                fitcont_broad.complist]])[0]
                main_abscomp.mandatory = True

            #fitting
            fitcont_broad.global_fit(split_fit=split_fit,method=cont_fit_method,fit_scorpeon=True,fit_SAA_norm=True)

            mod_fitcont=allmodel_data()

            chi2_cont=Fit.statistic

            chi2_cont_bkn=0

            if chi2_cont==0 and chi2_cont_bkn==0:

                print_xlog('\nProblem during broad band fit. Skipping line detection for this exposure...')
                return ['\nProblem during broad band fit. Skipping line detection for this exposure...']
            # else:
            #     if chi2_cont<chi2_cont_bkn:
            #         model_load(mod_fitcont)

            #storing the absorption of the broad fit if there is absorption
            abs_incl_comps = (np.array(fitcont_broad.complist)[[elem.absorption and elem.included for elem in \
                                                           [elem_comp for elem_comp in fitcont_broad.complist if
                                                            elem_comp is not None]]])
            if len(abs_incl_comps) != 0:
                main_abs_comp = abs_incl_comps[0]
                broad_absval=main_abs_comp.xcomps[0].nH.values[0]
                broad_abscomp=main_abs_comp.compname
            else:
                main_abs_comp = None
                broad_absval=0
                broad_abscomp=''

            if 'disk_nthcomp' in [comp.compname for comp in \
                                  [elem for elem in fitcont_broad.includedlist if elem is not None]]:
                broad_gamma_nthcomp=fitcont_broad.disk_nthcomp.xcomps[0].Gamma.values[0]
            else:
                broad_gamma_nthcomp=2.5
            for i_sp in range(len(epoch_files_good)):
                if line_cont_ig_indiv[i_sp] != '':
                    AllData(i_sp+1).notice(line_cont_ig_indiv[i_sp])

            curr_store_fit(mode='broadband',fitmod=fitcont_broad)

            #storing the class
            fitcont_broad.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadband.pkl')

            #saving the model
            data_broad=allmodel_data()
            print_xlog('\nComputing HID broad fit...')

            for i_sp in range(len(epoch_files_good)):
                if line_cont_ig_indiv[i_sp] != '':
                    AllData(i_sp+1).ignore(line_cont_ig_indiv[i_sp])

            if broad_HID_mode:
                fitcont_hid=fitcont_broad

            else:
                AllModels.clear()

                #reloading the scorpeon save (if there is one, aka if with NICER),
                # from the broad fit and freezing it to avoid further variations
                xscorpeon.load(scorpeon_save=data_broad.scorpeon,frozen=True)

                ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                                  sat_high_groups=e_sat_high_indiv,glob_ignore_bands=ignore_bands_indiv)

                #if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
                if not flag_lowSNR_line:
                    for i_sp in range(len(epoch_files_good)):
                        if line_cont_ig_indiv[i_sp] != '':
                            AllData(i_sp+1).ignore(line_cont_ig_indiv[i_sp])

                #creating the automatic fit class for the standard continuum
                # (without absorption if it didn't get included)
                if broad_absval!=0:
                    fitcont_hid=fitmod(comp_cont,curr_logfile,curr_logfile_write,absval=broad_absval)
                else:
                    #creating the fitcont without the absorption component if it didn't exist in the broad model
                    fitcont_hid=fitmod([elem for elem in comp_cont if elem!=broad_abscomp],
                                       curr_logfile,curr_logfile_write)

                # forcing the absorption component to be included for the broad band fit
                if sat_glob == 'NuSTAR' and freeze_nH:
                    main_abscomp = (np.array(fitcont_hid.complist)[[elem.absorption for elem in \
                                                                      fitcont_hid.complist]])[0]
                    main_abscomp.mandatory = True

                #fitting
                fitcont_hid.global_fit(split_fit=split_fit,method=cont_fit_method)

                mod_fitcont=allmodel_data()

            chi2_cont=Fit.statistic

            if chi2_cont==0:

                print_xlog('\nProblem during hid band fit. Skipping line detection for this exposure...')
                return ['\nProblem during hid band fit. Skipping line detection for this exposure...']

            spflux_single,spflux_high=hid_fit_infos(fitcont_hid,broad_absval)

            return spflux_single,broad_absval,broad_abscomp,data_broad,fitcont_broad,broad_gamma_nthcomp,spflux_high

        AllModels.clear()

        #frozen Scorpeon for now (unfreezed during the broad fit)
        xscorpeon.load(frozen=True)

        ignore_data_indiv(e_sat_low_indiv,e_sat_high_indiv,reset=True,glob_ignore_bands=ignore_bands_indiv)

        baseload_path=outdir+'/'+epoch_observ[0]+'_baseload.xcm'
        if os.path.isfile(baseload_path):
            os.remove(baseload_path)

        Xset.save(outdir+'/'+epoch_observ[0]+'_baseload.xcm',info='a')

        result_broad_fit=broad_fit()

        if len(result_broad_fit)==1:
            return fill_result(result_broad_fit)
        else:
            main_spflux,broad_absval,broad_abscomp,data_broad,fitcont_broad,broad_gamma_nthcomp,main_spflux_high\
                =result_broad_fit

        if max(e_sat_high_indiv)>=20:
            np.savetxt(outdir + '/' + epoch_observ[0] + '_main_spflux_high.txt',main_spflux_high)

        #reloading the frozen scorpeon data\
        # (which won't change anything if it hasn't been fitted but will help otherwise)
        xscorpeon.load(scorpeon_save=data_broad.scorpeon,frozen=True)

        '''
        more advanced automatic detector exclusions to avoid crashing xspec should be done later if needed
        note the scorpeon model can crash, but in the simple case below we won't ever have Suzaku-NICER
        for more general automatic detector exclusion this will need to be adresed
        '''
        #removing PIN from the detectors if it is there
        if 'PIN' in epoch_dets_good:
            reload_sp(baseload_path,keyword_skip='pin',newbl_keyword='autofit')
            mask_nodeload = epoch_dets_good!='PIN'
        else:
            mask_nodeload = np.repeat(True,AllData.nGroups)

        result_high_fit=high_fit(broad_absval,broad_abscomp,broad_gamma_nthcomp)

        #if the function returns an array of length 1, it means it returned an error message
        if len(result_high_fit)==1:
            return fill_result(result_high_fit)
        else:
            data_mod_high,fitmod_cont=result_high_fit

        # re-limiting to the line search energy range
        ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                          sat_low_groups=e_sat_low_indiv[mask_nodeload],
                          sat_high_groups=e_sat_high_indiv[mask_nodeload],
                          glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

        #changing back to the auto rescale of xspec
        Plot.commands=()
        Plot.addCommand('rescale')

        print_xlog('\nStarting line search...')

        cont_abspeak,cont_peak_points,cont_peak_widths,cont_peak_delchis,cont_peak_eqws,chi_dict_init=\
            narrow_line_search(data_mod_high,'cont',line_search_e=line_search_e,line_search_norm=line_search_norm,
                               e_sat_low_indiv=e_sat_low_indiv[mask_nodeload],
                               peak_thresh=peak_thresh,peak_clean=peak_clean,
                               line_cont_range=line_cont_range,trig_interval=trig_interval,
                               scorpeon_save=data_broad.scorpeon)

        with open(outdir+'/'+epoch_observ[0]+'_chi_dict_init.pkl','wb') as file:
            dill.dump(chi_dict_init,file)

        #same for autofit, then specific mode

        plot_line_search(chi_dict_init, outdir, sat_glob,suffix='cont', epoch_observ=epoch_observ)

        '''
        Automatic line fitting
        '''

        #### Autofit

        if autofit and (cont_abspeak or force_autofit) and not flag_lowSNR_line:

            '''
            See the fitmod code for a detailed description of the auto fit process
            '''

            #reloading the continuum fitcomp
            data_mod_high.load()

            #feching the list of components we're gonna use
            comp_lines=model_list(autofit_model)

            #creating a new logfile for the autofit
            curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log_autofit.log')
            curr_logfile_write.reconfigure(line_buffering=True)
            curr_logfile=open(curr_logfile_write.name,'r')

            #creating the fitmod object with the desired components (we currently do not use comp groups)
            fitlines=fitmod(comp_lines,curr_logfile,curr_logfile_write,prev_fitmod=fitmod_cont,sat_list=sat_indiv_good)

            # (just in case) inputting the fixed values to avoid issues during component deletion
            fitlines.fixed_abs = broad_absval
            fitlines.fixed_gamma = broad_gamma_nthcomp

            #global fit, with MC only if no continuum refitting
            fitlines.global_fit(chain=not refit_cont,directory=outdir,observ_id=epoch_observ[0],split_fit=split_fit,
                                no_abslines=no_abslines)

            #storing the final fit
            data_autofit=allmodel_data()

            '''
            ####Refitting the continuum
            
            if refit_cont is set to True, we add a 3 step process to better estimate the continuum from the autofit lines.
            
            First, we relaunch a global fit iteration while blocking all line parameters in the broad band.
            We then refreeze the absorption and relaunch two global fit iterations in 3-10 (for the HID) and 4-10 (for the autofit continuum)
            '''

            if refit_cont:

                #don't know what's happening
                from xspec import AllChains

                AllChains.clear()

                #saving the initial autofit result for checking purposes
                curr_store_fit(mode='autofit_init',fitmod=fitlines)

                #storing the fitmod class into a file
                fitlines.dump(outdir+'/'+epoch_observ[0]+'_fitmod_autofit_init.pkl')

                #updating the logfile for the second round of fitting
                curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log_autofit_recont.log')
                curr_logfile_write.reconfigure(line_buffering=True)
                curr_logfile=open(curr_logfile_write.name,'r')

                #updating it in the fitmod
                fitlines.logfile=curr_logfile
                fitlines.logfile_write=curr_logfile_write
                fitlines.update_fitcomps()

                #freezing every line component
                for comp in [elem for elem in fitlines.includedlist if elem is not None]:
                    if comp.line:

                        freeze(parlist=comp.unlocked_pars)

                #reloading the broad band data without overwriting the model
                Xset.restore(baseload_path)

                fitlines.update_fitcomps()

                '''
                To reload the broadband model and avoid issues, 
                we need to shift some datagroups and recreate the links properly
                This will need to be done eventually but for now with just pin going back and forth
                as the last datagroup this should be fine
                '''

                #refitting in broad band for the nH
                ignore_data_indiv(e_sat_low_indiv, e_sat_high_indiv, reset=True, glob_ignore_bands=ignore_bands_indiv)

                #merging the previous continuum if asked to
                if merge_cont:

                    fitlines.merge(fitcont_broad)

                #thawing the absorption to allow improving its value

                #first we remove the fixed value from the fitmod itself
                if not (sat_glob=='NuSTAR' and freeze_nH):
                    #we reset the value of the fixed abs to allow it to be free if it gets deleted and put again
                    fitlines.fixed_abs=None

                #then we attempt to thaw the component value if the component is included
                abs_incl_comps = (np.array(fitlines.complist)[[elem.absorption and elem.included for elem in \
                                                                    [elem_comp for elem_comp in fitlines.complist
                                                                     if
                                                                     elem_comp is not None]]])

                if len(abs_incl_comps)!=0 and not (sat_glob=='NuSTAR' and freeze_nH):
                    main_abscomp=abs_incl_comps[0]
                    main_abscomp.xcomps[0].nH.frozen=False

                    #releasing the n_unlocked_pars of the absorption component
                    fitlines.update_fitcomps()
                    main_abscomp.n_unlocked_pars_base=len(main_abscomp.unlocked_pars)

                #thawing similarly the nthcomp gamma (note that we can't just pars
                if 'disk_nthcomp' in fitlines.name_complist:
                    fitlines.fixed_gamma=None

                if 'disk_nthcomp' in [comp.compname for comp in \
                                      [elem for elem in fitlines.includedlist if elem is not None]]:
                    fitlines.disk_nthcomp.xcomps[0].Gamma.frozen=False

                    fitlines.update_fitcomps()
                    fitlines.disk_nthcomp.n_unlocked_pars_base=len(fitlines.disk_nthcomp.unlocked_pars)

                #fitting the model to the new energy band first
                calc_fit(logfile=fitlines.logfile)

                #autofit
                fitlines.global_fit(chain=False,lock_lines=True,directory=outdir,observ_id=epoch_observ[0],
                                    split_fit=split_fit,fit_scorpeon=True,fit_SAA_norm=fit_SAA_norm)

                #refreezing the scorpeon model
                xscorpeon.freeze()

                data_broad_postauto=allmodel_data()

                AllChains.clear()

                curr_store_fit(mode='broadband_post_auto',fitmod=fitlines)

                #storing the class
                fitlines.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadband_post_auto.pkl')

                #remaking this list just in case there was some deletion and addition and the components changed
                abs_incl_comps = (np.array(fitlines.complist)[[elem.absorption and elem.included for elem in \
                                                                    [elem_comp for elem_comp in fitlines.complist
                                                                     if
                                                                     elem_comp is not None]]])

                #re fixing the absorption parameter and storing the value to retain it if the absorption
                # gets taken off and tested again
                fitlines.fixed_abs=broad_absval
                if len(abs_incl_comps)!=0:
                    main_abscomp=abs_incl_comps[0]
                    main_abscomp.xcomps[0].nH.frozen=True
                    broad_absval=main_abscomp.xcomps[0].nH.values[0]

                    fitlines.update_fitcomps()
                    main_abscomp.n_unlocked_pars_base=len(main_abscomp.unlocked_pars)
                else:
                    broad_absval=0

                if not broad_HID_mode:
                    #refitting in hid band for the HID values
                    ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                                      sat_high_groups=e_sat_high_indiv, glob_ignore_bands=ignore_bands_indiv)

                    #fitting the model to the new energy band first
                    calc_fit(logfile=fitlines.logfile)

                    #autofit
                    fitlines.global_fit(chain=False,lock_lines=True,directory=outdir,observ_id=epoch_observ[0],split_fit=split_fit)

                AllChains.clear()

                main_spflux,main_spflux_high=hid_fit_infos(fitlines,broad_absval,post_autofit=True)

                if max(e_sat_high_indiv)>=20:
                    np.savetxt(outdir + '/' + epoch_observ[0] + '_main_spflux_high.txt', main_spflux_high)

                '''
                Refitting in the autofit range to get the newer version of the autofit and continuum
                
                first: freezing the nthcomp gamma if necessary
                 
                second: restoring the line freeze states
                here we restore the INITIAL component freeze state, effectively thawing all components pegged during the first autofit
                '''

                # freezing the gamma of the nthcomp to avoid nonsenses:
                fitlines.fixed_gamma = broad_gamma_nthcomp

                if 'disk_nthcomp' in [comp.compname for comp in \
                                      [elem for elem in fitlines.includedlist if elem is not None]]:
                    fitlines.disk_nthcomp.xcomps[0].Gamma.frozen = True

                    fitlines.update_fitcomps()
                    fitlines.disk_nthcomp.n_unlocked_pars_base = len(fitlines.disk_nthcomp.unlocked_pars)

                for comp in [elem for elem in fitlines.includedlist if elem is not None]:

                    if comp.line and not comp.calibration:
                        #unfreezing the parameter with the mask created at the first addition of the component

                        unfreeze(parlist=np.array(comp.parlist)[comp.unlocked_pars_base_mask])

                    if comp.calibration and comp.compname not in ['calNuSTAR_edge']:
                        fitlines.remove_comp(comp)

                #re-ignoring PIN if needed
                if 'PIN' in epoch_dets_good:
                    mod_pre_autofit=allmodel_data()
                    Xset.restore(baseload_path.replace('.xcm','_autofit.xcm'))
                    mask_nodeload = epoch_dets_good != 'PIN'

                    #restoring the rest of the datagroups because reloading a different number of datagroup
                    #resets the links of all but the first DGs
                    mod_pre_autofit.default=mod_pre_autofit.default[mask_nodeload]

                    #note that more things would need to be done for NICER/Suzaku and
                    #some things might not work with NuSTAR edge's links but let's see
                    mod_pre_autofit.load()
                else:
                    mask_nodeload = np.repeat(True,AllData.nGroups)

                fitlines.update_fitcomps()

                ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                                  sat_low_groups=e_sat_low_indiv[mask_nodeload],
                                  sat_high_groups=e_sat_high_indiv[mask_nodeload],
                                  glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

                #fitting the model to the new energy band first
                calc_fit(logfile=fitlines.logfile)

                #autofit
                fitlines.global_fit(chain=True,directory=outdir,observ_id=epoch_observ[0],split_fit=split_fit,
                                    no_abslines=no_abslines)

                #storing the final fit
                data_autofit=allmodel_data()

            #storing the final plot and parameters
            #screening the xspec plot
            Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+"_screen_xspec_autofit",
                        includedlist=fitlines.includedlist)

            if 'Chandra' in sat_indiv_good:

                #plotting a zoomed version for HETG spectra
                AllData.ignore('**-6.5 '+str(float(min(9,max(e_sat_high_indiv))))+'-**')

                Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+"_screen_xspec_autofit_zoom",
                            includedlist=fitlines.includedlist)

                #putting back the energy range
                AllData.notice(str(line_cont_range[0])+'-'+str(line_cont_range[1]))

            #saving the model str
            catch_model_str(curr_logfile,savepath=outdir+'/'+epoch_observ[0]+'_mod_autofit.txt')

            if os.path.isfile(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm'):
                os.remove(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm')

            #storing the current configuration and model
            Xset.save(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm',info='a')

            #storing the class
            fitlines.dump(outdir+'/'+epoch_observ[0]+'_fitmod_autofit.pkl')


    if autofit and (cont_abspeak or force_autofit) and not flag_lowSNR_line:

        # updating the logfile for the next type of computations
        curr_logfile_write = Xset.openLog(outdir + '/' + epoch_observ[0] + '_xspec_log_autofit_comput.log')
        curr_logfile_write.reconfigure(line_buffering=True)
        curr_logfile = open(curr_logfile_write.name, 'r')

        # updating it in the fitmod
        fitlines.logfile = curr_logfile
        fitlines.logfile_write = curr_logfile_write
        fitlines.update_fitcomps()

        '''
        Computing all the necessary info for the autofit plots and overlap tests
        Note: we only show the first (constant=1) model components because the others are identical modulo their respective constant factor 
        '''

        #storing the components of the model for the first data group only
        plot_autofit_data,plot_autofit_comps=store_plot('ldata',comps=True)
        plot_autofit_data=plot_autofit_data[0]
        plot_autofit_comps=plot_autofit_comps[0]

        #computing the names of the additive continuum components remaning in the autofit
        addcomps_cont=[comp.compname.split('_')[1] for comp in [elem for elem in fitlines.includedlist if elem is not None]\
                       if 'gaussian' not in comp.compname and not comp.multipl]

        #same with the lines
        addcomps_lines=[comp.compname.split('_')[0] for comp in [elem for elem in fitlines.includedlist if elem is not None]\
                        if 'gaussian' in comp.compname]

        addcomps_abslines=[comp.compname.split('_')[0] for comp in [elem for elem in fitlines.includedlist if elem is not None]\
                        if comp.named_absline]

        comp_absline_position=[]
        for id_comp_absline,comp_absline in enumerate(addcomps_abslines):
            comp_absline_position += [np.argwhere(np.array(addcomps_lines) == comp_absline)[0][0]]

        #rearranging the components in a format usable in the plot. The components start at the index 2
        #(before it's the entire model x and y values)
        plot_autofit_cont=plot_autofit_comps[:2+len(addcomps_cont)]

        #same for the line components
        plot_autofit_lines=plot_autofit_comps[2+len(addcomps_cont):]


        #taking off potential background components

        if 'nxb' in list(AllModels.sources.values()):
            plot_autofit_lines=plot_autofit_lines[:-2]

        if 'sky' in list(AllModels.sources.values()):
            plot_autofit_lines=plot_autofit_lines[:-2]

        '''
        #### Testing the overlap of absorption lines with emission lines
        We compare the overlapping area (in count space) of each absorption line to the sum of emission lines
        The final threshold is set in fraction of the absorption line area
        
        we fix the limit for emission lines to be considered as continuum (and thus not taken into account in this computation)
        when their width is more than 1/4 of the line detection range 
        This is the threshold for their 2 sigma width (or 95% of their area) to be under the line detection range
        '''

        #first, we compute the sum of the "narrow" emission line components
        sum_autofit_emlines=np.zeros(len(plot_autofit_cont[0]))

        included_linecomps=[elem for elem in [elem for elem in fitlines.includedlist if elem is not None] \
                            if 'gaussian' in elem.compname]

        for ind_line,line_fitcomp in enumerate(included_linecomps):

            #skipping absorption lines
            if line_fitcomp.named_absline:
                continue

            # #width limit test
            # of now that the line comps are not allowed to have stupid widths
            # if AllModels(1)(line_fitcomp.parlist[1])*8<(line_search_e[1]-line_search_e[0]):

            #adding the right component to the sum
            sum_autofit_emlines+=plot_autofit_lines[ind_line]

        abslines_em_overlap=np.zeros(n_absline)

        #and then the overlap for each absorption
        for ind_line,line_fitcomp in enumerate(included_linecomps):

            #skipping emission lines
            if not line_fitcomp.named_absline:
                continue

            #computing the overlapping bins
            overlap_bins=np.array([abs(plot_autofit_lines[ind_line]),sum_autofit_emlines]).min(0)

            #and the integral from that
            overlap_frac=trapezoid(overlap_bins,x=plot_autofit_cont[0])/\
                        abs(trapezoid(plot_autofit_lines[ind_line],x=plot_autofit_cont[0]))

            #fetching the index of the line being tested
            line_ind_std=np.argwhere(np.array(lines_std_names[3:])==line_fitcomp.compname.split('_')[0])[0][0]

            #and storing the value
            abslines_em_overlap[line_ind_std]=overlap_frac


        '''
        in order to compute the continuum ratio evolution we compute the sum of all the non absorption line components,
        then the ratio when adding each absorption line component
        '''

        plot_autofit_noabs=np.concatenate((([[plot_addline] for plot_addline in plot_autofit_comps[2:]\
                                             if not max(plot_addline)<=0]))).sum(axis=0)

        plot_autofit_ratio_lines=[(plot_autofit_noabs+plot_autofit_lines[i])/plot_autofit_noabs\
                                     for i in range(len(plot_autofit_lines)) if max(plot_autofit_lines[i])<=0.]

        '''
        Chain computation for the MC significance
        '''

        #drawing parameters for the MC significance test later
        autofit_drawpars=np.array([None]*nfakes)

        print_xlog('\nDrawing parameters from the Chain...')
        for i_draw in range(nfakes):

            curr_simpar=AllModels.simpars()

            #we restrict the simpar to the initial model because we don't really care about simulating the variations
            # of the bg since it's currently frozen
            autofit_drawpars[i_draw]=np.array(curr_simpar)[:AllData.nGroups*AllModels(1).nParameters]\
                                     .reshape(AllData.nGroups,AllModels(1).nParameters)

        #turning it back into a regular array
        autofit_drawpars=np.array([elem for elem in autofit_drawpars])

        #storing the parameter and errors of all the components, as well as their corresponding name
        autofit_parerrors,autofit_parnames=fitlines.get_usedpars_vals()

        print_xlog('\nComputing informations from the fit...')

        #### Computing line parameters

        #fetching informations about the absorption lines
        abslines_flux,abslines_eqw,abslines_bshift,abslines_delchi,abslines_bshift_distinct=fitlines.get_absline_info(autofit_drawpars)

        from xspec import AllChains

        #clearing the chain before doing anything else
        AllChains.clear()

        Fit.perform()

        #computing the 3-sigma width without the MC to avoid issues with values being too different from 0
        abslines_width=fitlines.get_absline_width()

        '''
        Saving a "continuum" version of the model without absorption
        '''

        #this first part is to get the mask to reduce autofit_drawpars_cont to its version without
        #the abslines

        #We store the indexes of the absgaussian parameters to shift the rest of the parameters accordingly after
        #deleting those components
        abslines_parsets=np.array([elem.parlist for elem in fitlines.includedlist if elem is not None and elem.named_absline])

        abslines_parids=ravel_ragged(abslines_parsets)

        #covering for the case where the list was mono-dimensional
        if type(abslines_parids)!=list:
            abslines_parids=abslines_parids.tolist()

        #switching to indexes instead of actual parameter numbers
        abslines_arrids=[elem-1 for elem in abslines_parids]

        #creating the list of continuum array indexes (note that this will be applied for each datagroups)
        #this is for A SINGLE DATAGROUP because autofit_drawpars has a adatagroup dimension
        continuum_arrids=[elem for elem in np.arange(AllModels(1).nParameters) if elem not in abslines_arrids]

        #and parameters
        continuum_parids=[elem+1 for elem in continuum_arrids]

        #to create a drawpar array which works with thereduced model (this line works with several data groups)
        autofit_drawpars_cont=autofit_drawpars.T[continuum_arrids].T

        #and now delete the abslines to properly get the forced continuum parameters in the 'no-abs' state
        full_includedlist=fitlines.includedlist

        #deleting all absorption components (reversed so we don't have to update the fitcomps)
        for comp in [elem for elem in fitlines.includedlist if elem is not None][::-1]:
            if comp.named_absline:
                #note that with no rollback we do not update the methods of the component
                # so it has still its included status and everything else
                ndel=comp.delfrommod(rollback=False)
                fitlines.includedlist=fitlines.includedlist[:-ndel]
                fitlines.update_fitcomps()

        #storing the no abs line 'continuum' model
        data_autofit_noabs=allmodel_data()

        #recording all the frozen/linked parameters in the model to refreeze them during the fakeit fit
        #note that here we don't care about how things are linked, just freezing them to their drawpared value
        #is enough because the drawpar itself will have been generated with the correct link
        continuum_forcedpars=[i_par for i_par in range(1,AllData.nGroups*AllModels(1).nParameters+1)\
                            if AllModels(par_degroup(i_par)[0])(par_degroup(i_par)[1]).frozen\
                            or AllModels(par_degroup(i_par)[0])(par_degroup(i_par)[1]).link!='']

        continuu_ranges=[]
        plot_ratio_autofit_noabs=store_plot('ratio')

        #resetting the states of fitlines
        fitlines.includedlist=full_includedlist
        data_autofit.load()
        fitlines.update_fitcomps()

        #and comming back to the noabs state
        data_autofit_noabs.load()

        #plotting the combined autofit plot

        def autofit_plot(data,data_noabs,addcomps_cont,comp_pos):

            '''
            The goal here is to plot the autofit in a way that shows the different lines
            '''

            gs_comb=GridSpec(2,1,hspace=0.)

            axes=[None,None]
            #first subplot is the ratio
            axes[0]=plt.subplot(gs_comb[0,0])
            axes[1]=plt.subplot(gs_comb[1,0])

            '''first plot (components)'''

            #We only plot this for the first data group, no need to show all of them since the only difference is a constant factor
            plot_line_comps(axes[0],plot_autofit_cont,addcomps_cont,plot_autofit_lines,addcomps_lines,combined=True)

            '''second plot (ratio + abslines ratio)'''

            plot_line_ratio(axes[1],data_autofit=data,data_autofit_noabs=data_noabs,
                            n_addcomps_cont=len(addcomps_cont),line_position=comp_pos,
                            line_search_e=line_search_e,line_cont_range=line_cont_range)

            plt.tight_layout()

        fig_autofit=plt.figure(figsize=(15,10))

        autofit_plot(data=data_autofit,data_noabs=data_autofit_noabs,addcomps_cont=addcomps_cont,
                     comp_pos=comp_absline_position)

        plt.savefig(outdir+'/'+epoch_observ[0]+'_autofit_components_plot_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png')
        plt.close(fig_autofit)

        '''
        Autofit residuals assessment
        '''

        chi_dict_autofit=narrow_line_search(data_autofit,'autofit',
                                            line_search_e=line_search_e,line_search_norm=line_search_norm,
                           e_sat_low_indiv=e_sat_low_indiv[mask_nodeload],
                                            peak_thresh=peak_thresh,peak_clean=peak_clean,
                           line_cont_range=line_cont_range,trig_interval=trig_interval,
                           scorpeon_save=data_broad.scorpeon,data_fluxcont=data_autofit_noabs)

        with open(outdir+'/'+epoch_observ[0]+'_chi_dict_autofit.pkl','wb') as file:
            dill.dump(chi_dict_init,file)

        plot_line_search(chi_dict_autofit,outdir,sat_glob,suffix='autofit',epoch_observ=epoch_observ)

        ####Paper plot

        def paper_plot(fig_paper,chi_dict_init,chi_dict_postauto,title=None):

            line_cont_range=chi_dict_init['line_cont_range']
            ax_paper=np.array([None]*4)
            fig_paper.suptitle(title)

            #gridspec creates a grid of spaces for subplots. We use 4 rows for the 4 plots
            #Second column is there to keep space for the colorbar. Hspace=0. sticks the plots together
            gs_paper=GridSpec(4,2,figure=fig_paper,width_ratios=[100,0],hspace=0.)

            #first plot is the data with additive components
            ax_paper[0]=plt.subplot(gs_paper[0,0])
            prev_plot_add=Plot.add
            Plot.add=True

            #reloading the pre-autofit continuum for display
            data_mod_high.load()

            xPlot('ldata',axes_input=ax_paper[0])

            #loading the no abs autofit
            data_autofit_noabs.load()

            Plot.add=prev_plot_add

            #second plot is the first blind search coltour
            ax_paper[1]=plt.subplot(gs_paper[1,0],sharex=ax_paper[0])
            ax_colorbar=plt.subplot(gs_paper[1,1])
            coltour_chi2map(fig_paper,ax_paper[1],chi_dict_init,combined='paper',ax_bar=ax_colorbar)
            ax_paper[1].set_xlim(line_cont_range)

            ax_paper[2]=plt.subplot(gs_paper[2,0],sharex=ax_paper[0])
            #third plot is the autofit ratio with lines added
            plot_line_ratio(ax_paper[2],mode='paper',data_autofit=data_autofit,data_autofit_noabs=data_autofit_noabs,
                            n_addcomps_cont=len(addcomps_cont),line_position=comp_absline_position,
                            line_search_e=line_search_e,line_cont_range=line_cont_range)

            #fourth plot is the second blind search coltour
            ax_paper[3]=plt.subplot(gs_paper[3,0],sharex=ax_paper[0])
            ax_colorbar=plt.subplot(gs_paper[3,1])

            # coltour_chi2map(fig_paper,ax_paper[3],chi_dict_postauto,combined='nolegend',ax_bar='bottom',norm=(251.5,12.6))

            #need to fix the colorbar here
            coltour_chi2map(fig_paper,ax_paper[3],chi_dict_postauto,combined='nolegend',ax_bar=ax_colorbar)

            ax_paper[3].set_xlim(line_cont_range)

            plot_std_ener(ax_paper[1], mode='chimap', plot_em=True)
            plot_std_ener(ax_paper[2], plot_em=True)
            plot_std_ener(ax_paper[3], mode='chimap', plot_em=True)

            # taking off the x axis of the first 3 axis to avoid ugly stuff
            for ax in ax_paper[:3]:
                ax.xaxis.set_visible(False)

            # adding panel names
            for ax, name in zip(ax_paper, ['A', 'B', 'C', 'D']):
                ax.text(0.02, 0.05, name, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=25)

        fig_paper=plt.figure(figsize=(14.5,22))

        paper_plot(fig_paper,chi_dict_init,chi_dict_autofit)

        plt.savefig(outdir+'/'+epoch_observ[0]+'_paper_plot_'+args.line_search_e.replace(' ','_')+
                    '_'+args.line_search_norm.replace(' ','_')+'.png')
        plt.savefig(outdir+'/'+epoch_observ[0]+'_paper_plot_'+args.line_search_e.replace(' ','_')+
                    '_'+args.line_search_norm.replace(' ','_')+'.pdf')

        plt.close(fig_paper)

        data_autofit_noabs.load()

        #we don't update the fitcomps here because it would require taking off the abslines from the includedlists
        #and we don't want that for the significance computation

        '''
        Absorption lines statistical significance assessment
        
        The goal here is to test the statistical significance of the data feature which we interpret as lines 
        (not the significance of adding a component to the model to describe it)
        
        Thus, we use fakeit to simulate a great number of fake spectra, which we then fit with the saved continuum model, 
        before fetching the biggest delchis possible with narrow gaussians in the fake spectrum
        
        We use the continuum in autofit to test the new lines 
        
        The standard condition for starting the process is if assess line is set to true AND if there is an absorption line in the autofit 
        We thus gain time when the initial peak detection was a fluke or we forced the autofit without an absorption, 
        in which cases there's no need to compute the fakes

        '''

        #### Significance assessment

        #computing an array of significantly non-zero widths
        sign_widths_arr=np.array([elem[0] if elem[0]-elem[1]>1e-6 else 0 for elem in abslines_width])

        abslines_sign=np.zeros((len(abslines_delchi)))
        abslines_eqw_upper=np.zeros((len(abslines_delchi)))

        is_absline=np.array([included_comp.named_absline for included_comp in \
                                     [comp for comp in fitlines.includedlist if comp is not None]]).any()

        #updating the logfile for the second round of fitting
        curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log_fakeits.log')
        curr_logfile_write.reconfigure(line_buffering=True)
        curr_logfile=open(curr_logfile_write.name,'r')

        #updating it in the fitmod
        fitlines.logfile=curr_logfile
        fitlines.logfile_write=curr_logfile_write

        Xset.logChatter=2

        '''
        In order to make the fakeit process as fast as possible, we don't write the fakes in file 
        and directly fit them as loaded every time
        For that, we specify fakeitsettings with given filename, bg, resp etc. 
        (since by default fakeit overwrite those while loading the spectra)
        and use the nowrite command to keep from overwriting the files
        By not giving an exposure, we assume the loaded spectra's exposures
        '''

        print_xlog('\nCreating fake spectra to assess line significance...')

        fakeset=[FakeitSettings(response='' if not isrmf_grp[i_grp-1] else AllData(i_grp).response.rmf,
                                arf='' if not isarf_grp[i_grp-1] else AllData(i_grp).response.arf,
                                background='' if not isbg_grp[i_grp-1] else AllData(i_grp).background.fileName,
                                fileName=AllData(i_grp).fileName) for i_grp in range(1,AllData.nGroups+1)]

        #array for delchi storing
        delchi_arr_fake=np.zeros((nfakes,round((line_search_e_space[-1]-line_search_e_space[0])/line_search_e[2])+1))
        delchi_arr_fake_line=np.zeros((6,nfakes))
        # eqw_arr_fake=np.zeros((nfakes,6))

        steppar_ind_list=[]

        line_id_list=[]

        '''
        Since we now have specific energy intervals for each line, we can restrict the delchi test to the interval
        of each line. Thus, to optimize computing time, we compute which indexes need to be computed for all lines
        and compute steppars for each interval among those indexes
        '''

        #assessing the range of steppar to use for of each line
        for i_line in range(len(abslines_sign)):

            #skipping the computation for lines above 8 keV in restrict mode when we don't go above 8keV anyway
            if restrict_graded and i_line>=2:
                continue

            #here we skip the first two emission lines
            line_name=list(lines_e_dict.keys())[i_line+3]

            #fetching the lower and upper bounds of the energies from the blueshifts
            #here we add a failsafe for the upper part of the steppar to avoid going beyond the energies ignored which crashes it

            line_lower_e=lines_e_dict[line_name][0]*(1+lines_e_dict[line_name][1]/c_light)
            line_upper_e=min(lines_e_dict[line_name][0]*(1+lines_e_dict[line_name][2]/c_light),max(e_sat_high_indiv))

            #computing the corresponding indexes in the delchi array
            line_lower_ind=int((line_lower_e-line_search_e_space[0])//line_search_e[2])
            line_upper_ind=int((line_upper_e-line_search_e_space[0])//line_search_e[2]+1)

            #skipping the interval if the line has not been detected
            if abslines_eqw[i_line][0]==0:
                continue

            #adding the parts of the line_search_e_space which need to be computed to an array
            steppar_ind_list+=np.arange(line_lower_ind,line_upper_ind+1).tolist()

            #adding the index to the list of line indexes to be tested
            line_id_list+=[i_line]

        loaded_fakes=False

        if is_absline and assess_line:

            # attempting to reload the fake delchi arr if allowed
            if reload_fakes and os.path.isfile(outdir + '/' + epoch_observ[0] + '_delchi_arr_fake_line.npy'):

                print_xlog('Complete fake computation detected. Reloading...')

                delchi_arr_fake_line=np.load(outdir + '/' + epoch_observ[0] + '_delchi_arr_fake_line.npy')
                loaded_fakes=True

            if not loaded_fakes:

                #now we compute the list of intervals that can be made from that
                steppar_ind_unique=np.unique(steppar_ind_list)

                steppar_ind_inter=list(interval_extract(steppar_ind_unique))

                #fake loop
                with tqdm(total=nfakes) as pbar:
                    for f_ind in range(nfakes):

                        #reloading the high energy continuum
                        mod_fake=data_autofit_noabs.load()

                        #Freezing it to ensure the fakeit doesn't make the parameters vary, and loading them from a steppar
                        for i_grp in range(1,AllData.nGroups+1):

                            #freezing doesn't change anything for linked parameters
                            freeze(AllModels(i_grp))

                            #computing which setpars need to be adjusted from the list
                            #in the case where the parsed value is out of the default continuum range

                            #skipping the drawpar elements which would give things already nomally linked,
                            #to avoid issues with parameter ranges ('' is the default value for SetPars)
                            curr_group_drawpar=autofit_drawpars_cont[f_ind][i_grp-1].tolist()
                            for group_par in range(1,AllModels(1).nParameters+1):
                                if AllModels(i_grp)(group_par).link!='' and \
                                        int(AllModels(i_grp)(group_par).link.replace('= p',''))==group_par:
                                    curr_group_drawpar[group_par-1]=''

                            AllModels(i_grp).setPars(curr_group_drawpar)

                        #replacing the current spectra with a fake with the same characteristics so this can be looped
                        #applyStats is set to true but shouldn't matter for now since everything is frozen

                        AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)

                        # limiting to the line search energy range
                        ignore_data_indiv(line_cont_range[0],line_cont_range[1], reset=True,
                                          sat_low_groups=e_sat_low_indiv[mask_nodeload],
                                          sat_high_groups=e_sat_high_indiv[mask_nodeload],
                                          glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

                        #adjusting the fit and storing the chi²

                        for i_grp in range(1,AllData.nGroups+1):
                            #unfreezing the model
                            unfreeze(AllModels(i_grp))

                        for par_forced in continuum_forcedpars:
                            AllModels(par_degroup(par_forced)[0])(par_degroup(par_forced)[1]).frozen=True

                            #previous method which wouldn't work with more complicated unlinks and free pars
                            #for multigroup models
                            # #keeping the initially frozen parameters frozen
                            # freeze(AllModels(i_grp),parlist=continuum_forcedpars)
                            #
                            # #keeping only the first constant factor frozen if necessary
                            # if i_grp>1 and AllModels(1).componentNames[0]=='constant':
                            #     AllModels(i_grp)(1).frozen=False


                        #no error computation to avoid humongus computation times
                        calc_fit(nonew=True,noprint=True)

                        for i_grp in range(1,AllData.nGroups+1):
                            #freezing the model again since we want to fit only specific parameters afterwards
                            freeze(AllModels(i_grp))

                        '''
                        Now we search for residual lines. We use an energy grid steppar with free normalisation set at 0 
                        The steppar will fit the free parameter at each energy for us
                        '''

                        #adding a narrow gaussian
                        mod_fake=addcomp('nagaussian')

                        #computing a steppar for each element of the list
                        Xset.chatter=0
                        Xset.logChatter=0

                        for steppar_inter in steppar_ind_inter:

                            #giving the width value of the corresponding line before computing the steppar
                            AllModels(1)(AllModels(1).nParameters-1).values=[sign_widths_arr[0]]+AllModels(1)(AllModels(1).nParameters-1).values[1:]

                            #exploring the parameters
                            try:
                                Fit.steppar('nolog '+str(mod_fake.nParameters-2)+' '+str(round(line_search_e_space[steppar_inter[0]],3))+\
                                            ' '+str(round(line_search_e_space[steppar_inter[1]],3))+' '\
                                           +str(steppar_inter[1]-steppar_inter[0]))

                                #updating the delchi array with the part of the parameters that got updated
                                delchi_arr_fake[f_ind][steppar_inter[0]:steppar_inter[1]+1]=\
                                    abs(np.array([min(elem,0) for elem in Fit.stepparResults('delstat')]))
                            except:
                                #can happen if there are issues in the data quality, we just don't consider the fakes then
                                pass

                        Xset.chatter=xchatter
                        Xset.logChatter=5

                        pbar.update(1)

            #assessing the significance of each line
            for i_line in range(len(abslines_sign)):

                '''
                Now we just compute the indexes corresponding to the lower and upper bound of each line's interval and compute the 
                probability from this space only (via a transposition)
                '''

                #skipping the computation for lines above 8 keV in restrict mode when we don't go above 8keV anyway
                if restrict_graded and i_line>=2:
                    continue

                #here we skip the first two emission lines
                line_name=list(lines_e_dict.keys())[i_line+3]

                #fetching the lower and upper bounds of the energies from the blueshifts
                line_lower_e=lines_e_dict[line_name][0]*(1+lines_e_dict[line_name][1]/c_light)
                line_upper_e=lines_e_dict[line_name][0]*(1+lines_e_dict[line_name][2]/c_light)

                #computing the corresponding indexes in the delchi array
                line_lower_ind=int((line_lower_e-line_search_e_space[0])//line_search_e[2])
                line_upper_ind=int((line_upper_e-line_search_e_space[0])//line_search_e[2]+1)

                #remaking delchi_arr_fake_line if it wasn't loaded
                if not loaded_fakes:
                    #restricting the array to those indexes
                    #we use max evaluation here because it could potentially lead to underestimating the significance
                    #if more than 1 delchi element in an iteration are above the chi threshold
                    delchi_arr_fake_line[i_line]=delchi_arr_fake.T[line_lower_ind:line_upper_ind+1].T.max(1)

                #we round to keep the precision to a logical value
                #we also add a condition to keep the significance at 0 when there's no line in order to avoid problems
                abslines_sign[i_line]=round(1-len(delchi_arr_fake_line[i_line][delchi_arr_fake_line[i_line]>abslines_delchi[i_line]])/nfakes,
                                            len(str(nfakes))) if abslines_delchi[i_line]!=0 else 0

                #giving the line a significance attribute
                line_comp=[comp for comp in [elem for elem in fitlines.complist if elem is not None] if line_name in comp.compname][0]

                line_comp.significant=abslines_sign[i_line]

                '''
                computing the UL for detectability at the given threshold
                Here we convert the delchi threshold for significance to the EW that we would obtain for a line at the maximum blueshift 
                '''

        #reloading the initial spectra for any following computations
        Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm')

        #and saving the delchi array if it wasn't already done previously
        if assess_line and not loaded_fakes:
            np.save(outdir+'/'+epoch_observ[0]+'_delchi_arr_fake_line.npy',delchi_arr_fake_line)

        '''
        ####Line fit upper limits
        '''

        #reloading the autofit model with no absorption to compute the upper limits
        data_autofit_noabs.load()

        #freezing the model to avoid it being affected by the missing absorption lines
        #note : it would be better to let it free when no absorption lines are there but we keep the same procedure for
        #consistency
        allfreeze()

        #computing a mask for significant lines
        mask_abslines_sign=abslines_sign>=sign_threshold

        if assess_line_upper:
            #computing the upper limits for the non significant lines
            abslines_eqw_upper=fitlines.get_eqwidth_uls(mask_abslines_sign,abslines_bshift,sign_widths_arr,pre_delete=True)

        #here will need to reload an accurate model before updating the fitcomps
        '''HTML TABLE FOR the pdf summary'''

        abslines_table_str=html_table_maker()

        with open(outdir+'/'+epoch_observ[0]+'_abslines_table.txt','w+') as abslines_table_file:
            abslines_table_file.write(abslines_table_str)

        def latex_table_maker():

            '''
            to be done
            '''

            def latex_value_maker(value_arr,is_shift=False):

                '''
                wrapper for making a latex-proof of the line abs values

                set is_shift to true for energy/blueshift values, for which 0 values or low uncertainties equal to the value
                are sign of being pegged to the blueshift limit
                '''

                #the first case is for eqwidths and blueshifts (argument is an array with the uncertainties)
                if type(value_arr)==np.ndarray:
                    #If the value array is entirely empty, it means the line is not detected and thus we put a different string
                    if len(np.nonzero(value_arr)[0])==0:
                        newstr='/'
                    else:
                        if is_shift==True and value_arr[1]=='linked':
                            #we do not show uncertainties for the linked parameters since it is just a repeat
                            newstr=str(round(value_arr[0],2))

                        else:
                            newstr=str(round(value_arr[0],2))+' -'+str(round(value_arr[1],2))+' +'+str(round(value_arr[2],2))

                #the second case is for the significance, which is a float
                else:
                    #same empty test
                    if value_arr==0:
                        newstr='/'
                    else:
                        newstr=(str(round(100*value_arr,len(str(nfakes)))) if value_arr!=1 else '>'+str((1-1/nfakes)*100))+'%'

                return newstr

            Xset.logChatter=10

        #storing line string
        autofit_store_str=epoch_observ[0]+'\t'+\
            str(abslines_eqw.tolist())+'\t'+str(abslines_bshift.tolist())+'\t'+str(abslines_delchi.tolist())+'\t'+\
            str(abslines_flux.tolist())+'\t'+str(abslines_sign.tolist())+'\t'+str(abslines_eqw_upper.tolist())+'\t'+\
            str(abslines_em_overlap.tolist())+'\t'+str(abslines_width.tolist())+'\t'+str(abslines_bshift_distinct.tolist())+'\t'+\
            str(autofit_parerrors.tolist())+'\t'+str(autofit_parnames.tolist())+'\n'

    else:
        autofit_store_str=epoch_observ[0]+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\n'


    '''Storing the results'''

    #### result storing

    #we test for the low SNR flag to ensure not overwriting line results for good quality data by mistake if launching the script without autofit
    if autofit or flag_lowSNR_line:

        autofit_store_header='Observ_id\tabslines_eqw\tabslines_bshift\tablines_delchi\tabslines_flux\t'+\
        'abslines_sign\tabslines_eqw_upper\tabslines_em_overlap\tabslines_width\tabslines_bshift_distinct'+\
            '\tautofit_parerrors\tautofit_parnames\n'

        file_edit(path=autofit_store_path,line_id=epoch_observ[0],line_data=autofit_store_str,header=autofit_store_header)

    if len(cont_peak_points)!=0:
        line_str=epoch_observ[0]+'\t'+str(cont_peak_points.T[0].tolist())+'\t'+str(cont_peak_points.T[1].tolist())+'\t'+\
                              str(cont_peak_widths.tolist())+'\t'+str(cont_peak_delchis)+'\t'+str(cont_peak_eqws)+'\t'+\
                              '\t'+str(main_spflux.tolist())+'\n'
    else:
        line_str=epoch_observ[0]+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+str(main_spflux.tolist())+'\n'

    line_store_header='Observ_id\tpeak_e\tpeak_norm\tpeak_widths\tpeak_delchis\tpeak_eqwidths\tbroad_flux\n'

    file_edit(path=line_store_path,line_id=epoch_observ[0],line_data=line_str,header=line_store_header)

    '''PDF creation'''

    if write_pdf:

        pdf_summary(epoch_files,fit_ok=True,summary_epoch=fill_result('Line detection complete.'),
                    e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

    #closing the logfile for both access and Xspec
    curr_logfile.close()
    Xset.closeLog()

    return fill_result('Line detection complete.')

'''
#### Epoch matching
'''

if sat_glob=='XMM':
    spfile_dates=np.array([[None,None]]*len(spfile_list))

    #storing the dates of all the exposures
    for file_index,elem_sp in enumerate(spfile_list):
        with fits.open(elem_sp) as fits_spec:
            spfile_dates[file_index][0]=Time(fits_spec[0].header['DATE-OBS'])
            spfile_dates[file_index][1]=Time(fits_spec[0].header['DATE-END'])

    def overlap_fraction(dates_1,dates_2):
        duration_1=dates_1[1]-dates_1[0]
        duration_2=dates_2[1]-dates_2[0]

        max_overlap=max(0,min(dates_1[1],dates_2[1])-max(dates_1[0],dates_2[0]))

        return max(max_overlap/duration_1,max_overlap/duration_2)

    epoch_list=[]
    #and matching them
    while len(ravel_ragged(epoch_list))!=len(spfile_list):

        elem_epoch=[]

        #taking a new spectrum
        curr_sp_id,curr_sp=[[i,spfile_list[i]] for i in range(len(spfile_list)) if spfile_list[i] not in ravel_ragged(epoch_list)][0]

        #adding it to a new epoch
        elem_epoch+=[curr_sp]

        #testing all remaining spectrum for overlap
        #we do this incrementally to test overlap between with all the spectra in the epoch
        id_ep=0
        while id_ep<len(elem_epoch):
            curr_tested_epoch=elem_epoch[id_ep]
            curr_tested_epoch_id=np.argwhere(spfile_list==curr_tested_epoch)[0][0]
            for elem_sp_id,elem_sp in [[i,spfile_list[i]] for i in range(len(spfile_list)) if
                                       (spfile_list[i] not in ravel_ragged(epoch_list) and spfile_list[i] not in elem_epoch)]:
                #fetching the index of each
                if overlap_fraction(spfile_dates[curr_tested_epoch_id],spfile_dates[elem_sp_id]).value>0.5:
                    elem_epoch+=[elem_sp]
            id_ep+=1

        '''
        ordering the epoch files with pn, mos1, mos2 (or any part of this)
        '''
        elem_epoch_sorted=[]
        for cam in ['pn','mos1','mos2']:
            for elem_sp in elem_epoch:
                if elem_sp.split('_')[1]==cam:
                    elem_epoch_sorted+=[elem_sp]

        epoch_list+=[elem_epoch_sorted]

elif sat_glob=='Chandra':
    epoch_list=[]
    epoch_list_started=[]
    obsid_list_chandra=np.unique([elem.split('_')[0] for elem in spfile_list])
    for obsid in obsid_list_chandra:
        epoch_list+=[[elem for elem in spfile_list if elem.startswith(obsid)]]

    obsid_list_started_chandra=np.unique([elem.split('_')[0] for elem in started_expos[1:]])
    for obsid in obsid_list_started_chandra.tolist():
        epoch_list_started+=[[elem,elem.replace('-1','1')] for elem in started_expos if elem.startswith(obsid)]

elif sat_glob=='NICER':
    epoch_list=[]
    tstart_list=[]
    for elem_file in spfile_list:
        with fits.open(elem_file) as hdul:

            start_obs_s = hdul[1].header['TSTART'] + hdul[1].header['TIMEZERO']

            # saving for titles later
            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')

        tstart_list+=[obs_start.to_value('jd')]

    #max delta between gti starts in sec
    max_delta=(TimeDelta(group_max_timedelta.split('_')[0],format='jd')+\
              TimeDelta(group_max_timedelta.split('_')[1],format='jd')/24+ \
              TimeDelta(group_max_timedelta.split('_')[2], format='jd')/(24*60)+ \
              TimeDelta(group_max_timedelta.split('_')[3], format='jd')/(24*3600)+ \
              TimeDelta(group_max_timedelta.split('_')[4], format='jd')/(24*3600*1e3)).to_value('jd')

    epoch_id_list_ravel=[]
    epoch_id_list=[]

    with tqdm(total=len(tstart_list)) as pbar:
        for id_elem,elem_tstart in enumerate(tstart_list):

            #skipping computation for already grouped elements
            if id_elem in epoch_id_list_ravel:
                continue

            elem_delta=[(elem-elem_tstart) for elem in tstart_list]

            elem_epoch_id=[id for id in range(len(tstart_list)) if\
                                     id not in epoch_id_list_ravel and elem_delta[id]>=0 and elem_delta[id]<max_delta]

            epoch_id_list_ravel+=elem_epoch_id

            if len(elem_epoch_id)>0:
                epoch_id_list+=[elem_epoch_id]

            pbar.update(n=len(elem_epoch_id))

    epoch_list=[spfile_list[elem] for elem in epoch_id_list]


    #skipping flares if asked for
    if skip_flares:
        epoch_list=[[subelem for subelem in elem if "F_sp" not in subelem] for elem in epoch_list]
        epoch_list=[elem for elem in epoch_list if len(elem)>0]

    epoch_list=np.array(epoch_list,dtype=object)

    #not needed atm
    # def str_to_epoch(str_epoch):
    #     str_epoch_list=[]
    #     for elem_obsid_str in str_epoch:
    #         if '-' not in elem_obsid_str:
    #             str_epoch_list+=elem_obsid_str
    #         else:
    #             str_epoch_list+=[elem_obsid_str.split('-')[0]+elem_obsid_str.split('-')[i]\
    #                              for i in range(1,len(elem_obsid_str.split('-')))]

    epoch_list_started=started_expos
    epoch_list_done=done_expos

elif sat_glob in ['Suzaku','Swift']:
    epoch_list=[]
    epoch_list_started=[]
    if sat_glob=='Swift':
        obsid_list=np.unique([elem[:11] for elem in spfile_list])
    else:
        obsid_list=np.unique([elem.split('_')[0] for elem in spfile_list])

    for obsid in obsid_list:

        epoch_list+=[[elem for elem in spfile_list if elem.startswith(obsid+'_')]]

    if sat_glob=='Swift':
        obsid_list_started=np.unique([elem.split('_')[0][:11] for elem in started_expos[1:]])
        for obsid in obsid_list_started.tolist():
            epoch_list_started+=[[elem] for elem in started_expos if elem.startswith(obsid)]

    if sat_glob=='Suzaku':
        #reversing the order to have the FI xis first, then the BI xis, then pin instead of the opposite
        epoch_list=[elem[::-1] for elem in epoch_list]

        epoch_list_started=[literal_eval(elem.split(']')[0]+']') for elem in started_expos[1:]]

elif sat_glob=='NuSTAR':

    epoch_list = []

    # skipping flares if asked for
    if skip_flares:
        spfile_list = np.array([elem for elem in spfile_list if "F_sp" not in elem])

    tstart_list = np.array([None] * len(spfile_list))
    det_list = np.array([None] * len(spfile_list))
    tstop_list = np.array([None] * len(spfile_list))

    for i_file, elem_file in enumerate(spfile_list):

        # for Suzaku this won't work for meugmi's xis0_xis3 files bc their header has been replaced
        # so we replace them by the xis1 to be able to load the exposure
        elem_file_load = elem_file.replace('xis0_xis3', 'xis1')

        with fits.open(elem_file_load) as hdul:
            if 'TELESCOP' in hdul[1].header:
                det_list[i_file] = hdul[1].header['TELESCOP'].replace('SUZAKU', 'Suzaku')
            else:
                # the only files without TELESCOP in the header should be the fused megumi_files suzaku sp
                assert megumi_files, 'Issue with detector handling'

                det_list[i_file] = 'Suzaku'

            if 'TIMEZERO' in hdul[1].header:
                start_obs_s = hdul[1].header['TSTART'] + hdul[1].header['TIMEZERO']
                stop_obs_s = hdul[1].header['TSTOP'] + hdul[1].header['TIMEZERO']
            else:
                start_obs_s = hdul[1].header['TSTART']
                stop_obs_s = hdul[1].header['TSTOP']
            # saving for titles later
            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')
            obs_stop = mjd_ref + TimeDelta(stop_obs_s, format='sec')

        tstart_list[i_file] = obs_start.to_value('jd')
        tstop_list[i_file] = obs_stop.to_value('jd')

    # max delta between gti starts in sec
    max_delta=(TimeDelta(group_max_timedelta.split('_')[0],format='jd')+\
              TimeDelta(group_max_timedelta.split('_')[1],format='jd')/24+ \
              TimeDelta(group_max_timedelta.split('_')[2], format='jd')/(24*60)+ \
              TimeDelta(group_max_timedelta.split('_')[3], format='jd')/(24*3600)+ \
              TimeDelta(group_max_timedelta.split('_')[4], format='jd')/(24*3600*1e3)).to_value('jd')

    epoch_id_list_ravel = []
    epoch_id_list = []

    tstart_list_base = tstart_list
    tstop_list_base = tstop_list
    det_list_base = det_list
    id_base = np.arange(len(spfile_list))

    with tqdm(total=len(tstart_list)) as pbar:
        for id_elem, (elem_tstart, elem_tstop, elem_det) in enumerate(
                zip(tstart_list_base, tstop_list_base, det_list_base)):

            # skipping computation for already grouped elements
            if id_base[id_elem] in epoch_id_list_ravel:
                continue

            elem_delta = np.array([-get_overlap([elem_tstart, elem_tstop], [other_start, other_stop], distance=True) for
                                   other_start, other_stop in zip(tstart_list, tstop_list)])

            # list of matchable epochs
            elem_epoch_id = np.array([id for id in range(len(tstart_list)) if \
                                      id not in epoch_id_list_ravel and elem_delta[id] < max_delta])

            epoch_id_list_ravel += elem_epoch_id.tolist()

            if len(elem_epoch_id) > 0:
                epoch_id_list += [elem_epoch_id]

            pbar.update(n=len(elem_epoch_id))

    epoch_list = [spfile_list[elem] for elem in epoch_id_list]

    epoch_list = np.array(epoch_list, dtype=object)

    epoch_list_started = started_expos
    epoch_list_done = done_expos

elif sat_glob=='multi':
    epoch_list=[]

    #skipping flares if asked for
    if skip_flares:
        spfile_list=np.array([elem for elem in spfile_list if "F_sp" not in elem])

    tstart_list=np.array([None]*len(spfile_list))
    det_list=np.array([None]*len(spfile_list))
    tstop_list=np.array([None]*len(spfile_list))

    for i_file,elem_file in enumerate(spfile_list):

        #for Suzaku this won't work for meugmi's xis0_xis3 files bc their header has been replaced
        # so we replace them by the xis1 to be able to load the exposure
        elem_file_load=elem_file.replace('xis0_xis3','xis1')

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
            mjd_ref = Time(hdul[1].header['MJDREFI'] + hdul[1].header['MJDREFF'], format='mjd')

            obs_start = mjd_ref + TimeDelta(start_obs_s, format='sec')
            obs_stop=mjd_ref+TimeDelta(stop_obs_s,format='sec')

        tstart_list[i_file]=obs_start.to_value('jd')
        tstop_list[i_file]=obs_stop.to_value('jd')

    #max delta between gti starts in sec
    max_delta=(TimeDelta(group_max_timedelta.split('_')[0],format='jd')+\
              TimeDelta(group_max_timedelta.split('_')[1],format='jd')/24+ \
              TimeDelta(group_max_timedelta.split('_')[2], format='jd')/(24*60)+ \
              TimeDelta(group_max_timedelta.split('_')[3], format='jd')/(24*3600)+ \
              TimeDelta(group_max_timedelta.split('_')[4], format='jd')/(24*3600*1e3)).to_value('jd')

    epoch_id_list_ravel=[]
    epoch_id_list=[]

    if multi_focus!=False:
        #restricting the match epochs to a specific satellite
        mask_multi_focus=[elem==multi_focus for elem in det_list]
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

            elem_delta=np.array([-get_overlap([elem_tstart,elem_tstop],[other_start,other_stop],distance=True) for other_start,other_stop in zip(tstart_list,tstop_list)])

            #list of matchable epochs
            elem_epoch_id=np.array([id for id in range(len(tstart_list)) if\
                                     id not in epoch_id_list_ravel and elem_delta[id]<max_delta])

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

            epoch_id_list_ravel+=elem_epoch_id

            if len(elem_epoch_id)>0:
                epoch_id_list+=[elem_epoch_id]

            pbar.update(n=len(elem_epoch_id))

    epoch_list=[spfile_list[elem] for elem in epoch_id_list]

    epoch_list=np.array(epoch_list,dtype=object)

    if restrict_combination:
        epoch_list=[epoch_list[id_epoch] for id_epoch in range(len(epoch_list)) if (np.unique(det_list[epoch_id_list[id_epoch]])==restrict_combination.split('+')).all()]

    if plot_multi_overlap:

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

            num_dates_start = mdates.date2num(Time(tstart_list.astype(float)[epoch_mask], format='jd').datetime)
            num_dates_stop = mdates.date2num(Time(tstop_list.astype(float)[epoch_mask], format='jd').datetime)

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

            for i_fome,elem_file in enumerate(elem_epoch):

                num_date_start_file=num_dates_start[spfile_list[epoch_mask]==elem_file][0]
                num_date_stop_file=num_dates_stop[spfile_list[epoch_mask]==elem_file][0]

                ax_exp.axvspan(xmin=num_date_start_file,
                               xmax=num_date_stop_file,
                               ymin=0.5, ymax=1, color=epoch_color,
                               label='', alpha=0.4)

            ax_exp.xaxis.set_major_formatter(date_format)
            for label in ax_exp.get_xticklabels(which='major'):
                label.set(rotation=45, horizontalalignment='right')

            #locking the x axis with a one day margin around the epoch
            ax_exp.set_xlim(ax_exp.get_xlim()[0]-1, ax_exp.get_xlim()[1]+1)

            #plotting the rest of the exposures (cheap way to show them without having to sort them
            mask_tel = np.array([np.array(det_list) == elem for elem in tel_col_list])

            num_dates_start = mdates.date2num(Time(tstart_list.astype(float), format='jd').datetime)
            num_dates_stop = mdates.date2num(Time(tstop_list.astype(float), format='jd').datetime)

            #cylcing through each telescope and their respective epochs to get different colors

            tel_col_shown=[]

            for i_det in range(len(tel_col_list)):

                for i_exp in range(sum(mask_tel[i_det])):

                    bar_in_plot=get_overlap([num_dates_start[mask_tel[i_det]][i_exp],num_dates_stop[mask_tel[i_det]][i_exp]],ax_exp.get_xlim())>0

                    ax_exp.axvspan(xmin=num_dates_start[mask_tel[i_det]][i_exp],
                                   xmax=num_dates_stop[mask_tel[i_det]][i_exp],
                                   ymin=0, ymax=0.5, color=telescope_colors[tel_col_list[i_det]],
                                   label=tel_col_list[i_det] if (bar_in_plot and tel_col_list[i_det] not in tel_col_shown) else '', alpha=0.4)

                    if bar_in_plot:
                        tel_col_shown+=[tel_col_list[i_det]]

            plt.tight_layout()
            plt.legend()

            #short epoch id
            short_ep_str=shorten_epoch([elem_sp.split('_gti')[0].split('_sp')[0].split('src')[0] for elem_sp in elem_epoch])
            plt.savefig(os.path.join(outdir,'_'.join(short_ep_str)+'_multi_matching.png'))
            plt.savefig(os.path.join(outdir,'_'.join(short_ep_str)+'_multi_matching.pdf'))
            plt.close()

    epoch_list_started=started_expos
    epoch_list_done=done_expos

if save_epoch_list:
    with open(os.path.join(outdir,'epoch_list.txt'),'w+') as f:
        f.writelines([str(elem_epoch.tolist())+'\n'  for elem_epoch in epoch_list])

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
    print('Loading functions mode activated. Stopping here...')
    breakpoint()

#### line detections for exposure with a spectrum
for epoch_id,epoch_files in enumerate(epoch_list):

    if hid_only:
        continue

    #bad spectrum prevention
    for i_sp,elem_sp in enumerate(epoch_files):
        if elem_sp in bad_flags:
            print('\nSpectrum previously set as bad. Skipping the spectrum...')
            epoch_files=epoch_files[:i_sp]+epoch_files[i_sp+1:]

    #we use the id of the first file as an identifier
    firstfile_id=epoch_files[0].split('_sp')[0]

    if sat_glob=='Suzaku' and megumi_files:
        file_ids = [elem.split('_spec')[0].split('_src')[0] for elem in epoch_files]
    else:
        file_ids=[elem.split('_sp')[0] for elem in epoch_files]

    short_epoch_id='_'.join(shorten_epoch(file_ids))

    if restrict and epoch_restrict!=['']:

        if sat_glob in ['NICER','Suzaku']:
            if short_epoch_id not in epoch_restrict:
                print(short_epoch_id)
                print('\nRestrict mode activated and at least one spectrum not in the restrict array')
                continue
        else:
            if len([elem_sp for elem_sp in epoch_files if elem_sp not in epoch_restrict])\
                    >max(len(epoch_files)-len(epoch_restrict),0):
                print('\nRestrict mode activated and at least one spectrum not in the restrict array')
                continue

    #skip start check
    if sat_glob in ['Suzaku']:

        sp_epoch=[elem_sp.split('_spec')[0].split('_src')[0] for elem_sp in epoch_files]

        started_epochs=[literal_eval(elem.split(']')[0]+(']')) for elem in started_expos]

        if skip_started and sp_epoch in started_epochs:
             print('\nSpectrum analysis already performed. Skipping...')
             continue

    elif sat_glob=='Swift':
        if skip_started and sum([[elem] in epoch_list_started for elem in epoch_files])>0:
             print('\nSpectrum analysis already performed. Skipping...')
             continue

    elif sat_glob in ['Chandra','XMM']:

        if (skip_started and len([elem_sp for elem_sp in epoch_files[:1] if elem_sp.split('_sp')[0] not in started_expos])==0) or \
           (skip_complete and len([elem_sp for elem_sp in epoch_files[:1] if elem_sp.split('_sp')[0] not in done_expos])==0):

            print('\nSpectrum analysis already performed. Skipping...')
            continue

    elif sat_glob in ['NICER','NuSTAR','multi']:

        if (skip_started and shorten_epoch(file_ids) in started_expos) or \
           (skip_complete and shorten_epoch(file_ids) in done_expos):

            print('\nSpectrum analysis already performed. Skipping...')
            continue

    #overwrite check
    if not overwrite:
        pdf_name=os.path.join(outdir, short_epoch_id+'_recap.pdf')

        if os.path.isfile(pdf_name):
            print('\nLine detection already computed for this exposure (recap PDF exists). Skipping...')
            continue

    #we don't use the error catcher/log file in restrict mode to avoid passing through bpoints
    if not restrict:

        if log_console:
            prev_stdout=sys.stdout
            log_text=open(outdir+'/'+epoch_files[0].split('_sp')[0]+'_terminal_log.log')

        if catch_errors:
            try:
                #main function
                summary_lines=line_detect(epoch_id)

            except:
                summary_lines=['unknown error']
        else:

            summary_lines=line_detect(epoch_id)

        if log_console:
            sys.stdout=prev_stdout
            log_text.close()

        #0 is the default value for skipping overwriting the summary file
        #note: we don't overwrite the summary file in pdf_only for now
        if summary_lines is not None and not pdf_only:
            #creating the text of the summary line for this observation


            # if '_' in firstfile_id:
            #     obsid_id=firstfile_id[:firstfile_id.find('_')]
            #     file_id=firstfile_id[firstfile_id.find('_')+1:]
            # else:
            #     obsid_id=firstfile_id
            #     file_id=obsid_id

            if sat_glob=='Suzaku' and megumi_files:
                epoch_files_suffix=np.unique([elem.split('_spec')[-1].split('_pin')[-1] for elem in epoch_files])
                epoch_files_suffix=epoch_files_suffix[::-1].tolist()
            else:
                epoch_files_suffix=np.unique([elem.split('_sp')[-1]for elem in epoch_files]).tolist()

            epoch_files_str=epoch_files_suffix

            if len(np.unique(summary_lines))==1:
                summary_lines_use=summary_lines[0]
            else:
                summary_lines_use=summary_lines.tolist()

            summary_content=str(shorten_epoch(file_ids))+'\t'+str(epoch_files_suffix)+'\t'+str(summary_lines_use)

            #adding it to the summary file
            file_edit(outdir+'/'+'summary_line_det.log',
                      str(shorten_epoch(file_ids))+'\t'+str(epoch_files_suffix),summary_content+'\n',summary_header)

    else:
        summary_lines=line_detect(epoch_id)

assert not compute_highflux_only,'Stopping the computation here because no need to rebuild the summaries for this'

#not creating the recap file in spread comput mode to avoid issues
assert spread_comput==1, 'Stopping the computation here to avoid conflicts when making the summary'

if multi_obj==False:
    #loading the diagnostic messages after the analysis has been done
    if os.path.isfile(os.path.join(outdir,'summary_line_det.log')):
        with open(os.path.join(outdir,'summary_line_det.log')) as sumfile:
            glob_summary_linedet=sumfile.readlines()[1:]

    #creating summary files for the rest of the exposures
    lineplots_files=[elem.split('/')[1] for elem in glob.glob(outdir+'/*',recursive=True)]

    if sat_glob=='XMM':
        aborted_epochs=[epoch for epoch in epoch_list if not epoch[0].split('_sp')[0]+'_recap.pdf' in lineplots_files]

        aborted_files=[epoch for epoch in epoch_list if not epoch[0].split('_sp')[0]+'_recap.pdf' in lineplots_files]

    elif sat_glob in ['Chandra','Swift']:
        aborted_epochs=[[elem.replace('_grp_opt'+('.pi' if sat_glob=='Swift' else '.pha'),'') for elem in epoch]\
                        for epoch in epoch_list if not epoch[0].split('_grp_opt'+('.pi' if sat_glob=='Swift' else '.pha'))[0]+'_recap.pdf'\
                            in lineplots_files]

        aborted_files=[epoch for epoch in epoch_list if\
                       not epoch[0].split('_grp_opt'+('.pi' if sat_glob=='Swift' else '.pha'))[0]+'_recap.pdf'\
                            in lineplots_files]
    elif sat_glob in ['NICER','NuSTAR']:

        sp_suffix='_sp_grp_opt.pha' if sat_glob=='NICER' else '_sp_src_grp_opt.pha'
        #updated with shorten_epoch
        epoch_ids=[[elem.replace(sp_suffix,'') for elem in epoch] for epoch in epoch_list]

        aborted_epochs=[elem_epoch_id for elem_epoch_id in epoch_ids if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

        aborted_files=[elem_epoch for elem_epoch,elem_epoch_id in zip(epoch_list,epoch_ids) if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

    elif sat_glob=='Suzaku':
        #updated with shorten_epoch
        epoch_ids=[[elem.replace('_grp_opt.pha','').replace('_src_dtcor','').replace('_gti_event_spec_src','')\
                    for elem in epoch] for epoch in epoch_list]

        aborted_epochs=[elem_epoch_id for elem_epoch_id in epoch_ids if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

        aborted_files=[elem_epoch.tolist() for elem_epoch,elem_epoch_id in zip(epoch_list,epoch_ids) if\
                        not '_'.join(shorten_epoch(elem_epoch_id))+'_recap.pdf' in lineplots_files]

    #for now should be fine because for now we won't multi on stuff that's too weak to be detected
    elif sat_glob=='multi':
        aborted_files=[]
        aborted_epochs=[]

    if write_aborted_pdf:
        for elem_epoch_files in aborted_files:

            #list conversion since we use epochs as arguments
            pdf_summary(elem_epoch_files)

            #not used for now
            # if sat_glob=='XMM':
            #     epoch_observ=[elem_file.split('_sp')[0] for elem_file in elem_epoch]
            # elif sat_glob in ['Chandra','Swift']:
            #     epoch_observ=[elem_file.split('_grp_opt')[0] for elem_file in elem_epoch]
            # elif sat_glob=='NICER':
            #     epoch_observ=[elem_file.split('_sp_grp_opt')[0] for elem_file in elem_epoch]
            #
            # elif sat_glob=='Suzaku':
            #     if megumi_files:
            #         epoch_observ=[elem_file.split('_src')[0].split('_gti')[0] for elem_file in elem_epoch]

'''''''''''''''''''''''''''''''''''''''
''''''Hardness-Luminosity Diagrams''''''
'''''''''''''''''''''''''''''''''''''''

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
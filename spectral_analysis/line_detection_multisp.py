#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:17:29 2021

Compute spectral analysis and attempts to detect absorption lines in the iron band
for all spectra in the current merge directory

When finished or if asked to, computes a global PDF merging the result of the line detection process for each exposure 
with previous data reduction results if any

Can also add very basic correlation/distribution of the line parameters
(these should be looked at with visual_line)

If using multi_obj, it is assumed the lineplots directory is outdir
"""

#general imports
import os,sys
import glob
import argparse
import time

import numpy as np
import pandas as pd

#matplotlib imports
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.widgets import Slider,RangeSlider,Button

from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

#other stuff
from ast import literal_eval
from copy import deepcopy

#powerful saves
import dill

#progress bar
from tqdm import tqdm

#pdf conversion with HTML parsin
from fpdf import FPDF, HTMLMixin

class PDF(FPDF, HTMLMixin):
    pass

#pdf merging
from PyPDF2 import PdfFileMerger

#trapezoid integration
from scipy.integrate import trapezoid

'''Astro'''
#general astro imports
from astropy.io import fits
from astropy.time import Time
from xspec import AllModels,AllData,Fit,Spectrum,Model,Plot,Xset,FakeitSettings,AllChains,Chain

#custom script with a few shorter xspec commands
from xspec_config_multisp import allmodel_data,model_data,model_load,addcomp,editmod,Pset,Pnull,rescale,reset,Plot_screen,store_plot,freeze,allfreeze,unfreeze,\
                         calc_error,delcomp,fitmod,fitcomp,model_list,calc_fit,plot_line_comps,\
                         xcolors_grp,comb_chi2map,plot_std_ener

#custom script with a some lines and fit utilities and variables
from fitting_tools import c_light,lines_std_names,lines_e_dict,ravel_ragged,n_absline,range_absline

#importing some graph tools from the streamlit script
from visual_line_tools import load_catalogs,dist_mass,obj_values,abslines_values,values_manip,distrib_graph,correl_graph,n_infos,incl_dic
                                

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

ap = argparse.ArgumentParser(description='Script to detect lines in XMM Spectra.\n)')

'''GENERAL OPTIONS'''

ap.add_argument('-satellite',nargs=1,help='telescope to fetch spectra from',default='XMM',type=str)
ap.add_argument("-cameras",nargs=1,help='Cameras to use for spectral analysis',default='pn',type=str)
ap.add_argument("-expmodes",nargs=1,help='restrict the analysis to a single type of exposure',default='all',type=str)
ap.add_argument("-grouping",nargs=1,help='specfile grouping for XMM spectra in [5,10,20] cts/bin',default='20',type=str)
ap.add_argument("-prefix",nargs=1,help='restrict analysis to a specific prefix',default='auto',type=str)

####output directory
ap.add_argument("-outdir",nargs=1,help="name of output directory for line plots",default="lineplots_newfit",type=str)

#overwrite
ap.add_argument('-overwrite',nargs=1,
            help='overwrite previously computed line detection products (if False, skips the computation if the recap PDF file exists)',
                default=True,type=bool)

ap.add_argument("-skipbg_timing",nargs=1,help='do not use background for the -often contaminated- timing backgrounds',
                default=True,type=bool)
ap.add_argument('-max_bg_imaging',nargs=1,help='maximal imaging bg rate compared to standard bg values',default=100,type=float)

ap.add_argument('-see_search',nargs=1,help='plot every single iteration of the line search',default=False,type=bool)

ap.add_argument('-log_console',nargs=1,help='log console output instead of displaying it on the screen',default=False,type=bool)
    
ap.add_argument('-catch_errors',nargs=1,help='catch errors when the line detection process crashes and continue for other exposures',
                default=False,type=bool)

ap.add_argument('-launch_cpd',nargs=1,help='launch cpd /xs window to be able to plot elements through xspec native commands',default=False,
                 type=bool)

ap.add_argument('-xspec_window',nargs=1,help='xspec window id (auto tries to pick it automatically)',default='auto',type=str)
#note: bash command to see window ids: wmctrl -l

ap.add_argument('-pre_reduced_NICER',nargs=1,help='change NICER data format to pre-reduced obsids',default=False,type=bool)

'''DIRECTORY SPECIFICS'''

ap.add_argument("-local",nargs=1,help='launch analysis in the current directory instead',default=True,type=bool)
ap.add_argument("-h_update",nargs=1,help='update the bg, rmf and arf file names in the grouped spectra headers',
                default=True,type=bool)

'''####ANALYSIS RESTRICTION'''

ap.add_argument('-restrict',nargs=1,help='restrict the computation to a number of predefined exposures',default=False,type=bool)
#in this mode, the line detection function isn't wrapped in a try, and the summary isn't updasted

observ_restrict=['0670673001_pn_U002_Timing_auto_sp_src_grp_20.ds'
                 ]

'''    
4U spectra where need to test noem 
    '22377_heg_-1_grp_opt.pha','22377_heg_1_grp_opt.pha'
    '22378_heg_-1_grp_opt.pha','22378_heg_1_grp_opt.pha'
    '19904_heg_-1_grp_opt.pha','19904_heg_1_grp_opt.pha'

H1743 high obs with weak to no abs embedded in em feature : 
    '3804_heg_-1_grp_opt.pha','3804_heg_1_grp_opt.pha'
H1743 low obs to compare with Miller2012:
    '11048_heg_-1_grp_opt.pha','11048_heg_1_grp_opt.pha'

GRS1915+105
Hard obs with detection: 
    '660_heg_-1_grp_opt.pha','660_heg_1_grp_opt.pha'
obs initially missing from tgcat
    '16711_heg_-1_grp_opt.pha','16711_heg_1_grp_opt.pha'
    
good fast 4U XMM test: 0670673001_pn_U002_Timing_auto_sp_src_grp_20.ds

GRS huge em spectra
    '0144090101_pn_U002_Timing_auto_sp_src_grp_20.ds']


1H abs off    
                 # '0605610201_pn_S003_Timing_auto_sp_src_grp_20.ds',
                 # '0692341401_pn_S003_Imaging_auto_sp_src_grp_20.ds',
                 # '0204730301_pn_U002_Timing_auto_sp_src_grp_20.ds',
                 # '0204730201_pn_U002_Timing_auto_sp_src_grp_20.ds']
'''

ap.add_argument('-SNR_min',nargs=1,help='minimum source Signal to Noise Ratio',default=50,type=float)
#shouldn't be needed now that we have a counts min limit + sometimes false especially in timing when the bg is the source

ap.add_argument('-counts_min',nargs=1,help='minimum source counts in the source region in the line continuum range',default=5000,type=float)
ap.add_argument('-fit_lowSNR',nargs=1,help='fit the continuum of low quality data to get the HID values',default=False,type=str)

ap.add_argument('-counts_min_HID',nargs=1,help='minimum counts for HID fitting in broad band',default=200,type=float)

ap.add_argument('-skip_started',nargs=1,help='skip all exposures listed in the local summary_line_det file',
                default=False,type=bool)
#note : will skip exposures for which the exposure didn't compute or with errors

ap.add_argument('-skip_complete',nargs=1,help='skip completed exposures listed in the local summary_line_det file',
                default=True,type=bool)

ap.add_argument('-skip_nongrating',nargs=1,help='skip non grating Chandra obs (used to reprocess with changes in the restrictions)',
                default=False,type=bool)

ap.add_argument('-write_pdf',nargs=1,help='overwrite finished pdf at the end of the line detection',default=True,type=bool)


'''MODES'''

ap.add_argument('-pdf_only',nargs=1,help='Updates the pdf with already existing elements but skips the line detection entirely',
                default=False,type=bool)

ap.add_argument('-hid_only',nargs=1,help='skip the line detection and directly plot the hid',
                default=False,type=bool)

ap.add_argument('-multi_obj',nargs=1,help='compute the hid for multiple obj directories inside the current directory',
                default=False)

ap.add_argument('-autofit',nargs=1,help='enable auto fit with lines if the peak search detected at least one absorption',
                default=True,type=bool)

ap.add_argument('-refit_cont',nargs=1,help='After the autofit, refit the continuum without excluding the iron region, using the lines found during the procedure, then re-estimate the fit parameters and HID.',default=True)

#line significance assessment parameter
ap.add_argument('-assess_line',nargs=1,help='use fakeit simulations to estimate the significance of each absorption line',default=True,type=bool)

ap.add_argument('-assess_ul_detec',nargs=1,help='use fakeit simulations to estimate the upper limit of each line',default=False,type=bool)


'''SPECTRUM PARAMETERS'''

#pile-up control
ap.add_argument("-plim","--pileup_lim",nargs=1,help='maximal pileup value',default=0.10,type=float)
ap.add_argument("-pmiss",nargs=1,help='include spectra with no pileup info',default=True,type=bool)

ap.add_argument("-hid_cont_range",nargs=1,help='min and max energies of the hid band fit',default='3 10',type=str)

ap.add_argument("-line_cont_range",nargs=1,help='min and max energies of the line continuum broand band fit',default='4 10',type=str)
ap.add_argument("-line_cont_ig_arg",nargs=1,help='min and max energies of the ignore zone in the line continuum broand band fit',
                default='iron',type=str)
ap.add_argument("-line_search_e",nargs=1,help='min, max and step of the line energy search',default='4 10 0.05',type=str)

ap.add_argument("-line_search_norm",nargs=1,help='min, max and nsteps (for one sign)  of the line norm search (which operates in log scale)',
                default='0.01 10 500',type=str)

#skips fakes testing at high energy to gain time
ap.add_argument('-restrict_fakes',nargs=1,help='restrict range of fake computation to 8keV max',default=True,type=bool)

#Chandra issues
ap.add_argument('-restrict_graded',nargs=1,help='restrict range of line analysis to 8keV max for old CC33_graded spectra',default=False,type=bool)

'''PEAK/MC DETECTION PARAMETERS'''

ap.add_argument('-peak_thresh',nargs=1,help='chi difference threshold for the peak detection',default=9.21,type=float)
ap.add_argument('-peak_clean',nargs=1,help='try to distinguish a width for every peak',default=False,type=bool)

ap.add_argument('-nfakes',nargs=1,help='number of simulations used. Limits the maximal significance tested to >1-1/nfakes',default=1e3,type=int)

ap.add_argument('-sign_threshold',nargs=1,help='data significance used to start the upper limit procedure and estimate the detectability',default=0.997,
                type=float)

'''AUTOFIT PARAMETERS'''

ap.add_argument('-force_autofit',nargs=1,help='force autofit even when there are no abs peaks detected',default=True,type=bool)
ap.add_argument('-trig_interval',nargs=1,help='interval restriction for the absorption peak to trigger the autofit process',default='6.5 9.1',
                type=str)

####autofit model
ap.add_argument('-autofit_model',nargs=1,help='model list to use for the autofit computation',default='lines_resolved',type=str)


ap.add_argument('-overlap_flag',nargs=1,help='overlap value to trigger the overlap flag, in absorption line area fraction',default=0.5,
                type=float)
#note: currently not used

'''VISUALISATION'''

ap.add_argument('-plot_mode',nargs=1,help='system used for the visualisation',default='matplotlib',type=str)
ap.add_argument('-paper_look',nargs=1,help='changes some visual elements for a more official look',default=True,type=bool)

'''GLOBAL PDF SUMMARY'''

ap.add_argument('-line_infos_pdf',nargs=1,help='write line infos in the global object pdf',default=True,type=bool)

args=ap.parse_args()

'''
Notes:
-Only works for the auto observations (due to prefix naming) for now

-For now we fix the masses of all the objets at 10M_sol

-Due to the way the number of steps is computed, we explore one less value for the positive side of the normalisation

-The norm_stepval argument is for a fixed flux band, and the value is scaled in the computation depending on the line energy step
'''
sat=args.satellite
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
nfakes=int(args.nfakes)
autofit=args.autofit
force_autofit=args.force_autofit
autofit_model=args.autofit_model
write_pdf=args.write_pdf
trig_interval=np.array(args.trig_interval.split(' ')).astype(float)
hid_cont_range=np.array(args.hid_cont_range.split(' ')).astype(float)
catch_errors=args.catch_errors
assess_ul_detec=args.assess_ul_detec
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

#assessing upper limits require the autofit to proceed
if assess_ul_detec:
    force_autofit=True
    
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
    
def file_edit(path,line_id,line_data,header):
    
    '''
    Edits (or create) the file given in the path and replaces/add the line(s) where the line_id str/LIST is with the line-content str/LIST.
    line_id should be included in line_content.
    Header is the first line of the file, with usually different informations.
    '''
    
    lines=[]
    if type(line_id)==str or type(line_id)==np.str_:
        line_identifier=[line_id]
    else:
        line_identifier=line_id
        
    if type(line_data)==str or type(line_data)==np.str_:
        line_content=[line_data]
    else:
        line_content=line_data
        
    if os.path.isfile(path):
        with open(path) as file:
            lines=file.readlines()
            
            #loop for all the lines to add
            for single_identifier,single_content in zip(line_identifier,line_content):
                line_exists=False
                if not single_content.endswith('\n'):
                    single_content+='\n'
                #loop for all the lines in the file
                for l,single_line in enumerate(lines):
                    if single_identifier in single_line:
                        lines[l]=single_content
                        line_exists=True
                if line_exists==False:
                    lines+=[single_content]
            
    else:
        #adding everything
        lines=line_content

    with open(path,'w+') as file:
        if lines[0]==header:
            file.writelines(lines)
        else:
            file.writelines([header]+lines)

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
    except:
        launched_expos=[]
        completed_expos=[]
        
    return launched_expos,completed_expos

def catch_model_str(logfile,savepath=None):
    
    '''
    catches the current model's paremeters and fit (AllModels(1).show + Fit.show) from the current logfile
    
    If savepath is not None, saves the str to the given path
    '''
    
    #saving the previous chatter state and ensuring the log chatter is correctly set
    prev_logchatter=Xset.logChatter
    
    Xset.logChatter=10

    #flushing the readline to get to the current point
    logfile.readlines()
    
    #Displaying the elements we're interested in once again
    AllModels.show()
    Fit.show()
    
    #catching them
    mod_str=logfile.readlines()
    
    #and writing them into a txt
    if savepath is not None:
        with open(savepath,'w') as mod_str_file:
            mod_str_file.writelines(mod_str)
        
    #switching back to the initial logchatter value
    Xset.logChatter=prev_logchatter
    
    return mod_str

                    
#for the current directory:
started_expos,done_expos=folder_state()

if sat=='NICER':
    started_expos=[elem.split('_')[0] for elem in started_expos]
    done_expos=[elem.split('_')[0] for elem in done_expos]

#bad spectrum manually taken off
bad_flags=[]

'''initialisation'''

#readjusting the variables in lists
if sat=='XMM':
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

elif sat=='Chandra':
    cameras=['hetg']
elif sat=='NICER':
    cameras=['xti']
elif sat=='Suzaku':
    cameras=['XIS']
elif sat=='Swift':
    cameras=['xrt']
        
if expmodes=='all':
    expmodes=['Imaging','Timing']
else:
    expmodes=[expmodes]
    if 'timing' in expmodes[0] or 'Timing' in expmodes[0]:
        expmodes=expmodes+['Timing']
    if 'imaging' in expmodes[0] or 'Imaging' in expmodes[0]:
        expmodes=expmodes+['Imaging']
    expmodes=expmodes[1:]

####Energy bands

#lower limit to broad band spectra depending on the instrument (HETG is way worse at lower E)
if sat in ['XMM','NICER','Suzaku','Swift']:
    
    if sat=='Suzaku':
        e_sat_low=1.5
    else:
        e_sat_low=0.3
    if sat in ['XMM','Suzaku','Swift']:
        e_sat_high_init=10.
    else:
        e_sat_high_init=10.
        
elif sat=='Chandra':
    e_sat_low=1.5
    e_sat_high_init=10.

e_sat_high=e_sat_high_init

'''
computing the line ignore values, which we cap from the lower and upper bound of the global energy ranges to avoid issues 
we also avoid getting upper bounds lower than the lower bounds because xspec reads it in reverse and still ignores the band you want to keep
####should eventually be expanded to include the energies of each band as for the lower bound they are higher and we could have the opposite issue with re-noticing low energies
'''

if line_cont_ig_arg=='iron':
    
    if sat in ['XMM','Chandra','NICER','Swift','Suzaku']:
        
        line_cont_ig=''
        if e_sat_high>6.5:
            
            line_cont_ig+='6.5-'+str(min(7.1,e_sat_high))
            
            if e_sat_high>7.7:
                line_cont_ig+=',7.7-'+str(min(8.3,e_sat_high))
        else:
            #failsafe in case the e_sat_high is too low, we ignore the very first channel of the spectrum
            line_cont_ig=str(1)
                        
    else:
        line_cont_ig='6.-8.'
        
if local==False:
    os.chdir('bigbatch')

if launch_cpd:
    #the weird constant error is just to avoid having an error detection in Spyder due to xspec_id not being created at this point
    if 1==0:
        xspec_id=1
    
    try:
        Pnull(xspec_id)
    except:
        Pnull()

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
    
    if sat=='XMM':
        for elem_cam in cameras:
            for elem_exp in expmodes:
                spfile_list=spfile_list+glob.glob('*'+elem_cam+'*'+elem_exp+'_'+prefix+'_sp_src_grp_'+grouping+'*')
                #taking of modified spectra with background checked
                spfile_list=[elem for elem in spfile_list if 'bgtested' not in elem]
    elif sat in ['Chandra','NICER','Suzaku','Swift']:
        # if pre_reduced_NICER and sat=='NICER':
        #     spfile_list=glob.glob('*.grp')
        # else:
        spfile_list=glob.glob('*_grp_opt'+('.pi' if sat=='Swift' else '.pha') )
        
    if launch_cpd:
        #obtaining the xspec window id. It is important to call the variable xspec_id, since it is called by default in other functions
        #we set yLog as False to use ldata in the plot commands and avoid having log delchis
        xspec_id=Pset(xlog=True,ylog=False)
        if xspec_window!='auto':
            xspec_id=xspec_window
            Plot.xLog=True
    else:
        Pset(window=None,xlog=True,ylog=False)
        
    #creating the output directory
    os.system('mkdir -p '+outdir)
    
    #listing the exposure ids in the bigbatch directory
    bigbatch_files=glob.glob('**')
    
    if sat=='XMM':
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
Xset.chatter=5

#defining the standard number of fit iterations
Fit.nIterations=100

#summary file header
summary_header='Obsid\tFile identifier\tSpectrum extraction result\n'
            
def pdf_summary(epoch_observ,fit_ok=False,summary_epoch=None):
    
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
        
    pdf=PDF(orientation="landscape")
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    
    pdf.cell(1,1,'Spectra informations:\n',align='C',center=True)
    
    #line skip
    pdf.ln(10)
        
    if sat=='XMM':
        #global flare lightcurve screen (computed in filter_evt)
        rate_name_list=[elem_observ.replace(elem_observ.split('_')[1],'rate'+elem_observ.split('_')[1]) for elem_observ in epoch_observ]
    
        rate_name_list=[rate[:rate.rfind('_')]+'_screen.png' for rate in rate_name_list]
    
    epoch_inf=[elem_observ.split('_') for elem_observ in epoch_observ]
    is_sp=[]
    is_cleanevt=[]
    filename_list=[]
    exposure_list=[]
    expmode_list=[]
    for i_obs,elem_observ in enumerate(epoch_observ):
        if sat=='XMM':
            if os.path.isfile(elem_observ+'_sp_src_grp_20.ds'):
                is_sp+=[True]
                is_cleanevt+=[True]
                filename_list+=[elem_observ+'_sp_src_grp_20.ds']
            elif os.path.isfile(elem_observ+'_evt_save.ds'):
                is_sp+=[False]
                is_cleanevt+=[True]
                filename_list+=[elem_observ+'_evt_save.ds']
            else:
                is_sp+=[False]
                is_cleanevt+=[False]
                filename_list+=[elem_observ.replace('_screen.png','.ds')]
        elif sat in ['Chandra','NICER','Suzaku','Swift']:
            is_sp+=[True]
            is_cleanevt+=[False]
            
            # if sat=='NICER' and pre_reduced_NICER:
            #     filename_list+=[elem_observ]
            # else:
            filename_list+=[elem_observ+('_sp' if sat in ['NICER','Suzaku'] else '')+'_grp_opt'+('.pi' if sat=='Swift' else '.pha')]

        with fits.open(filename_list[0]) as hdul:
            
            try:
                exposure_list+=[hdul[1].header['EXPOSURE']]
            except:
                try:
                    exposure_list+=[hdul[1].header['ONTIME']]
                except:
                    pass
                
            if sat=='Chandra':
                epoch_grating=hdul[1].header['GRATING']
                expmode_list+=[hdul[1].header['DATAMODE']]
            else:   
                expmode_list+=[''] if pre_reduced_NICER else [hdul[0].header['DATAMODE']]

            if sat=='NICER' and pre_reduced_NICER:
                    pdf.cell(1,1,'Object: '+obj_name+' | Date: '+Time(hdul[1].header['MJDSTART'],format='mjd').isot+
                             ' | Obsid: '+epoch_inf[i_obs][0],align='C',center=True)
            else:
                pdf.cell(1,1,'Object: '+obj_name+' | Date: '+hdul[0].header['DATE-OBS'].split('T')[0]+' | Obsid: '+epoch_inf[i_obs][0],
                          align='C',center=True)
            pdf.ln(10)
            if sat=='XMM':
                pdf.cell(1,1,'exposure: '+epoch_inf[i_obs][2]+' | camera: '+epoch_inf[i_obs][1]+' | mode: '+epoch_inf[i_obs][3]+
                      ' | submode: '+hdul[0].header['SUBMODE']+' | clean exposure time: '+str(round(exposure_list[i_obs]))+
                      's',align='C',center=True)
            elif sat=='Chandra':

                pdf.cell(1,1,'grating: '+epoch_grating+' | mode: '+expmode_list[0]+
                         ' clean exposure time: '+str(round(exposure_list[i_obs]))+'s',align='C',center=True)
            elif sat in ['NICER','Suzaku','Swift']:
                pdf.cell(1,1,'mode: '+expmode_list[0]+
                         ' clean exposure time: '+str(round(exposure_list[i_obs]))+'s',align='C',center=True)

            pdf.ln(10)
        
            #we only show the third line for XMM spectra with spectrum infos if there is an actual spectrum 
            if epoch_SNR[i_obs]!='X':
                
                grouping_str='SNR: '+str(round(epoch_SNR[i_obs],3))+' | Spectrum bin grouping: '+epoch_inf[i_obs][-1].split('.')[0]+' cts/bin | '
                try:
                    pileup_lines=fits.open(elem_observ+'_sp_src.ds')[0].header['PILE-UP'].split(',')
                    pdf.cell(1,1,grouping_str+'pile-up values:'+pileup_lines[-1][10:],align='C',center=True)
                except:
                    pdf.cell(1,1,grouping_str+'no pile-up values for this exposure',align='C',center=True)
    
        pdf.ln(10)
    
        #turned off for now        
        # if flag_bg[i_obs]:
        #     pdf.cell(1,1,'FLAG : EMPTY BACKGROUND')
    
    
    '''Line detection infos'''
    if summary_epoch is None:
            
        #the replace avoid problems with using full chandra/NICER file names as epoch loggings in the summary files
        result_epoch=literal_eval([elem.split('\t')[2] for elem in glob_summary_linedet \
        if (elem.split('\t')[0] if sat in ['NICER','Swift'] else '_'.join(\
            [elem.split('\t')[0],elem.split('\t')[1].replace('_grp_opt','').replace('sp_grp_opt','').replace('.pha','').replace('.pi','')]))==\
           '_'.join(epoch_inf[0])][0])
    else:
        result_epoch=summary_epoch
        
    def disp_broadband_data():
        
        '''
        display the different raw spectra in the epoch
        '''
        
        pdf.cell(1,1,'Broad band data ('+str(e_sat_low)+'-'+str(e_sat_high)+' keV)',align='C',center=True)
        sp_displayed=False
        
        if len(epoch_observ)==1 or sat!='XMM':
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
        for i_elem,elem_result in enumerate(result_epoch):
            pdf.cell(1,1,epoch_observ[i_elem]+': '+elem_result,align='C',center=True)
            pdf.ln(10)
            
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
            
            '''
            scans model lines for multiple data groups and displays only the first lines of each data group after the first
            (to show the constant factors)
            '''
            lineid_grp_arr=np.argwhere(np.array(['Data group' in elem for elem in np.array(lines)])).T[0]
            
            #no need to do anything for less than two datagroups
            if len(lineid_grp_arr)<2:
                return lines
            else:
                lines_cleaned=[]
                #we display u to the second data group, then only 2 lines at a time
                for i_grp,lineid_grp in enumerate(lineid_grp_arr):
                    if i_grp==0:
                        i_begin=0
                        i_end=lineid_grp_arr[i_grp+1]
                    else:
                        i_begin=lineid_grp
                        i_end=i_begin+2
                    lines_cleaned+=lines[i_begin:i_end]
                
                #adding everything after the end of the model (besides the last line which is just a 'model not fit yet' line)
                lines_cleaned+=lines[i_begin+lineid_grp_arr[1]-lineid_grp_arr[0]:-1]
                
                return lines_cleaned
            
        def display_fit(fit_type):
            if 'broadband' in fit_type:
                fit_title='broad band'
                fit_ener=str(e_sat_low)+'-'+str(e_sat_high)
            elif 'broadhid' in fit_type:
                fit_title='HID'
                fit_ener='3.-10.'
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
                pdf.ln(10)

                pdf.cell(1,1,fit_title+' ('+fit_ener+' keV):',align='C',center=True)
                
                if fit_type=='broadband':
                    #displaying the colors for the upcoming plots in the first fit displayed
                    pdf.cell(1,10,'        '.join([xcolors_grp[i_good_sp]+': '+'_'.join(good_sp[i_good_sp].split('_')[1:3])\
                                                   for i_good_sp in range(len(good_sp))]),align='C',center=True)
                        
                pdf.image(image_path,x=0,y=50,w=150)       
                
                #fetching the model unless in zoom mode where the model was displayed on the page before
                if 'zoom' not in fit_type:
                    #and getting the model lines from the saved file
                    with open(outdir+'/'+epoch_observ[0]+'_mod_'+fit_type+'.txt') as mod_txt:
                        fit_lines=mod_txt.readlines()
                        pdf.set_font('helvetica', 'B', 8-int(len(disp_multigrp(fit_lines))/10))
                        pdf.multi_cell(150,5,'\n'*max(0,int(15-2*(len(disp_multigrp(fit_lines))**2/100)))+''.join(disp_multigrp(fit_lines)))
                    
                    #in some cases due to the images between some fit displays there's no need to add a page
                    if 'linecont' not in fit_type:
                        pdf.add_page()
                    else:
                        #restrictingthe addition of a new page to very long models
                        if len(fit_lines)>35:
                            pdf.add_page()
                else:
                    pass
                 
        display_fit('broadband_post_auto')
        display_fit('broadhid_post_auto')
        display_fit('autofit')
        display_fit('autofit_zoom')
            
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
            
        display_fit('broadband')
        display_fit('broadhid')
        display_fit('broadband_linecont')
        
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
    elif sat=='XMM':
        
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
        
    '''extraction images'''

    #### XMM Data reduction display
    
    if sat=='XMM':
        for i_obs,elem_observ in enumerate(epoch_observ):
            if is_sp[i_obs]:
                pdf.add_page()
                pdf.set_font('helvetica', 'B', 16)
                pdf.cell(1,10,'Data reduction for observation '+elem_observ,align='C',center=True)
                pdf.ln(10)
                pdf.cell(1,30,'Initial region definition                                        '+
                          'Post pile-up excision (if any) region definition',align='C',center=True)
                pdf.image(elem_observ+'_reg_screen.png',x=2,y=50,w=140)
                
                if os.path.isfile(elem_observ+'_reg_excised_screen.png'):
                    pdf.image(elem_observ+'_reg_excised_screen.png',x=155,y=50,w=140)
    
                if expmode_list[i_obs]=='IMAGING':
                    pdf.add_page()
                    pdf.image(elem_observ+'_opti_screen.png',x=1,w=280)
                    
                    #adding a page for the post-pileup computation if there is one
                    if os.path.isfile(elem_observ+'_opti_excised_screen.png'):
                        pdf.add_page()
                        pdf.image(elem_observ+'_opti_excised_screen.png',x=1,w=280)
                        
                elif expmode_list[i_obs]=='TIMING' or expmode_list[i_obs]=='BURST':
                    pdf.add_page()
                    pdf.cell(1,30,'SNR evolution for different source regions, first iteration',align='C',center=True)
                    pdf.image(elem_observ+'_opti_screen.png',x=10,y=50,w=140)
                    
                    #adding a page for the post-pileup computation if there is one
                    if os.path.isfile(elem_observ+'_opti_excised_screen.png'):
                        pdf.image(elem_observ+'_opti_excised_screen.png',x=150,y=50,w=140)
                        
            elif is_cleanevt[i_obs]:
                pdf.set_font('helvetica', 'B', 16)
                if expmode_list[i_obs]=='IMAGING':
                    pdf.add_page()
                    pdf.cell(1,30,'Raw image                     '+'              position catalog cropping zone          '+
                              '            cropped region zoom',align='C',center=True)
                    try:
                        pdf.image(elem_observ+'_img_screen.png',x=1,y=70,w=90)
                        pdf.image(elem_observ+'_catal_reg_screen.png',x=100,y=70,w=90)
                        pdf.image(elem_observ+'_catal_crop_screen.png',x=190,y=65,w=120)
                    except:
                        pass
                if expmode_list[i_obs]=='TIMING' or expmode_list[i_obs]=='BURST':
                    pdf.add_page()
                    pdf.cell(1,30,'Raw image',align='C',center=True)
                    try:
                        pdf.image(elem_observ+'_img_screen.png',x=70,y=50,w=150)
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
        pdf.output(outdir+'/'+epoch_observ[0]+'_aborted_recap.pdf')
    else:
        pdf.output(outdir+'/'+epoch_observ[0]+'_recap.pdf')
    
def line_detect(epoch_id):
    
    Xset.logChatter=10
    
    '''
    line detection for a single object
    
    we use the index as an argument to fill the chi array 
    '''
    
    epoch_files=epoch_list[epoch_id]
    
    epoch_observ=[elem.split('_sp')[0] if sat=='XMM' else elem.split('_grp_opt')[0] if sat in ['Chandra','Swift'] else elem.split('_sp_grp_opt')[0] if sat in ['NICER','Suzaku'] else '' for elem in epoch_files]
    
    print('\nStarting line detection for files ')
    print(epoch_files)

    if restrict and observ_restrict!=[''] and len([elem_sp for elem_sp in epoch_files if elem_sp not in observ_restrict])>max(len(epoch_files)-len(observ_restrict),0):
        print('\nRestrict mode activated and at least one spectrum not in the restrict array')
        return ''
    
    #reset the xspec config
    reset()
    
    #test variable for resetting the energy range, the reset happens unless specific mode requires specific energy ranges
    reset_ener=True
    
    #same thing for skipping old graded obs
    obs_grating=False
    #churazov weight for HETG spectra to compensate the lack of grouping besides Kastra

    ###TODO: Change this to C-stat
    Fit.weight='churazov'
    
    if sat=='Chandra':
        
        with fits.open(epoch_files[0]) as hdul:
            datamode=hdul[1].header['DATAMODE']
            obsdate=hdul[1].header['DATE-OBS']
            
        if datamode=='CC33_GRADED' and int(obsdate.split('-')[0])<=2014:
            
            obs_grating=True
            
            print('\nOld CC33 obs detected.')
            if restrict_graded:
                print('\nRestricting energy range...')
                e_sat_high=8.
                hid_cont_range[1]=8.
                line_cont_range[1]=8.
                
                line_search_e_space=np.arange(line_search_e[0],8.+line_search_e[2]/2,line_search_e[2])
                #this one is here to avoid adding one point if incorrect roundings create problem
                line_search_e_space=line_search_e_space[line_search_e_space<=8.]

                reset_ener=False
        
    if reset_ener:
        
        e_sat_high=e_sat_high_init
        hid_cont_range[1]=e_sat_high
        line_cont_range[1]=e_sat_high
        
        #note: we add half a step to get rid of rounding problems and have the correct steps
        line_search_e_space=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2]/2,line_search_e[2])
        #this one is here to avoid adding one point if incorrect roundings create problem
        line_search_e_space=line_search_e_space[line_search_e_space<=line_search_e[1]]

    #skipping observation if asked
    if sat=='Chandra' and skip_nongrating and not obs_grating:
        return None
        
    #useful for later
    spec_inf=[elem_sp.split('_') for elem_sp in epoch_files]
    
    #Step 0 is to readjust the response and bg file names if necessary (i.e. files got renamed)
    if h_update and sat=='XMM':
        for i_sp,elem_sp in enumerate(epoch_files):
            with fits.open(elem_sp,mode='update') as hdul:
                hdul[1].header['BACKFILE']='_'.join(spec_inf[i_sp][:-4])+'_sp_bg.ds'
                hdul[1].header['RESPFILE']='_'.join(spec_inf[i_sp][:-4])+'.rmf'
                hdul[1].header['ANCRFILE']='_'.join(spec_inf[i_sp][:-4])+'.arf'
                #saving changes
                hdul.flush()
    
    '''Setting up a log file and testing the properties of each spectra'''
    
    curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log.log')
    #ensuring the log information gets in the correct place in the log file by forcing line to line buffering
    curr_logfile_write.reconfigure(line_buffering=True)
    
    curr_logfile=open(curr_logfile_write.name,'r')
    
    #list containing the epoch files rejected by the test
    epoch_files_good=[]
    
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

    if pdf_only:
        
        try:
            
            pdf_summary(epoch_observ,fit_ok=True,summary_epoch=fill_result('Line detection complete.'))
        
            #closing the logfile for both access and Xspec
            curr_logfile.close()    
            Xset.closeLog()
                
            return fill_result('Line detection complete.')
        except:
            return fill_result('Missing elements to compute PDF.')
        
    for i_sp,elem_sp in enumerate(epoch_files):
        
        '''Those checks are exclusively for XMM Spectra'''
        
        if sat!='XMM':
            continue
        
        AllData.clear()
        
        bg_off_flag=False
        try:
            curr_spec=Spectrum(elem_sp)
        except:
            try:
                curr_spec=Spectrum(elem_sp,backfile=None)
                print('\nLoaded the spectrum '+elem_sp+' with no background')
                bg_off_flag=True
            except:
                print("\nCouldn't load the spectrum "+elem_sp+"  with Xspec. Negative exposure time can cause this. Skipping the spectrum...")
                epoch_result[i_sp]="Couldn't load the spectrum with Xspec."
                continue
        
        #this is useful for the check plots 
        Plot.background=True
        Plot.add=True

        #saving a screen of the spectrum for verification purposes    
        AllData.ignore('**-'+str(e_sat_low)+' '+str(e_sat_high)+'-**')
        
        #checking if the spectrum is empty after ignoring outside of the broad interval
        if curr_spec.rate[0]==0:
                print("\nSpectrum "+elem_sp+" empty in the ["+str(e_sat_low)+"-"+str(e_sat_high)+"] keV range. Skipping the spectrum...")
                epoch_result[i_sp]="Spectrum empty in the ["+str(e_sat_low)+"-"+str(e_sat_high)+"]keV range."
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
                print('\nPile-up value for spectrum '+elem_sp+' above the given limit of '+str(round(100*pileup_lim,1))+
                      '%. Skipping the spectrum...')
                epoch_result[i_sp]='Pile-up above the given limit ('+str(round(100*pileup_lim,1))+'%).'
                continue
            else:
                print('\nPile-up value for spectrum '+elem_sp+' OK.')
                
        except:
            print('\nNo pile-up information available for spectrum '+elem_sp)
            pileup_value=-1        
            if pileup_missing==False:
                print('\nSkipping the spectrum...')
                epoch_result[i_sp]='No pile-up info available'
                continue
                
        '''Unlading Imaging backgrounds if they are too bright compared to blank field standards'''
        
        with fits.open(elem_sp) as hdul:
            curr_expmode=hdul[0].header['DATAMODE']
            curr_cam=hdul[1].header['INSTRUME'][1:].swapcase()
            
        if curr_expmode=='IMAGING' and bg_off_flag==False:
        
            AllData.ignore('**-'+str(e_sat_low)+' '+str(e_sat_high)+'.-**')
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
                print('\nIntense background detected, probably contaminated by the source. Unloading it to avoid affecting the source...')
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
                print('\nSpectrum  '+elem_sp+' Signal to Noise Ratio below the limit. Skipping the spectrum ...')
                epoch_result[i_sp]='Spectrum SNR below the limit ('+str(SNR_min)+')'
                continue
        
        #saving the spectra if it passed all the test
        epoch_files_good+=[elem_sp]

    #testing if all spectra have been taken off
    if len(epoch_files_good)==0 and sat=='XMM':
        return epoch_result 

    #### Data load
    if sat=='XMM':
        AllData(''.join([str(i_sp+1)+':'+str(i_sp+1)+' '+elem_sp.replace('.ds','_bgtested.ds')+' '\
                     for i_sp,elem_sp in enumerate(epoch_files_good)]))
    elif sat=='Chandra':
        
        #loading both 1st order grating of the HETG
        AllData('1:1 '+epoch_files[0]+' 2:2 '+epoch_files[1])
        
        #should not be needed
        # AllData(1).response=epoch_files[0].replace('_grp_opt.pha','.rmf')
        # AllData(2).response=epoch_files[1].replace('_grp_opt.pha','.rmf')
        
        AllData(1).response.arf=epoch_files[0].replace('_grp_opt.pha','.arf')
        AllData(2).response.arf=epoch_files[1].replace('_grp_opt.pha','.arf')
        
        #screening the spectra
        AllData.ignore('**-'+str(e_sat_low)+' '+str(e_sat_high)+'-**')
        Plot_screen("ldata",outdir+'/'+epoch_observ[0]+"_screen_xspec_spectrum")
        
    elif sat=='NICER':
        
        #loading the spectrum and the response
        
        # if pre_reduced_NICER:
        #     curr_sp=Spectrum(epoch_files[0])
        
        # else:
        curr_sp=Spectrum(epoch_files[0],backFile=epoch_observ[0]+'_bkg.pi',respFile=epoch_observ[0]+'.rmf',arfFile=epoch_observ[0]+'.arf')
        try:
                curr_sp.background=epoch_observ[0]+'_bkg.pi'
        except:
            pass
        curr_sp.response=epoch_observ[0]+'.rmf'
        curr_sp.response.arf=epoch_observ[0]+'.arf'
        
    elif sat=='Suzaku':
        
        AllData('1:1 '+epoch_files[0]+' 2:2 '+epoch_files[1])
        
        AllData(1).background=epoch_observ[0]+'_bkg.pi'
        AllData(2).background=epoch_observ[1]+'_bkg.pi'
        
        AllData(1).response.arf='0sp.arf'
        AllData(2).response.arf='sp.arf'
        
    elif sat=='Swift':
        if len(epoch_files)==2:
            AllData('1:1 '+epoch_files[0]+' 2:2 '+epoch_files[1])
            AllData(1).response.arf=epoch_observ[0].replace('source','')+'.arf'
            AllData(1).background=epoch_observ[0].replace('source','')+('back.pi')
            
            AllData(2).response.arf=epoch_observ[1].replace('source','')+'.arf'
            AllData(2).background=epoch_observ[1].replace('source','')+('back.pi')
        
        elif len(epoch_files)==1:
            AllData('1:1 '+epoch_files[0])
            AllData(1).response.arf=epoch_observ[0].replace('source','')+'.arf'
            AllData(1).background=epoch_observ[0].replace('source','')+('back.pi')
            
    if sat!='XMM':
        
        #screening the spectra
        AllData.ignore('**-'+str(e_sat_low)+' '+str(e_sat_high)+'-**')
        Plot_screen("ldata",outdir+'/'+epoch_observ[0]+"_screen_xspec_spectrum")
        
    '''
    Testing the amount of raw source counts in the line detection range for all datagroups combined
    '''
    
    #### Testing if the data is above the count limit
    
    #for the line detection
    AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')
    glob_counts=0
    indiv_counts=[]
    
    for i_grp in range(1,AllData.nGroups+1):
        indiv_counts+=[round(AllData(i_grp).rate[2]*AllData(i_grp).exposure)]
        glob_counts+=indiv_counts[-1]
    if glob_counts<counts_min:
        flag_lowSNR_line=True
        if not fit_lowSNR:
            print('\nInsufficient counts ('+str(round(glob_counts))+' < '+str(round(counts_min))+
                  ') in line detection range.')
            return fill_result('Insufficient counts ('+str(round(glob_counts))+' < '+str(round(counts_min))+\
                               ') in line detection range.')
    else:
        flag_lowSNR_line=False
        
    #for the continuum fit
    AllData.notice('all')
    AllData.ignore('bad')
    
    #limiting to the line search energy range
    AllData.ignore('**-'+str(hid_cont_range[0])+' '+str(hid_cont_range[1])+'-**')
     
    glob_counts=0
    indiv_counts=[]
    
    for i_grp in range(1,AllData.nGroups+1):
        indiv_counts+=[round(AllData(i_grp).rate[2]*AllData(i_grp).exposure)]
        glob_counts+=indiv_counts[-1]
    if glob_counts<counts_min_HID:
        print('\nInsufficient counts ('+str(round(glob_counts))+' < '+str(round(counts_min_HID))+
              ') in HID detection range.')
        return fill_result('Insufficient counts ('+str(round(glob_counts))+' < '+str(round(counts_min_HID))+\
                           ') in HID detection range.')
            
    #creating the continuum model list
    comp_cont=model_list('cont')
    
    #taking off the constant factor if there's only one data group
    if AllData.nGroups==1:
        comp_cont=comp_cont[1:]
    
    isbg_grp=[]
    for i_grp in range(1,AllData.nGroups+1):
        try:
            AllData(i_grp).background
            isbg_grp+=[True]
        except:
            isbg_grp+=[False]
                    
    '''Continuum fits'''
    
    def high_fit(broad_absval=None):
        
        '''
        high energy fit and flux array computation
        '''
        
        AllModels.clear()
        
        print('\nComputing line continuum fit...')
        AllData.notice('all')
        AllData.ignore('bad')
        
        #limiting to the line search energy range
        AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')

        #if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
        if not flag_lowSNR_line:
            #ignoring the 6-8keV energy range for the fit to avoid contamination by lines
            AllData.ignore(line_cont_ig)
        
        #comparing different continuum possibilities with a broken powerlaw or a combination of diskbb and powerlaw
        
        #creating the automatic fit class for the standard continuum
        if broad_absval!=0:
            fitcont_high=fitmod(comp_cont,curr_logfile,curr_logfile_write,absval=broad_absval)
        else:
            #creating the fitcont without the absorption component if it didn't exist in the broad model
            fitcont_high=fitmod([elem for elem in comp_cont if elem!='glob_phabs'],curr_logfile,curr_logfile_write)
            
        # try:
            #fitting
        fitcont_high.global_fit()
        
        # mod_fitcont=allmodel_data()
        
        chi2_cont=Fit.statistic
        # except:
        #     pass
        #     chi2_cont=0
        
        # AllModels.clear()

        #not used currently        
        # #with the broken powerlaw continuum
        # fitcont_high_bkn=fitmod(comp_cont_bkn,curr_logfile)
        
        # try:
        #     #fitting
        #     fitcont_high_bkn.global_fit()
            
        #     chi2_cont_bkn=Fit.statistic
        # except:
        #     pass
        chi2_cont_bkn=0
            
        if chi2_cont==0 and chi2_cont_bkn==0:
            
            print('\nProblem during line continuum fit. Skipping line detection for this exposure...')
            return ['\nProblem during line continuum fit. Skipping line detection for this exposure...']
        # else:
        #     if chi2_cont<chi2_cont_bkn:
        #         model_load(mod_fitcont)
                
        
        #renoticing the line energy range
        AllData.notice(line_cont_ig)

        #saving the model data to reload it after the broad band fit if needed
        mod_high_dat=allmodel_data()
        
        fitcont_high.save()
        
        #rescaling before the prints to avoid unecessary loggings in the screen
        rescale(auto=True)
        
        #screening the xspec plot and the model informations for future use
        Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+"_screen_xspec_broadband_linecont")
        
        #saving the model str
        catch_model_str(curr_logfile,savepath=outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.txt')
    
        #deleting the model file since Xset doesn't have a built-in overwrite argument and crashes when asking manual input
        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.xcm'):
            os.remove(outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.xcm')
            
        #storing the current configuration and model
        Xset.save(outdir+'/'+epoch_observ[0]+'_mod_broadband_linecont.xcm',info='a')
        
        #storing the class
        fitcont_high.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadband_linecont.pkl')
        
        '''
        Computing the local flux for every step of the energy space :
        This will be used to adapt the line energy to the continuum and avoid searching in wrong norm spaces
        We store the flux of the continuum for a width of one step of the energy space, around each step value
        
        Note : We do this for the first spectrum only even with multi data groups
        '''
        
        flux_base=np.zeros(len(line_search_e_space))
        
        #here to avoir filling the display with information we're already storing
        # with redirect_stdout(open(os.devnull, 'w')):
         
        for ind_e,energy in enumerate(line_search_e_space):
        
            AllModels.calcFlux(str(energy-line_search_e[2]/2)+" "+str(energy+line_search_e[2]/2))
            flux_base[ind_e]=AllData(1).flux[0]
            
            #this is required because the buffer is different when redirected
            sys.stdout.flush()
            
        return [mod_high_dat,flux_base,fitcont_high]
        
    def store_fit(mode='broadband'):
        
        '''
        plots and saves various informations about a fit
        '''
        
        #Since the automatic rescaling goes haywire when using the add command, we manually rescale (with our own custom command)
        rescale(auto=True)
        
        Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+'_screen_xspec_'+mode)
        
        #saving the model str
        catch_model_str(curr_logfile,savepath=outdir+'/'+epoch_observ[0]+'_mod_'+mode+'.txt')
        
        if os.path.isfile(outdir+'/'+epoch_observ[0]+'_mod_'+mode+'.xcm'):
            os.remove(outdir+'/'+epoch_observ[0]+'_mod_'+mode+'.xcm')
            
        #storing the current configuration and model
        Xset.save(outdir+'/'+epoch_observ[0]+'_mod_'+mode+'.xcm',info='a')
        
    def hid_fit_infos(fitmodel,broad_absval,post_autofit=False):
        
        '''
        computes various informations about the fit
        '''

        if post_autofit:
            add_str='_post_auto'
        else:
            add_str=''
        #freezing what needs to be to avoid problems with the Chain
        calc_error(curr_logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),timeout=60,freeze_pegged=True,indiv=True)
        
        Fit.perform()
    
        fitmodel.update_fitcomps()

        #storing the flux and HR with the absorption to store the errors
        #We can only show one flux in the HID so we use the first one, which should be the most 'precise' with our order (pn first)
        
        AllChains.defLength=50000
        AllChains.defBurn=20000
        AllChains.defWalkers=10
        
        #deleting the previous chain to avoid conflicts
        AllChains.clear()
        
        if os.path.exists(outdir+'/'+epoch_observ[0]+'_chain_hid'+add_str+'.fits'):
            os.remove(outdir+'/'+epoch_observ[0]+'_chain_hid'+add_str+'.fits')
        
        try:
            #Creating a chain to avoid problems when computing the errors
            Chain(outdir+'/'+epoch_observ[0]+'_chain_hid'+add_str+'.fits')
        except:
            #trying to freeze pegged parameters again in case the very last fit created peggs
            
            calc_error(curr_logfile,param='1-'+str(AllModels(1).nParameters*AllData.nGroups),timeout=60,freeze_pegged=True,indiv=True)
            
            Fit.perform()
        
            fitmodel.update_fitcomps()
            #Creating a chain to avoid problems when computing the errors
            Chain(outdir+'/'+epoch_observ[0]+'_chain_hid'+add_str+'.fits')
            
        #computing and storing the flux for the full luminosity and two bands for the HR
        spflux_single=[None]*5
        
        AllModels.calcFlux(str(hid_cont_range[0])+' '+str(hid_cont_range[1])+" err 1000 90")
        spflux_single[0]=AllData(1).flux[0],AllData(1).flux[0]-AllData(1).flux[1],AllData(1).flux[2]-AllData(1).flux[0]
        AllModels.calcFlux("3. 6. err 1000 90")
        spflux_single[1]=AllData(1).flux[0],AllData(1).flux[0]-AllData(1).flux[1],AllData(1).flux[2]-AllData(1).flux[0]
        AllModels.calcFlux("6. 10. err 1000 90")
        spflux_single[2]=AllData(1).flux[0],AllData(1).flux[0]-AllData(1).flux[1],AllData(1).flux[2]-AllData(1).flux[0]
        AllModels.calcFlux("1. 3. err 1000 90")
        spflux_single[3]=AllData(1).flux[0],AllData(1).flux[0]-AllData(1).flux[1],AllData(1).flux[2]-AllData(1).flux[0]
        AllModels.calcFlux("3. 10. err 1000 90")
        spflux_single[4]=AllData(1).flux[0],AllData(1).flux[0]-AllData(1).flux[1],AllData(1).flux[2]-AllData(1).flux[0]
        
        spflux_single=np.array(spflux_single)
        
        AllChains.clear()
        
        AllData.notice(line_cont_ig)
        
        store_fit(mode='broadhid'+add_str)
        
        #storing the fitmod class into a file
        fitmodel.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadhid'+add_str+'.pkl')
                
        #taking off the absorption (if it is in the final components) before computing the flux
        if broad_absval!=0:
            if 'glob_phabs' in [elem.compname for elem in [comp for comp in fitmodel.includedlist if comp is not None]]:
                if fitmodel.glob_phabs.included:    
                    fitmodel.glob_phabs.xcomps[0].nH=0
            elif 'cont_phabs' in [elem.compname for elem in [comp for comp in fitmodel.includedlist if comp is not None]]:
                if fitmodel.cont_phabs.included:
                    fitmodel.cont_phabs.xcomps[0].nH=0
                
        #and replacing the main values with the unabsorbed flux values 
        #(conservative choice since the other uncertainties are necessarily higher)
        AllModels.calcFlux(str(hid_cont_range[0])+' '+str(hid_cont_range[1]))
        spflux_single[0][0]=AllData(1).flux[0]
        AllModels.calcFlux("3. 6.")
        spflux_single[1][0]=AllData(1).flux[0]
        AllModels.calcFlux("6. 10.")
        spflux_single[2][0]=AllData(1).flux[0]
        AllModels.calcFlux("1. 3.")
        spflux_single[3][0]=AllData(1).flux[0]
        AllModels.calcFlux("3. 10.")
        spflux_single[4][0]=AllData(1).flux[0]
        
        spflux_single=spflux_single.T
        
        #reloading the absorption values to avoid modifying the fit
        if broad_absval!=0:
            if 'glob_phabs' in [elem.compname for elem in [comp for comp in fitmodel.includedlist if comp is not None]]:
                if fitmodel.glob_phabs.included:    
                    fitmodel.glob_phabs.xcomps[0].nH=broad_absval
            elif 'cont_phabs' in [elem.compname for elem in [comp for comp in fitmodel.includedlist if comp is not None]]:
                if fitmodel.cont_phabs.included:
                    fitmodel.cont_phabs.xcomps[0].nH=broad_absval
                    
        return spflux_single
        
    def broad_fit():
        
        '''Broad band fit to get the HR ratio and Luminosity'''
    
        #first broad band fit in e_sat_low-10 to see the spectral shape
        print('\nComputing broad band fit for visualisation purposes...')
        AllData.notice('all')
        AllData.ignore('bad')
        AllData.ignore('**-'+str(e_sat_low)+' '+str(e_sat_high)+'-**')
        
        #if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
        if not flag_lowSNR_line:
            AllData.ignore(line_cont_ig)
        

        
        #creating the automatic fit class for the standard continuum
        fitcont_broad=fitmod(comp_cont,curr_logfile,curr_logfile_write)
        
        # try:
        #fitting
        fitcont_broad.global_fit()
        
        mod_fitcont=allmodel_data()
        
        chi2_cont=Fit.statistic
        # except:
        #     pass
        #     chi2_cont=0
        
        # AllModels.clear()
        
        # #with the broken powerlaw continuum
        # fitcont_broad_bkn=fitmod(comp_cont_bkn,curr_logfile)
        
        # try:
        #     #fitting
        #     fitcont_broad_bkn.global_fit()
            
        #     chi2_cont_bkn=Fit.statistic
        # except:
        #     pass
        
        chi2_cont_bkn=0
            
        if chi2_cont==0 and chi2_cont_bkn==0:
            
            print('\nProblem during broad band fit. Skipping line detection for this exposure...')
            return ['\nProblem during broad band fit. Skipping line detection for this exposure...']
        # else:
        #     if chi2_cont<chi2_cont_bkn:
        #         model_load(mod_fitcont)

        #storing the absorption of the broad fit
        if fitcont_broad.glob_phabs.included:
            broad_absval=AllModels(1)(fitcont_broad.glob_phabs.parlist[0]).values[0]
        else:
            broad_absval=0
            
        AllData.notice(line_cont_ig)
        
        store_fit()
        
        #storing the class
        fitcont_broad.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadband.pkl')
        
        print('\nComputing HID broad fit...')
        AllModels.clear()
        AllData.notice('all')
        AllData.ignore('**-'+str(hid_cont_range[0])+' '+str(hid_cont_range[1])+'-**')
        AllData.ignore('bad')
        
        #if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
        if not flag_lowSNR_line:
            AllData.ignore(line_cont_ig)

        #creating the automatic fit class for the standard continuum
        if broad_absval!=0:
            fitcont_hid=fitmod(comp_cont,curr_logfile,curr_logfile_write,absval=broad_absval)
        else:
            #creating the fitcont without the absorption component if it didn't exist in the broad model
            fitcont_hid=fitmod([elem for elem in comp_cont if elem!='glob_phabs'],curr_logfile,curr_logfile_write)
        
        # try:
        #fitting
        fitcont_hid.global_fit()
        
        mod_fitcont=allmodel_data()
        
        chi2_cont=Fit.statistic
        # except:
        #     pass
        #     chi2_cont=0
        
        # AllModels.clear()
        
        # #with the broken powerlaw continuum
        # fitcont_hid_bkn=fitmod(comp_cont_bkn,curr_logfile)
        
        # try:
        #     #fitting
        #     fitcont_hid_bkn.global_fit()
            
        #     chi2_cont_bkn=Fit.statistic
        # except:
        #     pass
        
        chi2_cont_bkn=0
            
        if chi2_cont==0 and chi2_cont_bkn==0:
            
            print('\nProblem during hid band fit. Skipping line detection for this exposure...')
            return ['\nProblem during hid band fit. Skipping line detection for this exposure...']
        # else:
        #     if chi2_cont<chi2_cont_bkn:
        #         model_load(mod_fitcont)
        
        spflux_single=hid_fit_infos(fitcont_hid,broad_absval)
        
        return spflux_single,broad_absval

    
    AllModels.clear()
    
    result_broad_fit=broad_fit()
    
    if len(result_broad_fit)==1:
        return fill_result(result_broad_fit)
    else:
        main_spflux,broad_absval=result_broad_fit
    
    result_high_fit=high_fit(broad_absval)
    
    #if the function returns an array of length 1, it means it returned an error message
    if len(result_high_fit)==1:
        return fill_result(result_high_fit)
    else:
        data_mod_high,flux_cont,fitmod_cont=result_high_fit
        
    #creation of the eqwidth conversion variable
    eqwidth_conv=np.zeros(len(line_search_e_space))

    #for the line plots we don't need the background nor the rest anymore
    Plot.background=False
    
    #re-limiting to the line search energy range
    AllData.notice('all')
    AllData.ignore('bad')
    AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')
    
    #changing back to the auto rescale of xspec
    Plot.commands=()
    Plot.addCommand('rescale')
                
    print('\nStarting line search...')
    
    def narrow_line_search(data_cont,suffix):
        
        '''
        Wrapper for all the line search codeand associated visualisation
        
        Explores the current model in a given range by adding a line of varying normalisation and energy and mapping the associated
        2D delchi map
        '''
        
        #defining the chi array for each epoch
        chi_arr=np.zeros((len(line_search_e_space),norm_nsteps))

        #reseting the model 
        AllModels.clear()
        
        #reloading the broad band model
        model_load(data_cont)

        plot_ratio_values=store_plot('ratio')
        
        chi_base=Fit.statistic
        
        #adding the gaussian with constant factors and cflux for variations
        #since we cannot use negative logspaces in steppar, we use one constant factor for the sign and a second for the value
        addcomp('constant(constant(cflux(gauss)))')
        
        mod_gauss=AllModels(1)
        
        #freezing everything but the second constant factor to avoid problems during steppar
        allfreeze()
        
        #since there might be other components with identical names in the model, we retrieve each of the added xspec components
        #from their position as the last components in the component list:
        comp_cfactor_1,comp_cfactor_2,comp_gauss_cflux,comp_gauss=[getattr(mod_gauss,mod_gauss.componentNames[-4+i]) for i in range(4)]
        
        #unlocking negative constant factors for the first one
        comp_cfactor_1.factor.values=[1.0, -0.01, -1e6, -1e6, 1e6, 1e6]

        #getting the constant factor index and unfreezing it
        index_cfactor_2=comp_cfactor_2.factor.index
        comp_cfactor_2.factor.frozen=0
        
        #adjusting the cflux to be sure we cover the entire flux of the gaussian component
        comp_gauss_cflux.Emin=str(e_sat_low)
        comp_gauss_cflux.Emax=12.
    
        #narrow line locked
        comp_gauss.Sigma=0
        comp_gauss.Sigma.frozen=1
        
        #tqdm creates a progress bar display:
        with tqdm(total=len(line_search_e_space)) as pbar:
            
            for j,energy in enumerate(line_search_e_space):
                    
                #exploring the parameter space for energy
                comp_gauss.LineE=energy
                
                #resetting the second constant factor value
                comp_cfactor_2.factor=1
            
                '''
                getting the equivalent width conversion for every energy
                careful: this only gives the eqwidth of the unabsorbed line
                
                for that we set the gaussian cflux to the continuum flux at this energy 
                since norm_par_space is directly in units of local continuum flux it will make it much easier to get all the eqwidths afterwards)
                '''
                comp_gauss_cflux.lg10Flux=np.log10(flux_cont[j])
    
                #Computing the eqwidth of a component works even with the cflux dependance.
                AllModels.eqwidth(len(AllModels(1).componentNames))
                
                #conversion in eV from keV included since the result is in keV
                eqwidth_conv[j]=AllData(1).eqwidth[0]*10**3
                
                '''
                exploring the norm parameter space in units of the continuum flux at this energy
                In order to do that, we add 2 steppar computations (for positive and negative norms) where we vary the constant factor
                in a similar manner to the norm par space
                '''
                
                #turning off the chatter to avoid spamming the console
                prev_xchatter=Xset.chatter
                
                Xset.chatter=0
                
                #first steppar in negative norm space 
                # -1 in the number of computations because steppar adds 1 
                comp_cfactor_1.factor=-1
                Fit.steppar('log '+str(index_cfactor_2)+' '+str(line_search_norm[1])+' '+str(line_search_norm[0])+\
                            ' '+str(int(line_search_norm[2]/2)-1))
    
                negchi_arr=np.array(Fit.stepparResults('statistic'))
                
                #second steppar in positive norm space
                comp_cfactor_1.factor=1
                Fit.steppar('log '+str(index_cfactor_2)+' '+str(line_search_norm[0])+' '+str(line_search_norm[1])+\
                            ' '+str(int(line_search_norm[2]/2)-1))
                    
                poschi_arr=np.array(Fit.stepparResults('statistic'))
                
                #returning the chatter to the previous value
                Xset.chatter=prev_xchatter
                
                chi_arr[j]=np.concatenate((negchi_arr,np.array([chi_base]),poschi_arr))
    
                pbar.update(1)
        
        #to compute the contour chi, we start from a chi with a fit with a line normalisation of 0
        chi_contours=[chi_base-9.21,chi_base-4.61,chi_base-2.3]
        
        #unused for now
        # #computing the negative (i.e. improvement) part of the delchi map for the autofit return
        # chi_arr_impr=np.where(chi_arr>=chi_base,0,abs(chi_base-chi_arr))
        
        '''Peak computation'''
        
        def peak_search(array_arg):
            
            #safeguard in case of extreme peaks dwarfing the other peaks
            if np.max(array_arg)>=2e2:
                array=np.where(array_arg>=1,array_arg**(1/2),array_arg)
            else:
                array=array_arg
                
            #choosing a method
            peak_finder= findpeaks(method='topology',whitelist='peak',denoise=None)
            peak_finder.fit(array)
            peak_result=peak_finder.results['persistence']
            
            #cutting the peaks for which the birth level (is zero) to avoid false detections
            peak_validity=peak_result['score']>0
            peak_result=np.array(peak_result)
            peak_result=peak_result[peak_validity]
            peak_result=peak_result.T[:2].astype(int)[::-1].T
            
            #computing the polygon points of the peak regions
            #since the polygons created from the mask are "inside" the edge of ther polygon, this can cause problems for thin polygons
            #We thus do one iteration of binary dilation to expand the mask by one pixel.
            #This is equivalent to having the polygon cover the exterior edges of the pixels.
            peak_result_points=Mask(binary_dilation(array_arg)).polygons().points
            
            return peak_result,peak_result_points
        
        #The library is very good at finding peaks but garbage at finding valleys, so we cut what's below
        #the chi difference limit and swap the valleys into peaks
        
        chi_arr_sub_thresh=np.where(chi_arr>=chi_base-peak_thresh,0,abs(chi_base-chi_arr))
        peak_points_raw,peak_polygons_points=peak_search(chi_arr_sub_thresh)
        
        peak_points=[]
        #limiting to a single peak per energy step by selecting the peak with the lowest chi squared for each energy bin
        for peak_e in np.unique(peak_points_raw.T[0]):
            
            chi_peak=chi_base
            
            for peak_norm in peak_points_raw.T[1][peak_points_raw.T[0]==peak_e]:
                if chi_arr[peak_e][peak_norm]<chi_peak:
                    chi_peak=chi_arr[peak_e][peak_norm]
                    maxpeak_pos=[peak_e,peak_norm]
            peak_points.append(maxpeak_pos)
            
        peak_points=np.array(peak_points)
        
        if peak_polygons_points!=[]:
        
            #making sure the dimension is correct even if a single polygon is detected 
            if type(peak_polygons_points[0][0])==np.int64:
                peak_polygons_points=np.array(peak_polygons_points)
        
            if len(peak_polygons_points)!=len(peak_points):
                print('\nThe number of peak and polygons is not identical.')
                
                if peak_clean:
                    print('\nRefining...')
                    #we refine by progressively deleting the remaining elements of the array until the shape splits
                    chi_arr_refine=chi_arr_sub_thresh.copy()
                    chi_refine_values=chi_arr_sub_thresh[chi_arr_sub_thresh.nonzero()]
                    chi_refine_values.sort()
                    index_refine=-1
                    peak_eq_pol=False
                    peak_points_ref=peak_points.copy()
                    
                    #the failure stop condition is when a peak gets deleted, which only happens if the refining was too strict
                    while not peak_eq_pol and len(peak_points_ref)==len(peak_points):
                        
                        index_refine+=1
        
                        #refining the chi array
                        chi_arr_refine=np.where(chi_arr_refine>chi_refine_values[index_refine],chi_arr_refine,0)
                        
                        #recomputing the peaks
                        peak_points_ref,peak_polygons_points_ref=peak_search(chi_arr_refine)
                        
                        #testing if the process worked. We use peak_points instead of peak_points_ref to avoid soliving deleting peaks
                        peak_eq_pol=len(peak_points)==len(peak_polygons_points_ref)
                
                    #We only replace the polygons if the refining did work
                    if len(peak_points_ref)==len(peak_points):
        
                        peak_polygons_points=peak_polygons_points_ref
            
            #creating a list of the equivalent shapely polygons and peaks
            peak_polygons=np.array([None]*min(len(peak_points),len(peak_polygons_points)))
            peak_points_shapely=np.array([None]*min(len(peak_points),len(peak_polygons_points)))
            
            for i_poly,elem_poly in enumerate(peak_polygons_points):
                
                if i_poly>=len(peak_points):
                    continue
                
                peak_polygons[i_poly]=Polygon(elem_poly)
                peak_points_shapely[i_poly]=Point(peak_points[i_poly][::-1])
            
            #linking each peak to its associated polygon and storing its width
            peak_widths=np.zeros(len(peak_points))
            
            for elem_point in enumerate(peak_points_shapely):
                for elem_poly in peak_polygons:
                    if elem_poly.contains(elem_point[1]):
                        #we substract 1 from the width since the bouding box considers one more pixel as in
                        peak_widths[elem_point[0]]=elem_poly.bounds[3]-elem_poly.bounds[1]-1
        else:
            peak_widths=[]
            
        #storing the chi² differences of the peaks
        peak_delchis=[]
        peak_eqws=[]
        if len(peak_points)!=0:
            for coords in peak_points:
                peak_delchis.append(chi_base-chi_arr[coords[0]][coords[1]])
                
                #since the stored eqwidth is for the continuum flux, multiplying it by norm_par_space and the step size ratio to 0.1
                #directly gives us all the eqwidths for an energy, since they scale linearly with the norm/cflux 
                peak_eqws.append(eqwidth_conv[coords[0]]*norm_par_space[coords[1]])
                
        if len(peak_points)>0:
            is_abspeak=((np.array(peak_eqws)<0) & (line_search_e_space[peak_points.T[0]]>=trig_interval[0]) &\
                        (line_search_e_space[peak_points.T[0]]<=trig_interval[1])).any()
        else:
            is_abspeak=False
        
        '''''''''''''''''
        ######PLOTS######
        '''''''''''''''''
        
        #creating some necessary elements

        #line threshold is the threshold for the symlog axis
        chi_dict_plot={
                    'chi_arr':chi_arr,
                    'chi_base':chi_base,
                    'line_threshold':line_search_norm[0],
                    'line_search_e':line_search_e,
                    'line_search_norm':line_search_norm,
                    'line_search_e_space':line_search_e_space,
                    'norm_par_space':norm_par_space,
                    'peak_points':peak_points,
                    'peak_widths':peak_widths,
                    'line_cont_range':line_cont_range,
                    'sat':sat,
                    'plot_ratio_values':plot_ratio_values,
            }
        
        color_title=r'color plot of the $\chi^2$ evolution for observ '+epoch_observ[0]+\
                    '\n with line par '+args.line_search_e+' and norm par'+args.line_search_norm+\
                    ' in continuum units'
        contour_title=r'contour plot of the $\chi^2$ evolution for for observ '+epoch_observ[0]+\
                      '\n with line par '+args.line_search_e+' and norm par'+args.line_search_norm+\
                      ' in continuum units'
        coltour_title=r'contour plot of the $\chi^2$ evolution for ofor observ '+epoch_observ[0]+\
                      '\n with line par '+args.line_search_e+' and norm par'+args.line_search_norm+\
                      ' in continuum units'
            
        comb_title=r' Blind search visualisation for observ '+epoch_observ[0]+'\n with line par '+args.line_search_e+\
                      ' and norm par'+args.line_search_norm+' in continuum units'

        comb_label=[]
        for i_grp in range(AllData.nGroups):
            if sat=='Chandra':
                label_grating=str(-1+2*i_grp)
            else:
                label_grating=''

            comb_label+=[('_'.join(epoch_observ[i_grp].split('_')[1:3])) if sat=='XMM' else\
                                        epoch_observ[0]+' order '+label_grating if sat=='Chandra' else '']
                
        #creating the figure                
        figure_comb=plt.figure(figsize=(15,10))
        
        comb_chi2map(figure_comb,chi_dict_plot,title=comb_title,comb_label=comb_label)
        
        #saving it and closing it
        plt.savefig(outdir+'/'+epoch_observ[0]+'_'+suffix+'_line_comb_plot_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png')
        plt.close(figure_comb)
        
        if suffix=='cont':
            return is_abspeak,peak_points,peak_widths,peak_delchis,peak_eqws

    cont_abspeak,cont_peak_points,cont_peak_widths,cont_peak_delchis,cont_peak_eqws=narrow_line_search(data_mod_high,'cont')
    
    '''
    Automatic line fitting
    '''
    
    #### Autofit
    
    abslines_table_str=None

    if autofit and (cont_abspeak or force_autofit) and not flag_lowSNR_line:
        
        '''
        See the fitmod code for a detailed description of the auto fit process
        '''
        
        #reloading the continuum fitcomp
        fitmod_cont.reload()
        
        #feching the list of components we're gonna use
        comp_lines=model_list('lines_resolved' if sat=='Chandra' and autofit_model=='lines' else autofit_model)
        
        #creating a new logfile for the autofit
        curr_logfile_write=Xset.openLog(outdir+'/'+epoch_observ[0]+'_xspec_log_autofit.log')
        curr_logfile_write.reconfigure(line_buffering=True)
        curr_logfile=open(curr_logfile_write.name,'r')
        
        #creating the fitmod object with the desired componets (we currently do not use comp groups)
        fitlines=fitmod(comp_lines,curr_logfile,curr_logfile_write,prev_fitmod=fitmod_cont)
        
        #global fit
        fitlines.global_fit(chain=True,directory=outdir,observ_id=epoch_observ[0])
            
        #storing the final fit
        data_autofit=allmodel_data()
                
        '''
        ####Refitting the continuum
        
        if refit_cont is set to True, we add a 3 step process to better estimate the continuum from the autofit lines.
        
        First, we relaunch a global fit iteration while blocking all line parameters in the broad band.
        We then refreeze the absorption and relaunch two global fit iterations in 3-10 (for the HID) and 4-10 (for the autofit continuum)
        '''
        
        if refit_cont:
            
            AllChains.clear()

            #saving the initial autofit result for checking purposes
            store_fit(mode='autofit_init')
            
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
            
            linespar_freeze=[]
            #storing the previous freeze states from each line component
            for comp in [elem for elem in fitlines.includedlist if elem is not None]:
                if comp.line:
                    linespar_freeze+=[[elem in comp.unlocked_pars for elem in comp.parlist]]

                    freeze(parlist=comp.unlocked_pars)

            #refitting in broad band for the nH
            AllData.notice('all')
            AllData.ignore('bad')
            AllData.ignore('**-'+str(e_sat_low)+' '+str(e_sat_high)+'-**')
            
            #thawing the absorption to allow improving its value
            if fitlines.glob_phabs.included:
                AllModels(1)(fitlines.glob_phabs.parlist[0]).frozen=False
                
            #we reset the value of the fixed abs to allow it to be free if it gets deleted and put again
            fitlines.fixed_abs=None

            fitlines.global_fit(chain=True,lock_lines=True,directory=outdir,observ_id=epoch_observ[0])
            
            AllChains.clear()
            
            store_fit(mode='broadband_post_auto')
        
            #storing the class
            fitlines.dump(outdir+'/'+epoch_observ[0]+'_fitmod_broadband_post_auto.pkl')
                
            #re fixing the absorption parameter and storing the value to retain it if the absorption gets taken off and tested again

            if fitlines.glob_phabs.included:
                broad_absval=AllModels(1)(fitlines.glob_phabs.parlist[0]).values[0]
                AllModels(1)(fitlines.glob_phabs.parlist[0]).frozen=True
            else:
                broad_absval=0
            
            fitlines.fixed_abs=broad_absval
            
            #refitting in hid band for the HID values
            AllData.notice('all')
            AllData.ignore('bad')
            AllData.ignore('**-'+str(hid_cont_range[0])+' '+str(hid_cont_range[1])+'-**')
            fitlines.global_fit(chain=True,lock_lines=True,directory=outdir,observ_id=epoch_observ[0])
                
            AllChains.clear()
            main_spflux=hid_fit_infos(fitlines,broad_absval,post_autofit=True)
            

            #restoring the line freeze states
            #we can safely assume than even if the parameter numbers changed, the line will keep the same position so we can apply the mask to the 
            #new set of parameters using the same component order as before
            i_line_thaw=0
            for comp in [elem for elem in fitlines.includedlist if elem is not None]:
                if comp.line:
                    #unfreezing the parameter with the mask created before
                    unfreeze(parlist=np.array(comp.parlist)[linespar_freeze[i_line_thaw]])

                    #updating the index of the line compoent to keep in line with the mask
                    i_line_thaw+=1
                    
            #refitting in the autofit range to get the newer version of the autofit and continuum
            AllData.notice('all')
            AllData.ignore('bad')
            AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')            
            fitlines.global_fit(chain=True,directory=outdir,observ_id=epoch_observ[0])
            
            #storing the final fit
            data_autofit=allmodel_data()
        
            
        #storing the final plot and parameters
        #screening the xspec plot 
        Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+"_screen_xspec_autofit")
        
        if sat=='Chandra':
            
            #plotting a zoomed version for HETG spectra
            AllData.ignore('**-6.5 '+str(float(min(9,e_sat_high)))+'-**')
            
            Plot_screen("ldata,ratio,delchi",outdir+'/'+epoch_observ[0]+"_screen_xspec_autofit_zoom")
            
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
            
        #rearranging the components in a format usable in the plot. The components start at the index 2 
        #(before it's the entire model x and y values)
        plot_autofit_cont=plot_autofit_comps[:2+len(addcomps_cont)]
        
        #same for the line components
        plot_autofit_lines=plot_autofit_comps[2+len(addcomps_cont):]
        
        '''
        Testing the overlap of absorption lines with emission lines
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
            
            #width limit test
            if AllModels(1)(line_fitcomp.parlist[1])*8<(line_search_e[1]-line_search_e[0]):
                
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
    
        #recording all the frozen parameters in the model to refreeze them during the fakeit fit
        #the parameters in the other data groups are all linked besides the constant factor which isn't frozen so there should be no problem
        autofit_forcedpars=[i for i in range(1,AllModels(1).nParameters+1) if AllModels(1)(i).frozen or AllModels(1)(i).link!='']
        
        '''
        Chain computation for the MC significance
        '''        

        #drawing parameters for the MC significance test later
        autofit_drawpars=np.array([None]*nfakes)
        
        print('\nDrawing parameters from the Chain...')
        for i_draw in range(nfakes):
            
            curr_simpar=AllModels.simpars()
            
            autofit_drawpars[i_draw]=np.array(curr_simpar).reshape(AllData.nGroups,AllModels(1).nParameters)
        
        #turning it back into a regular array
        autofit_drawpars=np.array([elem for elem in autofit_drawpars])
        
        #storing the parameter and errors of all the components, as well as their corresponding name
        autofit_parerrors,autofit_parnames=fitlines.get_usedpars_vals()
        
        print('\nComputing informations from the fit...')
        
        #### Computing line parameters
        
        #fetching informations about the absorption lines
        abslines_flux,abslines_eqw,abslines_bshift,abslines_delchi=fitlines.get_absline_info(autofit_drawpars)

        #clearing the chain before doing anything else
        AllChains.clear()
        
        Fit.perform()
        
        #computing the 3-sigma width without the MC to avoid issues with values being too different from 0
        abslines_width=fitlines.get_absline_width()
                
        '''
        Saving a "continuum" version of the model without absorption
        '''
        
        #We store the indexes of the absgaussian parameters to shift the rest of the parameters accordingly after
        #deleting those components
        abslines_parsets=np.array([elem.parlist for elem in fitlines.includedlist if elem is not None and elem.named_absline])

        abslines_parids=ravel_ragged(abslines_parsets)
        
        #covering for the case where the list was mono-dimensional 
        if type(abslines_parids)!=list:
            abslines_parids=abslines_parids.tolist()
        
        #switching to indexes instead of actual parameter numbers
        abslines_arrids=[elem-1 for elem in abslines_parids]
        
        #creating the list of continuum array indexes
        continuum_arrids=[elem for elem in np.arange(AllModels(1).nParameters) if elem not in abslines_arrids]
        
        #and parameters
        continuum_parids=[elem+1 for elem in continuum_arrids]
        
        #to create a drawpar array which works with thereduced model (this line works with several data groups)
        autofit_drawpars_cont=autofit_drawpars.T[continuum_arrids].T
        
        #same thing with the forced parameters list
        continuum_forcedpars=[np.argwhere(np.array(continuum_parids)==elem)[0].tolist()[0]+1 for elem in\
                              [elem2 for elem2 in autofit_forcedpars if elem2 in continuum_parids]]
        
        #deleting all absorption components (reversed so we don't have to update the fitcomps)
        for comp in [elem for elem in fitlines.includedlist if elem is not None][::-1]:
            if comp.named_absline:
                #note that with no rollback we do not update the values of the component so it has still its included status and everything else 
                comp.delfrommod(rollback=False)
            
        #storing the no abs line 'continuum' model
        data_autofit_noabs=allmodel_data()
        
        plot_ratio_autofit_noabs=store_plot('ratio')
        
        #plotting the combined autofit plot
        
        def autofit_plot(fig):
            
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
            
            axes[1].set_xlabel('Energy (keV)')
            axes[1].xaxis.set_label_position('bottom') 
            axes[1].set_ylabel('Fit ratio compared the sum of continuum and all emission lines')
            axes[1].set_xlim(line_cont_range)
            
            #we put the x axis on top to avoid it being hidden by the second subplot2aaa
            axes[1].xaxis.tick_bottom()
            axes[1].xaxis.set_label_position('bottom')
            for i_grp in range(AllData.nGroups):
                axes[1].errorbar(plot_ratio_autofit_noabs[i_grp][0][0], plot_ratio_autofit_noabs[i_grp][1][0],
                                        xerr=plot_ratio_autofit_noabs[i_grp][0][1],yerr=plot_ratio_autofit_noabs[i_grp][1][1],
                                        color=xcolors_grp[i_grp],ecolor=xcolors_grp[i_grp],linestyle='None')
            axes[1].axhline(y=1,xmin=0,xmax=1,color='green')
            
            #limiting the plot to the range of the line energy search
            axes[1].set_xlim(line_search_e_space[0],line_search_e_space[-1])
            
            plot_ratio_xind_rel=[np.array([elem for elem in np.where(plot_ratio_autofit_noabs[i_grp][0][0]>=line_search_e[0])[0]\
                                          if elem in np.where(plot_ratio_autofit_noabs[i_grp][0][0]<=line_search_e_space[-1])[0]])\
                                 for i_grp in range(AllData.nGroups)]
                
            #rescaling with errorbars (which are not taken into account by normal rescaling)

            plot_ratio_y_up=np.array([(plot_ratio_autofit_noabs[i_grp][1][0]+plot_ratio_autofit_noabs[i_grp][1][1])[plot_ratio_xind_rel[i_grp]] 
                                      for i_grp in range(AllData.nGroups)])

                
            plot_ratio_y_dn=np.array([(plot_ratio_autofit_noabs[i_grp][1][0]-plot_ratio_autofit_noabs[i_grp][1][1])[plot_ratio_xind_rel[i_grp]]
                                      for i_grp in range(AllData.nGroups)])
            axes[1].set_ylim(0.95*np.min(ravel_ragged(plot_ratio_y_dn)),1.05*np.max(ravel_ragged(plot_ratio_y_up)))
            
            #linestyles
            l_styles=['solid','dotted','dashed','dashdot']
    
            #plotting the delta ratio of the absorption components
            for i_line,ratio_line in enumerate(plot_autofit_ratio_lines):
                
                #fetching the position of the line compared to other line components to get identical alpha and ls values
                
                i_line_comp=np.argwhere(np.array(addcomps_lines)==addcomps_abslines[i_line])[0][0]
                
                #plotting each ratio when it is significantly different from the continuum, 
                #and with the same color coding as the component plot bove
                axes[1].plot(plot_autofit_cont[0][ratio_line<=1-1e-3],ratio_line[ratio_line<=1-1e-3],color='red',
                             alpha=1-i_line_comp*0.1,linestyle=l_styles[i_line_comp%4])
                
            '''Plotting the Standard absorption line energies'''
            
            plot_std_ener(axes[1])
            
            plt.tight_layout()
            
        fig_autofit=plt.figure(figsize=(15,10))
        
        autofit_plot(fig_autofit)
        
        plt.savefig(outdir+'/'+epoch_observ[0]+'_autofit_components_plot_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png')
        plt.close(fig_autofit)

        '''
        Autofit residuals assessment
        '''
    
        narrow_line_search(data_autofit,'autofit')
        
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
        
        The computation can be forced by setting assess_ul_detec to True
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
        
        Xset.logChatter=2
        
        '''
        In order to make the fakeit process as fast as possible, we don't write the fakes in file 
        and directly fit them as loaded every time
        For that, we specify fakeitsettings with given filename, bg, resp etc. 
        (since by default fakeit overwrite those while loading the spectra)
        and use the nowrite command to keep from overwriting the files
        By not giving an exposure, we assume the loaded spectra's exposures
        '''
        
        print('\nCreating fake spectra to assess line significance...')
        
        fakeset=[FakeitSettings(response=AllData(i_grp).response.rmf,arf=AllData(i_grp).response.arf,
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
            line_upper_e=min(lines_e_dict[line_name][0]*(1+lines_e_dict[line_name][2]/c_light),e_sat_high)
            
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
                        
        if is_absline and assess_line:      

            #now we compute the list of intervals that can be made from that
            steppar_ind_unique=np.unique(steppar_ind_list)
            
            steppar_ind_inter=list(interval_extract(steppar_ind_unique))
        
            #fake loop
            with tqdm(total=nfakes) as pbar:
                for f_ind in range(nfakes):
                    
                    #reloading the high energy continuum
                    mod_fake=model_load(data_autofit_noabs)
                    
                    #Freezing it to ensure the fakeit doesn't make the parameters vary, and loading them from a steppar
                    for i_grp in range(1,AllData.nGroups+1):
                        
                        #freezing doesn't change anything for linked parameters
                        freeze(AllModels(i_grp))
                            
                        AllModels(i_grp).setPars(autofit_drawpars_cont[f_ind][i_grp-1].tolist())
                        
                    #replacing the current spectra with a fake with the same characteristics so this can be looped
                    #applyStats is set to true but shouldn't matter for now since everything is frozen

                    AllData.fakeit(settings=fakeset,applyStats=True,noWrite=True)

                        
                    #ensuring we are in the correct energy range
                    AllData.notice('all')
                    AllData.ignore('bad')
                    AllData.ignore('**-'+str(line_cont_range[0])+' '+str(line_cont_range[1])+'-**')
            
                    #adjusting the fit and storing the chi² 
                    
                    for i_grp in range(1,AllData.nGroups+1):
                        #unfreezing the model 
                        unfreeze(AllModels(i_grp))
                        
                        #keeping the initially frozen parameters frozen
                        freeze(AllModels(i_grp),parlist=continuum_forcedpars)
                                        
                        #keeping the first constant factor frozen if necessary
                        if i_grp>1 and AllModels(1).componentNames[0]=='constant':
                            AllModels(i_grp)(1).frozen=False
                            
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
                        Fit.steppar('nolog '+str(mod_fake.nParameters-2)+' '+str(line_search_e_space[steppar_inter[0]])+\
                                    ' '+str(line_search_e_space[steppar_inter[1]])+' '\
                                   +str(steppar_inter[1]-steppar_inter[0]))
                            
                        #updating the delchi array with the part of the parameters that got updated
                        delchi_arr_fake[f_ind][steppar_inter[0]:steppar_inter[1]+1]=\
                            abs(np.array([min(elem,0) for elem in Fit.stepparResults('delstat')]))
                    
                    Xset.chatter=5
                    Xset.logChatter=5
                    
                    #not used anymore because it was not what we wanted
                    # '''
                    # computing an upper limit distribution for the photon noise 90\% EQW UL of each line
                    # '''
                                        
                    # if assess_ul_detec:
                    #     for i_line in range(len(abslines_sign)):
                        
                    #         #skipping the computation for lines beyond the iron Ka complex in restrict mode when we don't go above 8keV anyway
                    #         if restrict_graded or restrict_fakes and i_line>=2:
                    #             continue
                            
                    #         #here we skip the first two emission lines
                    #         line_name=list(lines_e_dict.keys())[i_line+3]
                            
                    #         #fetching the lower and upper bounds of the energies from the blueshifts
                    #         line_upper_e=lines_e_dict[line_name][0]*(1+lines_e_dict[line_name][2]/c_light)
                            
                    #         #putting the energy at the line maximum energy
                    #         mod_fake(AllModels(1).nParameters-2).values=[line_upper_e]+mod_fake(AllModels(1).nParameters-2).values[1:]
                            
                    #         mod_fake(AllModels(1).nParameters-2).frozen=True
                    #         Fit.perform()
                            
                    #         #computing the 90% error on the eqw
                    #         #note: this crashes in multigroup but passing the error makes it work for some reason
                    #         try:
                    #             AllModels.eqwidth(len(AllModels(1).componentNames),err=True,number=100,level=90)
                    #         except:
                    #             pass
                            
                    #         #and storing the absolute value of the lower bound (the eqwidth tuple has the lower bound in second)
                    #         #note: we always take the value of the first data group as it is the one with the 'correct' (constant=1) normalisation
                    #         eqw_arr_fake[f_ind][i_line]=abs(AllData(1).eqwidth[1])*1e3
                                
                            
                    pbar.update(1)
                    
            # #storing the EQW array
            # np.save(outdir+'/'+epoch_observ[0]+'_eqw_arr_fake.npy',eqw_arr_fake)
            
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
                
                # #assessing the EQW upper limit for each line
                # if assess_ul_detec:
                #     eqw_arr_fake_line=eqw_arr_fake.T[i_line]
                #     eqw_arr_fake_line.sort()
                    
                '''
                computing the UL for detectability at the given threshold
                Here we convert the delchi threshold for significance to the EW that we would obtain for a line at the maximum blueshift 
                '''
                
        #reloading the initial spectra for any following computations
        Xset.restore(outdir+'/'+epoch_observ[0]+'_mod_autofit.xcm')
            
        #and saving the delchi array
        np.save(outdir+'/'+epoch_observ[0]+'_delchi_arr_fake_line.npy',delchi_arr_fake_line)
              
        '''
        ####Line fit upper limits
        '''
        
        #reloading the autofit model with no absorption to compute the upper limits
        model_load(data_autofit_noabs)
        
        #computing a mask for significant lines
        mask_abslines_sign=abslines_sign>sign_threshold
                
        #computing the upper limits for the non significant lines
        try:
            abslines_eqw_upper=fitlines.get_eqwidth_uls(mask_abslines_sign,abslines_bshift,sign_widths_arr)
        except:
            breakpoint()                    
        #here will need to reload an accurate model before updating the fitcomps
        '''HTML TABLE FOR the pdf summary'''
                    
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
                <th width="14%">blueshift<br></th>
                <th width="9%">width</th>
                <th width="14%">equivalent width</th>
                <th width="9%">EW '''+str(sign_threshold)+''' UL</th>
                <th width="6%">significance</th>
                <th width="6%">delchi²</th>
              </tr>
            </thead>
            <thead>
              <tr>
                <th width="10%"> </th>
                <th width="10%"> keV </th>
                <th width="10%"> </th>
                <th width="15%">1e-12 erg/s/cm²</th>
                <th width="15%">km/s<br></th>
                <th width="9%">eV(+-3sigma)</th>
                <th width="15%">eV</th>
                <th width="10%">eV </th>
                <th width="7%"></th>
                <th width="7%"></th>
              </tr>
            </thead>
            <tbody>
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
        autofit_store_str=epoch_observ[0]+'\t'+str(abslines_eqw.tolist())+'\t'+str(abslines_bshift.tolist())+'\t'+\
            str(abslines_delchi.tolist())+'\t'+str(abslines_flux.tolist())+'\t'+str(abslines_sign.tolist())+'\t'+\
            str(abslines_eqw_upper.tolist())+'\t'+str(abslines_em_overlap.tolist())+'\t'+str(abslines_width.tolist())+'\t'+str(autofit_parerrors.tolist())+'\t'+\
            str(autofit_parnames.tolist())+'\n'
            
    else:
        autofit_store_str=epoch_observ[0]+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\t'+'\n'
    
    
    '''Storing the results'''
    
    #### result storing
    
    #we test for the low SNR flag to ensure not overwriting line results for good quality data by mistake if launching the script without autofit
    if autofit or flag_lowSNR_line:
        
        autofit_store_header='Observ_id\tabslines_eqw\tabslines_bshift\tablines_delchi\tabslines_flux\t'+\
        'abslines_sign\tabslines_eqw_upper\tabslines_em_overlap\tabslines_width\tautofit_parerrors\tautofit_parnames\n'
        
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
            
        pdf_summary(epoch_observ,fit_ok=True,summary_epoch=fill_result('Line detection complete.'))
                
    #closing the logfile for both access and Xspec
    curr_logfile.close()    
    Xset.closeLog()
    
    return fill_result('Line detection complete.')

'''
#### Epoch matching
'''


if sat=='XMM':
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
            curr_tested_epoch_id=np.argwhere(np.array(spfile_list)==curr_tested_epoch)[0][0]
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
        
elif sat=='Chandra':
    epoch_list=[]
    epoch_list_started=[]
    obsid_list_chandra=np.unique([elem.split('_')[0] for elem in spfile_list])
    for obsid in obsid_list_chandra:
        epoch_list+=[[elem for elem in spfile_list if elem.startswith(obsid)]]

    obsid_list_started_chandra=np.unique([elem.split('_')[0] for elem in started_expos[1:]])
    for obsid in obsid_list_started_chandra.tolist():
        epoch_list_started+=[[elem,elem.replace('-1','1')] for elem in started_expos if elem.startswith(obsid)]
        
elif sat in ['NICER','Suzaku','Swift']:
    epoch_list=[]
    epoch_list_started=[]
    if sat=='Swift':
        obsid_list=np.unique([elem[:11] for elem in spfile_list])       
    else:
        obsid_list=np.unique([elem.split('_')[0] for elem in spfile_list])
        
    for obsid in obsid_list:

        epoch_list+=[[elem for elem in spfile_list if elem.startswith(obsid)]]
    
    if sat=='Swift':
        obsid_list_started=np.unique([elem.split('_')[0][:11] for elem in started_expos[1:]])
    else:
        obsid_list_started=np.unique([elem.split('_')[0] for elem in started_expos[1:]])
        
    for obsid in obsid_list_started.tolist():
        epoch_list_started+=[[elem] for elem in started_expos if elem.startswith(obsid)]
    
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

    
    #skip start check
    if sat in ['Suzaku']:
        if skip_started and epoch_files in epoch_list_started:
             print('\nSpectrum analysis already performed. Skipping...')
             continue
    elif sat=='Swift':
        if skip_started and sum([[elem] in epoch_list_started for elem in epoch_files])>0:
             print('\nSpectrum analysis already performed. Skipping...')
             continue
         
    elif sat in ['Chandra','NICER','XMM']:
        
        ###TODO: same thing for XMM
        if (skip_started and len([elem_sp for elem_sp in epoch_files[:1] if elem_sp.split('_sp')[0] not in started_expos])==0) or \
           (skip_complete and len([elem_sp for elem_sp in epoch_files[:1] if elem_sp.split('_sp')[0] not in done_expos])==0):
            print('\nSpectrum analysis already performed. Skipping...')
            continue
    
    #overwrite check
    if overwrite==False and os.path.isfile(outdir+'/'+firstfile_id+'_recap.pdf'):
        print('\nLine detection already computed for this exposure. Skipping...')
        continue
    
    #we don't use the error catcher/log file in restrict mode to avoid passing through bpoints
    if restrict==False:

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
        if summary_lines is not None:
            #creating the text of the summary line for this observation
            
            if '_' in firstfile_id:
                obsid_id=firstfile_id[:firstfile_id.find('_')]
                file_id=firstfile_id[firstfile_id.find('_')+1:]
            else:
                obsid_id=firstfile_id
                file_id=obsid_id
                
            summary_content=obsid_id+'\t'+file_id+'\t'+str(summary_lines.tolist())
            
            #adding it to the summary file
            file_edit(outdir+'/'+'summary_line_det.log',
                      obsid_id+'\t'+file_id,summary_content+'\n',summary_header)
        
    else:
        summary_lines=line_detect(epoch_id)

if multi_obj==False:
    #loading the diagnostic messages after the analysis has been done
    if os.path.isfile(os.path.join(outdir,'summary_line_det.log')):
        with open(os.path.join(outdir,'summary_line_det.log')) as sumfile:
            glob_summary_linedet=sumfile.readlines()[1:]
    
    #creating summary files for the rest of the exposures
    lineplots_files=[elem.split('/')[1] for elem in glob.glob(outdir+'/*',recursive=True)]
    
    if sat=='XMM':
        aborted_epochs=[epoch for epoch in epoch_list if not epoch[0].split('_sp')[0]+'_recap.pdf' in lineplots_files]
        
    elif sat in ['Chandra','Swift']:
        aborted_epochs=[[elem.replace('_grp_opt'+('.pi' if sat=='Swift' else '.pha'),'') for elem in epoch]\
                        for epoch in epoch_list if not epoch[0].split('_grp_opt'+('.pi' if sat=='Swift' else '.pha'))[0]+'_recap.pdf'\
                            in lineplots_files]
    elif sat in ['NICER','Suzaku']:
        aborted_epochs=[[elem.replace('_sp_grp_opt.pha','') for elem in epoch]\
                        for epoch in epoch_list if not epoch[0].split('_sp_grp_opt.pha')[0]+'_recap.pdf' in lineplots_files]        
    for elem_epoch in aborted_epochs:
        if sat=='XMM':
            epoch_observ=[elem_file.split('_sp')[0] for elem_file in elem_epoch]
        elif sat in ['Chandra','Swift']:
            epoch_observ=[elem_file.split('_grp_opt')[0] for elem_file in elem_epoch]
        elif sat in ['NICER','Suzaku']:
            epoch_observ=[elem_file.split('_sp_grp_opt')[0] for elem_file in elem_epoch]
        #list conversion since we use epochs as arguments        
        pdf_summary(epoch_observ)
     
'''''''''''''''''''''''''''''''''''''''
''''''Hardness-Luminosity Diagrams''''''
'''''''''''''''''''''''''''''''''''''''

'Distance and Mass determination'

catal_blackcat,catal_watchdog,catal_blackcat_obj,catal_watchdog_obj,catal_maxi_df,catal_maxi_simbad=load_catalogs()
    
all_files=glob.glob('**',recursive=True)
lineval_id='line_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
lineval_files=[elem for elem in all_files if outdir+'/' in elem and lineval_id in elem]

abslines_id='autofit_values_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.txt'
abslines_files=[elem for elem in all_files if outdir+'/' in elem and abslines_id in elem]

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
    'cameras':cameras,
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
observ_list,lineval_list,flux_list,date_list,instru_list=obj_values(lineval_files,Edd_factor,dict_linevis)

dict_linevis['flux_list']=flux_list

#the values here are for each observation
abslines_infos,autofit_infos=abslines_values(abslines_files,dict_linevis)

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
        flux_list[i]=np.delete(flux_list[i],bad_index,axis=0)
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
        flux_list[0]=np.delete(flux_list[0],bad_index,axis=0)
        # link_list[0]=np.delete(link_list[0],bad_index)

#checking if the obsid identifiers of every index is in the bad flag list or if there's just no file
if len(observ_list.ravel())==0:
    print('\nNo line detection to build HID graph.')
    flag_noexp=True
else:
    flag_noexp=False
'''Diagram creation'''

'''2D'''

mpl.rcParams.update({'font.size': 10})

fig_hid,ax_hid=plt.subplots(1,1,figsize=(16,8))

plt.subplots_adjust(bottom=0.25)

#axe definition for the sliders (which require their own 'axe')
ax_slider_width = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_slider_e = plt.axes([0.25, 0.15, 0.65, 0.03])

# slider for the width limit
valinit_widthlim=5
slider_width=Slider(ax=ax_slider_width,label='Maximal allowed width (keV)',valmin=0.1,valmax=line_search_e[1]-line_search_e[0],
                    valinit=valinit_widthlim) 

#slider for the energy limits
valinit_e=(6.,9.)
val_allowed_e=line_search_e_space[:-1]
slider_e=RangeSlider(ax=ax_slider_e,label='Energy limits threshold (keV)',valmin=line_search_e[0],valmax=line_search_e[1],
                     valstep=val_allowed_e,color='blue',valinit=valinit_e)



#booleans for displaying the emission/absorption lines
if paper_look:
    disp_pos=False
else:
    disp_pos=True

disp_neg=True
disp_noline=True

#switch to only plot the cb once
cb_plot=0

#to avoid a lot of complexity, we simply recreate the entire graph whenever we update the sliders or buttons
def update_graph(val=None):

    #detecting if a colorbar exists and clearing it if so
    for elem_axes in fig_hid.get_axes():
        #the colorbar will have this position
        if str(elem_axes)[5:9]=='0.92':
            elem_axes.remove()

    #ax definition for the colorbar
    if disp_chi2:
        ax_cb=plt.axes([0.92, 0.25, 0.02, 0.63])
        
    width_lim=slider_width.val
    elim_low=slider_e.val[0]
    elim_high=slider_e.val[1]
    
    ax_hid.clear()
    
    #log x scale for an easier comparison with Ponti diagrams
    ax_hid.set_xscale('log')
    ax_hid.set_xlabel('Hardness Ratio ([6-10]/[3-6] keV bands)')
    ax_hid.set_ylabel(r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units')
    ax_hid.set_yscale('log')
    
    '''Dichotomy'''
    
    #determining the indexes for which there are emission/absorption lines for each object and storing the corresponding values
    
    ind_line_pos_list=np.array([None]*len(obj_list))
    ind_line_neg_list=np.array([None]*len(obj_list))
    ind_noline_list=np.array([None]*len(obj_list))
    
    max_delchi_pos_list=np.array([None]*len(obj_list))
    max_delchi_neg_list=np.array([None]*len(obj_list))
    
    max_delchi_eqw_pos_list=np.array([None]*len(obj_list))
    max_delchi_eqw_neg_list=np.array([None]*len(obj_list))
    
    scatter_noline=np.array([None]*len(obj_list))
    scatter_em=np.array([None]*len(obj_list))
    scatter_abs=np.array([None]*len(obj_list))
    
    all_delchis=[]
    
    for o in range(len(obj_list)):
    
        #passing objects which do not have any point
        if len(lineval_list[o])==0:
               continue
        ind_line_pos=[]
        ind_line_neg=[]
        max_delchi_pos=[]
        max_delchi_neg=[]
        
        #the maximum eqwidths can be for peaks that are not the most significant  in terms of chi**2
        #so we store the values of the eqwidths of the peaks with the biggest delchi
        max_delchi_eqw_pos=[]
        max_delchi_eqw_neg=[]
        
        for i,values in enumerate(lineval_list[o]):
            
            curr_max_delchi_pos=[]
            curr_max_delchi_neg=[]
            
            curr_max_eqw_pos=[]
            curr_max_eqw_neg=[]
            
            for j,peaks in enumerate(values[0]):
                if peaks in np.argwhere(line_search_e_space>=elim_low) and peaks in np.argwhere(line_search_e_space<=elim_high):
                    
                    #limiting the detection depending on the width of the line
                    if values[2][j]*line_search_e[2]<=width_lim:
                        
                    #dichotomy between positive and negative norm values
                        if norm_par_space[values[1][j]]>=0:
                            if disp_pos:
                                #limiting to a single index per energy
                                if i not in ind_line_pos:
                                    ind_line_pos.append(i)
                                    
                                #saving the norm
                                curr_max_delchi_pos.append(values[3][j])
                                
                                #and the related eqwidth
                                #here we directly adapt it (minimum at 5eV and size times 20) for plotting purposes
                                #we use abs since we store absorption lines as negative eqws
                                curr_max_eqw_pos.append(20*max(abs(values[4][j]),5))
                        else:
                            if disp_neg:
                                if  i not in ind_line_neg and disp_neg:
                                    ind_line_neg.append(i)
                                    
                                curr_max_delchi_neg.append(values[3][j])
                                
                                #and the related eqwidth
                                #here we directly adapt it (minimum at 5eV and size times 20) for plotting purposes
                                curr_max_eqw_neg.append(20*max(abs(values[4][j]),5))
            
            #now we add the extremal values to the lists
            if curr_max_delchi_pos!=[]:
                max_delchi_pos.append(max(curr_max_delchi_pos))
                max_delchi_eqw_pos.append(curr_max_eqw_pos[np.array(curr_max_delchi_pos).argmax()])
            if curr_max_delchi_neg!=[]:
                max_delchi_neg.append(max(curr_max_delchi_neg))
                max_delchi_eqw_neg.append(curr_max_eqw_neg[np.array(curr_max_delchi_neg).argmax()])
                
        # #normalisation of the norm values to be used as marker sizes
        # max_norm_pos_plot=np.array(max_norm_pos)*500/max(norm_par_space)
        # max_norm_neg_plot=np.array(max_norm_neg)*500/max(norm_par_space)
        
        # #transforming the widths into the real values for the graph
        # max_width_pos_plot=np.array(max_width_pos)*line_search_e[2]
        # max_width_neg_plot=np.array(max_width_neg)*line_search_e[2]
        
        # #getting the extremes of the widths to get a common color bar
        # all_widths=np.concatenate([max_width_pos_plot,max_width_neg_plot],axis=0)
        # min_width,max_width=all_widths.min(),all_widths.max()
            
        #only computable if either of the lines are selected
        if ind_line_pos!=[] or ind_line_neg!=[]:
            #getting the extremes of the delchi differences to get a common color bar
            all_delchis=np.concatenate((all_delchis,max_delchi_pos,max_delchi_neg),axis=0)
        
        #creating indexes for non detections
        if disp_noline:
            ind_noline=[elem for elem in range(len(observ_list[o])) if (elem not in ind_line_pos and elem not in ind_line_neg)]
        else:
            ind_noline=[]
            
        #since there is a bug where single plots in scatter don't register for link creations we double the points if there are alone
        if len(ind_noline)==1:
            ind_noline+=ind_noline
        if len(ind_line_pos)==1:
            ind_line_pos+=ind_line_pos
        if len(ind_line_neg)==1:
            ind_line_neg+=ind_line_neg
            
        ind_line_pos_list[o]=ind_line_pos
        ind_line_neg_list[o]=ind_line_neg
        ind_noline_list[o]=ind_noline
        
        max_delchi_pos_list[o]=max_delchi_pos
        max_delchi_neg_list[o]=max_delchi_neg

        max_delchi_eqw_pos_list[o]=max_delchi_eqw_pos
        max_delchi_eqw_neg_list[o]=max_delchi_eqw_neg
        
    if len(all_delchis)!=0:
        min_delchi,max_delchi=all_delchis.min(),all_delchis.max()
        
    #switching to square roots if the max delchi is too big
        if max_delchi>=1e3:
            bigpeak_flag=1
            min_delchi,max_delchi=min_delchi**(1/2),max_delchi**(1/2)
            all_delchis=all_delchis**(1/2)
            
        else:
            bigpeak_flag=0
        
    '''Actual plotting'''
    
    #creating the color coding for the multi object plot
    cmap_space_plasma=mpl.cm.get_cmap('tab20',len(obj_list))
    cmap_vars_plasma=cmap_space_plasma(range(len(obj_list)))
    
    #in obj chi2 mode we plot a single label per type of detection so we don't reset the label counter in the object loop
    label_noline_plotted=0
    label_abs_plotted=0
    label_em_plotted=0
    
    if len(obj_list)==1:
        fig_hid.suptitle('Hardness Luminosity diagram with line detections for '+obj_list[0]+'\n with '+args.cameras+' camera(s), with line pars '\
                      +args.line_search_e+' and norm pars '+args.line_search_norm+' in continuum units')
    else:
        fig_hid.suptitle('Global Hardness Luminosity diagram with line detections\nwith '+args.cameras+' camera(s), line pars'\
                      +args.line_search_e+' and norm pars '+args.line_search_norm+' in continuum units')

    if paper_look:
        marker_abs='o'
        size_noline=50
        marker_noline='d'
    else:
        marker_abs='x'
        size_noline=300
        marker_noline='.'

    for o in range(len(obj_list)):
        
        #in obj color mode we plot one object per label so we reset the label counter at each object
        label_obj_plotted=0
        
        if ind_noline_list[o]!=[]:
            
            #label definition
            if disp_chi2 and label_noline_plotted==0:                    
                    label_noline='no detection above 3 sigma'
                    label_noline_plotted=1
            elif disp_chi2==False and label_obj_plotted==0:
                    label_noline=obj_list[o]
                    label_obj_plotted=1
            else:
                label_noline=None
               
            #scatter plot for the non detections

            scatter_noline[o]=ax_hid.scatter(flux_list[o].T[2][0][ind_noline_list[o]]/flux_list[o].T[1][0][ind_noline_list[o]],
                                                     flux_list[o].T[4][0][ind_noline_list[o]],
                                                     marker=marker_noline,s=size_noline,color='grey' if disp_chi2 else cmap_vars_plasma[o],
                                                     label=label_noline)
            # scatter_noline[o].set_urls(link_list[o][ind_noline_list[o]])
        
        
        #scatter plot for the absorption lines
        if ind_line_neg_list[o]!=[]:
            
            #label definition
            if disp_chi2 and label_abs_plotted==0:
                    label_abs='Absorption lines'
                    label_abs_plotted=1
            elif disp_chi2==False and label_obj_plotted==0:
                label_abs=obj_list[o]
                label_obj_plotted=1
            else:
                label_abs=None
            
            if disp_chi2:
                
                if len(max_delchi_neg_list[o])==1:
                    cmap_chi_values_abs=max_delchi_neg_list[o]+max_delchi_neg_list[o]
                else:
                    cmap_chi_values_abs=max_delchi_neg_list[o]
                
                scatter_abs[o]=ax_hid.scatter(flux_list[o].T[2][0][ind_line_neg_list[o]]/flux_list[o].T[1][0][ind_line_neg_list[o]],
                                                  flux_list[o].T[4][0][ind_line_neg_list[o]],marker=marker_abs,
                                                  s=max_delchi_eqw_neg_list[o],
                                                  c=cmap_chi_values_abs,cmap='plasma',label=label_abs)

                
                #adapting the scatter color to the global chi square range of the plot
                scatter_abs[o].set_clim(min_delchi,max_delchi)
            else:
                scatter_abs[o]=ax_hid.scatter(flux_list[o].T[2][0][ind_line_neg_list[o]]/flux_list[o].T[1][0][ind_line_neg_list[o]],
                                                  flux_list[o].T[4][0][ind_line_neg_list[o]],marker=marker_abs,
                                                      s=max_delchi_eqw_neg_list[o],
                                                      color=cmap_vars_plasma[o],label=label_abs)
            
            # scatter_abs[o].set_urls(link_list[o][ind_line_neg_list[o]])
        
            #creating the colorbar if it hasn't been created yet
            if cb_plot==0 and disp_chi2:
            
                cb=plt.colorbar(scatter_abs[o],cax=ax_cb)
                
                if bigpeak_flag==1:
                    cb.set_label(r'$\sqrt{\Delta\chi^2}$ of the most significant line',labelpad=10)
                else:
                    cb.set_label(r'$\Delta\chi^2$ of the most significant line',labelpad=10)


        #scatter plot for the emission lines
        if ind_line_pos_list[o]!=[] and cb_plot==0:
            
            if disp_chi2 and label_em_plotted==0:
                    label_em='Emission lines'
                    label_em_plotted=1
            elif disp_chi2==False and label_obj_plotted==0:
                label_em=obj_list[o]
                label_obj_plotted=1
            else:
                label_em=None
                
            if disp_chi2:
                
                if len(max_delchi_pos_list[o])==1:
                    cmap_chi_values_em=max_delchi_pos_list[o]+max_delchi_pos_list[o]
                else:
                    cmap_chi_values_em=max_delchi_pos_list[o]
                    
                scatter_em[o]=ax_hid.scatter(flux_list[o].T[2][0][ind_line_pos_list[o]]/flux_list[o].T[1][0][ind_line_pos_list[o]],
                                             flux_list[o].T[4][0][ind_line_pos_list[o]],marker='+',
                                             s=max_delchi_eqw_pos_list[o],
                                             c=cmap_chi_values_em,cmap='plasma',label=label_em)
            else:
                scatter_em[o]=ax_hid.scatter(flux_list[o].T[2][0][ind_line_pos_list[o]]/flux_list[o].T[1][0][ind_line_pos_list[o]],
                                             flux_list[o].T[4][0][ind_line_pos_list[o]],marker='+',
                                             s=max_delchi_eqw_pos_list[o],
                                             color=cmap_vars_plasma[o],label=label_em)
                
            # scatter_em[o].set_urls(link_list[o][ind_line_pos_list[o]])
            scatter_em[o].set_clim(min_delchi,max_delchi)
            
            if cb_plot==0 and disp_chi2:
                cb=plt.colorbar(scatter_em[o],cax=ax_cb)
                
                if bigpeak_flag==1:
                    cb.set_label(r'$\sqrt{\Delta\chi^2}$ of the most significant line',labelpad=10)
                else:
                    cb.set_label(r'$\Delta\chi^2$ of the most significant line',labelpad=10)

    #setting the limits of the graph
    
    #limits of the points
    hid_min_x=min([min(flux_list[i].T[2][0]/flux_list[i].T[1][0]) for i in range(len(flux_list))])
    hid_max_x=max([max(flux_list[i].T[2][0]/flux_list[i].T[1][0]) for i in range(len(flux_list))])
    
    hid_min_y=min([min(flux_list[i].T[4][0]) for i in range(len(flux_list))])
    hid_max_y=max([max(flux_list[i].T[4][0]) for i in range(len(flux_list))])
    
    #putting the axis limits at standard bounds or the points if the points extend further
    ax_hid.set_xlim((min(hid_min_x*0.9,0.1),max(hid_max_x*1.1,3)))
    ax_hid.set_ylim((min(hid_min_y*0.9,1e-5),max(hid_max_y*1.1,1)))
    
    
    if disp_chi2: 
        hid_legend=fig_hid.legend(framealpha=1)
    else:
        if paper_look:
            old_legend_size=mpl.rcParams['legend.fontsize']
            mpl.rcParams['legend.fontsize']=7
            hid_legend=fig_hid.legend(loc='upper right',ncol=1,bbox_to_anchor=(0.999,0.89))
            mpl.rcParams['legend.fontsize']=old_legend_size
        else:
            hid_legend=fig_hid.legend(loc='upper left',ncol=2,bbox_to_anchor=(0.01,0.99))
            
    #maintaining a constant marker size in the legend (but only for markers)
    for elem_legend in hid_legend.legendHandles:
        if type(elem_legend)==mpl.collections.PathCollection:
            if len(elem_legend._sizes)!=0:
                for i in range(len(elem_legend._sizes)):
                    elem_legend._sizes[i]=50

    #displaying few examples of the size to eqw conversion in a second legend
    if paper_look:
        hid_size_examples=[(Line2D([0],[0],marker=marker_abs,color='black',markersize=100**(1/2),linestyle='None')),
                            (Line2D([0],[0],marker=marker_abs,color='black',markersize=400**(1/2),linestyle='None')),
                            (Line2D([0],[0],marker=marker_abs,color='black',markersize=1000**(1/2),linestyle='None'))]
    else:
        #displaying few examples of the size to eqw conversion in a second legend
        hid_size_examples=[(Line2D([0],[0],marker='+',color='black',markersize=100**(1/2),linestyle='None'),
                            Line2D([0],[0],marker=marker_abs,color='black',markersize=100**(1/2),linestyle='None')),
                            (Line2D([0],[0],marker='+',color='black',markersize=400**(1/2),linestyle='None'),
                            Line2D([0],[0],marker=marker_abs,color='black',markersize=400**(1/2),linestyle='None')),
                            (Line2D([0],[0],marker='+',color='black',markersize=1000**(1/2),linestyle='None'),
                            Line2D([0],[0],marker=marker_abs,color='black',markersize=1000**(1/2),linestyle='None'))]
    
    fig_hid.legend(handles=hid_size_examples,loc='center left',labels=['<5 eV','20 eV','50 eV'],title='Equivalent widths',
               bbox_to_anchor=(0.001,0.2),handler_map = {tuple:mpl.legend_handler.HandlerTuple(None)},
               handlelength=6,handleheight=4,columnspacing=1.1)
    fig_hid.show()
    
slider_width.on_changed(update_graph)
slider_e.on_changed(update_graph)

#some naming variables for the files
save_dir='glob_batch' if multi_obj else outdir

if multi_obj:
    save_str_prefix=''
else:
    save_str_prefix=obj_list[0]+'_'
'''
BUTTONS
'''

if multi_obj:
    disp_chi2=False
    
    #defining a small delchi display button to show the save pdf button near
    ax_switch_chi2=plt.axes([0.01, 0.025, 0.05, 0.04])
    button_switch_chi2=Button(ax_switch_chi2,r'see $\Delta\chi^2$',hovercolor='grey')
    def switch_chi2(event):
        global disp_chi2
        if disp_chi2:
            disp_chi2=False
        else:
            disp_chi2=True
        update_graph()
    button_switch_chi2.on_clicked(switch_chi2)
    
    ax_save_pdf=plt.axes([0.07,0.025,0.08,0.04])
    
else:
    #there's no need to switch between object display and HID display here so we can have a bigger pdf display buttopn
    ax_save_pdf=plt.axes([0.03,0.025,0.1,0.04])
    
button_save_pdf=Button(ax_save_pdf,'Save pdf',hovercolor='grey')
    
def save_pdf(event=None):
    
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
    ax_hid=plt.gca().get_figure().get_axes()[0]
    
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
                args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+\
                '_elims_'+str(round(slider_e.val[0],2))+'-'+str(round(slider_e.val[1],2))+'_wlim_'+str(round(slider_width.val,3))+\
                '_em_'+str(int(disp_pos))+'_abs_'+str(int(disp_neg))+'_noline_'+str(int(disp_noline))+'.pdf'
                        
    #the main part of the summary creation is only gonna work if we have points in the graph
    if not flag_noexp:
        
        #rassembling all points
        #we concatenate all of the points array of the (up to 3) groups, and restrict uniques to avoid two points for absorption/emissions
        points_hid=np.unique(np.concatenate([linecols_hid[i].get_offsets().data for i in range(len(linecols_hid))]),axis=0)
        
        #saving the current graph in a pdf format
        plt.savefig(save_dir+'/curr_hid.png')
    
        #before adding it to the pdf
        pdf.image(save_dir+'/curr_hid.png',x=1,w=280)
        
        #resticting other infos to if they are asked
        #note : these are very outdated compared to everything visual_line does nowadays so we don't plot these by default
        
        if glob_summary_save_line_infos:
            
            #adding the initial statistic graphs
            pdf.add_page()
            
            page_blindsearch=pdf.page_no()
        
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(1,30,'Blind search repartition:',align='C',center=True)
        
            pdf.image(save_dir+'/graphs/'+save_str_prefix+'repartition_cam_'+args.cameras+'_'+args.line_search_e.replace(' ','_')+\
                      '_'+args.line_search_norm.replace(' ','_')+'.png',
                      x=-7,y=60,w=110)
            try:
                pdf.image(save_dir+'/graphs/'+save_str_prefix+'distrib_cam_'+args.cameras+'_emi_'+args.line_search_e.replace(' ','_')+\
                          '_'+args.line_search_norm.replace(' ','_')+'.png',
                          x=90,y=60,w=110)
            except:
                pass
            
            try:
                pdf.image(save_dir+'/graphs/'+save_str_prefix+'distrib_cam_'+args.cameras+'_abs_'+args.line_search_e.replace(' ','_')+\
                          '_'+args.line_search_norm.replace(' ','_')+'.png',
                          x=190,y=60,w=110)
            except:
                pass
            
            '''
            Autofit distribution graphs
            '''
            #only where absorption lines are detected
            if not flag_noabsline:
                #adding the 3 global graphs
                pdf.add_page()
                
                page_autofit_distrib=pdf.page_no()
                
                pdf.cell(1,30,'Parameter distributions for all the lines',align='C',center=True)
                graph_infos=['lineflux','eqw','bshift','ener']
                graph_title_str=['Line flux','Equivalent width','Blueshift','Energy']
                
                for ind,elem_info in enumerate(graph_infos):
                    pdf.image(save_dir+'/graphs/distrib/'+save_str_prefix+'autofit_distrib_'+elem_info+'_all_cam_'+args.cameras+'_'+\
                                    args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                    x=75*(ind%(len(graph_infos))),y=40+80*(ind//(len(graph_infos))),w=70)
                
                #Individual graphs
                for ind,elem_info in enumerate(graph_infos):
                    
                    pdf.add_page()
                    
                    pdf.cell(1,30,graph_title_str[ind]+' distributions for individual lines',align='C',center=True)
                    
                    for ind_line in range_absline:
                        pdf.image(save_dir+'/graphs/distrib/'+save_str_prefix+'autofit_distrib_'+elem_info+'_'+lines_std_names[3+ind_line]+'_cam_'+args.cameras+'_'+\
                                        args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                       x=95*(ind_line%3),y=30+90*(ind_line//3),w=90)
                
                '''
                Intrinsic correlation graphs
                '''
                
                #adding the 3 global graphs
                pdf.add_page()
                
                page_autofit_correl=pdf.page_no()
                
                pdf.cell(1,30,'Intrinsic line parameter scatter plots for all the lines',align='C',center=True)
                
                graph_infos=['bshift_eqw','ener_eqw','lineflux_bshift','lineflux_eqw']
                graph_title_str=['Blueshift - Equivalent Width','Energy - Equivalent Width','Line flux - Blueshift',
                                 'Line flux - Equivalent Width']
                
                for ind,elem_info in enumerate(graph_infos):
                    pdf.image(save_dir+'/graphs/intrinsic/'+save_str_prefix+'autofit_correl_'+elem_info+'_all_cam_'+args.cameras+'_'+\
                                    args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                    x=75*ind,y=60,w=70)
                
                #Individual graphs
                for ind,elem_info in enumerate(graph_infos):
                    
                    pdf.add_page()
                    
                    pdf.cell(1,30,graph_title_str[ind]+' scatter plots for individual lines',align='C',center=True)
                    
                    for ind_line in range_absline:
                        pdf.image(save_dir+'/graphs/intrinsic/'+save_str_prefix+'autofit_correl_'+elem_info+'_'+lines_std_names[3+ind_line]+\
                                  '_cam_'+args.cameras+'_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                        x=95*(ind_line%3),y=30+90*(ind_line//3),w=90)
                
                '''
                HID 1D Correlation graphs
                '''
                
                page_autofit_correl_hid=pdf.page_no()
                
                #adding the 6 global graphs
                graph_infos=['lineflux_HR','lineflux_flux','eqw_HR','bshift_HR','ener_HR','eqw_flux','bshift_flux','ener_flux']
                graph_title_str=['Line flux - Hardness Ratio','Line flux - Flux','Equivalent Width - Hardness Ratio','Blueshift - Hardness Ratio','Energy - Hardness Ratio',
                                 'Equivalent Width - Flux','Blueshift - Flux','Energy - Flux']
                pdf.add_page()
                
                pdf.cell(1,30,'Line-HID parameter scatter plots for all the lines',align='C',center=True)
                
                for ind,elem_info in enumerate(graph_infos):
                    pdf.image(save_dir+'/graphs/observ/'+save_str_prefix+'autofit_correl_'+elem_info+'_all_cam_'+args.cameras+'_'+\
                                    args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                    x=75*(ind%(len(graph_infos)/2)),y=40+80*(ind//(len(graph_infos)/2)),w=70)
                                            
                #Individual graphs
                for ind,elem_info in enumerate(graph_infos):
                    
                    pdf.add_page()
                    
                    pdf.cell(1,30,graph_title_str[ind]+' scatter plots for individual lines',align='C',center=True)
                    
                    for ind_line in range_absline:
                        pdf.image(save_dir+'/graphs/observ/'+save_str_prefix+'autofit_correl_'+elem_info+'_'+lines_std_names[3+ind_line]+\
                                  '_cam_'+args.cameras+'_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                       x=95*(ind_line%3),y=30+90*(ind_line//3),w=90)
                
                '''
                HID inclination Correlation graphs
                '''
                
                if multi_obj:
                    page_autofit_correl_incl=pdf.page_no()
                    
                    #adding the 6 global graphs
                    graph_infos=['lineflux_incl','eqw_incl','bshift_incl','ener_incl']
                    graph_title_str=['Line flux - Inclination','Equivalent Width - Inclination','Blueshift - Inclination','Energy - Inclination']
                    pdf.add_page()
                    
                    pdf.cell(1,30,'Line-inclination parameter scatter plots for all the lines',align='C',center=True)
                    
                    for ind,elem_info in enumerate(graph_infos):
                        pdf.image(save_dir+'/graphs/source/'+save_str_prefix+'autofit_correl_'+elem_info+'_all_cam_'+args.cameras+'_'+\
                                        args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                        x=75*(ind%(len(graph_infos)/2)),y=40+80*(ind//(len(graph_infos)/2)),w=70)
                                                
                    #Individual graphs
                    for ind,elem_info in enumerate(graph_infos):
                        
                        pdf.add_page()
                        
                        pdf.cell(1,30,graph_title_str[ind]+' scatter plots for individual lines',align='C',center=True)
                        
                        for ind_line in range_absline:
                            pdf.image(save_dir+'/graphs/source/'+save_str_prefix+'autofit_correl_'+elem_info+'_'+lines_std_names[3+ind_line]+\
                                      '_cam_'+args.cameras+'_'+args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'.png',
                                            x=95*(ind_line%3),y=30+90*(ind_line//3),w=90)
        '''
        identification of the points
        '''
        
        #fetching the number of pages after all the graphs
        init_pages=pdf.page_no()
        
        #this is only for single object mode
        if not multi_obj:
            outline_circles=np.array([None]*len(points_hid))
            
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
                point_observ=observ_list[0][np.argwhere(flux_list[0].T[4][0]==single_point[1])[0][0]]
                
                pdf.image(save_dir+'/curr_hid_highlight_'+str(point_id)+'.png',x=1,w=280)
                
                outline_circles[point_id][0].remove()

            for i in range(len(points_hid)):
                os.remove(save_dir+'/curr_hid_highlight_'+str(i)+'.png')
                
            os.remove(save_dir+'/curr_hid.png')
        
        #adding unfinished analysis section
        pdf.add_page()
            
        pdf.output(pdf_path)
        
        #listing the files in the save_dir
        save_dir_list=glob.glob(save_dir+'/*',recursive=True)
        
        #creating the merger pdf
        merger=PdfFileMerger()
        
        #adding the first page
        merger.append(pdf_path,pages=(0,init_pages))
        
        if glob_summary_save_line_infos:
            
            #adding the bookmarks
            #Note : Somehow every initial bookmark is shifted by 1 so we just correct it manually
            merger.addBookmark('Blind search repartition',page_blindsearch-1)
            
            if not flag_noabsline:
                merger.addBookmark('Autofit distribution graphs',page_autofit_distrib-1)
                bkm_1dcorrel=merger.addBookmark('1D scatter plots',page_autofit_correl-1)
                merger.addBookmark('Intrinsic line parameters',page_autofit_correl-1,parent=bkm_1dcorrel)
                merger.addBookmark('Intrinsic line / HID',page_autofit_correl_hid-1,parent=bkm_1dcorrel)
                if multi_obj:
                    merger.addBookmark('Intrinsic line / inclination',page_autofit_correl_incl-1,parent=bkm_1dcorrel)
            
        #stopping the pdf creation here for multi_obj mode
        if multi_obj:
            merger.write(save_dir+'/temp.pdf')
            merger.close()
            
            os.remove(pdf_path)
            os.rename(save_dir+'/temp.pdf',pdf_path)
            print('\nHLD summary PDF creation complete.')
            return
            
        bkm_completed=merger.addBookmark('Completed Analysis',len(merger.pages))
        #second loop to insert the recaps
        for point_id,single_point in enumerate(points_hid):
    
            #once again fetching the exposure identier
            point_observ=observ_list[0][np.argwhere(flux_list[0].T[4][0]==single_point[1])[0][0]]
            
            #there should only be one element here
            point_recapfile=[elem for elem in save_dir_list if point_observ+'_recap.pdf' in elem][0]
            
            #adding the corresponding hid highlight page
            merger.addBookmark(point_observ,len(merger.pages),parent=bkm_completed)
            merger.append(pdf_path,pages=(point_id+init_pages,point_id+init_pages+1))
            
            #adding the recap
            merger.append(point_recapfile)
            
            # #adding the aborted analysis section to the merger
            # merger.append(pdf_path,pages=(point_id+3,point_id+4))
    else:
        pdf.output(pdf_path)
    
        #creating the merger pdf
        merger=PdfFileMerger()
        # #adding the (empty) pdf in order to be able to add a bookmark
        # merger.append(pdf_path)
        
    bkm_aborted=merger.addBookmark('Aborted Analysis',len(merger.pages))
    for elem_epoch in aborted_epochs:
        curr_pages=len(merger.pages)
        merger.append(save_dir+'/'+elem_epoch[0].split('_sp')[0]+'_aborted_recap.pdf')
        bkm_completed=merger.addBookmark(elem_epoch[0].split('_sp')[0],curr_pages,parent=bkm_aborted)
        
    #overwriting the pdf with the merger, but not directly to avoid conflicts
    merger.write(save_dir+'/temp.pdf')
    merger.close()
    
    os.remove(pdf_path)
    os.rename(save_dir+'/temp.pdf',pdf_path)
    
    print('\nHLD summary PDF creation complete.')
    
button_save_pdf.on_clicked(save_pdf)


ax_save_hld=plt.axes([0.15, 0.025, 0.1, 0.04])
button_save_hld=Button(ax_save_hld,'Save HLD',hovercolor='grey')
def save_hld(event):
    '''
    Saves the current graph in a svg (i.e. with clickable points) format.
    '''

    plt.savefig(save_dir+'/'+save_str_prefix+'HLD_cam_'+args.cameras+'_'+\
                args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+\
                '_elims_'+str(round(slider_e.val[0],2))+'-'+str(round(slider_e.val[1],2))+'_wlim_'+str(round(slider_width.val,3))+\
                '_em_'+str(int(disp_pos))+'_abs_'+str(int(disp_neg))+'_noline_'+str(int(disp_noline))+'.svg')

button_save_hld.on_clicked(save_hld)

#Absorption/Emission switches
ax_switch_neg=plt.axes([0.75, 0.025, 0.05, 0.04])
button_switch_neg=Button(ax_switch_neg,'abs',hovercolor='grey')
def switch_neg(event):
    global disp_neg
    disp_neg=not disp_neg
    update_graph()
button_switch_neg.on_clicked(switch_neg)

ax_switch_noline=plt.axes([0.80, 0.025, 0.05, 0.04])
button_switch_noline=Button(ax_switch_noline,'no det',hovercolor='grey')
def switch_noline(event):
    global disp_noline
    disp_noline=not disp_noline
    update_graph()
button_switch_noline.on_clicked(switch_noline)

ax_switch_pos=plt.axes([0.85, 0.025, 0.05, 0.04])
button_switch_pos=Button(ax_switch_pos,'em',hovercolor='grey')
def switch_pos(event):
    global disp_pos
    disp_pos=not disp_pos
    update_graph()
button_switch_pos.on_clicked(switch_pos)

#reset button
ax_reset_we=plt.axes([0.95, 0.025, 0.05, 0.04])
button_reset_we=Button(ax_reset_we,'Reset',hovercolor='0.975')
def reset_we(event):
    #the reset doesn't work for the slider range so we do it manually instead
    slider_e.set_val(valinit_e)
    slider_width.reset()
button_reset_we.on_clicked(reset_we)
        
if len(obj_list)>1:
    disp_chi2=False
else:
    disp_chi2=True

if multi_obj:
    os.system('mkdir -p glob_batch')
    
plt.savefig(save_dir+'/'+save_str_prefix+'HLD_cam_'+args.cameras+'_'+\
                args.line_search_e.replace(' ','_')+'_'+args.line_search_norm.replace(' ','_')+'_def.svg')  

    
if flag_noexp and not multi_obj:
    save_pdf()
    sys.exit()
else:
    update_graph()
    
'''''''''''''''''''''
      DIAGRAMS
'''''''''''''''''''''
os.system('mkdir -p '+save_dir+'/graphs')
os.system('mkdir -p '+save_dir+'/graphs/distrib')
os.system('mkdir -p '+save_dir+'/graphs/intrinsic')
os.system('mkdir -p '+save_dir+'/graphs/observ')
os.system('mkdir -p '+save_dir+'/graphs/source')
'''
First blind search
'''

#fetching the distribution of all emission lines energies for all objects with an ugly conversion
distrib_e_em=np.array([np.array(lineval_list[i][j][0])[np.array(lineval_list[i][j][4])>0]\
                     for i in range(len(lineval_list)) for j in range(len(lineval_list[i]))],dtype=object)

#same process with absorption
distrib_e_abs=np.array([np.array(lineval_list[i][j][0])[np.array(lineval_list[i][j][4])<0]\
                      for i in range(len(lineval_list)) for j in range(len(lineval_list[i]))],dtype=object)

#counting the occurences for each combination
#we distinguish between single, double and >3 detections for emission lines only, absorption lines only, both, and no detection at all
distrib_n_em=[0,0,0]
distrib_n_abs=[0,0,0]
distrib_n_both=[0,0,0]
distrib_n_noline=[0,0,0]
for ind in range(len(distrib_e_em)):
    if len(distrib_e_em[ind])==0 and len(distrib_e_abs[ind])==0:
        distrib_n_noline[0]+=1
    if len(distrib_e_em[ind])!=0 and len(distrib_e_abs[ind])==0:
        distrib_n_em[min(len(distrib_e_em[ind]),3)-1]+=1
    if len(distrib_e_em[ind])==0 and len(distrib_e_abs[ind])!=0:
        distrib_n_abs[min(len(distrib_e_abs[ind]),3)-1]+=1
    if len(distrib_e_em[ind])!=0 and len(distrib_e_abs[ind])!=0:
        distrib_n_both[min(len(distrib_e_abs[ind]),len(distrib_e_abs[ind]),3)-1]+=1

distrib_n=np.array([distrib_n_em,distrib_n_abs,distrib_n_both,distrib_n_noline])

def distrib_blind_e(distrib,linetype):
    fig_hist,ax_hist=plt.subplots(1,1,figsize=(10,8))
    ax_hist.set_xlabel('Line peak energy')
    ax_hist.set_ylabel(r'Number of $>3\sigma$ peak detections')
    fig_hist.suptitle('Repartition of '+linetype+' lines peak detections with energy')
    ax_hist.hist(distrib,bins=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2],2*line_search_e[2]))

    #forcing only integer ticks on the y axis
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.show()

    plt.savefig(save_dir+'/graphs/'+save_str_prefix+'distrib_cam_'+args.cameras+'_'+\
                    linetype[:3]+'_'+args.line_search_e.replace(' ','_')+'_'
                +args.line_search_norm.replace(' ','_')+'.png')
    plt.close()
    
#converting the energy distribution to 1d and supressing rows with no detection
if np.sum(distrib_n_em)!=0 or np.sum(distrib_n_both)!=0:
    distrib_e_em=np.concatenate(distrib_e_em[[len(elem)!=0 for elem in distrib_e_em]])
    #converting indexes to actual energies
    distrib_e_em=np.array([line_search_e[0]+elem*line_search_e[2] for elem in distrib_e_em])

    distrib_blind_e(distrib_e_em,'emission')

if np.sum(distrib_n_abs)!=0 or np.sum(distrib_n_both)!=0:
    #same with absorption
    distrib_e_abs=np.concatenate(distrib_e_abs[[len(elem)!=0 for elem in distrib_e_abs]])
    distrib_e_abs=np.array([line_search_e[0]+elem*line_search_e[2] for elem in distrib_e_abs])

    distrib_blind_e(distrib_e_abs,'absorption')

def pie_blind_lines(distrib):
    fig_pie,ax_pie=plt.subplots(1,1,figsize=(10,8))
    if multi_obj==False:
        fig_pie.suptitle('Repartition of peak detections for '+obj_name)
    else:
        fig_pie.suptitle('Repartition of peak detections for the sample')
    pie_width=0.3
    #colors (grey for linedet, blue for emission, red for absorption, purple for both)
    em_color=[0,0,1,1]
    abs_color=[1,0,0,1]
    both_color=[0.7,0,1,1]
    noline_color=[0.1,0.1,0.1,0.6]
    alpha_range=[0.2,0.35,0.5]
    
    #defining the labels
    labels_outer=['Emission only','Absorption only','Double detections','No detection']
    labels_iner=[['1','2',r'>=3'],['1','2',r'>=3'],['1','2',r'>=3'],['X','X','X']]
    
    #replacing the labels for which there is no pie chart by empty values
    labels_outer=[labels_outer[ind] if np.sum(distrib[ind])!=0 else '' for ind in range(len(distrib))]
    labels_inner=[labels_iner[i][j] if distrib[i][j]!=0 else '' for i in range(len(distrib)) for j in range(len(distrib[i]))]
    
    #defining the color arrays
    outer_colors=np.array([em_color,abs_color,both_color,noline_color])
    inner_colors=np.array([em_color[:-1]+[elem] for elem in alpha_range]+[abs_color[:-1]+[elem] for elem in alpha_range]+
                           [both_color[:-1]+[elem] for elem in alpha_range]+[noline_color[:-1]+[elem] for elem in alpha_range])
    
    #and plotting
    ax_pie.pie(distrib.sum(axis=1),radius=1,colors=outer_colors,wedgeprops=dict(width=pie_width, edgecolor='w'),
               labels=labels_outer)
    ax_pie.pie(distrib.flatten(),radius=1-pie_width,colors=inner_colors,\
               wedgeprops=dict(width=pie_width, edgecolor='w'),labels=labels_inner,labeldistance=1-pie_width+0.1)
    plt.tight_layout()

    plt.savefig(save_dir+'/graphs/'+save_str_prefix+'repartition_cam_'+args.cameras+'_'+\
                    args.line_search_e.replace(' ','_')+'_'
                +args.line_search_norm.replace(' ','_')+'.png')

    plt.close()
    
pie_blind_lines(distrib_n)

'''
AUTOFIT LINES
'''

#Transforming the perline array into something that's easier to plot

flag_noabsline=False

    
'''
in this form, the new order is:
    -the info (4 rows, eqw/bshift/delchi/sign)
    -it's uncertainty (3 rows, main value/neg uncert/pos uncert,useless for the delchi and sign)
    -each habsorption line
    -the number of sources
    -the number of obs for each source
'''

#bin values for all the histograms below
#for the blueshift and energies the range is locked so we can use a global binning for all the diagrams
bins_bshift=np.concatenate(([-499],np.linspace(1,1e4,num=21,endpoint=True)))
bins_ener=np.arange(line_search_e[0],line_search_e[1]+line_search_e[2],2*line_search_e[2])

abslines_infos_perline,abslines_infos_perobj,abslines_plot,abslines_ener,flux_plot,hid_plot,incl_plot,width_plot,nh_plot=values_manip(abslines_infos,dict_linevis,autofit_infos)

#adding some dictionnary elements
dict_linevis['mask_lines']=dict_linevis['mask_lines']=np.repeat(True,n_absline)
dict_linevis['bins_bshift']=bins_bshift
dict_linevis['bins_ener']=bins_ener
dict_linevis['save_dir']=save_dir
dict_linevis['save_str_prefix']= save_str_prefix
dict_linevis['abslines_ener']=abslines_ener
dict_linevis['abslines_plot']=abslines_plot
dict_linevis['mask_obj']=np.repeat(True,len(obj_list))
dict_linevis['observ_list']=observ_list

'''Distributions'''

distrib_graph(abslines_plot,'lineflux',dict_linevis,save=True,close=True)
distrib_graph(abslines_plot,'lineflux',dict_linevis,indiv=True,save=True,close=True)
distrib_graph(abslines_plot,'eqw',dict_linevis,save=True,close=True)
distrib_graph(abslines_plot,'eqw',dict_linevis,indiv=True,save=True,close=True)
distrib_graph(abslines_plot,'bshift',dict_linevis,save=True,close=True)
distrib_graph(abslines_plot,'bshift',dict_linevis,indiv=True,save=True,close=True)
distrib_graph(abslines_plot,'ener',dict_linevis,abslines_ener,save=True,close=True)
distrib_graph(abslines_plot,'ener',dict_linevis,abslines_ener,indiv=True,save=True,close=True)

'''1-1 Correlations'''

'''Intrinsic line parameters'''
            
#plotting the intrinsic graphs
correl_graph(abslines_plot,'bshift_eqw',abslines_ener,dict_linevis,save=True,close=True)           
correl_graph(abslines_plot,'bshift_eqw',abslines_ener,dict_linevis,indiv=True,save=True,close=True)
correl_graph(abslines_plot,'ener_eqw',abslines_ener,dict_linevis,save=True,close=True)
correl_graph(abslines_plot,'ener_eqw',abslines_ener,dict_linevis,indiv=True,save=True,close=True)
correl_graph(abslines_plot,'lineflux_bshift',abslines_ener,dict_linevis,save=True,close=True)
correl_graph(abslines_plot,'lineflux_bshift',abslines_ener,dict_linevis,indiv=True,save=True,close=True)
correl_graph(abslines_plot,'lineflux_eqw',abslines_ener,dict_linevis,save=True,close=True)
correl_graph(abslines_plot,'lineflux_eqw',abslines_ener,dict_linevis,indiv=True,save=True,close=True)

#and the hid ones
correl_graph(abslines_plot,'lineflux_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'lineflux_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'lineflux_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'lineflux_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'eqw_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'eqw_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'eqw_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'eqw_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'bshift_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'bshift_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'bshift_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'bshift_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'ener_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'ener_HR',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)
correl_graph(abslines_plot,'ener_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',save=True,close=True)
correl_graph(abslines_plot,'ener_flux',abslines_ener,dict_linevis,mode_vals=hid_plot,mode='observ',indiv=True,save=True,close=True)


#and the inclination ones
if multi_obj:
    correl_graph(abslines_plot,'lineflux_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',save=True,close=True)
    correl_graph(abslines_plot,'lineflux_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',indiv=True,save=True,close=True) 
    correl_graph(abslines_plot,'eqw_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',save=True,close=True)
    correl_graph(abslines_plot,'eqw_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',indiv=True,save=True,close=True)     
    correl_graph(abslines_plot,'bshift_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',save=True,close=True)
    correl_graph(abslines_plot,'bshift_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',indiv=True,save=True,close=True)     
    correl_graph(abslines_plot,'ener_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',save=True,close=True)
    correl_graph(abslines_plot,'ener_incl',abslines_ener,dict_linevis,mode_vals=incl_plot,mode='source',indiv=True,save=True,close=True)     

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

from general_tools import file_edit,ravel_ragged,shorten_epoch,expand_epoch,get_overlap,interval_extract

from pdf_summary import pdf_summary
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

def reload_sp(baseload_path,keyword_skip=None,write_baseload=True,newbl_keyword='skip',method='new',mask=None):
    '''
    Reloads a baseload (with a list of spectra) but excluding files with a specific keyword in their names
    works by parsing the baseload and rewriting a new version without the corresponding line before loading them
    Note that this assumes that no bg model is after these spectra (otherwise it could create a mess)

    -baseload_path: base data loading file

    -keywod_skip: basic filter for the data file names to know which files to filter out

    -write_baseload: delete the baseload or not after loading the data

    -newbl_keyword: name for the new baseload file


    -method: "new" simply edits the AllData class. Requires a mask of datagroups to KEEP .
             "old" method was editing the baseload file
    '''

    new_baseload_path = baseload_path.replace('.xcm', '_' + newbl_keyword + '.xcm')

    if method=='new':

        #to avoid definition problems
        from xspec_config_multisp import AllData

        for i in np.arange(1,AllData.nGroups+1)[~mask][::-1]:

            #argument calling to be more explicit
            AllData.__isub__(int(i))
        if write_baseload:
            if os.path.isfile(new_baseload_path):
                os.remove(new_baseload_path)

            Xset.save(new_baseload_path)

    elif method=='old':
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
        with open(new_baseload_path,'w+') as f_baseload_skip:
            f_baseload_skip.writelines(new_baseload_lines)

        Xset.restore(new_baseload_path)

        if not write_baseload:
            os.remove(new_baseload_path)

def line_detect(epoch_id,arg_dict):
    '''
    line detection for a single epoch
    we use the index as an argument to fill the chi array
    '''

    nfakes=arg_dict['nfakes']
    epoch_list=arg_dict['epoch_list']
    sat_glob=arg_dict['sat_glob']

    megumi_files=arg_dict['megumi_files']
    outdir=arg_dict['outdir']
    broad_HID_mode=arg_dict['broad_HID_mode']

    #the arg one
    line_search_e_arg=arg_dict['line_search_e_arg']
    line_search_norm_arg=arg_dict['line_search_norm_arg']

    sign_threshold=arg_dict['sign_threshold']
    fitstat=arg_dict['fitstat']
    skip_nongrating=arg_dict['skip_nongrating']
    h_update=arg_dict['h_update']
    pdf_only=arg_dict['pdf_only']
    line_ul_only=arg_dict['line_ul_only']
    reload_autofit=arg_dict['reload_autofit']
    pileup_lim=arg_dict['pileup_lim']
    pileup_missing=arg_dict['pileup_missing']
    max_bg_imaging=arg_dict['max_bg_imaging']
    skipbg_timing=arg_dict['skip_bg_timing']
    SNR_min=arg_dict['SNR_min']
    restrict_order=arg_dict['restrict_order']
    NICER_bkg=arg_dict['NICER_bkg']
    force_ener_bounds=arg_dict['force_ener_bounds']
    hid_cont_range=arg_dict['hid_cont_range']
    line_cont_range=arg_dict['line_cont_range']
    line_cont_range_arg=arg_dict['line_cont_range_arg']
    line_search_e=arg_dict['line_search_e']
    counts_min=arg_dict['counts_min']
    fit_lowSNR=arg_dict['fit_lowSNR']
    counts_min_HID=arg_dict['counts_min_HID']
    catch_errors=arg_dict['catch_errors']
    filter_NuSTAR_SNR=arg_dict['filter_NuSTAR_SNR']
    cameras=arg_dict['cameras']
    expmodes=arg_dict['expmodes']
    autofit_store_path=arg_dict['autofit_store_path']
    write_pdf=arg_dict['write_pdf']
    cont_model=arg_dict['cont_model']
    split_fit=arg_dict['split_fit']
    compute_highflux_only=arg_dict['compute_highflux_only']
    peak_thresh=arg_dict['peak_thresh']
    line_search_norm=arg_dict['line_search_norm']
    freeze_nH=arg_dict['freeze_nh']
    peak_clean=arg_dict['peak_clean']
    trig_interval=arg_dict['trig_interval']
    mandatory_abs=arg_dict['mandatory_abs']
    cont_fit_method=arg_dict['cont_fit_method']
    fit_SAA_norm=arg_dict['fit_SAA_norm']
    freeze_nH_val=arg_dict['freeze_nH_val']
    force_autofit=arg_dict['force_autofit']
    autofit_model=arg_dict['autofit_model']
    refit_cont=arg_dict['refit_cont']
    no_abslines=arg_dict['no_abslines']
    autofit=arg_dict['autofit']
    merge_cont=arg_dict['merge_cont']
    restrict_graded=arg_dict['restrict_graded']
    assess_line=arg_dict['assess_line']
    reload_fakes=arg_dict['reload_fakes']
    xchatter=arg_dict['xchatter']
    xspec_query=arg_dict['xspec_query']
    line_store_path=arg_dict['line_store_path']
    assess_line_upper=arg_dict['assess_line_upper']
    fix_compt_gamma=arg_dict['fix_compt_gamma']
    diff_bands_NuSTAR_NICER=arg_dict['diff_bands_NuSTAR_NICER']
    low_E_NICER=arg_dict['low_E_NICER']
    suzaku_xis_ignore=arg_dict['suzaku_xis_ignore']
    suzaku_xis_range=arg_dict['suzaku_xis_range']
    suzaku_pin_range=arg_dict['suzaku_pin_range']
    line_cont_ig_arg=arg_dict['line_cont_ig_arg']

    def line_e_ranges(sat, det=None):
        '''
        Determines the energy range allowed, as well as the ignore energies for a given satellite

        DO NOT USE INTS else it will be taken as channels instead of energies
        ignore_bands are bands that will be ignored on top of the rest, in ALL bands
        '''
        ignore_bands = None

        if sat == 'NuSTAR':
            e_sat_low = 8. if (sat_glob == 'multi' and diff_bands_NuSTAR_NICER) else 4.
            e_sat_high = 79.

        if sat.upper()=='SWIFT':
            if det is not None and det.upper()=='BAT':
                e_sat_low=14.
                e_sat_high=195.
            else:
                e_sat_low=0.3
                e_sat_high=10.

        if sat.upper() in ['XMM', 'NICER']:

            if sat == 'NICER':
                e_sat_low = 0.3 if (sat_glob == 'multi' and diff_bands_NuSTAR_NICER) else low_E_NICER
            else:
                e_sat_low = 0.3

            if sat.upper() in ['XMM']:
                if sat == 'XMM':
                    e_sat_low = 2.

                e_sat_high = 10.
            else:
                if sat == 'NICER':
                    e_sat_high = 10.
                else:
                    e_sat_high = 10.

        elif sat == 'Suzaku':

            if det == None:
                e_sat_low = 1.9
                e_sat_high = 40.

                ignore_bands = suzaku_xis_ignore
            else:

                assert det in ['PIN', 'XIS'], 'Detector argument necessary to choose energy ranges for Suzaku'

                e_sat_low = suzaku_xis_range[0] if det == 'XIS' else suzaku_pin_range[0]
                e_sat_high = suzaku_xis_range[1] if det == 'XIS' else suzaku_pin_range[1]

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

        line_cont_ig = ''

        if line_cont_ig_arg == 'iron':

            if sat in ['XMM', 'Chandra', 'NICER', 'Swift', 'SWIFT', 'Suzaku', 'NuSTAR']:


                if e_sat_low > 6:
                    # not ignoring this band for high-E only e.g. high-E only NuSTAR, BAT, INTEGRAL spectra
                    pass
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

        return e_sat_low, e_sat_high, ignore_bands, line_cont_ig


    def file_to_obs(file, sat):
        if sat == 'Suzaku':
            if megumi_files:
                return file.split('_src')[0].split('_gti')[0]
        elif sat in ['XMM', 'NuSTAR']:
            return file.split('_sp')[0]
        elif sat in ['Chandra', 'Swift', 'SWIFT']:
            return file.split('_grp_opt')[0]
        elif sat in ['NICER']:
            return file.split('_sp_grp_opt')[0]

    def html_table_maker():

        def strmaker(value_arr, is_overlap=False):

            '''
            wrapper for making a string of the line abs values

            set is_shift to true for energy/blueshift values, for which 0 values or low uncertainties equal to the value
            are sign of being pegged to the blueshift limit
            '''

            # the first case is for eqwidths and blueshifts (argument is an array with the uncertainties)
            if type(value_arr) == np.ndarray:
                # If the value array is entirely empty, it means the line is not detected and thus we put a different string
                if len(np.nonzero(value_arr)[0]) == 0:
                    newstr = '/'
                else:
                    # maybe the is_shift test needs to be put back
                    if type(value_arr[1]) == str:
                        # we do not show uncertainties for the linked parameters since it is just a repeat
                        newstr = str(round(value_arr[0], 2))

                    else:
                        # to get a more logical display in cases where the error bounds are out of the interval, we test the
                        # sign of the errors to write the error range differently
                        str_minus = ' -' if str(round(value_arr[1], 1))[0] != '-' else ' +'
                        str_plus = ' +' if str(round(value_arr[1], 1))[0] != '-' else ' '
                        newstr = str(round(value_arr[0], 1)) + str_minus + str(abs(round(value_arr[1], 1))) + \
                                 str_plus + str(round(value_arr[2], 1))

            # the second case is for the significance, delchis and the eqw upper limit, which are floats
            else:
                # same empty test except for overlap values which can go to zero
                if value_arr == 0:
                    if not is_overlap:
                        newstr = '/'
                    else:
                        return '0'
                else:
                    # the significance is always lower than 1
                    if value_arr <= 1 and not is_overlap:
                        newstr = (str(round(100 * value_arr, len(str(nfakes)))) if value_arr != 1 else '>' + str(
                            (1 - 1 / nfakes) * 100)) + '%'
                    # and the delchis should always be higher than 1 else the component would have been deleted
                    else:
                        newstr = str(round(value_arr, 2))

            return newstr

        # emission lines to be added
        '''
        </tr>
          <tr>
            <td>emission</td>
            <td>Fe Ka Neutral</td>
            <td>6.40</td>
            <td>no</td>
            <td>/</td>
            <td>''' + ''''strmaker(abslines_ener[0])''' + '''</td>
            <td>''' + '''strmaker(abslines_bshift[0])''' + '''</td>
            <td>/</td>
          </tr>
          <tr>
            <td>emission</td>
            <td>Fe Kb Neutral</td>
            <td>7.06</td>
            <td>no</td>
            <td>/</td>
            <td>''' + '''strmaker(abslines_ener[1])''' + '''</td>
            <td>''' + '''strmaker(abslines_bshift[1])''' + '''</td>
            <td>/</td>
          </tr>
        '''

        html_summary = \
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
                <th width="9%">EW ''' + str(sign_threshold) + ''' UL</th>
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

        for i_line, line_name in enumerate(
                [elem for elem in lines_e_dict.keys() if 'em' not in elem and 'Si' not in elem\
                 and elem in [comp.compname.split('_')[0] for comp in fitlines.complist if comp is not None]]):
            html_summary += '''
          <tr>
            <td>''' + line_name + '''</td>
            <td>''' + str(lines_e_dict[line_name][0]) + '''</td>
            <td>''' + ('/' if strmaker(abslines_sign[i_line]) == '/' else \
                     strmaker(abslines_em_overlap[i_line],is_overlap=True)) + '''</td>
            <td>''' + strmaker(abslines_flux[i_line] * 1e12) + '''</td>
            <td>''' + strmaker(abslines_bshift[i_line]) + '''</td>
            <td>''' + str('/' if abslines_bshift_distinct[i_line] is None else abslines_bshift_distinct[i_line]) + '''</td>
            <td>''' + strmaker(abslines_width[i_line] * 1e3) + '''</td>
            <td>''' + strmaker(abslines_eqw[i_line]) + '''</td>
            <td>''' + strmaker(abslines_eqw_upper[i_line]) + '''</td>
            <td>''' + strmaker(abslines_sign[i_line]) + '''</td>
            <td>''' + strmaker(abslines_delchi[i_line]) + '''</td>
          </tr>
          '''

        html_summary += '''
        </tbody>
        </table>
        '''
        return html_summary

    epoch_files = epoch_list[epoch_id]

    # this variable will store the final message for each spectra
    epoch_result = np.array([None] * len(epoch_files))

    def fill_result(string, result_array=epoch_result):

        '''
        small wrapper to fill the non defined epoch results with a string
        '''

        # defining a copy of the result array to fill it
        result_arr = np.array(result_array)
        for i in range(len(result_arr)):
            if result_arr[i] is None:
                result_arr[i] = string

        return result_arr

    Pset(window=None,xlog=False,ylog=False)

    # reducing the amount of data displayed in the terminal (doesn't affect the log file)
    Xset.chatter = xchatter

    # defining the standard number of fit iterations
    Fit.nIterations = 100

    Fit.query = xspec_query

    # deprecated
    obs_grating = False

    # Switching fit to C-stat
    Fit.statMethod = fitstat

    # this variable will serve for custom energy changes between different datagroups
    epoch_dets = []

    # skipping observation if asked
    if sat_glob == 'Chandra' and skip_nongrating and not obs_grating:
        return None

    # useful for later
    spec_inf = [elem_sp.split('_') for elem_sp in epoch_files]

    if sat_glob == 'multi':
        sat_indiv_init = np.repeat([None], len(epoch_files))

        for id_epoch, elem_file in enumerate(epoch_files):
            # fetching the instrument of the individual element
            # note that we replace the megumi xis0_xis3 files by the xis1 because the merged xis0_xis3 have no header
            # we also replace SUZAKU in caps by Suzaku to have a better time matching strings
            sat_indiv_init[id_epoch] = fits.open(elem_file.replace('xis0_xis3', 'xis1'))[1].header['TELESCOP'] \
                .replace('SUZAKU', 'Suzaku')
    else:
        sat_indiv_init = np.repeat([sat_glob], len(epoch_files))

    # Step 0 is to readjust the response and bg file names if necessary (i.e. files got renamed)
    if h_update:
        for i_sp, (elem_sp, elem_sat) in enumerate(zip(epoch_files, sat_indiv_init)):

            file_obsid = elem_sp.split('_')[0]

            epoch_dets += [None]

            with fits.open(elem_sp, mode='update') as hdul:
                if elem_sat == 'XMM':
                    hdul[1].header['BACKFILE'] = elem_sp.split('_sp')[0] + '_sp_bg.ds'
                    hdul[1].header['RESPFILE'] = elem_sp.split('_sp')[0] + '.rmf'
                    hdul[1].header['ANCRFILE'] = elem_sp.split('_sp')[0] + '.arf'
                    # saving changes
                    hdul.flush()

                elif elem_sat == 'NICER':
                    hdul[1].header['RESPFILE'] = elem_sp.split('_sp')[0] + '.rmf'
                    hdul[1].header['ANCRFILE'] = elem_sp.split('_sp')[0] + '.arf'
                    # saving changes
                    hdul.flush()

                elif elem_sat == 'Suzaku':
                    if megumi_files:
                        # update for pin files
                        if 'PIN' in hdul[1].header['DETNAM']:
                            epoch_dets[-1] = 'PIN'
                            # for megumi files we always update the header
                            pin_rspfile = glob.glob(file_obsid + '_ae_hxd_**.rsp')[0]
                            hdul[1].header['RESPFILE'] = pin_rspfile
                            hdul[1].header['BACKFILE'] = elem_sp.replace('src_dtcor_grp_opt', 'nxb_cxb')

                        elif 'XIS' in hdul[1].header['INSTRUME'] or '_xis' in elem_sp:
                            epoch_dets[-1] ='XIS'
                            hdul[1].header['RESPFILE'] = elem_sp.replace('src_grp_opt.pha', 'rsp.rmf')
                            hdul[1].header['BACKFILE'] = elem_sp.replace('src_grp_opt', 'bgd')

                elif elem_sat== 'SWIFT':
                    epoch_dets[-1]=hdul[1].header['INSTRUME']



                # saving changes
                hdul.flush()

    '''Energy bands, ignores, and setup'''

    # used to have specific energy limits for different instruments. can be modified later

    e_sat_low_indiv_init = np.repeat([None], len(epoch_files))
    e_sat_high_indiv_init = np.repeat([None], len(epoch_files))
    ignore_bands_indiv_init = np.repeat([None], len(epoch_files))
    line_cont_ig_indiv_init = np.repeat([None], len(epoch_files))

    if sat_glob != 'multi':
        sat_indiv_init = np.repeat([sat_glob], len(epoch_files))

    for id_epoch, elem_file in enumerate(epoch_files):
        e_sat_low_indiv_init[id_epoch], e_sat_high_indiv_init[id_epoch], ignore_bands_indiv_init[id_epoch], \
            line_cont_ig_indiv_init[id_epoch] = line_e_ranges(sat_indiv_init[id_epoch], epoch_dets[id_epoch])

    if sat_glob == 'multi':
        epoch_observ = [file_to_obs(elem_file, elem_telescope) for elem_file, elem_telescope in \
                        zip(epoch_files, sat_indiv_init)]
    else:
        epoch_observ = [file_to_obs(elem_file, sat_glob) for elem_file in epoch_files]

    Xset.logChatter = 10

    print('\nStarting line detection for files ')
    print(epoch_files)

    # reset the xspec config
    reset()

    '''Setting up a log file and testing the properties of each spectra'''

    curr_logfile_write = Xset.openLog(outdir + '/' + epoch_observ[0] + '_xspec_log' +
                                      ('_pdf' if pdf_only else '_ul' if line_ul_only \
                                          else '_reload' if reload_autofit and os.path.isfile(
                                          outdir + '/' + epoch_observ[0] + '_fitmod_autofit.pkl') \
                                          else '') + '.log')
    # ensuring the log information gets in the correct place in the log file by forcing line to line buffering
    curr_logfile_write.reconfigure(line_buffering=True)

    curr_logfile = open(curr_logfile_write.name, 'r')

    # curr_logfile_linedet=open(outdir+'/'+epoch_observ[0]+'_linedet_log.log','w+')

    def print_xlog(string, logfile_write=curr_logfile_write):

        '''
        prints and logs info in the xspec log file, and flushed to ensure the logs are printed before the next xspec print
        Different log file from the main xspec one to (hopefully) avoid issues
        '''
        print(string)
        logfile_write.write(time.asctime() + '\n')
        logfile_write.write(string)
        # adding a line for lisibility
        logfile_write.write('\n')
        logfile_write.flush()

    # list containing the epoch files rejected by the test
    epoch_files_good = []
    sat_indiv_good = []
    e_sat_low_indiv = []
    e_sat_high_indiv = []
    ignore_bands_indiv = []
    line_cont_ig_indiv = []
    epoch_dets_good = []
    '''
    Pile-up and bg tests for XMM spectra
    '''
    for i_sp, (elem_sp, elem_sat) in enumerate(zip(epoch_files, sat_indiv_init)):

        if elem_sat == 'XMM':

            AllData.clear()

            bg_off_flag = False
            try:
                curr_spec = Spectrum(elem_sp)
            except:
                try:
                    curr_spec = Spectrum(elem_sp, backfile=None)
                    print_xlog('\nLoaded the spectrum ' + elem_sp + ' with no background')
                    bg_off_flag = True
                except:
                    print_xlog(
                        "\nCouldn't load the spectrum " + elem_sp + "  with Xspec. Negative exposure time can cause this. Skipping the spectrum...")
                    epoch_result[i_sp] = "Couldn't load the spectrum with Xspec."
                    continue

            # this is useful for the check plots
            Plot.background = True
            Plot.add = True

            # saving a screen of the spectrum for verification purposes
            AllData.ignore('**-' + str(e_sat_low_indiv_init[i_sp]) + ' ' + str(e_sat_high_indiv_init[i_sp]) + '-**')

            # checking if the spectrum is empty after ignoring outside of the broad interval
            if curr_spec.rate[0] == 0:
                print_xlog("\nSpectrum " + elem_sp + " empty in the [" + str(e_sat_low_indiv_init[i_sp]) + "-" + str(
                    e_sat_high_indiv_init[i_sp]) + "] keV range. Skipping the spectrum...")
                epoch_result[i_sp] = "Spectrum empty in the [" + str(e_sat_low_indiv_init[i_sp]) + "-" + str(
                    e_sat_high_indiv_init[i_sp]) + "]keV range."
                continue
            else:
                Plot_screen("ldata", outdir + '/' + epoch_observ[i_sp] + "_screen_xspec_spectrum")
            AllData.notice('all')

            '''Various Checks'''

            # pile-up test
            # we use the ungrouped spectra for the headers since part of the header is often changed during the regrouping
            try:
                pileup_lines = fits.open(epoch_observ[i_sp] + '_sp_src.ds')[0].header['PILE-UP'].split(',')
                pileup_value = pileup_val(pileup_lines[-1])
                if pileup_value > pileup_lim:
                    print_xlog('\nPile-up value for spectrum ' + elem_sp + ' above the given limit of ' + str(
                        round(100 * pileup_lim, 1)) +
                               '%. Skipping the spectrum...')
                    epoch_result[i_sp] = 'Pile-up above the given limit (' + str(round(100 * pileup_lim, 1)) + '%).'
                    continue
                else:
                    print_xlog('\nPile-up value for spectrum ' + elem_sp + ' OK.')

            except:
                print_xlog('\nNo pile-up information available for spectrum ' + elem_sp)
                pileup_value = -1
                if pileup_missing == False:
                    print_xlog('\nSkipping the spectrum...')
                    epoch_result[i_sp] = 'No pile-up info available'
                    continue

            '''Unloading Imaging backgrounds if they are too bright compared to blank field standards'''

            with fits.open(elem_sp) as hdul:
                curr_expmode = hdul[0].header['DATAMODE']
                curr_cam = hdul[1].header['INSTRUME'][1:].swapcase()

            if curr_expmode == 'IMAGING' and bg_off_flag == False:

                AllData.ignore(
                    '**-' + str(e_sat_low_indiv_init[i_sp]) + ' ' + str(e_sat_high_indiv_init[i_sp]) + '.-**')
                # for that we must fetch the size of the background
                with open(epoch_observ[i_sp] + '_reg.reg') as regfile:
                    bg_rad = float(regfile.readlines()[-1].split('")')[0].split(',')[-1])

                # these values were computed by summing (and renormalizing) the graph rates in https://www.cosmos.esa.int/web/xmm-newton/bs-countrate
                # for standard patterns and thin filter opening
                if curr_cam == 'pn':
                    bg_blank = 1.02e-7
                elif curr_cam in ['mos1', 'mos2']:
                    bg_blank = 1.08e-8

                # now we can compare to the standard rates
                if curr_spec.rate[1] / (bg_rad) ** 2 > max_bg_imaging * bg_blank:
                    print_xlog(
                        '\nIntense background detected, probably contaminated by the source. Unloading it to avoid affecting the source...')
                    bg_off_flag = True

            '''unloading timing backgrounds'''

            if (curr_expmode in ['TIMING', 'BURST'] and skipbg_timing):
                bg_off_flag = True

            '''
            creating a duplicate spectrum file without the background so as to load the spectra natively without background 
            (the goal here is to avoid the crashes that sometimes happen when loading with a background)
            '''

            if os.path.isfile(elem_sp.replace('.ds', '_bgtested.ds')):
                os.remove(elem_sp.replace('.ds', '_bgtested.ds'))
            with fits.open(elem_sp) as hdul:
                if bg_off_flag == True:
                    hdul[1].header['BACKFILE'] = ''
                hdul.writeto(elem_sp.replace('.ds', '_bgtested.ds'))

            # SNR limit (only tested when the bg is kept)
            if bg_off_flag == False:
                with open(epoch_observ[i_sp] + '_regex_results.txt', 'r') as regex_file:
                    regex_lines = regex_file.readlines()
                    curr_SNR = float(regex_lines[3].split('\t')[1])
                if curr_SNR < SNR_min:
                    print_xlog(
                        '\nSpectrum  ' + elem_sp + ' Signal to Noise Ratio below the limit. Skipping the spectrum ...')
                    epoch_result[i_sp] = 'Spectrum SNR below the limit (' + str(SNR_min) + ')'
                    continue

            # saving the spectra if it passed all the test
            epoch_files_good += [elem_sp]
            sat_indiv_good += [elem_sat]
            e_sat_low_indiv += [e_sat_low_indiv_init[i_sp]]
            e_sat_high_indiv += [e_sat_high_indiv_init[i_sp]]
            ignore_bands_indiv += [ignore_bands_indiv_init[i_sp]]
            line_cont_ig_indiv += [line_cont_ig_indiv_init[i_sp]]
            epoch_dets_good += [epoch_dets[i_sp]]
        else:
            epoch_files_good += [elem_sp]
            sat_indiv_good += [elem_sat]
            e_sat_low_indiv += [e_sat_low_indiv_init[i_sp]]
            e_sat_high_indiv += [e_sat_high_indiv_init[i_sp]]
            ignore_bands_indiv += [ignore_bands_indiv_init[i_sp]]
            line_cont_ig_indiv += [line_cont_ig_indiv_init[i_sp]]
            epoch_dets_good += [epoch_dets[i_sp]]

        # testing if all spectra have been taken off
        if len(epoch_files_good) == 0:
            return epoch_result

    epoch_files_good = np.array(epoch_files_good)
    sat_indiv_good = np.array(sat_indiv_good)
    e_sat_low_indiv = np.array(e_sat_low_indiv)
    e_sat_high_indiv = np.array(e_sat_high_indiv)
    ignore_bands_indiv = np.array(ignore_bands_indiv, dtype=object)
    line_cont_ig_indiv = np.array(line_cont_ig_indiv)
    epoch_dets_good = np.array(epoch_dets_good)

    '''
    Data load
    '''

    # creating the file load str in the xspec format
    data_load_str = ''

    for i_sp, (elem_sp, elem_sat) in enumerate(zip(epoch_files_good, sat_indiv_good)):

        index_str = str(i_sp + 1) + ':' + str(i_sp + 1)
        data_load_str += index_str

        if elem_sat == 'XMM':
            data_load_str += ' ' + elem_sp.replace('.ds', '_bgtested.ds') + ' '
        if elem_sat == 'Chandra':
            if 'heg_1' in elem_sp and restrict_order:
                data_load_str = data_load_str[:-len(index_str)]
                continue
            data_load_str += ' ' + elem_sp + ' '
        if elem_sat in ['NICER', 'Suzaku', 'Swift', 'SWIFT', 'NuSTAR']:
            data_load_str += ' ' + elem_sp + ' '

    AllData(data_load_str)

    # updating individual spectra bg, rmf arf if needed
    scorpeon_list_indiv = np.repeat(None, len(epoch_files_good))
    for i_sp, (elem_sp, elem_sat,elem_det) in enumerate(zip(epoch_files_good, sat_indiv_good,epoch_dets_good)):

        if elem_sat == 'Chandra':
            if 'heg_1' in elem_sp and restrict_order:
                continue
            AllData(i_sp + 1).response.arf = elem_sp.replace('_grp_opt.pha', '.arf')

        if elem_sat == 'NICER':
            if NICER_bkg == 'scorpeon_mod':
                # loading the background and storing the bg python path in xscorpeon
                # note that here we assume that all files have a valid scorpeon background
                scorpeon_list_indiv[i_sp] = elem_sp.replace('_sp_grp_opt.pha', '_bg.py')
        if elem_sat in ['Swift', 'SWIFT'] and elem_det!='BAT':
            AllData(i_sp + 1).response.arf = epoch_observ[0].replace('source', '') + '.arf'
            AllData(i_sp + 1).background = epoch_observ[0].replace('source', '') + ('back.pi')

    if 'NICER' in sat_indiv_good:
        xscorpeon.load(scorpeon_list_indiv, frozen=True)

    '''resetting the energy bands if needed'''
    if not force_ener_bounds:
        hid_cont_range[1] = max(e_sat_high_indiv)

        # here we just test for the snr and create the line searc hspace, this will re-adjusted esat_high later
        line_cont_range[1] = min(np.array(line_cont_range_arg.split(' ')).astype(float)[1], max(e_sat_high_indiv))

    # note: we add half a step to get rid of rounding problems and have the correct steps
    line_search_e_space = np.arange(line_search_e[0], line_search_e[1] + line_search_e[2] / 2, line_search_e[2])
    # this one is here to avoid adding one point if incorrect roundings create problem
    line_search_e_space = line_search_e_space[line_search_e_space <= line_search_e[1]]

    # outdated because now included in the line_e_ranges
    # for i_obs,elem_sat in enumerate(sat_indiv_good):
    #     #ad hoc way to only consider the suzaku detectors
    #     i_det_suzaku=0
    #     if elem_sat == 'Suzaku':
    #         e_sat_high_indiv[i_obs]=40. if epoch_dets[i_det_suzaku]=='PIN' else 9.
    #         e_sat_low_indiv[i_obs]=12. if epoch_dets[i_det_suzaku]=='PIN' else 1.9
    #         #test whether this works
    #         i_det_suzaku+=1

    # in preparation for the raw counts test
    if sat_glob == 'Suzaku':
        line_cont_range[1] = 9.

    if max(e_sat_high_indiv) > 12:
        Plot.xLog = True
    else:
        Plot.xLog = False

    if 'NuSTAR' in sat_indiv_good or 'SWIFT' in sat_indiv_good:
        Plot.background = True

    # Putting the right energy bands and screening the spectra (note that we don't ignore the ignore bands here on purpose)
    ignore_data_indiv(e_sat_low_indiv, e_sat_high_indiv)

    Plot_screen("ldata", outdir + '/' + epoch_observ[0] + "_screen_xspec_spectrum")

    '''
    Testing the amount of raw source counts in the line detection range for all datagroups combined
    '''

    #### Testing if the data is above the count limit

    # for the line detection
    AllData.ignore('**-' + str(line_cont_range[0]) + ' ' + str(line_cont_range[1]) + '-**')
    glob_counts = 0
    indiv_counts = []

    bg_counts = 0

    for id_grp, elem_sat in enumerate(sat_indiv_good):

        i_grp = id_grp + 1
        # for NICER we subtract the rate from the background which at this point is the entire model ([3])
        if elem_sat == 'NICER':
            indiv_counts += [round((AllData(i_grp).rate[0] - AllData(i_grp).rate[3]) * AllData(i_grp).exposure)]

            bg_counts += AllData(i_grp).rate[3] * AllData(i_grp).exposure
        else:
            indiv_counts += [round(AllData(i_grp).rate[0] * AllData(i_grp).exposure)]

            bg_counts += (AllData(i_grp).rate[2] - AllData(i_grp).rate[0]) * AllData(i_grp).exposure

        glob_counts += indiv_counts[-1]

    '''
    SNR formula from https://xmm-tools.cosmos.esa.int/external/sas/current/doc/specgroup.pdf 4.3.2 (1)
    #note: here indiv_count is a net count, so a full source counts is indiv_counts+ bg and
    #source + bg is thus net + 2*bg
    '''

    SNR = (sum(indiv_counts)) / np.sqrt(sum(indiv_counts) + 2 * bg_counts)

    if glob_counts < counts_min:
        flag_lowSNR_line = True
        if not fit_lowSNR:
            print_xlog('\nInsufficient net counts (' + str(round(glob_counts)) + ' < ' + str(round(counts_min)) +
                       ') in line detection range.')
            return fill_result('Insufficient net counts (' + str(round(glob_counts)) + ' < ' + str(round(counts_min)) + \
                               ') in line detection range.')

    elif SNR < SNR_min:
        if not fit_lowSNR:
            print_xlog('\nInsufficient SNR (' + str(round(SNR, 1)) + '<50) in line detection range.')
            return fill_result('Insufficient SNR (' + str(round(SNR, 1)) + '<50) in line detection range.')
    else:
        flag_lowSNR_line = False

    if not force_ener_bounds:
        # re-adjusting the line cont range to esathigh to allow fitting up to higher energies if needed
        line_cont_range[1] = min(np.array(line_cont_range_arg.split(' ')).astype(float)[1], max(e_sat_high_indiv))

    # limiting to the HID energy range
    ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                      sat_high_groups=e_sat_high_indiv, glob_ignore_bands=ignore_bands_indiv)

    glob_counts = 0
    indiv_counts = []

    for i_grp in range(1, AllData.nGroups + 1):
        # could add background here
        indiv_counts += [round(AllData(i_grp).rate[2] * AllData(i_grp).exposure)]
        glob_counts += indiv_counts[-1]
    if glob_counts < counts_min_HID:
        print_xlog('\nInsufficient counts (' + str(round(glob_counts)) + ' < ' + str(round(counts_min_HID)) +
                   ') in HID detection range.')
        return fill_result('Insufficient counts (' + str(round(glob_counts)) + ' < ' + str(round(counts_min_HID)) + \
                           ') in HID detection range.')

    # testing the NuSTAR bands if asked to
    if filter_NuSTAR_SNR != 0.:
        for i_exp, elem_sat in enumerate(sat_indiv_init):
            SNR_low = False
            if elem_sat == 'NuSTAR':
                # starting at 10keV (stops at 78)
                for val_keV in np.arange(10., 79.):
                    AllData(i_exp + 1).notice('all')
                    AllData(i_exp + 1).ignore('**-' + str(val_keV) + ' ' + str(val_keV + 1) + '-**')
                    # SNR formula from https://xmm-tools.cosmos.esa.int/external/sas/current/doc/specgroup.pdf 4.3.2 (1)

                    SNR_val = np.sqrt(AllData(i_exp + 1).exposure) * AllData(i_exp + 1).rate[0] \
                              / np.sqrt(
                        AllData(i_exp + 1).rate[0] + 2 * (AllData(i_exp + 1).rate[2] - AllData(i_exp + 1).rate[0]))

                    # stopping the computation if the SNR goes below the limit
                    if SNR_val < filter_NuSTAR_SNR:
                        SNR_low = True
                        break

                # resetting the energy band of the exposure
                AllData(i_exp + 1).notice('all')
                AllData(i_exp + 1).ignore(
                    '**-' + str(e_sat_low_indiv[i_exp]) + ' ' + str(e_sat_high_indiv[i_exp]) + '-**')

                # storing the final value
                e_sat_high_indiv[i_exp] = 79. if not SNR_low else val_keV

                print_xlog(
                    'High energy limit of exposure ' + str(i_exp + 1) + ' fixed to ' + str(e_sat_high_indiv[i_exp]) +
                    ' keV with a SNR limit of ' + str(filter_NuSTAR_SNR))

    '''re-resetting the energy bands if needed to adapt to the NuSTAR SNR cuts'''
    if not force_ener_bounds:
        hid_cont_range[1] = max(e_sat_high_indiv)

        # here we just test for the snr and create the line searc hspace, this will re-adjusted esat_high later
        line_cont_range[1] = min(np.array(line_cont_range_arg.split(' ')).astype(float)[1], max(e_sat_high_indiv))

    if pdf_only:

        if catch_errors:

            try:

                pdf_summary(epoch_files,arg_dict=arg_dict, fit_ok=True, summary_epoch=fill_result('Line detection complete.'),
                            e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

                # closing the logfile for both access and Xspec
                curr_logfile.close()
                Xset.closeLog()

                return fill_result('Line detection complete.')
            except:
                return fill_result('Missing elements to compute PDF.')

        else:

            pdf_summary(epoch_files,arg_dict=arg_dict, fit_ok=True, summary_epoch=fill_result('Line detection complete.'),
                        e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

            # closing the logfile for both access and Xspec
            curr_logfile.close()
            Xset.closeLog()

            return fill_result('Line detection complete.')

    if line_ul_only:

        # loading the autofit model
        # note that this also restores the data ignore states
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm')

        # reloading the fitlines class
        fitlines = load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_autofit.pkl')

        # updating fitcomps
        fitlines.update_fitcomps()

        # recreating the no abs version

        # deleting all absorption components (reversed so we don't have to update the fitcomps)
        for comp in [elem for elem in fitlines.includedlist if elem is not None][::-1]:
            if comp.named_absline:
                # note that with no rollback we do not update the values of the component so it has still its included status and everything else
                comp.delfrommod(rollback=False)

        # storing the no abs line 'continuum' model
        data_autofit_noabs = allmodel_data()

        # reloading previously computed information
        dict_linevis = {'visual_line': False,
                        'cameras': 'all' if sat_glob == 'multi' else cameras,
                        'expmodes': expmodes}

        # the goal here is to avoid importing streamlit if possible
        from visual_line_tools import abslines_values

        precomp_absline_vals, precomp_autofit_vals = abslines_values(autofit_store_path, dict_linevis,
                                                                     obsid=epoch_observ[0])

        # selecting object 0 and obs 0 aka this obs
        precomp_absline_vals = precomp_absline_vals[0][0]
        precomp_autofit_vals = precomp_autofit_vals[0][0]

        abslines_eqw, abslines_bshift, abslines_delchi, abslines_flux, abslines_sign = precomp_absline_vals[:5]

        abslines_eqw_upper = np.zeros(len(range_absline))

        abslines_em_overlap, abslines_width, abslines_bshift_distinct = precomp_absline_vals[6:]

        autofit_parerrors, autofit_parnames = precomp_autofit_vals

        sign_widths_arr = np.array([elem[0] if elem[0] - elem[1] > 1e-6 else 0 for elem in abslines_width])

        # fetching the ID of this observation
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
            pdf_summary(epoch_files,arg_dict=arg_dict, fit_ok=True, summary_epoch=fill_result('Line detection complete.'),
                        e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

        # closing the logfile for both access and Xspec
        curr_logfile.close()
        Xset.closeLog()

        return fill_result('Line detection complete.')

    # creating the continuum model list
    comp_cont = model_list(cont_model)

    # taking off the constant factor if there's only one data group
    if AllData.nGroups == 1:
        comp_cont = [elem for elem in comp_cont if elem != 'glob_constant']

    # testing the presence of arf, rmf etc in the datasets
    isbg_grp = []
    isrmf_grp = []
    isarf_grp = []
    for i_grp in range(1, AllData.nGroups + 1):
        try:
            AllData(i_grp).background
            isbg_grp += [True]
        except:
            isbg_grp += [False]

        try:
            AllData(i_grp).response
            isrmf_grp += [True]
        except:
            isrmf_grp += [False]

        try:
            AllData(i_grp).response.arf
            isarf_grp += [True]
        except:
            isarf_grp += [False]

    abslines_table_str = None

    def curr_store_fit(mode='broadband', fitmod=None):
        store_fit(mode=mode, epoch_id=epoch_observ[0], outdir=outdir, logfile=curr_logfile, fitmod=fitmod)

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

        # dont know why this causes an issue
        from xspec import AllChains

        AllChains.defLength = 10000
        AllChains.defBurn = 5000
        AllChains.defWalkers = 10

        # deleting the previous chain to avoid conflicts
        AllChains.clear()

        if os.path.exists(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits'):
            os.remove(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits')

        # we don't need to make a new chain if the broad fit has done it already
        if not broad_HID_mode:
            # Creating a chain to avoid problems when computing the errors
            Chain(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits')

        # trying to freeze pegged parameters again in case the very last fit created pegs
        calc_error(curr_logfile, param='1-' + str(AllModels(1).nParameters * AllData.nGroups), timeout=20,
                   freeze_pegged=True, indiv=True)

        Fit.perform()

        mod_hid = allmodel_data()

        fitmodel.update_fitcomps()

        # computing and storing the flux in several bands
        spflux_single = [None] * 5
        spflux_high = None

        # adding a cflux component after all calibration and absorption components
        # we need to parse the components in the model to know where to place it

        # first component from the first non calibration fitcomp
        first_stdcomp = [elem for elem in fitmodel.includedlist_main if not elem.absorption
                         and not elem.xcomps[0].name in xspec_globcomps and not elem.calibration][0].compnumbers[0]

        # last component from the last non calibration fitcomp
        # (could be vashift*gaussian here for example, hence the second [-1])
        last_stdcomp = [elem for elem in fitmodel.includedlist_main if not elem.absorption
                        and not elem.xcomps[0].name in xspec_globcomps and not elem.calibration][-1].compnumbers[-1]

        flux_bands = [[hid_cont_range[0], hid_cont_range[1]], [3, 6], [6., 10.], [1., 3.], [3., 10]]

        # adding the high energy band if it is relevant
        if max(e_sat_high_indiv) >= 20:
            flux_bands += [[15, 50]]

        # loop in each band:
        for i_band, elem_band in enumerate(flux_bands):

            mod_hid.load()

            # increasing the energy range if necessary (computing high-E flux) and no thcomp has been loaded
            if i_band > 4 and AllModels(1).energies(1)[-1] < 50:
                AllModels.setEnergies('0.1 100. 1000 log')

            # first addition of the cflux component to compute the main value
            allfreeze()

            # addding the component
            cflux_compnumber = addcomp('cflux', position=first_stdcomp, endmult=last_stdcomp, return_pos=True)[1][0]

            cflux_comp = getattr(AllModels(1), AllModels(1).componentNames[cflux_compnumber - 1])

            # setting the energies and a decent starting flux
            cflux_comp.Emin.values = elem_band[0]
            cflux_comp.Emax.values = elem_band[1]
            cflux_comp.lg10Flux.values = -10

            # fitting to get a first idea of the value
            calc_fit()
            cflux_stval = cflux_comp.lg10Flux.values[0]

            # resetting the model
            mod_hid.load()

            # freezing the first normalisation parameter found in the main components if there is one
            # to avoid issues when fitting cflux
            for i_comp in range(first_stdcomp, last_stdcomp + 1):
                stdcomp = getattr(AllModels(1), AllModels(1).componentNames[i_comp - 1])

                if 'norm' in stdcomp.parameterNames:
                    fitmodel.print_xlog('Freezing norm for component ' + stdcomp.name)
                    stdcomp.norm.frozen = True
                    break

            # reloading the component
            addcomp('cflux', position=first_stdcomp, endmult=last_stdcomp, return_pos=True)
            cflux_comp = getattr(AllModels(1), AllModels(1).componentNames[cflux_compnumber - 1])

            # setting the energies and a decent starting flux
            cflux_comp.Emin.values = elem_band[0]
            cflux_comp.Emax.values = elem_band[1]
            cflux_comp.lg10Flux.values = cflux_stval

            # fitting
            calc_fit()

            # computing the errors
            par_index = cflux_comp.lg10Flux.index
            err_vals = calc_error(fitmodel.logfile, param=str(par_index), give_errors=True, timeout=120)

            err_vals_decimal = 10 ** (np.array([cflux_comp.lg10Flux.values[0],
                                                cflux_comp.lg10Flux.values[0] - err_vals[0][par_index - 1][0],
                                                cflux_comp.lg10Flux.values[0] + err_vals[0][par_index - 1][1]]))

            if i_band > 4:
                spflux_high = np.array(err_vals_decimal)
            else:
                spflux_single[i_band] = [err_vals_decimal[0], err_vals_decimal[0] - err_vals_decimal[1],
                                         err_vals_decimal[2] - err_vals_decimal[1]]

        mod_hid.load()

        spflux_single = np.array(spflux_single).T

        for i_sp in range(len(epoch_files_good)):
            if line_cont_ig_indiv[i_sp] != '':
                try:
                    AllData(i_sp + 1).notice(line_cont_ig_indiv[i_sp])
                except:
                    breakpoint()
                    pass

        curr_store_fit(mode='broadhid' + add_str, fitmod=fitmodel)

        # storing the fitmod class into a file
        fitmodel.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid' + add_str + '.pkl')

        fitmodel.update_fitcomps()

        # if not broad_HID_mode:
        #     # Creating a chain to avoid problems when computing the errors
        #     Chain(outdir + '/' + epoch_observ[0] + '_chain_hid' + add_str + '.fits')
        #
        # # computing and storing the flux for the full luminosity and two bands for the HR
        # spflux_single = [None] * 5
        #
        # '''the first computation is ONLY to get the errors, the main values are overwritten below'''
        # # we still only compute the flux of the first model even with NICER because the rest is BG
        #
        # AllModels.calcFlux(str(hid_cont_range[0]) + ' ' + str(hid_cont_range[1]) + " err 1000 90")
        #
        # spflux_single[0] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
        #                                        AllData(1).flux[0]
        # AllModels.calcFlux("3. 6. err 1000 90")
        # spflux_single[1] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
        #                                        AllData(1).flux[0]
        # AllModels.calcFlux("6. 10. err 1000 90")
        # spflux_single[2] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
        #                                        AllData(1).flux[0]
        # AllModels.calcFlux("1. 3. err 1000 90")
        # spflux_single[3] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
        #                                        AllData(1).flux[0]
        # AllModels.calcFlux("3. 10. err 1000 90")
        # spflux_single[4] = AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
        #                                        AllData(1).flux[0]

        # #easier to save the fit here to reload it without having changed the energies
        # if os.path.isfile(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm'):
        #     os.remove(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm')
        # Xset.save(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm', info='a')
        #
        # if max(e_sat_high_indiv)>=20:
        #     #re-arranging the energies array to compute a high energy flux value
        #     AllModels.setEnergies('0.1 100. 1000 log')
        #     #need to remake a new fit after that
        #     calc_fit()
        #     AllModels.calcFlux("15. 50. err 1000 90")
        #     spflux_high = np.array([AllData(1).flux[0], AllData(1).flux[0] - AllData(1).flux[1], AllData(1).flux[2] - \
        #                                            AllData(1).flux[0]])
        # else:
        #     spflux_high=None
        #
        # Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '_withignore.xcm')
        #
        # spflux_single = np.array(spflux_single)
        #
        # AllChains.clear()
        #
        # mod_hid.load()
        #
        # for i_sp in range(len(epoch_files_good)):
        #     if line_cont_ig_indiv[i_sp] != '':
        #         try:
        #             AllData(i_sp+1).notice(line_cont_ig_indiv[i_sp])
        #         except:
        #             breakpoint()
        #             pass
        # curr_store_fit(mode='broadhid' + add_str, fitmod=fitmodel)
        #
        # # storing the fitmod class into a file
        # fitmodel.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid' + add_str + '.pkl')
        #
        # # taking off the absorption (if it is in the final included components) before computing the flux
        # abs_incl_comps = (np.array(fitmodel.complist)[[elem.absorption and elem.included for elem in \
        #                                                [elem_comp for elem_comp in fitmodel.complist if
        #                                                 elem_comp is not None]]])
        # if len(abs_incl_comps)!=0:
        #     main_abs_comp=abs_incl_comps[0]
        #     main_abs_comp.xcomps[0].nH.values=0
        # else:
        #     main_abs_comp=None
        #
        # #removing the calibration components and absorption lines, reverse included order to not have to update the fitcomps
        # for comp in [elem for elem in fitmodel.includedlist if elem is not None][::-1]:
        #     if comp.calibration or comp.named_absline:
        #         comp.delfrommod(rollback=False)
        #
        # #removing the absorption lines
        # # and replacing the main values with the unabsorbed flux values
        # # (conservative choice since the other uncertainties are necessarily higher)
        #
        # #this value is fully useless at this point
        # AllModels.calcFlux(str(hid_cont_range[0]) + ' ' + str(hid_cont_range[1]))
        #
        # spflux_single[0][0] = AllData(1).flux[0]
        # AllModels.calcFlux("3. 6.")
        # spflux_single[1][0] = AllData(1).flux[0]
        # AllModels.calcFlux("6. 10.")
        # spflux_single[2][0] = AllData(1).flux[0]
        # AllModels.calcFlux("1. 3.")
        # spflux_single[3][0] = AllData(1).flux[0]
        # AllModels.calcFlux("3. 10.")
        # spflux_single[4][0] = AllData(1).flux[0]
        #
        # #re-arranging the energies array to compute a high energy flux value
        # if max(e_sat_high_indiv)>=20:
        #     AllModels.setEnergies('0.1 100. 1000 log')
        #     AllModels.calcFlux("15. 50.")
        #     spflux_high[0]=AllData(1).flux[0]
        #
        # #and resetting the energies
        # Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadhid' + add_str + '.xcm')
        #
        #
        # spflux_single = spflux_single.T
        #
        # mod_hid.load()
        #
        # fitmodel.update_fitcomps()

        return spflux_single, spflux_high

    # reload previously stored autofits to gain time if asked to
    if reload_autofit and os.path.isfile(outdir + '/' + epoch_observ[0] + '_fitmod_autofit.pkl'):

        print_xlog('completed autofit detected...Reloading computation.')
        # reloading the broad band fit and model and re-storing associed variables
        fitlines_broad = load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_broadband_post_auto.pkl')
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadband_post_auto.xcm')
        fitlines_broad.update_fitcomps()
        data_broad = allmodel_data()
        
        # re fixing the absorption parameter and storing the value to retain it
        # if the absorption gets taken off and tested again
        abs_incl_comps = (np.array(fitlines_broad.complist)[[elem.absorption and elem.included for elem in \
                                                             [elem_comp for elem_comp in fitlines_broad.complist if
                                                              elem_comp is not None]]])
        if len(abs_incl_comps) != 0:
            main_abs_comp = abs_incl_comps[0]
            broad_absval = main_abs_comp.xcomps[0].nH.values[0]
        else:
            broad_absval = 0

        # reloading the hid band fit and model and re-storing associed variables
        fitlines_broad = load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_broadband_post_auto.pkl')
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadhid_post_auto.xcm')
        fitlines_broad.update_fitcomps()
        data_broad_post_auto = allmodel_data()
        
        if os.path.isfile(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid_post_auto.pkl'):
            fitlines_hid = load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid_post_auto.pkl')
            Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadhid_post_auto.xcm')
            fitlines_hid.update_fitcomps()
            data_broadhid = allmodel_data()

        else:
            fitlines_hid = fitlines_broad

            assert not broad_HID_mode, "This failsafe is for older computation which shouldn't run broad_HID_mode"

            # refitting in hid band for the HID values
            ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                              sat_high_groups=e_sat_high_indiv, glob_ignore_bands=ignore_bands_indiv)

            # fitting the model to the new energy band first
            calc_fit(logfile=fitlines_hid.logfile)

            # autofit
            fitlines_hid.global_fit(chain=True, lock_lines=True, directory=outdir, observ_id=epoch_observ[0],
                                    split_fit=split_fit)

            fitlines_hid.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadhid_post_auto.pkl')

        from xspec import AllChains

        AllChains.clear()

        # updating the logfile for the next type of computations
        curr_logfile_write = Xset.openLog(outdir + '/' + epoch_observ[0] + '_xspec_log_autofit_comput.log')
        curr_logfile_write.reconfigure(line_buffering=True)
        curr_logfile = open(curr_logfile_write.name, 'r')

        # updating it in the fitmod
        fitlines_hid.logfile = curr_logfile
        fitlines_hid.logfile_write = curr_logfile_write
        fitlines_hid.update_fitcomps()

        main_spflux, main_spflux_high = hid_fit_infos(fitlines_hid, broad_absval, post_autofit=True)

        if max(e_sat_high_indiv) >= 20:
            np.savetxt(outdir + '/' + epoch_observ[0] + '_main_spflux_high.txt', main_spflux_high)

        if compute_highflux_only:
            return fill_result('Line detection complete.')

        # restoring the linecont save
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_broadband_linecont.xcm')

        data_mod_high = allmodel_data()

        cont_abspeak, cont_peak_points, cont_peak_widths, cont_peak_delchis, cont_peak_eqws, chi_dict_init = \
            narrow_line_search(data_mod_high, 'cont', line_search_e=line_search_e, line_search_norm=line_search_norm,
                               e_sat_low_indiv=e_sat_low_indiv, peak_thresh=peak_thresh, peak_clean=peak_clean,
                               line_cont_range=line_cont_range, trig_interval=trig_interval,
                               scorpeon_save=data_mod_high.scorpeon)

        with open(outdir + '/' + epoch_observ[0] + '_chi_dict_init.pkl', 'wb') as file:
            dill.dump(chi_dict_init, file)

        # re-ignoring high-E only spectra for the autofit band
        mask_nodeload = e_sat_low_indiv<10

        # reloading the continuum models to get the saves back and compute the continuum infos
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm')

        # loading the autofit model
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm')

        # reloading the fitlines class
        fitlines = load_fitmod(outdir + '/' + epoch_observ[0] + '_fitmod_autofit.pkl')

        # updating fitcomps
        fitlines.update_fitcomps()

        # reloading the fit
        calc_fit()

        # reloading the chain
        AllChains += outdir + '/' + epoch_observ[0] + '_chain_autofit.fits'

        data_autofit = allmodel_data()

    else:

        '''Continuum fits'''

        def high_fit(broad_absval, broad_abscomp, broad_gamma_nthcomp,scorpeon_deload, thcomp_frac_frozen=False):

            '''
            high energy fit and flux array computation
            '''

            print_xlog('\nComputing line continuum fit...')

            AllModels.clear()
            xscorpeon.load('auto', scorpeon_save=scorpeon_deload, frozen=True)

            # limiting to the line search energy range
            ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                              sat_low_groups=e_sat_low_indiv[mask_nodeload],
                              sat_high_groups=e_sat_high_indiv[mask_nodeload],
                              glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

            # if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
            if not flag_lowSNR_line:
                for i_grp in range(len(epoch_files_good[mask_nodeload])):
                    # ignoring the line_cont_ig energy range for the fit to avoid contamination by lines
                    AllData(i_grp + 1).ignore(line_cont_ig_indiv[i_grp])

            # comparing different continuum possibilities with a broken powerlaw or a combination of diskbb and powerlaw

            # creating the automatic fit class for the standard continuum

            # Xset.closeLog()

            # Xset.openLog('lineplots_opt_nth/test.log')
            # curr_logfile = Xset.log
            # curr_logfile_write = Xset.log
            # curr_logfile_write.reconfigure(line_buffering=True)
            # curr_logfile = open(curr_logfile_write.name, 'r')

            # initial informations:
            # gamma to freeze the nthcomp to avoid nonsenses
            # and giving the info on whether the thcomp should be nullified if there is one

            if broad_absval != 0:
                fitcont_high = fitmod(comp_cont,
                                      curr_logfile, curr_logfile_write, absval=broad_absval,
                                      fixed_gamma=broad_gamma_nthcomp, thcomp_frac_frozen=thcomp_frac_frozen,
                                      mandatory_abs=mandatory_abs)
            else:

                # creating the fitcont without the absorption component if it didn't exist in the broad model
                fitcont_high = fitmod([elem for elem in comp_cont if elem != broad_abscomp],
                                      curr_logfile, curr_logfile_write,
                                      fixed_gamma=broad_gamma_nthcomp, thcomp_frac_frozen=thcomp_frac_frozen)

            # forcing the absorption component to be included for the broad band fit
            if sat_glob == 'NuSTAR' and freeze_nH:
                main_abscomp = (np.array(fitcont_high.complist)[[elem.absorption for elem in \
                                                                 fitcont_high.complist]])[0]
                main_abscomp.mandatory = True

            # check the thcomp_frac_frozen

            fitcont_high.global_fit(split_fit=split_fit, method=cont_fit_method)

            # check why thcomp isn't pegged when it should be

            # mod_fitcont=allmodel_data()

            chi2_cont = Fit.statistic
            # except:
            #     pass
            #     chi2_cont=0

            # AllModels.clear()
            # xscorpeon.load(scorpeon_save=data_broad.scorpeon,frozen=True)
            # not used currently
            # #with the broken powerlaw continuum
            # fitcont_high_bkn=fitmod(comp_cont_bkn,curr_logfile)

            # try:
            #     #fitting
            #     fitcont_high_bkn.global_fit(split_fit=split_fit))

            #     chi2_cont_bkn=Fit.statistic
            # except:
            #     pass
            chi2_cont_bkn = 0

            if chi2_cont == 0 and chi2_cont_bkn == 0:
                print_xlog('\nProblem during line continuum fit. Skipping line detection for this exposure...')
                return ['\nProblem during line continuum fit. Skipping line detection for this exposure...']
            # else:
            #     if chi2_cont<chi2_cont_bkn:
            #         model_load(mod_fitcont)

            # renoticing the line energy range
            # note: for now this is fine but might need to be udpated later with telescopes with global ignore bands
            # matching part of this

            for i_sp in range(len(epoch_files_good[mask_nodeload])):
                if line_cont_ig_indiv[i_sp] != '':
                    AllData(i_sp + 1).notice(line_cont_ig_indiv[i_sp])

            # saving the model data to reload it after the broad band fit if needed
            mod_high_dat = allmodel_data()

            # rescaling before the prints to avoid unecessary loggings in the screen
            rescale(auto=True)

            # screening the xspec plot and the model informations for future use
            Plot_screen("ldata,ratio,delchi", outdir + '/' + epoch_observ[0] + "_screen_xspec_broadband_linecont",
                        includedlist=fitcont_high.includedlist)

            # saving the model str
            catch_model_str(curr_logfile, savepath=outdir + '/' + epoch_observ[0] + '_mod_broadband_linecont.txt')

            # deleting the model file since Xset doesn't have a built-in overwrite argument and crashes when asking manual input
            if os.path.isfile(outdir + '/' + epoch_observ[0] + '_mod_broadband_linecont.xcm'):
                os.remove(outdir + '/' + epoch_observ[0] + '_mod_broadband_linecont.xcm')

            # storing the current configuration and model
            Xset.save(outdir + '/' + epoch_observ[0] + '_mod_broadband_linecont.xcm', info='a')

            # storing the class
            fitcont_high.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadband_linecont.pkl')

            return [mod_high_dat, fitcont_high]

        def broad_fit():

            '''Broad band fit to get the HR ratio and Luminosity'''

            # first broad band fit in e_sat_low-10 to see the spectral shape
            print_xlog('\nComputing broad band fit for visualisation purposes...')

            ignore_data_indiv(e_sat_low_indiv, e_sat_high_indiv, reset=True, glob_ignore_bands=ignore_bands_indiv)

            # if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
            if not flag_lowSNR_line:
                for i_sp in range(len(epoch_files_good)):
                    if line_cont_ig_indiv[i_sp] != '':
                        AllData(i_sp + 1).ignore(line_cont_ig_indiv[i_sp])

            # forcing an absorption value if asked to
            if sat_glob == 'NuSTAR' and freeze_nH:
                broad_absval = freeze_nH_val
            else:
                broad_absval = None

            # creating the automatic fit class for the standard continuum
            fitcont_broad = fitmod(comp_cont, curr_logfile, curr_logfile_write,
                                   absval=broad_absval, sat_list=sat_indiv_good,
                                   mandatory_abs=mandatory_abs)

            # forcing the absorption component to be included for the broad band fit
            if sat_glob == 'NuSTAR' and freeze_nH:
                main_abscomp = (np.array(fitcont_broad.complist)[[elem.absorption for elem in \
                                                                  fitcont_broad.complist]])[0]
                main_abscomp.mandatory = True

            # fitting
            fitcont_broad.global_fit(split_fit=split_fit, method=cont_fit_method, fit_scorpeon=True,
                                     fit_SAA_norm=fit_SAA_norm)

            # checking whether the comptonisation component is useless
            if 'disk_thcomp' in [elem.compname for elem in fitcont_broad.includedlist_main]:

                # checking if the component was pegged during the fit
                thcomp_frac_frozen = AllModels(1).thcomp.Gamma_tau.frozen
                if thcomp_frac_frozen:
                    # forcing the component to 0 values and freezing it
                    AllModels(1).thcomp.cov_frac.values = 0.
                    AllModels(1).thcomp.Gamma_tau.values = 3.5
                    AllModels(1).thcomp.Gamma_tau.frozen = True
                    calc_fit()

            else:
                thcomp_frac_frozen = False

            if fix_compt_gamma:
                if 'disk_nthcomp' in [comp.compname for comp in \
                                      [elem for elem in fitcont_broad.includedlist if elem is not None]]:
                    broad_gamma_compt = fitcont_broad.disk_nthcomp.xcomps[0].Gamma.values[0]
                elif 'disk_thcomp' in [comp.compname for comp in \
                                       [elem for elem in fitcont_broad.includedlist if elem is not None]]:
                    broad_gamma_compt = fitcont_broad.disk_thcomp.xcomps[0].Gamma_tau.values[0]
                else:
                    broad_gamma_compt = None
            else:
                broad_gamma_compt=None

            mod_fitcont = allmodel_data()

            chi2_cont = Fit.statistic

            chi2_cont_bkn = 0

            if chi2_cont == 0 and chi2_cont_bkn == 0:
                print_xlog('\nProblem during broad band fit. Skipping line detection for this exposure...')
                return ['\nProblem during broad band fit. Skipping line detection for this exposure...']
            # else:
            #     if chi2_cont<chi2_cont_bkn:
            #         model_load(mod_fitcont)

            # storing the absorption of the broad fit if there is absorption
            abs_incl_comps = (np.array(fitcont_broad.complist)[[elem.absorption and elem.included for elem in \
                                                                [elem_comp for elem_comp in fitcont_broad.complist if
                                                                 elem_comp is not None]]])
            if len(abs_incl_comps) != 0:
                main_abs_comp = abs_incl_comps[0]
                broad_absval = main_abs_comp.xcomps[0].nH.values[0]
                broad_abscomp = main_abs_comp.compname
            else:
                main_abs_comp = None
                broad_absval = 0
                broad_abscomp = ''

            for i_sp in range(len(epoch_files_good)):
                if line_cont_ig_indiv[i_sp] != '':
                    AllData(i_sp + 1).notice(line_cont_ig_indiv[i_sp])

            curr_store_fit(mode='broadband', fitmod=fitcont_broad)

            # storing the class
            fitcont_broad.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadband.pkl')

            # saving the model
            data_broad = allmodel_data()
            print_xlog('\nComputing HID broad fit...')

            for i_sp in range(len(epoch_files_good)):
                if line_cont_ig_indiv[i_sp] != '':
                    AllData(i_sp + 1).ignore(line_cont_ig_indiv[i_sp])

            if broad_HID_mode:
                fitcont_hid = fitcont_broad

            else:
                AllModels.clear()

                # reloading the scorpeon save (if there is one, aka if with NICER),
                # from the broad fit and freezing it to avoid further variations
                xscorpeon.load('auto', scorpeon_save=data_broad.scorpeon, frozen=True)

                ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                                  sat_high_groups=e_sat_high_indiv, glob_ignore_bands=ignore_bands_indiv)

                # if the stat is low we don't do the autofit anyway so we'd rather get the best fit possible
                if not flag_lowSNR_line:
                    for i_sp in range(len(epoch_files_good)):
                        if line_cont_ig_indiv[i_sp] != '':
                            AllData(i_sp + 1).ignore(line_cont_ig_indiv[i_sp])

                # creating the automatic fit class for the standard continuum
                # (without absorption if it didn't get included)
                if broad_absval != 0:
                    fitcont_hid = fitmod(comp_cont, curr_logfile, curr_logfile_write, absval=broad_absval,
                                         mandatory_abs=mandatory_abs)
                else:
                    # creating the fitcont without the absorption component if it didn't exist in the broad model
                    fitcont_hid = fitmod([elem for elem in comp_cont if elem != broad_abscomp],
                                         curr_logfile, curr_logfile_write)

                # forcing the absorption component to be included for the broad band fit
                if sat_glob == 'NuSTAR' and freeze_nH:
                    main_abscomp = (np.array(fitcont_hid.complist)[[elem.absorption for elem in \
                                                                    fitcont_hid.complist]])[0]
                    main_abscomp.mandatory = True

                # fitting
                fitcont_hid.global_fit(split_fit=split_fit, method=cont_fit_method)

                mod_fitcont = allmodel_data()

            chi2_cont = Fit.statistic

            if chi2_cont == 0:
                print_xlog('\nProblem during hid band fit. Skipping line detection for this exposure...')
                return ['\nProblem during hid band fit. Skipping line detection for this exposure...']

            spflux_single, spflux_high = hid_fit_infos(fitcont_hid, broad_absval)

            return spflux_single, broad_absval, broad_abscomp, data_broad, fitcont_broad, broad_gamma_compt, spflux_high, \
                thcomp_frac_frozen

        AllModels.clear()

        # frozen Scorpeon for now (unfreezed during the broad fit)
        xscorpeon.load('auto', frozen=True)

        ignore_data_indiv(e_sat_low_indiv, e_sat_high_indiv, reset=True, glob_ignore_bands=ignore_bands_indiv)

        baseload_path = outdir + '/' + epoch_observ[0] + '_baseload.xcm'
        if os.path.isfile(baseload_path):
            os.remove(baseload_path)

        Xset.save(outdir + '/' + epoch_observ[0] + '_baseload.xcm', info='a')

        result_broad_fit = broad_fit()

        if len(result_broad_fit) == 1:
            return fill_result(result_broad_fit)
        else:
            main_spflux, broad_absval, broad_abscomp, data_broad, fitcont_broad, broad_gamma_compt, main_spflux_high, \
                thcomp_frac_frozen = result_broad_fit

        if max(e_sat_high_indiv) >= 20:
            np.savetxt(outdir + '/' + epoch_observ[0] + '_main_spflux_high.txt', main_spflux_high)

        # reloading the frozen scorpeon data\
        # (which won't change anything if it hasn't been fitted but will help otherwise)
        xscorpeon.load('auto', scorpeon_save=data_broad.scorpeon, frozen=True)

        '''
        automatic detector exclusions base on their energy range to avoid crashes later
        '''
        mask_nodeload = e_sat_low_indiv<10
        reload_sp(baseload_path, newbl_keyword='autofit',method='new',mask=mask_nodeload)
        data_broad_deload=deepcopy(data_broad)
        scorpeon_deload=data_broad_deload.scorpeon

        data_broad_deload = deepcopy(data_broad)
        data_broad_deload.default = data_broad_deload.default[mask_nodeload]

        #note always defined at least as a list of nones even if no scorpeon model exists, so can be masked
        data_broad_deload.scorpeon.nxb_save_list = np.array(data_broad_deload.scorpeon.nxb_save_list)[mask_nodeload]
        data_broad_deload.scorpeon.sky_save_list = np.array(data_broad_deload.scorpeon.sky_save_list)[mask_nodeload]

        scorpeon_deload=data_broad_deload.scorpeon

        result_high_fit = high_fit(broad_absval, broad_abscomp, broad_gamma_compt,scorpeon_deload,
                                   thcomp_frac_frozen=thcomp_frac_frozen)

        # if the function returns an array of length 1, it means it returned an error message
        if len(result_high_fit) == 1:
            return fill_result(result_high_fit)
        else:
            data_mod_high, fitmod_cont = result_high_fit

        # re-limiting to the line search energy range
        ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                          sat_low_groups=e_sat_low_indiv[mask_nodeload],
                          sat_high_groups=e_sat_high_indiv[mask_nodeload],
                          glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

        # changing back to the auto rescale of xspec
        Plot.commands = ()
        Plot.addCommand('rescale')

        print_xlog('\nStarting line search...')

        cont_abspeak, cont_peak_points, cont_peak_widths, cont_peak_delchis, cont_peak_eqws, chi_dict_init = \
            narrow_line_search(data_mod_high, 'cont', line_search_e=line_search_e, line_search_norm=line_search_norm,
                               e_sat_low_indiv=e_sat_low_indiv[mask_nodeload],
                               peak_thresh=peak_thresh, peak_clean=peak_clean,
                               line_cont_range=line_cont_range, trig_interval=trig_interval,
                               scorpeon_save=data_mod_high.scorpeon)

        with open(outdir + '/' + epoch_observ[0] + '_chi_dict_init.pkl', 'wb') as file:
            dill.dump(chi_dict_init, file)

        # same for autofit, then specific mode

        plot_line_search(chi_dict_init, outdir, sat_glob, suffix='cont', epoch_observ=epoch_observ)

        '''
        Automatic line fitting
        '''

        #### Autofit

        if autofit and (cont_abspeak or force_autofit) and not flag_lowSNR_line:

            '''
            See the fitmod code for a detailed description of the auto fit process
            '''

            # reloading the continuum fitcomp
            data_mod_high.load()

            # feching the list of components we're gonna use
            comp_lines = model_list(autofit_model)

            # creating a new logfile for the autofit
            curr_logfile_write = Xset.openLog(outdir + '/' + epoch_observ[0] + '_xspec_log_autofit.log')
            curr_logfile_write.reconfigure(line_buffering=True)
            curr_logfile = open(curr_logfile_write.name, 'r')

            # creating the fitmod object with the desired components (we currently do not use comp groups)
            fitlines = fitmod(comp_lines, curr_logfile, curr_logfile_write, prev_fitmod=fitmod_cont,
                              sat_list=sat_indiv_good, fixed_gamma=broad_gamma_compt,
                              thcomp_frac_frozen=thcomp_frac_frozen,
                              mandatory_abs=mandatory_abs)

            # global fit, with MC only if no continuum refitting
            fitlines.global_fit(chain=not refit_cont, directory=outdir, observ_id=epoch_observ[0], split_fit=split_fit,
                                no_abslines=no_abslines)

            # storing the final fit
            data_autofit = allmodel_data()

            '''
            ####Refitting the continuum

            if refit_cont is set to True, we add a 3 step process to better estimate the continuum from the autofit lines.

            First, we relaunch a global fit iteration while blocking all line parameters in the broad band.
            We then refreeze the absorption and relaunch two global fit iterations in 3-10 (for the HID) and 4-10 (for the autofit continuum)
            '''

            if refit_cont:

                # don't know what's happening
                from xspec import AllChains

                AllChains.clear()

                # saving the initial autofit result for checking purposes
                curr_store_fit(mode='autofit_init', fitmod=fitlines)

                # storing the fitmod class into a file
                fitlines.dump(outdir + '/' + epoch_observ[0] + '_fitmod_autofit_init.pkl')

                # updating the logfile for the second round of fitting
                curr_logfile_write = Xset.openLog(outdir + '/' + epoch_observ[0] + '_xspec_log_autofit_recont.log')
                curr_logfile_write.reconfigure(line_buffering=True)
                curr_logfile = open(curr_logfile_write.name, 'r')

                # updating it in the fitmod
                fitlines.logfile = curr_logfile
                fitlines.logfile_write = curr_logfile_write
                fitlines.update_fitcomps()

                # freezing every line component
                for comp in [elem for elem in fitlines.includedlist if elem is not None]:
                    if comp.line:
                        freeze(parlist=comp.unlocked_pars)

                # reloading the broad band data without overwriting the model
                Xset.restore(baseload_path)

                fitlines.update_fitcomps()

                '''
                To reload the broadband model and avoid issues, 
                we need to shift some datagroups and recreate the links properly
                This will need to be done eventually but for now with just pin going back and forth
                as the last datagroup this should be fine
                '''

                # refitting in broad band for the nH
                ignore_data_indiv(e_sat_low_indiv, e_sat_high_indiv, reset=True, glob_ignore_bands=ignore_bands_indiv)

                # merging the previous continuum if asked to
                if merge_cont:
                    fitlines.merge(fitcont_broad)

                # thawing the absorption to allow improving its value

                # first we remove the fixed value from the fitmod itself
                if not (sat_glob == 'NuSTAR' and freeze_nH):
                    # we reset the value of the fixed abs to allow it to be free if it gets deleted and put again
                    fitlines.fixed_abs = None

                # then we attempt to thaw the component value if the component is included
                abs_incl_comps = (np.array(fitlines.complist)[[elem.absorption and elem.included for elem in \
                                                               [elem_comp for elem_comp in fitlines.complist
                                                                if
                                                                elem_comp is not None]]])

                if len(abs_incl_comps) != 0 and not (sat_glob == 'NuSTAR' and freeze_nH):
                    main_abscomp = abs_incl_comps[0]
                    main_abscomp.xcomps[0].nH.frozen = False

                    # releasing the n_unlocked_pars of the absorption component
                    fitlines.update_fitcomps()
                    main_abscomp.n_unlocked_pars_base = len(main_abscomp.unlocked_pars)

                # thawing similarly the nthcomp gamma and removing the fixed restriction
                if 'disk_nthcomp' in fitlines.name_complist or \
                        ('disk_thcomp' in fitlines.name_complist and not fitlines.thcomp_frac_frozen):
                    fitlines.fixed_gamma = None

                if 'disk_nthcomp' in [comp.compname for comp in \
                                      fitlines.includedlist_main]:
                    fitlines.disk_nthcomp.xcomps[0].Gamma.frozen = False

                    fitlines.update_fitcomps()
                    fitlines.disk_nthcomp.n_unlocked_pars_base = len(fitlines.disk_nthcomp.unlocked_pars)

                # same thing for thcomp, but only if the thcomp was not negligible to begin with
                elif 'disk_thcomp' in [comp.compname for comp in \
                                       fitlines.includedlist_main] and not fitlines.thcomp_frac_frozen:

                    fitlines.disk_thcomp.xcomps[0].Gamma_tau.frozen = False

                    fitlines.update_fitcomps()
                    fitlines.disk_thcomp.n_unlocked_pars_base = len(fitlines.disk_thcomp.unlocked_pars)

                # fitting the model to the new energy band first
                calc_fit(logfile=fitlines.logfile)

                # autofit
                fitlines.global_fit(chain=False, lock_lines=True, directory=outdir, observ_id=epoch_observ[0],
                                    split_fit=split_fit, fit_scorpeon=True, fit_SAA_norm=fit_SAA_norm)

                # refreezing the scorpeon model
                xscorpeon.freeze()

                data_broad_postauto = allmodel_data()

                AllChains.clear()

                curr_store_fit(mode='broadband_post_auto', fitmod=fitlines)

                # storing the class
                fitlines.dump(outdir + '/' + epoch_observ[0] + '_fitmod_broadband_post_auto.pkl')

                # remaking this list just in case there was some deletion and addition and the components changed
                abs_incl_comps = (np.array(fitlines.complist)[[elem.absorption and elem.included for elem in \
                                                               [elem_comp for elem_comp in fitlines.complist
                                                                if
                                                                elem_comp is not None]]])

                # re fixing the absorption parameter and storing the value to retain it if the absorption
                # gets taken off and tested again
                fitlines.fixed_abs = broad_absval
                if len(abs_incl_comps) != 0:
                    main_abscomp = abs_incl_comps[0]
                    main_abscomp.xcomps[0].nH.frozen = True
                    broad_absval = main_abscomp.xcomps[0].nH.values[0]

                    fitlines.update_fitcomps()
                    main_abscomp.n_unlocked_pars_base = len(main_abscomp.unlocked_pars)
                else:
                    broad_absval = 0

                if not broad_HID_mode:
                    # refitting in hid band for the HID values
                    ignore_data_indiv(hid_cont_range[0], hid_cont_range[1], reset=True, sat_low_groups=e_sat_low_indiv,
                                      sat_high_groups=e_sat_high_indiv, glob_ignore_bands=ignore_bands_indiv)

                    # fitting the model to the new energy band first
                    calc_fit(logfile=fitlines.logfile)

                    # autofit
                    fitlines.global_fit(chain=False, lock_lines=True, directory=outdir, observ_id=epoch_observ[0],
                                        split_fit=split_fit)

                AllChains.clear()

                main_spflux, main_spflux_high = hid_fit_infos(fitlines, broad_absval, post_autofit=True)

                if max(e_sat_high_indiv) >= 20:
                    np.savetxt(outdir + '/' + epoch_observ[0] + '_main_spflux_high.txt', main_spflux_high)

                '''
                Refitting in the autofit range to get the newer version of the autofit and continuum

                first: refreezing the nthcomp/thcomp gamma if necessary

                second: restoring the line freeze states
                here we restore the INITIAL component freeze state, effectively thawing all components pegged during the first autofit
                '''

                if fix_compt_gamma:

                    #getting the broad value in case of deletion
                    if 'disk_nthcomp' in [comp.compname for comp in \
                                          [elem for elem in fitlines.includedlist if elem is not None]]:
                        broad_gamma_compt = fitlines.disk_nthcomp.xcomps[0].Gamma.values[0]
                    elif 'disk_thcomp' in [comp.compname for comp in \
                                           [elem for elem in fitlines.includedlist if elem is not None]]:
                        broad_gamma_compt = fitlines.disk_thcomp.xcomps[0].Gamma_tau.values[0]
                    else:
                        broad_gamma_compt = None

                    #and ajusting the current values
                    if 'disk_nthcomp' in [comp.compname for comp in \
                                          fitlines.includedlist_main]:
                        fitlines.disk_nthcomp.xcomps[0].Gamma.frozen = True

                        fitlines.update_fitcomps()
                        fitlines.disk_nthcomp.n_unlocked_pars_base = len(fitlines.disk_nthcomp.unlocked_pars)

                    if 'disk_nthcomp' in [comp.compname for comp in \
                                          fitlines.includedlist_main]:
                        fitlines.disk_thcomp.xcomps[0].Gamma_tau.frozen = True

                        fitlines.update_fitcomps()
                        fitlines.disk_thcomp.n_unlocked_pars_base = len(fitlines.disk_thcomp.unlocked_pars)

                else:
                    broad_gamma_compt = None

                # setting the fixed gamma to its value it has been selected
                fitlines.fixed_gamma = broad_gamma_compt


                #removing irrelevant components
                for comp in [elem for elem in fitlines.includedlist if elem is not None]:

                    if comp.line and not comp.calibration:
                        # unfreezing the parameter with the mask created at the first addition of the component

                        unfreeze(parlist=np.array(comp.parlist)[comp.unlocked_pars_base_mask])

                    if comp.calibration and comp.compname not in ['calNuSTAR_edge']:
                        fitlines.remove_comp(comp)


                # re-ignoring high-E only telescopes
                mod_pre_autofit = allmodel_data()

                #reloading the high-E baseload
                Xset.restore(baseload_path.replace('.xcm', '_autofit.xcm'))

                # restoring the rest of the datagroups because reloading a different number of datagroup
                # resets the links of all but the first DGs
                mod_pre_autofit.default = mod_pre_autofit.default[mask_nodeload]
                mod_pre_autofit.scorpeon.nxb_save_list=np.array(mod_pre_autofit.scorpeon.nxb_save_list)[mask_nodeload]
                mod_pre_autofit.scorpeon.sky_save_list=np.array(mod_pre_autofit.scorpeon.sky_save_list)[mask_nodeload]

                # note that more things would need to be done for NICER/Suzaku and
                # some things might not work with NuSTAR edge's links but let's see
                mod_pre_autofit.load()

                fitlines.update_fitcomps()

                ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                                  sat_low_groups=e_sat_low_indiv[mask_nodeload],
                                  sat_high_groups=e_sat_high_indiv[mask_nodeload],
                                  glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

                # fitting the model to the new energy band first
                calc_fit(logfile=fitlines.logfile)

                # autofit
                fitlines.global_fit(chain=True, directory=outdir, observ_id=epoch_observ[0], split_fit=split_fit,
                                    no_abslines=no_abslines)

                # storing the final fit
                data_autofit = allmodel_data()


            # storing the final plot and parameters
            # screening the xspec plot
            Plot_screen("ldata,ratio,delchi", outdir + '/' + epoch_observ[0] + "_screen_xspec_autofit",
                        includedlist=fitlines.includedlist)

            if 'Chandra' in sat_indiv_good:
                # plotting a zoomed version for HETG spectra
                AllData.ignore('**-6.5 ' + str(float(min(9, max(e_sat_high_indiv)))) + '-**')

                Plot_screen("ldata,ratio,delchi", outdir + '/' + epoch_observ[0] + "_screen_xspec_autofit_zoom",
                            includedlist=fitlines.includedlist)

                # putting back the energy range
                AllData.notice(str(line_cont_range[0]) + '-' + str(line_cont_range[1]))

            # saving the model str
            catch_model_str(curr_logfile, savepath=outdir + '/' + epoch_observ[0] + '_mod_autofit.txt')

            if os.path.isfile(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm'):
                os.remove(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm')

            # storing the current configuration and model
            Xset.save(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm', info='a')

            # storing the class
            fitlines.dump(outdir + '/' + epoch_observ[0] + '_fitmod_autofit.pkl')

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

        # storing the components of the model for the first data group only
        plot_autofit_data, plot_autofit_comps = store_plot('ldata', comps=True)
        plot_autofit_data = plot_autofit_data[0]
        plot_autofit_comps = plot_autofit_comps[0]

        # computing the names of the additive continuum components remaning in the autofit
        addcomps_cont = [comp.compname.split('_')[1] for comp in
                         [elem for elem in fitlines.includedlist if elem is not None] \
                         if 'gaussian' not in comp.compname and not comp.multipl]

        # same with the lines
        addcomps_lines = [comp.compname.split('_')[0] for comp in
                          [elem for elem in fitlines.includedlist if elem is not None] \
                          if 'gaussian' in comp.compname]

        addcomps_abslines = [comp.compname.split('_')[0] for comp in
                             [elem for elem in fitlines.includedlist if elem is not None] \
                             if comp.named_absline]

        comp_absline_position = []
        for id_comp_absline, comp_absline in enumerate(addcomps_abslines):
            comp_absline_position += [np.argwhere(np.array(addcomps_lines) == comp_absline)[0][0]]

        # rearranging the components in a format usable in the plot. The components start at the index 2
        # (before it's the entire model x and y values)
        plot_autofit_cont = plot_autofit_comps[:2 + len(addcomps_cont)]

        # same for the line components
        plot_autofit_lines = plot_autofit_comps[2 + len(addcomps_cont):]

        # taking off potential background components

        if 'nxb' in list(AllModels.sources.values()):
            plot_autofit_lines = plot_autofit_lines[:-2]

        if 'sky' in list(AllModels.sources.values()):
            plot_autofit_lines = plot_autofit_lines[:-2]

        '''
        #### Testing the overlap of absorption lines with emission lines
        We compare the overlapping area (in count space) of each absorption line to the sum of emission lines
        The final threshold is set in fraction of the absorption line area

        we fix the limit for emission lines to be considered as continuum (and thus not taken into account in this computation)
        when their width is more than 1/4 of the line detection range 
        This is the threshold for their 2 sigma width (or 95% of their area) to be under the line detection range
        '''

        # first, we compute the sum of the "narrow" emission line components
        sum_autofit_emlines = np.zeros(len(plot_autofit_cont[0]))

        included_linecomps = [elem for elem in [elem for elem in fitlines.includedlist if elem is not None] \
                              if 'gaussian' in elem.compname]

        for ind_line, line_fitcomp in enumerate(included_linecomps):

            # skipping absorption lines
            if line_fitcomp.named_absline:
                continue

            # #width limit test
            # of now that the line comps are not allowed to have stupid widths
            # if AllModels(1)(line_fitcomp.parlist[1])*8<(line_search_e[1]-line_search_e[0]):

            # adding the right component to the sum
            sum_autofit_emlines += plot_autofit_lines[ind_line]

        abslines_em_overlap = np.zeros(n_absline)

        # and then the overlap for each absorption
        for ind_line, line_fitcomp in enumerate(included_linecomps):

            # skipping emission lines
            if not line_fitcomp.named_absline:
                continue

            # computing the overlapping bins
            overlap_bins = np.array([abs(plot_autofit_lines[ind_line]), sum_autofit_emlines]).min(0)

            # and the integral from that
            overlap_frac = trapezoid(overlap_bins, x=plot_autofit_cont[0]) / \
                           abs(trapezoid(plot_autofit_lines[ind_line], x=plot_autofit_cont[0]))

            # fetching the index of the line being tested
            line_ind_std = np.argwhere(np.array(lines_std_names[3:]) == line_fitcomp.compname.split('_')[0])[0][0]

            # and storing the value
            abslines_em_overlap[line_ind_std] = overlap_frac

        '''
        in order to compute the continuum ratio evolution we compute the sum of all the non absorption line components,
        then the ratio when adding each absorption line component
        '''

        plot_autofit_noabs = np.concatenate((([[plot_addline] for plot_addline in plot_autofit_comps[2:] \
                                               if not max(plot_addline) <= 0]))).sum(axis=0)

        plot_autofit_ratio_lines = [(plot_autofit_noabs + plot_autofit_lines[i]) / plot_autofit_noabs \
                                    for i in range(len(plot_autofit_lines)) if max(plot_autofit_lines[i]) <= 0.]

        '''
        Chain computation for the MC significance
        '''

        # drawing parameters for the MC significance test later
        autofit_drawpars = np.array([None] * nfakes)

        print_xlog('\nDrawing parameters from the Chain...')
        for i_draw in range(nfakes):
            curr_simpar = AllModels.simpars()

            # we restrict the simpar to the initial model because we don't really care about simulating the variations
            # of the bg since it's currently frozen
            autofit_drawpars[i_draw] = np.array(curr_simpar)[:AllData.nGroups * AllModels(1).nParameters] \
                .reshape(AllData.nGroups, AllModels(1).nParameters)

        # turning it back into a regular array
        autofit_drawpars = np.array([elem for elem in autofit_drawpars])

        # storing the parameter and errors of all the components, as well as their corresponding name
        autofit_parerrors, autofit_parnames = fitlines.get_usedpars_vals()

        print_xlog('\nComputing informations from the fit...')

        #### Computing line parameters

        # fetching informations about the absorption lines
        abslines_flux, abslines_eqw, abslines_bshift, abslines_delchi, abslines_bshift_distinct = fitlines.get_absline_info(
            autofit_drawpars)

        from xspec import AllChains

        # clearing the chain before doing anything else
        AllChains.clear()

        Fit.perform()

        # computing the 3-sigma width without the MC to avoid issues with values being too different from 0
        abslines_width = fitlines.get_absline_width()

        '''
        Saving a "continuum" version of the model without absorption
        '''

        # this first part is to get the mask to reduce autofit_drawpars_cont to its version without
        # the abslines

        # We store the indexes of the absgaussian parameters to shift the rest of the parameters accordingly after
        # deleting those components
        abslines_parsets = np.array(
            [elem.parlist for elem in fitlines.includedlist if elem is not None and elem.named_absline])

        abslines_parids = ravel_ragged(abslines_parsets)

        # covering for the case where the list was mono-dimensional
        if type(abslines_parids) != list:
            abslines_parids = abslines_parids.tolist()

        # switching to indexes instead of actual parameter numbers
        abslines_arrids = [elem - 1 for elem in abslines_parids]

        # creating the list of continuum array indexes (note that this will be applied for each datagroups)
        # this is for A SINGLE DATAGROUP because autofit_drawpars has a adatagroup dimension
        continuum_arrids = [elem for elem in np.arange(AllModels(1).nParameters) if elem not in abslines_arrids]

        # and parameters
        continuum_parids = [elem + 1 for elem in continuum_arrids]

        # to create a drawpar array which works with thereduced model (this line works with several data groups)
        autofit_drawpars_cont = autofit_drawpars.T[continuum_arrids].T

        # and now delete the abslines to properly get the forced continuum parameters in the 'no-abs' state
        full_includedlist = fitlines.includedlist

        # deleting all absorption components (reversed so we don't have to update the fitcomps)
        for comp in [elem for elem in fitlines.includedlist if elem is not None][::-1]:
            if comp.named_absline:
                # note that with no rollback we do not update the methods of the component
                # so it has still its included status and everything else
                ndel = comp.delfrommod(rollback=False)
                fitlines.includedlist = fitlines.includedlist[:-ndel]
                fitlines.update_fitcomps()

        # storing the no abs line 'continuum' model
        data_autofit_noabs = allmodel_data()

        # recording all the frozen/linked parameters in the model to refreeze them during the fakeit fit
        # note that here we don't care about how things are linked, just freezing them to their drawpared value
        # is enough because the drawpar itself will have been generated with the correct link
        continuum_forcedpars = [i_par for i_par in range(1, AllData.nGroups * AllModels(1).nParameters + 1) \
                                if AllModels(par_degroup(i_par)[0])(par_degroup(i_par)[1]).frozen \
                                or AllModels(par_degroup(i_par)[0])(par_degroup(i_par)[1]).link != '']

        continuu_ranges = []
        plot_ratio_autofit_noabs = store_plot('ratio')

        # resetting the states of fitlines
        fitlines.includedlist = full_includedlist
        data_autofit.load()
        fitlines.update_fitcomps()

        # and comming back to the noabs state
        data_autofit_noabs.load()

        # plotting the combined autofit plot

        def autofit_plot(data, data_noabs, addcomps_cont, comp_pos):

            '''
            The goal here is to plot the autofit in a way that shows the different lines
            '''

            gs_comb = GridSpec(2, 1, hspace=0.)

            axes = [None, None]
            # first subplot is the ratio
            axes[0] = plt.subplot(gs_comb[0, 0])
            axes[1] = plt.subplot(gs_comb[1, 0])

            '''first plot (components)'''

            # We only plot this for the first data group, no need to show all of them since the only difference is a constant factor
            plot_line_comps(axes[0], plot_autofit_cont, addcomps_cont, plot_autofit_lines, addcomps_lines,
                            combined=True)

            '''second plot (ratio + abslines ratio)'''

            plot_line_ratio(axes[1], data_autofit=data, data_autofit_noabs=data_noabs,
                            n_addcomps_cont=len(addcomps_cont), line_position=comp_pos,
                            line_search_e=line_search_e, line_cont_range=line_cont_range)

            plt.tight_layout()

        fig_autofit = plt.figure(figsize=(15, 10))

        autofit_plot(data=data_autofit, data_noabs=data_autofit_noabs, addcomps_cont=addcomps_cont,
                     comp_pos=comp_absline_position)

        plt.savefig(outdir + '/' + epoch_observ[0] + '_autofit_components_plot_' + line_search_e_arg.replace(' ',
                                                                                                              '_') + '_' + line_search_norm_arg.replace(
            ' ', '_') + '.png')
        plt.close(fig_autofit)

        '''
        Autofit residuals assessment
        '''

        chi_dict_autofit = narrow_line_search(data_autofit, 'autofit',
                                              line_search_e=line_search_e, line_search_norm=line_search_norm,
                                              e_sat_low_indiv=e_sat_low_indiv[mask_nodeload],
                                              peak_thresh=peak_thresh, peak_clean=peak_clean,
                                              line_cont_range=line_cont_range, trig_interval=trig_interval,
                                              scorpeon_save=data_autofit.scorpeon,
                                              data_fluxcont=data_autofit_noabs)

        with open(outdir + '/' + epoch_observ[0] + '_chi_dict_autofit.pkl', 'wb') as file:
            dill.dump(chi_dict_init, file)

        plot_line_search(chi_dict_autofit, outdir, sat_glob, suffix='autofit', epoch_observ=epoch_observ)

        ####Paper plot

        def paper_plot(fig_paper, chi_dict_init, chi_dict_postauto, title=None):

            line_cont_range = chi_dict_init['line_cont_range']
            ax_paper = np.array([None] * 4)
            fig_paper.suptitle(title)

            # gridspec creates a grid of spaces for subplots. We use 4 rows for the 4 plots
            # Second column is there to keep space for the colorbar. Hspace=0. sticks the plots together
            gs_paper = GridSpec(4, 2, figure=fig_paper, width_ratios=[100, 0], hspace=0.)

            # first plot is the data with additive components
            ax_paper[0] = plt.subplot(gs_paper[0, 0])
            prev_plot_add = Plot.add
            Plot.add = True

            # reloading the pre-autofit continuum for display
            data_mod_high.load()

            xPlot('ldata', axes_input=ax_paper[0])

            # loading the no abs autofit
            data_autofit_noabs.load()

            Plot.add = prev_plot_add

            # second plot is the first blind search coltour
            ax_paper[1] = plt.subplot(gs_paper[1, 0], sharex=ax_paper[0])
            ax_colorbar = plt.subplot(gs_paper[1, 1])
            coltour_chi2map(fig_paper, ax_paper[1], chi_dict_init, combined='paper', ax_bar=ax_colorbar)
            ax_paper[1].set_xlim(line_cont_range)

            ax_paper[2] = plt.subplot(gs_paper[2, 0], sharex=ax_paper[0])
            # third plot is the autofit ratio with lines added
            plot_line_ratio(ax_paper[2], mode='paper', data_autofit=data_autofit, data_autofit_noabs=data_autofit_noabs,
                            n_addcomps_cont=len(addcomps_cont), line_position=comp_absline_position,
                            line_search_e=line_search_e, line_cont_range=line_cont_range)

            # fourth plot is the second blind search coltour
            ax_paper[3] = plt.subplot(gs_paper[3, 0], sharex=ax_paper[0])
            ax_colorbar = plt.subplot(gs_paper[3, 1])

            # coltour_chi2map(fig_paper,ax_paper[3],chi_dict_postauto,combined='nolegend',ax_bar='bottom',norm=(251.5,12.6))

            # need to fix the colorbar here
            coltour_chi2map(fig_paper, ax_paper[3], chi_dict_postauto, combined='nolegend', ax_bar=ax_colorbar)

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

        fig_paper = plt.figure(figsize=(14.5, 22))

        paper_plot(fig_paper, chi_dict_init, chi_dict_autofit)

        plt.savefig(outdir + '/' + epoch_observ[0] + '_paper_plot_' + line_search_e_arg.replace(' ', '_') +
                    '_' + line_search_norm_arg.replace(' ', '_') + '.png')
        plt.savefig(outdir + '/' + epoch_observ[0] + '_paper_plot_' + line_search_e_arg.replace(' ', '_') +
                    '_' + line_search_norm_arg.replace(' ', '_') + '.pdf')

        plt.close(fig_paper)

        data_autofit_noabs.load()

        # we don't update the fitcomps here because it would require taking off the abslines from the includedlists
        # and we don't want that for the significance computation

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

        # computing an array of significantly non-zero widths
        sign_widths_arr = np.array([elem[0] if elem[0] - elem[1] > 1e-6 else 0 for elem in abslines_width])

        abslines_sign = np.zeros((len(abslines_delchi)))
        abslines_eqw_upper = np.zeros((len(abslines_delchi)))

        is_absline = np.array([included_comp.named_absline for included_comp in \
                               [comp for comp in fitlines.includedlist if comp is not None]]).any()

        # updating the logfile for the second round of fitting
        curr_logfile_write = Xset.openLog(outdir + '/' + epoch_observ[0] + '_xspec_log_fakeits.log')
        curr_logfile_write.reconfigure(line_buffering=True)
        curr_logfile = open(curr_logfile_write.name, 'r')

        # updating it in the fitmod
        fitlines.logfile = curr_logfile
        fitlines.logfile_write = curr_logfile_write

        Xset.logChatter = 2

        '''
        In order to make the fakeit process as fast as possible, we don't write the fakes in file 
        and directly fit them as loaded every time
        For that, we specify fakeitsettings with given filename, bg, resp etc. 
        (since by default fakeit overwrite those while loading the spectra)
        and use the nowrite command to keep from overwriting the files
        By not giving an exposure, we assume the loaded spectra's exposures
        '''

        print_xlog('\nCreating fake spectra to assess line significance...')

        fakeset = [FakeitSettings(response='' if not isrmf_grp[i_grp - 1] else AllData(i_grp).response.rmf,
                                  arf='' if not isarf_grp[i_grp - 1] else AllData(i_grp).response.arf,
                                  background='' if not isbg_grp[i_grp - 1] else AllData(i_grp).background.fileName,
                                  fileName=AllData(i_grp).fileName) for i_grp in range(1, AllData.nGroups + 1)]

        # array for delchi storing
        delchi_arr_fake = np.zeros(
            (nfakes, round((line_search_e_space[-1] - line_search_e_space[0]) / line_search_e[2]) + 1))
        delchi_arr_fake_line = np.zeros((6, nfakes))
        # eqw_arr_fake=np.zeros((nfakes,6))

        steppar_ind_list = []

        line_id_list = []

        '''
        Since we now have specific energy intervals for each line, we can restrict the delchi test to the interval
        of each line. Thus, to optimize computing time, we compute which indexes need to be computed for all lines
        and compute steppars for each interval among those indexes
        '''

        # assessing the range of steppar to use for of each line
        for i_line in range(len(abslines_sign)):

            # skipping the computation for lines above 8 keV in restrict mode when we don't go above 8keV anyway
            if restrict_graded and i_line >= 2:
                continue

            # here we skip the first two emission lines
            line_name = list(lines_e_dict.keys())[i_line + 3]

            # fetching the lower and upper bounds of the energies from the blueshifts
            # here we add a failsafe for the upper part of the steppar to avoid going beyond the energies ignored which crashes it

            line_lower_e = lines_e_dict[line_name][0] * (1 + lines_e_dict[line_name][1] / c_light)
            line_upper_e = min(lines_e_dict[line_name][0] * (1 + lines_e_dict[line_name][2] / c_light),
                               max(e_sat_high_indiv))

            # computing the corresponding indexes in the delchi array
            line_lower_ind = int((line_lower_e - line_search_e_space[0]) // line_search_e[2])
            line_upper_ind = int((line_upper_e - line_search_e_space[0]) // line_search_e[2] + 1)

            # skipping the interval if the line has not been detected
            if abslines_eqw[i_line][0] == 0:
                continue

            # adding the parts of the line_search_e_space which need to be computed to an array
            steppar_ind_list += np.arange(line_lower_ind, line_upper_ind + 1).tolist()

            # adding the index to the list of line indexes to be tested
            line_id_list += [i_line]

        loaded_fakes = False

        if is_absline and assess_line:

            # attempting to reload the fake delchi arr if allowed
            if reload_fakes and os.path.isfile(outdir + '/' + epoch_observ[0] + '_delchi_arr_fake_line.npy'):
                print_xlog('Complete fake computation detected. Reloading...')

                delchi_arr_fake_line = np.load(outdir + '/' + epoch_observ[0] + '_delchi_arr_fake_line.npy')
                loaded_fakes = True

            if not loaded_fakes:

                # now we compute the list of intervals that can be made from that
                steppar_ind_unique = np.unique(steppar_ind_list)

                steppar_ind_inter = list(interval_extract(steppar_ind_unique))

                # fake loop
                with tqdm(total=nfakes) as pbar:
                    for f_ind in range(nfakes):

                        # reloading the high energy continuum
                        data_autofit_noabs.load()

                        # Freezing it to ensure the fakeit doesn't make the parameters vary, and loading them from a steppar
                        for i_grp in range(1, AllData.nGroups + 1):

                            # computing which setpars need to be adjusted from the list
                            # in the case where the parsed value is out of the default continuum range

                            # skipping the drawpar elements which would give things already nomally linked,
                            # to avoid issues with parameter ranges ('' is the default value for SetPars)
                            curr_group_drawpar = autofit_drawpars_cont[f_ind][i_grp - 1].tolist()
                            curr_group_drawpar_dict = {}
                            for group_par in range(1, AllModels(1).nParameters + 1):
                                if AllModels(i_grp)(group_par).link != '' and \
                                        int(AllModels(i_grp)(group_par).link.replace('= p', '')) == group_par:
                                    continue
                                curr_group_drawpar_dict[group_par] = curr_group_drawpar[group_par - 1]

                            # has to switch to dictionnaries to avoid overwriting parameter values to their
                            # standard default due to link disappearances in the new pyxspec version
                            AllModels(i_grp).setPars(curr_group_drawpar)

                            # freezing doesn't change anything for linked parameters
                            freeze(AllModels(i_grp))

                        # replacing the current spectra with a fake with the same characteristics so this can be looped
                        # applyStats is set to true but shouldn't matter for now since everything is frozen

                        AllData.fakeit(settings=fakeset, applyStats=True, noWrite=True)

                        # energy modifications reset when using fakeit, so we update
                        # them in case there is a component requires it in the model
                        AllModels.setEnergies('0.01 1000.0 5000 log')

                        # limiting to the line search energy range
                        ignore_data_indiv(line_cont_range[0], line_cont_range[1], reset=True,
                                          sat_low_groups=e_sat_low_indiv[mask_nodeload],
                                          sat_high_groups=e_sat_high_indiv[mask_nodeload],
                                          glob_ignore_bands=ignore_bands_indiv[mask_nodeload])

                        # adjusting the fit and storing the chi²

                        for i_grp in range(1, AllData.nGroups + 1):
                            # unfreezing the model
                            unfreeze(AllModels(i_grp))

                        for par_forced in continuum_forcedpars:
                            AllModels(par_degroup(par_forced)[0])(par_degroup(par_forced)[1]).frozen = True

                            # previous method which wouldn't work with more complicated unlinks and free pars
                            # for multigroup models
                            # #keeping the initially frozen parameters frozen
                            # freeze(AllModels(i_grp),parlist=continuum_forcedpars)
                            #
                            # #keeping only the first constant factor frozen if necessary
                            # if i_grp>1 and AllModels(1).componentNames[0]=='constant':
                            #     AllModels(i_grp)(1).frozen=False

                        # no error computation to avoid humongus computation times
                        calc_fit(nonew=True, noprint=True)

                        for i_grp in range(1, AllData.nGroups + 1):
                            # freezing the model again since we want to fit only specific parameters afterwards
                            freeze(AllModels(i_grp))

                        '''
                        Now we search for residual lines. We use an energy grid steppar with free normalisation set at 0 
                        The steppar will fit the free parameter at each energy for us
                        '''

                        # adding a narrow gaussian
                        mod_fake = addcomp('nagaussian')

                        # computing a steppar for each element of the list
                        Xset.chatter = 0
                        Xset.logChatter = 0

                        for steppar_inter in steppar_ind_inter:

                            # giving the width value of the corresponding line before computing the steppar
                            AllModels(1)(AllModels(1).nParameters - 1).values = [sign_widths_arr[0]] + AllModels(1)(
                                AllModels(1).nParameters - 1).values[1:]

                            # exploring the parameters
                            try:
                                Fit.steppar('nolog ' + str(mod_fake.nParameters - 2) + ' ' + str(
                                    round(line_search_e_space[steppar_inter[0]], 3)) + \
                                            ' ' + str(round(line_search_e_space[steppar_inter[1]], 3)) + ' ' \
                                            + str(steppar_inter[1] - steppar_inter[0]))

                                # updating the delchi array with the part of the parameters that got updated
                                delchi_arr_fake[f_ind][steppar_inter[0]:steppar_inter[1] + 1] = \
                                    abs(np.array([min(elem, 0) for elem in Fit.stepparResults('delstat')]))
                            except:
                                # can happen if there are issues in the data quality, we just don't consider the fakes then
                                pass

                        Xset.chatter = xchatter
                        Xset.logChatter = 5

                        pbar.update(1)

            # assessing the significance of each line
            for i_line in range(len(abslines_sign)):



                '''
                Now we just compute the indexes corresponding to the lower and upper bound of each line's interval and compute the 
                probability from this space only (via a transposition)
                '''

                # skipping the computation for lines above 8 keV in restrict mode when we don't go above 8keV anyway
                if restrict_graded and i_line >= 2:
                    continue

                # here we skip the first two emission lines
                line_name = list(lines_e_dict.keys())[i_line + 3]

                # fetching the lower and upper bounds of the energies from the blueshifts
                line_lower_e = lines_e_dict[line_name][0] * (1 + lines_e_dict[line_name][1] / c_light)
                line_upper_e = lines_e_dict[line_name][0] * (1 + lines_e_dict[line_name][2] / c_light)

                # computing the corresponding indexes in the delchi array
                line_lower_ind = int((line_lower_e - line_search_e_space[0]) // line_search_e[2])
                line_upper_ind = int((line_upper_e - line_search_e_space[0]) // line_search_e[2] + 1)

                # remaking delchi_arr_fake_line if it wasn't loaded
                if not loaded_fakes:
                    # restricting the array to those indexes
                    # we use max evaluation here because it could potentially lead to underestimating the significance
                    # if more than 1 delchi element in an iteration are above the chi threshold
                    delchi_arr_fake_line[i_line] = delchi_arr_fake.T[line_lower_ind:line_upper_ind + 1].T.max(1)

                # we round to keep the precision to a logical value
                # we also add a condition to keep the significance at 0 when there's no line in order to avoid problems
                abslines_sign[i_line] = round(1 - len(
                    delchi_arr_fake_line[i_line][delchi_arr_fake_line[i_line] > abslines_delchi[i_line]]) / nfakes,
                                              len(str(nfakes))) if abslines_delchi[i_line] != 0 else 0

                # giving the line a significance attribute
                line_comp_list = [comp for comp in [elem for elem in fitlines.complist if elem is not None] if
                             line_name in comp.compname]

                if len(line_comp_list)==0:
                    continue
                else:
                    assert len(line_comp_list)==1,'Issue: should only be one matching line'
                    line_comp_list[0].significant = abslines_sign[i_line]

                # '''
                # computing the UL for detectability at the given threshold
                # Here we convert the delchi threshold for significance to the EW that we would obtain for a line at the maximum blueshift
                # '''

        # reloading the initial spectra for any following computations
        Xset.restore(outdir + '/' + epoch_observ[0] + '_mod_autofit.xcm')

        # and saving the delchi array if it wasn't already done previously
        if assess_line and not loaded_fakes:
            np.save(outdir + '/' + epoch_observ[0] + '_delchi_arr_fake_line.npy', delchi_arr_fake_line)

        '''
        ####Line fit upper limits
        '''

        # reloading the autofit model with no absorption to compute the upper limits
        data_autofit_noabs.load()

        # freezing the model to avoid it being affected by the missing absorption lines
        # note : it would be better to let it free when no absorption lines are there but we keep the same procedure for
        # consistency
        allfreeze()

        # computing a mask for significant lines
        mask_abslines_sign = abslines_sign >= sign_threshold

        if assess_line_upper:
            # computing the upper limits for the non significant lines
            abslines_eqw_upper = fitlines.get_eqwidth_uls(mask_abslines_sign, abslines_bshift, sign_widths_arr,
                                                          pre_delete=True)

        # here will need to reload an accurate model before updating the fitcomps
        '''HTML TABLE FOR the pdf summary'''

        abslines_table_str = html_table_maker()

        with open(outdir + '/' + epoch_observ[0] + '_abslines_table.txt', 'w+') as abslines_table_file:
            abslines_table_file.write(abslines_table_str)

        def latex_table_maker():

            '''
            to be done
            '''

            def latex_value_maker(value_arr, is_shift=False):

                '''
                wrapper for making a latex-proof of the line abs values

                set is_shift to true for energy/blueshift values, for which 0 values or low uncertainties equal to the value
                are sign of being pegged to the blueshift limit
                '''

                # the first case is for eqwidths and blueshifts (argument is an array with the uncertainties)
                if type(value_arr) == np.ndarray:
                    # If the value array is entirely empty, it means the line is not detected and thus we put a different string
                    if len(np.nonzero(value_arr)[0]) == 0:
                        newstr = '/'
                    else:
                        if is_shift == True and value_arr[1] == 'linked':
                            # we do not show uncertainties for the linked parameters since it is just a repeat
                            newstr = str(round(value_arr[0], 2))

                        else:
                            newstr = str(round(value_arr[0], 2)) + ' -' + str(round(value_arr[1], 2)) + ' +' + str(
                                round(value_arr[2], 2))

                # the second case is for the significance, which is a float
                else:
                    # same empty test
                    if value_arr == 0:
                        newstr = '/'
                    else:
                        newstr = (str(round(100 * value_arr, len(str(nfakes)))) if value_arr != 1 else '>' + str(
                            (1 - 1 / nfakes) * 100)) + '%'

                return newstr

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

    else:
        autofit_store_str = epoch_observ[0] + '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + '\n'

    '''Storing the results'''

    #### result storing

    # we test for the low SNR flag to ensure not overwriting line results for good quality data by mistake if launching the script without autofit
    if autofit or flag_lowSNR_line:
        autofit_store_header = 'Observ_id\tabslines_eqw\tabslines_bshift\tablines_delchi\tabslines_flux\t' + \
                               'abslines_sign\tabslines_eqw_upper\tabslines_em_overlap\tabslines_width\tabslines_bshift_distinct' + \
                               '\tautofit_parerrors\tautofit_parnames\n'

        file_edit(path=autofit_store_path, line_id=epoch_observ[0], line_data=autofit_store_str,
                  header=autofit_store_header)

    if len(cont_peak_points) != 0:
        line_str = epoch_observ[0] + '\t' + str(cont_peak_points.T[0].tolist()) + '\t' + str(
            cont_peak_points.T[1].tolist()) + '\t' + \
                   str(cont_peak_widths.tolist()) + '\t' + str(cont_peak_delchis) + '\t' + str(cont_peak_eqws) + '\t' + \
                   '\t' + str(main_spflux.tolist()) + '\n'
    else:
        line_str = epoch_observ[0] + '\t' + '\t' + '\t' + '\t' + '\t' + '\t' + str(main_spflux.tolist()) + '\n'

    line_store_header = 'Observ_id\tpeak_e\tpeak_norm\tpeak_widths\tpeak_delchis\tpeak_eqwidths\tbroad_flux\n'

    file_edit(path=line_store_path, line_id=epoch_observ[0], line_data=line_str, header=line_store_header)

    '''PDF creation'''

    if write_pdf:
        pdf_summary(epoch_files,arg_dict=arg_dict,fit_ok=True, summary_epoch=fill_result('Line detection complete.'),
                    e_sat_low_list=e_sat_low_indiv, e_sat_high_list=e_sat_high_indiv)

    # closing the logfile for both access and Xspec
    curr_logfile.close()
    Xset.closeLog()

    return fill_result('Line detection complete.')

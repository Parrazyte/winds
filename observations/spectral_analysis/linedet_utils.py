import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator

import sys
from tqdm import tqdm
from copy import deepcopy
from matplotlib.gridspec import GridSpec

from general_tools import ravel_ragged,MinorSymLogLocator

#fitting imports
from xspec import AllData,Plot,AllModels,Fit,Xset

from xspec_config_multisp import xcolors_grp,addcomp,store_plot,allfreeze,xscorpeon,\
                                model_load,xspec_globcomps,allmodel_data
from fitting_tools import lines_std,lines_e_dict, lines_std_names

#bipolar colormap from a custom library (https://github.com/endolith/bipolar-colormap)
#in the general tools folder
from bipolar import hotcold

'''peak detection'''

#2.6.6 version doesn't work, and 2.5 is out of date with newer scipy structure so have to modify
#findpeaks to put from scipy.fft import fft, ifft instead of from from scipy import fft, ifft
from findpeaks import findpeaks
#mask to polygon conversion
from imantics import Mask
from shapely.geometry import Polygon,Point
#mask propagation for the peak detection
from scipy.ndimage import binary_dilation
import dill
import time

def narrow_line_cycle(low_e,high_e,e_step=2e-2,plot_suffix='',baseload=None,
                      e_sat_low_indiv='auto',force_ylog_ratio=False,
                      ratio_bounds=None,title=True):

    '''
    Simple wrapper to compute a line search and make an associated plot

    baseload should be an XCM file with both the file and model

    e_sat_low_indiv here assumes that a RESOLVE and EXTEND spectra are loaded

    '''

    prev_chatter=Xset.chatter
    Xset.chatter=1

    if type(e_sat_low_indiv)==str and e_sat_low_indiv=='auto':
        ngroups=AllData.nGroups
        Plot('data')
        e_sat_low_indiv_use=[Plot.x(i_grp)[0] for i_grp in range(1,ngroups+1)]

    else:
        e_sat_low_indiv_use=e_sat_low_indiv


    mod_ls=allmodel_data()
    narrow_out_val=narrow_line_search(mod_ls,'mod_ls',e_sat_low_indiv_use,[low_e,high_e,e_step],
                                      line_cont_range=[low_e,high_e],scorpeon=False)

    if baseload is not None:
        baseload_str=baseload[:baseload.rfind('.')]
    else:
        baseload_str=str(time.time()).split('.')[0]
    with open(baseload_str+'_narrow_out_'+str(low_e)+'_'+str(high_e)+'.dill','wb+') as f:
        dill.dump(narrow_out_val,f)

    plot_line_search(narrow_out_val, './', 'XRISM', suffix='', save=True,
                     epoch_observ=[plot_suffix], format='pdf',
                     force_ylog_ratio=force_ylog_ratio,ratio_bounds=ratio_bounds,title=title)

    Xset.chatter=prev_chatter

def narrow_line_search(data_cont, suffix,e_sat_low_indiv,line_search_e=[4,10,0.05],line_search_norm=[0.01,10,500],
                       lw=0.,peak_thresh=9.21,peak_clean=False,line_cont_range=[4,10],trig_interval=[6.5,9.1],
                       scorpeon_save=None,data_fluxcont=None,scorpeon=True):

    '''
    Wrapper for all the line search code and associated visualisation

    Explores the current model in a given range by adding a line of varying normalisation and energy and mapping the associated
    2D delchi map

    can use datafluxcont to compute the normalisation from the continuum of another spectrum
    useful when negative gaussians at 0 width that go below zero in flux (which you don't care about normally
    because it is diluted by the instrumental response, but it crashes the log flux)

    Note: because of indexing the peak values are always shifted to 1 index lower (so 1 energy step lower)

    arg:
        -scorpeon. Tries to load the scorpeon background, checks all fits files and
         will fail if there was a change in folder after loading. Setting it to False desactivates the load.
    '''

    line_search_e_space = np.arange(line_search_e[0], line_search_e[1] + line_search_e[2] / 2, line_search_e[2])
    # this one is here to avoid adding one point if incorrect roundings create problem
    line_search_e_space = line_search_e_space[line_search_e_space <= line_search_e[1]+1e-5]

    norm_par_space = np.concatenate((-np.logspace(np.log10(line_search_norm[1]), np.log10(line_search_norm[0]),
                                                  int(line_search_norm[2] / 2)), np.array([0]),
                                     np.logspace(np.log10(line_search_norm[0]), np.log10(line_search_norm[1]),
                                                 int(line_search_norm[2] / 2))))
    norm_nsteps = len(norm_par_space)

    '''
    Computing the local flux for every step of the energy space :
    This will be used to adapt the line energy to the continuum and avoid searching in wrong norm spaces
    We store the flux of the continuum for a width of one step of the energy space, around each step value

    Note : We do this for the first spectrum only even with multi data groups
    '''

    if data_fluxcont is None:
        data_cont.load()
    else:
        data_fluxcont.load()

    flux_cont = np.zeros(len(line_search_e_space))

    #n_models=len(AllModels.sources.keys())

    # here to avoid filling the display with information we're already storing
    # with redirect_stdout(open(os.devnull, 'w')):

    for ind_e, energy in enumerate(line_search_e_space):
        AllModels.calcFlux(str(energy - line_search_e[2] / 2) + " " + str(energy + line_search_e[2] / 2))

        #summing on all the models including the background because here this normalization will be added to the whole continuum (also with the BG models)
        #NOTE: doesn't work because the models are adapted to serve as background
        #flux_cont[ind_e] = sum([AllData(1).flux[6*i] for i in range(n_models)])

        flux_cont[ind_e] = AllData(1).flux[0]

        # this is required because the buffer is different when redirected
        sys.stdout.flush()

    # creation of the eqwidth conversion variable
    eqwidth_conv = np.zeros(len(line_search_e_space))

    # defining the chi array for each epoch
    chi_arr = np.zeros((len(line_search_e_space), norm_nsteps))

    # reseting the model
    AllModels.clear()
    if scorpeon:
        xscorpeon.load('auto',scorpeon_save=scorpeon_save, frozen=True)

    # reloading the broad band model
    data_cont.load()

    plot_ratio_values = store_plot('ratio')

    chi_base = Fit.statistic

    # adding the gaussian with constant factors and cflux for variations
    # since we cannot use negative logspaces in steppar, we use one constant factor for the sign and a second for the value
    addcomp('constant(constant(cflux(gauss)))',
            position='lastin' if AllModels(1).componentNames[0] in xspec_globcomps else 'last')

    mod_gauss = AllModels(1)

    # freezing everything but the second constant factor to avoid problems during steppar
    allfreeze()

    # since there might be other components with identical names in the model, we retrieve each of the added xspec components
    # from their position as the last components in the component list:
    comp_cfactor_1, comp_cfactor_2, comp_gauss_cflux, comp_gauss = [getattr(mod_gauss, mod_gauss.componentNames[-4 + i])
                                                                    for i in range(4)]

    # unlocking negative constant factors for the first one
    comp_cfactor_1.factor.values = [1.0, -0.01, -1e6, -1e6, 1e6, 1e6]

    # getting the constant factor index and unfreezing it
    index_cfactor_2 = comp_cfactor_2.factor.index
    comp_cfactor_2.factor.frozen = 0

    # adjusting the cflux to be sure we cover the entire flux of the gaussian component
    comp_gauss_cflux.Emin = str(min(e_sat_low_indiv))
    comp_gauss_cflux.Emax = 12.

    # narrow line locked
    comp_gauss.Sigma = lw
    comp_gauss.Sigma.frozen = 1

    ####TODO: test if this affects things
    #Fit.steppar='on'

    # tqdm creates a progress bar display:
    with tqdm(total=len(line_search_e_space)) as pbar:

        for j, energy in enumerate(line_search_e_space):
            # exploring the parameter space for energy
            comp_gauss.LineE = energy

            # resetting the second constant factor value
            comp_cfactor_2.factor = 1

            '''
            getting the equivalent width conversion for every energy
            careful: this only gives the eqwidth of the unabsorbed line

            for that we set the gaussian cflux to the continuum flux at this energy 
            since norm_par_space is directly in units of local continuum flux it will make it much easier to get all the eqwidths afterwards)
            '''
            comp_gauss_cflux.lg10Flux = np.log10(flux_cont[j])

            # Computing the eqwidth of a component works even with the cflux dependance.
            AllModels.eqwidth(len(AllModels(1).componentNames))

            # conversion in eV from keV included since the result is in keV
            eqwidth_conv[j] = AllData(1).eqwidth[0] * 10 ** 3

            '''
            exploring the norm parameter space in units of the continuum flux at this energy
            In order to do that, we add 2 steppar computations (for positive and negative norms) where we vary the constant factor
            in a similar manner to the norm par space
            '''

            # turning off the chatter to avoid spamming the console
            prev_xchatter = Xset.chatter

            Xset.chatter = 0

            # first steppar in negative norm space
            # -1 in the number of computations because steppar adds 1
            comp_cfactor_1.factor = -1
            Fit.steppar(
                'log ' + str(index_cfactor_2) + ' ' + str(line_search_norm[1]) + ' ' + str(line_search_norm[0]) + \
                ' ' + str(int(line_search_norm[2] / 2) - 1))

            negchi_arr = np.array(Fit.stepparResults('statistic'))

            # second steppar in positive norm space
            comp_cfactor_1.factor = 1
            Fit.steppar(
                'log ' + str(index_cfactor_2) + ' ' + str(line_search_norm[0]) + ' ' + str(line_search_norm[1]) + \
                ' ' + str(int(line_search_norm[2] / 2) - 1))

            poschi_arr = np.array(Fit.stepparResults('statistic'))

            # returning the chatter to the previous value
            Xset.chatter = prev_xchatter

            chi_arr[j] = np.concatenate((negchi_arr, np.array([chi_base]), poschi_arr))

            pbar.update(1)

    # to compute the contour chi, we start from a chi with a fit with a line normalisation of 0
    chi_contours = [chi_base - 9.21, chi_base - 4.61, chi_base - 2.3]

    # unused for now
    # #computing the negative (i.e. improvement) part of the delchi map for the autofit return
    # chi_arr_impr=np.where(chi_arr>=chi_base,0,abs(chi_base-chi_arr))

    '''Peak computation'''

    def peak_search(array_arg):

        # safeguard in case of extreme peaks dwarfing the other peaks
        if np.max(array_arg) >= 2e2:
            array = np.where(array_arg >= 1, array_arg ** (1 / 2), array_arg)
        else:
            array = array_arg

        # choosing a method
        peak_finder = findpeaks(method='topology', whitelist='peak', denoise=None)
        peak_finder.fit(array)
        peak_result = peak_finder.results['persistence']

        # cutting the peaks for which the birth level (is zero) to avoid false detections
        peak_validity = peak_result['score'] > 0
        peak_result = np.array(peak_result)
        peak_result = peak_result[peak_validity]
        peak_result = peak_result.T[:2].astype(int)[::-1].T

        # computing the polygon points of the peak regions
        # since the polygons created from the mask are "inside" the edge of ther polygon, this can cause problems for thin polygons
        # We thus do one iteration of binary dilation to expand the mask by one pixel.
        # This is equivalent to having the polygon cover the exterior edges of the pixels.
        peak_result_points = Mask(binary_dilation(array_arg)).polygons().points

        return peak_result, peak_result_points

    # The library is very good at finding peaks but garbage at finding valleys, so we cut what's below
    # the chi difference limit and swap the valleys into peaks

    chi_arr_sub_thresh = np.where(chi_arr >= chi_base - peak_thresh, 0, abs(chi_base - chi_arr))
    peak_points_raw, peak_polygons_points = peak_search(chi_arr_sub_thresh)

    peak_points = []
    # limiting to a single peak per energy step by selecting the peak with the lowest chi squared for each energy bin
    for peak_e in np.unique(peak_points_raw.T[0]):

        chi_peak = chi_base

        for peak_norm in peak_points_raw.T[1][peak_points_raw.T[0] == peak_e]:
            if chi_arr[peak_e][peak_norm] < chi_peak:
                chi_peak = chi_arr[peak_e][peak_norm]
                maxpeak_pos = [peak_e, peak_norm]
        peak_points.append(maxpeak_pos)

    peak_points = np.array(peak_points)

    if peak_polygons_points != []:

        # making sure the dimension is correct even if a single polygon is detected
        if type(peak_polygons_points[0][0]) == np.int64:
            peak_polygons_points = np.array(peak_polygons_points)

        if len(peak_polygons_points) != len(peak_points):
            print('\nThe number of peak and polygons is not identical.')

            if peak_clean:
                print('\nRefining...')
                # we refine by progressively deleting the remaining elements of the array until the shape splits
                chi_arr_refine = chi_arr_sub_thresh.copy()
                chi_refine_values = chi_arr_sub_thresh[chi_arr_sub_thresh.nonzero()]
                chi_refine_values.sort()
                index_refine = -1
                peak_eq_pol = False
                peak_points_ref = peak_points.copy()

                # the failure stop condition is when a peak gets deleted, which only happens if the refining was too strict
                while not peak_eq_pol and len(peak_points_ref) == len(peak_points):
                    index_refine += 1

                    # refining the chi array
                    chi_arr_refine = np.where(chi_arr_refine > chi_refine_values[index_refine], chi_arr_refine, 0)

                    # recomputing the peaks
                    peak_points_ref, peak_polygons_points_ref = peak_search(chi_arr_refine)

                    # testing if the process worked. We use peak_points instead of peak_points_ref to avoid soliving deleting peaks
                    peak_eq_pol = len(peak_points) == len(peak_polygons_points_ref)

                # We only replace the polygons if the refining did work
                if len(peak_points_ref) == len(peak_points):
                    peak_polygons_points = peak_polygons_points_ref

        # creating a list of the equivalent shapely polygons and peaks
        peak_polygons = np.array([None] * min(len(peak_points), len(peak_polygons_points)))
        peak_points_shapely = np.array([None] * min(len(peak_points), len(peak_polygons_points)))

        for i_poly, elem_poly in enumerate(peak_polygons_points):

            if i_poly >= len(peak_points):
                continue

            peak_polygons[i_poly] = Polygon(elem_poly)
            peak_points_shapely[i_poly] = Point(peak_points[i_poly][::-1])

        # linking each peak to its associated polygon and storing its width
        peak_widths = np.zeros(len(peak_points))

        for elem_point in enumerate(peak_points_shapely):
            for elem_poly in peak_polygons:
                if elem_poly.contains(elem_point[1]):
                    # we substract 1 from the width since the bouding box considers one more pixel as in
                    peak_widths[elem_point[0]] = elem_poly.bounds[3] - elem_poly.bounds[1] - 1
    else:
        peak_widths = []

    # storing the chi² differences of the peaks
    peak_delchis = []
    peak_eqws = []
    if len(peak_points) != 0:
        for coords in peak_points:
            peak_delchis.append(chi_base - chi_arr[coords[0]][coords[1]])

            # since the stored eqwidth is for the continuum flux, multiplying it by norm_par_space and the step size ratio to 0.1
            # directly gives us all the eqwidths for an energy, since they scale linearly with the norm/cflux
            peak_eqws.append(eqwidth_conv[coords[0]] * norm_par_space[coords[1]])

    if len(peak_points) > 0:
        is_abspeak = ((np.array(peak_eqws) < 0) & (line_search_e_space[peak_points.T[0]] >= trig_interval[0]) & \
                      (line_search_e_space[peak_points.T[0]] <= trig_interval[1])).any()
    else:
        is_abspeak = False

    '''''''''''''''''
    ######PLOTS######
    '''''''''''''''''

    # creating some necessary elements

    # line threshold is the threshold for the symlog axis
    chi_dict_plot = {
        'chi_arr': chi_arr,
        'chi_base': chi_base,
        'line_threshold': line_search_norm[0],
        'line_search_e': line_search_e,
        'line_search_norm': line_search_norm,
        'line_search_e_space': line_search_e_space,
        'norm_par_space': norm_par_space,
        'peak_points': peak_points,
        'peak_widths': peak_widths,
        'line_cont_range': line_cont_range,
        'plot_ratio_values': plot_ratio_values,
    }

    if suffix == 'cont':
        return is_abspeak, peak_points, peak_widths, peak_delchis, peak_eqws, chi_dict_plot
    else:
        return chi_dict_plot

def plot_line_search(chi_dict_plot,outdir,sat,save=True,suffix=None,epoch_observ=None,format='png',
                     force_ylog_ratio=False,ratio_bounds=None,ener_show_range=None,show_indiv_transi=False,
                     title=True,squished_mode=False,local_chi_bounds=False,force_side_lines='none',
                     minor_locator=False,show_peak_pos=True,show_peak_width=True):

    line_search_e=chi_dict_plot['line_search_e']
    line_search_norm=chi_dict_plot['line_search_norm']
    n_groups=len(chi_dict_plot['plot_ratio_values'])
    #doing this to keep the same syntax as before on the file creation
    line_search_e_str='_'.join([str(int(elem) if int(elem)==elem else elem) for elem in line_search_e])
    line_search_norm_str='_'.join([str(int(elem) if int(elem)==elem else elem) for elem in line_search_norm])

    #back to the main computation
    curr_plot_bg_state = Plot.background

    # for the line plots we don't need the background
    Plot.background = False

    comb_title = r' Blind search visualisation for observ ' + ('' if epoch_observ is None else epoch_observ[0]) +\
                 '\n with line par ' + line_search_e_str + \
                 ' and norm par ' + line_search_norm_str + ' in continuum units'

    comb_label = []
    for i_grp in range(n_groups):
        if sat == 'Chandra':
            label_grating = str(-1 + 2 * i_grp)
        else:
            label_grating = ''

        comb_label += ['' if epoch_observ is None else ('_'.join(epoch_observ[i_grp].split('_')[1:3])) if sat == 'XMM'
                        else epoch_observ[0] + ' order ' + label_grating if sat == 'Chandra' else '']

    # creating the figure
    figure_comb = plt.figure(figsize=(15, 10*(0.5 if squished_mode else 1)))

    comb_chi2map(figure_comb, chi_dict_plot, title=comb_title if title else '', comb_label=comb_label,
                 force_ylog_ratio=force_ylog_ratio,ratio_bounds=ratio_bounds,ener_show_range=ener_show_range,
                 show_indiv_transi=show_indiv_transi,squished_mode=squished_mode,local_chi_bounds=local_chi_bounds,
                 force_side_lines=force_side_lines,minor_locator=minor_locator,
                 show_peak_pos=show_peak_pos,show_peak_width=show_peak_width)

    if save==True:
        # saving it and closing it
        plt.savefig(os.path.join(outdir,epoch_observ[0] + '_' + suffix + '_line_comb_plot_' +\
                    line_search_e_str.replace(' ','_') + '_' + line_search_norm_str.replace(' ', '_') + '.'+format))
        plt.close(figure_comb)

    # putting the background plotting to its previous state
    Plot.background = curr_plot_bg_state

def plot_line_ratio(axe,data_autofit,data_autofit_noabs,n_addcomps_cont,mode=None,
                    line_position=None,line_search_e=[4,10,0.05],line_cont_range=[4,10],plot_ener=True):

    '''
    plots the line ratio highlighting absorption components in the model
    '''

    # recreating the line search space
    line_search_e_space = np.arange(line_search_e[0], line_search_e[1] + line_search_e[2] / 2, line_search_e[2])
    # this one is here to avoid adding one point if incorrect roundings create problem
    line_search_e_space = line_search_e_space[line_search_e_space <= line_search_e[1]]

    '''
    storing the different ratio and creating the main variables
    '''

    # storing the no abs line 'continuum' model
    model_load(data_autofit_noabs)
    plot_ratio_autofit_noabs = store_plot('ratio')

    model_load(data_autofit)
    # storing the components of the model for the first data group only
    plot_autofit_comps = store_plot('ldata', comps=True)[1][0]

    # rearranging the components in a format usable in the plot. The components start at the index 2
    # (before it's the entire model x and y values)
    plot_autofit_cont = plot_autofit_comps[:2 + n_addcomps_cont]

    # same for the line components
    plot_autofit_lines = plot_autofit_comps[2 + n_addcomps_cont:]

    # taking off potential background components
    if 'nxb' in list(AllModels.sources.values()):
        plot_autofit_lines = plot_autofit_lines[:-2]

    if 'sky' in list(AllModels.sources.values()):
        plot_autofit_lines = plot_autofit_lines[:-2]

    #creating a sum of all the non absorption components
    plot_autofit_noabs = np.concatenate((([[plot_addline] for plot_addline in plot_autofit_comps[2:] \
                                           if not max(plot_addline) <= 0]))).sum(axis=0)

    #and creating a ratio of the lines addition to it
    plot_autofit_ratio_lines = [(plot_autofit_noabs + plot_autofit_lines[i]) / plot_autofit_noabs \
                                for i in range(len(plot_autofit_lines)) if max(plot_autofit_lines[i]) <= 0.]


    '''second plot (ratio + abslines ratio)'''

    axe.set_xlabel('Energy (keV)')
    axe.xaxis.set_label_position('bottom')

    # hiding the ticks values for the lower x axis if in paper mode
    if mode == 'paper':
        plt.setp(axe.get_xticklabels(), visible=False)

    # changing the axis for when in paper mode
    axe.set_ylabel('Fit ratio' if mode == 'paper' else 'Fit ratio compared the sum of continuum and all emission lines')
    axe.set_xlim(line_cont_range)

    # we put the x axis on top to avoid it being hidden by the second subplot2aaa
    axe.xaxis.tick_bottom()
    axe.xaxis.set_label_position('bottom')
    for i_grp in range(AllData.nGroups):
        axe.errorbar(plot_ratio_autofit_noabs[i_grp][0][0], plot_ratio_autofit_noabs[i_grp][1][0],
                     xerr=plot_ratio_autofit_noabs[i_grp][0][1], yerr=plot_ratio_autofit_noabs[i_grp][1][1],
                     color=xcolors_grp[i_grp], ecolor=xcolors_grp[i_grp], linestyle='None', alpha=0.7)
    axe.axhline(y=1, xmin=0, xmax=1, color='green')

    # limiting the plot to the range of the line energy search
    axe.set_xlim(line_search_e_space[0], line_search_e_space[-1])

    plot_ratio_xind_rel = [
        np.array([elem for elem in np.where(plot_ratio_autofit_noabs[i_grp][0][0] >= line_search_e[0])[0] \
                  if elem in np.where(plot_ratio_autofit_noabs[i_grp][0][0] <= line_search_e_space[-1])[0]]) \
        for i_grp in range(AllData.nGroups)]

    '''
    rescaling with errorbars (which are not taken into account by normal rescaling)
    we use 1 as a default value if nothing is in the correct range for a given datagroup so that it doesn't 
    affect the rest
    '''
    plot_ratio_y_up = np.array(
        [1 if len(plot_ratio_xind_rel[i_grp])==0 else (plot_ratio_autofit_noabs[i_grp][1][0] + plot_ratio_autofit_noabs[i_grp][1][1])[plot_ratio_xind_rel[i_grp]]
         for i_grp in range(AllData.nGroups)], dtype=object)

    plot_ratio_y_dn = np.array(
        [1 if len(plot_ratio_xind_rel[i_grp])==0 else (plot_ratio_autofit_noabs[i_grp][1][0] - plot_ratio_autofit_noabs[i_grp][1][1])[plot_ratio_xind_rel[i_grp]]
         for i_grp in range(AllData.nGroups)], dtype=object)
    axe.set_ylim(0.95 * np.min(ravel_ragged(plot_ratio_y_dn)), 1.05 * np.max(ravel_ragged(plot_ratio_y_up)))

    # linestyles
    l_styles = ['solid', 'dotted', 'dashed', 'dashdot']

    # plotting the delta ratio of the absorption components
    for i_line, ratio_line in enumerate(plot_autofit_ratio_lines):
        # fetching the position of the line compared to other line components to get identical alpha and ls values

        if line_position is not None:
            i_line_comp=line_position[i_line]
        else:
            i_line_comp=i_line

        # plotting each ratio when it is significantly different from the continuum,
        # and with the same color coding as the component plot above
        axe.plot(plot_autofit_cont[0][ratio_line <= 1 - 1e-3], ratio_line[ratio_line <= 1 - 1e-3], color='red',
                 alpha=1 - i_line_comp * 0.1, linestyle=l_styles[i_line_comp % 4])

    if plot_ener:
        '''Plotting the Standard absorption line energies'''
        plot_std_ener(axe)


def plot_std_ener(ax_ratio, ax_contour=None, plot_em=False, mode='ratio',exclude_last=False,plot_indiv_transi=False,
                  squished_mode=False,force_side='none'):
    '''
    Plots the current absorption (and emission if asked) standard lines in the current axis
    also used in the autofit plots further down

    -plot_indiv_transi:
            -True/False: plots only/none of the individual transitions for instead of the averaged energies
            -prio_resolved: plots the resolved transitions when available, otherwise the non-resolved
            -only[X+Y+...]: only plots the resolved lines for X, Y, ...
    -squished mode: makes the absorption text slightly higher to avoid it going lower than the plot

    -force_side: shows all lines either in emission if 'em' or in absorption if 'abs'
    '''
    # since for the first plot the 1. ratio is not necessarily centered, we need to fetch the absolute position of the y=1.0 line
    # in graph height fraction
    pos_ctr_ratio = 0.5 if mode=='chimap' else\
                    (1 - ax_ratio.get_ylim()[0]) / (ax_ratio.get_ylim()[1] - ax_ratio.get_ylim()[0])

    lines_names = np.array(lines_std_names)


    lines_abs_pos = ['abs' in elem for elem in lines_names]
    lines_em_pos = ['em' in elem for elem in lines_names]

    indiv_lines_nonresolved_std=np.unique([' '.join(elem.split(' ')[:2]) if len(elem.split(' '))<=2
                                    else '' for elem in lines_std.values()])
    indiv_lines_resolved_std=np.unique([' '.join(elem.split(' ')[:2]) if len(elem.split(' '))>2
                                    else '' for elem in lines_std.values()])

    indiv_lines_both_std=[elem for elem in indiv_lines_resolved_std if elem in indiv_lines_nonresolved_std]

    indiv_lines_both=[elem for elem in lines_names if lines_std[elem] in indiv_lines_both_std]

    #removing or restricting to resolved lines for the emission
    lines_resolved_mask=['(' in lines_std[elem] and ')' in lines_std[elem] for elem in lines_names]
    lines_resolved=lines_names[lines_resolved_mask]

    if type(plot_indiv_transi)==str and 'only' in plot_indiv_transi:
        plot_indiv_transi_lines=plot_indiv_transi.split('only')[1].split('+')
        lines_resolved_restrict=[elem for elem in lines_resolved if
                        sum([elem.startswith(subelem) for subelem in plot_indiv_transi_lines])>0]

    for i_line, line in enumerate(lines_names):

        if lines_e_dict[line][0] < ax_ratio.get_xlim()[0] or lines_e_dict[line][0] > ax_ratio.get_xlim()[1]:
            continue

        # skipping redundant indexes
        if line in ['FeKa25em','FeKa26em','FeKa0em','FeKb0em','calNICERSiem','FeDiazem']:
            continue

        # skipping Nika27, FeKa25em, FeKa26em:
        if i_line in [5,9,10]:
            continue

        # skipping display if emission lines are not asked
        if 'em' in line and not plot_em:
            continue

        if plot_indiv_transi=='prio_resolved':
            if  line in indiv_lines_both and not ('(' in lines_std[line] and ')' in lines_std[line]):
                continue
        elif type(plot_indiv_transi)==str and plot_indiv_transi.startswith('only'):
            if line in lines_resolved and line not in lines_resolved_restrict:
                continue
        else:
            if not 'em' in line and (line not in lines_resolved and plot_indiv_transi==True \
                                 or line in lines_resolved and plot_indiv_transi==False):
                continue



        # booleans for dichotomy in the plot arguments
        abs_bool = 'abs' in line and 'abs' in force_side
        em_bool = not abs_bool or 'em' in force_side

        # plotting the lines on the two parts of the graphs
        ax_ratio.axvline(x=lines_e_dict[line][0],
                         ymin=0 if mode not in ['ratio','chimap'] else pos_ctr_ratio if not abs_bool else 0.,
                         ymax=1 if mode not in ['ratio','chimap'] else pos_ctr_ratio if not em_bool else 1.,
                         color='blue' if em_bool else 'brown',
                         linestyle='dashed', linewidth=1.5,zorder=-1)
        if ax_contour is not None:
            ax_contour.axvline(x=lines_e_dict[line][0], ymin=0.5 if em_bool else 0, ymax=1 if em_bool else 0.5,
                               color='blue' if em_bool else 'brown', linestyle='dashed', linewidth=0.5)

        # small left horizontal shift to help the Nika27 display
        txt_hshift = 0.1 if 'Ni' in line else 0.006 if 'Si' in line else 0
        txt_line=lines_std[line]

        line_x_text=lines_e_dict[line][0] - txt_hshift
        if mode != 'noname':

            add_height_squished=0
            if plot_indiv_transi:
                line_full_name=lines_std[line]

                #ensuring the line is an individual transition
                if '(' in line_full_name and ')' in line_full_name:

                    #additional tester to move line_x_tex if none of the conditions below are true
                    bool_shift=True

                    #shifting it to the left if is the first one of a complex
                    if i_line<len(lines_names)-1 and lines_std[lines_names[i_line+1]].split('(')[0]\
                                                   ==lines_std[lines_names[i_line]].split('(')[0]:
                        line_x_text=lines_e_dict[line][0]-0.05-\
                                    (0.01 if 'P' in lines_std[lines_names[i_line]].split('(')[1] else 0)

                        #avoiding overlap with the NiKa27 complex display
                        if line=='FeKb25p1abs':
                            txt_line = '(' + lines_std[line].split('(')[1]
                            line_x_text+=0.04
                        bool_shift=False
                    #removing everything except the complex name otherwise
                    if i_line>0 and lines_std[lines_names[i_line]].split('(')[0]==\
                                    lines_std[lines_names[i_line-1]].split('(')[0]:
                        line_x_text=lines_e_dict[line][0]+0.01+\
                        (0.01 if 'P' in lines_std[lines_names[i_line]].split('(')[1] else 0)

                        if line!='FeKb25p3abs':
                            txt_line = '(' + lines_std[line].split('(')[1]
                        else:
                            line_x_text+=0.03
                        bool_shift=False
                        add_height_squished=0.01 if  'P' in lines_std[lines_names[i_line]].split('(')[1] else 0

                    if bool_shift:
                        line_x_text+=0.04
                else:
                    line_x_text+=0.04
            else:
                line_x_text += 0.04

            # but the legend on the top part only
            ax_ratio.text(x=line_x_text,
                          y=0.96 if not abs_bool else (0.06+(0.02+add_height_squished if squished_mode else 0)
                                                  if i_line % 2 == 1 else 0.12+(0.01 if squished_mode else 0)),
                          s=txt_line,
                          color='blue' if em_bool else 'brown',
                          transform=ax_ratio.get_xaxis_transform(), ha='center',
                          va='top')


def plot_line_comps(axe, comp_cont, names_cont, comp_lines, names_lines, combined=False):
    '''
    Wrapper for plotting model component contributions
    '''

    axe.set_ylabel(r'normalized counts s$^{-1}$ keV$^{-1}$')
    axe.set_yscale('log')

    # summing the continuum components to have baseline for the line plotting
    cont_sum = np.sum(comp_cont[2:], 0)

    # computing the extremal energy bin widths
    bin_widths = (comp_cont[0][1] - comp_cont[0][0]) / 2, (comp_cont[0][-1] - comp_cont[0][-2]) / 2

    # resizing the x axis range to the line continuum range
    axe.set_xlim(round(comp_cont[0][0] - bin_widths[0], 1), round(comp_cont[0][-1] + bin_widths[1], 1))

    # resizing the y axis range to slightly above the beginning of the continuum and an order of magnitude below the continuum to
    # see the emission lines if they are strong

    axe.set_ylim(5e-1 * min(cont_sum), 1.1 * max(cont_sum))

    if combined:
        axe.xaxis.set_label_position('top')
        axe.xaxis.tick_top()
    else:
        axe.set_xlabel('Energy (keV')

    # continuum colormap
    norm_colors_cont = mpl.colors.Normalize(vmin=1, vmax=len(comp_cont[2:]))

    colors_cont = mpl.cm.ScalarMappable(norm=norm_colors_cont, cmap=mpl.cm.viridis)

    # plotting the continuum components
    for i_cont in range(0, len(comp_cont[2:])):
        axe.plot(comp_cont[0], comp_cont[2 + i_cont], label=names_cont[i_cont], color=colors_cont.to_rgba(i_cont + 1))

    # linestyles
    l_styles = ['solid', 'dotted', 'dashed', 'dashdot']

    # loop for each line
    for i_line in range(len(comp_lines)):

        # fetching the position of the line in the standard line array from their component name
        line_name = names_lines[i_line]

        # selecting the parts of the curve for which the contribution of the line is significant
        # (we put stronger constraints on emission lines to avoid coloring the entirety of the curve)
        sign_bins = (abs(comp_lines[i_line]) >= 1e-3 * cont_sum) if 'em' in line_name else (
                    abs(comp_lines[i_line]) >= 1e-2 * cont_sum)

        # plotting the difference with the continuum when the line has a non zero value, with the appropriate color and name
        axe.plot(comp_cont[0][sign_bins],
                 cont_sum[sign_bins] + comp_lines[i_line][sign_bins],
                 label=lines_std[line_name], color='blue' if 'em' in line_name else 'red', alpha=1 - 0.1 * i_line,
                 linestyle=l_styles[i_line % 4])

        # plotting the strong emission lines by themselves independantly
        if min(comp_lines[i_line]) >= 0 and max(comp_lines[i_line]) >= 5e-1 * min(cont_sum):
            axe.plot(comp_cont[0], comp_lines[i_line], color='blue',
                     alpha=1 - 0.1 * i_line, linestyle=l_styles[i_line % 4])

        axe.legend()


def color_chi2map(fig, axe, chi_map, title='', combined=False, ax_bar=None):
    axe.set_ylabel('Line normalisation iteration')
    axe.set_xlabel('Line energy parameter iteration')

    if combined == False:
        axe.set_title(title)

    if np.max(chi_map) >= 1e3:
        chi_map = chi_map ** (1 / 2)
        bigline_flag = 1
    else:
        bigline_flag = 0

    img = axe.imshow(chi_map, interpolation='none', cmap='plasma', aspect='auto')

    if combined == False:
        colorbar = plt.colorbar(img, ax=axe)
        fig.tight_layout()

    else:
        colorbar = plt.colorbar(img, cax=ax_bar)

    if bigline_flag == 1:
        colorbar.set_label(r'$\sqrt{\Delta C}$')
    else:
        colorbar.set_label(r'$\Delta C$')


def contour_chi2map(fig, axe, chi_dict, title='', combined=False):
    chi_arr = chi_dict['chi_arr']
    chi_base = chi_dict['chi_base']
    line_threshold = chi_dict['line_threshold']
    line_search_e = chi_dict['line_search_e']
    line_search_e_space = chi_dict['line_search_e_space']
    line_search_norm = chi_dict['line_search_norm']
    norm_par_space = chi_dict['norm_par_space']
    peak_points = chi_dict['peak_points']
    peak_widths = chi_dict['peak_widths']

    chi_map = np.where(chi_arr >= chi_base, 0, chi_base - chi_arr)

    axe.set_ylabel('Gaussian line normalisation\n in units of local continuum Flux')
    axe.set_yscale('symlog', linthresh=line_threshold, linscale=0.1)
    if combined == False:
        axe.set_xlabel('Energy (keV)')

    chi_contours = [chi_base - 9.21, chi_base - 4.61, chi_base - 2.3]

    contours_var = axe.contour(line_search_e_space, norm_par_space, chi_map, levels=chi_contours, cmap='plasma')

    contours_var_labels = [r'99% conf. with 2 d.o.f.', r'90% conf. with 2 d.o.f.',
                           r'68% conf. with 2 d.o.f.']

    # avoiding error if there are no contours to plot
    for l in range(len(contours_var_labels)):
        try:
            contours_var.collections[l].set_label(contours_var_labels[l])
        except:
            pass

    contours_base = axe.contour(line_search_e_space, norm_par_space, chi_arr.T, levels=[chi_base + 0.5], colors='black',
                                linewidths=0.5, linestyles='dashed')
    contours_base_labels = [r'base level ($\Delta C=0.5$)']

    # avoiding error if there are no contours to plot
    if float(mpl.__version__.split('.')[0])>=3 and float(mpl.__version__.split('.')[1])>=8:
        contour_looper_base=contours_base.legend_elements()[0]
    else:
        contour_looper_base=contours_base.collections

    for l in range(len(contour_looper_base)):
        plt.plot([],[],label=contours_base_labels[l], colors='black',linewidths=0.5, linestyles='dashed')

    # for each peak and width, the coordinates need to be translated in real energy and norm coordinates
    try:
        for elem_point in enumerate(peak_points):
            point_coords = [line_search_e_space[elem_point[1][0]], norm_par_space[elem_point[1][1]]]

            segment_coords = [point_coords[0] - line_search_e[2] * peak_widths[elem_point[0]] / 2,
                              point_coords[0] + line_search_e[2] * peak_widths[elem_point[0]] / 2], [point_coords[1],
                                                                                                     point_coords[1]]

            axe.scatter(point_coords[0], point_coords[1], marker='X', color='black',
                        label='peak' if elem_point[0] == 0 else None)

            axe.plot(segment_coords[0], segment_coords[1], color='black',
                     label='max peak structure width' if elem_point[0] == 0 else None)

            # ununsed for now
            # arrow_coords_left=[point_coords[0]-line_search_e[2]*peak_widths[elem_point[0]]/2,point_coords[1]]
            # arrow_coords_right=[point_coords[0]+line_search_e[2]*peak_widths[elem_point[0]]/2,point_coords[1]]
            # arrow_coords_del=[line_search_e[2]*peak_widths[elem_point[0]],0]
            # axe.arrow(arrow_coords_left[0],arrow_coords_left[1],arrow_coords_del[0],arrow_coords_del[1],shape='full',
            #           head_width=line_search_e/10,head_length=line_search_e[2]/10,color='black',length_includes_head=True,
            #           label='max peak structure width' if elem_point[0]==0 else None)
            # axe.arrow(arrow_coords_right[0],arrow_coords_right[1],-arrow_coords_del[0],arrow_coords_del[1],
            #           shape='full',head_width=norm_nsteps/100,head_length=line_search_e[2]/10,color='black',fc='black',
            #           length_includes_head=True)
    except:
        pass

    # using a weird class to get correct tickers on the axes since it doesn't work natively
    axe.yaxis.set_minor_locator(MinorSymLogLocator(line_search_norm[0]))

    if combined == False:
        axe.legend()
        fig.tight_layout()
    else:
        axe.legend(loc='right', bbox_to_anchor=(1.25, 0.5))


def coltour_chi2map(fig, axe, chi_dict, title='', combined=False, ax_bar=None, norm=None,
                    squished_mode=False,local_chi_bounds=False,ener_show_range=None,
                    show_peak_pos=True,show_peak_width=True):
    '''
        squished_mode: adds a bunch of line separators in th colormap label to avoid display issues

    '''
    chi_arr = chi_dict['chi_arr']
    chi_base = chi_dict['chi_base']
    line_threshold = chi_dict['line_threshold']
    line_search_e = chi_dict['line_search_e']
    line_search_e_space = chi_dict['line_search_e_space']
    line_search_norm = chi_dict['line_search_norm']
    norm_par_space = chi_dict['norm_par_space']
    peak_points = deepcopy(chi_dict['peak_points'])
    peak_widths = chi_dict['peak_widths']

    chi_map = np.where(chi_arr >= chi_base, 0, chi_base - chi_arr)
    chi_map_full=chi_map
    if ener_show_range is not None:

        #rounds to avoid issues with precision
        ener_show_mask=(line_search_e_space.round(4)>=ener_show_range[0]) &\
                       (line_search_e_space.round(4)<=ener_show_range[1])
        chi_map=chi_map[ener_show_mask]
        chi_arr=chi_arr[ener_show_mask]

        line_search_e_space=line_search_e_space[ener_show_mask]

        if len(peak_points)>0:
            peak_points.T[0]-=np.argwhere(ener_show_mask).T[0][0]
            peak_points=peak_points[(peak_points.T[0]>0) & (peak_points.T[0]<len(line_search_e_space))]


    axe.set_ylabel('Gaussian line normalisation\n in units of local continuum Flux')
    axe.set_yscale('symlog', linthresh=line_threshold, linscale=0.1)

    if combined == False:
        axe.set_xlabel('Energy (keV)')
        axe.set_title(title)

    # hiding the ticks values for the lower x axis if in paper mode
    if combined == 'paper':
        plt.setp(axe.get_xticklabels(), visible=False)

    '''COLOR PLOT'''

    # here we do some more modifications
    chi_arr_plot = chi_map

    #for the norm
    chi_arr_plot_full=chi_map_full.copy()
    
    # swapping the sign of the delchis for the absorption and emission lines in order to display them
    # with both parts of the cmap + using a square root norm for easier visualisation

    for i in range(len(chi_arr_plot)):
        chi_arr_plot[i] = np.concatenate((-(chi_arr_plot[i][:int(len(chi_arr_plot[i]) / 2)]) ** (1 / 2),
                                          (chi_arr_plot[i][int(len(chi_arr_plot[i]) / 2):]) ** (1 / 2)))

    for i in range(len(chi_arr_plot_full)):
        chi_arr_plot_full[i] = np.concatenate((-(chi_arr_plot_full[i][:int(len(chi_arr_plot_full[i]) / 2)]) ** (1 / 2),
                                          (chi_arr_plot_full[i][int(len(chi_arr_plot_full[i]) / 2):]) ** (1 / 2)))

    if np.max(chi_arr_plot) >= 1e3 or (np.max(chi_arr_plot_full) >= 1e3 and not local_chi_bounds):
        chi_arr_plot = chi_arr_plot ** (1 / 2)
        chi_arr_plot_full=chi_arr_plot_full **(1/2)
        bigline_flag = 1
    else:
        bigline_flag = 0
        if norm is not None:
            norm_col = np.array(norm) ** (1 / 2)

    # creating the bipolar cm
    cm_bipolar = hotcold(neutral=1)

    # and the non symetric normalisation
    cm_norm = colors.TwoSlopeNorm(vcenter=0, 
                                  vmin=min(-np.sqrt(9.21),chi_arr_plot.min() if local_chi_bounds
                                          else chi_arr_plot_full.min()) if norm is None else -norm_col[0],
                          vmax=max(chi_arr_plot.max() if local_chi_bounds else
                                   chi_arr_plot_full.max(),np.sqrt(9.21)) if norm is None else norm_col[1])


    #should be tested if necessary
    #cm_norm = colors.TwoSlopeNorm(vcenter=0, vmin=min(-1,chi_arr_plot.min() if norm is None else -norm_col[0]),
    #                         vmax=max(1,chi_arr_plot.max() if norm is None else norm_col[1]))


    # create evenly spaced ticks with different scales in top and bottom
    cm_ticks = np.concatenate((np.linspace((chi_arr_plot.min() if local_chi_bounds else chi_arr_plot_full.min())
                                           if norm is None else -norm_col[0], 0, 6, endpoint=True),
                               np.linspace(0, (chi_arr_plot.max() if local_chi_bounds else chi_arr_plot_full.max())
                                           if norm is None else norm_col[1], 6, endpoint=True)))

    if cm_ticks[0]>-np.sqrt(7):
        cm_ticks=np.array([-np.sqrt(9.21)]+cm_ticks.tolist())

    if cm_ticks[-1]<np.sqrt(7):
        cm_ticks=np.array(cm_ticks.tolist()+[np.sqrt(9.21)])

    # renaming the ticks to positive values only since the negative side is only for the colormap + re-squaring to get the
    # right values
    cm_ticklabels = (cm_ticks ** 2).round(1).astype(str)

    # this allows to superpose the image to a log scale (imshow scals with pixels so it doesn't work)
    img = axe.pcolormesh(line_search_e_space, norm_par_space, chi_map.T, norm=cm_norm, cmap=cm_bipolar.reversed())

    if ax_bar != None:

        if ax_bar == 'bottom':
            colorbar = plt.colorbar(img, location='bottom', orientation='horizontal', spacing='proportional',
                                    ticks=cm_ticks,aspect=50)
            colorbar.ax.set_xticklabels(cm_ticklabels)
        elif combined == False:
            colorbar = plt.colorbar(img, ax=axe, spacing='proportional', ticks=cm_ticks)
            colorbar.ax.set_yticklabels(cm_ticklabels)
        else:
            colorbar = plt.colorbar(img, cax=ax_bar, spacing='proportional', ticks=cm_ticks)
            colorbar.ax.set_yticklabels(cm_ticklabels)

        if bigline_flag == 1:
            colorbar.set_label(r'$\sqrt{\Delta C}$'+('\n' if squished_mode else ' ')
                               +'with separated scales'+('\n' if squished_mode else ' ')+'for absorption and emission')
        else:
            colorbar.set_label(r'$\Delta C$'+('\n' if squished_mode else ' ')+
                               'with separated scales'+('\n' if squished_mode else '')+'for absorption and emission')

    '''CONTOUR PLOT'''

    chi_contours = [chi_base - 9.21, chi_base - 4.61, chi_base - 2.3]

    contours_var_labels = [r'99% conf. with 2 d.o.f.', r'90% conf. with 2 d.o.f.',
                           r'68% conf. with 2 d.o.f.']
    contours_var_ls = ['solid', 'dashed', 'dotted']

    contours_var = axe.contour(line_search_e_space, norm_par_space, chi_arr.T, levels=chi_contours, colors='black',
                               linestyles=contours_var_ls, label=contours_var_labels)

    # avoiding error if there are no contours to plot
    if float(mpl.__version__.split('.')[0])>=3 and float(mpl.__version__.split('.')[1])>=8:
        contour_looper=contours_var.legend_elements()[0]
    else:
        contour_looper=contours_var.collections

    for l in range(len(contour_looper)):
    # there is an issue in the current matplotlib version with contour labels crashing the legend so we use proxies instead
        axe.plot([], [], ls=contours_var_ls[l], label=contours_var_labels[l], color='black')

        # not using this
        # contours_var.collections[l].set_label(contours_var_labels[l])

    contours_base_labels = [r'base level ($\Delta C=0.5$)']
    contours_base_ls = ['dashed']

    contours_base = axe.contour(line_search_e_space, norm_par_space, chi_arr.T, levels=[chi_base + 0.5], colors='grey',
                                linewidths=0.5, linestyles=contours_base_ls)

    if float(mpl.__version__.split('.')[0])>=3 and float(mpl.__version__.split('.')[1])>=8:
        contour_looper_base=contours_base.legend_elements()[0]
    else:
        contour_looper_base=contours_base.collections

    for l in range(len( contour_looper_base)):
        # same workaround here
        axe.plot([], [], ls=contours_base_ls[l], lw=0.5, label=contours_base_labels[l], color='black')

        # not using this
        # contours_base.collections[l].set_label(contours_base_labels[l])

    # for each peak and width, the coordinates need to be translated in real energy and norm coordinates
    # try:
    for elem_point in enumerate(peak_points):
        point_coords = [line_search_e_space[elem_point[1][0]], norm_par_space[elem_point[1][1]]]

        segment_coords = [point_coords[0] - line_search_e[2] * peak_widths[elem_point[0]] / 2,
                          point_coords[0] + line_search_e[2] * peak_widths[elem_point[0]] / 2], [point_coords[1],
                                                                                                 point_coords[1]]

        if show_peak_pos:
            #offset of one index (aka 1 line_search_e[2]) because of the internal offset in the values
            axe.scatter(np.array(point_coords[0])+line_search_e[2], point_coords[1], marker='X', color='black',
                    label='peak' if elem_point[0] == 0 else None)

        if show_peak_width:
            #offset of one index (aka 1 line_search_e[2]) because of the internal offset in the values
            axe.plot(np.array(segment_coords[0])+line_search_e[2], segment_coords[1], color='black',
                 label='max peak structure width' if elem_point[0] == 0 else None)
    # except:
    #     pass

    # using a weird class to get correct tickers on the axes since it doesn't work natively
    axe.yaxis.set_minor_locator(MinorSymLogLocator(line_search_norm[0]))

    #removing a part of the central ticks to avoid overplotting
    full_yticks=axe.get_yticks()

    axe.set_yticks(full_yticks[:len(full_yticks)//2-1].tolist()+[0]+full_yticks[len(full_yticks)//2+2:].tolist())

    if combined == False:
        fig.tight_layout()
    else:
        if combined == 'paper':
            axe.legend(loc='upper left')
        elif combined == 'nolegend':
            pass
        else:
            axe.legend(title='Bottom panel labels', loc='right', bbox_to_anchor=(1.25, 1.5))


def comb_chi2map(fig_comb, chi_dict, title='', comb_label='',
                 force_ylog_ratio=False,ratio_bounds=None,ener_show_range=None,show_indiv_transi=False,
                 squished_mode=False,local_chi_bounds=False,force_side_lines='none',minor_locator=False,
                 show_peak_pos=True,show_peak_width=True):

    '''

    force_ylog_ratio, ratio bounds: for visual changes on the y axis

    ener_show_range: force a specific range for the x axis (useful when taking bigger chi_dicts as input)

    squished_mode: adds a bunch of line separators in th colormap label to avoid display issues
    '''

    line_cont_range = chi_dict['line_cont_range']
    plot_ratio_values = chi_dict['plot_ratio_values']
    line_search_e_space = chi_dict['line_search_e_space']
    n_groups=len(chi_dict['plot_ratio_values'])

    ax_comb = np.array([None] * 2)
    fig_comb.suptitle(title)

    # gridspec creates a grid of spaces for subplots. We use 2 rows for the 2 plots
    # Second column is there to keep space for the colorbar. Hspace=0. sticks the plots together
    gs_comb = GridSpec(2, 2, figure=fig_comb, width_ratios=[98, 2], hspace=0.)

    # first subplot is the ratio
    ax_comb[0] = plt.subplot(gs_comb[0, 0])
    ax_comb[0].set_xlabel('Energy (keV)')
    ax_comb[0].set_ylabel('Fit ratio')
    ax_comb[0].set_xlim(line_cont_range)
    # we put the x axis on top to avoid it being hidden by the second subplot2aaa
    ax_comb[0].xaxis.tick_top()
    ax_comb[0].xaxis.set_label_position('top')

    if force_ylog_ratio:
        ax_comb[0].set_yscale('log')



    # for now we only expect up to 3 data groups. The colors below are the standard xspec colors, for visual clarity with the xspec screen

    for i_grp in range(n_groups):
        ax_comb[0].errorbar(plot_ratio_values[i_grp][0][0], plot_ratio_values[i_grp][1][0],
                            xerr=plot_ratio_values[i_grp][0][1].clip(0), yerr=plot_ratio_values[i_grp][1][1].clip(0),
                            color=xcolors_grp[i_grp], ecolor=xcolors_grp[i_grp], linestyle='None',
                            label=comb_label[i_grp])

    ax_comb[0].axhline(y=1, xmin=0, xmax=1, color='green')

    # limiting the plot to the range of the line energy search
    ax_comb[0].set_xlim(line_search_e_space[0], line_search_e_space[-1])

    # not needed for now
    # #selecting the indexes of the points of the plot which are in the line_search_e energy range
    # plot_ratio_xind_rel=np.array([elem for elem in np.where(plot_ratio_values[0][0][0]>=line_search_e[0])[0]\
    #                               if elem in np.where(plot_ratio_values[0][0][0]<=line_search_e[1])[0]])

    # rescaling with errorbars (which are not taken into account by normal rescaling)
    plot_ratio_y_up = max(ravel_ragged(np.array([(plot_ratio_values[i_grp][1][0] + plot_ratio_values[i_grp][1][1])
                                                 for i_grp in range(n_groups)], dtype=object), mode=object))

    plot_ratio_y_dn = min(ravel_ragged(np.array([(plot_ratio_values[i_grp][1][0] - plot_ratio_values[i_grp][1][1])
                                                 for i_grp in range(n_groups)], dtype=object), mode=object))

    if ratio_bounds is None:

        ax_comb[0].set_ylim(0.95 * np.min(plot_ratio_y_dn), 1.05 * np.max(plot_ratio_y_up))
    else:
        ax_comb[0].set_ylim(ratio_bounds[0],ratio_bounds[1])

    '''second plot (contour)'''

    ax_comb[1] = plt.subplot(gs_comb[1, 0], sharex=ax_comb[0])
    ax_colorbar = plt.subplot(gs_comb[1, 1])
    coltour_chi2map(fig_comb, ax_comb[1], chi_dict, combined=True, ax_bar=ax_colorbar,
                    squished_mode=squished_mode,local_chi_bounds=local_chi_bounds,ener_show_range=ener_show_range,
                    show_peak_pos=show_peak_pos,show_peak_width=show_peak_width)

    ax_comb[1].set_xlim(line_cont_range)
    # #third plot (color), with a separate colorbar plot on the second column of the gridspec
    # #to avoid reducing the size of the color plot
    # ax_comb[2]=plt.subplot(gs_comb[2,0])
    # ax_colorbar=plt.subplot(gs_comb[2,1])
    # color_plot(fig_comb,ax_comb[2],combined=True,ax_bar=ax_colorbar)

    # #we currently do not map the confidence levels of the
    # #adding peak significance to the color plot
    # if assess_line==True:

    #     for i_peak in range(len(peak_sign)):
    #         #restricting the display to absorption lines
    #         if peak_eqws[i_peak]<0:
    #             #the text position we input is the horizontal symmetrical compared to the peak's position
    #             ax_comb[2].annotate((str(round(100*peak_sign[i_peak],len(str(nfakes)))) if peak_sign[i_peak]!=1 else\
    #                                  '>'+str((1-1/nfakes)*100))+'%',\
    #                                  xy=(peak_points[i_peak][0],len(norm_par_space)-peak_points[i_peak][1]),
    #                                  xytext=(peak_points[i_peak][0],peak_points[i_peak][1]),color='white',ha='center',
    #                                 arrowprops=dict(arrowstyle='->',color='white'))

    '''Plotting the Standard Line energies'''

    if ener_show_range is not None:
        ax_comb[0].set_xlim(ener_show_range[0],ener_show_range[1])
        ax_comb[1].set_xlim(ener_show_range[0],ener_show_range[1])

    if minor_locator is not False:
        ax_comb[0].xaxis.set_minor_locator(AutoMinorLocator(minor_locator))
        ax_comb[1].xaxis.set_minor_locator(AutoMinorLocator(minor_locator))


    plot_std_ener(ax_comb[0], ax_comb[1], plot_em=True,mode='chimap',plot_indiv_transi=show_indiv_transi,
                  squished_mode=squished_mode,force_side=force_side_lines)


def merge_chi_dict(chi_dict_files,skip_chi_base_equal=False):

    '''
    attempts to merge a series of chi_dict binary files saved by dill

    First checks if all the parameters that should be the same are the same, and if so, combines them into a new file

    bypass_chi_base can be useful when ls where run on subsections of the spectra and thus chi_base is not
    representative

    '''

    chi_dict_arr=[]

    for elem_file in chi_dict_files:
        with open(elem_file,'rb') as f:
            elem_chi_dict=dill.load(f)
            chi_dict_arr+=[elem_chi_dict]

    line_search_e_arr=np.array([elem['line_search_e'] for elem in chi_dict_arr])

    #sorting the list to an increasing order of energy bands
    chi_dict_arr=np.array(chi_dict_arr)[np.argsort(line_search_e_arr.T[0])]
    chi_dict_files_sorted=np.array(chi_dict_files)[np.argsort(line_search_e_arr.T[0])]

    #testing whether the files are compatible

    chi_base_equal=skip_chi_base_equal or len(np.unique([elem['chi_base'] for elem in chi_dict_arr]))==1
    line_threshold_equal=len(np.unique([elem['line_threshold'] for elem in chi_dict_arr]))==1
    line_search_norm_equal=len(np.unique([str(elem['line_search_norm']) for elem in chi_dict_arr]))==1

    #recreating the array since the order has changed
    line_search_e_arr=np.array([elem['line_search_e'] for elem in chi_dict_arr])

    ##testing whether the energies are compatible
    line_search_e_step_equal=len(np.unique(line_search_e_arr.T[-1]))==1
    line_search_bounds_compat=str(line_search_e_arr.T[0][1:])==str(line_search_e_arr.T[1][:-1])

    assert chi_base_equal and line_threshold_equal and line_search_norm_equal \
           and line_search_e_step_equal and line_search_bounds_compat,'Error: line search saves not compatible'

    chi_dict_merge={}
    #directly giving it the elements that remain common
    chi_dict_merge['chi_base']=1e4 if skip_chi_base_equal else chi_dict_arr[0]['chi_base']
    chi_dict_merge['line_threshold']=chi_dict_arr[0]['line_threshold']
    chi_dict_merge['line_search_norm']=chi_dict_arr[0]['line_search_norm']
    chi_dict_merge['norm_par_space']=chi_dict_arr[0]['norm_par_space']
    chi_dict_merge['plot_ratio_values']=chi_dict_arr[0]['plot_ratio_values']

    #merging the energy bounds
    chi_dict_merge['line_search_e']=np.array([line_search_e_arr[0][0],line_search_e_arr[-1][1],line_search_e_arr[0][-1]])
    chi_dict_merge['line_search_e_space']=np.arange(chi_dict_merge['line_search_e'][0],
                                        chi_dict_merge['line_search_e'][1]+chi_dict_merge['line_search_e'][2]/2,
                                                    chi_dict_merge['line_search_e'][2])
    chi_dict_merge['line_cont_range']=chi_dict_merge['line_search_e'][:2]

    chi_arr_merge=[]
    peak_points=[]
    peak_widths=[]

    #useful for moving the peak positions
    n_ener_added=0

    for i_elem,elem in enumerate(chi_dict_arr):

        #testing if the last element is in the array or not (failsafe for previous issue with narrow_line_search)

        complete_set=abs(elem['line_search_e_space'][-1]-elem['line_search_e'][1])<1e-5

        if complete_set:
            #-1 to avoid redundancy for the bridge energy value
            chi_arr_merge+=(1e4+elem['chi_arr'][:-1]-elem['chi_base']).tolist()
        else:
            #-1 to avoid redundancy for the bridge energy value
            chi_arr_merge+=(1e4+elem['chi_arr']-elem['chi_base']).tolist()

        elem_peak_points=elem['peak_points']
        if len(elem_peak_points)!=0:
            elem_peak_points.T[0]+=n_ener_added

        peak_points+=elem_peak_points.tolist()

        if len(elem['peak_widths'])!=0:
            peak_widths+=elem['peak_widths'].tolist()

        if complete_set:
            n_ener_added=len(chi_arr_merge[:-1])
        else:
            n_ener_added=len(chi_arr_merge)

    #adding the missing last element of the last band for the chi_arr
    chi_arr_merge+=(1e4+chi_dict_arr[-1]['chi_arr'][-1:]-chi_dict_arr[-1]['chi_base']).tolist()

    if len(chi_arr_merge)!=len(chi_dict_merge['line_search_e_space']):
        if len(chi_arr_merge)==len(chi_dict_merge['line_search_e_space'])-1:
            chi_arr_merge+=[chi_arr_merge[-1]]
        else:
            breakpoint()

    #reconverting into arrays
    chi_arr_merge=np.array(chi_arr_merge)
    peak_points=np.array(peak_points)
    peak_widths=np.array(peak_widths)

    chi_dict_merge['chi_arr']=chi_arr_merge
    chi_dict_merge['peak_points']=peak_points
    chi_dict_merge['peak_widths']=peak_widths

    #saving the array
    low_e=chi_dict_merge['line_search_e'][0]
    high_e=chi_dict_merge['line_search_e'][1]

    merge_save_path=chi_dict_files_sorted[0][:chi_dict_files_sorted[0].rfind(str(low_e))]+str(low_e)+'_'+str(high_e)+'.dill'

    with open(merge_save_path,'wb+') as f:
        dill.dump(chi_dict_merge,f)
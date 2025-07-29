#general imports
import sys,os
import glob
import getpass
username=getpass.getuser()

import numpy as np
import pandas as pd
from decimal import Decimal

import streamlit as st
#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from matplotlib.ticker import Locator

import matplotlib.dates as mdates



from astropy.time import Time


#correlation values and trend plots with MC distribution from the uncertainties
from custom_pymccorrelation import pymccorrelation
from lmplot_uncert import lmplot_uncert_a

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties,
# so I tweaked their 'perturb values' function


'''Astro'''

#rough way of testing if online or not
online=os.getcwd().startswith('/mount/src')
project_dir='/'.join(__file__.split('/')[:-3])

#to be tested online
sys.path.append(os.path.join(project_dir,'observations/spectral_analysis/'))
sys.path.append(os.path.join(project_dir,'general/'))


#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std, lines_std_names

from general_tools import ravel_ragged

#Catalogs and manipulation

from visual_line_tools import ratio_choices,ratio_choices_str,info_str,axis_str,telescope_colors,\
                              info_hid_str,axis_hid_str


def correl_graph(data_perinfo, infos, data_ener, dict_linevis, mode='intrinsic', mode_vals=None, conf_thresh=0.99,
                 indiv=False, save=False, close=False,
                 streamlit=False, compute_correl=False, bigger_text=False, show_linked=True, show_ul_ew=False):
    '''

    should be updated to reflect 4U paper changes

    Scatter plots for specific line parameters from all the observations in the current pool (i.e. all objects/obs/lines).

    infos:
        Intrinsic:
            -ewidth
            -bshift
            -ener
            -ewratio+25/26/Ka/Kb : ratio between the two 25/26/Ka/Kb lines. Always uses the highest line (in energy) as numerator
            -width Note: adding a number directly after the width is necessary to plot ewratio vs width as line for which
                            the width is plotted needs to be specified
            -nH

        observ:
            -HR
            -flux
            -Time
            'nthcomp-gamma' (only for high-energy obs)
            'highE-flux' (15-50 keV)
            'highE-HR' (15-50keV/3-6keV)

        source:
            -inclin
    Modes:
        -Intrinsic  Use the 'info' keyword to plot infos vs one another
        -observ     Use the 'info' keyword to plot infos vs HR or flux (observ info always on y axis)
        -source     Use the 'info' keyword to plot ewidth, bshift, ener or flux vs the source inclinations
        -ewcomp     Use the 'info' keyword to specify which the lines for which the ew will be compared

    In HID mode, requires the mode_vals variable to be set to the flux values
    in inclin mode, idem with the inclination values

    Color modes (not an argument, ruled by the dicionnary variable color_scatter):
        -None
        -Instrument
        -Sources
        -Time
        -HR
        -width
        -line structures (for 4U1630-47)
        -accretion states (for 4U1630-47)
        -outbursts

    (Note : displaying non significant results overrides the coloring)

    Use the 'indiv' keyword to plot for all lines simultaneously or 6 plots for the 6 single lines
    the streamlit mode uses the indiv=False since the lines are already selected in the data provided to the function

    Non detections are filtered via 0 values in significance

    Detections above and below the given threshold and linked values are highlighted if the checkbox to display non significant detectoins is set, else
    only significant detections are displayed

    It is possible to compute the spearman and pearson coefficients,
            to compute linear regressions for significantly correlated graphs, and to restrict the range of both
            computations with x and y bounds

    The person and spearman coefficients and the regressions are computed from significant detections only,
     with MC propagation of their uncertainties through a custom library adapted from
      https://github.com/privong/pymccorrelation
    As such, the distribution of the errors are assumed to be gaussian
    (which is not the case, but using the distributions themselves would be a nightmare)


    we ravel the last 2 dimensions with a custom function since for multi object plots the sequences can be ragged and the custom .ravel function
    doesn't work

    Note : there's an explicit call to the FeKg26 being the last of the tested abs line (and assumption that Kb25 and Kb26 are just before),
    so the uncertainty shift part will have to be changed if this is modified
    '''

    def forceAspect(ax, aspect=1):

        '''Forces an aspect ratio of 1 in matplotlib axes'''

        # lims=(abs(ax.get_xlim()[1]-ax.get_xlim()[0]),abs(ax.get_ylim()[1]-ax.get_ylim()[0]))
        # ax.set_aspect(2)
        pass

    range_absline = dict_linevis['range_absline']
    mask_lines = dict_linevis['mask_lines']

    if streamlit:
        display_nonsign = dict_linevis['display_nonsign']
        scale_log_hr = dict_linevis['scale_log_hr']
        scale_log_ew = dict_linevis['scale_log_ew']

        compute_regr = dict_linevis['compute_regr']
        regr_pval_threshold = dict_linevis['regr_pval_threshold']
        restrict_comput_scatter = dict_linevis['restrict_comput_scatter']
        comput_scatter_lims = dict_linevis['comput_scatter_lims']

        lock_lims_det = dict_linevis['lock_lims_det']
        display_pearson = dict_linevis['display_pearson']
        display_abserr_bshift = dict_linevis['display_abserr_bshift']
        display_std_abserr_bshift = dict_linevis['display_std_abserr_bshift']
        glob_col_source = dict_linevis['glob_col_source']
        display_th_width_ew = dict_linevis['display_th_width_ew']
        cmap_color_det = dict_linevis['cmap_color_det']
        common_observ_bounds_lines = dict_linevis['common_observ_bounds_lines']
        common_observ_bounds_dates = dict_linevis['common_observ_bounds_dates']
        use_alpha_ul = dict_linevis['use_alpha_ul']
    else:

        display_nonsign = True
        scale_log_hr = True
        scale_log_ew = True
        compute_regr = False
        lock_lims_det = False
        display_pearson = False
        display_abserr_bshift = False
        display_std_abserr_bshift = False
        glob_col_source = False
        display_th_width_ew = False
        cmap_color_det = mpl.cm.plasma
        common_observ_bounds_lines = False
        common_observ_bounds_dates = False
        use_alpha_ul = False

    mask_obj = dict_linevis['mask_obj']
    abslines_ener = dict_linevis['abslines_ener']
    abslines_plot = dict_linevis['abslines_plot']

    save_dir = dict_linevis['save_dir']
    save_str_prefix = dict_linevis['save_str_prefix']
    args_cam = dict_linevis['args_cam']
    line_search_e_str = dict_linevis['line_search_e_str']
    args_line_search_norm = dict_linevis['args_line_search_norm']

    if streamlit:
        slider_date = dict_linevis['slider_date']
        instru_list = dict_linevis['instru_list'][mask_obj]
        diago_color = dict_linevis['diago_color'][mask_obj]
        custom_states_color = dict_linevis['custom_states_color'][mask_obj]
        custom_outburst_color = dict_linevis['custom_outburst_color'][mask_obj]
        custom_outburst_number = dict_linevis['custom_outburst_number'][mask_obj]
        custom_outburst_dict = dict_linevis['custom_outburst_dict'][mask_obj]
        custom_ionization_color = dict_linevis['custom_ionization_color'][mask_obj]
        display_legend_correl = dict_linevis['display_legend_correl']

    if 'color_scatter' in dict_linevis:
        color_scatter = dict_linevis['color_scatter']
        obj_disp_list = dict_linevis['obj_list'][mask_obj]
        date_list = dict_linevis['date_list'][mask_obj]
        hid_plot = dict_linevis['hid_plot_restrict']
        width_plot_restrict = dict_linevis['width_plot_restrict']
        nh_plot_restrict = dict_linevis['nh_plot_restrict']

    else:
        color_scatter = 'None'
        date_list = None
        hid_plot = None
        width_plot_restrict = None
        nh_plot_restrict = None

    lum_plot_restrict = dict_linevis['lum_plot_restrict']
    observ_list = dict_linevis['observ_list'][mask_obj]

    # note that these one only get the significant BAT detections so no need to refilter
    lum_high_sign_plot_restrict = dict_linevis['lum_high_sign_plot_restrict']
    hr_high_sign_plot_restrict = dict_linevis['hr_high_sign_plot_restrict']

    gamma_nthcomp_plot_restrict = dict_linevis['gamma_nthcomp_plot_restrict']

    mask_added_regr_sign = dict_linevis['mask_added_regr_sign']

    correl_internal_mode=dict_linevis['correl_internal_mode']
    npert_rank=dict_linevis['npert_rank']
    paper_look_correl=dict_linevis['paper_look_correl']

    mask_lum_high_valid = ~np.isnan(ravel_ragged(lum_high_sign_plot_restrict[0]))

    # This considers all the lines
    mask_obs_sign = np.array([ravel_ragged(abslines_plot[4][0].T[mask_obj].T[i]).astype(float) >= conf_thresh \
                              for i in range(len(mask_lines))]).any(0)

    # note: quantities here are already restricted
    mask_intime_norepeat = (Time(ravel_ragged(date_list)) >= slider_date[0]) & (
                Time(ravel_ragged(date_list)) <= slider_date[1])

    mask_sign_intime_norepeat = mask_obs_sign & mask_intime_norepeat

    mask_bounds = ((mask_intime_norepeat) if not common_observ_bounds_dates else np.repeat(True,
                                                                                           len(mask_intime_norepeat))) & \
                  ((mask_obs_sign) if (common_observ_bounds_lines and not (show_ul_ew and not lock_lims_det)) \
                       else np.repeat(True, len(mask_intime_norepeat)))

    if sum(mask_bounds) > 0:
        bounds_hr = [0.9 * min(ravel_ragged(hid_plot[0][0])[mask_bounds] - ravel_ragged(hid_plot[0][1])[mask_bounds]),
                     1 / 0.9 * max(
                         ravel_ragged(hid_plot[0][0])[mask_bounds] + ravel_ragged(hid_plot[0][2])[mask_bounds])]
        bounds_flux = [0.9 * min(ravel_ragged(hid_plot[1][0])[mask_bounds] - ravel_ragged(hid_plot[1][1])[mask_bounds]),
                       1 / 0.9 * max(
                           ravel_ragged(hid_plot[1][0])[mask_bounds] + ravel_ragged(hid_plot[1][2])[mask_bounds])]

    else:
        bounds_hr = None
        bounds_flux = None

    n_obj = len(observ_list)

    infos_split = infos.split('_')
    alpha_ul = 0.3 if use_alpha_ul else 1.

    x_error = None
    y_error = None

    # using time changes the way things are plotted, uncertainties are computed and significance are used
    if 'time' in infos_split[0]:
        time_mode = True
        infos_split = infos_split[::-1]
    else:
        time_mode = False

    # using ratios changes the way things are plotted, uncertainties are computed and significance are used
    if 'ratio' in infos:
        ratio_mode = True

        if 'ratio' in infos_split[0]:
            # fetching the line indexes corresponding to the names
            ratio_indexes_x = ratio_choices[infos_split[0][-2:]]

            # keeping the same architecture in time mode even if the ratio is plotted on the y axis for simplicity
        elif 'ratio' in infos_split[1]:
            # fetching the line indexes corresponding to the names
            ratio_indexes_x = ratio_choices[infos_split[1][-2:]]
    else:
        ratio_mode = False

    if mode == 'ewcomp':
        line_comp_mode = True
    else:
        line_comp_mode = False

    # not showing upper limit for the width plot as there's no point currently
    if 'width' in infos:
        width_mode = True
        show_ul_ew = False
        if ratio_mode:
            # if comparing width with the ewratio, the line number for the width has to be specified for the width and is
            # retrieved (and -1 to compare to an index)
            width_line_id = int(infos[infos.find('width') + 5]) - 1

    else:
        width_mode = False

    # failsafe to prevent wrong colorings for intrinsic plots
    if (ratio_mode or line_comp_mode) and color_scatter == 'width':
        color_scatter = 'None'

    data_list = [data_perinfo[0], data_perinfo[1], data_ener, data_perinfo[3], date_list, width_plot_restrict]

    high_E_mode = False
    # infos and data definition depending on the mode
    if mode == 'intrinsic':

        ind_infos = [
            np.argwhere([elem in infos_split[i] for elem in ['ew', 'bshift', 'ener', 'lineflux', 'time', 'width']])[0][
                0] for i in [0, 1]]

        data_plot = [data_list[ind] for ind in ind_infos]

    elif mode == 'observ':
        # the 'None' is here to have the correct index for the width element
        ind_infos = [
            np.argwhere([elem in infos_split[0] for elem in ['ew', 'bshift', 'ener', 'lineflux', 'None', 'width']])[0][
                0],
            np.argwhere(np.array(['HR', 'flux', 'time', 'nthcomp-gamma', 'highE-flux', 'highE-HR']) == infos_split[1])[
                0][0]]

        if ind_infos[1] in [3, 4, 5]:
            high_E_mode = True

        second_arr_infos = [date_list, gamma_nthcomp_plot_restrict, lum_high_sign_plot_restrict,
                            hr_high_sign_plot_restrict]
        data_plot = [data_list[ind_infos[0]], second_arr_infos[ind_infos[1] - 2] if ind_infos[1] >= 2 \
            else mode_vals[ind_infos[1]]]

        if time_mode:
            data_plot = data_plot[::-1]

    elif mode == 'source':
        ind_infos = [
            np.argwhere([elem in infos_split[0] for elem in ['ew', 'bshift', 'ener', 'lineflux', 'None', 'width']])[0][
                0], -1]
        data_plot = [data_list[ind_infos[0]], mode_vals]

    elif line_comp_mode:
        ind_infos = [np.argwhere([elem in infos_split[i] for elem in lines_std_names[3:]])[0][0] for i in [0, 1]]
        data_plot = [data_perinfo[0], data_perinfo[0]]

    if indiv:
        graph_range = range_absline
    else:
        # using a list index for the global graphs allows us to keep the same structure
        # however we need to restrict it to the currently plotted lines in streamlit mode
        if streamlit:
            graph_range = [range(len([elem for elem in mask_lines if elem]))]
        else:
            graph_range = [range_absline]

    for i in graph_range:

        figsize_val = [8. + (-0.5 if mode != 'observation' else 0), 5.5] if color_scatter in ['Time', 'HR', 'width',
                                                                                              'nH', 'L_3-10'] else [
        6, 6]

        if paper_look_correl:
            figsize_val[0]-=1
            figsize_val[1]-=1


        # if color_scatter in ['Time','HR','width','nH']:
        #
        #     fig_scat = plt.figure(figsize=figsize_val if bigger_text else (11, 10.5))
        #     gs = GridSpec(1, 2, width_ratios=[4, 0.2])
        #     ax_scat=plt.subplot(gs[0])
        #     ax_cb=plt.subplot(gs[1])
        # else:
        fig_scat, ax_scat = plt.subplots(1, 1, figsize=figsize_val if bigger_text else (11, 10.5))

        legend_title = ''

        if 'width' in infos_split[1]:
            ax_scat.set_ylim(0, 5500)
        elif 'width' in infos_split[0]:
            ax_scat.set_xlim(0, 5500)

        if not bigger_text:
            if indiv:
                fig_scat.suptitle(info_str[ind_infos[0]] + ' - ' + (info_str[ind_infos[1]] if mode == 'intrinsic' else \
                                                                        info_hid_str[
                                                                            ind_infos[1]]) + ' scatter for the ' +
                                  lines_std[lines_std_names[3 + i]] +
                                  ' absorption line')
            else:
                fig_scat.suptitle(
                    ((infos_split[0] + ' - ' + infos_split[1] + ' equivalent widths') if line_comp_mode else \
                         ((info_str[ind_infos[0]] if not ratio_mode else infos_split[0]) + ' - ' + (
                             info_str[ind_infos[1]] if mode == 'intrinsic' else \
                                 (info_hid_str[ind_infos[1]]) if mode == 'observ' else 'inclination'))) \
                    + (' for currently selected ' + (
                        'lines and ' if not ratio_mode else '') + 'sources' if streamlit else ' for all absorption lines'))

        if not line_comp_mode:

            if sum(mask_lines) == 1:
                line_str = lines_std[np.array(lines_std_names)[3:9][mask_lines][0]]
            else:
                line_str = ''

            ax_scat.set_xlabel(
                'Time' if time_mode else (ratio_choices_str[infos_split[0][-2:]] + ' ' if ratio_mode else '') +
                                         (axis_str[ind_infos[0]].replace(' (eV)',
                                                                         ('' if ratio_mode else ' (eV)')) + ' ' + (
                                                      infos_split[0][-2:] + ' ratio')) \
                    if ratio_mode else line_str + ' ' + axis_str[ind_infos[0]])

        else:
            line_str = ''
            ax_scat.set_xlabel(infos_split[0] + ' ' + axis_str[ind_infos[0]])
            ax_scat.set_ylabel(infos_split[1] + ' ' + axis_str[ind_infos[0]])

        # putting a logarithmic y scale if it shows equivalent widths or one of the hid parameters
        # note we also don't put the y log scale for a gamma y axis (of course)
        if mode != 'source' and ((mode == 'observ' and scale_log_hr and not time_mode and not ind_infos[1] == [3]) \
                                 or ((ind_infos[0 if time_mode else 1] in [0, 3] or line_comp_mode) and scale_log_ew)):
            ax_scat.set_yscale('log')

        # putting a logarithmic x scale if it shows equivalent widths
        if ind_infos[0] in [0, 3] and scale_log_ew and not time_mode:
            ax_scat.set_xscale('log')

        if mode == 'intrinsic':
            ax_scat.set_ylabel(line_str + ' ' + axis_str[ind_infos[1]])
        elif mode == 'observ':
            ax_scat.set_ylabel(axis_str[ind_infos[0]] if (time_mode and not ratio_mode) else \
                                   axis_str[ind_infos[0]] + ' ' + (infos_split[0][-2:] + ' ratio') if time_mode else \
                                       axis_hid_str[ind_infos[1]])
        elif mode == 'source':
            ax_scat.set_ylabel('Source inclination (°)')
            ax_scat.set_ylim((0, 90))

        # creating a time variable for time mode to be used later

        if time_mode:
            # creating an appropriate date axis
            # manually readjusting for small durations because the AutoDateLocator doesn't work well
            time_range = min(mdates.date2num(slider_date[1]), max(mdates.date2num(ravel_ragged(date_list)))) - \
                         max(mdates.date2num(slider_date[0]), min(mdates.date2num(ravel_ragged(date_list))))

            #
            #
            # if time_range<150:
            #     date_format=mdates.DateFormatter('%Y-%m-%d')
            # elif time_range<1825:
            #     date_format=mdates.DateFormatter('%Y-%m')
            # else:
            #     date_format=mdates.AutoDateFormatter(mdates.AutoDateLocator())
            # ax_scat.xaxis.set_major_formatter(date_format)
            #
            # plt.xticks(rotation=70)

        date_list_repeat = np.array(
            [date_list for repeater in (i if type(i) == range else [i])]) if not ratio_mode else date_list

        mask_lum_high_valid_repeat = ravel_ragged(
            [mask_lum_high_valid for repeater in (i if type(i) == range else [i])]) \
            if not ratio_mode else mask_lum_high_valid

        if streamlit:
            mask_intime = (Time(ravel_ragged(date_list_repeat)) >= slider_date[0]) & (
                        Time(ravel_ragged(date_list_repeat)) <= slider_date[1])

        else:
            mask_intime = True

            mask_intime_norepeat = True

        # the boolean masks for detections and significance are more complex when using ratios instead of the standard data since
        # two lines need to be checked

        if line_comp_mode or ratio_mode:

            # we can use the data constructs used for the ew ratio mode to create the ratios in ratio_mode
            # we just need to get common indexing variable
            ind_ratio = ind_infos if line_comp_mode else ratio_indexes_x

            # in ew ratio mode we need to make sure than both lines are defined for each point so we must combine the mask of both lines

            bool_sign_x = ravel_ragged(data_perinfo[4][0][ind_ratio[0]]).astype(float)
            bool_sign_y = ravel_ragged(data_perinfo[4][0][ind_ratio[1]]).astype(float)

            # adding a width mask to ensure we don't show elements with no width
            if width_mode:
                bool_sign_ratio_width = ravel_ragged(width_plot_restrict[2][width_line_id]).astype(float) != 0
            else:
                bool_sign_ratio_width = True

            bool_det_ratio = (bool_sign_x != 0.) & (~np.isnan(bool_sign_x)) & (bool_sign_y != 0.) & (
                ~np.isnan(bool_sign_y)) & \
                             mask_intime_norepeat & (True if not high_E_mode else mask_lum_high_valid)

            # for whatever reason we can't use the bitwise comparison so we compute the minimum significance of the two lines before testing for a single arr
            bool_sign_ratio = np.array([bool_sign_x[bool_det_ratio], bool_sign_y[bool_det_ratio]]).T.min(
                1) >= conf_thresh

            # applying the width mask (or just a True out of width mode)
            bool_sign_ratio = bool_sign_ratio & (
                True if bool_sign_ratio_width is True else bool_sign_ratio_width[bool_det_ratio])

            # masks for upper limits (needs at least one axis to have detection and significance)
            bool_nondetsign_x = np.array(((bool_sign_x < conf_thresh).tolist() or (np.isnan(bool_sign_x).tolist()))) & \
                                (np.array((bool_sign_y >= conf_thresh).tolist()) & np.array(
                                    (~np.isnan(bool_sign_y)).tolist())) \
                                & mask_intime_norepeat & (True if not high_E_mode else mask_lum_high_valid)
            bool_nondetsign_y = np.array(((bool_sign_y < conf_thresh).tolist() or (np.isnan(bool_sign_y).tolist()))) & \
                                (np.array((bool_sign_x >= conf_thresh).tolist()) & np.array(
                                    (~np.isnan(bool_sign_x)).tolist())) \
                                & mask_intime_norepeat & (True if not high_E_mode else mask_lum_high_valid)

            # the boool sign and det are only used for the ratio in ratio_mode, but are global in ewratio mode
            if line_comp_mode:

                # converting the standard variables to the ratio ones
                bool_det = bool_det_ratio
                bool_sign = bool_sign_ratio

                # note: we don't care about the 'i' index of the graph range here since this graph is never made in indiv mode/not all lines mode

                # here we can build both axis variables and error variables identically
                x_data, y_data = [np.array([ravel_ragged(data_plot[0][0][ind_infos[i]])[bool_det][bool_sign],
                                            ravel_ragged(data_plot[0][0][ind_infos[i]])[bool_det][~bool_sign]],
                                           dtype=object) for i in [0, 1]]

                mask_added_regr_sign_use = None if mask_added_regr_sign is None else \
                    mask_added_regr_sign[bool_det][bool_sign]

                # same thing for the uncertainties
                x_error, y_error = [
                    np.array([[ravel_ragged(data_plot[i][l][ind_infos[i]])[bool_det][bool_sign] for l in [1, 2]],
                              [ravel_ragged(data_plot[i][l][ind_infos[i]])[bool_det][~bool_sign] for l in [1, 2]]],
                             dtype=object) for i in [0, 1]]

            # here we assume this is ratio_mode
            else:
                # this time data_plot is an array
                # here, in the ratio X is the numerator so the values are obtained by dividing X/Y

                # here we can directly create the data ratio
                if time_mode:

                    x_data = np.array([mdates.date2num(ravel_ragged(date_list_repeat))[bool_det_ratio][bool_sign_ratio],
                                       mdates.date2num(ravel_ragged(date_list_repeat))[bool_det_ratio][
                                           ~bool_sign_ratio]],
                                      dtype=object)

                    # in this case the ew ratio is on the Y axis
                    y_data = np.array(
                        [ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio] / \
                         ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                         ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio] / \
                         ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],
                        dtype=object)
                else:
                    x_data = np.array(
                        [ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio] / \
                         ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                         ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio] / \
                         ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]],
                        dtype=object)
        else:

            # these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others
            bool_sign = ravel_ragged(data_perinfo[4][0][i]).astype(float)

            # standard detection mask
            bool_det = (bool_sign != 0.) & (~np.isnan(bool_sign)) & (mask_intime) & \
                       (True if not high_E_mode else mask_lum_high_valid_repeat)

            bool_detsign_bounds = (bool_sign != 0.) & (~np.isnan(bool_sign)) & \
                                  (mask_intime if not common_observ_bounds_dates else True) & \
                                  (True if not high_E_mode else mask_lum_high_valid_repeat) & (bool_sign >= conf_thresh)

            # mask used for upper limits only
            bool_nondetsign = ((bool_sign < conf_thresh) | (np.isnan(bool_sign))) & (mask_intime) & \
                              (True if not high_E_mode else mask_lum_high_valid_repeat)

            # restricted significant mask, to be used in conjunction with bool_det
            bool_sign = bool_sign[bool_det] >= conf_thresh

            if width_mode:
                # creating a mask for widths that are not compatible with 0 at 3 sigma (for which this width value is pegged to 0)
                bool_sign_width = ravel_ragged(width_plot_restrict[2][i]).astype(float) != 0

                bool_sign = bool_sign & bool_sign_width[bool_det]

            if time_mode:

                x_data = np.array([mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][bool_sign],
                                   mdates.date2num(ravel_ragged(date_list_repeat))[bool_det][~bool_sign]],
                                  dtype=object)

                y_data = np.array([ravel_ragged(data_plot[1][0][i])[bool_det][bool_sign],
                                   ravel_ragged(data_plot[1][0][i])[bool_det][~bool_sign]],
                                  dtype=object)

                if len(x_data[0]) > 0:
                    x_data_bounds = mdates.date2num(ravel_ragged(date_list_repeat))[bool_detsign_bounds]
                else:
                    x_data_bounds = None

            else:
                x_data = np.array([ravel_ragged(data_plot[0][0][i])[bool_det][bool_sign],
                                   ravel_ragged(data_plot[0][0][i])[bool_det][~bool_sign]],
                                  dtype=object)

                if len(x_data[0]) > 0:
                    x_data_bounds = ravel_ragged(data_plot[0][0][i])[bool_detsign_bounds]
                else:
                    x_data_bounds = None

        # applying the same thing to the y axis if ratios are also plotted there
        if type(ind_infos[1]) not in [int, np.int64] and 'ratio' in ind_infos[1]:
            # note:not needed for now
            pass
        # #we fetch the index list corresponding to the info string at the end of the info provided
        # ratio_indexes_y=ratio_choices[ind_infos[1][-2:]]

        # #before using them to create the data ratio (here we don't bother using the non significant detections)
        # #we don't bother masking the infinites here because they are masked natively when plotting
        # y_data=np.array(
        #       [ravel_ragged(data_plot[0][0][ratio_indexes_y[0]])[bool_det[ratio_indexes_y[0]]][bool_sign[ratio_indexes_y[0]]]/\
        #        ravel_ragged(data_plot[0][0][ratio_indexes_y[1]])[bool_det[ratio_indexes_y[1]]][bool_sign[ratio_indexes_y[1]]],
        #        ravel_ragged(data_plot[0][0][ratio_indexes_y[0]])[bool_det[ratio_indexes_y[0]]][~bool_sign[ratio_indexes_y[0]]]/\
        #        ravel_ragged(data_plot[0][0][ratio_indexes_y[1]])[bool_det[ratio_indexes_y[1]]][~bool_sign[ratio_indexes_y[1]]]],dtype=object)
        else:

            if mode == 'intrinsic':

                if width_mode and ratio_mode:
                    # the width needs to be changed to a single line here
                    y_data = np.array([ravel_ragged(data_plot[1][0][width_line_id])[bool_det_ratio][bool_sign_ratio],
                                       ravel_ragged(data_plot[1][0][width_line_id])[bool_det_ratio][~bool_sign_ratio]],
                                      dtype=object)

                else:
                    y_data = np.array([ravel_ragged(data_plot[1][0][i])[bool_det][bool_sign],
                                       ravel_ragged(data_plot[1][0][i])[bool_det][~bool_sign]],
                                      dtype=object)

            elif mode == 'observ' and not time_mode:

                # since the hid data is the same no matter the line, we need to repeat it for the number of lines used
                # when plotting the global graph
                # index 1 since data_plot is [x_axis,y_axis], then 0 to get the main value
                y_data_repeat = np.array([data_plot[1][0] for repeater in (i if type(i) == range else [i])])

                # only then can the linked mask be applied correctly (doesn't change anything in individual mode)
                if ratio_mode:
                    # here we just select one of all the lines in the repeat and apply the ratio_mode mask onto it
                    y_data = np.array([ravel_ragged(y_data_repeat[ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio],
                                       ravel_ragged(y_data_repeat[ratio_indexes_x[0]])[bool_det_ratio][
                                           ~bool_sign_ratio]],
                                      dtype=object)

                    mask_added_regr_sign_use = None if mask_added_regr_sign is None else \
                        mask_added_regr_sign[bool_det_ratio][bool_sign_ratio]
                else:
                    # this is not implemented currently

                    y_data = np.array([ravel_ragged(y_data_repeat)[bool_det][bool_sign],
                                       ravel_ragged(y_data_repeat)[bool_det][~bool_sign]],
                                      dtype=object)

                    mask_added_regr_sign_use = None if mask_added_regr_sign is None else \
                        ravel_ragged([mask_added_regr_sign \
                                      for repeater in (i if type(i) == range else [i])])[bool_det][bool_sign]


            elif mode == 'source':
                y_data_repeat = np.array([data_plot[1][i_obj][0] for repeater in (i if type(i) == range else [i]) \
                                          for i_obj in range(n_obj) for i_obs in
                                          range(len(data_plot[0][0][repeater][i_obj])) \
                                          ]).ravel()

                y_data = np.array([y_data_repeat[bool_det][bool_sign], y_data_repeat[bool_det][~bool_sign]],
                                  dtype=object)

        #### upper limit computation

        if show_ul_ew:

            if line_comp_mode:
                # we use the same double definition here
                # in ratio_mode, x is the numerator so the ul_x case amounts to an upper limit
                y_data_ul_x = np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[0]])[bool_nondetsign_x],
                                       dtype=object)
                x_data_ul_x = np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[1]])[bool_nondetsign_x],
                                       dtype=object)

                # here in ratio mode ul_y amounts to a lower limit
                y_data_ul_y = np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[0]])[bool_nondetsign_y],
                                       dtype=object)
                x_data_ul_y = np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[1]])[bool_nondetsign_y],
                                       dtype=object)

                # same way of defining the errors
                y_error_ul_x = np.array(
                    [ravel_ragged(data_perinfo[0][l][ind_ratio[1]])[bool_nondetsign_x] for l in [1, 2]],
                    dtype=object)
                x_error_ul_y = np.array(
                    [ravel_ragged(data_perinfo[0][l][ind_ratio[0]])[bool_nondetsign_y] for l in [1, 2]],
                    dtype=object)

            elif ratio_mode:

                # we use the same double definition here
                # in ratio_mode, x is the numerator so the ul_x case amounts to an upper limit
                y_data_ul_x = np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[0]])[bool_nondetsign_x],
                                       dtype=object)
                x_data_ul_x = np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[1]])[bool_nondetsign_x],
                                       dtype=object)

                # here in ratio mode ul_y amounts to a lower limit
                y_data_ul_y = np.array(ravel_ragged(data_perinfo[0][0][ind_ratio[0]])[bool_nondetsign_y],
                                       dtype=object)
                x_data_ul_y = np.array(ravel_ragged(data_perinfo[5][0][ind_ratio[1]])[bool_nondetsign_y],
                                       dtype=object)

                # same way of defining the errors
                y_error_ul_x = np.array(
                    [ravel_ragged(data_perinfo[0][l][ind_ratio[1]])[bool_nondetsign_x] for l in [1, 2]],
                    dtype=object)
                x_error_ul_y = np.array(
                    [ravel_ragged(data_perinfo[0][l][ind_ratio[0]])[bool_nondetsign_y] for l in [1, 2]],
                    dtype=object)

                # computing two (upper and lower) limits for the ratios

                # the upper limit corresponds to points for which the numerator line is not detected, so it's the ul_y
                # similarly, the lower limit is for ul_x

                # this is only in ratio_mode
                if ratio_mode:
                    if time_mode:

                        y_data_ll = y_data_ul_y / x_data_ul_y
                        y_data_ul = y_data_ul_x / x_data_ul_x

                        x_data_ll = np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign_y],
                                             dtype=object)
                        x_data_ul = np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign_x],
                                             dtype=object)

                    else:
                        x_data_ll = y_data_ul_y / x_data_ul_y
                        y_data_ll = np.array(ravel_ragged(y_data_repeat[ind_ratio[1]])[bool_nondetsign_y], dtype=object)

                        x_data_ul = y_data_ul_x / x_data_ul_x
                        y_data_ul = np.array(ravel_ragged(y_data_repeat[ind_ratio[0]])[bool_nondetsign_x], dtype=object)

                        pass

            else:

                if time_mode:

                    y_data_ul = np.array(ravel_ragged(data_perinfo[5][0][i])[bool_nondetsign],
                                         dtype=object)
                    x_data_ul = np.array(mdates.date2num(ravel_ragged(date_list_repeat))[bool_nondetsign], dtype=object)

                else:

                    x_data_ul = np.array(ravel_ragged(data_perinfo[5][0][i])[bool_nondetsign],
                                         dtype=object)

                    # we can directly create the y_data_ul from y_data_repeat no matter which one it comes from
                    y_data_ul = np.array(ravel_ragged(y_data_repeat)[bool_nondetsign], dtype=object)

        # in the errors we want to distinguish the parameters which were linked so we start compute the masks of the 'linked' values

        # we don't really care about the linked state in ewratio mode
        if line_comp_mode:
            linked_mask = np.array([np.repeat(False, len(x_data[0])), np.repeat(False, len(x_data[1]))], dtype=object)
        else:
            if ratio_mode:

                linked_mask = np.array(
                    [np.array([type(elem) == str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[0]]) \
                        [bool_det_ratio][bool_sign_ratio]] and \
                              [type(elem) == str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[1]]) \
                                  [bool_det_ratio][bool_sign_ratio]]).astype(object),
                     np.array([type(elem) == str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[0]]) \
                         [bool_det_ratio][~bool_sign_ratio]] and \
                              [type(elem) == str for elem in ravel_ragged(data_perinfo[1][1][ratio_indexes_x[1]]) \
                                  [bool_det_ratio][~bool_sign_ratio]]).astype(object)],
                    dtype=object)
            else:
                linked_mask = np.array( \
                    [np.array([type(elem) == str for elem in \
                               ravel_ragged(data_perinfo[1][1][i])[bool_det][bool_sign].astype(object)]),
                     np.array([type(elem) == str for elem in \
                               ravel_ragged(data_perinfo[1][1][i])[bool_det][~bool_sign].astype(object)])],
                    dtype=object)

        # reconverting the array to bools and replacing it by true if it is empty to avoid conflicts when there is not data to plot
        # note : this doesn't work for arrays of length 1 so we add a reconversion when we use it just in case

        # resticting to cases where there's more than one resulting element
        if type(linked_mask[0]) != bool:
            linked_mask[0] = linked_mask[0].astype(bool)
        else:
            linked_mask[0] = np.array([linked_mask[0]])
        if type(linked_mask[1]) != bool:
            linked_mask[1] = linked_mask[1].astype(bool)
        else:
            linked_mask[1] = np.array([linked_mask[1]])

        # defining the jumps in indexes for linked values in the all line arrays, which is the number of lines shifted times the number of obs in
        # all current objects
        n_obs = len(ravel_ragged(data_perinfo[0][0][0]))
        jump_Kb = 3 * n_obs
        jump_Kg = 4 * n_obs

        # unused for now
        # #defining the number of lines to skip when not all lines are used
        # n_jump_Kg=len([elem for elem in mask_lines[:-1] if elem is True])
        # n_jump_Kb26=len([elem for elem in mask_lines[:-2] if elem is True])
        # n_jump_Kb25=len([elem for elem in mask_lines[:-3] if elem is True])

        # custom function so that the array indexation is not many lines long
        def linked_uncer(ind, ind_incer, ind_info):

            '''
            For all lines at once it is easier to use absolute index positions after developping the array
            We do the index changes before bool_det and bool_sign else we could lose the regularity in the shifts
            linked_ind is only applied in abslines_plot (and not the restricted array) because else the information of the line the current line
            is linked to can be missing if the other line is masked
            '''

            linked_line_ind_restrict = ind // n_obs

            # here we retrieve the index of the original line and translate it into a position for the non masked array
            linked_line_ind = np.argwhere(mask_lines)[linked_line_ind_restrict][0]

            linked_line_pos = ind % n_obs + n_obs * linked_line_ind

            '''
            Since the line shifts can be either positive or negative depending on if the first complexes are linked to the others or vice-versa, 
            we directly fetch all of the uncertainty values for each linked complex and pick the associated value
            We also make sure each line is linked properly by restricting the lines we check to the ones with the same base values
            '''

            ###TODO: update this for future linedets
            # #first we isolate the values for the same exposure
            # curr_exp_values=ravel_ragged(abslines_plot[ind_info][0].transpose(1,0,2)[mask_obj].transpose(1,0,2))[(ind//6)*6:(ind//6+1)*6]

            # then we identify

            if linked_line_ind > 2:
                val_line_pos = linked_line_pos - jump_Kg if linked_line_ind > 4 else linked_line_pos - jump_Kb
            else:
                # should not happen
                breakpoint()

            if ind_info == 2:
                # testing if there are 5 dimensions which would mean that we need to transpose with specific axes due to the array being regular
                if np.ndim(abslines_plot) == 5:
                    return ravel_ragged(abslines_ener[ind_incer].transpose(1, 0, 2)[mask_obj].transpose(1, 0, 2))[
                        val_line_pos]
                else:
                    return ravel_ragged(abslines_ener[ind_incer].T[mask_obj].T)[val_line_pos]
            else:
                if np.ndim(abslines_plot) == 5:
                    return \
                    ravel_ragged(abslines_plot[ind_info][ind_incer].transpose(1, 0, 2)[mask_obj].transpose(1, 0, 2))[
                        val_line_pos]
                else:
                    return ravel_ragged(abslines_plot[ind_info][ind_incer].T[mask_obj].T)[val_line_pos]

        #### error computation
        # (already done earlier in specific cases)

        # in ratio mode the errors are always for the ew and thus are simply computed from composing the uncertainties of each ew
        # then coming back to the ratio
        if ratio_mode:
            if time_mode:

                '''
                reminder of the logic since this is quite confusing
                the uncertainty is computed through the least squares because it's the correct formula for these kind of computation
                (correctly considers the skewness of the bivariate distribution)
                here, we can use ratio_indiexes_x[0] and ratio_indexes_x[1] independantly as long was we divide the uncertainty 
                ([i] which will either be [1] or [2]) by the main value [0]. However, as the uncertainties are asymetric,
                 each one must be paired with the opposite one to maintain consistency 
                (aka the + uncertaintiy of the numerator and the - incertainty of the denominator, when composed,
                 give the + uncertainty of the quotient)
                on top of that, the adressing is important as the minus uncertainty of the quotient needs to be first, 
                so the -num/+det must be first
                so for the first index in [1,2] which is i=1, we must have i for the num and 2 for the det
                the num is the one with ratio_indexes[0]
                '''

                y_error = np.array(
                    [[((ravel_ragged(data_plot[1][i][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask] / \
                        ravel_ragged(data_plot[1][0][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask]) ** 2 +
                       (ravel_ragged(data_plot[1][2 if i == 1 else 1][ratio_indexes_x[1]])[bool_det_ratio][
                            elem_sign_mask] / \
                        ravel_ragged(data_plot[1][0][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask]) ** 2) ** (
                                  1 / 2) *
                      y_data[i_sign] for i in [1, 2]] for i_sign, elem_sign_mask in
                     enumerate([bool_sign_ratio, ~bool_sign_ratio])], dtype=object)

            else:
                x_error = np.array(
                    [[((ravel_ragged(data_plot[0][i][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask] / \
                        ravel_ragged(data_plot[0][0][ratio_indexes_x[0]])[bool_det_ratio][elem_sign_mask]) ** 2 +
                       (ravel_ragged(data_plot[0][2 if i == 1 else 1][ratio_indexes_x[1]])[bool_det_ratio][
                            elem_sign_mask] / \
                        ravel_ragged(data_plot[0][0][ratio_indexes_x[1]])[bool_det_ratio][elem_sign_mask]) ** 2) ** (
                                  1 / 2) *
                      x_data[i_sign] for i in [1, 2]] for i_sign, elem_sign_mask in
                     enumerate([bool_sign_ratio, ~bool_sign_ratio])], dtype=object)

                if width_mode:
                    y_error = np.array([[elem if type(elem) != str else linked_uncer(ind_val, j, ind_infos[0]) \
                                         for ind_val, elem in enumerate(ravel_ragged(data_plot[1][j][width_line_id]))]
                                        for j in [1, 2]])

        # defining the errors and shifting the linked values accordingly for blueshifts
        # here we do not need the linked_ind shenanigans because this is not called during streamlit (and so all the lines are there)

        # THIS ALSO NEEDS TO CHANGE
        if indiv:
            x_error = np.array(
                [[elem if type(elem) != str else (ravel_ragged(data_plot[0][j][i - 3])[ind_val] if i < 5 else \
                                                      ravel_ragged(data_plot[0][1][i - 4])[ind_val]) for ind_val, elem
                  in enumerate(ravel_ragged(data_plot[0][j][i]))] for j in [1, 2]])

            if mode == 'intrinsic':
                y_error = np.array(
                    [[elem if type(elem) != str else (ravel_ragged(data_plot[1][j][i - 3])[ind_val] if i < 5 else \
                                                          ravel_ragged(data_plot[0][1][i - 4])[ind_val]) for
                      ind_val, elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1, 2]])

        elif not ratio_mode and not line_comp_mode:
            if time_mode:
                # swapping the error to the y axis
                y_error = np.array([[elem if type(elem) != str else linked_uncer(ind_val, j, ind_infos[1]) \
                                     for ind_val, elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1, 2]])
            else:
                x_error = np.array([[elem if type(elem) != str else linked_uncer(ind_val, j, ind_infos[0]) \
                                     for ind_val, elem in enumerate(ravel_ragged(data_plot[0][j][i]))] for j in [1, 2]])

            if mode == 'intrinsic':
                y_error = np.array([[elem if type(elem) != str else linked_uncer(ind_val, j, ind_infos[0]) \
                                     for ind_val, elem in enumerate(ravel_ragged(data_plot[1][j][i]))] for j in [1, 2]])

        '''
        in observ and source mode, we can compute the errors in the same formula in indiv mode or not since the values are the same for each line
        '''
        # note : all the cases of use of time mode are already created before
        if mode == 'observ' and not time_mode:

            # l is error index here
            y_err_repeat = np.array(
                [[data_plot[1][l] for repeater in (i if type(i) == range else [i])] for l in [1, 2]])

            # only then can the linked mask be applied correctly (doesn't change anything in individual mode)
            if ratio_mode:

                # here we just select one of all the lines in the repeat and apply the ratio_mode mask onto it
                y_error = np.array([[ravel_ragged(y_err_repeat[l][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio]
                                     for l in [0, 1]],
                                    [ravel_ragged(y_err_repeat[l][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio]
                                     for l in [0, 1]]], dtype=object)
            else:
                y_error = np.array([[ravel_ragged(y_err_repeat[l])[bool_det][bool_sign] for l in [0, 1]],
                                    [ravel_ragged(y_err_repeat[l])[bool_det][~bool_sign] for l in [0, 1]]],
                                   dtype=object)

        if mode == 'source':
            y_err_repeat = np.array([[data_plot[1][i_obj][l] for j in (i if type(i) == range else [i]) \
                                      for i_obj in range(n_obj) for i_obs in range(len(data_plot[0][0][j][i_obj]))] \
                                     for l in [1, 2]])

            y_error = np.array([[ravel_ragged(y_err_repeat[l])[bool_det][bool_sign] for l in [0, 1]],
                                [ravel_ragged(y_err_repeat[l])[bool_det][~bool_sign] for l in [0, 1]]],
                               dtype=object)

        # maybe missing something for time mode here
        ###TODO
        if show_ul_ew and not time_mode:
            if line_comp_mode:
                pass
                # note: pass because we rebuild it later anyway

            elif ratio_mode:
                # creation of y errors from both masks for upper and lower limits
                y_error_ul = np.array([ravel_ragged(y_err_repeat[l][ind_ratio[0]])[bool_nondetsign_x] for l in [0, 1]],
                                      dtype=object)
                y_error_ll = np.array([ravel_ragged(y_err_repeat[l][ind_ratio[1]])[bool_nondetsign_y] for l in [0, 1]],
                                      dtype=object)
            else:
                # creating y errors from the upper limit mask
                y_error_ul = np.array([ravel_ragged(y_err_repeat[l])[bool_nondetsign] for l in [0, 1]], dtype=object)

        # applying the det and sign masks
        if not line_comp_mode and not ratio_mode:

            if time_mode:
                y_error = [np.array([y_error[0][bool_det][bool_sign], y_error[1][bool_det][bool_sign]]),
                           np.array([y_error[0][bool_det][~bool_sign], y_error[1][bool_det][~bool_sign]])]
            else:

                if len(x_data[0]) > 0:
                    x_err_bounds = [x_error[0][bool_detsign_bounds], x_error[1][bool_detsign_bounds]]
                else:
                    x_err_bounds = None

                x_error = [np.array([x_error[0][bool_det][bool_sign], x_error[1][bool_det][bool_sign]]),
                           np.array([x_error[0][bool_det][~bool_sign], x_error[1][bool_det][~bool_sign]])]

        if mode == 'intrinsic':
            if width_mode and ratio_mode:
                y_error = [np.array(
                    [y_error[0][bool_det_ratio][bool_sign_ratio], y_error[1][bool_det_ratio][bool_sign_ratio]]).astype(
                    float),
                           np.array([y_error[0][bool_det_ratio][~bool_sign_ratio],
                                     y_error[1][bool_det_ratio][~bool_sign_ratio]]).astype(float)]
            else:
                y_error = [np.array([y_error[0][bool_det][bool_sign], y_error[1][bool_det][bool_sign]]).astype(float),
                           np.array([y_error[0][bool_det][~bool_sign], y_error[1][bool_det][~bool_sign]]).astype(float)]

        id_point_refit = 999999999999999
        # # changing the second brightest substructure point for the EW ratio plot to the manually fitted one
        # # with relaxed SAA filtering
        # #now modified directly in the files
        # if color_scatter == 'custom_line_struct' and 'ewratio' in infos_split[0]:
        #     x_point = 1.6302488583411436
        #     id_point_refit=np.argwhere(x_data[0]==x_point)[0][0]
        #
        #     #computed manually
        #     x_data[0][id_point_refit]=2.41
        #     x_error[0][0][id_point_refit]=1.09
        #     x_error[0][1][id_point_refit]=2.59

        # no need to do anything for y error in hid mode since it's already set to None

        data_cols = ['blue', 'grey']

        data_link_cols = ['dodgerblue', 'silver'] if show_linked else data_cols

        # data_labels=np.array(['detections above '+str(conf_thresh*100)+'% treshold','detections below '+str(conf_thresh*100)+'% treshold'])
        data_labels = ['', '']
        #### plots

        # plotting the unlinked and linked results with a different set of colors to highlight the differences
        # note : we don't use markers here since we cannot map their color

        if width_mode:
            # note: we only do this for significant unlinked detections
            uplims_mask = y_data[0].astype(float)[~(linked_mask[0].astype(bool))] == 0
        else:
            uplims_mask = None

        errbar_list = [
            '' if len(x_data[s]) == 0 else ax_scat.errorbar(x_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                                            y_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                                            xerr=None if x_error is None else \
                                                                np.array(
                                                                    [elem[~(linked_mask[s].astype(bool))] for elem in
                                                                     x_error[s]]).clip(0),
                                                            yerr=np.array(
                                                                [elem[~(linked_mask[s].astype(bool))] for elem in
                                                                 y_error[s]]).clip(0),
                                                            linewidth=1,
                                                            c=data_cols[s],
                                                            label=data_labels[s] if color_scatter == 'None' else '',
                                                            linestyle='',
                                                            uplims=uplims_mask,
                                                            marker='D' if color_scatter == 'None' else None,
                                                            alpha=None if color_scatter == 'custom_outburst' else 1,
                                                            zorder=1) \
            for s in ([0, 1] if display_nonsign else [0])]

        # note:deprecated
        errbar_list_linked = [
            '' if len(x_data[s]) == 0 else ax_scat.errorbar(x_data[s].astype(float)[linked_mask[s].astype(bool)],
                                                            y_data[s].astype(float)[linked_mask[s].astype(bool)],
                                                            xerr=None if x_error is None else \
                                                                np.array([elem[linked_mask[s].astype(bool)] for elem in
                                                                          x_error[s]]).clip(0),
                                                            yerr=np.array([elem[linked_mask[s].astype(bool)] for elem in
                                                                           y_error[s]]).clip(0),
                                                            linewidth=1,
                                                            color=data_link_cols[s], label='linked ' + data_labels[
                    s] if color_scatter == 'None' else '', linestyle='',
                                                            marker='D' if color_scatter == 'None' else None,
                                                            uplims=False,
                                                            alpha=None if color_scatter == 'custom_outburst' else 0.6 if show_linked else 1.0,
                                                            zorder=1000) \
            for s in ([0, 1] if display_nonsign else [0])]

        # locking the graph if asked to to avoid things going haywire with upper limits
        if lock_lims_det or not show_ul_ew:
            ax_scat.set_xlim(ax_scat.get_xlim())
            ax_scat.set_ylim(ax_scat.get_ylim())

        #### adding the absolute blueshift uncertainties for Chandra in the blueshift graphs

        # adding a line at 0 blueshift
        if infos_split[0] == 'bshift':
            if time_mode:
                ax_scat.axhline(y=0, xmin=0, xmax=1, color='grey', linestyle=':', lw=1.)
            else:
                ax_scat.axvline(x=0, ymin=0, ymax=1, color='grey', linestyle=':', lw=1.)

        # (-speed_abs_err[0],speed_abs_err[1],color='grey',label='Absolute error region',alpha=0.3)

        if display_std_abserr_bshift and infos_split[0] == 'bshift':

            # plotting the distribution mean and std (without the GRS exposures)
            v_mean = -200
            v_sigma = 360

            if time_mode:
                # reverses the axese so here we need to do it vertically (only case)
                # mean
                ax_scat.axhline(y=v_mean, xmin=0, xmax=1, color='brown', linestyle='-', label=r'$\overline{v}$',
                                lw=0.75)

                # span for the std
                ax_scat.axhspan(v_mean - v_sigma, v_mean + v_sigma, color='brown', label=r'$\sigma_v$', alpha=0.1)

            else:

                # mean
                ax_scat.axvline(x=v_mean, ymin=0, ymax=1, color='brown', linestyle='-', label=r'$\overline{v}$',
                                lw=0.75)

                # span for the std
                ax_scat.axvspan(v_mean - v_sigma, v_mean + v_sigma, color='brown', label=r'$\sigma_v$', alpha=0.1)

        if display_abserr_bshift and infos_split[0] == 'bshift':

            # computing the distribution mean and std
            if time_mode:
                bshift_data = y_data[0]
                bshift_err = y_error[0]
            else:
                bshift_data = x_data[0]
                bshift_err = x_error[0]

            # computing the mean and str
            bshift_mean = bshift_data.mean()
            bshift_sigma = bshift_data.std()
            bshift_mean_err = bshift_sigma / np.sqrt(len(bshift_data))

            # mean
            if time_mode:
                ax_scat.axhline(y=bshift_mean, xmin=0, xmax=1, color='orange', linestyle='-',
                                label=r'$\overline{v}_{curr}$',
                                lw=0.75)

                # span for the std
                ax_scat.axhspan(bshift_mean - bshift_sigma, bshift_mean - bshift_sigma, color='orange',
                                label=r'$\sigma_v_{curr}$',
                                alpha=0.1)

            else:
                ax_scat.axvline(x=bshift_mean, ymin=0, ymax=1, color='orange', linestyle='-',
                                label=r'$\overline{v}_{curr}$',
                                lw=0.75)

                # span for the std
                ax_scat.axvspan(bshift_mean - bshift_sigma, bshift_mean + bshift_sigma, color='orange',
                                label=r'$\sigma_{v_{curr}}$',
                                alpha=0.1)

            legend_title += r'$\overline{v}_{curr}=' + str(int(bshift_mean)) + '\pm' + str(int(bshift_mean_err)) + \
                            '$ \n $\sigma_{v_{curr}}=' + str(int(bshift_sigma)) + '$'
        #### Color definition

        # for the colors, we use the same logic than to create y_data with observation/source level parameters here

        # default colors for when color_scatter is set to None, will be overwriten
        color_arr = ['blue']
        color_arr_ul = 'black'
        color_arr_ul_x = color_arr_ul
        color_arr_ul_y = color_arr_ul

        if color_scatter == 'Instrument':

            # there's no need to repeat in ewratio since the masks are computed for a single line
            if line_comp_mode or ratio_mode:
                color_data_repeat = instru_list
            else:
                color_data_repeat = np.array([instru_list for repeater in (i if type(i) == range else [i])])

            if ratio_mode:
                color_data = np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                       ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]], dtype=object)

            else:

                color_data = np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],
                                       ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                      dtype=object)

                if not line_comp_mode:
                    # same thing for the upper limits
                    color_data_ul = ravel_ragged(color_data_repeat)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                # same thing for the upper limits in x and y
                color_data_ul_x = ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y = ravel_ragged(color_data_repeat)[bool_nondetsign_y]

            # computing the actual color array for the detections
            color_arr = np.array([np.array([telescope_colors[elem] for elem in color_data[s]]) for s in [0, 1]],
                                 dtype=object)

            # and for the upper limits if needed
            if show_ul_ew:

                if ratio_mode or line_comp_mode:
                    color_arr_ul_x = np.array([telescope_colors[elem] for elem in color_data_ul_x])

                    color_arr_ul_y = np.array([telescope_colors[elem] for elem in color_data_ul_y])

                else:
                    color_arr_ul = np.array([telescope_colors[elem] for elem in color_data_ul])

            # here we can keep a simple labeling
            label_dict = telescope_colors

        elif 'custom' in color_scatter:

            custom_color = diago_color if color_scatter == 'custom_line_struct' else \
                custom_states_color if color_scatter == 'custom_acc_states' else \
                    custom_ionization_color if color_scatter == 'custom_ionization' else \
                        custom_outburst_color if color_scatter == 'custom_outburst' else 'grey'

            # there's no need to repeat in ewratio since the masks are computed for a single line
            if line_comp_mode or ratio_mode:
                color_data_repeat = custom_color
            else:
                color_data_repeat = np.array([custom_color for repeater in (i if type(i) == range else [i])])

            if ratio_mode:
                color_data = np.array(
                    [ravel_ragged(color_data_repeat, ragtuples=False)[bool_det_ratio][bool_sign_ratio],
                     ravel_ragged(color_data_repeat, ragtuples=False)[bool_det_ratio][~bool_sign_ratio]], dtype=object)

            else:

                color_data = np.array([ravel_ragged(color_data_repeat, ragtuples=False)[bool_det][bool_sign],
                                       ravel_ragged(color_data_repeat, ragtuples=False)[bool_det][~bool_sign]],
                                      dtype=object)

                if not line_comp_mode:
                    # same thing for the upper limits
                    color_data_ul = ravel_ragged(color_data_repeat, ragtuples=False)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                # same thing for the upper limits in x and y
                color_data_ul_x = ravel_ragged(color_data_repeat, ragtuples=False)[bool_nondetsign_x]
                color_data_ul_y = ravel_ragged(color_data_repeat, ragtuples=False)[bool_nondetsign_y]

            # computing the actual color array for the detections
            color_arr = np.array([np.array([elem for elem in color_data[s]]) for s in [0, 1]],
                                 dtype=object)

            # and for the upper limits if needed
            if show_ul_ew:

                if ratio_mode or line_comp_mode:
                    color_arr_ul_x = np.array([elem for elem in color_data_ul_x])

                    color_arr_ul_y = np.array([elem for elem in color_data_ul_y])

                else:
                    color_arr_ul = np.array([elem for elem in color_data_ul])

            if color_scatter == 'custom_line_struct':
                label_dict = {'main structure': 'grey', 'substructure': 'orange', 'outliers': 'blue'}
            elif color_scatter == 'custom_acc_states':
                label_dict = {'undecided': 'grey',
                              'thermal dominated': 'green',
                              'intermediate': 'orange',
                              'SPL': 'red',
                              'canonical hard': 'blue',
                              'QRM': 'violet'}
            elif color_scatter == 'custom_ionization':
                label_dict = {'std': 'grey',
                              'outlier_diagonal_middle': 'red',
                              'outlier_diagonal_lower_floor': 'rosybrown',
                              'diagonal_lower_low_highE_flux': 'orange',
                              'diagonal_upper_mid_highE_flux': 'turquoise',
                              'diagonal_upper_high_highE_flux': 'pink',
                              'diagonal_upper_low_highE_flux': 'powderblue',
                              'SPL_whereline': 'forestgreen'}
            elif color_scatter == 'custom_outburst':
                # note: 0 here because this is only used in display_single mode
                label_dict = custom_outburst_dict[0]

        elif color_scatter in ['Time', 'HR', 'width', 'nH', 'L_3-10']:

            color_var_arr = date_list if color_scatter == 'Time' \
                else hid_plot[0][0] if color_scatter == 'HR' else hid_plot[1][0] if color_scatter == 'L_3-10' \
                else width_plot_restrict[0] if color_scatter == 'width' else nh_plot_restrict[0]

            # there's no need to repeat in ewratio since the masks are computed for a single line
            if color_scatter == 'width':
                color_data_repeat = color_var_arr
            else:
                if line_comp_mode or ratio_mode:
                    color_data_repeat = color_var_arr
                else:
                    color_data_repeat = np.array([color_var_arr for repeater in (i if type(i) == range else [i])])

            if ratio_mode:
                color_data = np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                       ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]], dtype=object)

            else:

                color_data = np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],
                                       ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                      dtype=object)

                if not line_comp_mode:
                    # same thing for the upper limits
                    color_data_ul = ravel_ragged(color_data_repeat)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                # same thing for the upper limits in x and y
                color_data_ul_x = ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y = ravel_ragged(color_data_repeat)[bool_nondetsign_y]

            # here we compute a conversion of the dates to numerical values in the case of a Time colormap
            if color_scatter == 'Time':
                c_arr = np.array([mdates.date2num(color_data[s]) for s in [0, 1]], dtype=object)

                if ratio_mode or line_comp_mode:
                    c_arr_ul_x = mdates.date2num(color_data_ul_x)
                    c_arr_ul_y = mdates.date2num(color_data_ul_y)

                else:
                    c_arr_ul = mdates.date2num(color_data_ul)
            else:
                c_arr = color_data

                if ratio_mode or line_comp_mode:
                    c_arr_ul_x = color_data_ul_x
                    c_arr_ul_y = color_data_ul_y
                else:
                    c_arr_ul = color_data_ul

            c_arr_tot = c_arr.tolist()

            # #adding the upper limits to the normalisation if necessary
            if show_ul_ew and ('ew' in infos or line_comp_mode):

                if line_comp_mode or ratio_mode:
                    c_arr_tot += [c_arr_ul_x, c_arr_ul_y]
                else:
                    c_arr_tot += [c_arr_ul]

                    # differing norms for Time and HR:
            if len(ravel_ragged(c_arr_tot)) > 0:
                if min(ravel_ragged(c_arr_tot)) == max(ravel_ragged(c_arr_tot)):
                    # safeguard to keep a middle range color when there's only one value
                    c_norm = mpl.colors.Normalize(vmin=min(ravel_ragged(c_arr_tot)) * 0.9,
                                                  vmax=max(ravel_ragged(c_arr_tot)) * 1.1)
                else:
                    if color_scatter in ['HR', 'nH', 'L_3-10']:
                        c_norm = colors.LogNorm(vmin=min(ravel_ragged(c_arr_tot)),
                                                vmax=max(ravel_ragged(c_arr_tot)))
                    else:

                        c_norm = mpl.colors.Normalize(vmin=min(ravel_ragged(c_arr_tot)),
                                                      vmax=max(ravel_ragged(c_arr_tot)))
            else:
                c_norm = None

            color_cmap = mpl.cm.plasma
            colors_func_date = mpl.cm.ScalarMappable(norm=c_norm, cmap=color_cmap)

            # computing the actual color array for the detections
            color_arr = np.array(
                [[colors_func_date.to_rgba(elem) for elem in c_arr[s]] for s in ([0, 1] if display_nonsign else [0])])

            # and for the upper limits
            if ratio_mode or line_comp_mode:

                # the axes swap in timemode requires swapping the indexes to fetch the uncertainty locations
                color_arr_ul_x = np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul_x])
                color_arr_ul_y = np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul_y])
            else:
                color_arr_ul = np.array([colors_func_date.to_rgba(elem) for elem in c_arr_ul])

        elif color_scatter == 'Source':

            # there's no need to repeat in ewratio since the masks are computed for a single line
            if line_comp_mode or ratio_mode:
                color_data_repeat = np.array([obj_disp_list[i_obj] for i_obj in range(n_obj) \
                                              for i_obs in range(len(data_perinfo[0][0][0][i_obj]))])

            else:
                color_data_repeat = np.array([obj_disp_list[i_obj] for repeater in (i if type(i) == range else [i]) \
                                              for i_obj in range(n_obj) for i_obs in
                                              range(len(data_perinfo[0][0][repeater][i_obj])) \
                                              ])

            if ratio_mode:

                color_data = np.array([ravel_ragged(color_data_repeat)[bool_det_ratio][bool_sign_ratio],
                                       ravel_ragged(color_data_repeat)[bool_det_ratio][~bool_sign_ratio]], dtype=object)

            else:

                color_data = np.array([ravel_ragged(color_data_repeat)[bool_det][bool_sign],
                                       ravel_ragged(color_data_repeat)[bool_det][~bool_sign]],
                                      dtype=object)

                if not line_comp_mode:
                    # same thing for the upper limits
                    color_data_ul = ravel_ragged(color_data_repeat)[bool_nondetsign]

            if line_comp_mode or ratio_mode:
                # same thing for the upper limits in x and y
                color_data_ul_x = ravel_ragged(color_data_repeat)[bool_nondetsign_x]
                color_data_ul_y = ravel_ragged(color_data_repeat)[bool_nondetsign_y]

            color_data_tot = [color_data[s] for s in ([0, 1] if display_nonsign else [0])]
            # global array for unique extraction
            if show_ul_ew and ('ew' in infos or line_comp_mode):

                if not ratio_mode and not line_comp_mode:
                    # same thing for the upper limits
                    color_data_tot += [color_data_ul]
                else:
                    # same thing for the upper limits in x and y
                    color_data_tot += [color_data_ul_x, color_data_ul_y]

            # we extract the number of objects with detection from the array of sources
            if glob_col_source:
                disp_objects = obj_disp_list
            else:
                disp_objects = np.unique(ravel_ragged(color_data_tot))

            # and compute a color mapping accordingly
            norm_colors_obj = mpl.colors.Normalize(vmin=0, vmax=len(disp_objects) - 1)
            colors_func_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_det)

            color_arr = np.array([colors_func_obj.to_rgba(np.argwhere(disp_objects == elem)[0][0]) for s in
                                  ([0, 1] if display_nonsign else [0]) for elem in color_data[s]])

            label_dict = {disp_objects[i]: colors_func_obj.to_rgba(i) for i in range(len(disp_objects))}

            if not display_nonsign:
                color_arr = np.array([color_arr, 'None'], dtype=object)

            # same for the upper limits if needed
            if show_ul_ew and ('ew' in infos or line_comp_mode):

                if ratio_mode or line_comp_mode:
                    color_arr_ul_x = np.array(
                        [colors_func_obj.to_rgba(np.argwhere(disp_objects == elem)[0][0]) for elem in color_data_ul_x])

                    color_arr_ul_y = np.array(
                        [colors_func_obj.to_rgba(np.argwhere(disp_objects == elem)[0][0]) for elem in color_data_ul_y])

                else:

                    color_arr_ul = np.array(
                        [colors_func_obj.to_rgba(np.argwhere(disp_objects == elem)[0][0]) for elem in color_data_ul])

        def plot_ul_err(lims_bool, x_data, y_data, x_err, y_err, col, label=''):

            # this construction allows us to maintain a string color name when not using color_scatter

            for i_err, (x_data_single, y_data_single, x_err_single, y_err_single, col_single) in \
                    enumerate(zip(x_data, y_data, x_err, y_err,
                                  col if color_scatter != 'None' else np.repeat(col, len(x_data)))):
                ax_scat.errorbar(x_data_single, y_data_single,
                                 xerr=np.array([x_err_single]).T if x_err_single is not None else x_err_single,
                                 yerr=np.array([y_err_single]).T if y_err_single is not None else y_err_single,
                                 xuplims=lims_bool[0] == 1, xlolims=lims_bool[0] == -1, uplims=lims_bool[1] == 1,
                                 lolims=lims_bool[1] == -1,
                                 color=col_single, linestyle='', marker='.' if color_scatter == 'None' else None,
                                 label=label if i_err == 0 else '', alpha=alpha_ul)

        ####plotting upper limits
        if show_ul_ew and ('ew' in infos or line_comp_mode):

            if line_comp_mode:

                # inverted because here we're not primarily using the x axis for the ratio
                # xuplims here
                plot_ul_err([1, 0], y_data_ul_x, x_data_ul_x, y_data_ul_x * 0.05, y_error_ul_x.T, color_arr_ul_x)

                # uplims here
                plot_ul_err([0, 1], y_data_ul_y, x_data_ul_y, x_error_ul_y.T, x_data_ul_y * 0.05, color_arr_ul_y)

            else:

                # else:
                #     ax_scat.set_xlim(mdates.date2num(['2012-01-01']),mdates.date2num(['2012-10-01']))

                if time_mode:
                    # uplims here, the upper limits display has the same construction no matter if in ratio mode or not
                    plot_ul_err([0, 1], x_data_ul, y_data_ul, [None] * len(x_data_ul),
                                y_data_ul * 0.05, color_arr_ul_x if ratio_mode else color_arr_ul)

                else:

                    # xuplims here, the upper limits display has the same construction no matter if in ratio mode or not
                    plot_ul_err([1, 0], x_data_ul, y_data_ul, x_data_ul * 0.05,
                                y_error_ul.T, color_arr_ul_x if ratio_mode else color_arr_ul)

                # adding the lower limits in ratio mode
                if ratio_mode:

                    if time_mode:
                        # lolims here
                        plot_ul_err([0, -1], x_data_ll, y_data_ll, [None] * len(x_data_ll), y_data_ll * 0.05,
                                    color_arr_ul_y)
                    else:
                        # lolims here
                        plot_ul_err([-1, 0], x_data_ll, y_data_ll, x_data_ll * 0.05, y_error_ll.T, color_arr_ul_y)

        if time_mode:
            # creating an appropriate date axis
            # manually readjusting for small durations because the AutoDateLocator doesn't work well
            if time_range < 10:
                date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            elif time_range < 365:
                date_format = mdates.DateFormatter('%Y-%m-%d')
            else:
                date_format = mdates.DateFormatter('%Y-%m')

            ax_scat.xaxis.set_major_formatter(date_format)

            # forcing 8 xticks along the ax
            ax_scat.set_xlim(ax_scat.get_xlim())

            n_xticks = 6
            # putting an interval in seconds (to avoid imprecisions when zooming)
            date_tick_inter = int((ax_scat.get_xlim()[1] - ax_scat.get_xlim()[0]) * 24 * 60 * 60 / n_xticks)

            # 10 days
            if date_tick_inter > 60 * 60 * 24 * 10:
                ax_scat.xaxis.set_major_locator(mdates.DayLocator(interval=date_tick_inter // (24 * 60 * 60)))
            # 1 day
            elif date_tick_inter > 60 * 60 * 24:
                ax_scat.xaxis.set_major_locator(mdates.HourLocator(interval=date_tick_inter // (60 * 60)))
            elif date_tick_inter > 60 * 60:
                ax_scat.xaxis.set_major_locator(mdates.MinuteLocator(interval=date_tick_inter // (60)))
            else:
                ax_scat.xaxis.set_major_locator(mdates.SecondLocator(interval=date_tick_inter))

            # and offsetting if they're too close to the bounds because otherwise the ticks can be missplaced
            if ax_scat.get_xticks()[0] - ax_scat.get_xlim()[0] > date_tick_inter / (24 * 60 * 60) * 3 / 4:
                ax_scat.set_xticks(ax_scat.get_xticks() - date_tick_inter / (2 * 24 * 60 * 60))

            if ax_scat.get_xticks()[0] - ax_scat.get_xlim()[0] < date_tick_inter / (24 * 60 * 60) * 1 / 4:
                ax_scat.set_xticks(ax_scat.get_xticks() + date_tick_inter / (2 * 24 * 60 * 60))

            # ax_lc.set_xticks(ax_lc.get_xticks()[::2])

            for label in ax_scat.get_xticklabels(which='major'):
                label.set(rotation=0 if date_tick_inter > 60 * 60 * 24 * 10 else 45, horizontalalignment='center')

        #### adding cropping on the EW ratio X axis to fix unknown issue

        # complicated restriction to take off all the elements of x_data no matter their dimension if they are empty arrays
        x_data_use = [elem for elem in x_data if len(np.array(np.shape(elem)).nonzero()[0]) == np.ndim(elem)]

        if ratio_mode and len(x_data) > 0:

            if len(x_data_use) != 0:
                if width_mode:
                    if min(ravel_ragged(x_data_use)) > 0.28 and max(ravel_ragged(x_data_use)) < 3.5:
                        ax_scat.set_xlim(0.28, 3.5)
                elif time_mode:
                    if min(ravel_ragged(x_data_use)) > 0.28:
                        ax_scat.set_ylim(0.28, ax_scat.get_ylim()[1])
                else:
                    if min(ravel_ragged(x_data_use)) > 0.28:
                        ax_scat.set_xlim(0.28, ax_scat.get_xlim()[1])

        #### Color replacements in the scatter to match the colormap
        if color_scatter != 'None':

            for s, elem_errbar in enumerate(errbar_list):
                # replacing indiviudally the colors for each point but the line

                # here the empty element is '' so we use a type comparison

                for elem_children in ([] if type(elem_errbar) == str else elem_errbar.get_children()[1:]):

                    if type(elem_children) == mpl.collections.LineCollection:

                        elem_children.set_colors(color_arr[s][~(linked_mask[s].astype(bool))])

                        # highlighting BAT/INT projected luminosities without having to worry about how the graphs
                        # were created
                        if mode == 'observ' and ind_infos[1] in [4, 5] and mask_added_regr_sign_use is not None:

                            ls_dist = np.array([np.where(bool, '--', '-') for bool in mask_added_regr_sign_use])

                            # older version
                            # ls_dist=np.array([np.where((elem in bat_lc_lum_scat.T[0] if ind_infos[1]==4 else bool),'--','-')\
                            #                   for elem,bool in zip(y_data[s],mask_added_regr_sign_use)])

                            ls_dist = ls_dist.tolist()

                            elem_children.set_linestyles(ls_dist)

                        #### distinguishing the 3 GRS obs
                        elif 'bshift' in infos_split:

                            data_bshift = np.array([x_data[s], y_data[s]])[np.array(infos_split) == 'bshift'][s]

                            ls_dist = np.repeat('-', len(x_data[s]))
                            # to avoid issues due to the type of the array
                            ls_dist = ls_dist.tolist()

                            facecol_dist = np.repeat('full', len(x_data[s]))

                            facecol_dist = facecol_dist.tolist()

                            bshift_obscured = [948.903977133402, 1356.3406485107535, 2639.060062961778]

                            index_bshift_GRS = np.argwhere([elem in bshift_obscured for elem in data_bshift]).T[0]

                            for i_obscured in index_bshift_GRS:
                                ls_dist[i_obscured] = '--'
                                facecol_dist[i_obscured] = 'none'

                            elem_children.set_linestyles(ls_dist)

                        # deprecated
                        # #dashing the errorbar for the 4U point with adjusted manual values for the EW ratio
                        # elif color_scatter == 'custom_line_struct' and 'ewratio' in infos_split[0]:
                        #     ls_dist=np.repeat('-',len(x_data[s]))
                        #     #to avoid issues due to the type of the array
                        #     ls_dist = ls_dist.tolist()
                        #
                        #     ls_dist[id_point_refit]='--'
                        #
                        #     elem_children.set_linestyles(ls_dist)

                    else:
                        elem_children.set_visible(False)

                        # this case is restricted to the upper limits in width mode
                        x_ul = x_data[0].astype(float)[~(linked_mask[0].astype(bool))][uplims_mask]

                        # using the max values (in yerr) as the upper limits so in y
                        y_ul = yerr_ul = \
                        np.array([elem[~(linked_mask[s].astype(bool))][uplims_mask] for elem in y_error[0]])[1]

                        xerr_ul = None if x_error is None else np.array(
                            [elem[~(linked_mask[s].astype(bool))][uplims_mask] for elem in x_error[0]]).T
                        yerr_ul = y_ul * 0.05

                        col_ul = color_arr[0][~(linked_mask[0])][uplims_mask]
                        plot_ul_err([0, 1], x_ul, y_ul, xerr_ul, yerr_ul, col_ul, label='')

            for s, elem_errbar_linked in enumerate(errbar_list_linked):
                # replacing indiviudally the colors for each point but the line

                for elem_children in ([] if type(elem_errbar_linked) == str else elem_errbar_linked.get_children()[1:]):
                    elem_children.set_colors(color_arr[s][(linked_mask[s].astype(bool))])

            if color_scatter in ['Time', 'HR', 'width', 'nH', 'L_3-10']:

                # adding space for the colorbar
                # breaks everything so not there currently
                # ax_cb=fig_scat.add_axes([0.99, 0.123, 0.02, 0.84 if not compute_correl else 0.73])
                # ax_cb = fig_scat.add_axes([0.99, 0.123, 0.02, 0.84])
                # divider = make_axes_locatable(ax_scat)
                # ax_cb = divider.append_axes('right', size='5%', pad=0.05)

                # scatter plot on top of the errorbars to be able to map the marker colors and create the colormap

                # no labels needed here
                scat_list = [ax_scat.scatter(x_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                             y_data[s].astype(float)[~(linked_mask[s].astype(bool))],
                                             c=c_arr[s][~(linked_mask[s].astype(bool))], cmap=color_cmap, norm=c_norm,
                                             marker='D', alpha=1, zorder=1) \
                             for s in ([0, 1] if display_nonsign else [0])]

                scat_list_linked = [ax_scat.scatter(x_data[s].astype(float)[(linked_mask[s].astype(bool))],
                                                    y_data[s].astype(float)[(linked_mask[s].astype(bool))],
                                                    c=c_arr[s][(linked_mask[s].astype(bool))], cmap=color_cmap,
                                                    norm=c_norm, marker='D', alpha=1, zorder=1) \
                                    for s in ([0, 1] if display_nonsign else [0])]

                # forcing a common normalisation for all the scatters
                scat_list_tot = scat_list + scat_list_linked

                for elem_scat in scat_list_tot:

                    if len(ravel_ragged(c_arr_tot)) > 0:
                        elem_scat.set_clim(min(ravel_ragged(c_arr_tot)), max(ravel_ragged(c_arr_tot)))

                if color_scatter == 'Time':
                    # creating the colormap

                    # manually readjusting for small durations because the AutoDateLocator doesn't work well
                    time_range = max(mdates.date2num(ravel_ragged(date_list)[mask_intime_norepeat])) \
                                 - min(mdates.date2num(ravel_ragged(date_list)[mask_intime_norepeat]))

                    if time_range < 150:
                        date_format = mdates.DateFormatter('%Y-%m-%d')
                    elif time_range < 1825:
                        date_format = mdates.DateFormatter('%Y-%m')
                    else:
                        date_format = mdates.AutoDateFormatter(mdates.AutoDateLocator())

                    # test = fig_scat.colorbar(scat_list[0],cax=ax_cb,format=mdates.DateFormatter('%Y-%m'))

                    ####TODO: reintroduce cax to get constant ax size
                    test = plt.colorbar(scat_list[0], ticks=mdates.AutoDateLocator(), format=date_format)

                elif color_scatter in ['HR', 'width', 'nH', 'L_3-10']:

                    # creating the colormap (we add a bottom extension for nH to indicate the values cut)
                    test = plt.colorbar(scat_list[0],
                                        label=r'nH ($10^{22}$ cm$^{-2}$)' if color_scatter == 'nH' else color_scatter,
                                        extend='min' if color_scatter == 'nH' else None, aspect=30)


            else:

                # scatter plot on top of the errorbars to be able to map the marker colors
                # The loop allows to create scatter with colors according to the labels
                for s in ([0, 1] if display_nonsign else [0]):

                    # for this we must split differently to ensure we keep the labeling for the
                    # first point displayed in each outburst while keeping the alpha colors for everything
                    # we do it now to make a dictionnary containing only the currently displayed points
                    if color_scatter == 'custom_outburst':

                        # creating a new dictionnary
                        label_dict_use = {}

                        outburst_mask_list = []
                        outburst_order_list = []
                        for outburst in list(label_dict.keys()):
                            # selecting the obs part of this outburst
                            outburst_select_mask = [elem.all() for elem in
                                                    (color_arr[s].T[:-1].T == label_dict[outburst][:-1])]

                            # saving the outburst masks for connectors
                            outburst_mask_list += [outburst_select_mask]

                            # skipping outbursts with no observations
                            if sum(outburst_select_mask) == 0:
                                continue

                            color_arr_outburst = color_arr[s][outburst_select_mask]
                            # computing the first point of the outburst (aka darkest) from the alpha orders
                            # we invert it to get the argosrt from the first to the last
                            outburst_order = (1 - color_arr_outburst.T[-1]).argsort()

                            outburst_order_list += [outburst_order]

                            outburst_init_id = outburst_order[0]

                            for i in range(len(color_arr_outburst)):
                                if outburst_init_id == i:
                                    # giving the outburst name
                                    label_dict_use[outburst] = color_arr_outburst[i]

                                # only adding dictionnary entries for element with different dates/colors
                                elif not np.array([(color_arr_outburst[i] == elem).all() \
                                                   for elem in np.array(list(label_dict_use.values()))]).any():
                                    # giving a modified outburst name with a . to now it shouldn't be displayed
                                    # point number in the displayed ones
                                    outburst_id = np.argwhere(outburst_order == i)[0][0]
                                    label_dict_use[outburst + '.' + str(outburst_id)] = color_arr_outburst[i]

                            # putting the connector with the outburst color
                            # note that we overwrite the alpha so we don't care about which color we're taking

                            x_outburst = x_data[s].astype(float)[~(linked_mask[s].astype(bool))] \
                                [outburst_select_mask][outburst_order]
                            y_outburst = y_data[s].astype(float)[~(linked_mask[s].astype(bool))] \
                                [outburst_select_mask][outburst_order]
                            ax_scat.plot(x_outburst, y_outburst,
                                         color=color_arr_outburst[0],
                                         label='', alpha=0.3, zorder=0)

                            xscale_lin = ax_scat.get_xscale() == 'linear'
                            yscale_lin = ax_scat.get_yscale() == 'linear'
                            # and arrows, using annotate to avoid issues with ax limits and resizing

                            x_pos = (x_outburst[1:] + x_outburst[:-1]) / 2 if xscale_lin else \
                                10 ** ((np.log10(x_outburst[1:]) + np.log10(x_outburst[:-1])) / 2)

                            y_pos = (y_outburst[1:] + y_outburst[:-1]) / 2 if yscale_lin else \
                                10 ** ((np.log10(y_outburst[1:]) + np.log10(y_outburst[:-1])) / 2)

                            x_dir = x_outburst[1:] - x_outburst[:-1] if xscale_lin else \
                                10 ** (np.log10(x_outburst[1:]) - np.log10(x_outburst[:-1]))
                            y_dir = y_outburst[1:] - y_outburst[:-1] if yscale_lin else \
                                10 ** (np.log10(y_outburst[1:]) - np.log10(y_outburst[:-1]))

                            # this is the offset from the position, since we make the arrow start at the middle point of
                            # the segment we don't want it to go any further so we put it almost at the same value
                            # note that for some reason this ends up with non uniform proportions in log scale, but
                            # that remains good enough for now

                            arrow_size_frac = 0.1

                            for X, Y, dX, dY in zip(x_pos, y_pos, x_dir, y_dir):
                                ax_scat.annotate("", xytext=(X, Y),
                                                 xy=(X + arrow_size_frac * dX if xscale_lin else \
                                                         10 ** (np.log10(X) + arrow_size_frac * np.log10(dX)),
                                                     Y + arrow_size_frac * dY if yscale_lin else \
                                                         10 ** (np.log10(Y) + arrow_size_frac * np.log10(dY))),
                                                 arrowprops=dict(arrowstyle='->', color=color_arr_outburst[0],
                                                                 alpha=0.3), size=10)

                        # replacing the dictionnary with this one
                        label_dict = label_dict_use
                        order_disp = np.array(list(label_dict.keys())).argsort()




                    else:
                        order_disp = np.arange(len(np.array(list(label_dict.keys()))))

                    for i_col, color_label in enumerate(np.array(list(label_dict.keys()))[order_disp]):

                        # creating a mask for the points of the right color
                        if color_scatter == 'Instrument':

                            color_mask = [(elem == label_dict[color_label]).all() for elem in
                                          color_arr[s][~(linked_mask[s].astype(bool))]]

                            color_mask_linked = [(elem == label_dict[color_label]).all() for elem in
                                                 color_arr[s][(linked_mask[s].astype(bool))]]

                        elif 'custom' in color_scatter:

                            if color_scatter == 'custom_outburst':
                                color_mask = [(elem == label_dict[color_label]).all() for elem in
                                              color_arr[s][~(linked_mask[s].astype(bool))]]

                                color_mask_linked = [(elem == label_dict[color_label]).all() for elem in
                                                     color_arr[s][(linked_mask[s].astype(bool))]]
                            else:

                                color_mask = [elem == label_dict[color_label] for elem in
                                              color_arr[s][~(linked_mask[s].astype(bool))]]

                                color_mask_linked = [elem == label_dict[color_label] for elem in
                                                     color_arr[s][(linked_mask[s].astype(bool))]]

                        # same idea but here since the color is an RGB tuple we need to convert the element before the comparison
                        elif color_scatter == 'Source':
                            color_mask = [tuple(elem) == label_dict[color_label] for elem in
                                          color_arr[s][~(linked_mask[s].astype(bool))]]
                            color_mask_linked = [tuple(elem) == label_dict[color_label] for elem in
                                                 color_arr[s][(linked_mask[s].astype(bool))]]

                        # checking if there is at least one upper limit:
                        # (a bit convoluted but we cannot concatenate 0 len arrays so we add a placeholder that'll never get recognized instead)

                        col_concat = (color_arr_ul_x.tolist() if type(color_arr_ul_x) != str else [] + \
                                                                                                  color_arr_ul_y.tolist() if type(
                            color_arr_ul_y) != str else []) \
                            if (ratio_mode or line_comp_mode) else color_arr_ul

                        no_ul_displayed = np.sum([tuple(elem) == label_dict[color_label] for elem in col_concat]) == 0

                        # not displaying color/labels that are not actually in the plot
                        if np.sum(color_mask) == 0 and np.sum(color_mask_linked) == 0 and (
                                not show_ul_ew or no_ul_displayed):
                            continue

                        # needs to be split to avoid indexation problem when calling color_mask behind
                        if uplims_mask is None:

                            # adding the marker color change for the BAT infered obs
                            if mode == 'observ' and ind_infos[1] in [4, 5] and mask_added_regr_sign_use is not None:

                                facecol_adjust_mask = mask_added_regr_sign_use[color_mask]

                                # older version
                                # facecol_adjust_mask = np.array([(elem in bat_lc_lum_scat.T[0]) if ind_infos[1]==4\
                                #                                 else bool\
                                #                     for elem,bool in zip(y_data[s][color_mask],
                                #                                          mask_added_regr_sign_use[color_mask])])

                                for id_obs in range(len(y_data[s][color_mask])):
                                    ax_scat.scatter(
                                        x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                                        y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                                        color=label_dict[color_label] if not facecol_adjust_mask[id_obs] else None,
                                        facecolor='none' if facecol_adjust_mask[id_obs] else None,
                                        label=color_label if id_obs == 0 else '',
                                        marker='D',
                                        edgecolor=label_dict[color_label] if facecol_adjust_mask[id_obs] else None,
                                        alpha=1, zorder=1)

                            # adding the marker color change for the obscured obs
                            elif 'bshift' in infos_split:

                                data_bshift_GRS = np.array([x_data[s][color_mask], y_data[s][color_mask]])[
                                    np.array(infos_split) == 'bshift'][s]

                                index_bshift_GRS_indiv = \
                                np.argwhere([elem in bshift_obscured for elem in data_bshift_GRS]).T[0]

                                for id_GRS in range(len(data_bshift_GRS)):
                                    ax_scat.scatter(
                                        x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_GRS],
                                        y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_GRS],
                                        color=label_dict[color_label] if id_GRS not in index_bshift_GRS_indiv else None,
                                        facecolor=None if id_GRS not in index_bshift_GRS_indiv else 'none',
                                        label='' if color_scatter == 'custom_outburst' and "." in color_label else \
                                            color_label if id_GRS == 0 else '',
                                        marker='D', edgecolor=None if id_GRS not in index_bshift_GRS_indiv else \
                                            label_dict[color_label],
                                        alpha=None if color_scatter == 'custom_outburst' else 1, zorder=1)

                            # # changing the marker color for the 4U point with adjusted manual values for the EW ratio
                            # elif  color_scatter == 'custom_line_struct' and 'ewratio' in infos_split[0]:
                            #
                            #     for id_obs,id_obs_full in enumerate(np.arange(len(y_data[s]))[color_mask]):
                            #         ax_scat.scatter(
                            #             x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                            #             y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask][id_obs],
                            #             color=label_dict[color_label] if id_obs_full!=id_point_refit else None,
                            #             facecolor='none' if id_obs_full==id_point_refit else None,
                            #             label=color_label if id_obs == 0 else '',
                            #             marker='D',
                            #             edgecolor=label_dict[color_label] if id_obs_full==id_point_refit else None,
                            #             alpha=1, zorder=1)

                            else:
                                ax_scat.scatter(
                                    x_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask],
                                    y_data[s].astype(float)[~(linked_mask[s].astype(bool))][color_mask],
                                    color=label_dict[color_label],
                                    label='' if color_scatter == 'custom_outburst' and "." in color_label else color_label,
                                    marker='D', alpha=None if color_scatter == 'custom_outburst' else 1, zorder=1)

                        else:
                            ax_scat.scatter(
                                x_data[s].astype(float)[~(linked_mask[s].astype(bool))][~uplims_mask][
                                    np.array(color_mask)[~uplims_mask]],
                                y_data[s].astype(float)[~(linked_mask[s].astype(bool))][~uplims_mask][
                                    np.array(color_mask)[~uplims_mask]],
                                color=label_dict[color_label],
                                label='' if color_scatter == 'custom_outburst' and "." in color_label else color_label,
                                marker='D', alpha=None if color_scatter == 'custom_outburst' else 1, zorder=1)

                        # No label for the second plot to avoid repetitions
                        ax_scat.scatter(x_data[s].astype(float)[(linked_mask[s].astype(bool))][color_mask_linked],
                                        y_data[s].astype(float)[(linked_mask[s].astype(bool))][color_mask_linked],
                                        color=label_dict[color_label],
                                        label='', marker='D', alpha=1, zorder=1)

        # ax_scat.set_xlim(mdates.date2num(['2012-01-01']),mdates.date2num(['2012-10-01']))

        # adjusting the axis sizes for ewratio mode to get the same scale
        if line_comp_mode:
            ax_scat.set_ylim(max(min(ax_scat.get_xlim()[0], ax_scat.get_ylim()[0]), 0),
                             max(ax_scat.get_xlim()[1], ax_scat.get_ylim()[1]))
            ax_scat.set_xlim(max(min(ax_scat.get_xlim()[0], ax_scat.get_ylim()[0]), 0),
                             max(ax_scat.get_xlim()[1], ax_scat.get_ylim()[1]))
            ax_scat.plot(ax_scat.get_xlim(), ax_scat.get_ylim(), ls='--', color='grey')


        # resizing for a square graph shape
        # forceAspect(ax_scat, aspect=0.8)

        #### theoretical line drawing for ew_width
        if infos == 'ew_width' and display_th_width_ew:

            # adding a blank line in the legend for cleaner display
            ax_scat.plot(np.NaN, np.NaN, '-', color='none', label='   ')

            line_id = np.argwhere(np.array(mask_lines))[0][0]
            breakpoint()
            #should be updated because this arborescence is likely outdated
            if line_id == 0:
                files = glob.glob('/home/'+username+'/Documents/Work/PhD/docs/atom/Fe25*_*.dat')
            elif line_id == 1:
                files = glob.glob('/home/'+username+'/Documents/Work/PhD/docs/atom/Fe26*_*.dat')

            nh_values = np.array([elem.split('/')[-1].replace('.dat', '').split('_')[1] for elem in files]).astype(
                float)

            nh_order = nh_values.argsort()

            nh_values.sort()

            ls_curve = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]

            colors_curve = ['black', 'black', 'black', 'black', 'black']

            for id_curve, nh_curve_path in enumerate(np.array(files)[nh_order]):
                nh_array = np.array(pd.read_csv(nh_curve_path, sep='      ', engine='python', header=None)).T

                ax_scat.plot(nh_array[0], nh_array[1], label=r'$N_i= $' + str(nh_values[id_curve]) + ' cm' + r'$^{-2}$',
                             color=colors_curve[id_curve],
                             alpha=0.5, ls=ls_curve[id_curve % len(ls_curve)])

        # logarithmic scale by default for ew ratios
        if ratio_mode:

            if time_mode:
                ax_scat.set_yscale('log')
                ax_scat.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                ax_scat.set_yticks(np.arange(0.3, 0.8, 0.2).tolist() + [1, 3, 5])

                ax_scat.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

            else:
                ax_scat.set_xscale('log')
                ax_scat.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                ax_scat.set_xticks(np.arange(0.3, 0.8, 0.2).tolist() + [1, 3, 5])

                ax_scat.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        ####forcing common observ bounds if asked
        # computing the common bounds if necessary
        if common_observ_bounds_lines:
            if mode == 'observ' and sum(mask_bounds) > 0 and not time_mode:
                plt.ylim(bounds_hr if infos_split[1] == 'HR' else bounds_flux if infos_split[1] == 'flux' else None)

        if common_observ_bounds_dates:
            if mode == 'observ' and not ratio_mode and not time_mode and x_data_bounds is not None:
                # for linspace only for now
                delta_x = (max(x_data_bounds + x_err_bounds[1]) - min(x_data_bounds - x_err_bounds[0])) * 0.05

                plt.xlim(min(x_data_bounds - x_err_bounds[0]) - delta_x, max(x_data_bounds + x_err_bounds[1]) + delta_x)

        #### Correlation values

        restrict_comput_mask = np.repeat(True, len(x_data[0]))

        if restrict_comput_scatter:
            # each "non-applicability" condition is having the same upper and lower limit value
            if comput_scatter_lims[0][0] != comput_scatter_lims[0][1]:
                restrict_comput_mask = (restrict_comput_mask) & (x_data[0] >= comput_scatter_lims[0][0]) \
                                       & (x_data[0] <= comput_scatter_lims[0][1])
            if comput_scatter_lims[1][0] != comput_scatter_lims[1][1]:
                restrict_comput_mask = (restrict_comput_mask) & (y_data[0] >= comput_scatter_lims[1][0]) \
                                       & (y_data[0] <= comput_scatter_lims[1][1])

            if sum(restrict_comput_mask) == 0:
                st.warning('No points remaining in trend bounds. Skipping bounds... ')
                restrict_comput_mask = np.repeat(True, len(x_data[0]))

        # giving default values to pearson and spearman to avoid issues when rerunning streamlit rapidly
        r_pearson = np.zeros(4).reshape(2, 2)
        r_spearman = np.zeros(4).reshape(2, 2)

        # computing the statistical coefficients for the significant portion of the sample (we don't care about the links here)
        if compute_correl and mode != 'source' and len(x_data[0]) > 1 and not time_mode:

            x_data_trend = x_data[0][restrict_comput_mask]
            y_data_trend = y_data[0][restrict_comput_mask]

            # we cannot transpose the whole arrays since in ewcomp/HID mode they are most likely ragged due to having both
            # the significant and unsignificant data, so we create regular versions manually
            x_error_sign_T = None if x_error is None else np.array([elem for elem in x_error[0]]).T[
                restrict_comput_mask]
            y_error_sign_T = None if y_error is None else np.array([elem for elem in y_error[0]]).T[
                restrict_comput_mask]

            # note: general p-score conversion: 1/scinorm.ppf((1 + error_percent/100) / 2)

            if display_pearson:
                r_pearson = np.array(
                    pymccorrelation(x_data_trend.astype(float), y_data_trend.astype(float),
                                    dx_init=x_error_sign_T / 1.65, dy_init=y_error_sign_T / 1.65,
                                    ylim_init=uplims_mask,
                                    Nperturb=npert_rank, coeff='pearsonr', percentiles=(50, 5, 95)))

                # switching back to uncertainties from quantile values
                r_pearson = np.array([[r_pearson[ind_c][0], r_pearson[ind_c][0] - r_pearson[ind_c][1],
                                       r_pearson[ind_c][2] - r_pearson[ind_c][0]] \
                                      for ind_c in [0, 1]])

                # no need to put uncertainties now
                # str_pearson=r'$r_{Pearson}='+str(round(r_pearson[0][0],2))+'_{-'+str(round(r_pearson[0][1],2))+'}^{+'\
                #                         +str(round(r_pearson[0][2],2))+'}$ '+'| $p='+'%.1e' % Decimal(r_pearson[1][0 ])\
                #                         +'_{-'+'%.1e' % Decimal(r_pearson[1][1])+'}^{+'+'%.1e' % Decimal(r_pearson[1][2])+'}$\n'

                str_pearson = r'$r_{Pearson}=$' + str(round(r_pearson[0][0], 2)) + '$\;|\; $p=' + '%.1e' % Decimal(
                    r_pearson[1][0]) + '$\n'
            else:
                str_pearson = ''

            r_spearman = np.array(
                pymccorrelation(x_data_trend.astype(float), y_data_trend.astype(float),
                                dx_init=x_error_sign_T / 1.65, dy_init=y_error_sign_T / 1.65,
                                ylim_init=uplims_mask,
                                Nperturb=npert_rank, coeff='spearmanr', percentiles=(50, 5, 95)))

            # switching back to uncertainties from quantile values
            r_spearman = np.array([[r_spearman[ind_c][0], r_spearman[ind_c][0] - r_spearman[ind_c][1],
                                    r_spearman[ind_c][2] - r_spearman[ind_c][0]] \
                                   for ind_c in [0, 1]])

            # str_spearman=r'$r_{Spearman}='+str(round(r_spearman[0][0],2))+'_{-'+str(round(r_spearman[0][1],2))\
            #                           +'}^{+'+str(round(r_spearman[0][2],2))+'}$ '+'| $p='+'%.1e' % Decimal(r_spearman[1][0])\
            #                           +'_{-'+'%.1e' % Decimal(r_spearman[1][1])+'}^{+'+'%.1e' % Decimal(r_spearman[1][2])+'}$ '

            str_spearman = r'$r_S \,=' + str(round(r_spearman[0][0], 2)) + '$\n$p_S=' + '%.1e' % Decimal(
                r_spearman[1][0]) + '$'

            legend_title += (str_pearson if display_pearson else '') + str_spearman

        else:
            # placeholder to avoid activating the trends
            r_spearman = [[], [1]]

            legend_title += ''

        if compute_regr and compute_correl and r_spearman[1][0] < regr_pval_threshold:
            # note: doesn't consider non-significant detections

            x_data_regr = x_data[0][restrict_comput_mask]
            y_data_regr = y_data[0][restrict_comput_mask]

            # note: this is the right shape for lmpplot_uncert
            x_err_regr = (np.array([elem for elem in x_error[0]]).T)[restrict_comput_mask].T
            y_err_regr = (np.array([elem for elem in y_error[0]]).T)[restrict_comput_mask].T

            # since we can't easily change the lin to log type of the axes, we recreate a linear
            with st.spinner('Computing linear regrs'):
                lmplot_uncert_a(ax_scat, x_data_regr, y_data_regr, x_err_regr, y_err_regr, percent=90,
                                nsim=2000, return_linreg=False, percent_regions=[68, 95, 99.7],
                                error_percent=90,
                                intercept_pos='auto', inter_color=['lightgrey', 'silver', 'darkgrey'],
                                infer_log_scale=True, xbounds=plt.xlim(), ybounds=plt.ylim(),
                                line_color='black')

        # updating the ticks of the y axis to avoid powers of ten when unnecessary
        if ax_scat.get_yscale() == 'log':
            if ax_scat.get_ylim()[0] > 0.01:
                ax_scat.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))
                ax_scat.yaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))

        #### legend display
        if show_linked or (
                compute_correl and mode != 'source' and len(x_data[0]) > 1 and not time_mode) or color_scatter not in [
            'Time', 'HR', 'width', 'nH', 'L_3-10', None] \
                and display_legend_correl:
            scat_legend = ax_scat.legend(fontsize=9 if infos == 'ew_width' and display_th_width_ew else 10,
                                         title=legend_title,
                                         ncol=2 if display_th_width_ew and infos == 'ew_width' else 1,
                                         loc='upper right' if not ratio_mode else 'upper right')
            plt.setp(scat_legend.get_title(), fontsize='small')

        if len(x_data_use) > 0:
            try:
                plt.tight_layout()
            except:
                st.rerun()
        else:
            # preventing log scales for graphs with no values, which crashes them
            plt.xscale('linear')
            plt.yscale('linear')

        #### custom things for the wind_review_global paper

        # #specific limits for the width graphs
        # ax_scat.set_xlim(0,90)

        if correl_internal_mode:
            ax_scat.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=True,  # ticks along the bottom edge are off
                right=True,  # ticks along the top edge are off
                labelleft=False,
                labelright=False,
                direction='in')
            ax_scat.set_ylabel('')

        #for constant scaling on HR hard correls for the 4U paper
        # plt.ylim(7e-3,7e-1)

        if save:
            if indiv:
                suffix_str = '_' + lines_std_names[3 + i]
            else:
                suffix_str = '_all'

            plt.savefig(
                save_dir + '/graphs/' + mode + '/' + save_str_prefix + 'autofit_correl_' + infos + suffix_str + '_cam_' + args_cam + '_' + \
                line_search_e_str.replace(' ', '_') + '_' + args_line_search_norm.replace(' ', '_') + '.png')
        if close:
            plt.close(fig_scat)

    # returning the graph for streamlit display
    if streamlit:
        return fig_scat


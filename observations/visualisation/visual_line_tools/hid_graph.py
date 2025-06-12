#general imports
import sys,os

import matplotlib.collections
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple

from matplotlib.lines import Line2D
from matplotlib.ticker import Locator

import matplotlib.dates as mdates



from astropy.time import Time

from copy import deepcopy
import matplotlib.path as mpath
mpl_circle = mpath.Path.unit_circle()
mpl_star = mpath.Path.unit_regular_star(6)
#from https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html

mpl_cut_star = mpath.Path(
    vertices=np.concatenate([mpl_circle.vertices, mpl_star.vertices[::-1, ...]]),
    codes=np.concatenate([mpl_circle.codes, mpl_star.codes]))

'''Astro'''

#rough way of testing if online or not
online=os.getcwd().startswith('/mount/src')
project_dir='/'.join(__file__.split('/')[:-3])

#to be tested online
sys.path.append(os.path.join(project_dir,'observations/spectral_analysis/'))
sys.path.append(os.path.join(project_dir,'general/'))

from general_tools import ravel_ragged,rescale_flex

from visual_line_tools import telescope_colors
from dist_mass_tools import dist_mass_indiv


def hid_graph(ax_hid, dict_linevis,
              display_single=False, display_nondet=True, display_upper=False,
              cyclic_cmap_nondet=False, cyclic_cmap_det=False, cyclic_cmap=False,
              cmap_incl_type=None, cmap_incl_type_str=None,
              radio_info_label=None,
              ew_ratio_ids=None,
              color_nondet=True,
              restrict_threshold=False, display_nonsign=False, display_central_abs=False,
              display_incl_inside=False, dash_noincl=False,
              display_hid_error=False, display_edgesource=False, split_cmap_source=True,
              display_evol_single=False, display_dicho=False,
              global_colors=True, alpha_abs=1,
              paper_look=False, bigger_text=True, square_mode=True, zoom=False,
              broad_mode=False,
              restrict_match_INT=False, broad_binning='day', orbit_bin_lim=1):
    '''

    in broad mode, replaces the standard HID axis by adding theoretical flues estimated from the bat catalog
    '''

    abslines_infos_perobj = dict_linevis['abslines_infos_perobj']
    abslines_plot = dict_linevis['abslines_plot']
    nh_plot = dict_linevis['nh_plot']
    kt_plot_restrict = dict_linevis['Tin_diskbb_plot_restrict']
    kt_plot = dict_linevis['Tin_diskbb_plot']
    hid_plot = dict_linevis['hid_plot']
    incl_plot = dict_linevis['incl_plot']
    mask_obj = dict_linevis['mask_obj']
    mask_obj_base = dict_linevis['mask_obj_base']
    mask_lines = dict_linevis['mask_lines']
    mask_lines_ul = dict_linevis['mask_lines_ul']
    obj_list = dict_linevis['obj_list']
    date_list = dict_linevis['date_list']
    instru_list = dict_linevis['instru_list']
    lum_list = dict_linevis['lum_list']
    choice_telescope = dict_linevis['choice_telescope']
    telescope_list = dict_linevis['telescope_list']
    bool_incl_inside = dict_linevis['bool_incl_inside']
    bool_noincl = dict_linevis['bool_noincl']
    slider_date = dict_linevis['slider_date']
    slider_sign = dict_linevis['slider_sign']
    radio_info_cmap = dict_linevis['radio_info_cmap']
    radio_cmap_i = dict_linevis['radio_cmap_i']
    cmap_color_source = dict_linevis['cmap_color_source']
    cmap_color_det = dict_linevis['cmap_color_det']
    cmap_color_nondet = dict_linevis['cmap_color_nondet']
    exptime_list = dict_linevis['exptime_list']
    display_minorticks = dict_linevis['display_minorticks']

    diago_color = dict_linevis['diago_color']
    custom_states_color = dict_linevis['custom_states_color']
    custom_outburst_color = dict_linevis['custom_outburst_color']
    custom_outburst_number = dict_linevis['custom_outburst_number']
    custom_ionization_color = dict_linevis['custom_ionization_color']

    hatch_unstable = dict_linevis['hatch_unstable']
    change_legend_position = dict_linevis['change_legend_position']

    hr_high_plot_restrict = dict_linevis['hr_high_plot_restrict']
    hid_log_HR = dict_linevis['hid_log_HR']
    flag_single_obj = dict_linevis['flag_single_obj']

    restrict_Ledd_low = dict_linevis['restrict_Ledd_low']
    restrict_Ledd_high = dict_linevis['restrict_Ledd_high']
    restrict_HR_low = dict_linevis['restrict_HR_low']
    restrict_HR_high = dict_linevis['restrict_HR_high']

    Edd_factor_restrict = dict_linevis['Edd_factor_restrict']

    additional_HLD_points_LEdd = dict_linevis['additional_HLD_points_LEdd']
    additional_HLD_points_flux = dict_linevis['additional_HLD_points_flux']
    additional_line_points = dict_linevis['additional_line_points']

    base_sample_points_bool=dict_linevis['base_sample_points_bool']

    # weird setup but due to the variable being either a bool or a str
    if not broad_mode == False:
        HR_broad_bands = dict_linevis['HR_broad_bands']
        lum_broad_bands = dict_linevis['lum_broad_bands']
        lum_plot = dict_linevis['lum_plot']

    if broad_mode == 'BAT':
        catal_bat_df = dict_linevis['catal_bat_df']
        catal_bat_simbad = dict_linevis['catal_bat_simbad']
        sign_broad_hid_BAT = dict_linevis['sign_broad_hid_BAT']

    if restrict_match_INT:
        lc_int_sw_dict = dict_linevis['lc_int_sw_dict']
        fit_int_revol_dict = dict_linevis['fit_int_revol_dict']

    observ_list = dict_linevis['observ_list']

    # note that these one only get the significant BAT detections so no need to refilter
    lum_high_1sig_plot_restrict = dict_linevis['lum_high_1sig_plot_restrict']
    lum_high_sign_plot_restrict = dict_linevis['lum_high_sign_plot_restrict']
    # global normalisations values for the points
    norm_s_lin = 5
    norm_s_pow = 1.15

    # extremal allowed values for kT in the fitting procedure(in keV)
    kt_min = 0.5
    kt_max = 3.

    # parameters independant of the presence of lines
    type_1_cm = ['Inclination', 'Time', 'nH', 'kT']

    # parameters without actual colorbars
    type_1_colorcode = ['Source', 'Instrument', 'custom_line_struct', 'custom_acc_states', 'custom_outburst',
                        'custom_ionization']

    fig_hid = ax_hid.get_figure()

    hid_plot_use = deepcopy(hid_plot)

    if restrict_match_INT:
        # currently limited to 4U1630-47
        int_lc_df = fit_int_revol_dict['4U1630-47']

        int_lc_mjd = np.array([Time(elem).mjd.astype(float) for elem in int_lc_df['ISOT']])

        obs_dates = Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd.astype(
            float)

        # computing which observations are within the timeframe of an integral revolution (assuming 3days-long)
        mask_withtime_INT = [min((int_lc_mjd - elem)[(int_lc_mjd - elem) >= 0]) < 3 for elem in obs_dates[0]]

    elif broad_mode == 'BAT':

        HR_broad_6_10 = HR_broad_bands == '([6-10]+[BAND])/[3-6]'
        lum_broad_soft = lum_broad_bands == '[3-10]+[BAND]'

        ax_hid.set_yscale('log')

        # no need for symlog now that we use upper limits
        # if not HR_broad_6_10:
        #     broad_x_linthresh=0.01
        #     ax_hid.set_xscale('symlog', linthresh=broad_x_linthresh, linscale=0.1)
        #     ax_hid.xaxis.set_minor_locator(MinorSymLogLocator(linthresh=broad_x_linthresh))
        # else:

        broad_x_linthresh = 0
        ax_hid.set_xscale('log')

        ax_hid.set_xlabel('Hardness Ratio in ' + HR_broad_bands.replace('BAND', '15-50') + ' keV bands)')
        ax_hid.set_ylabel(
            r'Luminosity in the ' + lum_broad_bands.replace('BAND', '15-50') + ' keV band in (L/L$_{Edd}$) units')

        # #currently limited to 4U1630-47
        # bat_lc_df_init = fetch_bat_lightcurve(catal_bat_df, catal_bat_simbad,['4U1630-47'], binning=broad_binning)
        #
        # #restricting to significant BAT detections or not
        #
        # mask_sign_bat = bat_lc_df_init[bat_lc_df_init.columns[1]] - bat_lc_df_init[
        #     bat_lc_df_init.columns[2]] * 2 > 0
        #
        # if sign_broad_hid_BAT:
        #     # significance test to only get good bat data
        #
        #     # applying the mask. Reset index necessary to avoid issues when calling indices later.
        #     # drop to avoid creating an index column that will ruin the column calling
        #     bat_lc_df = bat_lc_df_init[mask_sign_bat].reset_index(drop=True)
        # else:
        #     bat_lc_df=bat_lc_df_init
        #
        # if broad_binning=='day':
        #     bat_lc_mjd=np.array(bat_lc_df[bat_lc_df.columns[0]])
        #
        # elif broad_binning=='orbit':
        #     bat_lc_tstart=Time('51910',format='mjd')+TimeDelta(bat_lc_df['TIME'],format='sec')
        #     bat_lc_tend=Time('51910',format='mjd')+TimeDelta(bat_lc_df['TIMEDEL'],format='sec')
        #
        # #converting to 15-50keV luminosity in Eddington units, removing negative values
        # bat_lc_lum_nocorr=np.array([bat_lc_df[bat_lc_df.columns[1]],bat_lc_df[bat_lc_df.columns[2]],
        #                             bat_lc_df[bat_lc_df.columns[2]]]).clip(0).T\
        #             *convert_BAT_count_flux['4U1630-47']*Edd_factor_restrict
        #
        # #and applying the correction
        # bat_lc_lum=corr_factor_lbat(bat_lc_lum_nocorr)
        #
        # if broad_binning=='day':
        #     obs_dates=Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd.astype(int)
        #
        #
        #     mask_withtime_BAT=[elem in bat_lc_mjd for elem in obs_dates[0]]
        #
        #     #this one only considers BAT but also considers non-significant points
        #
        #     #not necessary now that we create different arrays beforehand
        #     # #getting an array with the bat flux of each observation date
        #     # lum_broad_single_BAT=np.array([np.array([np.nan, np.nan,np.nan]) if obs_dates[0][i_obs] not in bat_lc_mjd else bat_lc_lum[bat_lc_mjd==obs_dates[0][i_obs]][0]\
        #     #                           for i_obs in range(len(obs_dates[0]))]).T
        #
        # elif broad_binning=='orbit':
        #
        #     obs_tstart=Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str)).mjd
        #     obs_tend=(Time(np.array([date_list[mask_obj][0] for i in range(sum(mask_lines))]).astype(str))+\
        #             TimeDelta(np.array([exptime_list[mask_obj][0] for i in range(sum(mask_lines))],format='sec'))).mjd
        #
        #     #TO BE IMPROVED FOR MORE PRECISION
        #     # # getting an array with the bat flux of each observation date
        #     # lum_broad_single_BAT = np.array([np.array([np.nan, np.nan]) if obs_dates[0][i_obs] not in bat_lc_mjd else
        #     #                              bat_lc_lum[bat_lc_mjd == obs_dates[0][i_obs]][0] \
        #     #                              for i_obs in range(len(obs_dates[0]))]).T

        # now we combine the BAT non-significant elements and the already existing significant elements in
        # lum_high_plot_restrict

        if flag_single_obj:
            lum_broad_single = np.array([elem for elem in np.transpose(lum_high_1sig_plot_restrict, (1, 0, 2))[0]])

            # mask to highlight non-significant high Energy detections
            mask_sign_high_E = ~np.isnan(
                np.array([elem for elem in np.transpose(lum_high_sign_plot_restrict, (1, 0, 2))[0]],
                         dtype=float)[0])

        else:
            lum_broad_single = np.array([elem for elem in lum_high_1sig_plot_restrict.T[0]])

            # mask to highlight non-significant high Energy detections
            mask_sign_high_E = ~np.isnan(np.array([elem for elem in lum_high_sign_plot_restrict.T[0]])[0])

        lum_broad_single = lum_broad_single.T

        # transforming the BAT non-significant detections in 1 sigma upper limits or removing them
        for i_obs in range(len(lum_broad_single)):
            if not mask_sign_high_E[i_obs]:
                if not sign_broad_hid_BAT:
                    lum_broad_single[i_obs] = np.array([lum_broad_single[i_obs][0] + lum_broad_single[i_obs][1],
                                                        lum_broad_single[i_obs][0] + lum_broad_single[i_obs][1], 0.])
                else:
                    lum_broad_single[i_obs] = np.repeat(np.nan, 3)

        lum_broad_single = lum_broad_single.T

        # creating the mask to avoid plotting nans everywhere
        mask_with_broad = ~np.isnan(lum_broad_single[0])

        # this is the quantity that needs to be added if the numerator is broad+6-10 and not just broad
        hid_broad_add = lum_plot[2][0][mask_obj][0].astype(float) if HR_broad_6_10 else 0

        hid_broad_vals = (lum_broad_single[0] + hid_broad_add) \
                         / lum_plot[1][0][mask_obj][0].astype(float)

        # here the numerator is the quadratic uncertainty addition and then the fraction is for the quadratic ratio uncertainty
        # composition
        hid_broad_err = np.array(
            [((((lum_plot[2][i][mask_obj][0] if HR_broad_6_10 else 0) ** 2 + lum_broad_single[i] ** 2) ** (1 / 2) \
               / (lum_broad_single[0] + hid_broad_add)) ** 2 + \
              (lum_plot[1][i][mask_obj][0] / lum_plot[1][0][mask_obj][0]) ** 2) ** (1 / 2) * hid_broad_vals \
             for i in [1, 2]])

        # overwriting hid_plot's individual elements because overwriting the full obs array doesn't work
        # there's lot of issues if using mask_obj directly but here we should be in single object mode only
        # so we can do it differently
        i_obj_single = np.argwhere(mask_obj).T[0][0]

        hid_plot_use[0][0][i_obj_single] = hid_broad_vals
        hid_plot_use[0][1][i_obj_single] = hid_broad_err[0]
        hid_plot_use[0][2][i_obj_single] = hid_broad_err[1]

        if lum_broad_soft:
            hid_plot_use[1][0][i_obj_single] += lum_broad_single[0]

            hid_plot_use[1][1][i_obj_single] = ((hid_plot_use[1][1][i_obj_single]) ** 2 + \
                                                (lum_broad_single[1]) ** 2) ** (1 / 2)
            hid_plot_use[1][2][i_obj_single] = ((hid_plot_use[1][2][i_obj_single]) ** 2 + \
                                                (lum_broad_single[1]) ** 2) ** (1 / 2)

    else:

        # log x scale for an easier comparison with Ponti diagrams
        if hid_log_HR:
            ax_hid.set_xscale('log')

            if display_minorticks:
                if ax_hid.get_xlim()[0] > 0.1:
                    ax_hid.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
                    ax_hid.xaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
                else:
                    ax_hid.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))
                    ax_hid.xaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))

        ax_hid.set_xlabel('Hardness Ratio ([6-10]/[3-6] keV bands)')
        ax_hid.set_ylabel(r'Luminosity in the [3-10] keV band in (L/L$_{Edd}$) units')
        ax_hid.set_yscale('log')

    # recreating some variables

    mask_obs_intime_repeat = np.array(
        [np.repeat(((np.array([Time(subelem) for subelem in elem]) >= Time(slider_date[0])) & \
                    (np.array([Time(subelem) for subelem in elem]) <= Time(slider_date[1]))), sum(mask_lines)) for elem
         in date_list], dtype=object)

    # checking which sources have no detection in the current combination

    global_displayed_sign = np.array(
        [ravel_ragged(elem)[mask.astype(bool)] for elem, mask in
         zip(abslines_plot[4][0][mask_lines].T, mask_obs_intime_repeat)],
        dtype=object)

    if incl_plot is not None:
        incl_cmap = np.array([incl_plot.T[0], incl_plot.T[0] - incl_plot.T[1], incl_plot.T[0] + incl_plot.T[2]]).T
        incl_cmap_base = incl_cmap[mask_obj]
        incl_cmap_restrict = incl_cmap[mask_obj]
        incl_plot_restrict = incl_plot[mask_obj]

    nh_plot_restrict = deepcopy(nh_plot)

    nh_plot_restrict = nh_plot_restrict.T[mask_obj].T

    if len(mask_obj) == 1 and np.ndim(hid_plot_use) == 4:
        hid_plot_restrict = hid_plot_use
    else:
        hid_plot_restrict = hid_plot_use.T[mask_obj].T

    if display_nonsign:
        mask_obj_withdet = np.array([(elem > 0).any() for elem in global_displayed_sign])
    else:
        mask_obj_withdet = np.array([(elem >= slider_sign).any() for elem in global_displayed_sign])

    # storing the number of objects with detections
    n_obj_withdet = sum(mask_obj_withdet & mask_obj)

    # markers
    marker_abs = 'o'
    marker_nondet = 'd'
    marker_ul = 'h'
    marker_ul_top = 'H'

    #initializing the variables
    xlims=None
    ylims=None

    alpha_ul = 0.5

    if base_sample_points_bool:
        # computing the extremal values of the whole sample/plotted sample to get coherent colormap normalisations, and creating the range of object colors
        if global_colors:
            global_plotted_sign = abslines_plot[4][0].ravel()
            global_plotted_data = abslines_plot[radio_cmap_i][0].ravel()

            # objects colormap for common display
            norm_colors_obj = mpl.colors.Normalize(vmin=0,
                                                   vmax=max(0, len(abslines_infos_perobj) + (-1 if not cyclic_cmap else 0)))
            colors_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_source)

            norm_colors_det = mpl.colors.Normalize(vmin=0, vmax=max(0,
                                                                    n_obj_withdet + (-1 if not cyclic_cmap_det else 0) + (
                                                                        1 if n_obj_withdet == 0 else 0)))
            colors_det = mpl.cm.ScalarMappable(norm=norm_colors_det, cmap=cmap_color_det)

            norm_colors_nondet = mpl.colors.Normalize(vmin=0, vmax=max(0, len(abslines_infos_perobj) - n_obj_withdet + (
                -1 if not cyclic_cmap_nondet else 0)))
            colors_nondet = mpl.cm.ScalarMappable(norm=norm_colors_nondet, cmap=cmap_color_nondet)

            # the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
            global_plotted_datetime = np.array([elem for elem in date_list for i in range(len(mask_lines))], dtype='object')

            global_mask_intime = np.repeat(True, len(ravel_ragged(global_plotted_datetime)))

            global_mask_intime_norepeat = np.repeat(True, len(ravel_ragged(date_list)))

        else:
            global_plotted_sign = abslines_plot[4][0][mask_lines].T[mask_obj].ravel()
            global_plotted_data = abslines_plot[radio_cmap_i][0][mask_lines].T[mask_obj].ravel()

            # objects colormap
            norm_colors_obj = mpl.colors.Normalize(vmin=0, vmax=max(0, len(abslines_infos_perobj[mask_obj]) + (
                -1 if not cyclic_cmap else 0)))
            colors_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_source)

            norm_colors_det = mpl.colors.Normalize(vmin=0, vmax=max(0, n_obj_withdet + (-1 if not cyclic_cmap_det else 0)))
            colors_det = mpl.cm.ScalarMappable(norm=norm_colors_det, cmap=cmap_color_det)

            norm_colors_nondet = mpl.colors.Normalize(vmin=0,
                                                      vmax=max(0, len(abslines_infos_perobj[mask_obj]) - n_obj_withdet + (
                                                          -1 if not cyclic_cmap_nondet else 0)))
            colors_nondet = mpl.cm.ScalarMappable(norm=norm_colors_nondet, cmap=cmap_color_nondet)

            # adapting the plotted data in regular array for each object in order to help
            # global masks to take off elements we don't want in the comparison

            # the date is an observation-level parameter so it needs to be repeated to have the same dimension as the other global variables
            global_plotted_datetime = np.array([elem for elem in date_list[mask_obj] for i in range(sum(mask_lines))],
                                               dtype='object')

            global_mask_intime = (Time(ravel_ragged(global_plotted_datetime)) >= Time(slider_date[0])) & \
                                 (Time(ravel_ragged(global_plotted_datetime)) <= Time(slider_date[1]))

            global_mask_intime_norepeat = (Time(ravel_ragged(date_list[mask_obj])) >= Time(slider_date[0])) & \
                                          (Time(ravel_ragged(date_list[mask_obj])) <= Time(slider_date[1]))

        # global_nondet_mask=(np.array([subelem for elem in global_plotted_sign for subelem in elem])<slider_sign) & (global_mask_intime)

        global_det_mask = (np.array([subelem for elem in global_plotted_sign for subelem in elem]) > 0) & (
            global_mask_intime)

        global_sign_mask = (np.array([subelem for elem in global_plotted_sign for subelem in elem]) >= slider_sign) & (
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
            cmap_norm_info = colors.Normalize(0, 90)

        # putting the axis limits at standard bounds or the points if the points extend further
        lum_list_ravel = np.array([subelem for elem in lum_list for subelem in elem])
        bounds_x = [min(lum_list_ravel.T[2][0] / lum_list_ravel.T[1][0]),
                    max(lum_list_ravel.T[2][0] / lum_list_ravel.T[1][0])]
        bounds_y = [min(lum_list_ravel.T[4][0]), max(lum_list_ravel.T[4][0])]

        if zoom == 'auto' or broad_mode != False:
            # the nan values are the BAT non-matching points and thus need to be removed
            broad_notBAT_mask = ~np.isnan(ravel_ragged(hid_plot_restrict[0][0]))

            xlims = (min(ravel_ragged(hid_plot_restrict[0][0])[broad_notBAT_mask]),
                     max(ravel_ragged(hid_plot_restrict[0][0])[broad_notBAT_mask]))

            ylims = (min(ravel_ragged(hid_plot_restrict[1][0])[broad_notBAT_mask]),
                     max(ravel_ragged(hid_plot_restrict[1][0])[broad_notBAT_mask]))

            rescale_flex(ax_hid, xlims, ylims, 0.05)

        if type(zoom) == list:
            ax_hid.set_xlim(zoom[0][0], zoom[0][1])
            ax_hid.set_ylim(zoom[1][0], zoom[1][1])

        if not zoom and not broad_mode:
            rescale_flex(ax_hid, bounds_x, bounds_y, 0.05, std_x=[0.1, 2], std_y=[1e-5, 1])

        # creating space for the colorbar
        if radio_info_cmap not in type_1_colorcode:
            ax_cb = plt.axes([0.92, 0.105, 0.02, 0.775])

            # giving a default value to the colorbar variable so we can test if a cb has been generated later on
            cb = None

        # note: the value will finish at false for sources with no non-detections
        label_obj_plotted = np.repeat(False, len(abslines_infos_perobj[mask_obj]))

        is_colored_scat = False

        # creating the plotted colors variable#defining the mask for detections and non detection
        plotted_colors_var = []

        #### detections HID

        id_obj_det = 0

        #### Still issues with colormapping when restricting time

        # loop on the objects for detections (restricted or not depending on if the mode is detection only)
        for i_obj, abslines_obj in enumerate(abslines_infos_perobj[mask_obj]):

            # defining the index of the object in the entire array if asked to, in order to avoid changing colors
            if global_colors:
                i_obj_glob = np.argwhere(obj_list == obj_list[mask_obj][i_obj])[0][0]
            else:
                i_obj_glob = i_obj

            '''
            # The shape of each abslines_obj is (uncert,info,line,obs)
            '''

            # defining the hid positions of each point
            if broad_mode != False:
                x_hid = hid_plot_use[0][0][mask_obj][i_obj]

                # similar structure for the rest
                x_hid_err = np.array([hid_plot_use[0][1][mask_obj][i_obj], hid_plot_use[0][2][mask_obj][i_obj]])

                # we do it this way because a += on y_hid will overwite lum_list, which is extremely dangerous
                if lum_broad_soft:
                    y_hid = lum_list[mask_obj][i_obj].T[4][0] + lum_broad_single[0]
                    y_hid_err = (lum_list[mask_obj][i_obj].T[4][1:] ** 2 + lum_broad_single[1:] ** 2) ** (1 / 2)
                else:
                    y_hid = lum_list[mask_obj][i_obj].T[4][0]
                    y_hid_err = lum_list[mask_obj][i_obj].T[4][1:]
            else:
                x_hid = lum_list[mask_obj][i_obj].T[2][0] / lum_list[mask_obj][i_obj].T[1][0]

                x_hid_err = (((lum_list[mask_obj][i_obj].T[2][1:] / lum_list[mask_obj][i_obj].T[2][0]) ** 2 + \
                              (lum_list[mask_obj][i_obj].T[1][1:] / lum_list[mask_obj][i_obj].T[1][0]) ** 2) ** (
                                         1 / 2) * x_hid)

                y_hid = lum_list[mask_obj][i_obj].T[4][0]

                y_hid_err = lum_list[mask_obj][i_obj].T[4][1:]

            # defining the masks and shapes of the markers for the rest

            # defining the mask for the time interval restriction
            datelist_obj = Time(np.array([date_list[mask_obj][i_obj] for i in range(sum(mask_lines))]).astype(str))
            mask_intime = (datelist_obj >= Time(slider_date[0])) & (datelist_obj <= Time(slider_date[1]))

            if broad_mode == 'BAT':
                mask_intime = (mask_intime) & mask_with_broad

            # defining the mask for detections and non detection
            mask_det = (abslines_obj[0][4][mask_lines] > 0.) & (mask_intime)

            # defining the mask for significant detections
            mask_sign = (abslines_obj[0][4][mask_lines] >= slider_sign) & (mask_intime)

            # these ones will only be used if the restrict values chexbox is checked

            obj_val_cmap_sign = np.array(
                [np.nan if len(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_sign.T[i_obs]]) == 0 else \
                     (max(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_sign.T[i_obs]]) \
                          if radio_info_cmap != 'EW ratio' else \
                          np.nan if abslines_obj[0][radio_cmap_i][ew_ratio_ids[0]].T[i_obs] < slider_sign or \
                                    abslines_obj[0][radio_cmap_i][ew_ratio_ids[1]].T[i_obs] < slider_sign else \
                              abslines_obj[0][radio_cmap_i][ew_ratio_ids[1]].T[i_obs] / \
                              abslines_obj[0][radio_cmap_i][ew_ratio_ids[0]].T[i_obs]) \
                 for i_obs in range(len(abslines_obj[0][radio_cmap_i][mask_lines].T))])

            # the size is always tied to the EW
            obj_size_sign = np.array([np.nan if len(abslines_obj[0][0][mask_lines].T[i_obs][mask_sign.T[i_obs]]) == 0 else \
                                          max(abslines_obj[0][0][mask_lines].T[i_obs][mask_sign.T[i_obs]]) \
                                      for i_obs in range(len(abslines_obj[0][0][mask_lines].T))])

            # and we can create the plot mask from it (should be the same wether we take obj_size_sign or the size)
            obj_val_mask_sign = ~np.isnan(obj_size_sign)

            # creating a display order which is the reverse of the EW size order to make sure we do not hide part the detections
            obj_order_sign = obj_size_sign[obj_val_mask_sign].argsort()[::-1]

            if 'custom' in radio_info_cmap:
                if radio_info_cmap == 'custom_line_struct':
                    # reordering to have the substructures above the rest
                    colors_data_restrict = diago_color[mask_obj][i_obj][obj_val_mask_sign]
                    obj_order_sign_mainstruct = obj_size_sign[obj_val_mask_sign][colors_data_restrict == 'grey'].argsort()[
                                                ::-1]
                    obj_order_sign_substruct = obj_size_sign[obj_val_mask_sign][colors_data_restrict == 'orange'].argsort()[
                                               ::-1]
                    obj_order_sign_outliers = obj_size_sign[obj_val_mask_sign][colors_data_restrict == 'blue'].argsort()[
                                              ::-1]

                    len_arr = np.arange(len(obj_order_sign))
                    obj_order_sign = np.concatenate([len_arr[colors_data_restrict == 'grey'][obj_order_sign_mainstruct],
                                                     len_arr[colors_data_restrict == 'orange'][obj_order_sign_substruct],
                                                     len_arr[colors_data_restrict == 'blue'][obj_order_sign_outliers]])

                elif radio_info_cmap == 'custom_acc_states':
                    # reordering to have the substructures above the rest
                    colors_data_restrict = custom_states_color[mask_obj][i_obj][obj_val_mask_sign]

                    states_zorder = ['grey', 'red', 'orange', 'green', 'blue', 'purple']
                    obj_order_sign_states = [obj_size_sign[obj_val_mask_sign][colors_data_restrict == elem_state_zorder] \
                                                 .argsort()[::-1] for elem_state_zorder in states_zorder]

                    len_arr = np.arange(len(obj_order_sign))
                    obj_order_sign = np.concatenate([len_arr[colors_data_restrict == states_zorder[i_state]] \
                                                         [obj_order_sign_states[i_state]] \
                                                     for i_state in range(len(states_zorder))])

                elif radio_info_cmap == 'custom_ionization':
                    # reordering to have the substructures above the rest
                    colors_data_restrict = custom_ionization_color[mask_obj][i_obj][obj_val_mask_sign]

                    states_zorder = ['grey', 'red', 'rosybrown', 'orange', 'turquoise', 'pink', 'powderblue', 'forestgreen']
                    obj_order_sign_states = [obj_size_sign[obj_val_mask_sign][colors_data_restrict == elem_state_zorder] \
                                                 .argsort()[::-1] for elem_state_zorder in states_zorder]

                    len_arr = np.arange(len(obj_order_sign))
                    obj_order_sign = np.concatenate([len_arr[colors_data_restrict == states_zorder[i_state]] \
                                                         [obj_order_sign_states[i_state]] \
                                                     for i_state in range(len(states_zorder))])

            # same thing for all detections
            obj_val_cmap = np.array(
                [np.nan if len(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                     max(abslines_obj[0][radio_cmap_i][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                 for i_obs in range(len(abslines_obj[0][radio_cmap_i][mask_lines].T))])

            obj_size = np.array([np.nan if len(abslines_obj[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                                     max(abslines_obj[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                                 for i_obs in range(len(abslines_obj[0][0][mask_lines].T))])

            obj_val_mask = ~np.isnan(obj_size)

            # creating a display order which is the reverse of the EW size order to make sure we show as many detections as possible
            obj_order = obj_size[obj_val_mask].argsort()[::-1]

            # not used for now
            # else:
            #     #in single line mode we can directly fetch the single lines values and the mask for the specific line
            #     obj_val_cmap_sign=abslines_obj[0][radio_cmap_i][mask_lines][mask_sign[mask_lines]].astype(float)

            #     #and the mask
            #     obj_val_mask_sign=mask_sign[mask_lines]

            #     #in single line mode we can directly fetch the single lines values and the mask for the specific line
            #     obj_val_cmap=abslines_obj[0][radio_cmap_i][mask_lines][mask_det[mask_lines]].astype(float)

            #     obj_val_mask=mask_det[mask_lines]

            # this mask is used to plot 'unsignificant only' detection points
            obj_val_mask_nonsign = (obj_val_mask) & (~obj_val_mask_sign)

            # plotting everything

            # we put the color mapped scatter into a list to clim all of them at once at the end
            if i_obj == 0:
                scat_col = []

            # plotting the detection centers if asked for

            if len(x_hid[obj_val_mask]) > 0 and display_central_abs:

                if display_hid_error:
                    ax_hid.errorbar(x_hid[obj_val_mask], y_hid[obj_val_mask],
                                    xerr=x_hid_err.T[obj_val_mask].T, yerr=y_hid_err.T[obj_val_mask].T,
                                    marker='', ls='',
                                    color=colors_obj.to_rgba(i_obj_glob) \
                                        if radio_info_cmap == 'Source' else 'grey',
                                    label='', zorder=1000)

                ax_hid.scatter(x_hid[obj_val_mask], y_hid[obj_val_mask], marker=marker_abs,
                               color=colors_obj.to_rgba(i_obj_glob) \
                                   if radio_info_cmap == 'Source' else 'grey', label='', zorder=1000, edgecolor='black',
                               plotnonfinite=True)

            #### detection scatters
            # plotting statistically significant absorptions before values

            if radio_info_cmap == 'Instrument':
                color_instru = [telescope_colors[elem] for elem in
                                instru_list[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]]

                if display_nonsign:
                    color_instru_nonsign = [telescope_colors[elem] for elem in
                                            instru_list[mask_obj][i_obj][obj_val_mask_nonsign]]

            # note: there's no need to reorder for source level informations (ex: inclination) since the values are the same for all the points
            c_scat = None if radio_info_cmap == 'Source' else \
                mdates.date2num(
                    date_list[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign]) if radio_info_cmap == 'Time' else \
                    np.repeat(incl_cmap_restrict[i_obj][cmap_incl_type],
                              len(x_hid[obj_val_mask_sign])) if radio_info_cmap == 'Inclination' else \
                        color_instru if radio_info_cmap == 'Instrument' else \
                            nh_plot_restrict[0][i_obj][obj_val_mask_sign][obj_order_sign] if radio_info_cmap == 'nH' else \
                                kt_plot_restrict[0][i_obj][obj_val_mask_sign][
                                    obj_order_sign] if radio_info_cmap == 'kT' else \
                                    diago_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign] \
                                        if radio_info_cmap == 'custom_line_struct' else \
                                        custom_states_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign] \
                                            if radio_info_cmap == 'custom_acc_states' else \
                                            custom_ionization_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign] \
                                                if radio_info_cmap == 'custom_ionization' else \
                                                custom_outburst_color[mask_obj][i_obj][obj_val_mask_sign][obj_order_sign] \
                                                    if radio_info_cmap == 'custom_outburst' else \
                                                    obj_val_cmap_sign[obj_val_mask_sign][obj_order_sign]

            #### TODO : test the dates here with just IGRJ17451 to solve color problem

            # adding a failsafe to avoid problems when nothing is displayed
            if c_scat is not None and len(c_scat) == 0:
                c_scat = None

            if restrict_threshold:

                color_val_detsign = (colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                         colors_det.to_rgba(
                                             id_obj_det)) if radio_info_cmap == 'Source' else None

                if broad_mode == 'BAT' and not sign_broad_hid_BAT:

                    alpha_abs_sign_full = np.where(mask_sign_high_E, 1, 0.3)

                    alpha_abs_detsign = alpha_abs_sign_full[obj_val_mask_sign][obj_order_sign]

                    # in the case where alpha is not singular the len of the color keyword needs to match it
                    color_val_detsign = None if color_val_detsign is None else np.array(
                        [color_val_detsign] * len(alpha_abs_detsign))
                else:
                    alpha_abs_detsign = None

                if len(x_hid[obj_val_mask_sign][obj_order_sign]) > 0:
                    # displaying "significant only" cmaps/sizes
                    scat_col += [
                        ax_hid.scatter(x_hid[obj_val_mask_sign][obj_order_sign], y_hid[obj_val_mask_sign][obj_order_sign],
                                       marker=marker_abs, color=color_val_detsign,
                                       c=c_scat,
                                       s=norm_s_lin * obj_size_sign[obj_val_mask_sign][obj_order_sign] ** norm_s_pow,
                                       edgecolor='black' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                                       linewidth=1 + int(display_edgesource) / 2,
                                       norm=cmap_norm_info,
                                       label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and \
                                                                          (
                                                                                      radio_info_cmap == 'Source' or display_edgesource) and len(
                                           x_hid[obj_val_mask_sign]) > 0 else '',
                                       cmap=cmap_info, alpha=alpha_abs_detsign,
                                       plotnonfinite=True)]

                if (radio_info_cmap == 'Source' or display_edgesource) and len(x_hid[obj_val_mask_sign]) > 0:
                    label_obj_plotted[i_obj] = True

            # plotting the maximum value and hatch coding depending on if there's a significant abs line in the obs
            else:

                # displaying "all" cmaps/sizes but only where's at least one significant detection (so we don't hatch)
                scat_col += [
                    ax_hid.scatter(x_hid[obj_val_mask_sign][obj_order_sign], y_hid[obj_val_mask_sign][obj_order_sign],
                                   marker=marker_abs,
                                   color=(colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                              colors_det.to_rgba(id_obj_det)) if radio_info_cmap == 'Source' else None,
                                   c=c_scat, s=norm_s_lin * obj_size[obj_val_mask_sign][obj_order_sign] ** norm_s_pow,
                                   edgecolor='black' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                                   linewidth=1 + int(display_edgesource),
                                   norm=cmap_norm_info,
                                   label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and \
                                                                      (
                                                                              radio_info_cmap == 'Source' or display_edgesource) and len(
                                       x_hid[obj_val_mask_sign]) > 0 else '',
                                   cmap=cmap_info, alpha=alpha_abs,
                                   plotnonfinite=True)]

                if (radio_info_cmap == 'Source' or display_edgesource) and len(x_hid[obj_val_mask_sign]) > 0:
                    label_obj_plotted[i_obj] = True

            # adding the plotted colors into a list to create the ticks from it at the end
            plotted_colors_var += [elem for elem in
                                   (incl_cmap_restrict.T[cmap_incl_type] if radio_info_cmap == 'Inclination' else \
                                        (obj_val_cmap_sign[obj_val_mask_sign][obj_order_sign] if restrict_threshold \
                                             else obj_val_cmap[obj_val_mask_sign][obj_order_sign]).tolist()) if
                                   not np.isnan(elem)]

            if display_nonsign:

                c_scat_nonsign = None if radio_info_cmap == 'Source' else \
                    mdates.date2num(date_list[mask_obj][i_obj][obj_val_mask_nonsign]) if radio_info_cmap == 'Time' else \
                        np.repeat(incl_cmap_restrict[i_obj][cmap_incl_type],
                                  len(x_hid[obj_val_mask_nonsign])) if radio_info_cmap == 'Inclination' else \
                            nh_plot_restrict[0][i_obj][obj_val_mask_nonsign] if radio_info_cmap == 'nH' else \
                                kt_plot_restrict[0][i_obj][obj_val_mask_nonsign] if radio_info_cmap == 'kT' else \
                                    diago_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                        if radio_info_cmap == 'custom_line_struct' else \
                                        custom_states_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                            if radio_info_cmap == 'custom_acc_states' else \
                                            custom_ionization_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                                if radio_info_cmap == 'custom_ionization' else \
                                                custom_outburst_color[mask_obj][i_obj][obj_val_mask_nonsign] \
                                                    if radio_info_cmap == 'custom_outburst' else \
                                                    obj_val_cmap[obj_val_mask_nonsign]

                # adding a failsafe to avoid problems when nothing is displayed
                if c_scat is not None and len(c_scat) == 0:
                    c_scat = None

                # and "unsignificant only" in any case is hatched. Edgecolor sets the color of the hatch
                scat_col += [ax_hid.scatter(x_hid[obj_val_mask_nonsign], y_hid[obj_val_mask_nonsign], marker=marker_abs,
                                            color=(colors_obj.to_rgba(
                                                i_obj_glob) if not split_cmap_source else colors_det.to_rgba(
                                                id_obj_det)) if radio_info_cmap == 'Source' else None,
                                            c=c_scat_nonsign, s=norm_s_lin * obj_size[obj_val_mask_nonsign] ** norm_s_pow,
                                            hatch='///',
                                            edgecolor='grey' if not display_edgesource else colors_obj.to_rgba(i_obj_glob),
                                            linewidth=1 + int(display_edgesource),
                                            norm=cmap_norm_info,
                                            label=obj_list[mask_obj][i_obj] if not label_obj_plotted[i_obj] and \
                                                                               (
                                                                                       radio_info_cmap == 'Source' or display_edgesource) else '',
                                            cmap=cmap_info,
                                            alpha=alpha_abs,
                                            plotnonfinite=True)]
                if (radio_info_cmap == 'Source' or display_edgesource) and len(x_hid[obj_val_mask_nonsign]) > 0:
                    label_obj_plotted[i_obj] = True

                plotted_colors_var += [elem for elem in (
                    incl_cmap_restrict.T[cmap_incl_type] if radio_info_cmap == 'Inclination' else obj_val_cmap[
                        obj_val_mask_nonsign].tolist()) if not np.isnan(elem)]

            if len(x_hid[obj_val_mask_sign]) > 0 or (len(x_hid[obj_val_mask_nonsign]) > 0 and display_nonsign):
                id_obj_det += 1

            # resizing all the colors and plotting the colorbar, only done at the last iteration
            if radio_info_cmap not in type_1_colorcode and i_obj == len(abslines_infos_perobj[mask_obj]) - 1 and len(
                    plotted_colors_var) > 0:

                is_colored_scat = False

                for elem_scatter in scat_col:

                    # standard limits for the inclination and Time
                    if radio_info_cmap == 'Inclination':
                        elem_scatter.set_clim(vmin=0, vmax=90)
                    elif radio_info_cmap == 'Time':

                        if global_colors:
                            elem_scatter.set_clim(
                                vmin=min(mdates.date2num(ravel_ragged(date_list))),
                                vmax=max(mdates.date2num(ravel_ragged(date_list))))
                        else:
                            elem_scatter.set_clim(
                                vmin=max(
                                    min(mdates.date2num(ravel_ragged(date_list \
                                                                         [mask_obj])[global_mask_intime_norepeat])),
                                    mdates.date2num(slider_date[0])),
                                vmax=min(
                                    max(mdates.date2num(ravel_ragged(date_list \
                                                                         [mask_obj])[global_mask_intime_norepeat])),
                                    mdates.date2num(slider_date[1])))

                    elif radio_info_cmap == 'nH':

                        if global_colors:
                            elem_scatter.set_clim(vmin=min(ravel_ragged(nh_plot_restrict[0])),
                                                  vmax=max(ravel_ragged(nh_plot_restrict[0])))
                        else:
                            elem_scatter.set_clim(vmin=min(ravel_ragged(nh_plot_restrict[0])[global_mask_intime_norepeat]),
                                                  vmax=max(ravel_ragged(nh_plot_restrict[0])[global_mask_intime_norepeat]))
                    elif radio_info_cmap == 'kT':

                        elem_scatter.set_clim(vmin=0.5, vmax=3)

                    else:

                        # dynamical limits for the rest
                        if global_colors and radio_info_cmap not in ('EW ratio', 'Inclination', 'Time', 'nH', 'kT'):
                            if display_nonsign:
                                elem_scatter.set_clim(vmin=min(global_det_data), vmax=max(global_det_data))
                            else:
                                elem_scatter.set_clim(vmin=min(global_sign_data), vmax=max(global_sign_data))
                        else:
                            elem_scatter.set_clim(vmin=min(plotted_colors_var), vmax=max(plotted_colors_var))

                    if len(elem_scatter.get_sizes()) > 0:
                        is_colored_scat = True

                        # keeping the scatter to create the colorbar from it
                        elem_scatter_forcol = elem_scatter

                    # ax_cb.set_axis_off()

                # defining the ticks from the currently plotted objects

                if radio_cmap_i == 1 or radio_info_cmap == 'EW ratio':

                    cmap_min_sign = 1 if min(plotted_colors_var) == 0 else min(plotted_colors_var) / abs(
                        min(plotted_colors_var))

                    cmap_max_sign = 1 if min(plotted_colors_var) == 0 else max(plotted_colors_var) / abs(
                        max(plotted_colors_var))

                    # round numbers for the Velocity shift
                    if radio_info_cmap == 'Velocity shift':
                        bshift_step = 250 if choice_telescope == ['Chandra'] else 500

                        # the +1 are here to ensure we see the extremal ticks

                        cmap_norm_ticks = np.arange(((min(plotted_colors_var) // bshift_step) + 1) * bshift_step,
                                                    ((max(plotted_colors_var) // bshift_step) + 1) * bshift_step,
                                                    2 * bshift_step)
                        elem_scatter.set_clim(vmin=min(cmap_norm_ticks), vmax=max(cmap_norm_ticks))

                    else:
                        cmap_norm_ticks = np.linspace(cmap_min_sign * abs(min(plotted_colors_var)) ** (gamma_colors),
                                                      max(plotted_colors_var) ** (gamma_colors), 7, endpoint=True)

                    # adjusting to round numbers

                    if radio_info_cmap == 'EW ratio':
                        cmap_norm_ticks = np.concatenate((cmap_norm_ticks, np.array([1])))

                        cmap_norm_ticks.sort()

                    if radio_cmap_i == 1 and min(plotted_colors_var) < 0:
                        # cmap_norm_ticks=np.concatenate((cmap_norm_ticks,np.array([0])))
                        # cmap_norm_ticks.sort()
                        pass

                    if radio_info_cmap != 'Velocity shift':
                        # maintaining the sign with the square norm
                        cmap_norm_ticks = cmap_norm_ticks ** (1 / gamma_colors)

                        cmap_norm_ticks = np.concatenate((np.array([min(plotted_colors_var)]), cmap_norm_ticks))

                        cmap_norm_ticks.sort()


                else:
                    cmap_norm_ticks = None

                # only creating the colorbar if there is information to display
                if is_colored_scat and radio_info_cmap not in type_1_colorcode:

                    if radio_info_cmap == 'Time':

                        low_bound_date = max(
                            min(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj]) \
                                                    [global_mask_intime_norepeat])),
                            mdates.date2num(slider_date[0]))

                        high_bound_date = min(
                            max(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj]) \
                                                    [global_mask_intime_norepeat])),
                            mdates.date2num(slider_date[1]))

                        # manually readjusting for small durations because the AutoDateLocator doesn't work well
                        time_range = high_bound_date - low_bound_date

                        if time_range < 1:
                            date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
                        elif time_range < 5:
                            date_format = mdates.DateFormatter('%Y-%m-%d %Hh')
                        elif time_range < 150:
                            date_format = mdates.DateFormatter('%Y-%m-%d')
                        elif time_range < 1825:
                            date_format = mdates.DateFormatter('%Y-%m')
                        else:
                            date_format = mdates.AutoDateFormatter(mdates.AutoDateLocator())

                        cb = plt.colorbar(elem_scatter_forcol, cax=ax_cb, ticks=mdates.AutoDateLocator(),
                                          format=date_format, )
                    else:
                        cb = plt.colorbar(elem_scatter_forcol, cax=ax_cb, extend='min' if radio_info_cmap == 'nH' else None)
                        if cmap_norm_ticks is not None:
                            cb.set_ticks(cmap_norm_ticks)

                    # cb.ax.minorticks_off()

                    if radio_cmap_i == 1:
                        cb_add_str = ' (km/s)'
                    else:
                        cb_add_str = ''

                    if radio_info_cmap == 'Inclination':
                        cb.set_label(cmap_incl_type_str + ' of the source inclination (°)', labelpad=10)
                    elif radio_info_cmap == 'Time':
                        cb.set_label('Observation date', labelpad=30)
                    elif radio_info_cmap == 'nH':
                        cb.set_label(r'nH ($10^{22}$ cm$^{-2}$)', labelpad=10)
                    elif radio_info_cmap == 'kT':
                        cb.set_label(r'disk temperature (keV)', labelpad=10)
                    else:
                        if restrict_threshold:
                            cb.set_label(((
                                              'minimal ' if radio_cmap_i == 1 else 'maximal ') if radio_info_cmap != 'EW ratio' else '') + (
                                             radio_info_label[radio_cmap_i - 1].lower() if radio_info_cmap != 'Del-C' else
                                             radio_info_label[radio_cmap_i - 1]) +
                                         ' in significant detections\n for each observation' + cb_add_str, labelpad=10)
                        else:
                            cb.set_label(((
                                              'minimal ' if radio_cmap_i == 1 else 'maximal ') if radio_info_cmap != 'EW ratio' else '') + (
                                             radio_info_label[radio_cmap_i - 1].lower() if radio_info_cmap != 'Del-C' else
                                             radio_info_label[radio_cmap_i - 1]) +
                                         ' in all detections\n for each observation' + cb_add_str, labelpad=10)

        label_obj_plotted = np.repeat(False, len(abslines_infos_perobj[mask_obj]))

        #### non detections HID

        id_obj_det = 0
        id_obj_nondet = 0

        scatter_nondet = []

        # loop for non detection, separated to be able to restrict the color range in case of non detection
        for i_obj_base, abslines_obj_base in enumerate(abslines_infos_perobj[mask_obj]):

            # skipping everything if we don't plot nondetections
            if not display_nondet:
                continue

            # defining the index of the object in the entire array if asked to, in order to avoid changing colors
            if global_colors:
                i_obj_glob = np.argwhere(obj_list == obj_list[mask_obj][i_obj_base])[0][0]
            else:
                i_obj_glob = i_obj_base

            '''
            # The shape of each abslines_obj is (uncert,info,line,obs)
            '''

            # we use non-detection-masked arrays for non detection to plot them even while restricting the colors to a part of the sample

            x_hid_base = lum_list[mask_obj][i_obj_base].T[2][0] / lum_list[mask_obj][i_obj_base].T[1][0]
            y_hid_base = lum_list[mask_obj][i_obj_base].T[4][0]

            if len(mask_obj) == 1 and np.ndim(hid_plot) == 4:
                x_hid_uncert = hid_plot.transpose(2, 0, 1, 3)[i_obj][0]
                y_hid_uncert = hid_plot.transpose(2, 0, 1, 3)[i_obj][1]
            else:
                x_hid_uncert = hid_plot_use.T[mask_obj][i_obj_base].T[0]
                y_hid_uncert = hid_plot_use.T[mask_obj][i_obj_base].T[1]

            # reconstructing standard arrays
            x_hid_uncert = np.array([[subelem for subelem in elem] for elem in x_hid_uncert])
            y_hid_uncert = np.array([[subelem for subelem in elem] for elem in y_hid_uncert])

            if broad_mode != False:

                x_hid_base = hid_plot_use[0][0][mask_obj][i_obj_base]

                # done this way to avoid overwriting lum_list if using += on y_hid_base
                if lum_broad_soft:
                    y_hid_base = lum_list[mask_obj][i_obj_base].T[4][0] + lum_broad_single[0]

            # defining the non detection as strictly non detection or everything below the significance threshold
            if display_nonsign:
                mask_det = abslines_obj_base[0][4][mask_lines] > 0.

            else:
                mask_det = abslines_obj_base[0][4][mask_lines] >= slider_sign

            # defining the mask for the time interval restriction
            datelist_obj = Time(np.array([date_list[mask_obj][i_obj_base] \
                                          for i in range(sum(mask_lines_ul if display_upper else mask_lines))]).astype(str))
            mask_intime = (datelist_obj >= Time(slider_date[0])) & (datelist_obj <= Time(slider_date[1]))

            mask_intime_norepeat = (Time(date_list[mask_obj][i_obj_base].astype(str)) >= Time(slider_date[0])) & (
                    Time(date_list[mask_obj][i_obj_base].astype(str)) <= Time(slider_date[1]))

            if broad_mode == 'BAT':
                mask_intime = np.array([(elem) & mask_with_broad for elem in mask_intime])
                mask_intime_norepeat = (mask_intime_norepeat) & mask_with_broad
            if restrict_match_INT:
                mask_intime = np.array([(elem) & mask_withtime_INT for elem in mask_intime])
                mask_intime_norepeat = (mask_intime_norepeat) & mask_withtime_INT

            # defining the mask
            prev_mask_nondet = np.isnan(
                np.array([np.nan if len(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                              max(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                          for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))]))

            mask_nondet = (np.isnan(
                np.array([np.nan if len(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) == 0 else \
                              max(abslines_obj_base[0][0][mask_lines].T[i_obs][mask_det.T[i_obs]]) \
                          for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))]))) & (mask_intime_norepeat)

            # testing if the source has detections with current restrictions to adapt the color when using source colors, if asked to
            if obj_list[mask_obj][i_obj_base] not in obj_list[mask_obj_withdet]:
                source_nondet = True

            else:
                source_nondet = False

                # increasing the counter for sources with no non detections but detections
                if len(x_hid_base[mask_nondet]) == 0:
                    id_obj_det += 1

            if len(x_hid_base[mask_nondet]) > 0:
                # note: due to problems with colormapping of the edgecolors we directly compute the color of the edges with a normalisation
                norm_cmap_incl = mpl.colors.Normalize(0, 90)

                if global_colors:
                    norm_cmap_time = mpl.colors.Normalize(
                        min(mdates.date2num(ravel_ragged(date_list)[global_mask_intime_norepeat])),
                        max(mdates.date2num(ravel_ragged(date_list)[global_mask_intime_norepeat])))
                else:
                    norm_cmap_time = mpl.colors.Normalize(
                        min(mdates.date2num(ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                        max(mdates.date2num(ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])))
                if display_upper:

                    # we define the upper limit range of points independantly to be able to have a different set of lines used for detection and
                    # upper limits if necessary

                    mask_det_ul = (abslines_obj_base[0][4][mask_lines_ul] > 0.) & (mask_intime)
                    mask_det_ul = (abslines_obj_base[0][4][mask_lines_ul] >= slider_sign) & (mask_intime)

                    mask_nondet_ul = np.isnan(np.array( \
                        [np.nan if len(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]]) == 0 else \
                             max(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]]) \
                         for i_obs in range(len(abslines_obj_base[0][0][mask_lines].T))])) & (mask_intime_norepeat)

                    # defining the sizes of upper limits (note: capped to 75eV)
                    obj_size_ul = np.array(
                        [np.nan if len(abslines_obj_base[0][0][mask_lines_ul].T[i_obs][mask_det_ul.T[i_obs]]) != 0 else \
                             min(max(abslines_obj_base[0][5][mask_lines_ul].T[i_obs][~mask_det_ul.T[i_obs]]), 75) \
                         for i_obs in range(len(abslines_obj_base[0][0][mask_lines_ul].T))])

                    # creating a display order which is the reverse of the EW size order to make sure we do not hide part the ul
                    # not needed now that the UL are not filled colorwise
                    # obj_order_sign_ul=obj_size_ul[mask_nondet_ul].argsort()[::-1]

                    # there is no need to use different markers unless we display source per color, so we limit the different triangle to this case
                    marker_ul_curr = marker_abs if display_single else marker_ul_top if \
                        ((id_obj_nondet if source_nondet else id_obj_det) if split_cmap_source else i_obj_base) % 2 != 0 and \
                        radio_info_cmap == 'Source' else marker_ul

                    if radio_info_cmap == 'Instrument':
                        color_data = [telescope_colors[elem] for elem in
                                      instru_list[mask_obj][i_obj_base][mask_nondet_ul]]

                        edgec_scat = [colors.to_rgba(elem) for elem in color_data]
                    else:

                        edgec_scat = (colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                          (colors_nondet.to_rgba(id_obj_nondet) if source_nondet else \
                                               colors_det.to_rgba(
                                                   id_obj_det))) if radio_info_cmap == 'Source' and color_nondet else \
                            cmap_info(norm_cmap_incl(incl_cmap_base[i_obj_base][cmap_incl_type])) \
                                if radio_info_cmap == 'Inclination' else \
                                cmap_info(
                                    norm_cmap_time(mdates.date2num(date_list[mask_obj][i_obj_base][mask_nondet_ul]))) \
                                    if radio_info_cmap == 'Time' else \
                                    cmap_info(cmap_norm_info(nh_plot.T[mask_obj].T[0][i_obj_base][
                                                                 mask_nondet_ul])) if radio_info_cmap == 'nH' else \
                                        cmap_info(
                                            cmap_norm_info(kt_plot.T[mask_obj].T[0][i_obj_base][mask_nondet_ul])) if (
                                                1 and radio_info_cmap == 'kT') else \
                                            diago_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                                if radio_info_cmap == 'custom_line_struct' else \
                                                custom_states_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                                    if radio_info_cmap == 'custom_acc_states' else \
                                                    custom_ionization_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                                        if radio_info_cmap == 'custom_ionization' else \
                                                        custom_outburst_color[mask_obj][i_obj_base][mask_nondet_ul] \
                                                            if radio_info_cmap == 'custom_outburst' else \
                                                            'grey'

                    # adding a failsafe to avoid problems when nothing is displayed
                    if len(edgec_scat) == 0:
                        edgec_scat = None

                    if broad_mode == 'BAT' and not sign_broad_hid_BAT:
                        alpha_nondet_ul_full = np.where(mask_sign_high_E, 1, 0.3)

                        alpha_nondet_ul = alpha_nondet_ul_full[mask_nondet_ul]

                        # in the case where alpha is not singular the len of the color keyword needs to match it
                        color_val_nondet_ul = np.array([[0., 0., 0., 0.]] * len(alpha_nondet_ul))

                    else:
                        alpha_nondet_ul = alpha_abs - 0.2

                        color_val_nondet_ul = [0., 0., 0., 0.]

                    if hatch_unstable:
                        # for now done this way because hatch_color hasn't been introduced to matplotlib yet so linked to
                        # edgecolor
                        # displaying the 2021 unstable SEDs differently
                        unstable_list = ['4130010101-001',
                                         '4130010102-008',
                                         '4130010104-001',
                                         '4130010105-002',
                                         '4130010106-001',
                                         '4130010107-002',
                                         '4130010108-002',
                                         '4130010109-001',
                                         '4130010112-006',
                                         '4130010114-001']
                        hatch_mask = [elem in unstable_list for elem in observ_list[mask_obj][i_obj][mask_nondet_ul]]

                        # try:
                        # unstable scatter, sames as below but with zorder below and black hatch, no facecolor
                        # smaller zorder to only show the hatch inside of non-colored markers
                        # note that having no edgecolor and a hatch color only works with alpha=None, a repeating
                        # facolor and no edgecolor
                        elem_scatter_unstable = ax_hid.scatter(
                            x_hid_base[mask_nondet_ul][hatch_mask], y_hid_base[mask_nondet_ul][hatch_mask],
                            marker=marker_ul_curr,
                            facecolor=np.repeat('None', sum(hatch_mask)),
                            hatch='////',
                            s=norm_s_lin * obj_size_ul[mask_nondet_ul][hatch_mask] ** norm_s_pow,
                            label='',
                            zorder=0, alpha=None,
                            ls='--' if (
                                    display_incl_inside and not bool_incl_inside[mask_obj][
                                i_obj_base] or dash_noincl and bool_noincl[mask_obj][i_obj_base]) else 'solid',
                            plotnonfinite=True)

                    elem_scatter_nondet = ax_hid.scatter(
                        x_hid_base[mask_nondet_ul], y_hid_base[mask_nondet_ul], marker=marker_ul_curr,
                        facecolor=color_val_nondet_ul, edgecolor=edgec_scat,
                        s=norm_s_lin * obj_size_ul[mask_nondet_ul] ** norm_s_pow,
                        label='' if not color_nondet else (
                            obj_list[mask_obj][i_obj_base] if not label_obj_plotted[i_obj_base] and \
                                                              (radio_info_cmap == 'Source' or display_edgesource) else ''),
                        zorder=500, alpha=alpha_nondet_ul,
                        cmap=cmap_info if radio_info_cmap in ['Inclination', 'Time'] else None, ls='--' if (
                                display_incl_inside and not bool_incl_inside[mask_obj][
                            i_obj_base] or dash_noincl and bool_noincl[mask_obj][i_obj_base]) else 'solid',
                        plotnonfinite=True)

                    # we re-enforce the facecolor after otherwise because it can be overwirtten by alpha modifications
                    elem_scatter_nondet.set_facecolor('none')

                    scatter_nondet += [elem_scatter_nondet]

                else:

                    if radio_info_cmap == 'Instrument':
                        color_data = [telescope_colors[elem] for elem in
                                      instru_list[mask_obj][i_obj_base][mask_nondet]]

                        c_scat_nondet = [colors.to_rgba(elem) for elem in color_data]
                    else:

                        c_scat_nondet = np.array([(colors_obj.to_rgba(i_obj_glob) if not split_cmap_source else \
                                                       (colors_nondet.to_rgba(id_obj_nondet) if source_nondet else \
                                                            colors_det.to_rgba(
                                                                id_obj_det)))]) if radio_info_cmap == 'Source' and color_nondet else \
                            np.repeat(incl_cmap_base[i_obj_base][cmap_incl_type], sum(mask_nondet)) \
                                if radio_info_cmap == 'Inclination' else \
                                mdates.date2num(date_list[mask_obj][i_obj_base][mask_nondet]) \
                                    if radio_info_cmap == 'Time' else \
                                    nh_plot.T[mask_obj].T[0][i_obj_base][mask_nondet] if radio_info_cmap == 'nH' else \
                                        kt_plot.T[mask_obj].T[0][i_obj_base][mask_nondet] if radio_info_cmap == 'kT' else \
                                            diago_color[mask_obj][i_obj_base][mask_nondet] \
                                                if radio_info_cmap == 'custom_line_struct' else \
                                                custom_states_color[mask_obj][i_obj_base][mask_nondet] \
                                                    if radio_info_cmap == 'custom_acc_states' else \
                                                    custom_ionization_color[mask_obj][i_obj_base][mask_nondet] \
                                                        if radio_info_cmap == 'custom_ionization' else \
                                                        custom_outburst_color[mask_obj][i_obj_base][mask_nondet] \
                                                            if radio_info_cmap == 'custom_outburst' else \
                                                            'grey'

                    if broad_mode == 'BAT' and not sign_broad_hid_BAT:
                        alpha_nondet_full = np.where(mask_sign_high_E, 1, 0.3)

                        alpha_nondet = alpha_nondet_full[mask_nondet]



                    else:
                        alpha_nondet = alpha_abs - 0.2

                    elem_scatter_nondet = ax_hid.scatter(x_hid_base[mask_nondet], y_hid_base[mask_nondet],
                                                         marker=marker_nondet,
                                                         c=c_scat_nondet, cmap=cmap_info, norm=cmap_norm_info,
                                                         label='' if not color_nondet else (
                                                             obj_list[mask_obj][i_obj_base] if not label_obj_plotted[
                                                                 i_obj_base] and \
                                                                                               (
                                                                                                           radio_info_cmap == 'Source' or display_edgesource) else ''),
                                                         zorder=1000,
                                                         edgecolor='black', alpha=alpha_nondet,
                                                         plotnonfinite=True)

                    # note: the plot non finite allows to plot the nan values passed to the colormap with the color predefined as bad in
                    # the colormap

                    if display_hid_error:

                        # in order to get the same clim as with the standard scatter plots, we manually readjust the rgba values of the colors before plotting
                        # the errorbar "empty" and changing its color manually (because as of now matplotlib doesn't like multiple color inputs for errbars)
                        if radio_info_cmap in type_1_cm:
                            if radio_info_cmap == 'Inclination':
                                cmap_norm_info.vmin = 0
                                cmap_norm_info.vmax = 90
                            elif radio_info_cmap == 'Time':
                                cmap_norm_info.vmin = max(min(mdates.date2num(
                                    ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                    mdates.date2num(slider_date[0]))
                                cmap_norm_info.vmax = min(max(mdates.date2num(
                                    ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                    mdates.date2num(slider_date[1]))

                            elif radio_info_cmap == 'nH':
                                cmap_norm_info.vmin = min(
                                    ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat])
                                cmap_norm_info.vmax = max(
                                    ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat])
                            elif radio_info_cmap == 'kT':
                                cmap_norm_info.vmin = kt_min
                                cmap_norm_info.vmax = kt_max

                            colors_func = mpl.cm.ScalarMappable(norm=cmap_norm_info, cmap=cmap_info)

                            c_scat_nondet_rgba_clim = colors_func.to_rgba(c_scat_nondet)

                        elem_err_nondet = ax_hid.errorbar(x_hid_uncert[0][mask_nondet], y_hid_uncert[0][mask_nondet],
                                                          xerr=x_hid_uncert[1:].T[mask_nondet].T,
                                                          yerr=y_hid_uncert[1:].T[mask_nondet].T, marker='None',
                                                          linestyle='None', linewidth=0.5,
                                                          c=c_scat_nondet if radio_info_cmap not in type_1_cm else None,
                                                          label='', zorder=1000, alpha=1.)

                        if radio_info_cmap in type_1_cm:
                            for elem_children in elem_err_nondet.get_children()[1:]:
                                elem_children.set_colors(c_scat_nondet_rgba_clim)

                if radio_info_cmap == 'Source' and color_nondet:
                    label_obj_plotted[i_obj_base] = True

                if radio_info_cmap in type_1_cm:

                    if radio_info_cmap == 'Inclination':
                        elem_scatter_nondet.set_clim(vmin=0, vmax=90)

                        # if display_hid_error:
                        #     elem_err_nondet.set_clim(vmin=0,vmax=90)

                    elif radio_info_cmap == 'Time':
                        if global_colors:
                            elem_scatter_nondet.set_clim(
                                vmin=min(mdates.date2num(ravel_ragged(date_list))),
                                vmax=max(mdates.date2num(ravel_ragged(date_list))))
                        else:
                            elem_scatter_nondet.set_clim(
                                vmin=max(min(mdates.date2num(
                                    ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                    mdates.date2num(slider_date[0])),
                                vmax=min(max(mdates.date2num(
                                    ravel_ragged(date_list[mask_obj])[global_mask_intime_norepeat])),
                                    mdates.date2num(slider_date[1])))
                            # if display_hid_error:
                        #     elem_err_nondet.set_clim(
                        #     vmin=max(min(mdates.date2num(ravel_ragged(date_list[mask_obj][global_mask_intime_norepeat]))),mdates.date2num(slider_date[0])),
                        #     vmax=min(max(mdates.date2num(ravel_ragged(date_list[mask_obj][global_mask_intime_norepeat]))),mdates.date2num(slider_date[1])))

                    elif radio_info_cmap == 'nH':
                        if global_colors:
                            elem_scatter_nondet.set_clim(vmin=min(ravel_ragged(nh_plot[0])),
                                                         vmax=max(ravel_ragged(nh_plot[0])))
                        else:
                            elem_scatter_nondet.set_clim(
                                vmin=min(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]),
                                vmax=max(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]))
                    elif radio_info_cmap == 'kT':
                        elem_scatter_nondet.set_clim(vmin=kt_min,
                                                     vmax=kt_max)

                    if len(elem_scatter_nondet.get_sizes()) > 0:
                        is_colored_scat_nondet = True

                    # creating the colorbar at the end if it hasn't been created with the detections
                    if i_obj_base == len(
                            abslines_infos_perobj[mask_obj]) - 1 and not is_colored_scat and is_colored_scat_nondet:

                        # creating an empty scatter with a 'c' value to serve as base for the colorbar
                        elem_scatter_empty = ax_hid.scatter(x_hid_base[mask_nondet][False], y_hid_base[mask_nondet][False],
                                                            marker=None,
                                                            c=cmap_info(norm_cmap_time(mdates.date2num(
                                                                date_list[mask_obj][i_obj_base][mask_nondet])))[False],
                                                            label='', zorder=1000, edgecolor=None, cmap=cmap_info, alpha=1.)

                        if radio_info_cmap == 'Inclination':

                            elem_scatter_empty.set_clim(vmin=0, vmax=90)

                            cb = plt.colorbar(elem_scatter_empty, cax=ax_cb)

                            cb.set_label(cmap_incl_type_str + ' of the source inclination (°)', labelpad=10)
                        elif radio_info_cmap == 'Time':

                            low_bound_date = max(
                                min(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj]) \
                                                        [global_mask_intime_norepeat])),
                                mdates.date2num(slider_date[0]))

                            high_bound_date = min(
                                max(mdates.date2num(ravel_ragged(date_list[True if global_colors else mask_obj]) \
                                                        [global_mask_intime_norepeat])),
                                mdates.date2num(slider_date[1]))

                            elem_scatter_empty.set_clim(vmin=low_bound_date, vmax=high_bound_date)

                            # manually readjusting for small durations because the AutoDateLocator doesn't work well
                            time_range = high_bound_date - low_bound_date

                            if time_range < 150:
                                date_format = mdates.DateFormatter('%Y-%m-%d')
                            elif time_range < 1825:
                                date_format = mdates.DateFormatter('%Y-%m')
                            else:
                                date_format = mdates.AutoDateFormatter(mdates.AutoDateLocator())

                            cb = plt.colorbar(elem_scatter_empty, cax=ax_cb, ticks=mdates.AutoDateLocator(),
                                              format=date_format)

                            cb.set_label('Observation date', labelpad=10)

                        elif radio_info_cmap == 'nH':
                            elem_scatter_empty.set_clim(
                                vmin=min(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]),
                                vmax=max(ravel_ragged(nh_plot.T[mask_obj].T[0])[global_mask_intime_norepeat]))
                            cb = plt.colorbar(elem_scatter_empty, cax=ax_cb, extend='min')

                            cb.set_label(r'nH ($10^{22}$ cm$^{-2}$)')

                        elif radio_info_cmap == 'kT':
                            elem_scatter_empty.set_clim(vmin=kt_min, vmax=kt_max)

                            cb = plt.colorbar(elem_scatter_empty, cax=ax_cb)

                            cb.set_label(r'disk temperature (keV)')

                # only adding to the index if there are non detections
                if source_nondet:
                    id_obj_nondet += 1
                else:
                    id_obj_det += 1

        # taking off the axes in the colorbar axes if no colorbar was displayed
        if radio_info_cmap not in type_1_colorcode and cb is None:
            ax_cb.axis('off')

        # making a copy array of the single object date to avoid issues when adding new things
        datelist_time_evol = deepcopy(datelist_obj[0][mask_intime[0]])
        x_hid_time_evol = deepcopy(x_hid_base[mask_intime[0]])
        y_hid_time_evol = deepcopy(y_hid_base[mask_intime[0]])
    else:
        datelist_time_evol=[]
        x_hid_time_evol=np.array([])
        y_hid_time_evol=np.array([])

    # for the LEdd additional points only
    HLD_hard_rescale = False
    HLD_soft_rescale = False

    #dictionnaries to avoid repeating legend marker examples later when remaking the legend manually
    dict_markers_HLD_soft={}
    dict_markers_HLD_hard={}

    if not base_sample_points_bool:
        ax_hid.set_xlim(0.031114479663310157, 1.8280376809036172)
        ax_hid.set_ylim(2.4705851795648526e-05, 0.7014911926354677)

    if base_sample_points_bool and (not (zoom == 'auto' or broad_mode != False)):
        xlims=plt.xlim()
        ylims=plt.ylim()

    # displaying the additional points with LEdd values if any
    for i in range(len(additional_HLD_points_LEdd)):
        point_row = additional_HLD_points_LEdd.iloc[i]

        if point_row['L_3-10/L_Edd'] not in [0., None]:

            line_match_mask = np.array(np.array(additional_line_points[['Source','ObsID','Date (UTC)','Telescope']]) ==\
                                       np.array(point_row[['Source','ObsID','Date (UTC)','Telescope']])).all(1)

            line_match_row = line_match_mask.any()

            point_row_det = False
            point_row_size = 1e3
            point_row_size_UL = 1e3

            if line_match_row:
                point_row_line = additional_line_points[line_match_mask]

                # size from the maximum value between the two lines.
                # We assume the line is significant if these values are provided
                point_row_size = np.max(point_row_line[['EW_FeKa25', 'EW_FeKa26']])
                point_row_det = point_row_size != 0.
                if not point_row_det:
                    point_row_size_UL = np.max(point_row_line[['EW_FeKa25_UL', 'EW_FeKa26_UL']])

            point_marker=mpl_cut_star if point_row['obscured'] else "P" if not line_match_row else marker_abs
            point_color=point_row['color'] if point_row['color'] not in ['', None] else 'black'
            point_edge_color=None if point_row['obscured'] else \
                             point_color if line_match_row and not point_row_det else 'black'

            if point_row['HR_[6-10]/[3-6]'] not in [0., None] and not broad_mode:

                HLD_soft_rescale = True

                label_point = point_row['Source'] if point_row['Source'] not in dict_markers_HLD_soft.keys() or \
                        ([point_marker, point_color, point_row_det] \
                    not in dict_markers_HLD_soft[str(point_row['Source'])]) else ''

                if str(point_row['Source']) not in dict_markers_HLD_soft.keys():
                    dict_markers_HLD_soft[str(point_row['Source'])]=[[point_marker,point_color,point_row_det]]
                elif label_point!='':
                    dict_markers_HLD_soft[str(point_row['Source'])]+=[[point_marker,point_color,point_row_det]]

                ax_hid.scatter(point_row['HR_[6-10]/[3-6]'], point_row['L_3-10/L_Edd'],
                               color=point_color if (not line_match_row or point_row_det) else None,
                               facecolor='None' if line_match_row and not point_row_det else None,
                               # abs marker for points with line infos
                               marker=point_marker,
                               s=100 if not line_match_row else
                               norm_s_lin * (point_row_size if point_row_det else point_row_size_UL) ** norm_s_pow,
                               edgecolor=point_edge_color,
                               zorder=1e6,
                               label=label_point)


                # adding to x_hid_base if there is a date
                if point_row['Date (UTC)'] not in ['', None]:
                    if len(datelist_time_evol)==0:
                        datelist_time_evol=[Time(point_row['Date (UTC)'])]
                    else:
                        datelist_time_evol.insert(0, Time(point_row['Date (UTC)']))

                    x_hid_time_evol = np.concatenate([np.array([point_row['HR_[6-10]/[3-6]']]), x_hid_time_evol])
                    y_hid_time_evol = np.concatenate([np.array([point_row['L_3-10/L_Edd']]), y_hid_time_evol])

            if point_row['HR_[15-50]/[3-6]'] not in [0., None] and broad_mode:

                label_point = point_row['Source'] if point_row['Source'] not in dict_markers_HLD_hard.keys() or \
                        ([point_marker, point_color, point_row_det] \
                    not in dict_markers_HLD_hard[str(point_row['Source'])]) else ''

                if str(point_row['Source']) not in dict_markers_HLD_hard.keys():
                    dict_markers_HLD_hard[str(point_row['Source'])]=[[point_marker,point_color,point_row_det]]
                elif label_point!='':
                    dict_markers_HLD_hard[str(point_row['Source'])]+=[[point_marker,point_color,point_row_det]]

                HLD_hard_rescale = True

                ax_hid.scatter(point_row['HR_[15-50]/[3-6]'], point_row['L_3-10/L_Edd'],
                               color=point_color if (not line_match_row or point_row_det) else None,
                               facecolor='None' if line_match_row and not point_row_det else None,
                               # abs marker for points with line infos
                               marker=point_marker,
                               s=100 if not line_match_row else
                               norm_s_lin * (point_row_size if point_row_det else point_row_size_UL) ** norm_s_pow,
                               edgecolor=point_edge_color,
                               zorder=1e6,
                               label=label_point)

                # adding to x_hid_base if there is a date
                if point_row['Date (UTC)'] not in ['', None]:
                    if len(datelist_time_evol)==0:
                        datelist_time_evol=[Time(point_row['Date (UTC)'])]
                    else:
                        datelist_time_evol.insert(0, Time(point_row['Date (UTC)']))

                    x_hid_time_evol = np.concatenate([np.array([point_row['HR_[15-50]/[3-6]']]), x_hid_time_evol])
                    y_hid_time_evol = np.concatenate([np.array([point_row['L_3-10/L_Edd']]), y_hid_time_evol])

        if broad_mode != False or zoom == 'auto':
            # recomputing the x and ylims

            if HLD_hard_rescale:
                if not base_sample_points_bool:
                    xlims = [min(additional_HLD_points_LEdd['HR_[15-50]/[3-6]']),
                         max(additional_HLD_points_LEdd['HR_[15-50]/[3-6]'])]
                else:
                    xlims = [min(xlims[0], min(additional_HLD_points_LEdd['HR_[15-50]/[3-6]'])),
                         max(xlims[1], max(additional_HLD_points_LEdd['HR_[15-50]/[3-6]']))]
            if HLD_soft_rescale:
                if not base_sample_points_bool:
                    xlims = [min(additional_HLD_points_LEdd['HR_[6-10]/[3-6]']),
                             max(additional_HLD_points_LEdd['HR_[6-10]/[3-6]'])]
                else:
                    xlims = [min(xlims[0], min(additional_HLD_points_LEdd['HR_[6-10]/[3-6]'])),
                             max(xlims[1], max(additional_HLD_points_LEdd['HR_[6-10]/[3-6]']))]

            if HLD_hard_rescale or HLD_soft_rescale:
                if not base_sample_points_bool:
                    ylims = [min(additional_HLD_points_LEdd['L_3-10/L_Edd']),
                             max(additional_HLD_points_LEdd['L_3-10/L_Edd'])]
                else:
                    ylims = [min(ylims[0], min(additional_HLD_points_LEdd['L_3-10/L_Edd'])),
                             max(ylims[1], max(additional_HLD_points_LEdd['L_3-10/L_Edd']))]

                rescale_flex(ax_hid, xlims, ylims, 0.05)

    point_list_LEdd_3_10 = []
    point_list_LEdd_HR_soft = []
    point_list_LEdd_HR_hard = []

    # displaying the additional points with flux values if any
    for i in range(len(additional_HLD_points_flux)):
        point_row = additional_HLD_points_flux.iloc[i]


        column_3_10_df = ['flux_3-4', 'flux_4-5', 'flux_5-6', 'flux_6-7', 'flux_7-8', 'flux_8-9', 'flux_9-10']
        column_3_6_df = ['flux_3-4', 'flux_4-5', 'flux_5-6']
        column_6_10_df = ['flux_6-7', 'flux_7-8', 'flux_8-9', 'flux_9-10']

        if np.sum([point_row[column] in [0., None] for column in column_3_10_df]) == 0:

            line_match_mask = np.array(np.array(additional_line_points[['Source','ObsID','Date (UTC)','Telescope']]) ==\
                                       np.array(point_row[['Source','ObsID','Date (UTC)','Telescope']])).all(1)

            line_match_row = line_match_mask.any()

            point_row_det = False
            point_row_size = 1e3
            point_row_size_UL = 1e3

            if line_match_row:
                point_row_line = additional_line_points[line_match_mask]

                # size from the maximum value between the two lines.
                # We assume the line is significant if these values are provided
                point_row_size = np.max(point_row_line[['EW_FeKa25', 'EW_FeKa26']])
                point_row_det = point_row_size != 0.
                if not point_row_det:
                    point_row_size_UL = np.max(point_row_line[['EW_FeKa25_UL', 'EW_FeKa26_UL']])


            id_obj_obs_arr= np.argwhere([point_row['Source'] == obj_list[mask_obj]])

            if len(id_obj_obs_arr) != 0:
                edd_factor_point = Edd_factor_restrict[id_obj_obs_arr[0][0]]
            else:

                #here the names should match the dictionnaries
                d_obj_point,m_obj_point=dist_mass_indiv(dict_linevis,point_row['Source'],use_unsure_mass_dist=True)

                edd_factor_point = 4 * np.pi * (d_obj_point * 1e3 * 3.086e18) ** 2 / (1.26e38 * m_obj_point)


            point_LEdd_3_10 = np.sum([point_row[column] for column in column_3_10_df]) * edd_factor_point
            point_list_LEdd_3_10 += [point_LEdd_3_10]
            point_LEdd_3_6 = np.sum([point_row[column] for column in column_3_6_df]) * edd_factor_point
            point_LEdd_6_10 = np.sum([point_row[column] for column in column_6_10_df]) * edd_factor_point
            point_LEdd_HR_soft = point_LEdd_6_10 / point_LEdd_3_6
            point_list_LEdd_HR_soft += [point_LEdd_HR_soft]

            point_marker=mpl_cut_star if point_row['obscured'] else "P" if not line_match_row else marker_abs

            # for AO2
            point_marker=marker_ul

            point_color=point_row['color'] if point_row['color'] not in ['', None] else 'black'
            point_edge_color=None if point_row['obscured'] else \
                             point_color if line_match_row and not point_row_det else 'black'

            if not broad_mode:

                label_point = point_row['Source'] if point_row['Source'] not in dict_markers_HLD_soft.keys() or \
                        ([point_marker, point_color, point_row_det] \
                    not in dict_markers_HLD_soft[str(point_row['Source'])]) else ''

                if str(point_row['Source']) not in dict_markers_HLD_soft.keys():
                    dict_markers_HLD_soft[str(point_row['Source'])]=[[point_marker,point_color,point_row_det]]
                elif label_point!='':
                    dict_markers_HLD_soft[str(point_row['Source'])]+=[[point_marker,point_color,point_row_det]]

                ax_hid.scatter(point_LEdd_HR_soft, point_LEdd_3_10,
                               # no marker color for ULs
                               color=point_color if (not line_match_row or point_row_det) else None,
                               facecolor='None' if line_match_row and not point_row_det else None,
                               # abs marker for points with line infos
                               marker=point_marker,
                               s=100 if not line_match_row else
                               norm_s_lin * (point_row_size if point_row_det else point_row_size_UL) ** norm_s_pow,
                               edgecolor=point_edge_color, zorder=1e6,
                               label=label_point)


                # adding to x_hid_base if there is a date
                if point_row['Date (UTC)'] not in ['', None]:

                    if len(datelist_time_evol)==0:
                        datelist_time_evol=[Time(point_row['Date (UTC)'])]
                    else:
                        datelist_time_evol.insert(0, Time(point_row['Date (UTC)']))

                    x_hid_time_evol = np.concatenate([np.array([point_LEdd_HR_soft]), x_hid_time_evol])
                    y_hid_time_evol = np.concatenate([np.array([point_LEdd_3_10]), y_hid_time_evol])

            if broad_mode and point_row['Date (UTC)'] not in ['', None] and point_row['Source'] == '4U1630-47':

                bat_lc_df_scat = dict_linevis['bat_lc_df_scat']

                bat_lc_mjd_scat = np.array(bat_lc_df_scat[bat_lc_df_scat.columns[0]])

                mjd_point = int(Time(point_row['Date (UTC)']).mjd)

                if mjd_point in bat_lc_mjd_scat:
                    bat_rate_point = float(bat_lc_df_scat[bat_lc_df_scat.columns[1]] \
                                               [bat_lc_mjd_scat == mjd_point].iloc[0])

                    # using the more direct conversion here
                    point_LEdd_15_50 = bat_rate_point * 10 ** (-0.36 + np.log10(edd_factor_point / 7598382.454))

                    point_LEdd_HR_hard = point_LEdd_15_50 / point_LEdd_3_6

                    point_list_LEdd_HR_hard += [point_LEdd_HR_hard]

                    label_point = point_row['Source'] if point_row['Source'] not in dict_markers_HLD_hard.keys() or \
                                                         ([point_marker, point_color, point_row_det] \
                                                          not in dict_markers_HLD_hard[
                                                              str(point_row['Source'])]) else ''

                    if str(point_row['Source']) not in dict_markers_HLD_hard.keys():
                        dict_markers_HLD_hard[str(point_row['Source'])] = [[point_marker, point_color, point_row_det]]
                    elif label_point != '':
                        dict_markers_HLD_hard[str(point_row['Source'])] += [[point_marker, point_color, point_row_det]]

                    ax_hid.scatter(point_LEdd_HR_hard, point_LEdd_3_10,
                                   # no marker color for ULs
                                   color=point_color if (not line_match_row or point_row_det) else None,
                                   facecolor='None' if line_match_row and not point_row_det else None,
                                   # abs marker for points with line infos
                                   marker=point_marker,
                                   s=100 if not line_match_row else
                                   norm_s_lin * (point_row_size if point_row_det else point_row_size_UL) ** norm_s_pow,
                                   edgecolor=point_edge_color,
                                   zorder=1e6,
                                   label=label_point)

                    if len(datelist_time_evol)==0:
                        datelist_time_evol=[Time(point_row['Date (UTC)'])]
                    else:
                        datelist_time_evol.insert(0, Time(point_row['Date (UTC)']))

                    x_hid_time_evol = np.concatenate([np.array([point_LEdd_HR_hard]), x_hid_time_evol])
                    y_hid_time_evol = np.concatenate([np.array([point_LEdd_3_10]), y_hid_time_evol])

    if broad_mode != False or zoom!='manual':

        perform_unzoom=zoom=='auto'
            
        # recomputing the x and ylims
        if broad_mode != False and len(point_list_LEdd_HR_hard) > 0:
            if xlims is None:
                xlims = [min(point_list_LEdd_HR_hard),
                     max(point_list_LEdd_HR_hard)]
            else:
                xlims = [min(xlims[0], min(point_list_LEdd_HR_hard)),
                     max(xlims[1], max(point_list_LEdd_HR_hard))]
            if xlims[0]<plt.xlim()[0] or xlims[1]>plt.xlim()[1]:
                perform_unzoom=True
                    
        if broad_mode == False and len(point_list_LEdd_HR_soft) > 0:

            if xlims is None:
                xlims = [min(point_list_LEdd_HR_soft),
                     max(point_list_LEdd_HR_soft)]
            else:
                xlims = [min(xlims[0], min(point_list_LEdd_HR_soft)),
                     max(xlims[1], max(point_list_LEdd_HR_soft))]

            if xlims[0] < plt.xlim()[0] or xlims[1] > plt.xlim()[1]:
                perform_unzoom = True

        if len(point_list_LEdd_3_10) > 0:
            if ylims is None:
                ylims = [min(point_list_LEdd_3_10),
                     max(point_list_LEdd_3_10)]
            else:
                ylims = [min(ylims[0], min(point_list_LEdd_3_10)),
                     max(ylims[1], max(point_list_LEdd_3_10))]
        
        if xlims[0]<plt.xlim()[0] or xlims[1]<plt.xlim()[1]:
            perform_unzoom=True
            
        #aside from automode, we rescale only if it's necessary to not miss points
        if perform_unzoom:    
            rescale_flex(ax_hid, xlims, ylims, 0.05)

    #### Displaying arrow evolution if needed and if there are points
    if display_single and display_evol_single and sum(global_mask_intime_norepeat) > 1 and display_nondet:

        # odering the points depending on the observation date
        date_order = datelist_time_evol.argsort()

        # plotting the main line between all points
        ax_hid.plot(x_hid_time_evol[date_order], y_hid_time_evol[date_order], color='grey',
                    linewidth=0.5, alpha=0.5)

        # computing the position of the arrows to superpose to the lines
        xarr_start = x_hid_time_evol[date_order][range(len(x_hid_time_evol[date_order]) - 1)].astype(float)
        xarr_end = x_hid_time_evol[date_order][range(1, len(x_hid_time_evol[date_order]))].astype(float)
        yarr_start = y_hid_time_evol[date_order][range(len(y_hid_time_evol[date_order]) - 1)].astype(float)
        yarr_end = y_hid_time_evol[date_order][range(1, len(y_hid_time_evol[date_order]))].astype(float)

        # mask to know if we can do a log computation of the arrow positions or not (aka not broad mode or
        # x positions above the lintresh threshold
        x_arr_log_ok = np.array([not broad_mode or \
                                 (elem_x_s >= broad_x_linthresh and elem_x_e >= broad_x_linthresh) \
                                 for (elem_x_s, elem_x_e) in zip(xarr_start, xarr_end)])

        # preventing error when no point
        if sum(x_arr_log_ok) != 0:

            # linear version first
            xpos = (xarr_start + xarr_end) / 2
            ypos = (yarr_start + yarr_end) / 2

            xdir = xarr_end - xarr_start
            ydir = yarr_end - yarr_start

            # log version in the mask
            try:
                xpos[x_arr_log_ok] = 10 ** ((np.log10(xarr_start[x_arr_log_ok]) + np.log10(xarr_end[x_arr_log_ok])) / 2)
                ypos[x_arr_log_ok] = 10 ** ((np.log10(yarr_start[x_arr_log_ok]) + np.log10(yarr_end[x_arr_log_ok])) / 2)
            except:
                breakpoint()

            xdir[x_arr_log_ok] = 10 ** (np.log10(xarr_end[x_arr_log_ok]) - np.log10(xarr_start[x_arr_log_ok]))
            ydir[x_arr_log_ok] = 10 ** (np.log10(yarr_end[x_arr_log_ok]) - np.log10(yarr_start[x_arr_log_ok]))

            # this is the offset from the position, since we make the arrow start at the middle point of
            # the segment we don't want it to go any further so we put it almost at the same value
            arrow_size_frac = 0.001

            for X, Y, dX, dY, log_ok in zip(xpos, ypos, xdir, ydir, x_arr_log_ok):
                if log_ok:
                    ax_hid.annotate("", xytext=(X, Y), xy=(10 ** (np.log10(X) + arrow_size_frac * np.log10(dX)),
                                                           10 ** (np.log10(Y) + arrow_size_frac * np.log10(dY))),
                                    arrowprops=dict(arrowstyle='->', color='grey', alpha=0.5), size=10)
                else:
                    ax_hid.annotate("", xytext=(X, Y), xy=(X + arrow_size_frac * dX,
                                                           Y + arrow_size_frac * dY),
                                    arrowprops=dict(arrowstyle='->', color='grey', alpha=0.5), size=10)

        # else:
        #     xpos = (xarr_start + xarr_end) / 2
        #     ypos = (yarr_start + yarr_end) / 2
        #
        #     xdir = xarr_end - xarr_start
        #     ydir = yarr_end - yarr_start

    ####displaying the thresholds if asked to

    if display_dicho:

        if broad_mode == 'BAT':
            if HR_broad_6_10:
                # vertical
                pass
                # ax_hid.axline((0.1, 1e-6), (0.1, 10), ls='--', color='grey')
            else:
                # vertical
                ax_hid.axline((0.1, 1e-6), (0.1, 10), ls='--', color='grey')

        else:
            # horizontal
            ax_hid.axline((0.01, 1e-2), (10, 1e-2), ls='--', color='grey')

            # vertical
            ax_hid.axline((0.8, 1e-6), (0.8, 10), ls='--', color='grey')

        # restricting the graph to the portion inside the thrsesolds
        # ax_hid.set_xlim(ax_hid.get_xlim()[0],0.8)
        # ax_hid.set_ylim(1e-2,ax_hid.get_ylim()[1])

    # Shading the Ledd and HR limits if there are some
    if restrict_Ledd_low != 0:
        ax_hid.fill_between(ax_hid.get_xlim(), [ax_hid.get_ylim()[0], ax_hid.get_ylim()[0]],
                            [restrict_Ledd_low, restrict_Ledd_low],
                            color='grey', alpha=0.2)

    if restrict_Ledd_high != 0:
        ax_hid.fill_between(ax_hid.get_xlim(), [restrict_Ledd_high, restrict_Ledd_high],
                            [ax_hid.get_ylim()[1], ax_hid.get_ylim()[1]],
                            color='grey', alpha=0.2)

    if restrict_HR_low != 0:
        ax_hid.fill_between([ax_hid.get_xlim()[0], restrict_HR_low], [ax_hid.get_ylim()[0], ax_hid.get_ylim()[0]],
                            [ax_hid.get_ylim()[1], ax_hid.get_ylim()[1]],
                            color='grey', alpha=0.2)

    if restrict_HR_high != 0:
        ax_hid.fill_between([restrict_HR_high, ax_hid.get_xlim()[1]], [ax_hid.get_ylim()[0], ax_hid.get_ylim()[0]],
                            [ax_hid.get_ylim()[1], ax_hid.get_ylim()[1]],
                            color='grey', alpha=0.2)

    ''''''''''''''''''
    #### legends
    ''''''''''''''''''

    #for XRISM AO2
    # paper_look=False

    if radio_info_cmap == 'Source' or display_edgesource:

        # looks good considering the size of the graph
        n_col_leg_source = 4 if paper_look else (5 if sum(mask_obj) < 30 else 6)

        # #for AO2
        # n_col_leg_source = 2


        old_legend_size = mpl.rcParams['legend.fontsize']

        mpl.rcParams['legend.fontsize'] = (5.5 if sum(mask_obj) > 30 and radio_info_cmap == 'Source' else 7) + (
            3 if paper_look else 0)

        hid_legend = fig_hid.legend(loc='lower center', ncol=n_col_leg_source, bbox_to_anchor=(0.475, -0.11))

        elem_leg_source, labels_leg_source = plt.gca().get_legend_handles_labels()

        # selecting sources with both detections and non detections
        sources_uniques = np.unique(labels_leg_source, return_counts=True)
        sources_detnondet = sources_uniques[0][sources_uniques[1] != 1]

        # recreating the elem_leg and labels_leg with grouping but only if the colormaps are separated because then it makes sense
        if split_cmap_source:

            leg_source_gr = []
            labels_leg_source_gr = []

            for elem_leg, elem_label in zip(elem_leg_source, labels_leg_source):
                if elem_label in sources_detnondet:

                    # only doing it for the first iteration
                    if elem_label not in labels_leg_source_gr:
                        leg_source_gr += [tuple(np.array(elem_leg_source)[np.array(labels_leg_source) == elem_label])]
                        labels_leg_source_gr += [elem_label]

                else:
                    leg_source_gr += [elem_leg]
                    labels_leg_source_gr += [elem_label]

            # updating the handle list
            elem_leg_source = leg_source_gr
            labels_leg_source = labels_leg_source_gr

        n_obj_leg_source = len(elem_leg_source)

        def n_lines():
            return len(elem_leg_source) // n_col_leg_source + (1 if len(elem_leg_source) % n_col_leg_source != 0 else 0)

        # inserting blank spaces until the detections have a column for themselves
        while n_lines() < n_obj_withdet:
            # elem_leg_source.insert(5,plt.Line2D([],[], alpha=0))
            # labels_leg_source.insert(5,'')

            elem_leg_source += [plt.Line2D([], [], alpha=0)]
            labels_leg_source += ['']

        # removing the first version with a non-aesthetic number of columns
        hid_legend.remove()

        if len(elem_leg_source)>0:
            max_markers_legend=max([1 if type(elem) in [matplotlib.collections.PathCollection,matplotlib.lines.Line2D]
                                    else len(elem) for elem in elem_leg_source])
            # recreating it with updated spacing
            hid_legend = fig_hid.legend(elem_leg_source, labels_leg_source, loc='lower center',
                                        ncol=n_col_leg_source,
                                        bbox_to_anchor=(0.475, -0.02 * n_lines() - (
                                            0.02 * (6 - n_lines()) if paper_look else 0) - (0.1 if paper_look else 0)),
                                        handler_map={tuple: HandlerTuple(ndivide=None,
                                                                         pad=max_markers_legend**1.2-1.4)},
                                        handletextpad=0.1+max_markers_legend-1.3,
                                        columnspacing=(0.5 if paper_look else 1),
                                        frameon=max_markers_legend<=2)

            # #for AO2 plot
            # hid_legend = fig_hid.legend(elem_leg_source, labels_leg_source, loc='center left',
            #                             ncol=n_col_leg_source,
            #                             bbox_to_anchor=(0.125, 0.4),
            #                             handler_map={tuple: HandlerTuple(ndivide=None,
            #                                                              pad=max_markers_legend**1.2-1.4)},
            #                             handletextpad=0.1+max_markers_legend-1.3,
            #                             columnspacing=(0.5 if paper_look else 1),
            #                             frameon=max_markers_legend<=2)

            #mid-i
            # hid_legend = fig_hid.legend(elem_leg_source, labels_leg_source, loc='center left',
            #                             ncol=n_col_leg_source,
            #                             bbox_to_anchor=(0.505, 0.195),
            #                             handler_map={tuple: HandlerTuple(ndivide=None,
            #                                                              pad=max_markers_legend**1.2-1.4)},
            #                             handletextpad=0.1+max_markers_legend-1.3,
            #                             columnspacing=(0.5 if paper_look else 1),
            #                             frameon=max_markers_legend<=2)

        '''
        # maintaining a constant marker size in the legend (but only for markers)
        # note: here we cannot use directly legend_handles because they don't consider the second part of the legend tuples
        # We thus use the findobj method to search in all elements of the legend
        '''
        for elem_legend in hid_legend.findobj():

            #### find a way to change the size of this

            if type(elem_legend) == mpl.collections.PathCollection:
                if len(elem_legend._sizes) != 0:
                    for i in range(len(elem_legend._sizes)):
                        elem_legend._sizes[i] = 50 + (80 if paper_look else 0) + (
                            30 if n_lines() < 6 else 0) if display_upper else 30 + (40 if paper_look else 0) + (
                            10 if n_lines() < 6 else 0)

                    if paper_look and display_upper:
                        elem_legend.set_linewidth(2)

                    # changing the dash type of dashed element for better visualisation:
                    if elem_legend.get_dashes() != [(0.0, None)]:
                        elem_legend.set_dashes((0, (5, 1)))

        # old legend version
        # hid_legend=fig_hid.legend(loc='upper right',ncol=1,bbox_to_anchor=(1.11,0.895) if bigger_text and radio_info_cmap=='Source' \
        #                           and color_nondet else (0.9,0.88))

        mpl.rcParams['legend.fontsize'] = old_legend_size

    #for XRISM AO2
    # paper_look=True


    if display_single:
        hid_det_examples = [
            (Line2D([0], [0], marker=marker_abs, color='white',
                    markersize=50 ** (1 / 2), alpha=1., linestyle='None',
                    markeredgecolor='black', markeredgewidth=2)) \
                if display_upper else
            (Line2D([0], [0], marker=marker_nondet, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2)),
            (Line2D([0], [0], marker=marker_abs, color='grey', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2))]
    else:
        hid_det_examples = [
            ((Line2D([0], [0], marker=marker_ul, color='white', markersize=50 ** (1 / 2), alpha=alpha_ul,
                     linestyle='None',
                     markeredgecolor='black', markeredgewidth=2),
              Line2D([0], [0], marker=marker_ul_top, color='white', markersize=50 ** (1 / 2), alpha=alpha_ul,
                     linestyle='None', markeredgecolor='black', markeredgewidth=2)) \
                 if radio_info_cmap == 'Source' else Line2D([0], [0], marker=marker_ul, color='white',
                                                            markersize=50 ** (1 / 2), alpha=alpha_ul, linestyle='None',
                                                            markeredgecolor='black', markeredgewidth=2)) \
                if display_upper else
            (Line2D([0], [0], marker=marker_nondet, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2)),
            (Line2D([0], [0], marker=marker_abs, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='black', markeredgewidth=2))]

    if display_nonsign:
        hid_det_examples += [
            (Line2D([0], [0], marker=marker_abs, color='white', markersize=50 ** (1 / 2), linestyle='None',
                    markeredgecolor='grey', markeredgewidth=2))]

    if hatch_unstable:
        hid_det_examples += [
            (mpl.patches.Patch(facecolor='white', hatch='///'))]

    mpl.rcParams['legend.fontsize'] = 7 + (2 if paper_look and not zoom else 0)

    # marker legend

    # manual custom subplot adjust to get the same scale for the 3 sources with ULs and for the zoomed 5 sources with detection
    # to be put in the 5 sources
    custom = False

    fig_hid.legend(handles=hid_det_examples, loc='center left',
                   labels=['upper limit' if display_upper else 'non detection ',
                           'absorption line detection\n above ' + (r'3$\sigma$' if slider_sign == 0.997 else str(
                               slider_sign * 100) + '%') + ' significance'] +
                          (['absorption line detection below ' + str(slider_sign * 100) + ' significance.'] \
                               if display_nonsign else []) +
                          (['Unstable SED'] if hatch_unstable else []),
                   title='',
                   bbox_to_anchor=(0.69 if change_legend_position else 0.125, 0.829 - (
                       0.012 if paper_look and not zoom else 0) - (
                                       0.018 if hatch_unstable else 0)) if bigger_text and square_mode else (
                       0.125, 0.82), handler_map={tuple: mpl.legend_handler.HandlerTuple(None)},
                   handlelength=2, handleheight=2., columnspacing=1.)

    if radio_info_cmap == 'custom_acc_states':
        # making a legend for the accretion state colors
        custom_acc_states_example = [(mpl.patches.Patch(color='green')),
                                     (mpl.patches.Patch(color='orange')),
                                     (mpl.patches.Patch(color='red')),
                                     (mpl.patches.Patch(color='purple')),
                                     (mpl.patches.Patch(color='blue')),
                                     (mpl.patches.Patch(color='grey'))]

        fig_hid.legend(handles=custom_acc_states_example, loc='center left',
                       labels=['Soft', 'Intermediate', 'SPL', 'QRM', 'Hard', 'Unknown'],
                       title='Spectral state',
                       bbox_to_anchor=(0.125 if change_legend_position else 0.756, 0.754 - (
                           0.012 if paper_look and not zoom else 0) - (
                                           0.018 if hatch_unstable else 0)) if bigger_text and square_mode else (
                           0.125, 0.82), handler_map={tuple: mpl.legend_handler.HandlerTuple(None)},
                       handlelength=2, handleheight=2., columnspacing=1.)

    # note: upper left anchor (0.125,0.815)
    # note : upper right anchor (0.690,0.815)
    # note: 0.420 0.815

    # size legend

    if display_upper and not display_single:
        # displaying the
        if radio_info_cmap == 'Source':
            hid_size_examples = [(Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500),
                                  Line2D([0], [0], marker=marker_ul_top, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500),
                                  Line2D([0], [0], marker=marker_ul_top, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500),
                                  Line2D([0], [0], marker=marker_ul_top, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500))]
        else:
            hid_size_examples = [(Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500)),
                                 (Line2D([0], [0], marker=marker_abs, color='black',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None'),
                                  Line2D([0], [0], marker=marker_ul, color='None', markeredgecolor='grey',
                                         markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None',
                                         zorder=500))]
    else:
        hid_size_examples = [(Line2D([0], [0], marker=marker_abs, color='black',
                                     markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None')),
                             (Line2D([0], [0], marker=marker_abs, color='black',
                                     markersize=(norm_s_lin * 20 ** norm_s_pow) ** (1 / 2), linestyle='None')),
                             (Line2D([0], [0], marker=marker_abs, color='black',
                                     markersize=(norm_s_lin * 50 ** norm_s_pow) ** (1 / 2), linestyle='None'))]

    ew_legend = fig_hid.legend(handles=hid_size_examples, loc='center left', labels=['5 eV', '20 eV', '50 eV'],
                               title='Equivalent widths',
                               bbox_to_anchor=(0.125, 0.218 + (
                                   0.028 if paper_look and not zoom else 0)) if bigger_text and square_mode else (
                                   0.125, 0.218), handleheight=4, handlelength=4, facecolor='None')

    if radio_info_cmap == 'Instrument':
        instru_examples = np.array([Line2D([0], [0], marker=marker_abs, color=list(telescope_colors.values())[i],
                                           markeredgecolor='black',
                                           markersize=(norm_s_lin * 5 ** norm_s_pow) ** (1 / 2), linestyle='None') \
                                    for i in range(len(telescope_list))])

        telescope_choice_sort = np.array(choice_telescope)
        telescope_choice_sort.sort()

        instru_ind = [np.argwhere(np.array(telescope_list) == elem)[0][0] for elem in telescope_choice_sort]

        instru_legend = fig_hid.legend(handles=instru_examples[instru_ind].tolist(), loc='upper right',
                                       labels=telescope_choice_sort.tolist(),
                                       title=radio_info_cmap,
                                       bbox_to_anchor=(0.900, 0.21),
                                       # bbox_to_anchor=(0.900, 0.88) if bigger_text and square_mode else (0.825, 0.918),
                                       handleheight=1, handlelength=4, facecolor='None')

    # manual custom subplot adjust to get the same scale for the 3 visible sources plot and for the zoomed 5 sources with detection
    # elem=fig_hid.add_axes([0.5, 0.792, 0.1, 0.1])
    # mpl.rcParams.update({'font.size': 2})
    # elem.axis('off')

    # manual custom subplot adjust to get the same scale for the 3 sources with ULs and for the zoomed 5 sources with detection
    # to be put in the 5 sources

    if custom:
        plt.subplots_adjust(top=0.863)

    # note: 0.9 0.53
    # destacked version
    # fig_hid.legend(handles=hid_size_examples,loc='center left',labels=['5 eV','20 eV','50 eV'],title='Equivalent widths',
    #             bbox_to_anchor=(0.125,0.235) if bigger_text and square_mode else (0.125,0.235),handler_map = {tuple:mpl.legend_handler.HandlerTuple(None)},handlelength=8,handleheight=5,columnspacing=5.)


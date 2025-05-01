#general imports
import sys

import numpy as np

#matplotlib imports

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator,AutoMinorLocator

#pickle for the rxte lightcurve dictionnary

#needed to fetch the links to the lightcurves for MAXI data

#correlation values and trend plots with MC distribution from the uncertainties

#Note : as of the writing of this code, the standard pymccorrelation doesn't accept differing +/- uncertainties, so I tweaked their
#'perturb values' function

# import time

'''Astro'''

#local
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/general/')
sys.path.append('/home/parrama/Documents/Work/PhD/Scripts/Python/observations/spectral_analysis/')

#online
sys.path.append('/mount/src/winds/observations/spectral_analysis/')
sys.path.append('/mount/src/winds/general/')

#custom script with some lines and fit utilities and variables
from fitting_tools import lines_std, lines_std_names

from general_tools import ravel_ragged

from visual_line_tools import ratio_choices,ratio_choices_str,info_str,axis_str,telescope_colors
#Catalogs and manipulation

def distrib_graph(data_perinfo, info, dict_linevis, data_ener=None, conf_thresh=0.99, indiv=False, save=False,
                  close=False, streamlit=False, bigger_text=False, split=None):
    '''
    repartition diagram from all the observations in the current pool (i.e. all objects/obs/lines).

    Use the 'info' keyword to graph flux,ewidth, bshift or ener

    Use the 'indiv' keyword to plot for all lines simultaneously or 6 plots for the 6 single lines

    Non detections are filtered via 0 values in significance

    Detections above and below the given threshold are highlighted

    we ravel the last 2 dimensions with a custom function since for multi object plots the sequences can be ragged and the custom .ravel function
    doesn't work
    '''
    n_infos = dict_linevis['n_infos']
    range_absline = dict_linevis['range_absline']
    mask_lines = dict_linevis['mask_lines']
    list_id_lines = np.array(range_absline)[mask_lines]
    bins_bshift = dict_linevis['bins_bshift']
    bins_ener = dict_linevis['bins_ener']
    mask_obj = dict_linevis['mask_obj']

    if streamlit:
        display_nonsign = dict_linevis['display_nonsign']
        scale_log_ew = dict_linevis['scale_log_ew']
        obj_disp_list = dict_linevis['obj_list'][mask_obj]
        instru_list = dict_linevis['instru_list'][mask_obj]
        date_list = dict_linevis['date_list'][mask_obj]
        width_plot_restrict = dict_linevis['width_plot_restrict']
        glob_col_source = dict_linevis['glob_col_source']
        cmap_color_det = dict_linevis['cmap_color_det']
        split_dist_method = dict_linevis['split_dist_method']
    else:
        display_nonsign = True
        scale_log_ew = False
        glob_col_source = False
        cmap_color_det = mpl.cm.plasma
        split_dist_method = False

    save_dir = dict_linevis['save_dir']
    save_str_prefix = dict_linevis['save_str_prefix']
    args_cam = dict_linevis['args_cam']
    line_search_e_str = dict_linevis['line_search_e_str']
    args_line_search_norm = dict_linevis['args_line_search_norm']

    # range of the existing lines for loops
    range_line = range(len(list_id_lines))

    line_mode = info == 'lines'
    if not line_mode:
        ind_info = np.argwhere([elem in info for elem in ['ew', 'bshift', 'ener', 'lineflux', 'time', 'width']])[0][0]
    else:
        # we don't care here
        ind_info = 0

    split_off = split == 'Off'
    split_source = split == 'Source'
    split_instru = split == 'Instrument'

    # main data array
    data_plot = data_perinfo[ind_info] if ind_info not in [2, 4, 5] else data_ener if ind_info == 2 \
        else date_list if ind_info == 4 else width_plot_restrict

    ratio_mode = 'ratio' in info

    if indiv:
        graph_range = range_absline
    else:
        # using a list index for the global graphs allows us to keep the same structure
        # however we need to restrict it to the currently plotted lines in streamlit mode
        if streamlit:
            graph_range = [range(len([elem for elem in mask_lines if elem]))]
        else:
            graph_range = [range_absline]

    # computing the range of the ew bins from the global graph to get the same scale for all individual graphs)

    if scale_log_ew:
        bins_ew = np.geomspace(1, min(100, (max(ravel_ragged(data_perinfo[0][0])) // 5 + 1) * 5), 20)
    else:
        bins_ew = np.arange(5, min(100, (max(ravel_ragged(data_perinfo[0][0])) // 5 + 1) * 5), 2.5)
        # bins_ew=np.linspace(5,min(100,(max(ravel_ragged(data_perinfo[0][0]))//5+1)*5),20)

    bins_ewratio = np.linspace(0.2, 4, 20)

    if n_infos >= 5:
        if len(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0]) > 0]) > 0 and len(
                ravel_ragged(data_perinfo[3][0]).nonzero()[0]) != 0:

            bins_flux = np.geomspace(max(1e-16, min(1e-13, (
                        min(ravel_ragged(data_perinfo[3][0])[ravel_ragged(data_perinfo[4][0]) > 0]) * 0.9))),
                                     max(1e-10, (max(ravel_ragged(data_perinfo[3][0])[
                                                         ravel_ragged(data_perinfo[4][0]) > 0]) * 1.1)), 20)
        else:
            bins_flux = np.geomspace(1e-13, 1e-10, 20)
    else:
        bins_flux = None

    # for now
    bins_time = None

    # linear bin widths between 0 (where most widths value lie) and 0.05 which is the hard limit
    bins_width = np.linspace(0, 5500, 12)

    # sorting the bins in an array depending on the info asked
    bins_info = [bins_ew, bins_bshift, bins_ener, bins_flux, bins_time, bins_width]

    bins_ratio = [bins_ewratio]

    # and fetching the current one (or custom bins for other modes)
    hist_bins = bins_info[ind_info] if not ratio_mode else bins_ratio[ind_info] if not line_mode else range(graph_range)

    # fetching the global boolean array for the graph size definition
    bool_det_glob = ravel_ragged(data_perinfo[4][0]).astype(float)

    # the "and" keyword doesn't work for arrays so we use & instead (which requires parenthesis)
    bool_det_glob = (bool_det_glob != 0.) & (~np.isnan(bool_det_glob))

    # computing the maximal height of the graphs (in order to keep the same y axis scaling on all the individual lines graphs)
    max_height = max((np.histogram(ravel_ragged(data_plot[0])[bool_det_glob], bins=bins_info[ind_info]))[0])

    for i in graph_range:

        if ratio_mode:
            # we fetch the index list corresponding to the info string at the end of the info provided
            ratio_indexes_x = ratio_choices[info[-2:]]

        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(6, 4) if not split_off else (6, 6) if bigger_text else (10, 8))

        if not bigger_text:
            if indiv:
                fig_hist.suptitle('Repartition of the ' + info_str[ind_info] + ' of the ' + lines_std[
                    lines_std_names[3 + i]] + ' absorption line')
            else:
                if ratio_mode:
                    fig_hist.suptitle(
                        'Repartition of the ' + ratio_choices_str[info[-2:]] + ' ratio in currently selected sources')
                else:
                    fig_hist.suptitle('Repartition of the ' + info_str[ind_info] +
                                      (
                                          ' in currently selected lines and sources' if streamlit else ' absorption lines'))

        # using a log x axis if the
        ax_hist.set_ylabel(r'Detections')
        ax_hist.set_xlabel('' if line_mode else (ratio_choices_str[info[-2:]] + ' ' if ratio_mode else '') +
                                                axis_str[ind_info].replace(' (eV)', ('' if ratio_mode else ' (eV)')) +
                                                (' ratio' if ratio_mode else ''))

        if split_off:
            ax_hist.set_ylim([0, max_height + 0.25])

        if ind_info in [0, 3]:
            if ind_info == 3 or scale_log_ew:
                ax_hist.set_xscale('log')

        # these boolean arrays distinguish non detections (i.e. 0/nan significance) and statistically significant detections from the others
        bool_sign = ravel_ragged(data_perinfo[4][0][i]).astype(float)

        # single variable for more advanced bar plots with splitting
        bool_signdet = (bool_sign >= conf_thresh) & (~np.isnan(bool_sign))

        # split variables for old computations with significance or not
        bool_det = (bool_sign != 0.) & (~np.isnan(bool_sign))
        bool_sign = bool_sign[bool_det] >= conf_thresh

        if info == 'width':
            # creating a mask for widths that are not compatible with 0 at 3 sigma\
            # (different from the scatter here as we don't want the 0 values)
            bool_sign_width = ravel_ragged(data_perinfo[7][0][i]).astype(float) != 0

            bool_sign = bool_sign & bool_sign_width[bool_det]

        if not split_off:

            if split_source:
                # source by source version
                sign_det_split = np.array([ravel_ragged(data_perinfo[4][0][i].T[j]) for j in range(len(obj_disp_list))],
                                          dtype=object)

                # unused for now
                # #creating the bool det and bool sign masks
                # bool_det_split=np.array([(elem!=0.) & (~np.isnan(elem)) for elem in sign_det_split])

                # note: here we don't link bool sign to bool det because we won't bother plotting non significant lines
                bool_sign_split = np.array([(elem >= conf_thresh) & (~np.isnan(elem)) for elem in sign_det_split],
                                           dtype=object)

                if info == 'width':
                    # creating a mask for widths that are not compatible with 0 at 3 sigma\
                    # (different from the scatter here as we don't want the 0 values)
                    bool_sign_width_split = np.array([ravel_ragged(data_perinfo[7][0][i].T[j]).astype(float) != 0 \
                                                      for j in range(len(obj_disp_list))], dtype=object)

                    bool_sign_split = np.array(
                        [elem & elem2 for elem, elem2 in zip(bool_sign_split, bool_sign_width_split)], dtype=object)

            if split_instru:
                instru_list_repeat = np.array([instru_list for repeater in (i if type(i) == range else [i])])

                instru_unique = np.unique(ravel_ragged(instru_list))

                # here we re split bool_sign with values depending on the instrument to restrict it
                bool_sign_split = [(bool_signdet) & (ravel_ragged(instru_list_repeat) == instru_unique[i_instru]) \
                                   for i_instru in range(len(instru_unique))]

        #### creating the hist variables
        if line_mode:

            # computing the number of element for each line
            line_indiv_size = len(ravel_ragged(data_perinfo[4][0][0]))

            bool_sign_line = []
            for id_line, line in enumerate(list_id_lines):
                bool_sign_line += [
                    (bool_signdet) & (np.array(ravel_ragged(np.repeat(False, line_indiv_size * id_line)).tolist() + \
                                               ravel_ragged(np.repeat(True, line_indiv_size)).tolist() + \
                                               ravel_ragged(np.repeat(False, line_indiv_size * (
                                                           len(list_id_lines) - (id_line + 1)))).tolist()))]

            # here we create a 2D array with the number of lines detected per source and per line
            hist_data_splitsource = np.array([[sum((data_perinfo[4][0][i_line][i_obj] >= conf_thresh) & (
                ~np.isnan(data_perinfo[4][0][i_line][i_obj].astype(float)))) \
                                               for i_line in range_line] for i_obj in range(len(obj_disp_list))])

            # suming on the sources give the global per line number of detection
            hist_data = hist_data_splitsource.sum(axis=0)

            if split_source:
                hist_data_split = hist_data_splitsource

            if split_instru:
                hist_data_split = [[np.sum((bool_sign_line[i_line]) & (bool_sign_split[i_instru])) \
                                    for i_line in range_line] for i_instru in range(len(instru_unique))]

        # this time data_plot is an array
        elif ratio_mode:

            # we need to create restricted bool sign and bool det here, with only the index of each line
            sign_ratio_arr = np.array(
                [ravel_ragged(data_perinfo[4][0][ratio_indexes_x[i]]).astype(float) for i in [0, 1]])

            # here we create different mask variables to keep the standard bool_det/sign for the other axis if needed
            bool_det_ratio = (sign_ratio_arr[0] != 0.) & (~np.isnan(sign_ratio_arr[0])) & (sign_ratio_arr[1] != 0.) & (
                ~np.isnan(sign_ratio_arr[1]))

            # this doesn't work with bitwise comparison
            bool_sign_ratio = (sign_ratio_arr[0] >= conf_thresh) & (~np.isnan(sign_ratio_arr[0])) & (
                        sign_ratio_arr[1] >= conf_thresh) & (~np.isnan(sign_ratio_arr[1]))

            # making it clearer
            bool_sign_ratio = bool_sign_ratio[bool_det_ratio]

            # before using them to create the data ratio

            hist_data = np.array(
                [ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_ratio] / \
                 ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_ratio],
                 ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][~bool_sign_ratio] / \
                 ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][~bool_sign_ratio]], dtype=object)

            if not split_off:

                if split_source:
                    sign_det_ratio_split = [
                        [ravel_ragged(data_perinfo[4][0][ratio_indexes_x[i]].T[j]).astype(float) for j in
                         range(len(obj_disp_list))] for i in range(2)]

                    # creating the bool det and bool sign masks
                    # bool_det_split=np.array([(elem_num!=0.) & (~np.isnan(elem_num)) & (elem_denom!=0.) & (~np.isnan(elem_denom))\
                    #                              for elem_num,elem_denom in zip([elem for elem in sign_det_ratio_split])])

                    bool_sign_split = np.array([(elem_num >= conf_thresh) & (~np.isnan(elem_num)) & (
                                elem_denom >= conf_thresh) & (~np.isnan(elem_denom)) \
                                                for elem_num, elem_denom in
                                                zip(sign_det_ratio_split[0], sign_det_ratio_split[1])], dtype=object)

                    # computing the data array for the ratio (here we don't need to transpose because we select a single line with ratio_indexes_x
                    hist_data_split = np.array(
                        [ravel_ragged(data_plot[0][ratio_indexes_x[0]][i_obj])[bool_sign_split[i_obj]] / \
                         ravel_ragged(data_plot[0][ratio_indexes_x[1]][i_obj])[bool_sign_split[i_obj]] \
                         for i_obj in range(len(obj_disp_list))], dtype=object)

                elif split_instru:
                    bool_sign_split = [
                        (bool_sign_ratio) & ((ravel_ragged(instru_list) == instru_unique[i_instru])[bool_det_ratio]) \
                        for i_instru in range(len(instru_unique))]

                    hist_data_split = np.array(
                        [ravel_ragged(data_plot[0][ratio_indexes_x[0]])[bool_det_ratio][bool_sign_split[i_instru]] / \
                         ravel_ragged(data_plot[0][ratio_indexes_x[1]])[bool_det_ratio][bool_sign_split[i_instru]] \
                         for i_instru in range(len(instru_unique))], dtype=object)


        else:
            hist_data = [ravel_ragged(data_plot[0][i])[bool_det][bool_sign],
                         ravel_ragged(data_plot[0][i])[bool_det][~bool_sign]]

            if not split_off:

                #### should be changed to use the same method as split_instru
                if split_source:
                    # here we need to add a transposition
                    hist_data_split = np.array(
                        [ravel_ragged(data_plot[0][i].T[i_obj])[bool_sign_split[i_obj]] for i_obj in
                         range(len(obj_disp_list))], dtype=object)
                elif split_instru:
                    # no need for transposition here
                    hist_data_split = np.array([ravel_ragged(data_plot[0][i])[bool_sign_split[i_line]] for i_line in
                                                range(len(instru_unique))], dtype=object)

                # adjusting the length to avoid a size 1 array which would cause issue
                if len(hist_data_split) == 1:
                    hist_data_split = hist_data_split[0]

        hist_cols = ['blue', 'grey']
        hist_labels = ['detections above ' + str(conf_thresh * 100) + '% treshold',
                       'detections below ' + str(conf_thresh * 100) + '% treshold']

        #### histogram plotting and colors

        if display_nonsign:
            ax_hist.hist(hist_data, stacked=True, color=hist_cols, label=hist_labels, bins=hist_bins)
        else:
            if not split_off:

                # indexes of obj_list with significant detections
                id_obj_withdet = np.argwhere(np.array([sum(elem) > 0 for elem in bool_sign_split]) != 0).T[0]

                if glob_col_source:
                    n_obj_withdet = sum(mask_obj)
                else:
                    # number of sources with detections
                    n_obj_withdet = len(id_obj_withdet)

                # color mapping accordingly for source split
                norm_colors_obj = mpl.colors.Normalize(vmin=0, vmax=n_obj_withdet - 1)
                colors_func_obj = mpl.cm.ScalarMappable(norm=norm_colors_obj, cmap=cmap_color_det)

                if line_mode:

                    if split_source:

                        # creating the range of bars (we don't use histograms here because our counting is made arbitrarily on a given axis)
                        # recreate_5cmap
                        # for i_obj_det in range(n_obj_withdet-1):

                        for i_obj_det in range(n_obj_withdet):
                            i_obj = i_obj_det

                            # only for custom modifications to recreate the 5cmap
                            # i_obj=id_obj_withdet[i_obj_det]

                            # using a range in the list_id_lines allow to resize the graph when taking off lines
                            ax_hist.bar(
                                np.array(range(len(list_id_lines))) - 1 / 4 + i_obj_det / (4 / 3 * n_obj_withdet),
                                hist_data_split[i_obj],
                                width=1 / (4 / 3 * n_obj_withdet),
                                color=colors_func_obj.to_rgba(i_obj_det),
                                label=obj_disp_list[i_obj] if i_obj in id_obj_withdet else '')

                        # changing the tick and tick names to the actual line names
                        ax_hist.set_xticks(np.array(range(len(list_id_lines))))
                        ax_hist.set_xticklabels([lines_std[lines_std_names[i_line + 3]] for i_line in list_id_lines],
                                                rotation=60)

                    elif split_instru:
                        # creating the range of bars (we don't use histograms here because our counting is made arbitrarily on a given axis
                        # using a range in the list_id_lines allow to resize the graph when taking off lines
                        [ax_hist.bar(
                            np.array(range(len(list_id_lines))) - 1 / 4 + i_instru / (4 / 3 * len(instru_unique)),
                            hist_data_split[i_instru],
                            width=1 / (4 / 3 * len(instru_unique)),
                            color=telescope_colors[instru_unique[i_instru]], label=instru_unique[i_instru]) for i_instru
                         in range(len(instru_unique))]

                        # changing the tick and tick names to the actual line names
                        ax_hist.set_xticks(np.array(range(len(list_id_lines))))
                        ax_hist.set_xticklabels([lines_std[lines_std_names[i_line + 3]] for i_line in list_id_lines],
                                                rotation=60)

                else:

                    if split_source:

                        # skipping displaying bars for objects with no detection
                        source_withdet_mask = [sum(elem) > 0 for elem in bool_sign_split]

                        # resticting histogram plotting to sources with detection to havoid plotting many useless empty source hists
                        ax_hist.hist(hist_data_split[source_withdet_mask],
                                     color=np.array(
                                         [colors_func_obj.to_rgba(i_obj_det) for i_obj_det in range(n_obj_withdet)])[
                                         (source_withdet_mask if glob_col_source else np.repeat(True, n_obj_withdet))],
                                     label=obj_disp_list[source_withdet_mask], bins=hist_bins, rwidth=0.8, align='left',
                                     stacked=True)

                    elif split_instru:

                        try:
                            ax_hist.hist(hist_data_split,
                                         color=[telescope_colors[instru_unique[i_instru]] for i_instru in
                                                range(len(instru_unique))],
                                         label=instru_unique, bins=hist_bins, rwidth=0.8, align='left',
                                         stacked=not split_dist_method)
                        except:
                            breakpoint()

                        # if we want combined (not stacked) + transparent
                        # t=[ax_hist.hist(hist_data_split[i_instru],color=telescope_colors[instru_unique[i_instru]],
                        #              label=instru_unique[i_instru],bins=hist_bins,rwidth=0.8,align='left',alpha=0.5) for i_instru in range(len(instru_unique))]

                    # adding minor x ticks only (we don't want minor y ticks because they don't make sense for distributions)
                    ax_hist.xaxis.set_minor_locator(AutoMinorLocator())

            else:
                ax_hist.hist(hist_data[0], color=hist_cols[0], label=hist_labels[0], bins=hist_bins)

        # forcing only integer ticks on the y axis
        ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))

        if not bigger_text or not split_off:
            plt.legend(fontsize=10)

        plt.tight_layout()
        if save:
            if indiv:
                suffix_str = '_' + lines_std_names[3 + i]
            else:
                suffix_str = '_all'

            plt.savefig(
                save_dir + '/graphs/distrib/' + save_str_prefix + 'autofit_distrib_' + info + suffix_str + '_cam_' + args_cam + '_' + \
                line_search_e_str.replace(' ', '_') + '_' + args_line_search_norm.replace(' ', '_') + '.png')
        if close:
            plt.close(fig_hist)

    # returning the graph for streamlit display
    if streamlit:
        return fig_hist

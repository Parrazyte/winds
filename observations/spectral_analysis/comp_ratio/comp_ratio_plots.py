"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-24: plot_comp_ratio moved here from
xspec_config_multisp.py (Maxime's code) and revised.

Ratio plots of XSPEC additive components to a continuum made of other additive components.

Plot and Xset come from xspec_config_multisp, which sets them to None when PyXspec is not available
(streamlit mode), so this module can also be imported there.

old_plot_comp_ratio is the legacy version (unchanged body), kept for comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#plot_transi is imported under another name: plot_comp_ratio has a boolean argument called plot_transi
from xspec_config_multisp import Plot, Xset, set_ener, plot_std_ener, plot_transi as mark_transitions


def _per_comp(arg, default, n_comps, name, broadcast=True):
    '''
    Per-component list for a style argument of plot_comp_ratio:
        None                    -> n_comps copies of default
        a str or number         -> n_comps copies of it (if broadcast)
        an iterable             -> itself. If its length differs from n_comps a message is printed: the plotting
                                   loop zips all the lists, so it stops at the shortest one
    '''

    if arg is None:
        return [default]*n_comps

    if broadcast and isinstance(arg, (str, int, float)):
        return [arg]*n_comps

    arg = list(arg)
    if len(arg) != n_comps:
        print('plot_comp_ratio: %s has %d elements for %d other_addcomps. Components after the shortest list '
              'will not be plotted' % (name, len(arg), n_comps))
    return arg


def _ener_slice(x_vals, ener_low, ener_high):
    '''
    Slice of the bins between ener_low and ener_high, plus one bin on each side so that the curves reach the
    axis limits. Works for increasing or decreasing x (energy or wavelength axis)
    '''

    inside = np.flatnonzero((x_vals >= min(ener_low, ener_high)) & (x_vals <= max(ener_low, ener_high)))
    if len(inside) == 0:
        raise ValueError('plot_comp_ratio: no model bin between %s and %s (model grid from %s to %s)'
                         % (ener_low, ener_high, x_vals.min(), x_vals.max()))

    return slice(max(inside[0]-1, 0), min(inside[-1]+2, len(x_vals)))


def _sum_addcomps(comp_ids, x_sel, desc):
    '''
    Sum of the additive components comp_ids (xspec numbering, from 1) of the last Plot, restricted to x_sel
    With a single additive component xspec has no addcomp array (nAddComps is 0): component 1 is then the model
    '''

    if Plot.nAddComps() == 0 and list(comp_ids) == [1]:
        return np.array(Plot.model(1))[x_sel]

    try:
        return np.array([Plot.addComp(addCompNum=i_comp, plotGroup=1) for i_comp in comp_ids]).sum(0)[x_sel]
    except Exception as e:
        raise ValueError('plot_comp_ratio: could not get the additive components %s of %s (the model has %s)'
                         % (list(comp_ids), desc, Plot.nAddComps())) from e


def _plot_ratios(x_plot, other_addcomps_ratio, x_axis_unit, ener_low, ener_high, ylims,
                 other_addcomps_labels, other_addcomps_colors, figsize, other_addcomps_type,
                 other_addcomps_alpha, other_addcomps_ls, plot_transi, minor_locator, ylabel_prefix,
                 manual_bbox, ylog, ax, plot_transi_kwargs, bands=None, band_alpha=None, other_addcomps_lw=None,
                 transi_lw=None, legend_loc=None, transi_alternate=True):
    '''
    Plotting part of plot_comp_ratio (and of comp_ratio_plot_MC.plot_comp_ratio_MC). The style arguments are
    those of plot_comp_ratio.

    other_addcomps_ratio: one ratio array (on x_plot) or None (legend-only entry) per component

    bands: None, or one element per component: None (no band) or a list of (low, high) ratio arrays, one per
        confidence level, drawn in the color of the component's line with the matching band_alpha
    band_alpha: list of alpha values, one per element of each bands list
    other_addcomps_lw, transi_lw, legend_loc, transi_alternate: as for plot_comp_ratio

    Returns (fig, ax)
    '''

    n_comps = len(other_addcomps_ratio)
    if bands is None:
        bands = [None]*n_comps

    other_addcomps_labels_use = _per_comp(other_addcomps_labels, '', n_comps, 'other_addcomps_labels',
                                          broadcast=False)
    other_addcomps_colors_use = _per_comp(other_addcomps_colors, None, n_comps, 'other_addcomps_colors')
    other_addcomps_alpha_use = _per_comp(other_addcomps_alpha, None, n_comps, 'other_addcomps_alpha')
    other_addcomps_ls_use = _per_comp(other_addcomps_ls, '-', n_comps, 'other_addcomps_ls')
    other_addcomps_type_use = _per_comp(other_addcomps_type, 'em', n_comps, 'other_addcomps_type')
    other_addcomps_lw_use = _per_comp(other_addcomps_lw, None, n_comps, 'other_addcomps_lw')

    wrong_types = set(other_addcomps_type_use) - {'em', 'abs'}
    if wrong_types:
        raise ValueError('plot_comp_ratio: other_addcomps_type elements must be "em" or "abs", not %s'
                         % wrong_types)

    if ax is None:
        fig, ax_comp = plt.subplots(figsize=figsize)
    else:
        ax_comp = ax
        fig = ax.figure

    ax_comp.set_xlim(ener_low, ener_high)
    ax_comp.set_xlabel('Energy (' + x_axis_unit + ')')
    ax_comp.set_ylabel(ylabel_prefix + (' ' if ylabel_prefix != '' else '') + 'ratio to continuum')

    for elem_ratio, elem_color, elem_label, elem_type, elem_alpha, elem_ls, elem_lw, elem_bands in \
            zip(other_addcomps_ratio, other_addcomps_colors_use, other_addcomps_labels_use, other_addcomps_type_use,
                other_addcomps_alpha_use, other_addcomps_ls_use, other_addcomps_lw_use, bands):

        #None components are legend-only entries (no line)
        line = ax_comp.plot([] if elem_ratio is None else x_plot,
                            [] if elem_ratio is None else ((1 if elem_type == 'em' else 0)+elem_ratio),
                            color=elem_color, label=elem_label, lw=0 if elem_ratio is None else elem_lw,
                            alpha=elem_alpha, ls=elem_ls)[0]

        if elem_ratio is not None and elem_bands is not None:
            for (band_low, band_high), elem_band_alpha in zip(elem_bands, band_alpha):
                ax_comp.fill_between(x_plot, (1 if elem_type == 'em' else 0)+band_low,
                                     (1 if elem_type == 'em' else 0)+band_high,
                                     color=line.get_color(), alpha=elem_band_alpha, lw=0,
                                     zorder=line.get_zorder()-0.5)

    if ylog:
        ax_comp.set_yscale('log')

    if ylims is not None:
        ax_comp.set_ylim(ylims)

    if plot_transi:
        transi_kwargs = dict(color='grey', transi_alternate=transi_alternate)
        if transi_lw is not None:
            transi_kwargs.update(line_lw=transi_lw)
        if plot_transi_kwargs is not None:
            transi_kwargs.update(plot_transi_kwargs)
        mark_transitions(ax_comp, **transi_kwargs)

    if minor_locator is not False:
        ax_comp.xaxis.set_minor_locator(AutoMinorLocator(minor_locator))

    if ax is None:
        fig.tight_layout()

    if manual_bbox is not False:
        ax_comp.legend(loc='upper left' if legend_loc is None else legend_loc, bbox_to_anchor=manual_bbox)
    elif legend_loc is not None:
        ax_comp.legend(loc=legend_loc)
    else:
        ax_comp.legend()

    return fig, ax_comp


def old_plot_comp_ratio(cont_addcomps,other_addcomps,ener_low,ener_high,
                   plot_type='eemo',
                    ylims=None,
                   cont_addcomps_xcm='',other_addcomps_xcm=None,
                   other_addcomps_labels=None,other_addcomps_colors=None,figsize=(10,5),
                    other_addcomps_type=None,other_addcomps_alpha=None,other_addcomps_ls=None,
                    plot_transi=True,minor_locator=10,ylabel_prefix='',manual_bbox=False):

    '''
    LEGACY VERSION of plot_comp_ratio, kept unchanged (body identical to xspec_config_multisp.py before
    2026-10-24) to call and compare the old behaviour. Use plot_comp_ratio for new plots.
    Differences: always calls set_ener('thcomp', xrism=True) when an xcm is given, returns None, invisible
    lines without colors, breakpoint() on a wrong addcomp id, last model's continuum used for every ratio when
    several other_addcomps_xcm are given without cont_addcomps_xcm.

    Makes a ratio plot of individual components compared to the sum of others related to a specific weight

    plot_type (eemo or other xspec type where addcomp is expressed)
        will be used to get the addcomp values

    other_addcomps_xcm : path or array of paths or None
        will load a model or take the current one. If several models are put in an iterable, then will attempt
        to load one different model for each set of "other_addcomps" and assume a separate continuum model file

    cont_addcomps_xcm: path or ''
        if not '', will load a model to use for the continuum components only
        Note: in this case the cont_addcomps numbering must be adapted to the components of that xcm instead

    comp_addcomps: iterable of ints
        the list of addcomps ids (in xspec, so with indexes starting at 1)
        to be summed to create the continuum, to which
        all the "other_addcomps" will individually be normalized

    other_addcomps: iterable of either ints or iterables of ints
        the list of the addcomps ids (in xspec, so with indexes starting at 1)
        which will appear individually as the ratio on top of comp_addcomps
        note: can be combined by passing iterables within the list
        In this case, will sum all the addcomps in that iterable for a single displayed component
        if some elements are None, will not plot anything but keeps the label spot for additional lines in the legend


    ener_low/ener_high: x axis limits of the plot

    other_addcomps_names/colors/alpha/ls: None or iterable of same len as other_addcomps
        labels and colors for the additional components

    other_addcomps_type: None or iterable of same len as other_addcomps
        'em' for emission components, 'abs' for absorption components
        if None, all components are assumed in emission (1 is added to their ratio)
        Note that when combining absorption and emission it can be useful to use the "abs" type for emission components
        to see the overlap


    figsize:
        python plot figure size

    plot_transi: plot individual transitions with plt_std_ener (in grey)

    minor_locator: if not False, the sampling of the x axis minor locator

    ylabel_prefix: optional string to add before y label

    '''


    if cont_addcomps_xcm!='':
        #now doing the same thing with a second model for the continuum if requested

        Xset.restore(cont_addcomps_xcm)

        # generally better
        set_ener('thcomp', xrism=True)

        Plot(plot_type)
        cont_addcomps_sum = np.array([Plot.addComp(addCompNum=i_comp, plotGroup=1) for i_comp in cont_addcomps]).sum(0)


    if other_addcomps_xcm is not None:
        single_man_xcm=type(other_addcomps_xcm)==str
        if single_man_xcm:
            Xset.restore(other_addcomps_xcm)

            # generally better
        set_ener('thcomp', xrism=True)
    else:
        single_man_xcm=True

    other_addcomps_vals=[]

    #individual loop for each set of comps to load a model if needed
    for i_id,indiv_cont_id in enumerate(other_addcomps):

        if not single_man_xcm:
            Xset.restore(other_addcomps_xcm[i_id])

            #generally better
            set_ener('thcomp',xrism=True)

        Plot(plot_type)

        if cont_addcomps_xcm=='':
            cont_addcomps_sum = np.array([Plot.addComp(addCompNum=i_comp, plotGroup=1) for i_comp in cont_addcomps]).sum(0)
        else:
            assert cont_addcomps_xcm!='','several xcms without a separate continuum not implemented'

        if indiv_cont_id is None:
            other_addcomps_vals+=[None]
            continue

        if type(indiv_cont_id)==int:
            indiv_cont_id_list=[indiv_cont_id]
        else:
            indiv_cont_id_list=indiv_cont_id
        try:
            other_addcomps_vals+=[np.array([Plot.addComp(addCompNum=i_comp, plotGroup=1) for i_comp in indiv_cont_id_list]).sum(0)]
        except:
            breakpoint()
            pass

    other_addcomps_ratio=[None if elem is None else np.nan_to_num(elem/cont_addcomps_sum) for elem in other_addcomps_vals]


    plt.figure(figsize=figsize)
    ax_comp=plt.gca()

    plt.xlim(ener_low,ener_high)
    plt.xlabel('Energy '+Plot.xAxis)
    plt.ylabel(ylabel_prefix+(' ' if ylabel_prefix!='' else '')+'ratio to continuum')

    if other_addcomps_labels is None:
        other_addcomps_labels_use=np.repeat('',len(other_addcomps))
    else:
        other_addcomps_labels_use=other_addcomps_labels

    if other_addcomps_colors is None:
        other_addcomps_colors_use=np.repeat(None,len(other_addcomps))
    else:
        other_addcomps_colors_use=other_addcomps_colors

    if other_addcomps_alpha is None:
        other_addcomps_alpha_use=np.repeat(None,len(other_addcomps))
    else:
        other_addcomps_alpha_use=other_addcomps_alpha

    if other_addcomps_ls is None:
        other_addcomps_ls_use=np.repeat('-',len(other_addcomps))
    else:
        other_addcomps_ls_use=other_addcomps_ls

    if other_addcomps_type is None:
        other_addcomps_type_use=np.repeat('em',len(other_addcomps))
    else:
        other_addcomps_type_use =other_addcomps_type

    for elem_ratio,elem_color,elem_label,elem_type,elem_alpha,elem_ls in \
     zip(other_addcomps_ratio,other_addcomps_colors_use,other_addcomps_labels_use,other_addcomps_type_use,
         other_addcomps_alpha_use,other_addcomps_ls_use):
        plt.plot([] if elem_ratio is None else Plot.x(1),
                  [] if elem_ratio is None else ((1 if elem_type=='em' else 0)+elem_ratio),
                 color=elem_color,label=elem_label,lw=0 if elem_color==None else None,alpha=elem_alpha,
                 ls=elem_ls,
                 )

    if ylims is not None:
        plt.ylim(ylims)

    if plot_transi:
        plot_std_ener(ax_comp,plot_indiv_transi=True,force_side='none',
                      plot_em=True,squished_mode=False,color='grey')

    if minor_locator is not False:
        ax_comp.xaxis.set_minor_locator(AutoMinorLocator(minor_locator))

    plt.tight_layout()

    plt.tight_layout()


    if manual_bbox is not False:
        #plt.legend(loc='upper left', bbox_to_anchor=(0.045, 1), )
        plt.legend(loc='upper left', bbox_to_anchor=manual_bbox, )

    else:
        plt.legend()


def plot_comp_ratio(cont_addcomps, other_addcomps, ener_low, ener_high,
                    plot_type='eemo',
                    ylims=None,
                    cont_addcomps_xcm='', other_addcomps_xcm=None,
                    other_addcomps_labels=None, other_addcomps_colors=None, figsize=(10, 5),
                    other_addcomps_type=None, other_addcomps_alpha=None, other_addcomps_ls=None,
                    plot_transi=True, minor_locator=10, ylabel_prefix='', manual_bbox=False,
                    set_ener_str=None, set_ener_xrism=False, ylog=True, ax=None, plot_transi_kwargs=None,
                    other_addcomps_lw=None, transi_lw=None, legend_loc=None, transi_alternate=True):

    '''
    Makes a ratio plot of individual additive components (or sums of them) to a continuum, itself the sum of
    other additive components

    cont_addcomps: iterable of ints
        the addcomp ids (xspec numbering, starting at 1) summed to make the continuum, to which
        all the "other_addcomps" are individually normalized

    other_addcomps: iterable of None, ints or iterables of ints
        the addcomp ids (xspec numbering, starting at 1) shown individually as their ratio to the continuum
        an iterable of ids is summed into a single displayed component
        None elements plot nothing but keep a legend entry (label only, for additional lines in the legend)

    ener_low/ener_high: x axis limits of the plot (in the Plot.xAxis unit)

    plot_type: eemo or another xspec plot type where addcomp is available
        used to get the addcomp values. The E^k weighting of mo/emo/eemo cancels in the ratio

    ylims: None or [ymin,ymax]

    cont_addcomps_xcm: path or ''
        if not '', this model is loaded and used for the continuum components only
        Note: in this case the cont_addcomps numbering refers to the components of that xcm

    other_addcomps_xcm: None, path or iterable of paths
        None: uses the currently loaded model (after cont_addcomps_xcm, if given, this is that model)
        path: loads this model for all the other_addcomps
        iterable of paths (same length as other_addcomps): one model per element of other_addcomps.
            Each distinct path is loaded once. Requires cont_addcomps_xcm
        Without cont_addcomps_xcm, the continuum is computed from the (single) model of the other_addcomps

    other_addcomps_labels/colors/alpha/ls/lw: None or iterable with one element per element of other_addcomps
        colors, alpha, ls and lw can also be a single value used for all components
        None lw uses the matplotlib default (1.5)
        None colors/alpha use the matplotlib defaults, None ls is '-'

    other_addcomps_type: None, 'em', 'abs', or iterable of 'em'/'abs' (one per element of other_addcomps)
        'em': plots 1 + component/continuum (the component on top of the continuum)
        'abs': plots component/continuum (for a model absorber*continuum this is the transmission)
        None means 'em' for all components
        Note that when combining absorption and emission it can be useful to use the "abs" type for emission
        components to see the overlap

    figsize: figure size, only used when ax is None

    plot_transi: mark the individual transitions with xspec_config_multisp.plot_transi (in grey): dashed lines
        over the whole panel, transition names centered on their energy and rotated, ion names between the two
        label rows (old_plot_comp_ratio keeps the plot_std_ener display)
    plot_transi_kwargs: None or dict, overrides the plot_transi arguments (default color='grey'), e.g.
        transi_angle, text_y, row_gap, line_ls, full_height
    transi_lw: None (plot_transi default, 0.5) or width of the transition lines
    transi_alternate: if True, transition names too close to the previous one go on a second row above the ion
        names; if False, all the transition names stay on one row below the ion names

    minor_locator: if not False, the number of minor intervals of the x axis

    ylabel_prefix: optional string added before the y label

    manual_bbox: False or bbox_to_anchor tuple of the legend (in axis fraction)
    legend_loc: None or matplotlib legend loc. With manual_bbox, the point of the legend placed at manual_bbox
        (default 'upper left'); without, the legend position (default: matplotlib 'best')

    set_ener_str/set_ener_xrism: if set_ener_str is not None, set_ener(set_ener_str, xrism=set_ener_xrism) is called
        after loading each model (and on the current model if other_addcomps_xcm is None).
        The usual choice is set_ener_str='thcomp', set_ener_xrism=True
        All the models must end up on the same energy grid (checked)

    ylog: log scale on the y axis

    ax: None or a matplotlib axis to plot in. If None, a new figure of size figsize is created

    Returns (fig, ax)

    Side effects: leaves the last loaded xcm as the current xspec model, and plots with Plot.add=True
    (the previous Plot.add value is restored)
    '''

    n_comps = len(other_addcomps)

    if other_addcomps_xcm is None or isinstance(other_addcomps_xcm, str):
        comps_xcm = [other_addcomps_xcm]*n_comps
    else:
        comps_xcm = list(other_addcomps_xcm)
        if len(comps_xcm) != n_comps:
            raise ValueError('plot_comp_ratio: %d other_addcomps_xcm for %d other_addcomps'
                             % (len(comps_xcm), n_comps))
        if cont_addcomps_xcm == '':
            raise ValueError('plot_comp_ratio: several other_addcomps_xcm require a separate cont_addcomps_xcm')

    def load_model(xcm):
        #xcm=None keeps the current model
        if xcm is not None:
            Xset.restore(xcm)
        if set_ener_str is not None:
            set_ener(set_ener_str, xrism=set_ener_xrism)
        Plot(plot_type)
        return np.array(Plot.x(1))

    prev_plot_add = Plot.add
    Plot.add = True
    try:
        x_vals = None
        if cont_addcomps_xcm != '':
            x_vals = load_model(cont_addcomps_xcm)
            x_sel = _ener_slice(x_vals, ener_low, ener_high)
            cont_addcomps_sum = _sum_addcomps(cont_addcomps, x_sel, cont_addcomps_xcm)

        other_addcomps_vals = [None]*n_comps

        #each distinct model is loaded once, for all the components that use it
        for xcm in dict.fromkeys(comps_xcm):

            desc = 'the current model' if xcm is None else xcm
            x_curr = load_model(xcm)

            if x_vals is None:
                x_vals = x_curr
                x_sel = _ener_slice(x_vals, ener_low, ener_high)
            elif not np.array_equal(x_curr, x_vals):
                raise ValueError('plot_comp_ratio: the energy grid of %s differs from the continuum one. '
                                 'Use set_ener_str to put all the models on the same grid' % desc)

            if cont_addcomps_xcm == '':
                cont_addcomps_sum = _sum_addcomps(cont_addcomps, x_sel, desc)

            for i_comp, (indiv_cont_id, comp_xcm) in enumerate(zip(other_addcomps, comps_xcm)):
                if comp_xcm != xcm or indiv_cont_id is None:
                    continue

                indiv_cont_id_list = [indiv_cont_id] if isinstance(indiv_cont_id, (int, np.integer)) \
                    else indiv_cont_id
                other_addcomps_vals[i_comp] = _sum_addcomps(indiv_cont_id_list, x_sel, desc)

        x_axis_unit = Plot.xAxis
    finally:
        Plot.add = prev_plot_add

    x_plot = x_vals[x_sel]

    #bins with a zero continuum get a zero ratio
    other_addcomps_ratio = [None if elem is None else
                            np.divide(elem, cont_addcomps_sum, out=np.zeros_like(elem),
                                      where=cont_addcomps_sum > 0)
                            for elem in other_addcomps_vals]

    return _plot_ratios(x_plot, other_addcomps_ratio, x_axis_unit, ener_low, ener_high, ylims,
                        other_addcomps_labels, other_addcomps_colors, figsize, other_addcomps_type,
                        other_addcomps_alpha, other_addcomps_ls, plot_transi, minor_locator, ylabel_prefix,
                        manual_bbox, ylog, ax, plot_transi_kwargs, other_addcomps_lw=other_addcomps_lw,
                        transi_lw=transi_lw, legend_loc=legend_loc, transi_alternate=transi_alternate)

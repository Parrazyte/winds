#note: largely AI assisted.

import os
import numpy as np
from xspec_config_multisp import *
from xspec import Xset,AllModels
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# NOTE: Xset, AllModels, xPlot and set_ener are assumed to already be
# available in the calling environment (pyxspec + your local xPlot/set_ener
# helpers)

# ---------------------------------------------------------------------------
# Model registry: one entry per "dictionary name" the wrapper can look up.
# Each entry describes how to load the model in Xspec, and which "kind" of
# parameter layout it uses (this replaces the two hardcoded dicts + the
# separate thermal-wind branch that were scattered across the original
# functions).
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'Tanimoto25': dict(
        kind='mhd',
        path='/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/'
             'DiskWinds/launching/mhd/KeigoTanimoto/init',
        restore='baseload.xcm',
    ),
    '0p1': dict(
        kind='mhd',
        path='/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/'
             'DiskWinds/launching/mhd/KeigoTanimoto/Solutions/0p1_Ledd',
        restore='baseload.xcm',
    ),
    'thermal': dict(
        kind='thermal',
        path='/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/'
             'DiskWinds/launching/thermal',
        restore_main='baseload_total.xcm',  # used for every angle except the last
        restore_last='baseload_emi.xcm',  # used for the last (edge-on/scatt.-only) angle
    ),
}

# Hard parameter bounds for the thermal model's cos(theta) sightline
# parameter (param 1). Xspec raises "Parameter value must be between
# min/max" outside of this range - angles near 30 deg / 80 deg can land
# just outside it due to float rounding, so plot_model_config clips values
# that fall within THERMAL_COS_MARGIN of a bound back onto the bound, and
# raises a clear error for anything further out.
THERMAL_COS_BOUNDS = (0.174, 0.85)
THERMAL_COS_MARGIN = 1e-1


def _thermal_cos_value(angle, margin=THERMAL_COS_MARGIN, bounds=THERMAL_COS_BOUNDS):
    """cos(angle in deg), clipped onto `bounds` if within `margin` of one of
    them, else raises ValueError."""
    lo, hi = bounds
    val = np.cos(np.pi / 180 * angle)
    if val < lo:
        if val >= lo - margin:
            return lo
        raise ValueError(
            f"Thermal model: angle {angle}deg -> cos={val:.6f} is below the "
            f"parameter minimum ({lo}) by more than the {margin} margin.")
    if val > hi:
        if val <= hi + margin:
            return hi
        raise ValueError(
            f"Thermal model: angle {angle}deg -> cos={val:.6f} is above the "
            f"parameter maximum ({hi}) by more than the {margin} margin.")
    return val


# 'mhd'     -> inclination on param 2, norm on param 4, param 8 linked to p4
#              except for the last (most edge-on) angle, which unlinks and
#              forces the norm to 0 (mirrors the original "can't go below
#              80 degrees for Ryota's setup" special-casing).
# 'thermal' -> inclination as cos(angle) on param 1, norm on param 3; the
#              last angle needs a different .xcm restore (scattering-only
#              setup), the others share one restore.

def _dynamic_renorm_value(norm_index, energy_band, target, base_norm=1.):
    """
    Find the normalization value (for Xspec model parameter `norm_index` of
    the currently loaded model) that places the median 'eemo' continuum
    level within `energy_band` at `target`.

    Temporarily sets the parameter to `base_norm`, runs Plot('eemo'), takes
    the median of Plot.y() restricted to `energy_band`, and rescales
    linearly (the normalization parameter scales the output proportionally)
    so that the resulting curve is centered on `target`.
    """
    AllModels(1)(norm_index).values = base_norm
    Plot('eemo')
    x = np.array(Plot.x())
    y = np.array(Plot.model())
    mask = (x >= energy_band[0]) & (x <= energy_band[1])
    if not np.any(mask):
        raise ValueError(
            f"No Plot.x() values fall within energy_band={energy_band}; "
            f"cannot renormalize.")
    median_y = np.median(y[mask])

    if median_y == 0:
        raise ValueError(
            "Median continuum value in energy_band is zero; cannot "
            "renormalize (division by zero).")
    return base_norm * target / median_y


def plot_model_config(model_key, energy_band, angles, seps, color, alpha=0.5,
                       ls='-', label=None, plot_hlines=True, ax=None,
                       set_ener_str='thcomp', set_ener_xrism=True, figsize=(10, 8),
                       thermal_margin=THERMAL_COS_MARGIN, firstmod=True,
                       renorm_last=False, renorm_target=None):

    """
    Plot a single model's synthetic residuals-to-continuum ('eemo') spectrum
    across a set of inclination angles, on a shared axis.

    Parameters
    ----------
    model_key : str
        Key into MODEL_REGISTRY, e.g. 'Tanimoto25', '0p1', 'thermal'.
    energy_band : tuple(float, float)
        xlims passed to xPlot, e.g. (6, 7.1) or (2., 3.).
    angles : list of float
        Inclination angles (degrees), ordered face-on -> edge-on. The LAST
        entry is treated as the limiting/scattering-only case.
    seps : list of float
        Same length as `angles`. Used both as the normalization multiplier
        for each angle and as the y-value of the (optional) horizontal
        separator lines.
    color : str
        Color used for every angle of this model.
    ls : str
        Linestyle used for this model's legend proxy line (xPlot itself is
        only ever called with `model_colors=color` in the original code, so
        `ls` only affects the legend entry drawn here, not the spectra).
    label : str or None
        Legend label; defaults to `model_key`.
    plot_hlines : bool
        Whether to draw dashed horizontal lines at each value in `seps`.
    ax : matplotlib.axes.Axes or None
        Axis to draw into; a new figure/axis is created if None.
    set_ener_str : str
        Passed through to `set_ener`.
    set_ener_xrism : bool
        Passed through to `set_ener`.
    figsize : tuple(float, float)
        Figure size used when `ax` is None (i.e. a new figure is created).
    thermal_margin : float
        Only used for `kind='thermal'` models. cos(angle) values that fall
        outside [THERMAL_COS_BOUNDS] but within this margin of a bound are
        clipped onto the bound; values further out raise a ValueError.
    firstmod : bool
        Determines whether to plot at least one secondary axis.
    renorm_last : bool
        If True, the last (most edge-on / limiting, e.g. 80 deg) sightline
        is not plotted with its normalization taken directly from `seps`
        (or forced to 0, for 'mhd' models). Instead, the normalization is
        found dynamically with `_dynamic_renorm_value` so the plotted
        continuum's median over `energy_band` sits at `renorm_target`.
    renorm_target : float or None
        Target continuum height used when `renorm_last` is True; defaults
        to `seps[-1]` (i.e. the same height its horizontal separator line
        is drawn at) when not given.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    if len(angles) != len(seps):
        raise ValueError("`angles` and `seps` must have the same length")
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key '{model_key}'; "
                        f"available: {list(MODEL_REGISTRY)}")

    if renorm_last and renorm_target is None:
        renorm_target = seps[-1]

    config = MODEL_REGISTRY[model_key]
    kind = config['kind']
    n_angles = len(angles)

    if kind == 'thermal':
        thermal_cos_values = [_thermal_cos_value(a, margin=thermal_margin)
                               for a in angles]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if plot_hlines:
        for sep_val in seps:
            plt.axhline(sep_val, color='gray', lw=0.5, alpha=0.5, ls='--')

    currdir = os.getcwd()

    if kind == 'mhd':
        os.chdir(config['path'])
        Xset.restore(config['restore'])
        set_ener(set_ener_str, xrism=set_ener_xrism)

        for i, (angle, sep) in enumerate(zip(angles, seps)):
            is_last = (i == n_angles - 1)
            AllModels(1)(2).values = angle
            AllModels(1)(4).values = sep
            if i == 0:
                AllModels(1)(8).link = 'p4'
            if is_last:
                AllModels(1)(8).link = ''
                AllModels(1)(4).values = 0.
                if renorm_last:
                    AllModels(1)(8).values = _dynamic_renorm_value(
                        8, energy_band, renorm_target)
                else:
                    AllModels(1)(8).values=sep

            xPlot('eemo', xlims=energy_band, axes_input=ax, model_colors=color,
                  secondary_x=firstmod and i == 0, model_alpha=alpha,model_ls=[ls])

    elif kind == 'thermal':
        os.chdir(config['path'])
        Xset.restore(config['restore_main'])
        set_ener(set_ener_str, xrism=set_ener_xrism)

        for i, (angle, sep, cos_val) in enumerate(zip(angles, seps, thermal_cos_values)):
            is_last = (i == n_angles - 1)
            if is_last:
                os.chdir(config['path'])
                Xset.restore(config['restore_last'])
                set_ener(set_ener_str, xrism=set_ener_xrism)
            AllModels(1)(1).values = cos_val
            if is_last and renorm_last:
                AllModels(1)(3).values = _dynamic_renorm_value(
                    3, energy_band, renorm_target)
            else:
                AllModels(1)(3).values = sep
            xPlot('eemo', xlims=energy_band, axes_input=ax, model_colors=color,
                  secondary_x=firstmod and i == 0, model_alpha=alpha,model_ls=[ls])

    else:
        raise ValueError(f"Unknown model kind '{kind}' for '{model_key}'")

    plt.xscale('linear')
    ax.tick_params(labelbottom=True)

    plt.plot([], [], color=color, ls=ls, label=label or model_key)

    os.chdir(currdir)
    return ax


def plot_model_comparison(model_keys, energy_band, angles, seps,
                          colors,alpha=0.5, ls_list=None, labels=None,
                          save=None, ylim=None, figsize=(10, 8),
                          thermal_margin=THERMAL_COS_MARGIN,renorm_last=False,renorm_target=None,
                          legend_loc=None,legend_bbox=None,
                          swap_yaxis=False):
    """
    Overlay several models (as configured in MODEL_REGISTRY) on one figure,
    sharing a common energy band, angle list and separator list. Generalizes
    model_compa_MHD_highE/lowE and model_compa_highE/lowE to any number and
    mix of models, by repeatedly calling plot_model_config.

    Parameters
    ----------
    model_keys : list of str
    energy_band : tuple(float, float)
    angles : list of float
        Shared across all models.
    seps : list of float
        Shared across all models.
    colors : list of str
        One per model.
    ls_list : list of str or None
        One per model; defaults to '-' for all.
    labels : list of str or None
        One per model; defaults to `model_keys`.
    save : str or None
        If given, path to save the final figure to.
    ylim : tuple(float, float) or None
        Defaults to (1e-5, current auto max).
    figsize : tuple(float, float)
        Figure size for the comparison plot.
    thermal_margin : float
        Forwarded to `plot_model_config` for any 'thermal'-kind model.

    renorm_last : bool
        If True, the last (most edge-on / limiting, e.g. 80 deg) sightline
        is not plotted with its normalization taken directly from `seps`
        (or forced to 0, for 'mhd' models). Instead, the normalization is
        found dynamically with `_dynamic_renorm_value` so the plotted
        continuum's median over `energy_band` sits at `renorm_target`.
    renorm_target : float or None
        Target continuum height used when `renorm_last` is True; defaults
        to `seps[-1]` (i.e. the same height its horizontal separator line
        is drawn at) when not given.

    Returns
    -------
    fig, ax
    """
    n = len(model_keys)
    if ls_list is None:
        ls_list = ['-'] * n
    if labels is None:
        labels = list(model_keys)
    if not (len(colors) == len(ls_list) == len(labels) == n):
        raise ValueError("model_keys, colors, ls_list and labels must all have the same length")

    fig, ax = plt.subplots(figsize=figsize)

    for i, (key, color, ls, label) in enumerate(zip(model_keys, colors, ls_list, labels)):
        plot_model_config(
            key, energy_band, angles, seps, color,alpha=alpha,
            ls=ls, label=label,
            plot_hlines=(i == 0),  # draw the separator lines only once
            ax=ax,
            thermal_margin=thermal_margin,firstmod=i==0,
            renorm_last=renorm_last,renorm_target=renorm_target
        )
    if swap_yaxis:
        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            labelleft=False,
            labelright=True,
            direction='out')
        ax.yaxis.set_label_position("right")
    plt.ylabel('residuals to continuum (shifted for clarity)')


    ax_right = ax.secondary_yaxis('left' if swap_yaxis else 'right')
    order = np.argsort(seps)
    angle_labels = [f'{a:g}°' for a in angles]
    angle_labels[-1] += '\n(scatt.\nonly)'
    ax_right.set_yticks(np.array(seps)[order])
    ax_right.set_yticklabels(np.array(angle_labels)[order])
    ax_right.set_ylabel('sightline (0°=face-on)')

    if ylim is None:
        ylim = (1e-5, plt.ylim()[1])
    plt.ylim(*ylim)

    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    plt.tight_layout()
    plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)

    if save is not None:
        plt.savefig(save)

    return fig, ax


# ---------------------------------------------------------------------------
# Equivalents of the original four functions, expressed with the wrappers:
#
# model_compa_MHD_highE(['Tanimoto25', '0p1'], colors=['blue', 'cyan']) ==
#   plot_model_comparison(['Tanimoto25', '0p1'], (6, 7.1),
#                          [30, 50, 70, 80], [1, 0.1, 0.01, 0.001],
#                          colors=['blue', 'cyan'])
#
# model_compa_highE(model_mhd='0p1') ==
#   plot_model_comparison(['0p1', 'thermal'], (6, 7.1),
#                          [30, 50, 70, 80], [1, 0.1, 0.01, 0.001],
#                          colors=['blue', 'red'],
#                          labels=['cold MHD wind (no clumping)',
#                                  'thermal-radiative wind (no clumping)'])
#
# (lowE versions: same calls with energy_band=(2., 3.))
# ---------------------------------------------------------------------------

#combined setup for all three models:
if 0:

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching')
    plot_model_comparison(['0p1', 'Tanimoto25', 'thermal'], (2., 3.),
                          [30, 50, 70, 80], [1, 0.1, 0.01, 2e-5],
                          # labels=[r'MHD 0.1 L$_{Edd}$',
                          #         'MHD 0.3 L$_{Edd}$' + ' - truncated + very large disk',
                          #         'RHD 0.5 L$_{Edd}$'+ ' - large disk'],
                          labels=[r'MHD 0.1 L$_{Edd}$', 'MHD Tanimoto+25', 'RHD Tomaru+20'],
                          colors=['blue', 'cyan', 'red'], renorm_last=True, ylim=(1e-5, 3), figsize=(7, 7 * 8 / 10),
                          legend_loc='lower left', legend_bbox=[0.005, 0.543])

    plot_model_comparison(['0p1', 'Tanimoto25', 'thermal'], (6, 7.1),
                          [30, 50, 70, 80], [1, 0.1, 0.01, 2e-5],
                          labels=[r'MHD 0.1 L$_{Edd}$', 'MHD Tanimoto+25', 'RHD Tomaru+20'],
                          colors=['blue', 'cyan', 'red'], renorm_last=True, ylim=(1e-5, 3), figsize=(7, 7 * 8 / 10),
                          legend_loc='lower left', legend_bbox=[0.005, 0.18],swap_yaxis=True)


    plot_model_comparison(['Tanimoto25', '0p1', 'thermal'], (6, 7.1),
                          [30, 50, 70, 80], [1, 0.1, 0.01, 5e-5],
                          colors=['blue', 'cyan', 'red'], renorm_last=True, ylim=(3e-5, 3))

    plot_model_comparison(['Tanimoto25', '0p1','thermal'], (2., 3.),
                         [30, 50, 70, 80], [1, 0.1, 0.01, 0.001],
                         colors=['blue', 'cyan','red'])



#old functions


def model_compa_MHD_highE(model_mhd_list,colors=[],save=None):

    dict_mhd_baseload={'Tanimoto25':'/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto/init',
                       '0p1':'/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto/Solutions/0p1_Ledd'}

    currdir = os.getcwd()
    fig, ax = plt.subplots(figsize=(10, 8))

    for elem_mod,elem_color in zip(model_mhd_list,colors):

        # for mhd
        os.chdir(dict_mhd_baseload[elem_mod])

        Xset.restore('baseload.xcm')
        set_ener('thcomp', xrism=True)

        plt.axhline(1, color='gray', lw=0.5, alpha=0.5, ls='--')
        plt.axhline(0.1, color='gray', lw=0.5, alpha=0.5, ls='--')
        plt.axhline(0.01, color='gray', lw=0.5, alpha=0.5, ls='--')
        plt.axhline(0.001, color='gray', lw=0.5, alpha=0.5, ls='--')
        AllModels(1)(2).values = 30.
        AllModels(1)(4).values = 1.
        AllModels(1)(8).link = 'p4'
        xPlot('eemo', xlims=(6, 7.1), axes_input=ax, model_colors=elem_color)
        AllModels(1)(2).values = 50.
        AllModels(1)(4).values = 0.1
        xPlot('eemo', xlims=(6, 7.1), axes_input=ax, model_colors=elem_color)
        AllModels(1)(2).values = 70.
        AllModels(1)(4).values = 0.01
        xPlot('eemo', xlims=(6, 7.1), axes_input=ax, model_colors=elem_color)

        # since we cannot really go below 80 degrees for Ryota's setup
        AllModels(1)(2).values = 80
        AllModels(1)(4).values = 0.001
        AllModels(1)(8).link = ''
        AllModels(1)(4).values = 0.
        xPlot('eemo', xlims=(6, 7.1), axes_input=ax, model_colors=elem_color)
        plt.xscale('linear')
        plt.gca().get_children()[-3].remove()
        plt.gca().get_children()[-3].remove()
        plt.gca().get_children()[-3].remove()
        ax.tick_params(labelbottom=True)

        plt.plot([],[],color=elem_color,label=elem_mod)


        plt.legend()
        plt.ylabel('residuals to continuum (shifted for clarity)')
        ax_right=ax.secondary_yaxis('right')
        ax_right.set_yticks([1e-3,1e-2,1e-1,1])
        ax_right.set_yticklabels(['80°\n(scatt. \nonly)','70°','50°','30°',])
        ax_right.set_ylabel('sightline (0°=face-on)')
        plt.ylim(1e-5,plt.ylim()[1])
        plt.tight_layout()

        ax.xaxis.set_minor_locator(MultipleLocator(0.02))

        os.chdir(currdir)
        if save is not None:
            plt.savefig(save)

def model_compa_MHD_lowE(model_mhd_list,colors=[],save=None):

    dict_mhd_baseload={'Tanimoto25':'/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto/init',
                       '0p1':'/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto/Solutions/0p1_Ledd'}

    currdir = os.getcwd()
    fig, ax = plt.subplots(figsize=(10, 8))

    for elem_mod,elem_color in zip(model_mhd_list,colors):

        # for mhd
        os.chdir(dict_mhd_baseload[elem_mod])

        Xset.restore('baseload.xcm')
        set_ener('thcomp', xrism=True)

        plt.axhline(1, color='gray', lw=0.5, alpha=0.5, ls='--')
        plt.axhline(0.1, color='gray', lw=0.5, alpha=0.5, ls='--')
        plt.axhline(0.01, color='gray', lw=0.5, alpha=0.5, ls='--')
        plt.axhline(0.001, color='gray', lw=0.5, alpha=0.5, ls='--')
        AllModels(1)(2).values = 30.
        AllModels(1)(4).values = 1.
        AllModels(1)(8).link = 'p4'
        xPlot('eemo', xlims=(2., 3.), axes_input=ax, model_colors=elem_color)
        AllModels(1)(2).values = 50.
        AllModels(1)(4).values = 0.1
        xPlot('eemo', xlims=(2., 3.), axes_input=ax, model_colors=elem_color)
        AllModels(1)(2).values = 70.
        AllModels(1)(4).values = 0.01
        xPlot('eemo', xlims=(2., 3.), axes_input=ax, model_colors=elem_color)

        # since we cannot really go below 80 degrees for Ryota's setup
        AllModels(1)(2).values = 80
        AllModels(1)(4).values = 0.001
        AllModels(1)(8).link = ''
        AllModels(1)(4).values = 0.
        xPlot('eemo', xlims=(2., 3.), axes_input=ax, model_colors=elem_color)
        plt.xscale('linear')
        plt.gca().get_children()[-3].remove()
        plt.gca().get_children()[-3].remove()
        plt.gca().get_children()[-3].remove()

        plt.plot([],[],color=elem_color,label=elem_mod)


        plt.legend()
        plt.ylabel('residuals to continuum (shifted for clarity)')
        ax_right=ax.secondary_yaxis('right')
        ax_right.set_yticks([1e-3,1e-2,1e-1,1])
        ax_right.set_yticklabels(['80°\n(scatt. \nonly)','70°','50°','30°',])
        ax_right.set_ylabel('sightline (0°=face-on)')
        plt.ylim(1e-5,plt.ylim()[1])
        plt.tight_layout()

        ax.xaxis.set_minor_locator(MultipleLocator(0.02))

        # if combining
        # plt.gca().get_children()[-3].remove()
        ax.tick_params(labelbottom=True)
        plt.tight_layout()

        os.chdir(currdir)
        if save is not None:
            plt.savefig(save)


def model_compa_highE(save=None,model_mhd='0p1'):
    #for mhd

    currdir=os.getcwd()
    # for mhd
    if model_mhd=='old':
        os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto')
    elif model_mhd=='0p1':
        os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto/Solutions/0p1_Ledd')

    Xset.restore('baseload.xcm')
    set_ener('thcomp',xrism=True)

    fig,ax=plt.subplots(figsize=(10,8))
    plt.axhline(1,color='gray',lw=0.5,alpha=0.5,ls='--')
    plt.axhline(0.1,color='gray',lw=0.5,alpha=0.5,ls='--')
    plt.axhline(0.01,color='gray',lw=0.5,alpha=0.5,ls='--')
    plt.axhline(0.001,color='gray',lw=0.5,alpha=0.5,ls='--')
    AllModels(1)(2).values=30.
    AllModels(1)(4).values=1.
    AllModels(1)(8).link='p4'
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')
    AllModels(1)(2).values=50.
    AllModels(1)(4).values=0.1
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')
    AllModels(1)(2).values=70.
    AllModels(1)(4).values=0.01
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')

    #since we cannot really go below 80 degrees for Ryota's setup
    AllModels(1)(2).values=80
    AllModels(1)(4).values=0.001
    AllModels(1)(8).link=''
    AllModels(1)(4).values=0.
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='blue')
    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    ax.tick_params(labelbottom=True)

    plt.tight_layout()

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_total.xcm')
    set_ener('thcomp',xrism=True)
    # fig,ax=plt.subplots(figsize=(10,8))
    # plt.axhline(1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.01,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.001,color='gray',lw=0.5,alpha=0.5,ls='--')
    #basically 30 degrees
    AllModels(1)(1).values=0.85
    AllModels(1)(3).values=1.
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')
    AllModels(1)(1).values=np.cos(np.pi/180*50)
    AllModels(1)(3).values=0.1
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')
    AllModels(1)(1).values=np.cos(np.pi/180*70)
    AllModels(1)(3).values=0.01
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_emi.xcm')
    set_ener('thcomp',xrism=True)
    #basically 80 degrees
    AllModels(1)(1).values=0.16
    AllModels(1)(3).values=0.001
    xPlot('eemo',xlims=(6,7.1),axes_input=ax,model_colors='red')

    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()

    #if combining
    plt.gca().get_children()[-3].remove()
    ax.tick_params(labelbottom=True)
    plt.tight_layout()

    #replacing the legend
    plt.gca().get_children()[-2].remove()
    plt.plot([],[],color='red',label='thermal-radiative wind (no clumping)')
    plt.plot([],[],color='blue',label='cold MHD wind (no clumping)')
    plt.legend()
    plt.ylabel('residuals to continuum (shifted for clarity)')
    ax_right=ax.secondary_yaxis('right')
    ax_right.set_yticks([1e-3,1e-2,1e-1,1])
    ax_right.set_yticklabels(['80°\n(scatt. \nonly)','70°','50°','30°',])
    ax_right.set_ylabel('sightline (0°=face-on)')
    plt.ylim(1e-5,plt.ylim()[1])
    plt.tight_layout()

    # #if combining
    # plt.gca().get_children()[-3].remove()
    # ax.tick_params(labelbottom=True)
    # plt.tight_layout()

    ax.xaxis.set_minor_locator(MultipleLocator(0.02))

    os.chdir(currdir)
    if save is not None:
        plt.savefig(save)

def model_compa_lowE(save=None,model_mhd='0p1'):

    currdir=os.getcwd()

    # for mhd
    if model_mhd=='old':
        os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto')
    elif model_mhd=='0p1':
        os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/mhd/KeigoTanimoto/Solutions/0p1_Ledd')

    Xset.restore('baseload.xcm')
    set_ener('thcomp', xrism=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.axhline(1, color='gray', lw=0.5, alpha=0.5, ls='--')
    plt.axhline(0.1, color='gray', lw=0.5, alpha=0.5, ls='--')
    plt.axhline(0.01, color='gray', lw=0.5, alpha=0.5, ls='--')
    plt.axhline(0.001, color='gray', lw=0.5, alpha=0.5, ls='--')
    AllModels(1)(2).values = 30.
    AllModels(1)(4).values = 1.
    AllModels(1)(8).link = 'p4'
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')
    AllModels(1)(2).values = 50.
    AllModels(1)(4).values = 0.1
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')
    AllModels(1)(2).values = 70.
    AllModels(1)(4).values = 0.01
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')

    # since we cannot really go below 80 degrees for Ryota's setup
    AllModels(1)(2).values = 80
    AllModels(1)(4).values = 0.001
    AllModels(1)(8).link = ''
    AllModels(1)(4).values = 0.
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='blue')
    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()


    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_total.xcm')
    set_ener('thcomp', xrism=True)
    # fig,ax=plt.subplots(figsize=(10,8))
    # plt.axhline(1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.1,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.01,color='gray',lw=0.5,alpha=0.5,ls='--')
    # plt.axhline(0.001,color='gray',lw=0.5,alpha=0.5,ls='--')
    # basically 30 degrees
    AllModels(1)(1).values = 0.85
    AllModels(1)(3).values = 1.
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')
    AllModels(1)(1).values = np.cos(np.pi / 180 * 50)
    AllModels(1)(3).values = 0.1
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')
    AllModels(1)(1).values = np.cos(np.pi / 180 * 70)
    AllModels(1)(3).values = 0.01
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')
    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/NewAthena/SpecialIssue/DiskWinds/launching/thermal')
    Xset.restore('baseload_emi.xcm')
    set_ener('thcomp', xrism=True)
    # basically 80 degrees
    AllModels(1)(1).values = 0.16
    AllModels(1)(3).values = 0.001
    xPlot('eemo', xlims=(2.,3.), axes_input=ax, model_colors='red')

    plt.xscale('linear')
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()
    plt.gca().get_children()[-3].remove()

    # if combining
    plt.gca().get_children()[-3].remove()
    ax.tick_params(labelbottom=True)
    plt.tight_layout()

    # replacing the legend
    plt.gca().get_children()[-2].remove()
    plt.plot([], [], color='red', label='thermal-radiative wind (no clumping)')
    plt.plot([], [], color='blue', label='cold MHD wind (no clumping)')
    plt.legend()
    plt.ylabel('residuals to continuum (shifted for clarity)')
    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax_right.set_yticklabels(['80°\n(scatt. \nonly)', '70°', '50°', '30°', ])
    ax_right.set_ylabel('sightline (0°=face-on)')
    plt.ylim(3e-6, plt.ylim()[1])
    plt.tight_layout()

    ax.xaxis.set_minor_locator(MultipleLocator(0.02))

    os.chdir(currdir)
    if save is not None:
        plt.savefig(save)

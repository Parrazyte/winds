
'''
note: claude assisted
'''
import os,sys
import numpy as np
import matplotlib.pyplot as plt

from general_tools import ravel_ragged,edd_factor

from scipy.stats import linregress
from matplotlib.collections import LineCollection

import warnings

def colored_line(x, y, c, ax, **lc_kwargs):

    """
    from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # round caps/joins close the small notch that a sharp ~90° turn (e.g. a
    # Q-shape branch meeting a transition) leaves with the default butt caps.
    # At the point densities these branches use, round caps don't introduce
    # any visible beading on the straight stretches -- the cost is free here.
    default_kwargs = {"capstyle": "round", "joinstyle": "round"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def Q_shape_old(val_csv_hot=None,val_csv_warm=None,val_csv_cold=None,
                factor_soft=1,
                factor_hard=1,
                col_warm='magma_r',
                col_hot='viridis_r',

                #in 1e22 units
                nh_lim_hot=10,
                nh_lim_warm=10,

                d_kpc=8,M_CO=8,
                ledd_lowbranch=0.01,ledd_highbranch=0.1,lw=10,
                transi_width_margin=0.0065,
                figsize=(6,4)):

    '''
    Computes the Q-shape evolution of a given parameter according to a sampling of values for a given flux

    val_csv_warm: csv containing the values of the constraints with warm absorber with the hard state SED,
    val_csv_hot: csv containing the values of the constraints with a hot absorber with a hard state SED,

    factor_soft: flux multiplier for soft state SEDs. Used to draw the vertical line of the soft state and
                the horizontal lines linking the two together.
    factor_hard: flux multiplier for hard state SEDs.
                replaces factor_soft as an inverse value applied to hard state SEDs.
                only one of these values should be provided.

    col: colormap
    d_kpc,M_CO: used to normalize the position of the Q-shape

    nh_lim_hot/warm:
        use to define the lower limit of the hard stat descence. Stops when the hot or warm nh limit is above this value.
    outdated example:
    Q_shape_col('photo_nh_lim_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_0.1_1000_9_in_2p0_10p0_keV_mod_pion_abs_NS_0.005_warm.txt',
    'photo_nh_lim_mod_fit_hard_2e22_50ks_41_iter_line_FeKa26abs_flux_0.1_1000_9_in_2p0_10p0_keV_mod_pion_abs_NS_0.005_hot.txt',
    figsize=(6,4))

Q_shape_col('photo_nh_noise_mod_SED_soft_0p1Edd_2e22_5ks_1000_iter_flux_100_1_5_in_0p3_10p0_keV_mod_pion_abs_canon_soft.txt',
'photo_nh_noise_mod_SED_soft_0p1Edd_2e22_5ks_1000_iter_flux_100_1_5_in_0p3_10p0_keV_mod_pion_abs_canon_soft.txt',
factor_hard=0.5,nh_lim_hot=10,nh_lim_warm=10)


    '''

    assert factor_soft==1 or factor_hard==1,'Error: only one of the soft/hard slopes can be rescaled'

    val_hot=np.loadtxt(val_csv_hot)
    val_3sig_hot=val_hot.T[-1]
    val_flux_hot=val_hot.T[0]

    if val_csv_warm is not None:
        val_warm=np.loadtxt(val_csv_warm)
        val_3sig_warm=val_warm.T[-1]
        val_flux_warm=val_warm.T[0]


    edd_conv=edd_factor(d_kpc,M_CO).value

    '''
    #hot
    '''
    if val_csv_hot is not None:
        logslope_hot, logintercept_hot, r_value, p_value, std_err = linregress(np.log10(val_flux_hot*edd_conv),
                                                                                 np.log10(val_3sig_hot))

        #defining the y axis flux range for the hot hard state
        eddrange_hot_hard=np.logspace(-5,np.log10(ledd_highbranch),200)

        val_pred_hot_hard=10**(logslope_hot*np.log10(eddrange_hot_hard*factor_hard)+logintercept_hot)


        #defining the y axis flux range for the hot soft state
        eddrange_hot_soft=np.logspace(np.log10(ledd_lowbranch),np.log10(ledd_highbranch),100)

        #and computing the predicted limits considering  the soft factor
        val_pred_hot_soft=10**(logslope_hot*np.log10(eddrange_hot_soft*factor_soft)+logintercept_hot)


        #defining the x axis hr range for the state transitions
        hr_hot_transi=np.logspace(-1-transi_width_margin,0.+transi_width_margin,100)

        # this will make a logarithmic evolution between factor_soft and 1 if factor_hard is set to 1,
        # and between 1 and factor_hard if factor_soft is set to 1
        factor_norm_hot = (abs(np.log10(hr_hot_transi)) * (factor_soft - factor_hard) + factor_hard)

        val_pred_hot_transi_high = 10 ** (logslope_hot * \
                                           # normalizing for constant logarithmic step to *factor soft in soft
                                           # and *1 in hard
                                           np.log10(factor_norm_hot * ledd_highbranch)
                                           + logintercept_hot)

        val_pred_hot_transi_low = 10 ** (logslope_hot * \
                                          # normalizing for constant logarithmic step to *factor soft in soft
                                          # and *1 in hard
                                          np.log10(factor_norm_hot * ledd_lowbranch)
                                          + logintercept_hot)

    if val_csv_warm is not None:

        # Perform linear regression in log-log space with y in eddington units
        logslope_warm, logintercept_warm, r_value, p_value, std_err = linregress(np.log10(val_flux_warm*edd_conv),
                                                                                 np.log10(val_3sig_warm))

        #defining the y axis flux range for the warm hard state
        eddrange_warm_hard=np.logspace(-5,np.log10(ledd_highbranch),200)

        val_pred_warm_hard=10**(logslope_warm*np.log10(eddrange_warm_hard*factor_hard)+logintercept_warm)


        #defining the y axis flux range for the warm soft state
        eddrange_warm_soft=np.logspace(np.log10(ledd_lowbranch),np.log10(ledd_highbranch),100)

        #and computing the predicted limits considering the soft factor
        val_pred_warm_soft=10**(logslope_warm*np.log10(eddrange_warm_soft*factor_soft)+logintercept_warm)


        #defining the x axis hr range for the state transitions
        #note: this is not strictly correct since we should only add margins to the display but it won't matter much

        hr_warm_transi = np.logspace(-1 - transi_width_margin, 0. + transi_width_margin, 100)

        #this will make a logarithmic evolution between factor_soft and 1 if factor_hard is set to 1,
        #and between 1 and factor_hard if factor_soft is set to 1
        factor_norm_warm=(abs(np.log10(hr_warm_transi)) * (factor_soft - factor_hard) + factor_hard)

        val_pred_warm_transi_high = 10 ** (logslope_warm * \
                                           # normalizing for constant logarithmic step to *factor soft in soft
                                           # and *1 in hard
                                           np.log10(factor_norm_warm * ledd_highbranch)
                                           + logintercept_warm)

        val_pred_warm_transi_low = 10 ** (logslope_warm * \
                                          # normalizing for constant logarithmic step to *factor soft in soft
                                          # and *1 in hard
                                          np.log10(factor_norm_warm * ledd_lowbranch)
                                          + logintercept_warm)

        vrange_warm=np.log10(np.array([min(val_pred_warm_hard.tolist()+val_pred_warm_soft.tolist()),
                     max(val_pred_warm_hard.tolist() + val_pred_warm_soft.tolist())]))



    # fig,ax= plt.subplots(1, 3, figsize=figsize, width_ratios=[1,19, 1])
    fig,ax= plt.subplots(figsize=figsize)

    ax.set_xscale('log')
    ax.set_xlabel(r'Spectral Hardness ([6-10]/[3-6] keV flux)')
    ax.set_xlim(0.05,2)

    ax.set_yscale('log')
    ax.set_ylabel(r'Luminosity ($L/L_{Edd}$)')
    ax.set_ylim(1e-5,1)

    if val_csv_warm is not None:

        combined_x_warm = np.repeat(1, len(val_pred_warm_hard)).tolist() + hr_warm_transi.tolist()[::-1] + \
                          np.repeat(0.1, len(val_pred_warm_soft)).tolist()[::-1] + hr_warm_transi.tolist()

        combined_y_warm = eddrange_warm_hard.tolist() + np.repeat(ledd_highbranch, len(hr_warm_transi)).tolist()[::-1] + \
                          eddrange_warm_soft.tolist()[::-1] + np.repeat(ledd_lowbranch, len(hr_warm_transi)).tolist()

        combined_c_warm = np.log10(val_pred_warm_hard).tolist() + np.log10(val_pred_warm_transi_high).tolist()[::-1] + \
                          np.log10(val_pred_warm_soft).tolist()[::-1] + np.log10(val_pred_warm_transi_low).tolist()

        mask_use_warm = np.array(combined_c_warm) < np.log10(nh_lim_warm)
        vrange_warm = np.log10(np.array([min(10 ** np.array(combined_c_warm)[mask_use_warm]),
                                         max(10 ** np.array(combined_c_warm)[mask_use_warm])])).round(1)

        lines_warm = colored_line(np.array(combined_x_warm)[mask_use_warm],
                                  np.array(combined_y_warm)[mask_use_warm],
                                  np.array(combined_c_warm)[mask_use_warm],
                                  ax=ax, cmap=col_warm, lw=lw, clim=vrange_warm)

        # cbar_warm=fig.colorbar(val_warm_hard)
        cbar_warm = fig.colorbar(lines_warm, pad=0.04, )
        cbar_warm.ax.invert_yaxis()
        cbar_warm_ticks_adj = ['%.1e' % elem for elem in 10 ** (22 + cbar_warm.get_ticks())]
        cbar_warm.set_ticklabels(cbar_warm_ticks_adj)
        cbar_warm.set_label('warm wind \n NH', y=1.1, rotation=0, labelpad=-57)

        # lines_warm=colored_line(combined_x_warm,
        #                         combined_y_warm,
        #                         combined_c_warm,ax=ax,cmap=col_warm,lw=lw,clim=vrange_warm)

    if val_csv_hot is not None:

        margin_x_hot=0.22
        combined_x_hot=np.repeat(1*(1+margin_x_hot),len(val_pred_hot_hard)).tolist()+\
                        [1+margin_x_hot]+\
                        hr_hot_transi.tolist()[::-1]+\
                        [0.1/(1+margin_x_hot)] + \
                        np.repeat(0.1/(1+margin_x_hot),len(val_pred_hot_soft)).tolist()[::-1]+\
                        hr_hot_transi.tolist()

        combined_y_hot=(1.*eddrange_hot_hard).tolist()+\
                        [ledd_highbranch*1.8]+\
                        np.repeat(ledd_highbranch*1.8,len(hr_hot_transi)).tolist()[::-1] +\
                        [ledd_highbranch * 1.8] +\
                        ((1.)*eddrange_hot_soft).tolist()[::-1]+\
                       np.repeat(ledd_lowbranch/2,len(hr_hot_transi)).tolist()

        combined_c_hot=(np.log10(val_pred_hot_hard).tolist()+
                        [np.log10(val_pred_hot_hard[-1])]+
                        np.log10(val_pred_hot_transi_high).tolist()[::-1]+
                        [np.log10(val_pred_hot_soft[-1])]+\
                   np.log10(val_pred_hot_soft).tolist()[::-1]+np.log10(val_pred_hot_transi_low).tolist())



        mask_use_hot = np.array(combined_c_hot) < np.log10(nh_lim_hot)


        vrange_hot=np.log10(np.array([min(10**np.array(combined_c_hot)[mask_use_hot]),
                                      max(10**np.array(combined_c_hot)[mask_use_hot])])).round(1)

        lines_hot=colored_line(np.array(combined_x_hot)[mask_use_hot],
                               np.array(combined_y_hot)[mask_use_hot],
                               np.array(combined_c_hot)[mask_use_hot],
                               ax=ax,cmap=col_hot,lw=lw,clim=vrange_hot)


        cbar_hot=fig.colorbar(lines_hot,location='left',pad=0.16)
        cbar_hot.ax.invert_yaxis()
        cbar_hot_ticks_adj=['%.1e'%elem for elem in 10 ** (22 + cbar_hot.get_ticks())]
        cbar_hot.set_ticklabels(cbar_hot_ticks_adj)
        cbar_hot.set_label('hot wind \n NH',y=1.,rotation=0,labelpad=-55)
        cbar_hot.ax.yaxis.set_ticks_position('left')




    plt.tight_layout()



import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import linregress





def _nh_predict(ledd, factor, logslope, logintercept):
    """
    Predicted NH (in 1e22 cm^-2, same units as the input csv) at a given
    L/Ledd, for a branch whose incident SED normalization is rescaled by
    `factor` relative to the flux grid the csv was fitted on.
    """
    return 10 ** (logslope * np.log10(ledd * factor) + logintercept)


def _log_interp_factor(hr, hr_from, hr_to, factor_from, factor_to):
    """
    Interpolate the flux-rescaling factor linearly in log10(HR), used to
    walk the SED normalization smoothly across a state transition.
    """
    t = (np.log10(hr) - np.log10(hr_from)) / (np.log10(hr_to) - np.log10(hr_from))
    return factor_from + t * (factor_to - factor_from)


def build_q_path(logslope, logintercept,
                 factor_soft=1, factor_hard=1,
                 hr_soft=0.1, hr_hard=1.0,
                 ledd_lowbranch=0.01, ledd_highbranch=0.1,
                 ledd_tail=1e-5,
                 n_branch=150, n_transi=80,
                 hr_soft_disp=None, hr_hard_disp=None,
                 ledd_lowbranch_disp=None, ledd_highbranch_disp=None,
                 ledd_tail_disp=None):
    """
    Build a single continuous (HR, L/Ledd, NH) path tracing the classic
    XRB "q"-shaped hysteresis track.

    The physical NH calculation always uses the physical (undisplaced)
    HR and L/Ledd values.

    The *_disp arguments only control where the corresponding points are
    displayed. The hard branch is mapped piecewise so that BOTH
    ledd_lowbranch and ledd_highbranch remain exact junction points after
    the display displacement.
    """

    hr_soft_disp = hr_soft if hr_soft_disp is None else hr_soft_disp
    hr_hard_disp = hr_hard if hr_hard_disp is None else hr_hard_disp
    ledd_lowbranch_disp = (
        ledd_lowbranch if ledd_lowbranch_disp is None
        else ledd_lowbranch_disp
    )
    ledd_highbranch_disp = (
        ledd_highbranch if ledd_highbranch_disp is None
        else ledd_highbranch_disp
    )
    ledd_tail_disp = ledd_tail if ledd_tail_disp is None else ledd_tail_disp

    # ------------------------------------------------------------------
    # HARD BRANCH
    #
    # This is the important fix.
    #
    # The hard branch must pass exactly through:
    #
    #   physical ledd_lowbranch  -> display ledd_lowbranch_disp
    #
    # while still passing through:
    #
    #   physical ledd_highbranch -> display ledd_highbranch_disp
    #
    # Therefore the display mapping is piecewise in log(L).
    # ------------------------------------------------------------------

    # Allocate points according to the logarithmic lengths of the
    # two sections, while keeping an exact point at ledd_lowbranch.
    log_tail = np.log10(ledd_tail)
    log_low = np.log10(ledd_lowbranch)
    log_high = np.log10(ledd_highbranch)

    frac_low = (log_low - log_tail) / (log_high - log_tail)

    n_hard_low = max(2, int(round(n_branch * frac_low)))
    n_hard_high = max(2, n_branch - n_hard_low + 1)

    # Physical coordinates
    ledd_hard_low = np.logspace(
        np.log10(ledd_tail),
        np.log10(ledd_lowbranch),
        n_hard_low
    )

    ledd_hard_high = np.logspace(
        np.log10(ledd_lowbranch),
        np.log10(ledd_highbranch),
        n_hard_high
    )

    # Remove duplicate junction point from second section
    ledd_hard = np.concatenate([
        ledd_hard_low[:-1],
        ledd_hard_high
    ])

    # Display coordinates.
    #
    # First section:
    #   tail -> displaced low branch
    #
    ledd_hard_low_disp = np.logspace(
        np.log10(ledd_tail_disp),
        np.log10(ledd_lowbranch_disp),
        n_hard_low
    )

    # Second section:
    #   displaced low branch -> displaced high branch
    #
    ledd_hard_high_disp = np.logspace(
        np.log10(ledd_lowbranch_disp),
        np.log10(ledd_highbranch_disp),
        n_hard_high
    )

    ledd_hard_disp = np.concatenate([
        ledd_hard_low_disp[:-1],
        ledd_hard_high_disp
    ])

    hr_hard_branch_disp = np.full(len(ledd_hard), hr_hard_disp)

    # NH is ALWAYS evaluated using the physical coordinates.
    nh_hard = _nh_predict(
        ledd_hard,
        factor_hard,
        logslope,
        logintercept
    )

    # ------------------------------------------------------------------
    # TOP TRANSITION: hard -> soft
    # ------------------------------------------------------------------

    hr_top = np.logspace(
        np.log10(hr_hard),
        np.log10(hr_soft),
        n_transi
    )

    hr_top_disp = np.logspace(
        np.log10(hr_hard_disp),
        np.log10(hr_soft_disp),
        n_transi
    )

    factor_top = _log_interp_factor(
        hr_top,
        hr_hard,
        hr_soft,
        factor_hard,
        factor_soft
    )

    ledd_top_disp = np.full(
        n_transi,
        ledd_highbranch_disp
    )

    nh_top = _nh_predict(
        ledd_highbranch,
        factor_top,
        logslope,
        logintercept
    )

    # ------------------------------------------------------------------
    # SOFT BRANCH: top -> bottom
    # ------------------------------------------------------------------

    ledd_soft = np.logspace(
        np.log10(ledd_highbranch),
        np.log10(ledd_lowbranch),
        n_branch
    )

    ledd_soft_disp = np.logspace(
        np.log10(ledd_highbranch_disp),
        np.log10(ledd_lowbranch_disp),
        n_branch
    )

    hr_soft_branch_disp = np.full(
        n_branch,
        hr_soft_disp
    )

    nh_soft = _nh_predict(
        ledd_soft,
        factor_soft,
        logslope,
        logintercept
    )

    # ------------------------------------------------------------------
    # BOTTOM TRANSITION: soft -> hard
    # ------------------------------------------------------------------

    hr_bottom = np.logspace(
        np.log10(hr_soft),
        np.log10(hr_hard),
        n_transi
    )

    hr_bottom_disp = np.logspace(
        np.log10(hr_soft_disp),
        np.log10(hr_hard_disp),
        n_transi
    )

    factor_bottom = _log_interp_factor(
        hr_bottom,
        hr_soft,
        hr_hard,
        factor_soft,
        factor_hard
    )

    # Crucially, the bottom transition terminates exactly at the
    # displaced low-branch point of the hard branch.
    ledd_bottom_disp = np.full(
        n_transi,
        ledd_lowbranch_disp
    )

    nh_bottom = _nh_predict(
        ledd_lowbranch,
        factor_bottom,
        logslope,
        logintercept
    )

    # ------------------------------------------------------------------
    # Concatenate the complete Q
    # ------------------------------------------------------------------

    hr = np.concatenate([
        hr_hard_branch_disp,
        hr_top_disp,
        hr_soft_branch_disp,
        hr_bottom_disp
    ])

    ledd = np.concatenate([
        ledd_hard_disp,
        ledd_top_disp,
        ledd_soft_disp,
        ledd_bottom_disp
    ])

    nh = np.concatenate([
        nh_hard,
        nh_top,
        nh_soft,
        nh_bottom
    ])

    return hr, ledd, nh


def _fit_csv(val_csv, edd_conv):
    val = np.loadtxt(val_csv)
    flux, nh_3sig = val.T[0], val.T[-1]
    logslope, logintercept, r_value, p_value, std_err = linregress(
        np.log10(flux * edd_conv), np.log10(nh_3sig))
    return logslope, logintercept


def Q_shape_single(val_csv, factor_soft=1, factor_hard=1,
                    cmap='viridis_r', nh_lim=10,
                    d_kpc=8, M_CO=8,
                    ledd_lowbranch=0.01, ledd_highbranch=0.1,
                    lw=10, figsize=(6, 4), ax=None):
    """
    val_csv: csv with columns flux | 1/2/3-sigma max NH (1e22 cm^-2)
    nh_lim: NH values (1e22 cm^-2) above this are not drawn (the line is
        broken there via NaNs, rather than the points being silently
        dropped -- dropping points could draw a false straight segment
        across the gap).
    """

    assert factor_soft == 1 or factor_hard == 1, \
        'Error: only one of the soft/hard slopes can be rescaled'

    edd_conv = edd_factor(d_kpc, M_CO).value
    logslope, logintercept = _fit_csv(val_csv, edd_conv)

    hr, ledd, nh = build_q_path(logslope, logintercept,
                                 factor_soft=factor_soft, factor_hard=factor_hard,
                                 ledd_lowbranch=ledd_lowbranch,
                                 ledd_highbranch=ledd_highbranch)

    log_nh = np.log10(nh)

    # break rather than delete: points beyond nh_lim become NaN so the
    # colored line stops instead of jumping straight across the gap
    hr = np.where(log_nh >= np.log10(nh_lim), np.nan, hr)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_xscale('log')
    ax.set_xlabel(r'Spectral Hardness ([6-10]/[3-6] keV flux)')
    ax.set_xlim(0.05, 2)

    ax.set_yscale('log')
    ax.set_ylabel(r'Luminosity ($L/L_{Edd}$)')
    ax.set_ylim(1e-5, 1)

    valid = ~np.isnan(hr)
    vrange = None
    if np.any(valid):
        vrange = np.log10([nh[valid].min(), nh[valid].max()]).round(1)

    lines = colored_line(hr, ledd, log_nh, ax=ax, cmap=cmap, lw=lw, clim=vrange)

    cbar = fig.colorbar(lines, ax=ax, pad=0.04)
    cbar.ax.invert_yaxis()
    cbar.set_ticklabels(['%.1e' % v for v in 10 ** (22 + cbar.get_ticks())])
    cbar.set_label('NH', y=1.1, rotation=0, labelpad=-30)

    if own_fig:
        plt.tight_layout()

    return fig, ax

def Q_shape_mult(val_csvs, cmaps=('viridis_r', 'magma_r', 'cividis_r'),
                 labels=None,logxi_labels=None,
                 factor_soft=1, factor_hard=1,
                 nh_lims=10,
                 d_kpc=8, M_CO=8,
                 hr_soft=0.1, hr_hard=1.0,
                 ledd_lowbranch=0.01, ledd_highbranch=0.1,
                 ledd_tail=1e-5,
                 lws=10, ring_gap_pt=0.,
                 highlight_range_cmap=False,
                 figsize=(7, 5)):
    """
    Plot 2-3 Q-shapes as concentric rings using a common NH normalization.

    The colorbars are displayed as adjacent vertical strips on the
    right-hand side of the plot, with no gaps between them.

    All colorbars use the same NH normalization. Only the rightmost
    colorbar displays ticks and tick labels.

    Parameters
    ----------
    val_csvs : list
        CSV files, ordered from innermost to outermost ring.

    cmaps : tuple
        One matplotlib colormap name per ring.

    labels : list or None
        Optional labels for the individual colorbars. If None,
        no labels are displayed.

    nh_lims : scalar or list
        NH cutoff(s), in units of 1e22 cm^-2.

    lws : scalar or list
        Linewidth(s) of the rings, in points.

    ring_gap_pt : float
        Additional gap between neighbouring rings, in points.

    highlight_range_cmap : bool
        If True, add white horizontal dashed lines to each colorbar
        indicating the minimum and maximum NH values actually reached
        by the corresponding plotted ring.

    figsize : tuple
        Figure size.
    """

    n = len(val_csvs)

    assert n in (2, 3), 'Q_shape_mult takes 2 or 3 csvs'

    assert factor_soft == 1 or factor_hard == 1, \
        'Error: only one of the soft/hard slopes can be rescaled'

    cmaps = list(cmaps)[:n]

    lws = (
        [lws] * n
        if np.isscalar(lws)
        else list(lws)
    )

    nh_lims = (
        [nh_lims] * n
        if np.isscalar(nh_lims)
        else list(nh_lims)
    )

    # Keep labels=None as None so that no default "ring" labels
    # are generated.
    labels = (
        None
        if labels is None
        else list(labels)
    )

    edd_conv = edd_factor(d_kpc, M_CO).value

    # ==============================================================
    # FIT ALL RINGS AND DETERMINE COMMON NH NORMALIZATION
    # ==============================================================

    rings = []
    all_valid_log_nh = []

    #to test the interpolation uncomment this
    #plt.figure()
    # plt.xscale('log')
    # plt.yscale('log')
    # test_edd_range=np.logspace(-5, 1, 100)

    for csv, nh_lim in zip(val_csvs, nh_lims):

        logslope, logintercept = _fit_csv(
            csv,
            edd_conv
        )

        # to test the interpolation uncomment this
        # nh_hard = _nh_predict(test_edd_range,1, logslope,logintercept,)*1e22
        # plt.plot(np.loadtxt(csv).T[0]*edd_conv,np.loadtxt(csv).T[-1]*1e22,)
        # plt.plot(test_edd_range,nh_hard,ls='--')


        _, _, nh = build_q_path(
            logslope,
            logintercept,
            factor_soft=factor_soft,
            factor_hard=factor_hard,
            hr_soft=hr_soft,
            hr_hard=hr_hard,
            ledd_lowbranch=ledd_lowbranch,
            ledd_highbranch=ledd_highbranch,
            ledd_tail=ledd_tail
        )

        log_nh = np.log10(nh)

        valid = (
            log_nh <
            np.log10(nh_lim)
        )

        if np.any(valid):
            all_valid_log_nh.append(
                log_nh[valid]
            )

        rings.append(
            dict(
                logslope=logslope,
                logintercept=logintercept,
                nh_lim=nh_lim
            )
        )

    # Common color normalization across ALL rings.
    all_valid_log_nh = np.concatenate(
        all_valid_log_nh
    )

    vrange = np.log10([
        10 ** all_valid_log_nh.min(),
        10 ** all_valid_log_nh.max()
    ]).round(1)

    norm = plt.Normalize(
        vmin=vrange[0],
        vmax=vrange[1]
    )

    # ==============================================================
    # FIGURE AND MAIN AXES
    # ==============================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.set_xscale('log')
    ax.set_xlabel(
        r'Spectral Hardness ([6-10]/[3-6] keV flux)'
    )
    ax.set_xlim(0.05, 2)

    ax.set_yscale('log')
    ax.set_ylabel(
        r'Luminosity ($L/L_{Edd}$)'
    )
    ax.set_ylim(1e-5, 1)

    # ==============================================================
    # RESERVE SPACE FOR COLORBARS
    # ==============================================================

    # Total width of ALL colorbar strips = 5% of figure width.
    cbar_total_width = 0.075

    # Gap between main plot and first colorbar.
    cbar_pad = 0.012

    # Small margin for the tick labels of the rightmost colorbar.
    cbar_label_pad = 0.005

    pos = ax.get_position()

    new_width = (
        pos.width
        - cbar_pad
        - cbar_total_width
        - cbar_label_pad
    )

    if new_width <= 0:
        raise ValueError(
            "Figure is too narrow for the requested colorbar layout. "
            "Increase figsize[0]."
        )

    ax.set_position([
        pos.x0,
        pos.y0,
        new_width,
        pos.height
    ])

    # Get the final axes geometry after shrinking it.
    pos = ax.get_position()

    ax_left = pos.x0
    ax_bottom = pos.y0
    ax_width = pos.width
    ax_height = pos.height

    # ==============================================================
    # DRAW THE RINGS
    # ==============================================================

    # We need the final axes geometry before converting linewidths
    # from points to logarithmic coordinate offsets.

    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()

    bbox = ax.get_window_extent(
        renderer=renderer
    )

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    px_per_dex_x = (
        bbox.width /
        (np.log10(x1) - np.log10(x0))
    )

    px_per_dex_y = (
        bbox.height /
        (np.log10(y1) - np.log10(y0))
    )

    pt_to_px = fig.dpi / 72.

    # ==============================================================
    # CUMULATIVE RING OFFSETS
    # ==============================================================

    cum_pt = [0.] * n

    for i in range(1, n):

        cum_pt[i] = (
            cum_pt[i - 1]
            + lws[i - 1] / 2
            + lws[i] / 2
            + ring_gap_pt
        )

    # Store the actual NH ranges reached by each plotted ring.
    plotted_log_nh_ranges = []

    # ==============================================================
    # DRAW EACH Q-SHAPE
    # ==============================================================

    for i, (cmap, lw, ring) in enumerate(
        zip(cmaps, lws, rings)
    ):

        dex_x = (
            cum_pt[i] * pt_to_px
        ) / px_per_dex_x

        dex_y = (
            cum_pt[i] * pt_to_px
        ) / px_per_dex_y

        hr_disp, ledd_disp, nh = build_q_path(
            ring['logslope'],
            ring['logintercept'],

            factor_soft=factor_soft,
            factor_hard=factor_hard,

            hr_soft=hr_soft,
            hr_hard=hr_hard,

            ledd_lowbranch=ledd_lowbranch,
            ledd_highbranch=ledd_highbranch,
            ledd_tail=ledd_tail,

            # Display offsets
            hr_soft_disp=(
                hr_soft * 10 ** (-dex_x)
            ),

            hr_hard_disp=(
                hr_hard * 10 ** (dex_x)
            ),

            ledd_lowbranch_disp=(
                ledd_lowbranch * 10 ** (-dex_y)
            ),

            ledd_highbranch_disp=(
                ledd_highbranch * 10 ** (dex_y)
            ),

            ledd_tail_disp=ledd_tail
        )

        log_nh = np.log10(nh)

        # ----------------------------------------------------------
        # Determine the NH range actually visible in the plot.
        # ----------------------------------------------------------

        valid_plot = (
            np.isfinite(hr_disp)
            & np.isfinite(ledd_disp)
            & np.isfinite(log_nh)
            & (log_nh < np.log10(ring['nh_lim']))
        )

        if np.any(valid_plot):

            plotted_log_nh_ranges.append(
                (
                    np.nanmin(log_nh[valid_plot]),
                    np.nanmax(log_nh[valid_plot])
                )
            )

        else:

            plotted_log_nh_ranges.append(
                (np.nan, np.nan)
            )

        # ----------------------------------------------------------
        # Break the line when it exceeds the NH limit.
        # ----------------------------------------------------------

        hr_disp = np.where(
            log_nh >= np.log10(ring['nh_lim']),
            np.nan,
            hr_disp
        )

        colored_line(
            hr_disp,
            ledd_disp,
            log_nh,
            ax=ax,
            cmap=cmap,
            lw=lw,
            clim=vrange
        )

    # ==============================================================
    # MANUAL COLORBARS
    # ==============================================================

    # All colorbars together occupy exactly 5% of the figure width.
    cbar_width = cbar_total_width / n

    # No gap between neighbouring colorbars.
    cbar_gap = 0.0

    # Position of the first colorbar.
    cbar_x0 = (
        ax_left
        + ax_width
        + cbar_pad
    )

    # --------------------------------------------------------------
    # Create the individual colorbar strips.
    # --------------------------------------------------------------

    for i, cmap in enumerate(cmaps):

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        sm.set_array([])

        cbar_x = (
            cbar_x0
            + i * cbar_width
        )

        cax = fig.add_axes([
            cbar_x,
            ax_bottom,
            cbar_width,
            ax_height
        ])

        cbar = fig.colorbar(
            sm,
            cax=cax
        )

        # ----------------------------------------------------------
        # Optional logxi label above each colorbar
        # ----------------------------------------------------------

        logxi_labels = (
            None
            if logxi_labels is None
            else list(logxi_labels)
        )

        if logxi_labels is not None:
            assert len(logxi_labels) <= n, \
                'logxi_label can contain at most one value per colorbar'

        if logxi_labels is not None:

            cbar.ax.text(
                0.5,
                1.02,
                str(logxi_labels[i]),
                transform=cbar.ax.transAxes,
                ha='center',
                va='bottom',
                fontsize=9
            )

        if logxi_labels is not None:
            # Center logxi above the complete colorbar group
            cbar_center_x = (
                    cbar_x0 + cbar_total_width / 2
            )

            fig.text(
                cbar_center_x,
                ax_bottom + ax_height + 0.055,
                r'log$\xi$',
                ha='center',
                va='bottom',
                fontsize=10
            )

        # All colorbars have the same orientation.
        cbar.ax.invert_yaxis()

        # ----------------------------------------------------------
        # Highlight the actual NH range reached by this ring.
        # ----------------------------------------------------------

        if highlight_range_cmap:

            nh_min, nh_max = plotted_log_nh_ranges[i]

            if np.isfinite(nh_min):

                # Keep the markers within the common colorbar range.
                nh_min = np.clip(
                    nh_min,
                    vrange[0],
                    vrange[1]
                )

                nh_max = np.clip(
                    nh_max,
                    vrange[0],
                    vrange[1]
                )

                # Draw short white dashed horizontal markers.
                cbar.ax.axhline(
                    nh_min,
                    color='white',
                    linestyle='-',
                    linewidth=2.,
                    xmin=0.05,
                    xmax=0.95
                )

                cbar.ax.axhline(
                    nh_max,
                    color='white',
                    linestyle='-',
                    linewidth=2.,
                    xmin=0.05,
                    xmax=0.95
                )
                # --------------------------------------------------------------
                # Upper limit -> downward arrow
                # --------------------------------------------------------------

                tol = 1e-6
                arrow_x = 0.5
                arrow_length = 0.05 * (vrange[1] - vrange[0])

                cbar.ax.scatter(
                    0.5,
                    nh_max-0.08,
                    marker='^',
                    s=120,
                    color='white',
                    zorder=10,
                    clip_on=False
                )

                # --------------------------------------------------------------
                # Lower limit -> upward arrow
                # --------------------------------------------------------------

                cbar.ax.scatter(
                    0.5,
                    nh_min+0.08,
                    marker='v',
                    s=120,
                    color='white',
                    zorder=10,
                    clip_on=False
                )
        # ----------------------------------------------------------
        # Inner colorbars: no ticks.
        # ----------------------------------------------------------

        if i < n - 1:

            cbar.set_ticks([])

            cbar.ax.tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelright=False,
                length=0
            )

        # ----------------------------------------------------------
        # Rightmost colorbar: ticks on the right.
        # ----------------------------------------------------------

        else:

            cbar.ax.yaxis.set_ticks_position(
                'right'
            )

            ticks = cbar.get_ticks()

            cbar.set_ticklabels(
                [
                    '%.1e' % v
                    for v in 10 ** (22 + ticks)
                ]
            )

            # NH caption above the tick labels.
            cbar.ax.text(
                2.2,
                1.02,
                r'NH$_{\rm{det}}$',
                transform=cbar.ax.transAxes,
                ha='left',
                va='bottom',
                fontsize=10
            )

            # cbar.ax.set_title(
            #     r'$N_{\rm H}$',
            #     fontsize=10,
            #     pad=6
            # )

    # ==============================================================
    # OPTIONAL USER-PROVIDED COLORBAR LABELS
    # ==============================================================

    # If labels=None, nothing is added here.
    if labels is not None:

        for i, label in enumerate(labels[:n]):

            cbar = fig.axes[-n + i]

            cbar.ax.text(
                0.5,
                1.08,
                label,
                transform=cbar.ax.transAxes,
                ha='center',
                va='bottom',
                fontsize=9
            )

    # ==============================================================
    # RETURN
    # ==============================================================

    return fig, ax

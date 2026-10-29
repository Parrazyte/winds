"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-23.

Corner plots of posterior samples, with two drawing backends:
    'own'     matplotlib only (corner_plot)
    'corner'  the corner package (Foreman-Mackey 2016), wrapped by corner_plot_corner
Both get the same ranges, contour levels, quantiles, titles and decorations (hard limits,
best fit, decimal ticks on log axes, legend), so the two figures can be compared panel by panel.

Three levels:
    corner_plot / corner_plot_corner
                      general: one or several (n_samples, n_params) arrays, optional weights.
                      Lower triangle = 2D contours + density squares + points outside the
                      contours, diagonal = 1D histograms with quantiles, upper triangle = empty
                      or correlation coefficients.
    corner_chain      any chain written by chain_runner (FITS + <stem>.json sidecar): reads,
                      thins, transforms (log10 of the log-uniform parameters by default), draws
                      the hard limits and the best fit, can overlay consecutive parts of the
                      chain as a visual convergence check, or make a 'paper' figure without the
                      diagnostics text. Saves png and pdf.
    corner_maxij1543  the MAXI J1543-564 pion model (TBabs x 3 pion absorbers x (powerlaw +
                      diskbb) + pion emission): physical labels, cz in km/s, covering fraction,
                      parameter sets per absorber.

2D contour levels are the 2D-Gaussian equivalents of 1, 2, 3 sigma: the regions holding
1-exp(-s^2/2) = 39.3%, 86.5%, 98.9% of the samples (not 68/95/99.7%, which are the 1D values).
The corner package defaults to 0.5, 1, 1.5, 2 sigma: the levels are always passed explicitly.

Log-sampled parameters are plotted in log10 but, by default, labelled with round linear values
(0.1 / 0.3 / 1, 100 / 300 / 1000) and titled with linear values (quantiles are invariant under
the log, so the median and the 16/84% values are just 10**).
"""

import os
import re
import json
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

from chain_diagnostics import read_chain

#mass inside the 2D contours: 1, 2, 3 sigma of a 2D Gaussian
SIGMA_LEVELS_2D = tuple(1. - np.exp(-0.5 * np.array([1., 2., 3.]) ** 2))

DEFAULT_COLORS = ['#1f4e79', '#c0392b', '#2e8b57', '#8e44ad', '#d68910']

MARKER_STYLES = [dict(color='black', ls='-'), dict(color='#d68910', ls='-.'), dict(color='#7f8c8d', ls=':')]

#grid nodes of interpolated table parameters (option grid): thin lines over the contours, so that
#modes or valleys sitting on the nodes stand out (Lucchini et al. 2026: linear interpolation between
#the nodes can create a valley or a peak in the posterior at each node)
GRID_STYLE = dict(color='grey', ls=':', lw=0.9, alpha=0.8, zorder=4)

#hard limits (option limits): solid, so that they differ from the dotted grid nodes; the region
#beyond them is hatched
LIMIT_STYLE = dict(color='grey', ls='-', lw=1)

C_KMS = 299792.458


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------

def weighted_quantile(x, q, weights=None):
    '''
    Quantiles q (scalar or array in [0, 1]) of the samples x, with optional weights (nest output).
    '''

    x = np.asarray(x, dtype=float)
    if weights is None:
        return np.quantile(x, q)

    order = np.argsort(x)
    cum_w = np.cumsum(np.asarray(weights, dtype=float)[order])
    cum_w /= cum_w[-1]
    return np.interp(q, cum_w, x[order])


def _weighted_corr(x, y, weights=None):
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    return cov / (sx * sy) if sx > 0 and sy > 0 else np.nan


def _fmt_title(med, err_lo, err_hi):
    '''
    "med^{+hi}_{-lo}" rounded to 2 significant digits of the smaller error. A median below 0.01 or
    from 10^4 is written with a common factor: (8.85^{+0.57}_{-0.56}) x 10^-3, not 0.00885^{+0.00057}...
    '''

    k = 0
    if np.isfinite(med) and med != 0 and not 0.01 <= abs(med) < 1e4:
        k = int(np.floor(np.log10(abs(med))))
        med, err_lo, err_hi = med / 10. ** k, err_lo / 10. ** k, err_hi / 10. ** k

    err = min(err_lo, err_hi)
    if not np.isfinite(err) or err <= 0:
        core = '%.3g' % med
    else:
        n_dec = max(int(1 - np.floor(np.log10(err))), 0)
        fmt = '%.' + str(n_dec) + 'f'
        core = '%s^{+%s}_{-%s}' % (fmt % med, fmt % err_hi, fmt % err_lo)
    return r'$(%s)\times10^{%d}$' % (core, k) if k else '$%s$' % core


def _fmt_tick(val):
    '''
    Tick label of a round linear value: 0.003, 0.3, 300, 3000 as decimals, beyond as m x 10^k
    '''

    if 1e-3 <= abs(val) < 1e4:
        return '%g' % float('%.3g' % val)
    k = int(np.floor(np.log10(val)))
    mant = float('%.2g' % (val / 10 ** k))
    return r'$10^{%d}$' % k if mant == 1 else r'$%g\times10^{%d}$' % (mant, k)


def decimal_log_ticks(lo, hi, max_ticks=4):
    '''
    Ticks of a log10 axis spanning [lo, hi], placed at round LINEAR values: the sparsest of the
    sequences 1 x 10^k, (1, 3) x 10^k, (1, 2, 5) x 10^k giving at least 3 ticks (2 above 2
    decades), else the one giving the most (at least 2), else round linear values (1.8, 1.9, ...)
    for a narrow axis. Ticks within 3% of the edges are dropped (they would overlap the
    neighbouring panel's labels).

    Returns (positions in log10, labels).
    '''

    edge = 0.03 * (hi - lo)

    def inside(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[vals > 0]
        logs = np.log10(vals)
        return vals[(logs >= lo + edge) & (logs <= hi - edge)]

    decades = np.arange(np.floor(lo) - 1, np.ceil(hi) + 1)
    #3 ticks wanted below 2 decades (2 ticks there leave most of the axis unlabelled)
    needed = 3 if hi - lo < 2 else 2
    vals, best = np.array([]), np.array([])
    for mantissas in ([1.], [1., 3.], [1., 2., 5.]):
        vals = inside(sorted(m * 10. ** d for d in decades for m in mantissas))
        if len(vals) >= needed:
            break
        if len(vals) > len(best):
            best = vals
    if len(vals) < needed:
        vals = best
    if len(vals) < 2:
        vals = inside(MaxNLocator(4, steps=[1, 2, 5, 10]).tick_values(10. ** lo, 10. ** hi))
    if len(vals) > max_ticks:
        vals = vals[::int(np.ceil(len(vals) / max_ticks))]
    return np.log10(vals), [_fmt_tick(elem) for elem in vals]


def _contour_thresholds(hist, masses):
    '''
    Density thresholds of the highest-density regions holding the fractions masses of hist.
    Returned in increasing order (= decreasing mass), strictly increasing as contourf requires.
    '''

    flat = np.sort(hist.ravel())[::-1]
    cum = np.cumsum(flat)
    if cum[-1] <= 0:
        return None
    cum /= cum[-1]
    thr = [flat[min(np.searchsorted(cum, elem), len(flat) - 1)] for elem in masses]
    thr = np.sort(thr)
    #a very peaked histogram can give equal thresholds: nudge them apart
    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1]:
            thr[i] = thr[i - 1] * (1 + 1e-6) + 1e-300
    return thr


def _auto_range(datasets, i_par, limits, pad=0.04):
    '''
    Range of parameter i_par: extent of all datasets, padded, and widened to show a hard limit
    lying just outside it (a pile-up against a prior edge is then visible as such).
    '''

    lo = min(np.min(elem[:, i_par]) for elem in datasets)
    hi = max(np.max(elem[:, i_par]) for elem in datasets)
    if hi <= lo:
        width = abs(lo) * 0.01 if lo != 0 else 1.
        return lo - width, hi + width

    width = hi - lo
    lo, hi = lo - pad * width, hi + pad * width
    if limits is not None and limits[i_par] is not None:
        lim_lo, lim_hi = limits[i_par]
        if lim_lo is not None and np.isfinite(lim_lo) and lo - 0.25 * width <= lim_lo <= lo + pad * width:
            lo = min(lo, lim_lo - pad * width)
        if lim_hi is not None and np.isfinite(lim_hi) and hi - pad * width <= lim_hi <= hi + 0.25 * width:
            hi = max(hi, lim_hi + pad * width)
    return lo, hi


def _prepare(samples, weights, colors, ranges, limits, summary_samples, summary_weights):
    '''
    Common input parsing of both backends: list of datasets, weights, colors, ranges and the
    samples used for the quantiles/titles/correlations.
    '''

    datasets = [np.asarray(samples, dtype=float)] if not isinstance(samples, (list, tuple)) \
        else [np.asarray(elem, dtype=float) for elem in samples]
    n_par = datasets[0].shape[1]
    if any(elem.ndim != 2 or elem.shape[1] != n_par for elem in datasets):
        raise ValueError('every dataset must be a (n_samples, %d) array' % n_par)

    if weights is None or not isinstance(weights, (list, tuple)):
        weights = [weights] + [None] * (len(datasets) - 1)
    colors = colors or DEFAULT_COLORS[:len(datasets)]
    rng = [(ranges[i] if ranges is not None and ranges[i] is not None
            else _auto_range(datasets, i, limits)) for i in range(n_par)]

    #histogram range: the axis range cut at the hard limits. The histograms (and so the smoothing,
    #contours and density squares) stop at the prior edge instead of leaking into the hatched
    #region; the smoothing reflects at the edge (gaussian_filter mode 'reflect'), i.e. treats it as
    #a wall, which is what a hard prior limit is
    hist_rng = []
    for i, (lo, hi) in enumerate(rng):
        if limits is not None and limits[i] is not None:
            lim_lo, lim_hi = limits[i]
            if lim_lo is not None and np.isfinite(lim_lo):
                lo = max(lo, lim_lo)
            if lim_hi is not None and np.isfinite(lim_hi):
                hi = min(hi, lim_hi)
        hist_rng.append((lo, hi))

    if summary_samples is None:
        summary_samples, summary_weights = datasets[0], weights[0]
    return datasets, weights, colors, rng, hist_rng, np.asarray(summary_samples, dtype=float), summary_weights


def _layout(fig, n_par, title_lines):
    '''
    Margins in inches (tick labels + axis label on the left/bottom, diagonal titles on top)
    '''

    fig_w, fig_h = fig.get_size_inches()
    fig.subplots_adjust(left=1.1 / fig_w, bottom=1.1 / fig_h, right=1 - 0.15 / fig_w,
                        top=1 - (0.25 + 0.2 * title_lines) / fig_h, wspace=0.06, hspace=0.06)


def _decorate(fig, axes, rng, labels, titles, colors, n_set, summary_samples, summary_weights,
              quantiles, show_titles, title_lines, markers, limits, log_flags, show_corr, names,
              suptitle, label_size, tick_size, title_size, grid=None):
    '''
    Everything drawn identically by both backends, after the histograms and contours:
    y range of the diagonal (both backends normalize the histograms to unit area), quantile
    lines and titles on the diagonal, hard limits, table grid nodes, markers, correlation
    coefficients, decimal ticks on the log axes, legend and description text.
    '''

    n_par = len(rng)
    markers = markers or {}
    log_flags = log_flags or [False] * n_par

    for i_row in range(n_par):
        for i_col in range(i_row + 1):
            ax = axes[i_row, i_col]

            # diagonal: same y range in both backends, 1.1 x the highest histogram of all datasets
            #(the histograms are the only patches at this point; corner would otherwise keep the
            #range of its LAST call, which can clip an earlier, taller dataset)
            if i_col == i_row:
                heights = [elem.get_path().vertices[:, 1].max() for elem in ax.patches]
                if heights:
                    ax.set_ylim(0, 1.1 * max(heights))

            # diagonal: quantiles and title (from the summary samples)
            if i_col == i_row and quantiles:
                q_vals = weighted_quantile(summary_samples[:, i_col], quantiles, summary_weights)
                for elem in q_vals:
                    ax.axvline(elem, color=colors[0] if n_set == 1 else 'black', ls='--', lw=0.8)
                if show_titles and len(quantiles) >= 3:
                    lin = 10. ** np.asarray(q_vals) if log_flags[i_col] else np.asarray(q_vals)
                    med, q_lo, q_hi = lin[len(lin) // 2], lin[0], lin[-1]
                    ax.set_title(titles[i_col] + (' = ' if title_lines == 1 else '\n')
                                 + _fmt_title(med, med - q_lo, q_hi - med), fontsize=title_size)

            # hard limits: solid line (LIMIT_STYLE), region beyond hatched
            dims = [(i_col, ax.axvline, ax.axvspan)]
            if i_col != i_row:
                dims.append((i_row, ax.axhline, ax.axhspan))
            for i_par, axline, axspan in dims:
                if limits is None or limits[i_par] is None:
                    continue
                p_rng = rng[i_par]
                for i_side, lim in enumerate(limits[i_par]):
                    if lim is None or not np.isfinite(lim) or not p_rng[0] <= lim <= p_rng[1]:
                        continue
                    axline(lim, **LIMIT_STYLE)
                    axspan(*((p_rng[0], lim) if i_side == 0 else (lim, p_rng[1])),
                           facecolor='none', edgecolor='0.75', hatch='////', lw=0)

            # table grid nodes: a line at each node inside the panel
            for i_par, axline, _ in dims:
                if grid is None or grid[i_par] is None:
                    continue
                for val in grid[i_par]:
                    if rng[i_par][0] < val < rng[i_par][1]:
                        axline(val, **GRID_STYLE)

            # markers (best fit, ...)
            for i_mark, (m_name, m_vals) in enumerate(markers.items()):
                style = MARKER_STYLES[i_mark % len(MARKER_STYLES)]
                if np.isfinite(m_vals[i_col]):
                    ax.axvline(m_vals[i_col], lw=0.9, **style)
                if i_col != i_row and np.isfinite(m_vals[i_row]):
                    ax.axhline(m_vals[i_row], lw=0.9, **style)
                    if np.isfinite(m_vals[i_col]):
                        ax.plot(m_vals[i_col], m_vals[i_row], 's', ms=3, color=style['color'])

    # upper triangle: correlation coefficients
    if show_corr:
        for i_row in range(n_par):
            for i_col in range(i_row + 1, n_par):
                rho = _weighted_corr(summary_samples[:, i_col], summary_samples[:, i_row], summary_weights)
                if np.isfinite(rho):
                    axes[i_row, i_col].text(0.5, 0.5, '%.2f' % rho, transform=axes[i_row, i_col].transAxes,
                                            ha='center', va='center', fontsize=title_size + 2 * abs(rho),
                                            color=plt.cm.RdBu_r(0.5 + 0.5 * rho) if abs(rho) > 0.2 else '0.5')

    # decimal ticks on the log axes: ticks on every panel, labels on the bottom row / left column
    for i_par in range(n_par):
        if not log_flags[i_par]:
            continue
        pos, tick_labels = decimal_log_ticks(*rng[i_par])
        for i_row in range(i_par, n_par):
            ax = axes[i_row, i_par]
            ax.set_xticks(pos)
            ax.set_xticklabels(tick_labels if i_row == n_par - 1 else [], rotation=45, fontsize=tick_size)
            ax.set_xlim(rng[i_par])
        for i_col in range(i_par):
            ax = axes[i_par, i_col]
            ax.set_yticks(pos)
            ax.set_yticklabels(tick_labels if i_col == 0 else [], rotation=45, fontsize=tick_size)
            ax.set_ylim(rng[i_par])

    fig.align_ylabels(axes[1:, 0])
    fig.align_xlabels(axes[-1, :])

    # legend and description in the empty upper-right triangle, top-right corner of the last
    # column; with show_corr the triangle is used: they go above the plot
    handles = []
    if names is not None:
        handles += [Line2D([], [], color=colors[i], lw=1.5, label=names[i]) for i in range(n_set)]
    for i_mark, m_name in enumerate(markers):
        style = MARKER_STYLES[i_mark % len(MARKER_STYLES)]
        handles.append(Line2D([], [], marker='s', ms=4, lw=0.9, label=m_name, **style))
    if limits is not None and any(elem is not None for elem in limits):
        handles.append(Line2D([], [], label='hard limit (prior edge)', **LIMIT_STYLE))
    if grid is not None and any(elem is not None for elem in grid):
        handles.append(Line2D([], [], label='table grid node', **GRID_STYLE))

    fig_h = fig.get_size_inches()[1]
    top_right = axes[0, -1].get_position()
    x_anchor, y_anchor = top_right.x1, top_right.y1
    if show_corr:
        y_anchor = 1 - 0.05 / fig_h
    text_top = y_anchor
    if handles:
        leg = fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(x_anchor, y_anchor),
                         fontsize=label_size, frameon=False, borderaxespad=0)
        fig.canvas.draw()
        text_top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.1 / fig_h
    if suptitle:
        fig.text(x_anchor, text_top, suptitle, ha='right', va='top', fontsize=title_size, color='0.3')


# ---------------------------------------------------------------------------------------------
# backend 'own': matplotlib only
# ---------------------------------------------------------------------------------------------

def corner_plot(samples, labels=None, titles=None, weights=None, names=None, colors=None, ranges=None,
                bins=30, bins_1d=None, smooth=1., levels=SIGMA_LEVELS_2D,
                quantiles=(0.16, 0.5, 0.84), show_titles=True, fill=False, density=True,
                plot_points=True, max_points=1000, rasterize=False, markers=None, limits=None,
                log_flags=None, show_corr=False, summary_samples=None, summary_weights=None,
                title_lines=None, panel_size=1.6, label_size=11, tick_size=8, title_size=9,
                suptitle=None, grid=None, fig=None):
    '''
    Corner plot of one or several sets of posterior samples, matplotlib only.

    samples:     (n_samples, n_params) array, or a list of such arrays (overlays, same parameters)
    labels:      axis label per parameter; titles: name in the diagonal titles (default = labels)
    weights:     None, a (n_samples,) array, or a list (one per dataset, None allowed)
    names:       legend name per dataset (no legend entry if None)
    colors:      color per dataset (DEFAULT_COLORS)
    ranges:      list of (lo, hi) or None per parameter; None = auto (extent of the samples,
                 widened to a hard limit lying just outside it)
    bins:        bins per axis of the 2D histograms (= size of the density squares); bins_1d for
                 the diagonal (default = bins)
    smooth:      gaussian smoothing of the 2D histograms, in bins (0 or None: no smoothing)
    levels:      2D contour masses (SIGMA_LEVELS_2D: 1, 2, 3 sigma of a 2D Gaussian)
    quantiles:   vertical dashed lines on the diagonal, and the title values (median and the two
                 outer quantiles) if show_titles
    fill:        filled contours (bool or one per dataset), instead of the density squares
    density:     density squares inside the outermost contour, in the color of each dataset
                 (semi-transparent when there are several datasets)
    plot_points: points outside the outermost contour (selected among all the samples), at most
                 max_points per dataset and panel
    rasterize:   rasterize the points and squares. Off by default: in a pdf, each rasterized artist
                 costs a full-figure image in memory until the file is written (matplotlib keeps
                 the cropped view of it), i.e. ~13 GB for 17 parameters; the vector version is small
    markers:    {name: array of n_params values (nan to skip)}, e.g. the best fit
    limits:      list of (lo, hi) or None per parameter: hard limits, solid grey line + hatching beyond
    log_flags:   one bool per parameter: the samples are log10 values, ticks and titles are given
                 as linear values (decimal_log_ticks)
    show_corr:   weighted Pearson correlation of each pair in the upper triangle
    summary_samples, summary_weights: samples used for the quantiles, titles and correlations
                 instead of the first dataset (e.g. the whole chain when the datasets are parts of it)
    title_lines: 1 = "title = value" above each diagonal panel, 2 = on two lines
                 (default: 2 above 8 parameters, where one line overflows on the next panel)
    suptitle:    text written in the empty upper-right triangle, under the legend
    grid:        list of node values (in plotted coordinates) or None per parameter: the grid of an
                 interpolated table model, drawn as thin lines on the histograms and 2D panels
                 (GRID_STYLE), to see whether modes or valleys of the posterior sit on the nodes
    fig:         existing empty figure to draw in (default: n_params*panel_size + 1.4 inches)

    Returns (fig, axes), axes of shape (n_params, n_params).
    '''

    datasets, weights, colors, rng, hist_rng, summary_samples, summary_weights = \
        _prepare(samples, weights, colors, ranges, limits, summary_samples, summary_weights)
    n_set, n_par = len(datasets), datasets[0].shape[1]
    labels = labels or ['p%d' % i for i in range(n_par)]
    titles = titles or labels
    bins_1d = bins_1d or bins
    title_lines = title_lines or (2 if n_par > 8 else 1)
    if not isinstance(fill, (list, tuple)):
        fill = [fill] * n_set
    alpha_max = 0.85 if n_set == 1 else 0.5

    if fig is None:
        size = n_par * panel_size + 1.4
        fig = plt.figure(figsize=(size, size))
    axes = fig.subplots(n_par, n_par, squeeze=False)
    _layout(fig, n_par, title_lines)

    for i_row in range(n_par):
        for i_col in range(n_par):
            ax = axes[i_row, i_col]
            if i_col > i_row:
                ax.set_axis_off()
                continue
            x_rng = rng[i_col]

            # diagonal: 1D histograms
            if i_col == i_row:
                for i_set, (data, w) in enumerate(zip(datasets, weights)):
                    ax.hist(data[:, i_col], bins=bins_1d, range=hist_rng[i_col], weights=w, density=True,
                            histtype='step', color=colors[i_set], lw=1.2)
                ax.set_yticks([])
                ax.set_ylim(bottom=0)

            # lower triangle: points outside, density squares inside, contour lines
            else:
                y_rng = rng[i_row]
                for i_set, (data, w) in enumerate(zip(datasets, weights)):
                    x, y = data[:, i_col], data[:, i_row]
                    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=w,
                                                            range=[hist_rng[i_col], hist_rng[i_row]])
                    if smooth:
                        hist = gaussian_filter(hist, smooth)
                    thr = _contour_thresholds(hist, levels)
                    if thr is None:
                        continue
                    #contour grid: bin centres, extended to the panel edges with the edge values
                    #(as corner does), so that the contours reach the edges, e.g. a pile-up against
                    #a hard limit is closed by the panel edge and not cut half a bin before it
                    x_g = np.concatenate([[x_edges[0]], 0.5 * (x_edges[1:] + x_edges[:-1]), [x_edges[-1]]])
                    y_g = np.concatenate([[y_edges[0]], 0.5 * (y_edges[1:] + y_edges[:-1]), [y_edges[-1]]])
                    hist_g = np.pad(hist, 1, mode='edge')

                    if plot_points:
                        #points outside the outermost contour, tested against the filled region of
                        #that contour computed on the same grid (holes included): a test on the
                        #histogram bin instead leaves points inside the drawn line, which cuts
                        #through the bins
                        region = ax.contourf(x_g, y_g, hist_g.T, levels=[thr[0], hist.max() * (1 + 1e-6)])
                        inside = region.get_paths()[0].contains_points(np.column_stack([x, y])) \
                            if len(region.get_paths()) else np.zeros(len(x), bool)
                        region.remove()
                        outside = np.where(~inside)[0]
                        outside = outside[::max(1, int(np.ceil(len(outside) / max_points)))]
                        ax.scatter(x[outside], y[outside], s=1.5, color=colors[i_set], alpha=0.35,
                                   lw=0, rasterized=rasterize, zorder=1)
                    if density and not fill[i_set]:
                        cmap = LinearSegmentedColormap.from_list(
                            'density', [to_rgba(colors[i_set], 0.), to_rgba(colors[i_set], alpha_max)])
                        ax.pcolormesh(x_edges, y_edges, np.ma.masked_less(hist.T, thr[0]), cmap=cmap,
                                      vmin=0, vmax=hist.max(), shading='flat', lw=0, rasterized=rasterize,
                                      zorder=2)
                    if fill[i_set]:
                        #outer (largest mass) level = lightest
                        shades = [to_rgba(colors[i_set], alpha) for alpha in np.linspace(0.2, 0.75, len(thr))]
                        ax.contourf(x_g, y_g, hist_g.T, levels=list(thr) + [hist.max() * (1 + 1e-6)],
                                    colors=shades, zorder=2)
                    ax.contour(x_g, y_g, hist_g.T, levels=thr, colors=[colors[i_set]],
                               linewidths=np.linspace(0.6, 1.2, len(thr)), zorder=3)
                ax.set_ylim(y_rng)

            ax.set_xlim(x_rng)

            # ticks and labels: bottom row and left column only
            ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
            ax.tick_params(labelsize=tick_size, direction='in', top=True, right=i_col != i_row)
            if i_row == n_par - 1:
                ax.set_xlabel(labels[i_col], fontsize=label_size)
                for elem in ax.get_xticklabels():
                    elem.set_rotation(45)
            else:
                ax.set_xticklabels([])
            if i_col != i_row:
                ax.yaxis.set_major_locator(MaxNLocator(3, prune='both'))
                if i_col == 0:
                    ax.set_ylabel(labels[i_row], fontsize=label_size)
                    for elem in ax.get_yticklabels():
                        elem.set_rotation(45)
                else:
                    ax.set_yticklabels([])

    _decorate(fig, axes, rng, labels, titles, colors, n_set, summary_samples, summary_weights,
              quantiles, show_titles, title_lines, markers, limits, log_flags, show_corr, names,
              suptitle, label_size, tick_size, title_size, grid=grid)
    return fig, axes


# ---------------------------------------------------------------------------------------------
# backend 'corner': the corner package (written for corner 2.3.0)
# ---------------------------------------------------------------------------------------------

def corner_plot_corner(samples, labels=None, titles=None, weights=None, names=None, colors=None,
                       ranges=None, bins=30, bins_1d=None, smooth=1., levels=SIGMA_LEVELS_2D,
                       quantiles=(0.16, 0.5, 0.84), show_titles=True, fill=False, density=True,
                       plot_points=True, max_points=None, rasterize=None, markers=None, limits=None,
                       log_flags=None, show_corr=False, summary_samples=None, summary_weights=None,
                       title_lines=None, panel_size=1.6, label_size=11, tick_size=8, title_size=9,
                       suptitle=None, grid=None, fig=None):
    '''
    Same as corner_plot (same arguments and outputs), drawn by corner.corner: one call per dataset
    on the same figure, then the shared decorations. The ranges, levels and smoothing are passed
    explicitly, so the contours are comparable with the 'own' backend.

    Differences inherent to corner:
        -points: corner draws every sample, always rasterized, and hides the ones inside the
         outermost contour under a fill of the background color; max_points and rasterize are
         ignored. A pdf of a large figure then needs a lot of memory (see pdf_raster_memory):
         corner_chain skips the pdf when it would exceed its pdf_max_gb
        -density squares: drawn as vector (pcolor), not rasterized
        -several datasets: that background fill would hide the previous dataset, so it is turned
         off, the points are not drawn, and the density colormap is made half-transparent
        -titles and quantile lines are drawn by _decorate, not by corner (corner would compute
         them per dataset and in log10)
    '''

    import corner

    datasets, weights, colors, rng, hist_rng, summary_samples, summary_weights = \
        _prepare(samples, weights, colors, ranges, limits, summary_samples, summary_weights)
    n_set, n_par = len(datasets), datasets[0].shape[1]
    labels = labels or ['p%d' % i for i in range(n_par)]
    titles = titles or labels
    title_lines = title_lines or (2 if n_par > 8 else 1)
    several = n_set > 1

    if fig is None:
        size = n_par * panel_size + 1.4
        fig = plt.figure(figsize=(size, size))

    for i_set, (data, w) in enumerate(zip(datasets, weights)):
        n_before = [len(ax.collections) for ax in fig.axes]
        #fresh kwargs dicts at each call: corner writes the color into them
        corner.corner(data, bins=bins, range=list(hist_rng), weights=w, color=colors[i_set], smooth=smooth,
                      hist_bin_factor=(bins_1d or bins) / bins, levels=levels, labels=labels,
                      label_kwargs={'fontsize': label_size}, show_titles=False, quantiles=[],
                      fig=fig, max_n_ticks=3,
                      plot_datapoints=plot_points and not several, plot_density=density and not fill,
                      plot_contours=True, fill_contours=fill, no_fill_contours=several,
                      #corner's default alpha 0.1 makes the ~1% of points beyond 3 sigma invisible
                      data_kwargs={'ms': 2., 'alpha': 0.3},
                      hist_kwargs={'density': True, 'lw': 1.2},
                      contour_kwargs={'linewidths': 1.})
        #overlays: corner's density colormap goes from opaque color (peak) to transparent; an alpha
        #kwarg would REPLACE that per-square transparency (and veil the whole panel), so the colormap
        #of the new pcolor is swapped for a half-transparent version instead
        if several and density and not fill:
            semi = LinearSegmentedColormap.from_list('density_semi', [to_rgba(colors[i_set], 0.5),
                                                                      to_rgba('white', 0.)])
            for ax, n_old in zip(fig.axes, n_before or [0] * len(fig.axes)):
                for coll in ax.collections[n_old:]:
                    if isinstance(coll, PolyCollection):
                        coll.set_cmap(semi)

    axes = np.array(fig.axes).reshape(n_par, n_par)

    #corner drew everything over the histogram range (cut at the hard limits), but extends its
    #contour grid 2 bins beyond it: clip the 2D artists to the allowed rectangle, then give the
    #axes back the full range, where the hatching shows the excluded region
    for i_row in range(n_par):
        for i_col in range(i_row + 1):
            ax = axes[i_row, i_col]
            if i_col != i_row:
                (x_lo, x_hi), (y_lo, y_hi) = hist_rng[i_col], hist_rng[i_row]
                for elem in ax.collections + ax.lines:
                    elem.set_clip_path(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                                 transform=ax.transData))
                ax.set_ylim(rng[i_row])
            ax.set_xlim(rng[i_col])

    _layout(fig, n_par, title_lines)
    for ax in axes.ravel():
        ax.tick_params(labelsize=tick_size, direction='in')

    _decorate(fig, axes, rng, labels, titles, colors, n_set, summary_samples, summary_weights,
              quantiles, show_titles, title_lines, markers, limits, log_flags, show_corr, names,
              suptitle, label_size, tick_size, title_size, grid=grid)
    return fig, axes


BACKENDS = {'own': corner_plot, 'corner': corner_plot_corner}


def pdf_raster_memory(fig, dpi):
    '''
    Estimated memory (bytes) needed to write fig as a pdf: matplotlib draws each rasterized artist
    on a full-figure RGBA image at dpi and keeps it until the file is written (the pdf holds a
    cropped VIEW of it). Measured: 12 rasterized artists on a 10.2-inch figure at 120 dpi
    added 70 MB, the estimate gives 72 MB.
    '''

    n_raster = len(fig.findobj(lambda elem: elem.get_rasterized()))
    width, height = fig.get_size_inches() * dpi
    return n_raster * width * height * 4


# ---------------------------------------------------------------------------------------------
# XSPEC chains written by chain_runner
# ---------------------------------------------------------------------------------------------

def find_sidecar(chain_file):
    '''
    <stem>.json of a chain file, also for the backups of continue_chain (<stem>.fits.bak_cont<k>).
    Returns None if it does not exist.
    '''

    base = re.sub(r'\.bak_cont\d+$', '', chain_file)
    stem = base[:-5] if base.endswith('.fits') else os.path.splitext(base)[0]
    path = stem + '.json'
    return path if os.path.exists(path) else None


def table_grid_values(table_file):
    '''
    Grid nodes of the interpolated parameters of an XSPEC table model file (OGIP PARAMETERS
    extension): {parameter name: node values}. Names have their spaces replaced by '_', as XSPEC
    does ('v_rms [km/s]' -> 'v_rms_[km/s]'), so they match the sidecar names. The additional
    parameters and the redshift and norm that XSPEC adds are not in the grid.
    '''

    from astropy.io import fits
    with fits.open(table_file) as hdul:
        par = hdul['PARAMETERS']
        n_int = par.header['NINTPARM']
        return {str(row['NAME']).strip().replace(' ', '_'): np.array(row['VALUE'][:row['NUMBVALS']], dtype=float)
                for row in par.data[:n_int]}


def uniform_prior_weights(samples, keys, table, uniform_prior_keys):
    '''
    Importance weights replacing the jeffreys (log-uniform) prior of some parameters by a uniform
    prior between the same hard limits, without a new chain (same method as
    comp_ratio_plot_MC.load_draws).

    The chain samples follow likelihood x old prior; weighting them by new prior / old prior gives
    likelihood x new prior. jeffreys has density 1/x and uniform a constant, so the weight of a
    sample is x for each reweighted parameter (the product over several). Only the priors enter
    the weights, never the likelihood.

    samples:  (..., n_params) array, any leading shape (e.g. walker, step)
    keys:     parameter key of each column (read_chain keys)
    table:    {str(key): sidecar parameter row}, to check that each parameter had a jeffreys prior
    uniform_prior_keys: keys (model parameter indices) to switch to a uniform prior

    Returns (probabilities with the leading shape of samples, summing to 1; Kish effective sample
    size fraction (sum w)^2 / sum w^2 / n). A small fraction (a few % or less) means the old
    samples do not cover the region the new prior favours: that prior needs its own chain.
    '''

    keys = [str(elem) for elem in keys]
    log_w = 0.
    for key in uniform_prior_keys:
        key = str(key)
        if key not in keys:
            raise ValueError('parameter %s is not a free parameter of the chain' % key)
        prior = table.get(key, {}).get('prior')
        if not prior or prior[0] != 'jeffreys':
            raise ValueError('parameter %s was sampled with prior %s, not jeffreys' % (key, prior))
        #in log to avoid overflows with several parameters
        log_w = log_w + np.log(samples[..., keys.index(key)])
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    return weights, 1. / np.sum(weights ** 2) / weights.size


def gaussian_equivalent_widths(lines, continuum=('powerlaw', 'diskbb'), model='', n_nodes=200, de=5e-3):
    '''
    Equivalent widths of gaussian lines, for corner_chain(derived=...): a function
    f(params, table) -> {norm key: (EW in eV, (lo, hi) limits)}.

    EW = line norm (photons/cm2/s) / continuum photon flux density (photons/cm2/s/keV) at the
    observed line centre E0 (1 - v/c) (XSPEC vashift: positive v = redshift), same sign as the norm
    (negative for an absorption line, as XSPEC eqwidth). The continuum is the sum of the additive
    components named in continuum, without any line; a multiplicative component applied to both the
    lines and the continuum (TBabs) cancels in the ratio. XSPEC eqwidth differs: its continuum is the
    model minus the one line, so it includes the neighbouring lines.

    lines:     {norm key: key of the free or frozen vashift velocity of the line, or None (unshifted)}.
               The line energy is the LineE of the norm's component, which must be frozen
    continuum: additive components of the chain model (matched by component name in model, '' = the
               source model of the sidecar table), evaluated with XSPEC on an energy grid of bin
               width de keV, as bin flux / de interpolated at the line centre. Not finer: XSPEC's
               diskbb has bin-to-bin noise (rms 2e-3 at 0.14 eV bins, 3.5e-4 at 1 eV, measured
               2026-10-29), while the bias of 5 eV bins on these smooth continua is ~1e-5. Their
               norms enter linearly; each may have at most one other free
               parameter, whose shape is tabulated on n_nodes values across its sampled range and
               interpolated linearly (frozen parameters are set to their value; links are refused)

    Limits: a norm hard limit at 0 maps to EW = 0, any other one to no limit (it depends on the
    continuum of each sample).

    Needs PyXspec and an empty XSPEC session (it changes the model energies and defines a model):
    run it in its own process, as corner_chain's callers here do.
    '''

    def ew(params, table):
        from xspec import AllData, AllModels, Model, Xset
        from scipy.interpolate import RegularGridInterpolator

        if AllData.nSpectra or AllModels.sources:
            raise RuntimeError('gaussian_equivalent_widths needs an empty XSPEC session (no data, no model)')
        rows = [row for row in table.values() if row.get('model', '') == model]
        n_samp = len(next(iter(params.values())))

        def value(row):
            #samples of a free parameter, value of a frozen one
            if row['link']:
                raise ValueError('linked parameter %s.%s is not supported' % (row['component'], row['name']))
            return params[str(row['key'])] if row['free'] else np.full(n_samp, row['value'])

        # observed line centres
        energies = {}
        for key, v_key in lines.items():
            comp = table[str(key)]['component']
            line_e = [row for row in rows if row['component'] == comp and row['name'] == 'LineE']
            if len(line_e) != 1 or line_e[0]['free']:
                raise ValueError('norm %s: its component %s needs one frozen LineE' % (key, comp))
            vel = 0. if v_key is None else value(table[str(v_key)])
            energies[key] = line_e[0]['value'] * (1. - vel / C_KMS) * np.ones(n_samp)
        e_all = np.concatenate(list(energies.values()))
        e_lo, e_hi = e_all.min() - 10 * de, e_all.max() + 10 * de
        n_bins = int(np.ceil((e_hi - e_lo) / de))
        e_mid = e_lo + de * (np.arange(n_bins) + 0.5)

        # continuum: sum over the components of norm x shape (shape tabulated over its free parameter)
        chatter, Xset.chatter = Xset.chatter, 0
        AllModels.setEnergies('%.6f %.6f %d lin' % (e_lo, e_lo + n_bins * de, n_bins))
        cont = {key: np.zeros(n_samp) for key in lines}
        try:
            xs_model = Model(' + '.join(continuum))
            for comp in continuum:
                comp_rows = [row for row in rows if row['component'] == comp]
                if not comp_rows:
                    raise ValueError('continuum component %s is not in the chain model' % comp)
                free_shape = [row for row in comp_rows if row['name'] != 'norm' and row['free']]
                if len(free_shape) > 1:
                    raise NotImplementedError('%s has %d free shape parameters, at most 1 is supported'
                                              % (comp, len(free_shape)))
                #only this component: norm 1 here, 0 for the others
                for other in continuum:
                    getattr(xs_model, other).norm = 1. if other == comp else 0.
                for row in comp_rows:
                    if row['name'] != 'norm':
                        value(row)
                        getattr(getattr(xs_model, comp), row['name']).values = row['value']
                if free_shape:
                    x = value(free_shape[0])
                    nodes = np.linspace(x.min(), x.max(), n_nodes) if x.max() > x.min() else x[:1]
                else:
                    x, nodes = np.zeros(n_samp), np.zeros(1)
                shapes = []
                for node in nodes:
                    if free_shape:
                        getattr(getattr(xs_model, comp), free_shape[0]['name']).values = node
                    shapes.append(np.array(xs_model.values(0)) / de)
                shapes = np.array(shapes)
                norm = value([row for row in comp_rows if row['name'] == 'norm'][0])
                for key, ener in energies.items():
                    if len(nodes) > 1:
                        shape = RegularGridInterpolator((nodes, e_mid), shapes)(np.column_stack([x, ener]))
                    else:
                        shape = np.interp(ener, e_mid, shapes[0])
                    cont[key] += norm * shape
        finally:
            AllModels.clear()
            AllModels.setEnergies('reset')
            Xset.chatter = chatter

        out = {}
        for key in lines:
            row = table[str(key)]
            limits = tuple(0. if lim == 0 else None for lim in (row['hard_min'], row['hard_max']))
            out[key] = (1e3 * params[str(key)] / cont[key], limits)
        return out

    return ew


def compare_equivalent_widths(chain_file, lines, continuum=('powerlaw', 'diskbb'), n_samples=20, discard=0,
                              seed=0, narrow=5e-4, spectrum=1, verbose=True):
    '''
    Compares gaussian_equivalent_widths with XSPEC eqwidth on the same parameter values: the best fit
    of the chain (sidecar values) and n_samples random samples of the chain after discard.

    XSPEC eqwidth is computed three ways, on the chain's best-fit xcm (sidecar paths['bestfit_xcm'],
    restored from the run directory, provenance cwd) with the free parameters set to each vector:
        'xspec'           default eqwidth: continuum = all the other components (other lines included),
                          integrated over E (1 +- 0.05) around the line peak
        'xspec_narrow'    same, over E (1 +- narrow): removes the averaging over the continuum curvature
        'xspec_isolated'  narrow, with the norm of every other line component set to 0 (their links
                          undone; lines = the additive components not in continuum):
                          the definition of gaussian_equivalent_widths, which should agree to ~1e-4
                          (its 5 eV grid and XSPEC's diskbb bin noise)

    lines, continuum: as gaussian_equivalent_widths. Only models whose free parameters have integer
    keys (no named models) are supported.

    Needs PyXspec and an empty XSPEC session (gaussian_equivalent_widths): run it in its own process.
    It loads the data and model of the xcm; the session is cleared at the end.

    Returns {'ew': {method: {norm key: array (best fit first, then the samples)}}, 'vectors': the
    (1 + n_samples, n_free) parameter values, 'keys': their keys}. With verbose, prints the best-fit
    values and, over all vectors, the largest relative difference of each XSPEC method to
    gaussian_equivalent_widths.

    Why gaussian_equivalent_widths is much faster than XSPEC on chain samples: XSPEC eqwidth itself is
    quick (0.02 s on the Resolve model), but bringing the full model to each sample is not: every
    parameter change recomputes the whole model (TBabs, ~30 lines in vashift, continuum, the NXB model
    on the second response), folds it through the response and recomputes the statistic, 0.1-0.5 s
    (measured 2026-10-29), i.e. ~20 h for 192k samples. gaussian_equivalent_widths evaluates only the
    continuum components on a bare energy grid (no data, response, NXB or statistic; the lines and
    TBabs are not needed, TBabs cancels in the ratio), uses the linearity of the continuum in its norms
    (one XSPEC shape per component at norm 1) and tabulates the shape over its single free shape
    parameter (diskbb Tin, 200 nodes, 2.3 ms per evaluation): ~200 XSPEC calls in all, the rest is
    numpy, 1.4 s for 192k samples. This relies on at most one free shape parameter per continuum
    component (refused otherwise).

    Test run 2026-10-29 (Claude Code), MAXI J1543-564 empirical XRISM/Resolve chain, through
    posterior_empi_MAXIJ1543.py with WHAT = 'ew_check' (step 12: one subprocess calling this function
    with lines = EW_LINES, the 12 free line norms each mapped to the velocity of its zone, n_samples =
    EW_CHECK_SAMPLES = 20, discard = DISCARD = 4000), CORNER_FILE set to the backup
    MAXIJ1543_empi_all_free.fits.bak_cont2 (48 walkers x 68118 steps; the chain file was being written
    by continuation 2). 213 s. EW in eV at the best fit, then the largest |XSPEC / this method - 1| over
    the best fit and the 20 samples:
        norm  line              mine      xspec     narrow   isolated |  xspec  narrow  isolated
        7     Fe XXVI Lya2    -11.089   -12.294   -13.886   -11.088   | 1.2e-1  8.0e-1  3.1e-4
        10    Fe XXVI Lya1    -14.047   -15.451   -15.209   -14.049   | 1.1e-1  4.5e-1  2.4e-4
        14    Fe XXV w        -16.821   -17.980   -17.146   -16.823   | 7.7e-2  3.1e-2  2.5e-4
        18    Fe XXV y         -7.446    -8.067    -8.102    -7.448   | 9.4e-2  1.9e-1  2.5e-4
        26    Fe XXV x         -5.079    -5.515    -6.219    -5.078   | 9.8e-2  3.0e-1  2.3e-4
        33    Fe XXV Heb      -12.216   -12.045   -12.215   -12.215   | 1.9e-2  3.4e-4  3.4e-4
        50    Fe XXVI Lyb      -2.546    -2.586    -3.149    -2.546   | 2.2e-2  3.8e-1  4.2e-4
        37    S XVI Lya2       -0.513    -0.515    -0.557    -0.513   | 6.3e-3  9.8e-2  1.7e-4
        60    8.2955 keV       -9.835    -9.777    -9.836    -9.836   | 1.6e-2  2.2e-4  2.2e-4
        64    broad Lya2       -4.637    -5.179    -5.190    -4.637   | 1.3e-1  2.4e-1  2.8e-4
        70    6.660 keV        -6.406    -6.938    -7.171    -6.408   | 8.9e-2  2.1e-1  2.5e-4
        73    6.616 keV        -3.279    -3.437    -3.279    -3.279   | 5.6e-2  3.1e-5  3.1e-5
    -isolated agrees with this method to <= 4.2e-4 on every line and vector: same definition, the
     residual is the 5 eV grid and XSPEC's diskbb bin noise
    -default XSPEC differs by 0.6-13%: its continuum includes the other lines and is averaged over
     +-5% of the energy (only the averaging for the lines without neighbours: Heb, 8.2955, 6.616 keV,
     where narrow = isolated)
    -narrow without zeroing the other lines is the worst for blended lines (up to 80% for Fe XXVI
     Lya2, which sits in the wings of Lya1 and of the broad component): the "continuum" at the line
     peak is depressed by the neighbouring absorption lines
    '''

    from xspec import AllData, AllModels, Xset

    chain = load_chain(chain_file, discard=discard)
    side, table = chain['sidecar'], chain['table']
    if side is None:
        raise ValueError('no sidecar found for %s' % chain_file)
    keys = chain['keys']
    if any(not isinstance(key, int) for key in keys):
        raise NotImplementedError('only free parameters with integer keys (source model) are supported')

    idx = np.random.default_rng(seed).choice(len(chain['samples']), n_samples, replace=False)
    best = np.array([table[str(key)]['value'] for key in keys])
    vectors = np.vstack([best, chain['samples'][idx]])
    params = {str(key): vectors[:, j] for j, key in enumerate(keys)}

    # this module's method first: it needs the empty session
    ew = {'mine': {key: val for key, (val, _) in
                   gaussian_equivalent_widths(lines, continuum)(params, table).items()}}
    methods = ('xspec', 'xspec_narrow', 'xspec_isolated')
    ew.update({method: {key: np.zeros(len(vectors)) for key in lines} for method in methods})

    cwd_orig = os.getcwd()
    chatter, Xset.chatter = Xset.chatter, 0
    try:
        os.chdir(side['provenance']['cwd'])
        Xset.restore(side['paths']['bestfit_xcm'])
        os.chdir(cwd_orig)
        model = AllModels(1)
        names = list(model.componentNames)
        line_comps = [name for name in names if name not in continuum
                      and 'norm' in getattr(model, name).parameterNames]

        def eqwidth(comp, range_frac):
            AllModels.eqwidth(names.index(comp) + 1, rangeFrac=range_frac)
            return 1e3 * AllData(spectrum).eqwidth[0]

        #each parameter change recomputes the model and statistic (0.1-0.5 s on the Resolve model, eqwidth
        #itself 0.02 s): the changes are grouped. Pass 1, model as in the xcm
        for i_vec, vec in enumerate(vectors):
            model.setPars({key: val for key, val in zip(keys, vec)})
            for key in lines:
                comp = table[str(key)]['component']
                ew['xspec'][key][i_vec] = eqwidth(comp, 0.05)
                ew['xspec_narrow'][key][i_vec] = eqwidth(comp, narrow)

        #pass 2: every line norm unlinked and set to 0 (the session is cleared at the end), then one line
        #at a time is given its value
        line_norms = [getattr(model, comp).norm for comp in line_comps]
        for par in line_norms:
            if par.link:
                par.untie()
        model.setPars({par.index: 0. for par in line_norms})
        norm_cols = [j for j, key in enumerate(keys) if table[str(key)]['component'] in line_comps
                     and table[str(key)]['name'] == 'norm']
        for i_vec, vec in enumerate(vectors):
            model.setPars({keys[j]: vec[j] for j in range(len(keys)) if j not in norm_cols})
            previous = None
            for key in lines:
                change = {key: vec[keys.index(key)]}
                if previous is not None:
                    change[previous] = 0.
                model.setPars(change)
                ew['xspec_isolated'][key][i_vec] = eqwidth(table[str(key)]['component'], narrow)
                previous = key
            model.setPars({previous: 0.})
    finally:
        os.chdir(cwd_orig)
        AllData.clear()
        AllModels.clear()
        Xset.chatter = chatter

    if verbose:
        print('equivalent widths (eV) at the best fit; max |XSPEC / this module - 1| over the best fit and'
              ' %d chain samples' % n_samples)
        print('%-6s %-14s %10s %10s %10s %10s   %9s %9s %9s' % ('norm', 'component', 'mine', 'xspec', 'narrow',
                                                              'isolated', 'xspec', 'narrow', 'isolated'))
        for key in lines:
            rel = [np.abs(ew[method][key] / ew['mine'][key] - 1).max() for method in methods]
            print('%-6s %-14s %10.4f %10.4f %10.4f %10.4f   %9.1e %9.1e %9.1e'
                  % ((key, table[str(key)]['component']) + tuple(ew[method][key][0] for method in ('mine',) + methods)
                     + tuple(rel)))
    return {'ew': ew, 'vectors': vectors, 'keys': keys}


def load_chain(chain_file, sidecar=None, discard=0, thin=None, max_samples=200000,
               uniform_prior_keys=None, seed=0):
    '''
    Reads a chain_runner chain and its sidecar, for plotting.

    discard:      steps per walker removed at the start (the burn-in is already NOT in the file)
    thin:         keep one step per walker out of thin; None = smallest stride giving at most
                  max_samples samples in total. Plotting does not need independent samples:
                  thinning only bounds the cost, the shape of the contours is unchanged
                  (their noise is set by the ESS, not by the number of rows).
    uniform_prior_keys: None, or keys of jeffreys-prior parameters whose prior is replaced by a
                  uniform one (uniform_prior_weights). Instead of the regular thinning, max_samples
                  samples are drawn WITH replacement from all the samples after discard, with
                  probabilities proportional to the weights (importance resampling, as in
                  comp_ratio_plot_MC.load_draws): the result is an unweighted sample of the new
                  posterior, so the plots need no weights (corner.corner's points ignore weights).
                  Drawn from the full chain, the repeats are rare. thin is not used.
    seed:         random seed of that resampling (fixed by default: the same figures at each run)

    Returns a dict:
        samples   (n_samples, n_params) flattened over walkers and steps (step-major)
        step      (n_samples,) step index of each sample (counted after discard)
        walker    (n_samples,) walker index
        statistic (n_samples,)
        keys, labels   as read_chain
        table     {str(key): sidecar parameter row} for the chain columns (empty without sidecar)
        sidecar   the sidecar dict (None if not found)
        n_walkers, n_steps (steps per walker in the file, before discard and thinning), thin
        uniform_prior_keys, weight_ess_frac (Kish ESS fraction of the weights, None without
        reweighting)
    '''

    chain = read_chain(chain_file)
    samples, stat = chain['samples'], chain['statistic']        #(walker, step, par), (walker, step)
    n_walk, n_steps, n_par = samples.shape

    sidecar_path = sidecar if sidecar is not None else find_sidecar(chain_file)
    side = None
    table = {}
    if sidecar_path is not None:
        with open(sidecar_path) as f:
            side = json.load(f)
        table = {str(row['key']): row for row in side.get('parameters', [])}

    if discard:
        samples, stat = samples[:, discard:], stat[:, discard:]
    n_kept = samples.shape[1]

    weight_ess_frac = None
    if uniform_prior_keys:
        probs, weight_ess_frac = uniform_prior_weights(samples, chain['keys'], table, uniform_prior_keys)
        idx = np.random.default_rng(seed).choice(probs.size, max_samples, replace=True, p=probs.ravel())
        i_walk, i_step = np.unravel_index(idx, probs.shape)
        #step-major order, as without reweighting
        order = np.lexsort((i_walk, i_step))
        i_walk, i_step = i_walk[order], i_step[order]
        flat, stat_flat = samples[i_walk, i_step], stat[i_walk, i_step]
        thin = None
        print('load_chain: uniform prior on %s by importance resampling, effective sample size of the'
              ' weights %.2f of the samples' % (list(uniform_prior_keys), weight_ess_frac))
    else:
        if thin is None:
            thin = max(1, int(np.ceil(n_walk * n_kept / max_samples)))
        samples, stat = samples[:, ::thin], stat[:, ::thin]
        step_idx = np.arange(n_kept)[::thin]
        #back to step-major order, so that a slice of rows is a slice of steps
        flat, stat_flat = samples.transpose(1, 0, 2).reshape(-1, n_par), stat.T.ravel()
        i_step, i_walk = np.repeat(step_idx, n_walk), np.tile(np.arange(n_walk), len(step_idx))

    return {'samples': flat,
            'step': i_step,
            'walker': i_walk,
            'statistic': stat_flat,
            'uniform_prior_keys': None if not uniform_prior_keys else list(uniform_prior_keys),
            'weight_ess_frac': weight_ess_frac,
            'keys': chain['keys'],
            'labels': chain['labels'],
            'table': table,
            'sidecar': side,
            'sidecar_path': sidecar_path,
            'n_walkers': n_walk,
            'n_steps': n_steps,
            'discard': discard,
            'thin': thin}


def _resolve_params(params, chain):
    '''
    Column indices of params: keys (1, 'src:1'), 'component.name' strings, or None for all.
    '''

    keys = [str(elem) for elem in chain['keys']]
    if params is None:
        return list(range(len(keys)))

    out = []
    for elem in params:
        if str(elem) in keys:
            out.append(keys.index(str(elem)))
            continue
        matches = [i for i, key in enumerate(keys) if key in chain['table'] and
                   chain['table'][key]['component'] + '.' + chain['table'][key]['name'] == elem]
        if len(matches) != 1:
            raise ValueError('parameter %r: %d matches among the free parameters of the chain'
                             % (elem, len(matches)))
        out.append(matches[0])
    return out


def chain_summary_text(chain):
    '''
    Description of the plotted chain, with the sidecar verdict if it describes this file.
    '''

    text = '%s\n%d walkers x %d steps' % (os.path.basename(chain.get('file', '')),
                                         chain['n_walkers'], chain['n_steps'])
    if chain['discard']:
        text += ', first %d discarded' % chain['discard']
    if chain.get('uniform_prior_keys'):
        names = [chain['table'][str(key)]['component'] + '.' + chain['table'][str(key)]['name']
                 if str(key) in chain['table'] else str(key) for key in chain['uniform_prior_keys']]
        #wrapped: the text sits in the upper-right triangle, which is narrow on small figures
        text += '\n' + textwrap.fill('uniform prior (was jeffreys) on ' + ', '.join(names), 50) + \
                '\n%d samples drawn by importance resampling, weight ESS %.2f' \
                % (len(chain['samples']), chain['weight_ess_frac'])
    else:
        text += ', thinned x%d (%d samples)' % (chain['thin'], len(chain['samples']))

    side = chain['sidecar']
    if side is not None:
        side_steps = side.get('settings', {}).get('steps_per_walker_total',
                                                  side.get('settings', {}).get('steps_per_walker'))
        diag = side.get('diagnostics', {})
        #a reassessment promoted by chain_diagnostics.reassess_chain only holds for its own discard
        diag_discard = diag.get('discard') or 0
        if diag_discard and side_steps == chain['n_steps'] and chain['discard'] != diag_discard:
            text += '\n(sidecar verdict is for the first %d steps discarded: not shown)' % diag_discard
        elif side_steps == chain['n_steps'] and 'rhat' in diag:
            rhat = list(diag['rhat'].values())
            ess = list(diag.get('ess_bulk', {}).values())
            text += '\nverdict %s, R-hat %.3f-%.3f' % (diag.get('verdict'), min(rhat), max(rhat))
            if ess:
                text += '\nbulk ESS %.0f-%.0f' % (min(ess), max(ess))
            #the tau of the verdict's length rule (walker mean since 2026-10-24), else the largest walker tau
            if 'tau_used_all' in diag:
                text += ', tau %.0f steps' % diag['tau_used_all']
            elif 'tau_max_all' in diag:
                text += ', tau max %.0f steps' % diag['tau_max_all']
        elif side_steps is not None:
            text += '\n(sidecar diagnostics are for %d steps per walker: not shown)' % side_steps
    return text


def corner_chain(chain_file, params=None, labels=None, titles=None, log=None, scale=None, sidecar=None,
                 discard=0, thin=None, max_samples=200000, uniform_prior_keys=None, seed=0,
                 split=None, paper=False, backend='both',
                 decimal_log=True, show_limits=True, show_bestfit=True, show_minstat=False, title=True,
                 table_grids=None, derived=None, save=None, formats=('png', 'pdf'), dpi=120, pdf_max_gb=1.,
                 **corner_kw):
    '''
    Corner plot of a chain_runner chain file.

    params:   parameters to plot (keys like 1 or 'src:1', or 'component.name'); None = all free
    labels:   {key or 'component.name': axis label}; default '<component> <name>', with 'log '
              for the log10 parameters unless decimal_log
    titles:   {key or 'component.name': name in the diagonal titles}; default = labels
    log:      keys/names to plot in log10; None = the parameters whose prior in the plot is
              log-uniform: jeffreys in the sidecar and not in uniform_prior_keys (without sidecar,
              the positive parameters spanning > 2 decades)
    scale:    {key or 'component.name': factor} applied before the log10 (e.g. z -> cz in km/s)
    discard, thin, max_samples, uniform_prior_keys, seed: see load_chain (uniform_prior_keys: plot
              the posterior with uniform instead of jeffreys priors on these parameters)
    split:   None, or an integer k >= 2: overlay k consecutive parts of the chain (in steps),
              a visual stationarity check (parts that disagree = the chain still drifts). The
              quantiles and titles stay those of the whole chain (black dashed lines)
    paper:    figure for a paper: no description/diagnostics text (file, verdict, R-hat, ESS);
              the hard limits and the best fit stay. Only for a single (non-split) dataset.
    backend:  'own', 'corner' or 'both' (both figures; if the corner package is not importable,
              'both' warns and makes only 'own')
    decimal_log: ticks and titles of the log10 parameters as round linear values
    show_limits:  draw the hard limits from the sidecar (the prior ranges of chain_runner runs)
    show_bestfit: mark the sidecar values (the refit before sampling, without priors)
    show_minstat: mark the sample of lowest FIT_STATISTIC in the plotted samples
    title:    True = description of the chain and of its sidecar verdict, or a string
    table_grids: {model component name: table model file}: draw the grid nodes of the table
              parameters of these components (table_grid_values, matched by parameter name), with
              the same scale and log10 as the samples. Parameters outside the table grid (z, norm)
              get no lines
    derived:  None, or a function f(params, table) -> {key: (values, (lo, hi))} replacing the plotted
              values of some parameters by quantities derived from the chain (e.g.
              gaussian_equivalent_widths). params = {str(key): array} for every chain column, table =
              the sidecar rows. It is called on the samples, then on the best fit (arrays of 1); its
              (lo, hi) (None = no limit) replace the hard limits of these parameters. scale and log
              still apply after it
    save:     base path of the images (an extension is stripped): writes <save>_<backend>.<fmt>
              for each format in formats, at dpi (the directory is created)
    pdf_max_gb: the pdf is skipped (with a message) when pdf_raster_memory predicts more than this
              (the corner backend's rasterized points on a large figure: ~6 GB for 17 parameters)
    corner_kw: passed to the backend (bins, smooth, show_corr, density, plot_points, fill,
              panel_size, ...)

    Returns (figures, chain): figures = {backend: (fig, axes)}, chain as returned by load_chain
    plus 'columns' (the plotted column indices) and 'plotted' (the transformed samples).
    '''

    if paper and split:
        raise ValueError('paper=True is for a single dataset: remove split')
    if paper:
        title = None

    backends = ['own', 'corner'] if backend == 'both' else [backend]
    if 'corner' in backends:
        try:
            import corner  # noqa: F401
        except ImportError:
            if backend != 'both':
                raise
            print('corner_chain: the corner package is not importable, only the "own" backend is drawn')
            backends.remove('corner')

    chain = load_chain(chain_file, sidecar=sidecar, discard=discard, thin=thin, max_samples=max_samples,
                       uniform_prior_keys=uniform_prior_keys, seed=seed)
    chain['file'] = chain_file
    cols = _resolve_params(params, chain)
    keys = [str(chain['keys'][i]) for i in cols]
    rows = [chain['table'].get(key) for key in keys]

    def lookup(mapping, i):
        if not mapping:
            return None
        key, row = keys[i], rows[i]
        if key in mapping:
            return mapping[key]
        for elem in mapping:
            if str(elem) == key or (row is not None and elem == row['component'] + '.' + row['name']):
                return mapping[elem]
        return None

    data = chain['samples'][:, cols].copy()
    derived_limits = {}
    if derived is not None:
        all_params = {str(key): chain['samples'][:, j] for j, key in enumerate(chain['keys'])}
        for key, (vals, lims) in derived(all_params, chain['table']).items():
            if str(key) in keys:
                data[:, keys.index(str(key))] = vals
                derived_limits[str(key)] = lims
    is_log = []
    for i in range(len(cols)):
        factor = lookup(scale, i)
        if factor is not None:
            data[:, i] *= factor
        if log is not None:
            use_log = lookup({elem: True for elem in log}, i) is not None
        elif rows[i] is not None and 'prior' in rows[i]:
            #log10 where the prior of the plot is log-uniform: jeffreys in the chain and not
            #reweighted to a uniform prior
            use_log = rows[i]['prior'][0] == 'jeffreys' and \
                keys[i] not in [str(elem) for elem in (uniform_prior_keys or [])]
        else:
            x = data[:, i]
            use_log = bool(x.min() > 0 and x.max() / x.min() > 100)
        if use_log:
            data[:, i] = np.log10(data[:, i])
        is_log.append(use_log)

    def transform(vals, i):
        vals = np.array(vals, dtype=float)
        factor = lookup(scale, i)
        if factor is not None:
            vals = vals * factor
        if is_log[i]:
            with np.errstate(divide='ignore', invalid='ignore'):
                vals = np.where(vals > 0, np.log10(np.where(vals > 0, vals, 1.)), np.nan)
        return vals

    plot_labels, plot_titles = [], []
    for i in range(len(cols)):
        lab = lookup(labels, i)
        if lab is None:
            base = (rows[i]['component'] + ' ' + rows[i]['name']) if rows[i] is not None \
                else chain['labels'][cols[i]]
            lab = ('log ' if is_log[i] and not decimal_log else '') + base
        plot_labels.append(lab)
        tit = lookup(titles, i)
        plot_titles.append(lab if tit is None else tit)

    limits = None
    if show_limits and any(elem is not None for elem in rows):
        #a None limit (derived quantities) becomes nan: not drawn
        limits = [tuple(transform(derived_limits.get(keys[i], (row['hard_min'], row['hard_max'])), i))
                  if row is not None else None for i, row in enumerate(rows)]

    markers = {}
    if show_bestfit and any(elem is not None for elem in rows):
        best = [row['value'] if row is not None and keys[i] not in derived_limits else np.nan
                for i, row in enumerate(rows)]
        if derived_limits and all(str(key) in chain['table'] for key in chain['keys']):
            best_params = {str(key): np.array([chain['table'][str(key)]['value']]) for key in chain['keys']}
            for key, (vals, _) in derived(best_params, chain['table']).items():
                if str(key) in keys:
                    best[keys.index(str(key))] = vals[0]
        markers['best fit (no prior)'] = np.array([transform(val, i) for i, val in enumerate(best)])
    if show_minstat:
        markers['lowest statistic sample'] = data[int(np.argmin(chain['statistic']))]

    grid = None
    if table_grids:
        nodes = {comp: table_grid_values(path) for comp, path in table_grids.items()}
        grid = [transform(nodes[row['component']][row['name']], i)
                if row is not None and row['name'] in nodes.get(row['component'], {}) else None
                for i, row in enumerate(rows)]

    names = None
    if split:
        edges = np.linspace(0, chain['step'].max() + 1, int(split) + 1).astype(int)
        parts = [(chain['step'] >= edges[k]) & (chain['step'] < edges[k + 1]) for k in range(int(split))]
        datasets = [data[elem] for elem in parts]
        names = ['steps %d-%d' % (edges[k] + discard, edges[k + 1] - 1 + discard) for k in range(int(split))]
        #quantiles and titles from the whole chain, not from its first part
        corner_kw.setdefault('summary_samples', data)
    else:
        datasets = data

    if title is True:
        title = chain_summary_text(chain)

    figures = {}
    for name in backends:
        fig, axes = BACKENDS[name](datasets, labels=plot_labels, titles=plot_titles, names=names,
                                   markers=markers, limits=limits,
                                   log_flags=is_log if decimal_log else None,
                                   suptitle=title or None, grid=grid, **corner_kw)
        figures[name] = (fig, axes)
        if save:
            base = re.sub(r'\.(png|pdf)$', '', save)
            os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
            for fmt in formats:
                if fmt == 'pdf' and pdf_raster_memory(fig, dpi) > pdf_max_gb * 1e9:
                    print('corner_chain: %s_%s.pdf skipped, it would need about %.1f GB of memory'
                          ' (rasterized artists; raise pdf_max_gb to force it)'
                          % (os.path.basename(base), name, pdf_raster_memory(fig, dpi) / 1e9))
                    continue
                fig.savefig('%s_%s.%s' % (base, name, fmt), dpi=dpi)

    chain['columns'] = cols
    chain['plotted'] = data
    return figures, chain


# ---------------------------------------------------------------------------------------------
# MAXI J1543-564 pion model
# ---------------------------------------------------------------------------------------------

MAXIJ1543_DIR = '/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion/posterior'
MAXIJ1543_CHAIN = os.path.join(MAXIJ1543_DIR, 'MAXIJ1543_3pion_all_free_long.fits')

#pion component -> zone number used in the labels and set names. Zones are ordered by decreasing
#N_H (posterior medians of the 21000-step long chain): zone 1 = pion_abs_3 (N_H ~ 6.6e22, the one
#with the covering factor), zone 2 = pion_abs_4 (N_H ~ 4.5e22, redshifted), zone 3 = pion_abs
#(N_H ~ 0.6e22, log xi ~ 3.8 < 4). Insertion order = zone order.
MAXIJ1543_ZONES = {'pion_abs_3': '1', 'pion_abs_4': '2', 'pion_abs': '3'}

#pion table of each component (fit_3comp1em_gamma2.xcm), for the grid nodes (table_grid=True). Both
#tables have the same grid: rlogxi 1-6 in steps of 0.2 (linear interpolation), Nh and v_rms in steps
#of 0.2 dex (log interpolation). The z are added by XSPEC (not on the grid); pion_emit's table
#parameters are linked to zone 1 and its only free parameter, the norm (covering fraction), is not
#a table parameter
MAXIJ1543_PION_DIR = os.path.dirname(MAXIJ1543_DIR)
MAXIJ1543_TABLES = {'pion_abs': os.path.join(MAXIJ1543_PION_DIR, 'pionabsMAXIJ1543.fits'),
                    'pion_abs_3': os.path.join(MAXIJ1543_PION_DIR, 'pionabsMAXIJ1543.fits'),
                    'pion_abs_4': os.path.join(MAXIJ1543_PION_DIR, 'pionabsMAXIJ1543.fits'),
                    'pion_emit': os.path.join(MAXIJ1543_PION_DIR, 'pionemitMAXIJ1543.fits')}


def maxij1543_param_sets(table):
    '''
    Named parameter sets of the MAXI J1543 pion model, from the sidecar parameter table
    ({str(key): row}): 'all', 'continuum', 'absorbers', 'zone1', 'zone2', 'zone3', 'zones_nh_v'
    (N_H and v_rms of every zone). The covering factor (pion_emit.norm) goes with the zone whose
    parameters the pion_emit component is linked to (zone 1).
    Order: zone 1 (+ covering factor), zone 2, zone 3, then the continuum.
    '''

    free = [row for row in table.values() if row.get('free')]

    #pion_emit links: '= p6' -> parameter 6 -> its component
    emit_parent = None
    linked = [row['link'] for row in table.values() if row['component'] == 'pion_emit' and row['link']]
    if linked:
        target = re.sub(r'[^0-9]', '', linked[0])
        emit_parent = table.get(target, {}).get('component')

    sets = {}
    for comp, zone in MAXIJ1543_ZONES.items():
        sets['zone' + zone] = [row['key'] for row in free if row['component'] == comp]
        if comp == emit_parent:
            sets['zone' + zone] += [row['key'] for row in free if row['component'] == 'pion_emit']
    zone_keys = [key for zone in sorted(MAXIJ1543_ZONES.values()) for key in sets['zone' + zone]]
    sets['absorbers'] = zone_keys + [row['key'] for row in free if row['group'] == 'absorber'
                                     and row['key'] not in zone_keys]
    sets['continuum'] = [row['key'] for row in free if row['group'] == 'continuum']
    sets['all'] = sets['absorbers'] + sets['continuum']
    nh_v = {row['key'] for row in free if row['name'].split('/')[0].split('_')[0] in ('Nh', 'v')}
    sets['zones_nh_v'] = [key for key in zone_keys if key in nh_v]
    return sets


def maxij1543_labels_scales(table, velocity=True, covering_fraction=True, decimal_log=True, linear_keys=()):
    '''
    Axis labels, title names and scale factors of the MAXI J1543 parameters for corner_chain.
    velocity:          plot the absorber z as v_out = cz in km/s (same sign as z: negative =
                       blueshift)
    covering_fraction: plot pion_emit.norm divided by its hard maximum (the 4 pi value), i.e. the
                       covering fraction (labelled Omega, 1 = full covering)
    decimal_log:       labels of the log-sampled parameters without "log" (linear tick values);
                       log xi stays log xi (it is the sampled parameter itself)
    linear_keys:       keys of jeffreys parameters plotted on a linear axis (reweighted to a uniform
                       prior): never labelled "log"
    Returns (labels, titles, scale), dicts keyed by str(key).
    '''

    linear_keys = [str(elem) for elem in linear_keys]

    def logged(name, unit=''):
        #(axis label, title name) of a log-sampled quantity (key: the parameter of the loop below)
        if decimal_log or key in linear_keys:
            return name + (' (%s)' % unit if unit else ''), name
        return r'$\log$ ' + name, r'$\log$ ' + name

    labels, titles, scale = {}, {}, {}
    for key, row in table.items():
        comp, name = row['component'], row['name']
        lab = None
        if comp in MAXIJ1543_ZONES:
            sub = MAXIJ1543_ZONES[comp]
            if name == 'rlogxi':
                lab = (r'$\log\xi_{%s}$' % sub,) * 2
            elif name.startswith('Nh'):
                lab = logged(r'$N_{\rm H,%s}$' % sub, r'$10^{22}$ cm$^{-2}$')
            elif name.startswith('v_rms'):
                lab = logged(r'$v_{\rm rms,%s}$' % sub, 'km/s')
            elif name == 'z':
                if velocity:
                    lab = (r'$v_{\rm out,%s}$ (km/s)' % sub, r'$v_{\rm out,%s}$' % sub)
                    scale[key] = C_KMS
                else:
                    lab = (r'$z_{%s}$' % sub,) * 2
        elif comp == 'pion_emit' and name == 'norm':
            if covering_fraction:
                lab = logged(r'$\Omega$')
                scale[key] = 1. / row['hard_max']
            else:
                lab = logged('emit norm')
        elif comp == 'TBabs' and name == 'nH':
            lab = logged(r'$N_{\rm H}^{\rm ISM}$', r'$10^{22}$ cm$^{-2}$')
        elif comp == 'powerlaw' and name == 'norm':
            lab = logged('PL norm')
        elif comp == 'diskbb' and name == 'Tin':
            lab = (r'$T_{\rm in}$ (keV)', r'$T_{\rm in}$')
        elif comp == 'diskbb' and name == 'norm':
            lab = logged('disk norm')
        if lab is not None:
            labels[key], titles[key] = lab
    return labels, titles, scale


#parameters whose jeffreys prior is replaced by a uniform one with uniform_priors=True: the covering
#fraction (a fraction: uniform on its physical range) and the turbulent velocity of the three zones.
#With jeffreys priors, the weakly constrained Omega and zone 3 v_rms depend on their arbitrary lower
#limits (0.001 and the table's 3 km/s); the three v_rms get the same prior, for consistency (zones 1
#and 2 are well measured and move by < 0.3 sigma, 2026-10-25). Same choice as
#comp_ratio_plot_MC.plot_comp_ratio_MC_MAXIJ1543 (parameters 4, 8, 12, 22)
MAXIJ1543_UNIFORM_PRIOR_PARAMS = ('pion_emit.norm', 'pion_abs_3.v_rms_[km/s]', 'pion_abs_4.v_rms_[km/s]',
                                  'pion_abs.v_rms_[km/s]')

#output sub-directory of <chain dir>/corner and file name tag of each prior choice
MAXIJ1543_PRIOR_OUTPUT = {False: ('omega_vrms_loguniform_priors', 'logu'),
                          True: ('omega_vrms_uniform_priors', 'unif')}


def corner_maxij1543(chain_file=MAXIJ1543_CHAIN, select='all', velocity=True, covering_fraction=True,
                     uniform_priors=False, table_grid=False, save=None, **kw):
    '''
    Corner plot of a MAXI J1543 pion chain.

    select:  a set name of maxij1543_param_sets ('all', 'continuum', 'absorbers', 'zone1', 'zone2',
             'zone3', 'zones_nh_v'), several joined by '+' (e.g. 'zone1+zone2'), or an explicit
             list of keys / 'component.name'
    uniform_priors: plot the posterior with uniform instead of jeffreys (log-uniform) priors on the
             covering fraction and the v_rms of the 3 zones (MAXIJ1543_UNIFORM_PRIOR_PARAMS), by importance
             resampling of the chain (load_chain, uniform_prior_keys)
    table_grid: draw the grid nodes of the pion tables (MAXIJ1543_TABLES) on the log xi, N_H and
             v_rms panels
    save:    True = <chain dir>/corner/<prior dir>/<chain file name>_<select>[_split<k>][_paper]
             [_grid]_<prior tag>_<backend>.png and .pdf, with (prior dir, prior tag) from
             MAXIJ1543_PRIOR_OUTPUT ('omega_vrms_loguniform_priors', 'logu' or
             'omega_vrms_uniform_priors', 'unif'), or a base path
    kw:      passed to corner_chain and to the backend (backend='own'/'corner'/'both', split=2,
             paper=True, discard, thin, show_corr=True, bins, smooth, show_minstat, ...)

    log10 is applied to the jeffreys-prior parameters (as sampled: uniform in log), shown with
    round linear tick values; the zones are numbered by decreasing N_H (MAXIJ1543_ZONES), their z
    shown as v_out = cz in km/s and the covering factor as the covering fraction Omega (see
    maxij1543_labels_scales). Returns (figures, chain) as corner_chain.
    '''

    sidecar = kw.pop('sidecar', None) or find_sidecar(chain_file)
    if sidecar is None:
        raise ValueError('no sidecar found for %s: pass sidecar=<path of the .json>' % chain_file)
    with open(sidecar) as f:
        table = {str(row['key']): row for row in json.load(f)['parameters']}

    sets = maxij1543_param_sets(table)
    if isinstance(select, str):
        #'zone1+zone2': union of the sets, in the order given
        params = []
        for name in select.split('+'):
            params += [elem for elem in sets[name.strip()] if elem not in params]
    else:
        params = list(select)
    if uniform_priors:
        by_name = {row['component'] + '.' + row['name']: row['key'] for row in table.values()}
        kw['uniform_prior_keys'] = [by_name[elem] for elem in MAXIJ1543_UNIFORM_PRIOR_PARAMS]

    #the reweighted parameters are plotted on a linear axis (corner_chain), their labels follow
    labels, titles, scale = maxij1543_labels_scales(table, velocity=velocity,
                                                    covering_fraction=covering_fraction,
                                                    decimal_log=kw.get('decimal_log', True),
                                                    linear_keys=kw.get('uniform_prior_keys') or ())
    labels.update(kw.pop('labels', None) or {})
    titles.update(kw.pop('titles', None) or {})

    if save is True:
        prior_dir, prior_tag = MAXIJ1543_PRIOR_OUTPUT[bool(uniform_priors)]
        tag = select if isinstance(select, str) else 'custom'
        if kw.get('split'):
            tag += '_split%d' % kw['split']
        if kw.get('paper'):
            tag += '_paper'
        if table_grid:
            tag += '_grid'
        save =os.path.join(os.path.dirname(chain_file), 'corner', prior_dir,
                            os.path.basename(chain_file).replace('.fits', '') + '_' + tag + '_' + prior_tag)

    #absorber panels are few: larger panels by default
    kw.setdefault('panel_size', 1.6 if len(params) > 8 else 2.2)
    if table_grid:
        kw['table_grids'] = MAXIJ1543_TABLES
    return corner_chain(chain_file, params=params, labels=labels, titles=titles, scale=scale,
                        sidecar=sidecar, save=save, **kw)


#the figure set of the paper: 'own' backend for the whole model (also without the diagnostics text),
#each zone and each pair of zones; 'corner' backend for each zone
MAXIJ1543_FIGURE_SET = ([('own', dict(select='all')), ('own', dict(select='all', paper=True))]
                        + [('own', dict(select=sel)) for sel in ('zone1', 'zone2', 'zone3', 'zone1+zone2',
                                                                 'zone1+zone3', 'zone2+zone3')]
                        + [('corner', dict(select=sel)) for sel in ('zone1', 'zone2', 'zone3')])

#each zone with the pion table grid nodes (are the bimodalities interpolation artefacts?), both backends
MAXIJ1543_GRID_FIGURE_SET = [(backend, dict(select=sel, table_grid=True)) for sel in ('zone1', 'zone2', 'zone3')
                             for backend in ('own', 'corner')]


def make_maxij1543_figures(chain_file=MAXIJ1543_CHAIN, discard=10000, uniform_priors=(False, True),
                           figure_set=MAXIJ1543_FIGURE_SET):
    '''
    Makes the figure set (MAXIJ1543_FIGURE_SET) of a MAXI J1543 chain for each prior choice:
    uniform_priors False = the priors of the chain (jeffreys on the covering fraction and the v_rms
    of the 3 zones), True = uniform priors on these (MAXIJ1543_UNIFORM_PRIOR_PARAMS, importance
    resampling, plotted on linear axes). Each choice goes to its
    own directory with its own file tag (MAXIJ1543_PRIOR_OUTPUT), in png and pdf.

    discard: 10000 = the discard for which the final chain's verdict is OK (sidecar 'diagnostics',
    promoted reassessment).

    Each figure is made in its own process: loading the final chain (7.88M rows) peaks at 3.4 GB
    (measured 2026-10-25), and in one process the memory does not all come back between figures.
    '''

    import subprocess
    import sys
    for uniform in uniform_priors:
        for backend, run in figure_set:
            code = ("from contour_maker import *\n"
                    "corner_maxij1543(%r, discard=%d, uniform_priors=%r, save=True, backend=%r, **%r)"
                    % (chain_file, discard, uniform, backend, run))
            print('make_maxij1543_figures: %s priors, %s, %s' % ('uniform' if uniform else 'chain', backend, run),
                  flush=True)
            subprocess.run([sys.executable, '-c', code], cwd=os.path.dirname(os.path.abspath(__file__)),
                           check=True)


if __name__ == '__main__':
    make_maxij1543_figures()

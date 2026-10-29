"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-24.

Chain-based version of comp_ratio_plots.plot_comp_ratio: ratios of additive components to a continuum, with
pointwise credible bands from parameter draws of an XSPEC MCMC chain written by posterior/chain_runner.py.

Method:
    -the model is the chain's own best-fit xcm (sidecar paths['bestfit_xcm']): chain column k is model parameter k,
     and the frozen and linked values are those of the chain run
    -curves that are not additive components of the full model (e.g. the unabsorbed continuum, or the
     transmission of a single absorber) come from the same model with only some components kept (the others are
     removed with XSPEC delcomp, which keeps the values and order of the remaining parameters)
    -the same draws, in the same order, are evaluated in every derived model. The ratio is computed for each
     draw, then its quantiles are taken in each energy bin over the draws
    -the draws are unweighted by default: chain samples are already distributed as the posterior, so no Delta-C
     cut and no likelihood weights (see the 2026-10-24 assessment of Kammoun et al. 2025, Appendix B)
    -optionally (uniform_prior_keys), the jeffreys (log-uniform) prior of some parameters is replaced by a uniform
     prior without rerunning the chain, by importance resampling with weights new prior / old prior (load_draws)
    -the bands are pointwise: a single draw curve can leave a 68% band in places

The chains are read with posterior/contour_maker.load_chain (discard, thin, sidecar).

plot_comp_ratio_MC_broken (2026-10-26) draws several energy ranges side by side on one broken x axis (brokenaxes).
"""

import os
import sys
import json
import secrets
import re

import numpy as np
from tqdm import tqdm

from xspec import _pyXspec
from xspec_config_multisp import Plot, Xset, AllModels, set_ener
from comp_ratio.comp_ratio_plots import _ener_slice, _sum_addcomps, _plot_ratios

#posterior/ is not a package: its modules import each other by plain name (same as chain_runner.py)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'posterior'))
from contour_maker import load_chain, find_sidecar, uniform_prior_weights


def load_draws(chain_file, n_draws=1000, discard=0, thin=1, draw_mode='random', seed=None, uniform_prior_keys=None):
    '''
    Parameter draws from one or several chain_runner chain files (several files must have the same free
    parameters, e.g. a chain and its continuation backups).

    discard: steps per walker removed at the start of each file (the burn-in is already not in the file)
    thin: keep one step per walker out of thin before drawing
    draw_mode: 'random' (n_draws samples without replacement, from seed) or 'sequential' (n_draws samples evenly
        spaced through the retained, step-major samples)
    seed: random seed for 'random'. None draws one, which is printed and returned

    uniform_prior_keys: None, or keys (model parameter indices, as in the chain columns) of parameters sampled
        with a jeffreys prior, whose prior is replaced by a uniform one between the same hard limits.
        Method (importance resampling): the chain samples follow posterior = likelihood x old prior. Weighting
        each sample by new prior / old prior turns them into samples of likelihood x new prior. For jeffreys
        (density proportional to 1/x) -> uniform (constant), the weight is x for each such parameter, and the
        product over the parameters when there are several. The draws are then taken at random WITH replacement,
        with probabilities proportional to the weights, so that they are unweighted samples of the new posterior
        (and need no change downstream). The likelihood is not used in the weights (unlike the Delta-C weights
        of Kammoun et al.): only the ratio of the priors changes.
        Valid if the old samples cover the region the new posterior favours: the effective sample size of the
        weights (Kish, (sum w)^2 / sum w^2) is printed and returned. A few percent of the retained samples or less
        means the new prior needs its own chain. Requires draw_mode='random'.

    Returns a dict: samples (n_draws, n_free), keys (chain column keys), sidecar (of the first file),
    sample_idx (indices in the retained samples), n_retained, seed, uniform_prior_keys, weight_ess (Kish ESS of
    the weights over the retained samples, None without reweighting)
    '''

    chain_files = [chain_file] if isinstance(chain_file, str) else list(chain_file)

    chains = [load_chain(elem, discard=discard, thin=thin) for elem in chain_files]
    keys = chains[0]['keys']
    for elem_file, elem_chain in zip(chain_files[1:], chains[1:]):
        if elem_chain['keys'] != keys:
            raise ValueError('load_draws: %s has free parameters %s, %s has %s'
                             % (chain_files[0], keys, elem_file, elem_chain['keys']))
    if chains[0]['sidecar'] is None:
        raise ValueError('load_draws: no sidecar (.json) found for %s' % chain_files[0])

    samples = np.concatenate([elem['samples'] for elem in chains])
    n_retained = len(samples)
    sidecar = chains[0]['sidecar']

    weights = None
    weight_ess = None
    if uniform_prior_keys:
        if draw_mode != 'random':
            raise ValueError('load_draws: uniform_prior_keys requires draw_mode="random"')

        #weight = product of x over the reweighted parameters, normalized; the same function makes the
        #reweighted corner plots (contour_maker.load_chain), and checks that each parameter is free and
        #was sampled with a jeffreys prior
        table = {str(row['key']): row for row in sidecar['parameters']}
        weights, weight_ess_frac = uniform_prior_weights(samples, keys, table, uniform_prior_keys)
        weight_ess = weight_ess_frac*n_retained
        print('load_draws: uniform prior on %s by importance resampling, effective sample size of the weights %d '
              '(%.2f of the %d retained samples)' % (list(uniform_prior_keys), weight_ess, weight_ess/n_retained,
                                                     n_retained))
        if weight_ess < 10*n_draws:
            print('load_draws: WARNING the effective sample size of the weights is less than 10 x n_draws, '
                  'many draws will be repeated')

    if weights is not None:
        if seed is None:
            seed = secrets.randbits(31)
            print('load_draws: random draws with seed %d' % seed)
        sample_idx = np.sort(np.random.default_rng(seed).choice(n_retained, n_draws, replace=True, p=weights))
    elif n_draws >= n_retained:
        print('load_draws: %d draws asked for %d retained samples, using all of them' % (n_draws, n_retained))
        sample_idx = np.arange(n_retained)
    elif draw_mode == 'random':
        if seed is None:
            seed = secrets.randbits(31)
            print('load_draws: random draws with seed %d' % seed)
        sample_idx = np.sort(np.random.default_rng(seed).choice(n_retained, n_draws, replace=False))
    elif draw_mode == 'sequential':
        sample_idx = np.unique(np.linspace(0, n_retained-1, n_draws).round().astype(int))
    else:
        raise ValueError('load_draws: draw_mode must be "random" or "sequential", not %s' % draw_mode)

    return {'samples': samples[sample_idx], 'keys': keys, 'sidecar': sidecar,
            'sample_idx': sample_idx, 'n_retained': n_retained, 'seed': seed,
            'uniform_prior_keys': None if not uniform_prior_keys else list(uniform_prior_keys),
            'weight_ess': weight_ess}


def full_parameter_vectors(draws):
    '''
    Values of all the parameters of the chain model (frozen and linked ones included) for each draw, from the
    sidecar parameter table (chain start values) and the drawn free parameters.
    Links must be simple ("= p<n>"), which is what chain_runner models use.

    Returns (n_draws, n_parameters) array, column i is parameter i+1
    '''

    table = sorted(draws['sidecar']['parameters'], key=lambda row: row['key'])
    if any(not isinstance(row['key'], int) for row in table):
        raise NotImplementedError('full_parameter_vectors: named models (keys like "src:1") are not supported')

    vectors = np.tile([row['value'] for row in table], (len(draws['samples']), 1)).astype(float)
    vectors[:, np.array(draws['keys'])-1] = draws['samples']

    for row in table:
        if row['link'] == '':
            continue
        link_match = re.fullmatch(r'=\s*p?(\d+)', row['link'].strip())
        if link_match is None:
            raise NotImplementedError('full_parameter_vectors: parameter %d has a link expression (%s), only '
                                      '"= p<n>" links are supported' % (row['key'], row['link']))
        vectors[:, row['key']-1] = vectors[:, int(link_match.group(1))-1]

    return vectors


def keep_components(keep):
    '''
    Deletes (XSPEC delcomp) every component of the current model that is not in keep, and returns the indices
    (from 1, in the model before deletion) of the remaining parameters, in their new order.

    keep: iterable of component names as in AllModels(1).componentNames before deletion (e.g. 'pion_abs_3')
    '''

    model = AllModels(1)
    names = list(model.componentNames)

    missing = [elem for elem in keep if elem not in names]
    if missing:
        raise ValueError('keep_components: %s not in the model components %s' % (missing, names))

    #global parameter indices of each component (components are numbered in expression order)
    par_idx = {}
    i_par = model.startParIndex
    for name in names:
        n_comp_pars = len(getattr(model, name).parameterNames)
        par_idx[name] = list(range(i_par, i_par+n_comp_pars))
        i_par += n_comp_pars

    kept_names = [elem for elem in names if elem in keep]
    kept_pars = [i for name in kept_names for i in par_idx[name]]
    kept_par_names = [model(i).name for i in kept_pars]

    #highest index first, so that the indices of the components still to delete do not change
    for i_comp in sorted((names.index(elem)+1 for elem in names if elem not in keep), reverse=True):
        _pyXspec.doXspecCmd(['delcomp', str(i_comp)])

    model = AllModels(1)
    if model.nParameters != len(kept_pars) or \
            [model(i).name for i in range(1, model.nParameters+1)] != kept_par_names:
        raise RuntimeError('keep_components: unexpected parameters after delcomp (%s)' % model.expression)

    return kept_pars


def _set_parameters(par_map, vector):
    '''
    Sets the parameters of the current model from a full chain-model parameter vector.
    par_map[j-1] is the chain-model index of parameter j. Linked parameters are left to XSPEC
    '''

    model = AllModels(1)
    model.setPars({j: vector[i_chain-1] for j, i_chain in enumerate(par_map, start=1) if model(j).link == ''})


def plot_comp_ratio_MC(cont_addcomps, other_addcomps, ener_low, ener_high,
                       plot_type='eemo',
                       ylims=None,
                       cont_addcomps_xcm='', other_addcomps_xcm=None,
                       other_addcomps_labels=None, other_addcomps_colors=None, figsize=(10, 5),
                       other_addcomps_type=None, other_addcomps_alpha=None, other_addcomps_ls=None,
                       plot_transi=True, minor_locator=10, ylabel_prefix='', manual_bbox=False,
                       set_ener_str=None, set_ener_xrism=False, ylog=True, ax=None, plot_transi_kwargs=None,
                       chain_file=None, cont_keepcomps=None, other_keepcomps=None,
                       n_draws=1000, discard=0, thin=1, draw_mode='random', seed=None,
                       conf_level=(0.68, 0.95), band_alpha=(0.3, 0.15), band_type='eti',
                       center='bestfit',
                       combined_index=None, plot_combined_band=False, draws_cache=None, uniform_prior_keys=None,
                       other_addcomps_lw=None, transi_lw=None, legend_loc=None, transi_alternate=True,
                       ener_show=None):

    '''
    plot_comp_ratio with credible bands from an MCMC chain, and components selected by name.

    All the arguments of comp_ratio_plots.plot_comp_ratio are kept, with the same meaning, except that with
    chain_file the models come from the chain (cont_addcomps_xcm and other_addcomps_xcm must then be left empty).

    chain_file: None, or a chain_runner chain FITS file, or a list of them (same free parameters)
        None: no bands, best-fit lines only (plot_comp_ratio with the *_keepcomps options)
        otherwise every curve is computed from the chain's best-fit xcm (sidecar paths['bestfit_xcm']), restored
        from the run directory of the chain (sidecar provenance cwd: the xcm names its data and tables by
        relative path). The previous working directory is restored at the end

    cont_keepcomps: None or list of component names (as in AllModels(1).componentNames of the full model,
        e.g. ['powerlaw','diskbb']). The continuum is computed from the model with only these components kept.
        None keeps the full model. cont_addcomps then refers to the additive components of that reduced model
    other_keepcomps: None, or one element per element of other_addcomps: None (full model) or list of component
        names, as for cont_keepcomps. Requires an xcm (chain_file, or cont_addcomps_xcm/other_addcomps_xcm),
        since the components are deleted from the loaded model
        Multiplicative components wrapping everything (e.g. TBabs) can be kept everywhere: they cancel in the ratios

    n_draws, discard, thin, draw_mode, seed: parameter draws, see load_draws

    uniform_prior_keys: None, or keys of chain parameters sampled with a jeffreys (log-uniform) prior, whose prior
        is replaced by a uniform one (same hard limits) without rerunning the chain: the draws are taken from the
        chain samples with probabilities proportional to new prior / old prior = product of the parameter values
        (importance resampling, see load_draws). Useful when the data constrain a parameter weakly and the
        log-uniform prior, with its arbitrary lower bound, drives the result (e.g. a covering fraction, or the
        turbulent velocity of unresolved lines). The best-fit curves do not depend on the priors

    conf_level: credible level or tuple of levels, shared by all components. Default (0.68, 0.95)
    band_alpha: alpha of the bands, one per conf_level. Default (0.3, 0.15)
    band_type: 'eti' (equal-tailed, quantiles (1-c)/2 and (1+c)/2) or 'hdi' (highest density, arviz_stats.hdi)
    center: 'bestfit' (curves at the chain's best fit) or 'median' (pointwise median over the draws)
        The pointwise median curve is not the curve of any parameter set

    combined_index: None or index (in other_addcomps) of the element that is the combination of the others
    plot_combined_band: draw the bands of combined_index (off by default, they can hide the individual bands)

    draws_cache: None or path of a .npz file. If it exists and was made with the same settings, the curves are
        read from it instead of being recomputed; otherwise they are computed and saved there

    ener_show: None or (low, high) inside [ener_low, ener_high]: x range of the plot (and of the transition labels).
        The curves are still computed (or read from the cache) on ener_low-ener_high, so a sub-range of a cached
        range only redraws

    Returns (fig, ax, results), results a dict with x, cont_best, cont_draws, other_best, other_draws (lists,
    None for placeholder elements), ratio_center, ratio_bands, and draw_info (sample_idx, n_retained, seed, keys,
    uniform_prior_keys, weight_ess)
    '''

    n_comps = len(other_addcomps)

    other_keep_use = [None]*n_comps if other_keepcomps is None else list(other_keepcomps)
    if len(other_keep_use) != n_comps:
        raise ValueError('plot_comp_ratio_MC: %d other_keepcomps for %d other_addcomps'
                         % (len(other_keep_use), n_comps))

    conf_levels = [conf_level] if np.isscalar(conf_level) else list(conf_level)
    band_alphas = [band_alpha]*len(conf_levels) if np.isscalar(band_alpha) else list(band_alpha)
    if len(band_alphas) != len(conf_levels):
        raise ValueError('plot_comp_ratio_MC: %d band_alpha for %d conf_level' % (len(band_alphas), len(conf_levels)))
    if band_type not in ('eti', 'hdi'):
        raise ValueError('plot_comp_ratio_MC: band_type must be "eti" or "hdi", not %s' % band_type)
    if center not in ('bestfit', 'median'):
        raise ValueError('plot_comp_ratio_MC: center must be "bestfit" or "median", not %s' % center)
    show_low, show_high = (ener_low, ener_high) if ener_show is None else ener_show
    if not min(ener_low, ener_high) <= min(show_low, show_high) <= max(show_low, show_high) <= max(ener_low, ener_high):
        raise ValueError('plot_comp_ratio_MC: ener_show %s is not inside the computed range %s'
                         % ((show_low, show_high), (ener_low, ener_high)))

    #model source of every curve: (xcm path or None for the current model, tuple of kept components or None)
    if chain_file is not None:
        if cont_addcomps_xcm != '' or other_addcomps_xcm is not None:
            raise ValueError('plot_comp_ratio_MC: with chain_file, the models come from the chain\'s best-fit xcm. '
                             'Leave cont_addcomps_xcm and other_addcomps_xcm empty')
        first_file = chain_file if isinstance(chain_file, str) else chain_file[0]
        sidecar_path = find_sidecar(first_file)
        if sidecar_path is None:
            raise ValueError('plot_comp_ratio_MC: no sidecar (.json) found for %s' % first_file)
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        bestfit_xcm = sidecar['paths']['bestfit_xcm']
        run_dir = sidecar['provenance']['cwd']
        comps_xcm = [bestfit_xcm]*n_comps
        cont_xcm = bestfit_xcm
    else:
        run_dir = None
        if other_addcomps_xcm is None or isinstance(other_addcomps_xcm, str):
            comps_xcm = [other_addcomps_xcm]*n_comps
        else:
            comps_xcm = list(other_addcomps_xcm)
            if len(comps_xcm) != n_comps:
                raise ValueError('plot_comp_ratio_MC: %d other_addcomps_xcm for %d other_addcomps'
                                 % (len(comps_xcm), n_comps))
            if cont_addcomps_xcm == '':
                raise ValueError('plot_comp_ratio_MC: several other_addcomps_xcm require a separate cont_addcomps_xcm')
        cont_xcm = cont_addcomps_xcm if cont_addcomps_xcm != '' else comps_xcm[0]

    cont_source = (cont_xcm, None if cont_keepcomps is None else tuple(cont_keepcomps))
    comps_source = [(elem_xcm, None if elem_keep is None else tuple(elem_keep))
                    for elem_xcm, elem_keep in zip(comps_xcm, other_keep_use)]
    for elem_xcm, elem_keep in [cont_source]+comps_source:
        if elem_xcm is None and elem_keep is not None:
            raise ValueError('plot_comp_ratio_MC: *_keepcomps needs an xcm (the current model would be modified)')

    settings = json.dumps({'chain_file': chain_file, 'chain_size': None if chain_file is None else
                           [os.path.getsize(elem) for elem in ([chain_file] if isinstance(chain_file, str)
                                                                else chain_file)],
                           'n_draws': n_draws, 'discard': discard, 'thin': thin, 'draw_mode': draw_mode,
                           'seed': seed,
                           'cont_addcomps': cont_addcomps, 'other_addcomps': other_addcomps,
                           'cont_source': cont_source, 'comps_source': comps_source, 'plot_type': plot_type,
                           #floats: (2, 3) and (2., 3.) use the same cache
                           'set_ener': [set_ener_str, set_ener_xrism], 'ener': [float(ener_low), float(ener_high)],
                           #only when set, so that caches made before this option stay valid
                           **({} if not uniform_prior_keys else {'uniform_prior_keys': list(uniform_prior_keys)})},
                          default=str, sort_keys=True)

    curves = None
    if draws_cache is not None and os.path.exists(draws_cache):
        with np.load(draws_cache, allow_pickle=True) as cache:
            if str(cache['settings']) == settings:
                curves = cache['curves'].item()
                print('plot_comp_ratio_MC: curves read from %s' % draws_cache)
            else:
                print('plot_comp_ratio_MC: %s was made with other settings, recomputing' % draws_cache)

    if curves is None:
        draws = None if chain_file is None else \
            load_draws(chain_file, n_draws=n_draws, discard=discard, thin=thin, draw_mode=draw_mode, seed=seed,
                       uniform_prior_keys=uniform_prior_keys)
        curves = _compute_curves(cont_addcomps, other_addcomps, ener_low, ener_high, plot_type, cont_source,
                                 comps_source, set_ener_str, set_ener_xrism, draws, run_dir)
        if draws is not None:
            curves['draw_info'] = {elem: draws[elem] for elem in ('sample_idx', 'n_retained', 'seed', 'keys',
                                                                  'uniform_prior_keys', 'weight_ess')}
        if draws_cache is not None:
            np.savez(draws_cache, settings=settings, curves=np.array(curves, dtype=object))
            print('plot_comp_ratio_MC: curves saved to %s' % draws_cache)

    #ratios per draw, then their quantiles in each bin
    def ratio(num, den):
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    ratio_best = [None if elem is None else ratio(elem, curves['cont_best']) for elem in curves['other_best']]
    ratio_center = ratio_best
    ratio_bands = [None]*n_comps

    if chain_file is not None:
        ratio_draws = [None if elem is None else ratio(elem, curves['cont_draws']) for elem in curves['other_draws']]

        if center == 'median':
            ratio_center = [None if elem is None else np.median(elem, axis=0) for elem in ratio_draws]

        if band_type == 'hdi':
            import arviz_stats

        for i_comp, elem in enumerate(ratio_draws):
            if elem is None or (i_comp == combined_index and not plot_combined_band):
                continue
            ratio_bands[i_comp] = []
            for elem_level in conf_levels:
                if band_type == 'eti':
                    ratio_bands[i_comp].append(tuple(np.quantile(elem, [(1-elem_level)/2, (1+elem_level)/2], axis=0)))
                else:
                    elem_hdi = arviz_stats.hdi(elem, prob=elem_level, axis=0)
                    ratio_bands[i_comp].append((elem_hdi[..., 0], elem_hdi[..., 1]))

    fig, ax_comp = _plot_ratios(curves['x'], ratio_center, curves['x_axis_unit'], show_low, show_high, ylims,
                                other_addcomps_labels, other_addcomps_colors, figsize, other_addcomps_type,
                                other_addcomps_alpha, other_addcomps_ls, plot_transi, minor_locator, ylabel_prefix,
                                manual_bbox, ylog, ax, plot_transi_kwargs, bands=ratio_bands, band_alpha=band_alphas,
                                other_addcomps_lw=other_addcomps_lw, transi_lw=transi_lw, legend_loc=legend_loc,
                                transi_alternate=transi_alternate)

    results = dict(curves, ratio_center=ratio_center, ratio_bands=ratio_bands, conf_level=conf_levels,
                   band_type=band_type, center=center)

    return fig, ax_comp, results


def _compute_curves(cont_addcomps, other_addcomps, ener_low, ener_high, plot_type, cont_source, comps_source,
                    set_ener_str, set_ener_xrism, draws, run_dir):
    '''
    Best-fit and per-draw curves (restricted to the plotted energy range) of the continuum and of each element
    of other_addcomps. Each distinct model source (xcm, kept components) is loaded once, then all the draws are
    evaluated in it. See plot_comp_ratio_MC for the arguments
    '''

    vectors = None if draws is None else full_parameter_vectors(draws)

    def addcomp_ids(elem):
        return [elem] if isinstance(elem, (int, np.integer)) else elem

    cwd_orig = os.getcwd()
    prev_plot_add = Plot.add
    prev_chatter = (Xset.chatter, Xset.logChatter)
    Plot.add = True

    curves = {'x': None, 'cont_best': None, 'cont_draws': None,
              'other_best': [None]*len(other_addcomps), 'other_draws': [None]*len(other_addcomps)}
    try:
        if run_dir is not None:
            os.chdir(run_dir)

        for source in dict.fromkeys([cont_source]+comps_source):
            elem_xcm, elem_keep = source
            desc = ('the current model' if elem_xcm is None else elem_xcm) + \
                   ('' if elem_keep is None else ' (keeping %s)' % list(elem_keep))

            if elem_xcm is not None:
                Xset.restore(elem_xcm)

            #chain-model index of each parameter of the loaded model
            par_map = list(range(1, AllModels(1).nParameters+1)) if elem_keep is None else keep_components(elem_keep)

            if vectors is not None and elem_keep is None and len(par_map) != vectors.shape[1]:
                raise RuntimeError('the model of %s has %d parameters, the chain sidecar %d'
                                   % (desc, len(par_map), vectors.shape[1]))

            if set_ener_str is not None:
                set_ener(set_ener_str, xrism=set_ener_xrism)

            #what to read in this model: the continuum and/or some of the other_addcomps elements
            targets = []
            if source == cont_source:
                targets.append(('cont', cont_addcomps))
            targets += [(i_comp, addcomp_ids(elem)) for i_comp, elem in enumerate(other_addcomps)
                        if elem is not None and comps_source[i_comp] == source]

            Plot(plot_type)
            x_curr = np.array(Plot.x(1))
            if curves['x'] is None:
                x_sel = _ener_slice(x_curr, ener_low, ener_high)
                x_full = x_curr
                curves['x'] = x_curr[x_sel]
                curves['x_axis_unit'] = Plot.xAxis
            elif not np.array_equal(x_curr, x_full):
                raise ValueError('plot_comp_ratio_MC: the energy grid of %s differs from the first model. '
                                 'Use set_ener_str to put all the models on the same grid' % desc)

            #best fit: the parameters of the xcm (the chain start for a chain_runner xcm)
            best = {target: _sum_addcomps(ids, x_sel, desc) for target, ids in targets}

            if vectors is not None:
                Xset.chatter, Xset.logChatter = 0, 0
                draw_vals = {target: np.empty((len(vectors), len(curves['x']))) for target, ids in targets}
                for i_draw, vector in enumerate(tqdm(vectors, desc=os.path.basename(desc))):
                    _set_parameters(par_map, vector)
                    Plot(plot_type)
                    for target, ids in targets:
                        draw_vals[target][i_draw] = _sum_addcomps(ids, x_sel, desc)
                Xset.chatter, Xset.logChatter = prev_chatter

            for target, ids in targets:
                if target == 'cont':
                    curves['cont_best'] = best[target]
                    if vectors is not None:
                        curves['cont_draws'] = draw_vals[target]
                else:
                    curves['other_best'][target] = best[target]
                    if vectors is not None:
                        curves['other_draws'][target] = draw_vals[target]

    finally:
        Plot.add = prev_plot_add
        Xset.chatter, Xset.logChatter = prev_chatter
        if run_dir is not None:
            os.chdir(cwd_orig)
            print('plot_comp_ratio_MC: the loaded xspec model names its files relative to %s. '
                  'cd there before using it' % run_dir)

    return curves


def plot_comp_ratio_MC_broken(ener_ranges, ener_ranges_show=None, axis_stretch=None, panel_kwargs=None,
                              legend_panel=0, figsize=(10, 5), margins=None, wspace=0.05, d=0.008, xlabel_pad=22,
                              xtick_base=None, tight_pad=0.3, panel_func=None, **kwargs):
    '''
    plot_comp_ratio_MC on several energy ranges, side by side on one broken x axis (brokenaxes library).
    Each range is a separate plot_comp_ratio_MC call in its own panel, so each range keeps its own curves and cache.

    ener_ranges: list of (low, high) energy ranges of the curves (computed, or read from their caches), in
        increasing order
    ener_ranges_show: None (same as ener_ranges) or one (low, high) per panel, inside the matching ener_ranges
        element: x range shown in each panel (ener_show of plot_comp_ratio_MC). Cropping a cached range only redraws
    axis_stretch: None or one factor per range. None: 1 keV has the same width in every panel (panel widths
        proportional to the shown ranges). Each factor multiplies the width of its panel, e.g. (0.5, 1, 2): 1 keV is
        half as long in the first panel and twice as long in the third as in the second
    panel_kwargs: None, or one dict per range of arguments for that panel only (e.g. its draws_cache), added to kwargs
    legend_panel: index of the panel that keeps its legend (the legends of the other panels are removed)
    figsize: figure size
    margins: None or dict of fig.subplots_adjust arguments (left, right, bottom, top) of the group of panels,
        replacing the defaults left 0.07, right 0.98, bottom 0.12, top 0.8 (room for the transition labels above).
        Only the starting layout if tight_pad is not None
    tight_pad: None (keep the margins) or padding around the figure contents, in font size units: once everything
        is drawn, the margins are set so that the panels and all their labels (transition labels and legend
        included) fill the figure up to this padding, as fig.tight_layout (which does not work with brokenaxes)
    wspace: gap between the panels, as a fraction of the mean panel width
    d: length of the diagonal break marks, as a fraction of the figure width (see brokenaxes)
    xlabel_pad: distance of the x label below the panels, in points
    xtick_base: None, or one element per panel: None (the common spacing of the major x ticks, the largest automatic
        spacing of the panels, as brokenaxes) or the major tick spacing of that panel (e.g. 0.1 to get two tick labels
        in a panel narrower than the common spacing)
    panel_func: None (plot_comp_ratio_MC) or a function with the same return value, called for each panel as
        panel_func(ener_low=..., ener_high=..., ener_show=..., ax=..., **kwargs, **panel_kwargs[i])
    kwargs: arguments of panel_func shared by all the panels (chain_file, ylims, ylog, styles...). The panels share
        the y axis

    Only the first panel keeps its y label and tick labels. The x label is centered below the group.
    Returns (fig, bax, results): bax the brokenaxes.BrokenAxes (bax.axs: the panels), results the list of the
    results of each panel
    '''

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import AutoLocator, LogLocator, MultipleLocator
    from brokenaxes import brokenaxes

    n_ranges = len(ener_ranges)
    ranges_show = list(ener_ranges) if ener_ranges_show is None else list(ener_ranges_show)
    stretch = [1.]*n_ranges if axis_stretch is None else list(axis_stretch)
    panel_kwargs_use = [{}]*n_ranges if panel_kwargs is None else list(panel_kwargs)
    tick_bases = [None]*n_ranges if xtick_base is None else list(xtick_base)
    if len(ranges_show) != n_ranges or len(stretch) != n_ranges or len(panel_kwargs_use) != n_ranges or \
            len(tick_bases) != n_ranges:
        raise ValueError('plot_comp_ratio_MC_broken: ener_ranges_show, axis_stretch, panel_kwargs and xtick_base '
                         'need one element per range (%d ranges)' % n_ranges)
    if panel_func is None:
        panel_func = plot_comp_ratio_MC

    fig = plt.figure(figsize=figsize)
    #before brokenaxes: the break marks are drawn at the panel positions of its creation
    fig.subplots_adjust(**{'left': 0.07, 'right': 0.98, 'bottom': 0.12, 'top': 0.8, **(margins or {})})
    bax = brokenaxes(xlims=ranges_show, fig=fig, despine=False, d=d, wspace=wspace,
                     width_ratios=[(high-low)*elem for (low, high), elem in zip(ranges_show, stretch)])

    results = []
    for i_range, ((ener_low, ener_high), ener_show, ax) in enumerate(zip(ener_ranges, ranges_show, bax.axs)):
        results.append(panel_func(ener_low=ener_low, ener_high=ener_high, ener_show=ener_show, ax=ax,
                                  **{**kwargs, **panel_kwargs_use[i_range]})[2])
        if i_range == 0:
            xlabel = ax.get_xlabel()
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        if i_range != legend_panel and ax.get_legend() is not None:
            ax.get_legend().remove()

    #the panels changed their scales and limits: same tick spacing in all of them again, and no y tick labels in
    #the inner ones (as brokenaxes does after its own plotting calls)
    for ax in bax.axs:
        ax.xaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_locator(LogLocator() if ax.get_yscale() == 'log' else AutoLocator())
    bax.standardize_ticks()
    for ax, elem_base in zip(bax.axs, tick_bases):
        if elem_base is not None:
            ax.xaxis.set_major_locator(MultipleLocator(elem_base))
    bax.set_spines()

    #brokenaxes' default labelpad (15 points) overlaps the tick labels
    bax.set_xlabel(xlabel, labelpad=xlabel_pad)

    if tight_pad is not None:
        #fig.tight_layout ignores the panels (the brokenaxes gridspec is not attached to the figure): same idea by
        #hand, the margins of the group of panels move by the free space around everything drawn. Two passes: the
        #labels placed in axis fraction move with the panel widths. The gap between the panels (wspace) is kept
        width, height = fig.get_size_inches()
        pad = tight_pad*rcParams['font.size']/72
        for i_pass in range(2):
            box = fig.get_tightbbox()
            par = fig.subplotpars
            fig.subplots_adjust(left=par.left-(box.x0-pad)/width, right=par.right+(width-pad-box.x1)/width,
                                bottom=par.bottom-(box.y0-pad)/height, top=par.top+(height-pad-box.y1)/height)
        #the break marks are placed in figure coordinates when drawn: redrawn at the new panel positions
        for elem in bax.diag_handles:
            elem.remove()
        bax.draw_diags()

    return fig, bax, results


def plot_comp_ratio_MC_MAXIJ1543(n_draws=5000, uniform_priors=True, folded=False,
                                 ener_range=(6.38, 7.13),
                                 ylog=True, ylims=(0.,1.0),
                                 conf_level=(0.68, 0.95), band_alpha=(0.3, 0.15), lw=None, transi_lw=None,
                                 transi_alternate=False,
                                 legend_bbox='FeKa_FeXXVz', legend_loc='lower center', save_dir=None, version=None,
                                 ax=None, ener_show=None):
    '''
    MAXI J1543 photoionization plot (continuum, the three pion_abs zones, combined and emission) from the final
    long all-free chain (40 walkers x 197000 steps, first 10000 discarded: the discard of its OK verdict).
    Bands of the combined curve are off (combined_index=0).
    The curves are cached next to the chain (one cache per chain file, prior choice and folding), so a second call
    only redraws. Returns (fig, ax, results) of plot_comp_ratio_MC

    uniform_priors: if True (default, the priors of the results, same as the corner plots), the jeffreys priors of the covering fraction (pion_emit norm, parameter 22) and of the
        turbulent velocity of the three pion_abs zones (parameters 4, 8, 12) are replaced by uniform priors, by
        importance resampling of the chain (see plot_comp_ratio_MC / load_draws). Reasons: Omega/4pi is a fraction
        (uniform on 0-1 is the natural prior), and the pion_abs lines are unresolved (v_rms < ~35 km/s), so a
        log-uniform prior on v_rms mostly reflects its arbitrary 3 km/s lower bound; the three v_rms get the same
        prior for consistency (contour_maker.MAXIJ1543_UNIFORM_PRIOR_PARAMS, same choice for the corner plots).
        Test of 2026-10-24: Omega/4pi median 0.50 -> 0.75, pion_abs v_rms 95% upper limit 30 -> 35 km/s, the other
        parameters move by about 0.2 sigma or less. 2026-10-25, adding parameters 8 and 12: v_rms of pion_abs_3
        127 -> 134 km/s, of pion_abs_4 650 -> 686 km/s, Nh of pion_abs_3 7.0 -> 7.3 and of pion_abs_4 4.5 -> 4.2
        (all < 0.3 sigma), weight ESS fraction 0.50 -> 0.45. False = the jeffreys priors of the chain

    folded: False = unfolded model ratios (plot_type 'eemo': the intrinsic model on the 0.5 eV grid, unresolved
        lines appear with their intrinsic depth). True = ratios of the components folded through the Resolve
        response (plot_type 'data': what the instrument sees, lines broadened by the line-spread function, so the
        bands reflect what the data can distinguish). Both are unabsorbed: TBabs is removed from every curve.
        With folded=True, plot rebinning is switched off (Plot.setRebin(0, 1)) for the session, otherwise the folded
        curves would be grouped like the plotted data
    ener_range: (low, high) energy range in keV of the curves. Each range has its own cache (a new range recomputes
        the curves); ranges other than the default (6.38, 7.13) get a '_<low>-<high>keV' tag in the cache name
    ener_show: None (ener_range) or (low, high) inside ener_range: x range of the plot. Use it to show a part of an
        already computed range: the curves are read from the cache of ener_range. Shown ranges other than
        (6.38, 7.13) get a '_<low>-<high>keV' tag in the figure names
    ylog: log (True) or linear (False) y axis. Only the drawing changes: both scales use the same cached curves
    ylims: None for the defaults, (0.01, 1.2) in log and (0, 1.2) in linear, or a (ymin, ymax) pair
    conf_level, band_alpha: credible level(s) of the bands and their alpha, see plot_comp_ratio_MC
    lw: None (matplotlib default, 1.5) or line width of all the curves (single value or one per element)
    transi_lw: None (plot_transi default, 0.5) or width of the dashed grey transition lines (over the whole panel)
    transi_alternate: False (default here: the transition names fit on one row) or True (close transitions on a
        second row above the ion names), see plot_comp_ratio
    legend_bbox, legend_loc: legend position. legend_bbox: None (matplotlib 'best'), a point in axis fraction
        (the legend's legend_loc point is put there), or 'FeKa_FeXXVz' (default): horizontally halfway between the
        FeI Kalpha P3/2 and FeXXV z energies, at the bottom of the panel (with legend_loc='lower center', the same
        bottom margin as matplotlib's own lower positions), or a pair of fitting_tools.lines_e_dict names: the same,
        halfway between their energies ('FeKa_FeXXVz' is ('FeKa1p3em', 'FeKa25Zabs')). If the shown range does not
        contain both energies, the legend falls back to matplotlib's automatic placement ('best')
    Only the drawing changes with ylog, ylims, conf_level, band_alpha and lw: they reuse the cached curves
    save_dir: None, or directory (created if needed) where the figure is saved as
        comp_ratio_MC_<unfolded|folded>_<uniform|logu>_<log|lin>[_<low>-<high>keV][_<version>].pdf and .png
    version: None or a string appended to the file names (e.g. 'v2'), to keep earlier versions
    ax: None (new figure) or matplotlib axis to draw in (e.g. a panel of plot_comp_ratio_MC_MAXIJ1543_broken)

    Styles: zones numbered as in contour_maker.MAXIJ1543_ZONES and colored from the plasma colormap: zone 1
    (pion_abs_3) blue, zone 2 (pion_abs_4) red, zone 3 (pion_abs) orange; the zone 1 emission (pion_emit) dark grey
    (its bands look light grey). Best fits of the components dashed, combined model dotted black, without bands.

    Memory: reading the final chain (7.88M rows, 1.13 GB) peaks at about 3.4 GB on top of the XSPEC session with
    the pion tables (measured 2026-10-25 in contour_maker). thin=10 keeps 748,000 samples to draw from (tau 1540
    steps, so still about 150 per autocorrelation time) and avoids a further 1 GB copy of the full chain.
    Time per draw is unknown for the pion model: for an estimate, call with n_draws=50 first (its cache has other
    settings, so the full run recomputes).
    '''

    from matplotlib import colormaps

    posterior_dir = '/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion/posterior'
    chain_name = 'MAXIJ1543_3pion_all_free_long.fits'

    #TBabs multiplies every component, so it cancels in all the ratios: removed everywhere (including the
    #combined and emission elements, which would otherwise keep it while the continuum does not)
    def keep(*comps):
        return [*comps, 'powerlaw', 'diskbb']

    all_comps = keep('pion_abs', 'pion_abs_3', 'pion_abs_4', 'pion_emit')

    #v_rms of pion_abs, pion_abs_3, pion_abs_4 and the pion_emit norm (covering fraction)
    uniform_keys = [4, 8, 12, 22]

    plasma = colormaps['plasma']
    zone_colors = [plasma(0.05), plasma(0.55), plasma(0.82)]

    if folded:
        Plot.setRebin(0, 1)

    ener_low, ener_high = ener_range
    show_low, show_high = ener_range if ener_show is None else ener_show
    band_tag = '' if tuple(ener_range) == (6.38, 7.13) else '_%g-%gkeV' % (ener_low, ener_high)
    show_tag = '' if (show_low, show_high) == (6.38, 7.13) else '_%g-%gkeV' % (show_low, show_high)

    if isinstance(legend_bbox, str) and legend_bbox == 'FeKa_FeXXVz':
        legend_bbox = ('FeKa1p3em', 'FeKa25Zabs')
    if isinstance(legend_bbox, (tuple, list)) and all(isinstance(elem, str) for elem in legend_bbox):
        from fitting_tools import lines_e_dict
        ener_ref = [lines_e_dict[elem][0] for elem in legend_bbox]
        if show_low <= min(ener_ref) and max(ener_ref) <= show_high:
            legend_bbox = ((np.mean(ener_ref) - show_low) / (show_high - show_low), 0.)
        else:
            legend_bbox, legend_loc = None, None

    fig, ax, results = plot_comp_ratio_MC(
        [1, 2], [[1, 2, 3], [1, 2], [1, 2], [1, 2], [3]], ener_low, ener_high,
        plot_type='data' if folded else 'eemo',
        chain_file=os.path.join(posterior_dir, chain_name),
        discard=10000, thin=10, n_draws=n_draws, seed=1,
        cont_keepcomps=keep(),
        #order: combined, zone 1, zone 2, zone 3, zone 1 emission
        other_keepcomps=[all_comps, keep('pion_abs_3'), keep('pion_abs_4'), keep('pion_abs'), all_comps],
        combined_index=0, other_addcomps_type='abs',
        other_addcomps_labels=['combined', 'zone 1', 'zone 2', 'zone 3', 'zone 1 emission'],
        other_addcomps_colors=['black', *zone_colors, 'dimgrey'],
        other_addcomps_ls=[':', '--', '--', '--', '--'],
        ylabel_prefix='folded' if folded else 'unfolded',
        set_ener_str='thcomp', set_ener_xrism=True, ylog=ylog, ax=ax, ener_show=(show_low, show_high),
        conf_level=conf_level, band_alpha=band_alpha, other_addcomps_lw=lw, transi_lw=transi_lw,
        manual_bbox=False if legend_bbox is None else legend_bbox, legend_loc=legend_loc,
        transi_alternate=transi_alternate,
        ylims=ylims if ylims is not None else ((0.01, 1.2) if ylog else (0, 1.2)),
        uniform_prior_keys=uniform_keys if uniform_priors else None,
        #chain, prior choice and folding in the name: other caches (e.g. comp_ratio_MC_cache.npz, made from the
        #.bak_cont3 chain with jeffreys priors) are kept, not overwritten
        draws_cache=os.path.join(posterior_dir, 'comp_ratio_MC_cache_%s%s%s%s.npz'
                                 % (chain_name.replace('.fits', ''),
                                    '_uniform_' + '_'.join(str(elem) for elem in uniform_keys) if uniform_priors else '',
                                    '_folded' if folded else '', band_tag)))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.join(save_dir, 'comp_ratio_MC_%s_%s_%s' % ('folded' if folded else 'unfolded',
                                                                  'uniform' if uniform_priors else 'logu',
                                                                  'log' if ylog else 'lin')
                            + show_tag + ('' if version is None else '_' + version))
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png', dpi=200)
        print('plot_comp_ratio_MC_MAXIJ1543: saved %s.pdf/.png' % stem)

    return fig, ax, results


def plot_comp_ratio_MC_MAXIJ1543_broken(ener_ranges=((2, 3), (6.38, 7.13),(6.38, 7.13),(6.38, 7.13), (7.5, 9)),
                                        ener_ranges_show=((2.575, 2.675), (6.38, 6.46), (6.59, 6.75),(6.88,7.05),
                                                          (7.7, 8.8)),
                                        axis_stretch=[1,3,3,3,1],
                                        uniform_priors=True, folded=True, ylog=False,
                                        ylims=(0.,1),
                                        legend_panel=4,
                                        legend_bbox=('FeKg25Wabs', 'FeKg26p1abs'), legend_loc='lower center',
                                        xtick_base=(None, None, 0.1, 0.1, None),
                                        figsize=(14, 7), margins=None,
                                        save_dir='/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion/posterior/comp_plot_MC',
                                        version=None, **kwargs):
    '''
    plot_comp_ratio_MC_MAXIJ1543 on several energy ranges, side by side on one broken x axis
    (plot_comp_ratio_MC_broken). Each panel uses the cache of its range, so ranges already plotted alone only redraw.

    ener_ranges: list of (low, high) energy ranges in keV of the curves, in increasing order. They select the caches:
        keep them equal to already computed ranges (default: the three ranges cached on 2026-10-26) to only redraw
    ener_ranges_show: None (ener_ranges) or one (low, high) per panel, inside the matching ener_ranges element: x
        range shown in each panel
    axis_stretch: None (1 keV has the same width in every panel) or one width factor per range, see
        plot_comp_ratio_MC_broken
    uniform_priors, folded, ylog: as plot_comp_ratio_MC_MAXIJ1543. folded=True by default here: as of 2026-10-26 the
        caches of the three default ranges exist for the folded curves only (an unfolded call computes the missing
        ranges, about 30 min each)
    legend_panel, legend_bbox, legend_loc: panel with the legend, and its position in that panel (see
        plot_comp_ratio_MC_MAXIJ1543). Default: in the last panel, at the bottom, halfway between FeXXV Kgamma (w)
        and FeXXVI Kgamma (P1/2). The legend is about 2 inches wide: it needs a panel at least that wide
    xtick_base: None or one major x tick spacing (or None) per panel, see plot_comp_ratio_MC_broken. Default: 0.1 in
        the 6.59-6.75 and 6.88-7.05 panels, which the common spacing leaves with a single tick label
    figsize, margins: see plot_comp_ratio_MC_broken
    save_dir: None, or directory (created if needed) where the figure is saved as
        comp_ratio_MC_<unfolded|folded>_<uniform|logu>_<log|lin>_broken_<low>-<high>_..._<low>-<high>keV
        [_stretch<f1>-<f2>-...][_<version>].pdf and .png, with the shown ranges
    version: None or a string appended to the file names
    kwargs: other arguments of plot_comp_ratio_MC_MAXIJ1543 (n_draws, ylims, conf_level, band_alpha, lw, transi_lw,
        transi_alternate), shared by all the panels

    Returns (fig, bax, results) of plot_comp_ratio_MC_broken

        #previous attempts with

ener_ranges=((2, 3), (6.38, 7.13), (7.5, 9)),
                                        ener_ranges_show=((2.55, 2.65), (6.38, 7.03), (7.7, 8.8))



    '''

    def panel(ener_low, ener_high, ax, **panel_args):
        return plot_comp_ratio_MC_MAXIJ1543(ener_range=(ener_low, ener_high), ax=ax, **panel_args)

    ranges_show = ener_ranges if ener_ranges_show is None else ener_ranges_show

    fig, bax, results = plot_comp_ratio_MC_broken(ener_ranges, ener_ranges_show=ener_ranges_show,
                                                  axis_stretch=axis_stretch, legend_panel=legend_panel,
                                                  xtick_base=xtick_base,
                                                  figsize=figsize, margins=margins, panel_func=panel,
                                                  uniform_priors=uniform_priors, folded=folded, ylog=ylog,
                                                  legend_bbox=legend_bbox, legend_loc=legend_loc, ylims=ylims,
                                                  **kwargs)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.join(save_dir, 'comp_ratio_MC_%s_%s_%s_broken_%skeV'
                            % ('folded' if folded else 'unfolded', 'uniform' if uniform_priors else 'logu',
                               'log' if ylog else 'lin', '_'.join('%g-%g' % tuple(elem) for elem in ranges_show))
                            + ('' if axis_stretch is None else '_stretch' + '-'.join('%g' % elem for elem in axis_stretch))
                            + ('' if version is None else '_' + version))
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png', dpi=200)
        print('plot_comp_ratio_MC_MAXIJ1543_broken: saved %s.pdf/.png' % stem)

    return fig, bax, results

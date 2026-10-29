"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-22.
Design specification: docs/chain_runner_spec.md (written with Claude Opus 5.5 in claude.ai).
Reviewed and maintained by Maxime.

Chain-file reading and convergence/length checks used by chain_runner.run_chain.

Nothing here imports PyXspec, so these functions also work on a finished chain file
outside of an XSPEC session (e.g. from the corner-plot code).

Chain FITS layout (checked on XSPEC 13.0.1 / PyXspec 2.2.0, Goodman-Weare, see the
"layout" entry written in every sidecar):
    -one CHAIN binary table extension, header keywords CHAINTYP ('GoodmanWeare' or
     'MetropolisHastings') and NWALKERS
    -one column per free parameter, named <parname>__<globalidx> for the unnamed model and
     <parname>__<modelname>__<globalidx> for named models, then FIT_STATISTIC
    -the burn-in is NOT stored: the file holds exactly walkers*steps_per_walker rows
    -rows are step-major: row = step*n_walkers + walker
     (consecutive rows cycle through the walkers; a rejected move repeats the same values
     exactly n_walkers rows later)
    -the same xset seed gives a bit-identical file

The thresholds and their reasoning are listed in the chain_runner module docstring.
"""

import json
import math
import os

import numpy as np
from astropy.io import fits

#walker acceptance below this fraction of the ensemble median flags the walker as sticky
STICKY_ACCEPT_FRAC = 0.5

#typical GW acceptance range (Foreman-Mackey et al. 2013), advisory only
GW_ACCEPT_RANGE = (0.2, 0.5)

#the chain diag walker tau ratio threshold (tau_max/median over walkers)
MAX_WALKER_TAU_RATIO = 10.

#the other chain diag thresholds (Xspec/src/XSFit/MCMC/Diagnostics.h, HEASoft 6.37.1), used by chain_diag
CHAIN_DIAG_RHAT = 1.01
CHAIN_DIAG_ESS = 400.
CHAIN_DIAG_TAU_FRACTION = 50.

#which tau the length and burn rules of assess_chain use:
#    'walker_mean'  mean over walkers of the per-walker tau (chain diag tau_mean): the tau that sets the
#                   ESS (N_total/tau_mean matched the bulk ESS within 10% on the MAXI J1543 chain)
#    'walker_max'   largest per-walker tau (chain diag tau_max, the "tau" column of the chain diag table).
#                   One walker's tau is a noisy estimate, and the max over 40 of them is biased high: on
#                   the same chain it was 2-5 x tau_mean, and the 50 tau rule asked for 3.5 x the length
#                   that the ESS and the stability of the posterior between chain blocks justified
TAU_RULES = ('walker_mean', 'walker_max')


def parse_column_name(col_name):
    '''
    Maps a chain column name to the parameter key used everywhere in chain_runner:
        'nH__1'            -> 1          (unnamed model, global index)
        'PhoIndex__src__1' -> 'src:1'    (named model 'src', index 1 of that model)
    Returns None for non-parameter columns (FIT_STATISTIC)
    '''

    if col_name == 'FIT_STATISTIC':
        return None

    parts = col_name.split('__')
    if len(parts) == 2:
        return int(parts[1])
    if len(parts) == 3:
        return parts[1] + ':' + parts[2]

    raise ValueError('Unexpected chain column name: ' + col_name)


def read_chain(path):
    '''
    Reads an XSPEC chain FITS file into walker-separated arrays.

    Returns a dict with:
        samples:    array of shape (n_chains, n_steps, n_params)
                    GW: one chain per walker. MH and nest (CHAIN extension): a single chain.
                    HMC: one chain per CHAIN_ID of the HMC_DIAG extension, truncated to the
                    shortest chain.
        statistic:  array of shape (n_chains, n_steps)
        keys:       parameter keys in column order (int or 'model:idx', see parse_column_name)
        labels:     parameter names in column order
        chain_type: CHAINTYP header value ('' if absent)
        n_walkers:  NWALKERS header value (None if absent)
    '''

    with fits.open(path) as hdul:
        chain_hdu = hdul['CHAIN']
        header = chain_hdu.header
        data = chain_hdu.data
        cols = data.columns.names

        par_cols = [elem for elem in cols if elem != 'FIT_STATISTIC']
        flat = np.column_stack([np.asarray(data[elem], dtype=float) for elem in par_cols])
        stat = np.asarray(data['FIT_STATISTIC'], dtype=float)

        chain_type = str(header.get('CHAINTYP', '')).strip()
        n_walkers = header.get('NWALKERS', None)

        chain_ids = None
        if 'HMC_DIAG' in hdul and 'CHAIN_ID' in hdul['HMC_DIAG'].data.columns.names:
            chain_ids = np.asarray(hdul['HMC_DIAG'].data['CHAIN_ID'], dtype=int)

    n_rows, n_par = flat.shape

    if chain_type == 'GoodmanWeare' and n_walkers:
        n_walkers = int(n_walkers)
        if n_rows % n_walkers != 0:
            raise ValueError('%s: %d rows is not a multiple of NWALKERS=%d'
                             % (path, n_rows, n_walkers))
        n_steps = n_rows // n_walkers
        #step-major rows: (step, walker, par) -> (walker, step, par)
        samples = flat.reshape(n_steps, n_walkers, n_par).transpose(1, 0, 2)
        statistic = stat.reshape(n_steps, n_walkers).T

    elif chain_ids is not None:
        ids = sorted(set(chain_ids))
        per = [np.where(chain_ids == elem)[0] for elem in ids]
        n_steps = min(len(elem) for elem in per)
        samples = np.stack([flat[elem[:n_steps]] for elem in per])
        statistic = np.stack([stat[elem[:n_steps]] for elem in per])

    else:
        samples = flat[None, :, :]
        statistic = stat[None, :]

    return {'samples': samples,
            'statistic': statistic,
            'keys': [parse_column_name(elem) for elem in par_cols],
            'labels': [elem.split('__')[0] for elem in par_cols],
            'chain_type': chain_type,
            'n_walkers': n_walkers}


def walker_acceptance(samples):
    '''
    Acceptance fraction per chain/walker, from the fraction of steps where the position
    changed (a rejected GW stretch move repeats the previous position exactly).

    samples: (n_chains, n_steps, n_params) as returned by read_chain
    returns: (n_chains,) array

    This is not part of the chain diag output, but the spec asks to flag walkers whose
    acceptance is far below the ensemble's: such a walker is stuck (often in a low-probability
    region or a secondary mode) and biases the ensemble statistics (Foreman-Mackey et al. 2013).
    '''

    if samples.shape[1] < 2:
        return np.full(samples.shape[0], np.nan)

    moved = np.any(np.diff(samples, axis=1) != 0, axis=2)
    return moved.mean(axis=1)


def arviz_crosscheck(samples, keys):
    '''
    Rank-normalized R-hat and bulk/tail ESS per parameter computed with arviz, as an
    independent check of the XSPEC chain diag numbers (Vehtari et al. 2021 in both cases).

    Returns a dict {key: {'rhat':, 'ess_bulk':, 'ess_tail':}}, or {'error': message} if arviz
    is not importable or fails. Never raises: this is a cross-check, not the primary diagnostic.
    '''

    try:
        import arviz as az
    except ImportError as exc:
        return {'error': 'arviz not importable: %s' % exc}

    out = {}
    try:
        for i_par, key in enumerate(keys):
            #arviz wants (chain, draw) for a single variable
            arr = np.ascontiguousarray(samples[:, :, i_par])
            #the tail quantiles are passed explicitly: on numpy input, arviz-stats 1.3.3 does not
            #apply its default prob for method='tail' (TypeError: _ess_tail() missing 'prob')
            out[str(key)] = {'rhat': float(az.rhat(arr, method='rank')),
                             'ess_bulk': float(az.ess(arr, method='bulk')),
                             'ess_tail': float(az.ess(arr, method='tail', prob=(0.05, 0.95)))}
    except Exception as exc:
        return {'error': 'arviz failed: %r' % exc}

    return out


def integrated_tau(x, c=5.):
    '''
    Integrated autocorrelation time of each row of x (one walker per row), in steps, as chain diag
    computes it per walker (Diagnostics.cxx integratedTau): biased FFT autocorrelation, tau(m) =
    1 + 2 sum_{t=1..m} rho_t, emcee/Sokal automatic window m >= c tau(m), floor at 1.

    x: (n_walkers, n_steps). Returns an (n_walkers,) array.
    '''

    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    n_fft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(x - x.mean(axis=1, keepdims=True), n=n_fft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=n_fft, axis=1)[:, :n]
    del f

    out = np.ones(len(x))
    lags = np.arange(n)
    for i, row in enumerate(acov):
        if not row[0] > 0:
            continue
        taus = 2. * np.cumsum(row / row[0]) - 1.
        ok = (lags >= 1) & (lags >= c * taus)
        out[i] = max(taus[np.argmax(ok)] if ok.any() else taus[-1], 1.)
    return out


def chain_diag(samples, keys, discard=0, is_gw=True):
    '''
    Python version of XSPEC chain diag (AllChains.diagnostics()) on samples[:, discard:], for the
    checks after discarding the start of a chain: chain diag itself always uses the whole file.

    samples: (n_walkers, n_steps, n_params) from read_chain, keys: its parameter keys.
    Returns a dict with the keys of AllChains.diagnostics() that assess_chain uses (param_labels,
    rhat_split, rhat_rank_norm, bulk_ess, tail_ess, tau_mean, tau_max, walker_tau_ratio, n_sub_chains,
    sub_chain_length, verdict_status, failing_checks, is_gw), plus 'discard' and 'source'.

    R-hat and ESS come from arviz (Vehtari et al. 2021, as chain diag), tau from integrated_tau. On
    the full MAXI J1543 chain (discard=0) the result was compared with chain diag: see the
    chain_runner module docstring.
    '''

    import arviz as az

    discard = int(discard)
    sub = samples[:, discard:, :]
    n_w, n, n_par = sub.shape
    if n < 8:
        raise ValueError('only %d steps per walker left after discarding %d' % (n, discard))

    out = {'param_labels': [str(elem) for elem in keys],
           'rhat_split': [], 'rhat_rank_norm': [], 'bulk_ess': [], 'tail_ess': [],
           'tau_mean': [], 'tau_max': [], 'walker_tau_ratio': [],
           'n_sub_chains': [n_w] * n_par, 'sub_chain_length': [n] * n_par,
           'is_gw': bool(is_gw), 'discard': discard, 'source': 'chain_diagnostics.chain_diag (python)'}

    for i_par in range(n_par):
        arr = np.ascontiguousarray(sub[:, :, i_par])
        out['rhat_split'].append(float(az.rhat(arr, method='split')))
        out['rhat_rank_norm'].append(float(az.rhat(arr, method='rank')))
        out['bulk_ess'].append(float(az.ess(arr, method='bulk')))
        out['tail_ess'].append(float(az.ess(arr, method='tail', prob=(0.05, 0.95))))
        taus = integrated_tau(arr)
        out['tau_mean'].append(float(taus.mean()))
        out['tau_max'].append(float(taus.max()))
        out['walker_tau_ratio'].append(float(taus.max() / np.median(taus)) if is_gw and n_w > 1 else float('nan'))

    #verdict as in chain diag (Diagnostics.cxx aggregate): N/tau with N the total length
    labels = out['param_labels']
    failing = []
    i_max = int(np.argmax(out['rhat_rank_norm']))
    if out['rhat_rank_norm'][i_max] > CHAIN_DIAG_RHAT:
        failing.append('split-R-hat = %.3g > %g (par %s)' % (out['rhat_rank_norm'][i_max], CHAIN_DIAG_RHAT,
                                                             labels[i_max]))
    for ess_type, name in (('bulk_ess', 'bulk-ESS'), ('tail_ess', 'tail-ESS')):
        i_min = int(np.argmin(out[ess_type]))
        if out[ess_type][i_min] < CHAIN_DIAG_ESS:
            failing.append('%s = %.4g < %g (par %s)' % (name, out[ess_type][i_min], CHAIN_DIAG_ESS, labels[i_min]))
    frac = [n_w * n / elem for elem in out['tau_max']]
    i_min = int(np.argmin(frac))
    if frac[i_min] < CHAIN_DIAG_TAU_FRACTION:
        failing.append('N/tau = %.3g < %g (par %s)' % (frac[i_min], CHAIN_DIAG_TAU_FRACTION, labels[i_min]))
    if is_gw:
        i_max = int(np.nanargmax(out['walker_tau_ratio']))
        if out['walker_tau_ratio'][i_max] > MAX_WALKER_TAU_RATIO:
            failing.append('sticky walker: tau_max/tau_median = %.3g > %g (par %s)'
                           % (out['walker_tau_ratio'][i_max], MAX_WALKER_TAU_RATIO, labels[i_max]))

    out['verdict_status'] = 'WARN' if failing else 'OK'
    out['failing_checks'] = failing
    return out


def load_pilot_tau(pilot_path):
    '''
    Reads the per-parameter integrated autocorrelation times stored by a test (pilot) run.

    Returns (tau_dict, tau_reliable, free_keys) where tau_dict maps the parameter key (as a
    string, JSON keys are strings) to tau_max in steps per walker.
    '''

    with open(pilot_path) as f:
        pilot = json.load(f)

    diag = pilot.get('diagnostics', {})
    if 'tau_max' not in diag:
        raise ValueError('Pilot sidecar %s has no diagnostics.tau_max (status: %s)'
                         % (pilot_path, pilot.get('status')))

    free_keys = [str(elem['key']) for elem in pilot.get('parameters', []) if elem.get('free')]

    return diag['tau_max'], bool(diag.get('tau_reliable', False)), free_keys


def check_length_from_pilot(pilot_path, walkers, steps_per_walker, burn_per_walker,
                            min_tau_multiple=50, min_burn_tau=5, min_ess=400, free_keys=None):
    '''
    Pre-run length check from a pilot (test=True) run. Raises ValueError if the requested
    settings are too short, with the minimum values in the message.

    tau cannot be known before running, so it is taken from the pilot. A pilot shorter than
    min_tau_multiple*tau only gives a lower bound on tau (the estimate is biased low when the
    chain has not explored its correlation length), so in that case the planning tau is the
    pilot tau inflated by x2.

    Rules (tau = largest per-parameter tau_max, in steps per walker):
        -steps per walker after burn >= min_tau_multiple * tau
         (emcee autocorrelation guide: estimates of tau, and hence of the Monte Carlo error,
         are only trustworthy for chains longer than ~50 tau)
         NOTE: XSPEC's chain length already excludes the burn-in (the file holds only post-burn
         rows), so the requirement is on steps_per_walker alone, not 50 tau + burn.
        -burn per walker >= min_burn_tau * tau
         (several tau are needed to forget the starting ball; Hogg & Foreman-Mackey 2018)
        -predicted ESS = walkers * steps_per_walker / tau >= min_ess
         (bulk and tail ESS >= 400 for stable quantiles; Vehtari et al. 2021)

    Returns a dict with the planning numbers, written to the sidecar.
    '''

    tau_dict, tau_reliable, pilot_free = load_pilot_tau(pilot_path)

    tau_pilot = max(float(elem) for elem in tau_dict.values())
    tau_plan = tau_pilot if tau_reliable else 2. * tau_pilot

    req_steps = int(math.ceil(min_tau_multiple * tau_plan))
    req_burn = int(math.ceil(min_burn_tau * tau_plan))
    pred_ess = walkers * steps_per_walker / tau_plan
    req_steps_ess = int(math.ceil(min_ess * tau_plan / walkers))

    out = {'pilot': pilot_path,
           'tau_pilot_max': tau_pilot,
           'pilot_tau_reliable': tau_reliable,
           'tau_planning': tau_plan,
           'required_steps_per_walker': max(req_steps, req_steps_ess),
           'required_burn_per_walker': req_burn,
           'predicted_ess': pred_ess,
           'pilot_free_keys_match': None}

    if free_keys is not None and pilot_free:
        out['pilot_free_keys_match'] = sorted(pilot_free) == sorted(str(elem) for elem in free_keys)

    problems = []
    if steps_per_walker < req_steps:
        problems.append('steps_per_walker=%d < %d (= %g x tau %.3g)'
                        % (steps_per_walker, req_steps, min_tau_multiple, tau_plan))
    if burn_per_walker < req_burn:
        problems.append('burn_per_walker=%d < %d (= %g x tau %.3g)'
                        % (burn_per_walker, req_burn, min_burn_tau, tau_plan))
    if pred_ess < min_ess:
        problems.append('predicted ESS %.0f < %d: needs steps_per_walker >= %d with %d walkers'
                        % (pred_ess, min_ess, req_steps_ess, walkers))

    if problems:
        raise ValueError('Chain too short for the tau measured by the pilot %s'
                         '%s:\n  ' % (pilot_path, '' if tau_reliable else
                                      ' (pilot shorter than %g tau, tau inflated x2)' % min_tau_multiple)
                         + '\n  '.join(problems)
                         + '\nMinimum: steps_per_walker=%d, burn_per_walker=%d'
                         % (out['required_steps_per_walker'], req_burn))

    return out


def assess_chain(xspec_diag, steps_per_walker, burn_per_walker, acceptance=None,
                 min_tau_multiple=50, min_burn_tau=5, min_ess=400, max_rhat=1.01,
                 tau_rule='walker_mean', discard=0):
    '''
    Post-run verdict from the XSPEC chain diag results (AllChains.diagnostics()) plus the
    checks of the spec that chain diag does not apply in the same form.

    Checks (each failing one is listed in 'failing'):
        -the chain diag verdict itself (rank-normalized R-hat < 1.01, bulk/tail ESS >= 400,
         N >= 50 tau with N the TOTAL chain length, GW walker tau ratio < 10)
        -steps per walker >= min_tau_multiple * tau per parameter. Stricter than chain diag,
         which uses the total length: with an ensemble, the walkers are not independent chains,
         and the emcee guide asks for each walker to be longer than ~50 tau.
        -burn per walker >= min_burn_tau * tau
        -R-hat < max_rhat and bulk/tail ESS >= min_ess with the thresholds passed here (equal
         to the chain diag ones by default, so the two never disagree)
    Advisory only (listed in 'notes'):
        -walkers with acceptance < STICKY_ACCEPT_FRAC x the ensemble median
        -ensemble acceptance outside 0.2-0.5

    tau_rule: which tau the length and burn checks use, 'walker_mean' (default) or 'walker_max'
    (the rule before 2026-10-24), see TAU_RULES. 'tau_max' in the output is always the walker_max one.

    discard: steps per walker dropped from the START of the stored chain after the run (post-hoc
    burn-in). steps_per_walker is the stored length; the length check uses steps_per_walker - discard
    and the burn check burn_per_walker + discard. xspec_diag must then be computed on the chain without
    these steps (chain_diag(..., discard=discard)): XSPEC chain diag cannot drop them.

    A run is never marked OK on R-hat alone: every check above has to pass.
    If any check fails, the recommended length is discard + 1.5 x min_tau_multiple x tau per walker,
    with the tau just measured.
    '''

    if tau_rule not in TAU_RULES:
        raise ValueError('tau_rule must be one of %s' % (TAU_RULES,))
    discard = int(discard)
    steps_used = steps_per_walker - discard
    burn_total = burn_per_walker + discard

    labels = [str(elem) for elem in xspec_diag['param_labels']]
    tau_max = dict(zip(labels, [float(elem) for elem in xspec_diag['tau_max']]))
    tau_all = max(tau_max.values())
    tau_key = 'tau_mean' if tau_rule == 'walker_mean' else 'tau_max'
    tau_used = dict(zip(labels, [float(elem) for elem in xspec_diag[tau_key]]))
    tau_used_all = max(tau_used.values())

    failing = []
    notes = []

    if xspec_diag['verdict_status'] != 'OK':
        failing += ['chain diag [%s]: %s' % (xspec_diag['verdict_status'], elem)
                    for elem in xspec_diag['failing_checks']] \
            or ['chain diag verdict %s' % xspec_diag['verdict_status']]

    for i_par, lab in enumerate(labels):
        if steps_used < min_tau_multiple * tau_used[lab]:
            failing.append('par %s: steps per walker %d < %g x tau (%.3g)'
                           % (lab, steps_used, min_tau_multiple, tau_used[lab]))
        if burn_total < min_burn_tau * tau_used[lab]:
            failing.append('par %s: burn per walker %d < %g x tau (%.3g)'
                           % (lab, burn_total, min_burn_tau, tau_used[lab]))
        if not xspec_diag['rhat_rank_norm'][i_par] < max_rhat:
            failing.append('par %s: R-hat %.4f >= %g' % (lab, xspec_diag['rhat_rank_norm'][i_par], max_rhat))
        for ess_type in ['bulk_ess', 'tail_ess']:
            if not xspec_diag[ess_type][i_par] >= min_ess:
                failing.append('par %s: %s %.0f < %d' % (lab, ess_type, xspec_diag[ess_type][i_par], min_ess))
        if xspec_diag['is_gw'] and not xspec_diag['walker_tau_ratio'][i_par] < MAX_WALKER_TAU_RATIO:
            failing.append('par %s: walker tau ratio %.2f >= %g'
                           % (lab, xspec_diag['walker_tau_ratio'][i_par], MAX_WALKER_TAU_RATIO))

    sticky = []
    if acceptance is not None and len(acceptance) > 1:
        med = float(np.median(acceptance))
        sticky = [int(i) for i in np.where(acceptance < STICKY_ACCEPT_FRAC * med)[0]]
        if sticky:
            notes.append('walkers %s have acceptance < %g x ensemble median (%.3f)'
                         % (sticky, STICKY_ACCEPT_FRAC, med))
        mean_acc = float(np.mean(acceptance))
        if not GW_ACCEPT_RANGE[0] <= mean_acc <= GW_ACCEPT_RANGE[1]:
            notes.append('ensemble acceptance %.3f outside the typical %g-%g'
                         % (mean_acc, GW_ACCEPT_RANGE[0], GW_ACCEPT_RANGE[1]))

    out = {'verdict': 'OK' if not failing else 'WARN',
           'xspec_verdict': xspec_diag['verdict_status'],
           'failing': failing,
           'notes': notes,
           'tau_max': tau_max,
           'tau_max_all': tau_all,
           'tau_rule': tau_rule,
           'tau_used': tau_used,
           'tau_used_all': tau_used_all,
           'discard': discard,
           #used by the pilot planning (check_length_from_pilot), which keeps the largest walker tau
           'tau_reliable': bool(steps_used >= min_tau_multiple * tau_all),
           'rhat': dict(zip(labels, [float(elem) for elem in xspec_diag['rhat_rank_norm']])),
           'ess_bulk': dict(zip(labels, [float(elem) for elem in xspec_diag['bulk_ess']])),
           'ess_tail': dict(zip(labels, [float(elem) for elem in xspec_diag['tail_ess']])),
           'walker_tau_ratio': dict(zip(labels, [float(elem) for elem in xspec_diag['walker_tau_ratio']])),
           'sticky_walkers': sticky,
           'recommended_steps_per_walker': None,
           'thresholds': {'min_tau_multiple': min_tau_multiple, 'min_burn_tau': min_burn_tau,
                          'min_ess': min_ess, 'max_rhat': max_rhat,
                          'max_walker_tau_ratio': MAX_WALKER_TAU_RATIO}}

    if failing:
        out['recommended_steps_per_walker'] = discard + int(math.ceil(1.5 * min_tau_multiple * tau_used_all))
    if burn_total < min_burn_tau * tau_used_all:
        notes.append('the burn check passes with discard >= %d steps per walker (post-hoc, no rerun)'
                     % int(math.ceil(min_burn_tau * tau_used_all - burn_per_walker)))

    return out


def promote_reassessment(sidecar, entry):
    '''
    Makes a passing reassessment (an element of sidecar['reassessments']) the sidecar's 'diagnostics',
    i.e. the verdict shown by the corner plots and read by continue_chain as the previous verdict.
    The diagnostics it replaces, from chain diag on the whole file, are kept in 'diagnostics_whole_file'
    (not overwritten by a later promotion, only by a new whole-file one). Changes sidecar in place.
    '''

    if entry.get('verdict') != 'OK':
        raise ValueError('only an OK reassessment is promoted (this one: %s)' % entry.get('verdict'))
    current = sidecar.get('diagnostics')
    if current is not None and not current.get('discard'):
        sidecar['diagnostics_whole_file'] = current
    sidecar['diagnostics'] = entry


def reassess_chain(chain_file, discard, burn_per_walker=None, tau_rule='walker_mean', write=True,
                   promote_if_ok=True, min_tau_multiple=50, min_burn_tau=5, min_ess=400, max_rhat=1.01):
    '''
    Verdict of a finished gw chain file after discarding its first `discard` steps per walker, with
    chain_diag (python version of chain diag) and assess_chain. No PyXspec needed.

    burn_per_walker: the burn-in XSPEC dropped before writing the file; read from the sidecar
    <stem>.json when None. write=True appends the result to the sidecar's 'reassessments' list and,
    if the verdict is OK and promote_if_ok, also makes it the sidecar's 'diagnostics' (see
    promote_reassessment: the whole-file diagnostics move to 'diagnostics_whole_file'). The chain must
    then be used with the same discard. Do not write while a run or continue_chain is using the same
    file: they rewrite the sidecar at the end (continue_chain puts whole-file diagnostics back).

    Returns the assess_chain dict, with 'chain_diag' (the chain_diag output) and 'acceptance'.
    '''

    import datetime

    stem = chain_file[:-5] if chain_file.endswith('.fits') else chain_file
    sidecar_path = stem + '.json'
    sidecar = None
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as f:
            sidecar = json.load(f)
    if burn_per_walker is None:
        if sidecar is None:
            raise FileNotFoundError('no sidecar %s: pass burn_per_walker' % sidecar_path)
        burn_per_walker = int(sidecar['settings']['burn_per_walker'])

    chain = read_chain(chain_file)
    is_gw = chain['chain_type'] == 'GoodmanWeare'
    n_steps = chain['samples'].shape[1]
    xs = chain_diag(chain['samples'], chain['keys'], discard=discard, is_gw=is_gw)
    acc = walker_acceptance(chain['samples'][:, discard:]) if is_gw else None
    diag = assess_chain(xs, n_steps, burn_per_walker, acceptance=acc, min_tau_multiple=min_tau_multiple,
                        min_burn_tau=min_burn_tau, min_ess=min_ess, max_rhat=max_rhat,
                        tau_rule=tau_rule, discard=discard)
    diag['chain_diag'] = xs
    diag['acceptance'] = [float(elem) for elem in acc] if acc is not None else None

    if write and sidecar is not None:
        entry = {'date': datetime.datetime.now().isoformat(timespec='seconds'),
                 'steps_per_walker_total': n_steps, 'burn_per_walker': burn_per_walker}
        entry.update(diag)
        sidecar.setdefault('reassessments', []).append(entry)
        if promote_if_ok and diag['verdict'] == 'OK':
            promote_reassessment(sidecar, entry)
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=1)

    return diag

"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-22.
Assisted by Claude Code, model Sonnet 5.5 (claude-sonnet-5-5), 2026-10-22/23: cwd handling, jeffreys check, continue_chain().
Design specification: docs/chain_runner_spec.md (written with Claude Opus 5.5 in claude.ai).
Reviewed and maintained by Maxime.

run_chain(): runs an MCMC chain (or nested sampling / HMC) over every thawed parameter of the
model loaded in PyXspec 13 (or restored from an .xcm), and saves the chain together with enough
metadata to make the corner plot later without guessing. Plotting is NOT done here.

Each run writes, with a common stem:
    <stem>.fits         the chain as written by XSPEC
    <stem>.json         sidecar: provenance, sampler settings, parameter table (with priors and
                        absorber/continuum groups), best fit, profile errors, diagnostics
    <stem>_bestfit.xcm  the post-fit state (Xset.save, info='a'), i.e. the chain starting point
    <stem>.log          the XSPEC log of the run (chain diag report, warnings)

Scientific choices built in:
    -all parameters stay free by default, continuum included, so the absorber parameters are
     marginalized over the continuum. Freezing (freeze=) is explicit, logged, and flags the
     sidecar as conditional.
    -default sampler gw (Goodman-Weare). The pion grids are OGIP table models, which use
     finite-difference gradients, so hmc is opt-in only (force_hmc). nest for multimodal cases.
    -C-stat expected; a warning is printed otherwise.
    -profile-likelihood errors (profile_errors=) are computed BEFORE any chain is loaded: with a
     chain loaded, Fit.error returns chain credible intervals instead.

Calling run_chain changes the session: the new chain is loaded, and the Bayes settings and
seed are left as set by the run. Frozen flags are restored. PyXspec is a single global session:
do not wrap this in Python multiprocessing, use n_parallel (XSPEC's own parallel workers).

Chain settings: rules enforced (pre-run, from a pilot) and checked (post-run)
    Setting                       Rule                                       Basis
    walkers (gw)                  even; >= 2D, ideally 2D-4D                 XSPEC 13.0.1 manual Ch. 3;
                                                                             Foreman-Mackey et al. 2013
    steps per walker after burn   >= 50 tau                                  emcee autocorrelation guide;
                                                                             chain diag (N >= 50 tau)
    burn per walker               >= 5 tau                                   emcee guide; Hogg &
                                                                             Foreman-Mackey 2018
    effective sample size         bulk and tail ESS >= 400 per parameter    Vehtari et al. 2021; chain diag
    convergence                   rank-normalized R-hat < 1.01               Vehtari et al. 2021; chain diag
    walker consistency            walker tau ratio < 10; flag walkers with   chain diag; Foreman-Mackey
                                  acceptance far below the ensemble's        et al. 2013 (0.2-0.5 typical)
    thinning                      not done (not needed for the corner plot)  Hogg & Foreman-Mackey 2018
tau_max is the largest integrated autocorrelation time over parameters, in steps per walker.
Post-run, the tau of a parameter is the MEAN over walkers of the per-walker tau (chain diag tau_mean,
tau_rule='walker_mean' of assess_chain, since 2026-10-24). Before, it was the largest per-walker tau
(the "tau" column of chain diag): on the 97000-step MAXI J1543 chain that one was 2-5 x the mean,
the mean matched N_total/bulk ESS within 10%, and the rule asked for 335,574 steps per walker while
the posterior was already stable between chain blocks. The pilot planning below still uses the largest
per-walker tau (more conservative on a short pilot).
Burn-in can also be dropped after the run: chain_diagnostics.reassess_chain(file, discard) recomputes
the verdict without the first `discard` steps per walker (counted as burn), with chain_diag, a python
version of chain diag (it cannot drop steps). Checked on the same chain with discard=0: R-hat, tau and
walker tau ratio equal to chain diag's to 1e-12, bulk ESS within -2% (+11% on one parameter) and tail
ESS 2-13% lower (arviz vs XSPEC implementation of the ESS: the python verdict is the stricter one).
tau is unknown before running, hence two tiers:
    -pilot: test=True runs a short chain and stores tau per parameter in its sidecar, flagged
     tau_reliable=False when the pilot is shorter than 50 tau (the estimate is then a lower bound,
     and is inflated x2 for planning)
    -production: pilot=<pilot sidecar> is checked before sampling (see
     chain_diagnostics.check_length_from_pilot). Without a pilot, skip_length_check=True is required.
    -post-run: AllChains.diagnostics() (the chain diag suite) plus the per-walker checks of
     chain_diagnostics.assess_chain. A failure marks the sidecar diag WARN and prints the
     recommended rerun length (1.5 x 50 tau per walker).
R-hat in the AllChains.diagnostics() report: the first R-hat is the classical split R-hat
(Gelman et al. 2013), the second the rank-normalized R-hat of Vehtari et al. 2021
(doi:10.1214/20-BA1221, rhat_rank_norm in the returned dict). The second is the one to keep
below 1.01, as recommended by Vehtari et al. themselves, and the one checked here (max_rhat).
The default thresholds equal the chain diag ones so the two never disagree. Note that chain diag's
N >= 50 tau uses the TOTAL length N, while the rule above is per walker (stricter).
For D ~ 16, affine-invariant ensemble samplers lose efficiency with dimension (Huijser et al. 2022):
expect long tau. A long tau is a reason to run longer or switch to nest, not to loosen thresholds.

Priors (priors= dict or JSON path; keys = XSPEC global index as in show param, 'model:idx' for
named models, or 'component.parameter'; values = [family, *hyperparameters]),
following the 13.0.1 manual Sec. 3.2.2-3.2.3:
    -scale parameters whose order of magnitude is uncertain (normalizations, column densities
     spanning decades) -> jeffreys (uniform in log theta), the uninformative default
    -lognormal is informative (a center and a width in ln theta): only with independent prior
     information, never with settings tuned on the same data
    -parameters already stored as log10 in a table (e.g. log xi, possibly the pion columns) -> cons,
     since flat in log is already log-uniform. Never jeffreys on a log parameter
    -location parameters with physical limits (Gamma, Tin, velocities) -> cons, or gauss/cauchy
     from an independent measurement
    -bounded parameters (covering fractions) -> beta, or cons on [0, 1]
When priors are given they are authoritative: free parameters not listed get a flat (cons) prior
within their hard limits, and this is logged. Bayesian inference is switched on only after all
priors are set, and checked afterwards.

API check (XSPEC 13.0.1, PyXspec 2.2.0, HEASoft 6.37.1), names and signatures used here:
    xspec.runChain(file_name, length=None, walkers=None, burn=None, algorithm=None,
                   proposal=None, temperature=None, rand=None, rescale=None) -> ChainResult
        length/burn are totals (all walkers), length is post-burn
    xspec.runHMC(file_name, samples=None, warmup=None, chains=None, ..., overwrite=True) -> HMCResult
    xspec.runNested(file_name, nLive=400, dlogZTol=0.5, maxClusters=16, enlargement=1.2,
                    resampleN=4000, overwrite=True, checkpointEvery=200, parallel=None) -> NestedResult
        no seed argument: 'nest seed <N>' is issued through _pyXspec.doXspecCmd (as runNested
        itself does for its other settings); nest does not use the xset seed generator
    Fit.globalFit(maxGen=None, popSize=None); Fit.globalBasins: list of (stat, deltaStat, params)
    Fit.bayes ('on'/'off'/'cons'); Fit.bayesPriors (listing, GET only)
    Parameter.prior = family or (family, *hyperpars): passed to XSPEC unchecked (all 13 families
        work, although the docstring lists only 4)
    AllChains.diagnostics() -> dict (param_labels, rhat_rank_norm, bulk_ess, tail_ess, tau_mean,
        tau_max, walker_tau_ratio, n_sub_chains, sub_chain_length, verdict_status, failing_checks...)
    Xset.seed (SET only), Xset.parallel.walkers, Xset.openLog/closeLog/log,
    Xset.save(fileName, info='a'), Xset.restore, Model.startParIndex, AllModels.sources
    'parallel hmc <N>' has no PyXspec attribute: issued through _pyXspec.doXspecCmd

References (written from memory in the spec, to be checked before going into a paper):
    -XSPEC 13.0.1 manual, Chapter 3 and the chain, chain diag, bayes, hmc, nest entries
     (docs/XSPEC13.0.1_manual_MCMC_excerpts.pdf)
    -Goodman & Weare 2010, Comm. App. Math. Comp. Sci. 5, 65, doi:10.2140/camcos.2010.5.65
    -Foreman-Mackey et al. 2013, PASP 125, 306, doi:10.1086/670067, arXiv:1202.3665
    -emcee documentation, Autocorrelation analysis & convergence tutorial (emcee.readthedocs.io)
    -Vehtari et al. 2021, Bayesian Analysis 16, 667, doi:10.1214/20-BA1221, arXiv:1903.08008
    -Gelman & Rubin 1992, Statistical Science 7, 457, doi:10.1214/ss/1177011136
    -Gelman et al. 2013, Bayesian Data Analysis, 3rd ed., CRC Press (split R-hat)
    -Hogg & Foreman-Mackey 2018, ApJS 236, 11, doi:10.3847/1538-4365/aab76e, arXiv:1710.06068
    -Huijser, Goodman & Brewer 2022, Aust. N. Z. J. Stat., arXiv:1509.02230
    -Cash 1979, ApJ 228, 939, doi:10.1086/156922
    -ArviZ documentation (R-hat, ESS, MCSE), python.arviz.org
"""

import datetime
import json
import math
import os
import secrets
import shutil
import socket
import sys
import time

import numpy as np

import xspec
from xspec import AllChains, AllData, AllModels, Fit, Xset
from xspec import _pyXspec

#flat import of the sibling module, as elsewhere in the repo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chain_diagnostics import (arviz_crosscheck, assess_chain, check_length_from_pilot,
                               read_chain, walker_acceptance)

#number of hyperparameters per bayes prior family (bayes command entry, XSPEC 13.0.1 manual)
PRIOR_NARGS = {'cons': 0, 'exp': 1, 'jeffreys': 0, 'gauss': 2, 'lognormal': 2, 'gamma': 2,
               'cauchy': 2, 'beta': 2, 'studentst': 1, 'halfnormal': 1, 'powerlaw': 3,
               'isotropic': 0, 'shiftedlognormal': 3}

#families requiring a strictly positive support
POSITIVE_PRIORS = ('jeffreys', 'lognormal')

#test=True chain length
TEST_STEPS, TEST_BURN = 100, 20

#proposal used if the fit covariance cannot initialize the walkers
FALLBACK_PROPOSAL = 'uniform deltas 10'

#XSPEC log messages of a failed walker initialization from the fit covariance
COVARIANCE_FAILURE_MESSAGES = ('Unable to use fit covariance matrix for randomization',
                               'Cannot use the requested proposal distribution for walker initialization')


def _check_no_ipython(n_parallel, caller):
    '''
    XSPEC's parallel workers crash an IPython session (e.g. the PyCharm Python console with IPython):
    refuse n_parallel > 1 there, before anything is touched
    '''
    if n_parallel is None or int(n_parallel) <= 1:
        return
    ipy = sys.modules.get('IPython')
    if ipy is not None and ipy.get_ipython() is not None:
        raise RuntimeError('%s: n_parallel=%s inside an IPython session, where XSPEC parallel runs crash.'
                           ' Run from a plain Python interpreter, or set n_parallel=None.'
                           % (caller, n_parallel))


def _say(summary, msg):
    print('[run_chain] ' + msg)
    summary['log'].append(msg)


def _choose(summary, name, value, why):
    '''
    Records a value the function picked itself (no silent defaults)
    '''
    summary['auto_choices'][name] = value
    _say(summary, 'auto: %s = %s (%s)' % (name, value, why))


def _log_size(log_path):
    try:
        return os.path.getsize(log_path)
    except OSError:
        return 0


def _log_since(log_path, offset):
    '''
    XSPEC log text written after offset (empty string if the log cannot be read)
    '''
    try:
        if hasattr(Xset.log, 'flush'):
            Xset.log.flush()
        with open(log_path, errors='replace') as f:
            f.seek(offset)
            return f.read()
    except Exception:
        return ''


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(elem) for elem in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _write_json(path, content):
    with open(path, 'w') as f:
        json.dump(_json_safe(content), f, indent=1)


def _par_key(model_name, gidx):
    '''
    Parameter key: XSPEC global index (int) for the unnamed model, 'name:idx' for named models,
    i.e. the same form as in show param, freeze, error and the chain column names
    '''
    return gidx if model_name == '' else '%s:%d' % (model_name, gidx)


def _span_decades(row):
    '''
    Decades spanned by the hard-limit range, for scale-like parameters:
        log10(max/min) if min > 0
        log10(max/|delta|) if min == 0 (a range starting at 0, e.g. norms and columns)
        None if min < 0 (location parameter crossing zero, not scale-like)
    '''
    lo, hi, delta = row['hard_min'], row['hard_max'], abs(row['delta'])
    if hi <= 0 or lo < 0:
        return None
    if lo > 0:
        return math.log10(hi / lo)
    return math.log10(hi / delta) if delta > 0 else None


def _limit_position(row):
    '''
    Position of the value within its hard limits, in [0, 1]. Computed in log space for
    positive ranges spanning more than 2 decades (otherwise a norm at 1e-2 in [0, 1e24] would
    always look "at the lower limit"), linearly otherwise.
    '''
    lo, hi, val, delta = row['hard_min'], row['hard_max'], row['value'], abs(row['delta'])
    if hi == lo:
        return 0.

    span = _span_decades(row)
    if span is not None and span > 2:
        lo_eff = lo if lo > 0 else delta
        if val <= lo_eff:
            return 0.
        return (math.log10(val) - math.log10(lo_eff)) / (math.log10(hi) - math.log10(lo_eff))

    return (val - lo) / (hi - lo)


def build_param_table(absorber_components=('mtable', 'pion_abs')):
    '''
    Parameter table over every model source and data group of the current session.

    One row per parameter: key (global index, or 'model:idx'), datagroup, model name, component,
    parameter name, unit, value, delta, soft/hard limits, frozen, link, free (= not frozen and
    not linked) and group ('absorber' if the component name starts with one of
    absorber_components, 'continuum' otherwise).
    '''

    rows = []
    for mod_name in [AllModels.sources[elem] for elem in sorted(AllModels.sources)]:
        for i_grp in range(1, AllData.nGroups + 1):
            try:
                mod = AllModels(i_grp, mod_name)
            except Exception:
                #this model is not attached to this data group
                continue

            for comp_name in mod.componentNames:
                comp = getattr(mod, comp_name)
                for par_name in comp.parameterNames:
                    par = getattr(comp, par_name)
                    vals = par.values
                    link = par.link.strip()
                    rows.append({'key': _par_key(mod_name, mod.startParIndex + par.index - 1),
                                 'datagroup': i_grp,
                                 'model': mod_name,
                                 'local_index': par.index,
                                 'component': comp_name,
                                 'name': par_name,
                                 'unit': par.unit,
                                 'value': vals[0],
                                 'delta': vals[1],
                                 'hard_min': vals[2], 'soft_min': vals[3],
                                 'soft_max': vals[4], 'hard_max': vals[5],
                                 'frozen': par.frozen,
                                 'link': link,
                                 'free': (not par.frozen) and link == '',
                                 'group': 'absorber' if any(comp_name.lower().startswith(elem.lower())
                                                            for elem in absorber_components)
                                 else 'continuum'})
    return rows


def _get_par(row):
    return AllModels(row['datagroup'], row['model'])(row['local_index'])


def _refresh_values(table):
    for row in table:
        par = _get_par(row)
        row['value'], row['delta'] = par.values[0], par.values[1]


def _resolve_key(key, table):
    '''
    Resolves a user-given parameter reference (int, '3', 'model:idx' or 'component.parameter')
    to a row of the table
    '''

    by_key = {str(row['key']): row for row in table}

    if isinstance(key, (int, np.integer)) or (isinstance(key, str) and key.isdigit()):
        key = str(int(key))
    if str(key) in by_key:
        return by_key[str(key)]

    if isinstance(key, str) and '.' in key:
        comp_name, par_name = key.split('.', 1)
        matches = [row for row in table if row['component'] == comp_name and row['name'] == par_name
                   and row['link'] == '']
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError('"%s" is ambiguous (keys %s): use the global index instead'
                             % (key, [row['key'] for row in matches]))

    raise KeyError('Parameter "%s" not found in the current model (keys %s...)'
                   % (key, [row['key'] for row in table[:5]]))


def load_priors(priors, table):
    '''
    Reads priors (dict, or path to a JSON file kept in the project so prior choices are
    version-controlled) and resolves them to {row key: (family, [hyperparameters])}.

    Validation:
        -unknown family or wrong number of hyperparameters (bayes command entry) -> raise
        -prior on a frozen or linked parameter -> raise
        -jeffreys/lognormal on a parameter whose hard lower limit is <= 0 -> raise: both need a
         strictly positive support; set a small positive limit with newpar first
        -jeffreys on a parameter stored as a log (name containing 'log', e.g. rlogxi) -> raise: flat in
         log is already log-uniform, use cons
    '''

    if priors is None:
        return {}

    source = None
    if isinstance(priors, str):
        source = priors
        with open(priors) as f:
            priors = json.load(f)

    resolved = {}
    for user_key, spec in priors.items():
        row = _resolve_key(user_key, table)

        if isinstance(spec, str):
            spec = [spec]
        family, hyper = str(spec[0]).lower(), [float(elem) for elem in spec[1:]]

        if family not in PRIOR_NARGS:
            raise ValueError('Unknown prior family "%s" for %s. Known: %s'
                             % (family, user_key, ', '.join(PRIOR_NARGS)))
        if len(hyper) != PRIOR_NARGS[family]:
            raise ValueError('Prior %s for %s takes %d hyperparameter(s), got %d'
                             % (family, user_key, PRIOR_NARGS[family], len(hyper)))
        if not row['free']:
            raise ValueError('Prior set on %s (%s.%s), which is frozen or linked'
                             % (row['key'], row['component'], row['name']))
        if family in POSITIVE_PRIORS and row['hard_min'] <= 0:
            raise ValueError('%s prior on %s (%s.%s) needs a positive hard lower limit, currently %g.'
                             ' Set a small positive limit with newpar first.'
                             % (family, row['key'], row['component'], row['name'], row['hard_min']))
        #'in', not startswith, so that the pion 'rlogxi' is caught too
        if family == 'jeffreys' and 'log' in row['name'].lower():
            raise ValueError('jeffreys prior on %s (%s.%s), a log parameter: flat in log is already'
                             ' log-uniform, use cons.' % (row['key'], row['component'], row['name']))

        resolved[row['key']] = (family, hyper)

    if source is not None:
        resolved['_source'] = source

    return resolved


def _check_nest_priors(table, resolved):
    '''
    nest needs a proper, bounded prior on every free parameter: XSPEC refuses flat priors over
    hard ranges spanning more than 4 decades. Raise listing the offenders.
    '''
    bad = []
    for row in table:
        if not row['free'] or row['key'] in resolved:
            continue
        span = _span_decades(row)
        if span is not None and span > 4:
            bad.append('%s (%s.%s, flat over %.1f decades)' % (row['key'], row['component'],
                                                               row['name'], span))
    if bad:
        raise ValueError('sampler="nest" needs a proper bounded prior on every free parameter.'
                         ' Set jeffreys (with a positive hard lower limit) or narrow the limits'
                         ' with newpar for: ' + ', '.join(bad))


def _noticed_ranges(spec):
    '''
    Noticed energy ranges of a spectrum as [[e_lo, e_hi], ...] in keV
    '''
    #Spectrum.energies only lists the noticed channels, in the order of Spectrum.noticed
    ranges = []
    for chan, (e_lo, e_hi) in zip(spec.noticed, spec.energies):
        if ranges and chan == prev + 1:
            ranges[-1][1] = e_hi
        else:
            ranges.append([e_lo, e_hi])
        prev = chan
    return ranges


def _provenance(xcm, set_ener_str, set_ener_xrism):
    spectra = []
    for i_sp in range(1, AllData.nSpectra + 1):
        spec = AllData(i_sp)
        entry = {'index': i_sp, 'file': spec.fileName, 'datagroup': spec.dataGroup,
                 'noticed_keV': _noticed_ranges(spec)}
        try:
            entry['rmf'] = spec.response.rmf
            entry['arf'] = spec.response.arf
        except Exception:
            pass
        spectra.append(entry)

    return {'pyxspec_version': Xset.version[0], 'xspec_version': Xset.version[1],
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'xcm_input': os.path.abspath(xcm) if xcm is not None else None,
            'set_ener': None if set_ener_str is None else {'mode': set_ener_str, 'xrism': set_ener_xrism},
            'statistic': Fit.statMethod,
            'spectra': spectra}


def run_chain(
        outfile,                 # path of the chain file to write (.fits)
        xcm=None,                # optional .xcm to restore (data + model + settings)
        xcm_set_ener_str=None,   # optional mode of xspec_config_multisp.set_ener, applied after loading
        xcm_set_ener_xrism=True, # xrism argument of set_ener
        sampler="gw",           # "gw" | "nest" (| "hmc", opt-in only)
        walkers=None,            # gw only; default = smallest even number >= 2*D
        steps_per_walker=2000,   # gw only; total length = walkers * steps_per_walker
        burn_per_walker=500,     # gw only; total burn = walkers * burn_per_walker
        seed=None,               # int; None draws one and records it
        n_parallel=None,         # parallel workers for the chain; None = serial
        priors=None,             # dict or JSON path; keys = index or "comp.par"; see Priors
        prior_variants=None,     # list of alternative prior files for sensitivity reruns
        freeze=None,             # list of param indices to freeze for THIS run only
        refit=True,              # run Fit.perform() twice before sampling
        global_fit=False,        # run the DE global optimizer first (multimodality check)
        profile_errors=None,     # list of param indices for Fit.error before the chain
        save_xcm=True,           # write the post-fit state next to the chain
        overwrite=False,
        test=False,              # short chain: steps_per_walker=100, burn=20
        absorber_components=("mtable", "pion_abs"),  # component-name prefixes tagged "absorber"
                                 # (table components are named after the table's MODLNAME,
                                 # e.g. pion_abs, pion_abs_3, not "mtable")
        force_hmc=False,         # required to run hmc on table models
        pilot=None,              # sidecar of a test run, gives tau for length checks
        skip_length_check=False, # proceed without a pilot (logged)
        min_tau_multiple=50,     # steps after burn >= 50 * tau
        min_burn_tau=5,          # burn >= 5 * tau
        min_ess=400,             # bulk and tail ESS per parameter
        max_rhat=1.01,           # rank-normalized R-hat
) -> dict:                      # summary: paths, seed, D, walkers, diag verdict
    '''
    Runs a chain over all thawed parameters of the current (or restored) model and writes the
    chain, a JSON sidecar, the best-fit xcm and the XSPEC log with a common stem. See the module
    docstring for the rules, priors guidance and outputs.

    outfile: chain path; the stem (without .fits) names the other outputs
    xcm: restored with Xset.restore, from the xcm's own directory so its relative data paths
         work. This replaces the current session. The run stays in that directory until it ends
         (the cwd is restored afterwards): the pion tables are opened by relative name.
    xcm_set_ener_str, xcm_set_ener_xrism: if xcm_set_ener_str is not None,
         set_ener(xcm_set_ener_str, xcm_set_ener_xrism) from xspec_config_multisp is run after
         loading (model energy grid, e.g. the extended grid needed by convolution models)
    sampler: 'gw' (default), 'nest', or 'hmc' (requires force_hmc=True)
    walkers, steps_per_walker, burn_per_walker: gw only, expressed per walker so the totals are
         automatically divisible by the walker count. Ignored for nest and hmc, whose own settings
         (runNested / runHMC defaults) are logged instead.
    seed: xset seed (and nest seed); drawn with secrets.randbits(31) if None, always recorded
    n_parallel: Xset.parallel.walkers (gw), parallel nest (nest) or parallel hmc (hmc)
    priors: see the module docstring. prior_variants: list of alternative prior files, each rerun
         with the same seed and settings into <stem>_prior-<variant>.fits
    freeze: parameter keys frozen for this run only (conditional chain), restored at the end
    profile_errors: keys for Fit.error at delta-stat 1.0, computed before any chain is loaded
    test: short pilot chain (100 steps, 20 burn per walker); its sidecar gives tau for pilot=
    pilot / skip_length_check / min_* / max_rhat: length checks, see the module docstring

    Every value picked by the function (walkers, seed, priors left flat, proposal fallback) is
    printed and written to the sidecar.
    Returns the summary dict (also the core of the sidecar).
    '''

    _check_no_ipython(n_parallel, 'run_chain')

    if sampler not in ('gw', 'nest', 'hmc'):
        raise ValueError('sampler must be "gw", "nest" or "hmc", not "%s"' % sampler)

    if sampler == 'hmc' and not force_hmc:
        raise ValueError('sampler="hmc": the pion mtables use finite-difference gradients (about D+1'
                         ' likelihoods per gradient, noisy on tabulated models). Pass force_hmc=True'
                         ' to run it anyway.')

    if test:
        steps_per_walker, burn_per_walker = TEST_STEPS, TEST_BURN

    #all paths absolute before any chdir
    outfile = os.path.abspath(outfile)
    stem = outfile[:-5] if outfile.endswith('.fits') else outfile
    paths = {'chain': outfile, 'sidecar': stem + '.json', 'log': stem + '.log'}
    if save_xcm:
        paths['bestfit_xcm'] = stem + '_bestfit.xcm'
    if pilot is not None:
        pilot = os.path.abspath(pilot)
    if xcm is not None:
        xcm = os.path.abspath(xcm)
    cwd_orig = os.getcwd()

    run_kwargs = dict(locals())

    existing = [elem for elem in paths.values() if os.path.exists(elem)]
    if existing and not overwrite:
        raise FileExistsError('Refusing to overwrite %s (pass overwrite=True)' % existing)

    summary = {'status': 'starting', 'paths': paths, 'sampler': sampler, 'test': test,
               'log': [], 'auto_choices': {}}

    frozen_orig = {}
    log_opened = False
    table = []

    #no interactive "continue fitting?" prompt during the run (it blocks unattended runs)
    query_orig = Fit.query
    Fit.query = 'yes'
    _choose(summary, 'Fit.query', 'yes', 'no interactive prompts during the run, restored to "%s" at'
                                          ' the end' % query_orig)

    try:
        # 1. load state
        if xcm is not None:
            if not os.path.isfile(xcm):
                raise FileNotFoundError(xcm)
            _say(summary, 'WARNING: restoring %s replaces the current session' % xcm)
            #stay in the xcm directory until the end of the run (restored in the finally block):
            #the xcm names its tables relatively (mtable{pion...fits}) and the 2.8 GB pion tables
            #are, we suspect, read from disk during the fit, so a cwd change after the restore
            #leaves heasp::table::getValues without its file (segfault seen in the fit)
            os.chdir(os.path.dirname(xcm))
            Xset.restore(os.path.basename(xcm))
        elif len(AllModels.sources) == 0 or AllData.nSpectra == 0:
            raise RuntimeError('No model or no spectrum loaded: pass xcm= or load them first')

        if xcm_set_ener_str is not None:
            #imported here: xspec_config_multisp is heavy and only needed for this
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from xspec_config_multisp import set_ener
            set_ener(xcm_set_ener_str, xcm_set_ener_xrism)
            _say(summary, 'model energies set with set_ener("%s", xrism=%s)'
                 % (xcm_set_ener_str, xcm_set_ener_xrism))

        #log next to the chain, unless the user already has a log open
        if Xset.log is None:
            for elem in [paths['log']] + [paths[k] for k in paths if k != 'log']:
                if overwrite and os.path.exists(elem):
                    os.remove(elem)
            Xset.openLog(paths['log'])
            log_opened = True
        else:
            paths['log'] = getattr(Xset.log, 'name', 'already-open user log')
            _say(summary, 'an XSPEC log is already open (%s): not opening another' % paths['log'])
            for elem in [paths[k] for k in paths if k != 'log']:
                if overwrite and os.path.exists(elem):
                    os.remove(elem)

        # 2. pre-checks
        if 'cstat' not in Fit.statMethod.lower():
            _say(summary, 'WARNING: fit statistic is "%s", not cstat' % Fit.statMethod)

        table = build_param_table(absorber_components)
        frozen_orig = {str(row['key']): row['frozen'] for row in table}

        frozen_now = []
        for user_key in (freeze or []):
            row = _resolve_key(user_key, table)
            if not row['free']:
                raise ValueError('freeze: %s is already frozen or linked' % row['key'])
            _get_par(row).frozen = True
            row['frozen'], row['free'] = True, False
            frozen_now.append({'key': row['key'], 'component': row['component'],
                               'name': row['name'], 'value': row['value']})
            _say(summary, 'frozen for this run: %s %s.%s = %g' % (row['key'], row['component'],
                                                                 row['name'], row['value']))
        summary['conditional'] = len(frozen_now) > 0
        summary['frozen_for_run'] = frozen_now

        free_rows = [row for row in table if row['free']]
        n_free = len(free_rows)
        if n_free == 0:
            raise RuntimeError('No free parameter to sample')
        if not any(row['group'] == 'absorber' for row in free_rows):
            _say(summary, 'WARNING: no free parameter tagged "absorber": absorber_components=%s matches'
                          ' none of the components %s' % (list(absorber_components),
                                                          sorted(set(row['component'] for row in table))))

        # 3. refit
        if refit or global_fit:
            if global_fit:
                if seed is None:
                    seed = secrets.randbits(31)
                    _choose(summary, 'seed', seed, 'secrets.randbits(31)')
                Xset.seed = seed
                Fit.globalFit()
                basins = [list(elem) for elem in Fit.globalBasins]
                summary['global_basins'] = [{'stat': elem[0], 'dstat': elem[1], 'params': elem[2]}
                                            for elem in basins]
                competitive = [elem for elem in basins[1:] if elem[1] < 9]
                if competitive:
                    _say(summary, 'WARNING: %d competitive basin(s) within delta-stat < 9 (%s): the'
                                  ' posterior may be multimodal, consider sampler="nest"'
                         % (len(competitive), ', '.join('%.2f' % elem[1] for elem in competitive)))
            if refit:
                Fit.perform()
                Fit.perform()
        try:
            summary['best_fit'] = {'statistic': Fit.statistic, 'dof': Fit.dof, 'method': Fit.statMethod}
        except Exception:
            #refit=False on a session where no fit has been run yet
            summary['best_fit'] = {'statistic': None, 'dof': None, 'method': Fit.statMethod}
            _say(summary, 'WARNING: no valid fit (refit=False): the chain starts from unfitted values'
                          ' and the gw walkers cannot use the fit covariance')
        _refresh_values(table)

        for row in free_rows:
            if not 0.01 <= _limit_position(row) <= 0.99:
                _say(summary, 'WARNING: %s (%s.%s = %g) is within 1%% of a hard limit [%g, %g]:'
                              ' bad prior range or pegged fit?'
                     % (row['key'], row['component'], row['name'], row['value'],
                        row['hard_min'], row['hard_max']))

        _say(summary, 'D = %d free parameters:' % n_free)
        for row in table:
            print('    %-8s %-12s %-10s %-10s %13.6g  [%g, %g]  %s%s'
                  % (row['key'], row['component'], row['name'], row['unit'], row['value'],
                     row['hard_min'], row['hard_max'],
                     'free' if row['free'] else ('linked ' + row['link'] if row['link'] else 'frozen'),
                     ' (%s)' % row['group'] if row['free'] else ''))

        # 4. profile errors, before any chain is loaded
        if profile_errors:
            AllChains.clear()
            stat_before = Fit.statistic
            errors = {}
            for user_key in profile_errors:
                row = _resolve_key(user_key, table)
                Fit.error('1.0 %s' % row['key'])
                lo, hi, flags = _get_par(row).error
                errors[str(row['key'])] = {'low': lo, 'high': hi, 'flags': flags}
            summary['profile_errors'] = {'kind': 'profile-likelihood, delta-stat 1.0, computed before'
                                                 ' the chain', 'errors': errors}
            if Fit.statistic < stat_before - 1e-3:
                _say(summary, 'WARNING: Fit.error found a new minimum (stat %.4f -> %.4f)'
                     % (stat_before, Fit.statistic))
                summary['best_fit']['statistic_after_errors'] = Fit.statistic
                _refresh_values(table)

        # 5. priors
        resolved = load_priors(priors, table)
        prior_source = resolved.pop('_source', None)
        if resolved:
            for row in free_rows:
                family, hyper = resolved.get(row['key'], ('cons', []))
                _get_par(row).prior = tuple([family] + hyper) if hyper else family
                row['prior'] = [family] + hyper
                if row['key'] not in resolved:
                    _say(summary, 'prior: %s (%s.%s) left flat within hard limits [%g, %g]'
                         % (row['key'], row['component'], row['name'], row['hard_min'], row['hard_max']))
            Fit.bayes = 'on'
            if Fit.bayes != 'on':
                raise RuntimeError('Bayesian inference did not switch on (Fit.bayes = %s)' % Fit.bayes)
        else:
            #no priors given: keep (and record) whatever the session has
            for row in free_rows:
                row['prior'] = list(_get_par(row).prior) if Fit.bayes != 'off' else \
                    'flat within hard limits'
        summary['bayes'] = {'state': Fit.bayes, 'priors_source': prior_source,
                            'listing': Fit.bayesPriors}

        for row in free_rows:
            if row['prior'] == 'flat within hard limits' or \
                    [str(elem).lower() for elem in row['prior']] == ['cons']:
                span = _span_decades(row)
                if span is not None and span > 2:
                    _say(summary, 'WARNING: %s (%s.%s) is flat over %.1f decades of hard limits'
                         % (row['key'], row['component'], row['name'], span))

        if sampler == 'nest':
            _check_nest_priors(table, resolved)

        # 6. walkers, length, burn
        settings = {'sampler': sampler}
        if sampler == 'gw':
            if walkers is None:
                walkers = 2 * n_free
                _choose(summary, 'walkers', walkers, 'smallest even number >= 2D')
            if walkers % 2 != 0:
                raise ValueError('walkers must be even, got %d' % walkers)
            if not 2 * n_free <= walkers <= 4 * n_free:
                _say(summary, 'WARNING: %d walkers for D=%d, outside the recommended 2D-4D'
                     % (walkers, n_free))
            settings.update({'walkers': walkers, 'steps_per_walker': steps_per_walker,
                             'burn_per_walker': burn_per_walker,
                             'length_total': walkers * steps_per_walker,
                             'burn_total': walkers * burn_per_walker,
                             'proposal': 'gaussian fit', 'thin': 1})
            _say(summary, 'gw: %d walkers, total length %d, total burn %d'
                 % (walkers, walkers * steps_per_walker, walkers * burn_per_walker))
        else:
            _say(summary, '%s: walkers/steps/burn ignored, sampler defaults used' % sampler)

        # length pre-check
        if sampler == 'gw' and not test:
            if pilot is not None:
                summary['length_check'] = check_length_from_pilot(
                    pilot, walkers, steps_per_walker, burn_per_walker, min_tau_multiple,
                    min_burn_tau, min_ess, free_keys=[row['key'] for row in free_rows])
                if summary['length_check']['pilot_free_keys_match'] is False:
                    _say(summary, 'WARNING: the pilot free parameters differ from this run')
            elif skip_length_check:
                _say(summary, 'WARNING: no pilot, chain length not validated (skip_length_check=True)')
                summary['length_check'] = 'skipped'
            else:
                raise ValueError('No pilot given: the chain length cannot be validated. Run with'
                                 ' test=True first and pass pilot=<its .json>, or pass'
                                 ' skip_length_check=True.')

        # 7. seed and parallelism
        if seed is None:
            seed = secrets.randbits(31)
            _choose(summary, 'seed', seed, 'secrets.randbits(31)')
        Xset.seed = seed
        settings['seed'] = seed
        if n_parallel:
            if sampler == 'gw':
                Xset.parallel.walkers = n_parallel
                if walkers % n_parallel != 0:
                    _say(summary, 'WARNING: walkers (%d) not divisible by n_parallel (%d): poor load'
                                  ' balance' % (walkers, n_parallel))
            elif sampler == 'hmc':
                _pyXspec.doXspecCmd(['parallel', 'hmc', str(int(n_parallel))])
        settings['n_parallel'] = n_parallel
        summary['settings'] = settings

        if save_xcm:
            Xset.save(paths['bestfit_xcm'], info='a')

        summary['provenance'] = _provenance(xcm, xcm_set_ener_str, xcm_set_ener_xrism)

        # 8. run
        summary['status'] = 'running'
        AllChains.clear()
        if sampler == 'gw':
            AllChains.defAlgorithm = 'gw'
            AllChains.defFileType = 'fits'
            AllChains.defWalkers = walkers
            AllChains.defLength = walkers * steps_per_walker
            AllChains.defBurn = walkers * burn_per_walker
            AllChains.defProposal = 'gaussian fit'
            log_offset = _log_size(paths['log'])
            try:
                xspec.runChain(outfile, length=walkers * steps_per_walker, walkers=walkers,
                               burn=walkers * burn_per_walker, algorithm='gw', proposal='gaussian fit')
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                #XSPEC raises the same generic exception for a Ctrl-C and for a failed walker
                #initialization: only the log says which one it was
                new_log = _log_since(paths['log'], log_offset)
                if 'Chain run has been interrupted' in new_log:
                    raise KeyboardInterrupt('XSPEC chain run interrupted (Ctrl-C), partial chain'
                                            ' file not usable: %s' % outfile)
                if not any(elem in new_log for elem in COVARIANCE_FAILURE_MESSAGES):
                    raise

                #ill-conditioned fit covariance: retry once without it
                _say(summary, 'WARNING: chain with "gaussian fit" failed (%r, fit covariance unusable'
                              ' for the walker initialization), retrying with "%s"'
                     % (exc, FALLBACK_PROPOSAL))
                settings['proposal'] = FALLBACK_PROPOSAL
                settings['proposal_fallback_reason'] = repr(exc)
                AllChains.clear()
                #XSPEC refuses to rewrite an existing chain file name in the same session (reported
                #as "Invalid chars in file name"), so the partial file of the failed attempt goes
                if os.path.exists(outfile):
                    os.remove(outfile)
                Xset.seed = seed
                xspec.runChain(outfile, length=walkers * steps_per_walker, walkers=walkers,
                               burn=walkers * burn_per_walker, algorithm='gw',
                               proposal=FALLBACK_PROPOSAL)

            chain = read_chain(outfile)
            n_steps_written = chain['samples'].shape[1]
            if n_steps_written < steps_per_walker:
                #XSPEC catches Ctrl-C itself and leaves a truncated file
                raise KeyboardInterrupt('chain stopped after %d of %d steps per walker'
                                        % (n_steps_written, steps_per_walker))
            summary['layout'] = {'rows': 'step-major, row = step*n_walkers + walker',
                                 'columns': '<par>__<idx> or <par>__<model>__<idx>, then FIT_STATISTIC',
                                 'burn_stored': False,
                                 'keys_in_column_order': chain['keys']}

        elif sampler == 'nest':
            ckpt = outfile + '.ckpt.fits'
            if os.path.exists(ckpt) and not overwrite:
                _say(summary, 'nest: resuming from checkpoint %s' % ckpt)
            _pyXspec.doXspecCmd(['nest', 'seed', str(int(seed))])
            res = xspec.runNested(outfile, overwrite=overwrite, parallel=n_parallel)
            summary['nest'] = {'logZ': res.logZ, 'logZerr': res.logZerr, 'info_H': res.info_H,
                               'n_live': res.n_live, 'n_like': res.n_like, 'n_iter': res.n_iter,
                               'n_modes': res.n_modes, 'converged': res.converged,
                               'termination': res.termination}
            settings.update({'nLive': res.n_live, 'resampleN': len(res.equal_weight_samples)})

        else:
            res = xspec.runHMC(outfile, overwrite=overwrite)
            summary['hmc'] = {'n_chains': res.n_chains, 'n_divergent': res.n_divergent,
                              'mean_accept': res.mean_accept, 'n_grad': res.n_grad,
                              'eps_adapted': res.eps_adapted, 'termination': res.termination}
            settings.update({'chains': res.n_chains, 'samples_total': res.samples.shape[0]})

        # 9. diagnostics
        if sampler == 'nest':
            summary['diagnostics'] = {'verdict': 'N/A',
                                      'notes': ['nested sampling: no MCMC convergence diagnostics;'
                                                ' see nest (logZ, converged)']}
        else:
            try:
                xs_diag = AllChains.diagnostics()
            except Exception:
                xs_diag = {}
            if not xs_diag.get('param_labels'):
                #hmc output is chain-load compatible but not loaded by runHMC
                AllChains.clear()
                AllChains.__iadd__(outfile)  #AllChains += outfile, without rebinding the name
                xs_diag = AllChains.diagnostics()

            if sampler == 'gw':
                acc = walker_acceptance(chain['samples'])
                diag = assess_chain(xs_diag, steps_per_walker, burn_per_walker, acceptance=acc,
                                    min_tau_multiple=min_tau_multiple, min_burn_tau=min_burn_tau,
                                    min_ess=min_ess, max_rhat=max_rhat)
                diag['acceptance'] = acc
                diag['arviz'] = arviz_crosscheck(chain['samples'], chain['keys'])
            else:
                diag = assess_chain(xs_diag, xs_diag['sub_chain_length'][0], 0,
                                    min_tau_multiple=min_tau_multiple, min_burn_tau=0,
                                    min_ess=min_ess, max_rhat=max_rhat)
                #divergences should stay below ~1% of the samples (13.0.1 manual, Sec. 3.7)
                div_frac = summary['hmc']['n_divergent'] / settings['samples_total']
                if div_frac > 0.01:
                    diag['failing'].append('hmc: %d divergent transitions (%.1f%% > 1%%): raise'
                                           ' target_accept or reparameterize'
                                           % (summary['hmc']['n_divergent'], 100 * div_frac))
                    diag['verdict'] = 'WARN'
            diag['xspec_diagnostics'] = xs_diag
            summary['diagnostics'] = diag

            _say(summary, 'diagnostics: %s (chain diag %s)' % (diag['verdict'], diag['xspec_verdict']))
            for elem in diag['failing']:
                _say(summary, '    ' + elem)
            for elem in diag['notes']:
                _say(summary, '    note: ' + elem)
            if diag['recommended_steps_per_walker']:
                _say(summary, 'recommended rerun: steps_per_walker >= %d (1.5 x %g x tau %.3g, %s)'
                     % (diag['recommended_steps_per_walker'], min_tau_multiple, diag['tau_used_all'],
                        diag['tau_rule']))
            if test:
                _say(summary, 'test run: tau_max = %.3g, tau_reliable = %s'
                     % (diag['tau_max_all'], diag['tau_reliable']))

        summary['status'] = 'complete'

    except KeyboardInterrupt as exc:
        summary['status'] = 'interrupted'
        summary['error'] = repr(exc)
        raise

    except Exception as exc:
        #nest/hmc interrupts also come back as generic XSPEC exceptions
        interrupted = summary['status'] == 'running' and \
            'interrupted' in _log_since(paths['log'], 0).lower()
        summary['status'] = 'interrupted' if interrupted else 'failed'
        summary['error'] = repr(exc)
        raise

    finally:
        Fit.query = query_orig
        os.chdir(cwd_orig)

        #restore the frozen flags of this session
        for row in table:
            if str(row['key']) in frozen_orig and row['frozen'] != frozen_orig[str(row['key'])]:
                try:
                    _get_par(row).frozen = frozen_orig[str(row['key'])]
                except Exception:
                    pass

        summary['parameters'] = table
        summary['D'] = len([row for row in table if row['free']])
        if log_opened:
            Xset.closeLog()
        if os.path.isdir(os.path.dirname(paths['sidecar'])):
            _write_json(paths['sidecar'], summary)

    _say(summary, 'written: ' + ', '.join(paths.values()))

    # prior sensitivity reruns: same seed and settings, separate labelled outputs
    if prior_variants:
        summary['prior_variants'] = []
        for variant in prior_variants:
            var_name = os.path.splitext(os.path.basename(variant))[0]
            var_kwargs = {k: v for k, v in run_kwargs.items()
                          if k in run_chain.__code__.co_varnames[:run_chain.__code__.co_argcount]}
            var_kwargs.update({'outfile': stem + '_prior-%s.fits' % var_name, 'priors': variant,
                               'prior_variants': None, 'seed': seed})
            if test:
                var_kwargs['test'] = True
            var_summary = run_chain(**var_kwargs)
            summary['prior_variants'].append({'priors': variant, 'paths': var_summary['paths'],
                                              'verdict': var_summary.get('diagnostics', {}).get('verdict')})
        _write_json(paths['sidecar'], summary)

    return summary


def continue_chain(chain_file, steps_per_walker, priors, seed=None, n_parallel=None,
                   absorber_components=('mtable', 'pion_abs'), backup=True,
                   min_tau_multiple=50, min_burn_tau=5, min_ess=400, max_rhat=1.01):
    '''
    Appends steps_per_walker more steps per walker to a finished gw chain written by run_chain
    (XSPEC "chain run >file", PyXspec Chain.run(append=True)). The walkers restart from their last
    rows in the file, there is no new burn-in, and the number of walkers is the one of the file.

    The session must be the one of the original run: same model, same free parameters, same hard
    limits, and priors= is the same dict (or JSON path) as the original. This is checked against the
    sidecar <stem>.json and the file before anything is written, and the function raises on:
        -free parameters different from the file's columns
        -hard limits different from the sidecar's parameter table
        -the XSPEC prior listing different from the recorded one
    Parameters frozen for the original run (conditional chain) are frozen again from the sidecar.

    Safety: the chain file is copied to <file>.bak_cont<k> BEFORE appending (backup=True). XSPEC writes
    the chain to the file as it runs, so an interrupted or crashed append keeps the rows written so far;
    the sidecar is only rewritten after a complete, checked append, so it always describes the last
    good state. A crashed append k leaves <stem>_cont<k>.log and the backup: the function refuses to
    start again until these are moved (copy the backup over the chain file to redo that append, or
    keep the partial file after checking it with chain_diagnostics.read_chain).

    seed: a new seed for every append (None draws one); recorded. n_parallel: Xset.parallel.walkers.
    Diagnostics are recomputed on the whole file and the sidecar gets a "continuations" list.
    Returns the updated sidecar dict.
    '''

    _check_no_ipython(n_parallel, 'continue_chain')

    chain_file = os.path.abspath(chain_file)
    stem = chain_file[:-5] if chain_file.endswith('.fits') else chain_file
    sidecar_path = stem + '.json'
    if not (os.path.isfile(chain_file) and os.path.isfile(sidecar_path)):
        raise FileNotFoundError('%s and its sidecar %s are both needed' % (chain_file, sidecar_path))
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    old_settings = sidecar.get('settings', {})
    if old_settings.get('sampler') != 'gw':
        raise ValueError('only a gw chain can be continued, this one is "%s"' % old_settings.get('sampler'))
    walkers = int(old_settings['walkers'])
    burn_per_walker = int(old_settings['burn_per_walker'])
    continuations = list(sidecar.get('continuations', []))
    k = len(continuations) + 1
    log_path = '%s_cont%d.log' % (stem, k)
    backup_path = '%s.bak_cont%d' % (chain_file, k)
    leftovers = [elem for elem in ([log_path, backup_path] if backup else [log_path]) if os.path.exists(elem)]
    if leftovers:
        raise FileExistsError('%s exist: continuation %d was already started (crashed or interrupted?).'
                              ' Copy the backup over the chain file to redo it, or keep the partial'
                              ' chain after checking it, then move these files away.' % (leftovers, k))

    if len(AllModels.sources) == 0 or AllData.nSpectra == 0:
        raise RuntimeError('No model or no spectrum loaded: restore the original session first')
    if int(steps_per_walker) < 1:
        raise ValueError('steps_per_walker must be >= 1')
    steps_per_walker = int(steps_per_walker)

    summary = {'log': [], 'auto_choices': {}}
    table = build_param_table(absorber_components)
    frozen_orig = {str(row['key']): row['frozen'] for row in table}
    log_opened = False

    try:
        # 1. same frozen parameters as the original run
        for rec in sidecar.get('frozen_for_run', []):
            row = _resolve_key(rec['key'], table)
            if not row['free']:
                raise ValueError('%s was frozen for the original run but is already frozen or linked here'
                                 % row['key'])
            _get_par(row).frozen = True
            row['frozen'], row['free'] = True, False
            _say(summary, 'frozen as in the original run: %s %s.%s' % (row['key'], row['component'],
                                                                      row['name']))
        free_rows = [row for row in table if row['free']]

        # 2. the session must match the file and the sidecar
        chain = read_chain(chain_file)
        if chain['chain_type'] != 'GoodmanWeare' or chain['n_walkers'] != walkers:
            raise ValueError('%s: chain type "%s" with %s walkers, the sidecar says gw with %d walkers'
                             % (chain_file, chain['chain_type'], chain['n_walkers'], walkers))
        if [str(elem) for elem in chain['keys']] != [str(row['key']) for row in free_rows]:
            raise ValueError('The free parameters of this session differ from the columns of the chain.'
                             ' File: %s. Session: %s' % ([str(e) for e in chain['keys']],
                                                         [str(row['key']) for row in free_rows]))
        recorded = {str(row['key']): row for row in sidecar.get('parameters', [])}
        for row in free_rows:
            rec = recorded.get(str(row['key']))
            if rec is None:
                raise ValueError('Parameter %s is not in the sidecar parameter table' % row['key'])
            for lim in ('hard_min', 'hard_max'):
                if rec[lim] is None or not np.isclose(row[lim], rec[lim], rtol=1e-9, atol=0.):
                    raise ValueError('%s (%s.%s): %s is %g here and %s in the original run: the limits'
                                     ' are the prior range, they must be identical'
                                     % (row['key'], row['component'], row['name'], lim, row[lim], rec[lim]))

        # 3. priors, as in run_chain step 5, then compared with the recorded listing
        resolved = load_priors(priors, table)
        resolved.pop('_source', None)
        if not resolved:
            raise ValueError('priors= is required: pass the same priors as the original run')
        for row in free_rows:
            family, hyper = resolved.get(row['key'], ('cons', []))
            _get_par(row).prior = tuple([family] + hyper) if hyper else family
        Fit.bayes = 'on'
        if Fit.bayes != 'on':
            raise RuntimeError('Bayesian inference did not switch on (Fit.bayes = %s)' % Fit.bayes)
        if Fit.bayesPriors != sidecar.get('bayes', {}).get('listing'):
            raise ValueError('The priors differ from the original run.\nRecorded:\n%s\nNow:\n%s'
                             % (sidecar.get('bayes', {}).get('listing'), Fit.bayesPriors))

        # 4. backup, seed, log
        n_before = chain['samples'].shape[1]
        if backup:
            shutil.copy2(chain_file, backup_path)
            _say(summary, 'backup before appending: %s' % backup_path)
        if seed is None:
            seed = secrets.randbits(31)
            _choose(summary, 'seed', seed, 'secrets.randbits(31)')
        Xset.seed = seed
        if n_parallel:
            Xset.parallel.walkers = n_parallel
            if walkers % n_parallel != 0:
                _say(summary, 'WARNING: walkers (%d) not divisible by n_parallel (%d)' % (walkers, n_parallel))
        if Xset.log is None:
            Xset.openLog(log_path)
            log_opened = True
        else:
            log_path = getattr(Xset.log, 'name', 'already-open user log')

        # 5. append: load the chain, then run with append=True (the walkers start from the file rows)
        _say(summary, 'appending %d steps per walker (%d rows) to %s, now %d steps per walker, seed %d'
             % (steps_per_walker, walkers * steps_per_walker, chain_file, n_before, seed))
        AllChains.clear()
        AllChains.defFileType = 'fits'
        AllChains.__iadd__(chain_file)  #AllChains += chain_file, without rebinding the name
        chain_obj = AllChains(1)
        chain_obj.runLength = walkers * steps_per_walker
        chain_obj.proposal = FALLBACK_PROPOSAL     #not used for gw: the walkers start from the file
        t_run = time.time()
        run_error = None
        try:
            chain_obj.run(append=True)
        except KeyboardInterrupt:
            _say(summary, 'interrupted: the chain file keeps the rows written so far, backup before this'
                          ' append: %s' % backup_path)
            raise
        except Exception as exc:
            run_error = exc
        seconds = time.time() - t_run

        try:
            chain = read_chain(chain_file)
        except Exception as exc:
            raise RuntimeError('chain file unreadable after the append (%r): restore it from %s'
                               % (exc, backup_path))
        n_after = chain['samples'].shape[1]
        if run_error is not None or n_after - n_before < steps_per_walker:
            raise RuntimeError('append stopped after %d of %d steps per walker (%r). The file keeps the'
                               ' rows written, the backup before the append is %s'
                               % (n_after - n_before, steps_per_walker, run_error, backup_path))

        # 6. diagnostics on the whole chain, as in run_chain step 9
        AllChains.clear()
        AllChains.__iadd__(chain_file)
        xs_diag = AllChains.diagnostics()
        acc = walker_acceptance(chain['samples'])
        diag = assess_chain(xs_diag, n_after, burn_per_walker, acceptance=acc,
                            min_tau_multiple=min_tau_multiple, min_burn_tau=min_burn_tau,
                            min_ess=min_ess, max_rhat=max_rhat)
        diag['acceptance'] = acc
        #the arviz cross-check is not what decides the verdict: on a long chain it is thinned to about
        #25000 steps per walker (tau is of the order of 2000 steps) and a failure must not lose the
        #sidecar update of a finished append
        thin = max(1, int(np.ceil(n_after / 25000.)))
        try:
            diag['arviz'] = arviz_crosscheck(chain['samples'][:, ::thin], chain['keys'])
            diag['arviz_thinned_by'] = thin
        except Exception as exc:
            diag['arviz'] = {'error': repr(exc)}
        diag['xspec_diagnostics'] = xs_diag

        _say(summary, 'diagnostics on %d steps per walker: %s (chain diag %s)'
             % (n_after, diag['verdict'], diag['xspec_verdict']))
        for elem in diag['failing']:
            _say(summary, '    ' + elem)
        for elem in diag['notes']:
            _say(summary, '    note: ' + elem)
        if diag['recommended_steps_per_walker']:
            _say(summary, 'recommended length: steps_per_walker >= %d (1.5 x %g x tau %.3g, %s)'
                 % (diag['recommended_steps_per_walker'], min_tau_multiple, diag['tau_used_all'],
                    diag['tau_rule']))

        # 7. sidecar: only now, after a complete and checked append
        previous = sidecar.get('diagnostics', {})
        continuations.append({'index': k,
                              'date': datetime.datetime.now().isoformat(timespec='seconds'),
                              'seed': seed, 'n_parallel': n_parallel,
                              'steps_per_walker_added': steps_per_walker,
                              'steps_per_walker_before': n_before, 'steps_per_walker_after': n_after,
                              'seconds': seconds, 'log': log_path,
                              'backup': backup_path if backup else None,
                              'previous_verdict': previous.get('verdict'),
                              'previous_tau_max_all': previous.get('tau_max_all'),
                              'verdict': diag['verdict'], 'tau_max_all': diag['tau_max_all'],
                              'messages': summary['log']})
        sidecar['continuations'] = continuations
        sidecar['diagnostics'] = diag
        sidecar['settings']['steps_per_walker_total'] = n_after
        sidecar['status'] = 'complete'
        _write_json(sidecar_path, sidecar)
        _say(summary, 'sidecar updated: %s' % sidecar_path)

    finally:
        #restore the frozen flags of this session
        for row in table:
            if str(row['key']) in frozen_orig and row['frozen'] != frozen_orig[str(row['key'])]:
                try:
                    _get_par(row).frozen = frozen_orig[str(row['key'])]
                except Exception:
                    pass
        if log_opened:
            Xset.closeLog()

    return sidecar

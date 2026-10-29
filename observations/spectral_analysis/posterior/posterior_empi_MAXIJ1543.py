"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-27.

MCMC chain of the empirical XRISM/Resolve model of MAXI J1543-564 (obsid 902003010):
TBabs(powerlaw + diskbb + absorption lines, most of them in vashift and in tied doublets), C-stat, with
the NXB model nxb1 on the second response. Running the whole file restores the xcm, checks its
structure, sets the prior ranges and checks the starting fit, then does the ONE run or check chosen by
WHAT below. Blocks that only read a finished chain (reassess, summary, corner) skip XSPEC entirely.

The frozen / free / tied choices are those of the xcm and are not changed here: D = 24 free parameters,
all in the source model. Every nxb1 parameter is frozen (the chain is conditional on the NXB model),
as are all line energies and the powerlaw index (3.5).

Priors (agreed with Maxime 2026-10-27; 'jeffreys' = uniform in ln(theta), 'cons' = uniform in theta).
The hard limits are the prior ranges, set in step 3:
    jeffreys  TBabs nH [0.1, 10], powerlaw norm [0.02, 2], diskbb norm [10, 1000]: scale parameters,
              all data-dominated (the range only bounds the walkers)
    cons      diskbb Tin [0.3, 3] keV
    cons      velocities: +-3000 km/s, except the 8.29 keV pair (54) +-1000 km/s: wider, its single
              free line (p60) could move onto the Fe XXVI Lyb energies (1550-1780 km/s away) and take
              the place of the weak Lyb doublet. The +3000 km/s edge of the broad Fe XXVI component
              (61) is kept as physical (nothing near 1500 km/s in the pion v_rms of zone 2)
    cons      widths [0, 0.025] keV, broad Fe XXVI component (63) [0, 0.05] keV. Uniform and not
              log-uniform: below the Resolve line spread (sigma ~2 eV) the width is unconstrained, and a
              log prior would put its mass there, depending on an arbitrary lower cut
    cons      absorption norms [-1e-3, 0]: absorption only, -1e-3 is far deeper than any line (the
              deepest, S XVI, is -5e-5 +- 1e-5). A line consistent with zero piles up at 0, which gives
              a one-sided limit. Not log-uniform: the range contains 0
Priors are keyed by global index: 'powerlaw.norm' or 'gaussian.norm' would also match components of
nxb1, and chain_runner refuses ambiguous names.

Known features of the posterior (fit covariance with all 24 parameters free, 2026-10-27): the broad Fe
XXVI component (61, 63, 64) is correlated at |r| 0.8-0.97 with the narrow Lya norms (7, 10) and their
width (6): the data fix the total Fe XXVI Lya absorption better than its split between the narrow and
broad components. Its norm is only 1.4 sigma from 0 in that approximation (3 sigma away with the other
parameters frozen). Expect a longer tau in that group, and look at the 'fe_broad' corner plot.

The xcm sets DISABLE_ANALYTIC_GRAD yes (finite-difference gradients). It only matters for the fits
(start check, run_chain's refit and the fit covariance that starts the gw walkers), not for the
likelihood the samplers evaluate. It is set again in step 1 so the file does not depend on it.

n_parallel > 1 crashes XSPEC inside IPython (e.g. the PyCharm Python console): run this file with a
plain python interpreter (chain_runner refuses otherwise).
"""

import itertools
import os
import subprocess
import sys

POSTERIOR_DIR = '/home/parrazyte/Documents/Work/Scripts/Python/winds/observations/spectral_analysis/posterior'
sys.path.insert(0, POSTERIOR_DIR)

SP_DIR = '/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/902003010_repro/analysis_tr/sp'
XCM = 'prefit_setup_MC.xcm'
OUT_DIR = os.path.join(SP_DIR, 'posterior')
PILOT = os.path.join(OUT_DIR, 'MAXIJ1543_empi_pilot.fits')
CHAIN_FILE = os.path.join(OUT_DIR, 'MAXIJ1543_empi_all_free.fits')
NEST_FILE = os.path.join(OUT_DIR, 'MAXIJ1543_empi_nest.fits')
CORNER_DIR = os.path.join(OUT_DIR, 'corner')

#WHAT TO RUN, one value:
#    'setup'        steps 1-4 only: restore, structure check, prior ranges, starting-fit check
#    'gw_pilot'     step 5: 100 steps per walker (time per step, walker start, sticky walkers)
#    'gw_prod'      step 6: production chain, needs the pilot sidecar
#    'gw_continue'  step 7: append steps to the production chain (if its verdict is WARN)
#    'nest'         step 8: nested sampling, independent check for missed modes (optional)
#    'reassess', 'summary', 'corner'   steps 9-11: read the finished chain, no XSPEC
#    'ew_check'     step 12: equivalent widths of the corner plots vs XSPEC eqwidth (own process)
WHAT = 'gw_continue'

#gw: 48 walkers = 2D. Each half-ensemble is 24 proposals, split over min(n_parallel, 24) workers in
#rounds (XSPEC Chain.cxx): 12 workers = 2 even rounds. The machine has 12 physical cores.
WALKERS = 48
N_PARALLEL = 12
#post-run burn-in, in steps per walker, used by the read-only blocks: set it from 'reassess'
DISCARD = 4000

#source model keys (global indices, as in show param)
CONTINUUM_KEYS = (1, 3, 74, 75)
VELOCITY_KEYS = (4, 34, 54, 61)
WIDTH_KEYS = (6, 13, 36, 63)
NORM_KEYS = (7, 10, 14, 18, 26, 33, 37, 50, 60, 64, 70, 73)
FREE_KEYS = sorted(CONTINUUM_KEYS + VELOCITY_KEYS + WIDTH_KEYS + NORM_KEYS)
#components tagged 'absorber' in the sidecar (corner-plot groups): all the lines
LINE_COMPONENTS = ('vashift', 'gaussian')

#prior ranges = hard limits (soft limits set equal)
LIMITS = {1: (0.1, 10.), 3: (0.02, 2.), 74: (0.3, 3.), 75: (10., 1000.),
          4: (-3000., 3000.), 34: (-3000., 3000.), 61: (-3000., 3000.), 54: (-1000., 1000.),
          6: (0., 0.025), 13: (0., 0.025), 36: (0., 0.025), 63: (0., 0.05)}
LIMITS.update({key: (-1e-3, 0.) for key in NORM_KEYS})

PRIORS = {key: 'cons' for key in FREE_KEYS}
PRIORS.update({1: 'jeffreys', 3: 'jeffreys', 75: 'jeffreys'})

#rest energy (keV) used to show each width as a velocity: the widths are tied in keV, so the lines
#sharing one (e.g. 13: Fe XXV Hea, Heb, the 8.29 keV pair, 6.66 and 6.616 keV) differ in km/s
C_KMS = 299792.458
WIDTH_REF_KEV = {6: 6.9732, 13: 6.7004, 36: 2.6227, 63: 6.9732}

#axis labels (rest energies in keV; '2x' = the other line of the doublet is tied at twice this norm)
LABELS = {1: 'nH (1e22 cm-2)', 3: 'powerlaw norm', 74: 'Tin (keV)', 75: 'diskbb norm',
          4: 'v Fe XXV/XXVI (km/s)', 34: 'v S XVI (km/s)', 54: 'v 8.29 pair (km/s)',
          61: 'v broad Fe XXVI (km/s)',
          6: 'sigma Fe XXVI (km/s)', 13: 'sigma Fe XXV (km/s)', 36: 'sigma S XVI (km/s)',
          63: 'sigma broad Fe XXVI (km/s)',
          7: 'N 6.952 Fe XXVI Lya2', 10: 'N 6.973 Fe XXVI Lya1', 14: 'N 6.700 Fe XXV w',
          18: 'N 6.682 Fe XXV y', 26: 'N 6.668 Fe XXV x', 33: 'N 7.881 Fe XXV Heb',
          37: 'N 2.620 S XVI (2x)', 50: 'N 8.246 Fe XXVI Lyb (2x)', 60: 'N 8.2955',
          64: 'N broad 6.952 (2x)', 70: 'N 6.660', 73: 'N 6.616'}

#velocity zones: each free vashift velocity with the lines it shifts (including the vashifts of the xcm
#linked to it: 11, 15, 19, 23, 27, 47 = p4), numbered in model order. A zone holds the free widths of
#its lines even when they are tied to a width of another zone (the 8.29 keV pair has the Fe XXV width 13)
ZONES = {'zone1': [4, 6, 13, 7, 10, 14, 18, 26, 33, 50],   #Fe XXVI Lya and Lyb, Fe XXV w, y, x and Heb
         'zone2': [34, 36, 37],                              #S XVI Lya
         'zone3': [54, 13, 60],                              #8.29 keV pair
         'zone4': [61, 63, 64]}                              #broad Fe XXVI Lya
#lines outside any vashift (fixed energies), with their width (tied to 13)
UNSHIFTED_KEYS = [13, 70, 73]

CORNER_SETS = dict(ZONES)
for zone_a, zone_b in itertools.combinations(ZONES, 2):
    CORNER_SETS[zone_a + '+' + zone_b] = list(dict.fromkeys(ZONES[zone_a] + ZONES[zone_b]))
CORNER_SETS.update({'unshifted': UNSHIFTED_KEYS, 'continuum': list(CONTINUUM_KEYS),
                    #narrow / broad Fe XXVI Lya degeneracy
                    'fe_broad': [4, 6, 7, 10, 61, 63, 64]})
CORNER_SETS['all'] = list(dict.fromkeys(sum(ZONES.values(), []) + UNSHIFTED_KEYS + list(CONTINUUM_KEYS)))
if sorted(CORNER_SETS['all']) != FREE_KEYS:
    raise RuntimeError('the zones and other corner sets do not cover the free parameters exactly')

#figures of the corner block, as for the pion chain (contour_maker.MAXIJ1543_FIGURE_SET): matplotlib-only
#backend for every set, plus a 'paper' version of 'all' (no diagnostics text), corner package for each zone
CORNER_FIGURES = ([('own', name, False) for name in CORNER_SETS] + [('own', 'all', True)]
                  + [('corner', name, False) for name in ZONES])

#figure labels and titles, in the style of the pion figures (contour_maker.maxij1543_labels_scales).
#Velocities with the XSPEC vashift sign, widths in km/s (WIDTH_REF_KEV), line norms in 1e-5 ph/cm2/s
#(negative = absorption); a doublet's norm is that of the line given, the other is tied at twice it
PLOT_TITLES = {1: r'$N_{\rm H}^{\rm ISM}$', 3: 'PL norm', 74: r'$T_{\rm in}$', 75: 'disk norm',
               6: r'$\sigma_{\rm Fe\,XXVI}$', 13: r'$\sigma_{\rm Fe\,XXV}$', 36: r'$\sigma_{\rm S\,XVI}$',
               63: r'$\sigma_{\rm broad}$',
               7: r'$N_{\rm XXVI\,Ly\alpha_2}$', 10: r'$N_{\rm XXVI\,Ly\alpha_1}$', 14: r'$N_{\rm XXV\,w}$',
               18: r'$N_{\rm XXV\,y}$', 26: r'$N_{\rm XXV\,x}$', 33: r'$N_{\rm XXV\,He\beta}$',
               50: r'$N_{\rm XXVI\,Ly\beta}$', 37: r'$N_{\rm S\,XVI\,Ly\alpha_2}$', 60: r'$N_{8.296}$',
               64: r'$N_{\rm broad\,Ly\alpha_2}$', 70: r'$N_{6.660}$', 73: r'$N_{6.616}$'}
PLOT_TITLES.update({key: r'$v_{%s}$' % zone[-1] for zone, keys in ZONES.items() for key in keys
                    if key in VELOCITY_KEYS})
PLOT_LABELS = dict(PLOT_TITLES)
PLOT_LABELS.update({1: PLOT_TITLES[1] + r' ($10^{22}$ cm$^{-2}$)', 74: PLOT_TITLES[74] + ' (keV)'})
PLOT_LABELS.update({key: PLOT_TITLES[key] + ' (km/s)' for key in VELOCITY_KEYS + WIDTH_KEYS})
PLOT_LABELS.update({key: PLOT_TITLES[key] + r' ($10^{-5}$)' for key in NORM_KEYS})
PLOT_SCALE = {key: C_KMS / energy for key, energy in WIDTH_REF_KEV.items()}
PLOT_SCALE.update({key: 1e5 for key in NORM_KEYS})

#CORNER_EW: plot each free line norm as the equivalent width of its gaussian, in eV (negative =
#absorption, as XSPEC eqwidth): norm / (powerlaw + diskbb photon flux density at the observed line
#centre), per sample (contour_maker.gaussian_equivalent_widths; TBabs cancels). For a doublet, the EW
#of the line in the label (the other one, tied at twice the norm, has about twice the EW). Needs
#PyXspec in the plotting processes. File names get '_ew'.
CORNER_EW = True
#norm key -> velocity key of its zone (None: unshifted line)
EW_LINES = {key: ([elem for elem in keys if elem in VELOCITY_KEYS] + [None])[0]
            for keys in list(ZONES.values()) + [UNSHIFTED_KEYS] for key in keys if key in NORM_KEYS}
EW_TITLES = {key: PLOT_TITLES[key].replace('$N_', r'${\rm EW}_') for key in NORM_KEYS}
EW_LABELS = {key: EW_TITLES[key] + ' (eV)' for key in NORM_KEYS}

READ_ONLY = ('reassess', 'summary', 'corner', 'ew_check')

if WHAT not in READ_ONLY:
    import faulthandler; faulthandler.enable()
    from xspec import AllModels, Fit, Xset
    from chain_runner import build_param_table, run_chain

    # 1. load the model from the xcm directory (relative data and response paths), and stay in it
    os.chdir(SP_DIR)
    Xset.restore(XCM)
    Xset.addModelString('DISABLE_ANALYTIC_GRAD', 'yes')
    if 'cstat' not in Fit.statMethod.lower() or Fit.bayes != 'off':
        raise RuntimeError('expected cstat with bayes off after the restore, got %s, bayes %s'
                           % (Fit.statMethod, Fit.bayes))

    # 2. structure: the priors below are keyed by index, so the free parameters must be these ones
    table_xcm = build_param_table(LINE_COMPONENTS)
    free_now = sorted(row['key'] for row in table_xcm if row['free'])
    if free_now != FREE_KEYS:
        raise RuntimeError('free parameters of the xcm %s differ from the expected %s' % (free_now, FREE_KEYS))
    print('structure: D = %d free, all in the source model; nxb1: %d parameters, none free; %d links'
          % (len(free_now), sum(1 for row in table_xcm if row['model'] == 'nxb1'),
             sum(1 for row in table_xcm if row['link'])))

    # 3. prior ranges = hard limits
    for key, (lo, hi) in sorted(LIMITS.items()):
        par = AllModels(1)(key)
        val, delta = par.values[:2]
        if not lo <= val <= hi:
            raise ValueError('par %d = %g is outside its prior range [%g, %g]' % (key, val, lo, hi))
        par.values = [val, delta, lo, lo, hi, hi]

    #reference for the end-of-run check: frozen flags, links and limits of every parameter (nxb1
    #included), and the values of the frozen ones
    reference = {str(row['key']): row for row in build_param_table(LINE_COMPONENTS)}

    # 4. starting fit: is the xcm at the minimum, is the covariance complete, is any parameter
    #within 3 sigma of its prior range (a truncated posterior)?
    stat_xcm = Fit.statistic
    Fit.query = 'yes'
    Fit.perform()
    Fit.perform()
    print('start: C = %.3f (xcm values) -> %.3f after two fits, %d dof' % (stat_xcm, Fit.statistic, Fit.dof))
    if stat_xcm - Fit.statistic > 0.1:
        print('    the xcm was not at the minimum: run_chain starts from this refit')
    n_free = len(FREE_KEYS)
    if len(Fit.covariance) != n_free * (n_free + 1) // 2:
        print('WARNING: fit covariance has %d elements, %d expected: the "gaussian fit" walker start will'
              ' fall back to uniform deltas' % (len(Fit.covariance), n_free * (n_free + 1) // 2))
    print('widths in keV here (km/s only in the plots); lo/hi dist = distance to the limits in sigma')
    print('%-4s %-26s %12s %12s %10s %8s %8s %8s' % ('par', 'label', 'xcm', 'fit', 'sigma', 'shift/s',
                                                    'lo dist', 'hi dist'))
    for key in FREE_KEYS:
        par = AllModels(1)(key)
        val, lo, hi, sig = par.values[0], par.values[2], par.values[5], par.sigma
        if not sig > 0:
            print('%-4d %-26s %12.5g  no error (zero gradient?)' % (key, LABELS[key], val))
            continue
        flag = '  <- within 3 sigma of a limit' if min(val - lo, hi - val) < 3 * sig else ''
        print('%-4d %-26s %12.5g %12.5g %10.3g %8.2f %8.1f %8.1f%s'
              % (key, LABELS[key], reference[str(key)]['value'], val, sig,
                 (val - reference[str(key)]['value']) / sig, (val - lo) / sig, (hi - val) / sig, flag))


def check_preserved():
    '''
    After a run: frozen flags, links and hard limits of every parameter (nxb1 included), and the values
    of the frozen parameters, must be those set up in steps 1-3
    '''
    now = {str(row['key']): row for row in build_param_table(LINE_COMPONENTS)}
    fields = ('frozen', 'link', 'hard_min', 'hard_max')
    bad = [key for key, ref in reference.items()
           if key not in now or any(now[key][elem] != ref[elem] for elem in fields)
           or (ref['frozen'] and now[key]['value'] != ref['value'])]
    print('preserved frozen/linked/limits (%d parameters): %s' % (len(reference), 'PASS' if not bad else
                                                                  'FAIL for ' + ', '.join(bad)))


# 5. gw pilot: time per step, proposal actually used ("gaussian fit" or the fallback), sticky walkers.
#Its tau is a lower bound (10-100x too low on the pion model): it only sets a minimum length.
if WHAT == 'gw_pilot':
    os.makedirs(OUT_DIR, exist_ok=True)
    pilot = run_chain(PILOT, xcm=None, test=True, seed=4026, overwrite=True, priors=PRIORS,
                      absorber_components=LINE_COMPONENTS, walkers=WALKERS, n_parallel=N_PARALLEL)
    print(pilot['settings']['proposal'], pilot['diagnostics']['verdict'],
          pilot['diagnostics']['recommended_steps_per_walker'])
    check_preserved()

# 6. production chain: 10000 steps per walker after 2000 of burn, sized for tau up to about 200. If the
#verdict is WARN, drop more burn-in with 'reassess' first, then append with 'gw_continue'.
#run_chain refuses to overwrite an existing chain of this name.
if WHAT == 'gw_prod':
    os.makedirs(OUT_DIR, exist_ok=True)
    prod = run_chain(CHAIN_FILE, xcm=None, seed=4027, priors=PRIORS, absorber_components=LINE_COMPONENTS,
                     pilot=PILOT[:-5] + '.json', walkers=WALKERS, steps_per_walker=30000,
                     burn_per_walker=2000, n_parallel=N_PARALLEL)
    check_preserved()

# 7. append to the production chain (walkers restart from their last rows, no new burn). New seed for
#every append (5001, 5002, ...); a copy <file>.bak_cont<k> is made first.
#   production (seed 4027): 30000 steps per walker, burn 2000. WARN: R-hat 1.02-1.04, tau 580-930
#   continuation 1 (seed 5001, 40000 asked): stopped by hand after 38118 steps; the incomplete last step
#       was cut (NAXIS2 + truncation, untrimmed copy .fits.interrupted_cont1), recorded as 'interrupted'
#       in the sidecar -> 68118 steps per walker. Discard 4000: everything passes except R-hat 1.009-1.019
#   continuation 2 (seed 5002, 70000 asked): killed by a system shutdown after 14707 steps; trimmed the
#       same way (untrimmed copy .fits.interrupted_cont2), recorded 'interrupted' -> 82825 steps per walker.
#       Discard 4000: R-hat 1.007-1.016, 7 parameters still >= 1.01 (worst 7, 63, 64, 10, 61, 36)
#   continuation 3 (seed 5003): 60000 steps, about 19 h at 0.023 s per walker-step -> 142825 in total.
#       Max R-hat - 1 goes as about 1.25 / (thousands of steps after discard) on this chain (20k: 0.057,
#       44k: 0.024, 64k: 0.019, 79k: 0.016), so about 0.009 is expected (R-hat is noisy: little margin).
#       Check with WHAT='reassess' (DISCARD = 4000) and append again if needed.
CONTINUE_SEED = 5003
CONTINUE_STEPS = 70000
if WHAT == 'gw_continue':
    from chain_runner import continue_chain
    continue_chain(CHAIN_FILE, steps_per_walker=CONTINUE_STEPS, priors=PRIORS, seed=CONTINUE_SEED,
                   n_parallel=N_PARALLEL, absorber_components=LINE_COMPONENTS)
    check_preserved()

# 8. nested sampling, as a check for modes gw would miss (e.g. the broad/narrow Fe XXVI split). Every
#prior is bounded. Its logZ depends on the prior ranges, so it is not an evidence for model comparison
#unless the ranges are physical. NEST_PARALLEL = worker processes; untested on this model (None = 1).
NEST_PARALLEL = None
if WHAT == 'nest':
    os.makedirs(OUT_DIR, exist_ok=True)
    nest_run = run_chain(NEST_FILE, xcm=None, sampler='nest', seed=4029, priors=PRIORS,
                         absorber_components=LINE_COMPONENTS, n_parallel=NEST_PARALLEL)
    print(nest_run['nest'])
    check_preserved()

# ---- read-only blocks: a finished gw chain, no XSPEC

# 9. verdict after dropping the first DISCARD steps per walker (post-hoc burn-in). An OK verdict is
#written to the sidecar as its diagnostics; the other blocks must then use the same DISCARD.
if WHAT == 'reassess':
    from chain_diagnostics import reassess_chain
    diag = reassess_chain(CHAIN_FILE, discard=DISCARD)
    print('verdict %s (discard %d), tau %.0f steps, recommended steps per walker %s'
          % (diag['verdict'], DISCARD, diag['tau_used_all'], diag['recommended_steps_per_walker']))
    for elem in diag['failing'] + ['note: ' + elem for elem in diag['notes']]:
        print('    ' + elem)

# 10. posterior summary per parameter: median and 68% interval, the 90% one-sided bound away from 0 for
#the norms (deepest 10%) and widths (widest 10%), and the distance from the 0.5% and 99.5% quantiles to
#the lower and upper limits, in posterior standard deviations (in log for the jeffreys parameters): a
#distance near 0 = the posterior reaches the limit, i.e. is cut by the prior range. (A fraction of
#samples in 2% of the range, the first version, was meaningless on the wide [-1e-3, 0] norm ranges.)
if WHAT == 'summary':
    import numpy as np
    from contour_maker import load_chain
    chain = load_chain(CHAIN_FILE, discard=DISCARD)
    print('%d samples (%d walkers, discard %d, thin %d); widths in keV'
          % (len(chain['samples']), chain['n_walkers'], DISCARD, chain['thin']))
    print('%-4s %-26s %12s %25s %12s %7s %7s' % ('par', 'label', 'median', '68% interval', '90% 1-sided',
                                                'lo dist', 'hi dist'))
    for i, key in enumerate(chain['keys']):
        x = chain['samples'][:, i]
        row = chain['table'][str(key)]
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        one_sided = np.percentile(x, 10) if key in NORM_KEYS else np.percentile(x, 90) if key in WIDTH_KEYS \
            else np.nan
        lo, hi = row['hard_min'], row['hard_max']
        if row['prior'][0] == 'jeffreys':
            x, lo, hi = np.log10(x), np.log10(lo), np.log10(hi)
        q_lo, q_hi = np.percentile(x, [0.5, 99.5])
        print('%-4s %-26s %12.5g [%11.5g, %11.5g] %12.5g %7.2f %7.2f'
              % (key, LABELS.get(key, key), q50, q16, q84, one_sided, (q_lo - lo) / x.std(),
                 (hi - q_hi) / x.std()))

# 11. corner plots (CORNER_FIGURES) into CORNER_DIR, as <chain file name>_<set>[_split<k>][_paper]
#_<backend>.png/.pdf: zones of linked velocity, pairs of zones, the unshifted lines, the continuum, all.
#jeffreys parameters in log10. CORNER_SPLIT = k overlays k consecutive parts of the chain (stationarity
#check; no paper figure then). 'all' (24 parameters) uses only the matplotlib backend: the corner package
#needs several GB for a figure that size. One process per figure, as for the pion chain: the memory of
#a large chain does not all come back between figures.
#CORNER_FILE: the chain to plot. While a continuation runs, the chain file is being written: use its
#backup <file>.bak_cont<k> (the chain before that continuation, with the same sidecar)
CORNER_FILE = CHAIN_FILE
CORNER_SPLIT = None
if WHAT == 'corner':
    stem = os.path.join(CORNER_DIR, os.path.basename(CORNER_FILE).replace('.fits', ''))
    for backend, name, paper in CORNER_FIGURES:
        if paper and CORNER_SPLIT:
            continue
        keys = CORNER_SETS[name]
        use_ew = CORNER_EW and any(key in NORM_KEYS for key in keys)
        tag = name + ('_split%d' % CORNER_SPLIT if CORNER_SPLIT else '') + ('_paper' if paper else '') \
            + ('_ew' if use_ew else '')
        labels, titles, scale = dict(PLOT_LABELS), dict(PLOT_TITLES), dict(PLOT_SCALE)
        if use_ew:
            labels.update(EW_LABELS)
            titles.update(EW_TITLES)
            scale = {key: val for key, val in scale.items() if key not in NORM_KEYS}
        kw = dict(params=keys, labels=labels, titles=titles, scale=scale, discard=DISCARD,
                  split=CORNER_SPLIT, paper=paper, backend=backend, panel_size=1.6 if len(keys) > 8 else 2.2,
                  save=stem + '_' + tag)
        print('corner: %s, %s' % (tag, backend), flush=True)
        derived = 'derived=gaussian_equivalent_widths(%r), ' % EW_LINES if use_ew else ''
        subprocess.run([sys.executable, '-c', 'from contour_maker import corner_chain, gaussian_equivalent_widths\n'
                        'corner_chain(%r, %s**%r)' % (CORNER_FILE, derived, kw)], cwd=POSTERIOR_DIR, check=True)

# 12. the equivalent widths of the corner plots (CORNER_EW) against XSPEC eqwidth, at the best fit and on
#EW_CHECK_SAMPLES chain samples of CORNER_FILE (contour_maker.compare_equivalent_widths: XSPEC default,
#narrow continuum range, narrow with the other lines set to 0). Runs in its own process (it needs an empty
#XSPEC session, then loads the data of the best-fit xcm).
EW_CHECK_SAMPLES = 20
if WHAT == 'ew_check':
    subprocess.run([sys.executable, '-c', 'from contour_maker import compare_equivalent_widths\n'
                    'compare_equivalent_widths(%r, %r, n_samples=%d, discard=%d)'
                    % (CORNER_FILE, EW_LINES, EW_CHECK_SAMPLES, DISCARD)], cwd=POSTERIOR_DIR, check=True)

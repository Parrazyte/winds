"""
Created by Claude Code, model Sonnet 5.5 (claude-sonnet-5-5), 2026-10-22.

Test of run_chain() on the MAXI J1543-564 XRISM/Resolve pion model (TBabs x 3 pion absorbers x
(powerlaw + diskbb) + pion emission table). Running the whole file does: restore the model, set the
parameter limits and priors, then the ONE run (or check) chosen by the WHAT variable below (default:
nested sampling, step 4). Each other run or check is a block selected by WHAT (steps 5-11).

Why nest and not gw: the first all-free gw chain (40 walkers x 1500 steps) did not converge (R-hat
1.3-1.45, ESS 60-90, tau about 200). The cause is absorber 1: its (log Nh, log v_rms) posterior is a
narrow curved ridge about 1.4 dex long, clipped at the v_rms = 3 km/s hard limit (step 11 plots it),
and gw samples in linear Nh and v_rms where such a ridge is a very poor shape for an affine-invariant
ensemble. nest needs no tau and no shape assumption. Its cost on this model is unknown (only tested on
a fake spectrum so far): watch the first minutes.

Why the model is restored here and not by run_chain(xcm=...): the prior ranges are the HARD limits
of the parameters, and a log-uniform prior needs a positive lower limit. The limits are set below
after the restore, and run_chain(xcm=...) would restore the xcm again and overwrite them. The
session must stay in the pion directory (the xcm names the tables by relative path, moving away
crashes the fit); run_chain does not move it when xcm is None.

Priors (family 'jeffreys' = uniform in ln(theta), p(theta) ~ 1/theta: equal probability per decade;
family 'cons' = uniform in theta):
    jeffreys  TBabs nH, powerlaw norm, diskbb norm, pion Nh and v_rms of the 3 absorbers, and the
              covering factor (pion_emit.norm)
    cons      rlogxi (already a log10, so flat is already log-uniform: jeffreys on it would be
              wrong and load_priors refuses it), absorber z (a velocity, linear in z), diskbb Tin
Only the limits of the data-constrained parameters (nH, norms, Tin) are changed, to a window far
wider than their posterior; the Nh and v_rms limits stay those of the pion tables.
"""

import os
import sys

POSTERIOR_DIR = '/home/parrazyte/Documents/Work/Scripts/Python/winds/observations/spectral_analysis/posterior'
sys.path.insert(0, POSTERIOR_DIR)

import faulthandler; faulthandler.enable()
from xspec import AllModels, Xset
from chain_runner import run_chain

PION_DIR = '/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion'
XCM = 'fit_3comp1em_gamma2.xcm'
OUT_DIR = os.path.join(PION_DIR, 'posterior')
PILOT = os.path.join(OUT_DIR, 'MAXIJ1543_3pion_pilot.fits')
ALL_FREE = os.path.join(OUT_DIR, 'MAXIJ1543_3pion_all_free.fits')
#chain file read by the 'drift' and 'plot' blocks: the long gw run (3 h 53 min, R-hat 1.03-1.06, ESS 320-1130)
CHAIN_FILE = os.path.join(OUT_DIR, 'MAXIJ1543_3pion_all_free_long.fits')
#the covering factor is part of the absorber, tagged as such in the sidecar (corner plot groups)
ABSORBERS = ('mtable', 'pion_abs', 'pion_emit')
CONTINUUM = ('TBabs.nH', 'powerlaw.norm', 'diskbb.Tin', 'diskbb.norm')

#WHAT TO RUN: one value, the rest of the file always does steps 1-3 (restore, limits, priors) first.
#    'nest'            step 4: nested sampling, ONE process, about 12 GB of memory, 1 core
#    'gw_long'         step 7: long gw chain, 20 workers, about 4 GB, 20 cores
#    'gw_pilot', 'gw_prod', 'gw_conditional', 'gw_medium'   the other gw runs (steps 5, 6, 8, 9)
#    'drift', 'plot'   steps 10, 11: read an existing chain file, no sampling
#    'gw_continue_test'  step 12: test of the append on a copy of the pilot chain (3 min)
#    'gw_continue'       step 13: append 63125 steps per walker to the long chain (about 11 h, 10 workers)
#    anything else (e.g. 'setup')   only steps 1-3
#'nest' and 'gw_long' can run AT THE SAME TIME, each in its own console (24 cores, about 16-17 GB in
#total, keep 5 GB free); they write to different files, so set WHAT in each console before running.
WHAT = 'gw_continue'
#gw evaluates the 20 proposals of each half-ensemble on min(n_parallel, 20) workers, in rounds (XSPEC
#Chain.cxx, ProcessManager.cxx): 10 workers = 2 rounds, as 14 does. Measured 0.0151 s per walker-step
#with 20 workers and 0.0163 s with 14: above 10 the machine (12 physical cores) is the limit, and each
#busy worker caches about 0.25 GB of pion table privately.
n_parallel_standard=10
#if the machine runs out of memory the kernel kills the process with the highest oom score first,
#normally the biggest one (nest, 12 GB, hours of work). The gw runs are the cheaper ones to lose and
#they use little memory, so they are made preferred victims: the score adjustment is inherited by the
#XSPEC worker processes forked later. Raising your own adjustment needs no privilege.
if WHAT.startswith('gw'):
    with open('/proc/self/oom_score_adj', 'w') as f:
        f.write('500')

# 1. load the model, from the pion directory, and stay in it
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(PION_DIR)
Xset.restore(XCM)

#finite-difference gradients instead of the XSPEC 13.0.1 analytic ones. The analytic gradient is zero
#for the mtable z of absorbers 1 and 3 (params 5, 13): LM then cannot move them and the fit covariance
#is 15 x 15 instead of 17 x 17, which breaks the "gaussian fit" walker start of gw. It only affects
#the refit that run_chain does before sampling (best fit, covariance), not the likelihood that nest
#and gw evaluate, so nest does not need it: it is left on so that the refit and the best fit saved
#in the sidecar are correct. Comment this line to go back to the XSPEC default.
Xset.addModelString('DISABLE_ANALYTIC_GRAD', 'yes')

# 2. hard limits = prior range
def set_limits(key_component, par_name, lo=None, hi=None):
    par = getattr(getattr(AllModels(1), key_component), par_name)
    val, delta, hard_min, _, _, hard_max = par.values
    lo = hard_min if lo is None else lo
    hi = hard_max if hi is None else hi
    par.values = [val, delta, lo, lo, hi, hi]
    print('%s.%s limits [%g, %g]' % (key_component, par_name, lo, hi))

set_limits('TBabs', 'nH', lo=0.01, hi=100)           #best fit 1.85 +- 0.05, 1e22 cm-2
set_limits('powerlaw', 'norm', lo=1e-5, hi=10)       #best fit 9e-3 +- 6e-4
set_limits('diskbb', 'norm', lo=1, hi=1e5)           #best fit 160 +- 4
set_limits('diskbb', 'Tin', lo=0.1, hi=5)            #best fit 0.903 +- 0.003 keV
#covering factor: the upper limit is the physical one (4 pi covering, already in the xcm), the
#lower one is 1/1000 of it: below that the covering factor is unconstrained by the data
cov_max = AllModels(1).pion_emit.norm.values[5]
set_limits('pion_emit', 'norm', lo=cov_max / 1000.)

# 3. priors
priors = {
    'TBabs.nH': 'jeffreys',
    'powerlaw.norm': 'jeffreys',
    'diskbb.norm': 'jeffreys',
    'diskbb.Tin': 'cons',
    'pion_emit.norm': 'jeffreys',
}
for comp in ('pion_abs', 'pion_abs_3', 'pion_abs_4'):
    priors[comp + '.rlogxi'] = 'cons'
    priors[comp + '.Nh/1.0e22'] = 'jeffreys'
    priors[comp + '.v_rms_[km/s]'] = 'jeffreys'
    priors[comp + '.z'] = 'cons'

# 4. nested sampling (the run of this file). Handles the curved absorber-1 ridge and the bimodality
#flag on the covering factor without tau, and gives the evidence logZ. Every free parameter has a
#bounded prior, as nest requires. nLive (400) and the termination tolerance are XSPEC defaults, not
#run_chain arguments. There is no MCMC verdict: read nest_run['nest'] (converged, n_modes, n_like,
#logZ) and the equal-weight resampled posterior samples in the fits file.
#run_chain refuses to start if the .fits, .json, .log or _bestfit.xcm of this name already exist
#(also from an interrupted run), so rerunning the file stops here instead of overwriting hours of
#work: pass overwrite=True to start over. Resuming from the nest checkpoint (<file>.ckpt.fits) is
#untested: from the code it needs the .json, .log and _bestfit.xcm of the interrupted run deleted
#(keep the checkpoint), then a rerun with overwrite=False.
#NEST_PARALLEL: number of WORKER PROCESSES (XSPEC "parallel nest K", capped at nLive/10 = 40).
#With 20 the first attempt (2026-10-23) was killed by the kernel (out of memory, 30 GB, no swap)
#about 30 s after the start, at iteration ~300. The exact cause is not established: gw runs with 20
#parallel walkers (also XSPEC "processes") did not run out of memory, so nest's workers must use
#much more memory each than gw's (own copy of the model and of the 2.8 GB pion tables is the
#suspect), or something else was holding memory. None = a single process (safe: what the first gw
#pilots used). To speed up: start with None, look at the RSS of the python
#process (`ps -o rss= -p <pid>`, in kB, after the first minute), then set
#K <= (free memory in GB - 4) / (RSS in GB), and close the other XSPEC consoles first.
NEST_PARALLEL = None
if WHAT == 'nest':
    nest_run = run_chain(os.path.join(OUT_DIR, 'MAXIJ1543_3pion_nest.fits'), xcm=None, sampler='nest',
                         seed=2029, priors=priors, absorber_components=ABSORBERS,
                         n_parallel=NEST_PARALLEL)
    print(nest_run['nest'])

# ---- the blocks below run only if WHAT selects them. Blocks 5-9 need steps 1-3 run in the same
# ---- console (restore, limits, priors, always done); blocks 10-11 only read an existing chain file.

# 5. gw pilot (100 steps per walker): gives tau, time per step and the proposal actually used. The
#tau of a 100-step pilot is a large underestimate here (14 against about 200 measured later).
if WHAT == 'gw_pilot':
    pilot = run_chain(PILOT, xcm=None, test=True, seed=2026, overwrite=True, priors=priors,
                      absorber_components=ABSORBERS, walkers=40, n_parallel=n_parallel_standard)
    print(pilot['settings']['proposal'], pilot['diagnostics']['verdict'],
          pilot['diagnostics']['recommended_steps_per_walker'])

# 6. gw production chain, all parameters free (26 min, did NOT converge at this length). walkers=40
#(D=17: 2D=34 <= 40 <= 4D=68) divides evenly over n_parallel=n_parallel_standard: 2 rounds, no idle core. The pilot
#tau is flagged unreliable, so the length check plans with 2 x tau: with the 100-step pilot,
#steps_per_walker >= 1440, burn >= 144. Run step 5 first for the pilot sidecar.
if WHAT == 'gw_prod':
    prod = run_chain(ALL_FREE, xcm=None, seed=2027, priors=priors, absorber_components=ABSORBERS,
                     pilot=PILOT[:-5] + '.json', walkers=40, steps_per_walker=1500,
                     burn_per_walker=300, n_parallel=n_parallel_standard)

# 7. much longer gw chain, all parameters free (about 5.7 h: 40 walkers x 23100 steps at 0.022 s).
#Using the first chain's sidecar as the pilot: its tau (208) is unreliable (1500 steps = 7 tau), so
#the check plans with 2 x 208 = 416: steps_per_walker >= 20800, burn >= 2080. The tau of THIS run
#will again be a lower bound if it is shorter than 50 tau: read the post-run verdict.
if WHAT == 'gw_long':
    long_run = run_chain(os.path.join(OUT_DIR, 'MAXIJ1543_3pion_all_free_long.fits'), xcm=None, seed=2030,
                         priors=priors, absorber_components=ABSORBERS, pilot=ALL_FREE[:-5] + '.json',
                         walkers=40, steps_per_walker=21000, burn_per_walker=2100, n_parallel=n_parallel_standard)

# 8. conditional gw chain: continuum frozen at its best fit (D = 13), only the absorbers and the
#covering factor free. Flagged as conditional in the sidecar; its priors cannot include the frozen
#parameters. The pilot check warns that the free parameters differ from the pilot's: expected.
if WHAT == 'gw_conditional':
    cond_long = run_chain(os.path.join(OUT_DIR, 'MAXIJ1543_3pion_conditional_long.fits'), xcm=None,
                          seed=2031, priors={k: v for k, v in priors.items() if k not in CONTINUUM},
                          freeze=CONTINUUM, absorber_components=ABSORBERS, pilot=ALL_FREE[:-5] + '.json',
                          walkers=40, steps_per_walker=21000, burn_per_walker=2100, n_parallel=n_parallel_standard)

# 9. medium gw chain (40 walkers x 3000 steps, burn 500, about 50 min) with the finite-difference
#"gaussian fit" start, as an A/B test against the first all-free chain. Probably not informative any
#more: the ridge of absorber 1 is not fixed by a better starting ball. Needs step 5 for the pilot.
if WHAT == 'gw_medium':
    medium = run_chain(os.path.join(OUT_DIR, 'MAXIJ1543_3pion_fd_medium.fits'), xcm=None, seed=2032,
                       priors=priors, absorber_components=ABSORBERS, pilot=PILOT[:-5] + '.json',
                       walkers=40, steps_per_walker=3000, burn_per_walker=500, n_parallel=n_parallel_standard)

# 10. is a finished gw chain still drifting? read-only, works on any gw chain file of this model.
#per parameter: shift of the walker-averaged position between the first and last third of the
#chain, in units of the overall std (>> 0.3: not stationary), and the std ratio last/first third
#(far from 1: the ensemble was still expanding or contracting). Wide positive parameters in log10.
if WHAT == 'drift':
    import numpy as np
    from chain_diagnostics import read_chain
    chain = read_chain(CHAIN_FILE)
    samples, stat = chain['samples'], chain['statistic']       #(walker, step, par), (walker, step)
    n_third = samples.shape[1] // 3
    print('%-14s %8s %9s' % ('par', 'drift/sd', 'sd ratio'))
    for j, label in enumerate(chain['labels']):
        x = samples[:, :, j]
        if x.min() > 0 and x.max() / x.min() > 100:
            x = np.log10(x)
        first, last = x[:, :n_third], x[:, -n_third:]
        print('%-14s %8.2f %9.2f' % (label[:14], (last.mean() - first.mean()) / x.std(),
                                     last.std() / first.std()))
    print('FIT_STATISTIC: min %.2f, median %.2f, first third median %.2f, last third median %.2f'
          % (stat.min(), np.median(stat), np.median(stat[:, :n_third]), np.median(stat[:, -n_third:])))

# 11. shape of the posterior in (log Nh, log v_rms) for each of the three absorbers, from a gw chain
#file. Result on the first chain: a narrow curved ridge in absorber 1 clipped at v_rms = 3 km/s,
#compact round clouds in absorbers 3 and 4. Colour = step (blue early, yellow late): a cloud still
#growing in the late steps means the chain has not converged.
if WHAT == 'plot':
    import numpy as np
    import matplotlib.pyplot as plt
    from chain_diagnostics import read_chain
    chain = read_chain(CHAIN_FILE)
    samples, keys = chain['samples'], list(chain['keys'])
    n_steps = samples.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key_nh, key_v), title in zip(axes, [(3, 4), (7, 8), (11, 12)],
                                          ['absorber 1', 'absorber 3', 'absorber 4']):
        x = np.log10(samples[:, :, keys.index(key_nh)])
        y = np.log10(samples[:, :, keys.index(key_v)])
        step = np.broadcast_to(np.arange(n_steps), x.shape)
        pts = ax.scatter(x.ravel(), y.ravel(), c=step.ravel(), s=2, cmap='viridis')
        ax.set_xlabel('log10 Nh / 1e22'); ax.set_ylabel('log10 v_rms (km/s)'); ax.set_title(title)
    fig.colorbar(pts, ax=axes, label='step')
    #covering factor (pion_emit.norm, key 22): log10 histogram, with the two hard limits. The bimodality
    #flag (BC 0.63) may only be the pile-up at the physical upper limit plus the flat log-uniform tail
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(np.log10(samples[:, :, keys.index(22)].ravel()), bins=80)
    for lim in (1.73001e-13, 1.73001e-10):
        ax2.axvline(np.log10(lim), color='r', ls='--')
    ax2.set_xlabel('log10 covering factor norm (pion_emit.norm)')
    plt.show()

# 12. test of continue_chain on a COPY of the 100-step pilot chain (about 3 minutes): appends 40 steps
#per walker and checks that (1) the rows written before are untouched, (2) the file has exactly the
#expected number of rows, (3) the first appended step of each walker continues from its last
#position (a restart from a fresh random cloud would move every walker by about one posterior width),
#(4) the sidecar of the copy gets one "continuations" entry and the backup file exists. The real chain
#is not touched. Needs the same session as the pilot_fd run: steps 1-3 above (priors and limits).
if WHAT == 'gw_continue_test':
    import shutil
    import numpy as np
    from chain_runner import continue_chain
    from chain_diagnostics import read_chain
    src = os.path.join(OUT_DIR, 'MAXIJ1543_3pion_pilot_fd')
    test_dir = os.path.join(OUT_DIR, 'append_test')
    os.makedirs(test_dir, exist_ok=True)
    dst = os.path.join(test_dir, 'append_test')
    for suffix in ('.fits', '.json'):
        shutil.copy2(src + suffix, dst + suffix)
    for leftover in ('_cont1.log', '.fits.bak_cont1'):          #from an earlier test, if any
        if os.path.exists(dst + leftover):
            os.remove(dst + leftover)
    before = read_chain(dst + '.fits')['samples']                #(walker, step, par)
    side = continue_chain(dst + '.fits', steps_per_walker=40, priors=priors, seed=3001, n_parallel=n_parallel_standard,
                          absorber_components=ABSORBERS)
    after = read_chain(dst + '.fits')['samples']
    n_w, n_before, n_par = before.shape
    print('rows: walkers %d, steps %d -> %d (expected %d)' % (n_w, n_before, after.shape[1], n_before + 40))
    print('PASS' if after.shape == (n_w, n_before + 40, n_par) else 'FAIL', ': shape')
    print('PASS' if np.array_equal(after[:, :n_before], before) else 'FAIL', ': old rows untouched')
    #a gw walker either moves (all parameters change) or rejects the move and keeps its row EXACTLY,
    #so continuity is tested on the fraction of walkers whose row is unchanged from one step to the
    #next: at the junction it must be like inside the old part (about 1 - acceptance), whereas a
    #restart from a fresh random cloud would give about 0 (a median of |differences| would be 0 in
    #both cases, as acceptance is below 50%: that first version of this test was empty)
    unchanged = np.all(after[:, 1:n_before] == after[:, :n_before - 1], axis=2).mean()
    at_junction = np.all(after[:, n_before] == after[:, n_before - 1], axis=1).mean()
    print('fraction of walkers with an unchanged row: old part %.2f, at the junction %.2f' %
          (unchanged, at_junction))
    print('PASS' if at_junction > 0.5 * unchanged > 0 else 'FAIL', ': the appended steps continue from the last rows')
    print('PASS' if len(side['continuations']) == 1 and os.path.exists(dst + '.fits.bak_cont1') else 'FAIL',
          ': sidecar entry and backup')

# 13. the real continuation of the long chain, in segments, restarting the console in between (this frees
#the table cache memory, and a crash costs at most one segment). Target 173000 steps per walker, as the
#172596 recommended by the first diagnostics.
#   continuation 1 (seed 3001, 14 workers, 2026-10-23): OOM-killed after 12875 of 76000 steps; the file was
#       trimmed to 21000 + 12875 = 33875 steps per walker and recorded as 'crashed' in the sidecar
#   continuation 2 (seed 3002): 76000 - 12875 = 63125 steps per walker, about 11 h at 0.016 s per walker-step
#   continuation 3 (seed 3003): 76000 steps per walker, reaching 173000
#Safety: the chain file is copied to <file>.bak_cont<k> first, XSPEC writes the rows to the file as it
#runs (a crash keeps what was written), the sidecar is only updated after a complete append. Needs the
#same session as the long run: steps 1-3 above. Close the other applications first (memory), and do not
#make plots of the chain while it runs.
CONTINUE_SEED = 3003
CONTINUE_STEPS = 100000
if WHAT == 'gw_continue':
    from chain_runner import continue_chain
    continue_chain(CHAIN_FILE, steps_per_walker=CONTINUE_STEPS, priors=priors, seed=CONTINUE_SEED, n_parallel=n_parallel_standard,
                   absorber_components=ABSORBERS)

"""
Created by Claude Code, model Sonnet 5.5 (claude-sonnet-5-5), 2026-10-23.

How do XSPEC's parallel workers behave in gw (Xset.parallel.walkers) and in nest (parallel nest K)
on the MAXI J1543-564 pion model? The first nest run with 20 workers was killed by the kernel
(out of memory, 30 GB, no swap) 30 s after the start, while gw runs with 20 were fine.

Each configuration (sampler, K workers) runs in a FRESH python process (the XSPEC session is global
and keeps its memory), launched and watched by this script, which records every second:
    -system memory: MemAvailable from /proc/meminfo (page cache counts as available, so the drop is
     the memory really taken by the processes)
    -PSS of the process tree (psutil + /proc/<pid>/smaps_rollup): memory shared between forked
     workers is counted once, unlike RSS; the largest single process and the number of processes
    -nest: the iteration reached over time, parsed from the XSPEC log ("nest iter N")
    -gw: the duration of the chain only (log markers: "All chains are now removed" at the start,
     "is now loaded" at the end), not the model loading and the refit
and kills the whole process group if MemAvailable falls under FLOOR_GB, or when a nest run has had
NEST_SECONDS since its first iteration. It does not start a configuration whose predicted memory
(from the previous K of the same sampler, scaled linearly in K) would go under the floor.

Run it when nothing else heavy is running, and with the other XSPEC consoles closed: their memory
would distort the numbers. About 10 minutes in MEMORY_ONLY mode (the default), 30 with False. Results: a table, and parallel_test.json.
Not run yet (written without being able to execute python).

Usage: run this file (PyCharm "Run", or `python test_parallel_MAXIJ1543.py` with the astro venv).
The `--child` mode is internal.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time

POSTERIOR_DIR = '/home/parrazyte/Documents/Work/Scripts/Python/winds/observations/spectral_analysis/posterior'
SCRIPT = os.path.join(POSTERIOR_DIR, 'test_parallel_MAXIJ1543.py')
PION_DIR = '/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion'
XCM = 'fit_3comp1em_gamma2.xcm'
TEST_DIR = os.path.join(PION_DIR, 'posterior', 'parallel_test')
ABSORBERS = ('mtable', 'pion_abs', 'pion_emit')

#(sampler, workers); 1 = a single process. Increasing K per sampler: a larger K is only started
#if the memory predicted from the previous one stays above the floor
MEMORY_ONLY = False      #memory only (about 10 min): no refit, each run is stopped OBSERVE_SECONDS after its
                        #workers start. False: also measure the speed (the gw test chain to its end, nest
                        #for NEST_SECONDS), about 30 min
OBSERVE_SECONDS = 30
CONFIGS = [('gw', 1), ('nest', 1), ('gw', 4), ('nest', 4),
           ('gw', 8), ('nest', 8)]     #20 left out on purpose: the nest run with 20 was the crash
FLOOR_GB = 5.0          #kill the run if MemAvailable falls under this
NEST_SECONDS = 150      #a nest run is stopped this long after its first iteration line
CAP_SECONDS = {'gw': 1800, 'nest': 900}

MARKERS = {'chain_start': 'All chains are now removed', 'chain_end': 'is now loaded',
           'nest_start': 'nest: starting run', 'nest_pool': 'parallel pool with'}


def setup_session():
    '''Steps 1-3 of posterior_pion_MAXIJ1543.py: restore, limits, priors. Returns the priors.'''
    from xspec import AllModels, Xset

    os.chdir(PION_DIR)
    Xset.restore(XCM)
    Xset.addModelString('DISABLE_ANALYTIC_GRAD', 'yes')     #complete covariance for the refit

    def set_limits(comp, par_name, lo=None, hi=None):
        par = getattr(getattr(AllModels(1), comp), par_name)
        val, delta, hard_min, _, _, hard_max = par.values
        lo = hard_min if lo is None else lo
        hi = hard_max if hi is None else hi
        par.values = [val, delta, lo, lo, hi, hi]

    set_limits('TBabs', 'nH', lo=0.01, hi=100)
    set_limits('powerlaw', 'norm', lo=1e-5, hi=10)
    set_limits('diskbb', 'norm', lo=1, hi=1e5)
    set_limits('diskbb', 'Tin', lo=0.1, hi=5)
    set_limits('pion_emit', 'norm', lo=AllModels(1).pion_emit.norm.values[5] / 1000.)

    priors = {'TBabs.nH': 'jeffreys', 'powerlaw.norm': 'jeffreys', 'diskbb.norm': 'jeffreys',
              'diskbb.Tin': 'cons', 'pion_emit.norm': 'jeffreys'}
    for comp in ('pion_abs', 'pion_abs_3', 'pion_abs_4'):
        priors[comp + '.rlogxi'] = 'cons'
        priors[comp + '.Nh/1.0e22'] = 'jeffreys'
        priors[comp + '.v_rms_[km/s]'] = 'jeffreys'
        priors[comp + '.z'] = 'cons'
    return priors


def child(sampler, k, stem):
    '''One configuration, in this fresh process. gw: the 100-step test chain with 40 walkers.'''
    sys.path.insert(0, POSTERIOR_DIR)
    from chain_runner import run_chain

    priors = setup_session()
    kwargs = dict(xcm=None, priors=priors, absorber_components=ABSORBERS, overwrite=True,
                  refit=not MEMORY_ONLY,
                  seed=1, n_parallel=None if k == 1 else k)
    if sampler == 'gw':
        run_chain(stem + '.fits', test=True, walkers=40, **kwargs)
    else:
        run_chain(stem + '.fits', sampler='nest', **kwargs)


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576.
    return float('nan')


def pss_gb(pid):
    try:
        with open('/proc/%d/smaps_rollup' % pid) as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1]) / 1048576.
    except OSError:
        pass
    return 0.


def read_text(path):
    try:
        with open(path, errors='replace') as f:
            return f.read()
    except OSError:
        return ''


def kill_group(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        time.sleep(3)


def run_config(sampler, k):
    import psutil

    stem = os.path.join(TEST_DIR, 'ptest_%s_k%d' % (sampler, k))
    log_path = stem + '.log'
    for suffix in ('.fits', '.json', '.log', '_bestfit.xcm', '.fits.ckpt.fits'):
        if os.path.exists(stem + suffix):
            os.remove(stem + suffix)

    avail0 = mem_available_gb()
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, SCRIPT, '--child', sampler, str(k), stem],
                            start_new_session=True)
    root = psutil.Process(proc.pid)

    res = {'sampler': sampler, 'k': k, 'avail_start_gb': avail0, 'status': 'finished',
           'marks': {}, 'iters': [], 'chain_used_gb': 0., 'chain_pss_tree_gb': 0.,
           'chain_pss_max_proc_gb': 0., 'chain_n_proc': 1, 'used_gb_overall': 0.}

    while proc.poll() is None:
        time.sleep(1.)
        t = time.time() - t0
        avail = mem_available_gb()
        res['used_gb_overall'] = max(res['used_gb_overall'], avail0 - avail)

        text = read_text(log_path)
        for name, pattern in MARKERS.items():
            if name not in res['marks'] and pattern in text:
                res['marks'][name] = t
        iters = re.findall(r'nest iter\s+(\d+)', text)
        if iters and (not res['iters'] or int(iters[-1]) > res['iters'][-1][1]):
            res['iters'].append((t, int(iters[-1])))

        in_chain = 'chain_start' in res['marks'] or 'nest_start' in res['marks']
        if in_chain:
            try:
                pids = [root.pid] + [p.pid for p in root.children(recursive=True)]
            except psutil.NoSuchProcess:
                pids = []
            pss = [pss_gb(pid) for pid in pids]
            res['chain_used_gb'] = max(res['chain_used_gb'], avail0 - avail)
            res['chain_pss_tree_gb'] = max(res['chain_pss_tree_gb'], sum(pss))
            res['chain_pss_max_proc_gb'] = max(res['chain_pss_max_proc_gb'], max(pss, default=0.))
            res['chain_n_proc'] = max(res['chain_n_proc'], len(pids))

        if avail < FLOOR_GB:
            res['status'] = 'KILLED: MemAvailable %.1f GB < floor' % avail
            break
        started = res['marks'].get('chain_start' if sampler == 'gw' else 'nest_start')
        if MEMORY_ONLY and started is not None and t - started > OBSERVE_SECONDS:
            res['status'] = 'stopped after %d s in the chain' % OBSERVE_SECONDS
            break
        if (not MEMORY_ONLY) and sampler == 'nest' and res['iters'] and t - res['iters'][0][0] > NEST_SECONDS:
            res['status'] = 'stopped after %d s of nest iterations' % NEST_SECONDS
            break
        if t > CAP_SECONDS[sampler]:
            res['status'] = 'KILLED: time cap'
            break

    if proc.poll() is None:
        kill_group(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    elif proc.returncode != 0:
        res['status'] = 'child failed (return code %d)' % proc.returncode

    res['wall_s'] = time.time() - t0
    marks = res['marks']
    if sampler == 'gw' and 'chain_start' in marks and 'chain_end' in marks:
        res['chain_s'] = marks['chain_end'] - marks['chain_start']
    res['log_pool_line'] = [line.strip() for line in read_text(log_path).splitlines()
                            if 'parallel pool' in line][:1]
    time.sleep(5)                       #let the memory be released before the next measurement
    return res


def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    results = []
    for sampler, k in CONFIGS:
        previous = [r for r in results if r['sampler'] == sampler and r['chain_used_gb'] > 0]
        if k > 1 and previous:
            last = previous[-1]
            predicted = last['chain_used_gb'] * k / last['k']
            if mem_available_gb() - predicted < FLOOR_GB + 2.:
                print('SKIP %s K=%d: predicted %.1f GB used (%.1f GB for K=%d), %.1f GB available'
                      % (sampler, k, predicted, last['chain_used_gb'], last['k'], mem_available_gb()))
                continue
        print('--- %s, K=%d ---' % (sampler, k), flush=True)
        res = run_config(sampler, k)
        results.append(res)
        print('    %s | memory used during the chain %.1f GB | %d process(es)'
              % (res['status'], res['chain_used_gb'], res['chain_n_proc']), flush=True)
        if res['status'].startswith('KILLED: MemAvailable'):
            print('memory floor hit: stopping the matrix here')
            break

    #nest: time to reach the same iteration in every run (a later iteration costs more
    #likelihoods than an early one, so rates at different iterations are not comparable)
    nest_runs = [r for r in results if r['sampler'] == 'nest' and r['iters']]
    common = min((r['iters'][-1][1] for r in nest_runs), default=None)
    for r in nest_runs:
        r['t_to_common'] = next((t - r['marks'].get('nest_start', r['iters'][0][0])
                                 for t, n in r['iters'] if n >= common), None)

    print('\nsampler  K  status                                     chain  chain mem  PSS tree  PSS max'
          '  procs')
    print('                                                       time   used GB    GB        proc GB')
    for r in results:
        if r['sampler'] == 'gw':
            t = '%5.0f s (%d steps)' % (r['chain_s'], 40 * 120) if 'chain_s' in r else '?'
        else:
            t = ('%5.0f s to iter %d' % (r['t_to_common'], common)) if r.get('t_to_common') is not None else '?'
        print('%-6s %3d  %-42s %-24s %6.1f   %6.1f   %6.1f   %3d'
              % (r['sampler'], r['k'], r['status'][:42], t, r['chain_used_gb'],
                 r['chain_pss_tree_gb'], r['chain_pss_max_proc_gb'], r['chain_n_proc']))

    with open(os.path.join(TEST_DIR, 'parallel_test.json'), 'w') as f:
        json.dump(results, f, indent=1, default=str)
    print('written:', os.path.join(TEST_DIR, 'parallel_test.json'))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()

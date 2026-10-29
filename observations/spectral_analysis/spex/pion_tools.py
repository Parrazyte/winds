"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-28.

Grids of SPEX pion photoionisation spectra computed from a tabulated SED, and their conversion into
XSPEC table models. The computation and the table writing are adapted from R. Tomaru's
make_pion_mtable.py and piontxt2_xspec_mtable_new.py (V4641 Sgr version, emailed 2025-01-07, in
Scripts/Python/Ryota/script_pion_model/script_pion_model/).

Two outputs per grid point, both on the SPEX energy grid of the setup:
    abs   transmission = pion output / input, with the absorber covering the source (fcov_abs,
          default 1) and a negligible emission (omeg_abs, 1e-8 in Ryota's MAXI J1543 tables).
          Written as an mtable.
    emit  emission only: pion with fcov = 0 (nothing absorbed on the line of sight) and omeg_emit
          (default 1 = full 4 pi shell), minus the input. Stored as emitted photons per bin divided
          by the irradiating luminosity in erg/s over emit_band_keV (default 0.01-100 keV), so that
          the atable norm is F_irr x Omega/4pi, F_irr being the unabsorbed flux in erg/cm^2/s over
          the same band (the convention of the MAXI J1543 fits: Omega/4pi = 1 <-> norm 1.73e-10).
          Written as an atable. Whether Ryota's pionemit tables use the same convention is untested.

Differences with Ryota's scripts:
    - the input is a `file` component (tabulated SED) instead of dbb + comt
    - spectra are read as source-frame luminosities, so the distance (and the small cosmological
      redshift SPEX derives from it) does not enter the ratios or the emission
    - bins with zero input (outside the SED range) are set to 0 in abs (Ryota: NaN -> 0, inf kept)
    - the grid, the physics and the SPEX version are in one setup (setup.json in the run
      directory); a run only resumes if the stored setup is identical
    - each spectrum is written to a temporary file then renamed, so a killed run leaves no
      truncated file that would be taken as done
    - one task per grid point on a process pool (one SPEX session per worker, kept for all its
      points) instead of one static chunk per core
    - the table is assembled in float32 (the FITS format) instead of float64

Run directory layout:
    setup.json             full setup, see make_setup
    sed_spex.txt           the SED in the SPEX file-model format (write_spex_sed)
    energies.npy           (2, n_bins) bin edges in keV, [lo, hi]
    input.npy              input photon luminosity on the grid (ph/s/keV)
    input_info.json        irradiating luminosity used for the emission normalisation
    spectra/<kind>_<key><i>_..._<key><k>.npy   float32 spectrum of each grid point
    progress.log

SPEX pion units: nh in 1e24 cm^-2 (1e28 m^-2), hden in 1e14 cm^-3 (1e20 m^-3), v (rms) in km/s,
xil = log10 xi (xi in erg cm/s). Grid values are given in table units (e.g. N_H in 1e22 cm^-2) with
a factor to SPEX units (spex_factor, 0.01 for N_H).
"""

import os
import json
import time
import shutil
import ctypes
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.io import fits

#SPEX luminosity unit (photons/s/keV): file-model values and the model luminosity array
SPEX_FILE_UNIT = 1e44
#SPEX hden unit in cm^-3
SPEX_HDEN_UNIT = 1e14
#thread-count variables of the libraries SPEX uses (libgomp, system libopenblas), see _set_threads
THREAD_ENV_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
#longest file name passed to SPEX that was tested to work (see spex_session)
SPEX_PATH_MAX = 100
KEV_TO_ERG = 1.602176634e-9
OUTPUT_KINDS = ('abs', 'emit')
#XSPEC header fields of the two table kinds (as in Ryota's tables)
TABLE_HEADER = {'abs': ('pion abs', False), 'emit': ('pion emit', True)}


# ======================================================
# Setup
# ======================================================

def grid_param(name, key, spex_par, values, method, initial, delta, spex_factor=1.0):
    """One grid parameter.

    name         NAME in the XSPEC table (12 characters max), e.g. 'Nh/1.0e22'
    key          short label used in the spectrum file names, e.g. 'nh'
    spex_par     pion parameter name in SPEX ('xil', 'nh', 'v', 'hden', ...)
    values       grid values in table units, increasing
    method       XSPEC interpolation: 0 linear, 1 logarithmic
    initial, delta   INITIAL and DELTA of the XSPEC table
    spex_factor  SPEX value = table value x spex_factor
    """
    values = [float(v) for v in values]
    if np.any(np.diff(values) <= 0):
        raise ValueError('grid values of %s are not strictly increasing' % name)
    if len(name) > 12:
        raise ValueError('table parameter name %s is longer than 12 characters' % name)
    return {'name': name, 'key': key, 'spex_par': spex_par, 'values': values, 'method': int(method),
            'initial': float(initial), 'delta': float(delta), 'spex_factor': float(spex_factor)}


def spex_version():
    """SPEX version of the installation in $SPEX90 (read from its directory name, e.g. SPEX-3.08.3)."""
    return os.path.basename(os.path.normpath(os.environ['SPEX90'])).replace('SPEX-', '')


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_setup(grid, sed_file, hden=0.1, omeg_abs=1e-8, fcov_abs=1.0, omeg_emit=1.0, dist_kpc=8.0,
               egrid=(1e-3, 1e3, 65536, True), outputs=OUTPUT_KINDS, emit_band_keV=(0.01, 100.),
               abundance=None, element_abundances=None, var_calc='new'):
    """Full description of a pion grid computation (stored as setup.json in the run directory).

    grid           list of grid_param dicts, the first one varying slowest (XSPEC row order)
    sed_file       SED with two columns: energy (keV, bin centres, increasing) and photon
                   luminosity (ph/s/keV, > 0). Only its shape matters for abs and for the emission
                   per unit irradiating luminosity.
    hden           pion hydrogen density in SPEX units of 1e14 cm^-3 (default 0.1 = 1e13 cm^-3)
    omeg_abs, fcov_abs   pion omeg and fcov of the transmission run
    omeg_emit      pion omeg of the emission run (fcov = 0)
    dist_kpc       source distance (does not enter the outputs, which are source-frame)
    egrid          (emin keV, emax keV, number of bins, logarithmic)
    outputs        subset of ('abs', 'emit')
    emit_band_keV  band of the irradiating luminosity that normalises the emission
    abundance      SPEX abundance set (pyspex Session.abun), None = SPEX default (Lodders et al. 2009)
    element_abundances  {atomic number: abundance relative to the set} for the pion component,
                   None = pion defaults (1, but 0 for Li, Be, B, F, P, Cl, K, Sc, Ti, V, Cr, Mn, Co,
                   Cu, Zn). Ryota's MAXI J1543 tables include Cr, Mn, Co, Ti, V, P, Cl lines.
    var_calc       argument of the SPEX 'var calc' command (Ryota: 'new'), None to skip
    """
    if not set(outputs) <= set(OUTPUT_KINDS):
        raise ValueError('outputs must be in %s' % (OUTPUT_KINDS,))
    sed_file = os.path.abspath(sed_file)
    return {'grid': grid,
            'sed_file': sed_file, 'sed_sha256': file_sha256(sed_file),
            'hden': float(hden), 'omeg_abs': float(omeg_abs), 'fcov_abs': float(fcov_abs),
            'omeg_emit': float(omeg_emit), 'dist_kpc': float(dist_kpc),
            'egrid': {'emin': float(egrid[0]), 'emax': float(egrid[1]), 'nbins': int(egrid[2]),
                      'log': bool(egrid[3])},
            'outputs': list(outputs), 'emit_band_keV': [float(x) for x in emit_band_keV],
            'abundance': abundance,
            'element_abundances': None if element_abundances is None
            else {str(int(z)): float(a) for z, a in sorted(element_abundances.items())},
            'var_calc': var_calc,
            'spex_version': spex_version(), 'spex_root': os.environ['SPEX90']}


def read_setup(run_dir):
    with open(os.path.join(run_dir, 'setup.json')) as f:
        return json.load(f)


def _comparable(setup):
    #setup as it is written in json (tuples -> lists), without the creation date
    out = json.loads(json.dumps(setup))
    out.pop('created', None)
    return out


def grid_shape(setup):
    return tuple(len(p['values']) for p in setup['grid'])


def grid_values(setup, idx):
    """Table-unit parameter values of the grid point of indices idx."""
    return [p['values'][i] for p, i in zip(setup['grid'], idx)]


def point_file(run_dir, setup, kind, idx):
    name = '_'.join('%s%02d' % (p['key'], i) for p, i in zip(setup['grid'], idx))
    return os.path.join(run_dir, 'spectra', '%s_%s.npy' % (kind, name))


def point_done(run_dir, setup, idx):
    return all(os.path.exists(point_file(run_dir, setup, kind, idx)) for kind in setup['outputs'])


def write_spex_sed(sed_file, out_path):
    """Write a two-column SED (keV, ph/s/keV) in the SPEX file-model format: number of points on the
    first line, then energy (keV) and spectrum in 1e44 ph/s/keV. SPEX interpolates it linearly in
    log E - log S and sets it to 0 outside the tabulated range."""
    energy, spec = np.loadtxt(sed_file, unpack=True)
    if np.any(spec <= 0) or np.any(np.diff(energy) <= 0):
        raise ValueError('%s: the SPEX file model needs strictly positive values on an increasing '
                         'energy grid' % sed_file)
    with open(out_path, 'w') as f:
        f.write('%d\n' % len(energy))
        np.savetxt(f, np.c_[energy, spec / SPEX_FILE_UNIT], fmt='%.10e')


def prepare_run_dir(setup, run_dir):
    """Create the run directory for setup, or check that the one already there was made with the
    same setup (otherwise the existing spectra belong to another computation: raise)."""
    path = os.path.join(run_dir, 'setup.json')
    if os.path.exists(path):
        stored = read_setup(run_dir)
        if _comparable(stored) != _comparable(setup):
            stored_c, new_c = _comparable(stored), _comparable(setup)
            diff = sorted(k for k in set(stored_c) | set(new_c) if stored_c.get(k) != new_c.get(k))
            raise ValueError('%s holds a run with another setup (differences in %s): use a new '
                             'run directory' % (run_dir, ', '.join(diff)))
        return
    os.makedirs(os.path.join(run_dir, 'spectra'), exist_ok=True)
    write_spex_sed(setup['sed_file'], os.path.join(run_dir, 'sed_spex.txt'))
    _save_json(dict(setup, created=time.strftime('%Y-%m-%d %H:%M:%S')), path)


def _save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _save_npy(arr, path):
    #temporary file then rename: an interrupted write never leaves a file under the final name
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _log(run_dir, text):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + text
    print(line, flush=True)
    with open(os.path.join(run_dir, 'progress.log'), 'a') as f:
        f.write(line + '\n')


# ======================================================
# SPEX computation
# ======================================================

def spex_session(setup, sed_spex):
    """Start a pyspex session with the input SED (component 1, `file`) feeding a pion component
    (component 2), on the energy grid of the setup. Only one session can exist per process.

    Adapted from R. Tomaru's make_pion_mtable.py (process_chunk), with a tabulated SED instead of
    the dbb + comt continuum.

    sed_spex must be at most SPEX_PATH_MAX characters long: longer file names stop SPEX with a
    Fortran "End of record" error (api_model_parameter.f90, SPEX 3.08.03: 100 characters work,
    120 do not). Use a path relative to the working directory if needed.
    """
    if len(sed_spex) > SPEX_PATH_MAX:
        raise ValueError('SPEX file names are limited to %d characters, %s has %d'
                         % (SPEX_PATH_MAX, sed_spex, len(sed_spex)))
    from pyspex import spex
    s = spex.Session()
    if setup['var_calc']:
        s.command('var calc %s' % setup['var_calc'])
    if setup['abundance']:
        s.abun(setup['abundance'])
    s.com('file', isect=1)
    s.com('pion', isect=1)
    s.com_rel(1, 1, np.array([2]))
    s.par_text(1, 1, 'file', sed_spex)
    s.dist(1, setup['dist_kpc'], 'kpc')
    eg = setup['egrid']
    s.egrid(eg['emin'], eg['emax'], eg['nbins'], 'kev', eg['log'])
    s.par(1, 2, 'hden', setup['hden'])
    for z, a in (setup.get('element_abundances') or {}).items():
        s.par(1, 2, '%02d' % int(z), a)
    s.par(1, 2, 'zv', 0.)
    return s


def model_luminosity(isect=1):
    """Bin edges (keV) and source-frame photon luminosity (ph/s/keV) of the last computed model.

    Reads the same pyspex arrays as pyspex.model.Spectrum.get, without its per-bin Python loop
    (7 s for 65536 bins). The raw luminosity array is in 1e44 ph/s/keV (Spectrum.get puts the 1e44
    in the astropy unit, so its .value is in ph/s/keV, as returned here).
    """
    from pyspex import pyspex_f2py
    api = pyspex_f2py.api_model_spectrum
    api.api_model_spectrum_total(isect)
    n = int(api.nbins)
    hi = np.array(api.energy_bupp[:n], dtype=float)
    width = np.array(api.energy_width[:n], dtype=float)
    lum = np.array(api.luminosity[:n], dtype=float) * SPEX_FILE_UNIT
    api.api_model_spectrum_final()
    return hi - width, hi, lum


def input_spectrum(s):
    """Input luminosity on the grid: pion made transparent (fcov = 0) and without emission."""
    s.par(1, 2, 'fcov', 0.)
    s.par(1, 2, 'omeg', 0.)
    #Fortran calculation only: s.calc() also copies the model back to Python, which is not used here
    #(SPEX help, "Optimize for speed")
    s.command('calc')
    return model_luminosity()


def irradiating_luminosity(lo, hi, lum, band_keV):
    """Luminosity in erg/s of the bins whose centre is in band_keV."""
    centre = np.sqrt(lo * hi)
    m = (centre >= band_keV[0]) & (centre <= band_keV[1])
    return float(np.sum(lum[m] * (hi - lo)[m] * centre[m]) * KEV_TO_ERG)


def pion_point(s, setup, values, lo, hi, lum_in, l_irr):
    """abs and/or emit spectra (see the module docstring) at one point of table-unit values.

    Adapted from R. Tomaru's make_pion_mtable.py (process_chunk loop): one calculation per output.
    """
    for p, v in zip(setup['grid'], values):
        s.par(1, 2, p['spex_par'], v * p['spex_factor'])
    out = {}
    if 'abs' in setup['outputs']:
        s.par(1, 2, 'fcov', setup['fcov_abs'])
        s.par(1, 2, 'omeg', setup['omeg_abs'])
        s.command('calc')
        lum = model_luminosity()[2]
        out['abs'] = np.divide(lum, lum_in, out=np.zeros_like(lum), where=lum_in > 0)
    if 'emit' in setup['outputs']:
        s.par(1, 2, 'fcov', 0.)
        s.par(1, 2, 'omeg', setup['omeg_emit'])
        s.command('calc')
        lum = model_luminosity()[2]
        #with fcov = 0 the output is input + emission; negative values are rounding noise
        out['emit'] = np.maximum(lum - lum_in, 0.) * (hi - lo) / l_irr
    return out


#state of a pool worker: its SPEX session and the input spectrum
_WORKER = {}


def _set_threads(n):
    #SPEX uses OpenMP and the system OpenBLAS, which ignores OMP_NUM_THREADS: unset, OpenBLAS starts
    #one thread per core in each worker (24 here), and 6 workers ran about 100x slower (2026-10-28)
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)


def _worker_init(setup, run_dir, omp_threads):
    #pyspex is imported here, after the thread variables are set, and never in the parent process
    _set_threads(omp_threads)
    #SPEX writes scratch files (*.dum) in the working directory
    scratch = os.path.join(run_dir, 'scratch', str(os.getpid()))
    os.makedirs(scratch, exist_ok=True)
    os.chdir(scratch)
    #relative path: the absolute one can exceed the SPEX file-name length limit
    s = spex_session(setup, os.path.join('..', '..', 'sed_spex.txt'))
    #the pyspex found on the path must belong to the SPEX installation of the setup ($SPEX90 when the
    #setup was made), otherwise two versions are mixed
    import pyspex
    if not os.path.realpath(pyspex.__file__).startswith(os.path.realpath(setup['spex_root']) + os.sep):
        raise RuntimeError('pyspex imported from %s, but the setup is for %s: set PYTHONPATH and '
                           'LD_LIBRARY_PATH to the same SPEX' % (pyspex.__file__, setup['spex_root']))
    #also set at run time, in case the parent had already loaded OpenBLAS before the fork (the
    #environment variable is only read when the library starts)
    try:
        ctypes.CDLL('libopenblas.so.0').openblas_set_num_threads(int(omp_threads))
    except (OSError, AttributeError):
        pass
    lo, hi, lum_in = input_spectrum(s)
    #one-off check of the fast reader against pyspex
    sp = s.mod_spectrum
    sp.get(1)
    if not (np.allclose(sp.energy_upper.value, hi, rtol=1e-12)
            and np.allclose(sp.luminosity.value, lum_in, rtol=1e-12)):
        raise RuntimeError('model_luminosity disagrees with pyspex mod_spectrum.get')
    l_irr = irradiating_luminosity(lo, hi, lum_in, setup['emit_band_keV'])
    path_e = os.path.join(run_dir, 'energies.npy')
    if os.path.exists(path_e):
        lo0, hi0 = np.load(path_e)
        lum0 = np.load(os.path.join(run_dir, 'input.npy'))
        if not (np.array_equal(lo0, lo) and np.array_equal(hi0, hi) and np.allclose(lum0, lum_in, rtol=1e-10)):
            raise RuntimeError('input spectrum differs from the one stored in %s' % run_dir)
    else:
        _save_npy(np.array([lo, hi]), path_e)
        _save_npy(lum_in, os.path.join(run_dir, 'input.npy'))
        _save_json({'L_irr_erg_s': l_irr, 'band_keV': setup['emit_band_keV'],
                    'L_total_erg_s': irradiating_luminosity(lo, hi, lum_in, (0., np.inf))},
                   os.path.join(run_dir, 'input_info.json'))
    _WORKER.update(s=s, setup=setup, run_dir=run_dir, lo=lo, hi=hi, lum_in=lum_in, l_irr=l_irr)


def _worker_point(idx):
    w = _WORKER
    t0 = time.time()
    out = pion_point(w['s'], w['setup'], grid_values(w['setup'], idx), w['lo'], w['hi'], w['lum_in'],
                     w['l_irr'])
    for kind, spec in out.items():
        _save_npy(spec.astype(np.float32), point_file(w['run_dir'], w['setup'], kind, idx))
    return idx, time.time() - t0


def compute_pion_grid(setup, run_dir, points=None, n_workers=20, omp_threads=1):
    """Compute the abs/emit spectra of a pion grid into run_dir, skipping the points already done.

    setup        from make_setup
    run_dir      run directory, created if needed; if it exists, its setup.json must be identical
    points       list of index tuples (one index per grid parameter), None = the whole grid
    n_workers    number of processes, each with its own SPEX session (1-4 GB of RAM each)
    omp_threads  OpenMP and OpenBLAS threads per SPEX session (SPEX help: 4 is usually the fastest
                 for folding and matrix operations; n_workers x omp_threads should not exceed the
                 physical cores). One pion point: 19 s with 1 thread, 22 s with 24 OpenBLAS threads.

    Adapted from R. Tomaru's make_pion_mtable.py (main): the static chunks become one task per point
    on a pool whose workers keep their session. The SPEX multiprocessing guide restarts a process per
    task (maxtasksperchild=1) only to guarantee one session per process; here the session is made
    once per process by the pool initializer, which avoids restarting SPEX at every point. The pool
    uses the 'fork' start method so that the calling script needs no __main__ guard; the parent
    never imports pyspex. The workers' scratch directories (SPEX *.dum files) are removed at the end.
    """
    prepare_run_dir(setup, run_dir)
    shape = grid_shape(setup)
    points = list(np.ndindex(*shape)) if points is None else points
    points = [tuple(int(i) for i in p) for p in points]
    for p in points:
        if len(p) != len(shape) or any(not 0 <= i < n for i, n in zip(p, shape)):
            raise ValueError('point %s outside the grid of shape %s' % (p, shape))
    todo = [p for p in dict.fromkeys(points) if not point_done(run_dir, setup, p)]
    _log(run_dir, '%d points requested, %d to compute' % (len(points), len(todo)))
    if not todo:
        return
    _set_threads(omp_threads)
    n_workers = min(n_workers, len(todo))
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('fork'),
                             initializer=_worker_init, initargs=(setup, run_dir, omp_threads))
    try:
        futures = [ex.submit(_worker_point, p) for p in todo]
        for n_done, f in enumerate(as_completed(futures), 1):
            idx, dt = f.result()
            elapsed = time.time() - t0
            _log(run_dir, '%d/%d %s %s: %.1f s (elapsed %.0f s, remaining about %.0f s)'
                 % (n_done, len(todo), idx, grid_values(setup, idx), dt, elapsed,
                    elapsed / n_done * (len(todo) - n_done)))
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    shutil.rmtree(os.path.join(run_dir, 'scratch'), ignore_errors=True)


# ======================================================
# XSPEC tables
# ======================================================

def grid_to_xspec_table(run_dir, kind, out_path=None, overwrite=False):
    """Write the computed grid of run_dir as an XSPEC table model: mtable for kind 'abs', atable
    for 'emit'. Same conventions as Ryota's tables (parameter names, interpolation methods, units,
    REDSHIFT = T), taken from the grid of setup.json; the setup is recorded in HISTORY cards.
    Default output: <run_dir>/<run name>_<kind>.fits.

    Adapted from R. Tomaru's piontxt2_xspec_mtable_new.py (writeImage), assembled in float32.
    """
    setup = read_setup(run_dir)
    if kind not in setup['outputs']:
        raise ValueError('%s was not computed in %s' % (kind, run_dir))
    if out_path is None:
        out_path = os.path.join(run_dir, '%s_%s.fits' % (os.path.basename(os.path.normpath(run_dir)), kind))
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(out_path)
    grid = setup['grid']
    shape = grid_shape(setup)
    points = list(np.ndindex(*shape))
    missing = [p for p in points if not os.path.exists(point_file(run_dir, setup, kind, p))]
    if missing:
        raise RuntimeError('%d of %d %s spectra missing in %s (first: %s)'
                           % (len(missing), len(points), kind, run_dir, missing[0]))
    lo, hi = np.load(os.path.join(run_dir, 'energies.npy'))
    spectra = np.empty((len(points), len(lo)), dtype=np.float32)
    for row, p in enumerate(points):
        spectra[row] = np.load(point_file(run_dir, setup, kind, p))
    parmval = np.array([grid_values(setup, p) for p in points], dtype=np.float32)

    base = [('HDUCLASS', 'OGIP'), ('HDUCLAS1', 'XSPEC TABLE MODEL'), ('HDUVERS', '1.0.0')]
    model_name, additive = TABLE_HEADER[kind]
    primary = fits.PrimaryHDU(np.zeros((2, 2), dtype=int), fits.Header(
        base + [('MODLNAME', model_name), ('MDLUNIT', 'ph cm-2 s-1'), ('REDSHIFT', True),
                ('ADDMODEL', additive)]))
    for line in _history_lines(setup, kind):
        primary.header.add_history(line)

    n_max = max(shape)
    value = np.zeros((len(grid), n_max), dtype=np.float32)
    for i, p in enumerate(grid):
        value[i, :len(p['values'])] = p['values']
    par_cols = [fits.Column(name='NAME', format='12A', array=[p['name'] for p in grid]),
                fits.Column(name='METHOD', format='J', array=[p['method'] for p in grid]),
                fits.Column(name='INITIAL', format='E', array=[p['initial'] for p in grid]),
                fits.Column(name='DELTA', format='E', array=[p['delta'] for p in grid]),
                fits.Column(name='MINIMUM', format='E', array=[p['values'][0] for p in grid]),
                fits.Column(name='BOTTOM', format='E', array=[p['values'][0] for p in grid]),
                fits.Column(name='MAXIMUM', format='E', array=[p['values'][-1] for p in grid]),
                fits.Column(name='TOP', format='E', array=[p['values'][-1] for p in grid]),
                fits.Column(name='NUMBVALS', format='J', array=list(shape)),
                fits.Column(name='VALUE', format='%dE' % n_max, array=value)]
    par_hdu = fits.BinTableHDU.from_columns(par_cols, header=fits.Header(
        base + [('NINTPARM', len(grid)), ('NADDPARM', 0), ('HDUCLAS2', 'PARAMETERS')]))
    par_hdu.header['EXTNAME'] = 'PARAMETERS'
    e_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='ENERG_LO', format='E', array=lo), fits.Column(name='ENERG_HI', format='E', array=hi)],
        header=fits.Header(base + [('HDUCLAS2', 'ENERGIES')]))
    e_hdu.header['EXTNAME'] = 'ENERGIES'
    sp_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='PARMVAL', format='%dE' % len(grid), array=parmval),
         fits.Column(name='INTPSPEC', format='%dE' % len(lo), array=spectra)],
        header=fits.Header(base + [('HDUCLAS2', 'MODEL SPECTRA')]))
    sp_hdu.header['EXTNAME'] = 'SPECTRA'
    fits.HDUList([primary, par_hdu, e_hdu, sp_hdu]).writeto(out_path, overwrite=overwrite)
    return out_path


def _history_lines(setup, kind):
    lines = ['pion %s table, SPEX %s, written by pion_tools.py' % (kind, setup['spex_version']),
             'SED %s' % setup['sed_file'], 'SED sha256 %s' % setup['sed_sha256'],
             'hden %.4g (n_H %.4g cm-3), dist %.4g kpc' % (setup['hden'], setup['hden'] * SPEX_HDEN_UNIT,
                                                         setup['dist_kpc']),
             'abundances %s, var calc %s' % (setup['abundance'] or 'SPEX default', setup['var_calc']),
             'pion element abundances %s' % (','.join('%s:%g' % kv for kv in setup['element_abundances'].items())
                                              if setup.get('element_abundances') else 'pion default')]
    if kind == 'abs':
        lines.append('output/input, omeg %.4g, fcov %.4g' % (setup['omeg_abs'], setup['fcov_abs']))
    else:
        lines.append('emission (fcov 0, omeg %.4g) in photons per bin per erg/s of input over '
                     '%g-%g keV' % (setup['omeg_emit'], *setup['emit_band_keV']))
    return lines


def read_table_params(path):
    """Parameters of an XSPEC table: list of dicts (name, method, initial, delta, values) and the
    ADDMODEL and REDSHIFT flags. Only the small extensions are read."""
    with fits.open(path, memmap=True) as h:
        params = [{'name': row['NAME'].strip(), 'method': int(row['METHOD']),
                   'initial': float(row['INITIAL']), 'delta': float(row['DELTA']),
                   'values': np.array(row['VALUE'][:row['NUMBVALS']], dtype=float)}
                  for row in h['PARAMETERS'].data]
        return {'params': params, 'addmodel': bool(h[0].header['ADDMODEL']),
                'redshift': bool(h[0].header.get('REDSHIFT', False))}


def read_table_energies(path):
    """(lo, hi) bin edges in keV of an XSPEC table."""
    with fits.open(path, memmap=True) as h:
        e = h['ENERGIES'].data
        return np.array(e['ENERG_LO'], dtype=float), np.array(e['ENERG_HI'], dtype=float)


def table_node_index(params, values, rtol=1e-5):
    """Grid indices of the node at values (one per parameter); raises if values is not a node."""
    idx = []
    for p, v in zip(params, values):
        hit = np.flatnonzero(np.isclose(p['values'], v, rtol=rtol, atol=0))
        if len(hit) != 1:
            raise ValueError('%s = %g is not a node of the table grid' % (p['name'], v))
        idx.append(int(hit[0]))
    return tuple(idx)


def table_spectrum(path, values):
    """Spectrum of an XSPEC table at a grid node (only that row is read from disk)."""
    params = read_table_params(path)['params']
    idx = table_node_index(params, values)
    row = int(np.ravel_multi_index(idx, [len(p['values']) for p in params]))
    with fits.open(path, memmap=True) as h:
        rec = h['SPECTRA'].data[row]
        if not np.allclose(rec['PARMVAL'], [p['values'][i] for p, i in zip(params, idx)], rtol=1e-5):
            raise RuntimeError('row %d of %s is not the node %s' % (row, path, values))
        return np.array(rec['INTPSPEC'], dtype=float)


def check_grid_against_table(setup, path, rtol=1e-6):
    """Raise if the grid of setup (names, methods, values) is not the one of the XSPEC table."""
    params = read_table_params(path)['params']
    if len(params) != len(setup['grid']):
        raise ValueError('%d table parameters, %d in the setup' % (len(params), len(setup['grid'])))
    for p, q in zip(setup['grid'], params):
        if p['name'] != q['name'] or p['method'] != q['method'] or len(p['values']) != len(q['values']) \
                or not np.allclose(p['values'], q['values'], rtol=rtol, atol=0):
            raise ValueError('grid parameter %s differs from table parameter %s' % (p['name'], q['name']))


# ======================================================
# Interpolation and folding
# ======================================================

def multilinear_weights(grid_values_list, methods, values):
    """Nodes and weights of the XSPEC table-model interpolation at values: for each parameter the two
    bracketing nodes, linear in the value (METHOD 0) or in its log10 (METHOD 1), combined as a
    multilinear (tensor-product) interpolation. Returns a list of (index tuple, weight), weights > 0.
    Raises if a value is outside its grid."""
    import itertools
    per_param = []
    for vals, method, v in zip(grid_values_list, methods, values):
        x = np.log10(vals) if method == 1 else np.asarray(vals, dtype=float)
        xv = np.log10(v) if method == 1 else float(v)
        if not x[0] <= xv <= x[-1]:
            raise ValueError('value %g outside the grid [%g, %g]' % (v, vals[0], vals[-1]))
        i = min(int(np.searchsorted(x, xv, side='right')) - 1, len(x) - 2)
        f = (xv - x[i]) / (x[i + 1] - x[i])
        #on a node (up to rounding of the log): that node only
        f = 0. if abs(f) < 1e-9 else 1. if abs(1. - f) < 1e-9 else f
        per_param.append([(i, 1. - f), (i + 1, f)])
    out = []
    for combo in itertools.product(*per_param):
        w = float(np.prod([c[1] for c in combo]))
        if w > 0:
            out.append((tuple(c[0] for c in combo), w))
    return out


def table_interp(path, values):
    """Spectrum of an XSPEC table interpolated at values as XSPEC does (multilinear_weights with the
    table's METHOD column); only the needed rows are read."""
    params = read_table_params(path)['params']
    shape = [len(p['values']) for p in params]
    weights = multilinear_weights([p['values'] for p in params], [p['method'] for p in params], values)
    with fits.open(path, memmap=True) as h:
        data = h['SPECTRA'].data
        return sum(w * np.array(data[int(np.ravel_multi_index(idx, shape))]['INTPSPEC'], dtype=float)
                   for idx, w in weights)


def grid_interp(run_dir, kind, values):
    """Same interpolation as table_interp on the spectra computed in run_dir (all the bracketing
    nodes must have been computed). Returns the spectrum and the list of nodes used."""
    setup = read_setup(run_dir)
    weights = multilinear_weights([p['values'] for p in setup['grid']], [p['method'] for p in setup['grid']],
                                  values)
    missing = [idx for idx, _ in weights if not os.path.exists(point_file(run_dir, setup, kind, idx))]
    if missing:
        raise RuntimeError('nodes %s not computed in %s' % (missing, run_dir))
    spec = sum(w * np.load(point_file(run_dir, setup, kind, idx)).astype(float) for idx, w in weights)
    return spec, [idx for idx, _ in weights]


def read_response(rmf_path, arf_path, e_range):
    """RMF x ARF as a sparse (energy bin, channel) matrix in cm^2, for the energy bins inside e_range
    (keV). Only the first MATRIX extension is used, as XSPEC does (the Resolve 'comb' RMFs have a
    second one). Channels are numbered from 0 (row index = channel - TLMIN of F_CHAN).

    Returns a dict: e_lo, e_hi (keV) of the kept energy bins, matrix (scipy.sparse.csr_matrix),
    ch_lo, ch_hi (keV, EBOUNDS of all channels)."""
    from scipy import sparse
    with fits.open(rmf_path, memmap=True) as h:
        mat = h['MATRIX', 1]
        tlmin = int(mat.header.get('TLMIN%d' % (mat.columns.names.index('F_CHAN') + 1), 0))
        e_lo_all = np.array(mat.data['ENERG_LO'], dtype=float)
        e_hi_all = np.array(mat.data['ENERG_HI'], dtype=float)
        rows = np.flatnonzero((e_lo_all >= e_range[0]) & (e_hi_all <= e_range[1]))
        eb = h['EBOUNDS', 1].data
        ch_lo, ch_hi = np.array(eb['E_MIN'], dtype=float), np.array(eb['E_MAX'], dtype=float)
        r_idx, c_idx, vals = [], [], []
        for k, row in enumerate(rows):
            rec = mat.data[row]
            m = np.asarray(rec['MATRIX'], dtype=float)
            pos = 0
            for f, n in zip(np.atleast_1d(rec['F_CHAN']), np.atleast_1d(rec['N_CHAN'])):
                r_idx.append(np.full(n, k))
                c_idx.append(np.arange(f - tlmin, f - tlmin + n))
                vals.append(m[pos:pos + n])
                pos += n
    with fits.open(arf_path) as h:
        arf = h['SPECRESP'].data
        if not np.allclose(np.array(arf['ENERG_LO'], dtype=float), e_lo_all, rtol=0, atol=1e-6):
            raise ValueError('ARF and RMF energy grids differ')
        area = np.array(arf['SPECRESP'], dtype=float)[rows]
    matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
                               shape=(len(rows), len(ch_lo)))
    matrix = sparse.diags(area) @ matrix
    return {'e_lo': e_lo_all[rows], 'e_hi': e_hi_all[rows], 'matrix': matrix.tocsr(), 'ch_lo': ch_lo, 'ch_hi': ch_hi}


def rebin_counts(lo, hi, per_bin, new_lo, new_hi):
    """Rebin a per-bin quantity (e.g. photons per bin) from the contiguous bins (lo, hi) onto
    (new_lo, new_hi) by interpolating its cumulative sum (uniform density inside each old bin)."""
    edges = np.append(lo, hi[-1])
    cumul = np.concatenate([[0.], np.cumsum(per_bin)])
    return np.interp(new_hi, edges, cumul) - np.interp(new_lo, edges, cumul)


def fold_spectrum(response, lo, hi, photons_per_bin):
    """Count rate per channel (counts/s, for photons_per_bin in photons/cm^2/s per bin on the grid
    lo, hi) through a response from read_response."""
    flux = rebin_counts(lo, hi, photons_per_bin, response['e_lo'], response['e_hi'])
    return response['matrix'].T @ flux


# ======================================================
# Comparison
# ======================================================

def compare_spectra(lo, hi, new, ref, bands, floor_rel=1e-4):
    """Statistics of new / ref in energy bands (dict name -> (emin, emax) in keV, bin centres).

    Bins where ref is below floor_rel x the band maximum (saturated line cores, empty emission)
    are left out and counted in n_masked. Per band: number of bins used, median, 1st and 99th
    percentiles of the ratio, max |ratio - 1| and the energy where it occurs.
    """
    centre = np.sqrt(lo * hi)
    stats = {}
    for name, (emin, emax) in bands.items():
        m = (centre >= emin) & (centre < emax)
        ok = m & (ref > floor_rel * ref[m].max()) if m.any() and ref[m].max() > 0 else np.zeros_like(m)
        if not ok.any():
            stats[name] = {'n': 0, 'n_masked': int(m.sum())}
            continue
        r = new[ok] / ref[ok]
        dev = np.abs(r - 1)
        stats[name] = {'n': int(ok.sum()), 'n_masked': int(m.sum() - ok.sum()),
                       'median': float(np.median(r)), 'p01': float(np.percentile(r, 1)),
                       'p99': float(np.percentile(r, 99)), 'max_dev': float(dev.max()),
                       'e_max_dev': float(centre[ok][np.argmax(dev)])}
    return stats


def plot_comparison(lo, hi, new, ref, bands, title, path, floor_rel=1e-4, log_y=False):
    """One column per band: new and ref spectra (top) and new / ref (bottom, masked like
    compare_spectra). Saved to path.

    Drawn on a matplotlib Figure without pyplot, so no GUI backend is started (the Qt backend of
    an interactive console can abort the process when the first window is created)."""
    from matplotlib.figure import Figure
    centre = np.sqrt(lo * hi)
    fig = Figure(figsize=(5 * len(bands), 6))
    axes = fig.subplots(2, len(bands), sharex='col', squeeze=False, gridspec_kw={'height_ratios': [2, 1]})
    for col, (name, (emin, emax)) in enumerate(bands.items()):
        m = (centre >= emin) & (centre < emax)
        ax, axr = axes[0, col], axes[1, col]
        ax.step(centre[m], ref[m], where='mid', color='black', lw=0.8, label='reference table')
        ax.step(centre[m], new[m], where='mid', color='red', lw=0.8, ls='--', label='new')
        ok = m & (ref > floor_rel * ref[m].max())
        ratio = np.full(m.sum(), np.nan)
        ratio[ok[m]] = new[ok] / ref[ok]
        axr.step(centre[m], ratio, where='mid', color='red', lw=0.8)
        axr.axhline(1, color='grey', lw=0.6)
        ax.set_title(name)
        axr.set_xlabel('E (keV)')
        if log_y:
            ax.set_yscale('log')
        if emax / emin > 5:
            ax.set_xscale('log')
    axes[0, 0].legend()
    axes[1, 0].set_ylabel('new / reference')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)

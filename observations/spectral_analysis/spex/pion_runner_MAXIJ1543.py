"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-28.

Reproduction of R. Tomaru's MAXI J1543-564 pion tables (photo/pion/pionabsMAXIJ1543.fits and
pionemitMAXIJ1543.fits, May 2026) with pion_tools, before using the same tools for the off-node
check and the local fine zone-1 grid (later sections). The existing tables are only read.

Setup inferred from the tables, the SED and Ryota's emails (no setup is recorded in the tables):
    input       SED_cont_MAXIJ1543_scaled_8Msun.txt (0.01-997.7 keV, ph/s/keV despite its header,
                8.85e37 erg/s): the abs table is 0 in the last 10 bins, where the SED is not defined
    grid        rlogxi 1-6 (26, linear), Nh/1.0e22 0.01-1000 (26, log), v_rms 3-3000 km/s (16, log),
                65536 log bins over 1e-3-1e3 keV (the egrid of Ryota's make_pion_mtable.py)
    abs         output / input with omeg 1e-8 (Ryota, 2026-07-22), fcov 1
    emit        emission only, fcov 0, omeg 1, in photons per bin per erg/s of input luminosity over
                the whole energy grid: with 0.01-100 keV (the band of the 1.73e-10 norm) the first
                tests were 0.28% above the table, the ratio L(grid) / L(0.01-100 keV) = 1.0027
    density     HDEN_REPRO below: hden 1 (1e14 cm^-3). Ryota wrote "n_p = 1e-14 cm^-3" (2026-07-22),
                which is also the SPEX default hden = 1e-14 (1 cm^-3); the two only differ on
                density-sensitive lines (Fe XXII ~1.05 keV): 'density_check'
    SPEX        3.08.03 (Ryota used 3.08.01 for MAXI J1744; 3.08.03 fixed the file-model
                interpolation)
    abundances  SPEX default (Lodders et al. 2009) with all 30 pion elements at 1: with the pion
                default (P, Cl, Ti, V, Cr, Mn, Co... at 0) the first test (2026-10-28, run
                repro_J1543_hden1e00) missed the Cr, Mn, Co, Ti, V, P, Cl lines of the tables

WHAT selects the run:
    'single_point'   the zone-1 node (log xi 4.8, N_H 10, v_rms 119.43): compute and compare
    'multi_point'    POINTS (14 nodes across the grid): compute and compare
    'compare'        only the comparisons, for the points of POINTS already computed
    'density_check'  the Fe XXII node of POINTS at hden 1e-14 and 1 (two run directories)
    'off_node'       zone-1 off-node check: exact pion at OFF_XI x OFF_V (N_H = NH_ZONE1) and the table
                     nodes around them, then compare_off_node
    'off_node_compare'  only compare_off_node
Each run is in its own subdirectory of RUNS_DIR; the comparison plots and statistics go in its
compare/ subdirectory.

Results (2026-10-28):
    single_point, multi_point (14 nodes): abs agrees to ~1e-5 for log xi >= 4 except just below the
        H/He-like edges (Fe XXV 8.80 keV, Fe XXVI 9.27 keV: up to a few 10% in single bins, likely the
        3.08.03 edge-smoothing fix); at log xi 2-3.4 single lines differ by a few to ~30%. emit agrees
        to ~1% (medians) at high xi, lines within a few %; at low xi single lines differ by factors
        0.3-5 (SPEX version, presumably). Time per point 23-80 s at log xi >= 3.4, 160-370 s below,
        3.05 GB per worker.
"""

import os
import sys
import json

import numpy as np

SPEX_DIR = '/home/parrazyte/Documents/Work/Scripts/Python/winds/observations/spectral_analysis/spex'
sys.path.insert(0, SPEX_DIR)

from astropy.io import fits

from pion_tools import (grid_param, make_setup, compute_pion_grid, read_setup, point_file, point_done,
                        grid_values, grid_shape, check_grid_against_table, read_table_energies, table_spectrum,
                        compare_spectra, plot_comparison, multilinear_weights, table_interp, grid_interp,
                        read_response, fold_spectrum, spex_version)

SOURCE_DIR = '/run/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564'
SED_FILE = os.path.join(SOURCE_DIR, 'common_fit', 'SED_cont_MAXIJ1543_scaled_8Msun.txt')
#Ryota's tables, read only
REF_TABLES = {'abs': os.path.join(SOURCE_DIR, 'photo', 'pion', 'pionabsMAXIJ1543.fits'),
              'emit': os.path.join(SOURCE_DIR, 'photo', 'pion', 'pionemitMAXIJ1543.fits')}
RUNS_DIR = os.path.join(SOURCE_DIR, 'photo', 'pion_runs')

WHAT = 'single_point'
#processes for the multi-point runs: 3.05 GB per SPEX session (measured), 29 GB of RAM (about 21 GB
#free with the usual desktop), 12 physical cores. Threads per process: 4 was not faster than 1 for
#one pion point (2026-10-28)
N_WORKERS = 6
OMP_THREADS = 1

#grid of Ryota's tables (checked against both tables below)
GRID = [grid_param('rlogxi', 'xi', 'xil', np.linspace(1., 6., 26), method=0, initial=4.0, delta=0.1),
        grid_param('Nh/1.0e22', 'nh', 'nh', np.logspace(-2., 3., 26), method=1, initial=10.0, delta=0.1,
                   spex_factor=0.01),
        grid_param('v_rms [km/s]', 'v', 'v', 3. * np.logspace(0., 3., 16), method=1, initial=200.0,
                   delta=10.0)]
#pion hden of the reproduction run, in units of 1e14 cm^-3 (see the docstring)
HDEN_REPRO = 1.0
#pion abundances of all elements H-Zn (relative to Lodders et al. 2009)
ELEMENT_ABUNDANCES = {z: 1.0 for z in range(1, 31)}


def repro_setup(hden, grid=GRID):
    return make_setup(grid, SED_FILE, hden=hden, omeg_abs=1e-8, fcov_abs=1.0, omeg_emit=1.0, dist_kpc=8.0,
                      egrid=(1e-3, 1e3, 65536, True), outputs=('abs', 'emit'),
                      element_abundances=ELEMENT_ABUNDANCES, emit_band_keV=(1e-3, 1e3))


def run_dir_for(hden):
    #e.g. repro_J1543_allZ_Lgrid_hden1e00 (Lgrid: emission normalised by the whole-grid luminosity), with
    #a _spex<version> suffix for SPEX versions other than 3.08.3 (the SPEX of $SPEX90: to use another
    #version, start Python with SPEX90, PYTHONPATH and LD_LIBRARY_PATH pointing to its installation)
    name = 'repro_J1543_allZ_Lgrid_hden' + ('%.0e' % hden).replace('+', '')
    if spex_version() != '3.08.3':
        name += '_spex' + spex_version()
    return os.path.join(RUNS_DIR, name)


#test nodes (indices in rlogxi, Nh, v_rms), spread over the grid
POINTS = {'zone1': (19, 15, 8),            #log xi 4.8, N_H 10, v 119: the zone-1 node
          'corner_low': (0, 0, 0),         #1.0, 0.01, 3
          'corner_high': (25, 25, 15),     #6.0, 1000, 3000
          'highxi_thin': (25, 0, 0),       #6.0, 0.01, 3
          'lowxi_thick': (0, 25, 15),      #1.0, 1000, 3000
          'lowxi': (5, 10, 3),             #2.0, 1, 12
          'FeXXII': (8, 15, 8),            #2.6, 10, 119: Fe XXII metastable lines ~1.05 keV
          'midxi_thick': (12, 20, 5),      #3.4, 100, 30
          'midxi_thin': (15, 5, 12),       #4.0, 0.1, 754
          'zone_like_1': (17, 18, 10),     #4.4, 39.8, 300
          'highxi_thick': (20, 22, 2),     #5.0, 251, 7.5
          'zone_like_2': (22, 12, 6),      #5.4, 1.58, 47.5
          'highxi_mid': (24, 8, 14),       #5.8, 0.40, 1892
          'lowmid_thick': (10, 25, 0)}     #3.0, 1000, 3

#comparison bands (keV) and the band where each kind is plotted in log y
BANDS = {'0.3-30 keV': (0.3, 30.), 'Fe XXII 1.0-1.1 keV': (1.0, 1.1), 'Fe K 6.4-7.2 keV': (6.4, 7.2),
         '10-100 keV': (10., 100.)}
PLOT_BANDS = {'0.3-30 keV': (0.3, 30.), 'Fe XXII 1.0-1.1 keV': (1.0, 1.1), 'Fe K 6.4-7.2 keV': (6.4, 7.2)}


def compare_points(run_dir, names):
    """Compare the computed spectra of the POINTS names with the reference tables: statistics
    printed and saved in compare/stats.json, one plot per point and kind in compare/."""
    setup = read_setup(run_dir)
    out_dir = os.path.join(run_dir, 'compare')
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = np.load(os.path.join(run_dir, 'energies.npy'))
    for kind in setup['outputs']:
        lo_ref, hi_ref = read_table_energies(REF_TABLES[kind])
        if not (np.allclose(lo, lo_ref, rtol=1e-6) and np.allclose(hi, hi_ref, rtol=1e-6)):
            raise RuntimeError('energy grid differs from the one of %s' % REF_TABLES[kind])
    path_stats = os.path.join(out_dir, 'stats.json')
    all_stats = json.load(open(path_stats)) if os.path.exists(path_stats) else {}
    for name in names:
        idx = POINTS[name]
        if not point_done(run_dir, setup, idx):
            print('%s %s not computed, skipped' % (name, idx))
            continue
        values = grid_values(setup, idx)
        for kind in setup['outputs']:
            new = np.load(point_file(run_dir, setup, kind, idx)).astype(float)
            ref = table_spectrum(REF_TABLES[kind], values)
            stats = compare_spectra(lo, hi, new, ref, BANDS)
            all_stats['%s %s' % (name, kind)] = {'idx': idx, 'values': values, 'bands': stats}
            print('%-13s %-4s logxi %.1f Nh %.3g v %.4g' % (name, kind, *values))
            for band, st in stats.items():
                if st['n'] == 0:
                    print('    %-20s no bin above the floor' % band)
                    continue
                print('    %-20s n %6d (masked %5d)  median %.6f  p01 %.6f  p99 %.6f  max|r-1| %.3g at %.4f keV'
                      % (band, st['n'], st['n_masked'], st['median'], st['p01'], st['p99'], st['max_dev'],
                         st['e_max_dev']))
            plot_comparison(lo, hi, new, ref, PLOT_BANDS,
                            '%s %s: log xi %.1f, N_H %.3g e22, v_rms %.4g km/s' % (name, kind, *values),
                            os.path.join(out_dir, '%s_%s.png' % (kind, name)), log_y=(kind == 'emit'))
    with open(path_stats, 'w') as f:
        json.dump(all_stats, f, indent=1)


# ======================================================
# Off-node check of zone 1
# ======================================================
#zone 1 (pion_abs_3) of the final long chain, uniform priors on Omega and the v_rms
#(posterior_remarks_MAXIJ1543.py, 2026-10-29): median N_H 7.26e22 and Omega/4pi 0.75; two modes at
#(log xi, v_rms) = (4.70, 113 km/s) and (4.90, 145 km/s), split at log xi 4.793 (table node 4.8)
NH_ZONE1 = 7.26
OMEGA_ZONE1 = 0.75
#exact pion points across the cells on both sides of the log xi 4.8 and v_rms 119.4 km/s nodes
OFF_XI = [4.7, 4.75, 4.8, 4.85, 4.9]
OFF_V = [95., 110., GRID[2]['values'][8], 130., 150.]
OFF_GRID = [grid_param('rlogxi', 'xi', 'xil', OFF_XI, method=0, initial=4.8, delta=0.1),
            grid_param('Nh/1.0e22', 'nh', 'nh', [NH_ZONE1], method=1, initial=NH_ZONE1, delta=0.1,
                       spex_factor=0.01),
            grid_param('v_rms [km/s]', 'v', 'v', OFF_V, method=1, initial=119.4, delta=10.0)]
OFF_RUN_DIR = os.path.join(RUNS_DIR, 'offnode_zone1_NH7.26')
#Resolve data and response of the fits (photo/pion copies, the ones the xcm files load)
FIT_DIR = os.path.join(SOURCE_DIR, 'photo', 'pion')
RSL_STEM = 'xa902003010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0and1_rmf_evtbase_2_12_0_05_60000_comb'
RSL_RMF = os.path.join(FIT_DIR, RSL_STEM + '.rmf')
RSL_ARF = os.path.join(FIT_DIR, RSL_STEM + '_POINT.arf')
RSL_PHA = os.path.join(FIT_DIR, 'xa902003010rsl_p0px1000_cl_RTS_pixel_branch_filter_grade_0to1_sp.ds')
#channel bands of the folded comparison: Fe K and the fit band (ignore 1-4000, 20000-60000)
FOLD_BANDS = {'Fe K 6.0-7.2 keV': (6.0, 7.2), 'fit band 2-10 keV': (2.0, 10.0)}


def off_node_table_nodes():
    """Indices (in GRID) of the table nodes around every off-node point."""
    nodes = []
    for xi in OFF_XI:
        for v in OFF_V:
            nodes += [idx for idx, _ in multilinear_weights([p['values'] for p in GRID],
                                                            [p['method'] for p in GRID], [xi, NH_ZONE1, v])]
    return list(dict.fromkeys(nodes))


def compare_off_node():
    """Exact pion (OFF_RUN_DIR) against the table interpolation at the same parameters, unfolded and
    folded through the Resolve response, for zone 1 alone on the input SED: abs only, and abs +
    Omega x emission (OMEGA_ZONE1). Two interpolations: of our own nodes (run_dir_for(HDEN_REPRO), same
    SPEX version: pure interpolation error) and of Ryota's tables (what the fits use).

    Folded comparison: the folded exact model is scaled to the data counts in the Fe K band, and
    each channel difference is divided by the Poisson sigma of the exact model counts. The sum of
    their squares is the chi^2 (about the C-stat) the data would add if the true spectrum were the
    exact one and the model the interpolated one at the same parameters, without refitting (a refit
    can absorb part of it by moving the parameters).

    Prints one line per point, saves compare/off_node_stats.json, a map of the Fe K chi^2 over
    (log xi, v_rms) and, per point, the folded Fe K counts and residuals."""
    from matplotlib.figure import Figure
    setup = read_setup(OFF_RUN_DIR)
    repro_dir = run_dir_for(HDEN_REPRO)
    out_dir = os.path.join(OFF_RUN_DIR, 'compare')
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = np.load(os.path.join(OFF_RUN_DIR, 'energies.npy'))
    if not np.array_equal(np.load(os.path.join(repro_dir, 'energies.npy')), np.array([lo, hi])):
        raise RuntimeError('energy grids of %s and %s differ' % (OFF_RUN_DIR, repro_dir))
    #input continuum in photons per bin per erg/s of input luminosity: the unit of the emission
    with open(os.path.join(OFF_RUN_DIR, 'input_info.json')) as f:
        l_irr = json.load(f)['L_irr_erg_s']
    cont = np.load(os.path.join(OFF_RUN_DIR, 'input.npy')) * (hi - lo) / l_irr

    resp = read_response(RSL_RMF, RSL_ARF, (1.8, 10.5))
    ch_e = 0.5 * (resp['ch_lo'] + resp['ch_hi'])
    with fits.open(RSL_PHA) as h:
        data_counts = np.array(h['SPECTRUM'].data['COUNTS'], dtype=float)
    band_ch = {b: (ch_e >= e0) & (ch_e < e1) for b, (e0, e1) in FOLD_BANDS.items()}
    fek = band_ch['Fe K 6.0-7.2 keV']

    stats = {}
    shape = grid_shape(setup)
    chi2_map = {(lab, il): np.zeros((len(OFF_XI), len(OFF_V))) for lab in ('abs', 'abs+emit')
                for il in ('own nodes', 'Ryota table')}
    print('%-5s %-7s | %-9s | %-26s | %-26s | %s' % ('logxi', 'v_rms', 'model', 'own nodes: chi2 FeK/2-10',
                                                      'Ryota table: chi2 FeK/2-10', 'max|z| FeK own/Ryota'))
    for idx in np.ndindex(*shape):
        values = grid_values(setup, idx)
        exact = {k: np.load(point_file(OFF_RUN_DIR, setup, k, idx)).astype(float) for k in ('abs', 'emit')}
        interp = {'own nodes': {k: grid_interp(repro_dir, k, values)[0] for k in ('abs', 'emit')},
                  'Ryota table': {k: table_interp(REF_TABLES[k], values) for k in ('abs', 'emit')}}
        key = 'logxi %.2f v %.1f' % (values[0], values[2])
        stats[key] = {'values': values}
        for lab, omega in (('abs', 0.), ('abs+emit', OMEGA_ZONE1)):
            f_exact = fold_spectrum(resp, lo, hi, cont * exact['abs'] + omega * exact['emit'])
            scale = data_counts[fek].sum() / f_exact[fek].sum()
            m_exact = f_exact * scale
            sig = np.sqrt(np.maximum(m_exact, 1e-30))
            res = {}
            for il, sp in interp.items():
                m_int = fold_spectrum(resp, lo, hi, cont * sp['abs'] + omega * sp['emit']) * scale
                z = (m_int - m_exact) / sig
                unf = compare_spectra(lo, hi, sp['abs'], exact['abs'], {'Fe K': (6.0, 7.2)})['Fe K']
                res[il] = {'z': z, 'unfolded_abs_max_dev_FeK': unf['max_dev']}
                for b, m in band_ch.items():
                    res[il]['chi2 ' + b] = float(np.sum(z[m] ** 2))
                    res[il]['max|z| ' + b] = float(np.abs(z[m]).max())
                    res[il]['e_max|z| ' + b] = float(ch_e[m][np.argmax(np.abs(z[m]))])
                chi2_map[(lab, il)][idx[0], idx[2]] = res[il]['chi2 Fe K 6.0-7.2 keV']
            stats[key][lab] = {il: {k: v for k, v in r.items() if k != 'z'} for il, r in res.items()}
            print('%-5.2f %-7.1f | %-9s | %10.1f / %-13.1f | %10.1f / %-13.1f | %.2f / %.2f'
                  % (values[0], values[2], lab, res['own nodes']['chi2 Fe K 6.0-7.2 keV'],
                     res['own nodes']['chi2 fit band 2-10 keV'], res['Ryota table']['chi2 Fe K 6.0-7.2 keV'],
                     res['Ryota table']['chi2 fit band 2-10 keV'], res['own nodes']['max|z| Fe K 6.0-7.2 keV'],
                     res['Ryota table']['max|z| Fe K 6.0-7.2 keV']))
            if lab == 'abs+emit':
                fig = Figure(figsize=(12, 6))
                ax, axr = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                ax.step(ch_e[fek], m_exact[fek], where='mid', color='black', lw=0.7, label='exact pion')
                for il, col in (('own nodes', 'red'), ('Ryota table', 'tab:blue')):
                    m_int = m_exact + res[il]['z'] * sig
                    ax.step(ch_e[fek], m_int[fek], where='mid', color=col, lw=0.7, ls='--',
                            label='interpolated, ' + il)
                    axr.step(ch_e[fek], res[il]['z'][fek], where='mid', color=col, lw=0.7)
                axr.axhline(0, color='grey', lw=0.6)
                ax.set_ylabel('counts per 0.5 eV channel')
                axr.set_ylabel('(interp - exact) / sigma')
                axr.set_xlabel('E (keV)')
                ax.legend()
                ax.set_title('zone 1 abs + %.2f emission, log xi %.2f, N_H %.2f e22, v_rms %.1f km/s, folded'
                             % (omega, *values))
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'folded_FeK_logxi%.2f_v%.1f.png' % (values[0], values[2])),
                            dpi=130)

    with open(os.path.join(out_dir, 'off_node_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    fig = Figure(figsize=(11, 9))
    axes = fig.subplots(2, 2)
    for ax, ((lab, il), chi2) in zip(axes.ravel(), chi2_map.items()):
        im = ax.imshow(chi2.T, origin='lower', cmap='viridis')
        for i in range(len(OFF_XI)):
            for j in range(len(OFF_V)):
                ax.text(i, j, '%.1f' % chi2[i, j], ha='center', va='center', color='white', fontsize=8)
        ax.set_xticks(range(len(OFF_XI)), ['%.2f' % x for x in OFF_XI])
        ax.set_yticks(range(len(OFF_V)), ['%.1f' % v for v in OFF_V])
        ax.set_xlabel('log xi (table node at 4.80)')
        ax.set_ylabel('v_rms km/s (table node at 119.4)')
        ax.set_title('%s, interpolation of %s' % (lab, il))
        fig.colorbar(im, ax=ax, label='chi2, Fe K 6.0-7.2 keV')
    fig.suptitle('Zone 1 off-node check: interpolated vs exact pion, folded, N_H %.2f e22' % NH_ZONE1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'off_node_chi2_map.png'), dpi=130)


if __name__ == '__main__':
    setup = repro_setup(HDEN_REPRO)
    run_dir = run_dir_for(HDEN_REPRO)
    for path in REF_TABLES.values():
        check_grid_against_table(setup, path)

    if WHAT == 'single_point':
        compute_pion_grid(setup, run_dir, points=[POINTS['zone1']], n_workers=1, omp_threads=OMP_THREADS)
        compare_points(run_dir, ['zone1'])

    elif WHAT == 'multi_point':
        compute_pion_grid(setup, run_dir, points=list(POINTS.values()), n_workers=N_WORKERS,
                          omp_threads=OMP_THREADS)
        compare_points(run_dir, list(POINTS))

    elif WHAT == 'compare':
        compare_points(run_dir, list(POINTS))

    elif WHAT == 'density_check':
        for hden in (1e-14, 1.):
            compute_pion_grid(repro_setup(hden), run_dir_for(hden), points=[POINTS['FeXXII']], n_workers=1,
                              omp_threads=OMP_THREADS)
            print('===== hden = %g (1e14 cm^-3)' % hden)
            compare_points(run_dir_for(hden), ['FeXXII'])

    elif WHAT in ('off_node', 'off_node_compare'):
        if WHAT == 'off_node':
            #table nodes around the points (for the interpolation of our own spectra), then the exact points
            compute_pion_grid(setup, run_dir, points=off_node_table_nodes(), n_workers=N_WORKERS,
                              omp_threads=OMP_THREADS)
            compute_pion_grid(repro_setup(HDEN_REPRO, OFF_GRID), OFF_RUN_DIR, n_workers=N_WORKERS,
                              omp_threads=OMP_THREADS)
        compare_off_node()

"""
Created by Claude Code, model Opus 5.5 (claude-opus-5-5), 2026-10-25.

Numbers of the "posterior remarks" Notion page (MAXIJ1543-564/photoionization): posterior medians and
90% intervals (5-95% quantiles) of the final long chain, discard 10000, with
    UNIF: uniform priors on Omega and the v_rms of the 3 zones (importance resampling, the main result)
    LOGU: the chain's own log-uniform (jeffreys) priors (sensitivity test)
plus the zone 1 mode split, the imposed limits and the best fit. Run from posterior/ with the astro venv.
Zones: 1 = pion_abs_3, 2 = pion_abs_4, 3 = pion_abs (contour_maker.MAXIJ1543_ZONES).
"""

import numpy as np
from contour_maker import load_chain, MAXIJ1543_CHAIN

C = 299792.458
PARS = [(6, 'log xi1'), (7, 'N_H1'), (8, 'v_rms1'), (9, 'v_out1'), (22, 'Omega'),
        (10, 'log xi2'), (11, 'N_H2'), (12, 'v_rms2'), (13, 'v_out2'),
        (2, 'log xi3'), (3, 'N_H3'), (4, 'v_rms3'), (5, 'v_out3'),
        (1, 'N_H ISM'), (15, 'PL norm'), (16, 'Tin'), (17, 'disk norm')]
#v_rms of zones 3, 1, 2 (pion_abs, pion_abs_3, pion_abs_4) and the covering fraction (pion_emit norm)
UNIFORM_KEYS = [4, 8, 12, 22]


def get(chain):
    #parameter name -> samples, z as v_out = cz in km/s and the covering factor as Omega = norm / 4pi max
    cols = [str(k) for k in chain['keys']]
    out = {}
    for key, name in PARS:
        x = chain['samples'][:, cols.index(str(key))].copy()
        if name.startswith('v_out'):
            x *= C
        if name == 'Omega':
            x /= chain['table']['22']['hard_max']
        out[name] = x
    out['step'] = chain['step']
    return out


def q90(x):
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return '%.4g +%.3g/-%.3g' % (med, hi - med, med - lo)


chain_logu = load_chain(MAXIJ1543_CHAIN, discard=10000)
chain_unif = load_chain(MAXIJ1543_CHAIN, discard=10000, uniform_prior_keys=UNIFORM_KEYS)
P = {'UNIF': get(chain_unif), 'LOGU': get(chain_logu)}
table = chain_logu['table']

#zone 1 modes: split at the dip of the log xi1 histogram between the two peaks (log-uniform posterior)
hist, edges = np.histogram(P['LOGU']['log xi1'], bins=60, range=(4.4, 5.2))
centres = 0.5 * (edges[1:] + edges[:-1])
sel = (centres > 4.65) & (centres < 4.9)
split = centres[sel][np.argmin(hist[sel])]
print('chain: %s, %d walkers x %d steps, discard 10000; samples UNIF %d, LOGU %d'
      % (MAXIJ1543_CHAIN, chain_logu['n_walkers'], chain_logu['n_steps'], len(P['UNIF']['Omega']),
         len(P['LOGU']['Omega'])))
print('UNIF weight ESS fraction %.2f; zone 1 mode split at log xi1 = %.3f' % (chain_unif['weight_ess_frac'], split))

print('\n%-10s %-30s %-30s %s' % ('param', 'UNIF median +90%/-90%', 'LOGU median +90%/-90%', 'best fit (ML, no prior)'))
for key, name in PARS:
    best = table[str(key)]['value'] * (C if name.startswith('v_out') else 1) / \
        (table['22']['hard_max'] if name == 'Omega' else 1)
    print('%-10s %-30s %-30s %.4g' % (name, q90(P['UNIF'][name]), q90(P['LOGU'][name]), best))

for lab in ('UNIF', 'LOGU'):
    p = P[lab]
    low = p['log xi1'] < split
    print('\n== %s' % lab)
    print('N_H1+N_H2 %s | corr(N_H1, N_H2) %.2f | corr(log v_rms3, log N_H3) %.2f'
          % (q90(p['N_H1'] + p['N_H2']), np.corrcoef(p['N_H1'], p['N_H2'])[0, 1],
             np.corrcoef(np.log10(p['v_rms3']), np.log10(p['N_H3']))[0, 1]))
    print('Omega lower limits: 90%% %.3f, 95%% %.3f | P(Omega>0.5) %.2f | P(Omega<0.1) %.3f'
          % (np.percentile(p['Omega'], 10), np.percentile(p['Omega'], 5), (p['Omega'] > 0.5).mean(),
             (p['Omega'] < 0.1).mean()))
    print('v_rms3 upper limits: 90%% %.1f, 95%% %.1f km/s | P(v_rms3<5 km/s) %.3f'
          % (np.percentile(p['v_rms3'], 90), np.percentile(p['v_rms3'], 95), (p['v_rms3'] < 5).mean()))
    print('log xi3: 95th percentile %.3f | P(log xi3>3.95) %.3f' % (np.percentile(p['log xi3'], 95),
                                                                     (p['log xi3'] > 3.95).mean()))
    print('v_out2: P(>1300 km/s) %.4f, P(>1400 km/s) %.4f' % ((p['v_out2'] > 1300).mean(), (p['v_out2'] > 1400).mean()))
    counts, _ = np.histogram(p['log xi1'], bins=60, range=(4.4, 5.2))
    dip = counts[np.argmin(np.abs(centres - split))]
    peaks = [counts[(centres > 4.55) & (centres < split)].max(), counts[(centres > split) & (centres < 5.05)].max()]
    print('zone 1 modes: low-xi1 fraction %.3f; dip / lower peak %.2f' % (low.mean(), dip / min(peaks)))
    quarters = np.linspace(0, p['step'].max() + 1, 5)
    print('low-xi1 fraction per chain quarter: %s'
          % ', '.join('%.3f' % low[(p['step'] >= quarters[i]) & (p['step'] < quarters[i + 1])].mean()
                      for i in range(4)))
    for name in ('log xi1', 'v_rms1', 'N_H1', 'v_out1', 'log xi2', 'N_H2', 'v_rms2', 'v_out2'):
        print('   %-8s low-xi1 mode %-28s high-xi1 mode %s' % (name, q90(p[name][low]), q90(p[name][~low])))

print('\nhard limits (sidecar):')
for key, name in PARS[:13]:
    row = table[str(key)]
    lo, hi = row['hard_min'], row['hard_max']
    if name.startswith('v_out'):
        lo, hi = lo * C, hi * C
    if name == 'Omega':
        lo, hi = lo / table['22']['hard_max'], hi / table['22']['hard_max']
    print('   %-8s [%.4g, %.4g] prior %s' % (name, lo, hi, row.get('prior')))

<!-- Exported on 2026-10-22 from the claude.ai doc https://claude.ai/artifact/F1368hV9Z8jftZ1qGp3PRF (revision 20). Edit the doc or this file, not both. -->

# PyXspec 13 chain-runner: coding instructions

2026-10-21 · Maxime

## Goal and scope

Build one function, `run_chain()`, that takes the model currently loaded in PyXspec 13.0.1 (or restores one from an `.xcm` file), runs an MCMC chain over all thawed parameters, and saves the chain plus enough metadata to make the corner plot later without guessing. Plotting is a separate function and is out of scope here.

Scientific choices this function must respect:

- All parameters stay free by default, continuum included (TBabs nH, diskbb Tin and norm, powerlaw Γ and norm), so the 11 pion absorber parameters are marginalized over the continuum. Freezing is an explicit, logged option, used for the conditional-chain comparison or for parameters fixed by outside knowledge.
- Default sampler is `gw` (Goodman–Weare). The pion grids are OGIP table models, which the 13.0.1 manual lists as using finite-difference gradients, so `hmc` is not the default. `nest` is an option for multimodal cases.
- Fit statistic is C-stat; the function warns if anything else is active.
- Profile-likelihood errors, if wanted, are computed BEFORE the chain is loaded, because with a chain loaded `error` returns chain credible intervals.

**What is verified and what is not.** The XSPEC 13.0.1 manual documents the commands (`chain`, `chain diag`, `bayes`, `fit global`, `parallel`, `xset seed`) and says PyXspec 13 has a Bayesian module with `runChain`, `runHMC`, shared result objects and arviz export. It does NOT give their Python signatures; those are in the separate PyXspec 13 manual. The long-standing PyXspec interface (`Xset.restore`, `Fit.perform`, `AllChains`, `Chain`, `Xset.seed`, `Xset.parallel`) is assumed to still exist. Before writing code, check every call named below against the installed PyXspec 13.0.1 (`help(xspec.Fit)`, `help(xspec.AllChains)`, the PyXspec manual) and adapt names; never invent an attribute that isn't there.

**Reference material in the project.** Two PDFs are in `../../../docs`: `XSPEC13_manual_MCMC_excerpts.pdf` (73 pages: "New in 13.0.1", Chapter 3, the MCMC walkthrough, and the `parallel`, `bayes`, `chain`, `compare`, `error`, `fit`, `hmc`, `margin`, `nest`, `sim` and `xset` entries) and `XspecManual.pdf` (the full 629-page 13.0.1 manual). Read the excerpts and this spec first. Search the full manual (`pdftotext` + `grep`) only for specific lookups it lacks, such as Appendix B (fit statistics, W-statistic), Appendix H (analytic gradients), Appendix I (fix list), or the `group` and `steppar` entries.

**AI attribution (required).** The module docstring must start with an attribution notice. Code the AI writes from scratch says "Created by"; changes the AI makes to code Maxime wrote say "Assisted by", next to the changed function or in the module notice. The notice names the tool and the model, plus the date and this spec as the design source:

```python
"""
Created by Claude Code (JetBrains extension), model: <exact model name>, <YYYY-MM-DD>.
Design specification: docs/chain_runner_spec.md (written with Claude Opus 5.5 in claude.ai).
Reviewed and maintained by Maxime.
"""
```

If the AI does not know its exact model name, it must ask Maxime rather than guess. Keep the notice when the file is edited later, and update it if a different model makes substantial changes.

## Signature and arguments

```python
def run_chain(
    outfile,                 # path of the chain file to write (.fits)
    xcm=None,                # optional .xcm to restore (data + model + settings)
    sampler="gw",            # "gw" | "nest" (| "hmc", opt-in only)
    walkers=None,            # gw only; default = smallest even number >= 2*D
    steps_per_walker=2000,   # gw only; total length = walkers * steps_per_walker
    burn_per_walker=500,     # gw only; total burn = walkers * burn_per_walker
    seed=None,               # int; None draws one and records it
    n_parallel=None,         # parallel workers for the chain; None = serial
    priors=None,             # dict or JSON path; keys = index or "comp.par"; see Priors input
    prior_variants=None,     # list of alternative prior files for sensitivity reruns
    freeze=None,             # list of param indices to freeze for THIS run only
    refit=True,              # run Fit.perform() twice before sampling
    global_fit=False,        # run the DE global optimizer first (multimodality check)
    profile_errors=None,     # list of param indices for Fit.error before the chain
    save_xcm=True,           # write the post-fit state next to the chain
    overwrite=False,
    test=False,              # short chain: steps_per_walker=100, burn=20
    absorber_components=("mtable",),  # component names tagged "absorber"
    force_hmc=False,         # required to run hmc on table models
    pilot=None,              # sidecar of a test run, gives tau for length checks
    skip_length_check=False, # proceed without a pilot (logged)
    min_tau_multiple=50,     # steps after burn >= 50 * tau_max
    min_burn_tau=5,          # burn >= 5 * tau_max
    min_ess=400,             # bulk and tail ESS per parameter
    max_rhat=1.01,           # rank-normalized R-hat
) -> dict:                   # summary: paths, seed, D, walkers, diag verdict
```

Notes for the implementer:

- Parameters are addressed by XSPEC global index (as in `show param`), so the same indices work for `freeze`, `priors` and `profile_errors`.
- Express length and burn per walker, then multiply. XSPEC requires totals divisible by the walker count; this design makes that automatic.
- `walkers` must be even. With 16 free parameters the default is 32; allow up to 4D (64).
- Keep defaults conservative; `test=True` is the first thing to run on a new model.

### Priors input

`priors` accepts either a dict or a path to a JSON file kept in the project (preferred, so prior choices are version-controlled). Keys are XSPEC global indices or `"component.parameter"` names; values give the `bayes` family and its arguments, e.g. `{"diskbb.norm": ["jeffreys"], "TBabs.nH": ["cons"]}`. The resolved priors, per parameter, are copied into the sidecar.

Guidance to put in the docstring, following the 13.0.1 manual (Sec. 3.2.2–3.2.3):

- Scale parameters whose order of magnitude is uncertain (normalizations, column densities spanning decades) → `jeffreys` (uniform in log θ). This is the uninformative default.
- `lognormal` is informative (a center and a width in ln θ). Use it only with independent prior information, never with settings tuned on the same data.
- Parameters already stored as log10 in a table (e.g. log ξ, possibly the pion columns) → `cons`; flat in log is already log-uniform. Never put `jeffreys` on a log parameter.
- Location parameters with physical limits (Γ, Tin, velocities) → `cons`, or `gauss` / `cauchy` from an independent measurement.
- Bounded parameters (covering fractions) → `beta` or `cons` on [0, 1].

Validation:

- Raise if `jeffreys` or `lognormal` is set on a parameter with hard lower limit ≤ 0; the message should say to set a small positive limit with `newpar` first.
- Warn for any free parameter left flat over a hard-limit range spanning more than two decades.
- Raise for unknown family names or wrong argument counts, using the family list in the `bayes` command entry.
- Turn on Bayesian inference only after all priors are set, and confirm afterwards that it is on.

Prior sensitivity: add an option `prior_variants=None` (a list of alternative prior files). For each, rerun with the same seed and settings and write a separate, clearly labelled chain, so the weakly constrained absorber parameters can be checked for prior dependence.

## Step-by-step behaviour

1. **Load state.** If `xcm` is given, check the file exists, warn that it replaces the current session, then `Xset.restore(xcm)`. If not, require at least one model (`AllModels`) and one spectrum (`AllData.nSpectra > 0`); otherwise raise with a message telling the user to pass `xcm`.
2. **Pre-checks.**
    - Fit statistic: read `Fit.statMethod`; warn if it is not `cstat`.
    - Build a parameter table over all models and data groups: global index, component name, parameter name, unit, value, frozen/free, linked (tied), soft and hard limits. Free = not frozen and not linked. This table is reused in steps 5 and 9.
    - Apply `freeze` (if given) and log each frozen index with its value. Keep the original frozen flags so they are restored at the end.
    - Warn for any free parameter within 1% of a hard limit; that often signals a bad prior range or a pegged fit.
    - Print D (number of free parameters) and the table.
3. **Refit.** If `refit`, call `Fit.perform()` twice so the covariance is fresh; record the statistic and best-fit values. If `global_fit`, run the DE global optimizer first (`Fit.globalFit` per the manual; check its signature) and record any competitive minima it reports (`Fit.globalBasins`) with their Δstat. If any basin is within Δstat < 9, print a warning that the posterior may be multimodal and suggest `sampler="nest"`.
4. **Profile errors.** If `profile_errors`, first make sure no chain is loaded (`AllChains.clear()`), then run `Fit.error` on those indices at Δstat = 1.0 and store the intervals and error flags. This must happen before step 8.
5. **Priors.** If `priors` is given, set each one with the PyXspec equivalent of `bayes <idx> <family> <args>` (the manual names `Fit.bayesPriors` for `show bayes`; find the setter in the PyXspec manual). Then turn Bayesian inference on (`bayes on`); without it XSPEC ignores the priors. Validate first: a `jeffreys` prior needs a positive hard lower limit on that parameter, and `nest` needs proper bounded priors on every free parameter. Unset parameters keep the default flat prior within hard limits; log that explicitly.
6. **Walkers, length, burn.** For gw: `walkers` defaults to the smallest even number ≥ 2D; raise if odd. Total length = walkers × steps_per_walker; total burn = walkers × burn_per_walker. Log all three. For nest and hmc these arguments are ignored and the sampler's own settings are logged instead.
7. **Seeding and parallelism.** If `seed` is None, draw one from `secrets.randbits(31)`. Set it with `Xset.seed` (the `xset seed` equivalent) and record it. If `n_parallel`, set chain workers via `Xset.parallel` (check the exact key for chain/walkers in 13.0.1).
8. **Run.**
    - gw: `AllChains.clear()`, set `AllChains.defAlgorithm = "gw"`, `defWalkers`, `defLength`, `defBurn`, `defFileType = "fits"`, proposal `gaussian fit` (walkers are initialized from the fit covariance). Then create the chain on `outfile` and run it. Prefer the documented 13 entry point `runChain` if its signature covers these settings; fall back to the `Chain` class otherwise.
    - If the covariance is reported ill-conditioned, retry once with the uniform-deltas proposal and log the switch.
    - nest / hmc: call the matching PyXspec 13 entry point (`runHMC`, or the nest equivalent) with output to `outfile`.
    - Refuse to overwrite an existing `outfile` unless `overwrite=True`.
9. **Diagnostics.** XSPEC runs `chain diag` at the end of chain runs; capture its report (rank-normalized R̂, bulk/tail ESS, τ, verdict) from the returned result object or the log. If it is not accessible from Python, load the chain with arviz and compute R̂ and ESS there. Store the verdict as `"OK"` / `"WARN"` with the failing parameters.
10. **Save and restore.** Write the outputs (next section). Restore the original frozen flags from step 2 so the session is unchanged apart from the loaded chain. Return the summary dict.

## Chain settings: good practice and length checks

The function must refuse, or loudly warn about, settings too short to give a converged chain, and must explain why in its docstring and messages. Put the rules below in the module docstring (with the references from the References section) and in the docstring of each check function, so the reasoning travels with the code.

**Rules to explain in the code**

| Setting | Rule | Basis |
|---|---|---|
| Walkers (gw) | Even; at least 2D, ideally 2D–4D (32–64 for D = 16) | XSPEC 13.0.1 manual, Ch. 3; Foreman-Mackey et al. 2013 |
| Steps per walker after burn | ≥ 50 τ_max, where τ_max is the largest integrated autocorrelation time over parameters | emcee autocorrelation guide; `chain diag` threshold (N ≥ 50τ) |
| Burn per walker | ≥ 5 τ_max (heuristic: several τ to forget the starting ball) | emcee autocorrelation guide; Hogg & Foreman-Mackey 2018 |
| Effective sample size | Bulk and tail ESS ≥ 400 per parameter | Vehtari et al. 2021; `chain diag` threshold |
| Convergence | Rank-normalized R̂ < 1.01 | Vehtari et al. 2021; `chain diag` threshold |
| Walker consistency | Ratio of largest to smallest per-walker τ < 10; flag walkers whose acceptance fraction is far below the ensemble's | `chain diag`; Foreman-Mackey et al. 2013 (typical acceptance 0.2–0.5) |
| Thinning | Not needed for the corner plot; if used, thin by about τ/2 and record it | Hogg & Foreman-Mackey 2018 |

τ cannot be known before running, so length checks work in two tiers:

1. **Pilot (test) run.** `test=True` runs a short chain and writes per-parameter τ estimates to its sidecar, flagged `tau_reliable: false` when the pilot is shorter than 50 τ (the estimate itself is then a lower bound, so the τ used for planning must be inflated, e.g. ×2).
2. **Pre-run check (production).** New argument `pilot=<sidecar path>`. Before sampling, compute from the pilot τ_max: required steps = 50 τ_max + burn, required burn = 5 τ_max, predicted ESS ≈ walkers × steps_after_burn / τ_max. Raise if the requested settings fall short, with a message giving the minimum values. Without a pilot, warn that length cannot be validated and require `skip_length_check=True` to proceed.
3. **Post-run check.** Read the `chain diag` results (or compute them with arviz) and apply the thresholds in the table. If any fails, mark the sidecar `diag: "WARN"` and print the recommended rerun length, 1.5 × 50 τ_max per walker using the τ just measured. Never mark a run OK on R̂ alone.

Add to the signature: `pilot=None`, `skip_length_check=False`, `min_tau_multiple=50`, `min_burn_tau=5`, `min_ess=400`, `max_rhat=1.01`. Keep the defaults equal to the `chain diag` thresholds so the two never disagree.

Note for D = 16: affine-invariant ensemble samplers lose efficiency as dimension grows (Huijser et al. 2022), so expect long τ for this model. A long τ is a reason to run longer or switch to `nest`, not to loosen the thresholds.

## Outputs

Each run writes three files with a common stem, so the corner-plot code never has to guess column meanings.

| File | Content | Why |
|---|---|---|
| `<stem>.fits` | The chain as written by XSPEC | Primary samples |
| `<stem>.json` | Sidecar metadata (below) | Maps chain columns to physical parameters; records how the run was made |
| `<stem>_bestfit.xcm` | Post-fit state via `Xset.save(..., info="all")` | Reproduces the starting point exactly |

The sidecar JSON holds:

- Provenance: XSPEC and PyXspec versions, date, hostname, `xcm` input path, data file names, statistic, energy ranges noticed per spectrum.
- Sampler settings: sampler, walkers, total length, total burn, proposal (and any fallback), seed, parallel workers.
- Parameter table from step 2: index, component, name, unit, best fit, hard limits, prior (family + arguments, or "flat within hard limits"), free/frozen/linked, and frozen values. Mark the 11 pion parameters with a `group: "absorber"` tag and the rest `"continuum"`, so the plot function can select them. Tag by component name through an extra argument, e.g. `absorber_components=("mtable",)`, rather than hard-coding indices.
- Best-fit statistic, any global-fit basins with Δstat, and profile errors from step 4 (labelled as profile-likelihood, computed before the chain).
- Diagnostics verdict and per-parameter R̂ / ESS.
- `conditional: true` plus the frozen list whenever `freeze` was used, so a conditional chain can never be mistaken for the main one.

## Error handling and guardrails

- **Never overwrite silently.** Existing `outfile`, sidecar or xcm → raise unless `overwrite=True`.
- **Interrupts.** Wrap the run in `try/finally`: on `KeyboardInterrupt` or an XSPEC exception, restore frozen flags, write the sidecar with `status: "interrupted"` or `"failed"` and the exception text, then re-raise. Don't try to recover a partial gw file; treat it as unusable.
- **hmc guard.** If `sampler="hmc"`, warn that the pion mtables use finite-difference gradients (cost about D+1 likelihoods per gradient, noisy on tabulated models) and require `force_hmc=True` to proceed. XSPEC itself refuses hmc for ungradiented convolutions and mixing models; surface that error unchanged.
- **nest guard.** Before `nest`, verify every free parameter has a proper bounded prior; raise listing the ones that don't.
- **Global state.** PyXspec is a single global session. Don't use Python multiprocessing around this function; use XSPEC's own parallel workers (step 7). Document that calling it changes the loaded chain and Bayes settings.
- **Chatter and log.** Open an XSPEC log file next to the chain (`Xset.openLog`) for the run, so the `chain diag` report and any warnings are kept even if Python parsing of them fails. Close it in `finally`.
- **No silent defaults.** Every value the function picks itself (walkers, seed, prior defaults, proposal fallback) is printed and written to the sidecar.

## Test plan

Run these in order; the first two answer the open questions the manual leaves undocumented.

1. **API check.** In PyXspec 13.0.1, list the real names and signatures of `runChain`, `runHMC`, the nest entry point, `Fit.globalFit`, `Fit.globalBasins`, the prior setter, `Xset.seed` and the parallel key for chains. Record them in a comment block at the top of the module.
2. **Layout discovery.** On a cheap model (e.g. TBabs*powerlaw on one spectrum), run `test=True` with 6 walkers (2D for 3 free parameters). Open the FITS file with astropy and record: extension names, column names and how they map to parameter indices, the statistic column, and header keywords. Then check walker row ordering: run the same seed twice and compare, and check whether consecutive rows cycle through walkers. Write the findings into the sidecar schema (they become the contract for `read_chain`).
3. **Reproducibility.** Same seed, same model → identical chain files. Different seeds → different files. Repeat with `n_parallel=2` to confirm parallel workers don't duplicate each other.
4. **Guardrails.** Odd `walkers` raises; existing `outfile` without `overwrite` raises; `sampler="nest"` with a flat-unbounded parameter raises; `sampler="hmc"` without `force_hmc` raises; Ctrl-C mid-run leaves a sidecar with `status: "interrupted"` and frozen flags restored.
5. **Profile-error ordering.** Run with `profile_errors` and confirm the stored intervals match a manual `error` run on a session with no chain loaded.
6. **Full model, test chain.** Your TBabs(diskbb+powerlaw) × pion model with `test=True` and 32 walkers: check D = 16 (plus any constants), the absorber/continuum tags, and the run time per step. Use that to size the production run.
7. **Production, then conditional.** Full chain until the diagnostics verdict is OK. Then the same seed and settings with `freeze` on the continuum, to measure how much of each absorber width comes from the continuum.

## References

These were written from memory without web access, so check each link and citation before it goes into the code or the paper.

- XSPEC 13.0.1 manual, Chapter 3 (priors, samplers, convergence diagnostics) and the `chain`, `chain diag`, `bayes`, `hmc`, `nest` command entries. Local PDF.
- Goodman, J. & Weare, J. 2010, *Ensemble samplers with affine invariance*, Comm. App. Math. Comp. Sci. 5, 65. [doi:10.2140/camcos.2010.5.65](https://doi.org/10.2140/camcos.2010.5.65)
- Foreman-Mackey, D. et al. 2013, *emcee: The MCMC Hammer*, PASP 125, 306. [doi:10.1086/670067](https://doi.org/10.1086/670067), [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- emcee documentation, *Autocorrelation analysis & convergence* tutorial. [emcee.readthedocs.io](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. 2021, *Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC*, Bayesian Analysis 16, 667. [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221), [arXiv:1903.08008](https://arxiv.org/abs/1903.08008)
- Gelman, A. & Rubin, D. B. 1992, *Inference from iterative simulation using multiple sequences*, Statistical Science 7, 457. [doi:10.1214/ss/1177011136](https://doi.org/10.1214/ss/1177011136)
- Hogg, D. W. & Foreman-Mackey, D. 2018, *Data analysis recipes: Using Markov chain Monte Carlo*, ApJS 236, 11. [doi:10.3847/1538-4365/aab76e](https://doi.org/10.3847/1538-4365/aab76e), [arXiv:1710.06068](https://arxiv.org/abs/1710.06068)
- Huijser, D., Goodman, J. & Brewer, B. J. 2022, *Properties of the affine-invariant ensemble sampler's 'stretch move' in high dimensions*, Aust. N. Z. J. Stat. [arXiv:1509.02230](https://arxiv.org/abs/1509.02230)
- Cash, W. 1979, *Parameter estimation in astronomy through application of the likelihood ratio*, ApJ 228, 939. [doi:10.1086/156922](https://doi.org/10.1086/156922)
- ArviZ documentation (R̂, ESS, MCSE). [python.arviz.org](https://python.arviz.org/)


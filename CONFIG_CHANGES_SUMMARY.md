# Config Changes Summary (Simplified)

Full version: `CONFIG_CHANGES.md`

---

## What changed

### Instance Generator — simplified

**Removed:**
- Profiles (baseline/realistic)
- Chemistry model (COLOUR_FAMILIES, CHEM_COMPAT, CHEM_FACTOR)
- Continuous colour model, customer segments
- `profile`, `tightness`, `proc_colour_corr`, `setup_time_scale` params

**Changed:**
- Processing times: fixed 5–31 hrs → capacity-based (fills 168-hr week)
- Due dates: tightness-based → proportional within week (SPT order)
- Setup time: cost-proportional → 1/8 of avg processing time
- Cost matrix: nonlinear `diff^1.5` → linear `diff × 10`; light→dark now `|diff| × 3`
- Noise: gamma(2,1) → uniform(0,2)

**Instance configs:** 11 → 8. Shifted from multi-machine (m=2,3,5,10) to single-machine (m=1) base.

### Evaluator — minor

- `estimate_scales()`: `rng` param removed, now deterministic (seed=0)
- New `extract_schedule()` function

### Unchanged

`ga_env.py`, `drl_agent.py`, `ga.py` — no changes. PPO hyperparameters, obs/action spaces, GA toolbox all identical.

### Experiment scripts

All 5 scripts: `--profile` removed, `--small` added. Output paths no longer profile-suffixed.

---

## Design rationale

1. Grounded instance generation in real 168-hr week operation
2. Linear cost model simpler and more interpretable
3. Single-machine base configs for clean scaling analysis
4. Deterministic normalisation for reproducibility
5. Simplified CLI with `--small` for quick local runs

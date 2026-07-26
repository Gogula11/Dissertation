# New Additions & Design Rationale

Focus: entirely new things added since `81922a6`. For full diff see `CONFIG_CHANGES.md`.

---

## 1. `weekly_hours` parameter (default 168.0)

Processing times now grounded in real operation: `avg_proc = (m × 168) / n`. Total work across all machines fills exactly one 24/7 week.

**Rationale:** Textile dyeing factories typically operate 24/7 with multi-shift patterns (e.g. three 8-hour shifts per day, or two 12-hour shifts), giving 168 available hours per week. The previous model used arbitrary integer ranges (5–31 hrs) with no physical meaning. Weekly capacity ties the model to actual factory scheduling horizons.

---

## 2. Five single-machine configs (m=1)

`n5_m1` (n=5), `n10_m1` (n=10), `n20_m1` (n=20), `n50_m1` (n=50), `n100_m1` (n=100).

**Rationale:** Single-machine isolates sequencing effect from assignment effect. Clean scaling analysis from n=5 to n=100 without machine allocation confounding.

---

## 3. `n500_m10` (n=500, m=10)

Largest config for stress-testing scalability.

**Rationale:** Tests whether GA/DRL methods generalise to realistic production scale (500 jobs across 10 machines).

---

## 4. Seeds set for reproducibility

All random processes now use fixed seeds. Same inputs always produce same outputs.

**Rationale:** Ensures experiments are reproducible across runs.

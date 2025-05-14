"""
Monte‑Carlo exploration of how Type I (FWER/FDR) and Type II error rates
behave as the number of simultaneous hypothesis tests grows, under three
procedures: no correction, Bonferroni, Benjamini–Hochberg.

© 2025  –  feel free to reuse.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm

# Simulation parameters
alpha   = 0.05            # nominal error rate
ns      = np.arange(1, 21)  # number of simultaneous tests
pi_alt  = 0.20            # proportion of true alternatives
mu_alt  = 2.8             # effect size under H1 (in SD units)
n_sim   = 5_000           # Monte‑Carlo repetitions per n
#

# Containers for metrics
fwer_no, fwer_bonf, fwer_bh, fdr_bh = [], [], [], []
type2_no, type2_bonf, type2_bh = [], [], []

for n in ns:
    # Counters reset each n
    fp_no = fp_bonf = tp_no = tp_bonf = tp_bh = 0
    fwer_no_cnt = fwer_bonf_cnt = fwer_bh_cnt = 0
    total_alt = 0
    fdr_runs = []                      # run‑level false‑discovery proportions

    for _ in range(n_sim):
        # Truth status of each hypothesis
        alt_mask  = np.random.rand(n) < pi_alt
        null_mask = ~alt_mask
        total_alt += alt_mask.sum()

        # Generate z‑statistics and p‑values
        z = np.random.randn(n)
        z[alt_mask] += mu_alt
        p = 2 * (1 - norm.cdf(np.abs(z)))

        # No correction 
        reject_no = p < alpha
        fp_no += np.sum(reject_no & null_mask)
        tp_no += np.sum(reject_no & alt_mask)
        if (reject_no & null_mask).any():
            fwer_no_cnt += 1

        # Bonferroni 
        reject_bonf = p < alpha / n
        fp_bonf += np.sum(reject_bonf & null_mask)
        tp_bonf += np.sum(reject_bonf & alt_mask)
        if (reject_bonf & null_mask).any():
            fwer_bonf_cnt += 1

        # Benjamini–Hochberg
        sort_idx  = np.argsort(p)
        p_sorted  = p[sort_idx]
        thresh    = (np.arange(1, n + 1) / n) * alpha
        below     = p_sorted <= thresh
        if below.any():
            cutoff = p_sorted[np.max(np.where(below))]
            reject_bh = p <= cutoff
        else:
            reject_bh = np.zeros_like(p, bool)

        # BH counts
        fp_bh_run = np.sum(reject_bh & null_mask)
        tp_bh    += np.sum(reject_bh & alt_mask)
        R_run     = reject_bh.sum()
        fdr_runs.append(fp_bh_run / R_run if R_run else 0)
        if fp_bh_run:                      # at least one FP ⇒ FWER event
            fwer_bh_cnt += 1

    # Aggregate over simulations
    # Type I
    fwer_no.append(   fwer_no_cnt   / n_sim)
    fwer_bonf.append( fwer_bonf_cnt / n_sim)
    fwer_bh.append(   fwer_bh_cnt   / n_sim)
    fdr_bh.append(    np.mean(fdr_runs))

    # Type II  (1 − power)
    power_no   = tp_no   / total_alt if total_alt else 0
    power_bonf = tp_bonf / total_alt if total_alt else 0
    power_bh   = tp_bh   / total_alt if total_alt else 0
    type2_no.append( 1 - power_no)
    type2_bonf.append(1 - power_bonf)
    type2_bh.append( 1 - power_bh)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Type II(Left axis)

ax1.plot(ns, fwer_no,   'ko-', label='FWER (none)')
ax1.plot(ns, fwer_bonf, 'ro-', label='FWER (Bonferroni)')
ax1.plot(ns, fwer_bh,   'g^-', label='FWER (BH)')
ax1.plot(ns, fdr_bh,    'bs-', label='FDR  (BH)')

ax1.set_xlabel("Number of simultaneous tests")
ax1.set_ylabel("Probability of Type I error")
ax1.set_yscale('log')                # log scale shows Bonf & BH separation
ax1.set_ylim(1e-4, 1)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Type II (right axis)
ax2.plot(ns, type2_no,   'k--', label='Type II (none)')
ax2.plot(ns, type2_bonf, 'r--', label='Type II (Bonferroni)')
ax2.plot(ns, type2_bh,   'b--', label='Type II (BH)')
ax2.set_ylabel("Probability of Type II error")
ax2.set_ylim(0, 1)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# x‑axis ticks
ax1.set_xticks(ns)

# unified legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           loc='upper right', fontsize=9)

plt.title("Type I vs Type II error as the number of tests grows")
plt.tight_layout()
plt.show()
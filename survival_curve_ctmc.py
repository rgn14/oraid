#!/usr/bin/env python3
# Robust survivability plot (use this exact script)
import numpy as np
import math
import matplotlib.pyplot as plt
from math import gamma

# ---------------------------------------
# Parameters
# ---------------------------------------
Ns = [12, 16, 20]               # constellation sizes
weib_k = 1.4
eta_years = 12.0
mean_weibull_years = eta_years * gamma(1.0 + 1.0/weib_k)
lambda_per_year = 1.0 / mean_weibull_years   # failure rate λ

MTTR_days = 180.0
rho = 1.0 / (MTTR_days / 365.25)             # repair rate ρ

# ---------------------------------------
# Transient generator for S0, S1, S2
# ---------------------------------------
def Q_TT(N, lam, rho):
    Q = np.array([
        [-N*lam,           N*lam,                  0.0],
        [rho,             -((N-1)*lam + rho),      (N-1)*lam],
        [0.0,             2*rho,                  -((N-2)*lam + 2*rho)]
    ], dtype=float)
    return Q

# ---------------------------------------
# Stable real-valued matrix exponential via eigen-decomposition
# ---------------------------------------
def expm_via_eig(A, t):
    vals, vecs = np.linalg.eig(A)
    expD = np.diag(np.exp(vals * t))
    V_inv = np.linalg.inv(vecs)
    M = vecs @ expD @ V_inv
    # Remove tiny imaginary parts
    if np.max(np.abs(M.imag)) < 1e-12:
        M = M.real
    return M

# ---------------------------------------
# Compute survivability curves
# ---------------------------------------
t_years = np.linspace(0.0, 10.0, 801)  # smooth timeline (801 samples)
results = {}

for N in Ns:
    Q = Q_TT(N, lambda_per_year, rho)
    p0 = np.array([1.0, 0.0, 0.0])
    R = []
    for t in t_years:
        M = expm_via_eig(Q, t)
        pt = p0 @ M
        R.append(pt.sum())    # S0 + S1 + S2
    results[N] = np.array(R)
    print(f"N={N}, Survivability at 10 years = {results[N][-1]:.6f}")

# ---------------------------------------
# Plot with legend
# ---------------------------------------
plt.figure(figsize=(7,5))

for N, style in zip(Ns, ['-', '--', ':']):
    plt.plot(t_years, results[N], style, linewidth=2, label=f"N = {N}")

plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Survivability $R_{sys}(t)$", fontsize=12)
plt.title("Survivability Curve Over 10-year Mission (CTMC Model)", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.ylim(0.0, 1.0)

plt.tight_layout()
plt.savefig("c:/users/lakmobile/survival_curve_ctmc_corrected.png", dpi=300)
"c:/users/lakmobile/survival_curve_ctmc_corrected.png"

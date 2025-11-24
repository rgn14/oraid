#!/usr/bin/env python3
"""
Robust CTMC MTTDL solver for ORAID (Models A, B, C)
- Enforces safe m (1 <= m <= N-2)
- Uses consistent units (years)
- Attempts np.linalg.solve then falls back to regularized solve or pseudo-inverse
- Prints diagnostics when fallbacks are used
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from math import gamma

# ---------------------------
# Parameters
# ---------------------------
Ns = [10, 12, 16, 20, 30]

weib_k = 1.4
eta_years = 12.0
mean_weibull_years = eta_years * gamma(1.0 + 1.0 / weib_k)
lambda_per_year = 1.0 / mean_weibull_years

MTTR_days_baseline = 180.0
MTTR_days_hot = 30.0
rho_baseline = 1.0 / (MTTR_days_baseline / 365.25)
rho_hot = 1.0 / (MTTR_days_hot / 365.25)

def build_Q(N, m, lam, rho):
    size = m + 1
    Q = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        if i < m:
            Q[i, i+1] = (N - i) * lam
        if i > 0:
            Q[i, i-1] = i * rho
        diag = 0.0
        if i < m:
            diag += (N - i) * lam
        if i > 0:
            diag += i * rho
        Q[i, i] = -diag
    return Q

def safe_solve_Q(Q):
    """
    Solve (-Q) t = 1 robustly.
    Returns t (vector) and a dict 'info' indicating method used.
    """
    ones = np.ones(Q.shape[0], dtype=np.float64)
    A = -Q
    info = {'method': 'direct', 'cond': None, 'eps_used': 0.0}
    # compute condition number (may be large)
    try:
        cond = np.linalg.cond(A)
    except Exception:
        cond = float('inf')
    info['cond'] = cond

    # if condition is OK, try direct solve
    if cond < 1e12:
        try:
            t = np.linalg.solve(A, ones)
            return t, info
        except np.linalg.LinAlgError:
            # fallthrough to fallback
            pass

    # regularized solve: add tiny eps to diagonal
    # choose eps relative to norm(A)
    normA = np.linalg.norm(A, ord=2)
    eps = 1e-12 * max(1.0, normA)  # tiny relative perturbation
    info['eps_used'] = eps
    try:
        A_reg = A.copy()
        A_reg[np.diag_indices_from(A_reg)] += eps
        t = np.linalg.solve(A_reg, ones)
        info['method'] = 'regularized_solve'
        return t, info
    except np.linalg.LinAlgError:
        # final fallback: pseudoinverse (least-squares)
        A_pinv = np.linalg.pinv(A)
        t = A_pinv.dot(ones)
        info['method'] = 'pseudo_inverse'
        return t, info

def compute_mttdl(N, m_raw, lam, rho):
    # sanitize m
    m = max(1, min(int(round(m_raw)), N - 2))
    Q = build_Q(N, m, lam, rho)
    tvec, info = safe_solve_Q(Q)
    t0 = float(tvec[0])
    return t0, tvec, m, Q, info

# Run models A,B,C
results = []
for N in Ns:
    # Model A: fixed m=2 (clipped)
    mA_req = 2
    mttdlA, tA, mA, QA, infoA = compute_mttdl(N, mA_req, lambda_per_year, rho_baseline)
    if infoA['method'] != 'direct':
        print(f"[WARN] N={N} ModelA used fallback method: {infoA}")

    # Model B: scaled parity m≈0.2N
    mB_req = round(0.2 * N)
    mttdlB, tB, mB, QB, infoB = compute_mttdl(N, mB_req, lambda_per_year, rho_baseline)
    if infoB['method'] != 'direct':
        print(f"[WARN] N={N} ModelB used fallback method: {infoB}")

    # Model C: same as B but hot spare
    mttdlC, tC, mC, QC, infoC = compute_mttdl(N, mB_req, lambda_per_year, rho_hot)
    if infoC['method'] != 'direct':
        print(f"[WARN] N={N} ModelC used fallback method: {infoC}")

    results.append({
        'N': N, 'mA': mA, 'MTTDL_A': mttdlA,
        'mB': mB, 'MTTDL_B': mttdlB,
        'mC': mC, 'MTTDL_C': mttdlC,
        'infoA': infoA, 'infoB': infoB, 'infoC': infoC
    })

# Save CSV
with open('mttdl_curves_ABC.csv', 'w', newline='') as f:
    import csv
    w = csv.writer(f)
    w.writerow(['N', 'mA', 'MTTDL_A_years', 'mB', 'MTTDL_B_years', 'mC', 'MTTDL_C_years'])
    for r in results:
        w.writerow([r['N'], r['mA'], r['MTTDL_A'], r['mB'], r['MTTDL_B'], r['mC'], r['MTTDL_C']])

# Print results
print("\n=== MTTDL RESULTS (years) ===")
for r in results:
    print(f"N={r['N']:>2} | A(m={r['mA']}): {r['MTTDL_A']:.6e} | "
          f"B(m={r['mB']}): {r['MTTDL_B']:.6e} | C(m={r['mC']} hot): {r['MTTDL_C']:.6e} | "
          f"infoA={r['infoA']['method']}, infoB={r['infoB']['method']}, infoC={r['infoC']['method']}")

# Plot
plt.figure(figsize=(7,4))
plt.plot([r['N'] for r in results], [r['MTTDL_A'] for r in results], 'o-', label='Model A: m=2')
plt.plot([r['N'] for r in results], [r['MTTDL_B'] for r in results], 's--', label='Model B: m≈0.2N')
plt.plot([r['N'] for r in results], [r['MTTDL_C'] for r in results], '^:', label='Model C: scaled + hot spare')
plt.yscale('log')
plt.xlabel('Constellation size N')
plt.ylabel('MTTDL (years, log scale)')
plt.title('CTMC MTTDL Curves (Robust)')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('mttdl_curves_ABC.png', dpi=300)
print('Saved: mttdl_curves_ABC.png, mttdl_curves_ABC.csv')


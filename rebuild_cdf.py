#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (realistic hours-scale)
# -----------------------------
sigma = 1.2e-6          # Rayleigh scale for pointing error (radians)
B_nom_Gbps = 0.001      # nominal ISL bandwidth (Gbps) -> 1 Mbps
samples = 1000_000       # Monte Carlo samples

# data to rebuild: (k-1)*S = 7 * 64 MB = 0.448 GB = 3.584 Gb
data_Gb = 0.448 * 8.0   # gigabits

# -----------------------------
# Random samples
# -----------------------------
Pe = np.random.rayleigh(scale=sigma, size=samples)            # radians
# simple misalignment model (first-order Gaussian beam-walk)
B_eff_Gbps = B_nom_Gbps * np.exp(-(Pe / sigma)**2)            # Gbps

alpha = np.random.uniform(0.45, 0.85, size=samples)           # availability
latency_ms = np.random.uniform(3.0, 18.0, size=samples)       # ms
latency_hours = latency_ms / 1000.0 / 3600.0                  # convert ms -> hours

# rebuild time in hours: data bits / (alpha * B_eff (Gbps) * 1e9 bits/s) -> seconds -> hours
# Note: data_Gb is in gigabits: multiply by 1e9 to get bits
Trebuild_hours = (data_Gb * 1e9 / (alpha * (B_eff_Gbps * 1e9))) / 3600.0 + latency_hours

# -----------------------------
# Create CDF
# -----------------------------
sorted_t = np.sort(Trebuild_hours)
p = np.linspace(0, 1, len(sorted_t))

plt.figure(figsize=(7,4))
plt.plot(sorted_t, p, color='tab:orange', lw=1.5)
plt.xlabel('Rebuild Time (hours)')
plt.ylabel('CDF')
plt.title('CDF of Effective Parity Rebuild Time (N=16)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('rebuild_cdf_realistic.png', dpi=300)
# Optionally save raw arrays for reproducibility
np.savez('rebuild_cdf_data.npz', trebuild=Trebuild_hours, pe=Pe, beff=B_eff_Gbps, alpha=alpha)
print('Saved: rebuild_cdf_realistic.png and rebuild_cdf_data.npz')

# compute percentiles (approx)
for q in [0.25, 0.5, 0.75, 0.99]:
    print(f'{int(q*100)}th percentile: {np.percentile(Trebuild_hours, q*100):.2f} hours')


#!/usr/bin/env python3
"""
O-RAID Parity Rebuild Simulation for Satellite Constellation (N = 16)
Models:
 - ISL misalignment (Rayleigh)
 - Effective bandwidth degradation
 - Availability factor
 - Latency impact
 - Rebuild time distribution (CDF)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (realistic hours-scale)
# -----------------------------
N = 16                # constellation size (fixed for this simulation)
sigma = 1.2e-6        # Rayleigh scale for pointing error (radians)
B_nom_Gbps = 0.001    # nominal ISL bandwidth = 1 Mbps (in Gbps)
samples = 200000      # Monte Carlo samples

# Data to rebuild: (k-1)*S = 7 * 64 MB = 448 MB = 0.448 GB
# Convert to gigabits: 1 byte = 8 bits
data_Gb = 0.448 * 8.0   # gigabits

# -----------------------------
# Monte Carlo Sampling
# -----------------------------

# Random Rayleigh pointing error (radians)
Pe = np.random.rayleigh(scale=sigma, size=samples)

# Effective bandwidth from beam-walk penalty
B_eff_Gbps = B_nom_Gbps * np.exp(-(Pe / sigma)**2)   # Gbps

# Availability (random variation of link uptime 45%–85%)
alpha = np.random.uniform(0.45, 0.85, size=samples)

# Random one-way latency impact (3 to 18 ms)
latency_ms = np.random.uniform(3.0, 18.0, size=samples)
latency_hours = latency_ms / 1000.0 / 3600.0  # seconds -> hours

# Rebuild time:
# time = data bits / throughput bits per second  -> seconds -> hours
Trebuild_hours = (data_Gb * 1e9 / (alpha * (B_eff_Gbps * 1e9))) / 3600.0 + latency_hours

# -----------------------------
# Plot CDF
# -----------------------------
sorted_t = np.sort(Trebuild_hours)
p = np.linspace(0, 1, len(sorted_t))

plt.figure(figsize=(7,4))
plt.plot(sorted_t, p, color='tab:orange', lw=1.5)
plt.xlabel('Rebuild Time (hours)')
plt.ylabel('CDF')
plt.title('CDF of Effective Parity Rebuild Time (N = 16)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('rebuild_cdf_realistic_N16.png', dpi=300)

# -----------------------------
# Save raw data
# -----------------------------
np.savez('rebuild_cdf_data_N16.npz', trebuild=Trebuild_hours,
         pe=Pe, beff=B_eff_Gbps, alpha=alpha)

print("\n Saved: rebuild_cdf_realistic_N16.png and rebuild_cdf_data_N16.npz")

# -----------------------------
# Print percentile statistics
# -----------------------------
print("\nRebuild Time Percentiles (Hours):")
for q in [0.25, 0.5, 0.75, 0.99]:
    print(f"  {int(q*100)}th percentile: {np.percentile(Trebuild_hours, q*100):.2f} hours")

print("\nSimulation Complete for N = 16\n")






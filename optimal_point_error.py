import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simulation parameters
# -------------------------------

sigma = 1.2e-6        # Rayleigh scale for pointing error (in radians)
B_nom = 10.0          # Nominal ISL bandwidth in Gbps
N = 100000            # Number of Monte Carlo samples

# Beam-walk / misalignment loss model:
# Effective bandwidth = B_nom * exp( - (Pe/sigma)^2 )
# Pe is drawn from Rayleigh(sigma)
# -------------------------------

Pe = np.random.rayleigh(scale=sigma, size=N)
B_eff = B_nom * np.exp(-(Pe / sigma)**2)

# -------------------------------
# Rebuild time model
# -------------------------------
# Data volume = (k-1)*S = 7 * 64 MB = 448 MB = 0.448 GB
# Convert to gigabits for bandwidth calculation
# -------------------------------

data_GB = 0.448
data_Gb = data_GB * 8.0  # gigabits

# Link availability uniform model
alpha = np.random.uniform(0.45, 0.85, size=N)

# Latency contribution: 3–18 ms, converted to hours
latency_hours = np.random.uniform(0.003, 0.018, size=N) / 3600.0

# Rebuild time in hours:
# T = (data_Gb / (alpha * B_eff * 1e9 bits/sec)) converted to hours
Trebuild_hours = (data_Gb / (alpha * B_eff * 1e9)) / 3600.0 + latency_hours

# -------------------------------
# Figure 1: Pointing Error vs Effective Bandwidth
# -------------------------------

plt.figure(figsize=(6,4))
plt.scatter(Pe * 1e6, B_eff, s=2, alpha=0.2)
plt.xlabel("Pointing Error (μrad)")
plt.ylabel("Effective Bandwidth (Gbps)")
plt.title("Pointing Error vs Effective ISL Bandwidth")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/data/pe_vs_bw.png", dpi=300)

# -------------------------------
# Figure 2: Pointing Error vs Rebuild Time
# -------------------------------

plt.figure(figsize=(6,4))
plt.scatter(Pe * 1e6, Trebuild_hours, s=2, alpha=0.2)
plt.xlabel("Pointing Error (μrad)")
plt.ylabel("Rebuild Time (hours)")
plt.title("Pointing Error vs Rebuild Time")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/data/pe_vs_rebuild.png", dpi=300)

# Output filenames for convenience
"/mnt/data/pe_vs_bw.png", "/mnt/data/pe_vs_rebuild.png"

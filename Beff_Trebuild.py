import numpy as np

# ================================================================
# 1) FUNCTION: simulate mean(Beff) for a given BISL mean
# ================================================================
def simulate_beff_mean(target_mean_BISL, 
                       N=1000000,
                       sigma_log=0.5,
                       sigma_pe=1.2e-6,
                       g=1.0,
                       seed=12345):
    """
    Monte Carlo simulation of E[Beff] for a given BISL parent mean.
    Returns the mean effective bandwidth (Gbps).
    """

    rng = np.random.default_rng(seed)

    # lognormal mu chosen so that E[BISL] = target_mean_BISL
    mu_log = np.log(target_mean_BISL) - 0.5 * sigma_log**2

    # Sample baseline ISL (Gbps)
    BISL = rng.lognormal(mean=mu_log, sigma=sigma_log, size=N)

    # Pointing error (Rayleigh) in radians
    Pe = rng.rayleigh(scale=sigma_pe, size=N)

    # Misalignment loss (dB)
    LP_dB = 4.3429 * g * (Pe**2)

    # Effective ISL bandwidth
    Beff = BISL * (10**(-LP_dB / 10.0))

    return Beff.mean()



# 2) BISECTION SEARCH: find BISL mean giving E[Beff] ≈ 6.3 Gbps

Beff_target = 6.30    # desired mean Beff
tol = 0.02            # tolerance
low, high = 0.5, 40.0 # search interval in Gbps

for _ in range(40):
    mid = 0.5 * (low + high)
    m = simulate_beff_mean(mid)
    # print(mid, m)   # uncomment to watch convergence
    if m > Beff_target:
        high = mid
    else:
        low = mid
    if abs(m - Beff_target) < tol:
        break

BISL_mean_calibrated = mid
print("\n===============================================")
print("CALIBRATED BISL mean (pre-misalignment):", BISL_mean_calibrated)
print("Produces simulated E[Beff] ≈", m, "Gbps")
print("===============================================\n")



# 3) STEP: compute data size needed for Trebuild ≈ 9.4 hours

Trebuild_target = 9.4   # hours

# First we must estimate current Trebuild mean using some initial guess.
# Use previous 17 TB as initial full-node rebuild size.
data_TB_guess = 17.0

def simulate_trebuild_mean(data_TB,
                           BISL_mean,
                           N=1000000,
                           sigma_log=0.5,
                           sigma_pe=1.2e-6,
                           g=1.0,
                           seed=54321):

    rng = np.random.default_rng(seed)

    # Lognormal ISL parameters
    mu_log = np.log(BISL_mean) - 0.5*sigma_log**2
    BISL = rng.lognormal(mean=mu_log, sigma=sigma_log, size=N)

    # Pointing error
    Pe = rng.rayleigh(scale=sigma_pe, size=N)

    # Misalignment loss
    LP_dB = 4.3429 * g * (Pe**2)
    Beff = BISL * 10**(-LP_dB/10)

    # Rebuild data size
    data_bytes = data_TB * 1024**4

    # Availability + latency
    alpha = rng.uniform(0.45, 0.85, size=N)
    latency_s = rng.uniform(3e-3, 18e-3, size=N)

    Trebuild_s = data_bytes * 8.0 / (alpha * Beff * 1e9) + latency_s
    Trebuild_h = Trebuild_s / 3600.0

    return np.mean(Trebuild_h), Trebuild_h


# Compute current Trebuild using guessed data size
mean_T_rebuild_guess, _ = simulate_trebuild_mean(data_TB_guess, BISL_mean_calibrated)
scale_factor = Trebuild_target / mean_T_rebuild_guess
data_TB_calibrated = data_TB_guess * scale_factor

print("Guessed data size:", data_TB_guess, "TB")
print("Simulated Trebuild_mean:", mean_T_rebuild_guess, "h")
print("Scale factor:", scale_factor)
print("CALIBRATED data size:", data_TB_calibrated, "TB\n")



# 4) FINAL CALIBRATED MONTE CARLO RUN

mean_Trebuild_final, Trebuild_samples = simulate_trebuild_mean(
    data_TB_calibrated, BISL_mean_calibrated, N=1000000
)

# Now compute final Beff using calibrated BISL mean
# (Reuse simulate_beff_mean with larger N)
final_Beff_mean = simulate_beff_mean(BISL_mean_calibrated, N=1000000)

# Get percentiles for reporting
final_Beff_samples = None
# generate full samples
rng = np.random.default_rng(111)
mu_log_final = np.log(BISL_mean_calibrated) - 0.5*(0.5**2)
BISL_full = rng.lognormal(mean=mu_log_final, sigma=0.5, size=1000000)
Pe_full = rng.rayleigh(scale=1.2e-6, size=1000000)
LP_full = 4.3429 * (Pe_full**2)
Beff_full = BISL_full * 10**(-LP_full/10)
final_Beff_samples = Beff_full


# 5) PRINT RESULTS

print("==============================================================")
print("FINAL CALIBRATED RESULTS")
print("--------------------------------------------------------------")
print(f"Calibrated BISL mean: {BISL_mean_calibrated:.4f} Gbps")
print(f"Final E[Beff]: {final_Beff_mean:.4f} Gbps")
print(f"Final median(Beff): {np.median(final_Beff_samples):.4f} Gbps")
print("Beff percentiles (Gbps):", np.percentile(final_Beff_samples,
                                              [1,5,25,50,75,95,99]))
print("--------------------------------------------------------------")
print(f"Calibrated Data Size: {data_TB_calibrated:.3f} TB")
print(f"Final E[T_rebuild]: {mean_Trebuild_final:.4f} hours")
print(f"Median(T_rebuild): {np.median(Trebuild_samples):.4f} hours")
print("Trebuild percentiles (hours):", np.percentile(Trebuild_samples,
                                                    [1,5,25,50,75,95,99]))
print("==============================================================")



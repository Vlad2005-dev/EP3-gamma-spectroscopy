#!/usr/bin/env python3
"""
Coincidence angular dependence analysis.

Folder structure assumed:

EP3 Gamma Submission/
├── Data/
│   └── Day 4 Coincidence/
│       ├── co_-20_5min_65mm.Spe
│       ├── co_-15_5min_65mm.Spe
│       ├── co_-10_5min_65mm.Spe
│       ├── co_-7_5min_65mm.Spe
│       ├── co_-5_5min_65mm.Spe
│       ├── co_0_5min_65mm.Spe
│       ├── co_5_5min_65mm.Spe
│       ├── co_7_5min_65mm.Spe
│       ├── co_10_5min_65mm.Spe
│       ├── co_15_5min_65mm.Spe
│       └── co_20_5min_65mm.Spe
├── Plots/
│   └── Coincidence Plots/
│       ├── Main Plots/
│       └── Photopeak Windows/     (per-angle spectra with peak window)
└── Python Code/
    └── coincidence_angles.py      (this script)

This script should be in: EP3 Gamma Submission/Python Code/
"""

from pathlib import Path
import re

import numpy as np
import numdifftools
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import scipy.linalg


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../Submission/Python Code
ROOT_DIR = BASE_DIR.parent                          # .../Submission

DATA_DIR = ROOT_DIR / "Data" / "Day 4 Coincidence"
PLOTS_ROOT_DIR = ROOT_DIR / "Plots" / "Coincidence Plots"
MAIN_PLOTS_DIR = PLOTS_ROOT_DIR / "Main Plots"
PEAK_PLOTS_DIR = PLOTS_ROOT_DIR / "Photopeak Windows"   # renamed from "Peaks"

MAIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PEAK_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Data reader
# -------------------------------------------------
def data_reader(filename):
    """
    Reads an ASCII .Spe file and returns (E, N):
        E = channel numbers (float bin centers)
        N = counts (int)
    Trims leading and trailing zeros in the counts.
    """
    filename = Path(filename)
    with filename.open("r") as f:
        lines = f.readlines()

    # Find $DATA block
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            data_start = i + 1
            break
    else:
        raise ValueError(f"No $DATA section in .Spe file: {filename}")

    # Channel range (e.g. "0 1023")
    low, high = map(int, lines[data_start].split())
    n_channels = high - low + 1

    # Read counts
    count_lines = lines[data_start + 1: data_start + 1 + n_channels]
    counts = np.array([int(x.strip()) for x in count_lines], dtype=int)

    # Trim zeros at start and end
    nonzero_indices = np.where(counts > 0)[0]
    if len(nonzero_indices) == 0:
        raise ValueError(f"All counts are zero — no spectrum found in {filename}.")

    first = nonzero_indices[0]
    last = nonzero_indices[-1]

    # Build trimmed outputs
    E = np.arange(low, high + 1, dtype=int)[first:last + 1] + 0.5
    N = counts[first:last + 1]

    return E, N


# -------------------------------------------------
# Global containers for results
# -------------------------------------------------
total_counts = []      # counts within photopeak window (signal)
total_background = []  # counts outside window (background)


# -------------------------------------------------
# Single-spectrum analysis: integrate photopeak counts for a given angle
# -------------------------------------------------
def histogram_na22(filename, peak_range):
    """
    For a given coincidence spectrum:
    - Extracts the detector angle from the filename.
    - Integrates counts within the photopeak window [peak_range[0], peak_range[1]].
    - Separately sums the "background" counts outside the window.
    - Plots the spectrum with vertical lines marking the window.
    - Appends results to 'total_counts' and 'total_background'.
    """
    filename = Path(filename)
    filename_str = filename.name

    # Extract angle from filename: e.g. "co_-20_5min_65mm.Spe" -> -20
    match = re.search(r"co_(-?\d+)_", filename_str)
    if match:
        angle = int(match.group(1))
        angle_label = f"{angle}°"
    else:
        angle = None
        angle_label = "Unknown angle"

    bin_centers, N = data_reader(filename)

    # Masks for peak and background
    bin_min, bin_max = peak_range
    mask_peak = (bin_centers >= bin_min) & (bin_centers <= bin_max)
    mask_background = ~mask_peak

    N_peak = np.zeros_like(N)
    N_bkg = np.zeros_like(N)
    N_peak[mask_peak] = N[mask_peak]
    N_bkg[mask_background] = N[mask_background]

    counts = np.sum(N_peak)
    background = np.sum(N_bkg)

    print(
        f"The number of counts between {bin_min} and {bin_max} "
        f"for angle {angle_label} is: {counts}"
    )
    print()

    total_counts.append(counts)
    total_background.append(background)

    # Plot spectrum with window
    plt.figure()
    plt.plot(bin_centers, N, drawstyle="steps-mid")
    plt.xlabel("Bin Number")
    plt.ylabel("Counts")
    plt.title(f"Angle {angle_label}: Counts vs Bin Number")
    plt.axvline(
        x=bin_min,
        color="red",
        linestyle="--",
        label=f"Left window edge = {bin_min:.0f}",
    )
    plt.axvline(
        x=bin_max,
        color="red",
        linestyle="--",
        label=f"Right window edge = {bin_max:.0f}",
    )
    plt.legend()

    # Save per-angle spectrum with window
    angle_safe = str(angle) if angle is not None else "unknown"
    save_path = PEAK_PLOTS_DIR / f"coincidence_spectrum_angle_{angle_safe}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------------------------------------------------
# Gaussian rate model for counts vs angle
# -------------------------------------------------
def rate(params, x):
    """Gaussian model: counts vs angle."""
    A, x0, sigma = params
    return A * np.exp(- (x - x0) ** 2 / (2 * sigma ** 2))


def neg_loglike(params, x, N):
    """Poisson negative log-likelihood for Gaussian rate model."""
    lam = rate(params, x)
    # Avoid log(0) issues by clipping lam if desired (data should prevent 0 anyway)
    logp = scipy.stats.poisson.logpmf(N, lam).sum()
    return -logp


def gaussian_cf(x, A, x0, sigma):
    """Gaussian for curve_fit."""
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


# -------------------------------------------------
# Main script
# -------------------------------------------------
def main():
    # --------------------------------------------
    # 1. Integrate photopeak counts for each angle
    # --------------------------------------------
    peak_range = (175, 230)  # bin window for photopeak

    coincidence_files = [
        DATA_DIR / "co_-20_5min_65mm.Spe",
        DATA_DIR / "co_-15_5min_65mm.Spe",
        DATA_DIR / "co_-10_5min_65mm.Spe",
        DATA_DIR / "co_-7_5min_65mm.Spe",
        DATA_DIR / "co_-5_5min_65mm.Spe",
        DATA_DIR / "co_0_5min_65mm.Spe",
        DATA_DIR / "co_5_5min_65mm.Spe",
        DATA_DIR / "co_7_5min_65mm.Spe",
        DATA_DIR / "co_10_5min_65mm.Spe",
        DATA_DIR / "co_15_5min_65mm.Spe",
        DATA_DIR / "co_20_5min_65mm.Spe",
    ]

    for f in coincidence_files:
        if f.exists():
            histogram_na22(f, peak_range)
        else:
            print(f"Warning: {f} not found, skipping.")

    print(f"The counts from coincidence: {total_counts}")
    print(f"The respective uncertainty of coincidence counts: {np.sqrt(total_counts)}")
    print(f"The background counts: {total_background}")
    print()

    # Angles corresponding to each file:
    x = np.array([-20, -15, -10, -7, -5, 0, 5, 7, 10, 15, 20], dtype=float)
    counts_arr = np.array(total_counts, dtype=float)
    x_err = np.full_like(x, 0.5)

    # --------------------------------------------
    # 2. Plot counts vs angle (data only)
    # --------------------------------------------
    plt.figure()
    plt.errorbar(
        x,
        counts_arr,
        xerr=x_err,
        yerr=np.sqrt(counts_arr),
        fmt="o",
        color="blue",
        capsize=3,
        label="Data with uncertainties",
    )
    plt.ylabel("Counts")
    plt.xlabel("Detector Angle (°)")
    plt.title("Counts vs Detector Angle")

    # --------------------------------------------
    # 3. MLE Gaussian fit (Poisson)
    # --------------------------------------------
    p0 = [np.max(counts_arr), 0.0, 5.0]  # A, x0, sigma

    ret = scipy.optimize.minimize(
        neg_loglike,
        p0,
        args=(x, counts_arr),
        method="L-BFGS-B",
    )

    best_params = ret.x
    A_ml, x0_ml, sigma_ml = best_params

    def myfunc(p):
        return neg_loglike(p, x, counts_arr)

    hess = numdifftools.Hessian(myfunc)(best_params)
    cov_ml = scipy.linalg.inv(hess)
    best_params_unc = np.sqrt(np.diag(cov_ml))
    dA_ml, dx0_ml, dsigma_ml = best_params_unc

    print(f"The best MLE parameters [A, x0, sigma]: {best_params}")
    print(f"And their uncertainties: {best_params_unc}")
    print()
    print(f"A     = {A_ml:.4g} ± {dA_ml:.3g}")
    print(f"x0    = {x0_ml:.4g} ± {dx0_ml:.3g}")
    print(f"sigma = {sigma_ml:.4g} ± {dsigma_ml:.3g}")
    print()

    x_dense = np.linspace(-25, 25, 10000)
    y_dense_ml = rate(best_params, x_dense)
    plt.plot(x_dense, y_dense_ml, label="MLE Gaussian Fit", color="red")

    # Vertical line at x0 (MLE)
    plt.axvline(
        x=x0_ml,
        color="green",
        linestyle="--",
        label=f"x0 (MLE) = {x0_ml:.2f}°",
    )

    # --------------------------------------------
    # 4. Least-squares Gaussian (curve_fit)
    # --------------------------------------------
    popt_cf, pcov_cf = scipy.optimize.curve_fit(
        gaussian_cf,
        x,
        counts_arr,
        p0=p0,
    )
    A_cf, x0_cf, sigma_cf = popt_cf
    dA_cf, dx0_cf, dsigma_cf = np.sqrt(np.diag(pcov_cf))

    y_dense_cf = gaussian_cf(x_dense, A_cf, x0_cf, sigma_cf)
    # If you want to plot the curve_fit result too, uncomment:
    # plt.plot(x_dense, y_dense_cf, label="curve_fit Gaussian Fit", color="purple")
    # plt.axvline(x=x0_cf, color="magenta", linestyle="--",
    #             label=f"x0 (curve_fit) = {x0_cf:.2f}°")

    print("curve_fit parameters [A, x0, sigma]:", popt_cf)
    print("curve_fit uncertainties [dA, dx0, dsigma]:", [dA_cf, dx0_cf, dsigma_cf])
    print()

    eq_text = (
        rf"$R(\theta) = |{A_ml:.3f}|\exp\left[-0.5\frac{{(n{x0_ml:.3f})^2}}{{{sigma_ml:.3f}^2}}\right]$"
    )
    model_text = (
        r"$R(\theta) = |A|\exp\left[-0.5\frac{(n - x_{0})^2}{s^2}\right]$"
    )

    plt.text(
        -0.6,
        0.35,
        eq_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.text(
        -0.6,
        0.65,
        model_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.legend()
    save_path = MAIN_PLOTS_DIR / "Coincidence_vs_Angle.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # --------------------------------------------
    # 5. Chi-squared for each Gaussian model
    # --------------------------------------------
    y_model_ml = rate(best_params, x)
    y_model_cf = rate(popt_cf, x)

    chi2_ml = np.sum((counts_arr - y_model_ml) ** 2 / np.clip(y_model_ml, 1e-10, None))
    chi2_cf = np.sum((counts_arr - y_model_cf) ** 2 / np.clip(y_model_cf, 1e-10, None))
    dof = counts_arr.size - len(best_params)
    chi2_red_ml = chi2_ml / dof
    chi2_red_cf = chi2_cf / dof

    print(f"Chi-squared (MLE): {chi2_ml:.3f}")
    print(f"Reduced chi-squared (MLE): {chi2_red_ml:.3f} with {dof} dof")
    print()
    print(f"Chi-squared (curve_fit): {chi2_cf:.3f}")
    print(f"Reduced chi-squared (curve_fit): {chi2_red_cf:.3f} with {dof} dof")
    print()

    # --------------------------------------------
    # 6. Residuals for MLE Gaussian fit
    # --------------------------------------------
    plt.figure()
    plt.errorbar(
        x,
        counts_arr - y_model_ml,
        yerr=np.sqrt(counts_arr),
        fmt="o",
        color="blue",
        capsize=3,
        label="Residuals",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Residuals")
    plt.xlabel("Detector Angle (°)")
    plt.title("Residuals vs Detector Angle (MLE Gaussian)")
    plt.legend()
    save_path = MAIN_PLOTS_DIR / "coincidence_residuals.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # (If you want normalized residuals as well, you could add them here.)


if __name__ == "__main__":
    main()
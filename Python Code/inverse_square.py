#!/usr/bin/env python3
"""
Inverse square law analysis for 22Na intensity vs distance.

Folder structure assumed:

EP3 Gamma Submission/
├── Data/
│   ├── Day 3 Inverse square law/
│   │   ├── 22Na_3_160mm.Spe
│   │   ├── 22Na_3_200mm.Spe
│   │   ├── 22Na_3_250mm.Spe
│   │   ├── 22Na_3_300mm.Spe
│   │   ├── 22Na_3_350mm.Spe
│   │   ├── 22Na_3_400mm.Spe
│   │   └── 22Na_3_450mm.Spe
├── Plots/
│   └── Inverse Square Plots/
│       ├── Main Plots/
│       └── MLE distances/
└── Python Code/
    └── inverse_square.py   (this script)

Place this script in: EP3 Gamma Submission/Python Code/
"""

from pathlib import Path
import re

import numpy as np
import numdifftools
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import scipy.linalg
from scipy.optimize import curve_fit


# -------------------------------------------------
# Paths
# -------------------------------------------------
# inverse_square.py is in:  Submission/Python Code/
BASE_DIR = Path(__file__).resolve().parent          # .../Submission/Python Code
ROOT_DIR = BASE_DIR.parent                          # .../Submission

DATA_DIR = ROOT_DIR / "Data" / "Day 3 Inverse square law"
PLOTS_ROOT_DIR = ROOT_DIR / "Plots" / "Inverse Square Plots"
MAIN_PLOTS_DIR = PLOTS_ROOT_DIR / "Main Plots"
MLE_PLOTS_DIR = PLOTS_ROOT_DIR / "MLE distances"

MAIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MLE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def sci(x):
    """Return number in format a × 10^b (e.g. 3.611 × 10^7)."""
    if x == 0:
        return "0"
    b = int(np.floor(np.log10(abs(x))))
    a = x / 10**b
    return rf"{a:.3f} \times 10^{{{b}}}"


def data_reader(filename):
    """
    Reads an ASCII .Spe file and returns trimmed (E, N):
        E = channel numbers (float, bin centers)
        N = counts (int)

    It automatically trims leading and trailing zeros,
    and also trims the first and last non-zero count bins.
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

    # Further trim first and last non-zero bin
    E = E[1:-1]
    N = N[1:-1]

    return E, N


# -------------------------------------------------
# Global lists to accumulate MLE results
# -------------------------------------------------
counts = []          # integrated counts (intensity) at each distance
mean_bin = []        # MLE centroid A4 for each distance
mean_bin_unc = []    # uncertainty on A4 for each distance


# -------------------------------------------------
# Na-22: fit photopeak at a given distance and integrate counts
# -------------------------------------------------
def na_22(filename):
    """
    Fit the Na-22 511 keV photopeak with a linear background + Gaussian,
    using Poisson maximum likelihood, then integrate counts within ±2σ
    around the peak to get intensity.
    """
    bin_centers, bin_counts = data_reader(filename)

    # Extract distance in mm from filename string
    filename_str = str(filename)
    match = re.search(r'_(\d+)mm', filename_str)
    if match:
        distance_mm = float(match.group(1))
        distance_label = f"{distance_mm:.0f} mm"
    else:
        distance_mm = np.nan
        distance_label = "Unknown distance"

    def rate(params, E):
        A1, A2, A3, A4, A5 = params
        background = np.abs(A1 + A2 * (E - A4))
        peak_amp = np.abs(A3)
        gauss = peak_amp * np.exp(-0.5 * ((E - A4) ** 2) / (A5 ** 2))
        return background + gauss

    def neg_loglike(params, E, N):
        lam = rate(params, E)
        logp = scipy.stats.poisson.logpmf(N, lam).sum()
        return -logp

    # Initial guess for parameters
    p0 = [10, -1, max(bin_counts), 330, 10]

    ret = scipy.optimize.minimize(
        neg_loglike,
        p0,
        args=(bin_centers, bin_counts),
        method="L-BFGS-B"
    )

    print()
    print()
    print(distance_label)
    print(
        f"Negative log likelihood of MLE fit: "
        f"{neg_loglike(ret.x, bin_centers, bin_counts):.4f}"
    )

    parameters = ret.x

    # Hessian for parameter uncertainties
    def myfunc(p):
        return neg_loglike(p, bin_centers, bin_counts)

    cov = numdifftools.Hessian(myfunc)(parameters)
    cov_inv = scipy.linalg.inv(cov)
    A_unc = np.sqrt(np.diag(cov_inv))

    print(f"The parameters A1, A2, A3, A4, A5 are: {parameters}")
    print(f"And their corresponding uncertainties are: {A_unc}")

    A1_est, A2_est, A3_est, A4_est, A5_est = parameters
    A4_unc = A_unc[3]
    A5_unc = A_unc[4]

    mean_bin.append(A4_est)
    mean_bin_unc.append(A4_unc)

    # Define window: ±2σ around peak center
    window_mask = np.abs(bin_centers - A4_est) < 2 * np.abs(A5_est)

    # Sum all counts in that window
    total_counts = np.sum(bin_counts[window_mask])
    counts.append(total_counts)

    # Determine the bin edges actually used for counting
    E_window = bin_centers[window_mask]
    left_edge = E_window[0]
    right_edge = E_window[-1]

    # Plot spectrum, model, centroid, and integration window
    plt.figure()
    plt.plot(bin_centers, bin_counts, drawstyle="steps-mid", label="Data")
    plt.errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties"
    )

    xgrid = np.linspace(bin_centers[0], bin_centers[-1], 10000)
    plt.plot(xgrid, rate(parameters, xgrid), label="Model")

    # Centroid line
    plt.axvline(
        A4_est,
        linestyle="--",
        color="green",
        alpha=0.8,
        label=f"Peak (511 keV) centroid, A4 = {A4_est:.2f}"
    )

    # Integration-window edges (bins used for counting)
    plt.axvline(
        left_edge,
        linestyle=":",
        color="red",
        alpha=0.9,
        label=f"Integration start (n ≈ {left_edge:.1f})"
    )
    plt.axvline(
        right_edge,
        linestyle=":",
        color="red",
        alpha=0.9,
        label=f"Integration end (n ≈ {right_edge:.1f})"
    )

    plt.ylabel("Counts")
    plt.xlabel("Bin Number")
    plt.title(f"{distance_label}: Counts vs Bin Number")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    # Save to MLE distances folder
    safe_name = distance_label.replace(" ", "_")
    save_path = MLE_PLOTS_DIR / f"{safe_name}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return parameters, A4_unc


# -------------------------------------------------
# Inverse-square model
# -------------------------------------------------
def invsq_model(r, k, r0, b):
    """Inverse square law model: I(r) = k / (r - r0)^2 + b"""
    return k / ((r - r0) ** 2) + b


# -------------------------------------------------
# Main script
# -------------------------------------------------
def main():
    # --------------------------------------------
    # 1. Run MLE fits for all distances
    # --------------------------------------------
    inverse_files = [
        DATA_DIR / "22Na_3_160mm.Spe",
        DATA_DIR / "22Na_3_200mm.Spe",
        DATA_DIR / "22Na_3_250mm.Spe",
        DATA_DIR / "22Na_3_300mm.Spe",
        DATA_DIR / "22Na_3_350mm.Spe",
        DATA_DIR / "22Na_3_400mm.Spe",
        DATA_DIR / "22Na_3_450mm.Spe",
    ]

    for f in inverse_files:
        if f.exists():
            na_22(f)
        else:
            print(f"Warning: {f} not found, skipping.")

    # --------------------------------------------
    # 2. Intensity (counts) vs distance r
    # --------------------------------------------
    # Distances in mm corresponding to the files above
    r = np.array([160, 200, 250, 300, 350, 400, 450], dtype=float)  # mm
    I = np.array(counts, dtype=float)

    print("\n\nThe total counts measured for each distance:")
    print(I)

    # Uncertainties
    sigma_r = np.full_like(r, 0.5)   # constant ±0.5 mm
    sigma_I = np.sqrt(I)             # Poisson: ±sqrt(N)
    print("\n\nAnd their respective poisson uncertainties:")
    print(sigma_I)

    # --------------------------------------------
    # 3. Weighted least squares fit of I(r)
    #     I(r) = k / (r - r0)^2 + b
    # --------------------------------------------
    p0 = [1e7, 100, 150]  # initial guess

    # Bounds: k>0, r0 in a reasonable range, b>0
    lower_bounds = [1e5, -500, 0]
    upper_bounds = [1e9, 500, 1000]

    popt, pcov = curve_fit(
        invsq_model,
        r,
        I,
        p0=p0,
        sigma=sigma_I,       # weights ~ 1 / sigma_I^2
        absolute_sigma=True,  # pcov gives real parameter variances
        bounds=(lower_bounds, upper_bounds),
    )

    k_fit, r0_fit, b_fit = popt
    perr = np.sqrt(np.diag(pcov))   # uncertainties on k, r0, b

    print("\nBest-fit parameters (WLS):")
    print(f"k  = {k_fit:.6g} ± {perr[0]:.2g}")
    print(f"r0 = {r0_fit:.6g} ± {perr[1]:.2g}")
    print(f"b  = {b_fit:.6g} ± {perr[2]:.2g}")

    # Chi-squared and reduced chi-squared
    model_I = invsq_model(r, *popt)
    chi2 = np.sum(((I - model_I) / sigma_I) ** 2)
    dof = len(I) - len(popt)
    chi2_red = chi2 / dof

    # --------------------------------------------
    # 4. Plot I vs r with fitted model
    # --------------------------------------------
    x_dense = np.linspace(140, 500, 1000)
    y_dense = invsq_model(x_dense, *popt)

    plt.figure(figsize=(8, 5))
    plt.plot(x_dense, y_dense, label="WLS inverse-square fit")
    plt.errorbar(
        r,
        I,
        xerr=sigma_r,
        yerr=sigma_I,
        fmt="o",
        capsize=3,
        label="Data with uncertainties",
    )
    plt.xlabel("Distance r (mm)")
    plt.ylabel("Intensity (Counts)")
    plt.title("Intensity vs Distance r")
    plt.legend()

    ax = plt.gca()
    eq_text = (
        r"$I(r) = \dfrac{k}{(r - r_0)^2} + b$"
        "\n"
        "\n"
        "\n"
        rf"$I(r) = \dfrac{{{k_fit:.4g}}}{{(r - ({r0_fit:.4g}))^2}} + {b_fit:.4g}$"
    )

    ax.text(
        0.6,
        0.45,
        eq_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    save_path = MAIN_PLOTS_DIR / "Inverse_square_plot.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # --------------------------------------------
    # 5. Residuals
    # --------------------------------------------
    I_model = invsq_model(r, k_fit, r0_fit, b_fit)
    residuals = I_model - I

    # Non-normalized residuals
    plt.figure()
    plt.errorbar(
        r,
        residuals,
        yerr=sigma_I,
        fmt="o",
        capsize=3,
        label="Residuals with uncertainties",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Distance r (mm)")
    plt.ylabel("Residuals (non-normalised)")
    plt.title("Residuals of Inverse-Square Law Fit")
    save_path = MAIN_PLOTS_DIR / "Inverse_residuals.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # Normalized residuals
    normalized_residuals = (I_model - I) / sigma_I
    plt.figure()
    plt.scatter(r, normalized_residuals)
    plt.xlabel("Distance r (mm)")
    plt.ylabel("Residuals (sigma units)")
    plt.title("Normalized residuals of Inverse-Square Law Fit")
    plt.axhline(0, color="black", linewidth=1)
    save_path = MAIN_PLOTS_DIR / "Inverse_normalized_residuals.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # --------------------------------------------
    # 6. Photopeak centroids vs r
    # --------------------------------------------
    average_centroid = sum(mean_bin) / len(mean_bin)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        r,
        mean_bin,
        yerr=mean_bin_unc,
        fmt="o",
        color="blue",
        capsize=3,
        label="A4 with uncertainty",
    )
    plt.axhline(average_centroid, color="red", linestyle="--",
                label=f"Average centroid ≈ {average_centroid:.2f}")
    plt.title("Photopeak centroid, A4 vs r")
    plt.xlabel("r (mm)")
    plt.ylabel("A4 (bin number)")
    plt.legend()

    save_path = MAIN_PLOTS_DIR / "Average_centroid.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print()
    print(f"The average photopeak centroid value is: {average_centroid:.4f}")
    print()
    print("COUNTS VS r WITH INVERSE-SQUARE MODEL")
    print(f"Chi-squared: {chi2:.3f}")
    print(f"Reduced chi-squared: {chi2_red:.3f} (dof = {dof})")
    print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Gamma spectroscopy calibration script.

Folder structure assumed:

EP3 Gamma Submission/
├── Data/
│   ├── Day 1/
│   │   └── 22Na_1.Spe
│   ├── Day 2 Calibration/
│   │   ├── 137Cs_2_gain_8_r1_200-320.Spe
│   │   ├── 60Co_2_gain_8_r3_400-550.Spe
│   │   └── 22Na_2_gain_8_r1_150-250.Spe
│   ├── Day 3 Inverse square law/
│   └── Day 4 Coincidence/
├── Plots/
│   ├── Calibration Plots/
│   ├── Coincidence Plots/
│   └── Inverse Square Plots/
└── Python Code/
    ├── gamma_calibration.py
    └── ...

This script should be placed in: EP3 Gamma Submission/Python Code/gamma_calibration.py
"""

from pathlib import Path

import numpy as np
import numdifftools
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import scipy.linalg
from scipy.odr import ODR, Model, RealData


# -------------------------------------------------
# Paths
# -------------------------------------------------
# gamma_calibration.py is in:  Submission/Python Code/
BASE_DIR = Path(__file__).resolve().parent          # .../Submission/Python Code
ROOT_DIR = BASE_DIR.parent                          # .../Submission

DATA_DIR = ROOT_DIR / "Data"
PLOTS_ROOT_DIR = ROOT_DIR / "Plots"
CALIB_PLOT_DIR = PLOTS_ROOT_DIR / "Calibration Plots"

CALIB_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helper functions
# ----------------------------
def sci_fmt(x):
    """Return a value in LaTeX scientific notation a × 10^{b}."""
    if x == 0:
        return "0"
    b = int(np.floor(np.log10(abs(x))))
    a = x / 10**b
    return rf"{a:.3g}\times 10^{{{b}}}"


def data_reader(filename):
    """
    Reads an ASCII .Spe file and returns trimmed (E, N):
        E = channel numbers (float, bin centers)
        N = counts (int)

    It automatically trims leading and trailing zeros and also
    trims the first and last non-zero count bins.
    """
    filename = Path(filename)

    with filename.open("r") as f:
        lines = f.readlines()

    # --- Find the $DATA block ---
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            data_start = i + 1
            break
    else:
        raise ValueError(f"No $DATA section in .Spe file: {filename}")

    # --- Read channel range (e.g. "0 1023") ---
    low, high = map(int, lines[data_start].split())
    n_channels = high - low + 1

    # --- Read counts ---
    count_lines = lines[data_start + 1: data_start + 1 + n_channels]
    counts = np.array([int(x.strip()) for x in count_lines], dtype=int)

    # --- Trim zeros at start and end ---
    nonzero_indices = np.where(counts > 0)[0]
    if len(nonzero_indices) == 0:
        raise ValueError(f"All counts are zero — no spectrum found in {filename}.")

    first = nonzero_indices[0]
    last = nonzero_indices[-1]

    # --- Build trimmed outputs ---
    E = np.arange(low, high + 1, dtype=int)[first:last + 1] + 0.5
    N = counts[first:last + 1]

    # Further trim first and last non-zero bin
    E = E[1:-1]
    N = N[1:-1]

    return E, N


# ----------------------------
# Plot: Na-22 with Compton edge
# ----------------------------
def plot_Na22_Compton_and_save(filename):
    """Plots Na-22 spectrum with visible Compton edges and saves as PDF."""
    bin_centers, bin_counts = data_reader(filename)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(bin_centers, bin_counts, drawstyle="steps-mid")
    plt.title("$^{22}$Na Spectrum with Compton Edge: Counts vs Bin Number")
    plt.xlabel("Bin Number")
    plt.ylabel("Counts")

    # Hard-coded lines as in your original script
    plt.axvline(67,  linestyle='--',  color='red',    alpha=0.8,
            label="Compton Scattering Edge 1")
    plt.axvline(400, linestyle='-.',  color='k',      alpha=0.8,
                label="511 keV peak")
    plt.axvline(770, linestyle=':',   color='green',  alpha=0.8,
                label="Compton Scattering Edge 2")
    plt.axvline(960, linestyle='-',   color='purple', alpha=0.8,
                label="1275 keV peak")
    plt.legend()

    save_path = CALIB_PLOT_DIR / "Na22_Compton.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ----------------------------
# Cs-137 fit
# ----------------------------
def cs_137(filename):
    bin_centers, bin_counts = data_reader(filename)

    # Restrict fit region to the peak
    mask = (bin_centers > 220) & (bin_centers < 280)
    bin_centers = bin_centers[mask]
    bin_counts = bin_counts[mask]

    def rate(params, E):
        A1, A2, A3, A4, A5 = params

        background = np.abs(A1 + A2 * (E - A4))
        peak_amp = np.abs(A3)

        gauss = peak_amp * np.exp(-0.5 * ((E - A4) ** 2) / (A5 ** 2))
        return background + gauss

    def neg_loglike(params, E, N):
        logp = scipy.stats.poisson.logpmf(N, rate(params, E)).sum()
        return -logp

    ret = scipy.optimize.minimize(
        neg_loglike,
        [0, 10, 4000, 250, 10],
        args=(bin_centers, bin_counts),
        method="L-BFGS-B"
    )

    print("\nCs137")
    parameters = ret.x
    print(f"The parameters A1, A2, A3, A4, A5 are: {parameters}")

    plt.figure(figsize=(8.0, 5.5))
    plt.plot(bin_centers, bin_counts, drawstyle="steps-mid")
    plt.errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties",
    )

    xgrid = np.linspace(bin_centers[0], bin_centers[-1], 10000)
    plt.plot(xgrid, rate(parameters, xgrid), label="MLE Model")


    # Hessian for uncertainties
    def myfunc(p):
        return neg_loglike(p, bin_centers, bin_counts)

    cov = numdifftools.Hessian(myfunc)(parameters)
    errs = np.sqrt(np.diag(scipy.linalg.inv(cov)))
    print(f"And their corresponding uncertainties are: {errs}")

    # vertical line at peak mean
    plt.axvline(
        parameters[3],
        linestyle='--',
        color='red',
        alpha=0.8,
        label=f"Peak (0.662 MeV) centroid, A4 = {parameters[3]:.2f}±{errs[3]:.2f}",
    )

    plt.title("$^{137}$Cs Spectrum: Counts vs Bin Number")
    plt.ylabel('Counts')
    plt.xlabel('Bin Number')
    plt.legend(loc="upper left", fontsize=8)


    # Text with real coefficients
    A1, A2, A3, A4, A5 = parameters
    eq_text = (
        rf"$R(n) = |{A1:.3g} + {A2:.3g}(n-{A4:.5g})|$" "\n"
        rf"$\quad + |{A3:.3g}|\exp\left[-0.5\frac{{(n-{A4:.5g})^2}}{{{A5:.3g}^2}}\right]$"
    )
    model_text = (
        r"$R(n) = |A_{1} + A_{2}(n - A_{4})|$" "\n"
        r"$\quad + |A_{3}|\exp\left[-0.5\frac{(n - A_{4})^2}{A_{5}^2}\right]$"
    )

    plt.text(
        -0.55,
        0.35,
        eq_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.text(
        -0.55,
        0.65,
        model_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    save_path = CALIB_PLOT_DIR / "Cs137_calibration.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    

    A4_est = parameters[3]
    A5_est = parameters[4]
    A4_unc = errs[3]
    A5_unc = errs[4]


    # Chi-squared
    model_counts = rate(parameters, bin_centers)
    sigma = np.sqrt(bin_counts)
    mask = sigma > 0
    chi2 = np.sum(((bin_counts[mask] - model_counts[mask]) ** 2) / (sigma[mask] ** 2))
    dof = np.count_nonzero(mask) - len(parameters)
    chi2_red = chi2 / dof

    print(f"Chi-squared: {chi2}")
    print(f"Reduced chi-squared: {chi2_red} (dof = {dof})")
    print(f"Negative log-likelihood at MLE: {neg_loglike(parameters, bin_centers, bin_counts)}")
    


    # Residuals
    residuals = (bin_counts - model_counts) / sigma
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(bin_centers, residuals, ".", markersize=4)
    plt.xlabel("Bin number")
    plt.ylabel("Residuals (sigma units)")
    plt.title("$^{137}$Cs Spectrum: Normalized residuals")
    save_path = CALIB_PLOT_DIR / "Cs137_calibration_residuals.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # Combined figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: spectrum
    axs[0].plot(bin_centers, bin_counts, drawstyle="steps-mid")
    axs[0].errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties",
    )
    axs[0].plot(xgrid, rate(parameters, xgrid), label="MLE Model")
    axs[0].axvline(
        parameters[3],
        linestyle='--',
        color='red',
        alpha=0.8,
        label=f"Peak (0.662 MeV) centroid, A4 = {parameters[3]:.2f}±{errs[3]:.2f}",
    )
    axs[0].set_title("$^{137}$Cs Spectrum: Counts vs Bin Number")
    axs[0].set_ylabel('Counts')
    axs[0].set_xlabel('Bin Number')
    axs[0].legend(
        loc="center right",
        bbox_to_anchor=(-0.25, 0.8),
        fontsize=8,
    )
    axs[0].text(
        -1,
        0.3,
        eq_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    axs[0].text(
        -1,
        0.6,
        model_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    # Right: residuals
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].plot(bin_centers, residuals, ".", markersize=4)
    axs[1].set_xlabel("Bin number")
    axs[1].set_ylabel("Residuals (sigma units)")
    axs[1].set_title("$^{137}$Cs Spectrum: Normalized residuals")

    save_path = CALIB_PLOT_DIR / "Cs137_calibration_full.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return [A4_est, A5_est], [A4_unc, A5_unc]


# ----------------------------
# Co-60 fit
# ----------------------------
def co_60(filename):
    bin_centers, bin_counts = data_reader(filename)

    # Restrict fit region to the peak
    mask = (bin_centers > 410) & (bin_centers < 530)
    bin_centers = bin_centers[mask]
    bin_counts = bin_counts[mask]

    def rate(params, E):
        A1, A2, A3, A4, A5, A6, A7, A8 = params

        background = np.abs(A1 + A2 * (E - A4))
        peak_amp_1 = np.abs(A3)
        peak_amp_2 = np.abs(A6)

        gauss_1 = peak_amp_1 * np.exp(-0.5 * ((E - A4) ** 2) / (A5 ** 2))
        gauss_2 = peak_amp_2 * np.exp(-0.5 * ((E - A7) ** 2) / (A8 ** 2))
        return background + gauss_1 + gauss_2

    def neg_loglike(params, E, N):
        logp = scipy.stats.poisson.logpmf(N, rate(params, E)).sum()
        return -logp

    ret = scipy.optimize.minimize(
        neg_loglike,
        [100, -1, 200, 440, 10, 200, 500, 20],
        bounds=[
            (0, None),     # A1
            (None, None),  # A2
            (0, None),     # A3
            (430, 450),    # A4 (mu1)
            (0.5, 20),     # A5 (sigma1)
            (0, None),     # A6
            (490, 520),    # A7 (mu2)
            (0.5, 20),     # A8 (sigma2)
        ],
        args=(bin_centers, bin_counts),
        method="L-BFGS-B",
    )

    print("Co60")
    parameters = ret.x
    print("The parameters A1, A2, A3, A4, A5, A6, A7, A8 are:")
    print(parameters)

    # Hessian for uncertainties
    def myfunc(p):
        return neg_loglike(p, bin_centers, bin_counts)

    cov = numdifftools.Hessian(myfunc)(parameters)
    errs = np.sqrt(np.diag(scipy.linalg.inv(cov)))
    print("And their corresponding uncertainties are:")
    print(errs)

    plt.figure(figsize=(8, 5.5))
    plt.plot(bin_centers, bin_counts, drawstyle="steps-mid")
    plt.errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties",
    )

    xgrid = np.linspace(bin_centers[0], bin_centers[-1], 10000)
    plt.plot(xgrid, rate(parameters, xgrid), label="MLE Model")

    A1, A2, A3, A4, A5, A6, A7, A8 = parameters
    eq_text = (
        rf"$R(n)= |{A1:.3g} + {A2:.3g}(n-{A4:.5g})|$" "\n"
        rf"$\quad + |{A3:.3g}|\exp\left[-0.5\frac{{(n-{A4:.5g})^2}}{{{A5:.3g}^2}}\right]$" "\n"
        rf"$\quad + |{A6:.3g}|\exp\left[-0.5\frac{{(n-{A7:.5g})^2}}{{{A8:.3g}^2}}\right]$"
    )
    model_text = (
        r"$R(n) = |A_{1} + A_{2}(n - A_{4})|$" "\n"
        r"$\quad + |A_{3}|\exp\left[-0.5\frac{(n - A_{4})^2}{A_{5}^2}\right]$" "\n"
        r"$\quad + |A_{6}|\exp\left[-0.5\frac{(n - A_{7})^2}{A_{8}^2}\right]$"
    )

    plt.text(
        -0.5,
        0.35,
        eq_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.text(
        -0.5,
        0.65,
        model_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    # Vertical lines at both peak means
    plt.axvline(
        parameters[3],
        linestyle=':',
        color='red',
        alpha=0.8,
        label=f"Peak 1 (1.17 MeV) centroid, A4 = {parameters[3]:.1f}±{errs[3]:.1f}",
    )
    plt.axvline(
        parameters[6],
        linestyle='--',
        color='purple',
        alpha=0.8,
        label=f"Peak 2 (1.33 MeV) centroid, A7 = {parameters[6]:.1f}±{errs[6]:.1f}",
    )
    plt.title("$^{60}$Co Spectrum: Counts vs Bin Number")
    plt.ylabel('Counts')
    plt.xlabel('Bin Number')
    plt.legend()

    save_path = CALIB_PLOT_DIR / "Co60_calibration.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


    A4_est, A5_est = parameters[3], parameters[4]
    A7_est, A8_est = parameters[6], parameters[7]
    A4_unc, A5_unc = errs[3], errs[4]
    A7_unc, A8_unc = errs[6], errs[7]

    # Chi-squared
    model_counts = rate(parameters, bin_centers)
    sigma = np.sqrt(bin_counts)
    mask = sigma > 0
    chi2 = np.sum(((bin_counts[mask] - model_counts[mask]) ** 2) / (sigma[mask] ** 2))
    dof = np.count_nonzero(mask) - len(parameters)
    chi2_red = chi2 / dof

    print(f"Chi-squared: {chi2}")
    print(f"Reduced chi-squared: {chi2_red} (dof = {dof})")
    print(f"Negative log-likelihood at MLE: {neg_loglike(parameters, bin_centers, bin_counts)}")

    # Residuals
    residuals = (bin_counts - model_counts) / sigma
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(bin_centers, residuals, ".", markersize=4)
    plt.xlabel("Bin number")
    plt.ylabel("Residuals (sigma units)")
    plt.title("$^{60}$Co Spectrum: Normalized residuals")
    save_path = CALIB_PLOT_DIR / "Co60_calibration_residuals.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # Combined figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(bin_centers, bin_counts, drawstyle="steps-mid")
    axs[0].errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties",
    )
    axs[0].plot(xgrid, rate(parameters, xgrid), label="MLE Model")
    axs[0].axvline(
        parameters[3],
        linestyle=':',
        color='red',
        alpha=0.8,
        label=f"Peak 1 (1.17 MeV) centroid, A4 = {parameters[3]:.1f}±{errs[3]:.1f}",
    )
    axs[0].axvline(
        parameters[6],
        linestyle='--',
        color='purple',
        alpha=0.8,
        label=f"Peak 2 (1.33 MeV) centroid, A7 = {parameters[6]:.1f}±{errs[6]:.1f}",
    )
    axs[0].set_title("$^{60}$Co Spectrum: Counts vs Bin Number")
    axs[0].set_ylabel('Counts')
    axs[0].set_xlabel('Bin Number')
    axs[0].legend(
        loc="center right",
        bbox_to_anchor=(-0.25, 0.8),
        fontsize=8,
    )
    axs[0].text(
        -1,
        0.2,
        eq_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    axs[0].text(
        -1,
        0.6,
        model_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].plot(bin_centers, residuals, ".", markersize=4)
    axs[1].set_xlabel("Bin number")
    axs[1].set_ylabel("Residuals (sigma units)")
    axs[1].set_title("$^{60}$Co Spectrum: Normalized residuals")

    save_path = CALIB_PLOT_DIR / "Co60_calibration_full.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return [A4_est, A5_est, A7_est, A8_est], [A4_unc, A5_unc, A7_unc, A8_unc]


# ----------------------------
# Na-22 fit
# ----------------------------
def na_22(filename):
    bin_centers, bin_counts = data_reader(filename)

    # Restrict fit region to the peak
    mask = (bin_centers > 170) & (bin_centers < 220)
    bin_centers = bin_centers[mask]
    bin_counts = bin_counts[mask]

    def rate(params, E):
        A1, A2, A3, A4, A5 = params

        background = np.abs(A1 + A2 * (E - A4))
        peak_amp = np.abs(A3)
        gauss = peak_amp * np.exp(-0.5 * ((E - A4) ** 2) / (A5 ** 2))
        return background + gauss

    def neg_loglike(params, E, N):
        logp = scipy.stats.poisson.logpmf(N, rate(params, E)).sum()
        return -logp

    ret = scipy.optimize.minimize(
        neg_loglike,
        [100, -1, 1400, 200, 10],
        args=(bin_centers, bin_counts),
        method="L-BFGS-B",
    )

    print("Na22")
    parameters = ret.x
    print(f"The parameters A1, A2, A3, A4, A5 are: {parameters}")

    # Hessian for uncertainties
    def myfunc(p):
        return neg_loglike(p, bin_centers, bin_counts)

    cov = numdifftools.Hessian(myfunc)(parameters)
    errs = np.sqrt(np.diag(scipy.linalg.inv(cov)))
    print(f"And their corresponding uncertainties are: {errs}")

    plt.figure(figsize=(8.0, 5.5))
    plt.plot(bin_centers, bin_counts, drawstyle="steps-mid")
    plt.errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties",
    )

    xgrid = np.linspace(bin_centers[0], bin_centers[-1], 10000)
    plt.plot(xgrid, rate(parameters, xgrid), label="MLE Model")

    A1, A2, A3, A4, A5 = parameters
    eq_text = (
        rf"$R(n) = |{A1:.3g} + {A2:.3g}(n-{A4:.5g})|$" "\n"
        rf"$\quad + |{A3:.3g}|\exp\left[-0.5\frac{{(n-{A4:.5g})^2}}{{{A5:.3g}^2}}\right]$"
    )
    model_text = (
        r"$R(n) = |A_{1} + A_{2}(n - A_{4})|$" "\n"
        r"$\quad + |A_{3}|\exp\left[-0.5\frac{(n - A_{4})^2}{A_{5}^2}\right]$"
    )

    plt.text(
        -0.5,
        0.35,
        eq_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.text(
        -0.5,
        0.65,
        model_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.axvline(
        parameters[3],
        linestyle='--',
        color='red',
        alpha=0.8,
        label=f"Peak (0.511 MeV) centroid, A4 = {parameters[3]:.2f}±{errs[3]:.2f}",
    )
    plt.title("$^{22}$Na Spectrum: Counts vs Bin Number")
    plt.ylabel('Counts')
    plt.xlabel('Bin Number')
    plt.legend(loc="upper left", fontsize=8)

    save_path = CALIB_PLOT_DIR / "Na22_calibration.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    

    A4_est = parameters[3]
    A5_est = parameters[4]
    A4_unc = errs[3]
    A5_unc = errs[4]

    # Chi-squared
    model_counts = rate(parameters, bin_centers)
    sigma = np.sqrt(bin_counts)
    mask = sigma > 0
    chi2 = np.sum(((bin_counts[mask] - model_counts[mask]) ** 2) / (sigma[mask] ** 2))
    dof = np.count_nonzero(mask) - len(parameters)
    chi2_red = chi2 / dof

    print(f"Chi-squared: {chi2}")
    print(f"Reduced chi-squared: {chi2_red} (dof = {dof})")
    print(f"Negative log-likelihood at MLE: {neg_loglike(parameters, bin_centers, bin_counts)}")

    # Residuals
    residuals = (bin_counts - model_counts) / sigma
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(bin_centers, residuals, ".", markersize=4)
    plt.xlabel("Bin number")
    plt.ylabel("Residuals (sigma units)")
    plt.title("$^{22}$Na Spectrum: Normalized residuals")
    save_path = CALIB_PLOT_DIR / "Na22_calibration_residuals.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # Combined figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(bin_centers, bin_counts, drawstyle="steps-mid")
    axs[0].errorbar(
        bin_centers,
        bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=2,
        alpha=0.6,
        label="Poisson uncertainties",
    )
    axs[0].plot(xgrid, rate(parameters, xgrid), label="MLE Model")
    axs[0].axvline(
        parameters[3],
        linestyle='--',
        color='red',
        alpha=0.8,
        label=f"Peak (0.511 MeV) centroid, A4 = {parameters[3]:.2f}±{errs[3]:.2f}",
    )
    axs[0].set_title("$^{22}$Na Spectrum: Counts vs Bin Number")
    axs[0].set_ylabel('Counts')
    axs[0].set_xlabel('Bin Number')
    axs[0].legend(
        loc="center right",
        bbox_to_anchor=(-0.25, 0.8),
        fontsize=8,
    )
    axs[0].text(
        -1,
        0.6,
        model_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    axs[0].text(
        -1,
        0.3,
        eq_text,
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].plot(bin_centers, residuals, ".", markersize=4)
    axs[1].set_xlabel("Bin number")
    axs[1].set_ylabel("Residuals (sigma units)")
    axs[1].set_title("$^{22}$Na Spectrum: Normalized residuals")

    save_path = CALIB_PLOT_DIR / "Na22_calibration_full.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return [A4_est, A5_est], [A4_unc, A5_unc]


# ----------------------------
# Monte Carlo for Na-22 energy uncertainty
# ----------------------------
def mc_energy(a, b, cov_ab, n_meas, n_sigma, N=50000):
    """
    Draws Monte Carlo samples of a, b (from covariance) and n (from its
    measurement uncertainty) and returns a distribution of E = a + b*n.
    """
    mean = np.array([a, b])
    ab_samples = np.random.multivariate_normal(mean, cov_ab, size=N)
    a_samples = ab_samples[:, 0]
    b_samples = ab_samples[:, 1]

    n_samples = np.random.normal(n_meas, n_sigma, size=N)
    E_samples = a_samples + b_samples * n_samples

    return E_samples, E_samples.mean(), E_samples.std(ddof=1)


# ----------------------------
# Main analysis
# ----------------------------
def main():
    # Files in: Submission/Data/...
    na22_compton_file = DATA_DIR / "Day 1" / "22Na_1.Spe"
    cs137_file        = DATA_DIR / "Day 2 Calibration" / "137Cs_2_gain_8_r1_200-320.Spe"
    co60_file         = DATA_DIR / "Day 2 Calibration" / "60Co_2_gain_8_r3_400-550.Spe"
    na22_file         = DATA_DIR / "Day 2 Calibration" / "22Na_2_gain_8_r1_150-250.Spe"

    # 1) Na-22 Compton overview plot
    if na22_compton_file.exists():
        plot_Na22_Compton_and_save(na22_compton_file)
    else:
        print(f"Warning: {na22_compton_file} not found, skipping Compton plot.")

    # 2) Calibration peaks: energies and bin centroids
    energies = np.array([661.657, 1173.2, 1332.5])  # keV
    n = np.zeros(3)
    n_unc = np.zeros(3)

    # variables for resolution calculation
    energies_res = np.array([511, 661.657, 1173.2, 1332.5])
    A5s = np.zeros(len(energies_res))
    A5s_unc = np.zeros(len(energies_res))

    # Cs-137
    res = cs_137(cs137_file)
    bestval, bestval_err = res
    print(f"The best value of A4 for Cs-137: {bestval[0]}")
    print(f"And its corresponding uncertainty: {bestval_err[0]}")
    n[0] = bestval[0]
    n_unc[0] = bestval_err[0]
    A5s[1] = bestval[1]
    A5s_unc[1] = bestval_err[1]
    print()

    # Co-60
    res = co_60(co60_file)
    bestval, bestval_err = res
    print(f"The best values of A4, A7 for Co-60: {bestval[0]}, {bestval[2]}")
    print(f"And their corresponding uncertainties: {bestval_err[0]}, {bestval_err[2]}")
    n[1] = bestval[0]
    n_unc[1] = bestval_err[0]
    n[2] = bestval[2]
    n_unc[2] = bestval_err[2]
    A5s[2] = bestval[1]
    A5s_unc[2] = bestval_err[1]
    A5s[3] = bestval[3]
    A5s_unc[3] = bestval_err[3]
    print()

    # Na-22 peak centroid (for energy inference)
    res = na_22(na22_file)
    bestval, bestval_err = res
    print(f"The best value of A4 for Na-22: {bestval[0]}")
    print(f"And its corresponding uncertainty: {bestval_err[0]}")
    na_22_mean_bin = bestval[0]
    na_22_mean_bin_error = bestval_err[0]
    A5s[0] = bestval[1]
    A5s_unc[0] = bestval_err[1]
    print()
    print(f"The A5s and A5s_unc: {A5s}\n{A5s_unc}")

    # Simple least-squares calibration (E = a + b*n)
    a_ls, b_ls = np.polyfit(n, energies, 1)
    na_22_energy_ls = a_ls * na_22_mean_bin + b_ls
    print(f"Na-22 photopeak energy (simple LS) ≈ {na_22_energy_ls:.2f} keV")
    print()

    # resolution = (2.335*A5_est)/(0.662)
    # print(f"The detector resolution at 0.662 MeV: {(:.2f}")

    # Orthogonal Distance Regression (ODR)
    print("Now doing Orthogonal Distance Regression (ODR) instead:")

    sigma_n = n_unc
    sigma_E = np.full_like(energies, 0.1)  # ~0 keV uncertainty on known energies

    def linear_model(beta, n_vals):
        a, b = beta
        return a + b * n_vals

    model = Model(linear_model)
    data = RealData(n, energies, sx=sigma_n, sy=sigma_E)
    beta0 = [0.0, 1.0]
    odr = ODR(data, model, beta0=beta0)
    output = odr.run()

    a, b = output.beta
    a_err, b_err = np.sqrt(np.diag(output.cov_beta))

    n_dense = np.linspace(min(n), max(n), 1000)
    y_dense = b * n_dense + a

    na_22_energy = b * na_22_mean_bin + a

    # Plot calibration line
    plt.errorbar(
        n,
        energies,
        xerr=n_unc,
        fmt="o",
        label="Data with uncertainties",
        capsize=3,
    )
    plt.plot(
        n_dense,
        y_dense,
        label=f"Linear Model, E = ({a:.4f}) * n + {b:.4f}",
    )
    plt.title("Energy vs Bin Number")
    plt.xlabel("Bin Number")
    plt.ylabel("Energy (keV)")
    plt.legend()
    save_path = CALIB_PLOT_DIR / "Final_calibration.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print("\n--- Calibration results (E = a + b*n, ODR) ---")
    print(f"a = {a:.5f} ± {a_err:.5f}  keV")
    print(f"b = {b:.5f} ± {b_err:.5f}  keV/channel")

    # Monte Carlo uncertainty on Na-22 energy
    cov_ab_scaled = output.cov_beta * output.res_var
    E_samples, E_mean, E_std = mc_energy(
        a,
        b,
        cov_ab_scaled,
        n_meas=na_22_mean_bin,
        n_sigma=na_22_mean_bin_error,
        N=50000,
    )

    #resolutions

    resolutions = (2.355*(b*A5s))/energies_res
    resolutions_unc = resolutions*(A5s_unc/A5s)

    print()
    print(f"Na-22 energy = {E_mean:.2f} ± {E_std:.2f} keV (1σ)")
    print("This was found using ODR and Monte Carlo.")
    print()
    for i,res in enumerate(resolutions):
        print(f"Detector resolution for {energies_res[i]} keV is: {res:.4f} ± {resolutions_unc[i]:.1g}")


if __name__ == "__main__":
    main()
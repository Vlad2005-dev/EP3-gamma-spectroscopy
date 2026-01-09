# Gamma Spectroscopy Lab – Code & Data

This repository contains the analysis code and data used for the gamma spectroscopy lab, including:

- Energy calibration using known gamma-ray sources
- Inverse-square law for intensity vs distance
- Angular dependence of coincidence counts

The code is written so that it can be run on any machine, as long as the folder structure and file names are preserved.

## TL;DR

Physics data analysis project using Python to analyze gamma spectroscopy data.

Key techniques:
- Poisson maximum likelihood estimation for spectral peak fitting
- Energy calibration with orthogonal distance regression (ODR)
- Inverse-square law validation
- Coincidence angular distribution analysis

See `Gamma_Spectroscopy_Lab_Report-4.pdf` for the full scientific report.

## Folder Structure

The expected directory layout is:

EP3 Gamma Submission/
├── Data/
│   ├── Day 1/
│   │   └── 22Na_1.Spe
│   ├── Day 2 Calibration/
│   │   ├── 137Cs_2_gain_8_r1_200-320.Spe
│   │   ├── 60Co_2_gain_8_r3_400-550.Spe
│   │   └── 22Na_2_gain_8_r1_150-250.Spe
│   ├── Day 3 Inverse square law/
│   │   ├── 22Na_3_160mm.Spe
│   │   ├── 22Na_3_200mm.Spe
│   │   ├── 22Na_3_250mm.Spe
│   │   ├── 22Na_3_300mm.Spe
│   │   ├── 22Na_3_350mm.Spe
│   │   ├── 22Na_3_400mm.Spe
│   │   └── 22Na_3_450mm.Spe
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
│
├── Plots/
│   ├── Calibration Plots/
│   │   ├── Cs137_calibration.pdf
│   │   ├── Cs137_calibration_residuals.pdf
│   │   ├── Cs137_calibration_full.pdf
│   │   ├── Co60_calibration.pdf
│   │   ├── Co60_calibration_residuals.pdf
│   │   ├── Co60_calibration_full.pdf
│   │   ├── Na22_calibration.pdf
│   │   ├── Na22_calibration_residuals.pdf
│   │   ├── Na22_calibration_full.pdf
│   │   ├── Na22_Compton.pdf
│   │   └── Final_calibration.pdf
│   │
│   ├── Inverse Square Plots/
│   │   ├── Main Plots/
│   │   │   ├── Inverse_square_plot.pdf
│   │   │   ├── Inverse_residuals.pdf
│   │   │   ├── Inverse_normalized_residuals.pdf
│   │   │   └── Average_centroid.pdf
│   │   └── MLE distances/
│   │       └── ... (per-distance spectra with MLE fits)
│   │
│   └── Coincidence Plots/
│       ├── Main Plots/
│       │   ├── Coincidence_vs_Angle.pdf
│       │   └── coincidence_residuals.pdf
│       └── Photopeak Windows/
│           └── ... (per-angle spectra with photopeak window)
│
└── Python Code/
    ├── gamma_calibration.py
    ├── inverse_square.py
    └── coincidence_angles.py



If any of the .Spe files are missing, the scripts will print a warning and skip those parts.



## Dependencies

The code requires:
	•	Python 3 (3.8+ recommended)
	•	The following Python packages:

    pip install numpy scipy matplotlib numdifftools

pathlib and os are part of the Python standard library.



## How to Run the Scripts 

All scripts should be run from inside the Python Code/ directory.

## Scripts description 

1. Energy Calibration (gamma_calibration.py)

    Script: Python Code/gamma_calibration.py
    Purpose:
        •	Read in spectra from Cs-137, Co-60, and Na-22 (Day 2 Calibration)
        •	Fit photopeaks using Poisson maximum likelihood (Gaussian + background)
        •	Compute peak centroids and uncertainties
        •	Perform linear energy calibration ( E = a + b n ) using ODR
        •	Use Monte Carlo to estimate the uncertainty on the Na-22 511 keV energy

    Run:
        cd Python Code
        python gamma_calibration.py

    Input data used:
    •	../Data/Day 2 Calibration/137Cs_2_gain_8_r1_200-320.Spe
    •	../Data/Day 2 Calibration/60Co_2_gain_8_r3_400-550.Spe
    •	../Data/Day 2 Calibration/22Na_2_gain_8_r1_150-250.Spe
    •	../Data/Day 1/22Na_1.Spe (for the Compton edge overview plot)

    Output:
        •	Calibration plots and residuals saved as PDFs in
        ../Plots/Calibration Plots/


2. Inverse Square Law (inverse_square.py)

    Script: Python Code/inverse_square.py
    Purpose:
        •	For each distance (Day 3 Inverse square law), fit the Na-22 511 keV photopeak with Poisson MLE
        •	Integrate counts within ±2σ around the peak to get intensity ( I(r) )
        •	Fit an inverse-square model
    
    I(r) = (k/([r - r_0]^2)) + b
    
    using weighted least squares
        •	Compute χ² and reduced χ² for the fit
        •	Check that the photopeak centroid is stable vs distance

    Run:
        cd Python Code
        python inverse_square.py

    Input data used:
	•	../Data/Day 3 Inverse square law/22Na_3_160mm.Spe
	•	../Data/Day 3 Inverse square law/22Na_3_200mm.Spe
	•	../Data/Day 3 Inverse square law/22Na_3_250mm.Spe
	•	../Data/Day 3 Inverse square law/22Na_3_300mm.Spe
	•	../Data/Day 3 Inverse square law/22Na_3_350mm.Spe
	•	../Data/Day 3 Inverse square law/22Na_3_400mm.Spe
	•	../Data/Day 3 Inverse square law/22Na_3_450mm.Spe

    Output:
	•	Per-distance MLE peak fits and windows in
    ../Plots/Inverse Square Plots/MLE distances/      
    •	Inverse-square fit and residuals in
    ../Plots/Inverse Square Plots/Main Plots/


3. Angular Coincidence Dependence (coincidence_angles.py)

    Script: Python Code/coincidence_angles.py
    Purpose:
        •	For each detector angle (Day 4 Coincidence), integrate counts in a chosen photopeak window (e.g. bins 175–230)
        •	Plot coincidence counts vs detector angle
        •	Fit a Gaussian model to the angular distribution using:
        •	Poisson MLE
        •	Least squares (curve_fit)
        •	Compute χ² and reduced χ² for both fits
        •	Plot residuals for the MLE Gaussian model

    Run:
        cd Python Code
        python coincidence_angles.py

    Input data used:
	•	../Data/Day 4 Coincidence/co_-20_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_-15_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_-10_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_-7_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_-5_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_0_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_5_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_7_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_10_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_15_5min_65mm.Spe
	•	../Data/Day 4 Coincidence/co_20_5min_65mm.Spe

    Output:
    •	Per-angle spectra with the integration window marked in
    ../Plots/Coincidence Plots/Photopeak Windows/      
    •	Gaussian fit to counts vs angle, and residuals, in
    ../Plots/Coincidence Plots/Main Plots/


## Notes
	•	All paths in the scripts are relative to the script locations; no machine-specific absolute paths are used.
	•	If you rename data files or change the folder structure, you will need to update the corresponding paths in the Python scripts.
	•	The statistical methods (MLE, Hessian-based uncertainties, ODR, Monte Carlo, χ²) are documented in the lab report; this code is provided as supporting material.


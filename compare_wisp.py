"""
Compare stacked images to WISP data.

Example:
    %run compare_wisp.py data/20200921_LHBP1/stacks/iSPEX_grey_iso1840_mean.npy data/20200921_LHBP1/wisp_Olivier_20201126.csv
"""

import numpy as np
from sys import argv
from spectacle import plot, io, wavelength, raw, general, calibrate
from ispex import general as ispex_general, plot as ispex_plot, validation
from matplotlib import pyplot as plt
from pathlib import Path

# Get the filenames from the command line
filename_ispex, filename_wisp = io.path_from_input(argv)

# For iSPEX, get the grey card, sky, and water filenames
# For now, assume the given filename was for the grey card
label_dataset = filename_ispex.parents[1].stem
parent_ispex = filename_ispex.parent
filename_grey = filename_ispex
filename_sky = parent_ispex / (filename_grey.stem.replace("grey", "sky") + filename_grey.suffix)
filename_water = parent_ispex / (filename_grey.stem.replace("grey", "water") + filename_grey.suffix)
filenames_ispex = [filename_grey, filename_sky, filename_water]

# Get the std filenames from the mean filenames
filenames_ispex_std = [parent_ispex / (filename.stem.replace("mean", "stds") + filename.suffix) for filename in filenames_ispex]

# Hard-coded calibration for now
root = Path(r"C:\Users\Burggraaff\SPECTACLE_data\iPhone_SE")

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Load image stacks
mean_grey, mean_sky, mean_water = [np.load(filename) for filename in filenames_ispex]
stds_grey, stds_sky, stds_water = [np.load(filename) for filename in filenames_ispex_std]

# Bias correction
mean_grey, mean_sky, mean_water = camera.correct_bias([mean_grey, mean_sky, mean_water])

# Flat-field correction
mean_grey, mean_sky, mean_water = camera.correct_flatfield([mean_grey, mean_sky, mean_water])

# Slice the data
slice_Qp, slice_Qm = ispex_general.find_spectrum(mean_grey)
top, middle, bottom = ispex_general.find_background(mean_grey)

# Show the bounding boxes for visualisation
ispex_plot.plot_bounding_boxes_Rrs(mean_grey, mean_sky, mean_water, label_file=filename_grey, saveto=Path("results")/f"{label_dataset}_bounding_boxes.pdf", vmax=500)

mean_grey_Qp, mean_sky_Qp, mean_water_Qp = mean_grey[slice_Qp], mean_sky[slice_Qp], mean_water[slice_Qp]
mean_grey_Qm, mean_sky_Qm, mean_water_Qm = mean_grey[slice_Qm], mean_sky[slice_Qm], mean_water[slice_Qm]

bayer_Qp, bayer_Qm = camera.bayer_map[slice_Qp], camera.bayer_map[slice_Qm]

x = np.arange(mean_grey_Qp.shape[1])
xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T
xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T
yp = np.arange(mean_grey_Qp.shape[0])
ym = np.arange(mean_grey_Qm.shape[0])

# Load the wavelength calibration
coefficients_Qp = np.load(Path("calibration_data")/"wavelength_calibration_Qp.npy")
coefficients_Qm = np.load(Path("calibration_data")/"wavelength_calibration_Qm.npy")

wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

# Demosaick the data and wavelength calibration data
wavelengths_split_Qp, mean_grey_Qp_RGBG, mean_sky_Qp_RGBG, mean_water_Qp_RGBG, xp_split = raw.demosaick(bayer_Qp, [wavelengths_Qp, mean_grey_Qp, mean_sky_Qp, mean_water_Qp, xp])
wavelengths_split_Qm, mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG, xm_split = raw.demosaick(bayer_Qm, [wavelengths_Qm, mean_grey_Qm, mean_sky_Qm, mean_water_Qm, xm])

# Smooth the data in RGBG space
# 1 pixel in RGBG space = 2 pixels in RGB space
mean_grey_Qp_RGBG, mean_sky_Qp_RGBG, mean_water_Qp_RGBG, mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG = [general.gauss_filter_multidimensional(d, sigma=(0,0,3)) for d in [mean_grey_Qp_RGBG, mean_sky_Qp_RGBG, mean_water_Qp_RGBG, mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG]]

# Plot the Qm data in a single row
row = 50
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG], ["Grey card", "Sky", "Water"]):
    for j, c in enumerate(plot.RGB_OkabeIto):
        ax.plot(xm_split[j,row], RGBG_Qm[j,row], color=c)
    ax.set_ylabel(f"{label}\nCounts [ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, RGBG_Qm[:,row,1000:].max()*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Pixel")
axs[-1].set_xlim(0, x.shape[0])
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{label_dataset}_row_gauss_Qm.pdf", bbox_inches="tight")
plt.close()

# Convert from pixels to nm
# To do: use apply_multi for interpolate_multi
lambdastep = 0.1
interpolation = lambda wavelengths_split, RGBG: wavelength.interpolate_multi(wavelengths_split, RGBG, lambdamin=200, lambdamax=800, lambdastep=lambdastep)

lambdarange, mean_grey_Qp_nm = interpolation(wavelengths_split_Qp, mean_grey_Qp_RGBG)
lambdarange, mean_sky_Qp_nm = interpolation(wavelengths_split_Qp, mean_sky_Qp_RGBG)
lambdarange, mean_water_Qp_nm = interpolation(wavelengths_split_Qp, mean_water_Qp_RGBG)

lambdarange, mean_grey_Qm_nm = interpolation(wavelengths_split_Qm, mean_grey_Qm_RGBG)
lambdarange, mean_sky_Qm_nm = interpolation(wavelengths_split_Qm, mean_sky_Qm_RGBG)
lambdarange, mean_water_Qm_nm = interpolation(wavelengths_split_Qm, mean_water_Qm_RGBG)

# Spectrum-based background subtraction
mean_grey_Qp_nm = ispex_general.background_subtraction_spectrum(lambdarange, mean_grey_Qp_nm)
mean_sky_Qp_nm = ispex_general.background_subtraction_spectrum(lambdarange, mean_sky_Qp_nm)
mean_water_Qp_nm = ispex_general.background_subtraction_spectrum(lambdarange, mean_water_Qp_nm)

mean_grey_Qm_nm = ispex_general.background_subtraction_spectrum(lambdarange, mean_grey_Qm_nm)
mean_sky_Qm_nm = ispex_general.background_subtraction_spectrum(lambdarange, mean_sky_Qm_nm)
mean_water_Qm_nm = ispex_general.background_subtraction_spectrum(lambdarange, mean_water_Qm_nm)

# Plot the Qm data in a single row, in nm
row = 50
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm], ["Grey card", "Sky", "Water"]):
    plot._rgbplot(lambdarange, RGBG_Qm[...,row], func=ax.plot)
    ax.set_ylabel(f"{label}\nCounts [ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, RGBG_Qm[...,row].max()*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{label_dataset}_row_Qm_nm.pdf", bbox_inches="tight")
plt.close()

# Import SRF
camera._load_spectral_response()
spectral_response = camera.spectral_response
spectral_response_RGBG = np.stack([np.interp(lambdarange, spectral_response[0], spectral_response[j]) for j in [1,2,3,2]])

# Plot Qm data in nm and the SRF
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm], ["Grey card", "Sky", "Water"]):
    SRF_normalised = spectral_response_RGBG * (RGBG_Qm[...,row].max(axis=1) / spectral_response_RGBG.max(axis=1))[..., np.newaxis]  # This would be much nicer if the axis order were reversed
    plot._rgbplot(lambdarange, RGBG_Qm[...,row], func=ax.plot)
    plot._rgbplot(lambdarange, SRF_normalised, func=ax.plot, ls="--")
    ax.set_ylabel(f"{label}\nCounts [ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, RGBG_Qm[...,row].max()*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{label_dataset}_row_Qm_vs_SRF.pdf", bbox_inches="tight")
plt.close()

# SRF calibration
spectral_response_RGBG_clipped = spectral_response_RGBG.copy()
spectral_response_RGBG_clipped[spectral_response_RGBG_clipped < 0.1] = np.nan

# for k, shift in enumerate(np.arange(-50, 50, 1)):
#     shift_lambda = shift * lambdastep
#     # Wavelength-shifted SRF calibration
#     SRF = spectral_response_RGBG_clipped[...,np.newaxis]
#     SRF = np.roll(SRF, shift, axis=1)
#     mean_grey_Qp_srf, mean_sky_Qp_srf, mean_water_Qp_srf, mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf = [arr/SRF for arr in [mean_grey_Qp_nm, mean_sky_Qp_nm, mean_water_Qp_nm, mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm]]

#     # Plot the Qm data in a single row, SRF-calibrated, with wavelength shift
#     fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
#     for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf], ["Grey card", "Sky", "Water"]):
#         for j, c in enumerate("rgb"):
#             ax.plot(lambdarange, RGBG_Qm[j,:,row], c=c)
#         ax.set_ylabel(f"{label}\nCounts [rel. ADU]")
#         ax.grid(ls="--")
#         ax.set_ylim(-5, np.nanmax(RGBG_Qm[...,row])*1.05)
#     for ax in axs[:-1]:
#         ax.tick_params(axis="x", bottom=False, labelbottom=False)
#     axs[-1].set_xlabel("Wavelength [nm]")
#     axs[-1].set_xlim(390, 700)
#     axs[0].set_title(f"Shift: {shift_lambda:.1f} nm")
#     plt.savefig(Path("results")/f"lambdashift/{label_dataset}_row_Qm_SRF_lambda_{k:04}.png", bbox_inches="tight")
#     plt.close()

# Regular SRF calibration
mean_grey_Qp_srf, mean_sky_Qp_srf, mean_water_Qp_srf, mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf = [arr/spectral_response_RGBG_clipped[...,np.newaxis] for arr in [mean_grey_Qp_nm, mean_sky_Qp_nm, mean_water_Qp_nm, mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm]]

# Plot the Qm data in a single row, SRF-calibrated
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf], ["Grey card", "Sky", "Water"]):
    plot._rgbplot(lambdarange, RGBG_Qm[...,row], func=ax.plot)
    ax.set_ylabel(f"{label}\nCounts [rel. ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, np.nanmax(RGBG_Qm[...,row])*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{label_dataset}_row_Qm_SRF.pdf", bbox_inches="tight")
plt.close()

mean_grey_Qp, mean_sky_Qp, mean_water_Qp, mean_grey_Qm, mean_sky_Qm, mean_water_Qm = [arr.mean(axis=2) for arr in [mean_grey_Qp_srf, mean_sky_Qp_srf, mean_water_Qp_srf, mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf]]

# Get the I spectrum
# Assume 100% transmission for now
mean_grey_I, mean_sky_I, mean_water_I = mean_grey_Qp + mean_grey_Qm, mean_sky_Qp + mean_sky_Qm, mean_water_Qp + mean_water_Qm

# Plot the stacked -Q, +Q, I data
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8,6), sharex=True, sharey="row")
for ax_row, RGBG_Qp, RGBG_Qm, RGBG_I, label in zip(axs, [mean_grey_Qp, mean_sky_Qp, mean_water_Qp], [mean_grey_Qm, mean_sky_Qm, mean_water_Qm], [mean_grey_I, mean_sky_I, mean_water_I], ["Grey card", "Sky", "Water"]):
    for ax, data in zip(ax_row, [RGBG_Qm, RGBG_Qp, RGBG_I/2.]):
        plot._rgbplot(lambdarange, data, func=ax.plot)
    ax_row[0].set_ylabel(f"{label}\nCounts [rel. ADU]")
    for ax in ax_row:
        ax.grid(ls="--")
        ymax = 1.05 * np.nanmax([RGBG_Qm, RGBG_Qp, RGBG_I/2.])
        ax.set_ylim(-5, ymax)
for ax in axs[:-1].ravel():
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
for ax in axs[-1]:
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim(390, 700)
axs[0,0].set_title("Stacked $-Q$ spectrum")
axs[0,1].set_title("Stacked $+Q$ spectrum")
axs[0,2].set_title("Stacked $I/2$ spectrum")
plt.savefig(Path("results")/f"{label_dataset}_stack.pdf", bbox_inches="tight")
plt.close()

# Mask points where Ed <= 0
mean_grey_I[mean_grey_I <= 0] = np.nan

# Calculate R_Rs naively
Ed = np.pi / 0.18 * mean_grey_I
Lw = mean_water_I - 0.028 * mean_sky_I
Rrs = Lw / Ed

# Plot Rrs
ymax = np.nanmax(Rrs) * 1.05
plot._rgbplot(lambdarange, Rrs)
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.xlim(390, 710)
plt.ylim(0, ymax)
plt.grid(ls="--")
plt.xlabel("Wavelength [nm]")
plt.title(f"Remote sensing reflectance ({label_dataset})")
plt.savefig(Path("results")/f"{label_dataset}_Rrs.pdf", bbox_inches="tight")
plt.close()

# Load WISP-3 data
wisp_wavelengths, wisp_lu, wisp_lu_err, wisp_ls, wisp_ls_err, wisp_ed, wisp_ed_err, wisp_rrs, wisp_rrs_err = validation.load_wisp_data(filename_wisp)

# Compare Rrs plots
ymax = np.nanmax([np.nanmax(Rrs), np.nanmax(wisp_rrs)])*1.05
plt.plot(wisp_wavelengths, wisp_rrs, c='k', label="WISP-3", lw=2)
for j, (c, pc) in enumerate(zip(plot.RGB, plot.RGB_OkabeIto)):
    plt.plot(lambdarange, Rrs[j], color=pc, label=f"iSPEX 2 {c}", lw=2)
plt.fill_between(wisp_wavelengths, wisp_rrs-wisp_rrs_err, wisp_rrs+wisp_rrs_err, facecolor="0.75", alpha=0.75)
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.xlim(390, 800)
plt.ylim(0, ymax)
plt.grid(ls="--")
plt.xlabel("Wavelength [nm]")
plt.title(f"Remote sensing reflectance ({label_dataset})")
plt.legend(loc="best", facecolor="white", edgecolor='k', framealpha=1)
plt.savefig(Path("results")/f"{label_dataset}_Rrs_WISP.pdf", bbox_inches="tight")
plt.close()

# Normalise smartphone data to WISP-3 data
Rrs_wisp_lambda = np.array([np.interp(wisp_wavelengths, lambdarange, R) for R in Rrs])  # Smartphone data interpolated to WISP-3 wavelengths
median_factor = np.nanmedian(Rrs_wisp_lambda / wisp_rrs)
print(f"Median ratio of iSPEX 2 / WISP-3: {100*median_factor:.1f}%")
Rrs_rescaled = Rrs / median_factor

ymax = np.nanmax([np.nanmax(Rrs_rescaled), np.nanmax(wisp_rrs)])*1.05
plt.figure(figsize=(5,3))
for j, (c, pc) in enumerate(zip(plot.RGB, plot.RGB_OkabeIto)):
    plt.plot(lambdarange, Rrs_rescaled[j], color=pc, label=f"iSPEX 2 {c}", lw=2)
plt.plot(wisp_wavelengths, wisp_rrs, c='k', label="WISP-3", lw=1)
plt.fill_between(wisp_wavelengths, wisp_rrs-wisp_rrs_err, wisp_rrs+wisp_rrs_err, facecolor="0.75", alpha=0.75)
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.xlim(390, 800)
plt.ylim(0, ymax)
plt.grid(ls="--")
plt.xlabel("Wavelength [nm]")
plt.title(f"Remote sensing reflectance ({label_dataset})")
plt.legend(loc="best", facecolor="white", edgecolor='k', framealpha=1)
plt.savefig(Path("results")/f"{label_dataset}_Rrs_WISP_normalised.pdf", bbox_inches="tight")
plt.close()

# Correlation plot for Rrs
MAD = np.nanmedian(np.abs(Rrs_wisp_lambda - wisp_rrs), axis=1)
MADrel = np.nanmedian(np.abs((Rrs_wisp_lambda - wisp_rrs)/Rrs_wisp_lambda), axis=1) * 100

plt.figure(figsize=(5,5))
for j, (c, pc) in enumerate(zip(plot.RGB, plot.RGB_OkabeIto)):
    label = (f"{c}: MAD = {MAD[j]:.4f} " "sr$^{-1}$" f" / {MADrel[j]:.1f} %")
    # plt.errorbar(wisp_rrs, Rrs_wisp_lambda[j], xerr=wisp_rrs_err, c=c, fmt="o")
    plt.scatter(wisp_rrs, Rrs_wisp_lambda[j], color=pc, label=label)
plt.plot([-1, 1], [-1, 1], c='k')
plt.xlabel("$R_{rs}$ WISP-3 [sr$^{-1}$]")
plt.ylabel("$R_{rs}$ iSPEX 2 [sr$^{-1}$]")
plt.legend(loc="best", facecolor="white", edgecolor='k', framealpha=1)
plt.grid(ls="--")
plt.xlim(-1e-3, ymax)
plt.ylim(-1e-3, ymax)
plt.title(f"WISP-3 vs iSPEX 2 ({label_dataset})")
plt.savefig(Path("results")/f"{label_dataset}_Rrs_correlation.pdf", bbox_inches="tight")
plt.close()

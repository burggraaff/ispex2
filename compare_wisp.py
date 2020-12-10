"""
Compare stacked images to WISP data.
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
mean_grey, mean_sky, mean_water = camera.correct_bias(mean_grey, mean_sky, mean_water)

# Flat-field correction
mean_grey, mean_sky, mean_water = camera.correct_flatfield(mean_grey, mean_sky, mean_water)

# Slice the data
slice_Qp, slice_Qm = ispex_general.find_spectrum(mean_grey)
top, middle, bottom = ispex_general.find_background(mean_grey)

# Show the bounding boxes for visualisation
ispex_plot.plot_bounding_boxes_Rrs(mean_grey, mean_sky, mean_water, label_file=filename_grey, saveto=Path("results")/f"{filename_grey.stem}_bounding_boxes.pdf", vmax=1000)

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
wavelengths_split_Qp, mean_grey_Qp_RGBG, mean_sky_Qp_RGBG, mean_water_Qp_RGBG, xp_split = raw.demosaick(bayer_Qp, wavelengths_Qp, mean_grey_Qp, mean_sky_Qp, mean_water_Qp, xp)
wavelengths_split_Qm, mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG, xm_split = raw.demosaick(bayer_Qm, wavelengths_Qm, mean_grey_Qm, mean_sky_Qm, mean_water_Qm, xm)

# Smooth the data in RGBG space
# 1 pixel in RGBG space = 2 pixels in RGB space
mean_grey_Qp_RGBG, mean_sky_Qp_RGBG, mean_water_Qp_RGBG, mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG = general.gauss_filter_multidimensional(mean_grey_Qp_RGBG, mean_sky_Qp_RGBG, mean_water_Qp_RGBG, mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG, sigma=(0,0,3))

# Plot the Qm data in a single row
row = 50
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_RGBG, mean_sky_Qm_RGBG, mean_water_Qm_RGBG], ["Grey card", "Sky", "Water"]):
    for j, c in enumerate("rgb"):
        ax.plot(xm_split[j,row], RGBG_Qm[j,row], c=c)
    ax.set_ylabel(f"{label}\nCounts [ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, RGBG_Qm[:,row,1000:].max()*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Pixel")
axs[-1].set_xlim(0, x.shape[0])
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{filename_ispex.stem}_row_gauss_Qm.pdf", bbox_inches="tight")
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
    for j, c in enumerate("rgb"):
        ax.plot(lambdarange, RGBG_Qm[j,:,row], c=c)
    ax.set_ylabel(f"{label}\nCounts [ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, RGBG_Qm[...,row].max()*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{filename_ispex.stem}_row_Qm_nm.pdf", bbox_inches="tight")
plt.close()

# Import SRF
camera._load_spectral_response()
spectral_response = camera.spectral_response
spectral_response_RGBG = np.stack([np.interp(lambdarange, spectral_response[0], spectral_response[j]) for j in [1,2,3,2]])

# Plot Qm data in nm and the SRF
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm], ["Grey card", "Sky", "Water"]):
    for j, c in enumerate("rgb"):
        ax.plot(lambdarange, RGBG_Qm[j,:,row], c=c)
        ax.plot(lambdarange, spectral_response_RGBG[j]*RGBG_Qm[j,:,row].max()/spectral_response_RGBG[j].max(), c=c, ls="--")
    ax.set_ylabel(f"{label}\nCounts [ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, RGBG_Qm[...,row].max()*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{filename_ispex.stem}_row_Qm_vs_SRF.pdf", bbox_inches="tight")
plt.close()

# SRF calibration
spectral_response_RGBG_clipped = spectral_response_RGBG.copy()
spectral_response_RGBG_clipped[spectral_response_RGBG_clipped < 0.1] = np.nan

for k, shift in enumerate(np.arange(-50, 50, 1)):
    shift_lambda = shift * lambdastep
    # Wavelength-shifted SRF calibration
    SRF = spectral_response_RGBG_clipped[...,np.newaxis]
    SRF = np.roll(SRF, shift, axis=1)
    mean_grey_Qp_srf, mean_sky_Qp_srf, mean_water_Qp_srf, mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf = [arr/SRF for arr in [mean_grey_Qp_nm, mean_sky_Qp_nm, mean_water_Qp_nm, mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm]]

    # Plot the Qm data in a single row, SRF-calibrated, with wavelength shift
    fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
    for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf], ["Grey card", "Sky", "Water"]):
        for j, c in enumerate("rgb"):
            ax.plot(lambdarange, RGBG_Qm[j,:,row], c=c)
        ax.set_ylabel(f"{label}\nCounts [rel. ADU]")
        ax.grid(ls="--")
        ax.set_ylim(-5, np.nanmax(RGBG_Qm[...,row])*1.05)
    for ax in axs[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    axs[-1].set_xlabel("Wavelength [nm]")
    axs[-1].set_xlim(390, 700)
    axs[0].set_title(f"Shift: {shift_lambda:.1f} $\\lambda$")
    plt.savefig(Path("results")/f"lambdashift/{filename_ispex.stem}_row_Qm_SRF_lambda_{k:04}.png", bbox_inches="tight")
    plt.close()

# Regular SRF calibration
mean_grey_Qp_srf, mean_sky_Qp_srf, mean_water_Qp_srf, mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf = [arr/spectral_response_RGBG_clipped[...,np.newaxis] for arr in [mean_grey_Qp_nm, mean_sky_Qp_nm, mean_water_Qp_nm, mean_grey_Qm_nm, mean_sky_Qm_nm, mean_water_Qm_nm]]

# Plot the Qm data in a single row, SRF-calibrated
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG_Qm, label in zip(axs, [mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf], ["Grey card", "Sky", "Water"]):
    for j, c in enumerate("rgb"):
        ax.plot(lambdarange, RGBG_Qm[j,:,row], c=c)
    ax.set_ylabel(f"{label}\nCounts [rel. ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, np.nanmax(RGBG_Qm[...,row])*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Pixel row {row}")
plt.savefig(Path("results")/f"{filename_ispex.stem}_row_Qm_SRF.pdf", bbox_inches="tight")
plt.close()

mean_grey_Qp, mean_sky_Qp, mean_water_Qp, mean_grey_Qm, mean_sky_Qm, mean_water_Qm = [arr.mean(axis=2) for arr in [mean_grey_Qp_srf, mean_sky_Qp_srf, mean_water_Qp_srf, mean_grey_Qm_srf, mean_sky_Qm_srf, mean_water_Qm_srf]]

# Plot the stacked Q+- data
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6,6), sharex=True, sharey="row")
for ax_row, RGBG_Qp, RGBG_Qm, label in zip(axs, [mean_grey_Qp, mean_sky_Qp, mean_water_Qp], [mean_grey_Qm, mean_sky_Qm, mean_water_Qm], ["Grey card", "Sky", "Water"]):
    for j, c in enumerate("rgb"):
        ax_row[0].plot(lambdarange, RGBG_Qm[j], c=c)
        ax_row[1].plot(lambdarange, RGBG_Qp[j], c=c)
        ax_row[0].set_ylabel(f"{label}\nCounts [rel. ADU]")
    for ax in ax_row:
        ax.grid(ls="--")
        ymax = 1.05 * np.nanmax([RGBG_Qm, RGBG_Qp])
        ax.set_ylim(-5, ymax)
for ax in axs[:-1].ravel():
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
for ax in axs[-1]:
    ax.set_xlabel("Wavelength [nm]")
    ax.set_xlim(390, 700)
axs[0,0].set_title("Stacked $-Q$ spectrum")
axs[0,1].set_title("Stacked $+Q$ spectrum")
plt.savefig(Path("results")/f"{filename_ispex.stem}_stack.pdf", bbox_inches="tight")
plt.close()

# Get the I spectrum
# Assume 100% transmission for now
mean_grey_I, mean_sky_I, mean_water_I = mean_grey_Qp + mean_grey_Qm, mean_sky_Qp + mean_sky_Qm, mean_water_Qp + mean_water_Qm

# Plot the stacked I spectra
fig, axs = plt.subplots(nrows=3, figsize=(6,6), sharex=True)
for ax, RGBG, label in zip(axs, [mean_grey_I, mean_sky_I, mean_water_I], ["Grey card", "Sky", "Water"]):
    for j, c in enumerate("rgb"):
        ax.plot(lambdarange, RGBG[j], c=c)
    ax.set_ylabel(f"{label}\nCounts [rel. ADU]")
    ax.grid(ls="--")
    ax.set_ylim(-5, np.nanmax(RGBG)*1.05)
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[-1].set_xlabel("Wavelength [nm]")
axs[-1].set_xlim(390, 700)
axs[0].set_title(f"Stacked $I$ spectrum")
plt.savefig(Path("results")/f"{filename_ispex.stem}_stack_I.pdf", bbox_inches="tight")
plt.close()

# Calculate R_Rs naively
Ed = np.pi / 0.23 * mean_grey_I
Lw = mean_water_I - 0.028 * mean_sky_I
Rrs = Lw / Ed
for j, c in enumerate("rgb"):
    plt.plot(lambdarange, Rrs[j], c=c)
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.ylim(ymin=0)
plt.grid(ls="--")
plt.xlabel("Wavelength [nm]")
plt.title(f"Remote sensing reflectance")
plt.savefig(Path("results")/f"{filename_ispex.stem}_Rrs.pdf", bbox_inches="tight")
plt.close()

# Load WISP-3 data
wisp_wavelengths, wisp_lu, wisp_lu_err, wisp_ls, wisp_ls_err, wisp_ed, wisp_ed_err, wisp_rrs, wisp_rrs_err = validation.load_wisp_data(filename_wisp)

# Compare Rrs plots
plt.plot(wisp_wavelengths, wisp_rrs, c='k', label="WISP-3", lw=2)
for j, c in enumerate("rgb"):
    plt.plot(lambdarange, Rrs[j], c=c, label=f"iSPEX 2 {c}", lw=2)
plt.fill_between(wisp_wavelengths, wisp_rrs-wisp_rrs_err, wisp_rrs+wisp_rrs_err, facecolor="0.75", alpha=0.75)
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.ylim(ymin=0)
plt.grid(ls="--")
plt.xlabel("Wavelength [nm]")
plt.title(f"Remote sensing reflectance")
plt.legend(loc="best")
plt.savefig(Path("results")/f"{filename_ispex.stem}_Rrs_WISP.pdf", bbox_inches="tight")
plt.close()

"""
Compare stacked images to WISP data.
"""

import numpy as np
from sys import argv
from spectacle import plot, io, wavelength, raw, general, calibrate
from ispex import general as ispex_general, plot as ispex_plot
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

# SLice out the Qp and Qm spectra
slice_Qp, slice_Qm = ispex_general.find_spectrum(mean_grey)

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

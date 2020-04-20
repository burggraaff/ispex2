"""
Calibrate the wavelength response of an iSPEX unit using a spectrum of a
fluorescent light.

Command line arguments:
    * `file`: location of a RAW photograph of a fluorescent light spectrum,
    taken with iSPEX.

This should either be made generic (for any spectrometric data) or be forked
into the iSPEX repository.

NOTE: May not function correctly due to changes to flat-fielding methods. This
will be fixed with the general overhaul for iSPEX 2.
"""

import numpy as np
from sys import argv
from spectacle import general, io, plot, wavelength, raw2
from ispex import general as ispex_general, wavelength as wvl
from pathlib import Path
from matplotlib import pyplot as plt

# Get the data folder from the command line
file = io.path_from_input(argv)
save_to_Qp = Path("calibration_data")/"wavelength_calibration_Qp.npy"
save_to_Qm = Path("calibration_data")/"wavelength_calibration_Qm.npy"

# Load the data
img = io.load_raw_file(file)
print("Loaded data")

data = img.raw_image.astype(np.float64)
bayer_map = img.raw_colors

# Rudimentary bias calibration
data = data - float(img.black_level_per_channel[0])

slice_Qp, slice_Qm = ispex_general.find_spectrum(data)

data_Qp, data_Qm = data[slice_Qp], data[slice_Qm]
bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]

RGB_Qp = raw2.pull_apart2(data_Qp, bayer_Qp)
RGB_Qm = raw2.pull_apart2(data_Qm, bayer_Qm)

x = np.arange(RGB_Qp.shape[2])
yp = np.arange(RGB_Qp.shape[1])
ym = np.arange(RGB_Qm.shape[1])

# Convolve the data with a Gaussian kernel on the wavelength axis to remove
# noise and fill in the gaps
gauss_Qp = general.gauss_nan(RGB_Qp, sigma=(0,0,10))
gauss_Qm = general.gauss_nan(RGB_Qm, sigma=(0,0,10))

# Find the locations of the line peaks in every row
x_offset = 1700
lines_Qp = wavelength.find_fluorescent_lines(gauss_Qp[...,x_offset:]) + x_offset
lines_Qm = wavelength.find_fluorescent_lines(gauss_Qm[...,x_offset:]) + x_offset

lines_fit_Qp = wavelength.fit_fluorescent_lines(lines_Qp, yp)
lines_fit_Qm = wavelength.fit_fluorescent_lines(lines_Qm, ym)

plot.plot_fluorescent_lines(yp, lines_Qp, lines_fit_Qp)
plot.plot_fluorescent_lines(ym, lines_Qm, lines_fit_Qm)

# Fit a wavelength relation for each row
wavelength_fits_Qp = wavelength.fit_many_wavelength_relations(yp, lines_fit_Qp)
wavelength_fits_Qm = wavelength.fit_many_wavelength_relations(ym, lines_fit_Qm)

# Fit a polynomial to the coefficients of the previous fit
coefficients_Qp, coefficients_fit_Qp = wavelength.fit_wavelength_coefficients(yp, wavelength_fits_Qp)
coefficients_Qm, coefficients_fit_Qm = wavelength.fit_wavelength_coefficients(ym, wavelength_fits_Qm)

# Plot the polynomial fits to the coefficients
plot.wavelength_coefficients(yp, wavelength_fits_Qp, coefficients_fit_Qp)
plot.wavelength_coefficients(ym, wavelength_fits_Qm, coefficients_fit_Qm)

# Save the coefficients to file
wavelength.save_coefficients(coefficients_Qp, saveto=save_to_Qp)
wavelength.save_coefficients(coefficients_Qm, saveto=save_to_Qm)
print(f"Saved wavelength coefficients to '{save_to_Qp}' and '{save_to_Qm}'")

# Convert the input spectrum to wavelengths and plot it, as a sanity check
wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

wavelengths_split_Qp,_ = raw2.pull_apart(wavelengths_Qp, bayer_Qp)
wavelengths_split_Qm,_ = raw2.pull_apart(wavelengths_Qm, bayer_Qm)
RGBG_Qp,_ = raw2.pull_apart(data_Qp, bayer_Qp)
RGBG_Qm,_ = raw2.pull_apart(data_Qm, bayer_Qm)

lambdarange, all_interpolated_Qp = wavelength.interpolate_multi(wavelengths_split_Qp, RGBG_Qp)
lambdarange, all_interpolated_Qm = wavelength.interpolate_multi(wavelengths_split_Qm, RGBG_Qm)

stacked_Qp = wavelength.stack(lambdarange, all_interpolated_Qp)
stacked_Qm = wavelength.stack(lambdarange, all_interpolated_Qm)
plot.plot_fluorescent_spectrum(stacked_Qp[0], stacked_Qp[1:])
plot.plot_fluorescent_spectrum(stacked_Qm[0], stacked_Qm[1:])

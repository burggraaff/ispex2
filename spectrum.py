import numpy as np
from sys import argv
from spectacle import plot, io, wavelength, raw2
from ispex import general as ispex_general
from matplotlib import pyplot as plt
from pathlib import Path

file = io.path_from_input(argv)

# Load the data
img = io.load_raw_file(file)
print("Loaded data")

data = img.raw_image.astype(np.float64)
bayer_map = img.raw_colors

data = img.raw_image.astype(np.float64) - float(img.black_level_per_channel[0])
#data = calibrate.correct_bias(root, data)

slice_Qp, slice_Qm = ispex_general.find_spectrum(data)

data_Qp, data_Qm = data[slice_Qp], data[slice_Qm]
bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]

x = np.arange(data_Qp.shape[1])
yp = np.arange(data_Qp.shape[0])
ym = np.arange(data_Qm.shape[0])

coefficients_Qp = np.load(Path("calibration_data")/"wavelength_calibration_Qp.npy")
coefficients_Qm = np.load(Path("calibration_data")/"wavelength_calibration_Qm.npy")

wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

wavelengths_split_Qp,_ = raw2.pull_apart(wavelengths_Qp, bayer_Qp)
wavelengths_split_Qm,_ = raw2.pull_apart(wavelengths_Qm, bayer_Qm)
RGBG_Qp,_ = raw2.pull_apart(data_Qp, bayer_Qp)
RGBG_Qm,_ = raw2.pull_apart(data_Qm, bayer_Qm)

lambdarange, all_interpolated_Qp = wavelength.interpolate_multi(wavelengths_split_Qp, RGBG_Qp)
lambdarange, all_interpolated_Qm = wavelength.interpolate_multi(wavelengths_split_Qm, RGBG_Qm)

for j, c in enumerate("rgb"):
    plt.plot(lambdarange, all_interpolated_Qp[j,:,50], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Counts [ADU]")
plt.savefig(Path("results")/f"{file.stem}_row_Qp.pdf", bbox_inches="tight")
plt.close()

for j, c in enumerate("rgb"):
    plt.plot(lambdarange, all_interpolated_Qm[j,:,50], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Counts [ADU]")
plt.savefig(Path("results")/f"{file.stem}_row_Qm.pdf", bbox_inches="tight")
plt.close()

stacked_Qp = wavelength.stack(lambdarange, all_interpolated_Qp)
stacked_Qm = wavelength.stack(lambdarange, all_interpolated_Qm)

plot.plot_spectrum(stacked_Qp[0], stacked_Qp[1:], saveto=Path("results")/f"{file.stem}_Qp.pdf")
plot.plot_spectrum(stacked_Qm[0], stacked_Qm[1:], saveto=Path("results")/f"{file.stem}_Qm.pdf")

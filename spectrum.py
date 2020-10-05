import numpy as np
from sys import argv
from spectacle import plot, io, wavelength, raw2, general, calibrate
from ispex import general as ispex_general, plot as ispex_plot
from matplotlib import pyplot as plt
from pathlib import Path

file = io.path_from_input(argv)

root = Path(r"C:\Users\Burggraaff\SPECTACLE_data\iPhone_SE")

# Load the data
img = io.load_raw_file(file)
print("Loaded data")

data = img.raw_image.astype(np.float64)
bayer_map = img.raw_colors

# Bias correction
try:
    data = calibrate.correct_bias(root, data)
except:
    data = img.raw_image.astype(np.float64) - float(img.black_level_per_channel[0])

# Flat-field correction
try:
    data = calibrate.correct_flatfield(root, data)
except:
    print("No flat-field correction done")

slice_Qp, slice_Qm = ispex_general.find_spectrum(data)

data_Qp, data_Qm = data[slice_Qp], data[slice_Qm]
bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]

x = np.arange(data_Qp.shape[1])
xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T
xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T
yp = np.arange(data_Qp.shape[0])
ym = np.arange(data_Qm.shape[0])

coefficients_Qp = np.load(Path("calibration_data")/"wavelength_calibration_Qp.npy")
coefficients_Qm = np.load(Path("calibration_data")/"wavelength_calibration_Qm.npy")

wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

wavelengths_split_Qp,_ = raw2.pull_apart2(wavelengths_Qp, bayer_Qp)
wavelengths_split_Qm,_ = raw2.pull_apart2(wavelengths_Qm, bayer_Qm)
RGBG_Qp,_ = raw2.pull_apart2(data_Qp, bayer_Qp)
RGBG_Qm,_ = raw2.pull_apart2(data_Qm, bayer_Qm)
xp_split,_ = raw2.pull_apart2(xp, bayer_Qp)
xm_split,_ = raw2.pull_apart2(xm, bayer_Qm)

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb"):
    plt.plot(xm_split[j,50], RGBG_Qm[j,50], c=c)
plt.xlabel("Pixel")
plt.ylabel("Counts [ADU]")
plt.grid(ls="--")
plt.ylim(-5, RGBG_Qm[:,50,1000:].max()*1.05)
plt.xlim(0, x.shape[0])
plt.savefig(Path("results")/f"{file.stem}_row_raw_Qm.pdf", bbox_inches="tight")
plt.close()

# 1 pixel in RGBG space = 2 pixels in RGB space
RGBG_Qp = general.gauss_nan(RGBG_Qp, sigma=(0,0,3))
RGBG_Qm = general.gauss_nan(RGBG_Qm, sigma=(0,0,3))

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb"):
    plt.plot(xm_split[j,50], RGBG_Qm[j,50], c=c)
plt.xlabel("Pixel")
plt.ylabel("Counts [ADU]")
plt.grid(ls="--")
plt.ylim(-5, RGBG_Qm[:,50,1000:].max()*1.05)
plt.xlim(0, x.shape[0])
plt.savefig(Path("results")/f"{file.stem}_row_gauss_Qm.pdf", bbox_inches="tight")
plt.close()

lambdarange, all_interpolated_Qp = wavelength.interpolate_multi(wavelengths_split_Qp, RGBG_Qp)
lambdarange, all_interpolated_Qm = wavelength.interpolate_multi(wavelengths_split_Qm, RGBG_Qm)

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb"):
    plt.plot(lambdarange, all_interpolated_Qp[j,:,50], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Counts [ADU]")
plt.grid(ls="--")
plt.ylim(-5, np.nanmax(all_interpolated_Qp[...,50])*1.05)
plt.xlim(390, 700)
plt.savefig(Path("results")/f"{file.stem}_row_Qp.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb"):
    plt.plot(lambdarange, all_interpolated_Qm[j,:,50], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Counts [ADU]")
plt.grid(ls="--")
plt.ylim(-5, np.nanmax(all_interpolated_Qm[...,50])*1.05)
plt.xlim(390, 700)
plt.savefig(Path("results")/f"{file.stem}_row_Qm.pdf", bbox_inches="tight")
plt.close()

spectral_response = np.load("calibration_data/spectral_response.npy")
spectral_response_RGBG = np.stack([np.interp(lambdarange, spectral_response[0], spectral_response[j]) for j in [1,2,3,2]])
spectral_response_RGBG[spectral_response_RGBG < 0.20] = np.nan

# Single wavelength
j = 50
plt.plot(all_interpolated_Qp[2,j,:])
plt.show()
plt.close()

all_interpolated_Qp = all_interpolated_Qp / spectral_response_RGBG[...,np.newaxis]
all_interpolated_Qm = all_interpolated_Qm / spectral_response_RGBG[...,np.newaxis]

fig, axs = plt.subplots(nrows=2, figsize=(4,2), gridspec_kw={"hspace": 0.05, "wspace": 0}, sharex=True, sharey=True)
for ax, spectrum in zip(axs, [all_interpolated_Qp[1], all_interpolated_Qm[1]]):
    indices = np.linspace(spectrum.shape[1]*0.05, spectrum.shape[1]*0.95-1, 4).astype(int)
    for ind in indices:
        ax.plot(lambdarange, spectrum[:,ind])
    ax.grid(ls="--")
    ax.set_yticks(np.arange(0,1400,500))
    ax.set_ylim(-5, 1400)
axs[1].set_ylabel(" "*15+"Radiance [a.u.]")
axs[1].set_xlabel("Wavelength [nm]")
axs[0].tick_params(axis="x", bottom=False)

plt.savefig(Path("results")/f"{file.stem}_Qp_vs_Qm.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb"):
    plt.plot(lambdarange, all_interpolated_Qp[j,:,50], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Radiance [a.u.]")
plt.grid(ls="--")
plt.ylim(-5, np.nanmax(all_interpolated_Qp[...,50])*1.05)
plt.xlim(390, 700)
plt.savefig(Path("results")/f"{file.stem}_row_norm_Qp.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb"):
    plt.plot(lambdarange, all_interpolated_Qm[j,:,50], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Radiance [a.u.]")
plt.grid(ls="--")
plt.ylim(-5, np.nanmax(all_interpolated_Qm[...,50])*1.05)
plt.xlim(390, 700)
plt.savefig(Path("results")/f"{file.stem}_row_norm_Qm.pdf", bbox_inches="tight")
plt.close()

stacked_Qp = wavelength.stack(lambdarange, all_interpolated_Qp)
stacked_Qm = wavelength.stack(lambdarange, all_interpolated_Qm)

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb", 1):
    plt.plot(stacked_Qp[0], stacked_Qp[j], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Radiance [a.u.]")
plt.grid(ls="--")
plt.ylim(-5, np.nanmax(stacked_Qp[1:])*1.05)
plt.xlim(390, 700)
plt.savefig(Path("results")/f"{file.stem}_stacked_norm_Qp.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,2))
for j, c in enumerate("rgb", 1):
    plt.plot(stacked_Qm[0], stacked_Qm[j], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Radiance [a.u.]")
plt.grid(ls="--")
plt.ylim(-5, np.nanmax(stacked_Qm[1:])*1.05)
plt.xlim(390, 700)
plt.savefig(Path("results")/f"{file.stem}_stacked_norm_Qm.pdf", bbox_inches="tight")
plt.close()

#stacked_Qp[1:] = stacked_Qp[1:] / spectral_response_RGB
#stacked_Qm[1:] = stacked_Qm[1:] / spectral_response_RGB
#
#plot.plot_spectrum(stacked_Qp[0], stacked_Qp[1:], saveto=Path("results")/f"{file.stem}_norm_Qp.pdf")
#plot.plot_spectrum(stacked_Qm[0], stacked_Qm[1:], saveto=Path("results")/f"{file.stem}_norm_Qm.pdf")

import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex import raw, plot, io, wavelength
from glob import glob

folder = argv[1]
files = glob(folder+"/*.dng")
for filename in files:
    print(filename)
    handle = "_".join(filename.replace("\\", "/").split("/")).split(".")[0]

    img = io.load_raw_file(filename)
    exif = io.load_exif(filename)
    image_cut  = raw.cut_out_spectrum(img.raw_image)
    colors_cut = raw.cut_out_spectrum(img.raw_colors)

    RGBG = raw.demosaick(colors_cut, image_cut)
    plot.RGBG_stacked(RGBG, extent=(raw.xmin, raw.xmax, raw.ymax, raw.ymin), show_axes=True, saveto="RGBG_cutout.png")

    coefficients = wavelength.load_coefficients("wavelength_solution.npy")
    wavelengths_cut = wavelength.calculate_wavelengths(coefficients, raw.x, raw.y)
    wavelengths_split = raw.demosaick(colors_cut, wavelengths_cut)

    lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)
    plot.RGBG_stacked(all_interpolated, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), show_axes=True, xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="RGBG_cutout_corrected.png")
    plot.RGBG_stacked_with_graph(all_interpolated, x=lambdarange, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="RGBG_cutout_corrected_spectrum.png")

    plot.RGBG(RGBG, vmax=800, saveto="RGBG_split.png", size=30)
    plot.RGBG(all_interpolated, vmax=800, saveto="RGBG_split_interpolated.png", size=30)

    stacked = wavelength.stack(lambdarange, all_interpolated)
    np.save(io.results_folder/"spectra/"+handle+"_spectrum.npy", stacked)

    plot.plot_spectrum(stacked[...,0], stacked[...,1:], saveto=io.results_folder/"spectra/"+handle+"_spectrum_rgb.png",
                       xlim=(340, 760), ylim=(528, None), title=exif["Image DateTime"].values)
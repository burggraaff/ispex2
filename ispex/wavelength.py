"""
Wavelength calibration module
"""

import numpy as np

fluorescent_lines = np.array([611.6, 544.45, 436.6])  # RGB, units: nm


def dispersion_fluorescent(lines_fit):
    dispersion = (fluorescent_lines[0] - fluorescent_lines[2]) / (lines_fit[0] - lines_fit[2])
    return dispersion


def resolution(data_RGB, dispersion):
    slit = data_RGB[1,:,:data_RGB.shape[2]//2] # Get the left half of the G image
    peak_height = np.nanmax(slit, axis=1)
    FWHMs_px = np.zeros_like(peak_height)
    for j,row in enumerate(slit):
        in_slit = np.where(row >= peak_height[j]/2)[0]
        FWHMs_px[j] = in_slit[-1] - in_slit[0]

    FWHMs_nm = FWHMs_px * dispersion

"""
General functions for iSPEX 2 data reduction
"""

import numpy as np

def find_spectrum(data, device="iPhone SE"):
    """
    Find the x and y limits that contain the two spectra in the image

    Hardcoded for now
    """
    # # WISP-3 comparison
    slice_Qp = np.s_[1000:1300]
    slice_Qm = np.s_[1750:2050]

    # SPIE paper
    # slice_Qp = np.s_[650:1400]
    # slice_Qm = np.s_[1550:2300]

    return slice_Qp, slice_Qm


def find_background(data, device="iPhone SE"):
    """
    Find the x and y limits for background light

    Hardcoded for now
    """

    top = np.s_[:600]
    middle = np.s_[1470:1600]
    bottom = np.s_[2500:]

    return top, middle, bottom


def background_subtraction_spectrum(lambdarange, data):
    """
    Do a simple polynomial-based background subtraction on the spectrum
    """
    background = data.copy()
    background_inds = np.where((lambdarange < 380) | (lambdarange > 710))[0]

    background = np.moveaxis(background, 1, 0)
    background = background.reshape(len(lambdarange), -1)

    f = np.polyfit(lambdarange[background_inds], background[background_inds], 1)
    background_fit = f[0] * lambdarange[:,np.newaxis] + f[1]

    background_fit = background_fit.reshape(len(lambdarange), 4, -1)
    background_fit = np.moveaxis(background_fit, 1, 0)

    data_corrected = data - background_fit

    return data_corrected

"""
General functions for iSPEX 2 data reduction
"""

import numpy as np

def find_spectrum(data, device="iPhone SE"):
    """
    Find the x and y limits that contain the two spectra in the image

    Hardcoded for now
    """

    slice_Qp = np.s_[1000:1300]
    slice_Qm = np.s_[1750:2050]

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

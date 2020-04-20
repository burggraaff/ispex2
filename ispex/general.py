"""
General functions for iSPEX 2 data reduction
"""

import numpy as np

def find_spectrum(data, device="iPhone SE"):
    """
    Find the x and y limits that contain the two spectra in the image

    Hardcoded for now
    """

    slice_Qp = np.s_[540:1400]
    slice_Qm = np.s_[1480:2300]

    return slice_Qp, slice_Qm

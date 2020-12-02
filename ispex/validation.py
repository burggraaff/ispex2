import numpy as np

def _convert_wisp_block(block):
    block_stripped = [line.strip().replace('\"', '') for line in block]
    block_split = [line.split(",") for line in block_stripped]
    block_arr = np.array(block_split, dtype=np.float64)
    wavelengths = block_arr[:,0]
    data = block_arr[:,1:]
    return wavelengths, data

def load_wisp_data(wisp_filename):
    """
    Get (ir)radiance and reflectance data from a WISPweb output file
    """
    with open(wisp_filename, "r") as file:
        lines = file.readlines()

    block_Ld = lines[425:826]
    block_Lu = lines[827:1228]
    block_Ed = lines[1229:]

    wvl, Ld = _convert_wisp_block(block_Ld)
    wvl, Lu = _convert_wisp_block(block_Lu)
    wvl, Ed = _convert_wisp_block(block_Ed)

    Ld_mean = Ld.mean(axis=1)
    Lu_mean = Lu.mean(axis=1)
    Ed_mean = Ed.mean(axis=1)

    Ld_err = Ld.std(axis=1)# / np.sqrt(Ld.shape[1])
    Lu_err = Lu.std(axis=1)# / np.sqrt(Lu.shape[1])
    Ed_err = Ed.std(axis=1)# / np.sqrt(Ed.shape[1])

    rho = 0.028
    Rrs = (Lu_mean - rho*Ld_mean) / Ed_mean
    Rrs_var = Lu_err**2 * (1/Ed_mean)**2 + Ld_err**2 * (-rho/Ed_mean)**2 + Ed_err**2 * (-Rrs/Ed_mean)**2
    Rrs_err = np.sqrt(Rrs_var)

    return wvl, Lu_mean, Lu_err, Ld_mean, Ld_err, Ed_mean, Ed_err, Rrs, Rrs_err

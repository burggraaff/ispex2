import numpy as np

def load_wisp_data(wisp_filename):
    # Get the integration times from the header, in seconds
    with open(wisp_filename, "r") as file:
        integration_times = np.array(file.readlines()[6].strip().split("\t")[1:], dtype=np.float16) * 1e-6

    wisp_data = np.genfromtxt(wisp_filename, skip_header=11, skip_footer=1, unpack=True)

    # Normalise the radiometry by the integration times
    wisp_data[1::2] /= integration_times[:,np.newaxis]

import numpy as np


class DataOut(object):
    def __init__(self):
        pass

    def WriteToCSV(self, nparray, filename, header, fmt='%5.4f'):
        np.savetxt(filename, nparray, delimiter=",", fmt=fmt,header=",".join(header),comments="")
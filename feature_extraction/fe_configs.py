import numpy as np

SAMPLING_RATE = 8000
N_MFCC = 13
N_FFT = 1024
eps = np.finfo(np.float64).eps


class NormFactor:
    def __init__(self, m, s):
        self.means = m
        self.std = s


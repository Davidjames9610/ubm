from processing.processing import Processing as Proc
import numpy as np


# create combinations of processing functions for pre- and post-processing
class ProcessMethodBase:

    def __init__(self, snr_db=None, reverb=False, signal_average_power=0):
        self.snr_db = snr_db
        self.reverb = reverb
        self.noise_ap = signal_average_power / np.power((snr_db / 10), 10)

    def __str__(self):
        return f'ProcessMethod'

    @staticmethod
    def pre_process(file):
        signal = Proc.file_to_signal(file)
        return Proc.normalize_signal(signal)

    def post_process(self, signal):
        if self.snr_db is not None:
            signal = Proc.add_noise(signal, self.noise_ap)
        if self.reverb:
            signal = Proc.add_reverb(signal)
        return signal

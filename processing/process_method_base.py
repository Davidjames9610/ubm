from processing.processing import Processing as Proc


# create combinations of processing functions for pre- and post-processing
class ProcessMethodBase:

    def __init__(self, snr=None, reverb=False):
        self.snr = snr
        self.reverb = reverb

    def __str__(self):
        return f'ProcessMethod'

    @staticmethod
    def pre_process(file):
        signal = Proc.file_to_signal(file)
        return Proc.normalize_signal(signal)

    def post_process(self, signal):
        if self.snr is not None:
            signal = Proc.add_noise(signal, self.snr)
        if self.reverb:
            signal = Proc.add_reverb(signal)
        return signal

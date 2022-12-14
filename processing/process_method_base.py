from processing.processing import Processing as Proc


# create combinations of processing functions for pre- and post-processing
class ProcessMethodBase:

    def __init__(self, snr_db, reverb=True):
        self.snr_db = snr_db
        self.reverb = reverb

    def __str__(self):
        return f'ProcessMethod'

    @staticmethod
    def pre_process(file):
        signal = Proc.file_to_signal(file)
        return Proc.normalize_signal(signal)

    def post_process(self, signal):
        return Proc.add_reverb(Proc.add_noise(signal, self.snr_db))

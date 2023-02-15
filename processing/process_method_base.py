from processing.processing import Processing as Proc
import numpy as np
import torch
import config

# create combinations of processing functions for pre- and post-processing
class ProcessMethodBase:

    def __init__(self, snr_db=None, reverb=False, signal_average_power=0.1, model=None, get_speech_timestamps=None):
        self.snr_db = snr_db
        self.reverb = reverb
        if snr_db:
            self.noise_ap = signal_average_power / np.power((snr_db / 10), 10)
        else:
            self.noise_ap = None
        self.model = model
        self.get_speech_timestamps = get_speech_timestamps

    def __str__(self):
        return f'ProcessMethod'

    # @staticmethod
    # def pre_process(file):
    #     signal = Proc.file_to_signal(file)
    #     return Proc.normalize_signal(signal)

    def pre_process(self, file):
        signal = Proc.normalize_signal(Proc.file_to_signal(file))
        speech_timestamps = self.get_speech_timestamps(signal, self.model, sampling_rate=config.SAMPLING_RATE)
        if self.model:
            segments = []
            for i in range(len(speech_timestamps)):
                start = speech_timestamps[i]['start']
                end = speech_timestamps[i]['end']
                segments.append(signal[start: end])
            segments_flattened = np.array([item for sublist in segments for item in sublist])
            if len(segments_flattened) > 0:
                return segments_flattened
            else:
                return signal
        else:
            return signal

    def post_process(self, signal):
        if self.snr_db is not None:
            signal = Proc.add_noise(signal, self.noise_ap)
        if self.reverb:
            signal = Proc.add_reverb(signal)
        return signal

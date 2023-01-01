import librosa
import numpy as np
import torch
import torchaudio
from config import *


# simple pre- and post-processing functions
class Processing:
    """
    import file to signal and resample to projects sampling frequency
    """

    @staticmethod
    def file_to_signal(file):
        signal, _ = librosa.load(file, sr=SAMPLING_RATE)
        return signal

    @staticmethod
    def normalize_signal(signal):
        signal_normalised = librosa.util.normalize(signal, axis=0)
        return signal_normalised

    @staticmethod
    def add_reverb(signal):
        signal_split = np.array([signal / 2, signal / 2])
        signal_torch = torch.Tensor(signal_split)

        effects = [
            ['gain', '-n'],
            ["reverb", "-w"],
        ]

        signal_torch_reverb, _ = torchaudio.sox_effects.apply_effects_tensor(signal_torch, SAMPLING_RATE,
                                                                             effects)
        signal_reverb = signal_torch_reverb[0].numpy() * 2
        return signal_reverb

    @staticmethod
    def add_noise(signal, snr, signal_average_power=0.1):
        noise_average_power = signal_average_power / snr
        noise = np.random.normal(0, np.sqrt(noise_average_power), len(signal))
        return signal + noise

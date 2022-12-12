import numpy as np
from scipy.io import wavfile
import spafe.features.mfcc as mfcc
import torch
import torchaudio
import torchaudio.functional as F
from IPython.display import Audio
import matplotlib.pyplot as plt
import spafe.features.mfcc as mfcc
import librosa
from torch.distributions import transforms
from torchaudio.utils import download_asset
import config


class PreProcesses:

    def __init__(self):
        self.noise = self.file_to_signal_resample(
            download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"))

    @staticmethod
    def file_to_signal(file):
        sr, signal = wavfile.read(file)
        return signal, sr

    @staticmethod
    def re_sample_signal(signal, sr):
        transform = torchaudio.transforms.Resample(sr, config.SAMPLING_RATE)
        signal = transform(signal)
        return signal

    def file_to_signal_resample(self, file):
        signal, sr = self.file_to_signal(file)
        return self.re_sample_signal(signal, sr)

    @staticmethod
    def add_reverb(signal):
        effects = [
            ["reverb", "-w"]
        ]
        # Apply effects
        signal, _ = torchaudio.sox_effects.apply_effects_tensor(signal, config.SAMPLING_RATE, effects)
        return signal

    def add_noise(self, signal, snr_db):
        noise = self.noise[:, : signal.shape[1]]

        speech_rms = signal.norm(p=2)
        noise_rms = noise.norm(p=2)

        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms
        noisy_signal = (scale * signal + noise) / 2

        return noisy_signal

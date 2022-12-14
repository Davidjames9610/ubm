import librosa
import numpy as np
import torch
import torchaudio
import config


# simple pre- and post-processing functions
class Processing:
    """
    import file to signal and resample to projects sampling frequency
    """

    @staticmethod
    def file_to_signal(file):
        signal, _ = librosa.load(file, sr=config.SAMPLING_RATE)
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

        signal_torch_reverb, _ = torchaudio.sox_effects.apply_effects_tensor(signal_torch, config.SAMPLING_RATE,
                                                                             effects)
        signal_reverb = signal_torch_reverb[0].numpy() * 2
        return signal_reverb

    @staticmethod
    def add_noise(signal, snr_db):
        signal_split = np.array([signal / 2, signal / 2])
        signal_torch = torch.Tensor(signal_split)
        speech_rms = signal_torch.norm(p=2)

        noise = np.random.normal(0, np.sqrt(speech_rms.numpy()), len(signal))
        noise_split = np.array([noise / 2, noise / 2])
        noise_torch = torch.Tensor(noise_split)
        noise_rms = noise_torch.norm(p=2)

        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms
        noisy_torch_signal = (scale * signal_torch + noise_torch) / 2
        noisy_signal = noisy_torch_signal[0].numpy() * 2

        return librosa.util.normalize(noisy_signal, axis=0)

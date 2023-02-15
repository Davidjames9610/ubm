import torchaudio
import numpy as np
import config
import os.path
import math
import os
import pathlib
import random
import torch

class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data

# class ComposeTransform:
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, audio_data):
#         for t in self.transforms:
#             audio_data = t(audio_data)
#         return audio_data
#
# compose_transform = ComposeTransform([
#     RandomClip(sample_rate=sample_rate,clip_length=64000),
#     RandomSpeedChange(sample_rate),
#     RandomBackgroundNoise(sample_rate, './noises_directory')])
#
# transformed_audio = compose_transform(audio_data)
class FileToTensor:
    def __init__(self, sample_rate=config.SAMPLING_RATE):
        self.sample_rate = sample_rate

    def __call__(self, file):
        sox_effects = [
            ['remix', '1'],  # convert to mono
            ['rate', str(self.sample_rate)],  # resample
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_file(file, sox_effects, normalize=True)
        return transformed_audio

class SileroVad:
    def __init__(self, sample_rate=config.SAMPLING_RATE):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False)
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        self.sample_rate = sample_rate
        self.model = model
        self.get_speech_timestamps = get_speech_timestamps
        self.save_audio = save_audio
        self.read_audio = read_audio
        self.VADIterator = VADIterator
        self.collect_chunks = collect_chunks

    def __call__(self, audio_data):
        speech_timestamps = self.get_speech_timestamps(audio_data, self.model, sampling_rate=self.sample_rate)
        transformed_audio = self.collect_chunks(speech_timestamps, audio_data)
        return transformed_audio

class NormalizeSox:
    def __init__(self, sample_rate=config.SAMPLING_RATE):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        sox_effects = [
            ['gain', '-n']  # normalises to 0dB
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            tensor=audio_data, sample_rate=self.sample_rate, effects=sox_effects)
        return transformed_audio

class NormalizeUsingVariance:
    def __init__(self):
        pass

    def __call__(self, audio_data):
        std = torch.std(audio_data) * 10  # > 97%
        transformed_audio = audio_data / std
        return transformed_audio

# add background noise of specific SNR DB by measuring signals power and using it to create white noise
# snr_db = 10log(signal_power/noise_power)
# normalize afterwards maybe ?
class AddBackgroundWhiteNoise:

    def __init__(self, snr_db):
        self.snr_db = snr_db

    def __call__(self, audio_data):
        snr = math.exp(self.snr_db / 10)
        audio_power = audio_data.norm(p=2)  # not sure about this ?
        noise_power = audio_power / snr
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio_power))
        return audio_data + noise

class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[offset:(offset + self.clip_length)]

        return self.vad(audio_data)  # remove silences at the begining/end


class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        print(speed_factor)
        if speed_factor == 1.0:  # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio


class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15, snr_db=20):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.snr_db = snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'],  # convert to mono
            ['rate', str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        # add in noise randomly if longer, or if shorter then zero pad end
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset:offset + audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise) / 2

import torchaudio
import numpy as np
import config
import os.path
import math
import os
import pathlib
import random
import torch
import utils
import warnings
import my_torch.torchio as tio
import torchaudio.functional as F
import torchaudio.transforms as T


class ComposeTransform:
    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data

class ComposeProcessTransform(ComposeTransform):
    def __init__(self, transforms):
        all_transforms = [ProcessMethodCheck()]
        for transform in transforms:
            all_transforms.append(transform)
        super().__init__(all_transforms)


class TensorToNumpy:
    def __init__(self):
        pass

    def __call__(self, audio):
        return audio.numpy().T


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
            ['rate', str(self.sample_rate)],  # resample # TODO check if this low pass filters first
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_file(file, sox_effects, normalize=True)
        return transformed_audio


class FileToTensorLPF:
    def __init__(self, sample_rate=config.SAMPLING_RATE, resampling_method="kaiser_window"):
        self.sample_rate = sample_rate
        self.resampling_method = resampling_method

    def __call__(self, file):
        sox_effects = [
            ['remix', '1'],  # convert to mono
        ]
        transformed_audio, original_sample_rate = torchaudio.sox_effects.apply_effects_file(file, sox_effects,
                                                                                            normalize=True)
        transformed_audio = F.resample(transformed_audio, original_sample_rate, self.sample_rate,
                                       resampling_method=self.resampling_method)
        return transformed_audio


class ProcessMethodCheck:
    """
    Process Methods receive tensors
    ...
    """
    def __init__(self):
        pass

    def __call__(self, audio_data):
        if torch.is_tensor(audio_data):
            return audio_data
        else:
            return torch.from_numpy(audio_data.T)

# can remove added to base fe method
class FeMethodCheck:
    """
    Fe Methods receive numpy arrays
    ...

    """
    def __init__(self):
        pass

    def __call__(self, audio_data):
        if torch.is_tensor(audio_data):
            return audio_data.numpy().T
        else:
            return audio_data

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


class SileroVad:
    """
    Use SileroVad

    implement torch audio vad as well sometime
    torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)
    ...

    Attributes
    ----------
    sample_rate : int
        target snr_db

    """

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
        transformed_audio = self.collect_chunks_custom(speech_timestamps, audio_data)
        return transformed_audio

    def collect_chunks_custom(self, speech_timestamps, audio_data):
        segments = []
        for i in range(len(speech_timestamps)):
            start = speech_timestamps[i]['start']
            end = speech_timestamps[i]['end']
            segments.append(audio_data[:, start: end])
        segments_flattened = torch.cat(segments, axis=1)
        if len(segments_flattened) > 0:
            return segments_flattened
        else:
            return audio_data


class AddGaussianWhiteNoise:
    """
    Used to add white noise to a signal to reach a given snr_db
    If the average_signal_power is not given then calculate it
    ...

    Attributes
    ----------
    snr_db : int
        target snr_db
    average_signal_power : int
        used to ensure the same noise power is added across multiple signals

    """

    def __init__(self, snr_db, average_signal_power=None):
        self.snr_db = snr_db
        self.average_signal_power = average_signal_power

    def __call__(self, audio_data):
        if self.average_signal_power is None:
            self.average_signal_power = utils.get_average_power(audio_data)

        snr = np.power(10, self.snr_db / 10)
        noise_power = self.average_signal_power / snr
        noise = torch.tensor(np.random.normal(0, np.sqrt(noise_power), audio_data.size(1))[np.newaxis, ...])
        # if not np.isclose(utils.snr_matlab(audio_data, noise), self.snr_db, atol=0.5):
        #     diff = utils.snr_matlab(audio_data, noise) - self.snr_db
        #     print('target snr db not met!', diff.item())
        return audio_data + noise


# TODO: this needs to be updated - create a base add-noise class and then expand to:
# [1] add gaussian noise - done
# [2] add noise from audio
# [3] add noise from file name
# [4] add noise random noise from list

class AddNoiseFile:
    def __init__(self, snr_db, average_signal_power=None, sample_rate=config.SAMPLE_RATE, noise_data=None,
                 verbose=False):
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.average_signal_power = average_signal_power
        self.noise_data = noise_data
        self.verbose = verbose
        # possibly used
        self.noise_files = config.noise_db
        self.noise_files_list = list(pathlib.Path(self.noise_files).glob('**/*.wav'))

    def get_noise_list(self):
        return self.noise_files_list


class AddNoiseFromFile(AddNoiseFile):
    def __init__(self, snr_db, noise_dir, average_signal_power=None, sample_rate=config.SAMPLE_RATE, noise_data=None,
                 verbose=False):
        super().__init__(snr_db, average_signal_power, sample_rate, noise_data, verbose)
        self.noise_dir = noise_dir
        self.noise_data = self.noise_file_to_audio(noise_dir)

    def noise_file_to_audio(self, noise_dir):
        effects = [
            ['remix', '1'],  # convert to mono
            ['rate', str(self.sample_rate)],  # resample
            ['gain', '-n']  # normalises to 0dB
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(noise_dir, effects, normalize=True)
        return noise

    def __call__(self, audio_data):
        if self.average_signal_power is None:
            if self.verbose:
                print('calculating average_signal_power')
            self.average_signal_power = utils.get_average_power(audio_data)

        noise_data_trimmed = None

        audio_length = audio_data.shape[-1]
        noise_length = self.noise_data.shape[-1]
        # add in noise randomly if longer, or if shorter than zero pad end
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise_data_trimmed = self.noise_data[..., offset:offset + audio_length]
        elif noise_length < audio_length:
            warnings.warn('noise_length < audio_length')
            noise_data_trimmed = torch.cat(
                [self.noise_data, torch.zeros((self.noise_data.shape[0], audio_length - noise_length))], dim=-1)

        snr = np.power(10, self.snr_db / 10)
        noise_target_power = self.average_signal_power / snr
        noise_power = utils.get_average_power(noise_data_trimmed)
        scale = noise_target_power / noise_power
        noise_data_trimmed = noise_data_trimmed * np.sqrt(scale)  # sqrt since power
        if self.verbose:
            print('target snr and calculated snr: ', self.snr_db, utils.snr_matlab(audio_data, noise_data_trimmed).item())
        if not np.isclose(utils.snr_matlab(audio_data, noise_data_trimmed), self.snr_db, atol=0.5):
            warnings.warn('target snr db not met!')
        return audio_data + noise_data_trimmed


# class AddNoiseFromFile:
#     def __init__(self, snr_db, noise=None, noise_file_index=None, average_signal_power=None, sample_rate=config.SAMPLE_RATE):
#         self.sample_rate = sample_rate
#         self.snr_db = snr_db
#         self.noise_file_index = noise_file_index
#         self.average_signal_power = average_signal_power
#         self.noise_files = r'/Users/david/Documents/data/MUSAN/musan/noise/free-sound'
#         self.noise_files_list = list(pathlib.Path(self.noise_files).glob('**/*.wav'))
#
#         if not os.path.exists(self.noise_files):
#             raise IOError(f'Noise directory `{self.noise_files}` does not exist')
#
#         if self.noise_file_index is None:
#             self.noise_file_index = random.randint(0, len(self.noise_files_list))
#
#         noise_file_path = self.noise_files_list[self.noise_file_index]
#
#         compose_transform = ComposeTransform([
#             FileToTensor(),
#             NormalizeSox(),
#         ])
#
#         self.transformed_noise = compose_transform(noise_file_path)
#
#     def plot_play_noise(self):
#         tio.plot_waveform(self.transformed_noise)
#         tio.play_audio(self.transformed_noise)
#
#     def __call__(self, audio_data):
#         if self.noise_file_index is None:
#             self.noise_file_index = random.randint(0, len(self.noise_files_list))
#
#         noise_file_path = self.noise_files_list[self.noise_file_index]
#
#         print(noise_file_path)
#
#         # effects = [
#         #     ['remix', '1'],  # convert to mono
#         #     ['rate', str(self.sample_rate)],  # resample
#         # ]
#         # noise, _ = torchaudio.sox_effects.apply_effects_file(noise_file_path, effects, normalize=True)
#         # audio_length = audio_data.shape[-1]
#         # noise_length = noise.shape[-1]
#         # # add in noise randomly if longer, or if shorter then zero pad end
#         # if noise_length > audio_length:
#         #     offset = random.randint(0, noise_length - audio_length)
#         #     noise = noise[..., offset:offset + audio_length]
#         # elif noise_length < audio_length:
#         #     noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)
#         #
#         # snr_db = random.randint(self.min_snr_db, self.max_snr_db)
#         # snr = math.exp(snr_db / 10)
#         # audio_power = audio_data.norm(p=2)
#         # noise_power = noise.norm(p=2)
#         # scale = snr * noise_power / audio_power
#         # audio_power / noise_power = snr / scale
#         #
#         # return (scale * audio_data + noise) / 2

class AddReverb:
    def __init__(self, sample_rate=config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        sox_effects = [
            ["reverb", "-w"],
            ['remix', '1'],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            tensor=audio_data, sample_rate=self.sample_rate, effects=sox_effects)
        return transformed_audio


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

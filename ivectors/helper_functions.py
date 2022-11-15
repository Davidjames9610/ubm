from python_speech_features import mfcc, delta
import numpy as np
from importlib import reload
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav
import misc.vad as vad

eps = np.finfo(np.float64).eps


# extract features

def fe(signal, sample_rate):
    mfcc_feats = mfcc(signal=signal, samplerate=sample_rate, ceplifter=0, preemph=0, appendEnergy=False,
                      winfunc=np.hanning, numcep=10, winlen=0.02, nfft=1024)
    # mfcc_feats_delta = delta(mfcc_feats, 9)
    # mfcc_feats_delta_delta = delta(mfcc_feats_delta, 9)
    # mfcc_complete = np.concatenate((mfcc_feats, mfcc_feats_delta, mfcc_feats_delta_delta), axis=1)
    return mfcc_feats


def helper_feature_extraction(raw_audio_file, norm=None):
    # read in file
    (signal_rate, signal) = wav.read(raw_audio_file)

    # normalise
    signal = signal / max(signal)

    # detect speech
    v = vad.VoiceActivityDetector(signal_rate, signal, 0.8)
    detected = v.detect_speech()
    idx, maxed_idx = v.convert_windows_to_readible_labels(detected)

    # extract and concatenate features
    if idx:
        feats = []
        for i in range(len(idx)):
            feat = fe(signal[idx[i][0]:idx[i][1]], signal_rate)
            feats.append(feat)
        mfcc_feats = np.concatenate(feats, axis=0)

        # feature normalisation and Cepstral mean subtraction (for channel noise)
        if norm:
            mfcc_feats = (mfcc_feats - norm.means) / norm.std
            mfcc_feats = mfcc_feats - np.mean(mfcc_feats)
            return mfcc_feats
        else:
            return mfcc_feats
    else:
        return []

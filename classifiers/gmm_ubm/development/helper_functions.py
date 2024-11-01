from sklearn.mixture import GaussianMixture, _gaussian_mixture
from scipy.special import logsumexp
import numpy as np
import librosa
import scipy.io.wavfile as wav
import whale.setup.constants as const

eps = np.finfo(np.float64).eps


class MyGmm:
    def __init__(self, num_comp):
        self.gmm = GaussianMixture(n_components=num_comp, covariance_type='diag')

    def get_gmm(self):
        return self.gmm

    def set_gmm(self, some_gmm):
        self.gmm = some_gmm

    def set_gmm_params(self, w, m, c):
        # quick check
        num_comp = self.gmm.n_components
        if w.shape[0] != num_comp or m.shape[0] != num_comp or c.shape[0] != num_comp:
            print('somethings wrong')
        else:
            self.gmm.weights_ = w
            self.gmm.means_ = m
            self.gmm.covariances_ = c
            self.gmm.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(c, 'diag')


def helper_expectation(feature, ubm: GaussianMixture):
    post = ubm._estimate_weighted_log_prob(feature)
    l = logsumexp(post, axis=1)
    lv2 = helper_log_sum_exp(post)

    gamma = np.exp((post.T - l)).T
    n = np.sum(gamma, axis=0)
    f = np.dot(feature.T, gamma)
    s = np.dot(np.multiply(feature.T, feature.T), gamma)
    l = np.sum(l)

    return n, f, s, l


def helper_maximization(n, f, s, nc):
    n = np.maximum(n, eps)
    new_gmm = MyGmm(num_comp=nc)
    weights = np.maximum(n / np.sum(n), eps)
    weights = np.squeeze(weights / np.sum(weights))
    means = (f / n)
    covars = np.maximum((s / n) - np.square(means), eps)
    new_gmm.set_gmm_params(w=weights.T, m=means.T, c=covars.T)

    return new_gmm.get_gmm()


def helper_log_sum_exp(x):
    x = x.T
    a = np.max(x, axis=0)
    y = a + np.sum(np.exp(x - a), axis=0)
    return y


def get_norm_factors(all_features):
    means = []
    std = []
    for feature in all_features:
        means.append(np.mean(feature, axis=0))
        std.append(np.std(feature, axis=0))

    means = np.array(means)
    means = np.mean(means, axis=0)

    std = np.array(std)
    std = np.mean(std, axis=0)

    class NormFactor:
        def __init__(self, m, s):
            self.means = m
            self.std = s

    return NormFactor(means, std)


# helper functions

# [2] feature extraction
# 	• Normalize the audio
# 	• Use detectSpeech to remove nonspeech regions from the audio
# 	• Extract features from the audio
# 	• Normalize the features
#   * Apply cepstral mean normalization


def helper_feature_extraction(raw_audio_file, norm=None):
    # read in file
    (signal_rate, signal) = wav.read(raw_audio_file)

    # normalise
    signal = librosa.util.normalize(signal)

    # detect / vad, not working currently

    # fe
    mfcc = librosa.feature.mfcc(y=signal, sr=const.SAMPLING_RATE, n_mfcc=13, n_fft=1024).T

    delta = librosa.feature.delta(mfcc, width=3)

    mfcc_feats = np.concatenate([mfcc, delta], axis=1)

    # feature normalisation and cepstral mean subtraction (for channel noise)
    if norm:
        mfcc_feats = (mfcc_feats - norm.means) / norm.std
        mfcc_feats = mfcc_feats - np.mean(mfcc_feats)
        return mfcc_feats
    else:
        return mfcc_feats

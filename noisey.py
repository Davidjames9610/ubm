import math
import numpy as np
import os
import os.path

from hmmlearn.hmm import GaussianHMM
from scipy.io import wavfile

## https://en.wikipedia.org/wiki/Signal-to-noise_ratio#:~:text=SNR%20is%20defined%20as%20the,by%20the%20Shannon%E2%80%93Hartley%20theorem.
def addNoiseToData(data, snr):
    n_signal_l = []
    n_l = []
    for sample in data:
        x_watts = sample ** 2
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        target_snr_db = snr
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate a sample of white noise
        mean_noise = 0
        n_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts)) # nb
        # Noise up the original signal
        n_signal = sample + n_volts
        n_signal_l.append(n_signal)
        n_l.append(n_volts)
    return n_signal_l, n_l

def get_signal_avg_db(signal):
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    return sig_avg_db

def get_noise_power_given_signal_avg_db(sig_avg_db, snr):
    target_snr_db = snr
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    return noise_avg_watts

def get_noise_avg_watts(data, snr):
    x_watts = data ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    target_snr_db = snr
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    return noise_avg_watts

# copy ?
def getNoiseAvgWatts(sample, snr):
    x_watts = sample ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    target_snr_db = snr
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    return noise_avg_watts

def addNoiseToDataAtEnds(samples, fs, noise_avg_watts):
    n_signal_l = []
    n_signal_silence = []
    for sample in samples:
        n_silence = np.zeros(math.floor(fs*0.1))
        new_signal = np.concatenate((n_silence, sample))
        new_signal = np.concatenate((new_signal, n_silence))
        n_volts = np.random.normal(0, np.sqrt(noise_avg_watts), (len(new_signal)))
        n_signal = new_signal + n_volts
        n_signal_l.append(n_signal)
        n_signal_silence.append(new_signal)
    return n_signal_l, n_signal_silence

# create random 3 state Gaussian noise given length and power
def generate_gaussian_noise(num_samples, power_a, power_b, power_c):
    # Define the parameters of the HMM
    startprob = np.array([0.3, 0.3, 0.4])
    transmat = np.array([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005], [0.0005,0.0005, 0.999]])  # Transition matrix
    means = np.array([[0.0], [0.0], [0.0]])  # Mean values for each state
    covars = np.array([[[power_a]],[[power_b]],[[power_c]]])  # Covariance matrices for each state

    # Create a two-state HMM
    model = GaussianHMM(n_components=len(startprob), covariance_type="full", n_iter=100)
    model.n_features = 1
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # Generate samples from the HMM
    noise, states = model.sample(num_samples)

    return noise.flatten(), states

def addTimeDomain(s1, s2, loca=0.5):
    # loca = location that sample02 is added to sample01
    insertSample = int(np.floor(len(s1) * loca))
    s3_len = insertSample + len(s1)
    s1_zeros = np.zeros(s3_len - len(s1))
    s1 = np.concatenate((s1, s1_zeros))
    s2_zeros = np.zeros(s3_len - len(s2))
    s2 = np.concatenate((s2_zeros, s2))
    s3 = s1 + s2
    return s3


def getNoiseMatrix(features, length, mean):
    mean = np.ones(features) * mean
    cov = np.eye(features)
    matrix = np.random.multivariate_normal(mean, cov, length).T
    return matrix

def __fast_scandir__(dirname):
    fpaths = []
    labels = []
    for filename in os.listdir(dirname):
      if filename.endswith(".wav"):
        # print(os.path.join(dirname + filename))
        fpaths.append(dirname + '/' + filename)
        currentLabel = filename
        labels.append(currentLabel)
      else:
        continue
    return fpaths, labels

def getAllData():
    fpaths, labels = __fast_scandir__("/Users/davidedwards/Documents/Masters/AADC/Python /utils/noise_data")

    data = []
    f = 0
    for n, file in enumerate(fpaths):
        f, d = wavfile.read(file)
        data.append(d)

    return data, labels, f

if __name__ == '__main__':
    data, labels, f = getAllData()





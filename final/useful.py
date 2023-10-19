import librosa
import torchaudio
from hmmlearn.hmm import GaussianHMM
from spafe.utils import vis
import numpy as np

# vis features
def vis_helper(feature, title=""):
    vis.show_features(feature, title, 'y', 'x', figsize=(6, 3), cmap="viridis")


# normalize audio and return
def file_to_audio(noise_dir, sample_rate):
    effects = [
        ['remix', '1'],  # convert to mono
        ['rate', str(sample_rate)],  # resample
        ['gain', '-n']  # normalises to 0dB
    ]
    noise, sr = torchaudio.sox_effects.apply_effects_file(noise_dir, effects, normalize=True)
    return noise.numpy().flatten(), sr


# get samples and labels from get_data_dictionary
def find_key_by_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
def get_samples(get_data_dictionary, sig_dict, label_dict):
    samples = []
    labels = []
    for key in sig_dict:
        # print('getting samples for: ', key)
        get_data_array = get_data_dictionary[key]
        for get_data_inst in get_data_array:
            for i in range(len(get_data_inst.annotations)):
                annot = get_data_inst.annotations[i]
                sample = get_data_inst.audio[annot.start:annot.end]
                samples.append(sample)
                labels.append(label_dict[key])

    unique_elements, counts = np.unique(labels, return_counts=True)
    print('__collected samples__')
    for i in range(len(unique_elements)):
        print(find_key_by_value(label_dict, unique_elements[i]),': ', counts[i])
    return samples, labels

# some fe methods
def get_log_power_feature(sample, nfft):
    return np.log(np.square(np.abs(librosa.stft(sample, n_fft=nfft)).T))


# methods for comparing deleting states in a hmm

import scipy.stats as st
def distributions_js(distribution_p, distribution_q, n_samples=10 ** 5):
    # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
    # all the logarithms are defined as log2 (because of information entrophy)
    X = distribution_p.rvs(n_samples)
    p_X = distribution_p.pdf(X)
    q_X = distribution_q.pdf(X)
    log_mix_X = np.log2(p_X + q_X)

    Y = distribution_q.rvs(n_samples)
    p_Y = distribution_p.pdf(Y)
    q_Y = distribution_q.pdf(Y)
    log_mix_Y = np.log2(p_Y + q_Y)

    return (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
            + np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.rvs(n_samples)
    log_p_X = gmm_p.logpdf(X)
    log_q_X = gmm_q.logpdf(X)
    return log_p_X.mean() - log_q_X.mean()

def find_similar_states_kl(hmm1, hmm2):
    # Adjust the threshold based on your similarity criterion

    dict_js = np.zeros((hmm1.n_components,hmm2.n_components))
    # Iterate through states in hmm_whale
    for i in range(hmm1.n_components):
        # Iterate through states in hmm_noise
        for j in range(hmm2.n_components):
            # Calculate KL divergence between the distributions of means and covariances
            gmm_1 = st.multivariate_normal(hmm1.means_[i],hmm1.covars_[i])
            gmm_2 = st.multivariate_normal(hmm2.means_[j],hmm2.covars_[j])

            kl_div = gmm_kl(gmm_1, gmm_2)
            dict_js[i,j]  = kl_div
    return dict_js

def delete_component(hmm1, component_to_delete):
    # Remove the component_to_delete
    print('deleting comp', component_to_delete)
    new_hmm = GaussianHMM(n_components=hmm1.n_components - 1, covariance_type=hmm1.covariance_type)
    new_hmm.n_features = hmm1.n_features
    transmat_ = np.delete(hmm1.transmat_, component_to_delete, axis=0)
    transmat_ = np.delete(transmat_, component_to_delete, axis=1)
    new_hmm.transmat_ = transmat_ / np.sum(transmat_, axis=1, keepdims=True)  # Normalization along axis=1

    startprob_ = np.delete(hmm1.startprob_, component_to_delete)
    new_hmm.startprob_ = startprob_ / np.sum(startprob_, axis=0, keepdims=True)  # Normalization along axis=1

    new_hmm.means_ = np.delete(hmm1.means_, component_to_delete, axis=0)

    covars_ = np.delete(hmm1.covars_, component_to_delete, axis=0)

    new_hmm.covars_ = np.array([np.diag(i) for i in covars_])

    return new_hmm

# accuracy measures
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    return_metrics = {
        'acc': ACC,
        'tpr': TPR,
        'fpr': FPR,
    }

    return return_metrics

import final.models.decode_combine as dc
class SampleHolder:
    def __init__(self, samples, sample_labels, features=None, feature_labels=None):
        if feature_labels is None:
            feature_labels = []
        if features is None:
            features = []
        self.samples = samples  # n x t x 1
        self.sample_labels = sample_labels  # n x 1
        self.features = features    # n x ft x n_features
        self.feature_labels = feature_labels # n x ft
        self.log_prob = None
        self.y_pred = None

    def update_feature_labels(self):
        if len(self.features) > 0:
            self.feature_labels = []
            for i in range(len(self.features)):
                self.feature_labels.append(np.ones(len(self.features[i])) * self.sample_labels[i])

    def update_decode(self, output):
        self.y_pred = output.y_pred
        self.log_prob = output.log_prob



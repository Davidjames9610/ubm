import librosa
import torchaudio
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
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

            kl_div = gmm_kl(gmm_1, gmm_2, 10**2)
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
    if np.sum(startprob_) == 0:
        startprob_[np.argmax(np.sum(new_hmm.transmat_, axis=0))] = 1
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


## plot spectogram with annotations

def find_label_changes(labels):
    changes = []
    changes.append((0, labels[0]))
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            changes.append((i, labels[i]))
    return changes

def plot_spectrogram(features, true_labels, pred_labels, label_type, label_abr):

    times = np.arange(features.shape[0])
    frequencies = np.arange(features.shape[1])
    freq_max = features.shape[1]
    true_changes = find_label_changes(true_labels)

    pred_changes = find_label_changes(pred_labels)

    for index, label in true_changes:
        t = plt.text(times[index + 8], frequencies[-7], label_abr[label], color='black', fontsize=10, verticalalignment='bottom')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=0))
        plt.vlines(times[index], ymin=freq_max / 2, ymax=freq_max, color='black', linestyles='dashed', linewidth=1)

    for index, label in pred_changes:
        if(index + 8 < len(times)):
            t = plt.text(times[index + 8], frequencies[3], label_abr[label], color='red', fontsize=10, verticalalignment='bottom')
        else:
            t = plt.text(times[index], frequencies[3], label_abr[label], color='red', fontsize=10,
                         verticalalignment='bottom')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=0))
        plt.vlines(times[index], ymin=0, ymax=freq_max / 2, color='red', linestyles='solid', linewidth=2)

    plt.pcolormesh(features.T, cmap='viridis')
    plt.ylabel('F bin')
    plt.xlabel('T bin')
    legend_labels = [label_abr[label] + ' - ' + label_type[label].title() for label in label_type]
    plt.legend(legend_labels, loc='upper right', facecolor='white', framealpha=1, handlelength=0)
    plt.colorbar(label='Intensity [dB]')
    plt.title('Annotated Spectrogram')
    plt.show()

def smooth_labels(labels, window_size=50, step_size=10, diff_size=20):

    smoothy_labels = labels.copy()
    arg_max = []
    arg_max_index = []

    for start in range(0, len(labels) - window_size + 1, step_size):
        end = start + window_size
        window = labels[start:end]

        unique_elements, counts = np.unique(window, return_counts=True)
        max_count_index = np.argmax(counts)
        dominant_label = unique_elements[max_count_index]
        arg_max.append(dominant_label)
        arg_max_index.append(start)

    changes = []
    changes_index = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            changes.append(labels[i])
            changes_index.append(i)

    fwd = changes[2]
    curr = changes[1]
    prev = changes[0]
    for i in range(2, len(changes)):
        fwd = changes[i]
        curr = changes[i - 1]
        prev = changes[i - 2]
        index_curr = changes_index[i - 1]
        index_fwd = changes_index[i]
        index_prev = changes_index[i - 2]
        diff = index_fwd - index_curr
        # two cases, if quick switch back to original state then just set state to backwards
        if prev != curr:
            if diff < diff_size and prev == fwd:
                smoothy_labels[index_curr: index_fwd] = np.ones(diff) * prev
                changes[i - 1] = prev
            elif diff < diff_size and prev != fwd and curr != fwd:
                arg_max_index_cur = np.argmin(np.abs(np.array(arg_max_index) - index_curr))
                arg_max_index_fwd = np.argmin(np.abs(np.array(arg_max_index) - index_fwd))
                if (arg_max[arg_max_index_cur] == arg_max[arg_max_index_fwd]):
                    smoothy_labels[index_curr: index_fwd] = np.ones(diff) * fwd
                    changes[i - 1] = fwd
    return smoothy_labels

def add_tiny_amount(matrix, tiny_amount=1e-5):
    # Add tiny_amount to elements less than or equal to 0
    matrix = np.where(matrix <= 0, matrix + tiny_amount, matrix)
    return matrix
def normalize_matrix(matrix):
    matrix = add_tiny_amount(matrix)
    return matrix / np.sum(matrix, axis=(matrix.ndim -1), keepdims=True)

import librosa
import torchaudio
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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
def distributions_js(distribution_p, distribution_q, n_samples=10**5):
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

def find_similar_states_js(hmm1, hmm2, n_samples=10**5):

    dict_js = np.zeros((hmm1.n_components,hmm2.n_components))
    # Iterate through states in hmm_whale
    for i in range(hmm1.n_components):
        # Iterate through states in hmm_noise
        for j in range(hmm2.n_components):
            # Calculate KL divergence between the distributions of means and covariances
            gmm_1 = st.multivariate_normal(hmm1.means_[i],hmm1.covars_[i])
            gmm_2 = st.multivariate_normal(hmm2.means_[j],hmm2.covars_[j])

            kl_div = distributions_js(gmm_1, gmm_2, n_samples)
            if np.isclose(kl_div, 0) or kl_div < 0:
                kl_div = 1e-5
            dict_js[i,j] = kl_div
    return np.log(dict_js)

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.rvs(n_samples)
    log_p_X = gmm_p.logpdf(X)
    log_q_X = gmm_q.logpdf(X)
    return log_p_X.mean() - log_q_X.mean()

def find_similar_states_kl(hmm1, hmm2, n_samples=10**5):
    # Adjust the threshold based on your similarity criterion

    dict_js = np.zeros((hmm1.n_components,hmm2.n_components))
    # Iterate through states in hmm_whale
    for i in range(hmm1.n_components):
        # Iterate through states in hmm_noise
        for j in range(hmm2.n_components):
            # Calculate KL divergence between the distributions of means and covariances
            gmm_1 = st.multivariate_normal(hmm1.means_[i],hmm1.covars_[i])
            gmm_2 = st.multivariate_normal(hmm2.means_[j],hmm2.covars_[j])

            kl_div = gmm_kl(gmm_1, gmm_2, n_samples)
            dict_js[i,j] = kl_div
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

def delete_component_indicies(original_hmm: GaussianHMM, zero_indices, verbose=False):
    # Remove the component_to_delete
    if verbose:
        print('deleting comp', zero_indices)

    # new_hmm = GaussianHMM(n_components=hmm1.n_components - 1, covariance_type=hmm1.covariance_type)
    # new_hmm.n_features = hmm1.n_features

    rem_ind = zero_indices.astype(int)
    A = original_hmm.transmat_[rem_ind][:, rem_ind]
    pi = original_hmm.startprob_[rem_ind]
    means = original_hmm.means_[rem_ind]
    covar = original_hmm.covars_[rem_ind]

    new_hmm = GaussianHMM(A.shape[0], covariance_type='diag')
    new_hmm.n_features = means.shape[1]
    new_hmm.transmat_, new_hmm.startprob_, new_hmm.means_ = normalize_matrix(A), normalize_matrix(pi), means
    new_hmm.covars_ = np.array([np.diag(i) for i in covar])

    return new_hmm

# reverse the fhmm proces to create an estimate of the added noise hmm, returns updated means
def reverse_fhmm(hmm_noise: GaussianHMM, hmm_background: GaussianHMM, threshold=0):
    noise_means = []
    # assume noise was made up of max operation
    for i in range(hmm_noise.n_components):
        mean_i = hmm_noise.means_[i]
        curr_mean = []
        for j in range(hmm_background.n_components):
            mean_n = np.zeros(len(mean_i))
            mean_j = hmm_background.means_[j]
            m_mask = (mean_i > mean_j + threshold)  # add threshold
            m_mask_indicis = np.where(m_mask)
            mean_n[m_mask_indicis] = mean_i[m_mask_indicis]
            curr_mean.append(mean_n)
        reduced_mean = np.maximum.reduce(curr_mean)
        noise_means.append(reduced_mean)

    new_noise_means = np.stack(noise_means)
    zero_mask = np.where(new_noise_means == 0)
    new_noise_means[zero_mask] = np.NINF
    return new_noise_means

def calculate_average(list_of_dicts):
    result_dict = {}
    dict_count = len(list_of_dicts)

    if dict_count == 0:
        return result_dict  # Return an empty dictionary if the input list is empty

    # Iterate through each dictionary in the list
    for data_dict in list_of_dicts:
        # Iterate through each key-value pair in the dictionary
        for key, value in data_dict.items():
            # Check if the key is already present in the result_dict
            if key in result_dict:
                # If present, add the value to the running sum
                result_dict[key] += value
            else:
                # If not present, initialize the key in the result_dict
                result_dict[key] = value

    # Calculate the average for each key
    for key in result_dict.keys():
        result_dict[key] /= dict_count

    return result_dict

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
        'ACC': ACC,
        'PPV': PPV,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
    }

    return return_metrics


# average confusion matrice

def get_average_cm(conf_matrices):

    # Initialize a variable to store the sum of confusion matrices
    sum_conf_matrix = np.zeros(conf_matrices[0].shape)

    # Sum up all confusion matrices
    for conf_matrix in conf_matrices:
        sum_conf_matrix += conf_matrix

    # Calculate the average confusion matrix
    avg_conf_matrix = sum_conf_matrix / len(conf_matrices)

    return avg_conf_matrix

import numpy as np
from sklearn.metrics import confusion_matrix

def perf_measure_multi(y_actual, y_hat, avg_only=False):
    # Initialize dictionaries to store metrics for each class
    metrics_dict = {i: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for i in set(y_actual)}
    return_dict = {i: {} for i in set(y_actual)}

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_actual, y_hat)

    for i in range(len(set(y_actual))):
        for j in range(len(set(y_actual))):
            if i == j:
                metrics_dict[i]['TP'] = conf_matrix[i, j]
            else:
                metrics_dict[i]['FN'] += conf_matrix[i, j]
                metrics_dict[j]['FP'] += conf_matrix[i, j]
                metrics_dict[i]['TN'] += np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, j])

    # Initialize variables for macro-average
    macro_avg_dict = {'ACC': 0, 'PPV': 0, 'TPR': 0, 'TNR': 0, 'FPR': 0, 'FNR': 0}

    # Compute metrics for each class
    for i in metrics_dict:
        TP = metrics_dict[i]['TP']
        FP = metrics_dict[i]['FP']
        TN = metrics_dict[i]['TN']
        FN = metrics_dict[i]['FN']

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        # Update macro-average variables
        macro_avg_dict['ACC'] += ACC
        macro_avg_dict['PPV'] += PPV
        macro_avg_dict['TPR'] += TPR
        macro_avg_dict['TNR'] += TNR
        macro_avg_dict['FPR'] += FPR
        macro_avg_dict['FNR'] += FNR

        # Add metrics to the class-specific dictionary
        metrics_dict[i].update({
            'ACC': ACC,
            'PPV': PPV,
            'TPR': TPR,
            'TNR': TNR,
            'FPR': FPR,
            'FNR': FNR,
        })

        return_dict[i].update({
            'ACC': ACC,
            'PPV': PPV,
            'TPR': TPR,
            'TNR': TNR,
            'FPR': FPR,
            'FNR': FNR,
        })

    # Calculate macro-average
    num_classes = len(metrics_dict)
    macro_avg_dict = {key: value / num_classes for key, value in macro_avg_dict.items()}

    return_dict['macro_avg'] = macro_avg_dict

    if avg_only:
        return macro_avg_dict
    else:
        return return_dict



def perf_measure_multi_all(y_actual, y_hat):
    # Initialize dictionaries to store metrics for each class
    metrics_dict = {i: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for i in set(y_actual)}
    return_dict = {i: {} for i in set(y_actual)}

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_actual, y_hat)

    for i in range(len(set(y_actual))):
        for j in range(len(set(y_actual))):
            if i == j:
                metrics_dict[i]['TP'] = conf_matrix[i, j]
            else:
                metrics_dict[i]['FN'] += conf_matrix[i, j]
                metrics_dict[j]['FP'] += conf_matrix[i, j]
                metrics_dict[i]['TN'] += np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, j])

    # Compute metrics for each class
    for i in metrics_dict:
        TP = metrics_dict[i]['TP']
        FP = metrics_dict[i]['FP']
        TN = metrics_dict[i]['TN']
        FN = metrics_dict[i]['FN']

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        # Add metrics to the dictionary
        metrics_dict[i].update({
            'ACC': ACC,
            'PPV': PPV,
            'TPR': TPR,
            'TNR': TNR,
            'FPR': FPR,
            'FNR': FNR,
        })

    return metrics_dict

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
        t = plt.text(times[index], frequencies[-7], label_abr[label], color='black', fontsize=10, verticalalignment='bottom')
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
    # plt.colorbar(label='Intensity [dB]')
    # plt.title('Annotated Spectrogram')
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

def get_colors():
    colors = [
        (0, 0, 255),  # Blue
        (0, 255, 0),  # Green
        (255, 0, 0),  # Red
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (255, 165, 0),  # Orange
        (70, 130, 180),  # Steel Blue
        (0, 128, 0),  # Green (Dark Green)
        (255, 69, 0),  # Red-Orange
        (30, 144, 255),  # Dodger Blue
        (255, 20, 147),  # Deep Pink
        (50, 205, 50),  # Lime Green
        (219, 112, 147),  # Pale Violet Red
        (0, 255, 127),  # Spring Green
        (135, 206, 250),  # Light Sky Blue
        (255, 99, 71)  # Tomato
    ]

    # Normalizing the values to the range [0, 1]
    colors_normalized = [(r / 255, g / 255, b / 255) for r, g, b in colors]
    return colors_normalized

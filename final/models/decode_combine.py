"""
class for combining x amount of hmms into a larger model,
at the moment only works for two, change a matrix calculation if needed.

"""

# from hmm import myHmm
# from hmm.general_hmm_utils import normalise_a_matrix, calculate_emission_matrix, calculate_emission_matrix_gmm, viterbi_algorithm
from hmmlearn.hmm import GaussianHMM, GMMHMM
import numpy as np
from scipy.stats import multivariate_normal as mvn

import collections

def get_accuracy_from_decoder(labels, decoder_output, correct_state):
    correct_state = correct_state * 1.0
    accuracy_array = []
    for index in range(len(labels)):
        start, end = labels[index].start_fs, labels[index].end_fs
        guess = decoder_output[start: end]
        count = collections.Counter(guess)
        accuracy_array.append(count[correct_state] / len(guess) * 100)
    return accuracy_array

class DecodeCombineBase:

    def __init__(self, array_of_hmms: [GaussianHMM]):
        self.models: [GaussianHMM] = array_of_hmms
        self.n_models = len(self.models)
        total_states = 0
        for i in range(self.n_models):
            total_states += self.models[i].n_components
        self.total_states = total_states
        self.a = self.__calculate_a_matrix()
        self.pie = self.__calculate_pie_matrix()
        self.map_states = self.__calculate_map_states()

    def __str__(self):
        return f"class for combining hmm models for decoding"

    def __calculate_map_states(self):
        map_states = {}
        z = 0
        for m in range(self.n_models):
            curr_model = self.models[m]
            for v in range(curr_model.n_components):
                map_states[z] = m
                z += 1
        return map_states

    def __calculate_a_matrix(self):
        noise_states = self.models[-1].n_components
        a = np.zeros((self.total_states, self.total_states))
        x, y = 0, 0
        for i in range(self.n_models):
            cur_model = self.models[i]
            states = cur_model.n_components
            a[x:x + states, y: y + states] = cur_model.transmat_
            if y + states < self.total_states:
                a[self.total_states - 1, x] = 1
                a[x + states - 1, self.total_states - noise_states] = 1
        #     if y + states < self.total_states:
        #         a[x + states - 1, y + states] = 1
            x += states
            y += states
        # a[self.total_states - 1, 0] = 1
        a = normalise_a_matrix(a)
        return a

    def __calculate_pie_matrix(self):
        pies = []
        for z in range(self.n_models):
            pies.append(self.models[z].startprob_)

        pie = np.concatenate(pies)
        pie = pie + 1e-10
        pie /= pie.sum()

        return pie

    def __calculate_combined_emission_matrix(self, data):
        pass

    def decode(self, data):
        pass

class DecodeCombineGaussian(DecodeCombineBase):
    def __init__(self, array_of_hmms: [GaussianHMM]):
        super().__init__(array_of_hmms)

    def __calculate_combined_emission_matrix(self, data):
        log_bs = []
        for z in range(self.n_models):
            log_bs.append(calculate_emission_matrix(data, self.models[z]))
        return np.concatenate(log_bs, axis=0)

    def decode(self, data):
        total_t = len(data)
        print('calculating emission matrix')
        log_b = self.__calculate_combined_emission_matrix(data)

        # perform Viterbi
        print('viterbi_algorithm')
        states, log_prob = viterbi_algorithm(self.pie, self.a, log_b, self.total_states, total_t)

        labels = np.zeros(total_t)
        for z in range(len(states)):
            labels[z] = self.map_states[states[z]]

        return states, np.array(labels, dtype=int), log_prob

def normalise_a_matrix(a_matrix, decimal_n=0.0001):
    rows = a_matrix.shape[0]

    for z in range(rows):
        for j in range(rows):
            if a_matrix[z, j] < 0.1:
                a_matrix[z, j] = decimal_n

    norm = np.abs(a_matrix).sum(axis=1)

    new_a = np.zeros((rows, rows))
    for z in range(rows):
        for j in range(rows):
            new_a[z, j] = a_matrix[z, j] * (1 / norm[z])
    return new_a


def calculate_emission_matrix(data, model: GaussianHMM):
    total_t = len(data)

    states = model.n_components
    means = model.means_
    covars = model.covars_

    # make the emission matrix B
    log_b = np.zeros((states, total_t))
    for j in range(states):
        for t in range(total_t):
            p = mvn.logpdf(data[t], means[j], covars[j])
            log_b[j, t] += p
    return log_b


def calculate_emission_matrix_gmm(data, model: GMMHMM):
    total_t = len(data)

    states = model.n_components
    means = model.means_
    covars = model.covars_
    mix = model.weights_
    n_mix = model.n_mix

    # make the emission matrix B
    log_b = np.zeros((states, total_t))
    for j in range(states):
        for t in range(total_t):
            for k in range(n_mix):
                p = np.log(mix[j, k]) + mvn.logpdf(data[t], means[j, k], covars[j, k])
                log_b[j, t] += p
    return log_b


def viterbi_algorithm(pie, a, log_b, total_states, total_t):
    delta = np.zeros((total_t, total_states))
    psi = np.zeros((total_t, total_states))

    # populate first row of delta
    for s in range(total_states):
        delta[0, s] = np.log(pie[s]) + log_b[s, 0]

    for t in range(1, total_t):
        for s in range(total_states):
            ao1_row = a[:, s]
            next_delta = delta[t - 1] + np.log(ao1_row)
            delta[t, s] = np.max(next_delta) + log_b[s, t]
            psi[t, s] = np.argmax(next_delta)

    # backtrack
    states = np.zeros(total_t, dtype=np.int32)
    states[total_t - 1] = np.argmax(delta[total_t - 1])
    for t in range(total_t - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    log_prob = delta[-1, states[-1]]

    return states, log_prob

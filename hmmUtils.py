# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:55:55 2022

@author: David
"""

from hmmlearn import hmm
import numpy as np
from scipy.stats import multivariate_normal as mvn
# import librosa
# from python_speech_features import mfcc

def trainHmm(dataList, hmmObj: hmm):
    lengths = []
    for n, i in enumerate(dataList):
        lengths.append(len(i))
    hmmObj.fit(np.concatenate(dataList), lengths)


def scoreHmm(dataList, hmmObj: hmm):
    lengths = []
    if type(dataList) == list:
        for n, i in enumerate(dataList):
            lengths.append(len(i))
        return hmmObj.score(np.concatenate(dataList), lengths)
    else:
        return hmmObj.score(dataList)


def trainHmms(train_dict, n, mix):
    print("training hmm")
    hmms = {}
    for k in train_dict:
        newHmm = hmm.GMMHMM(n_components=n, n_mix=mix, covariance_type="diag", n_iter=100, verbose=True, tol=5)
        # newHmm = hmm.GaussianHMM(n_components=n)
        trainHmm(train_dict[k], newHmm)
        print("trained hmm", k)
        hmms[k] = newHmm
    return hmms


def scoreHmms(hmm_dict, test_dict):
    length = len(hmm_dict)
    scores = np.zeros(shape=(length, length))
    x = 0
    y = 0
    for k1 in hmm_dict:
        for k2 in test_dict:
            scores[x, y] = (scoreHmm(test_dict[k2], hmm_dict[k1]))
            y = y + 1
        y = 0
        x = x + 1
        print("scored hmm", k1)
    return scores


def getParams(hmm01):
    if isinstance(hmm01, hmm.GaussianHMM):
        return hmm01.transmat_, hmm01.n_components, hmm01.means_, hmm01.covars_, hmm01.startprob_
    else:
        return hmm01['transmat_'], hmm01['n_components'], hmm01['means_'], hmm01['covars_'], hmm01['startprob_']



def normalise_t_matrix_for_hmms(hmm_dict, decimal_n):
    for k1 in hmm_dict:
        normalise_t_matrix(decimal_n, hmm_dict[k1])

def normalise_t_matrix(decimal_n, hmm_to_normalise):
    hmm_apple_t_m = hmm_to_normalise.transmat_
    rows = hmm_apple_t_m.shape[0]

    for x in range(rows):
        for y in range(rows):
            if(hmm_apple_t_m[x, y] < 0.1):
                hmm_apple_t_m[x, y] = decimal_n

    norm = np.abs(hmm_apple_t_m).sum(axis=1)

    new_t = np.zeros((rows, rows))
    for x in range(rows):
        for y in range(rows):
                new_t[x, y] = hmm_apple_t_m[x, y] * (1/norm[x])

    hmm_to_normalise.transmat_ = new_t
    print(np.abs(hmm_to_normalise.transmat_).sum(axis=1))


def doViterbiAlgorithm(features, hmm01):
    # returns the most likely state sequence given observed sequence x
    # using the Viterbi algorithm

    T = len(features)
    a_matrix01, m01, means01, covars01, pi01 = getParams(hmm01)

    # make the emission matrix B
    logB = np.zeros((m01, T))
    for j in range(m01):
        for t in range(T):
            p = mvn.logpdf(features[t], means01[j], covars01[j])
            logB[j, t] += p

    # perform Viterbi
    delta = np.zeros((T, m01))
    psi = np.zeros((T, m01))

    pi01 = pi01 + 1e-10
    pi01 /= pi01.sum()

    # populate first row of delta
    for x in range(m01):
        delta[0, x] = np.log(pi01[x]) + logB[x, 0]

    for t in range(1, T):
        for x in range(m01):
            ao1_row = a_matrix01[:, x]
            next_delta = delta[t - 1] + np.log(ao1_row)
            delta[t, x] = np.max(next_delta) + logB[x, t]
            psi[t, x] = np.argmax(next_delta)

    # backtrack
    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    log_prob = delta[-1, states[-1]]
    return states, log_prob


def getMandC(j,k, m1, m2):
    mean_j = m1.means_[j]
    mean_k = m2.means_[k]
    m = np.maximum(mean_j, mean_k)
    m_mask = (mean_j > mean_k)
    covar_j = np.diag(m1.covars_[j])
    covar_k = np.diag(m2.covars_[k])
    c = np.where(m_mask, covar_j, covar_k)
    return m, np.diag(c)


def getProbMatrix(model01:hmm.GaussianHMM, model02:hmm.GaussianHMM, some_features):
    states01 = model01.n_components
    states02 = model02.n_components
    T_len = len(some_features)

    states_dict = []
    logBjk = np.zeros((states01, states02, T_len))
    for j in range(states01):
        for k in range(states02):
            m, c = getMandC(j,k,model01, model02) # combine m and c for models
            for t in range(T_len):
                p = mvn.logpdf(some_features[t], m, c)
                logBjk[j, k, t] += p
            states_dict.append([j, k])

    # print(np.concatenate(logBjk, axis = 0), 'logbjk')

    return np.concatenate(logBjk, axis = 0), states_dict


def compute_pi_fhmm(list_pi):
    """
    Input: list_pi: List of PI's of individual learnt HMMs
    Output: Combined Pi for the FHMM
    """
    result=list_pi[0]
    for i in range(len(list_pi)-1):
        result=np.kron(result,list_pi[i+1])
    return result


def kronecker_list(list_A):
    '''
    Input: list_pi: List of PI's of individual learnt HMMs
    Output: Combined Pi for the FHMM
    '''
    result=list_A[0]
    for i in range(len(list_A)-1):
        result=np.kron(result,list_A[i+1])
    return result

def doViterbiAlgorithmParallel(features, hmm01, hmm02):
    # returns the most likely state sequence given observed sequence x for fhmm of hmm01 and hmm02
    # assume features are log - power and therefore max mean is used per state
    # using the Viterbi algorithm

    T = len(features)
    a_01, n01, m01, c01, pi01 = getParams(hmm01)
    a_02, n02, m02, c02, pi02 = getParams(hmm02)

    # make the emission matrix B
    logB, states_dict = getProbMatrix(hmm01, hmm02, features)  # i x (j x t) matrix of observations evaluated at pdf of states
    t_states = len(states_dict)

    # init delta and psi
    delta = np.zeros((T, t_states))
    psi = np.zeros((T, t_states))

    # get combined pi and a
    pi_combined = kronecker_list([pi01, pi02])

    pi_combined = pi_combined + 1e-10
    pi_combined /= pi_combined.sum()

    a_combined = kronecker_list([a_01, a_02]) + 1e-10
    a_combined /= a_combined.sum(axis=1)

    # compute first row of delta
    for x in range(t_states):
        delta[0, x] = np.log(pi_combined[x]) + logB[x, 0]

    for t in range(1, T):
        for x in range(t_states):
            ao1_row = a_combined[:, x]
            next_delta = delta[t - 1] + np.log(ao1_row)
            delta[t, x] = np.max(next_delta) + logB[x, t]
            psi[t, x] = np.argmax(next_delta)

    # backtracks
    states_decoded = np.zeros(T, dtype=np.int32)
    states_decoded[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states_decoded[t] = psi[t + 1, states_decoded[t + 1]]
    log_prob = delta[-1, states_decoded[-1]]

    # split into separate:

    ss01 = []
    ss02 = []
    for x in range(len(states_decoded)):
        temp = states_dict[states_decoded[x]]
        ss01.append(temp[0])
        ss02.append(temp[1])

    return log_prob, [ss01, ss02]


def get_fhmm(hmm01: hmm.BaseHMM, hmm02: hmm.BaseHMM):

    a_01, n01, m01, c01, pi01 = getParams(hmm01)
    a_02, n02, m02, c02, pi02 = getParams(hmm02)

    # get combined pi and a
    pi_combined = kronecker_list([pi01, pi02])

    pi_combined = pi_combined + 1e-10
    pi_combined /= pi_combined.sum()

    a_combined = kronecker_list([a_01, a_02])
    a_combined /= a_combined.sum(axis=0)

    fhmm = hmm.GaussianHMM()
def normalized_matrix(d1, d2):
    x = np.ones((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

def normalize_matrix(x):
    return x / x.sum(axis=0, keepdims=True)

def combineTwoHmmsSimple(hmm01: hmm.GaussianHMM, hmm02: hmm.GaussianHMM):
    means_01 = hmm01.means_
    n_components01 = hmm01.n_components
    covars_01 = hmm01.covars_
    start_01 = hmm01.startprob_

    means_02 = hmm02.means_
    n_components02 = hmm02.n_components
    covars_02 = hmm02.covars_
    start_02 = hmm02.startprob_

    total_n = n_components01 + n_components02

    means_comb = np.append(means_01, means_02, axis=0)
    transmat_comb = normalized_matrix(total_n, total_n)
    covars_comb = np.append(covars_01, covars_02, axis=0)
    start_comb = normalize_matrix(np.append(start_01, start_02))

    return {'means_': means_comb, 'transmat_': transmat_comb, 'covars_': covars_comb, 'n_components': total_n, 'startprob_': start_comb}

def debug():

    print('debug')

    # y, sr = librosa.load(librosa.ex('trumpet'))
    # features = mfcc(y[0:10000], sr)[:, 2:-1]
    # hmm_01 = hmm.GaussianHMM(4)
    # hmm_01.fit(features)

    # y02, sr02 = librosa.load(librosa.ex('vibeace'))
    # features02 = mfcc(y02[0:10000], sr02)[:, 2:-1]
    # hmm_02 = hmm.GaussianHMM(2)
    # hmm_02.fit(features02)

    # combineTwoHmmsSimple(hmm_01, hmm_02)

if __name__ == '__main__':
    debug()



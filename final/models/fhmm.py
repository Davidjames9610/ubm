# gaussian fhmm
import numpy as np
from hmmlearn.hmm import GaussianHMM

class FactorialHMM:
    def __init__(self, hmm_a: GaussianHMM, hmm_b: GaussianHMM):
        self.hmm_a = hmm_a
        self.hmm_b = hmm_b
        self.n_states_a = hmm_a.n_components
        self.n_states_b = hmm_b.n_components
        self.n_components = self.hmm_a.n_components * self.hmm_b.n_components
        self.n_features = self.hmm_a.n_features

        self.hmm = GaussianHMM(n_components=self.n_components, covariance_type="full")

        transmat = self.get_transmat()
        startprob = self.get_startprob()
        means, covars, states_dict = self.get_m_and_c()
        self.hmm.n_features, self.hmm.means_, self.hmm.transmat_, self.hmm.startprob_ = self.n_features, means, transmat, startprob
        self.hmm.covars_ = np.array([np.diag(i) for i in covars])
        self.states_dict = states_dict

    def decode(self, features):
        log_prob, states_decoded = self.hmm.decode(features)

        ss01 = []
        ss02 = []
        for x in range(len(states_decoded)):
            temp = self.states_dict[states_decoded[x]]
            ss01.append(temp[0])
            ss02.append(temp[1])

        return log_prob, [ss01, ss02]

    def kronecker_list(self, list_A):
        result=list_A[0]
        for i in range(len(list_A)-1):
            result=np.kron(result,list_A[i+1])
        return result

    def get_transmat(self):
        transmat = self.kronecker_list([self.hmm_a.transmat_, self.hmm_b.transmat_]) + 1e-10
        transmat /= transmat.sum(axis=1)
        return transmat

    def get_startprob(self):
        startprob = self.kronecker_list([self.hmm_a.startprob_, self.hmm_b.startprob_]) + 1e-10
        startprob /= startprob.sum(axis=0)
        return startprob

    def get_m_and_c(self):
        states_dict = []
        means = [] # np.zeros((self.n_components, self.n_features))
        covars = [] # np.zeros((self.n_components, self.n_features, self.n_features))
        # logBjk = np.zeros((, states02, T_len))
        for j in range(self.n_states_a):
            m_j = []
            c_j = []
            for k in range(self.n_states_b):
                mean_j = self.hmm_a.means_[j]
                mean_k = self.hmm_b.means_[k]
                m = np.maximum(mean_j, mean_k)
                m_mask = (mean_j > mean_k)
                covar_j = np.diag(self.hmm_a.covars_[j])
                covar_k = np.diag(self.hmm_b.covars_[k])
                c = np.where(m_mask, covar_j, covar_k)
                m_j.append(m)
                c_j.append(c)
                states_dict.append([j, k])
            means.append(m_j)
            covars.append(c_j)

        return np.concatenate(means, axis=0), np.concatenate(covars, axis = 0), states_dict

# my_fhmm = FactorialHMM(hmm_whale, hmm_noise)
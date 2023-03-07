import numpy as np
import hmmlearn.hmm as hmmlearn
from sklearn.neural_network import MLPClassifier

def elog(x):
    res = np.log(x, where=(x != 0))
    res[np.where(x == 0)] = -(10.0 ** 8)
    return res


def getExpandedData(data):
    T = data.shape[0]

    data_0 = np.copy(data[0])
    data_T = np.copy(data[T - 1])

    for i in range(3):
        data = np.insert(data, 0, data_0, axis=0)
        data = np.insert(data, -1, data_T, axis=0)

    data_expanded = np.zeros((T, 7 * data.shape[1]))
    for t in range(3, T + 3):
        np.concatenate((data[t - 3], data[t - 2], data[t - 1], data[t],
                        data[t + 1], data[t + 2], data[t + 3]), out=data_expanded[t - 3])

    return data_expanded


def logSumExp(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    x_diff = x - x_max
    sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
    return x_max + np.log(sumexp)


class DNNHMM:
    def __init__(self, n_mix=2, n_components=4):
        self.n_mix = n_mix
        self.n_components = n_components
        self.hmm = hmmlearn.GaussianHMM(n_components=n_components)
        self.mlp = None

    # features should not be concatenated
    def fit(self, features):
        self.train_hmm(features)
        sequences = self.viterbi_hmm(features)
        self.train_mlp(features, sequences)

    def train_hmm(self, features):
        lengths = []
        for n, i in enumerate(features):
            lengths.append(len(i))
        self.hmm.fit(np.concatenate(features), lengths)

    def viterbi_hmm(self, features):
        sequences = []
        for feature in features:
            sequences.append(self.hmm.predict(feature))
        return sequences

    def train_mlp(self, features, sequences):

        O = []
        S = []

        for data_u, seq in zip(features, sequences):
            data_u_expanded = getExpandedData(data_u)
            O.append(data_u_expanded)
            S.append(seq)

        O = np.vstack(O)
        S = np.concatenate(S, axis=0)

        mlp = MLPClassifier(hidden_layer_sizes=(256, 256), random_state=1, early_stopping=True, verbose=True,
                            validation_fraction=0.1)
        mlp.fit(O, S)

        self.mlp = mlp

    def mlp_predict(self, o):
        o_expanded = getExpandedData(o)
        return self.mlp.predict_log_proba(o_expanded)

    def viterbi_mlp(self, o):

        hmm = self.hmm
        pi = hmm.startprob_
        a = hmm.transmat_

        T = o.shape[0]
        J = len(pi)

        s_hat = np.zeros(T, dtype=int)

        log_delta = np.zeros((T, J))

        psi = np.zeros((T, J))

        log_delta[0] = elog(pi)

        mlp_ll = self.mlp_predict(o)

        log_delta[0] += np.array([mlp_ll[0][j] for j in range(J)])

        log_A = elog(a)

        for t in range(1, T):
            for j in range(J):
                temp = np.zeros(J)
                for i in range(J):
                    temp[i] = log_delta[t - 1, i] + log_A[i, j] + mlp_ll[t][j]
                log_delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(log_delta[t - 1] + log_A[:, j])

        s_hat[T - 1] = np.argmax(log_delta[T - 1])

        for t in reversed(range(T - 1)):
            s_hat[t] = psi[t + 1, s_hat[t + 1]]

        return s_hat

    def forward_mlp(self, o):

        hmm = self.hmm

        pi = hmm.startprob_
        a = hmm.transmat_

        T = o.shape[0]
        J = len(pi)

        log_alpha = np.zeros((T, J))
        log_alpha[0] = elog(pi)

        mlp_ll = self.mlp_predict(o)

        log_alpha[0] += np.array([mlp_ll[0][j] for j in range(J)])

        for t in range(1, T):
            for j in range(J):
                mlp_ll_t = mlp_ll[t][j]
                log_alpha[t, j] = mlp_ll_t + logSumExp(elog(a[:, j].T) + log_alpha[t - 1])

        return log_alpha

    def score(self, data):
        T = data.shape[0]
        log_alpha_t = self.forward_mlp(data)[T - 1]
        ll = logSumExp(log_alpha_t)
        return ll

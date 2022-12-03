from sklearn.mixture import GaussianMixture, _gaussian_mixture
from scipy.special import logsumexp
import numpy as np

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

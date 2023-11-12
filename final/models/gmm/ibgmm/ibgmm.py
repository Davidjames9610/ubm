import itertools

import seaborn as sns
from hmmlearn.stats import _log_multivariate_normal_density_diag
from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from final.models.hdphmm.hdphmmwl.numba_wl import compute_probabilities, multinomial

# sns.set()
import numpy as np
from scipy.stats import wishart, dirichlet, invwishart, multivariate_normal
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from scipy import linalg
from sklearn import mixture

from tuts.bhmm.plot_hmm import plot_hmm_data

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


class NiwMcPrep:
    def __init__(self, mus, sigmas, lpdf_mus, lpdf_sigmas):
        self.mus = mus
        self.sigmas = sigmas
        self.lpdf_mus = lpdf_mus
        self.lpdf_sigmas = lpdf_sigmas


class InfiniteGMMGibbsSampler:
    def __init__(self, X, K, burn_in=0, iterations=40, hyper_params=None, verbose=False,**kwargs):
        """
        [] input
        X = data n x d
        K = expected amount of components

        [] params = pi, z, mu, Sigma
        init z and pi using uniform alpha, mu and sigma using data and z assignments

        [] hyper-params = alpha, m0, V0, nu0, S0
        for uniform priors we set
        alpha0 -> small value
        m0 -> mean of init assignments
        V0 -> large eye matrix
        nu0 -> K + 1
        S0 -> data scatter matrix

        [] ss
        nk - count of data in each k
        x_bar - mean of data in each k
        """

        self.X = X  # data n x d
        self.K = K  # expected no of states
        self.burn_in = burn_in
        self.iterations = iterations
        self.N = X.shape[0]  # length of data
        self.D = X.shape[1]  # dimension of data
        self.n_mce = int(1e3)  # n for monte carlo estimate

        self.Z_true = kwargs.get("Z_true")

        # set alpha to uniform prior and draw pi and z
        self.alpha0 = 100  # Uniform prior
        self.alpha_k = self.alpha0 / self.K
        alpha_k = np.ones(self.K) * self.alpha_k
        self.pi = dirichlet.rvs(alpha=alpha_k, size=1).flatten()

        # use k mean to init z
        self.Z = np.zeros(self.N)

        # true means
        kmeans = KMeans(n_clusters=self.K, random_state=42, init='random')
        kmeans.fit(self.X)

        # shuffle labels
        num_labels_to_replace = int(0.1 * len(kmeans.labels_))
        # Generate random labels between 0 and k
        random_labels = np.random.randint(0, self.K, num_labels_to_replace)
        # Replace 10% of the labels with random numbers
        shuffled_labels = np.copy(kmeans.labels_)
        replace_indices = np.random.choice(len(shuffled_labels), num_labels_to_replace, replace=False)
        shuffled_labels[replace_indices] = random_labels
        # Assign the shuffled labels to self.Z
        self.Z = shuffled_labels
        if self.Z_true is not None:
            print('init ari', np.round(ari(self.Z_true, self.Z), 3))

        # sufficient stats
        self.nk = np.zeros(self.K, dtype=int)
        self.x_bar = np.zeros((self.K, self.D), dtype=float)
        for k in range(K):
            self.nk[k] = np.sum(self.Z == k)
            self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

        # init mu and Sigma
        self.mu = np.zeros((self.K, self.D))
        self.sigma = np.zeros((self.K, self.D, self.D))
        self.lambdas = np.zeros((self.K, self.D, self.D))
        diagsig = np.zeros((self.K, self.D, self.D))  # scatter matrix
        for k in range(K):
            x_k = self.X[self.Z == k]
            sig_bar = np.cov(x_k.T, bias=True)
            diagsig[k] = np.diag(np.diag(sig_bar))
            self.mu[k] = self.x_bar[k]
            self.sigma[k] = sig_bar
            self.lambdas[k] = np.linalg.inv(sig_bar)

        # Hyper parameters
        if (hyper_params is None):
            self.V0 = np.eye(self.D) * 1000  # D x D
            # Sigma
            self.nu0 = 10  # np.copy(self.D) + 2  # 1 Degrees of freedom IW
            # set m0 and S0 to global mean and covar
            sig_bar = np.cov(self.X.T, bias=True)
            self.S0 = np.diag(np.diag(sig_bar))
            self.m0 = np.mean(self.X, axis=0)  # 1 x D
        else:
            self.V0 = hyper_params['V0']
            self.S0 = hyper_params['S0']
            self.m0 = hyper_params['m0']
            self.nu0 = hyper_params['nu0']

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []
        self.gmm = None
        self.verbose = verbose

    def has_converged(self, previous_likelihood, current_likelihood, tolerance):
        # Calculate the relative change in likelihood
        relative_change = np.abs((current_likelihood - previous_likelihood) / previous_likelihood)
        return relative_change < tolerance

    def calculate_likelihood(self):
        likelihood = 0.0
        for i in range(self.N):
            point_likelihood = 0.0
            for k in range(self.K):
                component_likelihood = self.pi[k] * multivariate_normal.pdf(self.X[i], mean=self.mu[k],
                                                                            cov=self.sigma[k])
                point_likelihood += component_likelihood
            likelihood += np.log(point_likelihood)  # Use log likelihood to avoid numerical underflow
        return likelihood

    def sample_theta(self):
        for k in range(self.K):
            # mu
            Vk = (np.linalg.inv(np.linalg.inv(self.V0) + self.nk[k] * np.linalg.inv(self.sigma[k])))
            term1 = np.dot(np.linalg.inv(self.sigma[k]), self.nk[k] * self.x_bar[k])
            term2 = np.dot(np.linalg.inv(self.V0), self.m0)
            mk = (np.dot(Vk, term1 + term2))
            # sample mu
            mu_k = np.random.multivariate_normal(mean=mk, cov=Vk, size=1).flatten()
            if np.isnan(mu_k).any():
                print('oh dear')
            else:
                self.mu[k] = mu_k

            # sigma
            dif = (self.X[self.Z == k] - self.mu[k])
            Sk = (self.S0 + (np.dot(dif.T, dif)))
            nuk = self.nu0 + self.nk[k]
            # sample sigma
            self.sigma[k] = invwishart.rvs(size=1, df=nuk, scale=Sk)

    def sample_pi(self):
        alpha_k = self.nk + (np.ones(self.K) * self.alpha_k)
        self.pi = dirichlet.rvs(size=1, alpha=alpha_k).flatten()

    def niw_mc_estimate(self, x, m=None, v=None, nu=None, s=None, random_k=None):

        m = self.m0
        v = self.V0
        nu = self.nu0
        s = self.S0

        mus = []
        sigmas = []
        for i in range(self.n_mce):
            mus.append(np.random.multivariate_normal(mean=m, cov=v, size=1).flatten())
            sigmas.append(invwishart.rvs(size=1, df=nu, scale=s))
        p = 0
        for i in range(self.n_mce):
            p += multivariate_normal.pdf(x, mus[i], sigmas[i]) * multivariate_normal.pdf(mus[i], m, v) * invwishart.pdf(
                sigmas[i], df=nu, scale=s)
        return np.log((p / self.n_mce) + 1e-20)

    def niw_mc_estimate_v2(self, x, mce_prep: NiwMcPrep):
        x_prob = _log_multivariate_normal_density_diag(x[np.newaxis, ...], mce_prep.mus,
                                                       np.diagonal(mce_prep.sigmas, axis1=1, axis2=2)).flatten()
        # p = np.sum(x_prob + mce_prep.lpdf_mus + mce_prep.lpdf_sigmas)
        # val_1 = p - np.log(mce_prep.n_mce)
        # val_2 = np.log(np.sum(np.exp(x_prob) * np.exp(mce_prep.lpdf_mus) * np.exp(mce_prep.lpdf_sigmas)) / mce_prep.n_mce)
        return logsumexp(x_prob + mce_prep.lpdf_mus + mce_prep.lpdf_sigmas) - np.log(self.n_mce)

    def niw_mc_estimate_prep(self):
        mus = np.random.multivariate_normal(mean=self.m0, cov=self.V0, size=self.n_mce)
        sigmas = invwishart.rvs(size=self.n_mce, df=self.nu0, scale=self.S0)
        lpdf_mus = multivariate_normal.logpdf(mus, self.m0, self.V0)
        sigmas_t = np.transpose(sigmas, (1, 2, 0))
        lpdf_sigmas = invwishart.logpdf(sigmas_t, df=self.nu0, scale=self.S0)
        return NiwMcPrep(mus, sigmas, lpdf_mus, lpdf_sigmas)

    def calculate_crp(self):

        prob_k = self.nk / (self.N - 1 + self.alpha_k)
        prob_new = self.alpha_k / (self.N - 1 + self.alpha_k)
        return np.log(prob_k + 1e-60), np.log(prob_new)

    def calculate_posterior_theta(self, x, mc_pre: NiwMcPrep):

        # theta_k = np.zeros(self.K)
        # for k in range(self.K):
        #     theta_k[k] = multivariate_normal.logpdf(x, mean=self.mu[k], cov=self.sigma[k])
        theta_k = _log_multivariate_normal_density_diag(x[np.newaxis, ...], self.mu,
                                                        np.diagonal(self.sigma, axis1=1, axis2=2)).flatten()
        # theta_k = multivariate_normal.logpdf(x, mean=self.mu, cov=self.sigma)

        # theta_new = self.niw_mc_estimate(x)
        theta_new = self.niw_mc_estimate_v2(x, mc_pre)

        return theta_k, theta_new

    def sample_z(self):

        # every sample of z get new niw_mc
        mc_prep = self.niw_mc_estimate_prep()

        for i in range(self.N):
            x = self.X[i]
            self.remove_x_from_z(i)

            crp_k, crp_new = self.calculate_crp()
            theta_k, theta_new = self.calculate_posterior_theta(x, mc_prep)

            prob = np.hstack((crp_k + theta_k, crp_new + theta_new))

            post_cases_probs = compute_probabilities(prob)
            self.Z[i] = multinomial(post_cases_probs)

            # new state
            if self.Z[i] == self.K:
                if self.verbose:
                    print('new state')
                random_k = np.random.randint(0, self.K)
                self.nk = np.hstack((self.nk, [0]))
                self.pi = np.hstack((self.pi, [0]))
                self.x_bar = np.vstack((self.x_bar, np.array([0, 0])))
                self.mu = np.vstack((self.mu, x))
                # self.m0 = np.vstack((self.m0, x))
                self.sigma = np.concatenate((self.sigma, [self.sigma[random_k]]), 0)
                # self.S0 = np.concatenate((self.S0, [self.S0[random_k]]), 0)
                self.K += 1
                self.alpha_k = self.alpha0 / self.K

            self.add_x_to_z(i)
            self.handle_empty_components()

    def handle_empty_components(self):
        nk = np.zeros(self.K)
        for k in range(self.K):
            nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(nk == 0)[0]
        if len(zero_indices) > 0:
            if self.verbose:
                print('deleting component(s), ', zero_indices)
            rem_ind = np.unique(self.Z)
            d = {k: v for v, k in enumerate(sorted(rem_ind))}
            self.Z = np.array([d[x] for x in self.Z])
            self.K = len(rem_ind)
            self.sigma = self.sigma[rem_ind]
            # self.S0 = self.S0[rem_ind]
            self.mu = self.mu[rem_ind]
            self.nk = self.nk[rem_ind]
            # self.m0 = self.m0[rem_ind]
            self.x_bar = self.x_bar[rem_ind]
            self.pi = self.pi[rem_ind]
            self.alpha_k = self.alpha0 / self.K

    def remove_x_from_z(self, index):
        x_i = self.X[index]
        x_z = self.Z[index]

        self.Z[index] = -1  # avoid counting
        prev_x_count = self.x_bar[x_z] * self.nk[x_z]
        self.nk[x_z] -= 1
        if (self.nk[x_z] > 0):
            self.x_bar[x_z] = (prev_x_count - x_i) / self.nk[x_z]
        else:
            self.x_bar[x_z] = 0

    def add_x_to_z(self, index):
        x_i = self.X[index]
        x_z = self.Z[index]
        prev_x_count = self.x_bar[x_z] * self.nk[x_z]
        self.nk[x_z] += 1
        self.x_bar[x_z] = (prev_x_count + x_i) / self.nk[x_z]

    def update_ss(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            # avoid zero assignments
            if self.nk[k] > 0:
                self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)
            else:
                print('error empty component in update_ss')
                # Reassign an empty component randomly
                random_idx = np.random.randint(0, self.N)
                self.Z[random_idx] = k
                self.nk[k] = 1  # Increment the count
                self.x_bar[k] = self.X[random_idx]

    # one sweep of git sampler, return variables sampled
    def gibbs_sweep(self):
        self.sample_z()
        self.update_ss()
        self.sample_theta()

    def fit(self):
        max_iterations = 100
        convergence_threshold = 1e-5
        self.likelihood_history = []

        self.ARI = np.zeros((self.iterations))
        print('starting gibbs sampling')
        for it in range(self.iterations):
            self.gibbs_sweep()

            # save trace
            # if it > self.burn_in:
            self.mu_trace.append(self.mu)
            self.sigma_trace.append(self.sigma)
            self.pi_trace.append(self.pi)

            # Calculate ARI
            if self.Z_true is not None:
                self.ARI[it] = np.round(ari(self.Z_true, self.Z), 3)
                # print(f"ARI:{self.ARI[it]}")

            if it % 50 == 0:
                # check likelihood and break if needed
                current_likelihood = self.calculate_likelihood()
                self.likelihood_history.append(current_likelihood)
                print('it: ', it, ' likelihood: ', current_likelihood, ' counts:', self.K)
            if it % 25 == 0 or it == 0:
                # bgmm = mixture.GaussianMixture(n_components=self.K, covariance_type="full")
                # bgmm.means_, bgmm.covariances_, bgmm.weights_, bgmm.precisions_cholesky_ = self.mu, self.sigma, self.pi, _compute_precision_cholesky(
                #     self.sigma, "full")
                # bgmm.precisions_ = bgmm.covariances_ ** 2
                # bgmm.converged_ = True
                plot_hmm_data(self.X, self.Z, self.K, self.mu, self.sigma)
                print('it: ', it, ' counts: ', self.nk)
                # if self.has_converged(likelihood_history[-1], current_likelihood, convergence_threshold):
                #     print(f"Converged after {it} iterations.")
                #     break

        bgmm = mixture.GaussianMixture(n_components=self.K, covariance_type="full")
        bgmm.means_, bgmm.covariances_, bgmm.weights_, bgmm.precisions_cholesky_ = self.mu, self.sigma, self.pi, _compute_precision_cholesky(
            self.sigma, "full")
        bgmm.precisions_ = bgmm.covariances_ ** 2
        bgmm.converged_ = True

        self.gmm = bgmm
# my_gibbs = GMMGibbsSampler(x_i, 3, )
# np.sum(my_gibbs.Z == 0)

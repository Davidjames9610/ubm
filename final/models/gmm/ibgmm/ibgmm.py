import itertools
import math
from typing import List

import seaborn as sns
from hmmlearn.stats import _log_multivariate_normal_density_diag
from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from final.models.hdphmm.hdphmmwl.numba_wl import compute_probabilities, multinomial
import multiprocessing
from functools import partial

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
    def __init__(self, X, K, alpha0=1, burn_in=0, iterations=40, hyper_params=None, verbose=False, **kwargs):
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
        self.n_mce = int(1e2)  # n for monte carlo estimate
        self.truncate = 30
        self.total_iterations = 0

        self.Z_true = kwargs.get("Z_true")
        self.chain_id = kwargs.get("chain_id")

        # set alpha to uniform prior and draw pi and z
        self.alpha0 = alpha0  # Uniform prior
        self.alpha_k = self.alpha0 / self.K
        alpha_k = np.ones(self.K) * self.alpha_k
        self.pi = dirichlet.rvs(alpha=alpha_k, size=1).flatten()

        # use k mean to init z
        self.Z = np.zeros(self.N)

        # true means
        kmeans = KMeans(n_clusters=self.K, random_state=42, init='random')
        kmeans.fit(self.X)

        # shuffle labels
        num_labels_to_replace = int(0.2 * len(kmeans.labels_))
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

        sig_bar = np.cov(self.X.T, bias=True)
        self.S0 = np.diag(np.diag(sig_bar))
        self.m0 = np.mean(self.X, axis=0)  # 1 x D

        if hyper_params is None:
            self.nu0 = np.copy(self.D) + 2  # Degrees of freedom IW
        else:
            self.nu0 = hyper_params['nu0']

        if hyper_params is None:
            self.V0 = np.eye(self.D) * self.get_magnitude(self.S0[0, 0])
        else:
            self.V0 = hyper_params['V0']

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.nk_trace = []
        self.k_trace = []
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []
        self.gmm: mixture.GaussianMixture | None = None
        self.verbose = verbose
        self.finish = False

    @staticmethod
    def get_magnitude(number):
        # Handle the case where the number is 0 separately
        if number == 0:
            return 0

        # Calculate the magnitude using the logarithm
        magnitude = int(math.floor(math.log10(abs(number)))) + 1

        # Adjust the result to match the scientific notation format
        return 10 ** magnitude

    def has_converged(self, previous_likelihood, current_likelihood, tolerance):
        # Calculate the relative change in likelihood
        relative_change = np.abs((current_likelihood - previous_likelihood) / previous_likelihood)
        return relative_change < tolerance

    # def calculate_likelihood(self):
    #     likelihood = 0.0
    #     for i in range(self.N):
    #         point_likelihood = 0.0
    #         for k in range(self.K):
    #             component_likelihood = self.pi[k] * multivariate_normal.pdf(self.X[i], mean=self.mu[k],
    #                                                                         cov=self.sigma[k])
    #             point_likelihood += component_likelihood
    #         likelihood += np.log(point_likelihood)  # Use log likelihood to avoid numerical underflow
    #     return likelihood

    def calculate_likelihood(self, test_data=None, use_gmm=False):
        pi = self.pi
        mu = self.mu
        sigma = self.sigma

        if use_gmm:
            pi = self.gmm.weights_
            mu = self.gmm.means_
            sigma = self.gmm.covariances_

        likelihood = []
        if test_data is None:
            test_data = self.X
        for i in range(len(test_data)):
            point_likelihood = 0.0
            for k in range(self.K):
                component_likelihood = pi[k] * multivariate_normal.pdf(test_data[i], mean=mu[k], cov=sigma[k])
                point_likelihood += component_likelihood
            likelihood.append(np.log(point_likelihood))
        return np.mean(likelihood), np.std(likelihood), np.sum(likelihood)

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

            if self.finish:
                prob = crp_k + theta_k
            else:
                prob = np.hstack((crp_k + theta_k, crp_new + theta_new))

            post_cases_probs = compute_probabilities(prob)

            self.Z[i] = multinomial(post_cases_probs)

            # new state
            if self.Z[i] == self.K:
                if self.K < self.truncate:
                    # if self.verbose:
                    #     print('new state')
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
                else:
                    prob = np.hstack((crp_k + theta_k))
                    post_cases_probs = compute_probabilities(prob)
                    self.Z[i] = multinomial(post_cases_probs)

            self.add_x_to_z(i)
            self.handle_empty_components()

    def handle_empty_components(self):
        nk = np.zeros(self.K)
        for k in range(self.K):
            nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(nk == 0)[0]
        if len(zero_indices) > 0:
            # if self.verbose:
            #     print('deleting component(s), ', zero_indices)
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
        self.sample_pi()
        self.total_iterations +=1

        if self.total_iterations > self.burn_in:
            return np.copy(self.mu), np.copy(self.sigma), np.copy(self.pi)
        else:
            return [], [], []

    def fit(self):
        max_iterations = 100
        convergence_threshold = 1e-5
        self.likelihood_history = []
        alpha_diff = self.iterations / 10

        self.ARI = np.zeros((self.iterations))
        print('starting gibbs sampling')
        for it in range(self.iterations):
            self.gibbs_sweep()

            # save trace
            # if it > self.burn_in:
            self.mu_trace.append(self.mu)
            self.sigma_trace.append(self.sigma)
            self.pi_trace.append(self.pi)
            self.nk_trace.append(self.nk)
            self.k_trace.append(self.K)

            # Calculate ARI
            if self.Z_true is not None:
                self.ARI[it] = np.round(ari(self.Z_true, self.Z), 3)
                # print(f"ARI:{self.ARI[it]}")

            if it % 50 == 0 and self.verbose:
                # check likelihood and break if needed
                current_likelihood = self.calculate_likelihood()
                self.likelihood_history.append(current_likelihood)
                print('it: ', it, ' likelihood: ', current_likelihood, ' counts:', self.K)
            if (it % 50 == 0 or it == 0) and self.verbose:
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
            # if self.iterations - it < 10 and self.alpha0 > 1 and self.alpha0 > alpha_diff + 1:
            #     self.alpha0 = self.alpha0 - alpha_diff
            # if self.iterations - it == 100:
            #     self.finish = True
            #     self.alpha0 = 10
            if self.iterations - it == 100:
                self.finish = True
                self.alpha0 = 1

        self.gmm_from_trace()

    def gmm_from_trace(self, average_over=25):

        # last_counts = np.copy(self.nk_trace[-25:])
        last_ks =  np.copy(self.k_trace[-25:])
        n_components = min(last_ks)
        # k = min(last_ks)

        # from the last 'average_over' its, get the sum
        nk = np.copy(self.nk_trace)[-average_over:]
        nk_sum = np.sum(nk, axis=0)
        largest_indices = np.argpartition(nk_sum, -n_components)[-n_components:]

        mu_matrix = np.mean(np.array(self.mu_trace[-average_over:], dtype=object), axis=0)[largest_indices]
        sigma_matrix = np.mean(np.array(self.sigma_trace[-average_over:], dtype=object), axis=0)[largest_indices]
        pi_matrix = np.mean(np.array(self.pi_trace[-average_over:], dtype=object), axis=0)[largest_indices]

        bgmm = mixture.GaussianMixture(n_components=n_components, covariance_type="full")
        bgmm.means_, bgmm.covariances_, bgmm.weights_, bgmm.precisions_cholesky_ = mu_matrix, sigma_matrix, pi_matrix, _compute_precision_cholesky(
            sigma_matrix, "full")
        bgmm.precisions_ = bgmm.covariances_ ** 2
        bgmm.converged_ = True
        self.gmm = bgmm

    def plot(self):
        plot_hmm_data(self.X, self.Z, self.K, self.mu, self.sigma)

    def autocorrelation(self, trace, lag):
        """
        Calculate autocorrelation for a given trace at a specified lag.

        Parameters:
        - trace: list or array, the trace of a parameter over iterations
        - lag: int, the lag at which to calculate the autocorrelation

        Returns:
        - float, autocorrelation at the specified lag
        """
        mean_trace = np.mean(trace)
        numerator = np.sum((trace[:-lag] - mean_trace) * (trace[lag:] - mean_trace))
        denominator = np.sum((trace - mean_trace) ** 2)
        autocorr = numerator / denominator if denominator != 0 else 0
        return autocorr

    def plot_autocorrelation(self, trace, max_lag=50):
        """
        Plot autocorrelation function for a given trace.

        Parameters:
        - trace: list or array, the trace of a parameter over iterations
        - max_lag: int, the maximum lag to include in the plot
        """
        lags = np.arange(1, max_lag + 1)
        autocorrs = [self.autocorrelation(trace, lag) for lag in lags]

        plt.figure(figsize=(10, 6))
        plt.plot(lags, autocorrs, marker='o', linestyle='-', color='b')
        plt.title('Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()

    def get_n_largest_components(self, cut_off=0.90):
        # Sort the array in descending order
        sorted_indices = np.argsort(self.pi)[::-1]

        # Cumulative sum of the sorted array
        cumulative_sum = np.cumsum(self.pi[sorted_indices])

        # Find the indices that contribute to % of the total weight
        selected_indices = sorted_indices[cumulative_sum <= cut_off]

        return selected_indices

class InfiniteGMMGibbsSamplerParallel:

    def __init__(self, X, K, alpha0, num_chains, hyper_params=None, iterations_per_chain=100, burn_in=100):
        self.num_chains = num_chains
        self.X = X
        self.K = K
        self.alpha0 = alpha0
        self.hyper_params = hyper_params
        self.chains = []
        self.burn_in = burn_in
        self.iterations_per_chain = iterations_per_chain
        for c in range(num_chains):
            self.chains.append(self.initialize_chain(c))

        self.total_its = 0

        # Use Manager to create shared lists
        # manager = multiprocessing.Manager()
        # self.mu_trace = [[] for _ in range(self.num_chains)]
        # self.sigma_trace = [[] for _ in range(self.num_chains)]
        # self.pi_trace = [[] for _ in range(self.num_chains)]
        # self.nk_trace = [[] for _ in range(self.num_chains)]
        # self.r_hat_nks = []

    def initialize_chain(self, chain_id):
        """
        Initialize a single chain of the Gibbs sampler.

        Parameters:
        - chain_id: int, identifier for the chain

        Returns:
        - InfiniteGMMGibbsSampler instance
        """
        return InfiniteGMMGibbsSampler(
            self.X, self.K, self.alpha0, burn_in=self.burn_in, verbose=False, iterations=self.iterations_per_chain,
            hyper_params=self.hyper_params, chain_id=chain_id
        )

    def run_chain(self, chain: InfiniteGMMGibbsSampler):
        """
        Run a single chain of the Gibbs sampler.

        Parameters:
        - chain: InfiniteGMMGibbsSampler instance
        """
        # iterations_per_chain = chain.iterations
        # for it in range(self.iterations_per_chain):
        #     mu, sigma, pi = chain.gibbs_sweep()
        #     self.mu_trace[chain.chain_id].append(mu)
        #     self.sigma_trace[chain.chain_id].append(sigma)
        #     self.pi_trace[chain.chain_id].append(pi)
        #     self.nk_trace[chain.chain_id].append([chain.nk])
        chain.fit()
        print(f"Chain {chain.chain_id} completed.")

    def run_chains_parallel(self):
        """
        Run multiple chains of the Gibbs sampler in parallel.

        Parameters:
        - num_chains: int, number of chains to run in parallel
        - iterations_per_chain: int, number of iterations for each chain
        """

        # Use multiprocessing to run chains in parallel
        pool = multiprocessing.Pool(processes=self.num_chains)
        partial_run_chain = partial(self.run_chain)
        pool.map(partial_run_chain, self.chains)
        pool.close()
        pool.join()

        # self.mu_trace = [[] for _ in range(self.num_chains)]
        # self.sigma_trace = [[] for _ in range(self.num_chains)]
        # self.pi_trace = [[] for _ in range(self.num_chains)]
        # self.nk_trace = [[] for _ in range(self.num_chains)]

        # for it in range(self.iterations_per_chain):
        #     for c in range(self.num_chains):
        #         chain = self.chains[c]
        #         mu, sigma, pi = chain.gibbs_sweep()
        #         if len(mu) > 0:
        #             self.mu_trace[chain.chain_id].append(mu)
        #             self.sigma_trace[chain.chain_id].append(sigma)
        #             self.pi_trace[chain.chain_id].append(pi)
        #             sorted_weights = np.sort(chain.pi)[::-1]
        #             cumulative_sum = np.cumsum(sorted_weights) # always 1
        #             target_explained_variance = 0.85
        #             n_components_80_percent = np.argmax(cumulative_sum >= target_explained_variance) + 1
        #             self.nk_trace[chain.chain_id].append(n_components_80_percent)
        # print('01')
        # self.chains[0].plot()
        # print('02')
        # self.chains[1].plot()

    def fit_parallel(self):
        """
        Run multiple chains of the Gibbs sampler in parallel and compute the Gelman-Rubin diagnostic.
        """
        self.run_chains_parallel()
        self.gelman_rubin_diagnostic()

    def gelman_rubin_diagnostic(self):
        """
        Compute the Gelman-Rubin diagnostic for each parameter.
        """
        component_index = 0
        # Concatenate traces from different chains for each parameter

        mu_traces = []
        pie_traces = []
        nk_traces = []
        for c in range(self.num_chains):
            mu_traces.append(np.vstack([trace_chain[component_index, :, ] for trace_chain in self.mu_trace[c]])[-10:,:])
            pie_traces.append(np.vstack([trace_chain[component_index] for trace_chain in self.pi_trace[c]]))
            nk_traces.append(np.vstack([trace_chain for trace_chain in self.nk_trace[c][-15:]]))
        # mu_traces = [np.concatenate(trace_chain) for trace_chain in self.mu_trace]
        # sigma_traces = [np.concatenate(trace_chain) for trace_chain in self.sigma_trace]
        # pi_traces = [np.concatenate(trace_chain) for trace_chain in self.pi_trace]

        # Compute Gelman-Rubin diagnostic for each parameter
        r_hat_mu = self.gelman_rubin_statistic_2d(mu_traces)
        # r_hat_sigma = self.gelman_rubin_statistic(sigma_traces)
        r_hat_pi = self.gelman_rubin_statistic(pie_traces)

        r_hat_nk = self.gelman_rubin_statistic(nk_traces)

        print(f"Gelman-Rubin Diagnostic - mu: {r_hat_mu}, pi: {r_hat_pi}, nk: {r_hat_nk}") #, sigma: {r_hat_sigma}, pi: {r_hat_pi}")

        self.r_hat_nks.append(r_hat_nk)

        if r_hat_nk < 1.1:
            print('convergence!')
            self.converged = True

            # collect 20 more samples
            self.iterations_per_chain = 20


    @staticmethod
    def gelman_rubin_statistic(chains):
        """
        Calculate the Gelman-Rubin diagnostic for multiple chains.

        Parameters:
        - chains: List of arrays, each representing a separate chain of samples for a parameter.

        Returns:
        - float, Gelman-Rubin diagnostic value.
        """
        tiny_value = 1e-40
        num_chains = len(chains)
        num_samples = chains[0].shape[0]

        # Calculate within-chain variance
        within_chain_variances = [(np.var(chain, ddof=1) + tiny_value) for chain in chains]

        # Calculate between-chain variance
        overall_mean = np.mean(np.concatenate(chains))
        between_chain_variance = num_samples * np.var([np.mean(chain) for chain in chains], ddof=1)

        # Calculate the pooled posterior variance
        pooled_posterior_variance = ((num_samples - 1) / num_samples) * np.mean(within_chain_variances) + \
                                    (1 / num_samples) * between_chain_variance

        # Calculate potential scale reduction factor
        potential_scale_reduction = np.sqrt(pooled_posterior_variance + tiny_value / np.mean(within_chain_variances) + tiny_value)

        return potential_scale_reduction

        # Example usage:
        # chains = [my_gibbs.pi_trace[0], my_gibbs.mu_trace[0][:, 0], my_gibbs.sigma_trace[0][:, 0, 0]]
        # r_hat = gelman_rubin_diagnostic(chains)
        # print("Gelman-Rubin diagnostic (R-hat):", r_hat)

    @staticmethod
    def gelman_rubin_statistic_2d(chains):
        """
        Calculate the Gelman-Rubin diagnostic for multiple chains.

        Parameters:
        - chains: List of arrays, each representing a separate chain of samples for a parameter.

        Returns:
        - float, Gelman-Rubin diagnostic value.
        """
        tiny_value = 1e-40
        num_chains = len(chains)
        num_samples = chains[0].shape[0]
        num_dimensions = chains[0].shape[1]  # Assuming chains have the same number of dimensions

        # Calculate within-chain variance for each dimension
        within_chain_variances = [
            [(np.var(chain[:, dim], ddof=1) + tiny_value) for dim in range(num_dimensions)]
            for chain in chains
        ]

        # Calculate between-chain variance for each dimension
        overall_mean = np.mean(np.concatenate(chains), axis=0)
        between_chain_variance = num_samples * np.sum([
            np.var([np.mean(chain[:, dim]) for chain in chains], ddof=1)
            for dim in range(num_dimensions)
        ])

        # Calculate the pooled posterior variance for each dimension
        pooled_posterior_variance = np.zeros(num_dimensions)
        for dim in range(num_dimensions):
            pooled_posterior_variance[dim] = (
                    ((num_samples - 1) / num_samples) * np.mean(
                [within_chain_variances[i][dim] for i in range(num_chains)]) +
                    (1 / num_samples) * between_chain_variance / num_dimensions
            )

        # Calculate potential scale reduction factor for each dimension
        potential_scale_reduction = np.sqrt(
            np.sum(pooled_posterior_variance) / np.sum(
                [np.mean(within_chain_variances[i]) + tiny_value for i in range(num_chains)])
        )

        return potential_scale_reduction

# my_gibbs = GMMGibbsSampler(x_i, 3, )
# np.sum(my_gibbs.Z == 0)

import math

import seaborn as sns
from sklearn import mixture
from sklearn.cluster import KMeans

from tuts.bhmm.plot_hmm import plot_hmm_data

sns.set()
import numpy as np
from scipy.stats import  wishart, dirichlet, invwishart, multivariate_normal
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from hmmlearn.stats import _log_multivariate_normal_density_diag, _log_multivariate_normal_density_full

class GMMGibbsSampler():
    def __init__(self, X, K, burn_in=0, iterations=40, alpha0=100, V0=None, nu0=None, **kwargs):
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

        self.verbose = kwargs.get("verbose")
        self.X = X  # data n x d
        self.K = K  # expected no of states
        self.burn_in = burn_in
        self.iterations = iterations
        self.N = X.shape[0] # length of data
        self.D = X.shape[1] # dimension of data

        self.Z_true = kwargs.get("Z_true")
        self.alpha0 = alpha0
        self.nu0 = nu0
        self.V0 = V0

        # set alpha to uniform prior and draw pi and z
        self.alpha0_k = np.ones(K) * self.alpha0/self.K # Uniform prior
        self.pi = dirichlet.rvs(alpha=self.alpha0_k, size=1).flatten()


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
        diagsig = np.zeros((self.K, self.D, self.D)) # scatter matrix
        for k in range(K):
            x_k = self.X[self.Z == k]
            sig_bar = np.cov(x_k.T, bias=True)
            diagsig[k] = np.diag(np.diag(sig_bar))
            self.mu[k] = self.x_bar[k]
            self.sigma[k] = sig_bar
            self.lambdas[k] = np.linalg.inv(sig_bar)

        # Hyper parameters
        # Mu
        self.m0 = np.mean(self.X, axis=0)
        # Sigma
        sig_bar = np.cov(self.X.T, bias=True)
        self.S0 = np.diag(np.diag(sig_bar)) # D x D

        if self.nu0 is None:
            self.nu0 = np.copy(self.D) + 2  # Degrees of freedom IW

        if self.V0 is None:
            self.V0 = np.eye(self.D) * self.get_magnitude(self.S0[0,0])

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []
        self.gmm = None

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

    # np.sum(_log_multivariate_normal_density_diag(self.X, self.mu, np.diagonal(self.sigma, axis1=1, axis2=2)) + np.log(
    #     self.pi))
    def calculate_likelihood(self):
        likelihood = 0.0
        for i in range(self.N):
            point_likelihood = 0.0
            for k in range(self.K):
                component_likelihood = self.pi[k] * multivariate_normal.pdf(self.X[i], mean=self.mu[k], cov=self.sigma[k])
                point_likelihood += component_likelihood
            likelihood += np.log(point_likelihood)  # Use log likelihood to avoid numerical underflow
        return likelihood

    def calculate_likelihood_on_test_data(self, test_data):
        likelihood = []
        for i in range(len(test_data)):
            point_likelihood = 0.0
            for k in range(self.K):
                component_likelihood = self.pi[k] * multivariate_normal.pdf(test_data[i], mean=self.mu[k], cov=self.sigma[k])
                point_likelihood += component_likelihood
            likelihood.append(np.log(point_likelihood))  # Use log likelihood to avoid numerical underflow
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
        alpha_k = self.nk + self.alpha0_k
        self.pi = dirichlet.rvs(size=1, alpha=alpha_k).flatten()

    def sample_z(self):
        # [2] sample assignments using params
        # responsibilities
        res = np.exp(_log_multivariate_normal_density_full(self.X, self.mu,self.sigma) + np.log(self.pi))
        res = res / np.sum(res, axis=1, keepdims=True)
        # Sample z
        Z_mat = np.zeros((self.N, self.K))
        for n in range(self.N):
            try:
                Z_mat[n] = np.random.multinomial(n=1, pvals=res[n], size=1).flatten()
            except ValueError as e:
                print(e)
        _, self.Z = np.where(Z_mat == 1)

    def update_ss(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
        self.handle_empty_components()
        for k in range(self.K):
            self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

    def handle_empty_components(self):
        zero_indices = np.where(self.nk == 0)[0]
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
            self.alpha0_k = self.alpha0 / self.K

    # one sweep of git sampler, return variables sampled
    def gibbs_sweep(self):
        self.sample_theta()
        self.sample_pi()
        self.sample_z()
        self.update_ss()

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

            # check likelihood and break if needed
            if it % 100 == 0 and self.verbose:
                current_likelihood = self.calculate_likelihood()
                print('it: ', it, 'like: ', current_likelihood)
                self.likelihood_history.append(current_likelihood)

                plot_hmm_data(self.X, self.Z, self.K, self.mu, self.sigma)
                print('it: ', it, 'len: ', len(self.nk), 'counts: ', self.nk)

            # if it > 0:
            #     if self.has_converged(likelihood_history[-1], current_likelihood, convergence_threshold):
                    # print(f"Converged after {it} iterations.")
                    # break

        bgmm = mixture.GaussianMixture(n_components=self.K, covariance_type="full")
        bgmm.means_, bgmm.covariances_, bgmm.weights_, bgmm.precisions_cholesky_ = self.mu, self.sigma, self.pi, _compute_precision_cholesky(self.sigma, "full")
        bgmm.precisions_ = bgmm.covariances_** 2
        bgmm.converged_ = True
        self.gmm = bgmm
    def get_n_largest_components(self, cut_off=0.90):
        # Sort the array in descending order
        sorted_indices = np.argsort(self.pi)[::-1]

        # Cumulative sum of the sorted array
        cumulative_sum = np.cumsum(self.pi[sorted_indices])

        # Find the indices that contribute to % of the total weight
        selected_indices = sorted_indices[cumulative_sum <= cut_off]

        return selected_indices

# my_gibbs = GMMGibbsSampler(x_i, 3, )
# np.sum(my_gibbs.Z == 0)
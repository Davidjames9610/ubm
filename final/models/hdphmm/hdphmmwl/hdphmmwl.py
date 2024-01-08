import math
from random import random

from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from scipy.stats import dirichlet, invwishart
from sklearn.metrics.cluster import adjusted_rand_score as ari
import numpy as np

from final import useful
from final.models.hdphmm.hdphmmwl.jax_wl import backward_robust_jax
from final.models.hdphmm.hdphmmwl.numba_wl import backward_robust_mv, sample_states_likelihood, sample_states_numba
from numpy.random import binomial
from final.models.hdphmm.hdphmmwl.consts import *
import time
from final.models.hdphmm.hdphmmda.hdp_hmm_da_utils.utils import is_symmetric_positive_semidefinite
from hmmlearn.stats import _log_multivariate_normal_density_diag
from hmmlearn._hmmc import backward_log

from final.models.hdphmm.helpers.plot_hmm import plot_hmm_data, plot_hmm_learn
import matplotlib.pyplot as plt


class HDPHMMWL():
    def __init__(self, X, K, Z_true=None, burn_in=0, iterations=20, sbp=None,
                 feature_a=0, feature_b=1, outer_its=1, convergence_check=100, max_it=1000, **kwargs):
        """
        [] input
        X_list = list of X
        X = data n x d
        K = expected amount of states

        [] params = beta, pi, z, mu, Sigma
        beta ~ GEM(gamma)
        pi ~ DP(beta, k, alpha)
        z ~ HDP()
        mu ~ Gauss(m0, V0)
        Sigma ~ IW(nu0, S0)

        [] hyper-params:
        gamma - concentration for GEM
        alpha - concentration for 2nd GEM
        k - sticky-ness of A

        m0 -> mean of init assignments
        V0 -> large eye matrix
        nu0 -> K + 1
        S0 -> data scatter matrix

        [] ss - need to update probably
        nk - count of data in each k
        x_bar - mean of data in each k
        """

        if isinstance(X, list):
            print('multiple sequences given')
            self.X_list = X
            self.X = X[0]  # init to first sequence
            self.outer_its = outer_its
        else:
            print('single sequence given')
            self.X_list = [X]
            self.X = X
            self.outer_its = 1

        self.K = K  # expected / weak limit no of states
        self.burn_in = burn_in
        self.max_it = max_it
        self.iterations = iterations
        self.N = self.X.shape[0]  # length of data
        self.D = self.X.shape[1]  # dimension of data
        self.Z = None
        self.Z_true = Z_true
        self.sbp = sbp

        # init variables
        self.beta = None  # base gem draw
        self.A = None  # transition matrix
        self.pi = None  # init prob matrix
        self.mu = None  # init mu
        self.sigma = None  # init sigma

        # init hyper-params
        # SDB and sticky-ness
        self.gamma0 = None  # concentration param on GEM 1
        self.alpha0 = None  # concentration param on GEM 1
        self.kappa0 = None  # sticky-ness
        self.rho0 = None  # combination of alpha0 and kappa0

        # Gaussian mv emission
        self.m0 = None
        self.V0 = None
        self.nu0 = None  # Degrees of freedom IW
        self.S0 = None

        # Sufficient statistics, state count and mean
        self.nk = None
        self.x_bar = None
        self.n_mat = None
        self.n_ft = None

        # aux-vars
        self.m_mat = None  # number of tables in restaurant j that were served dish k
        self.m_init = None
        self.w_vec = None  # override variable for table t in restaurant j
        self.m_mat_bar = None  # number of tables in restaurant j that considered dish k

        self.kwargs = kwargs
        self.V0 = kwargs.get('V0', None)
        self.nu0 = kwargs.get('nu0', None)
        self.m0 = kwargs.get('m0', None)
        self.S0 = kwargs.get('S0', None)

        self.init_sbp()
        self.init_z()

        self.trace = {}
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []

        self.hmm = None

        self.feature_a = feature_a
        self.feature_b = feature_b

        self.convergence_check = convergence_check

    def init_z(self, **kwargs):

        Z_mat = np.random.multinomial(n=1, pvals=self.pi, size=self.N)
        _, self.Z = np.where(Z_mat == 1)  # N x 1 component number

        # true means
        kmeans = KMeans(n_clusters=self.K, random_state=42)
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
        self.Z = kmeans.labels_  # shuffled_labels
        if self.Z_true is not None:
            print('init ari', np.round(ari(self.Z_true, self.Z), 3))

        # sufficient stats
        self.nk = np.zeros(self.K, dtype=int)
        self.x_bar = np.zeros((self.K, self.D), dtype=float)
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

        # init mu and Sigma
        self.mu = np.zeros((self.K, self.D))
        self.sigma = np.zeros((self.K, self.D, self.D))
        self.lambdas = np.zeros((self.K, self.D, self.D))
        diagsig = np.zeros((self.K, self.D, self.D))  # scatter matrix
        for k in range(self.K):
            x_k = self.X[self.Z == k]
            if len(x_k) < 10:
                random_indices = np.random.randint(0, self.N, int(self.N * 0.8))
                x_k = self.X[random_indices]
            sig_bar = np.cov(x_k.T, bias=True)
            diagsig[k] = np.diag(np.diag(sig_bar))
            self.mu[k] = self.x_bar[k]
            self.sigma[k] = np.diag(np.diag(sig_bar))
            # self.lambdas[k] = np.linalg.inv(sig_bar)

        # Hyper-parameters for normals
        if self.V0 is None:
            self.V0 = np.eye(self.D) * 1

        if self.m0 is None:
            self.m0 = np.mean(self.X, axis=0)

        if self.S0 is None:
            uni_sig_bar = np.cov(self.X.T, bias=True)
            self.S0 = np.diag(np.diag(uni_sig_bar))

        if self.nu0 is None:
            self.nu0 = np.copy(self.D) + 2

        # * (self.get_magnitude(np.max(self.S0)) + 1) # self.S0 # np.eye(self.D) * 1 # self.get_magnitude(np.max(self.S0)) # np.diag(np.diag(np.cov(self.X.T, bias=True)))

    @staticmethod
    def get_magnitude(number):
        # Handle the case where the number is 0 separately
        if number == 0:
            return 0

        # Calculate the magnitude using the logarithm
        magnitude = int(math.floor(math.log10(abs(number))))

        # Adjust the result to match the scientific notation format
        return 10 ** magnitude

    def init_sbp(self):

        if self.sbp:
            print('sbp given')
            self.gamma0 = self.sbp[GAMMA0]  # concentration parameter for stick breaking
            self.kappa0 = self.sbp[KAPPA0]  # sticky-ness
            self.alpha0 = self.sbp[ALPHA0]  # concentration for 2nd stick breaking
        else:
            self.gamma0 = 10  # concentration parameter for stick breaking
            self.kappa0 = 10  # sticky-ness
            self.alpha0 = 10  # concentration for 2nd stick breaking

        self.rho0 = self.kappa0 / (self.kappa0 + self.alpha0)

        gem_a = np.ones(self.K) * (self.gamma0 / self.K)  # GEM 1
        gem_a[gem_a < 0.01] = 0.01
        self.beta = dirichlet.rvs(gem_a, size=1)[0]  # assume this is a draw from infinite SBP

        gem_b = self.alpha0 * self.beta
        gem_b[gem_b < 0.01] = 0.01
        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            stickyness_addition = np.zeros(self.K)
            stickyness_addition[k] = self.kappa0
            A[k] = dirichlet.rvs(gem_b + stickyness_addition, size=1)[0]
        self.A = self.normalize_matrix(A)
        pi = dirichlet.rvs(gem_b, size=1)[0]
        self.pi = self.normalize_matrix(pi)

    def sample_z(self):

        backwards, likelihood = self.get_backwards_hmmlearn()
        # Z_a = sample_states_numba(backwards,
        #                             self.pi,
        #                             self.X,
        #                             self.mu,
        #                             self.sigma,
        #                             self.A,
        #                             self.N)

        Z_b = sample_states_likelihood(backwards,
                                       self.pi,
                                       self.A,
                                       self.N,
                                       likelihood)

        self.Z = Z_b

    def get_backwards_hmmlearn(self):
        likelihood = _log_multivariate_normal_density_diag(self.X, self.mu, np.diagonal(self.sigma, axis1=1, axis2=2))
        bwdlattice = backward_log(
            self.pi, self.A, likelihood)
        return bwdlattice, likelihood

    def update_ss(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            if self.nk[k] > 0:
                self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

        # n_mat - count transitions from i to j
        n_mat = np.zeros((self.K, self.K))
        for i in range(self.K):
            # find indices of states that come right after state i
            indices = np.where(self.Z == i)[0] + 1  # indices of X_k

            # need to address the case for the last state in the sequence
            if self.N in indices:
                indices = np.delete(indices, np.where(indices == self.N))

            states = self.Z[indices]
            n_i = np.zeros(self.K)
            for j in range(self.K):
                n_i[j] = np.count_nonzero(states == j)
            n_mat[i, :] = n_i
        self.n_mat = n_mat

        # n_ft
        n_ft = np.zeros((self.K))
        n_ft[int(self.Z[0])] += 1
        self.n_ft = n_ft

    def sample_aux_vars(self):
        m_mat = np.zeros((self.K, self.K))

        for j in range(self.K):
            for k in range(self.K):
                if self.n_mat[j, k] == 0:
                    m_mat[j, k] = 0
                else:
                    # pretend multivariate to len(n_mat) at once and avoid loop
                    x_vec = binomial(1, (self.alpha0 * self.beta[k] + self.kappa0 * (j == k)) / (
                            np.arange(self.n_mat[j, k]) + self.alpha0 * self.beta[k] + self.kappa0 * (j == k)))
                    x_vec = np.array(x_vec).reshape(-1)
                    m_mat[j, k] = sum(x_vec)
        self.m_mat = m_mat

        w_vec = np.zeros(self.K)
        m_mat_bar = m_mat.copy()

        if self.kappa0 > 0:
            stick_ratio = self.rho0  # rho0/(rho0+alpha0);
            for j in range(self.K):
                if m_mat[j, j] > 0:
                    w_vec[j] = binomial(m_mat[j, j], stick_ratio / (stick_ratio + self.beta[j] * (1 - stick_ratio)))
                    m_mat_bar[j, j] = m_mat[j, j] - w_vec[j]
        self.w_vec = w_vec
        self.m_mat_bar = m_mat_bar

        # first time point
        m_init = np.zeros(self.K)
        for j in range(self.K):
            if self.n_ft[j] == 0:
                m_init[j] = 0
            else:
                x_vec = binomial(1, self.alpha0 * self.beta[j] / (np.arange(self.n_ft[j]) + self.alpha0 * self.beta[j]))
                x_vec = np.array(x_vec).reshape(-1)
                m_init[j] = sum(x_vec)

        self.m_init = m_init

    def sample_beta(self):
        prob_vec = self.m_mat_bar.sum(axis=0) + (self.gamma0 / self.K) + self.m_init
        prob_vec[prob_vec < 0.01] = 0.01
        beta_vec = dirichlet.rvs(prob_vec, size=1)[0]
        self.beta = beta_vec

    @staticmethod
    def add_tiny_amount(matrix, tiny_amount=1e-5):
        # Add tiny_amount to elements less than or equal to 0
        matrix = np.where(matrix <= 0, matrix + tiny_amount, matrix)
        return matrix

    def normalize_matrix(self, matrix):
        matrix = self.add_tiny_amount(matrix)
        return matrix / np.sum(matrix, axis=(matrix.ndim - 1), keepdims=True)

    def sample_A(self):
        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            prob_vec = (self.alpha0 * self.beta) + self.n_mat[k]
            prob_vec[k] += self.kappa0
            prob_vec[prob_vec < 0.01] = 0.01
            A[k] = dirichlet.rvs(prob_vec, size=1)[0]
        self.A = self.normalize_matrix(A)

        prob_vec = (self.alpha0 * self.beta) + self.n_ft
        prob_vec[prob_vec < 0.01] = 0.01
        pi = dirichlet.rvs(prob_vec, size=1)[0]
        self.pi = self.normalize_matrix(pi)

    def sample_theta(self):
        # [1] sample params using assignments
        for k in range(self.K):
            # if count is 0 then don't sample
            if self.nk[k] > 0:
                # mu
                Vk = (np.linalg.inv(np.linalg.inv(self.V0) + self.nk[k] * np.linalg.inv(self.sigma[k])))
                term1 = np.dot(np.linalg.inv(self.sigma[k]), self.nk[k] * self.x_bar[k])
                term2 = np.dot(np.linalg.inv(self.V0), self.m0)
                mk = (np.dot(Vk, term1 + term2))
                if is_symmetric_positive_semidefinite(Vk) is not True:
                    Vk = np.diag(np.diag(Vk))
                # sample mu
                mu_k = np.random.multivariate_normal(mean=mk, cov=Vk, size=1).flatten()

                self.mu[k] = mu_k

                # self.trace['mu'][-1][k]

                # sigma
                dif = (self.X[self.Z == k] - self.mu[k])
                Sk = (self.S0 + (np.dot(dif.T, dif)))
                nuk = self.nu0 + self.nk[k]

                if is_symmetric_positive_semidefinite(Sk) is not True:
                    Sk = np.diag(np.diag(Sk))
                # sample sigma
                self.sigma[k] = np.diag(np.diag(invwishart.rvs(size=1, df=nuk, scale=Sk)))

    def create_hmm(self, covariance_type='diag'):
        # use the latest samples to create a hmmLearn object

        A = np.copy(self.A)
        pi = np.copy(self.pi)
        means = np.copy(self.mu)
        covar = np.copy(self.sigma)
        n_components = np.copy(self.K)

        # remove states with zero counts
        nk = np.zeros(self.K)
        for k in range(self.K):
            nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(nk > 0)[0]
        if len(zero_indices) > 0:
            rem_ind = zero_indices.astype(int)
            A = A[rem_ind][:, rem_ind]
            pi = pi[rem_ind]
            means = means[rem_ind]
            covar = covar[rem_ind]
            n_components = len(rem_ind)

        # creat hmm
        hmm = GaussianHMM(n_components, covariance_type=covariance_type, init_params='')
        hmm.n_features = self.D
        hmm.transmat_, hmm.startprob_, hmm.means_ = self.normalize_matrix(A), self.normalize_matrix(pi), means
        if covariance_type == 'diag':
            hmm.covars_ = np.array([np.diag(i) for i in covar])
        else:
            hmm.covars_ = np.array([np.diag(np.diag(i)) for i in covar])

        self.hmm = hmm

    def hmm_from_trace(self, n_components, average_over=100):

        # from the last 'average_over' its, get the sum
        nk = np.copy(self.trace['nk'])[-average_over:]
        nk_sum = np.sum(nk, axis=0)

        largest_indices = np.argpartition(nk_sum, -n_components)[-n_components:]
        # nk_sum_norm = (nk_sum / np.sum(nk_sum)) * 100

        mu_matrix = np.mean(np.array(self.trace['mu'][-average_over:]), axis=0)[largest_indices]
        sigma_matrix = np.mean(np.array(self.trace['covar'][-average_over:]), axis=0)[largest_indices]
        A = np.mean(np.array(self.trace['A'][-average_over:]), axis=0)[largest_indices][:, largest_indices]
        pi = np.mean(np.array(self.trace['pie'][-average_over:]), axis=0)[largest_indices]

        hmm_trace = GaussianHMM(len(largest_indices), covariance_type='diag')
        hmm_trace.n_features = mu_matrix.shape[1]
        hmm_trace.transmat_, hmm_trace.startprob_, hmm_trace.means_ = self.normalize_matrix(
            A), self.normalize_matrix(pi), mu_matrix
        hmm_trace.covars_ = np.array([np.diag(i) for i in sigma_matrix])

        return hmm_trace

    def get_likelihood(self):
        self.create_hmm()
        log_prob, _ = self.hmm.decode(self.X[:200])  # update this for multiple sequences ...
        return log_prob

    def gibbs_sweep(self):
        self.sample_z()
        self.update_ss()
        self.sample_aux_vars()
        self.sample_beta()
        self.sample_A()
        self.sample_theta()

    def plot_components_trace(self):

        data = self.trace['n_components_all']
        # Define the window size for the rolling average
        window_size = 20

        # Create a kernel for the rolling average
        kernel = np.ones(window_size) / window_size

        # Use np.convolve to calculate the rolling average
        rolling_avg = np.convolve(data, kernel, mode='valid')

        # Create an array of indices corresponding to the original data for plotting
        indices = np.arange(window_size - 1, len(data))

        # Plot the data and rolling average

        plt.figure(figsize=(10, 6))
        plt.plot(indices, data[window_size - 1:], label='Original Values', marker='o')
        plt.plot(indices, rolling_avg, label=f'Rolling Average ({window_size} periods)',
                 color='red', linestyle='--', marker='o')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Rolling Average Plot')
        plt.legend()
        plt.show()

    def fit_multiple(self, verbose=False):

        print('starting gibbs sampling')

        # init trace
        self.trace['n_components'] = []
        self.trace['n_components_all'] = []
        self.trace['nk'] = []
        self.trace['A'] = []
        self.trace['pie'] = []
        self.trace['mu'] = []
        self.trace['covar'] = []
        self.trace['n_components_avg'] = []
        self.trace[TIME] = []
        self.trace[LL] = []

        converged = False
        total_its = 0
        current_component_count = self.K
        imbetween_component_count = self.K + 1
        self.ARI = np.zeros(self.iterations)
        start_time = time.time()
        chunk_length = 200
        chunk = False
        if len(self.X_list) == 1 and len(self.X) > 500:
            chunk = True

        for outer_it in range(self.outer_its):

            for i in range(len(self.X_list)):

                self.X = self.X_list[i]
                self.N = self.X.shape[0]

                for it in range(self.iterations):
                    # chunk
                    if chunk:
                        start_position = np.random.randint(0, self.X_list[0].shape[0] - chunk_length)
                        # Extract chunk from the concatenated sequence
                        self.X = self.X_list[0][start_position: start_position + chunk_length, :]
                        self.N = self.X.shape[0]

                    total_its += 1
                    start_time_gibbs = time.time()
                    self.gibbs_sweep()
                    end_time_gibbs = time.time()
                    self.trace[TIME].append(start_time_gibbs - end_time_gibbs)

                    # Calculate ARI
                    if self.Z_true is not None:
                        self.ARI[it] = np.round(ari(self.Z_true, self.Z), 3)

                    # add values to trace
                    cur_nk = np.zeros(self.K)
                    for k in range(self.K):
                        cur_nk[k] = np.sum(self.Z == k)
                    zero_indices = np.where(cur_nk > 0)[0]
                    n_components = len(zero_indices.astype(int))

                    self.trace['n_components_all'].append(len(zero_indices))

                    if total_its > self.burn_in - 1:
                        self.trace['n_components'].append(len(zero_indices))
                        self.trace['nk'].append(cur_nk)
                        self.trace['A'].append(np.copy(self.A))
                        self.trace['pie'].append(np.copy(self.pi))
                        self.trace['mu'].append(np.copy(self.mu))
                        self.trace['n_components_avg'].append(np.mean(self.trace['n_components'][-50:]))
                        self.trace['covar'].append(np.copy(self.sigma))

                    if total_its > 10:
                        diff = self.trace['n_components_avg'][-1] - self.trace['n_components_avg'][-10]
                        if np.abs(diff) < 0.5 and total_its > 100:
                            print('convergence criteria met!')
                            converged = True
                            break
                        print('avg: ', self.trace['n_components_avg'][-1],
                              'diff: ', self.trace['n_components_avg'][-1] - self.trace['n_components_avg'][-10])

                    if total_its % 50 == 0 and verbose:
                        cur_ll = self.get_likelihood()
                        self.trace[LL].append(cur_ll)
                        if verbose: print('it: ', total_its, ' || Likelihood: ', cur_ll, ' || n_components: ',
                                          n_components)
                    if total_its % 100 == 0 and verbose:
                        plot_hmm_learn(self.X, self.hmm, feature_a=self.feature_a, feature_b=self.feature_b)

                    # if total_its % 100 == 0 and verbose: self.plot_components_trace()

                    if total_its % self.convergence_check == 0:
                        # check for convergence
                        if current_component_count == n_components and imbetween_component_count == current_component_count:
                            print('convergence criteria met!')
                            converged = True
                            break
                        else:
                            current_component_count = n_components

                    if total_its % np.round(self.convergence_check / 2) == 0:
                        imbetween_component_count = n_components

                    if total_its == self.max_it:
                        print('max it met!')
                        converged = True
                        break
                if converged:
                    break
            if converged:
                break

        end_time = time.time()
        print('completed gibbs sampling in ', end_time - start_time)

        # set amount of components from trace
        n_comps = int(np.round(np.mean(self.trace['n_components_all'][-50:])))
        return self.hmm_from_trace(n_comps, 200)

    def check_for_close_states(self, js_threshold=-2):
        self.create_hmm()  # update hmm
        similar_states = (useful.find_similar_states_js(self.hmm, self.hmm, 10))
        np.fill_diagonal(similar_states, 0)
        smallest_index = np.unravel_index(np.argmin(similar_states, axis=None), similar_states.shape)
        if similar_states[smallest_index] < js_threshold:
            print('adding', smallest_index[0], 'to', smallest_index[1])
            index_a = np.where(self.hmm.means_[smallest_index[0]] == self.mu)[0][0]
            index_b = np.where(self.hmm.means_[smallest_index[1]] == self.mu)[0][0]
            self.Z[self.Z == index_a] = index_b
            # complete gibbs sweep
            self.update_ss()
            self.sample_aux_vars()
            self.sample_beta()
            self.sample_A()
            self.sample_theta()

    def fit(self, iterations=None, verbose=False):

        if iterations is not None:
            self.iterations = iterations

        # print('fitting using gibbs sampling - iterations: ', self.iterations)

        # init trace

        self.trace[TIME] = []
        self.trace[LL] = []
        self.ARI = np.zeros(self.iterations)
        start_time = time.time()

        for it in range(self.iterations):
            start_time_gibbs = time.time()
            self.gibbs_sweep()
            end_time_gibbs = time.time()
            self.trace[TIME].append(start_time_gibbs - end_time_gibbs)
            # save trace

            # Calculate ARI
            if self.Z_true is not None:
                self.ARI[it] = np.round(ari(self.Z_true, self.Z), 3)

            # remove states with zero counts
            cur_nk = np.zeros(self.K)
            for k in range(self.K):
                cur_nk[k] = np.sum(self.Z == k)
            zero_indices = np.where(cur_nk > 0)[0]
            self.trace['n_components'].append(len(zero_indices))

            if verbose:
                if it % 10 == 0:
                    cur_ll = self.get_likelihood()
                    self.trace[LL].append(cur_ll)
                    print('it: ', it, ' || Likelihood: ', cur_ll, ' || n_components: ', self.hmm.n_components)

                if it % 100 == 0 and it > 0:
                    plot_hmm_learn(self.X, self.hmm, feature_a=self.feature_a, feature_b=self.feature_b)

        end_time = time.time()
        print('completed gibbs sampling in ', end_time - start_time)

    def plot_hmmlearn(self, feature_a=0, feature_b=1, percent=0.1):
        plot_hmm_learn(self.X, self.hmm, feature_a=feature_a, feature_b=feature_b, percent=percent)


if __name__ == '__main__':
    print('demo')
    # my_hmm = InfiniteDirectSamplerHMM(loaded_data, 2, loaded_ss, iterations=40)
    # fit_vars = my_hmm.fit()
    # my_hdp_hmm = InfiniteHMM(loaded_data, 10, loaded_ss, iterations=20)
    # plot_hmm.plot_hmm_data(loaded_data, my_hdp_hmm.Z, my_hdp_hmm.K, my_hdp_hmm.mu, my_hdp_hmm.sigma)
    # my_hdp_hmm.gibbs_sweep()
    # my_hdp_hmm.fit()
    #
    # plt.plot(range(0,len(my_hdp_hmm.ARI)), my_hdp_hmm.ARI, marker="None")
    # plt.xlabel('iteration')
    # plt.ylabel('ARI')
    # #plt.savefig("./image/ari.png")
    # plt.show()
    # plt.close()

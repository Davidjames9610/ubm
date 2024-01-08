# init things
import math
import time
import warnings

from numpy import VisibleDeprecationWarning
from sklearn.cluster import KMeans
from scipy.stats import wishart, dirichlet, invwishart, multivariate_normal
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import numpy as np
from final.models.hdphmm.hdphmmwl.numba_wl import sample_states_likelihood
from hmmlearn.stats import _log_multivariate_normal_density_diag
from hmmlearn._hmmc import backward_log
from final.useful import *


class BayesianHMM:

    def __init__(self, X, K, Z_true=None, burn_in=0, iterations=20, alpha0=1000, kappa0=10, verbose=False, outer_its=1,
                 convergence_check=200, V0=None, nu0=None):
        """
        [] input
        X = data n x d
        K = expected amount of states

        [] params = pi, A, z, mu, Sigma
        init z and pi using uniform alpha, mu and sigma using data and z assignments

        [] hyper-params = rho, alpha, m0, V0, nu0, S0
        for uniform priors we set
        rh0, alpha0 -> small value to allow for states to collapse for pi and A
        m0 -> mean of init assignments
        V0 -> large eye matrix
        nu0 -> K + 1
        S0 -> data scatter matrix

        [] ss
        nk - count of data in each k
        x_bar - mean of data in each k
        """

        # is X a list or sequnece

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

        self.K = K  # expected no of states
        self.burn_in = burn_in
        self.iterations = iterations

        self.N = self.X.shape[0]  # length of data
        self.D = self.X.shape[1]  # dimension of data
        self.Z_true = Z_true

        # init z matrix randomly along with pie and A
        self.alpha0 = alpha0
        alpha_k = np.ones(K) * self.alpha0 / self.K  # Uniform prior
        self.pi = dirichlet.rvs(alpha=alpha_k, size=1).flatten()

        self.kappa0 = kappa0

        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            stickyness_addition = np.zeros(self.K)
            stickyness_addition[k] = self.kappa0
            A[k] = dirichlet.rvs(alpha_k + stickyness_addition, size=1)[0]
        self.A = self.normalize_matrix(A)

        Z_mat = np.random.multinomial(n=1, pvals=self.pi, size=self.N)
        _, self.Z = np.where(Z_mat == 1)  # N x 1 component number

        # self.rho0 = np.array([np.ones(K) * 10 / self.K for i in range(K)])  # small prior
        # update alpha0 here for weights to collapse

        # true means
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(self.X)

        self.true_means_ = kmeans.cluster_centers_
        self.true_covars_ = [np.cov(self.X[kmeans.labels_ == i], rowvar=False) for i in range(K)]

        # shuffle labels
        num_labels_to_replace = int(0.8 * len(kmeans.labels_))
        # Generate random labels between 0 and k
        random_labels = np.random.randint(0, K, num_labels_to_replace)
        # Replace 10% of the labels with random numbers
        shuffled_labels = np.copy(kmeans.labels_)
        replace_indices = np.random.choice(len(shuffled_labels), num_labels_to_replace, replace=False)
        shuffled_labels[replace_indices] = random_labels
        # Assign the shuffled labels to self.Z
        self.Z = kmeans.labels_
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
            if len(x_k) < 10:
                random_indices = np.random.randint(0, self.N, int(self.N * 0.8))
                x_k = self.X[random_indices]
            sig_bar = np.cov(x_k.T, bias=True)
            diagsig[k] = np.diag(np.diag(sig_bar))
            self.mu[k] = self.x_bar[k]
            self.sigma[k] = np.diag(np.diag(sig_bar))
            # self.lambdas[k] = np.linalg.inv(sig_bar)

        # sig_bar = np.cov(self.X.T, bias=True)
        universal_sig_bar = np.cov(self.X.T, bias=True)
        self.S0 = np.diag(np.diag(universal_sig_bar))
        max_S0 = np.max(self.S0)
        # Hyper-parameters for normals
        # Mu
        self.m0 = np.mean(self.X, axis=0)
        # self.m0 = np.mean(self.X, axis=0)  # 1 x D
        if V0 is not None:
            self.V0 = V0
        else:
            self.V0 = np.eye(self.D) * 1000  # self.get_magnitude(max_S0)

        if nu0 is not None:
            self.nu0 = nu0  # 1 Degrees of freedom IW
        else:
            self.nu0 = np.copy(self.D) + 2  # 1 Degrees of freedom IW

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.trace = {}
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []
        self.verbose = verbose
        self.convergence_check = convergence_check
        self.hmm: GaussianHMM | None = None

    @staticmethod
    def get_magnitude(number):
        # Handle the case where the number is 0 separately
        if number == 0:
            return 0

        # Calculate the magnitude using the logarithm
        magnitude = int(math.floor(math.log10(abs(number))))

        # Adjust the result to match the scientific notation format
        return 10 ** magnitude

    def sample_states(self):

        backwards, likelihood = self.get_backwards_hmmlearn()

        Z = sample_states_likelihood(backwards,
                                     self.pi,
                                     self.A,
                                     self.N,
                                     likelihood)

        self.Z = Z

    def get_backwards_hmmlearn(self):
        likelihood = _log_multivariate_normal_density_diag(self.X, self.mu, np.diagonal(self.sigma, axis1=1, axis2=2))
        bwdlattice = backward_log(
            self.pi, self.A, likelihood)
        return bwdlattice, likelihood

    # for A
    def get_rho_k(self):
        rho_k = np.zeros((self.K, self.K))
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
            rho_k[i, :] = n_i
        return rho_k

    def fit(self):

        print('starting gibbs sampling')

        self.likelihood_history = []

        self.trace['nk'] = []
        self.trace['A'] = []
        self.trace['pie'] = []
        self.trace['mu'] = []
        self.trace['covar'] = []
        self.trace['time'] = 0
        self.trace['n_components'] = []
        self.trace['n_components_avg'] = []

        converged = False
        total_its = 0
        current_component_count = self.K
        self.ARI = np.zeros(self.iterations)
        start_time = time.time()

        for outer_it in range(self.outer_its):

            if self.verbose: print('outer it: ', outer_it)

            for i in range(len(self.X_list)):
                self.X = self.X_list[i]
                self.N = self.X.shape[0]

                # update
                # kmeans = KMeans(n_clusters=self.K, random_state=42)
                # kmeans.fit(self.X)
                # self.Z = kmeans.labels_
                # self.handle_empty_components()
                # self.update_ss()
                # self.sample_theta()
                # self.sample_a()

                for it in range(self.iterations):
                    total_its += 1
                    self.gibbs_sweep()
                    cur_nk = np.zeros(self.K)
                    for k in range(self.K):
                        cur_nk[k] = np.sum(self.Z == k)
                    self.trace['nk'].append(cur_nk)
                    self.trace['A'].append(np.copy(self.A))
                    self.trace['pie'].append(np.copy(self.pi))
                    self.trace['mu'].append(np.copy(self.mu))
                    self.trace['covar'].append(np.copy(self.sigma))
                    self.trace['n_components'].append(self.K)
                    self.trace['n_components_avg'].append(np.mean(self.trace['n_components'][-10:]))

                    # save trace
                    # if it > self.burn_in:
                    self.mu_trace.append(self.mu)
                    self.sigma_trace.append(self.sigma)
                    self.pi_trace.append(self.pi)

                    if total_its % 10 == 0 and self.verbose:
                        curr_hmm = self.hmm_from_trace(self.K, 25)
                        ll = 0
                        if curr_hmm is not None:
                            ll = curr_hmm.score(self.X)
                            self.likelihood_history.append(ll)
                        print('it: ', total_its, 'score: ',ll, 'n-components: ', self.K)
                        print('\n')
                        print(self.nk)

                    if total_its % self.convergence_check == 0:
                        # check for convergence
                        if current_component_count == self.K:
                            print('convergence criteria met!')
                            converged = True
                            break
                        else:
                            current_component_count = self.K

                    if total_its > 10:
                        diff = self.trace['n_components_avg'][-1] - self.trace['n_components_avg'][-10]
                        # if np.abs(diff) < 0.5 and total_its > 50:
                        #     print('convergence criteria met! diff < 0.5: ', np.abs(diff))
                        #     converged = True
                        #     break
                        print('it: ', total_its, 'avg: ',  self.trace['n_components_avg'][-1],
                              'diff: ', self.trace['n_components_avg'][-1] - self.trace['n_components_avg'][-10])

                    # Calculate ARI
                    if self.Z_true is not None:
                        self.ARI[it] = np.round(ari(self.Z_true, self.Z), 3)
                        if converged:
                            break
                if converged:
                    break
            if converged:
                break
        end_time = time.time()

        self.trace['time'] = end_time - start_time

        print('Completed gibbs sampling -- Convergence: ', converged, ' -- In: ', end_time - start_time)

        if converged:
            new_hmm = self.hmm_from_trace(self.K, 20)
            if new_hmm is not None:
                self.hmm = new_hmm
            else:
                self.hmm = self.hmm_from_trace(self.K, 1)
            return self.hmm
        else:
            self.hmm = self.hmm_from_trace(self.K, 1)
            return self.hmm

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
                Vk = np.diag(np.diag(Vk))
                # sample mu
                mu_k = np.random.multivariate_normal(mean=mk, cov=Vk, size=1).flatten()

                self.mu[k] = mu_k

                # self.trace['mu'][-1][k]

                # sigma
                dif = (self.X[self.Z == k] - self.mu[k])
                Sk = (self.S0 + (np.dot(dif.T, dif)))
                nuk = self.nu0 + self.nk[k]

                Sk = np.diag(np.diag(Sk))
                # sample sigma
                self.sigma[k] = np.diag(np.diag(invwishart.rvs(size=1, df=nuk, scale=Sk)))

    def sample_a(self):
        # sample pi
        alpha_k = np.ones(self.K) * self.alpha0 / self.K
        alpha_pi_k = self.nk + alpha_k  # could update this
        self.pi = dirichlet.rvs(size=1, alpha=alpha_pi_k).flatten()

        # sample A
        # alpha_A_k = self.get_rho_k() + alpha_k + (np.eye(self.D) * 10)
        # self.A = [dirichlet.rvs(alpha=alpha_A_k[k], size=1).flatten() for k in range(self.K)]
        # self.A = np.vstack(self.A)
        # self.A[np.where(self.A < 1e-2)] = 1e-2
        # self.A = self.normalize_matrix(self.A)

        n_mat = self.get_rho_k()

        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            prob_vec = alpha_k + n_mat[k]
            prob_vec[k] += self.kappa0
            prob_vec[prob_vec < 0.01] = 0.01
            A[k] = dirichlet.rvs(prob_vec, size=1)[0]
        self.A = self.normalize_matrix(A)

    def update_ss(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            if self.nk[k] > 0:
                self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

    def gibbs_sweep(self):
        self.sample_states()
        self.handle_empty_components()
        self.update_ss()
        self.sample_theta()
        self.sample_a()

    def handle_empty_components(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(self.nk == 0)[0]
        if len(zero_indices) > 0:
            if self.verbose: print('removing empty component(s)')
            rem_ind = np.unique(self.Z).astype(int)
            d = {k: v for v, k in enumerate(sorted(rem_ind))}
            self.Z = np.array([d[x] for x in self.Z])

            A = self.A[rem_ind][:, rem_ind]
            self.A = normalize_matrix(A)

            self.pi = self.pi[rem_ind]
            self.mu = self.mu[rem_ind]
            self.sigma = self.sigma[rem_ind]

            # ss
            self.nk = self.nk[rem_ind]
            self.x_bar = self.x_bar[rem_ind]

            self.S0 = self.S0
            self.m0 = self.m0

            self.K = len(rem_ind)

    def hmm_from_trace(self, n_components, average_over=100):

        # from the last 'average_over' its, get the sum
        nk = np.copy(self.trace['nk'])[-average_over:]
        if np.all([len(arr) == len(nk[-1]) for arr in nk]):
            nk_sum = np.sum(nk, axis=0)

            largest_indices = np.argpartition(nk_sum, -n_components)[-n_components:]
            # nk_sum_norm = (nk_sum / np.sum(nk_sum)) * 100

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=VisibleDeprecationWarning)
            mu_matrix = np.mean(np.stack(self.trace['mu'][-average_over:], axis=0), axis=0)[largest_indices]
            sigma_matrix = np.mean(np.stack(self.trace['covar'][-average_over:], axis=0), axis=0)[largest_indices]
            A = np.mean(np.stack(self.trace['A'][-average_over:], axis=0), axis=0)[largest_indices][:, largest_indices]
            pi = np.mean(np.stack(self.trace['pie'][-average_over:], axis=0), axis=0)[largest_indices]

            hmm_trace = GaussianHMM(len(largest_indices), covariance_type='diag')
            hmm_trace.n_features = mu_matrix.shape[1]
            hmm_trace.transmat_, hmm_trace.startprob_, hmm_trace.means_ = self.normalize_matrix(
                A), self.normalize_matrix(pi), mu_matrix
            hmm_trace.covars_ = np.stack([np.diag(i) for i in sigma_matrix], axis=0)

            return hmm_trace
        else:
            return None

    def normalize_matrix(self, matrix):
        matrix = self.add_tiny_amount(matrix)
        return matrix / np.sum(matrix, axis=(matrix.ndim - 1), keepdims=True)

    @staticmethod
    def add_tiny_amount(matrix, tiny_amount=1e-5):
        # Add tiny_amount to elements less than or equal to 0
        matrix = np.where(matrix <= 0, matrix + tiny_amount, matrix)
        return matrix

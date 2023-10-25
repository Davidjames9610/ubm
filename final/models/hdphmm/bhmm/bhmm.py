# init things
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

    def __init__(self, X, K, Z_true=None,burn_in=0, iterations=20):
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
        self.X = X  # data n x d
        self.K = K  # expected no of states
        self.burn_in = burn_in
        self.iterations = iterations
        self.N = X.shape[0]  # length of data
        self.D = X.shape[1]  # dimension of data
        self.Z_true = Z_true

        # init z matrix randomly along with pie and A
        self.alpha0 = np.ones(K) * 1000 / self.K  # Uniform prior
        self.pi = dirichlet.rvs(alpha=self.alpha0, size=1).flatten()
        self.A = dirichlet.rvs(alpha=self.alpha0, size=K)
        Z_mat = np.random.multinomial(n=1, pvals=self.pi, size=self.N)
        _, self.Z = np.where(Z_mat == 1)  # N x 1 component number

        self.alpha0 = np.ones(K) * 10 / self.K  # small prior
        self.rho0 = np.array([np.ones(K) * 10 / self.K for i in range(K)])  # small prior
        # update alpha0 here for weights to collapse

        # true means
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(X)

        self.true_means_ = kmeans.cluster_centers_
        self.true_covars_ = [np.cov(X[kmeans.labels_ == i], rowvar=False) for i in range(K)]

        # shuffle labels
        num_labels_to_replace = int(0.05 * len(kmeans.labels_))
        # Generate random labels between 0 and k
        random_labels = np.random.randint(0, K, num_labels_to_replace)
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
            if len(x_k) < 10:
                random_indices = np.random.randint(0, self.N, int(self.N * 0.1))
                x_k = self.X[random_indices]
            sig_bar = np.cov(x_k.T, bias=True)
            diagsig[k] = np.diag(np.diag(sig_bar))
            self.mu[k] = self.x_bar[k]
            self.sigma[k] = sig_bar
            self.lambdas[k] = np.linalg.inv(sig_bar)
        # Hyper-parameters for normals
        # Mu
        self.m0 = np.copy(self.x_bar)  # K x D
        if np.isnan(self.m0).any():
            print('nan')
        self.V0 = [np.eye(self.D) * 1000 for _ in range(K)]  # K x D x D
        # Sigma
        self.S0 = diagsig  # K x D x D
        self.nu0 = np.copy(self.D) + 2  # 1 Degrees of freedom IW

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []


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
        max_iterations = 20
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

        print('completed gibbs sampling')

    # one sweep of git sampler, return variables sampled

    def sample_theta(self):
        for k in range(self.K):
            # if count is 0 then sample from prior
            if self.nk[k] > 0:
                Vk = (np.linalg.inv(np.linalg.inv(self.V0[k]) + self.nk[k] * np.linalg.inv(np.diag(np.diag(self.sigma[k])))))
                term1 = np.dot(np.linalg.inv(np.diag(np.diag(self.sigma[k]))), self.nk[k] * self.x_bar[k])
                term2 = np.dot(np.linalg.inv(self.V0[k]), self.m0[k])
                mk = (np.dot(Vk, term1 + term2))
                # sample mu
                mu_k = np.random.multivariate_normal(mean=mk, cov=np.diag(np.diag(Vk)), size=1).flatten()

                self.mu[k] = mu_k

                # sigma
                dif = (self.X[self.Z == k] - self.mu[k])
                Sk = np.diag(np.diag(self.S0[k] + (np.dot(dif.T, dif))))
                nuk = self.nu0 + self.nk[k]
                # sample sigma
                self.sigma[k] = invwishart.rvs(size=1, df=nuk, scale=Sk)

    def sample_a(self):
        # sample pi
        alpha_k = self.nk + self.alpha0
        self.pi = dirichlet.rvs(size=1, alpha=alpha_k).flatten()

        # sample A
        rho_k = self.rho0 + self.get_rho_k()
        self.A = [dirichlet.rvs(alpha=rho_k[k], size=1).flatten() for k in range(self.K)]
        self.A = np.vstack(self.A)

    def update_ss(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            if self.nk[k] > 0:
                self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)

    def gibbs_sweep(self):
        self.sample_theta()
        self.sample_a()
        self.sample_states()
        self.handle_empty_components()
        self.update_ss()

    def handle_empty_components(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(self.nk == 0)[0]
        if len(zero_indices) > 0:
            if len(zero_indices) >= 1:
                print('more than one zero component')
            rem_ind = np.unique(self.Z).astype(int)
            d = {k: v for v, k in enumerate(sorted(rem_ind))}
            self.Z = np.array([d[x] for x in self.Z])

            A = self.A[rem_ind][:, rem_ind]
            self.A = normalize_matrix(A)
            rho0 = self.rho0[rem_ind][:, rem_ind]
            self.rho0 = normalize_matrix(rho0)
            self.pi = self.pi[rem_ind]
            self.mu = self.mu[rem_ind]
            self.sigma = self.sigma[rem_ind]

            # ss
            self.nk = self.nk[rem_ind]
            self.x_bar = self.x_bar[rem_ind]

            # hp
            self.alpha0 = self.alpha0[rem_ind]

            self.K = len(rem_ind)

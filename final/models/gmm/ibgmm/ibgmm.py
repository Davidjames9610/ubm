import seaborn as sns;
from sklearn import mixture

from final.models.hdphmm.hdphmmwl.numba_wl import compute_probabilities, multinomial

sns.set()
import numpy as np
from scipy.stats import  wishart, dirichlet, invwishart, multivariate_normal
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

class InfiniteGMMGibbsSampler():
    def __init__(self, X, K, burn_in=0, iterations=40, **kwargs):
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
        self.N = X.shape[0] # length of data
        self.D = X.shape[1] # dimension of data
        self.n_mce = int(1e2)    # n for monte carlo estimate

        self.Z_true = kwargs.get("Z_true")

        # set alpha to uniform prior and draw pi and z
        self.alpha0 = 1000/self.K # Uniform prior
        alpha_k = np.ones(self.K) * self.alpha0
        self.pi = dirichlet.rvs(alpha=alpha_k, size=1).flatten()
        Z_mat = np.random.multinomial(n=1, pvals=self.pi, size=self.N) # N x K 'one-hot vectors'
        _, self.Z = np.where(Z_mat == 1) # N x 1 component number

        self.alpha0 = 100 / self.K  # tiny prior !

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
        self.m0 = np.copy(self.x_bar) # K x D
        if np.isnan(self.m0).any():
            print('nan')
        self.V0 = [np.eye(self.D) * 10 for _ in range(K)] # K x D x D
        # Sigma
        self.S0 = diagsig # K x D x D
        self.nu0 = np.copy(self.D) + 2  # 1 Degrees of freedom IW

        self.mu_trace = []
        self.sigma_trace = []
        self.pi_trace = []
        self.ARI = np.zeros((self.iterations))
        self.likelihood_history = []
        self.gmm = None

    def has_converged(self, previous_likelihood, current_likelihood, tolerance):
        # Calculate the relative change in likelihood
        relative_change = np.abs((current_likelihood - previous_likelihood) / previous_likelihood)
        return relative_change < tolerance

    def calculate_likelihood(self):
        likelihood = 0.0
        for i in range(self.N):
            point_likelihood = 0.0
            for k in range(self.K):
                component_likelihood = self.pi[k] * multivariate_normal.pdf(self.X[i], mean=self.mu[k], cov=self.sigma[k])
                point_likelihood += component_likelihood
            likelihood += np.log(point_likelihood)  # Use log likelihood to avoid numerical underflow
        return likelihood

    def sample_theta(self):
        for k in range(self.K):
            # mu
            Vk = (np.linalg.inv(np.linalg.inv(self.V0[k]) + self.nk[k] * np.linalg.inv(self.sigma[k])))
            term1 = np.dot(np.linalg.inv(self.sigma[k]), self.nk[k] * self.x_bar[k])
            term2 = np.dot(np.linalg.inv(self.V0[k]), self.m0[k])
            mk = (np.dot(Vk, term1 + term2))
            # sample mu
            mu_k = np.random.multivariate_normal(mean=mk, cov=Vk, size=1).flatten()
            if np.isnan(mu_k).any():
                print('oh dear')
            else:
                self.mu[k] = mu_k

            # sigma
            dif = (self.X[self.Z == k] - self.mu[k])
            Sk = (self.S0[k] + (np.dot(dif.T, dif)))
            nuk = self.nu0 + self.nk[k]
            # sample sigma
            self.sigma[k] = invwishart.rvs(size=1, df=nuk, scale=Sk)

    def sample_pi(self):
        alpha_k = self.nk + self.alpha0
        self.pi = dirichlet.rvs(size=1, alpha=alpha_k).flatten()

    def niw_mc_estimate(self, x, n_mce, m=None, v=None, nu=None, s=None,random_k=None):

        if(m is None):
            if random_k is None:
                random_k = np.random.randint(0, self.K)
            m = self.m0[random_k]
            v = self.V0[random_k]
            nu = self.nu0
            s = self.S0[random_k]

        mus = []
        sigmas = []
        for i in range(n_mce):
            mus.append(np.random.multivariate_normal(mean=m, cov=v, size=1).flatten())
            sigmas.append(invwishart.rvs(size=1, df=nu, scale=s))
        p = 0
        for i in range(n_mce):
            p += multivariate_normal.pdf(x, mus[i], sigmas[i]) * multivariate_normal.pdf(mus[i], m, v) * invwishart.pdf(
                sigmas[i], df=nu, scale=s)
        return np.log(p / n_mce)
    def calculate_crp(self):

        prob_k = self.nk / (self.N -1 + self.alpha0)
        prob_new = self.alpha0 / (self.N -1 + self.alpha0)
        return np.log(prob_k), np.log(prob_new)

    def calculate_posterior_theta(self, x):

        theta_k = np.zeros(self.K)
        for k in range(self.K):
            theta_k[k] = multivariate_normal.logpdf(x, mean=self.mu[k], cov=self.sigma[k])

        theta_new = self.niw_mc_estimate(x, int(1e2), random_k=0)

        return theta_k, theta_new

    def sample_z(self):

        for i in range(self.N):
            x = self.X[i]
            self.remove_x_from_z(i)

            crp_k, crp_new = self.calculate_crp()
            theta_k, theta_new = self.calculate_posterior_theta(x)

            prob = np.hstack((crp_k+theta_k,crp_new+theta_new))

            post_cases_probs = compute_probabilities(prob)
            self.Z[i] = multinomial(post_cases_probs)

            # new state
            if self.Z[i] == self.K:
                print('new state')
                self.nk = np.hstack((self.nk, [0]))
                self.x_bar = np.hstack((self.x_bar, [0]))
                self.K += 1

            self.add_x_to_z(i)
            self.handle_empty_components()

    def handle_empty_components(self):
        nk = np.zeros(self.K)
        for k in range(self.K):
            nk[k] = np.sum(self.Z == k)
        zero_indices = np.where(nk == 0)[0]
        if len(zero_indices) > 0:
            print('deleting component(s), ', zero_indices)
            rem_ind = np.unique(self.Z)
            d = {k: v for v, k in enumerate(sorted(rem_ind))}
            self.Z = np.array([d[x] for x in self.Z])
            self.sigma = self.sigma[rem_ind][:, rem_ind]
            self.mu = self.mu[rem_ind]
            self.K = len(rem_ind)

    def remove_x_from_z(self, index):
        x_i = self.X[index]
        x_z = self.Z[index]

        self.Z[index] = -1
        self.nk[x_z] -= 1
        self.x_bar[x_z] -= x_i

    def add_x_to_z(self, index):
        x_i = self.X[index]
        x_z = self.Z[index]

        self.nk[x_z] += 1
        self.x_bar[x_z] += x_i

    def update_ss(self):
        for k in range(self.K):
            self.nk[k] = np.sum(self.Z == k)
            # avoid zero assignments
            if self.nk[k] > 0:
                self.x_bar[k] = np.mean(self.X[self.Z == k], axis=0)
            else:
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
        self.sample_theta() # not strictly needed since it has been integrated out

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
            current_likelihood = self.calculate_likelihood()
            self.likelihood_history.append(current_likelihood)
            # if it > 0:
            #     if self.has_converged(likelihood_history[-1], current_likelihood, convergence_threshold):
                    # print(f"Converged after {it} iterations.")
                    # break

        bgmm = mixture.GaussianMixture(n_components=self.K, covariance_type="full")
        bgmm.means_, bgmm.covariances_, bgmm.weights_, bgmm.precisions_cholesky_ = self.mu, self.sigma, self.pi, _compute_precision_cholesky(self.sigma, "full")
        bgmm.precisions_ = bgmm.covariances_** 2
        bgmm.converged_ = True

        self.gmm = bgmm

# my_gibbs = GMMGibbsSampler(x_i, 3, )
# np.sum(my_gibbs.Z == 0)
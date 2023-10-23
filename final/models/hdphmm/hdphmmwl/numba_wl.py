import numpy as np
from numba import jit
from scipy.stats import multivariate_normal
from numba_stats import qgaussian
from jax.scipy.stats import multivariate_normal as jax_mvn
import time


@jit(nopython=True)
def backward_robust_mv(A, means, covariances, observations):
    num_states = A.shape[0]
    num_observed = observations.shape[0]
    beta = np.zeros((num_observed, num_states))
    beta[observations.shape[0] - 1] = np.zeros(num_states)
    for t in range(num_observed - 2, -1, -1):
        for i in range(num_states):
            logbeta = 10000  # this will act as a hacky substitute for None because numba can't take in None
            for j in range(num_states):
                obs_prob = logpdf(observations[t + 1], means[j], np.diag(np.diag(covariances[j])))
                logbeta = elnsum(logbeta, elnproduct(eln(A[j, i]), elnproduct(obs_prob, beta[t + 1, j])))
                if np.isnan(logbeta):
                    print('warning np.isnan(logbeta)')
            beta[t, i] = logbeta
    return beta

@jit(nopython=True)
def logpdf(x, mean, cov):
    vals, vecs = np.linalg.eigh(cov)
    logdet = np.sum(np.log(vals))
    valsinv = 1. / vals
    U = vecs * np.sqrt(valsinv)
    dim = len(vals)
    dev = x - mean
    maha = np.square(np.dot(dev, U)).sum()
    log2pi = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + maha + logdet)


# @jit(nopython=True)
def sample_states_numba_mv(beta: np.ndarray, initial_dist: np.ndarray, observations: np.ndarray,
                           mu: np.ndarray, sigma: np.ndarray, A: np.ndarray, num_obs: int) -> np.ndarray:
    """
    numba version of function to sample from state distribution

    :param beta: backward algorithm probability ndarray
    :param initial_dist: np array of initial state probabilities
    :param observations:
    :param mu: np array holding mus corresponding to each state
    :param sigma_invsq: inverse of variance
    :param A: stochastic matrix of transition probabilities
    :param num_obs: integer number representing number of observations
    :return: updated state path
    """

    # equation (6) in /literature/Bayesian\ Model.pdf
    log_probabilities = np.log(initial_dist) + np.array([logpdf(observations[0], mean=mean, cov=np.diag(np.diag(cov))) for mean, cov in zip(mu, sigma)]) + beta[0]
    probabilities = compute_probabilities(log_probabilities)

    # construct new np.array to hold the sampled state path
    new_state_path = np.empty(num_obs)

    new_state_path[0] = multinomial(probabilities)

    for i in range(1, num_obs):
        # equation (7) in /literature/Bayesian\ Model.pdf
        log_probabilities = np.log(A[int(new_state_path[i - 1]), :]) + np.array([logpdf(observations[i], mean=mean, cov=np.diag(np.diag(cov))) for mean, cov in zip(mu, sigma)]) + beta[i]
        probabilities = compute_probabilities(log_probabilities)
        new_state_path[i] = multinomial(probabilities)
    return new_state_path

@jit(nopython=True)
def eexp(x):
    """

    :param x: x
    :return: exp(x)
    """
    if x == 10000:
        return 0
    else:
        return np.exp(x)


@jit(nopython=True)
def eln(x):
    """

    :param x: x
    :return: ln(x)
    """
    if x < 0:
        print("negative input error")
        return
    if x == 0:
        return np.nan
    elif x > 0:
        return np.log(x)


@jit(nopython=True)
def elnsum(x, y):
    """

    :param x: eln(x)
    :param y: eln(y)
    :return: ln(x + y)
    """
    if x == 10000 or y == 10000:
        if x == 10000:
            return y
        else:
            return x

    else:
        if x > y:
            return x + eln(1 + np.exp(y - x))
        else:
            return y + eln(1 + np.exp(x - y))


@jit(nopython=True)
def elnproduct(x, y):
    """

    :param x: eln(x)
    :param y: eln(y)
    :return: ln(x) + ln(y)
    """
    if x == 10000 or y == 10000:
        return np.nan
    else:
        return x + y


@jit(nopython=True)
def compute_probabilities(log_probabilities):
    """
    Stable way of computing probabilities from log probabilities
    Given a log-likelihood vector, convert to probability vector using method found in:
    https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability

    :param log_probabilities: log likelihood vector
    :return: probability vector
    """
    probabilities = np.exp(log_probabilities) / np.sum(np.exp(log_probabilities))  # make probs add up to 1

    if np.sum(probabilities) == 1:  # just in case there is some underflow/overflow stuff
        return probabilities

    else:
        max = np.amax(log_probabilities)
        log_probabilities = log_probabilities - max
        valid_probabilities = np.empty(0)
        for i in range(len(log_probabilities)):
            if log_probabilities[i] > -38:
                valid_probabilities = np.append(valid_probabilities,
                                                np.exp(log_probabilities[i]))  # append likelihood, not log likelihood
            else:
                valid_probabilities = np.append(valid_probabilities, 0)

        return valid_probabilities / np.sum(valid_probabilities)


@jit(nopython=True)
def multinomial(probabilities):
    """
    wrote my own multinomial function because numba had trouble handling underflow/overflow variability
    more on the issue can be found here: https://github.com/numba/numba/issues/3426

    :param probabilities: probability numpy array
    :return:
    """
    select = np.random.uniform(0, 1)

    for i in range(0, len(probabilities)):
        if select <= np.sum(probabilities[:i + 1]):
            return i


import scipy.stats as stats
import numpy as np
from numba import jit


def normal_pdf(x, mu, sigma=0.5):
    """
    :param x: observation
    :param mu: mean
    :param sigma: standard deviation
    :return: probability
    """
    return stats.norm.pdf(x, mu, sigma)


import numpy as np
from numba import jit


# @jit(nopython=True)
def multivariate_normal_log_pdf(x, mu, cov):
    """
    Calculate the log PDF of a multivariate normal distribution.
    """
    n = len(x)

    # Check dimensions
    if len(mu) != n or cov.shape != (n, n):
        raise ValueError("Incompatible dimensions for mean and covariance matrix.")

    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return_val = float(part1 * np.exp(part2))
    if return_val <= 0:
        print('oh dear')
    try:
        # Call the function that may raise a RuntimeWarning
        return_val = np.log(float(part1 * np.exp(part2)))
    except RuntimeWarning as rw:
        # Handle the warning
        print(f"Caught a RuntimeWarning: {rw}")
    return return_val


@jit(nopython=True)
def normal_log_pdf(val, mean, variance):
    if np.any(variance) <= 0:
        raise ValueError("Tried Pass through a variance that is less than or equal to 0 for gene {} at iteration {} ")
    return -0.5 * np.log(2 * np.pi) - np.log(np.sqrt(variance)) - (0.5 / variance) * (val - mean) ** 2

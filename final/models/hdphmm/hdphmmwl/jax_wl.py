import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as jax_mvn

@jax.jit
def backward_robust_jax(A, means, covariances, observations):
    num_states = A.shape[0]
    num_observed = observations.shape[0]
    beta = jnp.zeros((num_observed, num_states))

    for t in range(num_observed - 2, -1, -1):
        for i in range(num_states):
            logbeta = jnp.inf  # Initialize to infinity

            for j in range(num_states):
                try:
                    obs_prob = jax_mvn.logpdf(observations[t + 1], means[j], covariances[j])
                    logbeta = elnsum(logbeta, elnproduct(eln(A[j, i]), elnproduct(obs_prob, beta[t + 1, j])))
                except Exception as e:
                    print(f"An error occurred while logbeta: {e}")

            beta.at[t, i].set(logbeta)

    return beta

@jax.jit
def elnsum(x, y):
    return jax.scipy.special.logsumexp(jnp.array([x, y]))

@jax.jit
def elnproduct(x, y):
    return x + y

@jax.jit
def eln(x):
    """
    Compute ln(x) in a numerically stable way.

    :param x: x
    :return: ln(x)
    """
    return jnp.where(x > 0, jnp.log(x), jnp.nan)
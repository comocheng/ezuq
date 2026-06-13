# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import scipy.stats
import SALib.sample.morris

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# +
# Variances for 10 parameters

np.random.seed(400)
A = np.random.normal(size=(10, 10))
S = A @ A.T  # always positive semi-definite

# Normalize to correlation matrix
d = np.sqrt(np.diag(S))
R = S / np.outer(d, d)  # diagonal is exactly 1, off-diagonals in (-1, 1)

# Apply variances
std = np.array([1.0, 2.0, 1.5, 0.8, 2.2, 1.3, 0.9, 1.7, 2.0, 1.1])
D = np.diag(std)
Sigma = D @ R @ D

plt.matshow(Sigma)

k = Sigma.shape[0]



# +
# Take some morris samples

problem = {
    'num_vars': k,
    'names': [str(i) for i in range(k)],
    'bounds': [[1e-15, 1 - 1e-15]] * k,  # unit uniforms, we'll handle the actual translation to valid perturbations later on
}

# Generate Morris samples (takes a minute)
N = 10000

X = SALib.sample.morris.sample(problem, N=N, num_levels=4, seed=400)


X = np.random.uniform(size=(N, k))


# +
L = np.linalg.cholesky(Sigma)
assert np.isclose(L @ L.T, Sigma).all()
L_inv = np.linalg.inv(L)
z = scipy.stats.norm.ppf(X)

perturbation = (L @ z.T).T

# -

nataf_cov = np.cov(perturbation.T)

nataf_cov.shape

mc_samples = np.random.multivariate_normal(mean=np.zeros(k), cov=Sigma, size=perturbation.shape[0])
mc_cov = np.cov(mc_samples.T)

assert np.isclose(mc_cov, Sigma, rtol=0.1, atol=0.1).all()
assert np.isclose(nataf_cov, Sigma, rtol=0.1, atol=0.1).all()

# +
confidence = 0.9999
alpha = (1 - confidence) / 2

print(alpha)

# X_clipped = np.clip(X, alpha, 1 - alpha)
# -

eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

eigenvalues



"""Development of Sobol sampling using Kucherenko 2012, trying to speed things up and make this efficient"""
import numpy as np


import scipy.stats


def f(X):
    if X.ndim == 1:
        return np.sum(X)
    return np.sum(X, axis=1)

rho = 0.8
rho = -0.5
sigma = 2.0

mean = np.array([0.0, 0.0, 0])
cov = np.array([
    [1., 0., 0.],
    [0., 1., rho * sigma],
    [0., rho * sigma, sigma * sigma],
])
k = cov.shape[0]


L = np.linalg.cholesky(cov)
assert np.isclose(L @ L.T, cov).all()
# assert np.isclose(np.cov((x_tilde @ L.T).T), cov, rtol=0.1, atol=0.01).all()
# assert not np.isclose(np.cov((x_tilde @ L).T), cov).all()


N = 1024 * 1024
all_N_indices = np.arange(N)


u = np.random.uniform(size=(N, k))
u_prime = np.random.uniform(size=(N, k))

# 2. Generate unit normals
x_tilde = scipy.stats.norm.ppf(u)
x_tilde_prime = scipy.stats.norm.ppf(u_prime)

x = x_tilde @ L.T + mean  # L does not equal its transpose, so you have to be really careful here. check that you're recreating the cov matrix
x_prime = x_tilde_prime @ L.T + mean


f_y_z = f(x)
D = np.var(f_y_z)
f_y_prime_z_prime = f(x_prime)

results = np.zeros((k, 2))
for q in range(k):
    y_indices = [q]  # compute 1st order index on first parameter
    z_indices = list(set(np.arange(k)) - set(y_indices))

    # partition the mean and covariance matrices
    mu_y = mean[y_indices]
    mu_z = mean[z_indices]
    Sigma_y = cov[np.ix_(y_indices, y_indices)]
    Sigma_z = cov[np.ix_(z_indices, z_indices)]
    Sigma_yz = cov[np.ix_(y_indices, z_indices)]
    Sigma_zy = cov[np.ix_(z_indices, y_indices)]

    Sigma_y_inv = np.linalg.inv(Sigma_y)
    Sigma_z_inv = np.linalg.inv(Sigma_z)

    Sigma_zc = Sigma_z - Sigma_zy @ Sigma_y_inv @ Sigma_yz
    Sigma_yc = Sigma_y - Sigma_yz @ Sigma_z_inv @ Sigma_zy

    A_zc = np.linalg.cholesky(Sigma_zc)
    A_yc = np.linalg.cholesky(Sigma_yc)

    v = u[np.ix_(all_N_indices, y_indices)]
    w = u[np.ix_(all_N_indices, z_indices)]
    v_prime = u_prime[np.ix_(all_N_indices, y_indices)]
    w_prime = u_prime[np.ix_(all_N_indices, z_indices)]

    # 3. Generate unconditional normals

    # split into subsets
    y = x[np.ix_(all_N_indices, y_indices)]
    z = x[np.ix_(all_N_indices, z_indices)]

    y_prime = x_prime[np.ix_(all_N_indices, y_indices)]
    z_prime = x_prime[np.ix_(all_N_indices, z_indices)]

    # 4. Generate conditional normals
    mu_zc = np.tile(mu_z.T, (N, 1)) + (Sigma_zy @ Sigma_y_inv @ (y - np.tile(mu_y.T, (N, 1))).T).T
    mu_yc = np.tile(mu_y.T, (N, 1)) + (Sigma_yz @ Sigma_z_inv @ (z - np.tile(mu_z.T, (N, 1))).T).T

    # partition the standard normals
    y_tilde = scipy.stats.norm.ppf(v)
    z_tilde = scipy.stats.norm.ppf(w)
    y_tilde_prime = scipy.stats.norm.ppf(v_prime)
    z_tilde_prime = scipy.stats.norm.ppf(w_prime)

    y_bar = y_tilde @ A_yc.T + mu_yc
    y_bar_prime = y_tilde_prime @ A_yc.T + mu_yc
    z_bar = z_tilde @ A_zc.T + mu_zc
    z_bar_prime = z_tilde_prime @ A_zc.T + mu_zc

    y_z_bar_prime = np.zeros((N, k))
    y_z_bar_prime[:, y_indices] = y
    y_z_bar_prime[:, z_indices] = z_bar_prime

    y_bar_prime_z = np.zeros((N, k))
    y_bar_prime_z[:, y_indices] = y_bar_prime
    y_bar_prime_z[:, z_indices] = z

    # Evaluate functions
    f_y_z_bar_prime = f(y_z_bar_prime)
    f_y_bar_prime_z = f(y_bar_prime_z)

    Sy_contributions = np.multiply(f_y_z, f_y_z_bar_prime - f_y_prime_z_prime)
    ST_contributions = np.float_power(f_y_z - f_y_bar_prime_z, 2.0)

    Sy = np.mean(Sy_contributions, axis=0) / D
    ST = np.mean(ST_contributions, axis=0) / (2 * D)

    results[q, 0] = Sy
    results[q, 1] = ST

print(results)

ref_n0p5 = np.array([
    [0.2499976110325509, 0.2502901003688888],
    [2.8568176248771638e-19, 0.18781557632708504],
    [0.5642942108393815, 0.7507467427333141]
])
if rho == -0.5 and sigma == 2.0:
    assert np.isclose(results, ref_n0p5, rtol=1e-2, atol=1e-3).all()
else:
    raise ValueError()

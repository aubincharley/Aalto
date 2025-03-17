"""Given an observation y, we want to find the AR order of the process that generated the observation.
"""

import numpy as np
import matplotlib.pyplot as plt
from EKF import *
from RTS import *

n_points = 400
t = np.linspace(0, 1, n_points)
random_seed = 726
np.random.seed(random_seed)
# print(random_seed)


sigma_e2 = 0.0001
sigma_w2 = 0.00001
AR_order = 8
q = 10


# Generate the initial AR weights
# a = np.array([0.55, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
# a = np.array([0.5, 0.3, 0.2, 0.1, 0.1])
a = np.array([0.55, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.008])
sigma_a2 = 0.0001

# Generate data
x = np.zeros(n_points)
s = np.random.normal(0, np.sqrt(sigma_w2), q)
x[:q] = s
state = np.concatenate(([s, a]))

for i in range(AR_order, n_points):

    ### ICI UNE ERREUR, q n'est pas sense influer sur la preparation de la data###
    a = state[q:]
    A = companion(a, q)
    # np.dot(A, state) + u(q,AR_order,sigma_a2,sigma_w2)
    state = forward(state, q) + u(q, AR_order, sigma_a2, sigma_w2)
    x[i] = state[0]

y = x + np.random.normal(0, np.sqrt(sigma_e2), n_points)

# --------------------This part differs from the ex2-------------------------#
sigma_a2_est = sigma_a2
sigma_e2_est = sigma_e2
sigma_w2_est = sigma_w2*10
AR_order_est = AR_order
# ---------------------------------------------------------------------------#
means, covs, all_state_means, all_states_covs = Extended_Kalman_Filter(
    sigma_w2_est, sigma_a2_est, sigma_e2_est, q, AR_order_est, n_points, y)

smoothed_means, smoothed_covs, filtered_means, filtered_covs = RTS_smoother(
    all_state_means, all_states_covs, q, AR_order, sigma_w2, sigma_a2, n_points)

mse_kalman = np.mean((means[AR_order:] - x[AR_order:])**2)
mse_rts = np.mean((filtered_means[AR_order:] - x[AR_order:])**2)
mse_noisy = np.mean((y[AR_order:] - x[AR_order:])**2)


fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(t, y, label="Noisy measurements",
           color="red", alpha=0.5, linewidth=0.7)
ax[0].plot(t, x, label="True signal", color="black", linewidth=0.7)
ax[0].set_title(
    "AR(8) process with noise on the measurements and on the AR weights. MSE : {:.2e}".format(mse_noisy))
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Signal")
ax[0].legend()


ax[1].plot(t[AR_order:], means[AR_order:],
           label="Extended Kalman filter", color="green", linewidth=0.7)
ax[1].fill_between(t[AR_order:], means[AR_order:] - 1.96*np.sqrt(covs[AR_order:]),
                   means[AR_order:] + 1.96*np.sqrt(covs[AR_order:]), color="green", alpha=0.2)
ax[1].plot(t[AR_order:], x[AR_order:], label="True signal",
           color="black", linewidth=0.7)
ax[1].plot(t[AR_order:], y[AR_order:],
           label="Noisy measurements", color="red", alpha=0.5, linewidth=0.7)
ax[1].legend()
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Signal")
ax[1].set_title("Extended Kalman filter. MSE: {:.2e}".format(mse_kalman))

ax[2].plot(t[AR_order:], filtered_means[AR_order:],
           label="RTS smoother", color="blue", linewidth=0.7)
ax[2].fill_between(t[AR_order:], filtered_means[AR_order:] - 1.96*np.sqrt(filtered_covs[AR_order:]),
                   filtered_means[AR_order:] + 1.96*np.sqrt(filtered_covs[AR_order:]), color="blue", alpha=0.2)
ax[2].plot(t[AR_order:], x[AR_order:], label="True signal",
           color="black", linewidth=0.7)
ax[2].plot(t[AR_order:], y[AR_order:],
           label="Noisy measurements", color="red", alpha=0.5, linewidth=0.7)
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Signal")
ax[2].set_title("RTS smoother. MSE: {:.2e}".format(mse_rts))
ax[2].legend()
fig.tight_layout()
plt.show()

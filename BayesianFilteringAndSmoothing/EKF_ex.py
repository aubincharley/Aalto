import numpy as np
import matplotlib.pyplot as plt
from EKF import *

n_points = 200
t = np.linspace(0, 1, n_points)
np.random.seed(7)

sigma_e2 = 0.0005
sigma_w2 = 0.001
AR_order = 10
q = 13


# Generate the initial AR weights
a = np.array([0.55, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
# a = np.array([0.5, 0.3, 0.2, 0.1, 0.1])
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
plt.plot(t, y, label="Noisy measurements", color="red", alpha=0.5)
plt.plot(t, x, label="True signal", color="blue", linewidth=0.3)
plt.title(r"AR(10) process with noise on the measurements and on the AR weights")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.show()

means, covs, all_state_means, all_states_covs = Extended_Kalman_Filter(
    sigma_w2, sigma_a2, sigma_e2, q, AR_order, n_points, y)

plt.plot(t[AR_order:], means[AR_order:],
         label="Extended Kalman filter", color="green")
plt.plot(t[AR_order:], x[AR_order:], label="True signal", color="blue")
plt.scatter(t[AR_order:], y[AR_order:],
            label="Noisy measurements", color="red", alpha=0.5)
plt.legend()
plt.show()
plt.plot(t[AR_order:], means[AR_order:],
         label="Extended Kalman filter", color="green")
plt.plot(t[AR_order:], y[AR_order:],
         label="Noisy measurements", color="red", alpha=0.5)
plt.legend()
plt.show()
plt.plot(t[AR_order:], means[AR_order:],
         label="Extended Kalman filter", color="green")
plt.plot(t[AR_order:], x[AR_order:],
         label="True signal", color="blue", alpha=0.6)
plt.legend()
plt.show()

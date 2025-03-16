import numpy as np
import matplotlib.pyplot as plt
from KF import Kalman_filter

# Generate some data
n_points = 200
t = np.linspace(0, 1, n_points)
np.random.seed(0)

sigma_e2 = 0.01
sigma_w2 = 0.001
AR_order = 10
# Generate the AR matrix (companion)
A = np.zeros((AR_order, AR_order))
for i in range(1, AR_order):
    A[i, i-1] = 1
A[0, :] = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])

H = np.zeros((1, AR_order))  # Matrice d'observation (1 x AR_order)
H[0, 0] = 1

# Generate data
x = np.zeros(n_points)
x[:AR_order] = np.random.normal(0, np.sqrt(sigma_w2), AR_order)
for i in range(AR_order, n_points):
    s = np.dot(A, x[i-AR_order:i]) + np.random.normal(0, np.sqrt(sigma_w2))
    x[i] = s[0]

y = x + np.random.normal(0, np.sqrt(sigma_e2), n_points)
means, covs = Kalman_filter(AR_order, sigma_w2, sigma_e2, y, A, H, n_points)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(t, x, label="True signal", color="blue")
ax[0].plot(t, y, label="Noisy measurements", color="red", alpha=0.5)
ax[0].legend()
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Signal")
ax[0].set_title("True signal and noisy measurements")
ax[1].plot(t, means, label="Kalman filter", color="green")
ax[1].fill_between(t, means - np.sqrt(covs), means +
                   np.sqrt(covs), color="green", alpha=0.5)
ax[1].plot(t, x, label="True signal", color="blue")
ax[1].plot(t, y, label="Noisy measurements", color="red", alpha=0.5)
ax[1].legend()
ax[1].set_title("Kalman filter, true signal and noisy measurements")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Signal")
fig.suptitle("Kalman filter example")
plt.show()

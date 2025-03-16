import numpy as np
import matplotlib.pyplot as plt

n_points = 2000
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
sigma_a2 = 0.00001

# Generate data
x = np.zeros(n_points)
x[:AR_order] = np.random.normal(0, np.sqrt(sigma_w2), AR_order)
for i in range(AR_order, n_points):
    A[0, :] += np.random.normal(0, np.sqrt(sigma_a2), AR_order)
    s = np.dot(A, x[i-AR_order:i]) + np.random.normal(0, np.sqrt(sigma_w2))
    x[i] = s[0]

y = x + np.random.normal(0, np.sqrt(sigma_e2), n_points)
plt.plot(t, y, label="Noisy measurements", color="red", alpha=0.5)
plt.plot(t, x, label="True signal", color="blue", linewidth=0.3)
plt.title(r"AR(10) process with noise on the measurements and on the AR weights")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.show()

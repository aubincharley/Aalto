import numpy as np
from EKF import *


def prediction_RTS(kalman_mean, kalman_cov, Q, q, AR_order, smoothed_mean, smoothed_cov):
    # print(kalman_mean)
    mat = companion(kalman_mean[q:], q)
    pred_mean = mat@kalman_mean
    Jac = jacobian(kalman_mean.reshape(-1, 1), AR_order, q)
    # print(Jac)
    pred_cov = Jac @ kalman_cov @ Jac.T + Q
    G = kalman_cov@Jac.T@np.linalg.inv(pred_cov)
    mean = kalman_mean + G@(smoothed_mean-pred_mean)
    cov = kalman_cov + G@(smoothed_cov-kalman_cov)@G.T
    # print(mean.shape,cov.shape)
    return mean, cov


def RTS_smoother(kalman_means, kalman_covs, q, AR_order, sigma_w2, sigma_a2, n_points, plot_progress=False):
    Q = np.zeros((AR_order+q, AR_order+q))
    Q[0, 0] = sigma_w2

    for i in range(q, q + AR_order):
        Q[i][i] = sigma_a2
    smoothed_means = np.zeros_like(kalman_means)
    smoothed_covs = np.zeros_like(kalman_covs)
    smoothed_means[-1] = kalman_means[-1]
    smoothed_covs[-1] = kalman_covs[-1]
    for i in range(len(kalman_means)-2, -1, -1):
        if plot_progress:
            if i in [k*n_points//10 for k in range(1, 10)]:
                print(f"----Smoothing: {100-100*i//n_points}%----")
            elif i == 0:
                print(f"----Smoothing: 100%----")
        smoothed_means[i], smoothed_covs[i] = prediction_RTS(
            kalman_means[i], kalman_covs[i], Q, q, AR_order, smoothed_means[i+1], smoothed_covs[i+1])

    filtered_means = np.zeros(n_points)
    filtered_means[:q] = smoothed_means[:q, 0]
    for i in range(q, n_points):
        filtered_means[i] = smoothed_means[i-q+1, 0]
    return smoothed_means, smoothed_covs, filtered_means

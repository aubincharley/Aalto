import numpy as np


def Kalman_filter(AR_order, sigma_w2, sigma_e2, y, A, H, n_points):

    # Matrice de covariance du bruit de processus
    Q_k = np.zeros((AR_order, AR_order))
    Q_k[0, 0] = sigma_w2  # Seulement la première composante a du bruit

    # Initialisation
    means = np.zeros(n_points)
    covs = np.zeros(n_points)

    # Initialisation des AR_order premiers points
    means[:AR_order] = y[:AR_order]
    # Incertitude initiale = variance du bruit de mesure
    covs[:AR_order] = sigma_e2

    # État initial (vecteur colonne)
    mean = y[:AR_order].reshape(AR_order, 1)
    cov = np.eye(AR_order) * sigma_e2  # Covariance initiale

    for i in range(AR_order, n_points):
        # Prédiction
        mean_predicted = A @ mean
        cov_predicted = A @ cov @ A.T + Q_k

        # Mise à jour
        v = y[i] - H @ mean_predicted  # Innovation (scalaire)
        # Covariance de l'innovation (scalaire)
        S = H @ cov_predicted @ H.T + sigma_e2
        K = cov_predicted @ H.T / S  # Gain de Kalman (vecteur colonne)

        # Correction
        mean = mean_predicted + K * v
        cov = (np.eye(AR_order) - K @ H) @ cov_predicted  # Formule correcte

        # Stockage des résultats
        means[i] = mean[0, 0]
        covs[i] = cov[0, 0]

    return means, covs

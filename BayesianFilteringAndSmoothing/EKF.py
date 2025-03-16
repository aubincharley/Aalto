import numpy as np


def companion(s, q):
    """Compute the companion matrix of size q,q, formed by the s vector of size p

    Args:
        s (array): sequence used to form the companion matrix
        q (int): size of the companion matrix

    Returns:
        array: companion matrix associated with the s vector
    """
    p = len(s)
    A = np.zeros((q+p, q+p))
    A[0, :p] = s
    for i in range(1, q+p):
        A[i, i-1] = 1
    return A


def test_companion_truncated():
    x_test = np.array([1, 2, 3, 4, 5])
    q_test = 5
    print(f"Companion test for x = {x_test} and q = {q_test} give companion(x,q): ", companion(
        x_test, q_test))


def jacobian(x, p, q):
    """Compute the Jacobian matrix of size n,n, formed by the x vector of size n. 
    The jacobian is the derivative of the function x -> companion( x[q:, 0] )@x

    Args:
        x (array): current state
        p (int): AR order
        q (int): size used to predict the state. len(x) = p+q

    Returns:
        array: jacobian matrix evaluated at x
    """
    s = x[:p, 0]
    Jac = np.zeros((len(x), len(x)))
    A = np.zeros((q, q))
    A[0, :q] = x[p:, 0]
    for i in range(q):
        Jac[i][:q] = A[i]
    for j in range(1, len(x)):
        Jac[j][j-1] = 1
    Jac[0][q:] = s
    return Jac


def test_jacobian():
    x_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10, 1)
    p_test = 5
    q_test = 5
    print(
        "\nJacobian test for x_test: {x_test}, p_test: {p_test}, q_test: {q_test} give jacobian(x_test, p_test, q_test): ", jacobian(x_test, p_test, q_test))


def forward(x, q):
    """Compute the forward prediction of the state x

    Args:
        x (array):state of length q + AR_order
        q (int): size of the state used to predict the next state

    Returns:
        array: predicted state
    """
    a = x[q:]
    A = companion(a, q)
    mat = np.zeros((len(x), len(x)))
    for i in range(q):
        mat[i] = A[i]
    for j in range(q, len(x)):
        mat[j][j] = 1
    return mat@x


def test_forward():
    x_test = np.array([1, 2, 3, 4, 5])
    q_test = 3
    print(
        f"\nforward test for x_test: {x_test} , q_test: {q_test} give forward(x_test,q_test): {forward(x_test, q_test)}")


def u(q, p, sigma_a2, sigma_w2):
    """Compute the noise vector u

    Args:
        q (int): size of the state used to predict the next state
        p (int): AR order
        sigma_a2 (float): variance of the AR coefficients
        sigma_w2 (float): variance of the white noise

    Returns:
        array: noise vector
    """
    n1 = np.random.normal(0, np.sqrt(sigma_w2))
    n2 = np.random.normal(0, np.sqrt(sigma_a2), p)
    noise = np.zeros(p+q)
    noise[q:] = n2
    noise[0] = n1
    return noise


def test_u():
    print("\nu test: ", u(5, 2, 0.1, 0.2))


def prediction_kalman(mean, cov, Q, H, jac, y, sigma_e2, q, AR_order):
    """Compute the prediction step of the Kalman filter

    Args:
        mean (array): mean of the state
        cov (array): covariance of the state
        Q (array): covariance of the noise u 
        H (array): observation matrix
        jac (array): jacobian of the observation function
        y (array): observation
        sigma_e2 (array): variance of the observation noise
        q (int):  size of the state used to predict the next state
        AR_order (int): AR order

    Returns:
       mean (array): updated mean
       cov (array): updated covariance
    """
    comp = companion(mean[q:, 0], q)
    mean_predicted = comp@mean
    cov_predicted = jac @ cov @ jac.T + Q

    # Mise à jour
    v = y - H @ mean_predicted  # Innovation (scalaire)
    # Covariance de l'innovation (scalaire)
    S = H @ cov_predicted @ H.T + sigma_e2
    # Gain de Kalman (vecteur colonne)
    K = cov_predicted @ H.T @ np.linalg.inv(S)

    # Correction
    mean = mean_predicted + K @ v
    cov = (np.eye(AR_order+q) - K @ H) @ cov_predicted  # Formule correcte

    return mean, cov


def Extended_Kalman_Filter(sigma_w2, sigma_a2, sigma_e2, q, AR_order, n_points, y, plot_progress=False):
    """Compute the EKF for a given AR process

    Args:
        sigma_w2 (int): variance of the white process noise
        sigma_a2 (int): variance of the AR coefficients
        sigma_e2 (int): variance of the observation noise
        q (int):  size of the state used to predict the next state
        AR_order (int): AR order
        n_points (int): number of points
        y (array): observations

    Returns:
        means (array): predicted value at each time step
        covs (array): predicted covariance at each time step
        all_state_means (array): mean of the state at each time step
        all_state_covs (array): covariance of the state at each time step
    """

    mean = np.zeros(q+AR_order)
    cov = np.eye(q+AR_order)

    all_state_means = np.zeros((n_points-q+1, q+AR_order))
    all_state_covs = np.zeros((n_points-q+1, q+AR_order, q+AR_order))

    all_state_means[0] = mean
    all_state_covs[0] = cov
    # mean = np.ones_like(state)

    means = np.zeros(n_points)
    means[:q] = y[:q]
    covs = np.zeros(n_points)
    covs[:q] = sigma_w2

    H = np.zeros((1, q+AR_order))
    H[0, 0] = 1

    Q = np.zeros((AR_order+q, AR_order+q))
    Q[0, 0] = sigma_w2

    for i in range(q, q + AR_order):
        Q[i][i] = sigma_a2
    mean = mean.reshape((AR_order+q, 1))

    for i in range(q, n_points):
        if plot_progress:
            if i in [k*n_points//10 for k in range(1, 10)]:
                print(f"----Filtering:{100*i//n_points}%---- ")
            elif i == n_points-1:
                print(f"----Filtering:100%---- ")
        Jac = jacobian(mean, AR_order, q)
        mean, cov = prediction_kalman(
            mean, cov, Q, H, Jac, y[i], sigma_e2, q, AR_order)

        means[i] = mean[0, 0]
        covs[i] = cov[0][0]
        all_state_means[i-q+1] = mean.reshape(-1)
        all_state_covs[i-q+1] = cov

    return means, covs, all_state_means, all_state_covs


if __name__ == "__main__":
    test_companion_truncated()
    test_forward()
    test_jacobian()
    test_u()

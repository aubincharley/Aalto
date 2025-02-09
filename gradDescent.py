import numpy as np
import matplotlib.pyplot as plt
import time


random_seed = 1

# let define a random matrix
np.random.seed(random_seed)


def generate_random_non_negative_matrix(dimension):
    A = np.random.rand(dimension, dimension)
    eig_val, eig_vec = np.linalg.eig(np.transpose(A)@A)
    if IS_INVERTIBLE:
        # max(eig_val)*np.eye(dimension)
        return np.transpose(A)@A + np.eye(dimension)

    return np.transpose(A)@A


def generate_random_vector(dimension):
    return np.random.rand(dimension, 1)


def quadratic_function(A, b, x):
    return 1/2*np.transpose(x)@A@x - np.transpose(b)@x

# the goal is to minimize this quadratic function

# --------------------------------------Question 1-------------------------------------#


def gradient_descent(A, b, x0, learning_rate, iterations, epsilon):
    x = x0.copy()
    start_time = time.time()
    log_grad = [np.log(np.linalg.norm(A@x - b))]
    values_f = [np.linalg.norm(quadratic_function(A, b, x))]
    i = 0
    while i < iterations and np.linalg.norm(A@x - b) > epsilon:
        x = x - learning_rate*(A@x - b)
        log_grad.append(np.log(np.linalg.norm(A@x - b)))
        values_f.append(np.linalg.norm(quadratic_function(A, b, x)))
        i += 1
    if i == iterations:
        print("Did not converge in {} iterations" .format(i))
    else:

        print("Converged in {} iterations" .format(i))

    print("Time taken for gradient descent: {} seconds. \n" .format(
        time.time() - start_time))
    return x, log_grad, values_f, i


# we plot the result

def plot_Q1():
    A = generate_random_non_negative_matrix(dimension)
    b = generate_random_vector(dimension)

    x0 = generate_random_vector(dimension)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    L = max(eigenvalues)
    mu = min(eigenvalues)
    learning_rate = 2/(mu+L)
    # learning_rate = 1/L
    iterations = 5000
    epsilon = 10**(-5)

    x, iterations, values_f, n_iter = gradient_descent(
        A, b, x0, learning_rate, iterations, epsilon)

    x = [i for i in range(len(iterations))]
    linear = np.polyfit(x, iterations, 1)

    if PLOT_GRAPH:
        plt.subplot(1, 2, 1)
        plt.plot(x, np.polyval(linear, x), marker="_", color='grey')
        plt.plot(iterations)
        plt.xlabel('iterations')
        plt.ylabel('log(||Ax-b||)')
        plt.subplot(1, 2, 2)
        plt.plot(values_f)
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.suptitle(
            'Gradient Descent, stopped at {} iterations'.format(n_iter))
        plt.show()


def plot_Q1_bis():
    for dimension in [10, 50, 100, 300, 500]:
        A = generate_random_non_negative_matrix(dimension)
        b = generate_random_vector(dimension)

        x0 = generate_random_vector(dimension)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        L = max(eigenvalues)
        mu = min(eigenvalues)
        learning_rate = 2/(mu+L)
        # learning_rate = 1/L
        iterations = 5000
        epsilon = 10**(-5)

        x, iterations, values_f, n_iter = gradient_descent(
            A, b, x0, learning_rate, iterations, epsilon)

        if PLOT_GRAPH:
            plt.plot(iterations, label="{}".format(dimension))
            plt.xlabel('iterations')
            plt.ylabel('log(||Ax-b||)')
    plt.legend()
    plt.show()


# --------------------------------------Question 2-------------------------------------#


def conjugate_gradient(A, b, x0, iterations, epsilon):
    # print(iterations)
    start_time = time.time()
    x = x0.copy()
    r = A@x - b
    p = -r
    k = 0
    x_list = [x]
    log_grad = [np.log(np.linalg.norm(A@x0 - b))]
    values_f = [np.linalg.norm(quadratic_function(A, b, x0))]
    while k < iterations and np.linalg.norm(A@x - b) > epsilon:
        alpha = -(r.T @ p) / (p.T @ A @ p)
        x = x + alpha * p
        r = A @ x - b
        beta = (r.T@A@p) / (p.T @ A @ p)
        p = -r + beta * p
        k += 1
        x_list.append(x)
        values_f.append(np.linalg.norm(quadratic_function(A, b, x)))
        log_grad.append(np.log(np.linalg.norm(A@x - b)))
    if k == iterations:
        print("Did not converge in {} iterations".format(iterations))
    else:
        print("Converged in {} iterations".format(k))
    print("Time taken for conjugate gradient descent: {} seconds. \n".format(
        time.time() - start_time))
    return x, log_grad, values_f, k


def plot_Q2():
    A = generate_random_non_negative_matrix(dimension)
    b = generate_random_vector(dimension)

    x0 = generate_random_vector(dimension)
    nb_iterations = 1000
    epsilon = 10**(-5)

    x, iterations, values_f, n_iter = conjugate_gradient(
        A, b, x0, nb_iterations, epsilon)
    print(iterations[-1])
    if PLOT_GRAPH:
        plt.subplot(1, 2, 1)
        plt.plot(iterations)
        plt.xlabel('iterations')
        plt.ylabel('log(||Ax-b||)')
        plt.subplot(1, 2, 2)
        plt.plot(values_f)
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.suptitle(
            'Conjugate Gradient Descent, stopped at {} iterations'.format(n_iter))
        plt.show()


def plot_Q2_bis():
    for dimension in [3000]:
        A = generate_random_non_negative_matrix(dimension)
        b = generate_random_vector(dimension)

        x0 = generate_random_vector(dimension)
        nb_iterations = 1000
        epsilon = 10**(-5)

        x, iterations, values_f, n_iter = conjugate_gradient(
            A, b, x0, nb_iterations, epsilon)
        print(iterations[-1])
        if PLOT_GRAPH:

            plt.plot(iterations, label='{}'.format(dimension))
            plt.xlabel('iterations')
            plt.ylabel('log(||Ax-b||)')
    plt.legend()
    plt.show()

# --------------------------------------Question 3-------------------------------------#


def FISTA_gradient_descent(A, b, x0, L, iterations, epsilon):
    start_time = time.time()
    y = x0.copy()
    t = 1
    t_list = [t]
    x_list = [x0]
    values_f = [np.linalg.norm(quadratic_function(A, b, x0))]
    log_grad = [np.log(np.linalg.norm(A@x0 - b))]
    i = 0
    while i < iterations and np.linalg.norm(A@y - b) > epsilon:
        x = y - (1/L)*(A @ y - b)
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2

        y = x + ((t_list[-1]-1)/t) * (x - x_list[-1])
        x_list.append(x)
        values_f.append(np.linalg.norm(quadratic_function(A, b, x)))
        log_grad.append(np.log(np.linalg.norm(A@x - b)))
        i += 1
        t_list.append(t)
    if i == iterations:
        print("Did not converge in {} iterations.".format(i))
    else:
        print("Converged in {} iterations".format(i))

    print("Time taken for FISTA gradient descent: {} seconds. \n".format(
        time.time() - start_time))
    return x_list[-1], log_grad, values_f, i


def plot_Q3():
    A = generate_random_non_negative_matrix(dimension)
    b = generate_random_vector(dimension)

    x0 = generate_random_vector(dimension)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    L = max(eigenvalues)
    # L = 5000
    iterations = 10000
    epsilon = 10**(-5)

    x, iterations, values_f, n_iter = FISTA_gradient_descent(
        A, b, x0, L, iterations, epsilon)
    if PLOT_GRAPH:
        plt.subplot(1, 2, 1)
        plt.plot(iterations)
        plt.xlabel('iterations')
        plt.ylabel('log(||Ax-b||)')
        plt.subplot(1, 2, 2)
        plt.plot(values_f)
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.suptitle(
            'FISTA Gradient Descent,stopped at {} iterations'.format(n_iter))
        plt.show()


def plot_Q3_bis():
    for dimension in [10, 50, 100, 200, 500]:
        A = generate_random_non_negative_matrix(dimension)
        b = generate_random_vector(dimension)

        x0 = generate_random_vector(dimension)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        L = max(eigenvalues)
        # L = 5000
        iterations = 5000
        epsilon = 10**(-5)

        x, iterations, values_f, n_iter = FISTA_gradient_descent(
            A, b, x0, L, iterations, epsilon)
        if PLOT_GRAPH:

            plt.plot(iterations, label='{}'.format(dimension))
            plt.xlabel('iterations')
            plt.ylabel('log(||Ax-b||)')
    plt.legend()
    plt.show()


# --------------------------------------Question 4-------------------------------------#

def coordinate_descent(A, b, x0, learning_rate, iterations, epsilon):
    dimension = len(x0)
    start_time = time.time()
    x = x0.copy()
    x_list = [x]

    i = 1
    while i <= iterations:  # and np.linalg.norm(A@x - b) > epsilon:
        random = np.random.randint(0, dimension)
        new_x = x_list[-1].copy()
        new_x[random] = new_x[random] - learning_rate * \
            (A[random, :]@new_x - b[random])
        i += 1
        x_list.append(new_x)
    if i == iterations:
        print("Coordinate descent did not converge after {} iterations.".format(iterations))
    else:
        print("Coordinate descent converged after {} iterations.".format(i))
    print("Elapsed time for coordinate descent: {} seconds. \n".format(
        time.time() - start_time))

    values_f = [np.linalg.norm(quadratic_function(A, b, y)) for y in x_list]
    log_grad = [np.log(np.linalg.norm(A@y - b)) for y in x_list]

    iteration_convergence = 0
    while values_f[iteration_convergence] > epsilon and iteration_convergence < len(values_f) - 1:
        iteration_convergence += 1
    # print("Converged in {} iterations".format(iteration_convergence))

    return x_list, log_grad, values_f, iteration_convergence


def plot_Q4():
    A = generate_random_non_negative_matrix(dimension)
    b = generate_random_vector(dimension)

    x0 = generate_random_vector(dimension)
    learning_rate = 0.0001
    iterations = 5000
    epsilon = 10**(-5)

    x, iterations, values_f, n_iter = coordinate_descent(
        A, b, x0, learning_rate, iterations, epsilon)
    print(values_f[-1])
    if PLOT_GRAPH:
        plt.subplot(1, 2, 1)
        plt.plot(iterations)
        plt.xlabel('iterations')
        plt.ylabel('log(||Ax-b||)')
        plt.subplot(1, 2, 2)
        plt.plot(values_f)
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.suptitle(
            'Coordinate Gradient Descent, stopped at {} iterations'.format(n_iter))
        plt.show()


def plot_Q4_bis():
    for dimension in [10, 50, 100, 200, 500]:
        A = generate_random_non_negative_matrix(dimension)
        b = generate_random_vector(dimension)

        x0 = generate_random_vector(dimension)
        learning_rate = 0.001
        iterations = 5000
        epsilon = 10**(-5)

        x, iterations, values_f, n_iter = coordinate_descent(
            A, b, x0, learning_rate, iterations, epsilon)
        print(values_f[-1])
        if PLOT_GRAPH:
            plt.plot(iterations, label='{}'.format(dimension))
            plt.xlabel('iterations')
            plt.ylabel('log(||Ax-b||)')

    plt.legend()
    plt.show()

# ------------------------------------------Question 5-----------------------------------------------#


def comparison():
    A = generate_random_non_negative_matrix(dimension)
    b = generate_random_vector(dimension)

    x0 = generate_random_vector(dimension)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    L = max(eigenvalues)
    mu = min(eigenvalues)
    print("mu : ", mu)
    L_fista = L
    learning_rate = 1/L
    iterations = 500
    epsilon = 10**(-5)
    iterations_coordinate = 1000

    x, iterations_gradient_descent, values_f_gradient_descent, n_iter_gradient_descent = gradient_descent(
        A, b, x0, learning_rate, iterations, epsilon)
    x, iterations_coordinate_descent, values_f_coordinate_descent, n_iter_coordinate_descent = coordinate_descent(
        A, b, x0, learning_rate, iterations, epsilon)
    x, iterations_conjugate_descent, values_f_conjugate_descent, n_iter_conjugate_descent = conjugate_gradient(
        A, b, x0, iterations, epsilon)
    x, iterations_fista, values_f_fista, n_iter_fista = FISTA_gradient_descent(
        A, b, x0, L_fista, iterations, epsilon)

    print(iterations_conjugate_descent[0], iterations_coordinate_descent[0],
          iterations_gradient_descent[0], iterations_fista[0])

    if PLOT_GRAPH:

        # PLOT LES FONCTIONS LINEAIRES

        plt.subplot(1, 2, 1)

        plt.plot(iterations_gradient_descent,
                 label='gradient descent', color='red')
        plt.plot(iterations_coordinate_descent,
                 label='coordinate descent', color='blue')
        plt.plot(iterations_fista, label='FISTA', color='black')
        plt.plot(iterations_conjugate_descent,
                 label='conjugate gradient', color='green')
        plt.xlabel('iterations')
        plt.ylabel('log(||Ax-b||)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(values_f_gradient_descent,
                 label='gradient descent', color='red')
        plt.plot(values_f_coordinate_descent,
                 label='coordinate descent', color='blue')
        plt.plot(values_f_fista, label='FISTA', color='black')
        plt.plot(values_f_conjugate_descent,
                 label='conjugate gradient', color='green')
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.legend()
        plt.suptitle("Comparison of all GD")
        plt.show()
        # plt.savefig("comparison{}.png".format(np.random.randint(0, 1000)))


def try_to_invert():
    A = np.random.rand(dimension, dimension)
    start_time = time.time()
    try:
        A_inv = np.linalg.inv(A)
        end_time = time.time()
        print("A is invertible")
        print("Time to compute A_inv: ", end_time - start_time)
    except np.linalg.LinAlgError:
        print("A is not invertible")


def time_comparison():
    A = generate_random_non_negative_matrix(dimension)
    b = generate_random_vector(dimension)

    x0 = generate_random_vector(dimension)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    L = max(eigenvalues)
    L_fista = L
    learning_rate = 1/L
    iterations = 5000
    epsilon = 10**(-5)
    iterations_coordinate = 5000

    x, iterations_gradient_descent, values_f_gradient_descent, n_iter_gradient_descent = gradient_descent(
        A, b, x0, learning_rate, iterations, epsilon)
    x, iterations_coordinate_descent, values_f_coordinate_descent, n_iter_coordinate_descent = coordinate_descent(
        A, b, x0, learning_rate, iterations_coordinate, epsilon)
    x, iterations_conjugate_descent, values_f_conjugate_descent, n_iter_conjugate_descent = conjugate_gradient(
        A, b, x0, iterations, epsilon)
    x, iterations_fista, values_f_fista, n_iter_fista = FISTA_gradient_descent(
        A, b, x0, L_fista, iterations, epsilon)


def fast_coordinate_descent():
    x0 = generate_random_vector(dimension)
    start_time = time.time()
    x = x0.copy()
    x_list = [x]
    iterations = 20000
    learning_rate = 10**(-5)
    A = np.random.rand(dimension, dimension)
    b = np.random.rand(dimension)
    i = 1
    while i <= iterations:  # and np.linalg.norm(A@x - b) > epsilon:
        random = np.random.randint(0, dimension)
        new_x = x_list[-1].copy()
        new_x[random] = new_x[random] - learning_rate * \
            (A[random, :]@new_x - b[random])
        i += 1
        x_list.append(new_x)
    print("Elapsed time for coordinate descent: {} seconds".format(
        time.time() - start_time))

    print(np.log(np.linalg.norm(quadratic_function(A, b, x_list[-1]))))


# for dimension = 10000 time is too long
if __name__ == "__main__":
    IS_INVERTIBLE = True
    PLOT_GRAPH = True
    dimension = 250
    plot_Q4_bis()
    # plot_Q2()
    # plot_Q3()
    # plot_Q4()
    # comparison()
    # try_to_invert()
    # time_comparison()
    # fast_coordinate_descent()

import sys
import argparse
import numpy as np
import show
from price_estimation import get_thetas_values

def dataset():
    try:
        lst = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
    except Exception as e:
        print(f"Warning: Failed to load file!\nError: {e}\nMake sure data.csv exists.\n")
        sys.exit(-1)
    
    m = len(lst)
    x = lst[:, 0].reshape(-1, 1)
    nx = (x - np.min(x)) / (np.max(x) - np.min(x))
    Y = lst[:, 1].reshape(-1, 1)
    t0, t1 = get_thetas_values()
    theta = np.array([[t0], [t1]])

    
    normX = np.hstack((nx, np.ones((m, 1))))
    
    return x, normX, Y, theta

def model(X, theta):
    F = X.dot(theta)
    return F

def cost_function(m, X, Y, theta):
    J = (1 / (2 * m)) * np.sum((model(X, theta) - Y) ** 2)
    return J

def gradient_descent(X, Y, theta, learning_rate, n_iterations):
    m = len(Y)
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * (1 / m) * X.T.dot(model(X, theta) - Y)
        cost_history[i] = cost_function(m, X, Y, theta)
    return theta, cost_history

def ft_linear_regression():
    learning_rate = 0.07
    n_iterations = 1000
    x, normX, Y, theta = dataset()
    final_theta, cost_history = gradient_descent(normX, Y, theta, learning_rate, n_iterations)
    prediction = model(normX, final_theta)
    return x, Y, prediction, cost_history, final_theta, n_iterations

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction", action="store_true", help="show the prediction curve")
    parser.add_argument("-ch", "--cost-history", action="store_true", help="show the cost history curve")
    parser.add_argument("-cd", "--coef-determination", action="store_true", help="show the coefficient determination")
    args = parser.parse_args()

    x, Y, prediction, cost_history, final_theta, n_iterations = ft_linear_regression()

    if args.prediction:
        show.prediction_curve(x, Y, prediction)
    elif args.cost_history:
        show.cost_history_curve(n_iterations, cost_history)
    elif args.coef_determination:
        show.coef_determination(Y, prediction)
    show.thetas_values(float(final_theta[1][0]), float(final_theta[0][0]))

if __name__ == '__main__':
    argument_parser()

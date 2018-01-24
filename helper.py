import numpy as np

def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigma_d(x):
    s = sigam(x)
    return s * (1 - s)


def softmax(x):
    dot_exp = np.exp(x)
    summed = np.sum(dot_exp, axis=1)
    summed = np.reshape(summed, (dot_exp.shape[0], 1))
    summed = np.repeat(summed, dot_exp.shape[1], axis=1)
    return (dot_exp / (1.0 * summed))


def get_one_hot(labels):
    potential_vals = list(set(labels))
    potential_vals.sort()
    return np.array([[int(l == p) for p in potential_vals] for l in labels])


def foo():
    pass

def norm_loss_function(w, x, y):
        y = get_one_hot(y)
        return (-1.0 / (x.shape[0] * w.shape[1])) * np.sum(y * softmax(x, w))

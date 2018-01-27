from __future__ import print_function
import numpy as np
import pdb

def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigma_d(x):
    s = sigma(x)
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

# given node outputs x and labels y, returns times x==y
def norm_loss_function(x, y):
    y = get_one_hot(y)
    return (-1.0 / (x.shape[0] * x.shape[1])) * np.sum(y * x)


def prefix_ones(some_array):
    return np.concatenate((np.ones((some_array.shape[0], 1)),
                           some_array), axis=1)


def evaluate(x, labels):
        # y = get_one_hot(labels)
        possible_categories = np.array(list(set(labels)))
        # print possible_categories
        ind = np.argmax(x, axis=1)
        pred = np.array([possible_categories[i] for i in ind])
        # print y
        # print pred
        return 100.0 - 100.0 * np.sum((pred != labels).astype(int)) / (1.0 * x.shape[0])

def perturb(input, epsilon, idx):
    out_dat = list(input)
    out_dat[idx[0]][idx[1]] += epsilon
    return out_dat

def norm_loss_one(x,y,potential_vals):
    y = np.array([int(y == p) for p in potential_vals])
    return (-1.0 / (x.shape[0])) * np.sum(y * x)



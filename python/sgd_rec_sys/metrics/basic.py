import numpy as np

def accuracy(y_hat, y):
    return sum(y == y_hat)/ y.size


def precision(y_hat, y, classi):
    return sum((y_hat == classi) & (y == classi)) / sum(y_hat == classi)

def recall(y_hat, y, classi):
    return sum((y_hat == classi) & (y == classi)) / sum(y==classi)

def f1(y_hat, y, classi):
    p = precision(y_hat, y, classi)
    r = recall(y_hat, y, classi)

    return 2 * p*r / (p+r)

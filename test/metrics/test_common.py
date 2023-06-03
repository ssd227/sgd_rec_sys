import sys 
sys.path.append('./python')
from sgd_rec_sys.metrics import accuracy, recall, precision, f1


import numpy as np
y_hat = np.array([1,0,0,1,1])
y = np.array([1,0,1,1,1])
    
    
def test_precision():
    assert precision(y_hat, y, 1) == 1

def test_f1():
    assert f1(y_hat, y, 1) == 0.8571428571428571

def test_recall():
    assert recall(y_hat, y, 1)== 0.75

def test_accuracy():
    assert accuracy(y_hat, y) == 0.8


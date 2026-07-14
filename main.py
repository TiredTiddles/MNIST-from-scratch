from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

training_data = pd.read_csv('data/train.csv').to_numpy()
test_data = pd.read_csv('data/test.csv').to_numpy()

print(len(training_data))
np.random.shuffle(training_data)

X_train = training_data[:, 1:] / 255
Y_train = training_data[:, 0]
Y_train = np.eye(10)[Y_train]

m = 42000

X_sample = X_train[0:m].reshape(784,-1)
Y_sample = Y_train[0:m].reshape(10,-1)

print(X_sample.shape)
print(Y_sample.shape)

def initialise_weights():
    W1 = np.random.randn(10, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((10, 1))

    W2 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)
    b2 = np.zeros((10, 1))

    W3 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)
    b3 = np.zeros((10, 1))

    return W1, b1, W2, b2, W3, b3

# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return np.where(Z > 0, 1, 0)

def Softmax(Z):
    Z_exp = np.exp(Z - np.max(Z))
    A = Z_exp / (Z_exp.sum(axis=0, keepdims=True) + 1e-7)
    return A

def forward_prop(W1, b1, W2, b2, W3, b3):
    # layer 1
    Z1 = W1 @ X_sample + b1
    # print("W1:        ",W1.shape)
    # print("X_sample:  ",X_sample.shape)
    # print("b1:        ",b1.shape)
    # print("Z1:        ",Z1.shape)
    A1 = ReLU(Z1)
    # print("A1:        ",A1.shape)
    # print("\n")

    # layer 2
    Z2 = W2 @ A1 + b2
    # print("W2:        ",W2.shape)
    # print("A2:        ",A1.shape)
    # print("b2:        ",b2.shape)
    # print("Z2:        ",Z2.shape)
    A2 = ReLU(Z2)
    # print("A2:        ",A2.shape)
    # print("\n")

    # layer 3
    Z3 = W3 @ A2 + b3
    # print("W3:        ",W3.shape)
    # print("A2:        ",A2.shape)
    # print("b3:        ",b3.shape)
    # print("Z3:        ",Z3.shape)
    A3 = Softmax(Z3)
    # print("A3:        ",A3.shape)
    # print("\n")

    return Z1, A1, Z2, A2, Z3, A3

def back_prop(Y_sample, X_sample, W1, W2, W3, A1, A2, A3, m):
    # calculate loss
    L = -(1/m) * (Y_sample * np.log(A3 + 1e-15)).sum()

    dZ3 = A3 - Y_sample
    dW3 = (1/m) * dZ3 @ A2.T
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * deriv_ReLU(A2)
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * deriv_ReLU(A1)
    dW1 = (1/m) * dZ1 @ X_sample.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW3, db3, dW2, db2, dW1, db1, L

def update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= dW1 * alpha
    b1 -= db1 * alpha
    W2 -= dW2 * alpha
    b2 -= db2 * alpha
    W3 -= dW3 * alpha
    b3 -= db3 * alpha

    return W1, b1, W2, b2, W3, b3

def gradient_descent(X_sample, Y_sample, epoches, m):
    W1, b1, W2, b2, W3, b3 = initialise_weights()
    for epoch in range(epoches):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3)
        dW3, db3, dW2, db2, dW1, db1, L = back_prop(Y_sample, X_sample, W1, W2, W3, A1, A2, A3, m)
        W1, b1, W2, b2, W3, b3 = update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, 1e-2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} completed.")
            print(f"Loss: {L}")

    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = gradient_descent(X_sample, Y_sample, 10001, m)
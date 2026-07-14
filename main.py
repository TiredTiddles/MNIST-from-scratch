import torch
import pandas as pd

training_data = torch.from_numpy(pd.read_csv('data/train.csv').to_numpy())
training_test = torch.from_numpy(pd.read_csv('data/test.csv').to_numpy())

training_data = training_data.T
m,n = training_data.shape

Y_train = torch.eye(10)[training_data[0]] # Y is now one hot encoded
X_train = training_data[1:n] / 255

print(Y_train)
print(X_train)

# Initialise parameters

def init_params():

    W1 = torch.rand(10,784) - 0.5
    b1 = torch.rand(10,1) - 0.5
    W2 = torch.rand(10,10) - 0.5
    b2 = torch.rand(10,1) - 0.5
    W3 = torch.rand(10,10) - 0.5
    b3 = torch.rand(10,1) - 0.5

    return W1, b1, W2, b2, W3, b3

# Activation Functions

def ReLU(Z):
    return torch.clamp(Z, min=0)

def ReLU_deriv(Z):
    return (Z > 0).float()

def Softmax(Z):
    A = torch.exp(Z) / torch.sum(torch.exp(Z), dim=0, keepdim=True)
    return A

# def Softmax_deriv(Z):
#     return

def forward_prop(W1, b1, W2, b2, W3, b3, X):

    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    print(f"Z1.shape: {Z1.shape}")
    print(f"W1.shape: {W1.shape}")
    print(f"X.shape:  {X.shape}")
    print(f"b1.shape: {b1.shape}")
    print(f"A1.shape: {A1.shape}")
    print("\n")

    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    print(f"Z2.shape: {Z2.shape}")
    print(f"W2.shape: {W2.shape}")
    print(f"A1.shape: {A1.shape}")
    print(f"b2.shape: {b2.shape}")
    print(f"A2.shape: {A2.shape}")
    print("\n")

    
    Z3 = W3 @ A2 + b3
    A3 = Softmax(Z3)
    print(f"Z3.shape: {Z3.shape}")
    print(f"W3.shape: {W3.shape}")
    print(f"A2.shape:  {A2.shape}")
    print(f"b3.shape: {b3.shape}")
    print(f"A3.shape: {A3.shape}")
    print("\n")

    return Z1, A1, Z2, A2, Z3, A3

def back_prop(Z1, A1, W1, Z2, A2, W2, Z3, A3, W3, X, Y):
    m = X.shape[1]

    dZ3 = A3 - Y
    dW3 = (1 / m) * dZ3 @ A2.T
    db3 = (1 / m) * torch.sum(dZ3, dim=1, keepdim=True)

    dZ2 = (W3.T @ dZ3) * ReLU_deriv(Z2)
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * torch.sum(dZ2, dim=1, keepdim=True)

    dZ1 = (W2.T @ dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * torch.sum(dZ1, dim=1, keepdim=True)

    return dW3, db3, dW2, db2, dW1, db1

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):

    W1 -= dW1 * alpha
    b1 -= db1 * alpha
    W2 -= dW2 * alpha
    b2 -= db2 * alpha
    W3 -= dW3 * alpha
    b3 -= db3 * alpha

    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = init_params()
print('parameter initialisation complete!\n')
Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_train)
print('forward propagation complete!\n')
dW3, db3, dW2, db2, dW1, db1 = back_prop(Z1, A1, W1, Z2, A2, W2, Z3, A3, W3, X_train, Y_train)
print('backward propagation complete!\n')
W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, 0.01)
print('parameters updated!\n')


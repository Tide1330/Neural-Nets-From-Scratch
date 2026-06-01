import pickle
import numpy as np
import pandas as pd

DATA = pd.read_csv("train.csv")
ARRAY = np.array(DATA)
M, N = ARRAY.shape
np.random.shuffle(ARRAY)
ARRAY_T = ARRAY.T
Y_TRAIN = ARRAY_T[0]
X_TRAIN = ARRAY_T[1:N] / 255.0


def INIT_PARAMS():
    W1 = np.random.randn(64, 784) * np.sqrt(2.0 / 784)
    B1 = np.zeros((64, 1))

    W2 = np.random.randn(32, 64) * np.sqrt(2.0 / 64)
    B2 = np.zeros((32, 1))

    W3 = np.random.randn(10, 32) * np.sqrt(2.0 / 32)
    B3 = np.zeros((10, 1))

    return W1, B1, W2, B2, W3, B3

# Using leaky ReLU because I was encountering a lot of dying ReLU problems
def LEAKY_RELU(Z):
    return np.where(Z > 0, Z, Z * 0.01)


def SOFTMAX(Z):
    SHIFTED_Z = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(SHIFTED_Z) / np.sum(np.exp(SHIFTED_Z), axis=0, keepdims=True)


def DERIVATIVE_LEAKY_RELU(Z):
    return np.where(Z > 0, 1.0, 0.01)

#Okay so this is actually a really beautiful solution from Samson Zhange for onehot
def ONE_HOT(Y):
    ONE_HOT_Y = np.zeros((Y.size, Y.max() + 1))
    ONE_HOT_Y[np.arange(Y.size), Y] = 1
    ONE_HOT_Y = ONE_HOT_Y.T
    return ONE_HOT_Y


def FORWARD_PROP(W1, B1, W2, B2, W3, B3, X):
    Z1 = W1.dot(X) + B1
    A1 = LEAKY_RELU(Z1)

    Z2 = W2.dot(A1) + B2
    A2 = LEAKY_RELU(Z2)

    Z3 = W3.dot(A2) + B3
    A3 = SOFTMAX(Z3)

    return Z1, A1, Z2, A2, Z3, A3


def BACK_PROP(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    M = Y.size
    ONE_HOT_Y = ONE_HOT(Y)
    DZ3 = A3 - ONE_HOT_Y
    DW3 = 1 / M * DZ3.dot(A2.T)
    DB3 = 1 / M * np.sum(DZ3, axis=1, keepdims=True)
    DZ2 = W3.T.dot(DZ3) * DERIVATIVE_LEAKY_RELU(Z2)
    DW2 = 1 / M * DZ2.dot(A1.T)
    DB2 = 1 / M * np.sum(DZ2, axis=1, keepdims=True)
    DZ1 = W2.T.dot(DZ2) * DERIVATIVE_LEAKY_RELU(Z1)
    #We use X val here to account for our inputs at the very begining of the net
    DW1 = 1 / M * DZ1.dot(X.T)
    DB1 = 1 / M * np.sum(DZ1, axis=1, keepdims=True)
    return DW1, DB1, DW2, DB2, DW3, DB3


def UPDATE_PARAMS(W1, B1, W2, B2, W3, B3, DW1, DB1, DW2, DB2, DW3, DB3, LR):
    W1 = W1 - LR * DW1
    B1 = B1 - LR * DB1
    W2 = W2 - LR * DW2
    B2 = B2 - LR * DB2
    W3 = W3 - LR * DW3
    B3 = B3 - LR * DB3
    return W1, B1, W2, B2, W3, B3


def GET_PREDICTIONS(A3):
    return np.argmax(A3, 0)


def GET_ACCURACY(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def GD(X, Y, LR, TargetACC=0.95, ITERATIONS=10000):
    W1, B1, W2, B2, W3, B3 = INIT_PARAMS()
    current_acc = 0.0
    i = 0

    while current_acc < TargetACC and i < ITERATIONS:
        Z1, A1, Z2, A2, Z3, A3 = FORWARD_PROP(W1, B1, W2, B2, W3, B3, X)
        DW1, DB1, DW2, DB2, DW3, DB3 = BACK_PROP(
            Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y
        )
        W1, B1, W2, B2, W3, B3 = UPDATE_PARAMS(
            W1, B1, W2, B2, W3, B3, DW1, DB1, DW2, DB2, DW3, DB3, LR
        )

        current_acc = GET_ACCURACY(GET_PREDICTIONS(A3), Y)

        if i % 50 == 0:
            print("Iteration", i)
            print("ACC", current_acc)

        i += 1

    model_data = {
        "W1": W1,
        "B1": B1,
        "W2": W2,
        "B2": B2,
        "W3": W3,
        "B3": B3,
    }

    if current_acc >= TargetACC:
        with open("mnistmodel.pkl", "wb") as f:
            pickle.dump(model_data, f)
        print("looks like your model worked with 95 percent accuracy")
    else:
        with open("mnistmodel.pkl", "wb") as f:
            pickle.dump(model_data, f)
        print("model trained but its prob not at 95 percent acc")

    return W1, B1, W2, B2, W3, B3


W1, B1, W2, B2, W3, B3 = GD(X_TRAIN, Y_TRAIN, 0.01)
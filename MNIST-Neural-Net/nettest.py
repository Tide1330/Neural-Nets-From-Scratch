import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open("mnistmodel.pkl", "rb") as f:
    loaded_data = pickle.load(f)
W1 = loaded_data["W1"]
B1 = loaded_data["B1"]
W2 = loaded_data["W2"]
B2 = loaded_data["B2"]
W3 = loaded_data["W3"]
B3 = loaded_data["B3"]

#############################################
def LEAKY_RELU(Z):
    return np.where(Z > 0, Z, Z * 0.01)
def SOFTMAX(Z):
    SHIFTED_Z = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(SHIFTED_Z) / np.sum(np.exp(SHIFTED_Z), axis=0, keepdims=True)
def GET_PREDICTIONS(A3):
    return np.argmax(A3, 0)
#############################################


def FORWARD_PROP(W1, B1, W2, B2, W3, B3, X):
    Z1 = W1.dot(X) + B1
    A1 = LEAKY_RELU(Z1)
    
    Z2 = W2.dot(A1) + B2
    A2 = LEAKY_RELU(Z2)
    
    Z3 = W3.dot(A2) + B3
    A3 = SOFTMAX(Z3)
    
    return Z1, A1, Z2, A2, Z3, A3


# Couldn't get the accuracy calc right(encountered a bunch of problems with the dataset)
TEST_DATA = pd.read_csv('test.csv') 
TEST_ARRAY = np.array(TEST_DATA).T
TEST = TEST_ARRAY/ 255.0
_, _, _, _, _, A3_FINAL = FORWARD_PROP(W1, B1, W2, B2, W3, B3, TEST)
PREDICTIONS = GET_PREDICTIONS(A3_FINAL)
running = True
plt.ion()
def run():
    state = {"index": 0}
    fig, ax = plt.subplots()

    def on_key(event):
        if event.key == 'up':
            state["index"] += 1
            if state["index"] < TEST.shape[1]:
                sample_image = TEST[:, state["index"]].reshape(28, 28)
                ax.cla()
                ax.imshow(sample_image, cmap='gray')
                ax.set_title(f"Model Guess: {PREDICTIONS[state['index']]}")
                fig.canvas.draw_idle()
        elif event.key == 'escape':
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    sample_image = TEST[:, state["index"]].reshape(28, 28)
    ax.imshow(sample_image, cmap='gray')
    ax.set_title(f"Model Guess: {PREDICTIONS[state['index']]}")
    plt.show(block=True)

run()
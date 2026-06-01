# MNIST Neural Network from Scratch

This is my first deep dive into the world of neural networks. I wanted to understand how deep learning actually works under the hood, so I built this multi-layer perceptron from scratch using only NumPy and Pandas. No high-level libraries like TensorFlow or PyTorch—just math and code.

## Acknowledgments

A huge thank you to **Samson Zhang** for his incredible video, [Building a Neural Network from Scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1699s). His tutorial was the spark that got me started. While I've expanded on his original architecture to improve performance, his explanation of backpropagation and the elegant way he handled one-hot encoding formed the core of this project.

## Architecture & Improvements

The network is a three-layer dense architecture designed to classify 28x28 grayscale images into ten digits (0-9).

### Solving the "Dying ReLU" Problem
In my early tests, I ran into the "dying ReLU" problem where neurons would just stop learning. To fix this, I switched to **Leaky ReLU** ($f(x) = \max(0.01x, x)$) for the hidden layers. This small change ensures that even if a neuron has a negative input, it still allows a tiny bit of gradient to flow, keeping the learning process alive and robust.

### Third layer for deep learning
In a dataset like this one there are many patterns your model has to be aware of. While Samson's original architecture used only two nodes, I realised that in order to get a higher accuracy it would be more benificial to add a third layer for better pattern recognition.

## Results

By tweaking the architecture—specifically adding Leaky ReLU, using He-style weight initialization, and expanding the hidden layers to 64 and 32 units—I managed to push the accuracy significantly further.

*   **Accuracy:** 95%
*   **Improvement:** A 10% jump from the 85% accuracy in the original tutorial.
*   As you can see I only ran the training process untill it hit the 95% threshhold I wanted, I suspect that given more time and a slight bit more optimization you can easily take this higher it would just take a longer time to train :)

## How to Run It

### 1. Train the Model
This will train the network and save the weights and biases to `mnistmodel.pkl`.
```bash
python3 neuralnet.py
```

### 2. Test & Visualize
Load the trained model and see how it performs on the test set.
```bash
python3 nettest.py
```
*   **Up Arrow:** Cycle through test images.
*   **Escape:** Exit the viewer.

---

## Final Note
This project marks my formal introduction to neural networks and deep learning. Building this from first principles has been a challenging but rewarding foundation for my journey into artificial intelligence.

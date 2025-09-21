import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

folder = r"C:\Users\Srijan Bhushan\Downloads\fashion-minst\final"

def load_data(X, y):
    """
    Loads images and one-hot encoded labels.
    [Q1] Why one-hot encoding?
    One-hot vectors represent classes as probability distributions needed for cross-entropy
    loss and prevent ordinal assumptions about classes.
    """
    """Rest all question answers are in seperate pdf file in github link"""
    for class_name in os.listdir(folder):
        if not class_name.isdigit():
            continue
        idx = int(class_name)
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        for fn in files:
            fp = os.path.join(class_path, fn)
            try:
                img = Image.open(fp).convert('L')  # grayscale
                img = img.resize((28, 28))
                X.append(np.array(img))
            except Exception as e:
                print(f"Skipped {fp}: {e}")

        label = [0] * 10
        label[idx] = 1
        y.extend([label] * len(files))
        print(f"Loaded class {class_name}")

X, y = [], []
load_data(X, y)

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], -1)  # Flatten images to 784

print("Data shapes:", X.shape, y.shape)

class NN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr, epochs, batch_size):
        """
        Two hidden layers network with batch normalization.
        [Q5] Learning rate controls weight update magnitude.
        """
        self.in_n = input_size
        self.hid1_n = hidden1_size
        self.hid2_n = hidden2_size
        self.out_n = output_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Initialize weights with He initialization
        self.wih1 = np.random.randn(self.hid1_n, self.in_n) * np.sqrt(2 / self.in_n)
        self.bih1 = np.zeros((self.hid1_n, 1))
        self.wih2 = np.random.randn(self.hid2_n, self.hid1_n) * np.sqrt(2 / self.hid1_n)
        self.bih2 = np.zeros((self.hid2_n, 1))
        self.who = np.random.randn(self.out_n, self.hid2_n) * np.sqrt(2 / self.hid2_n)
        self.bho = np.zeros((self.out_n, 1))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred), axis=0)

    def cross_entropy_derivative(self, y_true, y_pred):
        return y_pred - y_true

    def batch_norm(self, x):
        mu = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_norm = (x - mu) / (np.sqrt(var + 1e-8))
        return x_norm

    def forward(self, X):
        X = X.T
        X = X - np.mean(X, axis=0, keepdims=True)

        self.z1 = np.dot(self.wih1, X) + self.bih1
        self.a1 = self.batch_norm(self.z1)
        self.a1 = self.relu(self.a1)

        self.z2 = np.dot(self.wih2, self.a1) + self.bih2
        self.a2 = self.batch_norm(self.z2)
        self.a2 = self.relu(self.a2)

        self.z3 = np.dot(self.who, self.a2) + self.bho
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backprop(self, X, y):
        inputs = np.array(X, ndmin=2).T
        inputs = inputs - np.mean(inputs, axis=0, keepdims=True)
        targets = np.array(y, ndmin=2).T

        y_pred = self.forward(X)
        m = inputs.shape[1]

        dZ3 = self.cross_entropy_derivative(targets, y_pred)
        dW3 = np.dot(dZ3, self.a2.T) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m

        dA2 = np.dot(self.who.T, dZ3)
        dZ2 = dA2 * self.relu_derivative(self.a2)  # relu derivative on activated batch-normed output
        dW2 = np.dot(dZ2, self.a1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.wih2.T, dZ2)
        dZ1 = dA1 * self.relu_derivative(self.a1)
        dW1 = np.dot(dZ1, inputs.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.who -= self.lr * dW3
        self.bho -= self.lr * db3
        self.wih2 -= self.lr * dW2
        self.bih2 -= self.lr * db2
        self.wih1 -= self.lr * dW1
        self.bih1 -= self.lr * db1

        loss = self.cross_entropy_loss(targets, y_pred)
        return np.mean(loss)

    def fit(self, X, y, X_val, y_val):
        train_losses = []
        val_losses = []
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            epoch_loss = 0.0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                batch_loss = self.backprop(X_batch, y_batch)
                epoch_loss += batch_loss * X_batch.shape[0]

            epoch_loss /= n_samples
            train_losses.append(epoch_loss)

            val_pred = self.forward(X_val).T
            val_loss = np.mean(self.cross_entropy_loss(y_val.T, val_pred.T))
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        return train_losses, val_losses

    def predict(self, X):
        probs = self.forward(X).T
        return np.argmax(probs, axis=1)

# Shuffle data and split 70:20:10
p = np.random.permutation(len(X))
X, y = X[p], y[p]
s1, s2 = int(0.7 * len(X)), int(0.9 * len(X))
X_train, y_train = X[:s1], y[:s1]
X_val, y_val = X[s1:s2], y[s1:s2]
X_test, y_test = X[s2:], y[s2:]

# Hyperparameters: deeper network, batch norm, smaller LR, longer epochs
model = NN(784, 256, 128, 10, lr=0.005, epochs=300, batch_size=64)

train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)

print(f"Final Test Accuracy: {np.mean(y_pred == y_true):.4f}")

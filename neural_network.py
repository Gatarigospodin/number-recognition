import os
import pickle

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов и смещений случайными значениями
        self.weights_1 = np.random.randn(input_size, hidden_size) * 0.01
        self.biases_1 = np.zeros((1, hidden_size))
        self.weights_2 = np.random.randn(hidden_size, output_size) * 0.01
        self.biases_2 = np.zeros((1, output_size))

    def train(self, X, y, epochs, learning_rate, reg_lambda=0.0):  # Добавляем параметр reg_lambda
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y, learning_rate, reg_lambda)  # Передаем reg_lambda в backward

            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(y, predictions)
                accuracy = self.calculate_accuracy(X, y)  # Вычисляем точность
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")

    def backward(self, X, y, learning_rate, reg_lambda):
        m = X.shape[0]
        dz2 = self.a2 - y
        dw2 = (1 / m) * np.dot(self.a1.T, dz2) + (reg_lambda / m) * self.weights_2  # L2 регуляризация для w2
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.weights_2.T) * self.sigmoid_derivative(self.a1)
        dw1 = (1 / m) * np.dot(X.T, dz1) + (reg_lambda / m) * self.weights_1  # L2 регуляризация для w1
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        self.weights_1 -= learning_rate * dw1
        self.biases_1 -= learning_rate * db1
        self.weights_2 -= learning_rate * dw2
        self.biases_2 -= learning_rate * db2

    def calculate_accuracy(self, X, y):
        predictions = self.forward(X)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predicted_labels == true_labels)

    def forward(self, X):
        # Прямой проход: вычисление предсказаний
        self.z1 = np.dot(X, self.weights_1) + self.biases_1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights_2) + self.biases_2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_predicted):
        # Добавим маленькое число для предотвращения логарифма от нуля
        epsilon = 1e-12
        y_predicted = np.clip(y_predicted, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_predicted)) / y_true.shape[0]
        return loss


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    if os.path.exists(filename):  # Проверка на существование файла
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None

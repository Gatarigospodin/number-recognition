import neural_network as NN
import numpy as np
from dataset import train_X, train_y, test_y, test_X

model_filename = "trained_model.pkl"
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 1000
reg_lambda = 0.01

nn = NN.NeuralNetwork(input_size, hidden_size, output_size)
nn.train(train_X, train_y, epochs, learning_rate, reg_lambda)

# Тестируем нейросеть
predictions = nn.forward(test_X)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_y, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy}")

NN.save_model(nn, "trained_model.pkl")

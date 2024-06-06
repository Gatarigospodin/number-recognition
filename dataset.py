import numpy as np
from keras.src.datasets import mnist

# Загружаем данные MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Преобразуем данные в удобный формат:
# - Изображения (28x28 пикселей) преобразуем в векторы (784 элемента)
# - Значения пикселей нормализуем к диапазону [0, 1]
train_X = train_X.reshape((60000, 784))
test_X = test_X.reshape((10000, 784))

train_X[train_X != 0] = 1
test_X[test_X != 0] = 1

# Преобразуем метки классов в one-hot encoding
# (например, цифра "5" -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]

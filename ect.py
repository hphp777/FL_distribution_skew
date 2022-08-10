# import tensorflow as tf
import torch

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# print(x_train.shape)

print(torch.cuda.is_available())
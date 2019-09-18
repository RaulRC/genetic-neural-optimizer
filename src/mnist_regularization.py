from genetic_optimizer import *
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
from keras.datasets import mnist
import itertools
from tensorflow.keras.callbacks import EarlyStopping
from default_regularization import regularization
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


from config_mnist_regularization import *
OUTPUT_PATH = "outputs/mnist_regularization/"

comb = [generations_list, populations_list, elitism_list, mutables_list]

experiments = list(itertools.product(*comb))


def generate_model():
    result = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    result.compile(optimizer='sgd',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return result

cb = [EarlyStopping(monitor='val_acc', min_delta=0.1)]
cb = []
regularization(experiments, iterations, generate_model, EPOCHS, x_train, y_train, x_test, y_test, OUTPUT_PATH, cb)


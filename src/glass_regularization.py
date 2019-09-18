from genetic_optimizer import *
import pdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import itertools
import time
from tensorflow.keras.callbacks import EarlyStopping
from default_regularization import regularization


dataset = pd.read_csv("data/glass.data")

#Splitting the data into training and test test
X_original = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values

from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X_original)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


from config_regularization import *

OUTPUT_PATH = "outputs/glass_regularization/"
comb = [generations_list, populations_list, elitism_list, mutables_list]

experiments = list(itertools.product(*comb))


def generate_model():
    genetic_model = Sequential()
    genetic_model.add(Dense(128, input_shape=(9,), activation='relu'))
    genetic_model.add(Dense(64, activation='relu'))
    genetic_model.add(Dense(6, activation='relu'))
    opt = SGD(lr=0.1)
    genetic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return genetic_model


cb = [EarlyStopping(monitor='val_acc')]
cb = []

regularization(experiments, iterations, generate_model, EPOCHS, x_train, y_train, x_test, y_test, OUTPUT_PATH, cb)


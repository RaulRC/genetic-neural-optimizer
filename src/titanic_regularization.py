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

columns = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
dataset = pd.read_csv("data/titanic/train.csv")[columns]
dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked'])

columns = dataset.columns

X_original = dataset[columns[1:]]
Y = dataset[columns[:1]]

from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X_original)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from config_regularization import *

OUTPUT_PATH = "outputs/titanic_regularization/"
comb = [generations_list, populations_list, elitism_list, mutables_list]

experiments = list(itertools.product(*comb))

def generate_model():
    genetic_model = Sequential()
    genetic_model.add(Dense(10, input_shape=(9,), activation='sigmoid'))
    genetic_model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.1)
    genetic_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return genetic_model


cb = [EarlyStopping(monitor='val_acc')]
cb = []

regularization(experiments, iterations, generate_model, EPOCHS, x_train, y_train, x_test, y_test, OUTPUT_PATH, cb)

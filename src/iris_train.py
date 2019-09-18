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

dataset = pd.read_csv("data/Iris.csv")

#Splitting the data into training and test test
X_original = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X_original)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


from config_train import *

OUTPUT_PATH = "outputs/iris_train/"

comb = [generations_list, populations_list, elitism_list, mutables_list]

experiments = list(itertools.product(*comb))


def generate_model():
    genetic_model = Sequential()
    genetic_model.add(Dense(10, input_shape=(4,), activation='sigmoid'))
    genetic_model.add(Dense(3, activation='sigmoid'))
    opt = SGD(lr=0.1)
    genetic_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return genetic_model


f = open('{}results.csv'.format(OUTPUT_PATH), 'w')
f.write("generations,population,elitism,mutables,time,train_loss,test_loss,train_acc,test_acc\n")

for e in experiments:
    exp_train_loss = []
    exp_test_loss = []
    exp_train_acc = []
    exp_test_acc = []
    exp_time = []
    plt.figure()
    for i in range(iterations):
        model = generate_model()

        exp_generation = e[0]
        exp_population = e[1]
        exp_elite = int(e[2] * exp_population)
        exp_mutables = e[3]
        print("Starting experiment \n\tg{}\n\tp{}\n\te{}\n\tm{}".format(exp_generation,
                                                                        exp_population,
                                                                        exp_elite,
                                                                        exp_mutables))
        ga = GeneticNeuralOptimizer(model,
                                    mutation_prob=0.9,
                                    iterations=exp_generation,
                                    mutables=exp_mutables,
                                    elite=exp_elite,
                                    )
        pop = ga.generate_population(exp_population)
        start = time.time()
        best, best_value, history = ga.fit(pop, x_train, y_train, x_test[:100], y_test[:100])
        end = time.time()
        print("Best weights found: {}".format(best))
        print("Best value found: {}".format(best_value))

        test_loss, test_acc = ga.model.evaluate(x_test, y_test)
        train_loss, train_acc = ga.model.evaluate(x_train, y_train)
        print('Test accuracy: ' + str(test_acc))
        print('Train accuracy: ' + str(train_acc))
        f.write("{},{},{},{},{},{},{},{},{}\n".format(exp_generation,
                                                      exp_population,
                                                      exp_elite,
                                                      exp_mutables,
                                                      round(end - start, 2),
                                                      round(train_loss, 2),
                                                      round(test_loss, 2),
                                                      round(train_acc, 2),
                                                      round(test_acc, 2)
                                                      ))
        exp_train_loss.append(train_loss)
        exp_test_loss.append(test_loss)
        exp_train_acc.append(train_acc)
        exp_test_acc.append(test_acc)
        exp_time.append(end-start)
        plt.plot(history)
        plt.title("Evolution g{}_p{}_e{}_m{}".format(exp_generation, exp_population, exp_elite, exp_mutables))
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.savefig("{}g{}_p{}_e{}_m{}_result.png".format(OUTPUT_PATH, exp_generation, exp_population, exp_elite,
                                                          exp_mutables))

    f.write(",,,,{},{},{},{},{}\n".format(round(np.mean(exp_time), 2),
                                          round(np.mean(exp_train_loss), 2),
                                          round(np.mean(exp_test_loss), 2),
                                          round(np.mean(exp_train_acc), 2),
                                          round(np.mean(exp_test_acc), 2),
                                          ))

f.close()




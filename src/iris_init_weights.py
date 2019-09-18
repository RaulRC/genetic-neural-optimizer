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


from config_basic_init_weights import *

OUTPUT_PATH = "outputs/iris_init_weights/"

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
f.write("generations,population,elitism,mutables,time,base_train_acc,base_test_acc,ga_train_acc,ga_test_acc\n")

for e in experiments:
    exp_train_acc = []
    exp_test_acc = []
    exp_base_train_acc = []
    exp_base_test_acc = []
    exp_time = []

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

    for i in range(iterations):
        model = generate_model()
        model_base = generate_model()
        exp_generation = e[0]
        exp_population = e[1]
        exp_elite = int(e[2] * exp_population)
        exp_mutables = e[3]
        print("Starting experiment \n\tg{}\n\tp{}\n\te{}\n\tm{}".format(exp_generation,
                                                                        exp_population,
                                                                        exp_elite,
                                                                        exp_mutables))
        ga = GeneticNeuralWeightOptimizer(model,
                                          mutation_prob=0.9,
                                          iterations=exp_generation,
                                          mutables=exp_mutables,
                                          elite=exp_elite,
                                          epochs=EPOCHS
                                    )
        pop = ga.generate_population(exp_population)
        start = time.time()
        best, best_value, history_ga, history_bp = ga.fit(pop, x_train, y_train, x_test, y_test)
        end = time.time()
        history_base = model_base.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))


        print("Best weights found: {}".format(best))
        print("Best value found: {}".format(best_value))

        # Evaluation
        # - Genetic
        train_loss, train_acc = ga.model.evaluate(x_train, y_train)
        test_loss, test_acc = ga.model.evaluate(x_test, y_test)
        print('Train accuracy: ' + str(train_acc))
        print('Test accuracy: ' + str(test_acc))

        # - Standard Base model
        base_train_loss, base_train_acc = model_base.evaluate(x_train, y_train)
        base_test_loss, base_test_acc = model_base.evaluate(x_test, y_test)
        print("Base train accuracy: {}".format(base_train_acc))
        print("Base test accuracy: {}".format(base_test_acc))

        f.write("{},{},{},{},{},{},{},{},{}\n".format(exp_generation,
                                                      exp_population,
                                                      exp_elite,
                                                      exp_mutables,
                                                      round(end-start, 2),
                                                      round(base_train_acc, 2),
                                                      round(base_test_acc, 2),
                                                      round(train_acc, 2),
                                                      round(test_acc, 2)))
        exp_train_acc.append(train_acc)
        exp_test_acc.append(test_acc)
        exp_base_train_acc.append(base_train_acc)
        exp_base_test_acc.append(base_test_acc)
        exp_time.append(end-start)

        axs[0].set_title("Genetic Initialization")
        axs[1].set_title("Backpropagation")
        axs[0].set_ylim([0.0, 1.2])
        axs[1].set_ylim([0.0, 1.2])
        fig.suptitle("Evolution g{}_p{}_e{}_m{}".format(exp_generation,
                                                        exp_population,
                                                        exp_elite,
                                                        exp_mutables), y=1.0)

        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        plt.tight_layout()
        axs[0].plot(history_bp.history['val_acc'])
        axs[1].plot(history_base.history['val_acc'])
        plt.savefig("{}g{}_p{}_e{}_m{}_result.png".format(OUTPUT_PATH, exp_generation, exp_population, exp_elite,
                                                          exp_mutables))

    fig_s, axs_s = plt.subplots(nrows=1, ncols=2, sharey=True)
    axs_s[0].boxplot(exp_test_acc, labels=['Genetic Initialization'])
    axs_s[1].boxplot(exp_base_test_acc, labels=['Default Backpropagation'])
    # axs_s[0].set_ylim([0.0, 1.0])
    # axs_s[1].set_ylim([0.0, 1.0])

    fig_s.suptitle("Test partition evaluation g{}_p{}_e{}_m{}".format(exp_generation,
                                                                      exp_population,
                                                                      exp_elite,
                                                                      exp_mutables), y=1.0)
    axs_s[0].set_ylabel("Accuracy")
    axs_s[1].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("{}scatter_g{}_p{}_e{}_m{}_result.png".format(OUTPUT_PATH, exp_generation, exp_population,
                                                              exp_elite,
                                                              exp_mutables))

    f.write(",,,,{},{},{},{},{}\n".format(round(np.mean(exp_time), 2),
                                          round(np.mean(exp_base_train_acc), 2),
                                          round(np.mean(exp_base_test_acc), 2),
                                          round(np.mean(exp_train_acc), 2),
                                          round(np.mean(exp_test_acc), 2)
                                          ))

f.close()




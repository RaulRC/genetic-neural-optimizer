from genetic_optimizer import *
import pdb
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import itertools
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

from config_mnist_train import *

OUTPUT_PATH = "outputs/fashion_train/"

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




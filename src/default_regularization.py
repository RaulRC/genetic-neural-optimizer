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
from keras.optimizers import SGD

def regularization(experiments, iterations, generate_model, EPOCHS, x_train, y_train, x_test, y_test, OUTPUT_PATH, cb):
    f = open('{}results.csv'.format(OUTPUT_PATH), 'w')
    f.write(
        "generations,population,elitism,mutables,time,base_train_acc,base_test_acc,ga_train_acc,ga_test_acc,base_diff,ga_diff\n")

    for e in experiments:
        exp_train_acc = []
        exp_test_acc = []
        exp_base_train_acc = []
        exp_base_test_acc = []
        exp_time = []
        exp_diff = []
        exp_base_diff = []


        for i in range(iterations):
            fig, axs = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True)
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
            ga = GeneticRegularizator(model,
                                      mutation_prob=0.9,
                                      iterations=exp_generation,
                                      mutables=exp_mutables,
                                      elite=exp_elite,
                                      epochs=EPOCHS,
                                      callbacks=cb
                                      )
            pop = ga.generate_population(exp_population)
            start = time.time()
            best, best_value, history_ga, history_bp = ga.fit(pop, x_train, y_train, x_test, y_test)
            es_epoch = len(history_bp.history['val_loss'])
            end = time.time()
            # history_base = model_base.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))


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
            result_diff = abs(train_acc - test_acc)
            result_base_diff = abs(base_train_acc - base_test_acc)

            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(exp_generation,
                                                                exp_population,
                                                                exp_elite,
                                                                exp_mutables,
                                                                round(end-start, 2),
                                                                round(base_train_acc, 2),
                                                                round(base_test_acc, 2),
                                                                round(train_acc, 2),
                                                                round(test_acc, 2),
                                                                round(result_base_diff, 2),
                                                                round(result_diff, 2)
                                                                ))
            exp_train_acc.append(train_acc)
            exp_test_acc.append(test_acc)
            exp_base_train_acc.append(base_train_acc)
            exp_base_test_acc.append(base_test_acc)

            exp_diff.append(result_diff)
            exp_base_diff.append(result_base_diff)

            exp_time.append(end-start)

            axs.set_title("Genetic Regularization")
            axs.set_ylim([0.0, 1.2])
            fig.suptitle("Evolution g{}_p{}_e{}_m{}".format(exp_generation,
                                                            exp_population,
                                                            exp_elite,
                                                            exp_mutables), y=1.0)

            axs.set_xlabel("Epoch")
            axs.set_ylabel("Accuracy")
            plt.tight_layout()

            r_t = history_bp.history['acc'] + [train_acc]
            r_val = history_bp.history['val_acc'] + [test_acc]
            tacc = round(history_bp.history['acc'][-1], 2)
            testacc = round(history_bp.history['val_acc'][-1], 2)
            tdiff = abs(round(tacc-testacc, 2))
            axs.set_xlim(-1, len(r_t) + 1)
            l1 = 'train ({0:.2f})'.format(tacc)
            l2 = 'val (|{0:.2f} - {1:.2f}| = {2:.2f})'.format(
                tacc,
                testacc,
                tdiff
            )

            axs.plot(r_t, label=l1)
            axs.plot(r_val, label=l2)

            r_train_acc = round(train_acc, 2)
            r_test_acc = round(test_acc, 2)
            r_diff = round(abs(r_train_acc - r_test_acc), 2)

            l3 = 'ga_regularization (|{0:.2f} - {1:.2f}|) = {2:.2f}'.format(
                            r_train_acc,
                            r_test_acc,
                            r_diff
                        )

            plt.axvline(x=es_epoch-1, linewidth=1, color='r', ls='--',
                        label=l3)
            axs.legend(loc='upper left')
            plt.savefig("{}exp{}_g{}_p{}_e{}_m{}_result.png".format(OUTPUT_PATH, i, exp_generation, exp_population,
                                                                    exp_elite,
                                                                    exp_mutables))

        fig_s, axs_s = plt.subplots(nrows=1, ncols=2, sharey=True)
        axs_s[0].boxplot(exp_diff, labels=['Genetic Initialization'])
        axs_s[1].boxplot(exp_base_diff, labels=['Default Backpropagation'])
        # axs_s[0].set_ylim([0.0, 0.4])
        # axs_s[1].set_ylim([0.0, 0.4])

        fig_s.suptitle("Train/test difference g{}_p{}_e{}_m{}".format(exp_generation,
                                                                      exp_population,
                                                                      exp_elite,
                                                                      exp_mutables), y=1.0)
        axs_s[0].set_ylabel("Accuracy")
        axs_s[1].set_ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig("{}scatter_g{}_p{}_e{}_m{}_result.png".format(OUTPUT_PATH, exp_generation, exp_population,
                                                                  exp_elite,
                                                                  exp_mutables))

        f.write(",,,,{},{},{},{},{},{},{}\n".format(round(np.mean(exp_time), 2),
                                                    round(np.mean(exp_base_train_acc), 2),
                                                    round(np.mean(exp_base_test_acc), 2),
                                                    round(np.mean(exp_train_acc), 2),
                                                    round(np.mean(exp_test_acc), 2),
                                                    round(np.mean(exp_base_diff), 2),
                                                    round(np.mean(exp_diff), 2),
                                              ))

    f.close()

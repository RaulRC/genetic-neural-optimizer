import numpy as np
import random
import types

from abc import ABC


class GeneticOptimizer(ABC):
    """
    \nGeneticOptimizer Abstract class, define the
    basic functions of the rest of Genetic Optimizer classes\n
    - fit function: train using Genetic Algorithm\n
    - GA function\n
    Operations needed durante Genetic Algorithms\n
    - fitness function: evaluating a chromosome\n
    - generate_population\n
    - random_selection\n
    - reproduce\n
    - mutate\n
    """
    def fitness(self, args, **kwargs):
        pass

    def fit(self, args, **kwargs):
        pass

    def GA(self, args, **kwargs):
        pass

    def generate_population(self, individuals, distribution='uniform'):
        """

        :return: numpy array with
        """
        if distribution == 'uniform':
            dist = np.random.uniform
        elif distribution == 'normal':
            dist = np.random.normal
        elif distribution == 'crazy':
            dist = random.choice([np.random.uniform, np.random.normal])
        else:
            dist = distribution

        population = []
        for i in range(individuals):
            individual = []
            for vector_shape in self.weights_shape:
                individual.append(dist(-1.0, 1.0, vector_shape[0]*vector_shape[1]))
            population.append(individual)
        return population

    def get_pop_dist(self):
        if self.pop_dist == 'crazy':
            res = random.choice(['uniform', 'normal'])
        else:
            res = self.pop_dist
        return res

    def random_selection(self, population, distribution=None):
        """

        :param population: list of numpy array
        :param distribution: probability distribution used during selection
        :return: tuple of two more promising individuals (numpy vectors)
        """
        if not distribution:
            distribution = [1/float(x + 1) for x in range(1, len(population)+1)]

        pop_index = [x for x in range(len(population))]
        x = random.choices(population=pop_index, weights=distribution)[-1]
        y = x
        while y == x:
            y = random.choices(population=pop_index, weights=distribution)[-1]
        return population[x], population[y]

    def original_weights_shape(self):
        """

        :return: shape of each layer in the neural network (list of numpy vectors)
        """
        return [x.shape for x in self.model.get_weights()]

    @staticmethod
    def reproduce(x, y):
        """

        :param x: numpy vector of genes
        :param y: numpy vector of genes
        :return: random crossover of x with y as numpy vector
        """
        child_a = []
        child_b = []
        for i, item in enumerate(x):
            n = len(x[i])
            pivot = random.randint(0, n - 1)
            result1 = np.concatenate((x[i][0:pivot], y[i][pivot:n]))
            result2 = np.concatenate((x[i][pivot:n], y[i][0:pivot]))
            child_a.append(result1)
            child_b.append(result2)
        return child_a, child_b

    def mutate(self, x, dist=np.random.uniform):
        """

        :param x: individual chromosome to mute (numpy vector)
        :param dist: probability distribution used (default is Uniform)
        :return: mutated x (numpy vector)
        """
        for layer in x:
            for _ in range(self.mutables):
                if random.random() <= self.mutation_prob:
                    mut_gene = random.randint(0, layer.shape[0] - 1)
                    if self.mutation_rate:
                        layer[mut_gene] += (self.mutation_rate if random.randint(0, 1) == 0 else -self.mutation_rate)
                    else:
                        layer[mut_gene] = dist(-1, 1, 1)[0]
        return x


class GeneticNeuralOptimizer(GeneticOptimizer):
    """
    \nGeneticNeuralOptimizer class.\n
    Train a Neural Network using a Genetic Algorithm.
    """

    def fitness(self, individual, x, y):
        """

        :param individual: list of individuals
        :param x_train: train dataset (numpy array)
        :param y_train: labels for train dataset (numpy vector)
        :return: best chromosome found (numpy vector), best value found for that chromsome (float), training history
        (list of fitness values), keras.history for the backpropagation training
        """
        new_weights = self.transform_weights(individual)
        self.model.set_weights(new_weights)
        evaluation = self.model.evaluate(x, y)
        return self.fitness_function(evaluation)

    def __init__(self, model,
                 mutation_prob=0.5,
                 mutation_rate=None,
                 elite=2,
                 genetic_train=True,
                 stop_condition=None,
                 epochs=-1,
                 fitness_function=lambda x: x[1],
                 mutables=1,
                 original_mutation_prob=0.5,
                 pop_dist='uniform',
                 mutation_dist=np.random.uniform,
                 min_delta=1.0,
                 regularization_metric='val_acc',
                 iterations=None):
        assert isinstance(fitness_function,
                          types.FunctionType), ("fitness_function should be an user defined function instead of" +
                                                " {}".format(type(fitness_function)))

        assert mutables >=0, "mutables param should be greater or equal 0"
        self.original_mutation_prob=original_mutation_prob
        self.mutation_dist = mutation_dist
        self.pop_dist = pop_dist
        self.fitness_function = fitness_function
        self.mutables = mutables
        self.regularization_metric = regularization_metric
        self.model = model
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.stop_condition = stop_condition
        if self. stop_condition:
            assert not self.iterations, "stop_condition is not compatible with iterations"
        if self.iterations:
            assert not self.stop_condition, "stop_condition is not compatible with iterations"
        self.top_best_preserved = elite
        self.original_gene = self.weights_to_vector()
        self.weights_shape = self.one_vector_shape()
        self.min_delta = min_delta
        self.original_weights_shape = self.original_weights_shape()
        self.genetic_train = genetic_train
        self.epochs = epochs
        self.ranking = list()

    def transform_weights(self, individual):
        """

        :param individual: chromosome (numpy vector)
        :return: reshaped chromosome into Neural Network weight arrangement (list of numpy vectors)
        """
        result = []
        for i, x in enumerate(individual):
            result.append(individual[i].reshape(self.original_weights_shape[i]))
        return result

    def weights_to_vector(self):
        """

        :return: list of numpy vectors
        """
        original = []
        for vector in self.model.get_weights():
            one_dimension = 1
            for dim in vector.shape:
                one_dimension *= dim
            original.append(vector.reshape(one_dimension, 1))
        original = np.array(original)
        return original

    def weights_to_vector_alt(self):
        """

        :return: list of numpy vectors
        """
        original = []
        for vector in self.model.get_weights():
            one_dimension = 1
            for dim in vector.shape:
                one_dimension *= dim
            original.append(vector.reshape(one_dimension))
        original = np.array(original)
        return original

    def one_vector_shape(self):
        return [x.shape for x in self.original_gene]

    def fit(self, population, x_train, y_train, x_test, y_test):
        """

        :param population: initial population. List of numpy vectors
        :param x_train: train dataset (numpy array)
        :param y_train: labels for train dataset (numpy vector)
        :param x_test: test dataset (numpy array)
        :param y_test: labels for test dataset (numpy vector)
        :return: best chromosome found (numpy vector), best value found for that chromsome (float), training history
        (list of fitness values)
        """
        assert len(population) >= 2, "number of individuals in population should be >= 2"
        self.max_pop_length = len(population)
        self.ranking = [(x, self.fitness(x, x_train, y_train)) for x in population]
        best, best_value, history = self.GA(population, self.iterations, x_train, y_train, x_test, y_test)
        return best, best_value, history

    def GA(self, population, iterations, x, y, x_test, y_test):
        """

        :param population: initial population. List of numpy vectors
        :param iterations: integer. Number of generations
        :param x: train dataset (numpy array)
        :param y: labels for train dataset (numpy vector)
        :param x_test: test dataset (numpy array)
        :param y_test: labels for test dataset (numpy vector)
        :return: tuple (best individual found as numpy vector, max value found, history of bests)
        """
        self.best_of_generation(self.fitness, population, x, y)
        history = [0]
        all_the_bests = [self.model.get_weights()]

        for i in range(iterations):
            distribution = [1 / float(x + 1) for x in range(1, len(population) + 1)]
            print("Generation {}".format(i))
            new_population = []
            for _ in population:
                if len(new_population) < self.max_pop_length:
                    # parent_one, parent_two = self.random_selection(self.ranking)
                    parent_one, parent_two = self.random_selection(list(map(lambda x: x[0], self.ranking)), distribution)
                    c_a, c_b = self.reproduce(parent_one, parent_two)
                    new_population.append(self.mutate(c_a, self.mutation_dist))
                    new_population.append(self.mutate(c_b, self.mutation_dist))

            population = list()

            for item in range(self.top_best_preserved):
                population.append(self.ranking[item][0])

            population += new_population

            best, best_value = self.best_of_generation(self.fitness, population, x, y)
            history.append(best_value)
            all_the_bests.append(best)

        best_index = history.index(max(history[1:]))
        best = all_the_bests[best_index]
        best_value = history[best_index]
        self.fitness(best, x_test, y_test)
        return best, best_value, history

    def best_of_generation(self, fitness, population, x, y):
        """

        :param fitness: fitness function
        :param eq: equation to maximize
        :param population: list of individuals (numpy array)
        :param x: train dataset (numpy array)
        :param y: labels for train dataset (numpy vector)
        :return: tuple of numpy vector best individual and best value
        """
        best = None
        best_value = -9999999

        for i, p in enumerate(population[:self.top_best_preserved]):
            current_value = self.ranking[i][1]
            if current_value > best_value:
                best = p
                best_value = current_value

        self.ranking = self.ranking[:self.top_best_preserved]
        print("Best value: {} --- {}".format(best_value, list(map(lambda x: round(x[1], 3), self.ranking))))
        for p in population[self.top_best_preserved:]:
            current_value = fitness(p, x, y)
            self.ranking.append((p, current_value))
            if current_value > best_value:
                best = p
                best_value = current_value

        self.ranking.sort(key=lambda x : x[1], reverse=True)
        return best, best_value


class GeneticNeuralWeightOptimizer(GeneticOptimizer):
    """
    \nGeneticNeuralWeightOptimizer class\n
    Initialize Neural Network weights using genetic algorithm.
    """
    def fitness(self, individual, x, y):
        """

        :param individual: individual chromosome to mute (numpy vector)
        :param x: dataset to evaluate on the model
        :param y: labels for each point of the dataset x
        :return: fitness value (float)
        """
        new_weights = self.transform_weights(individual)
        self.model.set_weights(new_weights)
        evaluation = self.model.evaluate(x, y)
        return self.fitness_function(evaluation)

    def __init__(self, model,
                 mutation_prob=0.5,
                 mutation_rate=None,
                 elite=2,
                 stop_condition=None,
                 epochs=-1,
                 fitness_function=lambda x: x[1],
                 mutables=1,
                 mutation_dist=np.random.uniform,
                 pop_dist='uniform',
                 iterations=None):

        assert isinstance(fitness_function,
                          types.FunctionType), ("fitness_function should be an user defined function instead of" +
                                                " {}".format(type(fitness_function)))

        assert mutables >= 0, "mutables param should be greater or equal 0"
        self.mutation_dist = mutation_dist
        self.pop_dist = pop_dist
        self.fitness_function = fitness_function
        self.mutables = mutables
        self.model = model
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.stop_condition = stop_condition
        if self. stop_condition:
            assert not self.iterations, "stop_condition is not compatible with iterations"
        if self.iterations:
            assert not self.stop_condition, "stop_condition is not compatible with iterations"
        self.top_best_preserved = elite
        self.original_gene = self.weights_to_vector()
        self.weights_shape = self.one_vector_shape()
        self.original_weights_shape = self.original_weights_shape()
        self.epochs = epochs
        self.ranking = list()
        self.max_pop_length = -1

    def transform_weights(self, individual):
        """

        :param individual: chromosome (numpy vector)
        :return: reshaped chromosome into Neural Network weight arrangement (list of numpy vectors)
        """
        result = []
        for i, x in enumerate(individual):
            result.append(individual[i].reshape(self.original_weights_shape[i]))
        return result

    def weights_to_vector(self):
        """
        Transform neural network weights to a chromosome
        :return: list of numpy vectors
        """
        original = []
        for vector in self.model.get_weights():
            one_dimension = 1
            for dim in vector.shape:
                one_dimension *= dim
            original.append(vector.reshape(one_dimension, 1))
        original = np.array(original)
        return original

    def weights_to_vector_alt(self):
        """

        :return: list of numpy vectors
        """
        original = []
        for vector in self.model.get_weights():
            one_dimension = 1
            for dim in vector.shape:
                one_dimension *= dim
            original.append(vector.reshape(one_dimension))
        original = np.array(original)
        return original

    def one_vector_shape(self):
        return [x.shape for x in self.original_gene]

    def fit(self, population, x_train, y_train, x_test, y_test):
        """

        :param population: list of individuals 
        :param x_train: train dataset (numpy array)
        :param y_train: labels for train dataset (numpy vector)
        :param x_test: test dataset (numpy array)
        :param y_test: labels for test dataset (numpy vector)
        :return: best chromosome found (numpy vector), best value found for that chromsome (float), training history
        (list of fitness values), keras.history for the backpropagation training
        """
        assert len(population) >= 2, "number of individuals in population should be >= 2"
        self.max_pop_length = len(population)
        self.ranking = [(x, self.fitness(x, x_train, y_train))
                        for x in self.generate_population(self.top_best_preserved, self.get_pop_dist())]
        best, best_value, history = self.GA(population, self.iterations, x_train, y_train, x_test, y_test)
        self.model.set_weights(self.transform_weights(best))
        history_bp = self.model.fit(x_train, y_train, epochs=self.epochs, validation_data=(x_test, y_test))
        return self.transform_weights(best), best_value, history, history_bp

    def GA(self, population, iterations, x, y, x_test, y_test):
        """

        :param population: initial population. List of numpy vectors
        :param iterations: integer. Number of generations
        :param x: train dataset (numpy array)
        :param y: labels for train dataset (numpy vector)
        :param x_test: test dataset (numpy array)
        :param y_test: labels for test dataset (numpy vector)
        :return: tuple (best individual found as numpy vector, max value found, history of bests)
        """
        self.best_of_generation(self.fitness, population, x, y)
        history = [0]
        all_the_bests = [self.model.get_weights()]

        for i in range(iterations):
            distribution = [1 / float(x + 1) for x in range(1, len(population) + 1)]
            print("Generation {}".format(i))
            new_population = []
            for _ in population:
                if len(new_population) < self.max_pop_length:
                    parent_one, parent_two = self.random_selection(list(map(lambda x: x[0], self.ranking)), distribution)
                    # parent_one, parent_two = self.random_selection(population)
                    c_a, c_b = self.reproduce(parent_one, parent_two)
                    new_population.append(self.mutate(c_a, self.mutation_dist))
                    new_population.append(self.mutate(c_b, self.mutation_dist))
            population = list()
            for item in range(self.top_best_preserved):
                population.append(self.ranking[item][0])

            population += new_population

            best, best_value = self.best_of_generation(self.fitness, population, x, y)
            history.append(best_value)
            all_the_bests.append(best)

        best_index = history.index(max(history[1:]))
        best = all_the_bests[best_index]
        best_value = history[best_index]
        self.fitness(best, x_test, y_test)
        return best, best_value, history

    def best_of_generation(self, fitness, population, x, y):
        """

        :param fitness: fitness function
        :param population: list of individuals (numpy array)
        :param x: train dataset (numpy array)
        :param y: labels for train dataset (numpy vector)
        :return: tuple of numpy vector best individual and best value
        """
        best = None
        best_value = -9999999
        for i, p in enumerate(population[:self.top_best_preserved]):
            current_value = self.ranking[i][1]
            if current_value > best_value:
                best = p
                best_value = current_value

        self.ranking = self.ranking[:self.top_best_preserved]
        print("Best value: {} --- {}".format(best_value, list(map(lambda x: round(x[1], 3), self.ranking))))
        for p in population[self.top_best_preserved:]:
            current_value = fitness(p, x, y)
            self.ranking.append((p, current_value))
            if current_value > best_value:
                best = p
                best_value = current_value

        self.ranking.sort(key=lambda k: k[1], reverse=True)
        return best, best_value


class GeneticRegularizator(GeneticOptimizer):
    """
    \nGeneticRegularizator class\n
    Using during a backpropagation training, will regularize the whole training if \n

    - If callback is provided and then, the training process is interrupted, or\n
    - When the training process with backpropagation is finished

    """

    def fitness(self, individual, x, y):
        """

        :param individual: individual chromosome to mute (numpy vector)
        :param x: dataset to evaluate on the model
        :param y: labels for each point of the dataset x
        :return: fitness value (float)
        """
        new_weights = self.transform_weights(individual)
        self.model.set_weights(new_weights)
        evaluation = self.model.evaluate(x, y)
        return self.fitness_function(evaluation)

    def __init__(self, model,
                 mutation_prob=0.5,
                 mutation_rate=None,
                 elite=2,
                 genetic_train=True,
                 stop_condition=None,
                 epochs=-1,
                 fitness_function=lambda x: x[1],
                 mutables=1,
                 original_mutation_prob=0.5,
                 pop_dist='uniform',
                 mutation_dist=np.random.uniform,
                 callbacks = [],
                 iterations=None):
        assert isinstance(fitness_function,
                          types.FunctionType), ("fitness_function should be an user defined function instead of" +
                                                " {}".format(type(fitness_function)))

        assert mutables >=0, "mutables param should be greater or equal 0"
        self.original_mutation_prob=original_mutation_prob
        self.mutation_dist = mutation_dist
        self.pop_dist = pop_dist
        self.fitness_function = fitness_function
        self.mutables = mutables
        self.model = model
        self.callbacks = callbacks
        self.elite = elite
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.stop_condition = stop_condition
        if self. stop_condition:
            assert not self.iterations, "stop_condition is not compatible with iterations"
        if self.iterations:
            assert not self.stop_condition, "stop_condition is not compatible with iterations"
        self.top_best_preserved = elite
        self.original_gene = self.weights_to_vector()
        self.weights_shape = self.one_vector_shape()
        self.mode = None
        self.original_weights_shape = self.original_weights_shape()
        self.genetic_train = genetic_train
        self.epochs = epochs
        self.ranking = list()

    def transform_weights(self, individual):
        """

        :param individual: chromosome (numpy vector)
        :return: reshaped chromosome into Neural Network weight arrangement (list of numpy vectors)
        """
        result = []
        for i, x in enumerate(individual):
            result.append(individual[i].reshape(self.original_weights_shape[i]))
        return result


    def weights_to_vector(self):
        """

        :return: list of numpy vectors
        """
        original = []
        for vector in self.model.get_weights():
            one_dimension = 1
            for dim in vector.shape:
                one_dimension *= dim
            original.append(vector.reshape(one_dimension, 1))
        original = np.array(original)
        return original

    def weights_to_vector_alt(self):
        """

        :return: list of numpy vectors
        """
        original = []
        for vector in self.model.get_weights():
            one_dimension = 1
            for dim in vector.shape:
                one_dimension *= dim
            original.append(vector.reshape(one_dimension))
        original = np.array(original)
        return original

    def one_vector_shape(self):
        return [x.shape for x in self.original_gene]

    def fit(self, population, x_train, y_train, x_test, y_test):
        """

        :param population: initial population. List of numpy vectors
        :param x_train: train dataset (numpy array)
        :param y_train: labels for train dataset (numpy vector)
        :param x_test: test dataset (numpy array)
        :param y_test: labels for test dataset (numpy vector)
        :return: best chromosome found (numpy vector), best value found for that chromsome (float), training history
        (list of fitness values)
        """

        assert len(population) >= 2, "number of individuals in population should be >= 2"
        if self.genetic_train:
            assert self.epochs > 0, "number of epochs must be positive for genetic_train = True"
            history_bp = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=self.callbacks,
                                        epochs=self.epochs)
            original = self.weights_to_vector_alt()
            new_population = list()
            new_population.append(original)
            for i in range(len(population)-1):
                if random.random() <= self.original_mutation_prob:
                    new_population += self.generate_population(1, distribution='uniform')
                else:
                    new_population.append(self.mutate(original, self.mutation_dist))
            population = new_population

        self.max_pop_length = len(population)
        self.ranking = [(x, self.fitness(x, x_train, y_train)) for x in population]
        best, best_value, history = self.GA(population, self.iterations, x_train, y_train, x_test, y_test)
        return best, best_value, history, history_bp

    def GA(self, population, iterations, x, y, x_test, y_test):
        """

        :param population: initial population. List of numpy vectors
        :param iterations: integer. Number of generations
        :param x: train dataset (numpy array)
        :param y: labels for train dataset (numpy vector)
        :param x_test: test dataset (numpy array)
        :param y_test: labels for test dataset (numpy vector)
        :return: tuple (best individual found as numpy vector, max value found, history of bests)
        """
        self.best_of_generation(self.fitness, population, x, y)
        history = [0]
        all_the_bests = [self.model.get_weights()]

        for i in range(iterations):
            distribution = [1 / float(x + 1) for x in range(1, len(population) + 1)]
            print("Generation {}".format(i))
            new_population = []
            for _ in population:
                if len(new_population) < self.max_pop_length:
                    # parent_one, parent_two = self.random_selection(self.ranking)
                    parent_one, parent_two = self.random_selection(list(map(lambda x: x[0], self.ranking)), distribution)
                    c_a, c_b = self.reproduce(parent_one, parent_two)
                    new_population.append(self.mutate(c_a, self.mutation_dist))
                    new_population.append(self.mutate(c_b, self.mutation_dist))

            population = list()
            for item in range(self.top_best_preserved):
                population.append(self.ranking[item][0])

            population += new_population

            best, best_value = self.best_of_generation(self.fitness, population, x, y)
            history.append(best_value)
            all_the_bests.append(best)

        best_index = history.index(max(history[1:]))
        best = all_the_bests[best_index]
        best_value = history[best_index]
        self.fitness(best, x_test, y_test)
        return best, best_value, history

    def best_of_generation(self, fitness, population, x, y):
        """

        :param fitness: fitness function
        :param eq: equation to maximize
        :param population: list of individuals (numpy array)
        :param x: train dataset (numpy array)
        :param y: labels for train dataset (numpy vector)
        :return: tuple of numpy vector best individual and best value
        """
        best = None
        best_value = -9999999

        for i, p in enumerate(population[:self.top_best_preserved]):
            current_value = self.ranking[i][1]
            if current_value > best_value:
                best = p
                best_value = current_value

        self.ranking = self.ranking[:self.top_best_preserved]
        print("Best value: {} --- {}".format(best_value, list(map(lambda x: round(x[1], 3), self.ranking))))
        for p in population[self.top_best_preserved:]:
            current_value = fitness(p, x, y)
            self.ranking.append((p, current_value))
            if current_value > best_value:
                best = p
                best_value = current_value

        self.ranking.sort(key=lambda x : x[1], reverse=True)
        return best, best_value

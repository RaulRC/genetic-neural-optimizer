# Genetic Neural Optimizer

A Keras extension to use Genetic Algorithms for the optimization of Neural Network Trainings. 

## Use cases

Let's use a basic example with MNIST dataset. 

````python

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

````

```python

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

model = generate_model()
```

### 1. Genetic Training

Train a Neural Network using a Genetic Algoritm instead of Backpropagation. 

First of all, declare ```GeneticNeuralOptimizer``` object with the parameters needed. 

````python

MUT_PROB = 0.9
GENERATIONS = 100
MUTABLES = 1
ELITE = 0.2
POPULATION_LENGTH = 30
EPOCHS = 2

ga = GeneticNeuralOptimizer(model,
                            mutation_prob=MUT_PROB,
                            iterations=GENERATIONS,
                            mutables=MUTABLES,
                            elite=ELITE)
                            
pop = ga.generate_population(POPULATION_LENGTH)

````

Execute the Genetic Algorithm fit. 

````python
best, best_value, history = ga.fit(pop, x_train, y_train, x_test, y_test)

````

### 2. Genetic Weights Initialization

Initialize Weights of a Neural Network using Genetic Algorithms. 

```python

ga = GeneticNeuralWeightOptimizer(model,
                                  mutation_prob=MUT_PROB,
                                  iterations=GENERATIONS,
                                  mutables=MUTABLES,
                                  elite=ELITE,
                                  epochs=EPOCHS
                            )
pop = ga.generate_population(POPULATION_LENGTH)
best, best_value, history_ga, history_bp = ga.fit(pop, x_train, y_train, x_test, y_test)

```

### 3. Genetic Regularization

Regularize a Neural Network using Genetic Algorithms. The network will start training using the regular Backpropagation and at a certain moment, the training conmutes to Genetic Algoritm. Regularization may occur in two scenarios:

- A defined callback stops the execution due to overfitting detection, or
- Once the training has finished, the Genetic Algoritms continues from that point


```python

cb = [EarlyStopping(monitor='val_acc', min_delta=0.1)]

ga = GeneticRegularizator(model,
                          mutation_prob=MUT_PROB,
                          iterations=GENERATIONS,
                          mutables=MUTABLES,
                          elite=ELITE,
                          epochs=EPOCHS,
                          callbacks=cb
                          )
pop = ga.generate_population(POPULATION_LENGTH)
best, best_value, history_ga, history_bp = ga.fit(pop, x_train, y_train, x_test, y_test) 
```

## Tests

Some tests have been defined in order to prove the library. 

Run all tests: 

```bash
sh run_tests.sh 
```

Run basic tests only (Glass, Iris, Titnaic, and Wine datasets): 

```bash
sh basic_tests.sh 
```

Titanic, MNIST and Fashion-MNIST tests only: 

```bash
sh mnist_tests.sh
sh fashion_tests.sh
```

All the outputs will be generated in ```outputs/``` folder. 

Under ```src/``` folder the main library is implemented: ```genetic_optimizer.py```. Tests and configuration files for them are also included. 


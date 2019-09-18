#!/usr/bin/env bash

echo "Running MNIST tests..."
set -x

rm -rf outputs/mnist_*

mkdir outputs
mkdir outputs/mnist_init_weights
mkdir outputs/mnist_regularization
mkdir outputs/mnist_train

python3 src/mnist_init_weights.py
python3 src/mnist_regularization.py
python3 src/mnist_train.py
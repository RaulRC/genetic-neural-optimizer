#!/usr/bin/env bash

echo "Running Fashion MNIST tests..."
set -x

rm -rf outputs/fashion_*

mkdir outputs
mkdir outputs/fashion_init_weights
mkdir outputs/fashion_regularization
mkdir outputs/fashion_train

python3 src/fashion_init_weights.py
python3 src/fashion_regularization.py
python3 src/fashion_train.py
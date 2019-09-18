#!/usr/bin/env bash

echo "Running basic tests"
set -x

rm -rf outputs/glass_*
rm -rf outputs/iris_*
rm -rf outputs/titanic_*
rm -rf outputs/wine_*

mkdir outputs
mkdir outputs/glass_init_weights
mkdir outputs/glass_regularization
mkdir outputs/glass_train

mkdir outputs/iris_init_weights
mkdir outputs/iris_regularization
mkdir outputs/iris_train

mkdir outputs/titanic_init_weights
mkdir outputs/titanic_regularization
mkdir outputs/titanic_train

mkdir outputs/wine_init_weights
mkdir outputs/wine_regularization
mkdir outputs/wine_train

python3 src/glass_init_weights.py
python3 src/glass_regularization.py
python3 src/glass_train.py

python3 src/iris_init_weights.py
python3 src/iris_regularization.py
python3 src/iris_train.py

python3 src/titanic_init_weights.py
python3 src/titanic_regularization.py
python3 src/titanic_train.py

python3 src/wine_init_weights.py
python3 src/wine_regularization.py
python3 src/wine_train.py
#!/usr/bin/env bash

echo "RUN ALL TESTS"
set -x

rm -rf outputs
mkdir outputs

sh basic_tests.sh
sh mnist_tests.sh
sh fashion_mnist_tests.sh
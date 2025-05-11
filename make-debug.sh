#!/bin/bash

set -e
set -x

mkdir build-debug -p
cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make
cp ./benchmark ..

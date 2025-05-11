#!/bin/bash

set -e
set -x

mkdir build-release -p
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
cp ./benchmark ..

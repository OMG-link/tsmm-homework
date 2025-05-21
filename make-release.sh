#!/bin/bash

set -e
set -x

mkdir build-release -p
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++)
make

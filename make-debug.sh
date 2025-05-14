#!/bin/bash

set -e
set -x

mkdir build-debug -p
cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++)
make

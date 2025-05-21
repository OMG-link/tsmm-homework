#!/bin/bash

set -e
set -x

./make-debug.sh > /dev/null 2> /dev/null
cd build-debug
ctest -R "func"

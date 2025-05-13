#!/bin/bash

set -e
set -x

./make-release.sh > /dev/null 2> /dev/null
cd build-release
ctest -R "perf"

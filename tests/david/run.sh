#!/bin/bash
ver=$1
cd "$(dirname "$0")" || exit 1
cd ../../


cp original/*.py bin/
cp tests/david/ex3_check.py bin/
cp my/versions/${ver} bin/ex3.py



# run

cd bin
python3 ex3_check.py
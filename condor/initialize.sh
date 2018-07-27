#!/bin/bash

echo "Untarring Code and Python"
tar xzf code.tar.gz
tar xzf miniconda.tar.gz

echo "Setting PATH"
export PATH=$(pwd)/miniconda2/bin:$PATH

echo "Fixing Paths"
grep -rl '/var/lib/condor/execute/slot1/dir_1498242' miniconda2/bin/ | xargs sed -i "s,/var/lib/condor/execute/slot1/dir_1498242,$_CONDOR_SCRATCH_DIR,"
grep -rl '/var/lib/condor/execute/slot1/dir_1497179' miniconda2/bin/ | xargs sed -i "s,/var/lib/condor/execute/slot1/dir_1497179,$_CONDOR_SCRATCH_DIR,"
grep -rl '/var/lib/condor/execute/slot1/dir_1498242' miniconda2/envs/keras | xargs sed -i "s,/var/lib/condor/execute/slot1/dir_1498242,$_CONDOR_SCRATCH_DIR,"
grep -rl '/var/lib/condor/execute/slot1/dir_1497179' miniconda2/envs/keras | xargs sed -i "s,/var/lib/condor/execute/slot1/dir_1497179,$_CONDOR_SCRATCH_DIR,"

echo "Activating Keras Environment"
source activate keras

echo "Running Neural Net with input $1"
python main.py -c $1

echo "Deactivating Keras Environment"
source deactivate

echo "Cleanup"
rm __init__.py datasets.py docopt.py graph.py main.py models.py output.py shapes.py poly.py

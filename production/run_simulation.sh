#!/bin/bash
set +e  # stop if error

CATALOG_DIR=$1
SHARD_DIR=$2
EXPERIMENT_DIR=$3 
TEST=$4  # expects --test or blank
PANOPTES=$5  # expects --panoptes or blank

N_ITERATIONS = $6  # new

echo 'Creating experiment and instructions:'
./create_instruction.sh CATALOG_DIR SHARD_DIR EXPERIMENT_DIR TEST PANOPTES

THIS_ITERATION=0
while [ $THIS_ITERATION -lt $N_ITERATIONS ]
do  
    echo 'Running iteration:' $THIS_ITERATION_DIR
    ./run_iteration.sh $EXPERIMENT_DIR $INSTRUCTIONS_DIR $PREVIOUS_ITERATION $THIS_ITERATION

    PREVIOUS_ITERATION=$THIS_ITERATION
    THIS_ITERATION=$[$THIS_ITERATION+1]
done

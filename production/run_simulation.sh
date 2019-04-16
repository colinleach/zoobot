#!/bin/bash
set +e  # stop if error

CATALOG_DIR=$1
SHARD_DIR=$2
EXPERIMENT_DIR=$3 
TEST=$4  # expects --test or blank
PANOPTES=$5  # expects --panoptes or blank

SHARD_CONFIG=$SHARD_DIR'/shard_config.json'
INSTRUCTIONS_DIR=$EXPERIMENT_DIR/instructions

N_ITERATIONS=5

echo 'base directory: ' $EXPERIMENT_DIR
echo 'shard configuration json: ' $SHARD_CONFIG
echo 'instructions for each iteration: ' $INSTRUCTIONS_DIR 
echo --

mkdir $EXPERIMENT_DIR
mkdir $INSTRUCTIONS_DIR

python zoobot/active_learning/create_instructions.py  --catalog-dir=$CATALOG_DIR --shard-config=$SHARD_CONFIG --instructions-dir=$INSTRUCTIONS_DIR --baseline --warm-start $TEST $PANOPTES
RESULT=$?
if [ $RESULT -gt 0 ]
then
    echo "Failure!" $RESULT 
    exit 1
fi
echo "Instructions succesfully created at $INSTRUCTIONS_DIR"

THIS_ITERATION=0

while [ $THIS_ITERATION -lt $N_ITERATIONS ]
do  
    THIS_ITERATION_DIR=$EXPERIMENT_DIR'/iteration_'$THIS_ITERATION

    if [ $THIS_ITERATION -eq "0" ]
    then
        PREVIOUS_ITERATION_DIR=""
    else
        PREVIOUS_ITERATION_DIR=$EXPERIMENT_DIR'/iteration_'$PREVIOUS_ITERATION

    fi

    python zoobot/active_learning/run_iteration.py --instructions-dir=$INSTRUCTIONS_DIR --this-iteration-dir=$THIS_ITERATION_DIR --previous-iteration-dir=$PREVIOUS_ITERATION_DIR --test
    # stop if any error
    RESULT=$?
    echo $RESULT $RESULT $RESULT $RESULT $RESULT
    if [ $RESULT -gt 0 ]
    then
        echo "Failure!" $RESULT 
        exit 1
    fi

    echo $THIS_ITERATION_DIR $NEXT_ITERATION_DIR

    PREVIOUS_ITERATION=$THIS_ITERATION
    THIS_ITERATION=$[$THIS_ITERATION+1]
done

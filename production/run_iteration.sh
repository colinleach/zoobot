#!/bin/bash
set +e  # stop if error

EXPERIMENT_DIR=$1
INSTRUCTIONS_DIR=$2
PREVIOUS_ITERATION$3
THIS_ITERATION=$4
TEST=$5

THIS_ITERATION_DIR=$EXPERIMENT_DIR'/iteration_'$THIS_ITERATION

    if [ $THIS_ITERATION -eq "0" ]
    then
        PREVIOUS_ITERATION_DIR=""
    else
        PREVIOUS_ITERATION_DIR=$EXPERIMENT_DIR'/iteration_'$PREVIOUS_ITERATION

    fi

    python zoobot/active_learning/run_iteration.py --instructions-dir=$INSTRUCTIONS_DIR --this-iteration-dir=$THIS_ITERATION_DIR --previous-iteration-dir=$PREVIOUS_ITERATION_DIR $TEST
    # stop if any error
    RESULT=$?
    if [ $RESULT -gt 0 ]
    then
        echo "Failure!" $RESULT 
        exit 1
    fi
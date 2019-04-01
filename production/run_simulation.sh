#!/bin/bash
set +e  # stop if error

BASE_DIR='/Volumes/alpha/production_prototype'

SHARD_CONFIG='/Volumes/alpha/for_philipp/gz2_uint8_256px_smooth_20ktrain_2keval_30ktotal/shard_config.json'
INSTRUCTIONS_DIR=$BASE_DIR'/instructions'

N_ITERATIONS=5

echo 'base directory: ' $BASE_DIR
echo 'shard configuration json: ' $SHARD_CONFIG
echo 'instructions for each iteration: ' $INSTRUCTIONS_DIR 
echo --

python zoobot/active_learning/create_instructions.py --shard-config=$SHARD_CONFIG --instructions-dir=$INSTRUCTIONS_DIR --baseline --warm-start --test
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
    THIS_ITERATION_DIR=$BASE_DIR'/iteration_'$THIS_ITERATION

    if [ $THIS_ITERATION -eq "0" ]
    then
        PREVIOUS_ITERATION_DIR=""
    else
        PREVIOUS_ITERATION_DIR=$BASE_DIR'/iteration_'$PREVIOUS_ITERATION

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

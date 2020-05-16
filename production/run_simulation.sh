#!/bin/bash
set +e  # stop if error

n_iterations=$1
catalog_dir=$2
shard_dir=$3
experiment_dir=$4 
options=$5

echo 'N iterations: ' $n_iterations 'Catalog dir' $catalog_dir 'Shard dir' $shard_dir 'Experiment dir' 
echo 'Options: ' $options

echo 'Creating experiment and instructions:'
./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir $options
instructions_dir=$experiment_dir/instructions

# first iteration, 0 and blank
this_iteration=0
previous_iteration_dir='None'  # exactly blank doesn't count as an argument
this_iteration_dir=$experiment_dir/iteration_$this_iteration

while [ $this_iteration -lt $n_iterations ]
do  
    echo 'Running iteration:' $this_iteration

    ./production/run_iteration.sh $instructions_dir $this_iteration_dir $previous_iteration_dir $options

    # update ahead of next iteration
    this_iteration=$[$this_iteration+1]
    previous_iteration_dir=$experiment_dir/iteration_$(($this_iteration - 1))
    this_iteration_dir=$experiment_dir/iteration_$this_iteration


done

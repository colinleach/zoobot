#!/bin/bash
set +e  # stop if error

n_iterations=$1
catalog_dir=$2
shard_dir=$3
experiment_dir=$4 
baseline=$5
test=$6  # expects --test or blank
panoptes=$7  # expects --panoptes or blank

echo '\nN iterations: ' $n_iterations 'Catalog dir' $catalog_dir 'Shard dir' $shard_dir 'Experiment dir' $experiment_dir 'Baseline?' $baseline 'test?' $test 'panoptes?' $panoptes

echo 'Creating experiment and instructions:'
./production/create_instructions.sh $catalog_dir $shard_dir $experiment_dir $baseline $test $panoptes
instructions_dir=$experiment_dir/instructions

# first iteration, 0 and blank
this_iteration=0
previous_iteration_dir='None'  # exactly blank doesn't count as an argument
this_iteration_dir=$experiment_dir/iteration_$this_iteration

while [ $this_iteration -lt $n_iterations ]
do  
    echo 'Running iteration:' $this_iteration

    ./production/run_iteration.sh $instructions_dir $this_iteration_dir $previous_iteration_dir $test

    # update ahead of next iteration
    this_iteration=$[$this_iteration+1]
    previous_iteration_dir=$experiment_dir/iteration_$(($this_iteration - 1))
    this_iteration_dir=$experiment_dir/iteration_$this_iteration


done

#!/usr/bin/env bash

task_list=( qm9 )
running_index_list=(0 1 2)

for task in "${task_list[@]}"; do
    for running_index in "${running_index_list[@]}"; do
        mkdir -p ./output/"$task"/"$running_index"
        mkdir -p ./model_weight/"$task"/"$running_index"
	echo $task
        python node_embedding.py \
        --mode="$task" \
	--epochs="30"
        --running_index="$running_index" > ./output/"$task"/"$running_index"/node_embedding.out

        mkdir -p ../../datasets/"$task"/"$running_index"

        python graph_embedding.py \
        --mode="$task" \
	--epochs="30"
        --running_index="$running_index" > ./output/"$task"/"$running_index"/graph_embedding.out
    done
done


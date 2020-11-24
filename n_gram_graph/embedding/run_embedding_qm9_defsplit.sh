#!/usr/bin/env bash

task_list=( qm9 )
seed_list=(0 1 2 3 4)

for task in "${task_list[@]}"; do
    for seed in "${seed_list[@]}"; do

    	mkdir -p ./output/"$task"/"defsplit-seed$seed"
        mkdir -p ./model_weight/"$task"/"defsplit-seed$seed"

	echo "$task - $seed"

	python node_embedding_defsplit.py \
        --mode="$task" \
	--epochs="30" \
        --seed="$seed" > ./output/"$task"/"defsplit-seed$seed"/node_embedding.out

        mkdir -p ../../datasets/"$task"/"defsplit-seed$seed"

        python graph_embedding_defsplit.py \
        --mode="$task" \
        --seed="$seed" > ./output/"$task"/"defsplit-seed$seed"/graph_embedding.out

    done
done


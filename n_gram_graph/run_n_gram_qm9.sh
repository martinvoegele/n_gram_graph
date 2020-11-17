#!/usr/bin/env bash

task_list=( mu alpha homo lumo gap r2 zpve cv u0 u298 h298 g298 u0_atom u298_atom h298_atom g298_atom )
running_index_list=(0 1 2)
model_list=(n_gram_rf n_gram_xgb)

for task in "${task_list[@]}"; do
    for model in "${model_list[@]}"; do
        for running_index in "${running_index_list[@]}"; do
            mkdir -p ../output/"$model"/"$running_index"

            python main_regression.py \
            --task="$task" \
            --config_json_file=../hyper/"$model"/"$task".json \
            --weight_file=temp.pt \
            --running_index="$running_index" \
            --model="$model" > ../output/"$model"/"$running_index"/"$task".out
        done
    done
done

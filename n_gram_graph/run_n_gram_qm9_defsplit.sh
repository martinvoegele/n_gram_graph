#!/usr/bin/env bash

task_list=( u0_atom u298_atom h298_atom g298_atom mu alpha homo lumo gap r2 zpve cv u0 u298 h298 g298 )
running_index_list=(0 1 2 3 4)
model_list=(n_gram_xgb)

for task in "${task_list[@]}"; do
    for model in "${model_list[@]}"; do
        for running_index in "${running_index_list[@]}"; do
            mkdir -p ../output/"$model"/"defsplit-seed$running_index"

            python main_regression_defsplit.py \
            --task="$task" \
            --config_json_file=../hyper/"$model"/"$task".json \
            --weight_file=temp.pt \
            --running_index="$running_index" \
            --model="$model" > ../output/"$model"/"defsplit-seed$running_index"/"$task".out
        done
    done
done

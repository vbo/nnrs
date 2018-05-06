#!/usr/bin/env bash

id=`date "+%Y_%m_%d_%H_%M_%S"`
iter_num=1000
sessions_per_iter=100000
examples_per_iter=$((10*$sessions_per_iter))
num_epochs=100
rust_execute="time env RUSTFLAGS=-Awarnings RUST_BACKTRACE=1 cargo run --release --"

echo "Creating new model ${id}"
$rust_execute snake_new --model_output "models/${id}.json"

for (( i = 0; i <= $iter_num; i++ ))
do
    echo "Generating training data ${i}"
    $rust_execute snake_gen \
            --model_input "models/${id}.json" \
            --training_data "data/snake_t_${id}.dat" \
            --save="${sessions_per_iter}"
    echo "Teaching model"
    $rust_execute snake_train \
        --model_input "models/${id}.json" \
        --training_data "data/snake_t_${id}.dat" \
        --model_output "models/${id}t.json" \
        --training_data_max="${examples_per_iter}" \
        --num_epochs="${num_epochs}" \
        --write_every="${examples_per_iter}" \
        --log_every_n="${examples_per_iter}"
    mv "models/${id}t.json" "models/${id}.json"
    echo "done"
done

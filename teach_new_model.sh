#!/usr/bin/env bash
set -e

id=`date "+%Y_%m_%d_%H_%M_%S"`
iter_num=100
data_per_iter=100000
num_epochs=100
rust_execute="time env RUSTFLAGS=-Awarnings RUST_BACKTRACE=1 cargo run --release --"

echo "Creating new model ${id}"
$rust_execute snake_new --model_output "models/${id}.json"

for i in {0..$num_epochs}
do
    echo "Generating training data ${i}"
    $rust_execute snake_gen \
            --model_input "models/${id}.json" \
            --training_data "data/snake_t_${id}.dat" \
            --save="${data_per_iter}"
    echo "Teaching model"
    $rust_execute snake_train \
        --model_input "models/${id}.json" \
        --training_data "data/snake_t_${id}.dat" \
        --model_output "models/${id}t.json" \
        --training_data_max="${data_per_iter}" \
        --num_epochs="${num_epochs}" \
        --log_every_n=1000000
    mv "models/${id}t.json" "models/${id}.json"
done

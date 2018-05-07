#!/usr/bin/env bash

input_id=$1
iter_num=1000
sessions_per_iter=10000
examples_per_iter=$((10*$sessions_per_iter))
num_epochs=100
rust_execute="time env RUSTFLAGS=-Awarnings RUST_BACKTRACE=1 cargo run --release --"

if [ -z "$input_id" ]
then
    id=`date "+%Y_%m_%d_%H_%M_%S"`
    echo "Creating new model ${id}"
    $rust_execute snake_new --model_output "models/${id}.json"
else
    id="$input_id"
fi

training_data="data/snake_t_${id}.dat"

trap "exit" INT
for (( i = 0; i <= $iter_num; i++ ))
do
    echo "Generating training data ${i}"
    $rust_execute snake_gen \
            --model_input "models/${id}.json" \
            --training_data "data/snake_t_${id}.dat" \
            --save="${sessions_per_iter}"
    echo "Lines before dedup: `cat ${training_data} | wc -l`"
    sort -u "${training_data}" -o "${training_data}"
    echo "Lines after dedup: `cat ${training_data} | wc -l`"
    echo "Teaching model"
    $rust_execute snake_train \
        --model_input "models/${id}.json" \
        --training_data "${training_data}" \
        --model_output "models/${id}t.json" \
        --training_data_max="${examples_per_iter}" \
        --num_epochs="${num_epochs}" \
        --write_every="${examples_per_iter}" \
        --log_every_n="${examples_per_iter}"
    mv "models/${id}t.json" "models/${id}.json"
    echo "done"
done

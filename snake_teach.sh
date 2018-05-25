#!/usr/bin/env bash

input_id=$1
iter_num=1000
sessions_per_iter=10000
write_every_n=1000
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

training_data="data/snake_${id}.dat"

if [ -z "$input_id" ]
then
    touch "${training_data}"
fi

trap "exit" INT
for (( i = 0; i <= $iter_num; i++ ))
do
    echo "Generating training data ${i}"
    $rust_execute snake_gen \
            --model_input "models/${id}.json" \
            --training_data "data/snake_t_${id}.dat" \
            --save="${sessions_per_iter}"
    cat "data/snake_t_${id}.dat_"* >> "${training_data}"
    echo "Lines before dedup: `cat ${training_data} | wc -l`"
    sort -u "${training_data}" -o "${training_data}"
    lines=$(cat "${training_data}" | wc -l)
    echo "Lines after dedup: ${lines}"
    echo "Teaching model"
    $rust_execute snake_train \
        --model_input "models/${id}.json" \
        --training_data "${training_data}" \
        --model_output "models/${id}t.json" \
        --training_data_max="${lines}" \
        --num_epochs="${num_epochs}" \
        --write_every="${write_every_n}" \
        --log_every_n="${write_every_n}"
    mv "models/${id}t.json" "models/${id}.json"
    echo "done"
done

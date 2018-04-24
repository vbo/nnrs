extern crate rand;

use std::fmt;

use rand::Rng;
use rand::distributions::{IndependentSample, Range};

mod math;
use math::Matrix;
use math::Vector;

mod mnist_data;
use mnist_data::Dataset;

mod network;
use network::Network;

fn calc_weights_pd(error_grad_prefix: &Vector, previous_activations: &Vector, res: &mut Matrix) {
    assert!(
        error_grad_prefix.rows == res.rows && res.cols == previous_activations.rows,
        "Invalid dimentions for output weights PD: {}x1, {}x1, {}x{}",
        error_grad_prefix.rows,
        previous_activations.rows,
        res.rows,
        res.cols
    );

    for row in 0..res.rows {
        let row_start = row * res.cols;
        for col in 0..res.cols {
            res.mem[row_start + col] = error_grad_prefix.mem[row] * previous_activations.mem[col];
        }
    }
}

fn calc_grad_prefix(activations: &Vector, error: &Vector, res: &mut Vector) {
    assert!(
        activations.rows == error.rows && error.rows == res.rows,
        "Invalid dimentions for output error grad prefix PD: {}x1, {}x1, {}x1",
        activations.rows,
        error.rows,
        res.rows
    );

    for row in 0..activations.rows {
        res.mem[row] = 2.0 * error.mem[row] * activations.mem[row] * (1.0 - activations.mem[row]);
    }
}

fn render_training_example(label_data: &[f64], input_data: &[f64]) {
    for i in 0..10 {
        if label_data[i] > 0.0 {
            println!("Label: {}", i);
        }
    }

    for i in 0..28 {
        let row_start = i * 28;
        for j in 0..28 {
            let v = input_data[row_start + j];
            if v > 0.7 {
                print!("1");
            } else {
                print!("0");
            }
        }
        print!("\n");
    }
}

// Sizes
const N_INPUTS: usize = 28 * 28;
const N_L1: usize = 16;
const N_L2: usize = 16;
const N_OUTPUTS: usize = 10;

const LOG_EVERY_N: usize = 10_000;
const LEARNING_RATE: f64 = 0.1;
const NUM_EPOCHS: usize = 1000;

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn main() {
    let mut nn = Network::new(N_INPUTS, N_OUTPUTS);
    let inputs_id = nn.input_layer();
    let l1_id = nn.add_hidden_layer(N_L1);
    let l2_id = nn.add_hidden_layer(N_L2);
    let outputs_id = nn.output_layer();
    nn.add_layer_dependency(outputs_id, l2_id);
    nn.add_layer_dependency(l2_id, l1_id);
    nn.add_layer_dependency(l1_id, inputs_id);

    let mut training_data = mnist_data::load_mnist_training();
    assert!(training_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(training_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let mut hits = 0usize;
    let mut random_number_generator = rand::thread_rng();
    let mut examples_processed = 0usize;
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut avg_error: f64 = 0.0;
    for current_epoch in 1..NUM_EPOCHS + 1 {
        // Randomize example order
        random_number_generator.shuffle(&mut training_data.example_indices.as_mut_slice());

        let mut current_examples_cursor = 0usize;
        while current_examples_cursor < training_data.examples_count {
            let (input_data, label_data) = training_data.slices_for_cursor(current_examples_cursor);
            true_outputs.copy_from_slice(label_data);
            let outputs = nn.predict(input_data).clone();
            nn.backward_propagation(&true_outputs);

            // Update accuracy metrics
            true_outputs.sub(&outputs, &mut error);
            avg_error += error.calc_magnitude();
            let (max_i, max) = outputs.max_component();
            let (tmax_i, tmax) = true_outputs.max_component();
            if max_i == tmax_i {
                hits += 1;
            }

            if (current_examples_cursor + 1) % network::BATCH_SIZE == 0 {
                nn.apply_batch();
            }

            if examples_processed % LOG_EVERY_N == 0 {
                println!(
                    "error over last {}: {:8.4}",
                    LOG_EVERY_N,
                    avg_error / LOG_EVERY_N as f64
                );
                println!("hits {}%", (hits as f64) * 100.0 / (LOG_EVERY_N as f64));
                avg_error = 0.0;
                hits = 0;

                if examples_processed % (LOG_EVERY_N * 10) == 0 {
                    println!("True: {}, Outputs: {}", true_outputs, outputs);
                }
            }

            examples_processed += 1;
            current_examples_cursor += 1;
        }
    }
}

fn old_main() {
    let mut training_data = mnist_data::load_mnist_training();

    assert!(training_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(training_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let mut inputs = Vector::new(N_INPUTS).init_with(1.0);

    let mut l1_weights = Matrix::new(N_L1, N_INPUTS).init_rand();
    let mut l1_weights_pd = Matrix::new(N_L1, N_INPUTS).init_with(0.0);
    let mut l1_weights_batch_pd = Matrix::new(N_L1, N_INPUTS).init_with(0.0);
    let mut l1_grad_prefix = Vector::new(N_L1).init_with(0.0);

    let mut l1_bias = 0.0f64;
    let mut l1_bias_batch_pd = 0.0f64;
    let mut l1_activations = Vector::new(N_L1).init_with(0.0);

    let mut l1_error = Vector::new(N_L1).init_with(0.0);

    let mut l2_weights = Matrix::new(N_L2, N_L1).init_rand();
    let mut l2_weights_pd = Matrix::new(N_L2, N_L1).init_with(0.0);
    let mut l2_weights_batch_pd = Matrix::new(N_L2, N_L1).init_with(0.0);
    let mut l2_weights_t = Matrix::new(N_L1, N_L2).init_with(0.0);
    let mut l2_grad_prefix = Vector::new(N_L2).init_with(0.0);

    let mut l2_bias = 0.0f64;
    let mut l2_bias_batch_pd = 0.0f64;
    let mut l2_activations = Vector::new(N_L2).init_with(0.0);

    let mut l2_error = Vector::new(N_L2).init_with(0.0);

    let mut output_weights = Matrix::new(N_OUTPUTS, N_L2).init_rand();
    let mut output_weights_t = Matrix::new(N_L2, N_OUTPUTS).init_with(0.0);
    let mut output_weights_pd = Matrix::new(N_OUTPUTS, N_L2).init_with(0.0);
    let mut output_weights_batch_pd = Matrix::new(N_OUTPUTS, N_L2).init_with(0.0);
    let mut output_grad_prefix = Vector::new(N_OUTPUTS).init_with(0.0);

    let mut output_bias = 0.0f64;
    let mut output_bias_batch_pd = 0.0f64;
    let mut outputs = Vector::new(N_OUTPUTS).init_with(0.0);

    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut avg_error: f64 = 0.0;

    let avg_by_batch = |x| x * LEARNING_RATE / network::BATCH_SIZE as f64;
    let set_0 = |_: f64| 0.0;

    let mut hits = 0usize;
    let mut random_number_generator = rand::thread_rng();
    let mut examples_processed = 0usize;
    for current_epoch in 1..NUM_EPOCHS + 1 {
        // Randomize example order
        random_number_generator.shuffle(&mut training_data.example_indices.as_mut_slice());

        let mut current_examples_cursor = 0usize;
        while current_examples_cursor < training_data.examples_count {
            // Load current example
            let current_example_index = training_data.example_indices[current_examples_cursor];
            let input_data_offset = current_example_index * training_data.input_size;
            let input_data_end = input_data_offset + training_data.input_size;
            let input_data = &training_data.input_mem[input_data_offset..input_data_end];

            let label_data_offset = current_example_index * training_data.label_size;
            let label_data_end = label_data_offset + training_data.label_size;
            let label_data = &training_data.label_mem[label_data_offset..label_data_end];
            inputs.copy_from_slice(input_data);
            true_outputs.copy_from_slice(label_data);

            // Forward propagation
            l1_weights.dot_vec(&inputs, &mut l1_activations);
            l1_activations.apply(|x| x + l1_bias);
            l1_activations.apply(sigmoid);

            l2_weights.dot_vec(&l1_activations, &mut l2_activations);
            l2_activations.apply(|x| x + l2_bias);
            l2_activations.apply(sigmoid);

            output_weights.dot_vec(&l2_activations, &mut outputs);
            // outputs.add_to_me(&output_grad_prefix);
            outputs.apply(|x| x + output_bias);
            outputs.apply(sigmoid);

            // Error
            true_outputs.sub(&outputs, &mut error);
            avg_error += error.calc_magnitude();
            let (max_i, max) = outputs.max_component();
            let (tmax_i, tmax) = true_outputs.max_component();
            if max_i == tmax_i {
                hits += 1;
            }

            // Backward propagation
            // Output layer
            calc_grad_prefix(&outputs, &error, &mut output_grad_prefix);
            output_bias_batch_pd += output_grad_prefix.calc_sum();
            calc_weights_pd(&output_grad_prefix, &l2_activations, &mut output_weights_pd);
            output_weights_batch_pd.add(&output_weights_pd);
            output_weights.transpose(&mut output_weights_t);
            output_weights_t.dot_vec(&output_grad_prefix, &mut l2_error);

            // Hidden layer L2
            calc_grad_prefix(&l2_activations, &l2_error, &mut l2_grad_prefix);
            calc_weights_pd(&l2_grad_prefix, &l1_activations, &mut l2_weights_pd);
            l2_bias_batch_pd += l2_grad_prefix.calc_sum();
            l2_weights_batch_pd.add(&l2_weights_pd);
            l2_weights.transpose(&mut l2_weights_t);
            l2_weights_t.dot_vec(&l2_grad_prefix, &mut l1_error);

            // Hidden layer L1
            calc_grad_prefix(&l1_activations, &l1_error, &mut l1_grad_prefix);
            calc_weights_pd(&l1_grad_prefix, &inputs, &mut l1_weights_pd);
            l1_bias_batch_pd += l1_grad_prefix.calc_sum();
            l1_weights_batch_pd.add(&l1_weights_pd);

            if examples_processed % LOG_EVERY_N == 0 {
                println!(
                    "error over last {}: {:8.4}",
                    LOG_EVERY_N,
                    avg_error / LOG_EVERY_N as f64
                );
                println!("hits {}%", (hits as f64) * 100.0 / (LOG_EVERY_N as f64));
                avg_error = 0.0;
                hits = 0;

                if examples_processed % (LOG_EVERY_N * 10) == 0 {
                    //println!("l1_weights:{}", l1_weights);
                    //println!("l1_grad_prefix:{}", l1_grad_prefix);
                    //println!("l2_weights:{}", l2_weights);
                    //println!("l2_grad_prefix:{}", l2_grad_prefix);
                    //println!("output_weights:{}", output_weights);
                    //println!("output_grad_prefix:{}", output_grad_prefix);
                    println!("output:{}", outputs);
                }
            }

            if (current_examples_cursor + 1) % network::BATCH_SIZE == 0 {
                // Weights adjustment
                // Output
                output_bias += avg_by_batch(output_bias_batch_pd);
                output_bias_batch_pd = 0.0;

                output_weights_batch_pd.apply(&avg_by_batch);
                output_weights.add(&output_weights_batch_pd);
                output_weights_batch_pd.apply(&set_0);

                // L2
                l2_bias += avg_by_batch(l2_bias_batch_pd);
                l2_bias_batch_pd = 0.0;

                l2_weights_batch_pd.apply(&avg_by_batch);
                l2_weights.add(&l2_weights_batch_pd);
                l2_weights_batch_pd.apply(&set_0);

                // L1
                l1_bias += avg_by_batch(l1_bias_batch_pd);
                l1_bias_batch_pd = 0.0;

                l1_weights_batch_pd.apply(&avg_by_batch);
                l1_weights.add(&l1_weights_batch_pd);
                l1_weights_batch_pd.apply(&set_0);
            }

            examples_processed += 1;
            current_examples_cursor += 1;
        }
    }

    //println!("after l1_weights:{}", l1_weights);
    //println!("after l1_grad_prefix:{}", l1_grad_prefix);
    //println!("after l2_weights:{}", l2_weights);
    //println!("after l2_grad_prefix:{}", l2_grad_prefix);
    //println!("after output_weights:{}", output_weights);
    //println!("after output_grad_prefix:{}", output_grad_prefix);
    //println!("after error:{}", error);
}

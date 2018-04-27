extern crate rand;
extern crate serde;

#[macro_use]
extern crate serde_derive;
extern crate serde_json;

#[macro_use]
extern crate rand_derive;

use std::fmt;
use std::time;

use rand::Rng;

mod math;
use math::Matrix;
use math::Vector;

mod mnist_data;
use mnist_data::Dataset;

mod network;
use network::Network;

mod snake;

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

// TODO: parse command line arguments
const LOAD_FROM_FILE: bool = false;
const LOG_EVERY_N: usize = 10_000;
const TEST_EVERY_N: usize = 50_000;
const WRITE_EVERY_N: usize = 50_000;
const MODEL_OUTPUT_PATH: &str = "model.json";
const LEARNING_RATE: f64 = 0.1;
const NUM_EPOCHS: usize = 1000;
const NANOS_IN_SECOND: u64 = 1_000_000_000;

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn duration_as_total_secs(duration: &time::Duration) -> f64 {
    duration.as_secs() as f64 + (duration.subsec_nanos() as f64 / NANOS_IN_SECOND as f64)
}

fn main() {
    snake::main_snake_random_nn();
}

fn main_mnist() {
    let mut nn;
    if LOAD_FROM_FILE {
        nn = Network::load_from_file(MODEL_OUTPUT_PATH);
    } else {
        nn = Network::new(N_INPUTS, N_OUTPUTS);

        let inputs_id = nn.input_layer();
        let l1_id = nn.add_hidden_layer(N_L1);
        let l2_id = nn.add_hidden_layer(N_L2);
        let outputs_id = nn.output_layer();
        nn.add_layer_dependency(outputs_id, l2_id);
        nn.add_layer_dependency(l2_id, l1_id);
        nn.add_layer_dependency(l1_id, inputs_id);
    }

    let mut training_data = mnist_data::load_mnist_training();
    assert!(training_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(training_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let testing_data = mnist_data::load_mnist_testing();
    assert!(testing_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(testing_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let mut hits = 0usize;
    let mut random_number_generator = rand::thread_rng();
    let mut examples_processed = 0usize;
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut total_error: f64 = 0.0;
    let mut total_elapsed_secs: f64 = 0.0;
    let mut overall_stopwatch = time::Instant::now();
    for _ in 1..NUM_EPOCHS + 1 {
        // Randomize example order
        random_number_generator.shuffle(&mut training_data.example_indices.as_mut_slice());

        let mut current_examples_cursor = 0usize;
        while current_examples_cursor < training_data.examples_count {
            let (input_data, label_data) = training_data.slices_for_cursor(current_examples_cursor);
            true_outputs.copy_from_slice(label_data);

            let mut stopwatch = time::Instant::now();
            let outputs = nn.predict(input_data).clone();
            nn.backward_propagation(&true_outputs);
            total_elapsed_secs += duration_as_total_secs(&stopwatch.elapsed());

            // Update accuracy metrics
            true_outputs.sub(&outputs, &mut error);
            total_error += error.calc_magnitude();
            let (max_i, _) = outputs.max_component();
            let (tmax_i, _) = true_outputs.max_component();
            if max_i == tmax_i {
                hits += 1;
            }

            if (current_examples_cursor + 1) % network::BATCH_SIZE == 0 {
                nn.apply_batch();
            }

            if (examples_processed + 1) % LOG_EVERY_N == 0 {
                println!(
                    "error over last {}: {:8.4}",
                    LOG_EVERY_N,
                    total_error / LOG_EVERY_N as f64
                );
                println!("hits {}%", (hits as f64) * 100.0 / (LOG_EVERY_N as f64));
                println!(
                    "time/example: {:.4}ms",
                    1000.0 * total_elapsed_secs / LOG_EVERY_N as f64
                );
                println!(
                    "total time per {}k: {:.4}s",
                    LOG_EVERY_N / 1000,
                    duration_as_total_secs(&overall_stopwatch.elapsed())
                );
                total_error = 0.0;
                hits = 0;
                total_elapsed_secs = 0.0;
                overall_stopwatch = time::Instant::now();

                if examples_processed % (LOG_EVERY_N * 10) == 0 {
                    println!("True: {}, Outputs: {}", true_outputs, outputs);
                }
            }

            if examples_processed % TEST_EVERY_N == 0 {
                println!(
                    "Trained over {}k examples. Evaluation results: {}",
                    examples_processed / 1000,
                    evaluate(&mut nn, &testing_data)
                );
            }

            if examples_processed % WRITE_EVERY_N == 0 {
                nn.write_to_file(&MODEL_OUTPUT_PATH);
            }

            examples_processed += 1;
            current_examples_cursor += 1;
        }
    }
}

#[derive(Debug)]
pub struct EvaluationResult {
    hits_ratio: f64,
    avg_error: f64,
}

impl fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[hits: {:.2}%, avg_error: {:.4}]",
            self.hits_ratio * 100.0,
            self.avg_error
        )
    }
}

fn evaluate(predictor: &mut Network, test_dataset: &Dataset) -> EvaluationResult {
    let mut hits = 0usize;
    let mut total_error = 0.0f64;
    let mut current_examples_cursor = 0usize;
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    while current_examples_cursor < test_dataset.examples_count {
        let (input_data, label_data) = test_dataset.slices_for_cursor(current_examples_cursor);
        true_outputs.copy_from_slice(label_data);
        let outputs = predictor.predict(input_data).clone();
        true_outputs.sub(&outputs, &mut error);
        total_error += error.calc_magnitude();
        let (max_i, _) = outputs.max_component();
        let (tmax_i, _) = true_outputs.max_component();
        if max_i == tmax_i {
            hits += 1;
        }

        current_examples_cursor += 1;
    }

    EvaluationResult {
        hits_ratio: hits as f64 / test_dataset.examples_count as f64,
        avg_error: total_error / test_dataset.examples_count as f64,
    }
}

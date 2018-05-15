use rand;
use rand::Rng;
use std::fmt;
use std::time;

use network;
use network::Network;

use mnist_data;
use mnist_data::Dataset;

use math;
use math::Matrix;
use math::Vector;

use timing;
use timing::Timing;

const N_INPUTS: usize = 28 * 28;
const N_L1: usize = 16;
const N_L2: usize = 16;
const N_OUTPUTS: usize = 10;

const NUM_EPOCHS: usize = 1000;
const NANOS_IN_SECOND: u64 = 1_000_000_000;

pub fn main_mnist(
    model_input_path: &str,
    model_output_path: &str,
    write_every_n: usize,
    test_every_n: usize,
    log_every_n: usize,
) {
    let mut nn;
    if !model_input_path.is_empty() {
        nn = Network::load_from_file(model_input_path);
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
    let mut timing = Timing::new();
    for _ in 1..NUM_EPOCHS + 1 {
        // Randomize example order
        random_number_generator.shuffle(&mut training_data.example_indices.as_mut_slice());

        let mut current_examples_cursor = 0usize;
        while current_examples_cursor < training_data.examples_count {
            timing.start("overall");
            let (input_data, label_data) = training_data.slices_for_cursor(current_examples_cursor);
            true_outputs.copy_from_slice(label_data);

            timing.start("overall.predict");
            let outputs = nn.predict(input_data).clone();
            timing.stop("overall.predict");
            timing.start("overall.backward_propagation");
            nn.backward_propagation(&true_outputs);
            timing.stop("overall.backward_propagation");

            // Update accuracy metrics
            true_outputs.sub(&outputs, &mut error);
            total_error += error.calc_magnitude();
            let (max_i, _) = outputs.max_component();
            let (tmax_i, _) = true_outputs.max_component();
            if max_i == tmax_i {
                hits += 1;
            }

            if (current_examples_cursor + 1) % network::BATCH_SIZE == 0 {
                timing.start("overall.apply_batch");
                nn.apply_batch();
                timing.stop("overall.apply_batch");
            }

            timing.stop("overall");

            if (examples_processed + 1) % log_every_n == 0 {
                println!(
                    "error over last {}: {:8.4}",
                    log_every_n,
                    total_error / log_every_n as f64
                );
                println!("hits {}%", (hits as f64) * 100.0 / (log_every_n as f64));
                timing.dump_divided(examples_processed);
                total_error = 0.0;
                hits = 0;
                total_elapsed_secs = 0.0;
                overall_stopwatch = time::Instant::now();

                if examples_processed % (log_every_n * 10) == 0 {
                    println!("True: {}, Outputs: {}", true_outputs, outputs);
                }
            }

            if examples_processed % test_every_n == 0 {
                println!(
                    "Trained over {}k examples. Evaluation results: {}",
                    examples_processed / 1000,
                    evaluate(&mut nn, &testing_data)
                );
            }

            if !model_output_path.is_empty() && examples_processed % write_every_n == 0 {
                nn.write_to_file(&model_output_path);
            }

            examples_processed += 1;
            current_examples_cursor += 1;
        }
    }
}

fn duration_as_total_secs(duration: &time::Duration) -> f64 {
    duration.as_secs() as f64 + (duration.subsec_nanos() as f64 / NANOS_IN_SECOND as f64)
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

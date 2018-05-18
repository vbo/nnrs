use std::fmt;
use std::time;
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::channel;

use rand;
use rand::Rng;

use network;
use network::{Network, NetworkParameters, NetworkPredictor, NetworkTrainer};

use mnist_data;
use mnist_data::Dataset;

use math;
use math::Matrix;
use math::Vector;

const N_INPUTS: usize = 28 * 28;
const N_L1: usize = 16;
const N_L2: usize = 16;
const N_OUTPUTS: usize = 10;

const NUM_EPOCHS: usize = 1000;
const NANOS_IN_SECOND: u64 = 1_000_000_000;

const NUM_CPUS: usize = 4;

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

    let training_data = Arc::new(mnist_data::load_mnist_training());
    assert!(training_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(training_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let mut example_indices: Vec<_> = (0usize..training_data.examples_count).collect();

    let testing_data = mnist_data::load_mnist_testing();
    assert!(testing_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(testing_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let (mut nn_parameters, mut nn_predictor, mut nn_trainer) = nn.as_parts();
    let (job_sender, job_receiver) = channel();
    let (output_sender, output_receiver) = channel();
    let job_receiver = Arc::new(Mutex::new(job_receiver));
    let mut join_handles = Vec::new();
    for thread_no in 0..NUM_CPUS {
        let nn_predictor = nn_predictor.clone();
        let nn_trainer = nn_trainer.clone();
        let training_data = training_data.clone();
        let output_sender = output_sender.clone();
        let job_receiver = job_receiver.clone();
        let jh = thread::spawn(move || {
            loop {
                if let Ok((chunk, nn_params)) = job_receiver.lock().unwrap().recv() {
                    // do stuff
                    drop(nn_params);
                    output_sender.send(nn_trainer.clone());
                } else {
                    println!("Exiting child thread!");
                    break;
                }
            }
        });
        join_handles.push(jh);
    }
    let mut hits = 0usize;
    let mut random_number_generator = rand::thread_rng();
    let mut examples_processed = 0usize;
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut total_error: f64 = 0.0;
    let mut total_elapsed_secs: f64 = 0.0;
    let mut overall_stopwatch = time::Instant::now();
    let mut shared_nn_parameters = Arc::new(nn_parameters);
    for _ in 1..NUM_EPOCHS + 1 {
        // Randomize example order
        random_number_generator.shuffle(&mut example_indices.as_mut_slice());
        let indices_batches: Vec<_> = example_indices.chunks(network::BATCH_SIZE).map(Vec::from).collect();

        for indices_batch in &indices_batches {
            // TODO: check small last batch
            let batch_chunks: Vec<_> = indices_batch.chunks(indices_batch.len()/NUM_CPUS).map(Vec::from).collect();
            let batch_chunks_no = batch_chunks.len();
            for chunk in batch_chunks {
                job_sender.send((chunk, shared_nn_parameters.clone()));
            }

            let mut output_no = 0;
            for output in output_receiver.iter() {
                println!("Aggregate output");
                output_no += 1;
                if output_no == batch_chunks_no {
                    break;
                }
            }

            if let Some(nn_parameters) = Arc::get_mut(&mut shared_nn_parameters) {
                println!("Apply batch");
            } else {
                panic!("NN params should be freed after batch");
            }
        }

        /*
        let mut current_examples_cursor = 0usize;
        while current_examples_cursor < training_data.examples_count {
            let (input_data, label_data) = training_data.slices_for_cursor(current_examples_cursor);
            true_outputs.copy_from_slice(label_data);

            let mut stopwatch = time::Instant::now();
            let outputs = nn_predictor.predict(&nn_parameters, input_data).clone();
            nn_trainer.backward_propagation(&nn_parameters, &nn_predictor, &true_outputs);
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
                nn_trainer.apply_batch(&mut nn_parameters);
            }

            if (examples_processed + 1) % log_every_n == 0 {
                println!(
                    "error over last {}: {:8.4}",
                    log_every_n,
                    total_error / log_every_n as f64
                );
                println!("hits {}%", (hits as f64) * 100.0 / (log_every_n as f64));
                println!(
                    "time/example: {:.4}ms",
                    1000.0 * total_elapsed_secs / log_every_n as f64
                );
                println!(
                    "total time per {}k: {:.4}s",
                    log_every_n / 1000,
                    duration_as_total_secs(&overall_stopwatch.elapsed())
                );
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
                    evaluate(&nn_parameters, &mut nn_predictor, &testing_data)
                );
            }

            if !model_output_path.is_empty() && examples_processed % write_every_n == 0 {
                nn_parameters.write_to_file(&model_output_path);
            }

            examples_processed += 1;
            current_examples_cursor += 1;
        }
        */
    }

    println!("Main done!");
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

fn evaluate(nn_parameters: &NetworkParameters, nn_predictor: &mut NetworkPredictor, test_dataset: &Dataset) -> EvaluationResult {
    let mut hits = 0usize;
    let mut total_error = 0.0f64;
    let mut current_examples_cursor = 0usize;
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    while current_examples_cursor < test_dataset.examples_count {
        let (input_data, label_data) = test_dataset.slices_for_cursor(current_examples_cursor);
        true_outputs.copy_from_slice(label_data);
        let outputs = nn_predictor.predict(nn_parameters, input_data).clone();
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

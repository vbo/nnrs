use std::fmt;
use std::time;

use rand;
use rand::Rng;

use network;
use network::{Network, NetworkParameters, NetworkPredictor};

use parallel_trainer::{num_threads, ParallelTrainer};
use training_data::Dataset;

use mnist_data;

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

    let training_data = mnist_data::load_mnist_training();
    assert!(training_data.input_size() == N_INPUTS, "Wrong inputs!");
    assert!(training_data.label_size() == N_OUTPUTS, "Wrong outputs!");

    let mut example_indices: Vec<_> = (0usize..training_data.examples_count()).collect();

    let testing_data = mnist_data::load_mnist_testing();
    assert!(testing_data.input_size() == N_INPUTS, "Wrong inputs!");
    assert!(testing_data.label_size() == N_OUTPUTS, "Wrong outputs!");

    let mut parallel_trainer = ParallelTrainer::new(training_data, nn);

    let mut hits = 0usize;
    let mut random_number_generator = rand::thread_rng();
    let mut batches_processed = 0usize;
    let mut timing = Timing::new();
    for epoch_no in 0..NUM_EPOCHS {
        println!("Starting epoch {}.", epoch_no);
        // Randomize example order
        random_number_generator.shuffle(&mut example_indices.as_mut_slice());
        let indices_batches: Vec<_> = example_indices
            .chunks(network::BATCH_SIZE)
            .map(Vec::from)
            .collect();

        for indices_batch in &indices_batches {
            // TODO: check small last batch
            let batch_chunks: Vec<_> = indices_batch
                .chunks(indices_batch.len() / num_threads())
                .map(Vec::from)
                .collect();
            timing.start("train_batch");
            parallel_trainer.process_batch(batch_chunks);
            timing.stop("train_batch");

            if (batches_processed + 1) % test_every_n == 0 {
                let (params, predictor, _) = parallel_trainer.borrow_network_parts();
                println!(
                    "Trained over {}k examples. Evaluation results: {}",
                    batches_processed * network::BATCH_SIZE / 1000,
                    evaluate(params, predictor, &testing_data)
                );
                timing.dump_divided(batches_processed);
            }

            if !model_output_path.is_empty() && batches_processed % write_every_n == 0 {
                let (params, predictor, _) = parallel_trainer.borrow_network_parts();
                params.write_to_file(&model_output_path);
            }

            batches_processed += 1;
        }
    }

    println!("Main done!");
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

fn evaluate<T: Dataset>(
    nn_parameters: &NetworkParameters,
    nn_predictor: &mut NetworkPredictor,
    test_dataset: &T,
) -> EvaluationResult {
    let mut hits = 0usize;
    let mut total_error = 0.0f64;
    let mut current_examples_cursor = 0usize;
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    while current_examples_cursor < test_dataset.examples_count() {
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
        hits_ratio: hits as f64 / test_dataset.examples_count() as f64,
        avg_error: total_error / test_dataset.examples_count() as f64,
    }
}

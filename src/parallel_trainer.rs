use std::sync::mpsc::{channel, Receiver, RecvError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};

extern crate num_cpus;

use math::*;
use network::*;
use timing::Timing;
use training_data::Dataset;

pub fn num_threads() -> usize {
    num_cpus::get()
}

struct TrainingJob {
    examples_indices: Vec<usize>,
    network_parameters: Arc<NetworkParameters>,
}

pub struct ParallelTrainer<D> {
    shared_training_data: Arc<D>,

    shared_nn_parameters: Arc<NetworkParameters>,
    nn_predictor: NetworkPredictor,
    nn_trainer: NetworkTrainer,

    job_sender: Sender<TrainingJob>,
    job_receiver: Arc<Mutex<Receiver<TrainingJob>>>,
    output_sender: Sender<Arc<Mutex<NetworkTrainer>>>,
    output_receiver: Receiver<Arc<Mutex<NetworkTrainer>>>,
    join_handles: Vec<JoinHandle<()>>,
}

impl<D: Dataset> ParallelTrainer<D> {
    pub fn new(dataset: D, network: Network) -> Self {
        let (mut nn_parameters, mut nn_predictor, mut nn_trainer) = network.as_parts();
        let (job_sender, job_receiver) = channel();
        let (output_sender, output_receiver) = channel();

        let shared_training_data = Arc::new(dataset);
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let mut join_handles = Vec::new();

        let num_threads = num_threads();
        for thread_no in 0..num_threads {
            // Threadlocal working copies
            let mut local_nn_predictor = nn_predictor.clone();
            let mut local_nn_trainer = Arc::new(Mutex::new(nn_trainer.clone()));

            // Shared immutable data
            let shared_training_data = shared_training_data.clone();

            // Shared channels
            let output_sender = output_sender.clone();
            let job_receiver = job_receiver.clone();
            let jh = spawn(move || {
                worker_thread(
                    job_receiver,
                    output_sender,
                    local_nn_predictor,
                    local_nn_trainer,
                    shared_training_data,
                );
            });
            join_handles.push(jh);
        }

        ParallelTrainer {
            shared_training_data,
            shared_nn_parameters: Arc::new(nn_parameters),
            nn_predictor,
            nn_trainer,
            job_sender,
            job_receiver,
            output_sender,
            output_receiver,
            join_handles,
        }
    }

    pub fn process_batch(&mut self, batch_chunks: Vec<Vec<usize>>) {
        let batch_chunks_no = batch_chunks.len();
        for chunk in batch_chunks {
            self.job_sender.send(TrainingJob {
                examples_indices: chunk,
                network_parameters: self.shared_nn_parameters.clone(),
            });
        }

        let mut output_no = 0;
        for output_nn_trainer in self.output_receiver.iter() {
            {
                let mut output_nn_trainer = output_nn_trainer.lock().unwrap();
                self.nn_trainer.aggregate(&output_nn_trainer);
            }
            output_no += 1;
            if output_no == batch_chunks_no {
                break;
            }
        }

        if let Some(nn_parameters) = Arc::get_mut(&mut self.shared_nn_parameters) {
            self.nn_trainer.apply_batch(nn_parameters);
        } else {
            panic!("NN params should be freed after batch");
        }
    }

    pub fn borrow_network_parts(
        &mut self,
    ) -> (&NetworkParameters, &mut NetworkPredictor, &NetworkTrainer) {
        let nn_parameters = match Arc::get_mut(&mut self.shared_nn_parameters) {
            Some(params) => params,
            _ => panic!("Cannot borrow"),
        };

        (nn_parameters, &mut self.nn_predictor, &self.nn_trainer)
    }
}

fn worker_thread<D: Dataset>(
    mut job_receiver: Arc<Mutex<Receiver<TrainingJob>>>,
    mut output_sender: Sender<Arc<Mutex<NetworkTrainer>>>,
    mut local_nn_predictor: NetworkPredictor,
    mut local_nn_trainer: Arc<Mutex<NetworkTrainer>>,
    shared_training_data: Arc<D>,
) {
    let mut true_outputs = Vector::new(shared_training_data.label_size()).init_with(0.0);
    loop {
        {
            // NOTE(vbo): non-lexical lifetimes would allow ommiting block here.
            let next_job: Result<TrainingJob, RecvError> = { job_receiver.lock().unwrap().recv() };

            // TODO(vbo): express in monads.
            let training_job = match next_job {
                Ok(job) => job,
                _ => {
                    println!("Exiting child thread!");
                    break;
                }
            };

            let mut local_nn_trainer = local_nn_trainer.lock().unwrap();
            local_nn_trainer.reset();
            for example_index in training_job.examples_indices {
                let (input_data, label_data) =
                    shared_training_data.slices_for_cursor(example_index);
                true_outputs.copy_from_slice(label_data); // TODO: no copy
                let outputs = local_nn_predictor
                    .predict(&training_job.network_parameters, input_data)
                    .clone();
                local_nn_trainer.backward_propagation(
                    &training_job.network_parameters,
                    &local_nn_predictor,
                    &true_outputs,
                );
            }
        } // drop network parameters
        output_sender.send(local_nn_trainer.clone());
    }
}

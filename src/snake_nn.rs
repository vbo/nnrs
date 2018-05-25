#![feature(test)]
use math::*;
use network;
use network::{Network, NetworkParameters, NetworkPredictor};
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use serde_json;
use snake::*;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufRead;
use std::io::BufReader;
use std::io::prelude::*;
use std::io::stdout;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use timing::Timing;
use timing::duration_as_total_nanos;
use training_data::Dataset;
use parallel_trainer::{num_threads, ParallelTrainer};

const SLEEP_INTERVAL_MS: u32 = 200;
const FORGET_RATE: f64 = 0.2;
const GAME_OVER_COST: f64 = -0.5;
const MAP_WIDTH: usize = 3;
const MAP_HEIGHT: usize = 3;
const N_INPUTS: usize = MAP_WIDTH * MAP_HEIGHT * SNAKE_TILE_SIZE + SNAKE_INPUT_SIZE;
const RANDOM_MOVE_PROBABILITY: f64 = 0.2;

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionStep {
    state: GameState,
    action: SnakeInput,
    is_optimal: bool,
}

// TODO(vbo): impl network perdiction debugger
// TODO(lenny): impl training data scoring debugger. This would allow us to understand if it's
// the network that is not learning or is it us who provide wrong score.
// TODO(lenny): it'd be cool to create two networks:
// - to predict the next state
// - to evaluate the state
// This way we get more information which layer is failing us.
pub fn snake_train(
    model_input_path: &str,
    model_output_path: &str,
    training_data_path: &str,
    training_data_max: usize,
    write_every_n: usize,
    log_every_n: usize,
    num_epochs: usize,
) {
    let mut nn = network::Network::load_from_file(model_input_path);
    println!("Starting model loaded from {}", training_data_path);

    //TODO(vbo): training_data_max should be a buffer here, we still need to read all the file.
    let training_data = read_dataset_from_file(training_data_path, training_data_max);
    let mut example_indices: Vec<usize> = (0..training_data.examples_count()).collect();
    println!(
        "Training data working set loaded from {}",
        training_data_path
    );

    let mut parallel_trainer = ParallelTrainer::new(training_data, nn);

    let mut batches_processed = 0;
    let mut random_number_generator = rand::thread_rng();
    let mut timing = Timing::new();
    for epoch_no in 0..num_epochs {
        random_number_generator.shuffle(&mut example_indices.as_mut_slice());
        let indices_batches: Vec<_> = example_indices
            .chunks(network::BATCH_SIZE)
            .map(Vec::from)
            .collect();

        for indices_batch in &indices_batches {
            let batch_chunks: Vec<_> = indices_batch
                .chunks(indices_batch.len() / num_threads())
                .map(Vec::from)
                .collect();
            timing.start("train_batch");
            parallel_trainer.process_batch(batch_chunks);
            timing.stop("train_batch");

            if (batches_processed + 1) % write_every_n == 0 {
                let (params, predictor, _) = parallel_trainer.borrow_network_parts();
                params.write_to_file(&model_output_path);
            }

            if (batches_processed + 1) % log_every_n == 0 {
                let (params, predictor, _) = parallel_trainer.borrow_network_parts();
                println!("Batches processed: {}", batches_processed);
                timing.start("evaluate_on_random_games");
                evaluate_on_random_games(params, predictor, 1000);
                timing.stop("evaluate_on_random_games");

                timing.dump_divided(batches_processed);
            }

            batches_processed += 1;
        }
    }

    timing.dump_divided(batches_processed);
    let (params, predictor, _) = parallel_trainer.borrow_network_parts();
    params.write_to_file(&model_output_path);
    println!("Final model saved");
}

pub fn snake_gen(model_path: &str, training_data_path: &str, save_n: usize) {
    let mut nn = network::Network::load_from_file(model_path);
    println!("Model extracted from file...");

    let mut timing = Timing::new();
    timing.start("snake_gen");
    let mut join_handles = Vec::new();
    let cpus = num_threads();
    for thread_no in 0..cpus {
        let mut thread_nn = nn.clone();
        let thread_out_path = training_data_path.to_owned();
        let jh = thread::spawn(move || {
            println!("Thread {}", thread_no);
            let mut training_data_file =
                File::create(&format!("{}_{}", thread_out_path, thread_no)).unwrap();
            let mut sessions_processed = 0usize;
            while sessions_processed < save_n / cpus {
                let mut state = GameState {
                    map: SnakeMap::random(MAP_WIDTH, MAP_HEIGHT),
                    score: 0.0,
                };

                let mut session = play_random_game(&mut thread_nn, state);
                score_session(&mut session);
                for step in &session {
                    serde_json::to_writer(&training_data_file, step).unwrap();
                    write!(training_data_file, "\n");
                }
                sessions_processed += 1;
            }
        });

        join_handles.push(jh);
    }

    for jh in join_handles {
        jh.join();
    }

    timing.stop("snake_gen");
    timing.dump();
}

pub fn snake_demo(model_path: &str, games_to_play: usize, visualize: bool) {
    let mut nn = network::Network::load_from_file(model_path);
    println!("Model extracted from file...");
    let mut sessions_processed = 0;
    let mut avg_score: f64 = 0.0;
    while sessions_processed < games_to_play {
        let mut state = GameState {
            map: SnakeMap::random(MAP_WIDTH, MAP_HEIGHT),
            score: 0.0,
        };

        let mut done = false;
        let mut rng = rand::thread_rng();

        if visualize {
            if MAP_WIDTH > 3 {
                print!("\x1B[2J");
            }
            draw_ascii(&mut stdout(), &state.map);
            thread::sleep_ms(SLEEP_INTERVAL_MS);
        }
        let mut visited = HashSet::new();
        while !done {
            let (input, is_optimal) = {
                let (nn_params, nn_predictor, _) = nn.borrow_parts();
                get_next_input_with_strat(
                    nn_params,
                    nn_predictor,
                    &mut visited,
                    &state,
                    0.0,
                    &mut rng,
                    visualize)
            };
            if !is_optimal {
                panic!("Same state reached.");
            }
            if visualize {
                if MAP_WIDTH > 3 {
                    print!("\x1B[2J");
                    print!("\x1B[1;1H");
                }
                println!("{:?}", input);
            }
            let StepResult {
                state: new_state,
                game_over: game_over,
            } = snake_step(state, input);
            state = new_state;
            done = game_over;
            if done {
                avg_score += state.score;
            }
            if visualize && !done {
                draw_ascii(&mut stdout(), &state.map);
                thread::sleep_ms(SLEEP_INTERVAL_MS);
            }
        }
        sessions_processed += 1;
    }
    println!(
        "Avg score: {}, games played {}",
        avg_score / sessions_processed as f64,
        sessions_processed
    );
}

pub fn snake_new(model_output_path: &str) {
    let mut nn;
    println!("Creating new network");
    let shape = [N_INPUTS, 36, 1];
    nn = network::Network::new(shape[0], shape[shape.len() - 1]);
    let mut prev_layer = nn.input_layer();
    for i in 1..shape.len() - 1 {
        let mut layer = nn.add_hidden_layer(shape[i]);
        nn.add_layer_dependency(layer, prev_layer);
        prev_layer = layer;
    }
    let outputs_id = nn.output_layer();
    nn.add_layer_dependency(outputs_id, prev_layer);
    let inp_layer = nn.input_layer();
    nn.add_layer_dependency(outputs_id, inp_layer);
    nn.write_to_file(model_output_path);
    println!("Model saved");
}

fn play_random_game(nn: &mut network::Network, mut state: GameState) -> Vec<SessionStep> {
    let mut done = false;
    let mut rng = rand::thread_rng();

    let mut visited = HashSet::new();
    let mut session = Vec::new();
    while !done {
        let (input, is_optimal) = {
            let (nn_params, nn_predictor, _) = nn.borrow_parts();
            get_next_input_with_strat(
                nn_params,
                nn_predictor,
                &mut visited,
                &state,
                RANDOM_MOVE_PROBABILITY,
                &mut rng,
                false,
            )
        };

        session.push(SessionStep {
            state: state.clone(),
            action: input,
            is_optimal: is_optimal,
        });

        let StepResult {
            state: new_state,
            game_over: game_over,
        } = snake_step(state, input);
        state = new_state;
        done = game_over;
    }

    return session;
}

fn evaluate_on_random_games(
    nn_parameters: &NetworkParameters,
    nn_predictor: &mut NetworkPredictor,
    count: usize,
) {
    let mut sessions_processed = 0;
    let mut sum_score: f64 = 0.0;
    let mut loops = 0;
    while sessions_processed < count {
        let mut state = GameState {
            map: SnakeMap::random(MAP_WIDTH, MAP_HEIGHT),
            score: 0.0,
        };

        let mut done = false;
        let mut rng = rand::thread_rng();
        let mut visited = HashSet::new();
        while !done {
            let (input, is_optimal) =
                get_next_input_with_strat(nn_parameters, nn_predictor, &mut visited, &state, 0.0, &mut rng, false);

            if !is_optimal {
                loops += 1;
                break;
            }

            let StepResult {
                state: new_state,
                game_over: game_over,
            } = snake_step(state, input);
            state = new_state;
            done = game_over;
            if done {
                sum_score += state.score;
            }
        }
        sessions_processed += 1;
    }

    println!(
        "Played: {}, avg score: {}, avg loops: {}",
        count,
        sum_score / count as f64,
        loops as f64 / count as f64
    );
}

fn get_next_input_with_strat<R: rand::Rng>(
    nn_params: &NetworkParameters,
    nn_predictor: &mut NetworkPredictor,
    visited: &mut HashSet<u64>,
    state: &GameState,
    random_move_prob: f64,
    rng: &mut R,
    visualize: bool,
) -> (SnakeInput, bool) {
    // TODO(vbo): remember prediction
    use self::SnakeInput::*;
    let mut inputs = convert_state_to_network_inputs(state);
    let possible_snake_inputs = [Up, Down, Left, Right];
    if (rng.gen_range(0.0, 1.0) <= random_move_prob) {
        let i = rng.gen_range(0, SNAKE_INPUT_SIZE);
        return (possible_snake_inputs[i], false);
    }
    let mut snake_inputs_gains = vec![0.0; possible_snake_inputs.len()];
    for (i, snake_input) in possible_snake_inputs.iter().enumerate() {
        set_snake_input_in_network_inputs(&mut inputs, *snake_input);
        let outputs = nn_predictor.predict(nn_params, inputs.as_slice());
        assert!(outputs.rows == 1);
        snake_inputs_gains[i] = outputs.mem[0];
        if visualize {
            println!(
                "{:?}: {:?}",
                possible_snake_inputs[i], snake_inputs_gains[i]
            )
        }
    }
    let (i, max_gain) = get_max_with_pos(snake_inputs_gains.as_slice());
    let state_hash = get_state_action_hash(&state.map, possible_snake_inputs[i]);
    if visited.contains(&state_hash) {
        let i = rng.gen_range(0, SNAKE_INPUT_SIZE);
        return (possible_snake_inputs[i], false);
    } else {
        visited.insert(state_hash);
        return (possible_snake_inputs[i], true);
    }
}

fn get_max_with_pos(xs: &[f64]) -> (usize, f64) {
    assert!(xs.len() > 0);
    let mut max = xs[0];
    let mut max_i = 0;

    for i in 1..xs.len() {
        let val = xs[i];
        if val > max {
            max = val;
            max_i = i;
        }
    }

    return (max_i, max);
}

fn teach_nn(
    nn: &mut network::Network,
    state: &GameState,
    snake_input: SnakeInput,
    true_output: f64,
) {
    let mut inputs = convert_state_to_network_inputs(state);
    set_snake_input_in_network_inputs(&mut inputs, snake_input);
    {
        let out = nn.predict(inputs.as_slice());
        assert!(out.rows == 1);
    }
    let mut true_output_vec = Vector::new(1).init_with(0.0);
    true_output_vec.mem[0] = true_output;
    nn.backward_propagation(&true_output_vec);
}

struct SnakeDataset {
    examples_count: usize,
    input_mem: Vec<f64>,
    label_mem: Vec<f64>,
}

impl Dataset for SnakeDataset {
    fn slices_for_cursor(&self, current_example_index: usize) -> (&[f64], &[f64]) {
        let input_data_offset = current_example_index * self.input_size();
        let input_data_end = input_data_offset + self.input_size();
        let input_data = &self.input_mem[input_data_offset..input_data_end];

        let label_data_offset = current_example_index * self.label_size();
        let label_data_end = label_data_offset + self.label_size();
        let label_data = &self.label_mem[label_data_offset..label_data_end];

        return (input_data, label_data);
    }

    fn examples_count(&self) -> usize {
        self.examples_count
    }

    fn input_size(&self) -> usize {
        N_INPUTS
    }

    fn label_size(&self) -> usize {
        1
    }
}

fn read_dataset_from_file(path: &str, count: usize) -> impl Dataset {
    let session_steps = read_training_data_from_file(path, count);
    let examples_count = session_steps.len();
    let input_mem_size = examples_count * N_INPUTS;
    let mut input_mem = vec![0.0f64; input_mem_size];
    let mut label_mem = vec![0.0f64; examples_count];

    {
        let mut session_offset = 0;
        let input_mem_slice = input_mem.as_mut_slice();
        for (session_no, session_step) in session_steps.iter().enumerate() {
            let input_mem_slice = &mut input_mem_slice[session_offset..session_offset+N_INPUTS];
            for (i, tile) in session_step.state.map.tiles().iter().enumerate() {
                let offset = *tile as usize;
                // [b11, f11, ..,
                //  b12, f12, ..,
                //  bN1, fN1, ..,
                //  bNM, fNM, ..,
                //  i1, i2, .., iK]
                input_mem_slice[SNAKE_TILE_SIZE * i + offset] = 1.0;
            }

            let action_offset = session_step.action as usize;
            input_mem_slice[N_INPUTS - 1 - action_offset] = 1.0;

            label_mem[session_no] = session_step.state.score;

            session_offset += N_INPUTS;
        }
    }

    SnakeDataset {
        examples_count,
        input_mem,
        label_mem,
    }
}

fn read_training_data_from_file(path: &str, count: usize) -> Vec<SessionStep> {
    let mut result = Vec::with_capacity(count);
    let file = File::open(path).unwrap();
    let bufreader = BufReader::new(&file);
    let mut steps_read = 0usize;
    for line in bufreader.lines() {
        if steps_read >= count {
            break;
        }
        let line = line.unwrap();
        let session_step = serde_json::from_str(&line).unwrap();
        result.push(session_step);
        steps_read += 1;
    }

    result.shrink_to_fit();
    return result;
}

fn score_session(session: &mut Vec<SessionStep>) {
    let max_score = get_max_score();

    // Diff the scores
    for i in 0..session.len() - 1 {
        session[i].state.score = session[i + 1].state.score - session[i].state.score;
    }
    let session_len = session.len();
    session[session_len - 1].state.score = GAME_OVER_COST;

    // Propagate "future benefits"
    session.reverse();
    for i in 1..session.len() {
        let is_next_action_optimal = session[i - 1].is_optimal;
        let next_score = session[i - 1].state.score;
        let step = &mut session[i];
        // If next action would be random, we want only the score of the next state
        // without added value of future benefits - they are not trustworthy.
        // TODO(vbo) uncomment or delete the condition below
        //if is_next_action_optimal {
        step.state.score = step.state.score + FORGET_RATE * next_score;
        //}
    }

    // Normalize scores
    // To transform score to be from 0 to 1: (score - min_score) / (max_score - min_score)
    // This must be done as a last step to avoid passing positive values to previous score
    // for non-apple moves.
    for step in session {
        step.state.score =
            round_2dec((step.state.score - GAME_OVER_COST) / (max_score - GAME_OVER_COST));
    }
}

fn round_2dec(v: f64) -> f64 {
    return (v * 100.0).round() as f64 / 100.0;
}

fn get_max_score() -> f64 {
    let mut res = GAME_OVER_COST;
    for i in 0..MAP_WIDTH * MAP_HEIGHT - 1 {
        res = res * FORGET_RATE + SNAKE_FRUIT_GAIN;
    }
    return res;
}

fn get_state_action_hash(map: &SnakeMap, action: SnakeInput) -> u64 {
    let mut hasher = DefaultHasher::new();
    map.body().hash(&mut hasher);
    action.hash(&mut hasher);
    return hasher.finish();
}

fn set_snake_input_in_network_inputs(nn_input: &mut [f64], input: SnakeInput) {
    let offset = input as usize;
    nn_input[nn_input.len() - 1 - offset] = 1.0;
}

fn convert_state_to_network_inputs(state: &GameState) -> Vec<f64> {
    assert!(state.map.width() == MAP_WIDTH);
    assert!(state.map.height() == MAP_HEIGHT);
    let mut inputs = vec![0.0; N_INPUTS];

    for (i, tile) in state.map.tiles().iter().enumerate() {
        let offset = *tile as usize;
        // [b11, f11, ..,
        //  b12, f12, ..,
        //  bN1, fN1, ..,
        //  bNM, fNM, ..,
        //  i1, i2, .., iK]
        inputs[SNAKE_TILE_SIZE * i + offset] = 1.0;
    }

    return inputs;
}

#[cfg(nightly)]
#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_teach_nn(b: &mut Bencher) {
        b.iter(|| 2 + 2);
    }
}

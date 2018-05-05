use std::io::BufReader;
use std::io::BufRead;
use std::io::prelude::*;
use std::io::stdout;
use std::thread;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use math::Vector;
use network;

use serde_json;
use rand;

const SLEEP_INTERVAL_MS: u32 = 200;

const FRUIT_GAIN: f64 = 1.0;
const FORGET_RATE: f64 = 0.2;
const GAME_OVER_COST: f64 = -0.5;

const MAP_WIDTH: usize = 3;
const MAP_HEIGHT: usize = 3;
const N_INPUTS: usize = MAP_WIDTH * MAP_HEIGHT * SNAKE_TILE_SIZE + SNAKE_INPUT_SIZE;
const RANDOM_MOVE_PROBABILITY: f64 = 0.2;

/// TODO(vbo): impl network perdiction debugger
/// TODO(lenny): impl training data scoring debugger. This would allow us to understand if it's
/// the network that is not learning or is it us who provide wrong score.
/// TODO(lenny): it'd be cool to create two networks:
/// - to predict the next state
/// - to evaluate the state
/// This way we get more information which layer is failing us.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    map: SnakeMap,
    score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionStep {
    state: GameState,
    action: SnakeInput,
    is_optimal: bool,
}

struct StepResult {
    state: GameState,
    game_over: bool,
}

#[derive(Copy, Clone, Debug, Rand, Hash, Serialize, Deserialize)]
pub enum SnakeInput {
    Up,
    Down,
    Left,
    Right,
}

// TODO(vbo): create macros to get number of values in enum
const SNAKE_INPUT_SIZE: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SnakeTile {
    Empty,
    Fruit,
    Body,
    Head,
}

const SNAKE_TILE_SIZE: usize = 4;

pub trait TileMap<T> {
    fn get_tile_at(&self, x: usize, y: usize) -> T;
    fn set_tile_at(&mut self, xy: (usize, usize), tile: T);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnakeMap {
    height: usize,
    width: usize,
    tiles: Vec<SnakeTile>,
    body: Vec<(usize, usize)>,
}

impl TileMap<SnakeTile> for SnakeMap {
    fn get_tile_at(&self, x: usize, y: usize) -> SnakeTile {
        return self.tiles[y * self.width + x];
    }

    fn set_tile_at(&mut self, xy: (usize, usize), tile: SnakeTile) {
        self.tiles[xy.1 * self.width + xy.0] = tile;
    }
}

impl SnakeMap {
    pub fn new(width: usize, height: usize) -> Self {
        SnakeMap {
            width: width,
            height: height,
            tiles: vec![SnakeTile::Empty; width * height],
            body: Vec::<(usize, usize)>::new(),
        }
    }

    pub fn random(width: usize, height: usize) -> Self {
        let mut map = SnakeMap::new(width, height);
        let head_pos = get_random_empty_tile(&map).unwrap();
        map.set_tile_at(head_pos, SnakeTile::Head);
        map.body.push(head_pos);
        let fruit_pos = get_random_empty_tile(&map).unwrap();
        map.set_tile_at(fruit_pos, SnakeTile::Fruit);
        return map;
    }

    pub fn get_head_pos(&self) -> (usize, usize) {
        assert!(self.body.len() > 0);
        self.body[self.body.len() - 1]
    }

    fn get_new_pos(&self, pos: (usize, usize), input: SnakeInput) -> (usize, usize) {
        match input {
            SnakeInput::Up => (pos.0, (pos.1 + self.height - 1) % self.height),
            SnakeInput::Down => (pos.0, (pos.1 + 1) % self.height),
            SnakeInput::Right => ((pos.0 + 1) % self.width, pos.1),
            SnakeInput::Left => ((pos.0 + self.width - 1) % self.width, pos.1),
        }
    }
}

pub fn draw_ascii<T: Write>(writer: &mut T, map: &SnakeMap) {
    for y in 0..map.height {
        for x in 0..map.width {
            output_char(writer, map.get_tile_at(x, y));
        }
        writer.write("\n".as_bytes());
    }
    writer.flush();
}

pub fn output_char<T: Write>(writer: &mut T, tile: SnakeTile) {
    let character = match tile {
        SnakeTile::Empty => ".",
        SnakeTile::Body => "#",
        SnakeTile::Fruit => "",
        SnakeTile::Head => "@",
    };

    writer.write(character.as_bytes());
}

fn snake_step(mut state: GameState, input: SnakeInput) -> StepResult {
    // TODO: consider edge cases
    let old_pos = state.map.get_head_pos();
    let new_pos = state.map.get_new_pos(old_pos, input);
    let tail_pos = state.map.body[0];

    let next_tile = state.map.get_tile_at(new_pos.0, new_pos.1);
    match next_tile {
        SnakeTile::Head => {
            panic!("There can only be one head");
        }
        SnakeTile::Empty => {
            // add new head to body, remove tail
            state.map.body.push(new_pos);
            state.map.body.remove(0);
            state.map.set_tile_at(new_pos, SnakeTile::Head);
            state.map.set_tile_at(old_pos, SnakeTile::Body);
            state.map.set_tile_at(tail_pos, SnakeTile::Empty);
            StepResult {
                state: state,
                game_over: false,
            }
        }
        SnakeTile::Body => StepResult {
            state: state,
            game_over: true,
        },
        SnakeTile::Fruit => {
            state.score += FRUIT_GAIN;
            // add new head to body
            state.map.body.push(new_pos);
            state.map.set_tile_at(new_pos, SnakeTile::Head);
            state.map.set_tile_at(old_pos, SnakeTile::Body);

            let mut iters = 0;
            let mut game_over = false;
            let next_fruit_pos = get_random_empty_tile(&state.map);
            match next_fruit_pos {
                Some(pos) => {
                    state.map.set_tile_at((pos.0, pos.1), SnakeTile::Fruit);
                }
                None => game_over = true,
            }
            StepResult {
                state: state,
                game_over: game_over,
            }
        }
    }
}

fn get_random_empty_tile(map: &SnakeMap) -> Option<(usize, usize)> {
    let mut free_tiles = Vec::<(usize, usize)>::new();
    for y in 0..map.height {
        for x in 0..map.width {
            if map.get_tile_at(x, y) == SnakeTile::Empty {
                free_tiles.push((x, y));
            }
        }
    }
    if free_tiles.len() == 0 {
        return None;
    }
    let mut rng = rand::thread_rng();
    let index = Range::new(0, free_tiles.len()).ind_sample(&mut rng);
    return Some(free_tiles[index]);
}

pub fn _unused_main_snake_random() {
    let mut state = GameState {
        map: SnakeMap::random(10, 10),
        score: 0.0,
    };

    let mut rng = rand::thread_rng();
    let mut done = false;

    print!("\x1B[2J");
    draw_ascii(&mut stdout(), &state.map);
    thread::sleep_ms(SLEEP_INTERVAL_MS);

    while !done {
        let input: SnakeInput = rng.gen();
        print!("\x1B[2J");
        print!("\x1B[1;1H");
        println!("{:?}", input);
        let StepResult {
            state: new_state,
            game_over: game_over,
        } = snake_step(state, input);
        done = game_over;
        state = new_state;
        draw_ascii(&mut stdout(), &state.map);
        thread::sleep_ms(SLEEP_INTERVAL_MS);
    }
}

fn convert_state_to_network_inputs(state: &GameState) -> Vec<f64> {
    assert!(state.map.width == MAP_WIDTH);
    assert!(state.map.height == MAP_HEIGHT);
    let mut inputs = vec![0.0; N_INPUTS];

    for (i, tile) in state.map.tiles.iter().enumerate() {
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

fn set_snake_input_in_network_inputs(nn_input: &mut [f64], input: SnakeInput) {
    let offset = input as usize;
    nn_input[nn_input.len() - 1 - offset] = 1.0;
}

fn get_state_action_hash(map: &SnakeMap, action: SnakeInput) -> u64 {
    let mut hasher = DefaultHasher::new();
    map.body.hash(&mut hasher);
    action.hash(&mut hasher);
    return hasher.finish();
}

fn get_next_input_with_strat<R: rand::Rng>(
    nn: &mut network::Network,
    visited: &mut HashSet<u64>,
    state: &GameState,
    random_move_prob: f64,
    rng: &mut R,
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
        let outputs = nn.predict(inputs.as_slice());
        assert!(outputs.rows == 1);
        snake_inputs_gains[i] = outputs.mem[0];
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

pub fn main_snake_train(
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
    let training_data = read_training_data_from_file(training_data_path, training_data_max);
    let mut training_data_indices: Vec<usize> = (0..training_data.len()).collect();
    println!(
        "Training data working set loaded from {}",
        training_data_path
    );

    let mut examples_processed = 0;
    let mut random_number_generator = rand::thread_rng();
    for _ in 1..num_epochs + 1 {
        random_number_generator.shuffle(&mut training_data_indices.as_mut_slice());
        for training_data_index in &training_data_indices {
            let sample_step = &training_data[*training_data_index];

            teach_nn(
                &mut nn,
                &sample_step.state,
                sample_step.action,
                sample_step.state.score,
            );
            examples_processed += 1;
            // TODO(vbo): BATCH_SIZE should be app, not lib part.
            if examples_processed % network::BATCH_SIZE == 0 {
                nn.apply_batch();
            }

            if examples_processed % write_every_n == 0 {
                nn.write_to_file(&model_output_path);
                println!("Model saved");
            }

            if examples_processed % log_every_n == 0 {
                println!("Examples processed: {}", examples_processed);
                evaluate_on_random_games(&mut nn, 1000);
            }
        }
    }
}

pub fn main_snake_gen(model_path: &str, training_data_path: &str, save_n: usize) {
    let mut training_data_file = File::create(&training_data_path).unwrap();
    let mut nn = network::Network::load_from_file(model_path);
    println!("Model extracted from file...");
    let mut sessions_processed = 0;
    let mut avg_score: f64 = 0.0;
    while sessions_processed < save_n {
        let mut state = GameState {
            map: SnakeMap::random(MAP_WIDTH, MAP_HEIGHT),
            score: 0.0,
        };

        let mut done = false;
        let mut rng = rand::thread_rng();

        let mut visited = HashSet::new();
        let mut session = Vec::new();
        while !done {
            let (input, is_optimal) = get_next_input_with_strat(
                &mut nn,
                &mut visited,
                &state,
                RANDOM_MOVE_PROBABILITY,
                &mut rng,
            );
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
            if done {
                avg_score += state.score;
            }
        }

        score_session(&mut session);
        for step in &session {
            serde_json::to_writer(&training_data_file, step).unwrap();
            write!(training_data_file, "\n");
        }

        sessions_processed += 1;
    }
    println!(
        "Avg score: {}, games played {}",
        avg_score / sessions_processed as f64,
        sessions_processed
    );
}

fn evaluate_on_random_games(nn: &mut network::Network, count: usize) {
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
        let mut is_loop = false;
        while !done {
            let (input, is_optimal) =
                get_next_input_with_strat(nn, &mut visited, &state, 0.0, &mut rng);

            if !is_optimal {
                loops += 1;
                is_loop = true;
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
        if !is_loop {
            sessions_processed += 1;
        }
    }

    println!(
        "Played: {}, avg score: {}, avg loops: {}",
        count,
        sum_score / count as f64,
        loops as f64 / count as f64
    );
}

pub fn main_snake_demo_nn(model_path: &str, games_to_play: usize, visualize: bool) {
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
            print!("\x1B[2J");
            draw_ascii(&mut stdout(), &state.map);
            thread::sleep_ms(SLEEP_INTERVAL_MS);
        }
        let mut visited = HashSet::new();
        while !done {
            let (input, is_optimal) =
                get_next_input_with_strat(&mut nn, &mut visited, &state, 0.0, &mut rng);
            if !is_optimal {
                panic!("Same state reached.");
            }
            if visualize {
                print!("\x1B[2J");
                print!("\x1B[1;1H");
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
            if visualize {
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

/*
fn make_session_nn(
    nn: &mut network::Network) -> Vec<SessionStep> {
    Vec::new();
}
*/

fn visualize_session(session: &Vec<SessionStep>) {
    for step in session {
        draw_ascii(&mut stdout(), &step.state.map);
    }
}

fn generate_dataset() {}

pub fn main_snake_new_nn(model_output_path: &str) {
    let mut nn;
    println!("Creating new network");
    let shape = [N_INPUTS, 36, 16, 4, 1];
    nn = network::Network::new(shape[0], shape[shape.len() - 1]);
    let mut prev_layer = nn.input_layer();
    for i in 1..shape.len() - 1 {
        let mut layer = nn.add_hidden_layer(shape[i]);
        nn.add_layer_dependency(layer, prev_layer);
        prev_layer = layer;
    }
    let outputs_id = nn.output_layer();
    nn.add_layer_dependency(outputs_id, prev_layer);
    nn.write_to_file(model_output_path);
    println!("Model saved");
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
        step.state.score = (step.state.score - GAME_OVER_COST) / (max_score - GAME_OVER_COST);
    }
}

fn get_max_score() -> f64 {
    let mut res = GAME_OVER_COST;
    for i in 0..MAP_WIDTH * MAP_HEIGHT - 1 {
        res = res * FORGET_RATE + FRUIT_GAIN;
    }
    return res;
}

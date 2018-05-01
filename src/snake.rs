use std::io::prelude::*;
use std::io::stdout;
use std::thread;
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use math::Vector;
use network;

const SLEEP_INTERVAL_MS: u32 = 200;

#[derive(Clone, Debug)]
pub struct GameState {
    map: SnakeMap,
    score: f64,
}

#[derive(Debug)]
struct SessionStep {
    state: GameState,
    action: SnakeInput,
    is_optimal: bool,
}

struct StepResult {
    state: GameState,
    game_over: bool,
}

#[derive(Copy, Clone, Debug, Rand)]
pub enum SnakeInput {
    Up,
    Down,
    Left,
    Right,
}

// TODO(vbo): create macros to get number of values in enum
const SNAKE_INPUT_SIZE: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug)]
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
            state.score += 1.0;
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
                free_tiles.push((x,y));
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

const MAP_WIDTH: usize = 3;
const MAP_HEIGHT: usize = 3;
const N_INPUTS: usize = MAP_WIDTH * MAP_HEIGHT * SNAKE_TILE_SIZE + SNAKE_INPUT_SIZE;
const RANDOM_MOVE_PROBABILITY: f64 = 0.2;

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

fn get_next_input_with_strat<R: rand::Rng>(
    nn: &mut network::Network,
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
    return (possible_snake_inputs[i], true);
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

pub fn main_snake_demo_nn(model_path: &str, log_every_n: usize, visualize: bool) {
    let mut nn = network::Network::load_from_file(model_path);
    println!("Model extracted from file...");
    let mut sessions_processed = 0;
    let mut avg_score: f64 = 0.0;
    while sessions_processed < log_every_n {
        let head_pos = (0, 0);
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
        while !done {
            let (input, is_optimal) = get_next_input_with_strat(&mut nn, &state, 0.0, &mut rng);
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

pub fn main_snake_teach_nn(
    model_input_path: Option<&str>,
    model_output_path: Option<&str>,
    log_every_n: usize,
    write_every_n: usize,
    demo_mode: bool,
) {
    let mut nn;
    if let Some(model_input_path) = model_input_path {
        println!("Loading the network from file {}", model_input_path);
        nn = network::Network::load_from_file(model_input_path);
    } else {
        println!("Creating new network");
        let shape = [N_INPUTS, 32, 16, 1];
        nn = network::Network::new(shape[0], shape[shape.len() - 1]);
        let mut prev_layer = nn.input_layer();
        for i in 1..shape.len() - 1 {
            let mut layer = nn.add_hidden_layer(shape[i]);
            nn.add_layer_dependency(layer, prev_layer);
            prev_layer = layer;
        }
        let outputs_id = nn.output_layer();
        nn.add_layer_dependency(outputs_id, prev_layer);
    }
    let mut examples_processed = 0;
    let mut sessions_processed = 0;
    let mut avg_score: f64 = 0.0;
    loop {
        let mut state = GameState {
            map: SnakeMap::random(MAP_WIDTH, MAP_HEIGHT),
            score: 0.0,
        };

        let mut done = false;
        let mut rng = rand::thread_rng();
        let mut session: Vec<SessionStep> = Vec::new();
        while !done {
            let (input, is_optimal) =
                get_next_input_with_strat(&mut nn, &state, RANDOM_MOVE_PROBABILITY, &mut rng);
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
            if demo_mode {
                draw_ascii(&mut stdout(), &state.map);
                thread::sleep_ms(SLEEP_INTERVAL_MS);
            }
        }

        session.reverse();
        avg_score += session[0].state.score;
        if sessions_processed % log_every_n == 0 {
            println!(
                "Avg score: {}, games played {}",
                avg_score / log_every_n as f64,
                sessions_processed
            );
            avg_score = 0.0;
        }

        // TODO(vbo): collect a bunch of samples, extract random portion and train on it.
        const FORGET_RATE: f64 = 0.2;
        const GAME_OVER_COST: f64 = -1.0;
        let mut future_score = session[0].state.score + GAME_OVER_COST;
        for step in &mut session {
            let tmp_score = step.state.score;
            step.state.score = future_score - step.state.score;
            future_score = tmp_score;
        }
        for i in 1..session.len() {
            let next_score = session[i-1].state.score;
            let step = &mut session[i];
            step.state.score += FORGET_RATE * next_score;
            teach_nn(&mut nn, &step.state, step.action, step.state.score);
            examples_processed += 1;
            // TODO(vbo): BATCH_SIZE should be app, not lib part.
            if examples_processed % network::BATCH_SIZE == 0 {
                nn.apply_batch();
            }
        }
        // To transform score to be from 0 to 1: (score - min_score) / (max_score - min_score)
        // This mast be done as a last step to avoid passing positive values to previous score
        // for non-apple moves.
        let max_score = (MAP_WIDTH*MAP_HEIGHT) as f64 - 1.0;
        for step in &mut session {
            step.state.score = (step.state.score - GAME_OVER_COST) / (max_score - GAME_OVER_COST);
        }
        // println!("{:?}", session);
        // break;
        if sessions_processed % write_every_n == 0 {
            if let Some(model_output_path) = model_output_path {
                nn.write_to_file(&model_output_path);
                println!("Model saved");
            }
        }
        sessions_processed += 1;
    }
}

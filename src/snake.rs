use std::io::prelude::*;
use std::io::stdout;
use std::thread;
use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use math::Vector;

const SLEEP_INTERVAL_MS: u32 = 200;

pub struct GameState {
    map: SnakeMap,
    score: f64,
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

#[derive(Copy, Clone)]
pub enum SnakeTile {
    Empty,
    Fruit,
    Body,
    Head,
}

pub trait TileMap<T> {
    fn get_tile_at(&self, x: usize, y: usize) -> T;
    fn set_tile_at(&mut self, xy: (usize, usize), tile: T);
}

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
    pub fn new(width: usize, height: usize, head_pos: (usize, usize)) -> Self {
        SnakeMap {
            width: width,
            height: height,
            tiles: vec![SnakeTile::Empty; width * height],
            body: vec![head_pos; 1],
        }
    }

    pub fn example(dims: (usize, usize), head_pos: (usize, usize)) -> Self {
        let (width, height) = dims;
        assert!(width >= 8);
        assert!(height >= 8);
        let mut map = SnakeMap::new(width, height, head_pos);
        map.set_tile_at(head_pos, SnakeTile::Head);
        map.set_tile_at((4, 2), SnakeTile::Fruit);
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
            // add new head to body
            state.map.body.push(new_pos);
            state.map.set_tile_at(new_pos, SnakeTile::Head);
            state.map.set_tile_at(old_pos, SnakeTile::Body);

            let mut iters = 0;
            while iters < 100 {
                let mut rng = rand::thread_rng();
                let pos_x = Range::new(0, state.map.width).ind_sample(&mut rng);
                let pos_y = Range::new(0, state.map.height).ind_sample(&mut rng);
                if let SnakeTile::Empty = state.map.get_tile_at(pos_x, pos_y) {
                    state.map.set_tile_at((pos_x, pos_y), SnakeTile::Fruit);
                    break;
                }

                iters += 1;
            }
            assert!(iters < 100);

            StepResult {
                state: state,
                game_over: false,
            }
        }
    }
}

pub fn main_snake() {
    let head_pos = (4, 4);
    let mut state = GameState {
        map: SnakeMap::example((10, 10), head_pos),
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

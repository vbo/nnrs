use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use std::io::prelude::*;

use rand;
use serde_json;

pub const SNAKE_FRUIT_GAIN: f64 = 1.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    pub map: SnakeMap,
    pub score: f64,
}

pub struct StepResult {
    pub state: GameState,
    pub game_over: bool,
}

#[derive(Copy, Clone, Debug, Rand, Hash, Serialize, Deserialize)]
pub enum SnakeInput {
    Up,
    Down,
    Left,
    Right,
}

// TODO(vbo): create macros to get number of values in enum
pub const SNAKE_INPUT_SIZE: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SnakeTile {
    Empty,
    Fruit,
    Body,
    Head,
}

pub const SNAKE_TILE_SIZE: usize = 4;

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

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn body(&self) -> &[(usize, usize)] {
        self.body.as_slice()
    }
    pub fn tiles(&self) -> &[SnakeTile] {
        self.tiles.as_slice()
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

pub fn snake_step(mut state: GameState, input: SnakeInput) -> StepResult {
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
            state.score += SNAKE_FRUIT_GAIN;
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

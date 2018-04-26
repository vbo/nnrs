use std::io::prelude::*;
use std::io::stdout;
use math::Vector;

pub struct GameState {
    map: SnakeMap,
    score: f64,
}

struct StepResult {
    state: GameState,
    game_over: bool,
}

#[derive(Copy, Clone)]
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
        SnakeTile::Head => "@"
    };

    writer.write(character.as_bytes());
}

fn get_new_pos(pos: (usize, usize), input: SnakeInput) -> (usize, usize) {
    match input {
        SnakeInput::Up => (pos.0, pos.1 - 1),
        SnakeInput::Down => (pos.0, pos.1 + 1),
        SnakeInput::Right => (pos.0 + 1, pos.1),
        SnakeInput::Left => (pos.0 - 1, pos.1),
    }
}

fn snake_step(mut state: GameState, input: SnakeInput) -> StepResult {
    // TODO: consider edge cases
    let old_pos = state.map.get_head_pos();
    let new_pos = get_new_pos(old_pos, input);
    let tail_pos = state.map.body[0];

    let next_tile = state.map.get_tile_at(new_pos.0, new_pos.1);
    match next_tile {
        SnakeTile::Head => {
            panic!("There can only be one head");
        },
        SnakeTile::Empty => {
            // add new head to body, remove tail
            state.map.body.push(new_pos);
            state.map.body.remove(0);
            state.map.set_tile_at(new_pos, SnakeTile::Head);
            state.map.set_tile_at(old_pos, SnakeTile::Body);
            state.map.set_tile_at(tail_pos, SnakeTile::Empty);
            StepResult {
                state: state,
                game_over: false
            }
        },
        SnakeTile::Body => {
            StepResult {
                state: state,
                game_over: true
            }
        },
        SnakeTile::Fruit => {
            // add new head to body
            state.map.body.push(new_pos);
            state.map.set_tile_at(new_pos, SnakeTile::Head);
            state.map.set_tile_at(old_pos, SnakeTile::Body);
            StepResult {
                state: state,
                game_over: false
            }
        },
    }
}

pub fn main_snake() {
    let head_pos = (4, 4);
    let mut state = GameState {
        map: SnakeMap::example((10, 10), head_pos),
        score: 0.0,
    };

    draw_ascii(&mut stdout(), &state.map);
    println!("\n");
    let StepResult{state: state, game_over: game_over} = snake_step(state, SnakeInput::Up);
    draw_ascii(&mut stdout(), &state.map);
    println!("\n");
    let StepResult{state: state, game_over: game_over} = snake_step(state, SnakeInput::Up);
    draw_ascii(&mut stdout(), &state.map);
    println!("\n");
    let StepResult{state: state, game_over: game_over} = snake_step(state, SnakeInput::Up);
    draw_ascii(&mut stdout(), &state.map);
    println!("\n");
    let StepResult{state: state, game_over: game_over} = snake_step(state, SnakeInput::Right);
    draw_ascii(&mut stdout(), &state.map);
    println!("\n");
}

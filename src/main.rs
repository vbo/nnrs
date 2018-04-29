extern crate rand;
extern crate serde;

#[macro_use]
extern crate serde_derive;
extern crate serde_json;

#[macro_use]
extern crate rand_derive;

#[macro_use]
extern crate clap;

mod snake;
mod mnist;
mod network;
mod mnist_data;
mod math;

// TODO: parse command line arguments
pub const LOAD_FROM_FILE: bool = false;
pub const LOG_EVERY_N: usize = 50_000;
pub const WRITE_EVERY_N: usize = 50_000;
pub const MODEL_OUTPUT_PATH: &str = "model.json";

fn main() {
    snake::main_snake_random_nn(
        LOAD_FROM_FILE,
        MODEL_OUTPUT_PATH,
        LOG_EVERY_N,
        WRITE_EVERY_N,
    );
}

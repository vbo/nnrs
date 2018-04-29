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
pub const SERVING_MODE: bool = false;
pub const TEST_MODE: bool = false;

fn main() {
    let matches = clap_app!(myapp =>
        (version: "0.1")
        (about: "Homebrewed neural network experiments")
        (@subcommand mnist_train =>
            (about: "Train a model to recognize digits")
            (version: "0.1")
            (@arg model_input_path: -i --model_input +takes_value
             "Path to read a model from.")
            (@arg model_output_path: -o --model_output +takes_value
             "Path to write a model to.")
            (@arg write_every_n: --write_every +takes_value
             "Will write a model every N iterations.")
            (@arg log_every_n: --log_every +takes_value
             "Logs some debug info every N iterations.")
            (@arg test_every_n: --test_every +takes_value
             "Evaluate model and print results every N iterations.")
        )
        (@subcommand snake_train =>
            (about: "Train a model to play snake")
            (version: "0.1")
            (@arg model_input_path: -i --model_input +takes_value
             "Path to read a model from.")
            (@arg model_output_path: -o --model_output +takes_value
             "Path to write a model to.")
        )
    );
    if SERVING_MODE {
        snake::main_snake_demo_nn(
            MODEL_OUTPUT_PATH,
            LOG_EVERY_N,
            TEST_MODE);
    } else {
        snake::main_snake_teach_nn(
            LOAD_FROM_FILE,
            MODEL_OUTPUT_PATH,
            LOG_EVERY_N,
            WRITE_EVERY_N,
        );
    }
}

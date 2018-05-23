//#![feature(test)]

use std::str::FromStr;

//extern crate test;
extern crate rand;
extern crate serde;

#[macro_use]
extern crate serde_derive;
extern crate serde_json;

#[macro_use]
extern crate rand_derive;

#[macro_use]
extern crate clap;
use clap::App;
use clap::AppSettings;
use clap::ArgMatches;

mod math;
mod mnist;
mod mnist_data;
mod network;
mod parallel_trainer;
mod snake;
mod snake_nn;
mod timing;
mod training_data;

fn main() {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml)
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .get_matches();

    match matches.subcommand() {
        ("mnist_train", Some(submatches)) => execute_mnist_train(&matches, submatches),
        ("snake_new", Some(submatches)) => execute_snake_new(&matches, submatches),
        ("snake_train", Some(submatches)) => execute_snake_train(&matches, submatches),
        ("snake_demo", Some(submatches)) => execute_snake_demo(&matches, submatches),
        ("snake_gen", Some(submatches)) => execute_snake_gen(&matches, submatches),
        (command_name, Some(_)) => panic!("Command not implemented: {}.", command_name),
        _ => panic!("No subcommand supplied - this should not happen."),
    }
}

fn get_int_arg<T: FromStr>(matches: &ArgMatches, argname: &str) -> Option<T> {
    matches.value_of(argname).unwrap().parse::<T>().ok()
}

fn execute_mnist_train(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_input_path = submatches.value_of("model_input_path").unwrap();
    let model_output_path = submatches.value_of("model_output_path").unwrap();
    let write_every_n = get_int_arg(submatches, "write_every_n").unwrap();
    let log_every_n = get_int_arg(submatches, "log_every_n").unwrap();
    let test_every_n = get_int_arg(submatches, "test_every_n").unwrap();
    mnist::main_mnist(
        model_input_path,
        model_output_path,
        write_every_n,
        test_every_n,
        log_every_n,
    );
}

fn execute_snake_new(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_output_path = submatches.value_of("model_output_path").unwrap();
    snake_nn::snake_new(model_output_path);
}

fn execute_snake_train(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_input_path = submatches.value_of("model_input_path").unwrap();
    let model_output_path = submatches.value_of("model_output_path").unwrap();
    let write_every_n = get_int_arg(submatches, "write_every_n").unwrap();
    let log_every_n = get_int_arg(submatches, "log_every_n").unwrap();
    let training_data_path = submatches.value_of("training_data_path").unwrap();
    let training_data_max = get_int_arg(submatches, "training_data_max").unwrap();
    let num_epochs = get_int_arg(submatches, "num_epochs").unwrap();
    println!("Executing snake training...");
    snake_nn::snake_train(
        model_input_path,
        model_output_path,
        training_data_path,
        training_data_max,
        write_every_n,
        log_every_n,
        num_epochs,
    );
}

fn execute_snake_gen(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_input_path = submatches.value_of("model_input_path").unwrap();
    let training_data_path = submatches.value_of("training_data_path").unwrap();
    let save_n = get_int_arg(submatches, "save_n").unwrap();
    println!("Executing snake gen...");
    snake_nn::snake_gen(model_input_path, training_data_path, save_n);
}

fn execute_snake_demo(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_input_path = submatches.value_of("model_input_path").unwrap();
    let visualize = submatches.is_present("visualize");
    let games_to_play = get_int_arg(submatches, "games_to_play").unwrap();
    println!("Executing snake demo...");
    snake_nn::snake_demo(model_input_path, games_to_play, visualize);
}

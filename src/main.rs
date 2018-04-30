use std::str::FromStr;

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

mod snake;
mod mnist;
mod network;
mod mnist_data;
mod math;

fn main() {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml)
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .get_matches();

    match matches.subcommand() {
        ("mnist_train", Some(submatches)) => execute_mnist_train(&matches, submatches),
        ("snake_train", Some(submatches)) => execute_snake_train(&matches, submatches),
        ("snake_demo", Some(submatches)) => execute_snake_demo(&matches, submatches),
        (command_name, Some(_)) => panic!("Command not implemented: {}.", command_name),
        _ => panic!("No subcommand supplied - this should not happen."),
    }
}

fn get_int_arg<T: FromStr>(matches: &ArgMatches, argname: &str) -> Option<T> {
    matches.value_of(argname).unwrap().parse::<T>().ok()
}

fn execute_mnist_train(matches: &ArgMatches, submatches: &ArgMatches) {}

fn execute_snake_train(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_input_path = submatches.value_of("model_input_path");
    let model_output_path = submatches.value_of("model_output_path");
    let write_every_n = get_int_arg(submatches, "write_every_n").unwrap();
    let log_every_n = get_int_arg(matches, "log_every_n").unwrap();
    let visualize = matches.is_present("visualize");
    println!("Executing snake training..");
    snake::main_snake_teach_nn(
        model_input_path,
        model_output_path,
        log_every_n,
        write_every_n,
        visualize,
    );
}

fn execute_snake_demo(matches: &ArgMatches, submatches: &ArgMatches) {
    let model_input_path = submatches.value_of("model_input_path").unwrap();
    let visualize = matches.is_present("visualize");
    // TODO(lenny): reconsider naming for log_every_n
    let log_every_n = get_int_arg(matches, "log_every_n").unwrap();
    println!("Executing snake demo..");
    snake::main_snake_demo_nn(model_input_path, log_every_n, !visualize);
}

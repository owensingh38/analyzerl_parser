use anyhow::{anyhow, Result};
use std::env;

use _boxcars::{
    animate_json_command, frames_args, frames_command, index_args, index_command,
    inspect_flip_command, match_guid_args, match_guid_command, match_guids_args,
    match_guids_command, parse_args, parse_command,
};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);

    match args.next().as_deref() {
        Some("parse") => parse_command(parse_args(args.collect())?),
        Some("frames") => frames_command(frames_args(args.collect())?),
        Some("index") => index_command(index_args(args.collect())?),
        Some("match-guid") => match_guid_command(match_guid_args(args.collect())?),
        Some("match-guids") => match_guids_command(match_guids_args(args.collect())?),
        Some("animate-json") => animate_json_command(args.collect()),
        Some("inspect-flip") => inspect_flip_command(args.collect()),
        Some("--help") | Some("-h") | None => {
            print_help();
            Ok(())
        }
        Some(command) => Err(anyhow!("unknown command: {command}")),
    }
}

fn print_help() {
    println!("analyzerl_boxcars");
}

//! Print the MUSA operator resolution report.
//!
//! One JSON document per run, on stdout, so it can be diffed between revisions
//! and read by `python/apxinf_ref compare` alongside a stage probe. Progress and
//! diagnostics go to stderr; the same split the stage probe uses.

use std::process::ExitCode;

use apxinf_musa::resolve::report;
use apxinf_musa::spec::Policy;

fn main() -> ExitCode {
    let mut token_count = 10usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--token-count" => match arguments.next().and_then(|value| value.parse().ok()) {
                Some(value) => token_count = value,
                None => {
                    eprintln!("--token-count needs an integer");
                    return ExitCode::FAILURE;
                }
            },
            "--help" | "-h" => {
                println!(
                    "usage: operator-gaps [--token-count <n>]\n\n\
                     Emits apxinf.musa.operator-resolution.v1 on stdout: one row per\n\
                     semantic PI0.5 needs, with its resolution against an empty MUSA\n\
                     candidate registry."
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unrecognized argument {other:?}");
                return ExitCode::FAILURE;
            }
        }
    }

    let value = report(token_count, &Policy::default(), &Default::default());
    match serde_json::to_string_pretty(&value) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cannot serialize the resolution report: {error}");
            ExitCode::FAILURE
        }
    }
}

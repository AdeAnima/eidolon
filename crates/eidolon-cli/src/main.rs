//! `eidolon` executable: report matches without changing input files.

use std::io::Write as _;
use std::process::ExitCode;

fn main() -> ExitCode {
    match eidolon_cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(std::io::stderr().lock(), "eidolon: {error:#}");
            ExitCode::FAILURE
        }
    }
}

//! Explicit-URL manual smoke and transfer accounting, without the CLI detector stack.

use std::io::{Write as _, stdout};
use std::path::PathBuf;

use anyhow::{Context as _, ensure};
use eidolon_fetch::{FetchOptions, Fetcher, VideoMode};

fn main() -> anyhow::Result<()> {
    let mut options = FetchOptions::default();
    if let Some(path) = std::env::var_os("EIDOLON_FETCH_CACHE") {
        options.cache_dir = PathBuf::from(path);
    }
    let mut urls = Vec::new();
    for argument in std::env::args().skip(1) {
        if argument == "--full-cadence" {
            options.video_mode = VideoMode::FullCadence;
        } else {
            urls.push(argument);
        }
    }
    ensure!(
        !urls.is_empty(),
        "usage: measure [--full-cadence] HTTP_URL ...; EIDOLON_FETCH_CACHE overrides cache path"
    );
    let cache = options.cache_dir.clone();
    let fetcher = Fetcher::with_options(options)?;
    let mut output = stdout().lock();
    let mut successes = 0;
    for url in urls {
        match fetcher.resolve(&url) {
            Ok(result) => {
                for warning in &result.warnings {
                    eprintln!("warning: {warning}");
                }
                let row = serde_json::json!({ "input": url, "bytes_fetched": result.bytes_fetched, "total_size": result.total_size, "ratio": result.total_size.filter(|size| *size > 0).map(|size| result.bytes_fetched as f64 / size as f64), "artifacts": result.artifacts, "warnings": result.warnings });
                let bytes = serde_json::to_vec_pretty(&row)?;
                let name = format!("measurement-{}.json", successes + 1);
                std::fs::write(cache.join(name), &bytes).context("retaining measurement")?;
                writeln!(output, "{row}")?;
                successes += 1;
            }
            Err(error) => eprintln!("warning: {url}: {error:#}"),
        }
    }
    ensure!(
        successes > 0,
        "all URL inputs failed; fetched {} response-body bytes",
        fetcher.bytes_fetched()
    );
    Ok(())
}

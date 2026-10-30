//! Opt-in public-domain Wikimedia smoke tests; fetched sources stay in the cache.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use eidolon_fetch::{FetchOptions, Fetcher};

const IMAGE: &str = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Blue_Marble_2002.png/1280px-Blue_Marble_2002.png";
const VIDEO: &str = "https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f1/NASA-_60_Years_in_60_Seconds.webm/NASA-_60_Years_in_60_Seconds.webm.480p.vp9.webm";

fn smoke(url: &str, name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let cache = std::env::var_os("EIDOLON_FETCH_SMOKE_CACHE")
        .map_or_else(
            || FetchOptions::default().cache_dir.join("real-web-smoke"),
            PathBuf::from,
        )
        .join(format!("{name}-{now}"));
    let fetcher = Fetcher::with_options(FetchOptions {
        cache_dir: cache.clone(),
        ..FetchOptions::default()
    })
    .expect("client");
    let result = fetcher.resolve(url).expect("public-domain Wikimedia URL");
    assert_eq!(result.artifacts.len(), 1);
    assert!(result.bytes_fetched > 0);
    let size = result.total_size.expect("Wikimedia supplies source size");
    let measurement = serde_json::json!({ "url": url, "bytes_fetched": result.bytes_fetched, "total_size": size, "ratio": result.bytes_fetched as f64 / size as f64, "artifacts": result.artifacts, "warnings": result.warnings });
    std::fs::write(
        cache.join("measurement.json"),
        serde_json::to_vec_pretty(&measurement).expect("measurement JSON"),
    )
    .expect("retain measurement");
    eprintln!("MEASURE real-web-{name}: {measurement}");
    assert_eq!(
        fetcher
            .resolve(url)
            .expect("real-web revalidation")
            .bytes_fetched,
        0
    );
    result.artifacts[0].local_path.clone()
}

#[test]
#[ignore = "explicit URL network access; NASA public-domain image, cold cache"]
fn commons_public_domain_image() {
    let path = smoke(IMAGE, "image");
    let image = image::open(path).expect("real image decodes");
    assert_eq!(image.width(), 1280);
}

#[test]
#[ignore = "explicit URL network access; NASA public-domain 60-second indexed WebM"]
fn commons_public_domain_video() {
    let path = smoke(VIDEO, "video");
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .expect("ffprobe installed");
    assert!(output.status.success());
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).expect("probe JSON");
    assert!(
        metadata["format"]["duration"]
            .as_str()
            .expect("duration")
            .parse::<f64>()
            .expect("duration number")
            >= 60.0
    );
}

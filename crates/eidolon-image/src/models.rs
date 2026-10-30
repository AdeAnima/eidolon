//! Explicit, SHA-256-pinned downloads; detection itself never uses the network.

use std::fs::File;
use std::io::{BufReader, Read as _, Write as _};
use std::path::Path;
use std::time::Duration;

use anyhow::ensure;
use sha2::{Digest as _, Sha256};

/// Local filename of the FP32 DINOv2-small model (CPU-compatible, 88.5 MB).
pub const FILE_NAME: &str = "dinov2-small.onnx";
/// Immutable upstream revision; the ONNX graph contains its own weights.
pub const URL: &str = "https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx";
/// SHA-256 published as the model's Git LFS object identifier.
pub const SHA256: &str = "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a";

/// Fetch the pinned model, verifying existing and downloaded files before use.
///
/// Downloads stream into a temporary file in `model_dir`, then atomically
/// publish the verified model. This is the crate's only network entry point;
/// no telemetry, remote inference, or automatic model updates are used.
///
/// # Errors
/// Fails on network/filesystem errors or a SHA-256 mismatch. An invalid existing
/// model is not silently replaced.
pub fn fetch(model_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(model_dir)?;
    let target = model_dir.join(FILE_NAME);
    if target.try_exists()? {
        return verify(&target);
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300))
        .user_agent("eidolon/0.1 explicit-model-download")
        .build()?;
    let mut response = client.get(URL).send()?.error_for_status()?;
    let mut temporary = tempfile::NamedTempFile::new_in(model_dir)?;
    std::io::copy(&mut response, &mut temporary)?;
    temporary.flush()?;
    verify(temporary.path())?;
    temporary.as_file().sync_all()?;
    temporary.persist(&target).map_err(|error| error.error)?;
    Ok(())
}

pub(crate) fn verify(path: &Path) -> anyhow::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    ensure!(
        actual == SHA256,
        "SHA-256 mismatch for {}: expected {SHA256}, got {actual}",
        path.display()
    );
    Ok(())
}

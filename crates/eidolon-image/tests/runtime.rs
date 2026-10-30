//! Child-process checks for native runtime and compiled-cache startup outside Cargo.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail, ensure};
use eidolon_core::Embedder as _;

const CHILD_CASE: &str = "EIDOLON_IMAGE_RUNTIME_CHILD";
const MODEL_DIR: &str = "EIDOLON_IMAGE_RUNTIME_MODEL_DIR";
const TEST_NAME: &str = "standalone_runtime_loading_does_not_depend_on_cargo_environment";

#[test]
#[ignore = "requires the fetched DINOv2 model; checks child-process startup with a timeout"]
#[cfg(target_os = "macos")]
fn standalone_runtime_loading_does_not_depend_on_cargo_environment() -> anyhow::Result<()> {
    if let Some(case) = std::env::var_os(CHILD_CASE) {
        return load_in_child(case.to_str().context("child case must be UTF-8")?);
    }
    let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
        .join("Library/Caches/eidolon/models");
    ensure!(
        model_dir.join(eidolon_image::models::FILE_NAME).is_file(),
        "fetch the pinned model before running the explicit runtime regression"
    );
    let temp = tempfile::Builder::new()
        .prefix("eidolon-runtime-")
        .tempdir_in("/tmp")?;
    for case in [
        "cache-only",
        "fallback",
        "missing",
        "auto-empty",
        "auto-ready",
    ] {
        run_child(case, &model_dir, temp.path())?;
    }
    Ok(())
}

fn load_in_child(case: &str) -> anyhow::Result<()> {
    use eidolon_image::embed::{ExecutionDevice, OnnxEmbedder};

    let model_dir =
        PathBuf::from(std::env::var_os(MODEL_DIR).context("child model directory missing")?);
    let result = (|| {
        let mut model = if case.starts_with("auto-") {
            OnnxEmbedder::load(&model_dir)
        } else {
            OnnxEmbedder::load_with_device(&model_dir, ExecutionDevice::Cpu)
        }?
        .context("fetched model missing in child process")?;
        if case.starts_with("auto-") {
            model.set_workload(usize::MAX)?;
        }
        if case != "cache-only" {
            model.prepare()?;
        }
        Ok::<_, anyhow::Error>(model)
    })();
    if case == "missing" {
        let error = result
            .err()
            .context("a missing explicitly configured runtime must fail")?;
        ensure!(
            format!("{error:#}").contains("brew install onnxruntime"),
            "missing-runtime error lacks installation guidance: {error:#}"
        );
        eprintln!("missing runtime returned an actionable error without hanging");
        return Ok(());
    }
    let model = result?;
    if case.starts_with("auto-") {
        ensure!(
            model.device() == ExecutionDevice::CoreMl,
            "Auto must exercise the installed CoreML provider, not bypass cache checks via CPU"
        );
        let photo = eidolon_image::decode(Path::new(
            "/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic",
        ))?
        .to_rgb8();
        let vector = model.embed(&photo)?;
        ensure!(
            !vector.is_empty()
                && (vector.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-4,
            "real-photo embedding must be nonempty and L2-normalized"
        );
    } else {
        ensure!(
            model.device() == ExecutionDevice::Cpu,
            "CPU request changed"
        );
    }
    eprintln!("{case}: standalone model loaded on {:?}", model.device());
    Ok(())
}

fn run_child(case: &str, model_dir: &Path, temp: &Path) -> anyhow::Result<()> {
    let stdout_path = temp.join(format!("{case}.stdout"));
    let stderr_path = temp.join(format!("{case}.stderr"));
    let mut command = Command::new(std::env::current_exe()?);
    command
        .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
        .env_remove("ORT_DYLIB_PATH")
        .env(CHILD_CASE, case)
        .env(MODEL_DIR, model_dir)
        .env("HOME", temp.join("home"))
        .stdout(Stdio::from(std::fs::File::create(&stdout_path)?))
        .stderr(Stdio::from(std::fs::File::create(&stderr_path)?));
    if case == "missing" || case == "cache-only" {
        command.env("ORT_DYLIB_PATH", temp.join("absent-libonnxruntime.dylib"));
    }
    // Native-library startup stays bounded independently of permitted first compilation.
    let seconds = if case == "auto-empty" { 90 } else { 15 };
    let started = Instant::now();
    let mut child = command.spawn()?;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= Duration::from_secs(seconds) {
            child.kill()?;
            child.wait()?;
            bail!(
                "{case}: model load exceeded the {seconds}-second child-process deadline\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&std::fs::read(stdout_path)?),
                String::from_utf8_lossy(&std::fs::read(stderr_path)?),
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let stdout = std::fs::read(stdout_path)?;
    let stderr = String::from_utf8(std::fs::read(stderr_path)?)?;
    ensure!(
        status.success(),
        "{case}: child failed\nstdout: {}\nstderr: {stderr}",
        String::from_utf8_lossy(&stdout),
    );
    if case.starts_with("auto-") {
        ensure!(
            stderr.contains("compiling model for this Mac") == (case == "auto-empty"),
            "{case}: compiled-cache state did not match the isolated cold/ready case: {stderr}"
        );
    }
    eprintln!("{case}: {:.3}s\n{stderr}", started.elapsed().as_secs_f64());
    Ok(())
}

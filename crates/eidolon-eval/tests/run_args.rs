//! The runner forwards benchmark flags as argv and archives the effective options.

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::process::Command;

use serde_json::{Value, json};

fn run(
    extra: &[&str],
    harness: &[&str],
) -> anyhow::Result<(tempfile::TempDir, Value, Vec<String>)> {
    let root = tempfile::Builder::new()
        .prefix("eidolon-run-args-")
        .tempdir_in("/tmp")?;
    fs::create_dir(root.path().join("media"))?;
    fs::write(
        root.path().join("truth.json"),
        serde_json::to_vec(&json!({
            "version":1,"seed":1,"root":root.path(),"scale":"test",
            "sources":[],"files":[],"pairs":[],"hard_negatives":[]
        }))?,
    )?;
    let detector = root.path().join("detector with spaces");
    // This receiver exercises actual process argv, independently of detector implementation.
    fs::write(
        &detector,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf '%s\\n' '{{\"version\":1,\"files\":[],\"matches\":[],\"groups\":[],\"settings\":{{\"thumbnails\":true}}}}'\n",
            root.path().join("received-args.txt").display()
        ),
    )?;
    fs::set_permissions(&detector, fs::Permissions::from_mode(0o700))?;
    let output = Command::new(env!("CARGO_BIN_EXE_eidolon-eval"))
        .arg("run")
        .arg(root.path())
        .arg("--eidolon")
        .arg(detector)
        .arg("--json")
        .args(harness)
        .arg("--")
        .args(extra)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let saved: Value = serde_json::from_slice(&fs::read(root.path().join("eidolon-score.json"))?)?;
    assert_eq!(
        report, saved,
        "stdout and archived benchmark provenance must agree"
    );
    let received = fs::read_to_string(root.path().join("received-args.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    Ok((root, report, received))
}

#[test]
fn defaults_to_explicit_thumbnail_matching() -> anyhow::Result<()> {
    let (root, report, received) = run(&[], &[])?;
    assert_eq!(
        report["eidolon_args"],
        json!(["--json", "--thumbnails", "on"])
    );
    assert_eq!(&received[..3], ["--json", "--thumbnails", "on"]);
    assert_eq!(received[3], root.path().join("media").to_string_lossy());
    Ok(())
}

#[test]
fn forwards_overrides_and_preserves_arguments_containing_spaces() -> anyhow::Result<()> {
    let extra = [
        "--thumbnails",
        "off",
        "--fast",
        "--video-fps",
        "0.5",
        "--embed-device",
        "cpu",
        "--cache",
        "/tmp/eidolon-cache with spaces.sqlite",
    ];
    let (_, report, received) = run(&extra, &[])?;
    let expected: Vec<_> = std::iter::once("--json").chain(extra).collect();
    assert_eq!(report["eidolon_args"], json!(expected));
    assert_eq!(&received[..expected.len()], expected.as_slice());
    Ok(())
}

#[test]
fn accepts_equals_overrides_without_duplicate_managed_options() -> anyhow::Result<()> {
    let (_, report, received) = run(
        &[
            "--json",
            "--thumbnails=auto",
            "--remote-video-mode=keyframes",
        ],
        &["--urls", "--remote-video-mode", "full-cadence"],
    )?;
    assert_eq!(report["remote_video_mode"], "keyframes");
    assert_eq!(
        report["eidolon_args"],
        json!([
            "--json",
            "--thumbnails=auto",
            "--remote-video-mode=keyframes",
            "--stdin",
            "--null"
        ])
    );
    assert_eq!(
        received,
        [
            "--json",
            "--thumbnails=auto",
            "--remote-video-mode=keyframes",
            "--stdin",
            "--null"
        ]
    );
    Ok(())
}

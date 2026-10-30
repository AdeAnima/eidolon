//! Record native-library discovery for feature identities and installed binaries.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let output = Command::new("pkg-config")
        .args(["--modversion", "libturbojpeg"])
        .output()
        .expect("pkg-config is required to discover libturbojpeg");
    assert!(
        output.status.success(),
        "libturbojpeg >= 3 is required; install libjpeg-turbo"
    );
    let version = std::str::from_utf8(&output.stdout)
        .expect("libturbojpeg version must be UTF-8")
        .trim();
    assert!(!version.is_empty(), "libturbojpeg version was empty");
    println!("cargo:rustc-env=EIDOLON_TURBOJPEG_VERSION={version}");
    println!("cargo:rerun-if-env-changed=EIDOLON_ORT_DYLIB_PATH");
    let runtime = match env::var("EIDOLON_ORT_DYLIB_PATH") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(env::VarError::NotPresent) => homebrew_runtime(),
        Err(error) => panic!("EIDOLON_ORT_DYLIB_PATH must be UTF-8: {error}"),
    };
    if let Some(path) = runtime {
        println!("cargo:rustc-env=EIDOLON_ORT_DYLIB_PATH={}", path.display());
    }
}

fn homebrew_runtime() -> Option<PathBuf> {
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return None;
    }
    let output = Command::new("brew")
        .env("HOMEBREW_NO_ANALYTICS", "1")
        .env("HOMEBREW_NO_AUTO_UPDATE", "1")
        .args(["--prefix", "onnxruntime"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix = std::str::from_utf8(&output.stdout).ok()?.trim();
    if prefix.is_empty() {
        return None;
    }
    Some(PathBuf::from(prefix).join("lib/libonnxruntime.dylib"))
}

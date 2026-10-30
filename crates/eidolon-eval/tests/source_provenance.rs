//! Native source videos never inherit an image's ancestry from their numeric index.

use std::fs;
use std::process::Command;

use eidolon_eval::generate::refresh_truth;
use eidolon_eval::model::Truth;
use serde_json::json;
use sha2::{Digest as _, Sha256};

#[test]
fn video_image_ancestry_requires_an_explicit_source_record() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-source-ancestry-")
        .tempdir_in("/tmp")?;
    let root = temp.path().join("corpus");
    fs::create_dir_all(root.join("media/image"))?;
    fs::create_dir_all(root.join("media/video"))?;
    let image_path = "media/image/002_original.png";
    let video_path = "media/video/002_original.webm";
    image::RgbImage::from_fn(96, 64, |x, y| {
        image::Rgb([(x * 2) as u8, (y * 3) as u8, ((x + y) * 2) as u8])
    })
    .save(root.join(image_path))?;
    let video = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-threads", "1", "-loop", "1"])
        .arg("-i")
        .arg(root.join(image_path))
        .args([
            "-t",
            "16",
            "-r",
            "1",
            "-c:v",
            "libvpx-vp9",
            "-threads",
            "1",
            "-g",
            "8",
        ])
        .arg(root.join(video_path))
        .output()?;
    assert!(
        video.status.success(),
        "{}",
        String::from_utf8_lossy(&video.stderr)
    );
    let image_hash = format!("{:x}", Sha256::digest(fs::read(root.join(image_path))?));
    let video_hash = format!("{:x}", Sha256::digest(fs::read(root.join(video_path))?));
    let mut truth: Truth = serde_json::from_value(json!({
        "version":1,"seed":1,"root":root,"scale":"test",
        "sources":[
            {"origin":"iMac Blue.heic","retained":image_path,"sha256":image_hash},
            {"origin":"https://commons.wikimedia.org/wiki/File:Independent_video.webm","retained":video_path,"sha256":video_hash}
        ],
        "files":[
            {"path":image_path,"kind":"image","variant":"original","family":"image-002","sha256":image_hash},
            {"path":video_path,"kind":"video","variant":"original","family":"video-002","sha256":video_hash}
        ],"pairs":[],"hard_negatives":[]
    }))?;
    let (native, _) = refresh_truth(truth, &temp.path().join("native"))?;
    assert_ne!(
        native.files[0].design, native.files[1].design,
        "numeric index must not fabricate a source-image connection"
    );
    truth = native;
    truth.sources[1].origin = format!("ffmpeg ken-burns source={image_path} duration=16");
    let (derived, _) = refresh_truth(truth, &temp.path().join("derived"))?;
    assert_eq!(derived.files[0].design, derived.files[1].design);
    assert!(
        derived.pairs.iter().all(|pair| pair.relation != "frame_of"),
        "shared design alone must never create FrameOf annotations"
    );
    Ok(())
}

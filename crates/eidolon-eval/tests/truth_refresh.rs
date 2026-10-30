//! Refresh corrections operate on decoded inputs without changing any source media.

use std::fs;
use std::path::Path;

use eidolon_eval::generate::refresh_truth;
use eidolon_eval::model::Truth;
use serde_json::json;
use sha2::{Digest as _, Sha256};

#[test]
fn refresh_rebuilds_designs_and_negatives_without_mutating_old_media() -> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-truth-refresh-")
        .tempdir_in("/tmp")?;
    let source = directory.path().join("original");
    fs::create_dir(&source)?;
    let mut files = Vec::new();
    let mut sources = Vec::new();
    let names = [
        "Dome Light.heic",
        "Dome.heic",
        "Calibrate 5120x2880.jpg",
        "Calibrate 5120x3200.jpg",
        "Hello Metallic Blue.heic",
        "Hello Metallic Green.heic",
        "Chroma Blue Dark.heic",
        "Chroma Blue Light.heic",
    ];
    let mut original_bytes = Vec::new();
    for (index, origin) in names.iter().enumerate() {
        let path = format!("input-{index}.png");
        // Distinct decoded RGB inputs exceed the visibility threshold in every paired case.
        source_image(index).save(source.join(&path))?;
        let bytes = fs::read(source.join(&path))?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        original_bytes.push((path.clone(), bytes));
        files.push(json!({
            "path":path,"kind":"image","variant":"original",
            "family":format!("image-{index:03}"),"sha256":hash,"design":"obsolete"
        }));
        sources.push(json!({"origin":origin,"retained":path,"sha256":hash}));
    }
    let old = json!({
        "version":1,"seed":17,"root":source,"scale":"test","sources":sources,
        "files":files,
        "pairs":[{"a":"input-4.png","b":"input-5.png","relation":"variant",
            "kind":"image","variant":"source_variant","visible_rgb_mae":100.0}],
        "hard_negatives":[{"a":"input-0.png","b":"input-1.png"}]
    });
    let old_bytes = serde_json::to_vec_pretty(&old)?;
    fs::write(source.join("truth.json"), &old_bytes)?;
    let truth: Truth = serde_json::from_value(old)?;
    let corrected = directory.path().join("corrected");
    let (truth, validation) = refresh_truth(truth, &corrected)?;
    let related = |a: &str, b: &str| {
        truth.pairs.iter().any(|pair| {
            pair.relation == "variant"
                && ((pair.a == Path::new(a) && pair.b == Path::new(b))
                    || (pair.a == Path::new(b) && pair.b == Path::new(a)))
        })
    };
    assert!(related("input-0.png", "input-1.png"));
    assert!(related("input-2.png", "input-3.png"));
    assert!(!related("input-4.png", "input-5.png"));
    assert!(!related("input-6.png", "input-7.png"));
    assert!(truth.pairs.iter().all(|pair| pair.relation != "frame_of"));
    assert!(
        !truth.hard_negatives.iter().any(|pair| {
            pair.a == Path::new("input-0.png") && pair.b == Path::new("input-1.png")
        })
    );
    assert!(
        truth.hard_negatives.is_empty(),
        "uncertain collections must not become negatives"
    );
    assert_eq!(validation.files, 8);
    let output = serde_json::from_value(json!({
        "version":1,
        "files":[{"id":4,"path":"input-4.png"},{"id":5,"path":"input-5.png"},
                 {"id":6,"path":"input-6.png"},{"id":7,"path":"input-7.png"}],
        "matches":[{"a":4,"b":5,"relation":"variant","score":0.99},
                   {"a":6,"b":7,"relation":"reencoded","score":0.99}],
        "groups":[{"files":[6,7]}]
    }))?;
    let report = eidolon_eval::score::evaluate(&truth, &output)?;
    assert_eq!(report.overall.false_positives, 0);
    assert_eq!(report.overall.group_false_positives, 0);
    assert_eq!(report.variants.false_positives, 0);
    assert_eq!(report.ignored_sibling_predictions, 2);
    assert_eq!(fs::read(source.join("truth.json"))?, old_bytes);
    for (path, bytes) in original_bytes {
        assert_eq!(fs::read(source.join(path))?, bytes);
    }
    let refreshed: Truth = serde_json::from_slice(&fs::read(corrected.join("truth.json"))?)?;
    assert_eq!(refreshed.root, source);
    assert!(corrected.join("distinctness.json").is_file());
    assert!(
        refresh_truth(refreshed, &corrected).is_err(),
        "existing metadata must not be overwritten"
    );
    Ok(())
}

fn source_image(index: usize) -> image::RgbImage {
    image::RgbImage::from_fn(96, 72, |x, y| {
        let value = if index.is_multiple_of(2) {
            24 + index as u8 * 9
        } else {
            224 - index as u8 * 7
        };
        let shifted = if matches!(index, 5 | 7) {
            (x + if y < 24 {
                17
            } else if y < 48 {
                0
            } else {
                43
            }) % 96
        } else {
            x
        };
        let shape =
            ((shifted / 13 + y / 11) % 3 == 0) ^ (shifted.abs_diff(42) + y.abs_diff(39) < 21);
        image::Rgb([
            value,
            if shape { 210 } else { 25 },
            if shape { 175 } else { 35 },
        ])
    })
}

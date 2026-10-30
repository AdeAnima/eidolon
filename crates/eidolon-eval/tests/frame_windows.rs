//! Frame containment is derived from retained pan/zoom parameters, not detector output.

use eidolon_eval::frame_windows;
use eidolon_eval::model::Truth;
use serde_json::json;

fn truth(zoom: f64, timestamp: f64) -> anyhow::Result<Truth> {
    Ok(serde_json::from_value(json!({
        "version":1,"seed":7,"root":"/corpus","scale":"test",
        "sources":[
            {"retained":"video.mp4","sha256":"video","origin":format!("ffmpeg ken-burns source=still.jpg zoom=1+{zoom}*on pan=sin(on/40)*20 duration=80 size=960x540 fps=8")},
            {"retained":"thumb.jpg","sha256":"thumb","origin":format!("frame source=video.mp4 timestamp={timestamp}s variant=thumb_frame jpeg_quality=70")}
        ],
        "files":[
            {"path":"still.jpg","kind":"image","variant":"original","family":"image-005","sha256":"still","scene":{"source":"scene.png","x":0,"y":0,"width":720,"height":960}},
            {"path":"video.mp4","kind":"video","variant":"original","family":"video-005","sha256":"video"},
            {"path":"clip10.mp4","kind":"video","variant":"clip_10s","family":"video-005","sha256":"clip10"},
            {"path":"clip25.mp4","kind":"video","variant":"clip_25pct","family":"video-005","sha256":"clip25"},
            {"path":"thumb.jpg","kind":"image","variant":"thumb_frame","family":"video-005","sha256":"thumb"}
        ],
        "pairs":[{"a":"thumb.jpg","b":"video.mp4","relation":"frame_of","kind":"image","variant":"thumb_frame"}],
        "hard_negatives":[]
    }))?)
}

#[test]
fn finds_same_region_outside_time_and_keeps_partial_overlap_unscored() -> anyhow::Result<()> {
    let mut early = truth(0.0015, 3.75)?;
    let audit = frame_windows::annotate(&mut early)?;
    assert!(
        early
            .pairs
            .iter()
            .any(|pair| pair.a.to_str() == Some("thumb.jpg")
                && pair.b.to_str() == Some("clip10.mp4"))
    );
    let added = audit
        .comparisons
        .iter()
        .find(|pair| pair.video.to_str() == Some("clip10.mp4"))
        .expect("clip window proof");
    assert!(added.best_iou >= 0.8 && !added.timestamp_inside_clip);
    assert!(added.best_frame_seconds >= 5.0 && added.best_frame_seconds < 15.0);
    assert_eq!(audit.added_positive_pairs, 1);

    let mut partial = truth(0.0015, 10.875)?;
    let audit = frame_windows::annotate(&mut partial)?;
    let proof = audit
        .comparisons
        .iter()
        .find(|pair| pair.video.to_str() == Some("clip25.mp4"))
        .expect("partial clip");
    assert!((0.3..0.8).contains(&proof.best_iou));
    assert!(
        !partial
            .pairs
            .iter()
            .any(|pair| pair.b.to_str() == Some("clip25.mp4"))
    );
    assert!(partial.hard_negatives.is_empty());
    Ok(())
}

#[test]
fn separates_low_overlap_from_partial_and_preserves_time_proven_cropped_frames()
-> anyhow::Result<()> {
    let mut far = truth(0.005, 79.0)?;
    let audit = frame_windows::annotate(&mut far)?;
    let proof = audit
        .comparisons
        .iter()
        .find(|pair| pair.video.to_str() == Some("clip10.mp4"))
        .expect("far clip");
    assert!(proof.best_iou < 0.3);
    assert!(
        far.hard_negatives
            .iter()
            .any(|pair| pair.a.to_str() == Some("thumb.jpg")
                && pair.b.to_str() == Some("clip10.mp4"))
    );

    let mut cropped = truth(0.0015, 7.0)?;
    cropped.files.last_mut().expect("thumbnail").variant = "thumb_overlay".into();
    frame_windows::annotate(&mut cropped)?;
    assert!(
        cropped
            .pairs
            .iter()
            .any(|pair| pair.b.to_str() == Some("clip10.mp4")),
        "a known time-proven crop remains a frame"
    );
    Ok(())
}

#[test]
fn refresh_cli_never_opens_media_and_keeps_input_metadata_unchanged() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-frame-refresh-")
        .tempdir_in("/tmp")?;
    let input = temp.path().join("input.json");
    let out = temp.path().join("refreshed");
    let bytes = serde_json::to_vec_pretty(&truth(0.0015, 3.75)?)?;
    std::fs::write(&input, &bytes)?;
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_eidolon-eval"))
        .arg("refresh-frames")
        .arg(&input)
        .arg(&out)
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(std::fs::read(&input)?, bytes);
    let refreshed: Truth = serde_json::from_slice(&std::fs::read(out.join("truth.json"))?)?;
    assert_eq!(refreshed.root, std::path::Path::new("/corpus"));
    assert!(
        refreshed
            .pairs
            .iter()
            .any(|pair| pair.b.to_str() == Some("clip10.mp4"))
    );
    let audit: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("frame-windows.json"))?)?;
    assert_eq!(audit["added_positive_pairs"], 1);
    let second = frame_windows::annotate(&mut serde_json::from_slice::<Truth>(
        &serde_json::to_vec(&refreshed)?,
    )?)?;
    assert_eq!(second.added_positive_pairs, 0);
    Ok(())
}

#[test]
fn scorer_ignores_partial_overlap_but_penalizes_below_threshold_predictions() -> anyhow::Result<()>
{
    for (zoom, timestamp, clip, false_positives) in [
        (0.0015, 10.875, "clip25.mp4", 0),
        (0.005, 79.0, "clip10.mp4", 1),
    ] {
        let mut expected = truth(zoom, timestamp)?;
        frame_windows::annotate(&mut expected)?;
        let output = serde_json::from_value(json!({
            "version":1,
            "files":[{"id":0,"path":"thumb.jpg"},{"id":1,"path":clip}],
            "matches":[{"a":0,"b":1,"relation":"frame_of","score":0.99}],
            "groups":[]
        }))?;
        let report = eidolon_eval::score::evaluate(&expected, &output)?;
        assert_eq!(report.overall.false_positives, false_positives);
        assert_eq!(report.overall.group_false_positives, 0);
        assert_eq!(report.ignored_sibling_predictions, 1 - false_positives);
    }
    Ok(())
}

#[test]
fn source_ancestry_without_thumbnail_clip_candidates_needs_no_pan_expression() -> anyhow::Result<()>
{
    let mut metadata = truth(0.0015, 3.75)?;
    metadata.files.retain(|file| file.variant == "original");
    metadata
        .sources
        .retain(|source| !source.origin.starts_with("frame source="));
    metadata.sources[0].origin = "ffmpeg ken-burns source=still.jpg duration=80".into();
    metadata.pairs.clear();
    let audit = frame_windows::annotate(&mut metadata)?;
    assert!(audit.comparisons.is_empty());
    Ok(())
}

#[test]
fn direct_only_negatives_allow_shared_original_groups_without_weakening_legacy_negatives()
-> anyhow::Result<()> {
    for direct_only in [true, false] {
        let mut metadata = truth(0.005, 79.0)?;
        frame_windows::annotate(&mut metadata)?;
        let mut serialized = serde_json::to_value(metadata)?;
        for negative in serialized["hard_negatives"]
            .as_array_mut()
            .expect("negatives")
        {
            if direct_only {
                negative["direct_only"] = json!(true);
            } else {
                negative
                    .as_object_mut()
                    .expect("negative")
                    .remove("direct_only");
            }
        }
        let expected = serde_json::from_value(serialized)?;
        let output = serde_json::from_value(json!({
            "version":1,
            "files":[{"id":0,"path":"thumb.jpg"},{"id":1,"path":"clip10.mp4"},{"id":2,"path":"video.mp4"}],
            "matches":[
                {"a":0,"b":2,"relation":"frame_of","score":0.99},
                {"a":1,"b":2,"relation":"clip_of","score":0.99}
            ],
            "groups":[{"files":[0,1,2],"keep_hint":2}]
        }))?;
        let report = eidolon_eval::score::evaluate(&expected, &output)?;
        assert_eq!(report.overall.false_positives, 0);
        assert_eq!(report.hard_negative_false_positives, 0);
        assert_eq!(
            report.overall.group_false_positives,
            usize::from(!direct_only)
        );
        assert_eq!(
            report.hard_negative_group_false_positives,
            usize::from(!direct_only)
        );
    }
    Ok(())
}

#[test]
fn frame_window_negatives_explicitly_serialize_their_direct_only_scope() -> anyhow::Result<()> {
    let mut metadata = truth(0.005, 79.0)?;
    frame_windows::annotate(&mut metadata)?;
    let serialized = serde_json::to_value(metadata)?;
    let negative = serialized["hard_negatives"]
        .as_array()
        .expect("negatives")
        .iter()
        .find(|row| row["b"] == "clip10.mp4")
        .expect("low-overlap clip");
    assert_eq!(negative["direct_only"], true);
    Ok(())
}

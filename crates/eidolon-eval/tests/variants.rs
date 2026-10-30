//! Variant relationships are related versions, never duplicate-group edges.

use eidolon_eval::model::{Output, Truth};
use eidolon_eval::score::evaluate;
use serde_json::json;

fn truth() -> anyhow::Result<Truth> {
    Ok(serde_json::from_value(json!({
        "version":1,"seed":1,"root":"/corpus","scale":"test","sources":[],
        "files":[
            {"path":"a.jpg","kind":"image","variant":"original","family":"a","design":"one","sha256":""},
            {"path":"b.jpg","kind":"image","variant":"variant_hue","family":"b","design":"one","sha256":""},
            {"path":"c.jpg","kind":"image","variant":"exact","family":"b","design":"one","sha256":""},
            {"path":"d.jpg","kind":"image","variant":"original","family":"d","design":"two","sha256":""}
        ],
        "pairs":[
            {"a":"a.jpg","b":"b.jpg","relation":"variant","kind":"image","variant":"variant_hue"},
            {"a":"a.jpg","b":"c.jpg","relation":"variant","kind":"image","variant":"source_variant"},
            {"a":"b.jpg","b":"c.jpg","relation":"identical","kind":"image","variant":"exact"}
        ],"hard_negatives":[{"a":"a.jpg","b":"d.jpg"}]
    }))?)
}

fn output(matches: &serde_json::Value) -> anyhow::Result<Output> {
    Ok(serde_json::from_value(json!({
        "version":1,"files":[{"id":0,"path":"a.jpg"},{"id":1,"path":"b.jpg"},
            {"id":2,"path":"c.jpg"},{"id":3,"path":"d.jpg"}],
        "matches":matches,"groups":[],
        "variants":[{"a":0,"b":1,"score":0.9}]
    }))?)
}

#[test]
fn variant_edges_have_separate_precision_and_never_connect_duplicate_groups() -> anyhow::Result<()>
{
    let output = output(&json!([
        {"a":0,"b":1,"relation":"variant","score":0.9},
        {"a":1,"b":2,"relation":"identical","score":1.0},
        {"a":0,"b":3,"relation":"variant","score":0.8}
    ]))?;
    let report = evaluate(&truth()?, &output)?;
    assert_eq!(report.overall.expected, 1);
    assert_eq!(report.overall.direct_found, 1);
    assert_eq!(report.overall.group_found, 1);
    assert_eq!(report.overall.false_positives, 0);
    assert_eq!(report.overall.group_false_positives, 0);
    assert_eq!(report.variants.expected, 2);
    assert_eq!(report.variants.direct_found, 1);
    assert_eq!(report.variants.false_positives, 1);
    assert_eq!(report.variants.direct_recall, 0.5);
    assert_eq!(report.variants.direct_precision, 0.5);
    assert_eq!(report.variants.relation_accuracy, Some(1.0));
    assert_eq!(report.variants.group_expected, 0);
    assert_eq!(report.categories["image/variant"].expected, 2);
    Ok(())
}

#[test]
fn confusing_versions_with_duplicates_is_a_miss_and_wrong_channel_false_positive()
-> anyhow::Result<()> {
    let output = output(&json!([
        {"a":0,"b":1,"relation":"edited","score":0.9},
        {"a":1,"b":2,"relation":"variant","score":0.9}
    ]))?;
    let report = evaluate(&truth()?, &output)?;
    assert_eq!(report.overall.direct_found, 0);
    assert_eq!(report.overall.false_positives, 1);
    assert_eq!(report.overall.group_false_positives, 1);
    assert_eq!(report.overall.relation_accuracy, Some(0.0));
    assert_eq!(report.variants.direct_found, 0);
    assert_eq!(report.variants.false_positives, 1);
    assert_eq!(report.variants.relation_accuracy, Some(0.0));
    Ok(())
}

#[test]
fn ignores_unannotated_same_design_derivatives_and_subtle_versions() -> anyhow::Result<()> {
    let truth: Truth = serde_json::from_value(json!({
        "version":1,"seed":1,"root":"/corpus","scale":"test","sources":[],"pairs":[],"hard_negatives":[],
        "files":[
            {"path":"a.jpg","kind":"image","variant":"original","family":"one","design":"grid","sha256":""},
            {"path":"crop.jpg","kind":"image","variant":"crop_40","family":"two","design":"grid","sha256":""},
            {"path":"subtle.jpg","kind":"image","variant":"variant_relit","family":"two/relit","design":"grid","sha256":""}
        ]
    }))?;
    let output: Output = serde_json::from_value(json!({
        "version":1,"files":[{"id":0,"path":"a.jpg"},{"id":1,"path":"crop.jpg"},{"id":2,"path":"subtle.jpg"}],
        "matches":[{"a":0,"b":1,"relation":"variant","score":0.9},{"a":0,"b":2,"relation":"reencoded","score":0.9}],
        "groups":[{"files":[0,1,2]}]
    }))?;
    let report = evaluate(&truth, &output)?;
    assert_eq!(report.overall.false_positives, 0);
    assert_eq!(report.overall.group_false_positives, 0);
    assert_eq!(report.variants.false_positives, 0);
    assert_eq!(report.ignored_sibling_predictions, 2);
    Ok(())
}

#[test]
fn frame_of_is_directional_and_maps_remote_artifacts_by_original_url() -> anyhow::Result<()> {
    let truth: Truth = serde_json::from_value(json!({
        "version":1,"seed":1,"root":"/corpus","scale":"test","sources":[],"hard_negatives":[],
        "files":[
            {"path":"media/image/frame.jpg","kind":"image","variant":"thumb_frame","family":"one","sha256":""},
            {"path":"media/video/whole.mp4","kind":"video","variant":"original","family":"one","sha256":""}
        ],
        "pairs":[{"a":"media/image/frame.jpg","b":"media/video/whole.mp4","kind":"image","variant":"thumb_frame","relation":"frame_of"}]
    }))?;
    let output: Output = serde_json::from_value(json!({
        "version":1,
        "files":[
            {"id":12,"path":"/cache/thumbnail.jpg","url":{"url":"http://127.0.0.1:9876/media/image/frame.jpg","bytes_fetched":100,"total_size":1000}},
            {"id":13,"path":"/corpus/media/video/whole.mp4"}
        ],
        "matches":[{"a":13,"b":12,"relation":"frame_of","score":0.9}],"groups":[]
    }))?;
    let report = evaluate(&truth, &output)?;
    assert_eq!(report.overall.direct_recall, 1.0);
    assert_eq!(report.overall.group_recall, 1.0);
    assert_eq!(report.overall.direction_accuracy, Some(0.0));
    assert_eq!(report.wrong_directions.len(), 1);
    assert_eq!(
        report
            .url_traffic
            .as_ref()
            .map(|traffic| traffic.bytes_fetched_ratio),
        Some(0.1)
    );
    assert_eq!(report.frame_of["local_video"].direct_recall, 1.0);
    assert_eq!(
        report.input_transport["remote/image/thumb_frame"].expected,
        1
    );
    assert_eq!(
        report.input_transport["remote/image/thumb_frame"].direct_recall,
        1.0
    );
    assert_eq!(
        report
            .url_traffic
            .as_ref()
            .map(|traffic| traffic.files["media/image/frame.jpg"].requests),
        Some(None)
    );
    let mut remote_output = output;
    remote_output.files[1].url = Some(eidolon_eval::model::UrlOrigin {
        url: "http://127.0.0.1:9876/media/video/whole.mp4".into(),
        bytes_fetched: 2000,
        total_size: Some(10000),
    });
    let remote_report = evaluate(&truth, &remote_output)?;
    assert_eq!(
        remote_report.frame_of["remote_video/thumb_frame"].direct_recall,
        1.0
    );
    remote_output.files.pop();
    remote_output.matches.clear();
    remote_output.files[0].url = Some(eidolon_eval::model::UrlOrigin {
        url: "http://127.0.0.1:9876/media/image/frame.jpg".into(),
        bytes_fetched: 0,
        total_size: Some(1000),
    });
    let missing = evaluate(&truth, &remote_output)?;
    assert_eq!(missing.frame_of["missing_video"].expected, 1);
    assert_eq!(missing.frame_of["missing_video"].direct_recall, 0.0);
    assert_eq!(
        missing.input_transport["missing/image/thumb_frame"].expected,
        1
    );
    assert_eq!(
        missing
            .url_traffic
            .as_ref()
            .map(|traffic| (traffic.bytes_served, traffic.full_remote_bytes)),
        Some((0, 1000))
    );
    Ok(())
}

#[test]
fn accepts_declared_canvas_duplicate_labels_without_accepting_variant() -> anyhow::Result<()> {
    let mut truth = truth()?;
    truth.pairs.retain(|pair| pair.relation != "variant");
    truth.pairs[0].relation = "reencoded".into();
    truth.pairs[0].variant = "canvas_reframe".into();
    truth.pairs[0].accepted_relations.push("crop_of".into());
    let output = output(&json!([{"a":1,"b":2,"relation":"crop_of","score":0.9}]))?;
    assert_eq!(
        evaluate(&truth, &output)?.overall.relation_accuracy,
        Some(1.0)
    );
    Ok(())
}

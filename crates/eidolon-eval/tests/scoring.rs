//! Pair-level scoring through the serialized report boundary.

use eidolon_eval::score::evaluate;
use serde_json::json;

#[test]
fn reports_false_positive_miss_reversed_direction_and_group_only_find() {
    let truth = serde_json::from_value(json!({
        "version": 1, "seed": 42, "root": "/corpus", "scale": "small", "sources": [],
        "files": [
            {"path":"a.jpg","kind":"image","variant":"original","family":"one","sha256":""},
            {"path":"crop.jpg","kind":"image","variant":"crop_40","family":"one","sha256":""},
            {"path":"copy.jpg","kind":"image","variant":"exact","family":"one","sha256":""},
            {"path":"miss.jpg","kind":"image","variant":"watermark","family":"one","sha256":""},
            {"path":"negative.jpg","kind":"image","variant":"original","family":"two","sha256":""}
        ],
        "pairs": [
            {"a":"crop.jpg","b":"a.jpg","relation":"crop_of","kind":"image","variant":"crop_40"},
            {"a":"a.jpg","b":"copy.jpg","relation":"identical","kind":"image","variant":"exact"},
            {"a":"a.jpg","b":"miss.jpg","relation":"edited","kind":"image","variant":"watermark"}
        ], "hard_negatives": [{"a":"a.jpg","b":"negative.jpg"}]
    }))
    .expect("truth JSON");
    let output = serde_json::from_value(json!({
        "version":1,
        "files":[{"id":0,"path":"/corpus/a.jpg"},{"id":1,"path":"/corpus/crop.jpg"},
                 {"id":2,"path":"/corpus/copy.jpg"},{"id":3,"path":"/corpus/miss.jpg"},
                 {"id":4,"path":"/corpus/negative.jpg"}],
        "matches":[{"a":0,"b":1,"relation":"crop_of","score":0.9},
                   {"a":0,"b":4,"relation":"similar","score":0.8}],
        "groups":[{"files":[0,1,2],"keep_hint":0}]
    }))
    .expect("output JSON");
    let report = evaluate(&truth, &output).expect("valid evaluation");
    assert_eq!(report.overall.expected, 3);
    assert_eq!(report.overall.direct_found, 1);
    assert_eq!(report.overall.group_found, 2);
    assert_eq!(report.overall.false_positives, 1);
    assert_eq!(report.overall.direct_precision, 0.5);
    assert_eq!(report.overall.direct_recall, 1.0 / 3.0);
    assert_eq!(report.overall.group_recall, 2.0 / 3.0);
    assert_eq!(report.overall.relation_accuracy, Some(1.0));
    assert_eq!(report.overall.direction_accuracy, Some(0.0));
    assert_eq!(report.misses.len(), 1);
    assert_eq!(report.false_positives.len(), 1);
    assert_eq!(report.wrong_directions.len(), 1);
    assert_eq!(report.hard_negative_false_positives, 1);
}

#[test]
fn cli_json_preserves_misses_and_fail_under_gates_group_recall() -> anyhow::Result<()> {
    let directory = tempfile::Builder::new()
        .prefix("eidolon-score-test-")
        .tempdir_in("/tmp")?;
    let truth = json!({
        "version":1, "seed":1, "root":directory.path(), "scale":"test", "sources":[],
        "files":[
            {"path":"original.txt","kind":"text","variant":"original","family":"one","sha256":""},
            {"path":"excerpt.txt","kind":"text","variant":"excerpt_10","family":"one","sha256":""}
        ],
        "pairs":[{"a":"excerpt.txt","b":"original.txt","relation":"excerpt_of","kind":"text","variant":"excerpt_10"}],
        "hard_negatives":[]
    });
    let mut output = json!({
        "version":1,
        "files":[{"id":101,"path":"original.txt"},{"id":303,"path":"excerpt.txt"}],
        "matches":[], "groups":[{"files":[101,303],"keep_hint":101}]
    });
    let truth_path = directory.path().join("truth.json");
    let output_path = directory.path().join("output.json");
    std::fs::write(&truth_path, serde_json::to_vec(&truth)?)?;
    std::fs::write(&output_path, serde_json::to_vec(&output)?)?;
    let run = |threshold: Option<&str>| -> anyhow::Result<std::process::Output> {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_eidolon-eval"));
        command
            .arg("score")
            .arg(&truth_path)
            .arg(&output_path)
            .arg("--json");
        if let Some(threshold) = threshold {
            command.args(["--fail-under", threshold]);
        }
        Ok(command.output()?)
    };
    let found = run(Some("0.9"))?;
    assert!(
        found.status.success(),
        "{}",
        String::from_utf8_lossy(&found.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&found.stdout)?;
    assert_eq!(report["overall"]["direct_recall"], 0.0);
    assert_eq!(report["overall"]["group_recall"], 1.0);
    assert!(report["overall"]["direction_accuracy"].is_null());
    let human = std::process::Command::new(env!("CARGO_BIN_EXE_eidolon-eval"))
        .arg("score")
        .arg(&truth_path)
        .arg(&output_path)
        .output()?;
    assert!(human.status.success());
    let human = String::from_utf8(human.stdout)?;
    let original_row = human
        .lines()
        .find(|line| line.starts_with("text/original"))
        .ok_or_else(|| anyhow::anyhow!("missing original row"))?;
    let columns: Vec<_> = original_row.split_whitespace().collect();
    assert_eq!(&columns[2..4], &["n/a", "n/a"]);
    output["groups"] = json!([]);
    // Omit the excerpt from output entirely: it must still count as a miss.
    output["files"] = json!([{"id":101,"path":"original.txt"}]);
    std::fs::write(&output_path, serde_json::to_vec(&output)?)?;
    assert!(
        run(None)?.status.success(),
        "low scores alone must not fail"
    );
    let failed_gate = run(Some("0.9"))?;
    assert_eq!(failed_gate.status.code(), Some(2));
    let report: serde_json::Value = serde_json::from_slice(&failed_gate.stdout)?;
    assert_eq!(report["overall"]["expected"], 1);
    assert_eq!(report["overall"]["group_recall"], 0.0);
    assert_eq!(report["misses"].as_array().map(Vec::len), Some(1));
    Ok(())
}

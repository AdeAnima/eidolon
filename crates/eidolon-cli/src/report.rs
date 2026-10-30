//! Stable JSON contract and truthful direct/transitive human reports.

use std::collections::VecDeque;
use std::io::Write;

use eidolon_core::{Evidence, FileEntry, FileId, Match, Relation};
use serde::Serialize;

use crate::grouping::Group;
use crate::remote::{Sources, UrlInfo};
use crate::settings::Settings;
use crate::skipped::Skipped;

#[derive(Serialize)]
struct Report<'a> {
    version: u8,
    settings: &'a Settings,
    files: ReportedFiles<'a>,
    matches: &'a [Match],
    groups: &'a [Group],
    variants: Variants<'a>,
    skipped: &'a [Skipped],
}

struct ReportedFiles<'a> {
    files: &'a [FileEntry],
    sources: &'a Sources,
}

#[derive(Serialize)]
struct ReportedFile<'a> {
    #[serde(flatten)]
    file: &'a FileEntry,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<&'a UrlInfo>,
}

impl Serialize for ReportedFiles<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(self.files.iter().map(|file| ReportedFile {
            file,
            url: self.sources.by_id.get(&file.id),
        }))
    }
}

struct Variants<'a>(&'a [Match]);

#[derive(Serialize)]
struct VariantPair {
    a: FileId,
    b: FileId,
    score: f64,
}

impl Serialize for Variants<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(
            self.0
                .iter()
                .filter(|edge| edge.relation == Relation::Variant)
                .map(|edge| VariantPair {
                    a: edge.a,
                    b: edge.b,
                    score: edge.score,
                }),
        )
    }
}

pub(crate) fn write(
    json_output: bool,
    files: &[FileEntry],
    matches: &[Match],
    groups: &[Group],
    sources: &Sources,
    settings: &Settings,
    skipped: &[Skipped],
) -> anyhow::Result<()> {
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    if json_output {
        json(
            &mut stdout,
            files,
            matches,
            groups,
            sources,
            settings,
            skipped,
        )?;
        sources.write_summary(&mut std::io::stderr().lock())?;
    } else {
        human(&mut stdout, files, matches, groups, sources, skipped)?;
    }
    stdout.flush()?;
    Ok(())
}

pub(crate) fn json(
    writer: &mut impl Write,
    files: &[FileEntry],
    matches: &[Match],
    groups: &[Group],
    sources: &Sources,
    settings: &Settings,
    skipped: &[Skipped],
) -> anyhow::Result<()> {
    serde_json::to_writer(
        &mut *writer,
        &Report {
            version: 1,
            settings,
            files: ReportedFiles { files, sources },
            matches,
            groups,
            variants: Variants(matches),
            skipped,
        },
    )?;
    writeln!(writer)?;
    Ok(())
}

pub(crate) fn human(
    writer: &mut impl Write,
    files: &[FileEntry],
    matches: &[Match],
    groups: &[Group],
    sources: &Sources,
    skipped: &[Skipped],
) -> anyhow::Result<()> {
    let mut adjacency = vec![Vec::new(); files.len()];
    for (index, matched) in matches
        .iter()
        .enumerate()
        .filter(|(_, edge)| crate::grouping::is_duplicate(edge))
    {
        adjacency[matched.a.0].push((matched.b, index));
        adjacency[matched.b.0].push((matched.a, index));
    }
    let mut parents = vec![None; files.len()];
    for (index, group) in groups.iter().enumerate() {
        let keep = group.keep_hint;
        writeln!(writer, "Group {} ({} files)", index + 1, group.files.len())?;
        writeln!(
            writer,
            "  keep_hint: {} ({} bytes)",
            sources.label(&files[keep.0]),
            files[keep.0].size
        )?;
        let mut queue = VecDeque::from([keep]);
        parents[keep.0] = Some((keep, usize::MAX));
        while let Some(parent) = queue.pop_front() {
            for (child, edge) in &adjacency[parent.0] {
                if parents[child.0].is_none() {
                    parents[child.0] = Some((parent, *edge));
                    queue.push_back(*child);
                }
            }
        }
        for id in &group.files {
            if *id == keep {
                continue;
            }
            let direct = adjacency[id.0]
                .iter()
                .filter(|(other, _)| *other == keep)
                .max_by(|(_, a), (_, b)| matches[*a].score.total_cmp(&matches[*b].score));
            let (other, edge, transitive) = if let Some((other, edge)) = direct {
                (*other, *edge, false)
            } else if let Some((other, edge)) = parents[id.0] {
                (other, edge, true)
            } else {
                anyhow::bail!("duplicate group is not connected");
            };
            let link = &matches[edge];
            let strongest = strongest_evidence(link);
            let relation = describe_relation(link, *id);
            if transitive {
                writeln!(
                    writer,
                    "  {} — transitive to keep_hint (not directly verified); {relation} {}; link score {:.3}; {}",
                    sources.label(&files[id.0]),
                    sources.label(&files[other.0]),
                    link.score,
                    strongest,
                )?;
            } else {
                writeln!(
                    writer,
                    "  {} — {relation} keep_hint; score {:.3}; {}",
                    sources.label(&files[id.0]),
                    link.score,
                    strongest,
                )?;
            }
        }
        writeln!(writer)?;
    }
    let variant_count = write_variants(writer, files, matches, sources)?;
    let unverified_count = write_unverified(writer, files, matches, sources)?;
    let grouped: usize = groups.iter().map(|group| group.files.len()).sum();
    write!(
        writer,
        "{} files; {} duplicate matches; {variant_count} variants",
        files.len(),
        matches.len() - variant_count - unverified_count,
    )?;
    if unverified_count > 0 {
        write!(writer, "; {unverified_count} unverified candidates")?;
    }
    writeln!(writer, "; {} groups; {grouped} grouped files", groups.len())?;
    writeln!(
        writer,
        "{} skipped files (near-duplicate analysis)",
        skipped.len()
    )?;
    sources.write_summary(writer)?;
    Ok(())
}

fn write_variants(
    writer: &mut impl Write,
    files: &[FileEntry],
    matches: &[Match],
    sources: &Sources,
) -> anyhow::Result<usize> {
    let mut count = 0;
    for related in matches
        .iter()
        .filter(|edge| edge.relation == Relation::Variant && !crate::grouping::is_unverified(edge))
    {
        if count == 0 {
            writeln!(writer, "Variants (related, not duplicates)")?;
        }
        writeln!(
            writer,
            "  {} — variant of {}; score {:.3}; {}",
            sources.label(&files[related.a.0]),
            sources.label(&files[related.b.0]),
            related.score,
            strongest_evidence(related),
        )?;
        count += 1;
    }
    if count > 0 {
        writeln!(writer)?;
    }
    Ok(count)
}

fn write_unverified(
    writer: &mut impl Write,
    files: &[FileEntry],
    matches: &[Match],
    sources: &Sources,
) -> anyhow::Result<usize> {
    let mut count = 0;
    for candidate in matches
        .iter()
        .filter(|edge| crate::grouping::is_unverified(edge))
    {
        if count == 0 {
            writeln!(
                writer,
                "Unverified candidates (not pixel-confirmed; not duplicates)"
            )?;
        }
        writeln!(
            writer,
            "  {} — candidate {} {}; score {:.3}; {}",
            sources.label(&files[candidate.a.0]),
            describe_relation(candidate, candidate.a),
            sources.label(&files[candidate.b.0]),
            candidate.score,
            strongest_evidence(candidate),
        )?;
        count += 1;
    }
    if count > 0 {
        writeln!(writer)?;
    }
    Ok(count)
}

fn strongest_evidence(matched: &Match) -> String {
    matched
        .evidence
        .iter()
        .filter(|evidence| evidence.signal != "quality")
        .max_by_key(|evidence| evidence_priority(evidence))
        .map_or_else(
            || "no matching evidence".into(),
            |evidence| {
                format!(
                    "{}: {} (value {})",
                    evidence.signal, evidence.note, evidence.value
                )
            },
        )
}

fn evidence_priority(evidence: &Evidence) -> u8 {
    let signal = evidence.signal;
    if signal.contains("blake3") {
        10
    } else if signal.contains("orb") || signal.contains("homography") || signal == "thumbnail" {
        9
    } else if signal.contains("contain") || signal.contains("align") || signal.contains("excerpt") {
        8
    } else if signal.contains("normalized") || signal.contains("decoded") {
        7
    } else if signal.contains("dinov2") || signal.contains("embedding") {
        6
    } else if signal.contains("minhash") || signal.contains("shingle") {
        5
    } else if signal.contains("phash") || signal.contains("hash") {
        4
    } else {
        1
    }
}

fn describe_relation(matched: &Match, file: FileId) -> &'static str {
    match matched.relation {
        Relation::Identical => "identical to",
        Relation::SameContent => "same content as",
        Relation::Reencoded => "reencoded version of",
        Relation::Edited => "edited version of",
        Relation::Variant => "variant of",
        Relation::Similar => "similar to",
        Relation::CropOf if file == matched.a => "crop of",
        Relation::CropOf => "contains crop of",
        Relation::ClipOf if file == matched.a => "clip of",
        Relation::ClipOf => "contains clip of",
        Relation::ExcerptOf if file == matched.a => "excerpt of",
        Relation::ExcerptOf => "contains excerpt of",
        Relation::FrameOf if file == matched.a => "frame of",
        Relation::FrameOf => "contains frame shown in",
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;
    use eidolon_core::{Evidence, Kind};
    use serde::{Deserialize, Serialize};

    use super::*;

    fn settings() -> Settings {
        let args = crate::args::Args::try_parse_from(["eidolon", "--no-embeddings", "/file-0"])
            .expect("valid report settings");
        Settings::new(&args, None, false, 2)
    }

    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireReport {
        version: u8,
        settings: serde_json::Value,
        files: Vec<WireFile>,
        matches: Vec<WireMatch>,
        groups: Vec<WireGroup>,
        variants: Vec<WireVariant>,
        skipped: Vec<serde_json::Value>,
    }

    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireFile {
        id: usize,
        path: std::path::PathBuf,
        size: u64,
        mtime_ns: i128,
        kind: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<WireUrl>,
    }

    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireUrl {
        url: String,
        page_url: Option<String>,
        role: String,
        mode: String,
        mode_reason: String,
        initial_bytes_fetched: u64,
        refinement_bytes_fetched: u64,
        bytes_fetched: u64,
        total_size: Option<u64>,
        budget_exhausted: bool,
    }
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireMatch {
        a: usize,
        b: usize,
        relation: String,
        score: f64,
        evidence: Vec<WireEvidence>,
    }
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireEvidence {
        signal: String,
        value: f64,
        note: String,
    }
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireGroup {
        files: Vec<usize>,
        keep_hint: usize,
    }

    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct WireVariant {
        a: usize,
        b: usize,
        score: f64,
    }

    fn fixtures() -> (Vec<FileEntry>, Vec<Match>, Vec<Group>) {
        let files = (0..3)
            .map(|id| FileEntry {
                id: FileId(id),
                path: format!("/file-{id}").into(),
                size: 100,
                mtime_ns: 42,
                kind: Kind::Image,
            })
            .collect();
        let matches = [(0, 1), (1, 2)]
            .into_iter()
            .map(|(a, b)| Match {
                a: FileId(a),
                b: FileId(b),
                relation: Relation::CropOf,
                score: 0.95,
                evidence: vec![Evidence {
                    signal: "orb",
                    value: 0.9,
                    note: "90% inliers".into(),
                }],
            })
            .collect();
        let groups = vec![Group {
            files: vec![FileId(0), FileId(1), FileId(2)],
            keep_hint: FileId(0),
        }];
        (files, matches, groups)
    }

    #[test]
    fn human_output_prefers_thumbnail_proof_over_a_hash_nomination() -> anyhow::Result<()> {
        let (files, mut matches, groups) = fixtures();
        matches[0].evidence = vec![
            Evidence {
                signal: "thumbnail",
                value: 0.01,
                note: "aligned RGB agreement; stride=4; 32x32 samples".into(),
            },
            Evidence {
                signal: "phash",
                value: 0.99,
                note: "hash nomination".into(),
            },
        ];
        let mut bytes = Vec::new();
        human(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &[],
        )?;
        let text = String::from_utf8(bytes)?;
        assert!(text.contains("thumbnail: aligned RGB agreement; stride=4; 32x32 samples"));
        assert!(!text.contains("phash: hash nomination"));
        Ok(())
    }

    #[test]
    fn json_round_trips_into_the_exact_contract_shape() -> anyhow::Result<()> {
        let (files, matches, groups) = fixtures();
        let mut bytes = Vec::new();
        json(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &settings(),
            &[],
        )?;
        let parsed: WireReport = serde_json::from_slice(&bytes)?;
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.files[1].id, 1);
        assert_eq!(parsed.files[1].kind, "image");
        assert_eq!(parsed.matches[0].relation, "crop_of");
        assert_eq!(parsed.matches[0].evidence[0].signal, "orb");
        assert_eq!(parsed.groups[0].files, vec![0, 1, 2]);
        assert_eq!(parsed.groups[0].keep_hint, 0);
        assert_eq!(parsed.variants, Vec::<WireVariant>::new());
        let serialized = serde_json::to_vec(&parsed)?;
        assert_eq!(parsed, serde_json::from_slice::<WireReport>(&serialized)?);
        Ok(())
    }

    #[test]
    fn human_output_marks_reverse_containment_and_unverified_transitive_links() -> anyhow::Result<()>
    {
        let (files, matches, groups) = fixtures();
        let mut bytes = Vec::new();
        human(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &[],
        )?;
        let text = String::from_utf8(bytes)?;
        assert!(text.contains("contains crop of keep_hint"));
        assert!(text.contains("transitive to keep_hint (not directly verified)"));
        assert!(text.contains("score 0.950"));
        assert!(text.contains("90% inliers"));
        assert!(
            text.contains("3 files; 2 duplicate matches; 0 variants; 1 groups; 3 grouped files")
        );
        Ok(())
    }

    #[test]
    fn unverified_candidates_remain_in_json_but_not_duplicate_links_or_counts() -> anyhow::Result<()>
    {
        let (files, mut matches, _) = fixtures();
        let mut direct_candidate = matches[0].clone();
        direct_candidate.score = 0.99;
        direct_candidate.evidence.push(Evidence {
            signal: "unverified",
            value: 1.0,
            note: "pixel-level confirmation skipped".into(),
        });
        matches[1].evidence.push(Evidence {
            signal: "unverified",
            value: 1.0,
            note: "pixel-level confirmation skipped".into(),
        });
        matches.push(direct_candidate);
        let groups = crate::grouping::build(&files, &matches)?;
        let mut bytes = Vec::new();
        human(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &[],
        )?;
        let text = String::from_utf8(bytes)?;
        let (duplicates, candidates) = text
            .split_once("Unverified candidates (not pixel-confirmed; not duplicates)")
            .ok_or_else(|| anyhow::anyhow!("missing unverified candidates section: {text}"))?;
        assert!(duplicates.contains("score 0.950"));
        assert!(!duplicates.contains("score 0.990"));
        assert!(candidates.contains("/file-2"));
        assert!(candidates.contains("score 0.990"));
        assert!(candidates.contains(
            "3 files; 1 duplicate matches; 0 variants; 2 unverified candidates; 1 groups; 2 grouped files"
        ));
        let mut bytes = Vec::new();
        json(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &settings(),
            &[],
        )?;
        let parsed: WireReport = serde_json::from_slice(&bytes)?;
        assert_eq!(parsed.matches.len(), 3);
        assert_eq!(parsed.groups.len(), 1);
        assert_eq!(parsed.groups[0].files, vec![0, 1]);
        assert!(
            parsed.matches[1]
                .evidence
                .iter()
                .any(|evidence| evidence.signal == "unverified")
        );
        Ok(())
    }

    fn variant() -> Match {
        Match {
            a: FileId(0),
            b: FileId(2),
            relation: Relation::Variant,
            score: 0.98,
            evidence: vec![Evidence {
                signal: "composition",
                value: 0.98,
                note: "same design in a dark version".into(),
            }],
        }
    }

    #[test]
    fn json_projects_variants_and_retains_the_full_matches() -> anyhow::Result<()> {
        let (files, mut matches, groups) = fixtures();
        matches.push(variant());
        let mut bytes = Vec::new();
        json(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &settings(),
            &[],
        )?;
        let parsed: WireReport = serde_json::from_slice(&bytes)?;
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.matches.len(), 3);
        assert_eq!(parsed.matches[2].relation, "variant");
        assert_eq!(
            parsed.matches[2].evidence[0].note,
            "same design in a dark version"
        );
        assert_eq!(
            parsed.variants,
            vec![WireVariant {
                a: 0,
                b: 2,
                score: 0.98
            }]
        );
        Ok(())
    }

    #[test]
    fn human_separates_variants_without_using_them_as_duplicate_links() -> anyhow::Result<()> {
        let (files, mut matches, groups) = fixtures();
        matches.push(variant());
        let mut bytes = Vec::new();
        human(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &[],
        )?;
        let text = String::from_utf8(bytes)?;
        let (duplicates, variants) = text
            .split_once("Variants (related, not duplicates)")
            .ok_or_else(|| anyhow::anyhow!("missing variants section"))?;
        assert!(duplicates.contains("transitive to keep_hint (not directly verified)"));
        assert!(!duplicates.contains("variant of"));
        assert!(variants.contains("/file-0 — variant of /file-2; score 0.980"));
        assert!(variants.contains("composition: same design in a dark version"));
        assert!(
            variants
                .contains("3 files; 2 duplicate matches; 1 variants; 1 groups; 3 grouped files")
        );
        Ok(())
    }

    #[test]
    fn variant_only_files_are_reported_without_duplicate_groups() -> anyhow::Result<()> {
        let (files, _, _) = fixtures();
        let matches = [variant()];
        let groups = crate::grouping::build(&files, &matches)?;
        assert_eq!(groups.len(), 0);
        let mut bytes = Vec::new();
        human(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &Sources::default(),
            &[],
        )?;
        let text = String::from_utf8(bytes)?;
        assert!(text.starts_with("Variants (related, not duplicates)\n"));
        assert!(
            text.contains("3 files; 0 duplicate matches; 1 variants; 0 groups; 0 grouped files")
        );
        Ok(())
    }

    fn remote_sources() -> Sources {
        let mut sources = Sources::default();
        sources.by_id.insert(
            FileId(0),
            crate::remote::UrlInfo {
                url: "https://example.test/thumbnail.jpg".into(),
                page_url: Some("https://example.test/page".into()),
                role: "og:image".into(),
                mode: eidolon_fetch::FetchMode::Full,
                mode_reason: "complete_resource".into(),
                initial_bytes_fetched: 240,
                refinement_bytes_fetched: 0,
                bytes_fetched: 240,
                total_size: Some(240),
                budget_exhausted: false,
                complete_bytes: true,
            },
        );
        sources.transfer.requested_urls = 1;
        sources.transfer.bytes_fetched = 350;
        sources.transfer.total_size = Some(240);
        sources
    }

    #[test]
    fn json_adds_provenance_only_to_remote_file_entries() -> anyhow::Result<()> {
        let (files, matches, groups) = fixtures();
        let mut bytes = Vec::new();
        json(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &remote_sources(),
            &settings(),
            &[],
        )?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert!(value["files"][1].get("url").is_none());
        let report: WireReport = serde_json::from_slice(&bytes)?;
        let origin = report.files[0]
            .url
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("remote provenance is missing"))?;
        assert_eq!(origin.url, "https://example.test/thumbnail.jpg");
        assert_eq!(
            origin.page_url.as_deref(),
            Some("https://example.test/page")
        );
        assert_eq!(origin.role, "og:image");
        assert_eq!(origin.bytes_fetched, 240);
        assert_eq!(origin.mode, "full");
        assert_eq!(origin.mode_reason, "complete_resource");
        assert_eq!(origin.initial_bytes_fetched, origin.bytes_fetched);
        assert_eq!(origin.refinement_bytes_fetched, 0);
        assert_eq!(origin.total_size, Some(240));
        Ok(())
    }

    #[test]
    fn frame_report_uses_urls_and_roles_instead_of_remote_cache_paths() -> anyhow::Result<()> {
        let (mut files, _, _) = fixtures();
        files[1].kind = Kind::Video;
        let matches = [Match {
            a: FileId(0),
            b: FileId(1),
            relation: Relation::FrameOf,
            score: 0.96,
            evidence: vec![Evidence {
                signal: "orb",
                value: 0.96,
                note: "frame verified at 1.25 seconds".into(),
            }],
        }];
        let groups = crate::grouping::build(&files, &matches)?;
        assert_eq!(groups[0].keep_hint, FileId(1));
        let mut sources = remote_sources();
        sources.by_id.insert(
            FileId(1),
            crate::remote::UrlInfo {
                url: "https://example.test/video.mp4".into(),
                page_url: None,
                role: "video".into(),
                mode: eidolon_fetch::FetchMode::Sparse,
                mode_reason: "indexed_keyframes".into(),
                initial_bytes_fetched: 620,
                refinement_bytes_fetched: 0,
                bytes_fetched: 620,
                total_size: Some(999_760),
                budget_exhausted: false,
                complete_bytes: false,
            },
        );
        sources.transfer.requested_urls = 2;
        sources.transfer.bytes_fetched = 860;
        sources.transfer.total_size = Some(1_000_000);
        let mut bytes = Vec::new();
        human(&mut bytes, &files, &matches, &groups, &sources, &[])?;
        let text = String::from_utf8(bytes)?;
        assert!(text.contains("keep_hint: https://example.test/video.mp4 [video]"));
        assert!(
            text.contains("https://example.test/thumbnail.jpg [og:image] — frame of keep_hint")
        );
        assert!(text.contains("frame verified at 1.25 seconds"));
        assert!(!text.contains("/file-0"));
        assert!(!text.contains("/file-1"));
        assert!(
            text.contains("860 bytes fetched / 1000000 bytes total remote size; 0 failed URLs")
        );
        let mut bytes = Vec::new();
        json(
            &mut bytes,
            &files,
            &matches,
            &groups,
            &sources,
            &settings(),
            &[],
        )?;
        let report: WireReport = serde_json::from_slice(&bytes)?;
        assert_eq!(report.matches[0].relation, "frame_of");
        Ok(())
    }
}

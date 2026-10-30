//! Connected duplicate groups and deterministic keep-hint selection.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use anyhow::ensure;
use eidolon_core::{FileEntry, FileId, Kind, Match, Relation};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct Group {
    pub(crate) files: Vec<FileId>,
    pub(crate) keep_hint: FileId,
}

#[derive(Clone, Copy, Default)]
struct Quality {
    pixels: f64,
    bitrate: f64,
    length: f64,
}

/// Whether evidence explicitly marks this candidate as lacking pixel confirmation.
pub(crate) fn is_unverified(matched: &Match) -> bool {
    matched
        .evidence
        .iter()
        .any(|evidence| evidence.signal == "unverified")
}

/// Whether a match may connect duplicate groups and contribute quality hints.
pub(crate) fn is_duplicate(matched: &Match) -> bool {
    matched.relation != Relation::Variant && !is_unverified(matched)
}

pub(crate) fn build(files: &[FileEntry], matches: &[Match]) -> anyhow::Result<Vec<Group>> {
    let mut sets = UnionFind::new(files.len());
    let mut quality = vec![Quality::default(); files.len()];
    for matched in matches {
        ensure!(
            matched.a.0 < files.len() && matched.b.0 < files.len(),
            "match refers to an unknown file"
        );
        if !is_duplicate(matched) {
            continue;
        }
        sets.join(matched.a.0, matched.b.0);
        for evidence in &matched.evidence {
            if evidence.signal == "quality" {
                add_quality(&evidence.note, matched, &mut quality);
            }
        }
    }
    let mut members: BTreeMap<usize, Vec<FileId>> = BTreeMap::new();
    for file in files {
        ensure!(file.id.0 < files.len(), "file id is outside the input list");
        members
            .entry(sets.root(file.id.0))
            .or_default()
            .push(file.id);
    }
    let mut groups = Vec::new();
    for mut ids in members.into_values().filter(|ids| ids.len() > 1) {
        ids.sort_unstable();
        let keep_hint = *ids
            .iter()
            .max_by(|a, b| compare_quality(&files[a.0], &files[b.0], &quality))
            .ok_or_else(|| anyhow::anyhow!("empty duplicate group"))?;
        groups.push(Group {
            files: ids,
            keep_hint,
        });
    }
    groups.sort_unstable_by(|a, b| {
        b.files
            .len()
            .cmp(&a.files.len())
            .then_with(|| a.files.cmp(&b.files))
    });
    Ok(groups)
}

fn add_quality(note: &str, matched: &Match, quality: &mut [Quality]) {
    let Some(id) = note
        .split_whitespace()
        .find_map(|field| field.strip_prefix("file=")?.parse::<usize>().ok())
    else {
        return;
    };
    if id != matched.a.0 && id != matched.b.0 {
        return;
    }
    for field in note.split_whitespace() {
        let Some((name, value)) = field.split_once('=') else {
            continue;
        };
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        if !value.is_finite() || value < 0.0 {
            continue;
        }
        let slot = match name {
            "pixels" => &mut quality[id].pixels,
            "bitrate" => &mut quality[id].bitrate,
            "length" => &mut quality[id].length,
            _ => continue,
        };
        *slot = slot.max(value);
    }
}

fn compare_quality(a: &FileEntry, b: &FileEntry, quality: &[Quality]) -> Ordering {
    let qa = quality[a.id.0];
    let qb = quality[b.id.0];
    (a.kind == Kind::Video)
        .cmp(&(b.kind == Kind::Video))
        .then_with(|| qa.pixels.total_cmp(&qb.pixels))
        .then_with(|| qa.bitrate.total_cmp(&qb.bitrate))
        .then_with(|| qa.length.total_cmp(&qb.length))
        .then_with(|| a.size.cmp(&b.size))
        .then_with(|| b.mtime_ns.cmp(&a.mtime_ns))
        .then_with(|| b.id.cmp(&a.id))
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            rank: vec![0; count],
        }
    }

    fn root(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    fn join(&mut self, a: usize, b: usize) {
        let mut a = self.root(a);
        let mut b = self.root(b);
        if a == b {
            return;
        }
        if self.rank[a] < self.rank[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if self.rank[a] == self.rank[b] {
            self.rank[a] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use eidolon_core::{Evidence, Kind, Relation};

    use super::*;

    fn files() -> Vec<FileEntry> {
        [(100, 3), (200, 2), (300, 1), (400, 0)]
            .into_iter()
            .enumerate()
            .map(|(id, (size, mtime_ns))| FileEntry {
                id: FileId(id),
                path: format!("/file-{id}").into(),
                size,
                mtime_ns,
                kind: Kind::Image,
            })
            .collect()
    }

    fn matched(a: usize, b: usize) -> Match {
        Match {
            a: FileId(a),
            b: FileId(b),
            relation: Relation::Similar,
            score: 0.9,
            evidence: vec![Evidence {
                signal: "phash",
                value: 2.0,
                note: "hamming 2/64".into(),
            }],
        }
    }

    #[test]
    fn transitive_matches_form_one_group_without_unrelated_files() -> anyhow::Result<()> {
        let groups = build(&files(), &[matched(0, 1), matched(1, 2)])?;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files, vec![FileId(0), FileId(1), FileId(2)]);
        assert_eq!(groups[0].keep_hint, FileId(2));
        Ok(())
    }

    #[test]
    fn variant_edges_do_not_merge_duplicate_groups() -> anyhow::Result<()> {
        let mut variant = matched(1, 2);
        variant.relation = Relation::Variant;
        let groups = build(&files(), &[matched(0, 1), variant, matched(2, 3)])?;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].files, vec![FileId(0), FileId(1)]);
        assert_eq!(groups[1].files, vec![FileId(2), FileId(3)]);
        assert_eq!(groups[0].keep_hint, FileId(1));
        assert_eq!(groups[1].keep_hint, FileId(3));
        Ok(())
    }

    #[test]
    fn unverified_edges_neither_merge_groups_nor_influence_keep_hints() -> anyhow::Result<()> {
        let mut candidate = matched(0, 2);
        candidate.evidence.extend([
            Evidence {
                signal: "unverified",
                value: 1.0,
                note: "pixel-level confirmation skipped".into(),
            },
            Evidence {
                signal: "quality",
                value: 0.0,
                note: "file=0 pixels=1000000000".into(),
            },
        ]);
        let groups = build(&files(), &[matched(0, 1), candidate, matched(2, 3)])?;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].files, vec![FileId(0), FileId(1)]);
        assert_eq!(groups[1].files, vec![FileId(2), FileId(3)]);
        assert_eq!(groups[0].keep_hint, FileId(1));
        assert_eq!(groups[1].keep_hint, FileId(3));
        Ok(())
    }

    #[test]
    fn frame_matches_join_groups_and_prefer_video_over_a_larger_image() -> anyhow::Result<()> {
        let mut files = files();
        files[1].kind = Kind::Video;
        files[1].size = 1;
        let mut frame = matched(0, 1);
        frame.relation = Relation::FrameOf;
        frame.evidence.push(Evidence {
            signal: "quality",
            value: 0.0,
            note: "file=0 pixels=1000000000".into(),
        });
        let groups = build(&files, &[frame, matched(0, 2)])?;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files, vec![FileId(0), FileId(1), FileId(2)]);
        assert_eq!(groups[0].keep_hint, FileId(1));
        Ok(())
    }

    #[test]
    fn quality_evidence_beats_file_size_then_earlier_mtime_breaks_ties() -> anyhow::Result<()> {
        let mut files = files();
        let mut edge = matched(0, 1);
        edge.evidence.push(Evidence {
            signal: "quality",
            value: 0.0,
            note: "file=0 pixels=1000000".into(),
        });
        edge.evidence.push(Evidence {
            signal: "quality",
            value: 0.0,
            note: "file=1 pixels=250000".into(),
        });
        assert_eq!(build(&files, &[edge])?[0].keep_hint, FileId(0));
        files[0].size = files[1].size;
        assert_eq!(build(&files, &[matched(0, 1)])?[0].keep_hint, FileId(1));
        Ok(())
    }

    #[test]
    fn groups_sort_by_member_count_descending() -> anyhow::Result<()> {
        let mut files = files();
        files.push(FileEntry {
            id: FileId(4),
            path: "/fifth".into(),
            size: 1,
            mtime_ns: 0,
            kind: Kind::Text,
        });
        let groups = build(&files, &[matched(0, 1), matched(2, 3), matched(3, 4)])?;
        assert_eq!(
            groups
                .iter()
                .map(|group| group.files.len())
                .collect::<Vec<_>>(),
            vec![3, 2]
        );
        Ok(())
    }
}

//! Locality-sensitive frame indexing, followed by cheap signal verification.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use rayon::prelude::*;

use crate::frames::{Frame, Video, cosine, hash_similarity};

const HASH_TABLES: usize = 12;
const REGION_TABLES: usize = 4;
const FRAME_TABLES: usize = HASH_TABLES + REGION_TABLES * 4;
const COLOR_TABLES: usize = HASH_TABLES;
const EMBED_TABLES: usize = 8;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(crate) struct FrameRef {
    pub(crate) video: usize,
    pub(crate) frame: usize,
}

pub(crate) struct Anchor {
    pub(crate) a_time: f64,
    pub(crate) b_time: f64,
    pub(crate) similarity: f64,
    pub(crate) embedded: bool,
}

pub(crate) type Candidates = HashMap<(usize, usize), Vec<Anchor>>;

pub(crate) struct Index {
    buckets: HashMap<u32, Vec<FrameRef>>,
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn hash_keys(hash: u64) -> [u32; HASH_TABLES] {
    static PROJECTIONS: LazyLock<[[u32; 16]; HASH_TABLES]> = LazyLock::new(|| {
        let mut state = 0x8b41_273f_de51_640a;
        std::array::from_fn(|_| {
            let mut used = 0_u64;
            std::array::from_fn(|_| {
                loop {
                    let position = (random(&mut state) % 63) as u32;
                    if used & (1 << position) == 0 {
                        used |= 1 << position;
                        break position;
                    }
                }
            })
        })
    });
    std::array::from_fn(|table| {
        let key = PROJECTIONS[table]
            .iter()
            .enumerate()
            .fold(0, |key, (bit, position)| {
                key | (((hash >> position) & 1) as u32) << bit
            });
        ((table as u32) << 16) | key
    })
}

fn embedding_keys(vector: &[f32]) -> [u32; EMBED_TABLES] {
    std::array::from_fn(|table| {
        let mut state = 0x43a8_57ed_3216_f90b_u64 ^ ((table as u64 + 1) * 0x11ba_6157);
        let mut key = 0;
        for bit in 0..12 {
            let mut projection = 0.0;
            for _ in 0..8 {
                let value = random(&mut state);
                let index = value as usize % vector.len();
                projection += vector[index] * if value & (1 << 63) == 0 { -1.0 } else { 1.0 };
            }
            key |= u32::from(projection > 0.0) << bit;
        }
        (((FRAME_TABLES + COLOR_TABLES + table) as u32) << 16) | key
    })
}

fn basic_keys(frame: &Frame) -> [u32; FRAME_TABLES] {
    let mut keys = [0; FRAME_TABLES];
    keys[..HASH_TABLES].copy_from_slice(&hash_keys(frame.phash));
    for (region, hash) in frame.regions.iter().enumerate() {
        for (table, key) in hash_keys(*hash).into_iter().take(REGION_TABLES).enumerate() {
            let table = HASH_TABLES + region * REGION_TABLES + table;
            keys[table] = ((table as u32) << 16) | (key & 0xffff);
        }
    }
    keys
}

fn frame_keys(frame: &Frame, color_variants: bool) -> impl Iterator<Item = u32> + '_ {
    // Channel order may change under a hue shift. Rank-normalized planes
    // share tables; these keys nominate, never verify, a color variant.
    let hashes = frame.color_hashes.as_ref().filter(|_| color_variants);
    basic_keys(frame)
        .into_iter()
        .chain(hashes.into_iter().flatten().flat_map(|hash| {
            hash_keys(*hash)
                .into_iter()
                .enumerate()
                .map(|(table, key)| (((FRAME_TABLES + table) as u32) << 16) | (key & 0xffff))
        }))
}

fn query_keys(frame: &Frame, color_variants: bool) -> impl Iterator<Item = u32> + '_ {
    frame_keys(frame, color_variants)
        .enumerate()
        .flat_map(|(index, key)| {
            // A hue rotation may reverse a channel's contrast. Query both
            // signs instead of choosing a noise-sensitive canonical sign.
            // Probe one flipped projection bit to tolerate codec/tone changes
            // without weakening the independent RGB verification.
            let probes = if index >= FRAME_TABLES { 34 } else { 1 };
            (0..probes).map(move |probe| {
                let signed = if probe >= 17 { key ^ 0xffff } else { key };
                let bit = probe % 17;
                signed ^ if bit == 0 { 0 } else { 1 << (bit - 1) }
            })
        })
}

impl Index {
    pub(crate) fn new<'a>(
        videos: impl IntoIterator<Item = &'a Video>,
        color_variants: bool,
    ) -> Self {
        let mut index: HashMap<u32, Vec<FrameRef>> = HashMap::new();
        for (video_index, video) in videos.into_iter().enumerate() {
            let mut seen = HashSet::new();
            for (frame_index, frame) in video.frames.iter().enumerate() {
                let reference = FrameRef {
                    video: video_index,
                    frame: frame_index,
                };
                // Identical stationary frames need only one index entry, not a quadratic bucket.
                if seen.insert((
                    frame.phash,
                    frame.dhash,
                    frame.color,
                    frame.regions,
                    frame.color_hashes,
                )) {
                    for key in frame_keys(frame, color_variants) {
                        let bucket = index.entry(key).or_default();
                        if bucket.last() != Some(&reference) {
                            bucket.push(reference);
                        }
                    }
                }
                if let Some(vector) = &frame.embedding {
                    for key in embedding_keys(vector) {
                        index.entry(key).or_default().push(reference);
                    }
                }
            }
        }
        Self { buckets: index }
    }

    pub(crate) fn query(&self, frame: &Frame, hits: &mut HashSet<FrameRef>) {
        hits.clear();
        for key in basic_keys(frame).into_iter().chain(
            frame
                .embedding
                .as_ref()
                .into_iter()
                .flat_map(|vector| embedding_keys(vector)),
        ) {
            if let Some(bucket) = self.buckets.get(&key) {
                hits.extend(bucket.iter().copied());
            }
        }
    }
}

pub(crate) fn generate(videos: &[Video], fast: bool) -> Candidates {
    let index = Index::new(videos, true);
    let embedded = fast.then(|| {
        videos
            .iter()
            .map(|video| video.frames.iter().any(|frame| frame.embedding.is_some()))
            .collect::<Vec<_>>()
    });
    videos
        .par_iter()
        .enumerate()
        .flat_map_iter(|(video_index, video)| {
            let mut candidates = HashMap::new();
            let mut hits = HashSet::new();
            let mut seen = HashSet::new();
            for frame in &video.frames {
                hits.clear();
                if seen.insert((
                    frame.phash,
                    frame.dhash,
                    frame.color,
                    frame.regions,
                    frame.color_hashes,
                )) {
                    for key in query_keys(frame, true) {
                        if let Some(bucket) = index.buckets.get(&key) {
                            hits.extend(
                                bucket
                                    .iter()
                                    .filter(|reference| reference.video < video_index)
                                    .copied(),
                            );
                        }
                    }
                }
                if let Some(vector) = &frame.embedding {
                    for key in embedding_keys(vector) {
                        if let Some(bucket) = index.buckets.get(&key) {
                            hits.extend(
                                bucket
                                    .iter()
                                    .filter(|reference| reference.video < video_index)
                                    .copied(),
                            );
                        }
                    }
                }
                for reference in &hits {
                    let other = &videos[reference.video].frames[reference.frame];
                    let candidate = if embedded
                        .as_ref()
                        .is_some_and(|embedded| embedded[video_index] && embedded[reference.video])
                    {
                        nominate(other, frame)
                    } else {
                        similarity(other, frame, fast)
                    };
                    if let Some((similarity, embedded)) = candidate {
                        candidates
                            .entry((reference.video, video_index))
                            .or_insert_with(Vec::new)
                            .push(Anchor {
                                a_time: other.time,
                                b_time: frame.time,
                                similarity,
                                embedded,
                            });
                    }
                }
            }
            candidates
        })
        .collect()
}

// Fast nomination avoids global color fitting when both timelines have an
// embedding route. Temporal verification still requires its normal pixel/color
// proof; a semantic candidate is not an accepted match.
fn nominate(a: &Frame, b: &Frame) -> Option<(f64, bool)> {
    if let Some(similarity) =
        hash_similarity(a, b, true).or_else(|| crate::appearance::aligned(a, b))
    {
        return Some((similarity, false));
    }
    let similarity = cosine(a.embedding.as_ref()?, b.embedding.as_ref()?)?;
    (similarity >= 0.95).then_some((similarity, true))
}

pub(crate) fn similarity(a: &Frame, b: &Frame, fast: bool) -> Option<(f64, bool)> {
    if let Some(similarity) = hash_similarity(a, b, fast) {
        return Some((similarity, false));
    }
    if let Some(similarity) = crate::appearance::aligned(a, b) {
        return Some((similarity, false));
    }
    if let Some(crate::appearance::ColorChange::Variant(similarity)) =
        crate::appearance::color_change(a, b)
    {
        return Some((similarity, false));
    }
    let similarity = cosine(a.embedding.as_ref()?, b.embedding.as_ref()?)?;
    (similarity >= 0.95).then_some((similarity, true))
}

//! Unicode-aware text duplicate and excerpt detection.
//!
//! UTF-8 (lossy if invalid) and BOM-marked UTF-16 inputs use NFKC, full case
//! folding and collapsed whitespace for `SameContent`. Punctuation remains
//! significant for equality, but is stripped for similarity shingles.
//!
//! Similarity uses five-word shingles, or twelve-character shingles for short
//! or predominantly unspaced CJK text. 128 `MinHash` permutations with 32 four-row
//! LSH bands propose similar pairs. A rare-anchor inverted index additionally
//! proposes containment pairs, which ordinary Jaccard LSH would miss at low
//! length ratios. Only exact set intersection verifies a candidate.
//!
//! Jaccard of at least 0.78 means `Edited`; at least 0.50 means `Similar`.
//! `ExcerptOf` needs at least 0.92 containment and at most 0.80 length ratio.
//! Below 20 words (80 alphanumeric characters for character mode), only
//! normalized equality or a verbatim excerpt is accepted. Approximate excerpts
//! need 50 words or 160 characters, plus 32 distinct shared shingles. Verbatim
//! word excerpts need 15 words, 11 distinct shingles and a contiguous normalized
//! span in a whole of at least 50 words. Common anchors occurring in more than
//! max(32, files/10) documents are ignored before selecting up to 64 anchors per
//! document, so shared boilerplate cannot crowd out discriminative anchors.
//!
//! Complete typed alignment is `Variant` only with a changed explicitly labelled
//! identifier/amount or ID-like data in bounded HTML form fields, not plain numbers.
//! HTML name/address replacements require a changed ID/amount and fixed visible prose.
//! Variants are related documents, not duplicate edges. Normalized equality still wins;
//! ordinary prose edits, footer number/date updates and excerpts retain similarity labels.

mod fields;
mod template;

use std::collections::{BTreeSet, HashMap};

use eidolon_core::{Ctx, Evidence, FileEntry, Match, Relation};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use unicode_casefold::UnicodeCaseFold;
use unicode_normalization::UnicodeNormalization;

const FEATURE: &str = "text.shingles/v4";
const PERMUTATIONS: usize = 128;
const ANCHORS: usize = 64;

#[derive(Serialize, Deserialize)]
struct Features {
    normalized: String,
    shingles: Vec<u64>,
    signature: Vec<u64>,
    words: usize,
    characters: usize,
    character_mode: bool,
    fields: Vec<fields::Field>,
}

impl Features {
    fn extract(text: &str) -> Self {
        let normalized = normalize(text);
        let clean: String = normalized
            .chars()
            .filter(|character| {
                character.is_alphanumeric()
                    || character.is_whitespace()
                    || unicode_normalization::char::is_combining_mark(*character)
            })
            .collect();
        let words: Vec<_> = clean.split_whitespace().collect();
        let characters = clean
            .chars()
            .filter(|character| character.is_alphanumeric())
            .count();
        let cjk = clean.chars().filter(|character| matches!(u32::from(*character), 0x3040..=0x30ff | 0x3400..=0x9fff | 0xac00..=0xd7af | 0x20000..=0x3134f)).count();
        let character_mode = words.len() < 5 || cjk * 3 > characters;
        let mut shingles = if character_mode {
            let chars: Vec<_> = clean
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect();
            chars
                .windows(chars.len().clamp(1, 12))
                .map(|window| {
                    hash_bytes(
                        window
                            .iter()
                            .flat_map(|character| u32::from(*character).to_le_bytes()),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            words
                .windows(5)
                .map(|window| {
                    hash_bytes(
                        window
                            .iter()
                            .flat_map(|word| word.bytes().chain(std::iter::once(0))),
                    )
                })
                .collect::<Vec<_>>()
        };
        shingles.sort_unstable();
        shingles.dedup();
        let signature = (0..PERMUTATIONS)
            .map(|permutation| {
                let seed = mix(permutation as u64);
                shingles
                    .iter()
                    .map(|value| mix(*value ^ seed))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect();
        let word_count = words.len();
        let fields = fields::extract(&normalized);
        Self {
            normalized,
            shingles,
            signature,
            words: word_count,
            characters,
            character_mode,
            fields,
        }
    }

    fn substantial(&self, excerpt: bool) -> bool {
        if self.character_mode {
            self.characters >= if excerpt { 160 } else { 80 }
        } else {
            self.words >= if excerpt { 50 } else { 20 }
        }
    }

    fn verbatim_excerpt(&self) -> bool {
        !self.character_mode && self.words >= 15 && self.shingles.len() >= 11
    }

    fn length(&self) -> usize {
        if self.character_mode {
            self.characters
        } else {
            self.words
        }
    }
}

/// Find near-duplicate text files without modifying the inputs.
///
/// Per-file extraction uses `config.jobs` workers (zero means all cores).
/// Versioned cached features include normalized text, sorted shingle hashes and
/// `MinHash` signatures. Matches meet `config.min_score`, carry measured evidence,
/// and use canonical relation direction. Empty normalized texts are ignored.
///
/// # Errors
/// Returns [`eidolon_core::InputError`] for unreadable files.
/// Cache serialization/database and worker pool failures remain run-wide errors.
pub fn find(files: &[FileEntry], ctx: &Ctx<'_>) -> anyhow::Result<Vec<Match>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.config.jobs)
        .build()?;
    let features: Vec<Features> = pool.install(|| {
        files
            .par_iter()
            .map(|file| {
                let bytes = ctx.cache.get_or_compute(file, FEATURE, || {
                    let bytes =
                        std::fs::read(&file.path).map_err(|error| eidolon_core::InputError {
                            file: file.id,
                            reason: format!("reading {}: {error}", file.path.display()),
                        })?;
                    Ok(serde_json::to_vec(&Features::extract(&decode(&bytes)))?)
                })?;
                Ok(serde_json::from_slice(&bytes)?)
            })
            .collect::<anyhow::Result<_>>()
    })?;
    let mut matches = Vec::new();
    for (i, j) in candidates(&features) {
        if let Some(mut found) = verify(&features[i], &features[j], &files[i], &files[j]) {
            if found.score < ctx.config.min_score {
                continue;
            }
            for index in [i, j] {
                let length = features[index].normalized.chars().count();
                found.evidence.push(Evidence {
                    signal: "quality",
                    value: length as f64,
                    note: format!("file={} length={length}", files[index].id.0),
                });
            }
            matches.push(found);
        }
    }
    matches.sort_unstable_by_key(|found| (found.a, found.b));
    Ok(matches)
}

fn candidates(features: &[Features]) -> BTreeSet<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    let mut equals: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut bands: HashMap<(usize, [u64; 4]), Vec<usize>> = HashMap::new();
    for (index, feature) in features.iter().enumerate() {
        if feature.normalized.is_empty() {
            continue;
        }
        let equal = equals.entry(&feature.normalized).or_default();
        pairs.extend(equal.iter().map(|&other| (other, index)));
        equal.push(index);
        if !feature.substantial(false) && !feature.verbatim_excerpt() {
            continue;
        }
        for (band, chunk) in feature.signature.as_chunks::<4>().0.iter().enumerate() {
            let key = (band, *chunk);
            let bucket = bands.entry(key).or_default();
            pairs.extend(bucket.iter().map(|&other| (other, index)));
            bucket.push(index);
        }
    }
    // Count before selecting anchors, so common boilerplate cannot occupy every
    // slot. Singletons need no posting allocation; selected keys stay bounded
    // by ANCHORS per document and postings by common_limit.
    let common_limit = 32.max(features.len() / 10);
    let mut frequencies = HashMap::new();
    for feature in features {
        for hash in &feature.shingles {
            *frequencies.entry(*hash).or_insert(0_usize) += 1;
        }
    }
    let mut anchors = HashMap::new();
    for feature in features {
        if !feature.substantial(false) && !feature.verbatim_excerpt() {
            continue;
        }
        for hash in feature
            .shingles
            .iter()
            .filter(|hash| (2..=common_limit).contains(&frequencies[hash]))
            .take(ANCHORS)
        {
            anchors
                .entry(*hash)
                .or_insert_with(|| Vec::with_capacity(frequencies[hash]));
        }
    }
    drop(frequencies);
    for (index, feature) in features.iter().enumerate() {
        for hash in &feature.shingles {
            if let Some(documents) = anchors.get_mut(hash) {
                documents.push(index);
            }
        }
    }
    let mut votes = HashMap::new();
    for (index, feature) in features.iter().enumerate() {
        if !feature.substantial(false) && !feature.verbatim_excerpt() {
            continue;
        }
        votes.clear();
        for documents in feature
            .shingles
            .iter()
            .filter_map(|hash| anchors.get(hash))
            .take(ANCHORS)
        {
            for &other in documents {
                if index != other {
                    let count = votes.entry(other).or_insert(0);
                    *count += 1;
                    if *count == 3 {
                        pairs.insert((index.min(other), index.max(other)));
                    }
                }
            }
        }
    }
    pairs
}

fn verify(first: &Features, second: &Features, a: &FileEntry, b: &FileEntry) -> Option<Match> {
    let (mut a_id, mut b_id) = if a.id < b.id {
        (a.id, b.id)
    } else {
        (b.id, a.id)
    };
    let (relation, score, evidence) = if first.normalized == second.normalized {
        (
            Relation::SameContent,
            1.0,
            vec![Evidence {
                signal: "normalized_text",
                value: 1.0,
                note: "NFKC, full case folding and whitespace normalization are equal".into(),
            }],
        )
    } else {
        if first.character_mode != second.character_mode {
            return None;
        }
        let shared = intersection(&first.shingles, &second.shingles);
        let union = first.shingles.len() + second.shingles.len() - shared;
        if union == 0 {
            return None;
        }
        let jaccard = shared as f64 / union as f64;
        let (part, whole, part_id, whole_id) = if first.length() <= second.length() {
            (first, second, a.id, b.id)
        } else {
            (second, first, b.id, a.id)
        };
        let containment = shared as f64 / part.shingles.len().max(1) as f64;
        let ratio = part.length() as f64 / whole.length().max(1) as f64;
        let mut evidence = vec![
            Evidence {
                signal: "text_jaccard",
                value: jaccard,
                note: format!("{shared} shared / {union} union hashed shingles"),
            },
            Evidence {
                signal: "text_containment",
                value: containment,
                note: format!(
                    "{shared} shared / {} part shingles; length ratio {ratio:.4}",
                    part.shingles.len()
                ),
            },
        ];
        let verbatim = part.verbatim_excerpt()
            && whole.substantial(true)
            && shared == part.shingles.len()
            && whole
                .normalized
                .match_indices(&part.normalized)
                .any(|(start, matched)| {
                    let end = start + matched.len();
                    (start == 0 || whole.normalized.as_bytes()[start - 1] == b' ')
                        && (end == whole.normalized.len()
                            || whole.normalized.as_bytes()[end] == b' ')
                });
        if first.substantial(false) && second.substantial(false) && {
            let typed = template::same_template(&first.normalized, &second.normalized);
            typed.as_ref().is_some_and(|proof| proof.labelled_change)
                || fields::same_template(
                    &first.normalized,
                    &first.fields,
                    &second.normalized,
                    &second.fields,
                    typed.as_ref().is_some_and(|proof| proof.identity_change),
                )
        } {
            evidence.push(Evidence {
                signal: "text_template_alignment",
                value: 1.0,
                note: "same template, different data".into(),
            });
            (Relation::Variant, 1.0, evidence)
        } else if ratio <= 0.80
            && (verbatim || (containment >= 0.92 && part.substantial(true) && shared >= 32))
        {
            a_id = part_id;
            b_id = whole_id;
            (Relation::ExcerptOf, containment, evidence)
        } else if !first.substantial(false) || !second.substantial(false) {
            return None;
        } else if jaccard >= 0.78 {
            (Relation::Edited, jaccard, evidence)
        } else if jaccard >= 0.50 {
            (Relation::Similar, jaccard, evidence)
        } else {
            return None;
        }
    };
    Some(Match {
        a: a_id,
        b: b_id,
        relation,
        score,
        evidence,
    })
}

fn intersection(first: &[u64], second: &[u64]) -> usize {
    let (mut left, mut right, mut count) = (0, 0, 0);
    while left < first.len() && right < second.len() {
        match first[left].cmp(&second[right]) {
            std::cmp::Ordering::Less => left += 1,
            std::cmp::Ordering::Greater => right += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                left += 1;
                right += 1;
            }
        }
    }
    count
}

fn hash_bytes(bytes: impl Iterator<Item = u8>) -> u64 {
    mix(bytes.fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    }))
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn decode(bytes: &[u8]) -> String {
    let little_endian = bytes.starts_with(&[0xff, 0xfe]);
    if little_endian || bytes.starts_with(&[0xfe, 0xff]) {
        let mut text: String =
            char::decode_utf16(bytes[2..].as_chunks::<2>().0.iter().map(|pair| {
                if little_endian {
                    u16::from_le_bytes([pair[0], pair[1]])
                } else {
                    u16::from_be_bytes([pair[0], pair[1]])
                }
            }))
            .map(|result| result.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect();
        if !bytes.len().is_multiple_of(2) {
            text.push(char::REPLACEMENT_CHARACTER);
        }
        text
    } else {
        String::from_utf8_lossy(bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes))
            .into_owned()
    }
}

fn normalize(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut space = false;
    for character in text.nfkc().case_fold().nfkc() {
        if character.is_whitespace() {
            space = !normalized.is_empty();
        } else {
            if space {
                normalized.push(' ');
            }
            normalized.push(character);
            space = false;
        }
    }
    normalized
}

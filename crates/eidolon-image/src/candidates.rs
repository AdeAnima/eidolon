//! Exact Hamming retrieval followed by bounded, monotone candidate neighborhoods.

use std::collections::{BTreeSet, HashMap};

use rayon::prelude::*;

use crate::features::Retrieval;

#[derive(Clone, Copy)]
struct HashEntry {
    image: usize,
    hash: u64,
}

struct HashIndex {
    entries: Vec<HashEntry>,
    buckets: Vec<Vec<usize>>,
}

impl HashIndex {
    fn new(hashes: impl IntoIterator<Item = (usize, u64)>) -> Self {
        let entries = hashes
            .into_iter()
            .map(|(image, hash)| HashEntry { image, hash })
            .collect::<Vec<_>>();
        let mut buckets = vec![Vec::new(); 8 * 256];
        for (entry, value) in entries.iter().enumerate() {
            for (band, byte) in value.hash.to_le_bytes().into_iter().enumerate() {
                buckets[band * 256 + usize::from(byte)].push(entry);
            }
        }
        Self { entries, buckets }
    }

    fn search(
        &self,
        hash: u64,
        radius: u32,
        owner: usize,
        visited: &mut [u8],
        generation: u8,
        mut visit: impl FnMut(&HashEntry, u32, bool),
    ) {
        debug_assert!(radius <= 20);
        // Norouzi et al.: if every substring misses its probe radius, the full
        // distance is at least sum(probe_radius + 1) = radius + 1. The first
        // radius % 8 + 1 bands probe one bit farther than the remaining bands.
        for (band, byte) in hash.to_le_bytes().into_iter().enumerate() {
            let minimum_miss = radius / 8 + u32::from(band as u32 <= radius % 8);
            let probes = match minimum_miss {
                0 => 0,
                1 => 1,
                2 => 9,
                _ => 37,
            };
            for mask in &BYTE_MASKS[..probes] {
                let bucket = &self.buckets[band * 256 + usize::from(byte ^ mask)];
                for &index in bucket {
                    let entry = &self.entries[index];
                    if entry.image == owner || visited[index] == generation {
                        continue;
                    }
                    visited[index] = generation;
                    let difference = hash ^ entry.hash;
                    let distance = difference.count_ones();
                    if distance <= radius {
                        visit(entry, distance, difference.to_le_bytes().contains(&0));
                    }
                }
            }
        }
    }
}

const BYTE_MASKS: [u8; 37] = {
    let mut masks = [0; 37];
    let mut next = 1;
    let mut first = 0;
    while first < 8 {
        masks[next] = 1 << first;
        next += 1;
        first += 1;
    }
    first = 0;
    while first < 8 {
        let mut second = first + 1;
        while second < 8 {
            masks[next] = (1 << first) | (1 << second);
            next += 1;
            second += 1;
        }
        first += 1;
    }
    masks
};

#[derive(Clone, Copy)]
struct Distances {
    exact: u32,
    byte_eligible: Option<u32>,
}

fn observe(neighbors: &mut HashMap<usize, Distances>, image: usize, distance: u32, byte: bool) {
    let distances = neighbors.entry(image).or_insert(Distances {
        exact: distance,
        byte_eligible: None,
    });
    distances.exact = distances.exact.min(distance);
    if byte {
        distances.byte_eligible = Some(distances.byte_eligible.unwrap_or(distance).min(distance));
    }
}

fn retain_nearest(
    neighbors: impl Iterator<Item = (usize, Distances)>,
    limit: usize,
    image: usize,
    pairs: &mut BTreeSet<(usize, usize)>,
) {
    let mut exact = Vec::new();
    let mut previous = Vec::new();
    for (other, distances) in neighbors {
        exact.push((distances.exact, other));
        if let Some(distance) = distances.byte_eligible {
            previous.push((distance, other));
        }
    }
    // Retain both rankings: newly found close neighbors must not evict a
    // previously retrieved pair from a crowded, capped neighborhood.
    for mut neighbors in [exact, previous] {
        if neighbors.len() > limit {
            neighbors.select_nth_unstable(limit);
            neighbors.truncate(limit);
        }
        for (_, other) in neighbors {
            pairs.insert((image.min(other), image.max(other)));
        }
    }
}

pub(super) fn generate(
    features: &[Retrieval],
    frame_embeddings: &[Option<&[f32]>],
) -> Vec<(usize, usize)> {
    let full = HashIndex::new(
        features
            .iter()
            .enumerate()
            .map(|(index, image)| (index, image.visual.phash)),
    );
    let gradient = HashIndex::new(
        features
            .iter()
            .enumerate()
            .map(|(index, image)| (index, image.visual.dhash)),
    );
    let regions =
        HashIndex::new(features.iter().enumerate().flat_map(|(index, image)| {
            image.visual.regions.iter().map(move |hash| (index, *hash))
        }));
    features
        .par_iter()
        .enumerate()
        .map(|(index, image)| {
            let mut pairs = BTreeSet::new();
            let mut full_neighbors = HashMap::new();
            let mut regions_by_shape = [HashMap::new(), HashMap::new()];
            let mut visited = vec![0; regions.entries.len().max(features.len())];
            for (generation, (table, hash, radius, region)) in [
                (&full, image.visual.phash, 14, false),
                (&gradient, image.visual.dhash, 12, false),
                (&regions, image.visual.phash, 20, true),
                (&full, !image.visual.phash, 14, false),
                (&gradient, !image.visual.dhash, 12, false),
                (&regions, !image.visual.phash, 20, true),
            ]
            .into_iter()
            .enumerate()
            {
                table.search(
                    hash,
                    radius,
                    index,
                    &mut visited,
                    generation as u8 + 1,
                    |entry, distance, byte| {
                        let neighbors = if region {
                            let other = &features[entry.image].visual;
                            let same_shape = image.visual.width == other.width
                                && image.visual.height == other.height;
                            &mut regions_by_shape[usize::from(same_shape)]
                        } else {
                            &mut full_neighbors
                        };
                        observe(neighbors, entry.image, distance, byte);
                    },
                );
            }
            retain_nearest(full_neighbors.into_iter(), 64, index, &mut pairs);
            // Same-size encodings must not consume the containment neighborhood.
            for neighbors in regions_by_shape {
                retain_nearest(neighbors.into_iter(), 48, index, &mut pairs);
            }
            if let Some(vector) = &image.embedding {
                for other in super::embedding_neighbors(
                    features
                        .iter()
                        .map(|candidate| candidate.embedding.as_deref()),
                    index,
                    vector,
                ) {
                    pairs.insert((index.min(other), index.max(other)));
                }
            }
            if let Some(Some(vector)) = frame_embeddings.get(index) {
                for other in
                    super::embedding_neighbors(frame_embeddings.iter().copied(), index, vector)
                {
                    pairs.insert((index.min(other), index.max(other)));
                }
            }
            pairs
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use super::{Distances, HashIndex, retain_nearest};

    fn feature(phash: u64, dhash: u64, regions: Vec<u64>) -> crate::features::Retrieval {
        crate::features::Retrieval {
            visual: crate::features::RetrievalVisual {
                width: 1,
                height: 1,
                phash,
                dhash,
                regions,
            },
            orb: None,
            embedding: None,
            proof: None,
        }
    }

    #[test]
    fn existing_frame_embeddings_add_candidates_without_displacing_still_neighbors() {
        let mut features = vec![
            feature(0, 0, vec![]),
            feature(0x5555_5555_5555_5555, 0x5555_5555_5555_5555, vec![]),
            feature(0x3333_3333_3333_3333, 0x3333_3333_3333_3333, vec![]),
        ];
        features[0].embedding = Some(vec![0.0, 1.0]);
        features[2].embedding = Some(vec![0.0, 1.0]);
        let first = [1.0, 0.0];
        let unrelated = [0.0, 1.0];
        let supplemental = [
            Some(first.as_slice()),
            Some(first.as_slice()),
            Some(unrelated.as_slice()),
        ];
        let baseline = super::generate(&features, &[]);
        let mixed = super::generate(&features, &supplemental);
        assert_eq!(baseline, [(0, 2)]);
        assert!(baseline.iter().all(|pair| mixed.contains(pair)));
        assert!(mixed.contains(&(0, 1)), "frame neighbors must be nominated");
        assert!(
            !mixed.contains(&(1, 2)),
            "different frame content is not a neighbor"
        );
    }

    #[test]
    fn candidate_generation_obeys_independent_hash_and_region_radii_and_complements() {
        let far = 0x5555_5555_5555_5555;
        for (phash, dhash, regions, expected) in [
            (0x0303_0303_0303_0101, far, vec![], true),
            (!0x0303_0303_0303_0101, far, vec![], true),
            (far, 0x0101_0303_0303_0101, vec![], true),
            (far, !0x0101_0303_0303_0101, vec![], true),
            (far, far, vec![0x0707_0707_0303_0303], true),
            (far, far, vec![!0x0707_0707_0303_0303], true),
            (0x0703_0303_0303_0101, far, vec![], false),
            (far, 0x0301_0303_0303_0101, vec![], false),
            (far, far, vec![0x0f07_0707_0303_0303], false),
        ] {
            let features = [feature(0, 0, vec![]), feature(phash, dhash, regions)];
            assert_eq!(
                super::generate(&features, &[]),
                if expected { vec![(0, 1)] } else { vec![] }
            );
        }
    }

    fn search(index: &HashIndex, query: u64, radius: u32) -> BTreeSet<usize> {
        let mut result = BTreeSet::new();
        let mut visited = vec![0; index.entries.len()];
        index.search(query, radius, usize::MAX, &mut visited, 1, |entry, _, _| {
            assert!(
                result.insert(entry.image),
                "each indexed hash is visited once"
            );
        });
        result
    }

    #[test]
    fn retrieves_hashes_with_no_equal_byte_at_each_detector_radius() {
        // Every byte differs. The old exact-byte lookup cannot retrieve any pair.
        for (radius, difference) in [
            (12, 0x0101_0303_0303_0101_u64),
            (14, 0x0303_0303_0303_0101),
            (20, 0x0707_0707_0303_0303),
        ] {
            assert_eq!(difference.count_ones(), radius);
            let index = HashIndex::new([(0, 0), (1, difference)]);
            assert_eq!(search(&index, 0, radius), BTreeSet::from([0, 1]));
            assert_eq!(search(&index, 0, radius - 1), BTreeSet::from([0]));
        }
    }

    #[test]
    fn range_search_equals_exhaustive_popcount_including_threshold_boundaries() {
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut hashes = Vec::new();
        for _ in 0..512 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            hashes.push(state);
        }
        for distance in 0..=21 {
            let flips = (0..distance).fold(0_u64, |bits, bit| bits | (1 << (bit * 29 % 64)));
            hashes.push(hashes[0] ^ flips);
        }
        let index = HashIndex::new(hashes.iter().copied().enumerate());
        for query in [hashes[0], hashes[17], !hashes[0], 0, u64::MAX] {
            for radius in [0, 1, 7, 8, 12, 14, 20] {
                let expected = hashes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, hash)| {
                        ((query ^ hash).count_ones() <= radius).then_some(index)
                    })
                    .collect();
                assert_eq!(search(&index, query, radius), expected);
            }
        }
    }

    #[test]
    fn exact_neighbors_never_evict_previously_retrieved_crowded_neighbors() {
        let mut neighbors = HashMap::new();
        for image in 0..160 {
            let byte_eligible = image >= 80;
            neighbors.insert(
                image,
                Distances {
                    exact: if byte_eligible { 14 } else { 8 },
                    byte_eligible: byte_eligible.then_some(14),
                },
            );
        }
        let mut pairs = BTreeSet::new();
        retain_nearest(neighbors.into_iter(), 64, 160, &mut pairs);
        assert_eq!(pairs.len(), 128);
        for old_neighbor in 80..144 {
            assert!(pairs.contains(&(old_neighbor, 160)));
        }
        for exact_neighbor in 0..64 {
            assert!(pairs.contains(&(exact_neighbor, 160)));
        }
    }
}

//! Exact, bidirectional ORB descriptor matching without duplicate distance work.

use anyhow::ensure;

use crate::features::Orb;

#[derive(Clone, Copy)]
struct Neighbors {
    first: (u32, usize),
    second: u32,
}

impl Neighbors {
    fn new() -> Self {
        // A best distance must be <=64. Values above 80 cannot affect its
        // strict 0.80 ratio test, so larger second distances need not be retained.
        Self {
            first: (81, usize::MAX),
            second: 81,
        }
    }

    fn push(&mut self, distance: u32, index: usize) {
        if distance < self.first.0 {
            self.second = self.first.0;
            self.first = (distance, index);
        } else if distance < self.second {
            self.second = distance;
        }
    }

    fn accepted(self) -> Option<usize> {
        (self.first.0 <= 64 && 5 * self.first.0 < 4 * self.second).then_some(self.first.1)
    }
}

fn packed(orb: &Orb) -> &[[u64; 4]] {
    orb.packed.get_or_init(|| {
        orb.descriptors
            .as_chunks::<32>()
            .0
            .iter()
            .map(|descriptor| {
                std::array::from_fn(|index| {
                    let start = index * 8;
                    u64::from_ne_bytes(
                        descriptor[start..start + 8]
                            .try_into()
                            .expect("eight bytes"),
                    )
                })
            })
            .collect()
    })
}

/// Match both descriptor directions with the same exact Hamming/Lowe criteria.
///
/// The cached byte layout must contain one 32-byte descriptor per keypoint.
pub(crate) fn mutual(a: &Orb, b: &Orb) -> anyhow::Result<Vec<(usize, usize)>> {
    ensure!(
        a.descriptors.len() == a.points.len() * 32 && b.descriptors.len() == b.points.len() * 32,
        "invalid cached ORB descriptor dimensions"
    );
    if a.points.len() < 2 || b.points.len() < 2 {
        return Ok(Vec::new());
    }
    let source = packed(a);
    let target = packed(b);
    let mut forward = vec![Neighbors::new(); source.len()];
    let mut backward = vec![Neighbors::new(); target.len()];
    for (query, (first, neighbors)) in source.iter().zip(&mut forward).enumerate() {
        for (train, (second, reverse)) in target.iter().zip(&mut backward).enumerate() {
            observe(first, second, neighbors, reverse, query, train);
        }
    }
    Ok(collect(forward, &backward))
}

fn observe(
    first: &[u64; 4],
    second: &[u64; 4],
    forward: &mut Neighbors,
    backward: &mut Neighbors,
    query: usize,
    train: usize,
) {
    // A partial Hamming distance is a lower bound on the full one. Distances
    // at or above both second neighbours cannot change either ratio test.
    let limit = forward.second.max(backward.second);
    let mut distance = (first[0] ^ second[0]).count_ones() + (first[1] ^ second[1]).count_ones();
    if distance >= limit {
        return;
    }
    distance += (first[2] ^ second[2]).count_ones();
    if distance >= limit {
        return;
    }
    distance += (first[3] ^ second[3]).count_ones();
    forward.push(distance, train);
    backward.push(distance, query);
}

fn collect(forward: Vec<Neighbors>, backward: &[Neighbors]) -> Vec<(usize, usize)> {
    forward
        .into_iter()
        .enumerate()
        .filter_map(|(query, neighbors)| {
            let train = neighbors.accepted()?;
            (backward[train].accepted() == Some(query)).then_some((query, train))
        })
        .collect()
}

/// Run-local inverted byte bands for fast descriptor candidate retrieval.
///
/// The persistent ORB representation is unchanged. Positions fit in sixteen
/// bits; unusually large descriptor sets use the exhaustive matcher instead.
pub(crate) struct ByteIndex {
    offsets: Vec<[u16; 257]>,
    positions: Vec<u16>,
}

impl ByteIndex {
    fn build(descriptors: &[u8]) -> Self {
        let descriptors = descriptors.as_chunks::<32>().0;
        let mut offsets = vec![[0_u16; 257]; 32];
        let mut positions = vec![0_u16; descriptors.len() * 32];
        for (band, offsets) in offsets.iter_mut().enumerate() {
            for descriptor in descriptors {
                offsets[usize::from(descriptor[band]) + 1] += 1;
            }
            for value in 1..257 {
                offsets[value] += offsets[value - 1];
            }
            let mut cursor = *offsets;
            for (index, descriptor) in descriptors.iter().enumerate() {
                let value = usize::from(descriptor[band]);
                positions[band * descriptors.len() + usize::from(cursor[value])] = index as u16;
                cursor[value] += 1;
            }
        }
        Self { offsets, positions }
    }

    fn bucket(&self, band: usize, value: u8, count: usize) -> &[u16] {
        let offsets = &self.offsets[band];
        let base = band * count;
        &self.positions[base + usize::from(offsets[usize::from(value)])
            ..base + usize::from(offsets[usize::from(value) + 1])]
    }
}

/// Retrieve approximate neighbours, then apply exact Hamming and mutual ratios.
///
/// Search the twelve least-crowded byte bands; a band containing more than an
/// eighth of the target descriptors is uninformative and skipped. Small sets
/// use exhaustive comparisons. Retrieval may miss correspondences; it never
/// replaces RANSAC or pixel proof.
pub(crate) fn mutual_fast(a: &Orb, b: &Orb) -> anyhow::Result<Vec<(usize, usize)>> {
    ensure!(
        a.descriptors.len() == a.points.len() * 32 && b.descriptors.len() == b.points.len() * 32,
        "invalid cached ORB descriptor dimensions"
    );
    if a.points.len() > usize::from(u16::MAX)
        || b.points.len() > usize::from(u16::MAX)
        || a.points.len() * b.points.len() < 16_384
    {
        return mutual(a, b);
    }
    let source = packed(a);
    let target = packed(b);
    let index = b
        .byte_index
        .get_or_init(|| ByteIndex::build(&b.descriptors));
    let mut forward = vec![Neighbors::new(); source.len()];
    let mut backward = vec![Neighbors::new(); target.len()];
    let mut seen = vec![u16::MAX; target.len()];
    for (query, ((first, bytes), neighbors)) in source
        .iter()
        .zip(a.descriptors.as_chunks::<32>().0)
        .zip(&mut forward)
        .enumerate()
    {
        // Crowded bands mostly repeat weak descriptor candidates. Retain the
        // twelve smallest postings and ignore individually uninformative bands.
        let mut bands: [(usize, usize); 32] =
            std::array::from_fn(|band| (index.bucket(band, bytes[band], target.len()).len(), band));
        bands.select_nth_unstable(12);
        for &(count, band) in &bands[..12] {
            if count > target.len() / 8 {
                continue;
            }
            for &train in index.bucket(band, bytes[band], target.len()) {
                let train = usize::from(train);
                if seen[train] == query as u16 {
                    continue;
                }
                seen[train] = query as u16;
                observe(
                    first,
                    &target[train],
                    neighbors,
                    &mut backward[train],
                    query,
                    train,
                );
            }
        }
    }
    Ok(collect(forward, &backward))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;
    use std::time::{Instant, UNIX_EPOCH};

    use anyhow::Context as _;
    use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, cache::FeatureCache};
    use image::imageops::FilterType;
    use opencv::{core, features, prelude::*};

    use crate::features::{Orb, Retrieval};

    fn reference(a: &Orb, b: &Orb) -> anyhow::Result<Vec<(usize, usize)>> {
        if a.points.len() < 2 || b.points.len() < 2 {
            return Ok(Vec::new());
        }
        let a_flat = core::Mat::from_slice(&a.descriptors)?;
        let b_flat = core::Mat::from_slice(&b.descriptors)?;
        let a_matrix = a_flat.reshape(1, i32::try_from(a.points.len())?)?;
        let b_matrix = b_flat.reshape(1, i32::try_from(b.points.len())?)?;
        let matcher = features::BFMatcher::create(core::NORM_HAMMING, false)?;
        let mut forward = core::Vector::<core::Vector<core::DMatch>>::new();
        let mut backward = core::Vector::<core::Vector<core::DMatch>>::new();
        matcher.knn_train_match_def(&a_matrix, &b_matrix, &mut forward, 2)?;
        matcher.knn_train_match_def(&b_matrix, &a_matrix, &mut backward, 2)?;
        let accepted = |matches: &core::Vector<core::Vector<core::DMatch>>| {
            matches
                .iter()
                .filter_map(|pair| {
                    let (first, second) = (pair.get(0).ok()?, pair.get(1).ok()?);
                    (first.distance <= 64.0 && first.distance < 0.80 * second.distance)
                        .then_some((first.query_idx as usize, first.train_idx as usize))
                })
                .collect::<Vec<_>>()
        };
        let reverse = accepted(&backward)
            .into_iter()
            .map(|(a, b)| (b, a))
            .collect::<HashSet<_>>();
        Ok(accepted(&forward)
            .into_iter()
            .filter(|pair| reverse.contains(pair))
            .collect())
    }

    fn extract(path: &Path, ctx: &Ctx<'_>) -> anyhow::Result<Retrieval> {
        let metadata = path.metadata()?;
        crate::features::extract(
            &FileEntry {
                id: FileId(0),
                path: path.to_owned(),
                size: metadata.len(),
                mtime_ns: i128::try_from(
                    metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
                )?,
                kind: Kind::Image,
            },
            ctx,
        )
    }

    #[test]
    fn matches_opencv_on_real_photo_encodings_crops_and_unrelated_scenes() -> anyhow::Result<()> {
        let temp = tempfile::Builder::new()
            .prefix("eidolon-exact-orb-")
            .tempdir_in("/tmp")?;
        let base = Path::new("/System/Library/Desktop Pictures/.thumbnails");
        let photo = crate::decode(&base.join("Big Sur Coastline.heic"))?.to_rgb8();
        let unrelated = crate::decode(&base.join("Big Sur Mountains.heic"))?.to_rgb8();
        let mut overlay = photo.clone();
        for y in photo.height() * 4 / 5..photo.height() * 9 / 10 {
            for x in photo.width() / 2..photo.width() * 9 / 10 {
                overlay.put_pixel(x, y, image::Rgb([240, 32, 160]));
            }
        }
        let inputs = [
            photo.clone(),
            image::imageops::resize(&photo, 256, 256, FilterType::Triangle),
            image::imageops::crop_imm(&photo, 20, 30, 280, 280).to_image(),
            overlay,
            unrelated,
            image::RgbImage::from_pixel(256, 256, image::Rgb([80, 80, 80])),
        ];
        let cache = FeatureCache::in_memory()?;
        let config = Config {
            timings: false,
            remote_overlay: true,
            min_score: 0.75,
            model_dir: temp.path().to_owned(),
            no_embeddings: true,
            no_keypoints: false,
            jobs: 2,
            fast: false,
            video_fps: 2.0,
        };
        let ctx = Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: None,
        };
        let images = inputs
            .iter()
            .enumerate()
            .map(|(index, image)| {
                let path = temp.path().join(format!("{index}.jpg"));
                image::codecs::jpeg::JpegEncoder::new_with_quality(
                    std::fs::File::create(&path)?,
                    30,
                )
                .encode_image(image)?;
                extract(&path, &ctx)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let source = images[0].orb.as_ref().context("source keypoints")?;
        for (index, image) in images.iter().enumerate() {
            let target = image.orb.as_ref().context("target keypoints")?;
            let started = Instant::now();
            let actual = super::mutual(source, target)?;
            let elapsed = started.elapsed();
            let expected = reference(source, target)?;
            assert_eq!(actual, expected, "input {index}");
            if index < 4 {
                assert!(
                    actual.len() >= 12,
                    "a decoded photo transform must match: {index}"
                );
            }
            eprintln!(
                "exact ORB input {index}: {} pairs, {:.3}ms",
                actual.len(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }
}

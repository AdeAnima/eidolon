//! Source-design provenance and visible full-frame version annotations.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::model::{ExpectedPair, Truth};
use crate::registration::{
    Audit, MIN_CORRELATION, MIN_COVERAGE, Prepared, SourceLink, compare_prepared,
};
use anyhow::Context as _;

#[allow(clippy::too_many_lines)] // Visibility and registration jointly decide each emitted edge.
pub(crate) fn annotate(truth: &mut Truth) -> anyhow::Result<Audit> {
    let ancestry = video_ancestry(truth);
    let designs: HashMap<_, _> = truth
        .files
        .iter()
        .filter(|file| file.kind == "image" && file.variant == "original")
        .map(|file| {
            let design = truth
                .sources
                .iter()
                .find(|source| source.retained == file.path)
                .and_then(|source| {
                    let path = Path::new(&source.origin);
                    ambiguous_collection(path)
                        .map(str::to_owned)
                        .or_else(|| source_design(path))
                })
                .unwrap_or_else(|| file.family.clone());
            (file.family.clone(), design)
        })
        .collect();
    for file in &mut truth.files {
        let base = file.family.split('/').next().unwrap_or(&file.family);
        let design = if file.kind == "video" || file.kind == "image" && base.starts_with("video-") {
            ancestry
                .get(base)
                .and_then(|family| designs.get(family))
                .cloned()
                .unwrap_or_else(|| base.to_owned())
        } else if file.kind == "image" {
            designs
                .get(base)
                .cloned()
                .unwrap_or_else(|| base.to_owned())
        } else {
            base.to_owned()
        };
        let scope = if file.kind == "text" { "text" } else { "media" };
        file.design = Some(format!("{scope}:{design}"));
    }
    let paths: HashMap<_, _> = truth
        .files
        .iter()
        .enumerate()
        .map(|(index, file)| (&file.path, index))
        .collect();
    let previous: HashSet<_> = truth
        .pairs
        .iter()
        .filter(|pair| pair.relation == "variant")
        .filter_map(|pair| Some(ordered(*paths.get(&pair.a)?, *paths.get(&pair.b)?)))
        .collect();
    truth.pairs.retain(|pair| pair.relation != "variant");
    let mut registrations = Registrations::new(truth, &ancestry)?;
    let mut annotated: HashSet<_> = truth
        .pairs
        .iter()
        .filter_map(|pair| Some(ordered(*paths.get(&pair.a)?, *paths.get(&pair.b)?)))
        .collect();
    let samples: Vec<_> = truth
        .files
        .iter()
        .map(|file| {
            if crate::visibility::full_frame(file) {
                crate::visibility::Sample::load(truth, file).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<anyhow::Result<_>>()?;
    for (a, first) in truth.files.iter().enumerate() {
        let Some(first_sample) = &samples[a] else {
            continue;
        };
        for (b, second) in truth.files.iter().enumerate().skip(a + 1) {
            if first.kind != second.kind
                || first.design != second.design
                || first.family == second.family
                || !annotated.insert((a, b))
            {
                continue;
            }
            let Some(second_sample) = &samples[b] else {
                continue;
            };
            let difference = first_sample.difference(second_sample);
            if difference < crate::visibility::MIN_VISIBLE {
                continue;
            }
            let variant = if first.family.split('/').next() != second.family.split('/').next() {
                "source_variant"
            } else if first.variant.starts_with("variant_") {
                &first.variant
            } else {
                &second.variant
            };
            if variant == "source_variant" {
                let link = registrations.link(a, b)?;
                if !link.proof.accepted {
                    *link.unscored_pairs.entry(first.kind.clone()).or_default() += 1;
                    if previous.contains(&(a, b)) {
                        *link
                            .removed_positive_pairs
                            .entry(first.kind.clone())
                            .or_default() += 1;
                    }
                    continue;
                }
            }
            truth.pairs.push(ExpectedPair {
                a: first.path.clone(),
                b: second.path.clone(),
                relation: "variant".into(),
                kind: first.kind.clone(),
                variant: variant.into(),
                visible_rgb_mae: Some(difference),
                accepted_relations: Vec::new(),
            });
        }
    }
    Ok(registrations.audit)
}

struct Registrations {
    audit: Audit,
    origins: Vec<Option<usize>>,
    links: HashMap<(usize, usize), usize>,
}

impl Registrations {
    fn new(truth: &Truth, ancestry: &HashMap<String, String>) -> anyhow::Result<Self> {
        let originals: Vec<_> = truth
            .files
            .iter()
            .enumerate()
            .filter(|(_, file)| file.kind == "image" && file.variant == "original")
            .map(|(index, _)| index)
            .collect();
        let by_family: HashMap<_, _> = originals
            .iter()
            .map(|&index| (truth.files[index].family.as_str(), index))
            .collect();
        let origins = truth
            .files
            .iter()
            .map(|file| {
                let base = file.family.split('/').next().unwrap_or(&file.family);
                if let Some(index) = by_family.get(base) {
                    return Some(*index);
                }
                ancestry
                    .get(base)
                    .and_then(|family| by_family.get(family.as_str()))
                    .copied()
            })
            .collect();
        let mut features = HashMap::new();
        let mut audit = Audit {
            minimum_correlation: MIN_CORRELATION,
            minimum_coverage: MIN_COVERAGE,
            source_links: Vec::new(),
        };
        let mut links = HashMap::new();
        for (a, &first) in originals.iter().enumerate() {
            for &second in originals.iter().skip(a + 1) {
                if truth.files[first].design != truth.files[second].design {
                    continue;
                }
                for index in [first, second] {
                    if let std::collections::hash_map::Entry::Vacant(entry) = features.entry(index)
                    {
                        let image = image::open(truth.root.join(&truth.files[index].path))
                            .with_context(|| {
                                format!(
                                    "loading registration original {}",
                                    truth.files[index].path.display()
                                )
                            })?;
                        entry.insert(Prepared::new(&image));
                    }
                }
                links.insert(ordered(first, second), audit.source_links.len());
                audit.source_links.push(SourceLink {
                    a_family: truth.files[first].family.clone(),
                    b_family: truth.files[second].family.clone(),
                    proof: compare_prepared(&features[&first], &features[&second]),
                    unscored_pairs: std::collections::BTreeMap::new(),
                    removed_positive_pairs: std::collections::BTreeMap::new(),
                });
            }
        }
        Ok(Self {
            audit,
            origins,
            links,
        })
    }

    fn link(&mut self, a: usize, b: usize) -> anyhow::Result<&mut SourceLink> {
        let a = self.origins[a].context("cross-source version lacks an original")?;
        let b = self.origins[b].context("cross-source version lacks a paired original")?;
        let index = *self
            .links
            .get(&ordered(a, b))
            .context("cross-source version lacks an original registration")?;
        Ok(&mut self.audit.source_links[index])
    }
}

// Names qualify only where they identify a common composition, not merely a
// collection. In particular Hello colours and Chroma Blue/Red change geometry.
pub(crate) fn source_design(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?.to_lowercase();
    for composition in ["chroma blue", "chroma red", "dome"] {
        if stem
            .strip_prefix(composition)
            .is_some_and(|suffix| matches!(suffix, "" | " light" | " dark"))
        {
            return Some(composition.into());
        }
    }
    if let Some(dimensions) = stem.strip_prefix("calibrate ")
        && let Some((width, height)) = dimensions.split_once('x')
        && !width.is_empty()
        && !height.is_empty()
        && width.bytes().all(|byte| byte.is_ascii_digit())
        && height.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Some("calibrate".into());
    }
    for prefix in [
        "big sur graphic",
        "monterey graphic",
        "ventura graphic",
        "solar gradients",
        "light stream",
        "imac",
        "mac",
        "grid",
        "motion",
        "radial",
        "studio color",
    ] {
        if let Some(suffix) = stem.strip_prefix(prefix)
            && (suffix.is_empty() || suffix.starts_with(' '))
            && suffix.split_whitespace().all(|word| {
                matches!(
                    word,
                    "blue"
                        | "green"
                        | "orange"
                        | "pink"
                        | "purple"
                        | "red"
                        | "silver"
                        | "yellow"
                        | "black"
                        | "white"
                        | "grey"
                        | "gray"
                        | "magenta"
                        | "cyan"
                        | "light"
                        | "dark"
                        | "sky"
                )
            })
        {
            return Some(prefix.into());
        }
    }
    None
}

// Different 2D compositions do not prove independent underlying 3D assets.
// Do not silently turn a removed design equivalence into a hard negative.
pub(crate) fn ambiguous_collection(path: &Path) -> Option<&'static str> {
    let stem = path.file_stem()?.to_str()?.to_lowercase();
    ["hello metallic", "chroma"].into_iter().find(|prefix| {
        stem.strip_prefix(prefix)
            .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with(' '))
    })
}

// Synthetic Ken-Burns ancestry is recorded explicitly; native source video IDs carry no ancestry.
pub(crate) fn video_ancestry(truth: &Truth) -> HashMap<String, String> {
    let images: HashMap<_, _> = truth
        .files
        .iter()
        .filter(|file| file.kind == "image" && file.variant == "original")
        .map(|file| (file.path.as_path(), file.family.as_str()))
        .collect();
    truth
        .files
        .iter()
        .filter(|file| file.kind == "video" && file.variant == "original")
        .filter_map(|video| {
            let source = truth
                .sources
                .iter()
                .find(|source| source.retained == video.path)?;
            let path = source
                .origin
                .strip_prefix("ffmpeg ken-burns source=")?
                .split_whitespace()
                .next()?;
            Some((
                video.family.clone(),
                images.get(Path::new(path))?.to_string(),
            ))
        })
        .collect()
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn source_designs_cover_proven_grid_versions_without_collapsing_big_sur_photos() {
        let magenta = source_design(Path::new("Grid Magenta Light.heic"));
        assert_eq!(magenta, Some("grid".into()));
        assert_eq!(magenta, source_design(Path::new("Grid Magenta.heic")));
        assert_eq!(magenta, source_design(Path::new("Grid Green Light.heic")));
        assert_eq!(source_design(Path::new("Big Sur Aerial.heic")), None);
        assert_eq!(source_design(Path::new("Big Sur Coastline.heic")), None);
    }

    #[test]
    fn source_links_follow_composition_not_a_collection_or_numeric_file_id() {
        assert_eq!(
            source_design(Path::new("Chroma Blue Dark.heic")),
            source_design(Path::new("Chroma Blue Light.heic"))
        );
        assert_ne!(
            source_design(Path::new("Chroma Blue Light.heic")),
            source_design(Path::new("Chroma Red Light.heic"))
        );
        assert_eq!(source_design(Path::new("Hello Metallic Blue.heic")), None);
        assert_eq!(source_design(Path::new("Hello Metallic Green.heic")), None);
        assert_eq!(
            source_design(Path::new("Dome Light.heic")),
            Some("dome".into())
        );
        assert_eq!(
            source_design(Path::new("Dome.heic")),
            source_design(Path::new("Dome Dark.heic"))
        );
        assert_eq!(
            source_design(Path::new("Calibrate 5120x2880.jpg")),
            Some("calibrate".into())
        );
        assert_eq!(
            source_design(Path::new("Calibrate 5120x3200.jpg")),
            source_design(Path::new("Calibrate 5120x2880.jpg"))
        );
        assert_eq!(source_design(Path::new("Dome Mountain.heic")), None);
        assert_eq!(source_design(Path::new("Calibrate unrelated.jpg")), None);
    }

    #[test]
    fn variants_require_visible_full_frames_and_preserve_true_photo_negatives() -> anyhow::Result<()>
    {
        let directory = tempfile::Builder::new()
            .prefix("eidolon-design-")
            .tempdir_in("/tmp")?;
        let mut truth: Truth = serde_json::from_value(json!({
            "version":1,"seed":1,"root":directory.path(),"scale":"test",
            "sources":[
                {"origin":"Grid Magenta Light.heic","retained":"a.jpg","sha256":""},
                {"origin":"Grid Magenta.heic","retained":"b.jpg","sha256":""},
                {"origin":"Grid Green Light.heic","retained":"c.jpg","sha256":""},
                {"origin":"Big Sur Aerial.heic","retained":"photo1.jpg","sha256":""},
                {"origin":"Big Sur Coastline.heic","retained":"photo2.jpg","sha256":""},
                {"origin":"Grid Magenta Light.heic","retained":"f.jpg","sha256":""},
                {"origin":"ffmpeg ken-burns source=c.jpg duration=1","retained":"video2.mp4","sha256":""},
                {"origin":"ffmpeg ken-burns source=f.jpg duration=1","retained":"video5.mp4","sha256":""}
            ],
            "files":[
                {"path":"a.jpg","kind":"image","variant":"original","family":"image-000","sha256":""},
                {"path":"copy.jpg","kind":"image","variant":"exact","family":"image-000","sha256":""},
                {"path":"hue.jpg","kind":"image","variant":"variant_hue","family":"image-000/variant_hue","sha256":""},
                {"path":"faint.jpg","kind":"image","variant":"variant_relit","family":"image-000/variant_relit","sha256":""},
                {"path":"crop.jpg","kind":"image","variant":"crop_40","family":"image-001","sha256":""},
                {"path":"watermark.jpg","kind":"image","variant":"watermark","family":"image-001","sha256":""},
                {"path":"b.jpg","kind":"image","variant":"original","family":"image-001","sha256":""},
                {"path":"c.jpg","kind":"image","variant":"original","family":"image-002","sha256":""},
                {"path":"photo1.jpg","kind":"image","variant":"original","family":"image-003","sha256":""},
                {"path":"photo2.jpg","kind":"image","variant":"original","family":"image-004","sha256":""},
                {"path":"f.jpg","kind":"image","variant":"original","family":"image-005","sha256":""},
                {"path":"video2.mp4","kind":"video","variant":"original","family":"video-002","sha256":""},
                {"path":"video5.mp4","kind":"video","variant":"original","family":"video-005","sha256":""}
            ],"pairs":[],"hard_negatives":[]
        }))?;
        for (path, value) in [
            ("a.jpg", 64),
            ("copy.jpg", 64),
            ("hue.jpg", 200),
            ("faint.jpg", 69),
            ("crop.jpg", 180),
            ("watermark.jpg", 180),
            ("b.jpg", 180),
            ("c.jpg", 100),
            ("photo1.jpg", 40),
            ("photo2.jpg", 220),
            ("f.jpg", 200),
        ] {
            structured_source(value).save(directory.path().join(path))?;
        }
        for (path, source) in [("video2.mp4", "c.jpg"), ("video5.mp4", "f.jpg")] {
            let output = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-loop", "1", "-i"])
                .arg(directory.path().join(source))
                .args([
                    "-t",
                    "40",
                    "-vf",
                    "scale=64:36,fps=1",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                ])
                .arg(directory.path().join(path))
                .output()?;
            anyhow::ensure!(
                output.status.success(),
                "test video generation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        annotate(&mut truth)?;
        let related = |a: &str, b: &str| {
            truth.pairs.iter().any(|pair| {
                pair.relation == "variant"
                    && ((pair.a == Path::new(a) && pair.b == Path::new(b))
                        || (pair.a == Path::new(b) && pair.b == Path::new(a)))
            })
        };
        assert!(related("a.jpg", "b.jpg"));
        assert!(related("copy.jpg", "b.jpg"));
        assert!(related("copy.jpg", "hue.jpg"));
        assert!(related("video2.mp4", "video5.mp4"));
        assert!(!related("photo1.jpg", "photo2.jpg"));
        assert!(!related("a.jpg", "photo1.jpg"));
        assert!(!related("a.jpg", "copy.jpg"));
        assert!(!related("a.jpg", "faint.jpg"));
        assert!(!related("a.jpg", "crop.jpg"));
        assert!(!related("a.jpg", "watermark.jpg"));
        assert!(
            truth
                .pairs
                .iter()
                .all(|pair| pair.visible_rgb_mae.is_some_and(|value| value >= 20.0))
        );
        Ok(())
    }

    fn structured_source(value: u8) -> image::RgbImage {
        image::RgbImage::from_fn(128, 96, |x, y| {
            let shape = ((x / 19 + y / 13) % 3 == 0) ^ (x.abs_diff(59) + y.abs_diff(43) < 25);
            image::Rgb([
                value,
                if shape { 210 } else { 25 },
                if shape { 175 } else { 35 },
            ])
        })
    }
}

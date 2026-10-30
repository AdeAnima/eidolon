//! Cross-family leakage checks shared by source selection and final corpus validation.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context as _, ensure};
use image::imageops::FilterType;
use serde::Serialize;
use unicode_casefold::UnicodeCaseFold as _;
use unicode_normalization::UnicodeNormalization as _;
use unicode_normalization::char::is_combining_mark;

use crate::model::Truth;

/// Cross-family audit results. A successful audit has zero conflicting pairs.
#[derive(Debug, Default, Serialize)]
pub struct Distinctness {
    /// Independently selected text originals.
    pub text_families: usize,
    /// Independently decoded image originals.
    pub image_families: usize,
    /// Independently sampled video originals.
    pub video_families: usize,
    /// All text-original pairs checked.
    pub text_pairs_checked: usize,
    /// All image-original pairs checked.
    pub image_pairs_checked: usize,
    /// All video-original pairs checked.
    pub video_pairs_checked: usize,
    /// Annotated text hard negatives checked with the same strict limits.
    pub text_hard_negatives_checked: usize,
    /// Largest cross-family Jaccard similarity of normalized five-word shingles.
    pub max_text_5_shingle_jaccard: f64,
    /// Cross-family text pairs sharing at least fifteen consecutive normalized words.
    pub shared_text_15_word_pairs: usize,
    /// Cross-family image pairs with normalized RGB mean difference at most two bytes.
    pub near_identical_image_pairs: usize,
    /// Cross-family video pairs with sampled RGB mean difference at most two bytes.
    pub near_identical_video_pairs: usize,
    /// Repeated generated-video source descriptions, including lavfi seed/parameters.
    pub duplicate_video_parameters: usize,
    /// Deliberate derived-version and source-design variant annotations.
    pub expected_variant_pairs: usize,
    /// Variant annotations directly between independently selected originals.
    pub variant_original_pairs: usize,
    /// Same-design cross-lineage pairs intentionally ignored (derivatives or subtle changes).
    pub ignored_design_pairs: usize,
    /// Smallest measured aligned RGB change among declared Variant pairs.
    pub minimum_variant_rgb_mae: Option<f64>,
    /// Hard negatives incorrectly sharing a design class; zero on success.
    pub variant_hard_negative_pairs: usize,
}

/// Audit originals, not derivatives, for accidental cross-family identity/containment.
///
/// Text uses NFKC, full Unicode case folding, NFKC, and alphanumeric/combining-mark
/// tokens. Sharing fifteen consecutive tokens or five-shingle Jaccard above 0.1
/// is rejected. This is stricter than the thirty-word containment safeguard and
/// keeps annotated hard-negative boilerplate below fifteen words.
/// Images use 64×64 RGB thumbnails; videos use four 64×36 RGB temporal samples.
/// Media pairs with mean channel difference at most 2/255 are rejected, including
/// same-picture resized/re-encoded copies. Only visibly changed full-frame versions
/// are Variant positives; other same-design pairs are ignored, never hard negatives.
///
/// # Errors
/// Returns an error on unreadable media, failed ffmpeg sampling, duplicate source
/// parameters, or a cross-family conflict; the error names both source paths.
pub fn audit(truth: &Truth) -> anyhow::Result<Distinctness> {
    let mut result = Distinctness::default();
    audit_variant_policy(truth, &mut result)?;
    let mut texts = Vec::new();
    let mut images = Vec::new();
    let mut videos = Vec::new();
    let mut families = HashSet::new();
    let mut video_parameters = HashSet::new();
    for file in truth.files.iter().filter(|file| file.variant == "original") {
        ensure!(
            families.insert(&file.family),
            "duplicate original family {}",
            file.family
        );
        let path = truth.root.join(&file.path);
        match file.kind.as_str() {
            "text" => texts.push((&file.path, TextFingerprint::new(&fs::read_to_string(path)?))),
            "image" => images.push((&file.path, image_signature(&image::open(path)?))),
            "video" => {
                videos.push((&file.path, video_signature(&path)?));
                if let Some(source) = truth
                    .sources
                    .iter()
                    .find(|source| source.retained == file.path)
                {
                    ensure!(
                        video_parameters.insert(&source.origin),
                        "duplicate video parameters: {}",
                        source.origin
                    );
                }
            }
            _ => {}
        }
    }
    result.text_families = texts.len();
    result.image_families = images.len();
    result.video_families = videos.len();
    for a in 0..texts.len() {
        for b in a + 1..texts.len() {
            let overlap = texts[a].1.jaccard(&texts[b].1);
            result.max_text_5_shingle_jaccard = result.max_text_5_shingle_jaccard.max(overlap);
            ensure!(
                !texts[a].1.conflicts(&texts[b].1),
                "text family leakage: {} / {} (five-shingle Jaccard {overlap:.4}; shared 15-word span: {})",
                texts[a].0.display(),
                texts[b].0.display(),
                !texts[a].1.fifteen.is_disjoint(&texts[b].1.fifteen)
            );
            result.text_pairs_checked += 1;
        }
    }
    for pair in &truth.hard_negatives {
        if let (Some(a), Some(b)) = (
            texts.iter().find(|(path, _)| **path == pair.a),
            texts.iter().find(|(path, _)| **path == pair.b),
        ) {
            ensure!(
                !a.1.conflicts(&b.1),
                "text hard negative contains shared source content"
            );
            result.text_hard_negatives_checked += 1;
        }
    }
    for (name, samples, checked) in [
        ("image", &images, &mut result.image_pairs_checked),
        ("video", &videos, &mut result.video_pairs_checked),
    ] {
        for a in 0..samples.len() {
            for b in a + 1..samples.len() {
                ensure!(
                    !media_conflicts(&samples[a].1, &samples[b].1),
                    "near-identical {name} originals: {} / {}",
                    samples[a].0.display(),
                    samples[b].0.display()
                );
                *checked += 1;
            }
        }
    }
    Ok(result)
}

fn audit_variant_policy(truth: &Truth, result: &mut Distinctness) -> anyhow::Result<()> {
    let files: std::collections::HashMap<_, _> = truth
        .files
        .iter()
        .map(|file| (file.path.as_path(), file))
        .collect();
    let mut variants = HashSet::new();
    for pair in truth.pairs.iter().filter(|pair| pair.relation == "variant") {
        let a = files
            .get(pair.a.as_path())
            .context("variant endpoint absent from files")?;
        let b = files
            .get(pair.b.as_path())
            .context("variant endpoint absent from files")?;
        ensure!(
            a.kind == b.kind && a.family != b.family,
            "variant endpoints must be distinct versions of one media kind"
        );
        if let (Some(a), Some(b)) = (&a.design, &b.design) {
            ensure!(a == b, "variant endpoints have different design classes");
        }
        ensure!(
            crate::visibility::full_frame(a) && crate::visibility::full_frame(b),
            "variant truth may only connect full-frame versions"
        );
        let difference = pair
            .visible_rgb_mae
            .context("variant missing decoded visibility evidence")?;
        ensure!(
            difference.is_finite() && difference >= crate::visibility::MIN_VISIBLE,
            "variant RGB visibility must be at least20/255"
        );
        result.minimum_variant_rgb_mae = Some(
            result
                .minimum_variant_rgb_mae
                .map_or(difference, |minimum| minimum.min(difference)),
        );
        ensure!(
            variants.insert(path_pair(&pair.a, &pair.b)),
            "duplicate variant annotation"
        );
        result.expected_variant_pairs += 1;
        result.variant_original_pairs +=
            usize::from(a.variant == "original" && b.variant == "original");
    }
    for pair in &truth.hard_negatives {
        let a = files
            .get(pair.a.as_path())
            .context("hard-negative endpoint absent")?;
        let b = files
            .get(pair.b.as_path())
            .context("hard-negative endpoint absent")?;
        ensure!(
            !(variants.contains(&path_pair(&pair.a, &pair.b))
                || a.design.is_some() && a.design == b.design),
            "design variants cannot be hard negatives: {} / {}",
            pair.a.display(),
            pair.b.display()
        );
    }
    for a in 0..truth.files.len() {
        let first = &truth.files[a];
        if first.design.is_none() {
            continue;
        }
        for second in &truth.files[a + 1..] {
            if first.kind == second.kind
                && first.design == second.design
                && first.family != second.family
            {
                result.ignored_design_pairs +=
                    usize::from(!variants.contains(&path_pair(&first.path, &second.path)));
            }
        }
    }
    Ok(())
}

fn path_pair<'a>(a: &'a Path, b: &'a Path) -> (&'a Path, &'a Path) {
    if a < b { (a, b) } else { (b, a) }
}

pub(crate) struct TextFingerprint {
    five: HashSet<u64>,
    fifteen: HashSet<u64>,
}

impl TextFingerprint {
    pub(crate) fn new(text: &str) -> Self {
        let folded: String = text.nfkc().case_fold().nfkc().collect();
        let words: Vec<_> = folded
            .split(|character: char| !character.is_alphanumeric() && !is_combining_mark(character))
            .filter(|word| !word.is_empty())
            .map(hash_word)
            .collect();
        Self {
            five: shingles(&words, 5),
            fifteen: shingles(&words, 15),
        }
    }

    pub(crate) fn conflicts(&self, other: &Self) -> bool {
        !self.fifteen.is_disjoint(&other.fifteen) || self.jaccard(other) > 0.1
    }

    fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.five.intersection(&other.five).count();
        let union = self.five.len() + other.five.len() - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

pub(crate) fn image_signature(image: &image::DynamicImage) -> Vec<u8> {
    image
        .resize_exact(64, 64, FilterType::Triangle)
        .into_rgb8()
        .into_raw()
}

pub(crate) fn media_conflicts(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let difference: u64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum();
    difference <= (a.len() as u64) * 2
}

pub(crate) fn video_signature(path: &Path) -> anyhow::Result<Vec<u8>> {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
        ])
        .arg(path)
        .args([
            "-vf",
            "fps=1/10,scale=64:36",
            "-frames:v",
            "4",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .context("sampling original video for distinctness")?;
    ensure!(
        output.status.success(),
        "ffmpeg distinctness sampling failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    ensure!(
        !output.stdout.is_empty(),
        "no sampled video frames: {}",
        path.display()
    );
    Ok(output.stdout)
}

fn hash_word(word: &str) -> u64 {
    word.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn shingles(words: &[u64], width: usize) -> HashSet<u64> {
    words
        .windows(width)
        .map(|words| {
            words.iter().fold(0xcbf2_9ce4_8422_2325, |hash, word| {
                (hash ^ word).wrapping_mul(0x0100_0000_01b3)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_selection_rejects_embedded_license_body_but_allows_short_boilerplate() {
        let left = (0..200)
            .map(|index| format!("left{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let right = (0..200)
            .map(|index| format!("right{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let license = "THE TERMS OF THIS LICENSE APPLY TO EVERY COPY OF THE SOFTWARE AND ITS ASSOCIATED DOCUMENTATION PROVIDED WITH THE DISTRIBUTION WITHOUT ANY ADDITIONAL EXPRESS OR IMPLIED WARRANTIES";
        let a = TextFingerprint::new(&format!("{left} {license}"));
        let b = TextFingerprint::new(&format!("{right} {}", license.to_lowercase()));
        assert!(
            a.conflicts(&b),
            "same embedded license text is not a negative family"
        );
        let short = "Copyright notice distributed under these terms without additional warranties";
        assert!(
            !TextFingerprint::new(&format!("{left} {short}"))
                .conflicts(&TextFingerprint::new(&format!("{right} {short}")))
        );
        assert!(TextFingerprint::new("Straße ＬＩＣＥＮＳＥ terms with several conditions and limitations for users and copies of this software")
            .conflicts(&TextFingerprint::new("STRASSE LICENSE terms with several conditions and limitations for users and copies of this software")));
        let paragraphs: Vec<_> = (0..10)
            .map(|group| {
                (0..14)
                    .map(|word| format!("token{group}x{word}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let a = TextFingerprint::new(
            &paragraphs
                .iter()
                .enumerate()
                .map(|(group, paragraph)| format!("{paragraph} left{group}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        let b = TextFingerprint::new(
            &paragraphs
                .iter()
                .enumerate()
                .map(|(group, paragraph)| format!("{paragraph} right{group}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        assert!(
            a.fifteen.is_disjoint(&b.fifteen),
            "this fixture isolates the Jaccard limit"
        );
        assert!(
            a.conflicts(&b),
            "many shared short passages still exceed 0.1 Jaccard"
        );
    }

    #[test]
    fn image_fingerprints_reject_resized_copies_but_keep_different_series_colors() {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(480, 320, |x, y| {
            image::Rgb([(x / 4) as u8, (y / 2) as u8, 50])
        }));
        let resized = image.resize_exact(240, 160, FilterType::Lanczos3);
        assert!(media_conflicts(
            &image_signature(&image),
            &image_signature(&resized)
        ));
        let different =
            image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(480, 320, |x, y| {
                image::Rgb([(x / 4) as u8, (y / 2) as u8, 180])
            }));
        assert!(!media_conflicts(
            &image_signature(&image),
            &image_signature(&different)
        ));
    }
}

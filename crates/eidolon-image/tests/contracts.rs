//! Public input, cache, and model lifecycle contracts.

use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use anyhow::Context as _;
use eidolon_core::{Config, Ctx, FileEntry, FileId, Kind, Relation, cache::FeatureCache};
use image::{ImageFormat, Rgb, RgbImage};

fn file(id: usize, path: PathBuf) -> anyhow::Result<FileEntry> {
    let stat = path.metadata()?;
    Ok(FileEntry {
        id: FileId(id),
        path,
        size: stat.len(),
        mtime_ns: i128::try_from(stat.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())?,
        kind: Kind::Image,
    })
}

fn config() -> Config {
    Config {
        timings: false,
        remote_overlay: true,
        min_score: 0.80,
        model_dir: PathBuf::new(),
        no_embeddings: true,
        no_keypoints: false,
        jobs: 2,
        fast: false,
        video_fps: 2.0,
    }
}

fn drawing(seed: u32) -> RgbImage {
    RgbImage::from_fn(320, 240, |x, y| {
        let block = ((x / 16).wrapping_mul(7919) ^ (y / 16).wrapping_mul(1237) ^ seed)
            .wrapping_mul(2_654_435_761);
        Rgb([(block >> 24) as u8, (block >> 16) as u8, (block >> 8) as u8])
    })
}

#[test]
fn supports_all_raster_formats_by_content() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-formats-")
        .tempdir_in("/tmp")?;
    let image = drawing(37);
    for format in [
        ImageFormat::Jpeg,
        ImageFormat::Png,
        ImageFormat::WebP,
        ImageFormat::Gif,
        ImageFormat::Tiff,
        ImageFormat::Bmp,
    ] {
        let path = temp.path().join(format!("{format:?}.unknown"));
        image.save_with_format(&path, format)?;
        let decoded = eidolon_image::decode(&path)?;
        assert_eq!(decoded.width(), image.width(), "{format:?}");
        assert_eq!(decoded.height(), image.height(), "{format:?}");
    }
    Ok(())
}

#[test]
fn detects_lossless_resaves_and_invalidates_modified_features() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-cache-")
        .tempdir_in("/tmp")?;
    let image = drawing(37);
    let a = temp.path().join("a.png");
    let b = temp.path().join("b.bmp");
    image.save(&a)?;
    image.save(&b)?;
    let files = [file(9, a)?, file(3, b.clone())?];
    let cache = FeatureCache::in_memory()?;
    let mut config = config();
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    let found = matches
        .first()
        .context("lossless cross-format match missing")?;
    assert_eq!(
        (found.a, found.b, found.relation),
        (FileId(3), FileId(9), Relation::SameContent)
    );
    config.no_keypoints = true;
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    assert_eq!(matches.len(), 1);
    assert!(
        !matches[0]
            .evidence
            .iter()
            .any(|signal| signal.signal == "orb")
    );
    // Different dimensions guarantee a fresh size even on coarse-mtime volumes.
    image::imageops::resize(
        &drawing(927),
        240,
        200,
        image::imageops::FilterType::Nearest,
    )
    .save(&b)?;
    let changed = [files[0].clone(), file(3, b)?];
    assert!(
        eidolon_image::find(
            &changed,
            &Ctx {
                refiner: None,
                config: &config,
                cache: &cache,
                embedder: None
            },
            None
        )?
        .is_empty()
    );
    Ok(())
}

#[test]
fn rejects_distinct_pages_with_shared_chrome() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-ui-")
        .tempdir_in("/tmp")?;
    let page = |seed| {
        RgbImage::from_fn(640, 480, |x, y| {
            if y < 64 || x < 100 {
                return Rgb([35, 44, 55]);
            }
            if x % 72 < 3 || y % 42 < 3 {
                return Rgb([190, 195, 200]);
            }
            if y > 80 && y % 42 < 17 && x > 130 && (x / 6 + y / 42 * seed) % 7 < 4 {
                Rgb([40, 50, 65])
            } else {
                Rgb([245, 245, 245])
            }
        })
    };
    let first = temp.path().join("page1.png");
    let second = temp.path().join("page2.png");
    page(3).save(&first)?;
    page(11).save(&second)?;
    let files = [file(0, first)?, file(1, second)?];
    let cache = FeatureCache::in_memory()?;
    let config = config();
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    assert!(
        matches.is_empty(),
        "different page bodies were called duplicate: {matches:?}"
    );
    let mut fast_config = config.clone();
    fast_config.fast = true;
    let fast_matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &fast_config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    assert!(
        fast_matches.is_empty(),
        "thumbnail proof must reject different page bodies: {fast_matches:?}"
    );
    let mut different_theme = page(11);
    image::imageops::invert(&mut different_theme);
    different_theme.save(&files[1].path)?;
    let [first, second] = files;
    let files = [first, file(1, second.path)?];
    for config in [&config, &fast_config] {
        let themed_matches = eidolon_image::find(
            &files,
            &Ctx {
                refiner: None,
                config,
                cache: &cache,
                embedder: None,
            },
            None,
        )?;
        assert!(
            themed_matches.is_empty(),
            "different page bodies must not become variants merely because the theme changed: {themed_matches:?}"
        );
    }
    Ok(())
}

#[test]
fn missing_models_are_optional_and_corrupt_models_are_never_replaced() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-model-")
        .tempdir_in("/tmp")?;
    assert!(eidolon_image::embed::OnnxEmbedder::load(temp.path())?.is_none());
    let path = temp.path().join(eidolon_image::models::FILE_NAME);
    std::fs::write(&path, b"not a model")?;
    assert!(eidolon_image::embed::OnnxEmbedder::load(temp.path()).is_err());
    assert!(eidolon_image::models::fetch(temp.path()).is_err());
    assert_eq!(std::fs::read(path)?, b"not a model");
    Ok(())
}

#[test]
fn does_not_discard_alpha_when_comparing_visible_content() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-alpha-")
        .tempdir_in("/tmp")?;
    let opaque = drawing(37);
    let transparent = image::RgbaImage::from_fn(opaque.width(), opaque.height(), |x, y| {
        let color = opaque.get_pixel(x, y);
        image::Rgba([color[0], color[1], color[2], 0])
    });
    let first = temp.path().join("opaque.png");
    let second = temp.path().join("transparent.png");
    opaque.save(&first)?;
    transparent.save(&second)?;
    let cache = FeatureCache::in_memory()?;
    let config = config();
    let files = [file(0, first)?, file(1, second)?];
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    assert!(
        matches.is_empty(),
        "invisible pixels were called the same content: {matches:?}"
    );
    Ok(())
}

#[test]
fn keeps_sixteen_bit_pixel_differences_out_of_same_content() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-hdr-")
        .tempdir_in("/tmp")?;
    let original = drawing(37);
    let deep = |offset: u16| {
        image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_fn(
            original.width(),
            original.height(),
            |x, y| {
                let pixel = original.get_pixel(x, y);
                image::Rgb(
                    pixel
                        .0
                        .map(|channel| (u16::from(channel) * 257).saturating_add(offset)),
                )
            },
        )
    };
    let first = temp.path().join("native16.png");
    let second = temp.path().join("changed16.tiff");
    deep(0).save(&first)?;
    deep(1).save(&second)?;
    let cache = FeatureCache::in_memory()?;
    let config = config();
    let files = [file(0, first)?, file(1, second)?];
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &config,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    let found = matches
        .first()
        .context("near-identical HDR pixels should still match")?;
    assert_eq!(
        found.relation,
        Relation::Reencoded,
        "low bits must not be discarded for exact pixel equality"
    );
    Ok(())
}

#[test]
fn fast_thumbnail_proof_does_not_change_default_reports() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-")
        .tempdir_in("/tmp")?;
    let original = drawing(37);
    let first = temp.path().join("original.png");
    let second = temp.path().join("resized.jpg");
    original.save(&first)?;
    let resized =
        image::imageops::resize(&original, 160, 120, image::imageops::FilterType::Triangle);
    image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&second)?, 30)
        .encode_image(&resized)?;
    let files = [file(91, first)?, file(3, second)?];
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    let run = |settings: &Config| {
        eidolon_image::find(
            &files,
            &Ctx {
                refiner: None,
                config: settings,
                cache: &cache,
                embedder: None,
            },
            None,
        )
    };
    let verified = run(&settings)?;
    assert_eq!(verified.len(), 1, "{verified:?}");
    assert!(
        verified[0]
            .evidence
            .iter()
            .any(|item| item.signal == "pixels")
    );
    settings.fast = true;
    let fast = run(&settings)?;
    assert_eq!(fast.len(), 1, "{fast:?}");
    assert_eq!(
        (fast[0].a, fast[0].b, fast[0].relation),
        (FileId(3), FileId(91), Relation::Reencoded)
    );
    assert!(
        fast[0]
            .evidence
            .iter()
            .any(|item| item.signal == "thumbnail")
    );
    assert!(
        !fast[0]
            .evidence
            .iter()
            .any(|item| item.signal == "unverified" || item.signal == "pixels")
    );
    settings.no_keypoints = true;
    let hashes_only = run(&settings)?;
    assert_eq!(hashes_only.len(), 1, "{hashes_only:?}");
    assert!(
        !hashes_only[0]
            .evidence
            .iter()
            .any(|item| item.signal == "orb")
    );
    settings.min_score = 1.0;
    assert!(run(&settings)?.is_empty());
    settings.fast = false;
    settings.no_keypoints = false;
    settings.min_score = 0.80;
    assert_eq!(
        serde_json::to_string(&verified)?,
        serde_json::to_string(&run(&settings)?)?
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn assert_fast_crop(embedder: Option<&dyn eidolon_core::Embedder>) -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-crop-")
        .tempdir_in("/tmp")?;
    let original = eidolon_image::decode(std::path::Path::new(
        "/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic",
    ))?
    .to_rgb8();
    let first = temp.path().join("whole.png");
    let second = temp.path().join("crop.png");
    original.save(&first)?;
    image::imageops::crop_imm(&original, 35, 41, 276, 276)
        .to_image()
        .save(&second)?;
    let files = [file(17, first)?, file(3, second)?];
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.fast = true;
    settings.no_embeddings = embedder.is_none();
    let run = |settings: &Config| {
        eidolon_image::find(
            &files,
            &Ctx {
                refiner: None,
                config: settings,
                cache: &cache,
                embedder,
            },
            None,
        )
    };
    let matches = run(&settings)?;
    let found = matches.first().context("fast ORB crop missing")?;
    assert_eq!(
        (found.a, found.b, found.relation),
        (FileId(3), FileId(17), Relation::CropOf),
        "{found:?}"
    );
    for signal in ["phash", "dhash", "orb", "containment", "thumbnail"] {
        assert!(
            found.evidence.iter().any(|item| item.signal == signal),
            "{found:?}"
        );
    }
    assert!(
        !found
            .evidence
            .iter()
            .any(|item| item.signal == "pixels" || item.signal == "unverified")
    );
    if embedder.is_some() {
        assert!(found.evidence.iter().any(|item| item.signal == "dinov2"));
        settings.no_embeddings = true;
        let hashes_and_orb = run(&settings)?;
        assert_eq!(hashes_and_orb.len(), 1, "{hashes_and_orb:?}");
        assert!(
            !hashes_and_orb[0]
                .evidence
                .iter()
                .any(|item| item.signal == "dinov2")
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn fast_preserves_real_photo_crop_direction_without_embeddings() -> anyhow::Result<()> {
    assert_fast_crop(None)
}

#[test]
#[ignore = "requires the already-fetched pinned model; no network requests"]
#[cfg(target_os = "macos")]
fn fast_preserves_available_embeddings_and_respects_the_disable_flag() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
        .join("Library/Caches/eidolon/models");
    let model = eidolon_image::embed::OnnxEmbedder::load(&model_dir)?
        .context("fetch the pinned model before this acceptance test")?;
    assert_fast_crop(Some(&model))
}

#[test]
fn fast_uses_complemented_hashes_for_inverted_versions() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-fast-inverted-")
        .tempdir_in("/tmp")?;
    let original = drawing(37);
    let mut inverted = original.clone();
    image::imageops::invert(&mut inverted);
    let first = temp.path().join("original.png");
    let second = temp.path().join("inverted.png");
    original.save(&first)?;
    inverted.save(&second)?;
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.fast = true;
    settings.no_keypoints = true;
    let files = [file(91, first)?, file(3, second)?];
    let matches = eidolon_image::find(
        &files,
        &Ctx {
            refiner: None,
            config: &settings,
            cache: &cache,
            embedder: None,
        },
        None,
    )?;
    let found = matches.first().context("inverted hash version missing")?;
    assert_eq!(
        (found.a, found.b, found.relation),
        (FileId(3), FileId(91), Relation::Variant),
        "{found:?}"
    );
    assert!(found.evidence.iter().any(|item| item.signal == "thumbnail"));
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the recording embedder local to its cache-migration scenario.
fn automatic_cpu_reuses_coreml_cache_and_embeds_only_new_files() -> anyhow::Result<()> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counted {
        id: &'static str,
        alternate: Option<&'static str>,
        prepared: AtomicUsize,
        inputs: Mutex<Vec<u8>>,
    }
    impl eidolon_core::Embedder for Counted {
        fn model_id(&self) -> &'static str {
            self.id
        }
        fn cache_model_ids(&self) -> [Option<&str>; 2] {
            [Some(self.id), self.alternate]
        }
        fn prepare(&self) -> anyhow::Result<()> {
            self.prepared.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn embed(&self, image: &RgbImage) -> anyhow::Result<Vec<f32>> {
            self.inputs
                .lock()
                .expect("input recorder poisoned")
                .push(image.get_pixel(0, 0)[0]);
            Ok(vec![1.0, 0.0])
        }
    }
    let counted = |id, alternate| Counted {
        id,
        alternate,
        prepared: AtomicUsize::new(0),
        inputs: Mutex::new(Vec::new()),
    };
    let temporary = tempfile::tempdir()?;
    let mut files = Vec::new();
    for id in 0..3 {
        let path = temporary.path().join(format!("{id}.png"));
        drawing(id as u32 + 37).save(&path)?;
        files.push(file(id, path)?);
    }
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    settings.no_keypoints = true;
    settings.fast = true;
    let run = |files: &[FileEntry], model: &Counted| {
        eidolon_image::find(
            files,
            &Ctx {
                config: &settings,
                cache: &cache,
                embedder: Some(model),
                refiner: None,
            },
            None,
        )
    };
    let missing = |model: &Counted| {
        eidolon_image::uncached_embedding_work(
            &files,
            &Ctx {
                config: &settings,
                cache: &cache,
                embedder: Some(model),
                refiner: None,
            },
            false,
        )
    };
    // Obsolete preprocessing bytes must not be accepted by the new input policy.
    for file in &files[..2] {
        cache.put(
            file,
            "image.embedding/qualified-coreml/white/v2",
            &bincode::serialize(&vec![0.0_f32, 1.0])?,
        )?;
    }
    let coreml = counted("qualified-coreml", None);
    run(&files[..2], &coreml)?;
    assert_eq!(
        coreml.inputs.lock().expect("input recorder poisoned").len(),
        2
    );
    let automatic_cpu = counted("qualified-cpu", Some("qualified-coreml"));
    assert_eq!(missing(&automatic_cpu)?, 1);
    run(&files, &automatic_cpu)?;
    let new_pixel = drawing(39).get_pixel(0, 0)[0];
    assert_eq!(
        *automatic_cpu
            .inputs
            .lock()
            .expect("input recorder poisoned"),
        [new_pixel]
    );
    assert_eq!(missing(&counted("qualified-cpu", None))?, 2);
    assert_eq!(missing(&counted("qualified-coreml", None))?, 1);
    let warm = counted("qualified-cpu", Some("qualified-coreml"));
    run(&files, &warm)?;
    assert_eq!(warm.prepared.load(Ordering::Relaxed), 0);
    assert!(
        warm.inputs
            .lock()
            .expect("input recorder poisoned")
            .is_empty()
    );
    assert_eq!(missing(&warm)?, 0);
    Ok(())
}

#[test]
fn lazy_provider_fallback_writes_only_its_resolved_cache_identity() -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Default)]
    struct Fallback {
        prepared: AtomicBool,
        calls: AtomicUsize,
        requested_only: bool,
    }
    impl eidolon_core::Embedder for Fallback {
        fn model_id(&self) -> &'static str {
            if self.prepared.load(Ordering::Relaxed) {
                "fallback-cpu"
            } else {
                "requested-coreml"
            }
        }
        fn cache_model_ids(&self) -> [Option<&str>; 2] {
            [
                Some("requested-coreml"),
                (!self.requested_only).then_some("fallback-cpu"),
            ]
        }
        fn prepare(&self) -> anyhow::Result<()> {
            self.prepared.store(true, Ordering::Relaxed);
            Ok(())
        }
        fn embed(&self, _: &RgbImage) -> anyhow::Result<Vec<f32>> {
            assert!(self.prepared.load(Ordering::Relaxed));
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![1.0, 0.0])
        }
    }
    let temporary = tempfile::tempdir()?;
    let mut files = Vec::new();
    for id in 0..2 {
        let path = temporary.path().join(format!("{id}.png"));
        drawing(id as u32 + 37).save(&path)?;
        files.push(file(id, path)?);
    }
    let cache = FeatureCache::in_memory()?;
    let mut settings = config();
    settings.no_embeddings = false;
    settings.no_keypoints = true;
    settings.fast = true;
    let run = |model: &Fallback| {
        eidolon_image::find(
            &files,
            &Ctx {
                config: &settings,
                cache: &cache,
                embedder: Some(model),
                refiner: None,
            },
            None,
        )
    };
    let first = Fallback::default();
    run(&first)?;
    assert_eq!(first.calls.load(Ordering::Relaxed), 2);
    let requested_only = Fallback {
        requested_only: true,
        ..Fallback::default()
    };
    assert_eq!(
        eidolon_image::uncached_embedding_work(
            &files,
            &Ctx {
                config: &settings,
                cache: &cache,
                embedder: Some(&requested_only),
                refiner: None,
            },
            false,
        )?,
        files.len()
    );
    let warm = Fallback::default();
    run(&warm)?;
    assert_eq!(warm.calls.load(Ordering::Relaxed), 0);
    assert!(!warm.prepared.load(Ordering::Relaxed));
    Ok(())
}

#[test]
fn undecodable_single_input_has_a_recoverable_file_identity() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("truncated.jpg");
    std::fs::write(&path, [0xff, 0xd8, 0xff, 0xe0, 0, 16])?;
    let files = [file(7, path)?];
    let cache = FeatureCache::in_memory()?;
    let config = config();
    let error = eidolon_image::find(
        &files,
        &Ctx {
            config: &config,
            cache: &cache,
            embedder: None,
            refiner: None,
        },
        None,
    )
    .expect_err("truncated image must fail extraction, even without another input");
    let input = error
        .downcast_ref::<eidolon_core::InputError>()
        .context("decode failure must identify its input")?;
    assert_eq!(input.file, FileId(7));
    assert_ne!(input.reason, "");
    Ok(())
}

#[test]
fn mixed_source_center_bank_preserves_aspect_and_has_an_independent_lazy_cache()
-> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Model {
        whole: AtomicUsize,
        center: AtomicUsize,
    }
    impl eidolon_core::Embedder for Model {
        fn model_id(&self) -> &'static str {
            "source-center-contract"
        }
        fn embed(&self, _: &RgbImage) -> anyhow::Result<Vec<f32>> {
            self.whole.fetch_add(1, Ordering::Relaxed);
            Ok(vec![1.0, 0.0])
        }
        fn embed_frame(&self, image: &RgbImage) -> anyhow::Result<Vec<f32>> {
            self.center.fetch_add(1, Ordering::Relaxed);
            assert_eq!(image.width().max(image.height()), 800);
            Ok(vec![image.width() as f32, image.height() as f32])
        }
    }
    let temp = tempfile::tempdir()?;
    let mut files = Vec::new();
    for (id, width, height) in [(91, 1200, 800), (3, 800, 1200)] {
        let path = temp.path().join(format!("{id}.png"));
        RgbImage::from_fn(width, height, |x, y| Rgb([x as u8, y as u8, 71])).save(&path)?;
        files.push(file(id, path)?);
    }
    let cache = FeatureCache::in_memory()?;
    let model = Model::default();
    let mut settings = config();
    settings.no_embeddings = false;
    settings.no_keypoints = true;
    let ctx = Ctx {
        config: &settings,
        cache: &cache,
        embedder: Some(&model),
        refiner: None,
    };
    assert_eq!(
        eidolon_image::uncached_embedding_work(&files, &ctx, true)?,
        4
    );
    assert_eq!(model.center.load(Ordering::Relaxed), 0);
    let bank = eidolon_image::source_frame_embeddings(&files, &ctx)?;
    assert_eq!(bank.len(), 2);
    assert!(bank[&FileId(91)][0] > bank[&FileId(91)][1]);
    assert!(bank[&FileId(3)][0] < bank[&FileId(3)][1]);
    assert_eq!(model.center.load(Ordering::Relaxed), 2);
    assert_eq!(model.whole.load(Ordering::Relaxed), 0);
    assert_eq!(
        eidolon_image::uncached_embedding_work(&files, &ctx, true)?,
        2
    );
    let mut shifted = files[0].clone();
    shifted.id = FileId(7);
    assert!(eidolon_image::source_frame_embeddings(&[shifted], &ctx)?.contains_key(&FileId(7)));
    assert_eq!(model.center.load(Ordering::Relaxed), 2);
    eidolon_image::find(&files, &ctx, Some(&bank))?;
    assert_eq!(model.whole.load(Ordering::Relaxed), 2);
    assert_eq!(model.center.load(Ordering::Relaxed), 2);
    assert_eq!(
        eidolon_image::uncached_embedding_work(&files, &ctx, true)?,
        0
    );
    let mut missing = files[0].clone();
    missing.path = temp.path().join("missing.jpg");
    assert!(
        eidolon_image::source_frame_embeddings(std::slice::from_ref(&missing), &ctx)?.is_empty()
    );
    let disabled_settings = Config {
        timings: false,
        remote_overlay: true,
        no_embeddings: true,
        ..settings.clone()
    };
    let disabled = Ctx {
        config: &disabled_settings,
        ..ctx
    };
    assert!(eidolon_image::source_frame_embeddings(&[missing], &disabled)?.is_empty());
    assert_eq!(model.center.load(Ordering::Relaxed), 2);
    Ok(())
}

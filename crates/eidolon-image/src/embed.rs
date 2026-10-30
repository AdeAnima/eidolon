//! Reusable, local DINOv2-small embeddings for stills and video frames.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, LazyLock, Mutex, OnceLock, PoisonError};

use anyhow::{Context as _, ensure};
use eidolon_core::Embedder;
use image::imageops::{FilterType, crop_imm, resize};
use ort::environment::Environment;
use ort::ep::{CoreML, ExecutionProvider as _, coreml};
use ort::session::{
    Session,
    builder::{GraphOptimizationLevel, SessionBuilder},
};
use ort::value::Tensor;
use sha2::{Digest as _, Sha256};

use crate::models;

const SIDE: usize = 224;
const DIMENSIONS: usize = 384;
const INTRA_THREADS: usize = 2;
const MAX_SESSIONS: usize = 8;
const MODEL_BYTES: usize = 88_532_934;
static INITIALIZE: LazyLock<Result<(), String>> = LazyLock::new(load_runtime);
// Retained-package measurements: 29.891 s CoreML startup versus 0.302 s CPU,
// then 250.996 versus 57.206 embeddings/s. Use the slower observed cached startup.
const AUTO_CROSSOVER: usize = 2_193;
const COREML_POLICY: &str = "mlprogram-fp32io-gpuaccum32-static1x3x224x224-all-v1";

/// Requested execution policy for local `DINOv2` inference.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionDevice {
    /// Select CPU or `CoreML` from uncached work; retry CPU if `CoreML` fails.
    #[default]
    Auto,
    /// Use the FP32 CPU execution provider.
    Cpu,
    /// Require `CoreML` `MLProgram` execution; Apple may use its CPU, GPU, or ANE.
    CoreMl,
}

/// SHA-verified DINOv2-small inference with bounded local inference sessions.
///
/// Still-image matching resizes the entire image to 224 square. Frame matching
/// preserves the pinned center crop on both still thumbnails and video frames.
/// Both use `ImageNet` CHW normalization and the same local inference sessions.
/// The CLS token is L2-normalized. ONNX Runtime telemetry is explicitly disabled.
pub struct OnnxEmbedder {
    // Every lazy session sees the same verified graph, even if its file changes.
    model: Vec<u8>,
    requested: ExecutionDevice,
    planned: ExecutionDevice,
    pool: OnceLock<Result<SessionPool, String>>,
}

#[derive(Clone, Copy)]
enum Preprocessing {
    WholeImage,
    FrameMatch,
}

struct SessionPool {
    sessions: Vec<Mutex<Option<Session>>>,
    device: ExecutionDevice,
    coreml_cache: Option<PathBuf>,
    idle: Mutex<Vec<usize>>,
    ready: Condvar,
}

impl OnnxEmbedder {
    /// Load the pinned model with automatic device selection, without network requests.
    ///
    /// Returns `Ok(None)` when the model file is absent. See
    /// [`Self::load_with_device`] for execution and resource behaviour.
    ///
    /// # Errors
    /// Fails if the model cannot be read or its checksum is invalid.
    pub fn load(model_dir: &Path) -> anyhow::Result<Option<Self>> {
        Self::load_with_device(model_dir, ExecutionDevice::Auto)
    }

    /// Verify and retain the pinned graph without loading ONNX Runtime or sessions.
    ///
    /// Inference is initialized by [`Embedder::prepare`] or the first embedding.
    /// CPU then starts with one session and grows under contention, up to one
    /// per two available CPUs, capped at eight. `CoreML` uses one session and a
    /// persistent compiled graph. [`Self::set_workload`] selects the `Auto`
    /// backend before inference; without a workload estimate it selects CPU.
    /// Missing model files return `Ok(None)`.
    ///
    /// # Errors
    /// Fails if the graph cannot be read or its checksum is invalid, or explicit
    /// `CoreML` is unsupported on this platform. Runtime/provider errors occur
    /// during preparation. Explicit requests never silently switch providers.
    pub fn load_with_device(
        model_dir: &Path,
        requested: ExecutionDevice,
    ) -> anyhow::Result<Option<Self>> {
        let path = model_dir.join(models::FILE_NAME);
        if !path.try_exists()? {
            return Ok(None);
        }
        // Limit the snapshot read to the pinned graph size plus a sentinel byte,
        // so a corrupt or concurrently replaced file cannot grow memory without bound.
        let mut model = Vec::with_capacity(MODEL_BYTES + 1);
        std::fs::File::open(&path)?
            .take((MODEL_BYTES + 1) as u64)
            .read_to_end(&mut model)
            .with_context(|| format!("reading {}", path.display()))?;
        ensure!(
            model.len() == MODEL_BYTES,
            "pinned DINOv2 model must contain exactly {MODEL_BYTES} bytes"
        );
        let actual = format!("{:x}", Sha256::digest(&model));
        ensure!(
            actual == models::SHA256,
            "SHA-256 mismatch for {}: expected {}, got {actual}",
            path.display(),
            models::SHA256
        );
        Ok(Some(Self {
            model,
            requested,
            planned: plan_device(requested, CoreML::default().supported_by_platform(), 0)?,
            pool: OnceLock::new(),
        }))
    }

    /// Return the planned device, or the actual device after initialization.
    ///
    /// Never returns [`ExecutionDevice::Auto`]. An automatic `CoreML` plan may
    /// become CPU when preparation falls back.
    #[must_use]
    pub fn device(&self) -> ExecutionDevice {
        self.pool
            .get()
            .and_then(|result| result.as_ref().ok())
            .map_or(self.planned, |pool| pool.device)
    }

    /// Select automatic inference from the upper bound of uncached embeddings.
    ///
    /// Cache hits do not count. Explicit CPU/`CoreML` requests are unchanged.
    /// The choice is fixed before any detector starts; preparation may fall back.
    ///
    /// # Errors
    /// Fails if preparation has already been attempted.
    pub fn set_workload(&mut self, uncached_work: usize) -> anyhow::Result<()> {
        ensure!(
            self.pool.get().is_none(),
            "embedding preparation already started"
        );
        self.planned = plan_device(
            self.requested,
            CoreML::default().supported_by_platform(),
            uncached_work,
        )?;
        Ok(())
    }

    fn cache_ids_for(&self, preprocessing: Preprocessing) -> [Option<&str>; 2] {
        let device = self.device();
        let alternate = (self.requested == ExecutionDevice::Auto).then(|| {
            let alternate = if device == ExecutionDevice::Cpu {
                ExecutionDevice::CoreMl
            } else {
                ExecutionDevice::Cpu
            };
            model_id_for(alternate, preprocessing)
        });
        [Some(model_id_for(device, preprocessing)), alternate]
    }

    fn infer(
        &self,
        image: &image::RgbImage,
        preprocessing: Preprocessing,
    ) -> anyhow::Result<Vec<f32>> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "cannot embed an empty image"
        );
        let pool = self.pool()?;
        let pixels = match preprocessing {
            Preprocessing::WholeImage => preprocess(image),
            Preprocessing::FrameMatch => preprocess_frame(image),
        };
        let input = Tensor::from_array(([1_usize, 3, SIDE, SIDE], pixels))?;
        let slot = pool.acquire()?;
        let mut session = pool.sessions[slot.index]
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session lock poisoned"))?;
        let session = match &mut *session {
            Some(session) => session,
            empty => empty.insert(load_session(
                &self.model,
                pool.device,
                pool.coreml_cache.as_deref(),
            )?),
        };
        let outputs = session.run(ort::inputs!["pixel_values" => input])?;
        let output = outputs
            .get("pooler_output")
            .or_else(|| outputs.get("last_hidden_state"))
            .context("DINOv2 graph has neither pooler_output nor last_hidden_state")?;
        let (_, values) = output.try_extract_tensor::<f32>()?;
        ensure!(
            values.len() >= DIMENSIONS,
            "DINOv2 returned too few dimensions"
        );
        let mut embedding = values[..DIMENSIONS].to_vec();
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        ensure!(
            norm.is_finite() && norm > f32::EPSILON,
            "DINOv2 returned a non-finite or empty vector"
        );
        for value in &mut embedding {
            *value /= norm;
        }
        Ok(embedding)
    }

    fn pool(&self) -> anyhow::Result<&SessionPool> {
        self.pool
            .get_or_init(|| self.initialize_pool().map_err(|error| format!("{error:#}")))
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error}"))
    }

    fn initialize_pool(&self) -> anyhow::Result<SessionPool> {
        if let Err(message) = &*INITIALIZE {
            anyhow::bail!("{message}");
        }
        let requested =
            if self.requested == ExecutionDevice::Auto && self.planned == ExecutionDevice::CoreMl {
                ExecutionDevice::Auto
            } else {
                self.planned
            };
        let (device, (coreml_cache, first), fallback) = initialize_device(
            requested,
            CoreML::default().supported_by_platform(),
            |device| {
                let coreml_cache = if device == ExecutionDevice::CoreMl {
                    ensure!(
                        CoreML::default().is_available()?,
                        "CoreML execution provider unavailable in the loaded ONNX Runtime"
                    );
                    Some(prepare_coreml_cache()?)
                } else {
                    None
                };
                let first = load_session(&self.model, device, coreml_cache.as_deref())?;
                Ok((coreml_cache, first))
            },
        )?;
        if let Some(error) = fallback {
            eprintln!("{}", coreml_fallback_warning(&error));
        }
        let count = if device == ExecutionDevice::CoreMl {
            1
        } else {
            session_count(std::thread::available_parallelism().map_or(1, usize::from))
        };
        Ok(SessionPool {
            sessions: std::iter::once(Mutex::new(Some(first)))
                .chain((1..count).map(|_| Mutex::new(None)))
                .collect(),
            device,
            coreml_cache,
            // Reuse initialized slots before growing under contention.
            idle: Mutex::new((0..count).rev().collect()),
            ready: Condvar::new(),
        })
    }
}

impl SessionPool {
    fn acquire(&self) -> anyhow::Result<SessionSlot<'_>> {
        let mut idle = self
            .idle
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session pool lock poisoned"))?;
        loop {
            if let Some(index) = idle.pop() {
                return Ok(SessionSlot { pool: self, index });
            }
            idle = self
                .ready
                .wait(idle)
                .map_err(|_| anyhow::anyhow!("ONNX session pool lock poisoned"))?;
        }
    }
}

// The slot returns on inference errors and unwinding, without retaining the
// pool lock during preprocessing, inference, or output normalization.
struct SessionSlot<'a> {
    pool: &'a SessionPool,
    index: usize,
}

impl Drop for SessionSlot<'_> {
    fn drop(&mut self) {
        // Only index bookkeeping runs under this lock; preserve capacity even
        // during unwinding rather than stranding callers on the condition.
        let mut idle = self
            .pool
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        idle.push(self.index);
        drop(idle);
        self.pool.ready.notify_one();
    }
}

fn session_count(cpus: usize) -> usize {
    (cpus / INTRA_THREADS).clamp(1, MAX_SESSIONS)
}

fn plan_device(
    requested: ExecutionDevice,
    coreml_supported: bool,
    uncached_work: usize,
) -> anyhow::Result<ExecutionDevice> {
    if requested == ExecutionDevice::Auto && uncached_work <= AUTO_CROSSOVER {
        return Ok(ExecutionDevice::Cpu);
    }
    resolve_device(requested, coreml_supported)
}

fn resolve_device(
    requested: ExecutionDevice,
    coreml_available: bool,
) -> anyhow::Result<ExecutionDevice> {
    match requested {
        ExecutionDevice::Auto => Ok(if coreml_available {
            ExecutionDevice::CoreMl
        } else {
            ExecutionDevice::Cpu
        }),
        ExecutionDevice::Cpu => Ok(ExecutionDevice::Cpu),
        ExecutionDevice::CoreMl => {
            ensure!(
                coreml_available,
                "CoreML execution provider unavailable; select cpu or install a CoreML-enabled ONNX Runtime"
            );
            Ok(ExecutionDevice::CoreMl)
        }
    }
}

fn initialize_device<T>(
    requested: ExecutionDevice,
    coreml_supported: bool,
    mut initialize: impl FnMut(ExecutionDevice) -> anyhow::Result<T>,
) -> anyhow::Result<(ExecutionDevice, T, Option<anyhow::Error>)> {
    let device = resolve_device(requested, coreml_supported)?;
    match initialize(device) {
        Ok(value) => Ok((device, value, None)),
        Err(error) if requested == ExecutionDevice::Auto && device == ExecutionDevice::CoreMl => {
            let value = initialize(ExecutionDevice::Cpu)
                .with_context(|| format!("CPU initialization after CoreML failed ({error:#})"))?;
            Ok((ExecutionDevice::Cpu, value, Some(error)))
        }
        Err(error) => Err(error),
    }
}

fn coreml_fallback_warning(error: &anyhow::Error) -> String {
    let mut warning = format!("CoreML initialization failed: {error:#}; using CPU embeddings");
    warning.retain(|character| !matches!(character, '\r' | '\n'));
    warning
}

fn coreml_cache_path(base: &Path, runtime_build: &str, os_build: &str) -> PathBuf {
    let mut environment = Sha256::new();
    environment.update(runtime_build.as_bytes());
    environment.update([0]);
    environment.update(os_build.as_bytes());
    base.join("coreml")
        .join(models::SHA256)
        .join(format!("{:x}", environment.finalize()))
        .join(COREML_POLICY)
}

fn coreml_cache_ready(cache: &Path) -> anyhow::Result<bool> {
    let graphs = match std::fs::read_dir(cache) {
        Ok(graphs) => graphs,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    // ORT's MLProgram cache contains graph/partition/model packages. A failed
    // compile may leave directories behind, so their presence is insufficient.
    for graph in graphs {
        let graph = graph?;
        if !graph.file_type()?.is_dir() {
            continue;
        }
        for partition in std::fs::read_dir(graph.path())? {
            let partition = partition?;
            if !partition.file_type()?.is_dir() {
                continue;
            }
            let compiled = partition.path().join("model/compiled_model.mlmodelc");
            if ["coremldata.bin", "model.mil", "weights/weight.bin"]
                .into_iter()
                .all(|name| {
                    compiled
                        .join(name)
                        .metadata()
                        .is_ok_and(|file| file.is_file() && file.len() > 0)
                })
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn prepare_coreml_cache() -> anyhow::Result<PathBuf> {
    let base = PathBuf::from(
        std::env::var_os("HOME").context("HOME unavailable for the CoreML model cache")?,
    )
    .join("Library/Caches/eidolon");
    let build = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-buildVersion")
        .output()
        .context("reading the macOS build for the CoreML model cache")?;
    ensure!(
        build.status.success(),
        "reading the macOS build failed: {}",
        build.status
    );
    let os_build = std::str::from_utf8(&build.stdout)?.trim();
    ensure!(!os_build.is_empty(), "macOS build identifier was empty");
    // ort::info includes the loaded runtime version, revision and build flags.
    let cache = coreml_cache_path(&base, ort::info(), os_build);
    let needs_compilation = !coreml_cache_ready(&cache)?;
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("creating CoreML model cache {}", cache.display()))?;
    if needs_compilation {
        eprintln!("compiling model for this Mac, one-time, ~30s");
    }
    Ok(cache)
}

fn load_session(
    model: &[u8],
    device: ExecutionDevice,
    coreml_cache: Option<&Path>,
) -> anyhow::Result<Session> {
    session_builder(device, coreml_cache)?
        .commit_from_memory(model)
        .context("loading the verified DINOv2 graph")
}

fn session_builder(
    device: ExecutionDevice,
    coreml_cache: Option<&Path>,
) -> anyhow::Result<SessionBuilder> {
    let mut builder = Session::builder()?
        .with_intra_threads(INTRA_THREADS)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_intra_op_spinning(false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_inter_op_spinning(false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if device == ExecutionDevice::CoreMl {
        let cache = coreml_cache.context("CoreML model-cache directory missing")?;
        let cache = cache
            .to_str()
            .context("CoreML model-cache path must be UTF-8")?;
        let side = i64::try_from(SIDE)?;
        // Inputs are always [1, 3, 224, 224]. Resolving the pinned graph's
        // symbolic dimensions lets CoreML own the whole graph rather than
        // splitting every attention reshape into a separate CPU fallback.
        for (name, size) in [
            ("batch_size", 1),
            ("num_channels", 3),
            ("height", side),
            ("width", side),
        ] {
            builder = builder
                .with_dimension_override(name, size)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        }
        builder = builder
            .with_execution_providers([CoreML::default()
                .with_model_format(coreml::ModelFormat::MLProgram)
                .with_compute_units(coreml::ComputeUnits::All)
                .with_static_input_shapes(true)
                .with_low_precision_accumulation_on_gpu(false)
                .with_model_cache_dir(cache)
                .build()
                .error_on_failure()])
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    } else {
        // ort's default automatic policy may select CoreML even for CPU requests.
        // Bind the built-in CPU EP itself, not just a CPU-type device/preference:
        // CPU::register only configures its arena and does not register a native EP.
        let environment = Environment::current()?;
        let cpu = environment
            .devices()
            .find(|device| device.ep().is_ok_and(|name| name == "CPUExecutionProvider"))
            .context("CPUExecutionProvider unavailable in the loaded ONNX Runtime")?;
        builder = builder
            .with_no_environment_execution_providers()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .with_devices([cpu], None)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    Ok(builder)
}

fn load_runtime() -> Result<(), String> {
    let path = runtime_path().map_err(|error| format!("{error:#}"))?;
    ort::init_from(&path)
        .map_err(|error| {
            format!(
                "loading ONNX Runtime from {}; run `brew install onnxruntime` or set ORT_DYLIB_PATH: {error}",
                path.display()
            )
        })?
        .with_name("eidolon")
        .with_telemetry(false)
        .commit();
    Ok(())
}

fn runtime_path() -> anyhow::Result<PathBuf> {
    runtime_path_from(
        std::env::var_os("ORT_DYLIB_PATH").as_deref(),
        option_env!("EIDOLON_ORT_DYLIB_PATH"),
    )
}

fn runtime_path_from(
    explicit: Option<&std::ffi::OsStr>,
    compiled: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let path = explicit
        .map(PathBuf::from)
        .or_else(|| compiled.map(PathBuf::from))
        .context(
            "ONNX Runtime has no build-time library path; run `brew install onnxruntime` and rebuild, or set ORT_DYLIB_PATH",
        )?;
    ensure!(
        path.is_file(),
        "ONNX Runtime library not found at {}; run `brew install onnxruntime` or correct ORT_DYLIB_PATH",
        path.display()
    );
    path.canonicalize().with_context(|| {
        format!(
            "resolving ONNX Runtime at {}; run `brew install onnxruntime` or set ORT_DYLIB_PATH",
            path.display()
        )
    })
}

fn model_id_for(device: ExecutionDevice, preprocessing: Preprocessing) -> &'static str {
    match (preprocessing, device) {
        (Preprocessing::WholeImage, ExecutionDevice::CoreMl) => concat!(
            "dinov2-small@",
            "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
            "/imagenet-whole-squash224/v1/coreml-mlprogram-fp32io-gpuaccum32-static1x3x224x224-all/v1"
        ),
        (Preprocessing::WholeImage, ExecutionDevice::Cpu | ExecutionDevice::Auto) => concat!(
            "dinov2-small@",
            "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
            "/imagenet-whole-squash224/v1/cpu-ep-fp32/v2"
        ),
        (Preprocessing::FrameMatch, ExecutionDevice::CoreMl) => concat!(
            "dinov2-small@",
            "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
            "/imagenet-center224/v1/coreml-mlprogram-fp32io-gpuaccum32-static1x3x224x224-all/v1"
        ),
        (Preprocessing::FrameMatch, ExecutionDevice::Cpu | ExecutionDevice::Auto) => concat!(
            "dinov2-small@",
            "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
            "/imagenet-center224/v1/cpu-ep-fp32/v2"
        ),
    }
}

fn preprocess(image: &image::RgbImage) -> Vec<f32> {
    let resized = resize(image, SIDE as u32, SIDE as u32, FilterType::CatmullRom);
    normalized_chw(&resized)
}

fn preprocess_frame(image: &image::RgbImage) -> Vec<f32> {
    let short = image.width().min(image.height());
    let square = crop_imm(
        image,
        (image.width() - short) / 2,
        (image.height() - short) / 2,
        short,
        short,
    );
    let resized = resize(&*square, 256, 256, FilterType::CatmullRom);
    normalized_chw(&*crop_imm(&resized, 16, 16, SIDE as u32, SIDE as u32))
}

fn normalized_chw(image: &impl image::GenericImageView<Pixel = image::Rgb<u8>>) -> Vec<f32> {
    let means = [0.485_f32, 0.456, 0.406];
    let stds = [0.229_f32, 0.224, 0.225];
    let mut pixels = vec![0.0_f32; 3 * SIDE * SIDE];
    for (x, y, pixel) in image.pixels() {
        let offset = y as usize * SIDE + x as usize;
        for channel in 0..3 {
            pixels[channel * SIDE * SIDE + offset] =
                (f32::from(pixel[channel]) / 255.0 - means[channel]) / stds[channel];
        }
    }
    pixels
}

impl Embedder for OnnxEmbedder {
    fn model_id(&self) -> &'static str {
        model_id_for(self.device(), Preprocessing::WholeImage)
    }

    fn cache_model_ids(&self) -> [Option<&str>; 2] {
        self.cache_ids_for(Preprocessing::WholeImage)
    }

    fn frame_model_id(&self) -> &'static str {
        model_id_for(self.device(), Preprocessing::FrameMatch)
    }

    fn frame_cache_model_ids(&self) -> [Option<&str>; 2] {
        self.cache_ids_for(Preprocessing::FrameMatch)
    }

    fn prepare(&self) -> anyhow::Result<()> {
        self.pool().map(|_| ())
    }

    fn embed(&self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        self.infer(image, Preprocessing::WholeImage)
    }

    fn embed_frame(&self, image: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        self.infer(image, Preprocessing::FrameMatch)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::Barrier;

    use anyhow::Context as _;
    use eidolon_core::Embedder as _;
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

    use super::{
        ExecutionDevice, MODEL_BYTES, OnnxEmbedder, coreml_cache_path, coreml_cache_ready,
        coreml_fallback_warning, initialize_device, load_session, resolve_device, session_builder,
        session_count,
    };

    #[test]
    fn runtime_uses_the_build_time_default_without_a_runtime_override() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let library = temp.path().join("libonnxruntime.dylib");
        std::fs::write(&library, b"runtime location fixture")?;
        let resolved = super::runtime_path_from(None, library.to_str())?;
        assert_eq!(resolved, library.canonicalize()?);
        Ok(())
    }

    #[test]
    fn runtime_override_is_authoritative_even_when_missing() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let default = temp.path().join("libonnxruntime.dylib");
        std::fs::write(&default, b"runtime location fixture")?;
        let missing = temp.path().join("missing.dylib");
        let error = super::runtime_path_from(Some(missing.as_os_str()), default.to_str())
            .expect_err("an invalid explicit override must not fall back");
        assert!(error.to_string().contains("ORT_DYLIB_PATH"));
        assert!(error.to_string().contains("missing.dylib"));
        Ok(())
    }

    #[test]
    fn runtime_without_an_override_or_build_default_names_the_required_variable() {
        let error = super::runtime_path_from(None, None)
            .expect_err("a build without a default must require an explicit runtime path");
        assert!(error.to_string().contains("ORT_DYLIB_PATH"));
    }

    #[test]
    #[ignore = "requires the pinned local model and ONNX Runtime; stays offline"]
    fn frame_matching_preserves_legacy_center_crop() -> anyhow::Result<()> {
        let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models");
        let model = OnnxEmbedder::load_with_device(&model_dir, ExecutionDevice::Cpu)?
            .context("fetch the pinned model first")?;
        let white = RgbImage::from_pixel(super::SIDE as u32, super::SIDE as u32, Rgb([255; 3]));
        let expected = model.embed(&white)?;
        for (width, height) in [(800, 450), (450, 800)] {
            let image = RgbImage::from_fn(width, height, |x, y| {
                if x < width / 32
                    || x >= width * 31 / 32
                    || y < height / 32
                    || y >= height * 31 / 32
                {
                    Rgb([255, 0, 0])
                } else {
                    Rgb([255; 3])
                }
            });
            let frame = model.embed_frame(&image)?;
            assert!(
                frame
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "frame-match inference must exclude the legacy crop's outer margins"
            );
            let whole = model.embed(&image)?;
            assert!(
                whole
                    .iter()
                    .zip(&expected)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "still-image inference must retain those margins"
            );
        }
        Ok(())
    }

    #[test]
    fn whole_image_preprocessing_keeps_all_four_margins() -> anyhow::Result<()> {
        for (width, height) in [(1600, 900), (900, 1600), (4096, 224), (224, 4096)] {
            let image = RgbImage::from_fn(width, height, |x, y| {
                if x < width / 8 {
                    Rgb([255, 0, 0])
                } else if x >= width * 7 / 8 {
                    Rgb([0, 0, 255])
                } else if y < height / 8 {
                    Rgb([0, 255, 0])
                } else if y >= height * 7 / 8 {
                    Rgb([255, 255, 0])
                } else {
                    Rgb([128, 128, 128])
                }
            });
            let mut encoded = Cursor::new(Vec::new());
            DynamicImage::ImageRgb8(image).write_to(&mut encoded, ImageFormat::Png)?;
            let image = image::load_from_memory(encoded.get_ref())?.into_rgb8();
            let pixels = super::preprocess(&image);
            let side = super::SIDE;
            assert_eq!(pixels.len(), 3 * side * side);
            let sample = |channel, x, y| pixels[channel * side * side + y * side + x];
            // Independently colored margins cannot be replaced by center content.
            assert!((sample(0, 0, side / 2) - 2.248_908_3).abs() < 1e-5);
            assert!((sample(2, side - 1, side / 2) - 2.64).abs() < 1e-5);
            assert!((sample(1, side / 2, 0) - 2.428_571_5).abs() < 1e-5);
            assert!((sample(0, side / 2, side - 1) - 2.248_908_3).abs() < 1e-5);
            assert!((sample(1, side / 2, side - 1) - 2.428_571_5).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn whole_image_cache_ids_cannot_read_center_crop_embeddings() {
        for requested in [
            ExecutionDevice::Cpu,
            ExecutionDevice::CoreMl,
            ExecutionDevice::Auto,
        ] {
            let model = OnnxEmbedder {
                model: Vec::new(),
                requested,
                planned: if requested == ExecutionDevice::CoreMl {
                    ExecutionDevice::CoreMl
                } else {
                    ExecutionDevice::Cpu
                },
                pool: std::sync::OnceLock::new(),
            };
            let identities = model.cache_model_ids();
            assert_eq!(
                identities.iter().flatten().count(),
                if requested == ExecutionDevice::Auto {
                    2
                } else {
                    1
                }
            );
            for identity in identities.into_iter().flatten() {
                assert!(identity.contains("/imagenet-whole-squash224/v1/"));
                assert!(!identity.contains("center224"));
            }
            let frame_identities = model.frame_cache_model_ids();
            assert_eq!(frame_identities[0], Some(model.frame_model_id()));
            assert_eq!(
                frame_identities.iter().flatten().count(),
                identities.iter().flatten().count()
            );
            for identity in frame_identities.into_iter().flatten() {
                assert!(identity.contains("/imagenet-center224/v1/"));
                assert!(
                    !identities
                        .into_iter()
                        .flatten()
                        .any(|image| image == identity)
                );
            }
            assert!(model.pool.get().is_none(), "cache lookup must stay lazy");
        }
    }

    #[test]
    fn automatic_device_selection_uses_uncached_work_crossover() -> anyhow::Result<()> {
        let crossover = super::AUTO_CROSSOVER;
        assert!(crossover > 0);
        for work in [0, 1, crossover] {
            assert_eq!(
                super::plan_device(ExecutionDevice::Auto, true, work)?,
                ExecutionDevice::Cpu
            );
        }
        assert_eq!(
            super::plan_device(ExecutionDevice::Auto, true, crossover + 1)?,
            ExecutionDevice::CoreMl
        );
        assert_eq!(
            super::plan_device(ExecutionDevice::Auto, false, usize::MAX)?,
            ExecutionDevice::Cpu
        );
        assert_eq!(
            super::plan_device(ExecutionDevice::Cpu, true, usize::MAX)?,
            ExecutionDevice::Cpu
        );
        assert_eq!(
            super::plan_device(ExecutionDevice::CoreMl, true, 0)?,
            ExecutionDevice::CoreMl
        );
        Ok(())
    }

    #[test]
    fn explicit_device_requests_are_strict() -> anyhow::Result<()> {
        assert_eq!(
            resolve_device(ExecutionDevice::Cpu, false)?,
            ExecutionDevice::Cpu
        );
        assert_eq!(
            resolve_device(ExecutionDevice::Cpu, true)?,
            ExecutionDevice::Cpu
        );
        assert_eq!(
            resolve_device(ExecutionDevice::CoreMl, true)?,
            ExecutionDevice::CoreMl
        );
        assert!(resolve_device(ExecutionDevice::CoreMl, false).is_err());
        assert_eq!(
            resolve_device(ExecutionDevice::Auto, false)?,
            ExecutionDevice::Cpu
        );
        Ok(())
    }

    #[test]
    fn automatic_coreml_failure_uses_cpu_once() -> anyhow::Result<()> {
        let mut attempts = Vec::new();
        let (device, value, fallback) = initialize_device(ExecutionDevice::Auto, true, |device| {
            attempts.push(device);
            if device == ExecutionDevice::CoreMl {
                anyhow::bail!("CoreML registration or compilation failed");
            }
            Ok(42)
        })?;
        assert_eq!(attempts, [ExecutionDevice::CoreMl, ExecutionDevice::Cpu]);
        assert_eq!(device, ExecutionDevice::Cpu);
        assert_eq!(value, 42);
        assert!(
            fallback
                .context("missing fallback reason")?
                .to_string()
                .contains("registration or compilation failed")
        );
        Ok(())
    }

    #[test]
    fn automatic_initialization_uses_the_supported_provider() -> anyhow::Result<()> {
        for (supported, expected) in [
            (true, ExecutionDevice::CoreMl),
            (false, ExecutionDevice::Cpu),
        ] {
            let mut attempts = Vec::new();
            let (device, value, fallback) =
                initialize_device(ExecutionDevice::Auto, supported, |device| {
                    attempts.push(device);
                    Ok(device)
                })?;
            assert_eq!(attempts, [expected]);
            assert_eq!(device, expected);
            assert_eq!(value, expected);
            assert!(fallback.is_none());
        }
        Ok(())
    }

    #[test]
    fn explicit_initialization_errors_are_not_fallbacks() {
        for requested in [ExecutionDevice::Cpu, ExecutionDevice::CoreMl] {
            let mut attempts = Vec::new();
            let result = initialize_device::<()>(requested, true, |device| {
                attempts.push(device);
                anyhow::bail!("provider failed");
            });
            assert!(result.is_err());
            assert_eq!(attempts, [requested]);
        }
    }

    #[test]
    fn failed_cpu_fallback_preserves_both_initialization_errors() -> anyhow::Result<()> {
        let error = initialize_device::<()>(ExecutionDevice::Auto, true, |device| {
            anyhow::bail!("{device:?} failed");
        })
        .err()
        .context("initialization unexpectedly succeeded")?;
        let reason = format!("{error:#}");
        assert!(reason.contains("CoreMl failed"));
        assert!(reason.contains("Cpu failed"));
        Ok(())
    }

    #[test]
    fn automatic_fallback_warning_is_one_line() {
        let error = anyhow::anyhow!("provider failed\r\n  native compiler failed");
        let warning = coreml_fallback_warning(&error);
        assert_eq!(warning.lines().count(), 1);
        assert!(warning.contains("provider failed"));
        assert!(warning.contains("native compiler failed"));
        assert!(warning.ends_with("using CPU embeddings"));
    }

    #[test]
    fn compiled_cache_is_keyed_by_model_runtime_and_os_build() {
        let base = PathBuf::from("/cache/eidolon");
        let cache = coreml_cache_path(&base, "ORT rel-1.31.1", "26A123");
        assert!(cache.starts_with(base.join("coreml").join(crate::models::SHA256)));
        assert_eq!(cache, coreml_cache_path(&base, "ORT rel-1.31.1", "26A123"));
        assert_ne!(cache, coreml_cache_path(&base, "ORT rel-1.32.0", "26A123"));
        assert_ne!(cache, coreml_cache_path(&base, "ORT rel-1.31.1", "26A124"));
    }

    #[test]
    fn missing_and_partial_compiled_caches_require_preparation() -> anyhow::Result<()> {
        let temporary = tempfile::tempdir()?;
        let cache = temporary.path().join("cache");
        assert!(!coreml_cache_ready(&cache)?);
        let compiled = cache.join("ort-key/0_static_mlprogram/model/compiled_model.mlmodelc");
        std::fs::create_dir_all(compiled.join("weights"))?;
        std::fs::write(compiled.join("model.mil"), "compiled program")?;
        std::fs::write(compiled.join("coremldata.bin"), "")?;
        assert!(!coreml_cache_ready(&cache)?);
        std::fs::write(compiled.join("coremldata.bin"), "compiled metadata")?;
        std::fs::write(compiled.join("weights/weight.bin"), "compiled weights")?;
        assert!(coreml_cache_ready(&cache)?);
        std::fs::remove_file(compiled.join("weights/weight.bin"))?;
        assert!(!coreml_cache_ready(&cache)?);
        Ok(())
    }

    #[test]
    fn session_budget_handles_small_machines_and_bounds_model_memory() {
        assert_eq!(session_count(0), 1);
        assert_eq!(session_count(1), 1);
        assert_eq!(session_count(2), 1);
        assert_eq!(session_count(7), 3);
        assert_eq!(session_count(16), 8);
        assert_eq!(session_count(128), 8);
    }

    #[test]
    fn rejects_full_sized_corrupt_model_before_loading_runtime() -> anyhow::Result<()> {
        let temporary = tempfile::tempdir()?;
        let path = temporary.path().join(crate::models::FILE_NAME);
        // A sparse, correctly sized file exercises the checksum rather than
        // passing only because a truncated model was rejected by its length.
        std::fs::File::create(path)?.set_len(MODEL_BYTES as u64)?;
        let error = OnnxEmbedder::load(temporary.path())
            .err()
            .context("corrupt graph unexpectedly accepted")?;
        assert!(format!("{error:#}").contains("SHA-256 mismatch"));
        Ok(())
    }

    fn initialized_sessions(embedder: &OnnxEmbedder) -> anyhow::Result<usize> {
        let Some(pool) = embedder.pool.get() else {
            return Ok(0);
        };
        let pool = pool.as_ref().map_err(|error| anyhow::anyhow!("{error}"))?;
        pool.sessions.iter().try_fold(0, |count, session| {
            let session = session
                .lock()
                .map_err(|_| anyhow::anyhow!("ONNX session lock poisoned"))?;
            Ok(count + usize::from(session.is_some()))
        })
    }

    #[test]
    #[ignore = "requires the already-fetched pinned model and native ONNX Runtime; stays offline"]
    fn concurrent_embeddings_preserve_single_session_cpu_values() -> anyhow::Result<()> {
        let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models");
        let temporary = tempfile::tempdir()?;
        let copied_path = temporary.path().join(crate::models::FILE_NAME);
        std::fs::hard_link(model_dir.join(crate::models::FILE_NAME), &copied_path)?;
        let pooled = OnnxEmbedder::load_with_device(temporary.path(), ExecutionDevice::Cpu)?
            .context("fetch the pinned model first")?;
        // Lazy sessions must retain the verified graph rather than re-open a
        // path that may disappear or name different weights after load.
        std::fs::remove_file(copied_path)?;
        assert_eq!(initialized_sessions(&pooled)?, 0);
        let serial = OnnxEmbedder::load_with_device(&model_dir, ExecutionDevice::Cpu)?
            .context("fetch the pinned model first")?;
        let images = [(257, 143), (135, 311), (1024, 13), (1, 1)]
            .into_iter()
            .map(|(width, height)| {
                let image = RgbImage::from_fn(width, height, |x, y| {
                    Rgb([
                        (x * 17 + y * 31) as u8,
                        (x * 53 + y * 7) as u8,
                        (x * 11 + y * 97) as u8,
                    ])
                });
                let mut png = Cursor::new(Vec::new());
                DynamicImage::ImageRgb8(image).write_to(&mut png, ImageFormat::Png)?;
                Ok(image::load_from_memory(png.get_ref())?.to_rgb8())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let expected = images
            .iter()
            .map(|image| serial.embed(image))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for (image, expected) in images.iter().zip(&expected) {
            assert_eq!(pooled.embed(image)?, *expected);
        }
        assert_eq!(initialized_sessions(&pooled)?, 1);
        let barrier = Barrier::new(16);
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let mut workers = Vec::new();
            for worker in 0..16 {
                let (pooled, images, expected, barrier) = (&pooled, &images, &expected, &barrier);
                workers.push(scope.spawn(move || -> anyhow::Result<()> {
                    barrier.wait();
                    for repeat in 0..4 {
                        let index = (worker + repeat) % images.len();
                        assert_eq!(pooled.embed(&images[index])?, expected[index]);
                    }
                    Ok(())
                }));
            }
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("embedding worker panicked"))??;
            }
            Ok(())
        })?;
        assert_eq!(
            initialized_sessions(&pooled)?,
            pooled.pool()?.sessions.len()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the pinned model and CoreML-enabled native ONNX Runtime; stays offline"]
    fn coreml_vectors_and_cache_identity_are_distinct_from_cpu() -> anyhow::Result<()> {
        let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models");
        let cpu = OnnxEmbedder::load_with_device(&model_dir, ExecutionDevice::Cpu)?
            .context("fetch the pinned model first")?;
        let coreml = OnnxEmbedder::load_with_device(&model_dir, ExecutionDevice::CoreMl)?
            .context("fetch the pinned model first")?;
        assert_eq!(cpu.device(), ExecutionDevice::Cpu);
        assert_eq!(coreml.device(), ExecutionDevice::CoreMl);
        assert_ne!(cpu.model_id(), coreml.model_id());
        assert!(cpu.model_id().ends_with("/cpu-ep-fp32/v2"));
        assert!(
            coreml
                .model_id()
                .contains("/coreml-mlprogram-fp32io-gpuaccum32-static1x3x224x224")
        );
        coreml.prepare()?;
        let cache = coreml
            .pool()?
            .coreml_cache
            .as_ref()
            .context("CoreML cache path missing")?;
        let expected_base = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/coreml")
            .join(crate::models::SHA256);
        assert!(cache.starts_with(expected_base));
        assert!(std::fs::read_dir(cache)?.next().transpose()?.is_some());
        let alternate = tempfile::tempdir()?;
        std::fs::hard_link(
            model_dir.join(crate::models::FILE_NAME),
            alternate.path().join(crate::models::FILE_NAME),
        )?;
        let mut automatic =
            OnnxEmbedder::load(alternate.path())?.context("linked model missing")?;
        assert_eq!(automatic.device(), ExecutionDevice::Cpu);
        assert_eq!(initialized_sessions(&automatic)?, 0);
        automatic.set_workload(usize::MAX)?;
        automatic.prepare()?;
        assert_eq!(automatic.device(), ExecutionDevice::CoreMl);
        assert_eq!(automatic.pool()?.coreml_cache.as_ref(), Some(cache));

        let image = RgbImage::from_fn(317, 211, |x, y| {
            Rgb([
                (x * 17 + y * 31) as u8,
                (x * 43 + y * 7) as u8,
                (x * 3 + y * 61) as u8,
            ])
        });
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image).write_to(&mut encoded, ImageFormat::Png)?;
        let decoded = image::load_from_memory(encoded.get_ref())?.into_rgb8();
        let cpu = cpu.embed(&decoded)?;
        let coreml = coreml.embed(&decoded)?;
        let dot: f64 = cpu
            .iter()
            .zip(&coreml)
            .map(|(&left, &right)| f64::from(left) * f64::from(right))
            .sum();
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let cosine = dot / (norm(&cpu) * norm(&coreml));
        assert!(cosine >= 0.999, "CPU/CoreML cosine was {cosine}");
        Ok(())
    }

    #[test]
    #[ignore = "requires the pinned model and CoreML-enabled native ONNX Runtime; stays offline"]
    fn coreml_rebuilds_a_missing_cache_and_reuses_the_compiled_package() -> anyhow::Result<()> {
        let model_dir = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models");
        let cpu = OnnxEmbedder::load_with_device(&model_dir, ExecutionDevice::Cpu)?
            .context("fetch the pinned model first")?;
        let temporary = tempfile::tempdir()?;
        let cache = temporary.path().join("compiled");
        assert!(!coreml_cache_ready(&cache)?);
        std::fs::create_dir(&cache)?;
        drop(load_session(
            &cpu.model,
            ExecutionDevice::CoreMl,
            Some(&cache),
        )?);
        assert!(coreml_cache_ready(&cache)?);
        let graph = std::fs::read_dir(&cache)?
            .next()
            .context("compiled graph missing")??;
        let program = graph
            .path()
            .join("0_static_mlprogram/model/compiled_model.mlmodelc/model.mil");
        let modified = std::fs::metadata(&program)?.modified()?;
        drop(load_session(
            &cpu.model,
            ExecutionDevice::CoreMl,
            Some(&cache),
        )?);
        assert!(coreml_cache_ready(&cache)?);
        assert_eq!(std::fs::metadata(program)?.modified()?, modified);
        Ok(())
    }

    #[test]
    #[ignore = "requires the pinned model and native ONNX Runtime; stays offline"]
    fn cpu_provider_is_explicit_in_native_node_trace() -> anyhow::Result<()> {
        let model_path = PathBuf::from(std::env::var_os("HOME").context("HOME missing")?)
            .join("Library/Caches/eidolon/models")
            .join(crate::models::FILE_NAME);
        crate::models::verify(&model_path)?;
        let model = std::fs::read(model_path)?;
        if let Err(message) = &*super::INITIALIZE {
            anyhow::bail!("{message}");
        }
        let temporary = tempfile::tempdir()?;
        let image = RgbImage::from_fn(224, 224, |x, y| {
            Rgb([
                (x * 17 + y * 31) as u8,
                (x * 43 + y * 7) as u8,
                (x * 3 + y * 61) as u8,
            ])
        });
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image).write_to(&mut encoded, ImageFormat::Png)?;
        let decoded = image::load_from_memory(encoded.get_ref())?.into_rgb8();
        let mut pixels = vec![0.0_f32; 3 * super::SIDE * super::SIDE];
        for (x, y, pixel) in decoded.enumerate_pixels() {
            let offset = y as usize * super::SIDE + x as usize;
            for channel in 0..3 {
                pixels[channel * super::SIDE * super::SIDE + offset] =
                    f32::from(pixel[channel]) / 255.0;
            }
        }
        let input =
            ort::value::Tensor::from_array(([1_usize, 3, super::SIDE, super::SIDE], pixels))?;
        for requested in [ExecutionDevice::Cpu, ExecutionDevice::Auto] {
            let prefix = temporary.path().join(format!("{requested:?}"));
            let (device, mut session, fallback) = initialize_device(requested, true, |selected| {
                anyhow::ensure!(
                    selected == ExecutionDevice::Cpu,
                    "injected CoreML initialization failure"
                );
                Ok(session_builder(selected, None)?
                    .with_profiling(&prefix)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?
                    .commit_from_memory(&model)?)
            })?;
            assert_eq!(device, ExecutionDevice::Cpu);
            assert_eq!(fallback.is_some(), requested == ExecutionDevice::Auto);
            session.run(ort::inputs!["pixel_values" => &input])?;
            let profile = session.end_profiling()?;
            let trace: Vec<serde_json::Value> =
                serde_json::from_reader(std::fs::File::open(profile)?)?;
            let kernels: Vec<_> = trace
                .iter()
                .filter(|event| {
                    event["cat"] == "Node"
                        && event["name"]
                            .as_str()
                            .is_some_and(|name| name.ends_with("_kernel_time"))
                })
                .collect();
            assert!(
                !kernels.is_empty(),
                "{requested:?}: no native kernel events"
            );
            for kernel in &kernels {
                assert_eq!(
                    kernel["args"]["provider"].as_str(),
                    Some("CPUExecutionProvider"),
                    "{requested:?}: non-CPU kernel {}",
                    kernel["name"]
                );
            }
        }
        Ok(())
    }
}

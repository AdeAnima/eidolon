//! Bounded, streamed `FFmpeg` decoding and versioned per-video features.

use std::io::{ErrorKind, Read as _};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, OnceLock};

use anyhow::{Context as _, bail, ensure};
use eidolon_core::cache::FeatureCache;
use eidolon_core::{Ctx, Embedder, FileEntry, RefinedWindow};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub(crate) const WIDTH: u32 = 96;
pub(crate) const HEIGHT: u32 = 96;
const MAX_FRAMES: usize = 1200;
pub(crate) const EMBED_INTERVAL: f64 = 3.0;
const FEATURE: &str = "video.frames/v6;rgb=96x96;phash=32x32-8x8;dhash=9x8;thumbnail=32x32;rgb_thumbnail=32x32;border=uniform-edge2;color_hashes=rank-luma-max-min-rgb-contrast;embed=3s";

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct Video {
    pub(crate) duration: f64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) fps: f64,
    pub(crate) bitrate: Option<u64>,
    pub(crate) step: f64,
    /// Maximum extracted samples, scaled with the configured sampling rate.
    pub(crate) max_frames: usize,
    pub(crate) frames: Vec<Frame>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct Frame {
    pub(crate) time: f64,
    pub(crate) phash: u64,
    pub(crate) dhash: u64,
    pub(crate) color: [u8; 12],
    pub(crate) tiles: [[u8; 3]; 16],
    pub(crate) regions: [u64; 4],
    /// Nomination-only planes; omitted on unindexed native refinement frames.
    pub(crate) color_hashes: Option<[u64; 6]>,
    pub(crate) thumbnail: Vec<u8>,
    pub(crate) rgb_thumbnail: Vec<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pixels: Option<Pixels>,
    /// Only still-image descriptors carry a detected caption exclusion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) overlay: Option<crate::overlay::Band>,
    pub(crate) embedding: Option<Vec<f32>>,
    #[serde(skip)]
    source_overlay: OnceLock<Option<crate::overlay::Band>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct Pixels {
    pub(crate) rgb: Vec<u8>,
    pub(crate) bounds: (u32, u32, u32, u32),
    pub(crate) aspect: f64,
    #[serde(skip)]
    ranges: OnceLock<[Vec<[[u8; 3]; 2]>; 3]>,
    #[serde(skip)]
    pub(crate) preview: OnceLock<Vec<[f64; 3]>>,
    #[serde(skip)]
    pub(crate) source_overlay: OnceLock<Option<crate::overlay::Band>>,
}

impl Pixels {
    /// Returns exact RGB extrema for rectangles of at most four native 3×3 cells.
    pub(crate) fn range(
        &self,
        rows: std::ops::Range<usize>,
        columns: std::ops::Range<usize>,
    ) -> [[u8; 3]; 2] {
        let ranges = self.ranges.get_or_init(|| {
            let mut ranges = vec![[[255; 3], [0; 3]]; 32 * 32];
            for (y, row) in self
                .rgb
                .as_chunks::<{ WIDTH as usize * 3 }>()
                .0
                .iter()
                .enumerate()
            {
                for (x, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
                    let [minimum, maximum] = &mut ranges[(y / 3) * 32 + x / 3];
                    for channel in 0..3 {
                        minimum[channel] = minimum[channel].min(pixel[channel]);
                        maximum[channel] = maximum[channel].max(pixel[channel]);
                    }
                }
            }
            let mut levels = [ranges, Vec::new(), Vec::new()];
            for level in 1..3 {
                let size = 1 << (level - 1);
                let previous = &levels[level - 1];
                let mut next = vec![[[255; 3], [0; 3]]; 32 * 32];
                for y in 0..32 {
                    for x in 0..32 {
                        for row in [y, (y + size).min(31)] {
                            for column in [x, (x + size).min(31)] {
                                let [minimum, maximum] = previous[row * 32 + column];
                                for channel in 0..3 {
                                    next[y * 32 + x][0][channel] =
                                        next[y * 32 + x][0][channel].min(minimum[channel]);
                                    next[y * 32 + x][1][channel] =
                                        next[y * 32 + x][1][channel].max(maximum[channel]);
                                }
                            }
                        }
                    }
                }
                levels[level] = next;
            }
            levels
        });
        debug_assert!(!rows.is_empty() && !columns.is_empty());
        debug_assert!(rows.len() <= 4 && columns.len() <= 4);
        let level = rows.len().min(columns.len()).ilog2() as usize;
        let size = 1 << level;
        let mut result = [[255; 3], [0; 3]];
        for row in (rows.start..rows.end).step_by(size) {
            for column in (columns.start..columns.end).step_by(size) {
                let row = row.min(rows.end - size);
                let column = column.min(columns.end - size);
                let [minimum, maximum] = ranges[level][row * 32 + column];
                for channel in 0..3 {
                    result[0][channel] = result[0][channel].min(minimum[channel]);
                    result[1][channel] = result[1][channel].max(maximum[channel]);
                }
            }
        }
        result
    }
}

impl Frame {
    /// Locate a source-only occluder once per sampled or native pixel plane.
    pub(crate) fn source_overlay(&self) -> Option<crate::overlay::Band> {
        if let Some(pixels) = &self.pixels {
            return *pixels.source_overlay.get_or_init(|| {
                crate::overlay::Band::solid(pixels.rgb.as_chunks::<3>().0, pixels.bounds, WIDTH)
            });
        }
        *self
            .source_overlay
            .get_or_init(|| crate::overlay::Band::solid(&self.rgb_thumbnail, (0, 0, 32, 32), 32))
    }

    pub(crate) fn retain_pixels(&mut self, rgb: RgbImage, aspect: f64) {
        self.pixels = Some(Pixels {
            bounds: content_bounds(&rgb),
            rgb: rgb.into_raw(),
            aspect,
            ranges: OnceLock::new(),
            preview: OnceLock::new(),
            source_overlay: OnceLock::new(),
        });
    }
}

#[derive(Deserialize)]
pub(crate) struct Remote {
    version: u32,
    duration_seconds: f64,
    source_fps: f64,
    sample_pts_seconds: Vec<f64>,
}

impl Remote {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(".remote.json");
        let bytes = match std::fs::read(sidecar) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let metadata: Self = serde_json::from_slice(&bytes)?;
        ensure!(
            metadata.version == 1,
            "unsupported remote frame metadata version"
        );
        ensure!(
            metadata.duration_seconds.is_finite()
                && metadata.duration_seconds > 0.0
                && metadata.source_fps.is_finite()
                && metadata.source_fps > 0.0,
            "invalid remote video timing"
        );
        ensure!(
            !metadata.sample_pts_seconds.is_empty()
                && metadata
                    .sample_pts_seconds
                    .iter()
                    .all(|time| time.is_finite()
                        && *time >= 0.0
                        && *time <= metadata.duration_seconds)
                && metadata
                    .sample_pts_seconds
                    .windows(2)
                    .all(|pair| pair[0] <= pair[1]),
            "invalid remote sample timestamps"
        );
        Ok(Some(metadata))
    }

    fn fingerprint(&self) -> sha2::digest::Output<Sha256> {
        let mut hash = Sha256::new();
        hash.update(self.duration_seconds.to_le_bytes());
        hash.update(self.source_fps.to_le_bytes());
        for time in &self.sample_pts_seconds {
            hash.update(time.to_le_bytes());
        }
        hash.finalize()
    }
}

#[derive(Deserialize)]
struct Probe {
    streams: Vec<Stream>,
    format: Option<Format>,
}

#[derive(Deserialize)]
struct Stream {
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<String>,
    avg_frame_rate: Option<String>,
    bit_rate: Option<String>,
}

#[derive(Deserialize)]
struct Format {
    duration: Option<String>,
    bit_rate: Option<String>,
}

#[derive(Deserialize)]
struct FrameClock {
    frames: Vec<PresentationTime>,
}

#[derive(Deserialize)]
struct PresentationTime {
    best_effort_timestamp_time: Option<String>,
}

pub(crate) fn load(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<Option<Video>> {
    let remote = Remote::load(&file.path).map_err(|error| input_error(file, error))?;
    load_with_remote(file, ctx, remote.as_ref())
}

pub(super) fn feature_key(fps: f64, model: &str, remote: Option<&Remote>) -> String {
    let max_frames = (fps * 600.0).ceil().max(1.0) as usize;
    if let Some(remote) = remote {
        format!(
            "{FEATURE};fps={fps};cap={max_frames};model={model};sparse-source-clock/v1={:x}",
            remote.fingerprint()
        )
    } else {
        format!("{FEATURE};fps={fps};cap={max_frames};model={model}")
    }
}

pub(super) fn uncached_embedding_work(file: &FileEntry, ctx: &Ctx<'_>) -> anyhow::Result<usize> {
    let Some(model) = ctx.embedder.filter(|_| !ctx.config.no_embeddings) else {
        return Ok(0);
    };
    let fps = ctx.config.video_fps;
    ensure!(
        fps.is_finite() && fps > 0.0,
        "video_fps must be positive and finite"
    );
    let maximum = (fps * 600.0).ceil().max(1.0) as usize;
    // Bad sidecars are skipped by extraction; counting their maximum is conservative.
    let Ok(remote) = Remote::load(&file.path) else {
        return Ok(maximum);
    };
    for id in model.frame_cache_model_ids().into_iter().flatten() {
        if ctx
            .cache
            .contains(file, &feature_key(fps, id, remote.as_ref()))?
        {
            return Ok(0);
        }
    }
    Ok(remote.map_or(maximum, |remote| {
        remote.sample_pts_seconds.len().min(maximum)
    }))
}

pub(crate) fn load_with_remote(
    file: &FileEntry,
    ctx: &Ctx<'_>,
    remote: Option<&Remote>,
) -> anyhow::Result<Option<Video>> {
    let fps = ctx.config.video_fps;
    ensure!(
        fps.is_finite() && fps > 0.0,
        "video_fps must be positive and finite"
    );
    let max_frames = (fps * 600.0).ceil().max(1.0) as usize;
    let embedder = ctx.embedder.filter(|_| !ctx.config.no_embeddings);
    let ids = embedder.map_or([Some("none"), None], Embedder::frame_cache_model_ids);
    for id in ids.into_iter().flatten() {
        let key = feature_key(fps, id, remote);
        if let Some(bytes) = ctx.cache.get(file, &key)? {
            return serde_json::from_slice::<Option<Video>>(&bytes)
                .context("reading cached video features");
        }
    }
    let mut video = probe(&file.path).map_err(|error| input_error(file, error))?;
    if let Some(video) = video.as_mut() {
        video.max_frames = max_frames;
        video.step = (video.duration / max_frames as f64).max(1.0 / fps);
        if let Some(remote) = remote {
            video.duration = remote.duration_seconds;
            video.fps = remote.source_fps;
            let count = remote.sample_pts_seconds.len();
            let stride = count.div_ceil(max_frames);
            video.step = video.duration / count.div_ceil(stride) as f64;
        }
        video.frames = decode(
            &file.path,
            video,
            embedder,
            None,
            remote.map(|remote| remote.sample_pts_seconds.as_slice()),
            None,
        )
        .map_err(|error| input_error(file, error))?;
        if video.frames.is_empty() {
            return Err(eidolon_core::InputError {
                file: file.id,
                reason: "video stream decoded no frames".to_owned(),
            }
            .into());
        }
    }
    let key = feature_key(
        fps,
        embedder.map_or("none", Embedder::frame_model_id),
        remote,
    );
    ctx.cache.put(file, &key, &serde_json::to_vec(&video)?)?;
    Ok(video)
}

fn input_error(file: &FileEntry, error: anyhow::Error) -> anyhow::Error {
    // An absent FFmpeg executable is a run-wide setup failure, not a bad input.
    if error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == ErrorKind::NotFound)
    {
        return error;
    }
    eidolon_core::InputError {
        file: file.id,
        reason: format!("{error:#}"),
    }
    .into()
}

fn number(value: Option<&str>) -> Option<f64> {
    value?
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn probe(path: &Path) -> anyhow::Result<Option<Video>> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "V:0",
            "-show_entries",
            "stream=width,height,duration,avg_frame_rate,bit_rate:format=duration,bit_rate",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("launching ffprobe (install FFmpeg to inspect videos)")?;
    ensure!(
        output.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let metadata: Probe =
        serde_json::from_slice(&output.stdout).context("parsing ffprobe metadata")?;
    let Some(stream) = metadata.streams.first() else {
        return Ok(None);
    };
    let format = metadata.format.as_ref();
    let duration = number(stream.duration.as_deref())
        .or_else(|| number(format.and_then(|format| format.duration.as_deref())))
        .context("video has no finite positive duration")?;
    let width = stream
        .width
        .filter(|value| *value > 0)
        .context("video has no width")?;
    let height = stream
        .height
        .filter(|value| *value > 0)
        .context("video has no height")?;
    let fps = stream
        .avg_frame_rate
        .as_deref()
        .and_then(|rate| {
            let (numerator, denominator) = rate.split_once('/')?;
            Some(number(Some(numerator))? / number(Some(denominator))?)
        })
        .unwrap_or(1.0);
    let bitrate = stream
        .bit_rate
        .as_deref()
        .or_else(|| format.and_then(|format| format.bit_rate.as_deref()))
        .and_then(|value| value.parse().ok());
    Ok(Some(Video {
        duration,
        width,
        height,
        fps,
        bitrate,
        step: 1.0 / fps,
        max_frames: MAX_FRAMES,
        frames: Vec::new(),
    }))
}

fn frame_filter(
    video: &Video,
    timestamps: Option<&[f64]>,
    indices: Option<&[usize]>,
) -> (String, usize) {
    let stride = timestamps.map_or(1, |times| {
        if indices.is_some() {
            1
        } else {
            times.len().div_ceil(video.max_frames)
        }
    });
    let filter = if timestamps.is_some() {
        let select = indices.map_or_else(
            || format!("not(mod(n,{stride}))"),
            |indices| {
                indices
                    .iter()
                    .map(|index| format!("eq(n,{index})"))
                    .collect::<Vec<_>>()
                    .join("+")
            },
        );
        format!("select='{select}',scale={WIDTH}:{HEIGHT}:flags=area")
    } else {
        format!(
            "fps=fps={:.12}:start_time=0,scale={WIDTH}:{HEIGHT}:flags=area",
            1.0 / video.step
        )
    };
    (filter, stride)
}

fn decode(
    path: &Path,
    video: &Video,
    embedder: Option<&dyn Embedder>,
    start: Option<f64>,
    timestamps: Option<&[f64]>,
    indices: Option<&[usize]>,
) -> anyhow::Result<Vec<Frame>> {
    let (filter, stride) = frame_filter(video, timestamps, indices);
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        "file,pipe",
        "-threads",
        "1",
    ]);
    if let Some(start) = start {
        command.args(["-ss", &format!("{start:.9}")]);
    }
    command.arg("-i").arg(path);
    if start.is_some() {
        command.args(["-t", &format!("{:.9}", video.duration)]);
    }
    if timestamps.is_some() {
        command.args(["-fps_mode", "passthrough"]);
    }
    let mut child = command
        .args([
            "-map",
            "0:V:0",
            "-an",
            "-sn",
            "-dn",
            "-filter_threads",
            "1",
            "-vf",
            &filter,
            "-frames:v",
            &video.max_frames.to_string(),
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("launching ffmpeg for frame decoding")?;
    let mut stdout = child
        .stdout
        .take()
        .context("FFmpeg stdout pipe unavailable")?;
    let mut stderr = child
        .stderr
        .take()
        .context("FFmpeg stderr pipe unavailable")?;
    std::thread::scope(|scope| {
        let errors = scope.spawn(move || {
            let mut message = String::new();
            let _ = stderr.read_to_string(&mut message);
            message
        });
        let frames = read_frames(
            &mut stdout,
            video,
            embedder,
            path,
            timestamps,
            stride,
            start.is_some() || indices.is_some(),
        );
        if frames.is_err() {
            let _ = child.kill();
        }
        // Release the pipe before waiting, including on an early read failure.
        drop(stdout);
        let status = child.wait().context("waiting for FFmpeg")?;
        let message = errors
            .join()
            .unwrap_or_else(|_| "stderr reader failed".to_owned());
        ensure!(
            status.success(),
            "ffmpeg decoding failed: {}",
            message.trim()
        );
        frames
    })
}

fn read_frames(
    reader: &mut impl std::io::Read,
    video: &Video,
    mut embedder: Option<&dyn Embedder>,
    path: &Path,
    timestamps: Option<&[f64]>,
    stride: usize,
    retain_pixels: bool,
) -> anyhow::Result<Vec<Frame>> {
    let step = video.step;
    let mut rgb = RgbImage::new(WIDTH, HEIGHT);
    let capacity = timestamps.map_or_else(
        || (video.duration / step).ceil() as usize,
        |times| times.len().div_ceil(stride).min(video.max_frames),
    );
    let mut frames = Vec::with_capacity(capacity);
    let mut next_embedding = 0.0;
    loop {
        let pixels: &mut [u8] = rgb.as_mut();
        match reader.read(&mut pixels[..1]) {
            Ok(0) => break,
            Ok(_) => (),
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error).context("reading raw video frame"),
        }
        reader
            .read_exact(&mut pixels[1..])
            .context("truncated raw video frame")?;
        let time = if let Some(timestamps) = timestamps {
            *timestamps
                .get(frames.len() * stride)
                .context("remote proxy contains more frames than its source clock")?
        } else {
            frames.len() as f64 * step
        };
        let mut frame = describe_frame(&rgb, time, !retain_pixels);
        if time + 1e-6 >= next_embedding {
            if let Some(model) = embedder {
                match model
                    .prepare()
                    .and_then(|()| model.embed_frame(&rgb))
                    .and_then(normalize)
                {
                    Ok(vector) => frame.embedding = Some(vector),
                    Err(error) => {
                        eprintln!(
                            "eidolon-video: {}: embedding failed; retaining frame hashes: {error:#}",
                            path.display()
                        );
                        embedder = None;
                    }
                }
            }
            next_embedding = time
                + if timestamps.is_some() {
                    EMBED_INTERVAL
                } else {
                    EMBED_INTERVAL.max(step)
                };
        }
        if retain_pixels {
            frame.retain_pixels(
                std::mem::replace(&mut rgb, RgbImage::new(WIDTH, HEIGHT)),
                f64::from(video.width) / f64::from(video.height),
            );
        }
        frames.push(frame);
    }
    if let Some(timestamps) = timestamps {
        ensure!(
            frames.len() == timestamps.len().div_ceil(stride).min(video.max_frames),
            "remote proxy decoded {} frames but source clock expects {}",
            frames.len(),
            timestamps.len().div_ceil(stride).min(video.max_frames)
        );
    }
    Ok(frames)
}

pub(crate) fn native(
    file: &FileEntry,
    source: &Video,
    cache: &FeatureCache,
) -> anyhow::Result<Option<Vec<Frame>>> {
    if (source.duration * source.fps).ceil() > MAX_FRAMES as f64 {
        return Ok(None);
    }
    let Some(period) = native_period(file, cache)? else {
        return Ok(None);
    };
    if (period - 1.0 / source.fps).abs() > 2e-6 {
        return Ok(None);
    }
    refine(file, source, source.duration * 0.5, source.duration * 0.5).map(Some)
}

fn native_period(file: &FileEntry, cache: &FeatureCache) -> anyhow::Result<Option<f64>> {
    const KEY: &str = "video.native-clock/v1;cap=1200";
    if let Some(bytes) = cache.get(file, KEY)? {
        return serde_json::from_slice(&bytes).context("parsing cached native video cadence");
    }
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "V:0",
            "-read_intervals",
            "%+#1201",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
        ])
        .arg(&file.path)
        .stdin(Stdio::null())
        .output()
        .context("probing native video cadence")?;
    ensure!(
        output.status.success(),
        "probing native video cadence failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let clock: FrameClock =
        serde_json::from_slice(&output.stdout).context("parsing native video cadence")?;
    let valid_length = (2..=MAX_FRAMES).contains(&clock.frames.len());
    let mut times = clock.frames.into_iter().map(|frame| {
        frame
            .best_effort_timestamp_time?
            .parse::<f64>()
            .ok()
            .filter(|time| time.is_finite())
    });
    let period =
        times
            .next()
            .flatten()
            .zip(times.next().flatten())
            .and_then(|(first, mut previous)| {
                let period = previous - first;
                (valid_length
                    && period > 0.0
                    && times.all(|time| {
                        let Some(time) = time else {
                            return false;
                        };
                        let difference = time - previous;
                        previous = time;
                        (difference - period).abs() <= 2e-6
                    }))
                .then_some(period)
            });
    cache.put(file, KEY, &serde_json::to_vec(&period)?)?;
    Ok(period)
}

pub(crate) fn refine(
    file: &FileEntry,
    source: &Video,
    center: f64,
    radius: f64,
) -> anyhow::Result<Vec<Frame>> {
    refine_path(&file.path, source, center, radius)
}

fn refine_path(
    path: &Path,
    source: &Video,
    center: f64,
    radius: f64,
) -> anyhow::Result<Vec<Frame>> {
    let start = (center - radius).max(0.0);
    let duration = (center + radius).min(source.duration) - start;
    if duration <= 0.0 {
        return Ok(Vec::new());
    }
    let window = Video {
        duration,
        width: source.width,
        height: source.height,
        fps: source.fps,
        bitrate: source.bitrate,
        step: (1.0 / source.fps).max(duration / MAX_FRAMES as f64),
        max_frames: MAX_FRAMES,
        frames: Vec::new(),
    };
    let mut frames = decode(path, &window, None, Some(start), None, None)?;
    for frame in &mut frames {
        frame.time += start;
    }
    Ok(frames)
}

pub(crate) fn refined_window(
    window: &RefinedWindow,
    center: f64,
    radius: f64,
) -> anyhow::Result<Vec<Frame>> {
    ensure!(
        window.source_offset_s.is_finite() && window.source_offset_s >= 0.0,
        "refined window has an invalid source-clock offset"
    );
    let source = probe(&window.path)?.context("refined window has no video stream")?;
    let mut frames = refine_path(
        &window.path,
        &source,
        center - window.source_offset_s,
        radius,
    )?;
    for frame in &mut frames {
        frame.time += window.source_offset_s;
    }
    Ok(frames)
}

pub(crate) fn retained(
    file: &FileEntry,
    video: &Video,
    remote: &Remote,
    frames: &[usize],
) -> anyhow::Result<Vec<Frame>> {
    let stride = remote.sample_pts_seconds.len().div_ceil(video.max_frames);
    let mut indices: Vec<_> = frames.iter().map(|index| index * stride).collect();
    indices.sort_unstable();
    indices.dedup();
    let times: Vec<_> = indices
        .iter()
        .map(|index| remote.sample_pts_seconds[*index])
        .collect();
    decode(&file.path, video, None, None, Some(&times), Some(&indices))
}

pub(crate) fn normalize(mut vector: Vec<f32>) -> anyhow::Result<Vec<f32>> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if vector.is_empty() || !norm.is_finite() || norm <= 0.0 {
        bail!("embedder returned an empty or non-finite vector");
    }
    for value in &mut vector {
        *value /= norm;
    }
    Ok(vector)
}

fn black_border(
    length: u32,
    cross: std::ops::Range<u32>,
    pixel: impl Fn(u32, u32) -> [u8; 3],
) -> u32 {
    let reference = pixel(0, cross.start);
    if reference.into_iter().any(|value| value > 12) {
        return 0;
    }
    let count = (0..length / 4)
        .take_while(|position| {
            cross.clone().all(|coordinate| {
                pixel(*position, coordinate)
                    .into_iter()
                    .zip(reference)
                    .all(|(value, reference)| value <= 12 && value.abs_diff(reference) <= 3)
            })
        })
        .count() as u32;
    if count == 0 {
        return 0;
    }
    // A dark scene gradient is not a border. Require a uniform strip and a
    // visible boundary; mixed resampling rows may soften that boundary.
    let mut contrast = [0_i32; 3];
    for coordinate in cross.clone() {
        let before = pixel(count - 1, coordinate);
        let after = pixel((count + 2).min(length - 1), coordinate);
        for (channel, difference) in contrast.iter_mut().enumerate() {
            *difference += i32::from(after[channel]) - i32::from(before[channel]);
        }
    }
    if contrast
        .into_iter()
        .any(|difference| difference.unsigned_abs() >= (cross.end - cross.start) * 8)
    {
        // Once a uniform core and a real edge establish padding, include its
        // near-black codec/resampling transition rather than retaining a rim.
        count
            + (count..(count + 2).min(length / 4))
                .take_while(|position| {
                    cross.clone().all(|coordinate| {
                        pixel(*position, coordinate)
                            .into_iter()
                            .all(|value| value <= 12)
                    })
                })
                .count() as u32
    } else {
        0
    }
}

fn content_bounds(rgb: &RgbImage) -> (u32, u32, u32, u32) {
    let top = black_border(HEIGHT, 0..WIDTH, |y, x| rgb.get_pixel(x, y).0);
    let bottom = black_border(HEIGHT, 0..WIDTH, |y, x| rgb.get_pixel(x, HEIGHT - 1 - y).0);
    let left = black_border(WIDTH, top..HEIGHT - bottom, |x, y| rgb.get_pixel(x, y).0);
    let right = black_border(WIDTH, top..HEIGHT - bottom, |x, y| {
        rgb.get_pixel(WIDTH - 1 - x, y).0
    });
    (left, top, WIDTH - right, HEIGHT - bottom)
}

pub(crate) fn describe(rgb: &RgbImage, time: f64) -> Frame {
    describe_frame(rgb, time, true)
}

fn describe_frame(rgb: &RgbImage, time: f64, indexed: bool) -> Frame {
    let mut gray = [0.0_f32; 32 * 32];
    let mut colors = [[0_u32; 3]; 16];
    let mut rgb_sums = [[0_u16; 3]; 1024];
    let mut gray_counts = [0_u8; 1024];
    let mut color_counts = [0_u16; 16];
    let (left, top, right, bottom) = content_bounds(rgb);
    for y in top..bottom {
        for x in left..right {
            let pixel = rgb.get_pixel(x, y);
            let gx = ((x - left) * 32 / (right - left)) as usize;
            let gy = ((y - top) * 32 / (bottom - top)) as usize;
            let gray_index = gy * 32 + gx;
            gray[gray_index] += f32::from(pixel[0]) * 0.299
                + f32::from(pixel[1]) * 0.587
                + f32::from(pixel[2]) * 0.114;
            gray_counts[gray_index] += 1;
            let color_index = (gy / 8) * 4 + gx / 8;
            for channel in 0..3 {
                colors[color_index][channel] += u32::from(pixel[channel]);
                rgb_sums[gray_index][channel] += u16::from(pixel[channel]);
            }
            color_counts[color_index] += 1;
        }
    }
    for (value, count) in gray.iter_mut().zip(&gray_counts) {
        *value /= f32::from(*count);
    }
    let tiles = std::array::from_fn(|index| {
        colors[index].map(|sum| (sum / u32::from(color_counts[index])) as u8)
    });
    let color = std::array::from_fn(|index| {
        let quadrant = index / 3;
        let corner = (quadrant / 2) * 8 + (quadrant % 2) * 2;
        let channel = index % 3;
        ([corner, corner + 1, corner + 4, corner + 5]
            .iter()
            .map(|index| u32::from(tiles[*index][channel]))
            .sum::<u32>()
            / 4) as u8
    });
    let rgb_thumbnail: Vec<_> = rgb_sums
        .iter()
        .zip(&gray_counts)
        .map(|(rgb, count)| {
            rgb.map(|sum| ((sum + u16::from(*count) / 2) / u16::from(*count)) as u8)
        })
        .collect();
    Frame {
        time,
        phash: perceptual_hash(&gray),
        dhash: difference_hash(&gray, 0, 0, 32),
        color,
        tiles,
        regions: std::array::from_fn(|index| {
            difference_hash(&gray, (index % 2) * 16, (index / 2) * 16, 16)
        }),
        color_hashes: indexed.then(|| color_hashes(&gray, &rgb_thumbnail)),
        thumbnail: gray.iter().map(|value| value.round() as u8).collect(),
        rgb_thumbnail,
        pixels: None,
        overlay: None,
        embedding: None,
        source_overlay: OnceLock::new(),
    }
}

fn color_hashes(gray: &[f32; 1024], rgb: &[[u8; 3]]) -> [u64; 6] {
    std::array::from_fn(|channel| {
        let mut plane = [0.0; 1024];
        let mut counts = [0_u32; 256];
        for (index, pixel) in rgb.iter().enumerate() {
            let value = match channel {
                0 => gray[index].round() as u8,
                1 => pixel[0].max(pixel[1]).max(pixel[2]),
                2 => pixel[0].min(pixel[1]).min(pixel[2]),
                _ => pixel[channel - 3],
            };
            plane[index] = f32::from(value);
            counts[usize::from(value)] += 1;
        }
        let mut total = 0;
        let ranks = counts.map(|count| {
            let rank = (total as f32 + count as f32 * 0.5) * (255.0 / 1024.0);
            total += count;
            rank
        });
        for value in &mut plane {
            *value = ranks[*value as usize];
        }
        perceptual_hash(&plane)
    })
}

fn difference_hash(gray: &[f32; 1024], left: usize, top: usize, size: usize) -> u64 {
    let mut hash = 0;
    for y in 0..8 {
        let mut previous = 0.0;
        for x in 0..9 {
            let start = left + x * size / 9;
            let end = left + (x + 1) * size / 9;
            let mut total = 0.0;
            for row in top + y * size / 8..top + (y + 1) * size / 8 {
                total += gray[row * 32 + start..row * 32 + end].iter().sum::<f32>();
            }
            let value = total / ((end - start) * (size / 8)) as f32;
            if x > 0 && value > previous {
                hash |= 1 << (y * 8 + x - 1);
            }
            previous = value;
        }
    }
    hash
}

fn perceptual_hash(gray: &[f32; 1024]) -> u64 {
    static COSINES: LazyLock<[[f32; 32]; 8]> = LazyLock::new(|| {
        std::array::from_fn(|frequency| {
            std::array::from_fn(|position| {
                ((2 * position + 1) as f32 * frequency as f32 * std::f32::consts::PI / 64.0).cos()
            })
        })
    });
    let cosines = &*COSINES;
    let mut rows = [[0.0_f32; 8]; 32];
    for (y, row) in rows.iter_mut().enumerate() {
        for (frequency, value) in row.iter_mut().enumerate() {
            *value = gray[y * 32..(y + 1) * 32]
                .iter()
                .zip(&cosines[frequency])
                .map(|(pixel, cosine)| pixel * cosine)
                .sum();
        }
    }
    let mut coefficients = [0.0_f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            coefficients[y * 8 + x] = rows
                .iter()
                .zip(&cosines[y])
                .map(|(row, cosine)| row[x] * cosine)
                .sum();
        }
    }
    let mut sorted: [f32; 63] = std::array::from_fn(|index| coefficients[index + 1]);
    sorted.sort_unstable_by(f32::total_cmp);
    let median = sorted[31];
    coefficients[1..]
        .iter()
        .enumerate()
        .fold(0, |hash, (bit, value)| {
            hash | (u64::from(*value > median) << bit)
        })
}

pub(crate) fn hash_similarity(a: &Frame, b: &Frame, fast: bool) -> Option<f64> {
    let phash = (a.phash ^ b.phash).count_ones();
    if phash > 12 {
        return None;
    }
    let dhash = (a.dhash ^ b.dhash).count_ones();
    let color = if fast {
        0.0
    } else {
        a.color
            .iter()
            .zip(b.color)
            .map(|(a, b)| f64::from(a.abs_diff(b)))
            .sum::<f64>()
            / (12.0 * 255.0)
    };
    if dhash > 18 || color > 0.13 {
        return None;
    }
    Some(1.0 - f64::from(phash) / 126.0 - f64::from(dhash) / 256.0 - color * 0.5)
}

pub(crate) fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    (a.len() == b.len() && !a.is_empty())
        .then(|| f64::from(a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>()).clamp(-1.0, 1.0))
}

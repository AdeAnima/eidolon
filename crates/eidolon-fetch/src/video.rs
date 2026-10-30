use std::io::{Read, Seek, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, ensure};
use mp4::{Mp4Reader, TrackType, WriteBox as _};
use serde::Serialize;
use url::Url;

use crate::http::{Headers, RangeSource, Session, inspect_boxes};
use crate::{VideoMode, save};

#[derive(Serialize)]
struct SourceTimes<'a> {
    version: u8,
    duration_seconds: f64,
    source_fps: f64,
    sample_pts_seconds: &'a [f64],
}

pub(crate) struct Plan {
    pub(crate) track_id: u32,
    pub(crate) duration_ticks: u64,
    pub(crate) duration_seconds: f64,
    pub(crate) source_fps: f64,
    pub(crate) samples: Vec<u32>,
}

pub(crate) struct ProxyWriter {
    output: tempfile::NamedTempFile,
    movie: mp4::MoovBox,
    mdat_start: u64,
    offsets: Vec<u64>,
    sizes: Vec<u32>,
    sync: Vec<u32>,
    durations: Vec<(u32, u32)>,
    composition: Vec<(u32, i32)>,
}

pub(crate) fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".remote.json");
    PathBuf::from(name)
}

pub(crate) fn proxy(
    session: &Session<'_>,
    url: &Url,
    headers: &Headers,
    path: &Path,
    remaining_budget: u64,
) -> anyhow::Result<()> {
    let size = headers
        .total_size
        .context("indexed video needs Content-Length")?;
    let mut source = RangeSource::new(session, url, headers, remaining_budget)?;
    if let Some(index) = crate::fragment::index(&mut source, size)? {
        return crate::fragment_video::proxy(session, &mut source, &index, path);
    }
    source.rewind()?;
    inspect_boxes(&mut source, size)?;
    let mut reader = Mp4Reader::read_header(source, size).context("parsing MP4 sample index")?;
    let sparse = session.options.video_mode == VideoMode::Keyframes;
    let plan = plan(&reader, sparse, session.options.max_video_bytes)?;
    let mut writer = ProxyWriter::new(&reader, &plan, path)?;
    let mut source_pts = Vec::with_capacity(if sparse { plan.samples.len() } else { 0 });
    for (index, &sample_id) in plan.samples.iter().enumerate() {
        session.check_deadline()?;
        let sample = reader
            .read_sample(plan.track_id, sample_id)?
            .context("missing MP4 sample")?;
        let track = reader
            .tracks()
            .get(&plan.track_id)
            .context("missing MP4 track")?;
        let duration = if sparse {
            let end = plan
                .samples
                .get(index + 1)
                .map_or(Ok(plan.duration_ticks), |next| decode_time(track, *next));
            u32::try_from(
                end?.checked_sub(sample.start_time)
                    .context("non-monotonic MP4 samples")?,
            )?
        } else {
            sample.duration
        };
        writer.push(&sample, duration)?;
        if sparse {
            source_pts.push(presentation_time(
                track,
                sample.start_time,
                sample.rendering_offset,
                reader.timescale(),
            )?);
        }
    }
    writer.finish(path)?;
    retain_timeline(path, &plan, sparse.then_some(source_pts.as_slice()))
}

pub(crate) fn retain_timeline(
    path: &Path,
    plan: &Plan,
    points: Option<&[f64]>,
) -> anyhow::Result<()> {
    if let Some(sample_pts_seconds) = points {
        let metadata = SourceTimes {
            version: 1,
            duration_seconds: plan.duration_seconds,
            source_fps: plan.source_fps,
            sample_pts_seconds,
        };
        save(&sidecar(path), &serde_json::to_vec(&metadata)?)?;
    } else if sidecar(path).exists() {
        std::fs::remove_file(sidecar(path))?;
    }
    Ok(())
}

pub(crate) fn validate_codec(track: &mp4::Mp4Track) -> anyhow::Result<()> {
    ensure!(
        matches!(
            track.media_type().context("unsupported MP4 video codec")?,
            mp4::MediaType::H264 | mp4::MediaType::VP9
        ),
        "HEVC/unsupported decoder configuration is not preserved by the MP4 parser"
    );
    Ok(())
}

pub(crate) fn plan<R: Read + Seek>(
    reader: &Mp4Reader<R>,
    sparse: bool,
    cap: u64,
) -> anyhow::Result<Plan> {
    ensure!(
        !reader.is_fragmented(),
        "fragmented MP4 needs bounded full download"
    );
    let track = reader
        .tracks()
        .values()
        .filter(|track| {
            track
                .track_type()
                .is_ok_and(|kind| kind == TrackType::Video)
        })
        .min_by_key(|track| track.track_id())
        .context("MP4 has no video track")?;
    validate_codec(track)?;
    ensure!(
        track.timescale() > 0 && reader.timescale() > 0,
        "invalid MP4 timescale"
    );
    let duration_seconds = reader.moov.mvhd.duration as f64 / f64::from(reader.timescale());
    ensure!(
        duration_seconds > 0.0 && duration_seconds.is_finite(),
        "invalid MP4 duration"
    );
    let samples = select_samples(track, sparse, duration_seconds)?;
    ensure!(!samples.is_empty(), "MP4 has no independent sync samples");
    for &sample_id in &samples {
        let table = &track.trak.mdia.minf.stbl.stsz;
        let length = if table.sample_size > 0 {
            table.sample_size
        } else {
            *table
                .sample_sizes
                .get((sample_id - 1) as usize)
                .context("invalid MP4 sample size index")?
        };
        ensure!(
            u64::from(length) <= cap,
            "MP4 sample exceeds video transfer cap"
        );
    }
    ensure!(
        track
            .trak
            .mdia
            .minf
            .stbl
            .stsc
            .entries
            .iter()
            .all(|entry| entry.sample_description_index == 1),
        "multiple MP4 sample descriptions need full download"
    );
    Ok(Plan {
        track_id: track.track_id(),
        duration_ticks: track.trak.mdia.mdhd.duration,
        duration_seconds,
        source_fps: track.frame_rate(),
        samples,
    })
}

impl ProxyWriter {
    pub(crate) fn new<R: Read + Seek>(
        reader: &Mp4Reader<R>,
        plan: &Plan,
        path: &Path,
    ) -> anyhow::Result<Self> {
        let track = reader
            .tracks()
            .get(&plan.track_id)
            .context("missing MP4 track")?;
        let movie = mp4::MoovBox {
            mvhd: reader.moov.mvhd.clone(),
            meta: None,
            mvex: None,
            traks: vec![track.trak.clone()],
            udta: None,
        };
        let mut output =
            tempfile::NamedTempFile::new_in(path.parent().context("proxy path has no parent")?)?;
        reader.ftyp.write_box(output.as_file_mut())?;
        let mdat_start = output.stream_position()?;
        output.write_all(&1_u32.to_be_bytes())?;
        output.write_all(b"mdat")?;
        output.write_all(&0_u64.to_be_bytes())?;
        Ok(Self {
            output,
            movie,
            mdat_start,
            offsets: Vec::with_capacity(plan.samples.len()),
            sizes: Vec::with_capacity(plan.samples.len()),
            sync: Vec::new(),
            durations: Vec::new(),
            composition: Vec::new(),
        })
    }

    pub(crate) fn push(&mut self, sample: &mp4::Mp4Sample, duration: u32) -> anyhow::Result<()> {
        ensure!(duration > 0, "zero-duration MP4 sample");
        self.offsets.push(self.output.stream_position()?);
        self.sizes.push(u32::try_from(sample.bytes.len())?);
        if sample.is_sync {
            self.sync.push(u32::try_from(self.sizes.len())?);
        }
        append_run(&mut self.durations, duration);
        append_run(&mut self.composition, sample.rendering_offset);
        self.output.write_all(&sample.bytes)?;
        Ok(())
    }

    pub(crate) fn set_duration(&mut self, ticks: u64, scale: u32) -> anyhow::Result<()> {
        let duration = u64::try_from(
            u128::from(ticks) * u128::from(self.movie.mvhd.timescale) / u128::from(scale),
        )?;
        self.movie.mvhd.duration = duration;
        let track = self
            .movie
            .traks
            .first_mut()
            .context("missing output video track")?;
        track.tkhd.duration = duration;
        track.mdia.mdhd.duration = ticks;
        Ok(())
    }

    pub(crate) fn normalize_window(&mut self, ticks: u64, scale: u32) -> anyhow::Result<()> {
        self.set_duration(ticks, scale)?;
        let track = self
            .movie
            .traks
            .first_mut()
            .context("missing output video track")?;
        track.edts = None;
        if let Some(ctts) = &mut track.mdia.minf.stbl.ctts {
            ctts.version = 1;
        }
        Ok(())
    }

    pub(crate) fn finish(mut self, path: &Path) -> anyhow::Result<()> {
        let end = self.output.stream_position()?;
        self.output.seek(SeekFrom::Start(self.mdat_start + 8))?;
        self.output
            .write_all(&(end - self.mdat_start).to_be_bytes())?;
        self.output.seek(SeekFrom::Start(end))?;
        // Nested box types are private in mp4. Reuse inferred types and original
        // codec parameters instead of reconstructing lossy decoder configuration.
        let table = &mut self
            .movie
            .traks
            .first_mut()
            .context("missing output video track")?
            .mdia
            .minf
            .stbl;
        let template = table.stts.entries.first().cloned().unwrap_or_default();
        table.stts.entries = self
            .durations
            .into_iter()
            .map(|(count, delta)| {
                let mut entry = template.clone();
                entry.sample_count = count;
                entry.sample_delta = delta;
                entry
            })
            .collect();
        if table.ctts.is_some() || self.composition.iter().any(|(_, offset)| *offset != 0) {
            let ctts = table.ctts.get_or_insert_default();
            if self.composition.iter().any(|(_, offset)| *offset < 0) {
                ctts.version = 1;
            }
            let template = ctts.entries.first().cloned().unwrap_or_default();
            ctts.entries = self
                .composition
                .into_iter()
                .map(|(count, offset)| {
                    let mut entry = template.clone();
                    entry.sample_count = count;
                    entry.sample_offset = offset;
                    entry
                })
                .collect();
        }
        if self.sync.len() == self.sizes.len() {
            table.stss = None;
        } else {
            table.stss.get_or_insert_default().entries = self.sync;
        }
        table.stsz.sample_size = 0;
        table.stsz.sample_count = u32::try_from(self.sizes.len())?;
        table.stsz.sample_sizes = self.sizes;
        if let Some(stco) = &mut table.stco {
            stco.entries = self
                .offsets
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?;
            table.co64 = None;
        } else if let Some(co64) = &mut table.co64 {
            co64.entries = self.offsets;
        } else {
            table.co64.get_or_insert_default().entries = self.offsets;
        }
        table.stsc.entries.resize_with(1, Default::default);
        let chunk = table
            .stsc
            .entries
            .first_mut()
            .context("MP4 has no chunk index")?;
        chunk.first_chunk = 1;
        chunk.samples_per_chunk = 1;
        chunk.sample_description_index = 1;
        chunk.first_sample = 1;
        self.movie.write_box(self.output.as_file_mut())?;
        self.output.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

fn append_run<T: Copy + Eq>(entries: &mut Vec<(u32, T)>, value: T) {
    if let Some((count, previous)) = entries.last_mut()
        && *previous == value
    {
        *count += 1;
    } else {
        entries.push((1, value));
    }
}

fn select_samples(track: &mp4::Mp4Track, sparse: bool, duration: f64) -> anyhow::Result<Vec<u32>> {
    let count = track.sample_count();
    ensure!(
        count > 0 && count <= 2_000_000,
        "MP4 sample count exceeds supported index limit"
    );
    if !sparse {
        return Ok((1..=count).collect());
    }
    let table = &track.trak.mdia.minf.stbl;
    let mut samples = Vec::new();
    let step = (duration / 1200.0).max(0.5);
    let mut next = 0.0;
    let mut consider = |sample: u32| -> anyhow::Result<()> {
        ensure!(
            sample > 0 && sample <= count,
            "invalid MP4 sync-sample index"
        );
        let time = decode_time(track, sample)? as f64 / f64::from(track.timescale());
        if time + 1e-8 >= next && samples.len() < 1200 {
            samples.push(sample);
            next = time + step;
        }
        Ok(())
    };
    if let Some(sync) = &table.stss {
        ensure!(
            sync.entries.windows(2).all(|ids| ids[0] < ids[1]),
            "unordered MP4 sync-sample table"
        );
        for &sample in &sync.entries {
            consider(sample)?;
        }
    } else {
        for sample in 1..=count {
            consider(sample)?;
        }
    }
    Ok(samples)
}

pub(crate) fn decode_time(track: &mp4::Mp4Track, sample: u32) -> anyhow::Result<u64> {
    let mut remaining = u64::from(sample.checked_sub(1).context("sample IDs start at one")?);
    let mut time = 0_u64;
    for entry in &track.trak.mdia.minf.stbl.stts.entries {
        let count = remaining.min(u64::from(entry.sample_count));
        time = time
            .checked_add(count * u64::from(entry.sample_delta))
            .context("MP4 timestamp overflow")?;
        if remaining < u64::from(entry.sample_count) {
            return Ok(time);
        }
        remaining -= count;
    }
    anyhow::bail!("missing MP4 decode timestamp")
}

pub(crate) fn presentation_time(
    track: &mp4::Mp4Track,
    start: u64,
    offset: i32,
    movie_scale: u32,
) -> anyhow::Result<f64> {
    let pts = i128::from(start) + i128::from(offset);
    let scale = f64::from(track.timescale());
    let Some(edits) = track
        .trak
        .edts
        .as_ref()
        .and_then(|edits| edits.elst.as_ref())
    else {
        return Ok(pts as f64 / scale);
    };
    let mut timeline = 0.0;
    for edit in &edits.entries {
        ensure!(
            edit.media_rate == 1 && edit.media_rate_fraction == 0,
            "non-unit MP4 edit rate unsupported"
        );
        let duration = edit.segment_duration as f64 / f64::from(movie_scale);
        // Empty edits encode signed -1 using the version's unsigned field width.
        if edit.media_time != u64::MAX && edit.media_time != u64::from(u32::MAX) {
            let relative = (pts - i128::from(edit.media_time)) as f64 / scale;
            if relative >= -1e-8 && relative < duration + 1e-8 {
                return Ok(timeline + relative.max(0.0));
            }
        }
        timeline += duration;
    }
    anyhow::bail!("sync sample lies outside supported MP4 edit timeline")
}

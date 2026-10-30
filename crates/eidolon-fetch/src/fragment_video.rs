use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context as _, ensure};

use crate::VideoMode;
use crate::fragment::{FragmentIndex, Sample};
use crate::http::Session;
use crate::video::{self, Plan, ProxyWriter};

pub(crate) fn source_time(index: &FragmentIndex, sample: &Sample) -> anyhow::Result<f64> {
    let track = index
        .init
        .tracks()
        .get(&index.track_id)
        .context("missing fragmented video track")?;
    video::presentation_time(
        track,
        sample.start_time,
        sample.rendering_offset,
        index.init.timescale(),
    )
}

pub(crate) fn plan(index: &FragmentIndex, sparse: bool, cap: u64) -> anyhow::Result<Plan> {
    let track = index
        .init
        .tracks()
        .get(&index.track_id)
        .context("missing fragmented video track")?;
    video::validate_codec(track)?;
    let scale = track.timescale();
    ensure!(
        scale > 0 && index.init.timescale() > 0,
        "invalid fragmented MP4 timescale"
    );
    ensure!(
        !index.samples.is_empty(),
        "fragmented MP4 has no video samples"
    );
    let mut end = 0_u64;
    for sample in &index.samples {
        ensure!(
            sample.start_time == end && sample.duration > 0,
            "discontinuous fragmented decode timeline needs full download"
        );
        end = end
            .checked_add(u64::from(sample.duration))
            .context("fragmented timeline overflow")?;
    }
    let duration = if index.init.moov.mvhd.duration > 0 {
        index.init.moov.mvhd.duration as f64 / f64::from(index.init.timescale())
    } else {
        index
            .samples
            .iter()
            .try_fold(0.0_f64, |duration, sample| -> anyhow::Result<f64> {
                Ok(duration.max(
                    source_time(index, sample)? + f64::from(sample.duration) / f64::from(scale),
                ))
            })?
    };
    ensure!(
        duration > 0.0 && duration.is_finite(),
        "invalid fragmented video duration"
    );
    let mut samples = Vec::new();
    let step = (duration / 1200.0).max(0.5);
    let mut next = 0.0;
    for (position, sample) in index.samples.iter().enumerate() {
        if sparse {
            let time = sample.start_time as f64 / f64::from(scale);
            if !sample.is_sync || time + 1e-8 < next || samples.len() == 1200 {
                continue;
            }
            next = time + step;
        }
        ensure!(
            u64::from(sample.size) <= cap,
            "fragmented sample exceeds video transfer cap"
        );
        samples.push(u32::try_from(position + 1)?);
    }
    ensure!(
        !samples.is_empty(),
        "fragmented video has no independent sync samples"
    );
    let source_fps = index.samples.len() as f64 * f64::from(scale) / end as f64;
    Ok(Plan {
        track_id: index.track_id,
        duration_ticks: end,
        duration_seconds: duration,
        source_fps,
        samples,
    })
}

pub(crate) fn read_sample<R: Read + Seek>(
    source: &mut R,
    sample: &Sample,
    cap: u64,
) -> anyhow::Result<mp4::Mp4Sample> {
    ensure!(
        u64::from(sample.size) <= cap,
        "fragmented sample exceeds video byte cap"
    );
    let length = usize::try_from(sample.size)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length)?;
    bytes.resize(length, 0);
    source.seek(SeekFrom::Start(sample.offset))?;
    source.read_exact(&mut bytes)?;
    Ok(mp4::Mp4Sample {
        start_time: sample.start_time,
        duration: sample.duration,
        rendering_offset: sample.rendering_offset,
        is_sync: sample.is_sync,
        bytes: bytes.into(),
    })
}

pub(crate) fn proxy<R: Read + Seek>(
    session: &Session<'_>,
    source: &mut R,
    index: &FragmentIndex,
    path: &Path,
) -> anyhow::Result<()> {
    let sparse = session.options.video_mode == VideoMode::Keyframes;
    let plan = plan(index, sparse, session.options.max_video_bytes)?;
    let scale = index
        .init
        .tracks()
        .get(&index.track_id)
        .context("missing fragmented video track")?
        .timescale();
    let mut writer = ProxyWriter::new(&index.init, &plan, path)?;
    writer.set_duration(plan.duration_ticks, scale)?;
    let mut points = Vec::with_capacity(if sparse { plan.samples.len() } else { 0 });
    for (position, &id) in plan.samples.iter().enumerate() {
        session.check_deadline()?;
        let item = &index.samples[(id - 1) as usize];
        let duration = if sparse {
            let end = plan
                .samples
                .get(position + 1)
                .map_or(plan.duration_ticks, |next| {
                    index.samples[(*next - 1) as usize].start_time
                });
            u32::try_from(
                end.checked_sub(item.start_time)
                    .context("unordered fragmented samples")?,
            )?
        } else {
            item.duration
        };
        let sample = read_sample(source, item, session.options.max_video_bytes)?;
        writer.push(&sample, duration)?;
        if sparse {
            points.push(source_time(index, item)?);
        }
    }
    writer.finish(path)?;
    video::retain_timeline(path, &plan, sparse.then_some(points.as_slice()))
}

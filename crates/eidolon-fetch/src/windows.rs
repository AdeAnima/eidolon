use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, ensure};
use eidolon_core::RefinedWindow;
use mp4::Mp4Reader;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::http::{Headers, RangeSource, Session, inspect_boxes};
use crate::video::{Plan, ProxyWriter, decode_time, plan, presentation_time};
use crate::{digest, save};

#[derive(Deserialize, Serialize)]
struct Clock {
    source_offset_s: f64,
}

fn gop_bounds(
    points: impl Iterator<Item = anyhow::Result<(u32, f64)>>,
    start: f64,
    end: f64,
    count: u32,
) -> anyhow::Result<(u32, u32)> {
    let mut first = None;
    let mut previous = None;
    for point in points {
        let (id, time) = point?;
        ensure!(
            id > 0 && id <= count && time.is_finite(),
            "invalid sync sample"
        );
        if let Some((previous_id, previous_time)) = previous {
            ensure!(
                id > previous_id && time >= previous_time,
                "unordered sync samples"
            );
        }
        previous = Some((id, time));
        let first = first.get_or_insert(id);
        if time <= start + 1e-8 {
            *first = id;
        }
        if time > end + 1e-8 && id > *first {
            return Ok((*first, id));
        }
    }
    Ok((
        first.context("no independent samples for refinement")?,
        count.checked_add(1).context("sample count overflow")?,
    ))
}

// Presentation order can differ from decode order. Keep every dependency before
// the final requested presentation sample, including forward-reference pictures.
fn prefix_end(
    points: impl Iterator<Item = anyhow::Result<(u32, f64)>>,
    target: f64,
    first: u32,
) -> anyhow::Result<u32> {
    let mut last = first;
    for point in points {
        let (id, time) = point?;
        ensure!(time.is_finite(), "invalid native presentation time");
        if time <= target + 1e-8 {
            last = last.max(id);
        }
    }
    last.checked_add(1).context("native sample count overflow")
}

fn rendering_offset(track: &mp4::Mp4Track, id: u32) -> anyhow::Result<i32> {
    let Some(ctts) = &track.trak.mdia.minf.stbl.ctts else {
        return Ok(0);
    };
    let mut remaining = id - 1;
    for entry in &ctts.entries {
        if remaining < entry.sample_count {
            return Ok(entry.sample_offset);
        }
        remaining -= entry.sample_count;
    }
    anyhow::bail!("missing composition offset for sample")
}

fn window_path(proxy: &Path, headers: &Headers, first: u32, end: u32) -> anyhow::Result<PathBuf> {
    let revision = digest(&format!(
        "{}:{:?}:{:?}",
        headers.url, headers.etag, headers.last_modified
    ));
    let name = proxy
        .file_stem()
        .and_then(|name| name.to_str())
        .context("invalid proxy filename")?;
    Ok(proxy.with_file_name(format!(
        "{name}.window-{}-{first}-{end}.mp4",
        &revision[..16]
    )))
}

fn cached_window(path: &Path) -> Option<RefinedWindow> {
    if !path.is_file() {
        return None;
    }
    let clock: Clock =
        serde_json::from_slice(&std::fs::read(path.with_extension("json")).ok()?).ok()?;
    clock.source_offset_s.is_finite().then(|| RefinedWindow {
        path: path.to_owned(),
        source_offset_s: clock.source_offset_s,
    })
}

fn retain_clock(window: &RefinedWindow) -> anyhow::Result<()> {
    save(
        &window.path.with_extension("json"),
        &serde_json::to_vec(&Clock {
            source_offset_s: window.source_offset_s,
        })?,
    )
}

struct NativeClock {
    first_dts: u64,
    first_cts: i32,
    source_offset_s: f64,
    scale: u32,
}

impl NativeClock {
    fn push(
        &self,
        writer: &mut ProxyWriter,
        mut sample: mp4::Mp4Sample,
        source_time: f64,
    ) -> anyhow::Result<()> {
        let local_ticks = i128::from(sample.start_time) - i128::from(self.first_dts)
            + i128::from(sample.rendering_offset)
            - i128::from(self.first_cts);
        let expected = self.source_offset_s + local_ticks as f64 / f64::from(self.scale);
        ensure!(
            (expected - source_time).abs() < 0.000_001,
            "discontinuous source edits cannot form a native-cadence window"
        );
        sample.rendering_offset = sample
            .rendering_offset
            .checked_sub(self.first_cts)
            .context("native composition offset overflow")?;
        writer.push(&sample, sample.duration)
    }
}

fn copy_regular<R: Read + Seek>(
    session: &Session<'_>,
    reader: &mut Mp4Reader<R>,
    plan: &Plan,
    writer: &mut ProxyWriter,
    clock: &NativeClock,
) -> anyhow::Result<()> {
    for &id in &plan.samples {
        session.check_deadline()?;
        let track = reader
            .tracks()
            .get(&plan.track_id)
            .context("missing source track")?;
        let table = &track.trak.mdia.minf.stbl.stsz;
        let size = if table.sample_size > 0 {
            table.sample_size
        } else {
            *table
                .sample_sizes
                .get((id - 1) as usize)
                .context("missing native sample size")?
        };
        ensure!(
            u64::from(size) <= session.options.max_video_bytes,
            "native sample exceeds video byte cap"
        );
        let sample = reader
            .read_sample(plan.track_id, id)?
            .context("missing native sample")?;
        let track = reader
            .tracks()
            .get(&plan.track_id)
            .context("missing source track")?;
        let source_time = presentation_time(
            track,
            sample.start_time,
            sample.rendering_offset,
            reader.timescale(),
        )?;
        clock.push(writer, sample, source_time)?;
    }
    Ok(())
}

fn fragmented<R: Read + Seek>(
    session: &Session<'_>,
    source: &mut R,
    index: &crate::fragment::FragmentIndex,
    proxy: &Path,
    headers: &Headers,
    center: f64,
    radius: f64,
) -> anyhow::Result<RefinedWindow> {
    let mut plan = crate::fragment_video::plan(index, true, session.options.max_video_bytes)?;
    let start = (center - radius).clamp(0.0, plan.duration_seconds);
    let end = (center + radius).clamp(0.0, plan.duration_seconds);
    let points = index
        .samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.is_sync)
        .map(|(position, sample)| {
            session.check_deadline()?;
            Ok((
                u32::try_from(position + 1)?,
                crate::fragment_video::source_time(index, sample)?,
            ))
        });
    let (first, gop_end) = gop_bounds(points, start, end, u32::try_from(index.samples.len())?)?;
    let end = prefix_end(
        (first..gop_end).map(|id| {
            session.check_deadline()?;
            Ok((
                id,
                crate::fragment_video::source_time(index, &index.samples[(id - 1) as usize])?,
            ))
        }),
        end,
        first,
    )?;
    let path = window_path(proxy, headers, first, end)?;
    if let Some(window) = cached_window(&path) {
        return Ok(window);
    }
    plan.samples.clear();
    plan.samples.extend(first..end);
    let item = &index.samples[(first - 1) as usize];
    let track = index
        .init
        .tracks()
        .get(&index.track_id)
        .context("missing fragmented video track")?;
    let offset = crate::fragment_video::source_time(index, item)?;
    let clock = NativeClock {
        first_dts: item.start_time,
        first_cts: item.rendering_offset,
        source_offset_s: offset,
        scale: track.timescale(),
    };
    let end_dts = index
        .samples
        .get((end - 1) as usize)
        .map_or(plan.duration_ticks, |sample| sample.start_time);
    let mut writer = ProxyWriter::new(&index.init, &plan, &path)?;
    writer.normalize_window(
        end_dts
            .checked_sub(item.start_time)
            .context("invalid fragmented window duration")?,
        track.timescale(),
    )?;
    for item in &index.samples[(first - 1) as usize..(end - 1) as usize] {
        session.check_deadline()?;
        let sample =
            crate::fragment_video::read_sample(source, item, session.options.max_video_bytes)?;
        clock.push(
            &mut writer,
            sample,
            crate::fragment_video::source_time(index, item)?,
        )?;
    }
    writer.finish(&path)?;
    let window = RefinedWindow {
        path,
        source_offset_s: offset,
    };
    retain_clock(&window)?;
    Ok(window)
}

pub(crate) fn materialize(
    session: &Session<'_>,
    url: &Url,
    headers: &Headers,
    proxy: &Path,
    center: f64,
    radius: f64,
    budget: u64,
) -> anyhow::Result<RefinedWindow> {
    let size = headers
        .total_size
        .context("indexed video needs Content-Length")?;
    let mut source = RangeSource::new(session, url, headers, budget)?;
    if let Some(index) = crate::fragment::index(&mut source, size)? {
        return fragmented(session, &mut source, &index, proxy, headers, center, radius);
    }
    source.rewind()?;
    inspect_boxes(&mut source, size)?;
    let mut reader = Mp4Reader::read_header(source, size).context("reading native video index")?;
    let mut plan = plan(&reader, true, session.options.max_video_bytes)?;
    let track = reader
        .tracks()
        .get(&plan.track_id)
        .context("missing source track")?;
    let start = (center - radius).clamp(0.0, plan.duration_seconds);
    let end = (center + radius).clamp(0.0, plan.duration_seconds);
    let mut point = |id| {
        session.check_deadline()?;
        Ok((
            id,
            presentation_time(
                track,
                decode_time(track, id)?,
                rendering_offset(track, id)?,
                reader.timescale(),
            )?,
        ))
    };
    let (first, gop_end) = if let Some(sync) = &track.trak.mdia.minf.stbl.stss {
        gop_bounds(
            sync.entries.iter().copied().map(&mut point),
            start,
            end,
            track.sample_count(),
        )?
    } else {
        gop_bounds(
            (1..=track.sample_count()).map(point),
            start,
            end,
            track.sample_count(),
        )?
    };
    let end = prefix_end((first..gop_end).map(point), end, first)?;
    let path = window_path(proxy, headers, first, end)?;
    if let Some(window) = cached_window(&path) {
        return Ok(window);
    }
    plan.samples.clear();
    plan.samples.reserve((end - first) as usize);
    plan.samples.extend(first..end);
    let decode_origin = decode_time(track, first)?;
    let composition_shift = rendering_offset(track, first)?;
    let offset = presentation_time(track, decode_origin, composition_shift, reader.timescale())?;
    let end_dts = if end > track.sample_count() {
        track.trak.mdia.mdhd.duration
    } else {
        decode_time(track, end)?
    };
    let scale = track.timescale();
    let mut writer = ProxyWriter::new(&reader, &plan, &path)?;
    writer.normalize_window(
        end_dts
            .checked_sub(decode_origin)
            .context("invalid native window duration")?,
        scale,
    )?;
    let clock = NativeClock {
        first_dts: decode_origin,
        first_cts: composition_shift,
        source_offset_s: offset,
        scale,
    };
    copy_regular(session, &mut reader, &plan, &mut writer, &clock)?;
    writer.finish(&path)?;
    let window = RefinedWindow {
        path,
        source_offset_s: offset,
    };
    retain_clock(&window)?;
    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::gop_bounds;

    #[test]
    fn native_windows_keep_preceding_gop_and_include_the_right_endpoint() {
        let points = || {
            [(1, 0.0), (97, 4.0), (193, 8.0), (289, 12.0)]
                .into_iter()
                .map(Ok)
        };
        assert_eq!(
            gop_bounds(points(), 4.5, 5.5, 384).expect("interior window"),
            (97, 193)
        );
        assert_eq!(
            gop_bounds(points(), 4.5, 8.0, 384).expect("closed right endpoint"),
            (97, 289)
        );
        assert_eq!(
            gop_bounds(points(), 0.0, 0.0, 384).expect("source beginning"),
            (1, 97)
        );
        assert_eq!(
            gop_bounds(points(), 15.0, 16.0, 384).expect("source end"),
            (289, 385)
        );
    }
}

//! Bounded Matroska metadata, Cues keyframes and native dependency prefixes.
//!
//! Element IDs and segment-relative cue offsets follow the RFC 9559 schema:
//! <https://www.rfc-editor.org/rfc/rfc9559>. Packet payloads are copied, never decoded here.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context as _, ensure};
use eidolon_core::RefinedWindow;
use url::Url;

use crate::http::{Headers, RangeSource, Session};
use crate::video::{Plan, retain_timeline};
use crate::{VideoMode, digest};

const SEGMENT: u64 = 0x1853_8067;
const INFO: u64 = 0x1549_a966;
const TRACKS: u64 = 0x1654_ae6b;
const CUES: u64 = 0x1c53_bb6b;
const CLUSTER: u64 = 0x1f43_b675;
const METADATA_CAP: u64 = 16 * 1024 * 1024;
const ELEMENT_CAP: usize = 100_000;

#[derive(Clone, Copy)]
struct Element {
    id: u64,
    start: u64,
    data: u64,
    end: u64,
}

fn vint(source: &mut impl Read, identifier: bool) -> anyhow::Result<(u64, usize)> {
    let mut first = [0_u8];
    source.read_exact(&mut first)?;
    let length = first[0].leading_zeros() as usize + 1;
    ensure!(
        length <= if identifier { 4 } else { 8 },
        "invalid EBML integer"
    );
    let mut value = if identifier {
        u64::from(first[0])
    } else {
        u64::from(first[0]) & (0xff_u64 >> length)
    };
    for _ in 1..length {
        source.read_exact(&mut first)?;
        value = (value << 8) | u64::from(first[0]);
    }
    Ok((value, length))
}

fn element(source: &mut (impl Read + Seek), bound: u64) -> anyhow::Result<Element> {
    let start = source.stream_position()?;
    ensure!(start < bound, "EBML element outside parent");
    let (id, _) = vint(source, true)?;
    let (size, length) = vint(source, false)?;
    let data = source.stream_position()?;
    ensure!(data <= bound, "truncated EBML header");
    let unknown = size == (1_u64 << (7 * length)) - 1;
    ensure!(
        !unknown || id == SEGMENT,
        "unknown-sized Matroska element needs complete download"
    );
    let end = if unknown {
        bound
    } else {
        data.checked_add(size).context("EBML size overflow")?
    };
    ensure!(end <= bound, "EBML element exceeds parent");
    Ok(Element {
        id,
        start,
        data,
        end,
    })
}

fn children(source: &mut (impl Read + Seek), parent: Element) -> anyhow::Result<Vec<Element>> {
    let mut result = Vec::new();
    let mut position = parent.data;
    while position < parent.end {
        ensure!(
            result.len() < ELEMENT_CAP,
            "too many Matroska metadata elements"
        );
        source.seek(SeekFrom::Start(position))?;
        let item = element(source, parent.end)?;
        position = item.end;
        result.push(item);
    }
    Ok(result)
}

fn bytes(source: &mut (impl Read + Seek), start: u64, end: u64) -> anyhow::Result<Vec<u8>> {
    let size = end.checked_sub(start).context("invalid EBML byte range")?;
    ensure!(
        size <= METADATA_CAP,
        "Matroska metadata/block exceeds 16 MiB"
    );
    let mut result = vec![0; usize::try_from(size)?];
    source.seek(SeekFrom::Start(start))?;
    source.read_exact(&mut result)?;
    Ok(result)
}

fn uint(source: &mut (impl Read + Seek), item: Element) -> anyhow::Result<u64> {
    let length = usize::try_from(item.end - item.data)?;
    ensure!(length <= 8, "oversized EBML unsigned integer");
    let mut data = [0_u8; 8];
    source.seek(SeekFrom::Start(item.data))?;
    source.read_exact(&mut data[8 - length..])?;
    Ok(u64::from_be_bytes(data))
}

fn float(source: &mut (impl Read + Seek), item: Element) -> anyhow::Result<f64> {
    source.seek(SeekFrom::Start(item.data))?;
    match item.end - item.data {
        4 => {
            let mut data = [0_u8; 4];
            source.read_exact(&mut data)?;
            Ok(f64::from(f32::from_be_bytes(data)))
        }
        8 => {
            let mut data = [0_u8; 8];
            source.read_exact(&mut data)?;
            Ok(f64::from_be_bytes(data))
        }
        _ => anyhow::bail!("invalid EBML float"),
    }
}

#[derive(Clone, Copy)]
struct Cue {
    time: u64,
    cluster: u64,
    relative: Option<u64>,
    independent: bool,
}

struct Index {
    ebml: Vec<u8>,
    track: Vec<u8>,
    track_number: u64,
    scale: u64,
    duration: f64,
    default_ns: u64,
    fps: f64,
    reordered: bool,
    cues: Vec<Cue>,
    end: u64,
}

fn track(
    source: &mut (impl Read + Seek),
    root: Element,
) -> anyhow::Result<(Vec<u8>, u64, u64, f64, bool)> {
    for entry in children(source, root)?
        .into_iter()
        .filter(|entry| entry.id == 0xae)
    {
        let fields = children(source, entry)?;
        let mut number = 0;
        let mut kind = 0;
        let mut default_ns = 0;
        let mut fps = 0.0;
        let mut codec = Vec::new();
        for field in &fields {
            match field.id {
                0xd7 => number = uint(source, *field)?,
                0x83 => kind = uint(source, *field)?,
                0x86 => codec = bytes(source, field.data, field.end)?,
                0x0023_e383 => default_ns = uint(source, *field)?,
                0xe0 => {
                    for video in children(source, *field)? {
                        if video.id == 0x0023_83e3 {
                            fps = float(source, video)?;
                        }
                    }
                }
                _ => (),
            }
        }
        if kind != 1 {
            continue;
        }
        ensure!(number > 0, "invalid Matroska video track number");
        ensure!(
            matches!(codec.as_slice(), b"V_VP8" | b"V_VP9" | b"V_MPEG4/ISO/AVC"),
            "unsupported Matroska video codec"
        );
        for field in fields {
            match field.id {
                0x6d80 => anyhow::bail!("encoded/encrypted Matroska tracks need complete download"),
                0x0023_314f => ensure!(
                    (float(source, field)? - 1.0).abs() < 1e-12,
                    "scaled Matroska track clock unsupported"
                ),
                0x56aa | 0x537f => ensure!(
                    uint(source, field)? == 0,
                    "shifted Matroska track clock unsupported"
                ),
                _ => (),
            }
        }
        return Ok((
            bytes(source, entry.start, entry.end)?,
            number,
            default_ns,
            fps,
            codec == b"V_MPEG4/ISO/AVC",
        ));
    }
    anyhow::bail!("Matroska has no video track")
}

fn index_elements(
    source: &mut (impl Read + Seek),
    segment: Element,
) -> anyhow::Result<(Element, Element, Element)> {
    let mut pending = vec![segment.data];
    let mut info = None;
    let mut tracks = None;
    let mut cues = None;
    let mut inspected = std::collections::HashSet::new();
    let mut metadata_bytes = 0;
    while let Some(mut position) = pending.pop() {
        while position < segment.end {
            ensure!(
                inspected.len() < ELEMENT_CAP,
                "too many Matroska index elements"
            );
            if !inspected.insert(position) {
                break;
            }
            source.seek(SeekFrom::Start(position))?;
            let item = element(source, segment.end)?;
            if matches!(item.id, INFO | TRACKS | CUES | 0x114d_9b74) {
                metadata_bytes += item.end - item.start;
                ensure!(
                    metadata_bytes <= METADATA_CAP,
                    "Matroska index exceeds 16 MiB"
                );
            }
            match item.id {
                INFO => info = Some(item),
                TRACKS => tracks = Some(item),
                CUES => cues = Some(item),
                0x114d_9b74 => {
                    for seek in children(source, item)?
                        .into_iter()
                        .filter(|seek| seek.id == 0x4dbb)
                    {
                        for field in children(source, seek)? {
                            if field.id == 0x53ac {
                                let offset = segment
                                    .data
                                    .checked_add(uint(source, field)?)
                                    .context("Matroska seek overflow")?;
                                ensure!(offset < segment.end, "Matroska seek outside Segment");
                                pending.push(offset);
                            }
                        }
                    }
                }
                _ => (),
            }
            if info.is_some() && tracks.is_some() && cues.is_some() {
                break;
            }
            // Prefer SeekHead destinations to walking all Cluster headers.
            if !pending.is_empty() {
                pending.insert(0, item.end);
                break;
            }
            position = item.end;
        }
        if info.is_some() && tracks.is_some() && cues.is_some() {
            break;
        }
    }
    Ok((
        info.context("Matroska has no Info")?,
        tracks.context("Matroska has no Tracks")?,
        cues.context("Matroska has no Cues index")?,
    ))
}

fn index(source: &mut (impl Read + Seek), size: u64) -> anyhow::Result<Index> {
    source.rewind()?;
    let ebml = element(source, size)?;
    ensure!(ebml.id == 0x1a45_dfa3, "not an EBML container");
    let header = bytes(source, ebml.start, ebml.end)?;
    source.seek(SeekFrom::Start(ebml.end))?;
    let segment = element(source, size)?;
    ensure!(segment.id == SEGMENT, "missing Matroska Segment");
    let (info, tracks, cues) = index_elements(source, segment)?;
    let mut scale = 1_000_000;
    let mut duration_ticks = None;
    for field in children(source, info)? {
        match field.id {
            0x002a_d7b1 => scale = uint(source, field)?,
            0x4489 => duration_ticks = Some(float(source, field)?),
            _ => (),
        }
    }
    ensure!(scale > 0, "invalid Matroska timestamp scale");
    let duration = duration_ticks.context("Matroska has no duration")? * scale as f64 / 1e9;
    ensure!(
        duration.is_finite() && duration > 0.0,
        "invalid Matroska duration"
    );
    let (track, track_number, default_ns, fps, reordered) = track(source, tracks)?;
    let points = cue_points(source, cues, segment, track_number)?;
    let mut result = Index {
        ebml: header,
        track,
        track_number,
        scale,
        duration,
        default_ns,
        fps,
        reordered,
        cues: points,
        end: segment.end,
    };
    if result.default_ns > 0 {
        result.fps = 1e9 / result.default_ns as f64;
    }
    if !result.fps.is_finite() || result.fps <= 0.0 {
        let (_, blocks, _) = cluster_blocks(source, &result, result.cues[0], Some(32), None)?;
        let mut differences: Vec<_> = blocks
            .windows(2)
            .map(|pair| pair[1].time - pair[0].time)
            .filter(|delta| *delta > 0.0)
            .collect();
        differences.sort_unstable_by(f64::total_cmp);
        let period = differences
            .get(differences.len() / 2)
            .context("Matroska has no usable frame cadence")?;
        result.fps = 1.0 / period;
    }
    ensure!(
        result.fps.is_finite() && result.fps > 0.0,
        "invalid Matroska frame cadence"
    );
    Ok(result)
}

fn cue_points(
    source: &mut (impl Read + Seek),
    cues: Element,
    segment: Element,
    track_number: u64,
) -> anyhow::Result<Vec<Cue>> {
    let mut points = Vec::new();
    for point in children(source, cues)?
        .into_iter()
        .filter(|point| point.id == 0xbb)
    {
        let mut time = None;
        let mut positions = Vec::new();
        for field in children(source, point)? {
            if field.id == 0xb3 {
                time = Some(uint(source, field)?);
            }
            if field.id == 0xb7 {
                positions.push(field);
            }
        }
        for position in positions {
            let mut number = 0;
            let mut cluster = None;
            let mut relative = None;
            for field in children(source, position)? {
                match field.id {
                    0xf7 => number = uint(source, field)?,
                    0xf1 => cluster = Some(uint(source, field)?),
                    0xf0 => relative = Some(uint(source, field)?),
                    _ => (),
                }
            }
            if number != track_number {
                continue;
            }
            let cluster = segment
                .data
                .checked_add(cluster.context("cue has no cluster position")?)
                .context("cue offset overflow")?;
            ensure!(cluster < segment.end, "cue cluster outside Segment");
            let time = time.context("cue has no timestamp")?;
            ensure!(
                points
                    .last()
                    .is_none_or(|previous: &Cue| previous.time < time),
                "unordered Matroska cues"
            );
            points.push(Cue {
                time,
                cluster,
                relative,
                independent: true,
            });
        }
    }
    ensure!(!points.is_empty(), "Matroska has no video cues");
    Ok(points)
}

struct Block {
    item: Element,
    payload: Element,
    time: f64,
    ticks: u64,
    keyframe: bool,
    frames: usize,
}

fn block(
    source: &mut (impl Read + Seek),
    index: &Index,
    item: Element,
    cluster_time: u64,
) -> anyhow::Result<Option<Block>> {
    let mut payload = item;
    let mut keyframe = false;
    if item.id == 0xa0 {
        let fields = children(source, item)?;
        payload = *fields
            .iter()
            .find(|field| field.id == 0xa1)
            .context("BlockGroup has no Block")?;
        keyframe = !fields.iter().any(|field| field.id == 0xfb);
        ensure!(
            !fields.iter().any(|field| field.id == 0xa4),
            "Matroska CodecState changes need complete download"
        );
    }
    source.seek(SeekFrom::Start(payload.data))?;
    let (track, _) = vint(source, false)?;
    if track != index.track_number {
        return Ok(None);
    }
    let mut header = [0_u8; 3];
    source.read_exact(&mut header)?;
    ensure!(
        source.stream_position()? <= payload.end,
        "truncated Matroska block"
    );
    let timestamp =
        i128::from(cluster_time) + i128::from(i16::from_be_bytes([header[0], header[1]]));
    let ticks = u64::try_from(timestamp).context("negative Matroska video timestamp")?;
    let lace = header[2] & 0x06;
    let frames = if lace == 0 {
        1
    } else {
        ensure!(index.default_ns > 0, "laced video requires DefaultDuration");
        let mut count = [0];
        source.read_exact(&mut count)?;
        ensure!(
            source.stream_position()? <= payload.end,
            "truncated Matroska lace"
        );
        usize::from(count[0]) + 1
    };
    Ok(Some(Block {
        item,
        payload,
        time: ticks as f64 * index.scale as f64 / 1e9,
        ticks,
        keyframe: keyframe || (item.id == 0xa3 && header[2] & 0x80 != 0),
        frames,
    }))
}

fn cluster_blocks(
    source: &mut (impl Read + Seek),
    index: &Index,
    cue: Cue,
    maximum: Option<usize>,
    end_time: Option<f64>,
) -> anyhow::Result<(Element, Vec<Block>, bool)> {
    source.seek(SeekFrom::Start(cue.cluster))?;
    let cluster = element(source, index.end)?;
    ensure!(cluster.id == CLUSTER, "cue does not reference a Cluster");
    let mut position = cluster.data;
    let mut cluster_time = None;
    let mut blocks = Vec::new();
    for _ in 0..ELEMENT_CAP {
        if position >= cluster.end {
            return Ok((cluster, blocks, false));
        }
        source.seek(SeekFrom::Start(position))?;
        let item = element(source, cluster.end)?;
        position = item.end;
        if item.id == 0xe7 {
            cluster_time = Some(uint(source, item)?);
            if let Some(relative) = cue.relative {
                position = cluster
                    .data
                    .checked_add(relative)
                    .context("relative cue overflow")?;
                ensure!(
                    position >= item.end && position < cluster.end,
                    "relative cue outside Cluster"
                );
            }
        } else if matches!(item.id, 0xa3 | 0xa0)
            && let Some(block) = block(
                source,
                index,
                item,
                cluster_time.context("Cluster has no Timestamp")?,
            )?
        {
            if block.ticks < cue.time {
                continue;
            }
            if blocks.is_empty() && cue.independent {
                ensure!(
                    block.keyframe && block.ticks == cue.time,
                    "cue does not identify an independent video block"
                );
            }
            if end_time.is_some_and(|end| block.time > end + 1e-8)
                && (!index.reordered || block.keyframe)
            {
                return Ok((cluster, blocks, true));
            }
            blocks.push(block);
            if maximum.is_some_and(|maximum| blocks.len() >= maximum) {
                return Ok((cluster, blocks, false));
            }
        }
    }
    anyhow::bail!("too many Matroska Cluster elements")
}

fn encode_size(output: &mut impl Write, value: u64) -> anyhow::Result<()> {
    for length in 1..=8 {
        if value < (1_u64 << (7 * length)) - 1 {
            let encoded = value | (1_u64 << (7 * length));
            output.write_all(&encoded.to_be_bytes()[8 - length..])?;
            return Ok(());
        }
    }
    anyhow::bail!("Matroska element too large")
}

fn put(output: &mut impl Write, id: u64, value: &[u8]) -> anyhow::Result<()> {
    let width = ((64 - id.leading_zeros()) as usize).div_ceil(8);
    output.write_all(&id.to_be_bytes()[8 - width..])?;
    encode_size(output, value.len() as u64)?;
    output.write_all(value)?;
    Ok(())
}

fn put_uint(output: &mut impl Write, id: u64, value: u64) -> anyhow::Result<()> {
    let width = ((64 - value.leading_zeros()) as usize).div_ceil(8).max(1);
    put(output, id, &value.to_be_bytes()[8 - width..])
}

fn copy_bytes(
    source: &mut (impl Read + Seek),
    output: &mut impl Write,
    start: u64,
    end: u64,
) -> anyhow::Result<()> {
    let length = end
        .checked_sub(start)
        .context("invalid Matroska copy range")?;
    ensure!(length <= METADATA_CAP, "Matroska block exceeds 16 MiB");
    source.seek(SeekFrom::Start(start))?;
    let copied = std::io::copy(&mut source.take(length), output)?;
    ensure!(copied == length, "truncated Matroska block payload");
    Ok(())
}

fn copy_block(
    source: &mut (impl Read + Seek),
    output: &mut impl Write,
    block: &Block,
) -> anyhow::Result<()> {
    source.seek(SeekFrom::Start(block.payload.data))?;
    let (_, track_length) = vint(source, false)?;
    let timestamp = block.payload.data + track_length as u64;
    copy_bytes(source, output, block.payload.start, timestamp)?;
    output.write_all(&[0, 0])?;
    copy_bytes(source, output, timestamp + 2, block.payload.end)
}

struct Writer {
    output: tempfile::NamedTempFile,
}

impl Writer {
    fn new(index: &Index, path: &Path, duration: f64) -> anyhow::Result<Self> {
        let mut output =
            tempfile::NamedTempFile::new_in(path.parent().context("proxy path has no parent")?)?;
        output.write_all(&index.ebml)?;
        output.write_all(&SEGMENT.to_be_bytes()[4..])?;
        output.write_all(&[0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff])?;
        let mut info = Vec::new();
        put_uint(&mut info, 0x002a_d7b1, index.scale)?;
        put(
            &mut info,
            0x4489,
            &(duration * 1e9 / index.scale as f64).to_be_bytes(),
        )?;
        put(&mut info, 0x4d80, b"eidolon")?;
        put(&mut info, 0x5741, b"eidolon")?;
        put(&mut output, INFO, &info)?;
        put(&mut output, TRACKS, &index.track)?;
        Ok(Self { output })
    }

    fn push(
        &mut self,
        source: &mut (impl Read + Seek),
        block: &Block,
        origin: u64,
    ) -> anyhow::Result<()> {
        self.output.write_all(&CLUSTER.to_be_bytes()[4..])?;
        let size_position = self.output.stream_position()?;
        self.output.write_all(&[0x01, 0, 0, 0, 0, 0, 0, 0])?;
        let data_position = self.output.stream_position()?;
        put_uint(
            &mut self.output,
            0xe7,
            block
                .ticks
                .checked_sub(origin)
                .context("native timestamp before origin")?,
        )?;
        if block.item.id == 0xa0 {
            let fields = children(source, block.item)?;
            let length: u64 = fields
                .iter()
                .filter(|field| field.id != 0xbf)
                .map(|field| field.end - field.start)
                .sum();
            self.output.write_all(&[0xa0])?;
            encode_size(&mut self.output, length)?;
            for field in fields {
                if field.id == 0xa1 {
                    copy_block(source, &mut self.output, block)?;
                } else if field.id != 0xbf {
                    copy_bytes(source, &mut self.output, field.start, field.end)?;
                }
            }
        } else {
            copy_block(source, &mut self.output, block)?;
        }
        let end = self.output.stream_position()?;
        let length = end - data_position;
        ensure!(length < (1_u64 << 56) - 1, "Matroska Cluster size overflow");
        self.output.seek(SeekFrom::Start(size_position))?;
        self.output
            .write_all(&((1_u64 << 56) | length).to_be_bytes())?;
        self.output.seek(SeekFrom::Start(end))?;
        Ok(())
    }

    fn finish(self, path: &Path) -> anyhow::Result<()> {
        self.output.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

pub(crate) fn proxy(
    session: &Session<'_>,
    url: &Url,
    headers: &Headers,
    path: &Path,
    budget: u64,
) -> anyhow::Result<()> {
    ensure!(
        session.options.video_mode == VideoMode::Keyframes,
        "Matroska full cadence uses complete download"
    );
    let size = headers
        .total_size
        .context("Matroska index needs Content-Length")?;
    let mut source = RangeSource::new(session, url, headers, budget)?;
    let index = index(&mut source, size)?;
    let mut writer = Writer::new(&index, path, index.duration)?;
    let mut points = Vec::new();
    let spacing = (index.duration / 1200.0).max(0.5);
    let mut next = 0.0;
    for cue in &index.cues {
        session.check_deadline()?;
        let time = cue.time as f64 * index.scale as f64 / 1e9;
        if time + 1e-8 < next {
            continue;
        }
        let (_, blocks, _) = cluster_blocks(&mut source, &index, *cue, Some(1), None)?;
        let block = blocks.first().context("cue has no video block")?;
        ensure!(
            points.len() + block.frames <= 1200,
            "too many laced keyframes for sparse proxy"
        );
        writer.push(&mut source, block, 0)?;
        for frame in 0..block.frames {
            points.push(block.time + frame as f64 * index.default_ns as f64 / 1e9);
        }
        next = time + spacing;
    }
    ensure!(
        !points.is_empty(),
        "Matroska has no independent video blocks"
    );
    writer.finish(path)?;
    let plan = Plan {
        track_id: 0,
        duration_ticks: 0,
        duration_seconds: index.duration,
        source_fps: index.fps,
        samples: Vec::new(),
    };
    retain_timeline(path, &plan, Some(&points))
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
        .context("Matroska index needs Content-Length")?;
    let mut source = RangeSource::new(session, url, headers, budget)?;
    let index = index(&mut source, size)?;
    let start = (center - radius).clamp(0.0, index.duration);
    let end = (center + radius).clamp(0.0, index.duration);
    let first = index
        .cues
        .partition_point(|cue| cue.time as f64 * index.scale as f64 / 1e9 <= start + 1e-8)
        .saturating_sub(1);
    let origin = index.cues[first].time;
    let revision = digest(&format!(
        "{}:{:?}:{:?}:{origin}:{end:.9}",
        headers.url, headers.etag, headers.last_modified
    ));
    let path = proxy.with_file_name(format!(
        "{}.window-{}.mkv",
        proxy
            .file_stem()
            .and_then(|name| name.to_str())
            .context("invalid proxy name")?,
        &revision[..24]
    ));
    let offset = origin as f64 * index.scale as f64 / 1e9;
    if path.is_file() {
        return Ok(RefinedWindow {
            path,
            source_offset_s: offset,
        });
    }
    let mut selected = Vec::new();
    let mut last_position = 0;
    'cues: for cue in &index.cues[first..] {
        session.check_deadline()?;
        if cue.time as f64 * index.scale as f64 / 1e9 > end + 1e-8 {
            break;
        }
        let mut cue = *cue;
        loop {
            let (cluster, blocks, stopped) =
                cluster_blocks(&mut source, &index, cue, None, Some(end))?;
            for block in blocks {
                if block.item.start <= last_position {
                    continue;
                }
                last_position = block.item.start;
                selected.push(block);
                ensure!(
                    selected.len() <= ELEMENT_CAP,
                    "too many native Matroska blocks"
                );
            }
            if stopped {
                break 'cues;
            }
            let next = cluster.end;
            if next >= index.end {
                break;
            }
            source.seek(SeekFrom::Start(next))?;
            let item = element(&mut source, index.end)?;
            if item.id != CLUSTER || index.cues.iter().any(|point| point.cluster == next) {
                break;
            }
            cue = Cue {
                cluster: next,
                time: origin,
                relative: None,
                independent: false,
            };
        }
    }
    // Reordered H.264 keeps forward-reference pictures only when a requested
    // presentation sample depends on their earlier position in decode order.
    let last = selected
        .iter()
        .rposition(|block| block.time <= end + 1e-8)
        .context("native Matroska prefix has no requested frames")?;
    selected.truncate(last + 1);
    let mut writer = Writer::new(
        &index,
        &path,
        (end - offset + 1.0 / index.fps).max(1.0 / index.fps),
    )?;
    for block in selected {
        writer.push(&mut source, &block, origin)?;
    }
    ensure!(
        last_position > 0,
        "native Matroska prefix contains no video blocks"
    );
    writer.finish(&path)?;
    Ok(RefinedWindow {
        path,
        source_offset_s: offset,
    })
}

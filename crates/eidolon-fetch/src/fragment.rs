//! Bounded fragment metadata indexing. Media payloads remain in the supplied reader.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::ops::Range;

use anyhow::{Context as _, ensure};
use mp4::{Mp4Reader, TrackType};

const MAX_METADATA: u64 = 16 * 1024 * 1024;
const MAX_BOXES: usize = 100_000;
const MAX_SAMPLES: usize = 1_000_000;
const MAX_TRACKS: usize = 128;
const NON_SYNC: u32 = 0x0001_0000;

/// One sample in native decode order, with absolute source offsets and track ticks.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Sample {
    pub(crate) offset: u64,
    pub(crate) size: u32,
    pub(crate) start_time: u64,
    pub(crate) duration: u32,
    pub(crate) rendering_offset: i32,
    pub(crate) is_sync: bool,
}

/// Initialization boxes and the lowest-ID video track's native fragment samples.
pub(crate) struct FragmentIndex {
    pub(crate) init: Mp4Reader<Cursor<Vec<u8>>>,
    pub(crate) track_id: u32,
    pub(crate) samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
struct Header {
    kind: [u8; 4],
    start: u64,
    payload: u64,
    end: u64,
}

struct Atom<'a> {
    kind: [u8; 4],
    payload: &'a [u8],
}

struct Fields<'a>(&'a [u8]);

impl<'a> Fields<'a> {
    fn take(&mut self, count: usize) -> anyhow::Result<&'a [u8]> {
        let bytes = self.0.get(..count).context("truncated MP4 metadata")?;
        self.0 = &self.0[count..];
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> anyhow::Result<[u8; N]> {
        Ok(self.take(N)?.try_into()?)
    }

    fn byte(&mut self) -> anyhow::Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> anyhow::Result<u16> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_be_bytes(self.array()?))
    }

    fn full(&mut self, max_version: u8, allowed_flags: u32) -> anyhow::Result<(u8, u32)> {
        let value = self.u32()?;
        let version = (value >> 24) as u8;
        let flags = value & 0x00ff_ffff;
        ensure!(version <= max_version, "unsupported MP4 box version");
        ensure!(flags & !allowed_flags == 0, "unsupported MP4 box flags");
        Ok((version, flags))
    }

    fn finish(self) -> anyhow::Result<()> {
        ensure!(self.0.is_empty(), "unexpected trailing MP4 metadata");
        Ok(())
    }
}

fn atom<'a>(input: &mut &'a [u8]) -> anyhow::Result<Atom<'a>> {
    let mut fields = Fields(input);
    let short_size = fields.u32()?;
    let kind = fields.array()?;
    let (size, header_size) = if short_size == 1 {
        (fields.u64()?, 16)
    } else {
        (u64::from(short_size), 8)
    };
    // Zero-size nested boxes can make the dependency's container loops stall.
    ensure!(size >= header_size, "invalid nested MP4 box size");
    let length = usize::try_from(size)?;
    let bytes = input
        .get(..length)
        .context("MP4 child exceeds parent box")?;
    let payload = &bytes[usize::try_from(header_size)?..];
    *input = &input[length..];
    Ok(Atom { kind, payload })
}

fn header<R: Read + Seek>(reader: &mut R, start: u64, size: u64) -> anyhow::Result<Header> {
    ensure!(
        size.checked_sub(start).is_some_and(|left| left >= 8),
        "truncated MP4 box header"
    );
    reader.seek(SeekFrom::Start(start))?;
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    let mut fields = Fields(&bytes);
    let short_size = fields.u32()?;
    let kind = fields.array()?;
    let (length, header_size) = match short_size {
        0 => (size - start, 8),
        1 => {
            ensure!(size - start >= 16, "truncated extended MP4 box header");
            reader.read_exact(&mut bytes)?;
            (u64::from_be_bytes(bytes), 16)
        }
        value => (u64::from(value), 8),
    };
    ensure!(length >= header_size, "invalid MP4 box size");
    let end = start
        .checked_add(length)
        .context("MP4 box offset overflow")?;
    ensure!(end <= size, "MP4 box exceeds source size");
    Ok(Header {
        kind,
        start,
        payload: start + header_size,
        end,
    })
}

fn metadata<R: Read + Seek>(
    reader: &mut R,
    location: Header,
    remaining: &mut u64,
) -> anyhow::Result<Vec<u8>> {
    let size = location.end - location.start;
    *remaining = remaining
        .checked_sub(size)
        .context("MP4 metadata exceeds 16 MiB")?;
    let length = usize::try_from(size)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length)?;
    bytes.resize(length, 0);
    reader.seek(SeekFrom::Start(location.start))?;
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

struct Reference {
    range: Range<u64>,
    nested: bool,
}

fn sidx(bytes: &[u8], location: Header, size: u64) -> anyhow::Result<Vec<Reference>> {
    let mut fields = Fields(bytes);
    let (version, _) = fields.full(1, 0)?;
    fields.u32()?; // reference_ID is not necessarily the selected video track.
    ensure!(fields.u32()? != 0, "zero sidx timescale");
    let (mut time, offset) = if version == 0 {
        (u64::from(fields.u32()?), u64::from(fields.u32()?))
    } else {
        (fields.u64()?, fields.u64()?)
    };
    let mut position = location
        .end
        .checked_add(offset)
        .context("sidx offset overflow")?;
    ensure!(position <= size, "sidx first offset exceeds source size");
    ensure!(fields.u16()? == 0, "nonzero sidx reserved field");
    let count = usize::from(fields.u16()?);
    ensure!(count <= fields.0.len() / 12, "truncated sidx references");
    let mut references = Vec::with_capacity(count);
    for _ in 0..count {
        let value = fields.u32()?;
        let length = u64::from(value & 0x7fff_ffff);
        ensure!(length > 0, "empty sidx reference");
        let end = position
            .checked_add(length)
            .context("sidx reference overflow")?;
        ensure!(end <= size, "sidx reference exceeds source size");
        time = time
            .checked_add(u64::from(fields.u32()?))
            .context("sidx time overflow")?;
        fields.u32()?; // SAP metadata does not override native trun sync flags.
        references.push(Reference {
            range: position..end,
            nested: value >> 31 != 0,
        });
        position = end;
    }
    fields.finish()?;
    Ok(references)
}

#[derive(Clone, Copy, Default)]
struct Defaults {
    description: Option<u32>,
    duration: Option<u32>,
    size: Option<u32>,
    flags: Option<u32>,
}

struct Track {
    defaults: Defaults,
    next_time: u64,
}

fn defaults(bytes: &[u8]) -> anyhow::Result<BTreeMap<u32, Defaults>> {
    let mut input = bytes;
    let mut result = BTreeMap::new();
    while !input.is_empty() {
        let child = atom(&mut input)?;
        if child.kind != *b"mvex" {
            continue;
        }
        let mut extensions = child.payload;
        while !extensions.is_empty() {
            let extension = atom(&mut extensions)?;
            if extension.kind != *b"trex" {
                continue;
            }
            let mut fields = Fields(extension.payload);
            fields.full(0, 0)?;
            let track_id = fields.u32()?;
            ensure!(track_id != 0, "zero trex track ID");
            let value = Defaults {
                description: Some(fields.u32()?),
                duration: Some(fields.u32()?),
                size: Some(fields.u32()?),
                flags: Some(fields.u32()?),
            };
            fields.finish()?;
            ensure!(
                result.insert(track_id, value).is_none(),
                "duplicate trex track ID"
            );
            ensure!(result.len() <= MAX_TRACKS, "too many MP4 tracks");
        }
    }
    Ok(result)
}

struct FragmentHeader {
    track_id: u32,
    base: Option<u64>,
    flags: u32,
    defaults: Defaults,
}

fn tfhd(bytes: &[u8]) -> anyhow::Result<FragmentHeader> {
    let mut fields = Fields(bytes);
    let (_, flags) = fields.full(0, 0x0003_003b)?;
    let track_id = fields.u32()?;
    ensure!(track_id != 0, "zero tfhd track ID");
    let base = if flags & 1 != 0 {
        Some(fields.u64()?)
    } else {
        None
    };
    let mut optional = |flag| -> anyhow::Result<Option<u32>> {
        if flags & flag != 0 {
            Ok(Some(fields.u32()?))
        } else {
            Ok(None)
        }
    };
    let defaults = Defaults {
        description: optional(2)?,
        duration: optional(8)?,
        size: optional(0x10)?,
        flags: optional(0x20)?,
    };
    fields.finish()?;
    Ok(FragmentHeader {
        track_id,
        base,
        flags,
        defaults,
    })
}

struct Runs<'a> {
    base: u64,
    next_offset: Option<u64>,
    data_end: Option<u64>,
    time: u64,
    defaults: Defaults,
    selected: bool,
    size: u64,
    media: &'a [Range<u64>],
}

impl Runs<'_> {
    fn parse(
        &mut self,
        bytes: &[u8],
        samples: &mut Vec<Sample>,
        sample_count: &mut usize,
    ) -> anyhow::Result<()> {
        let mut fields = Fields(bytes);
        let (version, flags) = fields.full(1, 0x0000_0f05)?;
        ensure!(
            flags & 4 == 0 || flags & 0x400 == 0,
            "conflicting trun sample flags"
        );
        let count = usize::try_from(fields.u32()?)?;
        *sample_count = sample_count
            .checked_add(count)
            .context("MP4 sample count overflow")?;
        ensure!(
            *sample_count <= MAX_SAMPLES,
            "MP4 sample count exceeds limit"
        );
        let mut offset = if flags & 1 != 0 {
            let delta = i32::from_be_bytes(fields.array()?);
            self.base
                .checked_add_signed(i64::from(delta))
                .context("trun data offset overflow")?
        } else {
            self.next_offset.unwrap_or(self.base)
        };
        let first_flags = if flags & 4 != 0 {
            Some(fields.u32()?)
        } else {
            None
        };
        let entry_size = (flags & 0xf00).count_ones() as usize * 4;
        ensure!(
            count.checked_mul(entry_size) == Some(fields.0.len()),
            "invalid trun sample table size"
        );
        // Default-only runs can claim huge counts in tiny boxes. Check their
        // entire contiguous media extent before reserving the sample vector.
        if flags & 0x200 == 0 && count != 0 {
            let size = self.defaults.size.context("missing fragment sample size")?;
            self.media_end(offset, u64::try_from(count)? * u64::from(size))?;
        }
        if self.selected {
            samples.try_reserve(count)?;
        }
        for number in 0..count {
            let (sample, end) = self.sample(
                &mut fields,
                flags,
                version,
                first_flags.filter(|_| number == 0),
                offset,
            )?;
            self.time = self
                .time
                .checked_add(u64::from(sample.duration))
                .context("MP4 decode time overflow")?;
            if self.selected {
                samples.push(sample);
            }
            offset = end;
        }
        ensure!(offset <= self.size, "trun offset exceeds source size");
        self.next_offset = Some(offset);
        self.data_end = Some(self.data_end.map_or(offset, |end| end.max(offset)));
        fields.finish()
    }

    fn sample(
        &self,
        fields: &mut Fields<'_>,
        flags: u32,
        version: u8,
        first_flags: Option<u32>,
        offset: u64,
    ) -> anyhow::Result<(Sample, u64)> {
        let duration = if flags & 0x100 != 0 {
            fields.u32()?
        } else {
            self.defaults
                .duration
                .context("missing fragment sample duration")?
        };
        let size = if flags & 0x200 != 0 {
            fields.u32()?
        } else {
            self.defaults.size.context("missing fragment sample size")?
        };
        let sample_flags = if flags & 0x400 != 0 {
            fields.u32()?
        } else {
            first_flags
                .or(self.defaults.flags)
                .context("missing fragment sample flags")?
        };
        let rendering_offset = if flags & 0x800 != 0 {
            let value = fields.u32()?;
            if version == 1 {
                value.cast_signed()
            } else {
                i32::try_from(value).context("unsigned CTS exceeds supported i32 range")?
            }
        } else {
            0
        };
        let end = self.media_end(offset, u64::from(size))?;
        Ok((
            Sample {
                offset,
                size,
                start_time: self.time,
                duration,
                rendering_offset,
                is_sync: sample_flags & NON_SYNC == 0 && (sample_flags >> 24) & 3 != 1,
            },
            end,
        ))
    }

    fn media_end(&self, offset: u64, length: u64) -> anyhow::Result<u64> {
        let end = offset
            .checked_add(length)
            .context("MP4 sample offset overflow")?;
        ensure!(end <= self.size, "MP4 sample exceeds source size");
        let media = self
            .media
            .partition_point(|range| range.start <= offset)
            .checked_sub(1)
            .and_then(|index| self.media.get(index))
            .context("MP4 sample is outside mdat")?;
        ensure!(end <= media.end, "MP4 sample crosses mdat boundary");
        Ok(end)
    }
}

struct MovieFragments<'a> {
    selected: u32,
    tracks: BTreeMap<u32, Track>,
    samples: Vec<Sample>,
    sample_count: usize,
    media: &'a [Range<u64>],
    size: u64,
}

impl MovieFragments<'_> {
    fn parse(&mut self, bytes: &[u8], start: u64) -> anyhow::Result<()> {
        let mut children = bytes;
        let mut sequence = false;
        let mut previous_end = start;
        while !children.is_empty() {
            let child = atom(&mut children)?;
            match &child.kind {
                b"mfhd" => {
                    ensure!(!sequence, "duplicate mfhd");
                    let mut fields = Fields(child.payload);
                    fields.full(0, 0)?;
                    fields.u32()?;
                    fields.finish()?;
                    sequence = true;
                }
                b"traf" => previous_end = self.traf(child.payload, start, previous_end)?,
                _ => {}
            }
        }
        ensure!(sequence, "missing mfhd");
        Ok(())
    }

    fn traf(&mut self, bytes: &[u8], start: u64, previous_end: u64) -> anyhow::Result<u64> {
        let mut children = bytes;
        let mut fragment_header = None;
        let mut decode_time = None;
        let mut runs = Vec::new();
        while !children.is_empty() {
            let child = atom(&mut children)?;
            match &child.kind {
                b"tfhd" => {
                    ensure!(fragment_header.is_none(), "duplicate tfhd");
                    fragment_header = Some(tfhd(child.payload)?);
                }
                b"tfdt" => {
                    ensure!(decode_time.is_none(), "duplicate tfdt");
                    let mut fields = Fields(child.payload);
                    let (version, _) = fields.full(1, 0)?;
                    decode_time = Some(if version == 1 {
                        fields.u64()?
                    } else {
                        u64::from(fields.u32()?)
                    });
                    fields.finish()?;
                }
                b"trun" => runs.push(child.payload),
                _ => {}
            }
        }
        let head = fragment_header.context("missing tfhd")?;
        let track = self
            .tracks
            .get_mut(&head.track_id)
            .context("fragment references unknown track")?;
        let defaults = Defaults {
            description: head.defaults.description.or(track.defaults.description),
            duration: head.defaults.duration.or(track.defaults.duration),
            size: head.defaults.size.or(track.defaults.size),
            flags: head.defaults.flags.or(track.defaults.flags),
        };
        if head.track_id == self.selected {
            ensure!(
                defaults.description == Some(1),
                "unsupported fragment sample description"
            );
        }
        let base = head.base.unwrap_or(if head.flags & 0x0002_0000 != 0 {
            start
        } else {
            previous_end
        });
        ensure!(
            base <= self.size,
            "tfhd base data offset exceeds source size"
        );
        let mut parser = Runs {
            base,
            next_offset: None,
            data_end: None,
            time: decode_time.unwrap_or(track.next_time),
            defaults,
            selected: head.track_id == self.selected,
            size: self.size,
            media: self.media,
        };
        ensure!(
            parser.time >= track.next_time,
            "non-monotonic fragment decode time"
        );
        let before = self.sample_count;
        for run in runs {
            parser.parse(run, &mut self.samples, &mut self.sample_count)?;
        }
        ensure!(
            head.flags & 0x0001_0000 == 0 || before == self.sample_count,
            "nonempty duration-is-empty fragment"
        );
        track.next_time = parser.time;
        Ok(parser.data_end.unwrap_or(base))
    }
}

// Validate container boundaries before using mp4's codec/track parser: that
// parser assumes children advance and several leaf sizes cannot underflow.
fn validate_init(bytes: &[u8], depth: usize) -> anyhow::Result<()> {
    ensure!(depth <= 16, "MP4 initialization nesting exceeds limit");
    let mut input = bytes;
    let mut count = 0;
    let mut tracks = 0;
    while !input.is_empty() {
        count += 1;
        ensure!(count <= MAX_BOXES, "too many MP4 initialization boxes");
        let child = atom(&mut input)?;
        let mut fields = Fields(child.payload);
        match &child.kind {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts" | b"mvex" | b"udta"
            | b"dinf" | b"ilst" | b"\xa9nam" | b"\xa9day" | b"covr" | b"desc" => {
                if child.kind == *b"trak" {
                    tracks += 1;
                    ensure!(tracks <= MAX_TRACKS, "too many MP4 tracks");
                }
                validate_init(child.payload, depth + 1)?;
            }
            b"meta" => {
                if fields.0.starts_with(&[0; 4]) {
                    fields.take(4)?;
                }
                validate_init(fields.0, depth + 1)?;
            }
            b"stsd" | b"dref" => {
                fields.take(4)?;
                let count = usize::try_from(fields.u32()?)?;
                ensure!(count <= fields.0.len() / 8, "invalid MP4 entry count");
                if child.kind == *b"stsd" {
                    ensure!(count == 1, "unsupported MP4 sample descriptions");
                }
                let mut entries = fields.0;
                for _ in 0..count {
                    atom(&mut entries)?;
                }
                ensure!(
                    entries.is_empty(),
                    "MP4 entry count does not match box size"
                );
                validate_init(fields.0, depth + 1)?;
            }
            b"avc1" | b"hev1" | b"vp09" | b"mp4a" => {
                fields.take(if child.kind == *b"mp4a" { 28 } else { 78 })?;
                validate_init(fields.0, depth + 1)?;
            }
            _ => validate_init_leaf(&child)?,
        }
    }
    Ok(())
}

fn validate_init_leaf(child: &Atom<'_>) -> anyhow::Result<()> {
    let mut fields = Fields(child.payload);
    match &child.kind {
        b"mvhd" | b"tkhd" | b"mdhd" | b"mehd" => validate_versioned_leaf(child)?,
        b"stts" | b"ctts" | b"stsc" | b"stco" | b"co64" | b"stss" | b"elst" => {
            let version = fields.byte()?;
            fields.take(3)?;
            let count = usize::try_from(fields.u32()?)?;
            let width = match &child.kind {
                b"stts" | b"ctts" | b"co64" => 8,
                b"stsc" => 12,
                b"elst" => {
                    if version == 1 {
                        20
                    } else {
                        12
                    }
                }
                _ => 4,
            };
            ensure!(
                count.checked_mul(width) == Some(fields.0.len()),
                "invalid initialization sample table"
            );
        }
        b"stsz" => {
            fields.take(4)?;
            let size = fields.u32()?;
            let count = usize::try_from(fields.u32()?)?;
            let width = if size == 0 { 4 } else { 0 };
            ensure!(
                count.checked_mul(width) == Some(fields.0.len()),
                "invalid initialization sample sizes"
            );
        }
        b"avcC" => {
            fields.take(5)?;
            let count = fields.byte()? & 31;
            for _ in 0..count {
                let length = usize::from(fields.u16()?);
                fields.take(length)?;
            }
            let count = fields.byte()?;
            for _ in 0..count {
                let length = usize::from(fields.u16()?);
                fields.take(length)?;
            }
            // High-profile AVC may append chroma/bit-depth extensions.
        }
        b"esds" => {
            fields.take(4)?;
            validate_descriptors(fields.0, 0)?;
        }
        b"ftyp" => {
            fields.take(8)?;
            ensure!(
                fields.0.len().is_multiple_of(4),
                "invalid MP4 compatible brands"
            );
        }
        b"hdlr" | b"trex" => {
            fields.take(24)?;
        }
        b"vmhd" => {
            fields.take(12)?;
        }
        b"smhd" | b"data" => {
            fields.take(8)?;
        }
        b"url " => {
            fields.take(4)?;
        }
        b"hvcC" => {
            fields.take(1)?;
        }
        b"vpcC" => {
            fields.take(11)?;
        }
        b"tx3g" => {
            fields.take(38)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_versioned_leaf(child: &Atom<'_>) -> anyhow::Result<()> {
    let version = Fields(child.payload).byte()?;
    ensure!(version <= 1, "unsupported initialization box version");
    let (short, long) = match &child.kind {
        b"mvhd" => (100, 112),
        b"tkhd" => (84, 96),
        b"mdhd" => (24, 36),
        _ => (8, 12),
    };
    let minimum = if version == 1 { long } else { short };
    ensure!(
        child.payload.len() >= minimum,
        "truncated initialization fields"
    );
    Ok(())
}

fn validate_descriptors(bytes: &[u8], depth: usize) -> anyhow::Result<()> {
    ensure!(depth <= 4, "MP4 descriptor nesting exceeds limit");
    let mut fields = Fields(bytes);
    while !fields.0.is_empty() {
        let tag = fields.byte()?;
        let mut length = 0_usize;
        for number in 0..4 {
            let byte = fields.byte()?;
            length = (length << 7) | usize::from(byte & 0x7f);
            if byte & 0x80 == 0 {
                break;
            }
            ensure!(number < 3, "invalid MP4 descriptor length");
        }
        let mut descriptor = Fields(fields.take(length)?);
        match tag {
            3 => {
                descriptor.take(2)?;
                ensure!(descriptor.byte()? == 0, "unsupported ES descriptor flags");
                validate_descriptors(descriptor.0, depth + 1)?;
            }
            4 => {
                descriptor.take(13)?;
                validate_descriptors(descriptor.0, depth + 1)?;
            }
            5 => {
                let first = descriptor.byte()?;
                let second = descriptor.byte()?;
                let extended = first >> 3 == 31;
                let frequency = if extended {
                    (second >> 1) & 15
                } else {
                    ((first & 7) << 1) | (second >> 7)
                };
                if frequency == 15 {
                    descriptor.take(3)?;
                } else if extended {
                    descriptor.take(1)?;
                }
            }
            6 => {
                descriptor.take(1)?;
            }
            _ => {}
        }
    }
    Ok(())
}

struct Layout {
    boxes: Vec<Header>,
    media: Vec<Range<u64>>,
    remaining: u64,
}

fn walk<R: Read + Seek>(reader: &mut R, size: u64) -> anyhow::Result<Layout> {
    let mut position = 0;
    let mut boxes = Vec::new();
    let mut media = Vec::new();
    let mut references = Vec::new();
    let mut remaining = MAX_METADATA;
    while position < size {
        ensure!(boxes.len() < MAX_BOXES, "too many top-level MP4 boxes");
        let location = header(reader, position, size)?;
        match &location.kind {
            b"mdat" => media.push(location.payload..location.end),
            b"sidx" => {
                let bytes = metadata(reader, location, &mut remaining)?;
                let parsed = atom(&mut bytes.as_slice())?;
                references.extend(sidx(parsed.payload, location, size)?);
            }
            _ => {}
        }
        boxes.push(location);
        position = location.end;
    }
    for reference in references {
        let start = boxes
            .binary_search_by_key(&reference.range.start, |location| location.start)
            .map_err(|_| anyhow::anyhow!("sidx reference starts inside a box"))?;
        ensure!(
            !reference.nested || boxes[start].kind == *b"sidx",
            "nested sidx reference is not a sidx"
        );
        ensure!(
            reference.range.end == size
                || boxes
                    .binary_search_by_key(&reference.range.end, |location| location.start)
                    .is_ok(),
            "sidx reference ends inside a box"
        );
    }
    Ok(Layout {
        boxes,
        media,
        remaining,
    })
}

/// Index fragments without reading mdat bodies; ordinary non-fragmented MP4 returns `None`.
///
/// Metadata is capped at 16 MiB, top-level boxes at 100,000, and samples at one million.
/// Malformed sizes, unresolved defaults, unsupported CTS ranges, and invalid media offsets fail closed.
pub(crate) fn index<R: Read + Seek>(
    reader: &mut R,
    size: u64,
) -> anyhow::Result<Option<FragmentIndex>> {
    let Layout {
        boxes,
        media,
        mut remaining,
    } = walk(reader, size)?;
    if !boxes.iter().any(|location| location.kind == *b"moof") {
        return Ok(None);
    }
    let find = |kind| -> anyhow::Result<Header> {
        let mut matching = boxes.iter().filter(|location| location.kind == kind);
        let location = *matching.next().context("missing MP4 initialization box")?;
        ensure!(
            matching.next().is_none(),
            "duplicate MP4 initialization box"
        );
        Ok(location)
    };
    let ftyp = find(*b"ftyp")?;
    let moov = find(*b"moov")?;
    let mut init_bytes = metadata(reader, ftyp, &mut remaining)?;
    let ftyp_size = init_bytes.len();
    let movie_size = moov.end - moov.start;
    remaining = remaining
        .checked_sub(movie_size)
        .context("MP4 metadata exceeds 16 MiB")?;
    let movie_size = usize::try_from(movie_size)?;
    init_bytes.try_reserve_exact(movie_size)?;
    init_bytes.resize(ftyp_size + movie_size, 0);
    reader.seek(SeekFrom::Start(moov.start))?;
    reader.read_exact(&mut init_bytes[ftyp_size..])?;
    validate_init(&init_bytes, 0)?;
    let mut track_defaults = defaults(atom(&mut &init_bytes[ftyp_size..])?.payload)?;
    let init_size = init_bytes.len() as u64;
    let init = Mp4Reader::read_header(Cursor::new(init_bytes), init_size)
        .context("parsing fragmented MP4 initialization")?;
    ensure!(
        init.moov.traks.len() == init.tracks().len(),
        "duplicate MP4 track IDs"
    );
    ensure!(init.tracks().len() <= MAX_TRACKS, "too many MP4 tracks");
    let track_id = init
        .tracks()
        .values()
        .filter(|track| {
            track
                .track_type()
                .is_ok_and(|kind| kind == TrackType::Video)
        })
        .map(mp4::Mp4Track::track_id)
        .min()
        .context("MP4 has no video track")?;
    let mut tracks = BTreeMap::new();
    for (&id, track) in init.tracks() {
        let next_time =
            track
                .trak
                .mdia
                .minf
                .stbl
                .stts
                .entries
                .iter()
                .try_fold(0_u64, |time, entry| {
                    time.checked_add(u64::from(entry.sample_count) * u64::from(entry.sample_delta))
                        .context("initial sample decode time overflow")
                })?;
        tracks.insert(
            id,
            Track {
                defaults: track_defaults.remove(&id).unwrap_or_default(),
                next_time,
            },
        );
    }
    ensure!(track_defaults.is_empty(), "trex references unknown track");
    let mut fragments = MovieFragments {
        selected: track_id,
        tracks,
        samples: Vec::new(),
        sample_count: 0,
        media: &media,
        size,
    };
    for location in boxes.iter().filter(|location| location.kind == *b"moof") {
        let bytes = metadata(reader, *location, &mut remaining)?;
        fragments.parse(atom(&mut bytes.as_slice())?.payload, location.start)?;
    }
    ensure!(
        !fragments.samples.is_empty(),
        "MP4 has no video fragment samples"
    );
    Ok(Some(FragmentIndex {
        init,
        track_id,
        samples: fragments.samples,
    }))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::index;

    fn atom(kind: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut bytes = u32::try_from(payload.len() + 8)
            .expect("fixture box size")
            .to_be_bytes()
            .to_vec();
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn full_atom(kind: &[u8], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
        let mut content = ((u32::from(version) << 24) | flags).to_be_bytes().to_vec();
        content.extend_from_slice(payload);
        atom(kind, &content)
    }

    fn words(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect()
    }

    fn init(defaults: &[[u32; 5]]) -> Vec<u8> {
        let config = mp4::Mp4Config {
            major_brand: "isom".parse().expect("brand"),
            minor_version: 0,
            compatible_brands: vec!["isom".parse().expect("brand")],
            timescale: 1000,
        };
        let mut writer =
            mp4::Mp4Writer::write_start(Cursor::new(Vec::new()), &config).expect("fixture writer");
        for _ in defaults {
            writer
                .add_track(&mp4::TrackConfig::from(mp4::AvcConfig {
                    width: 32,
                    height: 32,
                    seq_param_set: vec![0x67, 0x64, 0, 0x0a],
                    pic_param_set: vec![0x68, 0],
                }))
                .expect("fixture video track");
        }
        writer.write_end().expect("empty fixture movie");
        let file = writer.into_writer().into_inner();
        let mut output = Vec::new();
        let mut position = 0;
        while position < file.len() {
            let length =
                u32::from_be_bytes(file[position..position + 4].try_into().expect("box size"))
                    as usize;
            match &file[position + 4..position + 8] {
                b"ftyp" => output.extend_from_slice(&file[position..position + length]),
                b"moov" => {
                    let mut movie = file[position + 8..position + length].to_vec();
                    let extensions: Vec<_> = defaults
                        .iter()
                        .flat_map(|values| full_atom(b"trex", 0, 0, &words(values)))
                        .collect();
                    movie.extend(atom(b"mvex", &extensions));
                    output.extend(atom(b"moov", &movie));
                }
                _ => {}
            }
            position += length;
        }
        output
    }

    fn moof(trafs: &[Vec<u8>]) -> Vec<u8> {
        let mut content = full_atom(b"mfhd", 0, 0, &words(&[1]));
        for traf in trafs {
            content.extend(atom(b"traf", traf));
        }
        atom(b"moof", &content)
    }

    #[test]
    fn preserves_native_times_offsets_flags_and_multiple_runs() {
        let mut source = init(&[[1, 1, 90, 4, 0x0101_0000]]);
        let mut traf = full_atom(b"tfhd", 0, 0x0002_0000, &words(&[1]));
        traf.extend(full_atom(b"tfdt", 1, 0, &900_u64.to_be_bytes()));
        let first_run = |offset| {
            full_atom(
                b"trun",
                1,
                0x0000_0f01,
                &words(&[
                    2,
                    offset,
                    40,
                    3,
                    0x0200_0000,
                    (-20_i32).cast_unsigned(),
                    60,
                    5,
                    0x0101_0000,
                    30,
                ]),
            )
        };
        let mut placeholder = traf.clone();
        placeholder.extend(first_run(0));
        placeholder.extend(full_atom(b"trun", 0, 4, &words(&[1, 0x0200_0000])));
        let offset = u32::try_from(moof(&[placeholder]).len() + 8).expect("data offset");
        traf.extend(first_run(offset));
        traf.extend(full_atom(b"trun", 0, 4, &words(&[1, 0x0200_0000])));
        let data_start = source.len() as u64 + u64::from(offset);
        source.extend(moof(&[traf]));
        source.extend(atom(b"mdat", &[0; 12]));
        let length = source.len() as u64;
        let result = index(&mut Cursor::new(source), length)
            .expect("fragment index")
            .expect("fragmented video");
        assert_eq!(result.track_id, 1);
        assert_eq!(result.init.tracks()[&1].width(), 32);
        let samples: Vec<_> = result
            .samples
            .iter()
            .map(|sample| {
                (
                    sample.offset,
                    sample.size,
                    sample.start_time,
                    sample.duration,
                    sample.rendering_offset,
                    sample.is_sync,
                )
            })
            .collect();
        assert_eq!(
            samples,
            vec![
                (data_start, 3, 900, 40, -20, true),
                (data_start + 3, 5, 940, 60, 30, false),
                (data_start + 8, 4, 1000, 90, 0, true),
            ]
        );
    }

    #[test]
    fn rejects_malformed_init_children_without_entering_dependency_parser() {
        for child in [
            words(&[0, u32::from_be_bytes(*b"free")]),
            words(&[7, u32::from_be_bytes(*b"free")]),
            words(&[1000, u32::from_be_bytes(*b"free")]),
            atom(b"mvhd", &[]),
            atom(b"stts", &words(&[0, u32::MAX])),
            atom(b"data", &[0; 4]),
        ] {
            let mut source = atom(b"ftyp", b"isom\0\0\0\0");
            source.extend(atom(b"moov", &child));
            source.extend(moof(&[]));
            assert!(index(&mut Cursor::new(&source), source.len() as u64).is_err());
        }
    }

    fn append_fragment(source: &mut Vec<u8>, count: u32) -> std::ops::Range<u64> {
        let make = |offset| {
            let mut traf = full_atom(b"tfhd", 0, 0x0002_0000, &words(&[1]));
            traf.extend(full_atom(b"trun", 0, 1, &words(&[count, offset])));
            moof(&[traf])
        };
        let length = make(0).len();
        let fragment = make(u32::try_from(length + 8).expect("data offset"));
        let start = (source.len() + length + 8) as u64;
        source.extend(fragment);
        source.extend(atom(b"mdat", &vec![0; count as usize * 4]));
        start..source.len() as u64
    }

    #[test]
    fn inherits_trex_and_continues_decode_time_when_tfdt_is_absent() {
        let mut source = init(&[[1, 1, 25, 4, 0x0200_0000]]);
        let first = append_fragment(&mut source, 2);
        let second = append_fragment(&mut source, 1);
        let result = index(&mut Cursor::new(&source), source.len() as u64)
            .expect("index")
            .expect("fragments");
        assert_eq!(
            result
                .samples
                .iter()
                .map(|sample| sample.start_time)
                .collect::<Vec<_>>(),
            [0, 25, 50]
        );
        assert_eq!(result.samples[0].offset, first.start);
        assert_eq!(result.samples[2].offset, second.start);
        assert!(result.samples.iter().all(|sample| sample.is_sync));
    }

    #[test]
    fn tfhd_overrides_trex_and_explicit_base_accepts_negative_run_offset() {
        let mut source = init(&[[1, 1, 90, 9, 0x0101_0000]]);
        let make = |base: u64| {
            let mut head = words(&[1]);
            head.extend(base.to_be_bytes());
            head.extend(words(&[1, 45, 4, 0x0200_0000]));
            let mut traf = full_atom(b"tfhd", 0, 0x3b, &head);
            traf.extend(full_atom(b"tfdt", 0, 0, &words(&[1234])));
            traf.extend(full_atom(
                b"trun",
                0,
                1,
                &words(&[1, (-4_i32).cast_unsigned()]),
            ));
            moof(&[traf])
        };
        let data_start = (source.len() + make(0).len() + 8) as u64;
        source.extend(make(data_start + 4));
        source.extend(atom(b"mdat", &[0; 4]));
        let result = index(&mut Cursor::new(&source), source.len() as u64)
            .expect("index")
            .expect("fragment");
        assert_eq!(
            result.samples[0],
            super::Sample {
                offset: data_start,
                size: 4,
                start_time: 1234,
                duration: 45,
                rendering_offset: 0,
                is_sync: true,
            }
        );
    }

    #[test]
    fn implicit_base_follows_previous_traf_data_for_different_track() {
        let mut source = init(&[[1, 1, 40, 4, 0x0200_0000], [2, 1, 20, 3, 0]]);
        let make = |offset| {
            let mut first = full_atom(b"tfhd", 0, 0, &words(&[2]));
            first.extend(full_atom(b"trun", 0, 1, &words(&[2, offset])));
            let mut second = full_atom(b"tfhd", 0, 0, &words(&[1]));
            second.extend(full_atom(b"trun", 0, 0, &words(&[1])));
            moof(&[first, second])
        };
        let offset = u32::try_from(make(0).len() + 8).expect("offset");
        let selected_start = source.len() as u64 + u64::from(offset) + 6;
        source.extend(make(offset));
        source.extend(atom(b"mdat", &[0; 10]));
        let result = index(&mut Cursor::new(&source), source.len() as u64)
            .expect("index")
            .expect("fragment");
        assert_eq!(result.track_id, 1);
        assert_eq!(result.samples.len(), 1);
        assert_eq!(result.samples[0].offset, selected_start);
    }

    #[test]
    fn unsigned_cts_and_dependency_flags_preserve_packet_independence() {
        let mut source = init(&[[1, 1, 40, 4, 0x0100_0000]]);
        let make = |offset| {
            let mut traf = full_atom(b"tfhd", 0, 0x0002_0000, &words(&[1]));
            traf.extend(full_atom(b"trun", 0, 0x801, &words(&[1, offset, 120])));
            moof(&[traf])
        };
        let offset = u32::try_from(make(0).len() + 8).expect("offset");
        source.extend(make(offset));
        source.extend(atom(b"mdat", &[0; 4]));
        let result = index(&mut Cursor::new(&source), source.len() as u64)
            .expect("index")
            .expect("fragment");
        assert_eq!(result.samples[0].rendering_offset, 120);
        assert!(
            !result.samples[0].is_sync,
            "depends_on=1 is not an independent sample"
        );
    }

    fn indexed_source(version: u8) -> Vec<u8> {
        let mut source = init(&[[1, 1, 40, 4, 0x0200_0000]]);
        let mut segment = Vec::new();
        append_fragment(&mut segment, 1);
        let mut payload = words(&[1, 1000]);
        if version == 0 {
            payload.extend(words(&[0, 8]));
        } else {
            payload.extend(0_u64.to_be_bytes());
            payload.extend(8_u64.to_be_bytes());
        }
        payload.extend([0, 0, 0, 1]);
        payload.extend(words(&[
            u32::try_from(segment.len()).expect("segment size"),
            40,
            0x9000_0000,
        ]));
        source.extend(full_atom(b"sidx", version, 0, &payload));
        source.extend(atom(b"free", &[]));
        source.extend(segment);
        source
    }

    #[test]
    fn sidx_versions_zero_and_one_resolve_first_offset_after_index_box() {
        for version in [0, 1] {
            let source = indexed_source(version);
            let result = index(&mut Cursor::new(&source), source.len() as u64)
                .expect("index")
                .expect("fragment");
            assert_eq!(result.samples.len(), 1);
            assert_eq!(result.samples[0].offset, source.len() as u64 - 4);
        }
    }

    #[test]
    fn rejects_sidx_offsets_sizes_versions_and_truncated_references() {
        for (word, value) in [
            (0, 0x0200_0000),
            (2, 0),
            (4, u32::MAX),
            (6, 0),
            (6, u32::MAX),
        ] {
            let mut source = indexed_source(0);
            let position = source
                .windows(4)
                .position(|window| window == b"sidx")
                .expect("sidx");
            let start = position + 4 + word * 4;
            source[start..start + 4].copy_from_slice(&value.to_be_bytes());
            assert!(index(&mut Cursor::new(&source), source.len() as u64).is_err());
        }
        let mut source = indexed_source(0);
        let position = source
            .windows(4)
            .position(|window| window == b"sidx")
            .expect("sidx");
        source[position + 4 + 5 * 4 + 2..position + 4 + 6 * 4]
            .copy_from_slice(&2_u16.to_be_bytes());
        assert!(index(&mut Cursor::new(&source), source.len() as u64).is_err());
    }

    #[test]
    fn rejects_bad_box_sizes_truncated_runs_huge_counts_and_invalid_media_offsets() {
        let mut valid = init(&[[1, 1, 40, 4, 0x0200_0000]]);
        append_fragment(&mut valid, 1);
        for (kind, relative, value) in [
            (*b"moof", -4, 7),
            (*b"tfhd", -4, 7),
            (*b"trun", -4, 1000),
            (*b"trun", 8, u32::MAX),
            (*b"trun", 12, 0),
            (*b"trun", 12, u32::MAX),
            (*b"tfhd", 8, 99),
        ] {
            let mut source = valid.clone();
            let position = source
                .windows(4)
                .position(|window| window == kind)
                .expect("target box");
            let start = position.checked_add_signed(relative).expect("field offset");
            source[start..start + 4].copy_from_slice(&value.to_be_bytes());
            assert!(
                index(&mut Cursor::new(&source), source.len() as u64).is_err(),
                "{kind:?} {relative}"
            );
        }
        for length in 1..valid.len() {
            let source = &valid[..length];
            // An exact init or moof boundary can be a valid non-fragmented file
            // or an empty segment; all other truncated structures must error.
            let result = index(&mut Cursor::new(source), length as u64);
            assert!(result.is_err() || result.is_ok_and(|value| value.is_none()));
        }
    }

    #[test]
    fn rejects_decode_time_overflow_conflicting_flags_and_unsupported_cts() {
        for (time, flags, entries) in [
            (u64::MAX, 1, vec![]),
            (0, 0x405, vec![0, 0]),
            (0, 0x801, vec![u32::MAX]),
        ] {
            let mut source = init(&[[1, 1, 40, 4, 0x0200_0000]]);
            let make = |offset| {
                let mut traf = full_atom(b"tfhd", 0, 0x0002_0000, &words(&[1]));
                traf.extend(full_atom(b"tfdt", 1, 0, &time.to_be_bytes()));
                let mut values = vec![1, offset];
                values.extend(&entries);
                traf.extend(full_atom(b"trun", 0, flags, &words(&values)));
                moof(&[traf])
            };
            let offset = u32::try_from(make(0).len() + 8).expect("offset");
            source.extend(make(offset));
            source.extend(atom(b"mdat", &[0; 4]));
            assert!(index(&mut Cursor::new(&source), source.len() as u64).is_err());
        }
    }

    struct MetadataOnly {
        source: Cursor<Vec<u8>>,
        forbidden: Vec<std::ops::Range<u64>>,
        bytes_read: usize,
    }

    impl std::io::Read for MetadataOnly {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            let start = self.source.position();
            let end = start + output.len() as u64;
            if self
                .forbidden
                .iter()
                .any(|range| start < range.end && end > range.start)
            {
                return Err(std::io::Error::other(
                    "index attempted to read media payload",
                ));
            }
            let count = std::io::Read::read(&mut self.source, output)?;
            self.bytes_read += count;
            Ok(count)
        }
    }

    impl std::io::Seek for MetadataOnly {
        fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
            std::io::Seek::seek(&mut self.source, position)
        }
    }

    #[test]
    fn walks_without_sidx_by_seeking_over_mdat_payloads() {
        let mut source = init(&[[1, 1, 40, 4, 0x0200_0000]]);
        let first = append_fragment(&mut source, 1);
        let second = append_fragment(&mut source, 1);
        let length = source.len() as u64;
        let mut reader = MetadataOnly {
            source: Cursor::new(source),
            forbidden: vec![first, second],
            bytes_read: 0,
        };
        let result = index(&mut reader, length)
            .expect("metadata-only reads")
            .expect("fragments");
        assert_eq!(result.samples.len(), 2);
        assert!(
            reader.bytes_read < 4096,
            "only box headers and metadata are read"
        );
    }

    #[test]
    fn ordinary_mp4_returns_none_without_reading_media() {
        let mut source = init(&[[1, 1, 40, 4, 0]]);
        let start = source.len() as u64 + 8;
        source.extend(atom(b"mdat", &[0; 1024]));
        let length = source.len() as u64;
        let mut reader = MetadataOnly {
            source: Cursor::new(source),
            forbidden: std::iter::once(start..length).collect(),
            bytes_read: 0,
        };
        assert!(index(&mut reader, length).expect("ordinary MP4").is_none());
        assert_eq!(reader.bytes_read, 24);
    }

    fn compare_packets(path: &std::path::Path) {
        let bytes = std::fs::read(path).expect("generated video");
        let result = index(&mut Cursor::new(&bytes), bytes.len() as u64)
            .expect("generated fragment index")
            .expect("fragmented MP4");
        let output = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pos,size,dts,pts,duration,flags",
                "-of",
                "json",
            ])
            .arg(path)
            .output()
            .expect("ffprobe");
        assert!(
            output.status.success(),
            "ffprobe: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let probe: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("ffprobe JSON");
        let packets = probe["packets"].as_array().expect("video packets");
        assert_eq!(result.samples.len(), packets.len());
        // The MOV demuxer may shift signed CTS for decoder timestamp constraints.
        // Compare relative PTS here; synthetic fixtures assert absolute native CTS.
        let first = result.samples.first().expect("first sample");
        let native_origin =
            i64::try_from(first.start_time).expect("first DTS") + i64::from(first.rendering_offset);
        let probe_origin = packets.first().expect("first packet")["pts"]
            .as_i64()
            .expect("first PTS");
        for (sample, packet) in result.samples.iter().zip(packets) {
            let offset: u64 = packet["pos"]
                .as_str()
                .expect("packet offset")
                .parse()
                .expect("offset number");
            let size: u32 = packet["size"]
                .as_str()
                .expect("packet size")
                .parse()
                .expect("size number");
            assert_eq!((sample.offset, sample.size), (offset, size));
            let dts = i64::try_from(sample.start_time).expect("DTS");
            assert_eq!(packet["dts"].as_i64(), Some(dts));
            assert_eq!(
                packet["pts"].as_i64().expect("PTS") - probe_origin,
                dts + i64::from(sample.rendering_offset) - native_origin,
            );
            assert_eq!(
                packet["duration"].as_u64(),
                Some(u64::from(sample.duration))
            );
            assert_eq!(
                sample.is_sync,
                packet["flags"].as_str().expect("flags").contains('K')
            );
        }
    }

    #[test]
    fn ffmpeg_fragments_and_dash_sidx_match_ffprobe_native_packets() {
        let version = std::process::Command::new("ffmpeg")
            .arg("-version")
            .output();
        if version
            .as_ref()
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        {
            return;
        }
        assert!(version.expect("ffmpeg availability").status.success());
        let directory = tempfile::tempdir().expect("fragment fixtures");
        for (number, flags) in [
            "frag_keyframe+empty_moov+default_base_moof",
            "frag_keyframe+empty_moov+default_base_moof+dash+negative_cts_offsets",
            "frag_keyframe+empty_moov+default_base_moof+global_sidx",
        ]
        .iter()
        .enumerate()
        {
            let path = directory.path().join(format!("fragment-{number}.mp4"));
            let output = std::process::Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x64:rate=12:duration=2",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-preset",
                    "veryfast",
                    "-g",
                    "6",
                    "-keyint_min",
                    "6",
                    "-sc_threshold",
                    "0",
                    "-bf",
                    "2",
                    "-movflags",
                    flags,
                ])
                .arg(&path)
                .output()
                .expect("ffmpeg fixture");
            assert!(
                output.status.success(),
                "ffmpeg: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let bytes = std::fs::read(&path).expect("fixture bytes");
            assert_eq!(
                bytes.windows(4).any(|window| window == b"sidx"),
                number != 0
            );
            compare_packets(&path);
        }
    }

    #[test]
    fn negative_offset_base_does_not_replace_actual_preceding_traf_end() {
        let mut source = init(&[[1, 1, 40, 4, 0x0200_0000], [2, 1, 20, 3, 0]]);
        let make = |base: u64| {
            let mut header = words(&[2]);
            header.extend(base.to_be_bytes());
            let mut first = full_atom(b"tfhd", 0, 1, &header);
            first.extend(full_atom(
                b"trun",
                0,
                1,
                &words(&[1, (-16_i32).cast_unsigned()]),
            ));
            let mut second = full_atom(b"tfhd", 0, 0, &words(&[1]));
            second.extend(full_atom(b"trun", 0, 0, &words(&[1])));
            moof(&[first, second])
        };
        let data_start = (source.len() + make(0).len() + 8) as u64;
        source.extend(make(data_start + 16));
        source.extend(atom(b"mdat", &[0; 32]));
        let result = index(&mut Cursor::new(&source), source.len() as u64)
            .expect("index")
            .expect("fragment");
        assert_eq!(result.samples[0].offset, data_start + 3);
    }

    #[test]
    fn first_sample_flags_do_not_leak_to_later_samples() {
        let mut source = init(&[[1, 1, 40, 4, 0x0101_0000]]);
        let make = |offset| {
            let mut traf = full_atom(b"tfhd", 0, 0x0002_0000, &words(&[1]));
            traf.extend(full_atom(b"trun", 0, 5, &words(&[2, offset, 0x0200_0000])));
            moof(&[traf])
        };
        let offset = u32::try_from(make(0).len() + 8).expect("offset");
        source.extend(make(offset));
        source.extend(atom(b"mdat", &[0; 8]));
        let result = index(&mut Cursor::new(&source), source.len() as u64)
            .expect("index")
            .expect("fragment");
        assert!(result.samples[0].is_sync);
        assert!(!result.samples[1].is_sync);
    }

    #[test]
    fn extended_size_and_to_eof_mdat_headers_skip_payloads() {
        for extended in [true, false] {
            let mut source = init(&[[1, 1, 40, 4, 0x0200_0000]]);
            let make = |offset| {
                let mut traf = full_atom(b"tfhd", 0, 0x0002_0000, &words(&[1]));
                traf.extend(full_atom(b"trun", 0, 1, &words(&[1, offset])));
                moof(&[traf])
            };
            let header_size = if extended { 16 } else { 8 };
            let offset = u32::try_from(make(0).len() + header_size).expect("offset");
            let data_start = source.len() as u64 + u64::from(offset);
            source.extend(make(offset));
            source.extend(u32::from(extended).to_be_bytes());
            source.extend(b"mdat");
            if extended {
                source.extend(20_u64.to_be_bytes());
            }
            source.extend([0; 4]);
            let length = source.len() as u64;
            let mut reader = MetadataOnly {
                source: Cursor::new(source),
                forbidden: std::iter::once(data_start..length).collect(),
                bytes_read: 0,
            };
            let result = index(&mut reader, length)
                .expect("headers")
                .expect("fragment");
            assert_eq!(result.samples[0].offset, data_start);
        }
    }
}

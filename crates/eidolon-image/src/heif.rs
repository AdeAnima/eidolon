//! HEIF native orientation takes precedence over redundant EXIF orientation.
//!
//! libheif-rs applies irot/imir but does not expose their presence. Inspect only
//! primary-item property associations, seeking past image data without loading
//! the entire container. Thumbnail transformations must not affect the primary.

use std::fs::File;
use std::io::{BufReader, Read as _, Seek as _, SeekFrom};
use std::ops::Range;
use std::path::Path;

use anyhow::{Context as _, ensure};

type Reader = BufReader<File>;

struct BoxInfo {
    kind: [u8; 4],
    content: Range<u64>,
}

fn next_box(reader: &mut Reader, position: &mut u64, end: u64) -> anyhow::Result<Option<BoxInfo>> {
    if *position == end {
        return Ok(None);
    }
    ensure!(
        end.saturating_sub(*position) >= 8,
        "truncated HEIF box header"
    );
    reader.seek(SeekFrom::Start(*position))?;
    let mut header = [0_u8; 8];
    reader.read_exact(&mut header)?;
    let size = u64::from(u32::from_be_bytes(header[..4].try_into()?));
    let kind = header[4..].try_into()?;
    let (size, header_size) = if size == 1 {
        let mut extended = [0_u8; 8];
        reader.read_exact(&mut extended)?;
        (u64::from_be_bytes(extended), 16)
    } else if size == 0 {
        (end - *position, 8)
    } else {
        (size, 8)
    };
    ensure!(
        size >= header_size && size <= end - *position,
        "invalid HEIF box size"
    );
    let content = *position + header_size..*position + size;
    *position += size;
    Ok(Some(BoxInfo { kind, content }))
}

fn find_box(
    reader: &mut Reader,
    parent: Range<u64>,
    kind: [u8; 4],
) -> anyhow::Result<Option<Range<u64>>> {
    let mut position = parent.start;
    while let Some(item) = next_box(reader, &mut position, parent.end)? {
        if item.kind == kind {
            return Ok(Some(item.content));
        }
    }
    Ok(None)
}

fn uint<const N: usize>(reader: &mut Reader) -> anyhow::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes[4 - N..])?;
    Ok(u32::from_be_bytes(bytes))
}

fn associations(
    reader: &mut Reader,
    range: Range<u64>,
    primary: u32,
    output: &mut Vec<u32>,
) -> anyhow::Result<()> {
    reader.seek(SeekFrom::Start(range.start))?;
    let full_box = uint::<4>(reader)?;
    let version = full_box >> 24;
    let wide = full_box & 1 != 0;
    let count = uint::<4>(reader)?;
    ensure!(
        u64::from(count) <= range.end.saturating_sub(reader.stream_position()?) / 3,
        "invalid HEIF association count"
    );
    for _ in 0..count {
        let item = if version == 0 {
            uint::<2>(reader)?
        } else {
            uint::<4>(reader)?
        };
        let count = uint::<1>(reader)?;
        for _ in 0..count {
            let property = if wide {
                uint::<2>(reader)? & 0x7fff
            } else {
                uint::<1>(reader)? & 0x7f
            };
            ensure!(
                reader.stream_position()? <= range.end,
                "truncated HEIF association"
            );
            if item == primary && property != 0 {
                output.push(property);
            }
        }
    }
    Ok(())
}

pub(crate) fn has_native_orientation(path: &Path) -> anyhow::Result<bool> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic[4..] != b"ftyp" {
        return Ok(false);
    }
    let Some(mut metadata) = find_box(&mut reader, 0..size, *b"meta")? else {
        return Ok(false);
    };
    metadata.start += 4; // meta is a FullBox.
    let Some(primary) = find_box(&mut reader, metadata.clone(), *b"pitm")? else {
        return Ok(false);
    };
    reader.seek(SeekFrom::Start(primary.start))?;
    let version = uint::<4>(&mut reader)? >> 24;
    let primary = if version == 0 {
        uint::<2>(&mut reader)?
    } else {
        uint::<4>(&mut reader)?
    };
    let Some(properties) = find_box(&mut reader, metadata, *b"iprp")? else {
        return Ok(false);
    };
    let property_table =
        find_box(&mut reader, properties.clone(), *b"ipco")?.context("HEIF iprp has no ipco")?;
    let mut associated = Vec::new();
    let mut position = properties.start;
    while let Some(item) = next_box(&mut reader, &mut position, properties.end)? {
        if item.kind == *b"ipma" {
            associations(&mut reader, item.content, primary, &mut associated)?;
        }
    }
    let mut position = property_table.start;
    let mut index = 0;
    while let Some(property) = next_box(&mut reader, &mut position, property_table.end)? {
        index += 1;
        if matches!(&property.kind, b"irot" | b"imir") && associated.contains(&index) {
            return Ok(true);
        }
    }
    Ok(false)
}

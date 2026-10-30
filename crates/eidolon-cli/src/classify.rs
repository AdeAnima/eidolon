//! Content sniffing and streaming text validation, independent of extensions.

use std::fs::File;
use std::io::{BufRead as _, BufReader, Read as _};
use std::path::Path;
use std::process::Command;

use anyhow::Context as _;
use eidolon_core::cache::FeatureCache;
use eidolon_core::{FileEntry, Kind};
use rayon::prelude::*;

const KIND_FEATURE: &str = "cli.kind/v3";

pub(crate) fn classify(files: &mut [FileEntry], cache: &FeatureCache) -> anyhow::Result<()> {
    files.par_iter_mut().try_for_each(|file| {
        let encoded = cache.get_or_compute(file, KIND_FEATURE, || {
            Ok(vec![match sniff(&file.path)
                .with_context(|| format!("classify {}", file.path.display()))?
            {
                Kind::Image => 0,
                Kind::Video => 1,
                Kind::Text => 2,
                Kind::Other => 3,
            }])
        })?;
        file.kind = match encoded.as_slice() {
            [0] => Kind::Image,
            [1] => Kind::Video,
            [2] => Kind::Text,
            [3] => Kind::Other,
            _ => anyhow::bail!("invalid cached classification for {}", file.path.display()),
        };
        Ok(())
    })
}

fn sniff(path: &Path) -> anyhow::Result<Kind> {
    let mut prefix = Vec::with_capacity(8192);
    File::open(path)?.take(8192).read_to_end(&mut prefix)?;
    if let Some(inferred) = infer::get(&prefix) {
        let mime = inferred.mime_type();
        if mime.starts_with("image/") {
            return Ok(Kind::Image);
        }
        if mime.starts_with("video/") {
            return Ok(Kind::Video);
        }
        // Documents/archives stay exact-only, including ASCII PDF and RTF.
        // Other signatures can be weak: infer calls any initial "BC" LLVM.
        // Let complete text validation reject binary applications/fonts, while
        // audio containers still reach the video fallback (for example Ogg).
        if matches!(
            inferred.matcher_type(),
            infer::MatcherType::Doc | infer::MatcherType::Archive | infer::MatcherType::Book
        ) {
            return Ok(Kind::Other);
        }
    }
    if is_text(path)? {
        return Ok(Kind::Text);
    }
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("run ffprobe (install ffmpeg with Homebrew)")?;
    if output.status.success() {
        let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        if value["streams"]
            .as_array()
            .is_some_and(|streams| streams.iter().any(|stream| stream["codec_type"] == "video"))
        {
            return Ok(Kind::Video);
        }
    }
    Ok(Kind::Other)
}

fn is_text(path: &Path) -> anyhow::Result<bool> {
    let mut reader = BufReader::new(File::open(path)?);
    let prefix = reader.fill_buf()?;
    let endian = if prefix.starts_with(&[0xff, 0xfe]) {
        Some(true)
    } else if prefix.starts_with(&[0xfe, 0xff]) {
        Some(false)
    } else {
        None
    };
    if let Some(little_endian) = endian {
        reader.consume(2);
        validate_utf16(reader, little_endian)
    } else {
        validate_utf8(reader)
    }
}

fn validate_utf8(mut reader: impl std::io::Read) -> anyhow::Result<bool> {
    let mut buffer = [0; 16_384];
    let mut carry = 0;
    loop {
        let count = reader.read(&mut buffer[carry..16_380 + carry])?;
        if count == 0 {
            return Ok(carry == 0);
        }
        let length = carry + count;
        let chunk = &buffer[..length];
        if chunk.contains(&0) {
            return Ok(false);
        }
        match std::str::from_utf8(chunk) {
            Ok(_) => carry = 0,
            Err(error) if error.error_len().is_none() => {
                let start = error.valid_up_to();
                carry = length - start;
                buffer.copy_within(start..length, 0);
            }
            Err(_) => return Ok(false),
        }
    }
}

fn validate_utf16(mut reader: impl std::io::Read, little_endian: bool) -> anyhow::Result<bool> {
    let mut buffer = [0; 16_384];
    let mut first_byte = None;
    let mut high_surrogate = false;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            return Ok(first_byte.is_none() && !high_surrogate);
        }
        for byte in &buffer[..count] {
            let Some(first) = first_byte.take() else {
                first_byte = Some(*byte);
                continue;
            };
            let unit = if little_endian {
                u16::from_le_bytes([first, *byte])
            } else {
                u16::from_be_bytes([first, *byte])
            };
            if unit == 0 {
                return Ok(false);
            }
            if high_surrogate {
                if !(0xdc00..=0xdfff).contains(&unit) {
                    return Ok(false);
                }
                high_surrogate = false;
            } else if (0xd800..=0xdbff).contains(&unit) {
                high_surrogate = true;
            } else if (0xdc00..=0xdfff).contains(&unit) {
                return Ok(false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    #[test]
    fn recognizes_decoded_images_and_text_with_misleading_extensions() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let image_path = dir.path().join("not-text.txt");
        let pixels = RgbImage::from_fn(24, 20, |x, y| Rgb([(x * 9) as u8, (y * 11) as u8, 90]));
        pixels.save_with_format(&image_path, image::ImageFormat::Png)?;
        let text_path = dir.path().join("not-image.png");
        std::fs::write(
            &text_path,
            "A valid text, despite its misleading extension. 😀",
        )?;
        assert_eq!(sniff(&image_path)?, Kind::Image);
        assert_eq!(sniff(&text_path)?, Kind::Text);
        Ok(())
    }

    #[test]
    fn validates_entire_utf8_and_bom_utf16_without_decoded_nuls() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("text");
        let mut bytes = vec![b'x'; 65_519];
        bytes.extend_from_slice("😀 after a buffer boundary".as_bytes());
        std::fs::write(&path, &bytes)?;
        assert!(is_text(&path)?);
        bytes.push(0xff);
        std::fs::write(&path, &bytes)?;
        assert!(!is_text(&path)?);
        for little_endian in [true, false] {
            let mut utf16 = if little_endian {
                vec![0xff, 0xfe]
            } else {
                vec![0xfe, 0xff]
            };
            for unit in "A UTF-16 sentence with 😀".encode_utf16() {
                utf16.extend_from_slice(&if little_endian {
                    unit.to_le_bytes()
                } else {
                    unit.to_be_bytes()
                });
            }
            std::fs::write(&path, &utf16)?;
            assert!(is_text(&path)?);
            utf16.extend_from_slice(&[0, 0]);
            std::fs::write(&path, &utf16)?;
            assert!(!is_text(&path)?);
        }
        std::fs::write(&path, b"not\0text")?;
        assert!(!is_text(&path)?);
        std::fs::write(&path, [0xff, 0xfe, 0x00, 0xd8])?;
        assert!(!is_text(&path)?);
        Ok(())
    }

    #[test]
    fn ffprobe_finds_video_in_an_unrecognized_container() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("movie.data");
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=48x32:rate=2",
                "-t",
                "1",
                "-c:v",
                "mpeg4",
                "-f",
                "nut",
            ])
            .arg(&path)
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "ffmpeg: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(infer::get(&std::fs::read(&path)?).is_none());
        assert_eq!(sniff(&path)?, Kind::Video);
        Ok(())
    }

    #[test]
    fn recognized_documents_and_archives_are_exact_only_even_with_old_cache() -> anyhow::Result<()>
    {
        let dir = tempfile::tempdir()?;
        let cache = FeatureCache::in_memory()?;
        for (id, bytes) in [
            b"%PDF-1.4\nA text-only PDF header\n%%EOF\n".as_slice(),
            b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n%%EOF\n".as_slice(),
            b"PK\x03\x04\0\0\0\0".as_slice(),
            br"{\rtf1\ansi A rich text document.}".as_slice(),
        ]
        .iter()
        .enumerate()
        {
            let path = dir.path().join(format!("{id}.mp4"));
            std::fs::write(&path, bytes)?;
            let mut files = [FileEntry {
                id: eidolon_core::FileId(id),
                path,
                size: bytes.len() as u64,
                mtime_ns: 0,
                kind: Kind::Other,
            }];
            // Scans must invalidate classifications made by the old fallback policy.
            cache.put(&files[0], "cli.kind/v1", &[1])?;
            classify(&mut files, &cache)?;
            assert_eq!(files[0].kind, Kind::Other, "fixture {id}");
            assert_eq!(sniff(&files[0].path)?, Kind::Other, "fixture {id}");
        }
        Ok(())
    }

    #[test]
    fn weak_application_magic_does_not_override_valid_text() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("manual.txt");
        let prose = b"BC(1) General Commands Manual\nNAME\nbc - arbitrary-precision calculator\n";
        std::fs::write(&path, prose)?;
        assert_eq!(
            infer::get(prose).map(|kind| kind.mime_type()),
            Some("application/x-llvm")
        );
        assert_eq!(sniff(&path)?, Kind::Text);
        let mut files = [FileEntry {
            id: eidolon_core::FileId(0),
            path,
            size: prose.len() as u64,
            mtime_ns: 0,
            kind: Kind::Other,
        }];
        let cache = FeatureCache::in_memory()?;
        cache.put(&files[0], "cli.kind/v2", &[3])?;
        classify(&mut files, &cache)?;
        assert_eq!(files[0].kind, Kind::Text);
        Ok(())
    }
}

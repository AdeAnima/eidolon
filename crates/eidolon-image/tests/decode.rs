//! Decoding checks use actual macOS desktop photographs, not hash constants.

use std::path::Path;
use std::process::Command;

use anyhow::{Context as _, ensure};
use image::GenericImageView as _;

#[test]
#[cfg(target_os = "macos")]
fn decodes_real_heic_and_matches_apple_dimensions() -> anyhow::Result<()> {
    let source = Path::new("/System/Library/Desktop Pictures/.thumbnails/Big Sur Coastline.heic");
    let temp = tempfile::Builder::new()
        .prefix("eidolon-decode-")
        .tempdir_in("/tmp")?;
    let reference = temp.path().join("reference.png");
    let output = Command::new("sips")
        .args(["-s", "format", "png"])
        .arg(source)
        .arg("--out")
        .arg(&reference)
        .output()
        .context("converting the real desktop photograph using sips")?;
    ensure!(
        output.status.success(),
        "sips: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let apple = image::open(reference)?;
    let decoded = eidolon_image::decode(source)?;
    assert_eq!(decoded.dimensions(), apple.dimensions());
    assert!(decoded.width() >= 100);
    Ok(())
}

#[test]
fn applies_jpeg_exif_orientation() -> anyhow::Result<()> {
    let temp = tempfile::Builder::new()
        .prefix("eidolon-orientation-")
        .tempdir_in("/tmp")?;
    let original = image::RgbImage::from_fn(80, 40, |x, _| {
        if x < 40 {
            image::Rgb([250, 10, 20])
        } else {
            image::Rgb([10, 20, 250])
        }
    });
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95).encode_image(&original)?;
    // APP1 EXIF, little-endian TIFF, one SHORT orientation entry (6 = rotate 90 CW).
    let exif = [
        0xff, 0xe1, 0, 34, b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12,
        1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    ];
    jpeg.splice(2..2, exif);
    let path = temp.path().join("rotated.jpg");
    std::fs::write(&path, jpeg)?;
    let decoded = eidolon_image::decode(&path)?.to_rgb8();
    assert_eq!(decoded.dimensions(), (40, 80));
    assert!(decoded.get_pixel(20, 10)[0] > 200);
    assert!(decoded.get_pixel(20, 70)[2] > 200);
    Ok(())
}

#[test]
fn applies_heif_orientation_once_when_container_and_exif_agree() -> anyhow::Result<()> {
    use libheif_rs::{
        Channel, ColorSpace, CompressionFormat, EncoderQuality, EncodingOptions, HeifContext,
        Image, ImageOrientation, LibHeif, RgbChroma,
    };

    let temp = tempfile::Builder::new()
        .prefix("eidolon-heif-orientation-")
        .tempdir_in("/tmp")?;
    let mut image = Image::new(80, 40, ColorSpace::Rgb(RgbChroma::Rgb))?;
    image.create_plane(Channel::Interleaved, 80, 40, 8)?;
    let planes = image.planes_mut();
    let plane = planes.interleaved.context("RGB plane missing")?;
    for y in 0..40 {
        for x in 0..80 {
            let index = y * plane.stride + x * 3;
            plane.data[index..index + 3].copy_from_slice(if x < 40 {
                &[250, 10, 20]
            } else {
                &[10, 20, 250]
            });
        }
    }
    for native_transform in [false, true] {
        let library = LibHeif::new();
        let mut encoder = library.encoder_for_format(CompressionFormat::Hevc)?;
        encoder.set_quality(EncoderQuality::Lossy(95))?;
        let mut context = HeifContext::new()?;
        let mut options = EncodingOptions::new()?;
        if native_transform {
            options.set_image_orientation(ImageOrientation::Rotate90Cw);
        }
        let handle = context.encode_image(&image, &mut encoder, Some(options))?;
        let exif = [
            b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ];
        context.add_exif_metadata(&handle, &exif)?;
        let path = temp.path().join(format!("native-{native_transform}.heic"));
        context.write_to_file(path.to_str().context("non-UTF8 test path")?)?;
        let decoded = eidolon_image::decode(&path)?.to_rgb8();
        assert_eq!(decoded.dimensions(), (40, 80), "native={native_transform}");
        assert!(
            decoded.get_pixel(20, 10)[0] > 200,
            "native={native_transform}"
        );
        assert!(
            decoded.get_pixel(20, 70)[2] > 200,
            "native={native_transform}"
        );
    }
    Ok(())
}

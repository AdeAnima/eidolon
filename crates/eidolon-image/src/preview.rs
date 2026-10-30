//! Reduced decoder inputs are retrieval signals, never canonical pixel evidence.

use std::fs::File;
use std::io::{BufReader, Read as _};
use std::path::Path;
use std::sync::LazyLock;

use anyhow::Context as _;
use image::{DynamicImage, RgbImage, metadata::Orientation};
use turbojpeg::{Colorspace, Decompressor, PixelFormat, ScalingFactor};

pub(crate) const LONG_SIDE: u32 = 800;
const ID: &str = "turbojpeg151-dct1248-imageio032-heif-srgb-fir610-bilinear800/v1";

pub(crate) fn identity() -> anyhow::Result<&'static str> {
    static IDENTITY: LazyLock<Result<String, String>> =
        LazyLock::new(|| decoder_identity().map_err(|error| format!("{error:#}")));
    IDENTITY
        .as_deref()
        .map_err(|error| anyhow::anyhow!("{error}"))
}

fn decoder_identity() -> anyhow::Result<String> {
    let identity = format!("{ID}/native-jpeg-{}", env!("EIDOLON_TURBOJPEG_VERSION"));
    #[cfg(target_os = "macos")]
    {
        let build = std::process::Command::new("/usr/bin/sw_vers")
            .arg("-buildVersion")
            .output()
            .context("reading the macOS build for ImageIO feature caching")?;
        anyhow::ensure!(
            build.status.success(),
            "reading the ImageIO macOS build failed"
        );
        let build = std::str::from_utf8(&build.stdout)?.trim();
        anyhow::ensure!(!build.is_empty(), "ImageIO macOS build was empty");
        Ok(format!("{identity}/macos-{build}"))
    }
    #[cfg(not(target_os = "macos"))]
    Ok(identity)
}

pub(crate) struct Decoded {
    pub image: DynamicImage,
    pub width: u32,
    pub height: u32,
    pub canonical: bool,
    pub profile: Option<moxcms::ColorProfile>,
}

pub(crate) fn decode(path: &Path, include_profile: bool) -> anyhow::Result<Decoded> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 12];
    let count = file.read(&mut magic)?;
    if count >= 2
        && magic[..2] == [0xff, 0xd8]
        && let Some(decoded) = jpeg(path)?
    {
        return Ok(decoded);
    }
    #[cfg(target_os = "macos")]
    if count >= 12
        && &magic[4..8] == b"ftyp"
        && matches!(
            &magic[8..12],
            b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1"
        )
    {
        return crate::imageio::decode(path);
    }
    let (image, profile) = crate::decode_with_profile(path, include_profile)?;
    Ok(Decoded {
        width: image.width(),
        height: image.height(),
        image,
        canonical: true,
        profile,
    })
}

fn jpeg(path: &Path) -> anyhow::Result<Option<Decoded>> {
    let bytes = std::fs::read(path)?;
    let mut decoder = Decompressor::new()?;
    let header = decoder.read_header(&bytes)?;
    if header.is_lossless || matches!(header.colorspace, Colorspace::CMYK | Colorspace::YCCK) {
        return Ok(None);
    }
    let Some(factor) = [
        ScalingFactor::ONE_EIGHTH,
        ScalingFactor::ONE_QUARTER,
        ScalingFactor::ONE_HALF,
    ]
    .into_iter()
    .find(|factor| factor.scale(header.width.max(header.height)) >= LONG_SIDE as usize) else {
        // Keep the canonical decoder when DCT scaling cannot save pixel work.
        return Ok(None);
    };
    let scaled = header.scaled(factor);
    let width = u32::try_from(header.width)?;
    let height = u32::try_from(header.height)?;
    let mut image = RgbImage::new(u32::try_from(scaled.width)?, u32::try_from(scaled.height)?);
    decoder.set_scaling_factor(factor)?;
    decoder.decompress(
        &bytes,
        turbojpeg::Image {
            pixels: image.as_mut(),
            width: scaled.width,
            pitch: scaled.width * 3,
            height: scaled.height,
            format: PixelFormat::RGB,
        },
    )?;
    let orientation = exif_orientation(path)?;
    let (width, height) = oriented_dimensions(width, height, orientation);
    let mut image = DynamicImage::ImageRgb8(image);
    image.apply_orientation(orientation);
    Ok(Some(Decoded {
        image,
        width,
        height,
        canonical: false,
        profile: None,
    }))
}

pub(crate) fn exif_orientation(path: &Path) -> anyhow::Result<Orientation> {
    let mut reader = BufReader::new(File::open(path)?);
    Ok(exif::Reader::new()
        .read_from_container(&mut reader)
        .ok()
        .and_then(|exif| {
            exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?
                .value
                .get_uint(0)
        })
        .and_then(|value| u8::try_from(value).ok())
        .and_then(Orientation::from_exif)
        .unwrap_or(Orientation::NoTransforms))
}

pub(crate) fn oriented_dimensions(width: u32, height: u32, orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (height, width),
        _ => (width, height),
    }
}

pub(crate) fn resize(image: &RgbImage, width: u32, height: u32) -> anyhow::Result<RgbImage> {
    let mut output = RgbImage::new(width, height);
    fast_image_resize::Resizer::new()
        .resize(
            image,
            &mut output,
            &fast_image_resize::ResizeOptions::new().resize_alg(
                fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear),
            ),
        )
        .context("resizing retrieval preview")?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use image::{GenericImageView as _, Rgb, RgbImage};

    #[test]
    fn decodes_large_jpeg_at_reduced_resolution_with_native_dimensions() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("photo.unknown");
        let image = RgbImage::from_fn(3200, 1600, |x, _| {
            if x < 1600 {
                Rgb([240, 10, 20])
            } else {
                Rgb([10, 20, 240])
            }
        });
        image.save_with_format(&path, image::ImageFormat::Jpeg)?;
        let decoded = super::decode(&path, false)?;
        assert_eq!((decoded.width, decoded.height), (3200, 1600));
        assert_eq!(decoded.image.dimensions(), (800, 400));
        assert!(!decoded.canonical);
        assert!(decoded.image.to_rgb8().get_pixel(100, 100)[0] > 200);
        assert!(decoded.image.to_rgb8().get_pixel(700, 100)[2] > 200);
        Ok(())
    }

    #[test]
    fn retains_canonical_decode_when_jpeg_scaling_cannot_save_pixels() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("small.jpg");
        RgbImage::from_pixel(960, 600, Rgb([10, 20, 240])).save(&path)?;
        let decoded = super::decode(&path, false)?;
        assert!(decoded.canonical);
        assert_eq!((decoded.width, decoded.height), (960, 600));
        assert_eq!(decoded.image.dimensions(), (960, 600));
        assert_eq!(decoded.image.to_rgb8(), crate::decode(&path)?.to_rgb8());
        Ok(())
    }

    #[test]
    fn reduced_jpeg_applies_exif_orientation_once() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("rotated.jpg");
        RgbImage::from_fn(3200, 1600, |x, y| {
            if y < 800 {
                Rgb([10, 240, 20])
            } else if x < 1600 {
                Rgb([240, 10, 20])
            } else {
                Rgb([10, 20, 240])
            }
        })
        .save(&path)?;
        let jpeg = std::fs::read(&path)?;
        let mut oriented = vec![0xff, 0xd8, 0xff, 0xe1, 0, 34];
        oriented.extend_from_slice(b"Exif\0\0");
        oriented.extend_from_slice(&[
            b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ]);
        oriented.extend_from_slice(&jpeg[2..]);
        std::fs::write(&path, oriented)?;
        let decoded = super::decode(&path, false)?;
        assert!(!decoded.canonical);
        assert_eq!((decoded.width, decoded.height), (1600, 3200));
        assert_eq!(decoded.image.dimensions(), (400, 800));
        let reference = crate::decode(&path)?.to_rgb8();
        let actual = decoded.image.to_rgb8();
        for (x, y) in [(1, 1), (1, 3), (3, 1), (3, 3)] {
            let pixel = actual.get_pixel(actual.width() * x / 4, actual.height() * y / 4);
            let expected =
                reference.get_pixel(reference.width() * x / 4, reference.height() * y / 4);
            assert!(
                pixel
                    .0
                    .into_iter()
                    .zip(expected.0)
                    .all(|(a, b)| a.abs_diff(b) < 3)
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn heif_preview_ignores_stale_thumbnail_and_applies_orientation_once() -> anyhow::Result<()> {
        use anyhow::Context as _;
        use libheif_rs::{
            Channel, ColorSpace, CompressionFormat, EncoderQuality, EncodingOptions, HeifContext,
            Image, ImageOrientation, LibHeif, RgbChroma,
        };

        let temp = tempfile::tempdir()?;
        let make_image = |width, height, stale| -> anyhow::Result<Image> {
            let mut image = Image::new(width, height, ColorSpace::Rgb(RgbChroma::Rgb))?;
            image.create_plane(Channel::Interleaved, width, height, 8)?;
            let mut planes = image.planes_mut();
            let plane = planes.interleaved.as_mut().context("RGB plane missing")?;
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let offset = y * plane.stride + x * 3;
                    let color = if stale {
                        [240, 10, 20]
                    } else if y < height as usize / 2 {
                        [10, 240, 20]
                    } else if x < width as usize / 2 {
                        [240, 10, 20]
                    } else {
                        [10, 20, 240]
                    };
                    plane.data[offset..offset + 3].copy_from_slice(&color);
                }
            }
            Ok(image)
        };
        let primary = make_image(1600, 1000, false)?;
        let thumbnail = make_image(160, 100, true)?;
        for (native, exif) in [(false, 0), (false, 6), (false, 8), (true, 6), (true, 8)] {
            let library = LibHeif::new();
            let mut encoder = library.encoder_for_format(CompressionFormat::Hevc)?;
            encoder.set_quality(EncoderQuality::Lossy(95))?;
            let mut context = HeifContext::new()?;
            let mut options = EncodingOptions::new()?;
            if native {
                options.set_image_orientation(ImageOrientation::Rotate90Cw);
            }
            let mut handle = context.encode_image(&primary, &mut encoder, Some(options))?;
            context.encode_thumbnail(&thumbnail, &handle, 80, &mut encoder, None)?;
            context.set_primary_image(&mut handle)?;
            if exif != 0 {
                context.add_exif_metadata(
                    &handle,
                    &[
                        b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, exif, 0, 0,
                        0, 0, 0, 0, 0,
                    ],
                )?;
            }
            let path = temp.path().join(format!("primary-{native}-{exif}.unknown"));
            context.write_to_file(path.to_str().context("non-UTF8 test path")?)?;
            let reduced = super::decode(&path, false)?;
            assert!(!reduced.canonical, "must use the bounded ImageIO path");
            let expected = if native || exif != 0 {
                (1000, 1600)
            } else {
                (1600, 1000)
            };
            assert_eq!((reduced.width, reduced.height), expected);
            assert_eq!(
                reduced.image.dimensions(),
                if native || exif != 0 {
                    (500, 800)
                } else {
                    (800, 500)
                }
            );
            let reference = crate::decode(&path)?.to_rgb8();
            let actual = reduced.image.to_rgb8();
            for (x, y) in [(1, 1), (1, 3), (3, 1), (3, 3)] {
                let pixel = actual.get_pixel(actual.width() * x / 4, actual.height() * y / 4);
                let expected =
                    reference.get_pixel(reference.width() * x / 4, reference.height() * y / 4);
                assert!(
                    pixel
                        .0
                        .iter()
                        .zip(expected.0)
                        .all(|(a, b)| a.abs_diff(b) < 12),
                    "native={native}, exif={exif}, ({x},{y}): {pixel:?} != {expected:?}"
                );
            }
        }
        Ok(())
    }
}

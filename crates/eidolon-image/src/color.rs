//! Explicit ICC conversion supplies an alternate proof, never a colour-shift exemption.

use std::fs::File;
use std::io::{BufReader, Read as _, Seek as _};
use std::path::Path;

use anyhow::Context as _;
use image::{ImageDecoder, RgbImage, imageops::FilterType};
use libheif_rs::{HeifContext, StreamReader};
use moxcms::{ColorProfile, DataColorSpace, Layout, RenderingIntent, TransformOptions};

use crate::features::PREVIEW;

pub(crate) fn profile(
    path: &Path,
    decoder: &mut dyn ImageDecoder,
) -> anyhow::Result<Option<ColorProfile>> {
    if let Some(icc) = decoder.icc_profile()? {
        return ColorProfile::new_from_slice(&icc)
            .context("parsing embedded ICC profile")
            .map(Some);
    }
    // The libheif image hook omits ICC metadata. Read its primary handle
    // without loading or decoding the image payload a second time.
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic)?;
    if &magic[4..] != b"ftyp" {
        return Ok(None);
    }
    let size = file.metadata()?.len();
    file.rewind()?;
    let reader = StreamReader::new(BufReader::new(file), size);
    let context = HeifContext::read_from_reader(Box::new(reader))?;
    context
        .primary_image_handle()?
        .color_profile_raw()
        .map(|icc| ColorProfile::new_from_slice(&icc.data).context("parsing HEIF ICC profile"))
        .transpose()
}

pub(crate) fn to_srgb(image: &RgbImage, profile: &ColorProfile) -> anyhow::Result<Option<Vec<u8>>> {
    // A decoder has already rendered CMYK/gray input into RGB; its original
    // non-RGB profile cannot be applied to those RGB samples a second time.
    if profile.color_space != DataColorSpace::Rgb {
        return Ok(None);
    }
    let transform = profile
        .create_transform_8bit(
            Layout::Rgb,
            &ColorProfile::new_srgb(),
            Layout::Rgb,
            TransformOptions {
                rendering_intent: RenderingIntent::RelativeColorimetric,
                ..TransformOptions::default()
            },
        )
        .context("creating ICC-to-sRGB transform")?;
    let mut converted = vec![0; image.as_raw().len()];
    transform
        .transform(image.as_raw(), &mut converted)
        .context("converting ICC RGB samples to sRGB")?;
    let converted = RgbImage::from_raw(image.width(), image.height(), converted)
        .context("invalid ICC RGB pixel layout")?;
    // Convert before preview resampling: gamut clipping and averaging do not commute.
    Ok(Some(
        image::imageops::resize(&converted, PREVIEW, PREVIEW, FilterType::Triangle).into_raw(),
    ))
}

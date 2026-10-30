//! macOS HEIF retrieval thumbnails. Canonical evidence still uses libheif.
#![allow(unsafe_code)] // ImageIO's typed-dictionary FFI has no safe thumbnail wrapper.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context as _, ensure};
use image::{DynamicImage, RgbImage, metadata::Orientation};
use libheif_rs::{HeifContext, StreamReader};
use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFString, CFType, CFURL, CGRect};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, kCGColorSpaceSRGB,
};
use objc2_image_io::{
    CGImageSource, kCGImageSourceCreateThumbnailFromImageAlways,
    kCGImageSourceCreateThumbnailWithTransform, kCGImageSourceShouldCache,
    kCGImageSourceThumbnailMaxPixelSize,
};

use crate::preview::{Decoded, LONG_SIDE};

pub(crate) fn decode(path: &Path) -> anyhow::Result<Decoded> {
    // Header-only primary-handle dimensions match the canonical libheif hook.
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let reader = StreamReader::new(BufReader::new(file), size);
    let context = HeifContext::read_from_reader(Box::new(reader))?;
    let handle = context.primary_image_handle()?;
    let native = crate::heif::has_native_orientation(path)?;
    let orientation = if native {
        Orientation::NoTransforms
    } else {
        crate::preview::exif_orientation(path)?
    };
    let (width, height) =
        crate::preview::oriented_dimensions(handle.width(), handle.height(), orientation);
    ensure!(width > 0 && height > 0, "empty HEIF primary image");
    let url = CFURL::from_file_path(path).context("invalid HEIF path")?;
    let limit = CFNumber::new_i32(i32::try_from(LONG_SIDE)?);
    // SAFETY: These immutable framework constants are valid CFString objects.
    let keys = unsafe {
        [
            kCGImageSourceCreateThumbnailFromImageAlways,
            kCGImageSourceCreateThumbnailWithTransform,
            kCGImageSourceThumbnailMaxPixelSize,
            kCGImageSourceShouldCache,
        ]
    };
    let options = CFDictionary::<CFString, CFType>::from_slices(
        &keys,
        &[
            CFBoolean::new(true).as_ref(),
            CFBoolean::new(native).as_ref(),
            AsRef::<CFType>::as_ref(&*limit),
            CFBoolean::new(false).as_ref(),
        ],
    );
    // SAFETY: URL is retained; no untyped options are passed to source creation.
    let source = unsafe { CGImageSource::with_url(&url, None) }
        .context("opening HEIF primary image with ImageIO")?;
    // SAFETY: This is a live source. ImageIO selects the primary item, not index zero
    // (which can identify a thumbnail in multi-item HEIF containers).
    let index = unsafe { source.primary_image_index() };
    // SAFETY: Keys are the documented ImageIO options and values are the required
    // CFBoolean/CFNumber types. FromImageAlways explicitly ignores embedded thumbnails.
    let thumbnail = unsafe { source.thumbnail_at_index(index, Some(options.as_opaque())) }
        .context("decoding HEIF primary-image thumbnail with ImageIO")?;
    let pixels = rgb(&thumbnail)?;
    ensure!(
        pixels.width().max(pixels.height()) <= LONG_SIDE,
        "unbounded ImageIO thumbnail"
    );
    let mut image = DynamicImage::ImageRgb8(pixels);
    // ImageIO does not expose all HEIF EXIF orientations. Disable its EXIF
    // transform when no native irot/imir exists, then apply our canonical fallback once.
    image.apply_orientation(orientation);
    Ok(Decoded {
        image,
        width,
        height,
        canonical: false,
        profile: None,
    })
}

fn rgb(image: &CGImage) -> anyhow::Result<RgbImage> {
    let width = CGImage::width(Some(image));
    let height = CGImage::height(Some(image));
    ensure!(
        width > 0 && height > 0 && width.max(height) <= LONG_SIDE as usize,
        "invalid ImageIO thumbnail dimensions"
    );
    // SAFETY: Immutable framework constant is a valid retained CFString.
    let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
        .context("creating sRGB retrieval colour space")?;
    // Opaque RGBX destination starts white, preserving visible alpha compositing.
    let mut pixels = vec![255_u8; width * height * 4];
    {
        // SAFETY: Buffer has exactly height rows of width*4 bytes, is suitably aligned
        // for 8-bit components and is not accessed or resized while context lives.
        // Context is dropped before the buffer is compacted or returned.
        let context = unsafe {
            CGBitmapContextCreate(
                pixels.as_mut_ptr().cast(),
                width,
                height,
                8,
                width * 4,
                Some(&space),
                CGImageAlphaInfo::NoneSkipLast.0 | CGImageByteOrderInfo::Order32Big.0,
            )
        }
        .context("creating sRGB retrieval bitmap")?;
        CGContext::draw_image(
            Some(&context),
            CGRect::new(
                objc2_core_foundation::CGPoint::ZERO,
                objc2_core_foundation::CGSize::new(width as f64, height as f64),
            ),
            Some(image),
        );
    }
    for index in 0..width * height {
        pixels.copy_within(index * 4..index * 4 + 3, index * 3);
    }
    pixels.truncate(width * height * 3);
    RgbImage::from_raw(u32::try_from(width)?, u32::try_from(height)?, pixels)
        .context("invalid sRGB retrieval pixel layout")
}

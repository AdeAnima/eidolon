//! Registration certifies shared two-dimensional composition, not a shared palette.

use std::io::Cursor;

use eidolon_eval::registration::compare;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage, imageops::FilterType};

fn decoded(image: &DynamicImage) -> anyhow::Result<DynamicImage> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png)?;
    Ok(image::load_from_memory(bytes.get_ref())?)
}

#[test]
fn registers_reencoded_recoloured_composition_but_not_nonrigid_rearrangement() -> anyhow::Result<()>
{
    let original = DynamicImage::ImageRgb8(RgbImage::from_fn(240, 180, |x, y| {
        let value = if ((x / 23 + y / 17) % 3 == 0) ^ (x.abs_diff(115) + y.abs_diff(83) < 57) {
            160 + (x % 23) as u8
        } else {
            35 + (y % 31) as u8
        };
        Rgb([value, value / 2 + 20, value / 3 + 25])
    }));
    let recoloured = DynamicImage::ImageRgb8(RgbImage::from_fn(240, 180, |x, y| {
        let pixel = original.as_rgb8().expect("RGB source").get_pixel(x, y);
        Rgb([pixel[0] / 2 + 20, pixel[1] / 2 + 40, pixel[2] / 2 + 60])
    }))
    .resize_exact(160, 120, FilterType::Triangle);
    let rearranged = DynamicImage::ImageRgb8(RgbImage::from_fn(240, 180, |x, y| {
        let offset = if y < 60 {
            47
        } else if y < 120 {
            7
        } else {
            103
        };
        *original
            .as_rgb8()
            .expect("RGB source")
            .get_pixel((x + offset) % 240, y)
    }));
    let original = decoded(&original)?;
    let registered = compare(&original, &decoded(&recoloured)?);
    assert!(
        registered.accepted,
        "same composition rejected: {registered:?}"
    );
    let unproven = compare(&original, &decoded(&rearranged)?);
    assert!(
        !unproven.accepted,
        "nonrigid layout falsely certified: {unproven:?}"
    );
    Ok(())
}

#[test]
fn finds_translation_and_rejects_featureless_colours() -> anyhow::Result<()> {
    let original = DynamicImage::ImageRgb8(RgbImage::from_fn(256, 192, |x, y| {
        let value = if (x / 21 + y / 17) % 3 == 0 { 190 } else { 45 };
        Rgb([
            value,
            value / 2 + (y % 37) as u8,
            value / 3 + (x % 41) as u8,
        ])
    }));
    let translated = DynamicImage::ImageRgb8(RgbImage::from_fn(256, 192, |x, y| {
        if x < 24 {
            Rgb([30, 30, 30])
        } else {
            *original.as_rgb8().expect("RGB source").get_pixel(x - 24, y)
        }
    }));
    let proof = compare(&decoded(&original)?, &decoded(&translated)?);
    assert!(proof.accepted, "translation not registered: {proof:?}");
    assert!(
        proof.dx.abs() > 4.0,
        "identity transform cannot explain this input"
    );
    let plain_a = DynamicImage::ImageRgb8(RgbImage::from_pixel(256, 192, Rgb([30, 80, 190])));
    let plain_b = DynamicImage::ImageRgb8(RgbImage::from_pixel(256, 192, Rgb([190, 80, 30])));
    assert!(!compare(&decoded(&plain_a)?, &decoded(&plain_b)?).accepted);
    Ok(())
}

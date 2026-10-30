//! Image-side caption masks; excluded pixels never contribute to spatial proof.

use serde::{Deserialize, Serialize};

/// A full-width, text-bearing band in the content-normalized 32px image grid.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(crate) struct Band {
    top: u8,
    bottom: u8,
}

impl Band {
    /// Find a uniform backing with repeated high-contrast, near-neutral strokes.
    ///
    /// Backing uniformity and horizontal edge runs follow the inexpensive text-band
    /// localization approach in Lienhart and Wernicke (2002). No OCR, model,
    /// platform service, or particular caption color/string is required.
    pub(crate) fn detect(
        rgb: &[[u8; 3]],
        (left, top, right, bottom): (u32, u32, u32, u32),
    ) -> Option<Self> {
        let width = right - left;
        let height = bottom - top;
        let rows: [Option<[u8; 3]>; 96] = std::array::from_fn(|offset| {
            let y = top + offset as u32;
            if y >= bottom {
                return None;
            }
            let a = rgb[(y * 96 + left) as usize];
            let b = rgb[(y * 96 + right - 1) as usize];
            if a.into_iter().zip(b).any(|(a, b)| a.abs_diff(b) > 12) {
                return None;
            }
            let background = std::array::from_fn(|channel| u8::midpoint(a[channel], b[channel]));
            let uniform = (left..right)
                .filter(|x| {
                    rgb[(y * 96 + x) as usize]
                        .into_iter()
                        .zip(background)
                        .all(|(a, b)| a.abs_diff(b) <= 12)
                })
                .count();
            (uniform * 10 >= width as usize).then_some(background)
        });
        let mut best = None;
        let mut start = 0;
        while start < rows.len() {
            let Some(background) = rows[start] else {
                start += 1;
                continue;
            };
            let mut end = start + 1;
            while end < rows.len()
                && rows[end].is_some_and(|row| {
                    row.into_iter()
                        .zip(background)
                        .all(|(a, b)| a.abs_diff(b) <= 12)
                })
            {
                end += 1;
            }
            if end - start >= 3 && (end - start) * 4 <= height as usize {
                let mut edges = 0;
                let mut strokes = 0;
                let mut text_rows = 0;
                let mut first = right;
                let mut last = left;
                for y in top + start as u32..top + end as u32 {
                    let mut previous = false;
                    let mut row_edges = 0;
                    for x in left..right {
                        let pixel = rgb[(y * 96 + x) as usize];
                        let low = *pixel.iter().min()?;
                        let high = *pixel.iter().max()?;
                        let neutral = high - low <= 40 && (low >= 110 || high <= 64);
                        let contrast = pixel
                            .into_iter()
                            .zip(background)
                            .any(|(a, b)| a.abs_diff(b) >= 64);
                        let stroke = neutral && contrast;
                        if stroke {
                            strokes += 1;
                            first = first.min(x);
                            last = last.max(x);
                            row_edges += usize::from(!previous);
                        }
                        previous = stroke;
                    }
                    edges += row_edges;
                    text_rows += usize::from(row_edges >= 4);
                }
                let area = width as usize * (end - start);
                if text_rows >= 2
                    && strokes >= area / 100
                    && strokes * 2 <= area
                    && last.saturating_sub(first) >= width / 3
                {
                    // Include the resampling halo, never synthetic replacement pixels.
                    let band = Self {
                        top: ((start * 32 / height as usize).saturating_sub(1)) as u8,
                        bottom: ((end * 32).div_ceil(height as usize) + 1).min(32) as u8,
                    };
                    if band.unmasked_fraction() >= 0.7
                        && best.is_none_or(|(_, previous)| edges > previous)
                    {
                        best = Some((band, edges));
                    }
                }
            }
            start = end;
        }
        best.map(|(band, _)| band)
    }

    /// Locate an abrupt, majority-width solid source occluder independently of
    /// the query. Ordinary texture or a gradual background is not evidence.
    pub(crate) fn solid(
        rgb: &[[u8; 3]],
        (left, top, right, bottom): (u32, u32, u32, u32),
        side: u32,
    ) -> Option<Self> {
        let width = right - left;
        let height = bottom - top;
        let key = |p: [u8; 3]| {
            usize::from(p[0] >> 6) * 16 + usize::from(p[1] >> 6) * 4 + usize::from(p[2] >> 6)
        };
        let rows: [Option<[u8; 3]>; 96] = std::array::from_fn(|offset| {
            let y = top + offset as u32;
            if y >= bottom {
                return None;
            }
            let mut bins = [0_u16; 64];
            for x in left..right {
                bins[key(rgb[(y * side + x) as usize])] += 1;
            }
            let (mode, count) = bins.iter().enumerate().max_by_key(|(_, count)| *count)?;
            if u32::from(*count) * 5 < width * 3 {
                return None;
            }
            let mut sum = [0_u32; 3];
            for x in left..right {
                let pixel = rgb[(y * side + x) as usize];
                if key(pixel) == mode {
                    for channel in 0..3 {
                        sum[channel] += u32::from(pixel[channel]);
                    }
                }
            }
            Some(sum.map(|value| (value / u32::from(*count)) as u8))
        });
        let mut best = None;
        let mut start = 0;
        while start < height as usize {
            let Some(color) = rows[start] else {
                start += 1;
                continue;
            };
            let mut end = start + 1;
            while end < height as usize
                && rows[end]
                    .is_some_and(|row| row.into_iter().zip(color).all(|(a, b)| a.abs_diff(b) <= 32))
            {
                end += 1;
            }
            if end - start >= 3 && (end - start) * 4 <= height as usize {
                let mut contrast = u32::MAX;
                for row in [
                    (start > 0).then_some(start.saturating_sub(2)),
                    (end < height as usize).then_some((end + 1).min(height as usize - 1)),
                ]
                .into_iter()
                .flatten()
                {
                    let mut mean = [0_u32; 3];
                    for x in left..right {
                        for (sum, value) in mean
                            .iter_mut()
                            .zip(rgb[((top + row as u32) * side + x) as usize])
                        {
                            *sum += u32::from(value);
                        }
                    }
                    let edge = mean
                        .into_iter()
                        .zip(color)
                        .map(|(sum, value)| (sum / width).abs_diff(u32::from(value)))
                        .max()
                        .unwrap_or(0);
                    contrast = contrast.min(edge);
                }
                let band = Self {
                    top: (start * 32 / height as usize).saturating_sub(1) as u8,
                    bottom: ((end * 32).div_ceil(height as usize) + 1).min(32) as u8,
                };
                if contrast >= 48
                    && band.unmasked_fraction() >= 0.7
                    && best.is_none_or(|(_, previous)| contrast > previous)
                {
                    best = Some((band, contrast));
                }
            }
            start = end;
        }
        best.map(|(band, _)| band)
    }

    /// Whether the entire sampled vertical footprint excludes the detected band.
    pub(crate) fn valid(self, y: f64, height: f64) -> bool {
        y + height <= f64::from(self.top) || y >= f64::from(self.bottom)
    }

    pub(crate) fn unmasked_fraction(self) -> f64 {
        1.0 - f64::from(self.bottom - self.top) / 32.0
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::Band;

    fn background() -> RgbImage {
        RgbImage::from_fn(96, 96, |x, y| {
            Rgb([
                (48 + (x * 17 + y * 3) % 144) as u8,
                (64 + (x * 5 + y * 11) % 128) as u8,
                (32 + (x * 7 + y * 13) % 128) as u8,
            ])
        })
    }

    fn caption(image: &mut RgbImage, top: u32, height: u32) {
        for y in top..top + height {
            for x in 0..96 {
                image.put_pixel(x, y, Rgb([20, 24, 28]));
            }
        }
        for glyph in 0..12 {
            for y in top + 3..top + height - 2 {
                for x in 3 + glyph * 7..6 + glyph * 7 {
                    image.put_pixel(x, y, Rgb([238, 238, 238]));
                }
            }
        }
    }

    #[test]
    fn masks_text_bands_at_any_vertical_position() {
        for top in [0, 42, 84] {
            let mut image = background();
            caption(&mut image, top, 12);
            let mask = Band::detect(image.as_raw().as_chunks::<3>().0, (0, 0, 96, 96))
                .expect("caption band");
            assert!(!mask.valid(f64::from(top) / 3.0, 4.0));
            assert!(mask.unmasked_fraction() >= 0.7);
            let other = if top == 0 { 28.0 } else { 0.0 };
            assert!(mask.valid(other, 1.0));
        }
    }

    #[test]
    fn does_not_mask_plain_padding_texture_or_an_excessive_caption() {
        let image = background();
        assert!(Band::detect(image.as_raw().as_chunks::<3>().0, (0, 0, 96, 96)).is_none());
        let mut padding = image.clone();
        for y in 84..96 {
            for x in 0..96 {
                padding.put_pixel(x, y, Rgb([20, 24, 28]));
            }
        }
        assert!(Band::detect(padding.as_raw().as_chunks::<3>().0, (0, 0, 96, 96)).is_none());
        let mut excessive = image;
        caption(&mut excessive, 48, 48);
        assert!(Band::detect(excessive.as_raw().as_chunks::<3>().0, (0, 0, 96, 96)).is_none());
    }

    #[test]
    fn localizes_a_uniform_source_occluder_without_using_query_disagreement() {
        let mut image = background();
        assert!(Band::solid(image.as_raw().as_chunks::<3>().0, (0, 0, 96, 96), 96).is_none());
        for y in 72..86 {
            for x in 10..86 {
                image.put_pixel(x, y, Rgb([224, 24, 232]));
            }
        }
        let mask = Band::solid(image.as_raw().as_chunks::<3>().0, (0, 0, 96, 96), 96)
            .expect("abrupt, majority-width solid source band");
        assert!(!mask.valid(24.0, 1.0));
        assert!(mask.valid(20.0, 1.0));
        assert!(mask.unmasked_fraction() >= 0.7);
        let preview =
            image::imageops::resize(&image, 32, 32, image::imageops::FilterType::Triangle);
        let coarse = Band::solid(preview.as_raw().as_chunks::<3>().0, (0, 0, 32, 32), 32)
            .expect("cached RGB nomination also excludes the source band");
        assert!(!coarse.valid(24.0, 1.0));
        assert!(coarse.unmasked_fraction() >= 0.7);
    }
}

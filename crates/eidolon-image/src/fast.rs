//! Cheap full-frame proofs avoid unnecessary ORB and phase work, not verification.

use crate::features::Visual;
use crate::geometry::IDENTITY;

use super::{Pixels, Placement, confidence, decoded_equal, needs_crop_registration, photometry};

pub(super) fn full_frame(
    a: &Visual,
    b: &Visual,
    hash_distance: u32,
    min_score: f64,
) -> Option<(Placement, Pixels)> {
    let pixels = photometry::<4>(a, b, &IDENTITY);
    let local = pixels.supported() && pixels.localized() && !needs_crop_registration(a, b, &pixels);
    if !pixels.strict_full(hash_distance) && !local {
        return None;
    }
    if confidence(decoded_equal(a, b), None, &pixels, None) < min_score {
        // ORB confirmation can still lift the score; do not lose that candidate.
        return None;
    }
    Some((
        Placement {
            transform: IDENTITY,
            reverse_crop: false,
            crop_area: None,
        },
        pixels,
    ))
}

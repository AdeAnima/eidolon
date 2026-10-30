use std::collections::HashSet;

use opencv::{core, geometry, prelude::*};

use crate::features::Orb;

pub(crate) const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub(crate) struct Alignment {
    pub transform: [f64; 9],
    pub inliers: usize,
    pub ratio: f64,
    pub coverage: f64,
}

pub(crate) fn verify(a: &Orb, b: &Orb, fast: bool) -> anyhow::Result<Option<Alignment>> {
    if a.points.len() < 12 || b.points.len() < 12 {
        return Ok(None);
    }
    let matches = if fast {
        crate::matching::mutual_fast(a, b)?
    } else {
        crate::matching::mutual(a, b)?
    };
    if matches.len() < 12 {
        return Ok(None);
    }
    let mut source = core::Vector::<core::Point2f>::new();
    let mut target = core::Vector::<core::Point2f>::new();
    for &(first, second) in &matches {
        let first = a.points[first];
        let second = b.points[second];
        source.push(core::Point2f::new(first[0] * 800.0, first[1] * 800.0));
        target.push(core::Point2f::new(second[0] * 800.0, second[1] * 800.0));
    }
    let mut mask = core::Mat::default();
    let matrix = geometry::find_homography(
        &source,
        &target,
        geometry::RANSAC,
        3.5,
        &mut mask,
        2000,
        0.995,
    )?;
    if matrix.empty() {
        return Ok(None);
    }
    let mask = mask.data_bytes()?;
    let inliers = mask.iter().filter(|value| **value != 0).count();
    let ratio = inliers as f64 / matches.len() as f64;
    if inliers < 12 || ratio < 0.45 {
        return Ok(None);
    }
    let mut a_cells = HashSet::new();
    let mut b_cells = HashSet::new();
    for (&(first, second), inlier) in matches.iter().zip(mask) {
        if *inlier == 0 {
            continue;
        }
        let a_point = a.points[first];
        let b_point = b.points[second];
        a_cells.insert(((a_point[0] * 4.0) as u8, (a_point[1] * 4.0) as u8));
        b_cells.insert(((b_point[0] * 4.0) as u8, (b_point[1] * 4.0) as u8));
    }
    let coverage = a_cells.len().min(b_cells.len()) as f64 / 16.0;
    // A shared toolbar or repeated icon must not establish a whole-page match.
    if coverage < 0.25 {
        return Ok(None);
    }
    let values = matrix.data_typed::<f64>()?;
    let transform = [
        values[0],
        values[1],
        values[2] / 800.0,
        values[3],
        values[4],
        values[5] / 800.0,
        values[6] * 800.0,
        values[7] * 800.0,
        values[8],
    ];
    if transform.iter().any(|value| !value.is_finite()) || inverse(&transform).is_none() {
        return Ok(None);
    }
    // An unusable fit must not veto an independent crop-registration attempt.
    if !full_frame(&transform)
        && contained_area(&transform).is_none()
        && inverse(&transform).is_none_or(|inverse| contained_area(&inverse).is_none())
        && !reframed_windows(&transform)
    {
        return Ok(None);
    }
    Ok(Some(Alignment {
        transform,
        inliers,
        ratio,
        coverage,
    }))
}

pub(crate) fn project(h: &[f64; 9], x: f64, y: f64) -> Option<[f64; 2]> {
    let denominator = h[6] * x + h[7] * y + h[8];
    if denominator.abs() < 1e-9 {
        return None;
    }
    let point = [
        (h[0] * x + h[1] * y + h[2]) / denominator,
        (h[3] * x + h[4] * y + h[5]) / denominator,
    ];
    point.iter().all(|value| value.is_finite()).then_some(point)
}

pub(crate) fn inverse(h: &[f64; 9]) -> Option<[f64; 9]> {
    let a = h[4] * h[8] - h[5] * h[7];
    let b = h[2] * h[7] - h[1] * h[8];
    let c = h[1] * h[5] - h[2] * h[4];
    let determinant = h[0] * a + h[3] * b + h[6] * c;
    if determinant.abs() < 1e-10 {
        return None;
    }
    Some(
        [
            a,
            b,
            c,
            h[5] * h[6] - h[3] * h[8],
            h[0] * h[8] - h[2] * h[6],
            h[2] * h[3] - h[0] * h[5],
            h[3] * h[7] - h[4] * h[6],
            h[1] * h[6] - h[0] * h[7],
            h[0] * h[4] - h[1] * h[3],
        ]
        .map(|value| value / determinant),
    )
}

pub(crate) fn full_frame(h: &[f64; 9]) -> bool {
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        .iter()
        .all(|&(x, y)| {
            project(h, x, y)
                .is_some_and(|point| (point[0] - x).abs().max((point[1] - y).abs()) <= 0.015)
        })
}

pub(crate) fn contained_area(h: &[f64; 9]) -> Option<f64> {
    let mut points = [[0.0; 2]; 4];
    for (output, (x, y)) in points
        .iter_mut()
        .zip([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    {
        *output = project(h, x, y)?;
        if output.iter().any(|value| !(-0.035..=1.035).contains(value)) {
            return None;
        }
    }
    let mut signed_crosses = [0.0; 4];
    let mut area = 0.0;
    for index in 0..4 {
        let first = points[index];
        let second = points[(index + 1) % 4];
        let third = points[(index + 2) % 4];
        signed_crosses[index] = (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0]);
        area += first[0] * second[1] - second[0] * first[1];
    }
    if signed_crosses.iter().any(|cross| *cross <= 0.0) {
        return None;
    }
    let area = area.abs() / 2.0;
    (0.025..0.97).contains(&area).then_some(area)
}

/// Whether both frames contain substantial shared scene and their own new scene.
pub(crate) fn reframed_windows(transform: &[f64; 9]) -> bool {
    let Some(inverse) = inverse(transform) else {
        return false;
    };
    [transform, &inverse].into_iter().all(|matrix| {
        let shared = (0..16)
            .flat_map(|y| (0..16).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                project(
                    matrix,
                    (f64::from(x) + 0.5) / 16.0,
                    (f64::from(y) + 0.5) / 16.0,
                )
                .is_some_and(|point| point.iter().all(|value| (0.0..=1.0).contains(value)))
            })
            .count();
        // At least half the scene must be shared in each direction. Requiring
        // new scene in both frames excludes full-frame and ordinary containment.
        (128..=240).contains(&shared)
    })
}

//! Local RGB agreement must cover informative content, not just a shared frame.

#[derive(Clone, Copy, Default)]
struct Cell {
    count: u32,
    sums: [[f64; 3]; 2],
    squares: [f64; 2],
    residual_squared: f64,
}

impl Cell {
    fn observe(&mut self, source: [f64; 3], target: [f64; 3]) {
        self.count += 1;
        for channel in 0..3 {
            self.sums[0][channel] += source[channel];
            self.sums[1][channel] += target[channel];
            self.squares[0] += source[channel] * source[channel];
            self.squares[1] += target[channel] * target[channel];
            self.residual_squared += (source[channel] - target[channel]).powi(2);
        }
    }

    fn agreement(&self) -> Option<bool> {
        let count = f64::from(self.count.max(1));
        let variance: [f64; 2] = std::array::from_fn(|image| {
            (self.squares[image]
                - self.sums[image].iter().map(|sum| sum * sum).sum::<f64>() / count)
                .max(0.0)
        });
        // A blank cell in either image cannot erase the other's foreground.
        if variance[0].max(variance[1]) < 3.0 * count * 0.02_f64.powi(2) {
            return None;
        }
        let bias_squared = (0..3)
            .map(|channel| (self.sums[0][channel] - self.sums[1][channel]).powi(2))
            .sum::<f64>()
            / count;
        let covariance =
            (variance[0] + variance[1] - (self.residual_squared - bias_squared).max(0.0)) / 2.0;
        let denominator = (variance[0] * variance[1]).sqrt();
        Some(denominator > f64::EPSILON && covariance >= 0.85 * denominator)
    }
}

/// Run-local moments for the central 75% of an aligned comparison.
pub(super) struct Content {
    cells: [Cell; 36],
}

impl Content {
    /// Accumulate bounded moments without allocating another preview.
    pub fn new() -> Self {
        Self {
            cells: [Cell::default(); 36],
        }
    }

    /// Observe the same aligned RGB samples used by the photometric verifier.
    pub fn observe(&mut self, x: usize, y: usize, source: [f64; 3], target: [f64; 3]) {
        let (column, row) = (x / 16, y / 16);
        if (1..7).contains(&column) && (1..7).contains(&row) {
            self.cells[(row - 1) * 6 + column - 1].observe(source, target);
        }
    }

    /// Ignore cells flat in both images; retain all changed informative cells.
    pub fn finish(&self) -> Support {
        let mut support = Support {
            regions: 0,
            informative: 0,
        };
        for (index, cell) in self.cells.iter().enumerate() {
            if let Some(agreed) = cell.agreement() {
                support.informative += 1;
                if agreed {
                    support.regions |= 1 << index;
                }
            }
        }
        support
    }
}

/// Spatial support independently of global RGB error or the overlay mask.
#[derive(Debug)]
pub(super) struct Support {
    /// Agreed cells in the six-by-six interior grid, in row-major order.
    pub regions: u64,
    /// Cells with nontrivial contrast in at least one image.
    pub informative: u32,
}

impl Support {
    /// Require half the informative interior cells, rounded down, and spatial spread.
    pub fn supported(&self) -> bool {
        let agreed = self.regions.count_ones();
        agreed >= 4
            && agreed >= self.informative / 2
            && (0..6)
                .filter(|row| self.regions & (0x3F << (row * 6)) != 0)
                .count()
                >= 2
            && (0..6)
                .filter(|column| self.regions & (0x4104_1041 << column) != 0)
                .count()
                >= 2
    }
}

//! Matrices kept on the NPU.

use crate::device::Npu;
use crate::error::{Error, Result};
use crate::ops::LinearWeights;
use crate::pool::NpuPool;
use crate::tensor::Tensor;

use super::matrix::Matrix;

/// How to cut an [`NpuMatrix`] into tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tiling {
    /// At most this many rows per tile (`None` = no limit). Row blocks are
    /// spread over the NPUs round-robin, so with several NPUs set this or use
    /// [`NpuMatrix::upload_sharded`].
    pub max_rows: Option<usize>,
    /// At most this many columns per tile (`None` = no limit). Column tiles
    /// produce partial sums that are added on the host in f32.
    pub max_cols: Option<usize>,
}

/// Rows scaled to unit peak magnitude, plus the scale of each row. This is
/// the one place where values are prepared for the NPU's limited precision
/// (and the fixed input range of analog hardware): matrix rows on upload,
/// vectors on every product.
#[derive(Debug)]
struct UnitRows {
    values: Vec<f32>,
    scales: Vec<f32>,
}

impl UnitRows {
    fn new<'a>(rows: impl Iterator<Item = &'a [f32]>) -> Self {
        let mut out = Self {
            values: Vec::new(),
            scales: Vec::new(),
        };
        for row in rows {
            let peak = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if peak > 0.0 { peak } else { 1.0 };
            out.scales.push(s);
            out.values.extend(row.iter().map(|v| v / s));
        }
        out
    }
}

/// One rectangular block of the matrix on one NPU.
#[derive(Debug)]
struct Tile {
    npu: usize,
    row0: usize,
    col0: usize,
    weights: LinearWeights,
    /// Per-row scale divided out before upload.
    row_scale: Vec<f32>,
}

impl Tile {
    fn rows(&self) -> usize {
        self.row_scale.len()
    }

    fn cols(&self) -> usize {
        self.weights.in_features()
    }

    /// This tile's columns of every vector in `x` through the tile; returns
    /// `x.rows() × self.rows()` with all scales undone.
    fn apply(&self, x: &Matrix) -> Result<Matrix> {
        let cols = self.col0..self.col0 + self.cols();
        let input = UnitRows::new(x.iter_rows().map(|r| &r[cols.clone()]));
        let y = self
            .weights
            .forward(&Tensor::from_f32(&input.values, [x.rows(), self.cols()])?)?
            .to_f32_vec();
        let mut out = Matrix::new(x.rows(), self.rows(), y)?;
        for (row, &vs) in out.iter_rows_mut().zip(&input.scales) {
            for (v, rs) in row.iter_mut().zip(&self.row_scale) {
                *v *= rs * vs;
            }
        }
        Ok(out)
    }
}

/// Runs `job` once per item on its own thread and collects the results in order.
fn parallel<I: Sync, T: Send>(items: &[I], job: impl Fn(&I) -> Result<T> + Sync) -> Result<Vec<T>> {
    std::thread::scope(|s| {
        let job = &job;
        let handles: Vec<_> = items.iter().map(|it| s.spawn(move || job(it))).collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::Unsupported("NPU thread panicked".into())))
            })
            .collect()
    })
}

/// How an [`NpuMatrix`] is laid out over its NPUs.
#[derive(Debug)]
enum Layout {
    /// Tiles, possibly on several NPUs (row blocks round-robin).
    Tiled(Vec<Tile>),
    /// A full copy per NPU; batches are split between them.
    Replicated(Vec<NpuMatrix>),
}

/// A matrix kept on one or more NPUs, for streaming many vectors through it.
///
/// Upload once, then call [`apply_batch`](Self::apply_batch) (or
/// [`apply`](Self::apply) / [`matmul`](Self::matmul)) as often as needed:
/// only the vectors move. Large matrices can be split into tiles (to respect
/// device memory) and across NPUs.
#[derive(Debug)]
pub struct NpuMatrix {
    rows: usize,
    cols: usize,
    npus: Vec<Npu>,
    layout: Layout,
}

impl NpuMatrix {
    /// Uploads `a` to `npu` as a single tile.
    pub fn upload(npu: &Npu, a: &Matrix) -> Result<Self> {
        Self::upload_tiled(std::slice::from_ref(npu), a, Tiling::default())
    }

    /// Uploads a full copy of `a` to every NPU of `pool`. Each batch is split
    /// between the copies, so every vector goes to one NPU only.
    ///
    /// This is the layout for throughput: use it whenever the matrix fits on
    /// one device. Use [`upload_sharded`](Self::upload_sharded) only when it
    /// doesn't, because a sharded matrix needs every vector on every NPU.
    pub fn upload_replicated(pool: &NpuPool, a: &Matrix) -> Result<Self> {
        let copies = pool
            .npus()
            .iter()
            .map(|npu| Self::upload(npu, a))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rows: a.rows(),
            cols: a.cols(),
            npus: pool.npus().to_vec(),
            layout: Layout::Replicated(copies),
        })
    }

    /// Uploads `a` split into one row block per NPU of `pool` (sharding);
    /// products run on all of them in parallel. Each NPU holds `1/n` of the
    /// matrix, but every vector is sent to every NPU, so prefer
    /// [`upload_replicated`](Self::upload_replicated) when the matrix fits.
    pub fn upload_sharded(pool: &NpuPool, a: &Matrix) -> Result<Self> {
        let per = a.rows().div_ceil(pool.len().max(1)).max(1);
        Self::upload_tiled(
            pool.npus(),
            a,
            Tiling {
                max_rows: Some(per),
                max_cols: None,
            },
        )
    }

    /// Uploads `a` cut into tiles of at most `tiling` size, row blocks spread
    /// over `npus` round-robin.
    pub fn upload_tiled(npus: &[Npu], a: &Matrix, tiling: Tiling) -> Result<Self> {
        if npus.is_empty() {
            return Err(Error::InvalidArgument("no NPUs given".into()));
        }
        let (rows, cols) = a.shape();
        if rows == 0 || cols == 0 {
            return Err(Error::Shape("cannot upload an empty matrix".into()));
        }
        let tr = tiling.max_rows.unwrap_or(rows).clamp(1, rows);
        let tc = tiling.max_cols.unwrap_or(cols).clamp(1, cols);
        let mut tiles = Vec::new();
        for (block, row0) in (0..rows).step_by(tr).enumerate() {
            let npu = block % npus.len();
            let block_rows = row0..(row0 + tr).min(rows);
            for col0 in (0..cols).step_by(tc) {
                let tile_cols = col0..(col0 + tc).min(cols);
                let unit = UnitRows::new(block_rows.clone().map(|r| &a.row(r)[tile_cols.clone()]));
                let shape = [block_rows.len(), tile_cols.len()];
                tiles.push(Tile {
                    npu,
                    row0,
                    col0,
                    weights: LinearWeights::upload(
                        &npus[npu],
                        &Tensor::from_f32(&unit.values, shape)?,
                    )?,
                    row_scale: unit.scales,
                });
            }
        }
        Ok(Self {
            rows,
            cols,
            npus: npus.to_vec(),
            layout: Layout::Tiled(tiles),
        })
    }

    /// Rows of the matrix (length of each output vector).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the matrix (length of each input vector).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of tiles (over all copies).
    pub fn tiles(&self) -> usize {
        match &self.layout {
            Layout::Tiled(t) => t.len(),
            Layout::Replicated(c) => c.iter().map(NpuMatrix::tiles).sum(),
        }
    }

    /// Number of NPUs holding (part of) the matrix.
    pub fn npus(&self) -> usize {
        self.npus.len()
    }

    /// `y = A · x` for one vector. Prefer [`apply_batch`](Self::apply_batch):
    /// a single vector moves `cols + rows` values for `rows · cols` MACs.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(self
            .apply_batch(&Matrix::new(1, x.len(), x.to_vec())?)?
            .into_vec())
    }

    /// `Y[i] = A · X[i]` for every row of `X` (`k × cols` → `k × rows`),
    /// i.e. `Y = X · Aᵀ`. Work on different NPUs runs in parallel.
    pub fn apply_batch(&self, x: &Matrix) -> Result<Matrix> {
        if x.cols() != self.cols {
            return Err(Error::Shape(format!(
                "vectors have {} values, the matrix has {} columns",
                x.cols(),
                self.cols
            )));
        }
        if x.rows() == 0 {
            return Ok(Matrix::zeros(0, self.rows));
        }
        match &self.layout {
            Layout::Tiled(tiles) => self.apply_tiled(tiles, x),
            Layout::Replicated(copies) => Self::apply_replicated(copies, x),
        }
    }

    /// One thread per NPU runs that NPU's tiles; partial sums are added here.
    fn apply_tiled(&self, tiles: &[Tile], x: &Matrix) -> Result<Matrix> {
        let npus: Vec<usize> = (0..self.npus.len()).collect();
        let per_npu = parallel(&npus, |&npu| {
            tiles
                .iter()
                .filter(|t| t.npu == npu)
                .map(|t| Ok((t, t.apply(x)?)))
                .collect::<Result<Vec<_>>>()
        })?;
        let mut y = Matrix::zeros(x.rows(), self.rows);
        for (tile, part) in per_npu.into_iter().flatten() {
            for (dst, src) in y.iter_rows_mut().zip(part.iter_rows()) {
                let dst = &mut dst[tile.row0..tile.row0 + tile.rows()];
                dst.iter_mut().zip(src).for_each(|(d, s)| *d += s);
            }
        }
        Ok(y)
    }

    /// Splits the rows of `x` between the copies, one thread per NPU.
    fn apply_replicated(copies: &[NpuMatrix], x: &Matrix) -> Result<Matrix> {
        let per = x.rows().div_ceil(copies.len()).max(1);
        let chunks: Vec<(&NpuMatrix, Matrix)> = copies
            .iter()
            .zip((0..x.rows()).step_by(per))
            .map(|(c, start)| (c, x.slice_rows(start..(start + per).min(x.rows()))))
            .collect();
        let parts = parallel(&chunks, |(copy, chunk)| copy.apply_batch(chunk))?;
        let mut y = Matrix::zeros(0, copies[0].rows);
        for part in &parts {
            for row in part.iter_rows() {
                y.push_row(row)?;
            }
        }
        Ok(y)
    }

    /// `A · B` with `B` of shape `cols × n`: streams the `n` columns of `B`.
    pub fn matmul(&self, b: &Matrix) -> Result<Matrix> {
        Ok(self.apply_batch(&b.transpose())?.transpose())
    }

    /// `A^steps · x`, keeping `A` on the device between steps (square
    /// matrices only). Each step's input is scaled to unit range before it
    /// goes to the NPU, so growing or shrinking vectors stay accurate in
    /// bf16. Useful for Markov chains, power iteration, linear recurrences
    /// and time-stepping.
    pub fn apply_repeated(&self, x: &[f32], steps: usize) -> Result<Vec<f32>> {
        if self.rows != self.cols {
            return Err(Error::Shape("apply_repeated needs a square matrix".into()));
        }
        (0..steps).try_fold(x.to_vec(), |v, _| self.apply(&v))
    }
}

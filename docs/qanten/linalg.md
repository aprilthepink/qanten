# Linear algebra (`qanten::linalg`)

Solvers whose heavy inner loops run on the NPU but whose results are accurate to
f64 level, using **mixed-precision iterative refinement**.

The NPU multiplies in bfloat16 (about 3 significant digits). The solvers let it do the
many cheap-to-be-wrong inner iterations, then compute a few exact f64 residuals on the
host to correct the result. As long as the matrix is reasonably conditioned
(condition number well below ~100), this converges to f64-level accuracy.

```rust
use qanten::compute::Matrix64;
use qanten::linalg::{LinearOperator, SolveOptions};
use qanten::Npu;

fn main() -> qanten::Result<()> {
    let npu = Npu::auto(0)?;
    // A symmetric positive definite system.
    let a = Matrix64::from_rows(&[
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ])?;
    let op = LinearOperator::new(&npu, &a)?;
    let (x, report) = op.conjugate_gradient(&[1.0, 2.0, 3.0], &SolveOptions::default())?;
    println!("x = {x:?}, residual {:.1e} after {} NPU products and {} refinements",
             report.relative_residual, report.npu_iterations, report.refinements);
    assert!(report.converged);
    Ok(())
}
```

## Solvers

| Method | Solves | NPU work | Host work |
|---|---|---|---|
| `op.conjugate_gradient(&b, &opts)` | `A·x = b`, `A` symmetric positive definite | CG iterations (`A·p`) | one exact residual per refinement |
| `op.least_squares(&b, &opts)` | `min ‖A·x − b‖₂`, any shape (op built with `with_transpose`) | CGLS iterations (`A·p`, `Aᵀ·q`) | exact normal-equation residual per refinement |
| `op.top_eigenpairs(k, max_iters, seed)` | `k` largest-magnitude eigenpairs of a symmetric `A` | block power iteration (`k+4` vectors per call) | orthonormalisation, Rayleigh-Ritz, final exact iterations |
| `Pca::fit(&npu, &data, k)` | principal components of a `Matrix64` (one sample per row) | power iteration on the covariance | covariance (exact), centring |

`LinearOperator::new(&npu, &a)` takes a `compute::Matrix64` and keeps it twice: on the
host in f64 and on the NPU as a `compute::NpuMatrix` (bf16, every row and input vector
scaled to unit range). `apply_npu` / `apply_t_npu` give fast approximate products of many
vectors (rows of a `Matrix`) for your own algorithms; `apply_exact` / `apply_t_exact` the
exact ones. Small dense eigenproblems that don't need the NPU at all have
`Matrix::symmetric_eigen()` (exact Jacobi on the CPU), which `top_eigenpairs` also uses
for its Rayleigh–Ritz step.

`SolveOptions { tol, max_refinements, inner_iters, inner_tol }` defaults to a target
relative residual of 1e-10, at most 30 refinements, 50 inner iterations aiming for
1e-2 each. `SolveReport` says whether it converged, the final exact residual, and how
many refinements and NPU iterations it took.

## Measured (tests on the simulator)

| Problem | Result |
|---|---|
| CG, 64×64 SPD | relative residual 3.9e-12, solution error 5.7e-12; 25 NPU products, 5 refinements |
| CG without refinement (1 pass) | stuck at bf16 accuracy (> 1e-6) |
| least squares 80×12 | normal-equation gradient `Aᵀ(A·x − b)` below 1e-9 of `‖Aᵀb‖` |
| top-3 eigenpairs, 40×40, known spectrum | eigenvalues to < 1e-9, vectors aligned to 1 − 1e-9; 3 NPU iterations + 4 exact |
| PCA, 500 × 5 points along a known direction | first component aligned > 0.999, > 95 % of variance |

**Where the NPU pays off.** For linear solves the NPU does the bulk: many inner
iterations per exact refinement. For eigenproblems the split depends on the spectrum:
with well-separated eigenvalues (as in the test) the power iteration converges in a
few steps either way. The NPU's share grows when eigenvalues are close together and
power iteration needs many steps.

## When refinement can't help

If `A` is badly conditioned (condition number around 1/ε_bf16 ≈ 100 or worse), the
inner solves can't reduce the residual and the solver stops with `converged: false`.
Precondition or rescale the problem first, or solve on the CPU.

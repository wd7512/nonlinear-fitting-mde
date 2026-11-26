//! symbfitrs: a fast CPU fitter for nonlinear least squares with < 100 parameters.
//!
//! * [`tape`]     vectorised expression tapes exported from SymPy (model, Jacobian, separable parts)
//! * [`problem`]  a bound fitting problem (data + tapes + starts)
//! * [`tracker`]  deadline, best-so-far, checkpoint and counter bookkeeping shared by all solvers
//! * [`linalg`]   rank-revealing QR→SVD factorisation used for every least-squares step
//! * [`solvers`]  Levenberg–Marquardt on full or variable-projected residuals
//! * [`fit`]      variable projection + LM with its structural degenerate cases

pub mod fit;
pub mod linalg;
pub mod problem;
pub mod solvers;
pub mod special;
pub mod tape;
pub mod tracker;

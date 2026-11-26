//! nlfit-core: a fast CPU inner optimiser for nonlinear least squares with < 100 parameters.
//!
//! * [`tape`]     vectorised expression tapes exported from SymPy (model, Jacobian, separable parts)
//! * [`problem`]  a bound fitting problem (data + tapes + starts)
//! * [`tracker`]  deadline, best-so-far, checkpoint and counter bookkeeping shared by all solvers
//! * [`linalg`]   rank-revealing QR→SVD factorisation used for every least-squares step
//! * [`solvers`]  LM / Gauss–Newton / GN→LM hybrid on full or variable-projected residuals
//! * [`policy`]   named policies, including `auto` (V0 = VarPro+LM)

pub mod features;
pub mod linalg;
pub mod policy;
pub mod problem;
pub mod solvers;
pub mod special;
pub mod tape;
pub mod tracker;

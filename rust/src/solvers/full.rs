//! The model's own residual and Jacobian over all p parameters.

use nalgebra::{DMatrix, DVector};

use super::Lsq;
use crate::problem::Problem;
use crate::tracker::{Res, Tracker};

pub struct FullLsq<'a> {
    pub prob: &'a mut Problem,
    pub tr: &'a mut Tracker,
}

impl Lsq for FullLsq<'_> {
    fn n(&self) -> usize {
        self.prob.p
    }
    fn m(&self) -> usize {
        self.prob.m
    }
    fn tracker(&mut self) -> &mut Tracker {
        self.tr
    }
    fn residual(&mut self, x: &DVector<f64>, r: &mut DVector<f64>) -> Res<f64> {
        let prob = &mut *self.prob;
        let sse = self.tr.timed(|| prob.residual(x.as_slice(), r.as_mut_slice()))?;
        self.tr.record(x.as_slice(), sse);
        Ok(sse)
    }
    fn jacobian(&mut self, x: &DVector<f64>, j: &mut DMatrix<f64>) -> Res<bool> {
        let prob = &mut *self.prob;
        self.tr.check()?;
        prob.jacobian(x.as_slice(), j);
        Ok(j.iter().all(|v| v.is_finite()))
    }
}

//! Fixed-structure AMOEBA 2018 analysis, independent of rendering and selection syntax.
//! Native template assignment, permanent multipoles, mutual polarization and buffered 14-7 vdW.
use super::Molecule;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;
mod energy;
pub mod parameters;
pub mod protonation;
pub mod repair;
mod search;
mod topology;

#[derive(Debug, Error)]
pub enum AmoebaError {
    #[error("AMOEBA: {0}")]
    Invalid(String),
    #[error("AMOEBA parameter I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("AMOEBA result: {0}")]
    Json(#[from] serde_json::Error),
    #[error("background operation canceled")]
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisSettings {
    pub cutoff: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for AnalysisSettings {
    fn default() -> Self {
        Self {
            cutoff: 6.0,
            tolerance: 1e-6,
            max_iterations: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HydrogenBondCandidate {
    pub donor: usize,
    pub hydrogen: usize,
    pub acceptor: usize,
    pub donor_group: Vec<usize>,
    pub acceptor_group: Vec<usize>,
    pub donor_position: [f32; 3],
    pub acceptor_position: [f32; 3],
    /// E_full - E_AB_decoupled, kJ/mol. Negative means stabilizing cross coupling.
    pub delta_energy: f64,
    pub permanent: f64,
    pub vdw: f64,
    pub polarization: f64,
    pub da_distance: f64,
    pub ha_distance: f64,
    pub dha_angle: f64,
    pub parameterization_status: String,
    pub scf_iterations: usize,
    pub scf_residual_debye: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResidueParameterization {
    pub residue: String,
    pub atoms: Vec<usize>,
    pub status: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HydrogenBondReport {
    pub version: u32,
    pub selection: String,
    pub force_field: String,
    pub cutoff: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
    /// Display filter only. All chemically eligible candidates retain their scores.
    pub energy_threshold: f64,
    pub candidates: Vec<HydrogenBondCandidate>,
    pub residues: Vec<ResidueParameterization>,
    pub full_permanent: f64,
    pub full_vdw: f64,
    pub full_polarization: f64,
    pub scf_iterations: usize,
    pub scf_residual_debye: f64,
}

impl HydrogenBondReport {
    pub fn displayed(&self) -> impl Iterator<Item = &HydrogenBondCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.delta_energy < self.energy_threshold)
    }

    pub fn validate(&self, atom_count: usize) -> Result<(), AmoebaError> {
        let invalid = |message: &str| AmoebaError::Invalid(message.to_owned());
        if self.version != 1
            || self.selection.is_empty()
            || !(5.0..=12.0).contains(&self.cutoff)
            || !(0.0..=0.001).contains(&self.tolerance)
            || self.tolerance == 0.0
            || !(1..=10000).contains(&self.max_iterations)
            || ![
                self.energy_threshold,
                self.full_permanent,
                self.full_vdw,
                self.full_polarization,
                self.scf_residual_debye,
            ]
            .iter()
            .all(|v| v.is_finite())
            || self.scf_residual_debye < 0.0
            || self.scf_residual_debye > self.tolerance
            || self.scf_iterations > self.max_iterations
        {
            return Err(invalid("invalid report settings or convergence"));
        }
        for c in &self.candidates {
            if [c.donor, c.hydrogen, c.acceptor]
                .iter()
                .any(|i| *i >= atom_count)
                || c.donor == c.hydrogen
                || c.donor == c.acceptor
                || c.hydrogen == c.acceptor
                || c.donor_group.is_empty()
                || c.acceptor_group.is_empty()
                || !c.donor_group.contains(&c.donor)
                || !c.donor_group.contains(&c.hydrogen)
                || !c.acceptor_group.contains(&c.acceptor)
                || c.donor_group
                    .iter()
                    .chain(&c.acceptor_group)
                    .any(|i| *i >= atom_count)
                || c.donor_group.iter().any(|i| c.acceptor_group.contains(i))
                || !c
                    .donor_position
                    .iter()
                    .chain(&c.acceptor_position)
                    .all(|v| v.is_finite())
                || ![
                    c.delta_energy,
                    c.permanent,
                    c.vdw,
                    c.polarization,
                    c.da_distance,
                    c.ha_distance,
                    c.dha_angle,
                    c.scf_residual_debye,
                ]
                .iter()
                .all(|v| v.is_finite())
                || c.da_distance <= 0.0
                || c.ha_distance <= 0.0
                || !(0.0..=180.0).contains(&c.dha_angle)
                || c.parameterization_status != "parameterized"
                || c.scf_residual_debye < 0.0
                || c.scf_residual_debye > self.tolerance
                || c.scf_iterations > self.max_iterations
                || (c.delta_energy - c.permanent - c.vdw - c.polarization).abs()
                    > 1e-6 * (1.0 + c.delta_energy.abs())
            {
                return Err(invalid(
                    "invalid candidate indices, energy decomposition or convergence",
                ));
            }
        }
        for r in &self.residues {
            if r.atoms.iter().any(|i| *i >= atom_count)
                || !matches!(r.status.as_str(), "parameterized" | "unparameterized")
            {
                return Err(invalid("invalid residue parameterization status"));
            }
        }
        Ok(())
    }
}

/// Coordinates stay fixed. Explicit H/template matching determines protonation.
/// The environment is the entire parameterizable structure, not just `selected`.
pub fn analyze(
    molecule: &Molecule,
    selection: &str,
    selected: &[usize],
    settings: AnalysisSettings,
    cancel: &AtomicBool,
) -> Result<HydrogenBondReport, AmoebaError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(AmoebaError::Cancelled);
    }
    if !(5.0..=12.0).contains(&settings.cutoff)
        || !(1e-10..=0.001).contains(&settings.tolerance)
        || !(1..=10000).contains(&settings.max_iterations)
    {
        return Err(AmoebaError::Invalid(
            "invalid neighbor radius or SCF settings".into(),
        ));
    }
    if selection.trim().is_empty()
        || selected.is_empty()
        || selected.iter().any(|i| *i >= molecule.atoms.len())
    {
        return Err(AmoebaError::Invalid(
            "select at least one valid atom".into(),
        ));
    }
    if molecule
        .bonds
        .iter()
        .any(|b| b.a >= molecule.atoms.len() || b.b >= molecule.atoms.len() || b.a == b.b)
    {
        return Err(AmoebaError::Invalid("invalid covalent bond indices".into()));
    }
    if molecule.atoms.iter().any(|a| !a.position.is_finite()) {
        return Err(AmoebaError::Invalid("nonfinite coordinates".into()));
    }
    let ff = parameters::ForceField::builtin()?;
    search::run(molecule, selection, selected, settings, cancel, &ff)
}

#[cfg(test)]
mod tests;

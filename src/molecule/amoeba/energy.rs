//! Nonperiodic AMOEBA reference equations. See data/amoeba/LICENSE for attribution.
use super::{
    AmoebaError, AnalysisSettings,
    parameters::{ForceField, Multipole},
    topology::{Topology, check_cancel},
};
use glam::{DMat3, DVec3};
use std::sync::atomic::AtomicBool;
const COULOMB: f64 = 138.93545764438198;
const DEBYE: f64 = 48.033324;
fn invalid(s: &str) -> AmoebaError {
    AmoebaError::Invalid(s.into())
}
fn unit(v: DVec3) -> Result<DVec3, AmoebaError> {
    if v.length() < 1e-12 {
        Err(invalid("degenerate AMOEBA local frame"))
    } else {
        Ok(v.normalize())
    }
}
#[derive(Clone)]
pub(super) struct Particle {
    pub q: f64,
    pub mu: DVec3,
    pub quad: DMat3,
    permanent_quad: DMat3,
    pub alpha: f64,
    damp: f64,
    thole: f64,
    class: usize,
    site: DVec3,
}
fn axes<'a>(
    i: usize,
    t: &Topology,
    ff: &'a ForceField,
) -> Result<(&'a Multipole, [Option<usize>; 3]), AmoebaError> {
    let records: Vec<_> = ff
        .multipoles
        .iter()
        .filter(|m| m.r#type == t.types[i])
        .collect();
    for shell in 0..2 {
        for &m in &records {
            let [kz, kx, ky] = m.axes.map(|x| x.unsigned_abs() as usize);
            for &z in &t.bonds[i] {
                if t.types[z] != kz {
                    continue;
                }
                for &x in &t.shells[i][shell] {
                    if x == z || t.types[x] != kx || (shell == 1 && !t.bonds[x].contains(&z)) {
                        continue;
                    }
                    if ky == 0 {
                        return Ok((m, [Some(z), Some(x), None]));
                    }
                    for &y in &t.shells[i][shell] {
                        if y != z
                            && y != x
                            && t.types[y] == ky
                            && (shell == 0 || t.bonds[y].contains(&z))
                        {
                            return Ok((m, [Some(z), Some(x), Some(y)]));
                        }
                    }
                }
            }
        }
    }
    for &m in &records {
        if m.axes[1] == 0
            && let Some(&z) = t.bonds[i]
                .iter()
                .find(|&&z| t.types[z] == m.axes[0].unsigned_abs() as usize)
        {
            return Ok((m, [Some(z), None, None]));
        }
    }
    for &m in &records {
        if m.axes[0] == 0 {
            return Ok((m, [None; 3]));
        }
    }
    Err(AmoebaError::Invalid(format!(
        "atom {} ({}): no compatible AMOEBA multipole local frame",
        t.original[i], t.names[i]
    )))
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalFrame {
    ZThenX,
    Bisector,
    ZBisect,
    ThreeFold,
    ZOnly,
    None,
}

impl Particle {
    fn build(i: usize, t: &Topology, ff: &ForceField) -> Result<Self, AmoebaError> {
        let (m, [z, x, y]) = axes(i, t, ff)?;
        let [kz, kx, ky] = m.axes;
        let mut axis = LocalFrame::ZThenX;
        if kz == 0 {
            axis = LocalFrame::None;
        }
        if kz != 0 && kx == 0 {
            axis = LocalFrame::ZOnly;
        }
        if kz < 0 || kx < 0 {
            axis = LocalFrame::Bisector;
        }
        if kx < 0 && ky < 0 {
            axis = LocalFrame::ZBisect;
        }
        if kz < 0 && kx < 0 && ky < 0 {
            axis = LocalFrame::ThreeFold;
        }
        let mut dip = DVec3::from_array(m.dipole);
        let mut quad = DMat3::from_cols_array(&m.quadrupole);
        if axis == LocalFrame::ZThenX
            && let (Some(z), Some(x), Some(y)) = (z, x, y)
            && (t.xyz[z] - t.xyz[y])
                .cross(t.xyz[x] - t.xyz[y])
                .dot(t.xyz[i] - t.xyz[y])
                < 0.
        {
            let reflect = DMat3::from_diagonal(DVec3::new(1., -1., 1.));
            dip = reflect * dip;
            quad = reflect * quad * reflect;
        }
        let mut rot = DMat3::IDENTITY;
        if let Some(z) = z {
            let mut vz = unit(t.xyz[z] - t.xyz[i])?;
            let mut vx = if axis == LocalFrame::ZOnly {
                if vz.x.abs() < 0.866 {
                    DVec3::X
                } else {
                    DVec3::Y
                }
            } else {
                t.xyz[x.ok_or_else(|| invalid("missing AMOEBA x axis"))?] - t.xyz[i]
            };
            if axis == LocalFrame::Bisector {
                vz = unit(vz + unit(vx)?)?;
            }
            if axis == LocalFrame::ZBisect || axis == LocalFrame::ThreeFold {
                let vy =
                    unit(t.xyz[y.ok_or_else(|| invalid("missing AMOEBA y axis"))?] - t.xyz[i])?;
                vx = unit(vx)?;
                if axis == LocalFrame::ZBisect {
                    vx = unit(vx + vy)?;
                } else {
                    vz = unit(vz + vx + vy)?;
                }
            }
            vx = unit(vx - vz * vz.dot(vx))?;
            rot = DMat3::from_cols(vx, vz.cross(vx), vz);
        }
        // Reference permanent energies use spherical Q20 = 3 Qzz and
        // Q22 proportional to Qxx-Qyy. Preserve that convention when rounded
        // parameter tables have a tiny nonzero trace. The induced-field
        // equations retain the original Cartesian tensor, as in Reference.
        let mut permanent_quad = quad;
        let half_trace = 0.5 * (quad.x_axis.x + quad.y_axis.y + quad.z_axis.z);
        permanent_quad.x_axis.x -= half_trace;
        permanent_quad.y_axis.y -= half_trace;
        let pol = &ff.polar[&t.types[i]];
        let class = ff.types[&t.types[i]].class;
        let vd = &ff.vdw[&class];
        let site = if t.elements[i] == "H" && t.bonds[i].len() == 1 {
            let parent = t.xyz[t.bonds[i][0]];
            parent + vd.reduction * (t.xyz[i] - parent)
        } else {
            t.xyz[i]
        };
        Ok(Self {
            q: m.charge,
            mu: rot * dip,
            quad: rot * quad * rot.transpose(),
            permanent_quad: rot * permanent_quad * rot.transpose(),
            alpha: pol.alpha,
            damp: if pol.thole == 0. {
                0.
            } else {
                pol.alpha.powf(1. / 6.)
            },
            thole: pol.thole,
            class,
            site,
        })
    }
}
// Cartesian derivatives of 1/r contracted with AMOEBA moments (Q has no
// extra factorial). r points from j to i. Tracelessness is not assumed here.
fn permanent(a: &Particle, b: &Particle, r: DVec3) -> f64 {
    let s = r.length_recip();
    let s3 = s.powi(3);
    let s5 = s.powi(5);
    let s7 = s.powi(7);
    let s9 = s.powi(9);
    let qa = a.permanent_quad;
    let qb = b.permanent_quad;
    let ar = qa * r;
    let br = qb * r;
    let ra = r.dot(ar);
    let rb = r.dot(br);
    let ta = qa.x_axis.x + qa.y_axis.y + qa.z_axis.z;
    let tb = qb.x_axis.x + qb.y_axis.y + qb.z_axis.z;
    let da = a.mu.dot(r);
    let db = b.mu.dot(r);
    let qq = qa
        .to_cols_array()
        .iter()
        .zip(qb.to_cols_array())
        .map(|(x, y)| x * y)
        .sum::<f64>();
    a.q * b.q * s - (b.q * da - a.q * db) * s3 + 3. * (a.q * rb + b.q * ra - da * db) * s5
        - (a.q * tb + b.q * ta - a.mu.dot(b.mu)) * s3
        - 15. * (da * rb - db * ra) * s7
        + 3. * (da * tb - db * ta + 2. * (a.mu.dot(br) - b.mu.dot(ar))) * s5
        + 105. * ra * rb * s9
        - 15. * (ta * rb + tb * ra + 4. * ar.dot(br)) * s7
        + 3. * (ta * tb + 2. * qq) * s5
}
fn damped(a: &Particle, b: &Particle, d: f64) -> [f64; 3] {
    let mut v = [d.powi(-3), 3. * d.powi(-5), 15. * d.powi(-7)];
    let damping = a.damp * b.damp;
    if damping != 0. {
        let e = -a.thole.min(b.thole) * (d / damping).powi(3);
        if e > -50. {
            let ex = e.exp();
            v[0] *= 1. - ex;
            v[1] *= 1. - (1. - e) * ex;
            v[2] *= 1. - (1. - e + 0.6 * e * e) * ex;
        }
    }
    v
}
fn field(p: &Particle, r: DVec3, v: [f64; 3]) -> DVec3 {
    let qr = p.quad * r;
    -r * (v[0] * p.q - v[1] * p.mu.dot(r) + v[2] * qr.dot(r)) - p.mu * v[0] + qr * (2. * v[1])
}
#[derive(Clone, Copy, Debug)]
pub(super) struct PolarEnergy {
    pub energy: f64,
    pub iterations: usize,
    pub residual: f64,
}
pub(super) struct Evaluator<'a> {
    pub particles: Vec<Particle>,
    t: &'a Topology,
    ff: &'a ForceField,
    settings: AnalysisSettings,
    cancel: &'a AtomicBool,
}
impl<'a> Evaluator<'a> {
    pub fn new(
        t: &'a Topology,
        ff: &'a ForceField,
        settings: AnalysisSettings,
        cancel: &'a AtomicBool,
    ) -> Result<Self, AmoebaError> {
        let mut particles = Vec::new();
        for i in 0..t.xyz.len() {
            check_cancel(cancel)?;
            particles.push(Particle::build(i, t, ff)?);
        }
        for i in 0..t.xyz.len() {
            check_cancel(cancel)?;
            for j in 0..i {
                if t.xyz[i].distance(t.xyz[j]) < 1e-8 {
                    return Err(invalid("overlapping AMOEBA atoms"));
                }
            }
        }
        Ok(Self {
            particles,
            t,
            ff,
            settings,
            cancel,
        })
    }
    fn cross(&self, i: usize, j: usize, groups: Option<(usize, usize)>) -> bool {
        groups.is_some_and(|(a, b)| {
            let x = self.t.groups[i];
            let y = self.t.groups[j];
            (x == a && y == b) || (x == b && y == a)
        })
    }
    /// Pairwise permanent/vdW terms can be differenced exactly by summing cross
    /// terms. Polarization must instead be solved again for the entire system.
    pub fn pairwise(&self, only_cross: Option<(usize, usize)>) -> Result<(f64, f64), AmoebaError> {
        let mut ep = 0.;
        let mut ev = 0.;
        for i in 0..self.particles.len() {
            check_cancel(self.cancel)?;
            for j in 0..i {
                if only_cross.is_some() && !self.cross(i, j, only_cross) {
                    continue;
                }
                let a = &self.particles[i];
                let b = &self.particles[j];
                let shells = &self.t.shells[i];
                let near = shells[0].contains(&j) || shells[1].contains(&j);
                let scale = if near {
                    0.
                } else if shells[2].contains(&j) {
                    0.4
                } else if shells[3].contains(&j) {
                    0.8
                } else {
                    1.
                };
                if scale != 0. {
                    ep += COULOMB * scale * permanent(a, b, self.t.xyz[i] - self.t.xyz[j]);
                }
                if !near {
                    let va = &self.ff.vdw[&a.class];
                    let vb = &self.ff.vdw[&b.class];
                    let (radius, epsilon) = if let Some(p) = self.ff.pairs.iter().find(|p| {
                        p.classes == [a.class, b.class] || p.classes == [b.class, a.class]
                    }) {
                        (p.radius, p.epsilon)
                    } else {
                        let sum = va.radius.powi(2) + vb.radius.powi(2);
                        let e = va.epsilon.sqrt() + vb.epsilon.sqrt();
                        (
                            if sum == 0. {
                                0.
                            } else {
                                2. * (va.radius.powi(3) + vb.radius.powi(3)) / sum
                            },
                            if e == 0. {
                                0.
                            } else {
                                4. * va.epsilon * vb.epsilon / e.powi(2)
                            },
                        )
                    };
                    if epsilon != 0. && radius != 0. {
                        let rho = a.site.distance(b.site) / radius;
                        ev += epsilon
                            * (1.07 / (rho + 0.07)).powi(7)
                            * (1.12 / (rho.powi(7) + 0.12) - 2.);
                    }
                }
            }
        }
        if !ep.is_finite() || !ev.is_finite() {
            return Err(invalid("nonfinite AMOEBA energy"));
        }
        Ok((ep, ev))
    }
    fn fields(
        &self,
        groups: Option<(usize, usize)>,
    ) -> Result<(Vec<DVec3>, Vec<DVec3>), AmoebaError> {
        let n = self.particles.len();
        let mut ed = vec![DVec3::ZERO; n];
        let mut ep = ed.clone();
        for i in 0..n {
            check_cancel(self.cancel)?;
            for j in 0..i {
                if self.cross(i, j, groups) {
                    continue;
                }
                let r = self.t.xyz[j] - self.t.xyz[i];
                let v = damped(&self.particles[i], &self.particles[j], r.length());
                let fi = field(&self.particles[j], r, v);
                let fj = field(&self.particles[i], -r, v);
                let same = self.t.groups[i] == self.t.groups[j];
                let ss = &self.t.shells[i];
                let ps = if ss[0].contains(&j) || ss[1].contains(&j) {
                    0.
                } else if same && ss[2].contains(&j) {
                    0.5
                } else {
                    1.
                };
                if !same {
                    ed[i] += fi;
                    ed[j] += fj;
                }
                ep[i] += ps * fi;
                ep[j] += ps * fj;
            }
        }
        Ok((ed, ep))
    }
    fn apply(
        &self,
        x: &[DVec3],
        groups: Option<(usize, usize)>,
    ) -> Result<Vec<DVec3>, AmoebaError> {
        let dip: Vec<_> = x
            .iter()
            .zip(&self.particles)
            .map(|(&x, p)| x * p.alpha.sqrt())
            .collect();
        let mut result = vec![DVec3::ZERO; x.len()];
        for i in 0..x.len() {
            check_cancel(self.cancel)?;
            for j in 0..i {
                if self.cross(i, j, groups) {
                    continue;
                }
                let r = self.t.xyz[j] - self.t.xyz[i];
                let v = damped(&self.particles[i], &self.particles[j], r.length());
                result[i] += -v[0] * dip[j] + v[1] * r * r.dot(dip[j]);
                result[j] += -v[0] * dip[i] + v[1] * r * r.dot(dip[i]);
            }
        }
        for i in 0..x.len() {
            result[i] = x[i] - result[i] * self.particles[i].alpha.sqrt();
        }
        Ok(result)
    }
    fn residual(&self, r: &[DVec3]) -> f64 {
        DEBYE
            * (r.iter()
                .zip(&self.particles)
                .map(|(v, p)| v.length_squared() * p.alpha)
                .sum::<f64>()
                / r.len() as f64)
                .sqrt()
    }
    fn converge(
        &self,
        field: &[DVec3],
        groups: Option<(usize, usize)>,
    ) -> Result<(Vec<DVec3>, usize, f64), AmoebaError> {
        let rhs: Vec<_> = field
            .iter()
            .zip(&self.particles)
            .map(|(&f, p)| f * p.alpha.sqrt())
            .collect();
        let mut x = vec![DVec3::ZERO; rhs.len()];
        let mut r = rhs.clone();
        let mut direction = r.clone();
        let mut norm = dot(&r, &r);
        for iteration in 0..=self.settings.max_iterations {
            check_cancel(self.cancel)?;
            if self.residual(&r) <= self.settings.tolerance {
                let ax = self.apply(&x, groups)?;
                let true_r: Vec<_> = rhs.iter().zip(ax).map(|(&b, a)| b - a).collect();
                let error = self.residual(&true_r);
                if error <= self.settings.tolerance {
                    return Ok((
                        x.iter()
                            .zip(&self.particles)
                            .map(|(&v, p)| v * p.alpha.sqrt())
                            .collect(),
                        iteration,
                        error,
                    ));
                }
                // Restart on accumulated CG roundoff rather than accept a false convergence.
                r = true_r;
                direction = r.clone();
                norm = dot(&r, &r);
            }
            if iteration == self.settings.max_iterations {
                break;
            }
            let product = self.apply(&direction, groups)?;
            let denom = dot(&direction, &product);
            if !denom.is_finite() || denom <= 0. {
                return Err(invalid(
                    "mutual polarization is unstable (non-positive SCF operator)",
                ));
            }
            let step = norm / denom;
            for i in 0..x.len() {
                x[i] += step * direction[i];
                r[i] -= step * product[i];
            }
            let next = dot(&r, &r);
            for i in 0..x.len() {
                direction[i] = r[i] + next / norm * direction[i];
            }
            norm = next;
        }
        Err(AmoebaError::Invalid(format!(
            "mutual polarization did not converge in {} iterations",
            self.settings.max_iterations
        )))
    }
    pub fn polarization(&self, groups: Option<(usize, usize)>) -> Result<PolarEnergy, AmoebaError> {
        if self.particles.is_empty() {
            return Ok(PolarEnergy {
                energy: 0.,
                iterations: 0,
                residual: 0.,
            });
        }
        let (ed, ep) = self.fields(groups)?;
        let (mud, nd, rd) = self.converge(&ed, groups)?;
        let (mup, np, rp) = self.converge(&ep, groups)?;
        let energy = -0.5 * COULOMB * dot(&mud, &ep);
        let reciprocal = -0.5 * COULOMB * dot(&mup, &ed);
        if !energy.is_finite() || (energy - reciprocal).abs() > 1e-5_f64.max(energy.abs() * 1e-6) {
            return Err(invalid(
                "polarization reciprocity check failed; tighten SCF tolerance",
            ));
        }
        Ok(PolarEnergy {
            energy,
            iterations: nd.max(np),
            residual: rd.max(rp),
        })
    }
}
fn dot(a: &[DVec3], b: &[DVec3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.dot(*y)).sum()
}

#[cfg(test)]
mod tests {
    use super::super::parameters::{AtomType, Polar, Vdw};
    use super::*;
    use std::collections::BTreeSet;
    #[derive(serde::Deserialize)]
    struct FrameOracle {
        xyz: Vec<[f64; 3]>,
        mu: Vec<[f64; 3]>,
        induced: Vec<[f64; 3]>,
        permanent: f64,
        polarization: f64,
    }
    #[test]
    fn every_frame_and_reflected_chirality_match_reference_dipoles_and_energy() {
        let cases: Vec<FrameOracle> =
            serde_json::from_str(include_str!("../../../tests/fixtures/amoeba_frames.json"))
                .unwrap();
        for o in cases {
            let mut ff = ForceField::builtin().unwrap();
            for i in 0..6 {
                let ty = 1000 + i;
                let z = (1000 + (i + 1) % 6) as i32;
                let x = (1000 + (i + 2) % 6) as i32;
                let y = (1000 + (i + 3) % 6) as i32;
                let axes = match i {
                    0 => [z, x, y],
                    1 => [-z, -x, 0],
                    2 => [z, -x, -y],
                    3 => [-z, -x, -y],
                    4 => [z, 0, 0],
                    _ => [0; 3],
                };
                ff.types.insert(
                    ty,
                    AtomType {
                        class: ty,
                        element: "C".into(),
                    },
                );
                ff.polar.insert(
                    ty,
                    Polar {
                        alpha: 0.001,
                        thole: 0.39,
                        groups: vec![],
                    },
                );
                ff.vdw.insert(
                    ty,
                    Vdw {
                        radius: 0.,
                        epsilon: 0.,
                        reduction: 1.,
                    },
                );
                ff.multipoles.push(Multipole {
                    r#type: ty,
                    axes,
                    charge: 0.05,
                    dipole: [0.001, 0.002, 0.003],
                    quadrupole: [
                        0.0001, 0.00002, 0., 0.00002, -0.00004, 0.00001, 0., 0.00001, -0.00006,
                    ],
                });
            }
            let bonds: Vec<Vec<_>> = (0..6)
                .map(|i| (0..6).filter(|&j| i != j).collect())
                .collect();
            let mut t = Topology {
                original: (0..6).collect(),
                xyz: o.xyz.iter().map(|&p| DVec3::from_array(p)).collect(),
                names: vec!["X".into(); 6],
                elements: vec!["C".into(); 6],
                templates: vec!["TEST".into(); 6],
                types: (1000..1006).collect(),
                shells: bonds
                    .iter()
                    .map(|b| {
                        [
                            b.iter().copied().collect(),
                            BTreeSet::new(),
                            BTreeSet::new(),
                            BTreeSet::new(),
                        ]
                    })
                    .collect(),
                bonds,
                groups: (0..6).collect(),
            };
            let particles: Vec<_> = (0..6)
                .map(|i| Particle::build(i, &t, &ff).unwrap())
                .collect();
            // The synthetic frame-defining graph is only an assignment fixture;
            // the reference energy has no covalent exclusions.
            t.shells = vec![Default::default(); 6];
            let cancel = AtomicBool::new(false);
            let ev = Evaluator {
                particles,
                t: &t,
                ff: &ff,
                settings: AnalysisSettings {
                    tolerance: 1e-10,
                    ..Default::default()
                },
                cancel: &cancel,
            };
            for (i, p) in ev.particles.iter().enumerate() {
                assert!((p.mu - DVec3::from_array(o.mu[i])).length() < 1e-12);
            }
            let (ed, _) = ev.fields(None).unwrap();
            let (dip, _, _) = ev.converge(&ed, None).unwrap();
            for (i, d) in dip.iter().enumerate() {
                assert!((*d - DVec3::from_array(o.induced[i])).length() < 2e-11);
            }
            assert!((ev.pairwise(None).unwrap().0 - o.permanent).abs() < 1e-10);
            assert!((ev.polarization(None).unwrap().energy - o.polarization).abs() < 1e-9);
        }
    }
}

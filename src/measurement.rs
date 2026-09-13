use glam::Vec3;

use crate::{DisplayColor, VisibilityOverride};

pub const DEFAULT_MEASUREMENT_COLOR: DisplayColor = [1.0, 0.84, 0.08, 1.0];
pub const DEFAULT_MEASUREMENT_LABEL_SIZE: f32 = 16.0;
pub const DEFAULT_MEASUREMENT_THICKNESS: f32 = 0.055;
pub const MAX_MEASUREMENT_THICKNESS: f32 = 10.0;

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementEndpoint {
    pub position: Vec3,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementLine {
    pub hydrogen_bonds: Option<crate::molecule::amoeba::HydrogenBondReport>,
    pub id: u64,
    pub name: String,
    pub first: MeasurementEndpoint,
    pub second: MeasurementEndpoint,
    pub color: Option<DisplayColor>,
    pub visibility: VisibilityOverride,
    pub label_size: f32,
    pub thickness: f32,
}

impl MeasurementLine {
    pub fn new(id: u64, first: MeasurementEndpoint, second: MeasurementEndpoint) -> Self {
        Self {
            hydrogen_bonds: None,
            id,
            name: format!("Line #{id}"),
            first,
            second,
            color: None,
            visibility: VisibilityOverride::Inherit,
            label_size: DEFAULT_MEASUREMENT_LABEL_SIZE,
            thickness: DEFAULT_MEASUREMENT_THICKNESS,
        }
    }

    /// A measurement object may contain a scored set of lines sharing one style.
    pub fn segments(&self) -> Box<dyn Iterator<Item = (Vec3, Vec3)> + '_> {
        match &self.hydrogen_bonds {
            Some(report) => {
                let mut seen = std::collections::BTreeSet::new();
                Box::new(report.displayed().filter_map(move |c| {
                    let key = (c.donor.min(c.acceptor), c.donor.max(c.acceptor));
                    seen.insert(key).then(|| {
                        (
                            Vec3::from_array(c.donor_position),
                            Vec3::from_array(c.acceptor_position),
                        )
                    })
                }))
            }
            None => Box::new(std::iter::once((self.first.position, self.second.position))),
        }
    }

    pub fn distance(&self) -> f32 {
        self.first.position.distance(self.second.position)
    }

    pub fn midpoint(&self) -> Vec3 {
        (self.first.position + self.second.position) * 0.5
    }

    pub fn effective_color(&self) -> DisplayColor {
        self.color.unwrap_or(DEFAULT_MEASUREMENT_COLOR)
    }

    pub fn is_visible(&self) -> bool {
        self.visibility != VisibilityOverride::Hide
    }

    pub fn set_label_size(&mut self, size: f32) {
        self.label_size = size.clamp(8.0, 48.0);
    }

    pub fn set_thickness(&mut self, thickness: f32) {
        self.thickness = thickness.clamp(0.01, MAX_MEASUREMENT_THICKNESS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_reports_distance_midpoint_and_clamps_label_size() {
        let mut line = MeasurementLine::new(
            1,
            MeasurementEndpoint {
                position: Vec3::ZERO,
                description: "first".into(),
            },
            MeasurementEndpoint {
                position: Vec3::new(0.0, 3.0, 4.0),
                description: "second".into(),
            },
        );
        assert_eq!(line.distance(), 5.0);
        assert_eq!(line.midpoint(), Vec3::new(0.0, 1.5, 2.0));
        line.set_label_size(100.0);
        assert_eq!(line.label_size, 48.0);
        line.set_thickness(100.0);
        assert_eq!(line.thickness, MAX_MEASUREMENT_THICKNESS);
    }
}

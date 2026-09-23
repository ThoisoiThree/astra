//! Measures molecular surface construction: `cargo run --release --example surface_timing
//! -- path/to/structure [resolution]`.

use std::{sync::atomic::AtomicBool, time::Instant};

use astra::{
    molecule::parse_structure,
    surface::{SurfaceKind, SurfaceSettings, SurfaceSphere, compute_surface},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let path = arguments
        .next()
        .unwrap_or_else(|| "examples/4R8P.pdb".into());
    let resolution = arguments
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(0.5);
    let bytes = std::fs::read(&path)?;
    let molecule = parse_structure(&bytes, &path)?.molecule;
    let spheres: Vec<_> = molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| !atom.hetero)
        .map(|(index, atom)| SurfaceSphere {
            center: atom.position,
            radius: atom.element.van_der_waals_radius(),
            atom: index as u32,
        })
        .collect();
    for kind in SurfaceKind::ALL {
        let settings = SurfaceSettings {
            kind,
            resolution,
            ..SurfaceSettings::default()
        };
        let start = Instant::now();
        let mesh =
            compute_surface(&spheres, &settings, &AtomicBool::new(false)).ok_or("cancelled")?;
        println!(
            "{path}: {} surface of {} atoms: {} triangles, {:.0} Å², spacing {:.2} Å, {:.0} ms",
            kind.label(),
            spheres.len(),
            mesh.triangle_count(),
            mesh.area(),
            mesh.spacing,
            start.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}

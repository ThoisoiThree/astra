//! Command-line options. Without `--export` Astra opens its window as usual.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use astra::{
    DisplayMode,
    render::{ExportBackground, ImageExport},
};

use crate::ui::ExportRequest;

pub const USAGE: &str = "\
Usage: astra [STRUCTURE] [OPTIONS]

Opens STRUCTURE (PDB, mmCIF, BinaryCIF, PDBML, GRO, XYZ or an Astra .mol scene).

Batch rendering (renders STRUCTURE and exits):
  --export PATH.png        write a PNG image
  --size WIDTHxHEIGHT      image size in pixels (default 1920x1080)
  --supersampling N        samples per pixel along each axis, 1-4 (default 3)
  --background KIND        viewport, white, black or transparent (default viewport)
  --dpi N                  resolution recorded in the PNG (default 300)
  --mode MODE              cartoon, ball-and-stick, licorice, spacefill or toon
  --assembly ID            render biological assembly ID instead of the file's model
  --command CMD            run an Astra command before rendering; repeatable, for example
                           --command \"show surface, protein\" --command \"show labels, ligand\"

  -h, --help               show this help
  -V, --version            show the version";

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub path: Option<PathBuf>,
    pub batch: Option<BatchExport>,
}

#[derive(Debug, Clone)]
pub struct BatchExport {
    pub output: PathBuf,
    pub request: ExportRequest,
    pub mode: Option<DisplayMode>,
    pub assembly: Option<String>,
    /// Commands such as `show surface, protein`, run in order before rendering.
    pub commands: Vec<String>,
}

pub enum Parsed {
    Run(Options),
    Exit(String),
}

pub fn parse(arguments: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Parsed> {
    let mut options = Options::default();
    let mut output = None;
    let mut request = ExportRequest {
        image: ImageExport {
            width: 1920,
            height: 1080,
            supersampling: 3,
            background: ExportBackground::Scene,
        },
        dots_per_inch: 300,
    };
    let mut mode = None;
    let mut assembly = None;
    let mut commands = Vec::new();
    let mut batch_option_used = false;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let text = argument.to_string_lossy().into_owned();
        let mut value = |name: &str| -> Result<String> {
            arguments
                .next()
                .map(|value| value.to_string_lossy().into_owned())
                .with_context(|| format!("{name} requires a value"))
        };
        match text.as_str() {
            "-h" | "--help" => return Ok(Parsed::Exit(USAGE.into())),
            "-V" | "--version" => {
                return Ok(Parsed::Exit(format!("astra {}", env!("CARGO_PKG_VERSION"))));
            }
            "--export" => output = Some(PathBuf::from(value("--export")?)),
            "--size" => {
                let size = value("--size")?;
                let (width, height) = size
                    .split_once(['x', 'X'])
                    .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                    .with_context(|| format!("invalid size '{size}'; use WIDTHxHEIGHT"))?;
                if width == 0 || height == 0 {
                    bail!("image size must be positive");
                }
                request.image.width = width;
                request.image.height = height;
                batch_option_used = true;
            }
            "--supersampling" => {
                let scale: u32 = value("--supersampling")?
                    .parse()
                    .context("--supersampling expects a number")?;
                if !(1..=astra::render::MAX_RENDER_SCALE).contains(&scale) {
                    bail!("--supersampling must be between 1 and 4");
                }
                request.image.supersampling = scale;
                batch_option_used = true;
            }
            "--background" => {
                request.image.background = match value("--background")?.as_str() {
                    "viewport" | "scene" => ExportBackground::Scene,
                    "white" => ExportBackground::White,
                    "black" => ExportBackground::Black,
                    "transparent" => ExportBackground::Transparent,
                    other => bail!("unknown background '{other}'"),
                };
                batch_option_used = true;
            }
            "--dpi" => {
                request.dots_per_inch =
                    value("--dpi")?.parse().context("--dpi expects a number")?;
                batch_option_used = true;
            }
            "--mode" => {
                mode = Some(match value("--mode")?.as_str() {
                    "cartoon" => DisplayMode::Cartoon,
                    "ball-and-stick" | "ballandstick" | "sticks" => DisplayMode::BallAndStick,
                    "licorice" => DisplayMode::Licorice,
                    "spacefill" | "spheres" => DisplayMode::Spacefill,
                    "toon" => DisplayMode::Toon,
                    other => bail!("unknown mode '{other}'"),
                });
                batch_option_used = true;
            }
            "--assembly" => {
                assembly = Some(value("--assembly")?);
                batch_option_used = true;
            }
            "--command" => {
                let command = value("--command")?;
                astra::command::parse_command(&command)
                    .with_context(|| format!("invalid --command '{command}'"))?;
                commands.push(command);
                batch_option_used = true;
            }
            option if option.starts_with("--") => bail!("unknown option '{option}'\n\n{USAGE}"),
            _ => {
                if options.path.is_some() {
                    bail!("only one structure can be given");
                }
                options.path = Some(PathBuf::from(argument));
            }
        }
    }
    match output {
        Some(output) => {
            if options.path.is_none() {
                bail!("--export requires a structure file");
            }
            options.batch = Some(BatchExport {
                output,
                request,
                mode,
                assembly,
                commands,
            });
        }
        None if batch_option_used => bail!("rendering options require --export"),
        None => {}
    }
    Ok(Parsed::Run(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(arguments: &[&str]) -> Result<Options> {
        match parse(arguments.iter().map(std::ffi::OsString::from))? {
            Parsed::Run(options) => Ok(options),
            Parsed::Exit(_) => bail!("unexpected exit"),
        }
    }

    #[test]
    fn parses_batch_rendering_options() {
        let options = run(&[
            "1abc.cif",
            "--export",
            "out.png",
            "--size",
            "800x600",
            "--background",
            "transparent",
            "--mode",
            "spacefill",
            "--assembly",
            "1",
            "--command",
            "show surface, protein",
        ])
        .unwrap();
        assert_eq!(options.path, Some(PathBuf::from("1abc.cif")));
        let batch = options.batch.unwrap();
        assert_eq!(batch.request.image.width, 800);
        assert_eq!(
            batch.request.image.background,
            ExportBackground::Transparent
        );
        assert_eq!(batch.mode, Some(DisplayMode::Spacefill));
        assert_eq!(batch.assembly.as_deref(), Some("1"));
        assert_eq!(batch.commands, ["show surface, protein"]);
        assert!(
            run(&[
                "a.pdb",
                "--export",
                "b.png",
                "--command",
                "show nothing, all"
            ])
            .is_err()
        );
    }

    #[test]
    fn rejects_inconsistent_arguments() {
        assert!(run(&["--export", "out.png"]).is_err());
        assert!(run(&["a.pdb", "--size", "10x10"]).is_err());
        assert!(run(&["a.pdb", "--size", "10"]).is_err());
        assert!(run(&["a.pdb", "--frobnicate"]).is_err());
        assert!(run(&["a.pdb"]).unwrap().batch.is_none());
    }
}

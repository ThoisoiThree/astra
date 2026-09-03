mod cartoon;
mod instances;
mod mesh;
mod pipelines;
mod postprocess;
mod renderer;
mod targets;

pub use renderer::{PreparedCartoon, RenderError, Renderer, SurfaceIssue, prepare_cartoon};

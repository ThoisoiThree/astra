mod cartoon;
mod instances;
mod mesh;
mod pipelines;
mod postprocess;
mod profiling;
mod renderer;
mod targets;

pub use renderer::{
    PreparedCartoon, RenderError, RenderStats, Renderer, SurfaceIssue, prepare_cartoon,
    prepare_cartoon_cached,
};

#[cfg(test)]
mod shader_tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn bundled_wgsl_modules_parse_and_validate() {
        for (name, source) in [
            ("geometry", include_str!("shader.wgsl")),
            ("toon", include_str!("toon_sphere.wgsl")),
            ("postprocess", include_str!("postprocess.wgsl")),
            ("ambient occlusion", include_str!("ambient_occlusion.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} shader did not parse: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} shader did not validate: {error:#?}"));
        }
    }
}

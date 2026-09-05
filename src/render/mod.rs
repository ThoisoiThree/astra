mod cartoon;
mod dof;
mod instances;
mod mesh;
mod pipelines;
mod postprocess;
mod profiling;
mod renderer;
mod targets;
mod viewport_cache;

pub use renderer::{
    PreparedCartoon, RenderError, RenderStats, Renderer, SurfaceIssue, prepare_cartoon,
    prepare_cartoon_cached,
};

#[cfg(test)]
mod shader_tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn bundled_wgsl_modules_parse_and_validate() {
        let optics = include_str!("optics.wgsl");
        let peel = include_str!("peel.wgsl");
        for (name, source) in [
            (
                "geometry",
                [optics, peel, include_str!("shader.wgsl")].concat(),
            ),
            (
                "toon",
                [optics, peel, include_str!("toon_sphere.wgsl")].concat(),
            ),
            (
                "postprocess",
                [optics, include_str!("postprocess.wgsl")].concat(),
            ),
            (
                "tiled DOF",
                [
                    optics,
                    include_str!("dof.wgsl"),
                    include_str!("dof_reduce.wgsl"),
                ]
                .concat(),
            ),
            (
                "ambient occlusion",
                [optics, include_str!("ambient_occlusion.wgsl")].concat(),
            ),
        ] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name} shader did not parse: {error:?}"));
            let (_, uniform) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some("PostUniform"))
                .expect("shared postprocess uniform");
            let naga::TypeInner::Struct { members, span } = &uniform.inner else {
                panic!("uniform must be a struct")
            };
            assert_eq!(
                *span as usize,
                std::mem::size_of::<super::postprocess::PostUniform>(),
                "{name} uniform size"
            );
            for (member, offset) in members.iter().zip([
                std::mem::offset_of!(super::postprocess::PostUniform, inverse_view_projection),
                std::mem::offset_of!(super::postprocess::PostUniform, eye_position),
                std::mem::offset_of!(super::postprocess::PostUniform, optical_axis),
                std::mem::offset_of!(super::postprocess::PostUniform, lens),
                std::mem::offset_of!(super::postprocess::PostUniform, aperture),
                std::mem::offset_of!(super::postprocess::PostUniform, ao),
                std::mem::offset_of!(super::postprocess::PostUniform, quality),
                std::mem::offset_of!(super::postprocess::PostUniform, background),
                std::mem::offset_of!(super::postprocess::PostUniform, viewport),
            ]) {
                assert_eq!(member.offset as usize, offset, "{name}: {:?}", member.name);
            }
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} shader did not validate: {error:#?}"));
        }
    }
}

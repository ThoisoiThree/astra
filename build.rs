fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the resolved version rather than duplicating the Cargo dependency range.
    let lock = std::fs::read_to_string("Cargo.lock")?;
    let wgpu_version = lock
        .split("[[package]]")
        .find(|package| package.lines().any(|line| line.trim() == "name = \"wgpu\""))
        .and_then(|package| {
            package
                .lines()
                .find_map(|line| line.trim().strip_prefix("version = \"")?.strip_suffix('"'))
        })
        .ok_or("could not find the resolved wgpu version in Cargo.lock")?;
    println!("cargo:rustc-env=ASTRA_WGPU_VERSION={wgpu_version}");
    println!("cargo:rerun-if-changed=Cargo.lock");
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc);
    config.compile_protos(&["schemas/molecule_1_0.proto"], &["schemas"])?;
    println!("cargo:rerun-if-changed=schemas/molecule_1_0.proto");
    Ok(())
}

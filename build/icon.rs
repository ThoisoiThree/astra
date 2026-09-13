use std::{env, error::Error, fs, path::PathBuf};

use resvg::{tiny_skia, usvg};

pub fn generate() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=resources/logo/astra_icon.svg");
    println!("cargo:rerun-if-changed=build/icon.rs");
    let output = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?);
    let svg = fs::read("resources/logo/astra_icon.svg")?;
    let tree = usvg::Tree::from_data(&svg, &usvg::Options::default())?;
    let render = |size: u32| -> Result<tiny_skia::Pixmap, Box<dyn Error>> {
        let mut pixels = tiny_skia::Pixmap::new(size, size).ok_or("invalid icon dimensions")?;
        // Fit the non-square artwork into a transparent square without cropping.
        let scale = size as f32 / tree.size().width().max(tree.size().height());
        let transform = tiny_skia::Transform::from_row(
            scale,
            0.0,
            0.0,
            scale,
            (size as f32 - tree.size().width() * scale) * 0.5,
            (size as f32 - tree.size().height() * scale) * 0.5,
        );
        resvg::render(&tree, transform, &mut pixels.as_mut());
        Ok(pixels)
    };

    // winit expects straight alpha; tiny-skia stores premultiplied pixels.
    let pixels = render(64)?;
    let rgba: Vec<u8> = pixels
        .pixels()
        .iter()
        .flat_map(|pixel| {
            let color = pixel.demultiply();
            [color.red(), color.green(), color.blue(), color.alpha()]
        })
        .collect();
    fs::write(output.join("astra-icon.rgba"), rgba)?;

    if env::var("CARGO_CFG_TARGET_OS")? == "windows" {
        // ICO accepts PNG images, including the 256px Explorer representation.
        let sizes = [16u32, 24, 32, 48, 64, 128, 256];
        let images = sizes
            .iter()
            .map(|&size| Ok(render(size)?.encode_png()?))
            .collect::<Result<Vec<Vec<u8>>, Box<dyn Error>>>()?;
        let mut ico = vec![0, 0, 1, 0];
        ico.extend_from_slice(&(sizes.len() as u16).to_le_bytes());
        let mut offset = 6 + sizes.len() as u32 * 16;
        for (&size, png) in sizes.iter().zip(&images) {
            ico.extend_from_slice(&[size as u8, size as u8, 0, 0]);
            ico.extend_from_slice(&1u16.to_le_bytes());
            ico.extend_from_slice(&32u16.to_le_bytes());
            ico.extend_from_slice(&(png.len() as u32).to_le_bytes());
            ico.extend_from_slice(&offset.to_le_bytes());
            offset += png.len() as u32;
        }
        for png in images {
            ico.extend_from_slice(&png);
        }
        let path = output.join("astra.ico");
        fs::write(&path, ico)?;
        winresource::WindowsResource::new()
            .set_icon(path.to_str().ok_or("non-UTF-8 icon resource path")?)
            .compile()?;
    }

    if env::var("CARGO_CFG_TARGET_OS")? == "macos" {
        let mut chunks = Vec::new();
        for (size, tag) in [
            (128, b"ic07"),
            (256, b"ic08"),
            (512, b"ic09"),
            (1024, b"ic10"),
        ] {
            let png = render(size)?.encode_png()?;
            chunks.extend_from_slice(tag);
            chunks.extend_from_slice(&(png.len() as u32 + 8).to_be_bytes());
            chunks.extend_from_slice(&png);
        }
        let mut icns = b"icns".to_vec();
        icns.extend_from_slice(&(chunks.len() as u32 + 8).to_be_bytes());
        icns.extend_from_slice(&chunks);
        fs::write(output.join("astra.icns"), icns)?;
    }
    Ok(())
}

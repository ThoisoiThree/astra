//! PNG encoding of rendered images.

use std::io::Cursor;

use crate::render::RenderedImage;

/// Encodes an 8-bit sRGB RGBA image as PNG, recording the sRGB color space and a resolution
/// of `dots_per_inch` for page layout software.
pub fn encode_png(
    image: &RenderedImage,
    dots_per_inch: u32,
) -> Result<Vec<u8>, png::EncodingError> {
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(Cursor::new(&mut output), image.width, image.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let pixels_per_meter = (f64::from(dots_per_inch.max(1)) / 0.0254).round() as u32;
        encoder.set_pixel_dims(Some(png::PixelDimensions {
            xppu: pixels_per_meter,
            yppu: pixels_per_meter,
            unit: png::Unit::Meter,
        }));
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&image.rgba)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_rgba_pixels() {
        let image = RenderedImage {
            width: 2,
            height: 1,
            rgba: vec![255, 0, 0, 255, 0, 0, 255, 128],
            supersampling: 1,
        };
        let bytes = encode_png(&image, 300).unwrap();
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels).unwrap();
        assert_eq!((info.width, info.height), (2, 1));
        assert_eq!(&pixels[..info.buffer_size()], image.rgba.as_slice());
        let dims = reader.info().pixel_dims.unwrap();
        assert_eq!(dims.xppu, 11_811);
    }
}

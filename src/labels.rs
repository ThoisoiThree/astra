//! Text labels attached to atoms and residues, drawn in the viewport and in exported
//! images.

use std::collections::HashMap;

use glam::{Mat4, Vec3};

use crate::{DisplayState, RepresentationMask, molecule::Molecule};

/// What a label shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabelContent {
    /// One label per residue, for example `LYS 42`.
    #[default]
    Residue,
    /// Atom name, for example `CA`.
    Atom,
    /// Chain, residue and atom, for example `A/LYS 42/NZ`.
    Full,
    Element,
}

impl LabelContent {
    pub const ALL: [Self; 4] = [Self::Residue, Self::Atom, Self::Full, Self::Element];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Residue => "Residue",
            Self::Atom => "Atom name",
            Self::Full => "Chain/residue/atom",
            Self::Element => "Element",
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::Residue => 0,
            Self::Atom => 1,
            Self::Full => 2,
            Self::Element => 3,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Residue),
            1 => Some(Self::Atom),
            2 => Some(Self::Full),
            3 => Some(Self::Element),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelSettings {
    pub content: LabelContent,
    /// Text height in points on screen.
    pub size: f32,
    /// sRGB text color.
    pub color: [u8; 3],
    /// Draw a dark rounded box behind each label.
    pub background: bool,
}

impl Default for LabelSettings {
    fn default() -> Self {
        Self {
            content: LabelContent::Residue,
            size: 14.0,
            color: [255, 255, 255],
            background: true,
        }
    }
}

impl LabelSettings {
    pub const SIZE_RANGE: std::ops::RangeInclusive<f32> = 6.0..=48.0;

    pub fn sanitized(self) -> Self {
        Self {
            size: if self.size.is_finite() {
                self.size
                    .clamp(*Self::SIZE_RANGE.start(), *Self::SIZE_RANGE.end())
            } else {
                Self::default().size
            },
            ..self
        }
    }
}

/// Labels beyond this count are not drawn; text layout would dominate the frame time.
pub const MAX_DRAWN_LABELS: usize = 5000;

#[derive(Debug, Clone, PartialEq)]
pub struct LabelItem {
    pub position: Vec3,
    pub text: String,
}

/// Labels for every visible atom carrying the label representation.
pub fn label_items(molecule: &Molecule, display: &DisplayState) -> Vec<LabelItem> {
    let labeled = |index: usize| {
        display.visible.contains(index)
            && display
                .representations
                .get(index)
                .is_some_and(|mask| mask.contains(RepresentationMask::LABEL))
    };
    let content = display.labels.content;
    if content != LabelContent::Residue {
        return molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| labeled(*index))
            .map(|(_, atom)| LabelItem {
                position: atom.position,
                text: match content {
                    LabelContent::Atom => atom.name.clone(),
                    LabelContent::Element => atom.element.symbol().to_owned(),
                    _ => format!(
                        "{}/{} {}{}/{}",
                        atom.chain_id,
                        atom.residue_name,
                        atom.residue_number,
                        atom.insertion_code.map(String::from).unwrap_or_default(),
                        atom.name
                    ),
                },
            })
            .collect();
    }
    // One label per residue, anchored on the main-chain atom when it is labeled.
    let mut residues: Vec<(String, Vec<usize>)> = Vec::new();
    let mut lookup = HashMap::<(&str, i32, Option<char>, &str), usize>::new();
    for (index, atom) in molecule.atoms.iter().enumerate() {
        if !labeled(index) {
            continue;
        }
        let key = (
            atom.chain_id.as_str(),
            atom.residue_number,
            atom.insertion_code,
            atom.residue_name.as_str(),
        );
        let slot = *lookup.entry(key).or_insert_with(|| {
            residues.push((
                format!(
                    "{} {}{}",
                    atom.residue_name,
                    atom.residue_number,
                    atom.insertion_code.map(String::from).unwrap_or_default()
                ),
                Vec::new(),
            ));
            residues.len() - 1
        });
        residues[slot].1.push(index);
    }
    residues
        .into_iter()
        .map(|(text, atoms)| {
            let anchor = atoms
                .iter()
                .copied()
                .find(|&index| matches!(molecule.atoms[index].name.as_str(), "CA" | "C1'" | "C1*"));
            let position = anchor.map_or_else(
                || {
                    atoms
                        .iter()
                        .map(|&index| molecule.atoms[index].position)
                        .sum::<Vec3>()
                        / atoms.len() as f32
                },
                |index| molecule.atoms[index].position,
            );
            LabelItem { position, text }
        })
        .collect()
}

/// Paints labels projected with `view_projection` into `viewport` (in points). `scale`
/// multiplies the text size, for exports larger than the screen.
pub fn paint_labels(
    painter: &egui::Painter,
    viewport: egui::Rect,
    view_projection: Mat4,
    items: &[LabelItem],
    settings: &LabelSettings,
    scale: f32,
) {
    if !viewport.is_positive() || items.is_empty() {
        return;
    }
    let settings = settings.sanitized();
    let mut projected: Vec<(f32, egui::Pos2, &str)> = items
        .iter()
        .filter_map(|item| {
            let clip = view_projection * item.position.extend(1.0);
            if clip.w <= 0.0 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            ((-1.0..=1.0).contains(&ndc.x)
                && (-1.0..=1.0).contains(&ndc.y)
                && (0.0..=1.0).contains(&ndc.z))
            .then(|| {
                (
                    ndc.z,
                    egui::pos2(
                        viewport.left() + (ndc.x + 1.0) * 0.5 * viewport.width(),
                        viewport.top() + (1.0 - ndc.y) * 0.5 * viewport.height(),
                    ),
                    item.text.as_str(),
                )
            })
        })
        .collect();
    // Nearest labels are drawn last so they stay readable where labels overlap.
    projected.sort_by(|a, b| b.0.total_cmp(&a.0));
    let skip = projected.len().saturating_sub(MAX_DRAWN_LABELS);
    let [r, g, b] = settings.color;
    let text_color = egui::Color32::from_rgb(r, g, b);
    let font = egui::FontId::proportional(settings.size * scale);
    for (_, center, text) in projected.into_iter().skip(skip) {
        let galley = painter.layout_no_wrap(text.to_owned(), font.clone(), text_color);
        // Labels sit just above and to the right of their atom.
        let offset = egui::vec2(4.0, -4.0) * scale;
        let rect = egui::Rect::from_min_size(
            center + offset - egui::vec2(0.0, galley.size().y),
            galley.size(),
        );
        if settings.background {
            painter.rect_filled(
                rect.expand2(egui::vec2(4.0, 2.0) * scale),
                3.0 * scale,
                egui::Color32::from_rgba_unmultiplied(22, 25, 27, 200),
            );
        }
        painter.galley(rect.min, galley, text_color);
    }
}

/// Draws labels over an exported image (8-bit sRGB, straight alpha, rows top to bottom).
/// Text is laid out by egui at the export's pixel density and rasterized on the CPU, so
/// labels look the same on every GPU backend.
pub fn draw_labels_on_image(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    view_projection: Mat4,
    items: &[LabelItem],
    settings: &LabelSettings,
    scale: f32,
) {
    if items.is_empty() || width == 0 || height == 0 {
        return;
    }
    let pixels_per_point = scale.max(0.1);
    let context = egui::Context::default();
    let screen = egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(width as f32, height as f32) / pixels_per_point,
    );
    let mut input = egui::RawInput {
        screen_rect: Some(screen),
        max_texture_side: Some(8192),
        ..Default::default()
    };
    input
        .viewports
        .entry(egui::ViewportId::ROOT)
        .or_default()
        .native_pixels_per_point = Some(pixels_per_point);
    let mut output = context.run_ui(input, |ui| {
        let painter = ui.ctx().layer_painter(egui::LayerId::background());
        paint_labels(&painter, screen, view_projection, items, settings, 1.0);
    });
    let mut textures = HashMap::<egui::TextureId, egui::ColorImage>::new();
    let deltas = std::mem::take(&mut output.textures_delta.set);
    output.textures_delta.clear();
    for (id, deltas) in &deltas {
        for delta in deltas {
            let egui::ImageData::Color(image) = &delta.image;
            match delta.pos {
                None => {
                    textures.insert(*id, egui::ColorImage::clone(image));
                }
                Some([x0, y0]) => {
                    if let Some(target) = textures.get_mut(id) {
                        for y in 0..image.size[1] {
                            for x in 0..image.size[0] {
                                if x0 + x < target.size[0] && y0 + y < target.size[1] {
                                    target.pixels[(y0 + y) * target.size[0] + x0 + x] =
                                        image.pixels[y * image.size[0] + x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let primitives =
        context.tessellate(std::mem::take(&mut output.shapes), output.pixels_per_point);
    let mut canvas = Canvas {
        rgba,
        width: width as usize,
        height: height as usize,
    };
    for primitive in primitives {
        let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
            continue;
        };
        let clip = primitive.clip_rect;
        canvas.draw_mesh(
            &mesh,
            textures.get(&mesh.texture_id),
            pixels_per_point,
            [
                clip.min.x * pixels_per_point,
                clip.min.y * pixels_per_point,
                clip.max.x * pixels_per_point,
                clip.max.y * pixels_per_point,
            ],
        );
    }
}

struct Canvas<'a> {
    rgba: &'a mut [u8],
    width: usize,
    height: usize,
}

impl Canvas<'_> {
    fn draw_mesh(
        &mut self,
        mesh: &egui::Mesh,
        texture: Option<&egui::ColorImage>,
        pixels_per_point: f32,
        clip: [f32; 4],
    ) {
        for triangle in mesh.indices.as_chunks::<3>().0 {
            let vertices = [0, 1, 2].map(|corner| &mesh.vertices[triangle[corner] as usize]);
            let points = vertices.map(|vertex| {
                [
                    vertex.pos.x * pixels_per_point,
                    vertex.pos.y * pixels_per_point,
                ]
            });
            let area = edge(points[0], points[1], points[2]);
            if area.abs() < 1e-8 {
                continue;
            }
            let min_x = points
                .iter()
                .map(|p| p[0])
                .fold(f32::INFINITY, f32::min)
                .max(clip[0]);
            let max_x = points
                .iter()
                .map(|p| p[0])
                .fold(f32::NEG_INFINITY, f32::max)
                .min(clip[2]);
            let min_y = points
                .iter()
                .map(|p| p[1])
                .fold(f32::INFINITY, f32::min)
                .max(clip[1]);
            let max_y = points
                .iter()
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max)
                .min(clip[3]);
            if min_x >= max_x || min_y >= max_y {
                continue;
            }
            let x_range =
                (min_x.floor().max(0.0) as usize)..(max_x.ceil().max(0.0) as usize).min(self.width);
            let y_range = (min_y.floor().max(0.0) as usize)
                ..(max_y.ceil().max(0.0) as usize).min(self.height);
            for y in y_range {
                for x in x_range.clone() {
                    let sample = [x as f32 + 0.5, y as f32 + 0.5];
                    let weights = [
                        edge(points[1], points[2], sample) / area,
                        edge(points[2], points[0], sample) / area,
                        edge(points[0], points[1], sample) / area,
                    ];
                    if weights.iter().any(|weight| *weight < -1e-5) {
                        continue;
                    }
                    let mut color = [0.0_f32; 4];
                    let mut uv = [0.0_f32; 2];
                    for (vertex, weight) in vertices.iter().zip(weights) {
                        for (channel, value) in color.iter_mut().zip(vertex.color.to_array()) {
                            *channel += weight * f32::from(value) / 255.0;
                        }
                        uv[0] += weight * vertex.uv.x;
                        uv[1] += weight * vertex.uv.y;
                    }
                    if let Some(texture) = texture {
                        let texel = sample_bilinear(texture, uv);
                        for (channel, value) in color.iter_mut().zip(texel) {
                            *channel *= value;
                        }
                    }
                    self.blend(x, y, color);
                }
            }
        }
    }

    /// Source-over blending of a premultiplied color onto the straight-alpha image.
    fn blend(&mut self, x: usize, y: usize, source: [f32; 4]) {
        let offset = (y * self.width + x) * 4;
        let pixel = &mut self.rgba[offset..offset + 4];
        let destination_alpha = f32::from(pixel[3]) / 255.0;
        let alpha = source[3] + destination_alpha * (1.0 - source[3]);
        if alpha <= 0.0 {
            return;
        }
        for channel in 0..3 {
            let destination = f32::from(pixel[channel]) / 255.0 * destination_alpha;
            let premultiplied = source[channel] + destination * (1.0 - source[3]);
            pixel[channel] = ((premultiplied / alpha).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        pixel[3] = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
}

fn edge(a: [f32; 2], b: [f32; 2], point: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
}

fn sample_bilinear(texture: &egui::ColorImage, [u, v]: [f32; 2]) -> [f32; 4] {
    let [width, height] = texture.size;
    if width == 0 || height == 0 {
        return [1.0; 4];
    }
    let x = (u * width as f32 - 0.5).clamp(0.0, (width - 1) as f32);
    let y = (v * height as f32 - 0.5).clamp(0.0, (height - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let texel = |x: usize, y: usize| {
        texture.pixels[y * width + x]
            .to_array()
            .map(|value| f32::from(value) / 255.0)
    };
    let [a, b, c, d] = [texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1)];
    std::array::from_fn(|channel| {
        let top = a[channel] + (b[channel] - a[channel]) * tx;
        let bottom = c[channel] + (d[channel] - c[channel]) * tx;
        top + (bottom - top) * ty
    })
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Element};

    fn molecule() -> Molecule {
        let atom = |name: &str, residue: i32, x: f32| Atom {
            name: name.into(),
            element: Element::C,
            residue_name: "LYS".into(),
            residue_number: residue,
            chain_id: "A".into(),
            position: Vec3::new(x, 0.0, 0.0),
            ..Default::default()
        };
        Molecule::new(
            vec![
                atom("N", 1, 0.0),
                atom("CA", 1, 1.0),
                atom("CB", 1, 2.0),
                atom("CA", 2, 5.0),
            ],
            Vec::new(),
        )
    }

    #[test]
    fn residue_labels_anchor_on_the_main_chain() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        assert!(label_items(&molecule, &display).is_empty());
        display.set_representation([0, 1, 2], RepresentationMask::LABEL, true);
        let items = label_items(&molecule, &display);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "LYS 1");
        assert_eq!(items[0].position, Vec3::X);
        display.labels.content = LabelContent::Full;
        let items = label_items(&molecule, &display);
        assert_eq!(items.len(), 3);
        assert_eq!(items[2].text, "A/LYS 1/CB");
    }

    #[test]
    fn exported_labels_are_rasterized_at_their_atoms() {
        let (width, height) = (160_u32, 80_u32);
        let mut rgba = vec![0_u8; (width * height * 4) as usize];
        let items = [LabelItem {
            position: Vec3::ZERO,
            text: "LYS 42".into(),
        }];
        draw_labels_on_image(
            &mut rgba,
            width,
            height,
            Mat4::IDENTITY,
            &items,
            &LabelSettings::default(),
            2.0,
        );
        let covered = |x0: u32, x1: u32, y0: u32, y1: u32| {
            (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .filter(|&(x, y)| rgba[((y * width + x) * 4 + 3) as usize] > 0)
                .count()
        };
        // The label box starts at the image center and extends up and to the right.
        assert!(covered(80, 160, 0, 40) > 500);
        assert_eq!(covered(0, 70, 0, 80), 0);
        assert_eq!(covered(0, 160, 50, 80), 0);
        // White text pixels exist inside the dark box.
        assert!(
            rgba.as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[0] > 200 && pixel[1] > 200 && pixel[3] > 200)
        );
    }
}

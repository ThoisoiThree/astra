use glam::{Mat4, Vec2, Vec3};

use crate::picking::Ray;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthOfField {
    pub enabled: bool,
    pub focus_point: Vec3,
    /// Lens focal length and sensor height define the projection and optical CoC.
    pub focal_length_mm: f32,
    pub sensor_height_mm: f32,
    /// Physical lens f-number. Lower values produce a wider aperture.
    pub f_stop: f32,
    /// Zero selects a circular iris; 3-12 selects a polygonal diaphragm.
    pub blade_count: u32,
    pub blade_rotation: f32,
    /// Safety/performance clamp for the physical circle of confusion.
    pub max_coc_pixels: f32,
}

impl Default for DepthOfField {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_point: Vec3::ZERO,
            focal_length_mm: 60.0,
            sensor_height_mm: 24.0,
            f_stop: 2.0,
            blade_count: 7,
            blade_rotation: 0.0,
            max_coc_pixels: 24.0,
        }
    }
}

impl Viewport {
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: width.max(1) as f32,
            height: height.max(1) as f32,
        }
    }

    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.x
            && point.y >= self.y
            && point.x < self.x + self.width
            && point.y < self.y + self.height
    }
}

#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub aspect: f32,
    pub field_of_view_y: f32,
    pub near: f32,
    pub far: f32,
    pub depth_of_field: DepthOfField,
}

impl OrbitCamera {
    pub fn new(aspect: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 12.0,
            yaw: 0.65,
            pitch: 0.35,
            aspect: aspect.max(0.01),
            field_of_view_y: 2.0 * (24.0_f32 / (2.0 * 60.0)).atan(),
            near: 1.0,
            far: 1_000.0,
            depth_of_field: DepthOfField::default(),
        }
    }

    pub fn eye(&self) -> Vec3 {
        let direction = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );
        self.target + direction * self.distance
    }

    pub fn view_projection(&self) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y);
        let projection = glam::camera::rh::proj::directx::perspective(
            self.field_of_view_y,
            self.aspect.max(0.01),
            self.near,
            self.far,
        );
        projection * view
    }

    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw -= delta.x * 0.008;
        self.pitch = (self.pitch + delta.y * 0.008).clamp(-1.54, 1.54);
    }

    pub fn pan(&mut self, delta: Vec2, viewport_height: f32) {
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let world_per_pixel =
            2.0 * self.distance * (self.field_of_view_y * 0.5).tan() / viewport_height.max(1.0);
        self.target += (-right * delta.x + up * delta.y) * world_per_pixel;
    }

    /// Changes the orbit pivot while preserving the current eye position.
    pub fn set_pivot(&mut self, pivot: Vec3) {
        let eye = self.eye();
        let offset = eye - pivot;
        let distance = offset.length();
        if distance <= f32::EPSILON {
            return;
        }
        let direction = offset / distance;
        self.target = pivot;
        self.distance = distance;
        self.pitch = direction.y.asin().clamp(-1.54, 1.54);
        self.yaw = direction.x.atan2(direction.z);
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        self.distance = (self.distance * (-scroll_delta * 0.0015).exp()).clamp(0.1, 50_000.0);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width.max(1) as f32 / height.max(1) as f32;
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.aspect = viewport.width.max(1.0) / viewport.height.max(1.0);
    }

    pub fn set_clip_planes(&mut self, near: f32, far: f32) {
        self.near = near.clamp(0.001, 100_000.0);
        self.far = far.clamp(self.near + 0.001, 1_000_000.0);
    }

    pub fn focus_distance(&self) -> f32 {
        self.eye().distance(self.depth_of_field.focus_point)
    }

    pub fn optical_axis(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or_zero()
    }

    /// Focus distance measured along the optical axis, defining a focal plane.
    pub fn focus_depth(&self) -> f32 {
        (self.depth_of_field.focus_point - self.eye())
            .dot(self.optical_axis())
            .max(0.001)
    }

    pub fn set_lens(&mut self, focal_length_mm: f32, sensor_height_mm: f32) {
        self.depth_of_field.focal_length_mm = focal_length_mm.clamp(5.0, 400.0);
        self.depth_of_field.sensor_height_mm = sensor_height_mm.clamp(4.0, 70.0);
        self.field_of_view_y = 2.0
            * (self.depth_of_field.sensor_height_mm / (2.0 * self.depth_of_field.focal_length_mm))
                .atan();
    }

    /// Signed thin-lens circle-of-confusion radius in physical output pixels.
    /// Negative values are in front of the focal plane, positive values behind it.
    pub fn circle_of_confusion_pixels(&self, object_distance: f32, image_height: f32) -> f32 {
        let focal_length =
            self.depth_of_field.focal_length_mm / self.depth_of_field.sensor_height_mm;
        let focus = self.focus_depth().max(focal_length + 1e-4);
        let object_distance = object_distance.max(focal_length + 1e-4);
        let sensor_coc = 0.5 * focal_length * focal_length * (1.0 - focus / object_distance)
            / (self.depth_of_field.f_stop.max(0.1) * (focus - focal_length));
        (sensor_coc * image_height).clamp(
            -self.depth_of_field.max_coc_pixels,
            self.depth_of_field.max_coc_pixels,
        )
    }

    pub fn screen_ray(&self, position: Vec2, viewport: Viewport) -> Option<Ray> {
        if !viewport.contains(position) {
            return None;
        }
        let normalized_x = (position.x - viewport.x) / viewport.width.max(1.0);
        let normalized_y = (position.y - viewport.y) / viewport.height.max(1.0);
        let clip_x = normalized_x * 2.0 - 1.0;
        let clip_y = 1.0 - normalized_y * 2.0;
        let inverse = self.view_projection().inverse();
        if !inverse.is_finite() {
            return None;
        }
        let near = inverse.project_point3(Vec3::new(clip_x, clip_y, 0.0));
        let far = inverse.project_point3(Vec3::new(clip_x, clip_y, 1.0));
        let direction = (far - near).normalize_or_zero();
        (direction != Vec3::ZERO).then_some(Ray {
            origin: self.eye(),
            direction,
        })
    }

    /// Frames a set of points by their bounding sphere around the box center, which is
    /// tighter than the box diagonal for elongated or sparse structures.
    pub fn fit_points(&mut self, points: impl IntoIterator<Item = Vec3> + Clone) {
        let (minimum, maximum) = points.clone().into_iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(low, high), point| (low.min(point), high.max(point)),
        );
        if minimum.x > maximum.x {
            return;
        }
        let center = (minimum + maximum) * 0.5;
        let radius = points
            .into_iter()
            .map(|point| point.distance(center))
            .fold(0.0_f32, f32::max)
            .max(0.8);
        self.target = center;
        self.distance = (radius / (self.limiting_fov() * 0.5).sin() * 1.08).max(2.0);
    }

    fn limiting_fov(&self) -> f32 {
        if self.aspect < 1.0 {
            2.0 * ((self.field_of_view_y * 0.5).tan() * self.aspect).atan()
        } else {
            self.field_of_view_y
        }
    }

    pub fn fit_bounds(&mut self, minimum: Vec3, maximum: Vec3) {
        self.target = (minimum + maximum) * 0.5;
        let radius = ((maximum - minimum).length() * 0.5).max(0.8);
        let limiting_fov = if self.aspect < 1.0 {
            2.0 * ((self.field_of_view_y * 0.5).tan() * self.aspect).atan()
        } else {
            self.field_of_view_y
        };
        self.distance = (radius / (limiting_fov * 0.5).sin() * 1.15).max(2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_centers_and_frames_bounds() {
        let mut camera = OrbitCamera::new(1.0);
        camera.fit_bounds(Vec3::new(-2.0, -1.0, 0.0), Vec3::new(4.0, 3.0, 2.0));
        assert_eq!(camera.target, Vec3::new(1.0, 1.0, 1.0));
        assert!(camera.distance > 4.0);
        assert_eq!((camera.near, camera.far), (1.0, 1_000.0));
    }

    #[test]
    fn center_screen_ray_points_at_target() {
        let camera = OrbitCamera::new(1.0);
        let viewport = Viewport::full(800, 800);
        let ray = camera
            .screen_ray(Vec2::new(400.0, 400.0), viewport)
            .unwrap();
        let expected = (camera.target - camera.eye()).normalize();
        assert!(ray.direction.dot(expected) > 0.999);
    }

    #[test]
    fn clipping_planes_are_kept_valid() {
        let mut camera = OrbitCamera::new(1.0);
        camera.set_clip_planes(-1.0, 0.0);
        assert_eq!(camera.near, 0.001);
        assert!(camera.far > camera.near);
    }

    #[test]
    fn thin_lens_coc_is_signed_and_zero_at_focus() {
        let camera = OrbitCamera::new(1.0);
        let focus = camera.focus_distance();
        assert!(camera.circle_of_confusion_pixels(focus, 800.0).abs() < 1e-4);
        assert!(camera.circle_of_confusion_pixels(focus * 0.8, 800.0) < 0.0);
        assert!(camera.circle_of_confusion_pixels(focus * 1.2, 800.0) > 0.0);
    }

    #[test]
    fn coc_radius_matches_thin_lens_image_plane_geometry() {
        let mut camera = OrbitCamera::new(1.0);
        camera.set_lens(24.0, 24.0);
        camera.depth_of_field.f_stop = 2.0;
        camera.depth_of_field.max_coc_pixels = 1000.0;
        camera.depth_of_field.focus_point = camera.eye() + camera.optical_axis() * 10.0;
        // f = 1, aperture diameter = 1/2, image planes at 10/9 and 5/4:
        // the radius is (1/4) * (5/4 - 10/9) / (5/4) = 1/36 sensor heights.
        assert!((camera.circle_of_confusion_pixels(5.0, 900.0) + 25.0).abs() < 1e-3);
        assert!((camera.circle_of_confusion_pixels(20.0, 900.0) - 12.5).abs() < 1e-3);
        assert!((camera.circle_of_confusion_pixels(f32::INFINITY, 900.0) - 25.0).abs() < 1e-3);
        camera.depth_of_field.max_coc_pixels = 10.0;
        assert_eq!(camera.circle_of_confusion_pixels(5.0, 900.0), -10.0);
    }

    #[test]
    fn wider_aperture_produces_larger_circle_of_confusion() {
        let mut camera = OrbitCamera::new(1.0);
        camera.depth_of_field.max_coc_pixels = 1_000.0;
        let distance = camera.focus_distance() * 1.2;
        camera.depth_of_field.f_stop = 1.4;
        let wide = camera.circle_of_confusion_pixels(distance, 800.0).abs();
        camera.depth_of_field.f_stop = 8.0;
        let narrow = camera.circle_of_confusion_pixels(distance, 800.0).abs();
        assert!(wide > narrow);
    }

    #[test]
    fn changing_pivot_preserves_eye_position() {
        let mut camera = OrbitCamera::new(1.0);
        let eye = camera.eye();
        let pivot = Vec3::new(2.0, -1.0, 3.0);
        camera.set_pivot(pivot);
        assert!(camera.eye().distance(eye) < 1e-4);
        assert_eq!(camera.target, pivot);
    }
}

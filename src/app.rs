use std::{collections::BTreeMap, fs, path::Path, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use glam::{Vec2, Vec3};
use molview::{
    DisplayLevel, DisplayState, RepresentationMask,
    camera::{OrbitCamera, Viewport},
    command::{Command, Representation, parse_command},
    molecule::{Molecule, MoleculeHierarchy, parse_pdb},
    picking::pick_atom_filtered,
    render::{RenderError, Renderer, SurfaceIssue},
    selection::{Selection, evaluate_with_named},
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::ModifiersState,
    window::{Window, WindowAttributes, WindowId},
};

use crate::ui::{
    CameraUpdate, FocusRequest, InspectionTarget, ManagerAction, PivotRequest, UiActions, UiInfo,
    UiState,
};

pub fn run(initial_path: Option<PathBuf>) -> Result<()> {
    let event_loop = EventLoop::new().context("could not create the native event loop")?;
    let mut application = MolviewApplication {
        runtime: None,
        initial_path,
        startup_error: None,
    };
    event_loop
        .run_app(&mut application)
        .context("native event loop failed")?;
    if let Some(error) = application.startup_error {
        Err(error)
    } else {
        Ok(())
    }
}

struct MolviewApplication {
    runtime: Option<Runtime>,
    initial_path: Option<PathBuf>,
    startup_error: Option<anyhow::Error>,
}

impl ApplicationHandler for MolviewApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }
        match Runtime::new(event_loop, self.initial_path.take()) {
            Ok(runtime) => self.runtime = Some(runtime),
            Err(error) => {
                self.startup_error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(runtime) = &mut self.runtime
            && runtime.window.id() == window_id
        {
            runtime.window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}

struct Runtime {
    window: Arc<Window>,
    renderer: Renderer,
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    molecule: Option<Molecule>,
    hierarchy: Option<MoleculeHierarchy>,
    display: Option<DisplayState>,
    named_selections: BTreeMap<String, Selection>,
    inspection: Option<InspectionTarget>,
    focus_description: String,
    pivot_description: String,
    loaded_filename: Option<String>,
    camera: OrbitCamera,
    ui: UiState,
    cursor: Option<PhysicalPosition<f64>>,
    viewport: Viewport,
    left_drag: bool,
    left_drag_distance: f32,
    right_drag: bool,
    modifiers: ModifiersState,
}

impl Runtime {
    fn new(event_loop: &ActiveEventLoop, initial_path: Option<PathBuf>) -> Result<Self> {
        let attributes = WindowAttributes::default()
            .with_title("molview")
            .with_inner_size(LogicalSize::new(1280.0, 800.0))
            .with_min_inner_size(LogicalSize::new(720.0, 480.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("could not create the molview window")?,
        );
        let renderer = pollster::block_on(Renderer::new(window.clone()))
            .context("could not initialize wgpu")?;
        let size = renderer.size();
        let camera = OrbitCamera::new(size.width as f32 / size.height.max(1) as f32);
        let egui_context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let mut runtime = Self {
            window,
            renderer,
            egui_context,
            egui_state,
            molecule: None,
            hierarchy: None,
            display: None,
            named_selections: BTreeMap::new(),
            inspection: None,
            focus_description: "World origin".into(),
            pivot_description: "World origin".into(),
            loaded_filename: None,
            camera,
            ui: UiState::default(),
            cursor: None,
            viewport: Viewport::full(size.width, size.height),
            left_drag: false,
            left_drag_distance: 0.0,
            right_drag: false,
            modifiers: ModifiersState::default(),
        };
        if let Some(path) = initial_path
            && let Err(error) = runtime.load_path(&path)
        {
            runtime.ui.latest_error = Some(error.to_string());
        }
        runtime.window.request_redraw();
        Ok(runtime)
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let egui_response = self.egui_state.on_window_event(&self.window, &event);
        if egui_response.repaint {
            self.window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size);
                self.viewport = Viewport::full(size.width, size.height);
                self.camera.set_viewport(self.viewport);
                self.window.request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                if path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("pdb"))
                {
                    if let Err(error) = self.load_path(&path) {
                        self.ui.latest_error = Some(error.to_string());
                    }
                    self.window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left if pressed && !egui_response.consumed => {
                        self.left_drag = true;
                        self.left_drag_distance = 0.0;
                    }
                    MouseButton::Left if !pressed => {
                        let should_pick = self.left_drag
                            && self.left_drag_distance <= 4.0
                            && !egui_response.consumed
                            && !self.egui_context.is_pointer_over_egui();
                        self.left_drag = false;
                        if should_pick && let Some(position) = self.cursor {
                            self.pick_at(position);
                            self.window.request_redraw();
                        }
                    }
                    MouseButton::Right => self.right_drag = pressed && !egui_response.consumed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(previous) = self.cursor {
                    let delta = Vec2::new(
                        (position.x - previous.x) as f32,
                        (position.y - previous.y) as f32,
                    );
                    if !egui_response.consumed && !self.egui_context.is_pointer_over_egui() {
                        if self.left_drag {
                            self.left_drag_distance += delta.length();
                        }
                        let modifier_pan = (self.right_drag || self.left_drag)
                            && (self.modifiers.control_key() || self.modifiers.super_key());
                        if modifier_pan
                            || self.right_drag
                            || (self.left_drag && self.modifiers.shift_key())
                        {
                            self.camera.pan(delta, self.viewport.height);
                            self.window.request_redraw();
                        } else if self.left_drag && self.left_drag_distance > 4.0 {
                            self.camera.orbit(delta);
                            self.window.request_redraw();
                        }
                    }
                }
                self.cursor = Some(position);
            }
            WindowEvent::MouseWheel { delta, .. }
                if !egui_response.consumed && !self.egui_context.is_pointer_over_egui() =>
            {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 60.0,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32,
                };
                self.camera.zoom(amount);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let info = UiInfo {
            filename: self.loaded_filename.as_deref(),
            atom_count: self
                .molecule
                .as_ref()
                .map_or(0, |molecule| molecule.atoms.len()),
            bond_count: self
                .molecule
                .as_ref()
                .map_or(0, |molecule| molecule.bonds.len()),
            selection_count: self
                .display
                .as_ref()
                .map_or(0, DisplayState::selection_count),
            molecule: self.molecule.as_ref(),
            hierarchy: self.hierarchy.as_ref(),
            display: self.display.as_ref(),
            named_selections: &self.named_selections,
            inspection: self.inspection,
            camera: &self.camera,
            focus_description: &self.focus_description,
            pivot_description: &self.pivot_description,
        };
        let context = self.egui_context.clone();
        let mut actions = UiActions::default();
        let full_output = context.run_ui(raw_input, |root| {
            actions = self.ui.show(root, info);
        });
        self.egui_state.handle_platform_output_with_event_loop(
            &self.window,
            event_loop,
            full_output.platform_output,
        );
        self.update_viewport(actions.viewport, full_output.pixels_per_point);
        self.handle_ui_actions(actions);

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = context.tessellate(full_output.shapes, pixels_per_point);
        let mut textures_delta = full_output.textures_delta;
        let render_result = self.renderer.render(
            &self.camera,
            self.viewport,
            &paint_jobs,
            &textures_delta,
            pixels_per_point,
        );
        textures_delta.clear();
        match render_result {
            Ok(()) => {}
            Err(RenderError::Surface(SurfaceIssue::Outdated)) => {
                self.renderer.resize(self.window.inner_size());
                self.window.request_redraw();
            }
            Err(RenderError::Surface(SurfaceIssue::Lost)) => {
                if let Err(error) = self.renderer.recover_surface() {
                    self.ui.latest_error = Some(error.to_string());
                } else {
                    self.window.request_redraw();
                }
            }
            Err(RenderError::Surface(SurfaceIssue::Timeout)) => self.window.request_redraw(),
            Err(RenderError::Surface(SurfaceIssue::Occluded)) => {}
            Err(error) => {
                self.ui.latest_error = Some(error.to_string());
                self.window.request_redraw();
            }
        }
    }

    fn handle_ui_actions(&mut self, actions: UiActions) {
        if actions.open
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Protein Data Bank", &["pdb", "ent"])
                .pick_file()
            && let Err(error) = self.load_path(&path)
        {
            self.ui.latest_error = Some(error.to_string());
        }
        if actions.fit {
            self.fit();
        }
        if actions.reset_colors
            && let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display)
        {
            display.reset_colors(molecule);
            self.renderer.update_instances(molecule, display);
            self.ui.latest_error = None;
        }
        if let Some(command) = actions.execute {
            match parse_command(&command)
                .map_err(anyhow::Error::from)
                .and_then(|command| self.apply_command(command))
            {
                Ok(()) => self.ui.latest_error = None,
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            }
        }
        if let Some(action) = actions.manager {
            self.handle_manager_action(action);
        }
        if let Some(update) = actions.camera_update {
            self.apply_camera_update(update);
        }
        if let Some(request) = actions.focus_request {
            match self.resolve_focus(request) {
                Ok((point, description)) => {
                    self.camera.depth_of_field.focus_point = point;
                    self.focus_description = description;
                    self.ui.latest_error = None;
                }
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            }
        }
        if let Some(request) = actions.pivot_request {
            match self.resolve_pivot(request) {
                Ok((point, description)) => {
                    self.camera.set_pivot(point);
                    self.pivot_description = description;
                    self.ui.latest_error = None;
                }
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            }
        }
    }

    fn load_path(&mut self, path: &Path) -> Result<()> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("could not read {}", path.display()))?;
        let molecule =
            parse_pdb(&contents).with_context(|| format!("could not parse {}", path.display()))?;
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        let display = DisplayState::for_molecule(&molecule);
        self.renderer.update_instances(&molecule, &display);
        self.loaded_filename = Some(path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into(),
        ));
        self.molecule = Some(molecule);
        self.hierarchy = Some(hierarchy);
        self.display = Some(display);
        self.named_selections.clear();
        self.inspection = None;
        self.fit();
        self.camera.depth_of_field.focus_point = self.camera.target;
        self.focus_description = "Molecule center".into();
        self.pivot_description = "Molecule center".into();
        self.ui.latest_error = None;
        Ok(())
    }

    fn fit(&mut self) {
        if let Some((minimum, maximum)) = self.molecule.as_ref().and_then(Molecule::bounds) {
            self.camera.fit_bounds(minimum, maximum);
        }
    }

    fn apply_command(&mut self, command: Command) -> Result<()> {
        let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) else {
            anyhow::bail!("load a PDB file before executing commands");
        };
        match command {
            Command::Select { name, selection } => {
                let selection = evaluate_with_named(&selection, molecule, &self.named_selections)?;
                display.selection = selection.flags().to_vec();
                if let Some(name) = name {
                    self.named_selections.insert(name, selection);
                }
                self.inspection = None;
            }
            Command::Color { color, selection } => {
                let indices: Vec<_> =
                    evaluate_with_named(&selection, molecule, &self.named_selections)?
                        .indices()
                        .collect();
                display.set_color_override(&indices, DisplayLevel::Atom, Some(color.0));
            }
            Command::Show {
                representation,
                selection,
            } => {
                let mask = representation_mask(representation);
                for index in
                    evaluate_with_named(&selection, molecule, &self.named_selections)?.indices()
                {
                    display.representations[index].insert(mask);
                }
            }
            Command::Hide {
                representation,
                selection,
            } => {
                let mask = representation_mask(representation);
                for index in
                    evaluate_with_named(&selection, molecule, &self.named_selections)?.indices()
                {
                    display.representations[index].remove(mask);
                }
            }
        }
        self.renderer.update_instances(molecule, display);
        Ok(())
    }

    fn update_viewport(&mut self, rect: egui::Rect, pixels_per_point: f32) {
        if !rect.is_positive() {
            return;
        }
        let size = self.renderer.size();
        let x = (rect.min.x * pixels_per_point).clamp(0.0, size.width as f32);
        let y = (rect.min.y * pixels_per_point).clamp(0.0, size.height as f32);
        let width = (rect.width() * pixels_per_point)
            .max(1.0)
            .min(size.width as f32 - x);
        let height = (rect.height() * pixels_per_point)
            .max(1.0)
            .min(size.height as f32 - y);
        self.viewport = Viewport {
            x,
            y,
            width,
            height,
        };
        self.camera.set_viewport(self.viewport);
    }

    fn pick_at(&mut self, position: PhysicalPosition<f64>) {
        let point = Vec2::new(position.x as f32, position.y as f32);
        let picked = self
            .camera
            .screen_ray(point, self.viewport)
            .and_then(|ray| {
                self.molecule.as_ref().and_then(|molecule| {
                    let visible = self.display.as_ref().map(|display| &display.visible);
                    pick_atom_filtered(molecule, ray, |index| {
                        visible.is_none_or(|flags| flags.get(index).copied().unwrap_or(false))
                    })
                })
            });
        match picked {
            Some(atom_index) => {
                self.select_indices(vec![atom_index], Some(InspectionTarget::Atom(atom_index)))
            }
            None if self.viewport.contains(point) => self.select_indices(Vec::new(), None),
            None => {}
        }
    }

    fn handle_manager_action(&mut self, action: ManagerAction) {
        match action {
            ManagerAction::SelectChain(chain_index) => {
                let indices = self
                    .hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.chains.get(chain_index))
                    .map(|chain| {
                        chain
                            .residues
                            .iter()
                            .flat_map(|residue| residue.atom_indices.iter().copied())
                            .collect()
                    })
                    .unwrap_or_default();
                self.select_indices(indices, Some(InspectionTarget::Chain(chain_index)));
            }
            ManagerAction::SelectResidue {
                chain_index,
                residue_index,
            } => {
                let indices = self
                    .hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.residue(chain_index, residue_index))
                    .map(|residue| residue.atom_indices.clone())
                    .unwrap_or_default();
                self.select_indices(
                    indices,
                    Some(InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    }),
                );
            }
            ManagerAction::SelectAtom(atom_index) => {
                self.select_indices(vec![atom_index], Some(InspectionTarget::Atom(atom_index)))
            }
            ManagerAction::SetColor { target, color } => {
                let (indices, level) = self.display_target(target);
                if let Some(display) = &mut self.display {
                    display.set_color_override(&indices, level, color);
                }
                self.refresh_instances();
            }
            ManagerAction::CycleVisibility(target) => {
                let (indices, level) = self.display_target(target);
                if let Some(display) = &mut self.display {
                    display.cycle_visibility(&indices, level);
                }
                self.refresh_instances();
            }
            ManagerAction::SetColoringMode(mode) => {
                if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                    display.set_coloring_mode(molecule, mode);
                }
                self.refresh_instances();
            }
            ManagerAction::SetUniformColor(color) => {
                if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                    display.set_uniform_color(molecule, color);
                }
                self.refresh_instances();
            }
            ManagerAction::ActivateNamed(name) => {
                if let Some(selection) = self.named_selections.get(&name) {
                    self.set_selection(selection.flags().to_vec(), None);
                }
            }
            ManagerAction::RemoveNamed(name) => {
                self.named_selections.remove(&name);
            }
        }
    }

    fn select_indices(&mut self, indices: Vec<usize>, inspection: Option<InspectionTarget>) {
        let atom_count = self
            .molecule
            .as_ref()
            .map_or(0, |molecule| molecule.atoms.len());
        let mut flags = vec![false; atom_count];
        for index in indices {
            if let Some(selected) = flags.get_mut(index) {
                *selected = true;
            }
        }
        self.set_selection(flags, inspection);
    }

    fn set_selection(&mut self, flags: Vec<bool>, inspection: Option<InspectionTarget>) {
        let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) else {
            return;
        };
        display.selection = flags;
        self.inspection = inspection;
        self.renderer.update_instances(molecule, display);
    }

    fn display_target(&self, target: InspectionTarget) -> (Vec<usize>, DisplayLevel) {
        match target {
            InspectionTarget::Chain(chain_index) => (
                self.hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.chains.get(chain_index))
                    .map(|chain| {
                        chain
                            .residues
                            .iter()
                            .flat_map(|residue| residue.atom_indices.iter().copied())
                            .collect()
                    })
                    .unwrap_or_default(),
                DisplayLevel::Chain,
            ),
            InspectionTarget::Residue {
                chain_index,
                residue_index,
            } => (
                self.hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.residue(chain_index, residue_index))
                    .map(|residue| residue.atom_indices.clone())
                    .unwrap_or_default(),
                DisplayLevel::Residue,
            ),
            InspectionTarget::Atom(index) => (vec![index], DisplayLevel::Atom),
        }
    }

    fn refresh_instances(&mut self) {
        if let (Some(molecule), Some(display)) = (&self.molecule, &self.display) {
            self.renderer.update_instances(molecule, display);
        }
    }

    fn apply_camera_update(&mut self, update: CameraUpdate) {
        self.camera.set_clip_planes(update.near, update.far);
        self.camera
            .set_lens(update.focal_length_mm, update.sensor_height_mm);
        self.camera.depth_of_field.enabled = update.dof_enabled;
        self.camera.depth_of_field.f_stop = update.f_stop.clamp(0.7, 22.0);
        self.camera.depth_of_field.blade_count = if update.blade_count < 3 {
            0
        } else {
            update.blade_count.clamp(3, 12)
        };
        self.camera.depth_of_field.blade_rotation =
            update.blade_rotation.rem_euclid(360.0_f32.to_radians());
        self.camera.depth_of_field.max_coc_pixels = update.max_coc_pixels.clamp(1.0, 64.0);
    }

    fn resolve_focus(&self, request: FocusRequest) -> Result<(Vec3, String)> {
        let molecule = self
            .molecule
            .as_ref()
            .context("load a PDB file before setting a focus target")?;
        match request {
            FocusRequest::Chain(chain_id) => {
                let chain_id = chain_id.trim();
                let indices: Vec<_> = molecule
                    .atoms
                    .iter()
                    .enumerate()
                    .filter_map(|(index, atom)| {
                        atom.chain_id
                            .eq_ignore_ascii_case(chain_id)
                            .then_some(index)
                    })
                    .collect();
                Ok((
                    centroid(molecule, &indices)
                        .with_context(|| format!("chain '{chain_id}' was not found"))?,
                    format!(
                        "Chain {}",
                        if chain_id.is_empty() {
                            "(blank)"
                        } else {
                            chain_id
                        }
                    ),
                ))
            }
            FocusRequest::Residue { chain, number } => {
                let number = parse_residue_number(&number)?;
                let indices = matching_residue_indices(molecule, chain.trim(), number, None);
                Ok((
                    centroid(molecule, &indices).with_context(|| {
                        format!(
                            "residue {} in chain '{}' was not found",
                            number,
                            chain.trim()
                        )
                    })?,
                    format!("Residue {} / chain {}", number, chain.trim()),
                ))
            }
            FocusRequest::Base {
                chain,
                name,
                number,
            } => {
                let number = parse_residue_number(&number)?;
                let name = name.trim();
                let indices = matching_residue_indices(molecule, chain.trim(), number, Some(name));
                Ok((
                    centroid(molecule, &indices).with_context(|| {
                        format!(
                            "base {} {} in chain '{}' was not found",
                            name,
                            number,
                            chain.trim()
                        )
                    })?,
                    format!("Base {} {} / chain {}", name, number, chain.trim()),
                ))
            }
            FocusRequest::AtomSerial(serial) => {
                let serial = serial.trim().parse::<u32>().with_context(|| {
                    format!("'{}' is not a valid PDB atom serial", serial.trim())
                })?;
                let atom = molecule
                    .atoms
                    .iter()
                    .find(|atom| atom.serial == serial)
                    .with_context(|| format!("atom with PDB serial {serial} was not found"))?;
                Ok((atom.position, format!("Atom #{} {}", serial, atom.name)))
            }
            FocusRequest::Inspected => self.focus_from_inspection(molecule),
        }
    }

    fn resolve_pivot(&self, request: PivotRequest) -> Result<(Vec3, String)> {
        let molecule = self
            .molecule
            .as_ref()
            .context("load a PDB file before setting the pivot")?;
        match request {
            PivotRequest::Inspected => self.focus_from_inspection(molecule),
            PivotRequest::Reset => {
                let (minimum, maximum) = molecule
                    .bounds()
                    .context("the loaded molecule contains no atoms")?;
                Ok(((minimum + maximum) * 0.5, "Molecule center".into()))
            }
        }
    }

    fn focus_from_inspection(&self, molecule: &Molecule) -> Result<(Vec3, String)> {
        let target = self
            .inspection
            .context("select a chain, residue, or atom first")?;
        match target {
            InspectionTarget::Atom(index) => {
                let atom = molecule
                    .atoms
                    .get(index)
                    .context("the inspected atom no longer exists")?;
                Ok((
                    atom.position,
                    format!("Atom #{} {}", atom.serial, atom.name),
                ))
            }
            InspectionTarget::Residue {
                chain_index,
                residue_index,
            } => {
                let residue = self
                    .hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.residue(chain_index, residue_index))
                    .context("the inspected residue no longer exists")?;
                Ok((
                    centroid(molecule, &residue.atom_indices)
                        .context("the inspected residue contains no atoms")?,
                    format!("Residue {}", residue.id.label()),
                ))
            }
            InspectionTarget::Chain(chain_index) => {
                let chain = self
                    .hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.chains.get(chain_index))
                    .context("the inspected chain no longer exists")?;
                let indices: Vec<_> = chain
                    .residues
                    .iter()
                    .flat_map(|residue| residue.atom_indices.iter().copied())
                    .collect();
                Ok((
                    centroid(molecule, &indices)
                        .context("the inspected chain contains no atoms")?,
                    format!("Chain {}", chain.id),
                ))
            }
        }
    }
}

fn parse_residue_number(value: &str) -> Result<i32> {
    value
        .trim()
        .parse::<i32>()
        .with_context(|| format!("'{}' is not a valid residue number", value.trim()))
}

fn matching_residue_indices(
    molecule: &Molecule,
    chain: &str,
    number: i32,
    residue_name: Option<&str>,
) -> Vec<usize> {
    molecule
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(index, atom)| {
            (atom.chain_id.eq_ignore_ascii_case(chain)
                && atom.residue_number == number
                && residue_name.is_none_or(|name| atom.residue_name.eq_ignore_ascii_case(name)))
            .then_some(index)
        })
        .collect()
}

fn centroid(molecule: &Molecule, indices: &[usize]) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0_u32;
    for &index in indices {
        if let Some(atom) = molecule.atoms.get(index) {
            sum += atom.position;
            count += 1;
        }
    }
    (count > 0).then(|| sum / count as f32)
}

fn representation_mask(representation: Representation) -> RepresentationMask {
    match representation {
        Representation::Spheres => RepresentationMask::SPHERES,
        Representation::Sticks => RepresentationMask::STICKS,
    }
}

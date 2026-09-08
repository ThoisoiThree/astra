use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env, fs,
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
    path::PathBuf,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::mpsc::{
        self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError, TrySendError,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use astra::{
    DisplayColor, DisplayLevel, DisplayMode, DisplayState, DisplayStateData, ModeOverride,
    NamedSelectionStyle, RepresentationMask, VisibilityOverride,
    camera::{OrbitCamera, Viewport},
    command::{Command, Representation, parse_command},
    diagnostics::StartupTrace,
    measurement::{MeasurementEndpoint, MeasurementLine},
    molecule::{
        MAX_DECOMPRESSED_STRUCTURE_SIZE, Molecule, MoleculeHierarchy, SecondaryStructure,
        assign_secondary_structure, parse_structure,
    },
    picking::AtomBvh,
    render::{PreparedCartoon, RenderError, Renderer, SurfaceIssue, prepare_cartoon_cached},
    scene::{
        SCENE_EXTENSION, SCENE_FORMAT_NAME, SceneDocument, SceneHierarchyTarget,
        decode as decode_scene, encode as encode_scene, is_scene_document,
    },
    selection::{
        Selection, SelectionStatus, evaluate_with_named, rename_named_reference,
        resolve_named_expressions, validate_unique_name,
    },
};
use glam::{Vec2, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, ModifiersState},
    window::{Window, WindowAttributes, WindowId},
};

mod actions;
mod closing;
mod history;
mod io_jobs;
mod recovery;
mod runtime;
mod session;
use actions::*;
use history::*;
use io_jobs::*;
use runtime::Runtime;
use session::*;

pub use runtime::run;

const EDIT_HISTORY_LIMIT: usize = 50;
const MAX_FETCH_SIZE: u64 = 512 * 1024 * 1024;
const FETCH_BUFFER_SIZE: usize = 64 * 1024;
const PARALLEL_FETCH_MIN_SIZE: u64 = 4 * 1024 * 1024;
const PARALLEL_FETCH_WORKERS: usize = 4;
const FETCH_CANCELLED: &str = "PDB download canceled";
const MAX_LOCAL_FILE_SIZE: u64 = MAX_DECOMPRESSED_STRUCTURE_SIZE;
const AUTOSAVE_DELAY: Duration = Duration::from_secs(10);
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

use crate::ui::{
    CameraUpdate, CloseAction, FocusRequest, HierarchySelectionGesture, InspectionTarget,
    ManagerAction, PivotRequest, RecoveryAction, SessionTab, UiActions, UiInfo, UiState,
};

#[derive(Debug, Clone, Copy)]
struct PendingPick {
    request_id: u64,
    session_id: u64,
    document_version: u64,
    cpu_fallback: Option<usize>,
}

impl Runtime {
    fn new(event_loop: &ActiveEventLoop, initial_path: Option<PathBuf>) -> Result<Self> {
        let trace = StartupTrace::new("startup");
        trace.mark("BEGIN window creation");
        let attributes = WindowAttributes::default()
            .with_title("Astra")
            .with_inner_size(LogicalSize::new(1280.0, 800.0))
            .with_min_inner_size(LogicalSize::new(720.0, 480.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("could not create the Astra window")?,
        );
        trace.mark("END window creation; BEGIN renderer initialization");
        let renderer = pollster::block_on(Renderer::new(window.clone()))
            .context("could not initialize wgpu")?;
        trace.mark("END renderer initialization; BEGIN UI initialization");
        let size = renderer.size();
        let viewport = Viewport::full(size.width, size.height);
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
        let (job_sender, job_requests) = mpsc::sync_channel(2);
        let (job_events, job_receiver) = mpsc::channel();
        let worker_window = window.clone();
        thread::spawn(move || background_worker(job_requests, job_events, worker_window));
        let mut runtime = Self {
            window,
            renderer,
            egui_context,
            egui_state,
            molecule: None,
            hierarchy: None,
            secondary_structure: None,
            atom_bvh: None,
            display: None,
            named_selections: BTreeMap::new(),
            named_selection_expressions: BTreeMap::new(),
            named_selection_styles: BTreeMap::new(),
            named_selection_statuses: BTreeMap::new(),
            measurement_lines: Vec::new(),
            next_measurement_id: 1,
            hierarchy_names: BTreeMap::new(),
            inspection: None,
            hierarchy_selection: BTreeSet::new(),
            hierarchy_selection_anchor: None,
            focus_description: "World origin".into(),
            pivot_description: "World origin".into(),
            loaded_filename: None,
            molecule_id: None,
            scene_path: None,
            dirty: false,
            recovery_path: None,
            autosave_due: None,
            document_version: 0,
            cartoon_generation: 0,
            needs_cartoon_refresh: false,
            next_pick_request_id: 1,
            pending_pick: None,
            inactive_sessions: Vec::new(),
            session_order: Vec::new(),
            active_session_id: 1,
            next_session_id: 2,
            fetch_receiver: None,
            fetching_pdb_id: None,
            fetch_progress: FetchProgress::default(),
            fetch_cancel: None,
            job_sender,
            job_receiver,
            background_jobs: BTreeMap::new(),
            next_job_id: 1,
            camera,
            ui: UiState::default(),
            cursor: None,
            viewport,
            left_drag: false,
            left_drag_distance: 0.0,
            right_drag: false,
            modifiers: ModifiersState::default(),
            focused: true,
            occluded: false,
            undo_history: VecDeque::new(),
            redo_history: VecDeque::new(),
            repaint_due: None,
            pending_close: None,
            exit_ready: false,
            recovery_scan_pending: true,
            recovery_candidates: VecDeque::new(),
        };
        if let Some(path) = initial_path
            && let Err(error) = runtime.start_load_path(&path)
        {
            runtime.ui.latest_error = Some(error.to_string());
        }
        trace.mark("END UI initialization; requesting first redraw (recovery deferred)");
        runtime.window.request_redraw();
        Ok(runtime)
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        self.poll_background_jobs();
        if self.exit_ready {
            event_loop.exit();
            return;
        }
        self.poll_fetch_result();
        self.poll_pick_result();
        let egui_response = self.egui_state.on_window_event(&self.window, &event);
        // egui-winit returns repaint=true for RedrawRequested itself. Echoing
        // that response creates a self-sustaining loop even with ControlFlow::Wait.
        if egui_response.repaint && !matches!(event, WindowEvent::RedrawRequested) {
            self.window.request_redraw();
        }
        if self.pending_close.is_some()
            && matches!(
                event,
                WindowEvent::KeyboardInput { .. }
                    | WindowEvent::MouseInput { .. }
                    | WindowEvent::MouseWheel { .. }
                    | WindowEvent::CursorMoved { .. }
                    | WindowEvent::DroppedFile(_)
            )
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                if self.request_exit() {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                self.renderer.resize(size);
                self.viewport = Viewport::full(size.width, size.height);
                self.camera.set_viewport(self.viewport);
                self.window.request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                if let Err(error) = self.start_load_path(&path) {
                    self.ui.latest_error = Some(error.to_string());
                }
                self.window.request_redraw();
            }
            WindowEvent::Focused(focused) => {
                self.focused = focused;
                self.left_drag = false;
                self.right_drag = false;
                self.cursor = None;
                self.modifiers = ModifiersState::default();
                if focused {
                    let size = self.window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        self.window.request_redraw();
                    }
                }
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded && self.focused {
                    let size = self.window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        self.window.request_redraw();
                    }
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && !egui_response.consumed
                    && (self.modifiers.control_key() || self.modifiers.super_key()) =>
            {
                let handled = match &event.logical_key {
                    Key::Character(value) if value.eq_ignore_ascii_case("z") => self.undo(),
                    Key::Character(value) if value.eq_ignore_ascii_case("r") => self.redo(),
                    _ => false,
                };
                if handled {
                    self.window.request_redraw();
                }
            }
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
                            self.request_gpu_pick(position);
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
                            self.mark_camera_dirty();
                            self.window.request_redraw();
                        } else if self.left_drag && self.left_drag_distance > 4.0 {
                            self.camera.orbit(delta);
                            self.mark_camera_dirty();
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
                self.mark_camera_dirty();
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested
                if (self.focused || self.pending_close.is_some())
                    && !self.occluded
                    && self.window.inner_size().width > 0
                    && self.window.inner_size().height > 0 =>
            {
                self.redraw(event_loop);
            }
            _ => {}
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let trace = StartupTrace::new("redraw");
        trace.mark("BEGIN UI frame");
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let session_tabs = self.session_tabs();
        let background_job = self
            .background_jobs
            .iter()
            .find(|(_, job)| job.session_id == self.active_session_id);
        let info = UiInfo {
            adapter_info: self.renderer.adapter_info(),
            filename: self.loaded_filename.as_deref(),
            molecule_id: self.molecule_id.as_deref(),
            session_tabs: &session_tabs,
            active_session_id: self
                .loaded_filename
                .as_ref()
                .map(|_| self.active_session_id),
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
            named_selection_expressions: &self.named_selection_expressions,
            named_selection_styles: &self.named_selection_styles,
            named_selection_statuses: &self.named_selection_statuses,
            measurement_lines: &self.measurement_lines,
            hierarchy_names: &self.hierarchy_names,
            inspection: self.inspection,
            hierarchy_selection: &self.hierarchy_selection,
            camera: &self.camera,
            focus_description: &self.focus_description,
            pivot_description: &self.pivot_description,
            fetching_pdb_id: self.fetching_pdb_id.as_deref(),
            fetch_downloaded_bytes: self.fetch_progress.downloaded_bytes,
            fetch_total_bytes: self.fetch_progress.total_bytes,
            fetch_bytes_per_second: self.fetch_progress.bytes_per_second,
            background_job_id: background_job.map(|(id, _)| *id),
            background_stage: background_job.map(|(_, job)| job.stage.as_str()),
            background_progress: background_job.map_or(0.0, |(_, job)| job.progress),
            render_stats: self.renderer.stats(),
            close_pending: self.pending_close.is_some(),
            close_busy: self
                .pending_close
                .as_ref()
                .is_some_and(closing::ClosePlan::is_busy),
            recovery_file: self.recovery_candidates.front().map(PathBuf::as_path),
        };
        let context = self.egui_context.clone();
        let mut actions = UiActions::default();
        let full_output = context.run_ui(raw_input, |root| {
            actions = self.ui.show(root, info);
        });
        self.repaint_due = full_output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .and_then(|output| Instant::now().checked_add(output.repaint_delay));
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
        trace.mark("END UI frame; BEGIN GPU frame");
        let render_result = self.renderer.render(
            &self.camera,
            self.display.as_ref(),
            self.viewport,
            &paint_jobs,
            &textures_delta,
            pixels_per_point,
        );
        textures_delta.clear();
        trace.mark(format_args!("END GPU frame: {render_result:?}"));
        match render_result {
            Ok(()) => self.start_recovery_scan_after_frame(),
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
            // Retrying immediately can create a hot redraw loop on Metal while macOS is
            // restoring the window's drawable after an app switch. A later OS event will
            // schedule the next frame without monopolizing WindowServer.
            Err(RenderError::Surface(SurfaceIssue::Timeout)) => {}
            Err(RenderError::Surface(SurfaceIssue::Occluded)) => {}
            Err(error) => {
                self.ui.latest_error = Some(error.to_string());
            }
        }
    }

    fn handle_ui_actions(&mut self, actions: UiActions) {
        if let Some(action) = actions.close_confirmation {
            self.handle_close_action(action);
            return;
        }
        if self.pending_close.is_some() {
            return;
        }
        if let Some(action) = actions.recovery {
            self.handle_recovery_action(action);
            return;
        }
        if let Some(id) = actions.close_session {
            self.request_close_session(id);
            return;
        }
        if let Some(id) = actions.activate_session {
            self.activate_session(id);
            return;
        }
        if actions.open
            && let Some(paths) = rfd::FileDialog::new()
                .add_filter(SCENE_FORMAT_NAME, &[SCENE_EXTENSION])
                .add_filter(
                    "Molecular structures",
                    &["pdb", "ent", "cif", "mmcif", "bcif", "xml", "gz"],
                )
                .add_filter("PDBx/mmCIF", &["cif", "mmcif", "gz"])
                .add_filter("BinaryCIF", &["bcif", "gz"])
                .add_filter("Legacy PDB", &["pdb", "ent", "gz"])
                .add_filter("PDBML/XML", &["xml", "gz"])
                .pick_files()
        {
            for path in paths {
                if let Err(error) = self.start_load_path(&path) {
                    self.ui.latest_error = Some(error.to_string());
                    break;
                }
            }
        }
        if let Some(id) = actions.fetch
            && let Err(error) = self.start_fetch(&id)
        {
            self.ui.latest_error = Some(error.to_string());
        }
        if actions.cancel_fetch {
            self.cancel_fetch();
        }
        if let Some(id) = actions.cancel_background_job {
            self.cancel_background_job(id);
        }
        if actions.save_as {
            if let Err(error) = self.save_scene_as() {
                self.ui.latest_error = Some(error.to_string());
            }
        } else if actions.save {
            let result = if let Some(path) = self.scene_path.clone() {
                self.save_scene_to(path)
            } else {
                self.save_scene_as()
            };
            if let Err(error) = result {
                self.ui.latest_error = Some(error.to_string());
            }
        }
        if actions.fit {
            self.fit();
            self.mark_camera_dirty();
        }
        if actions.reset_colors {
            let before = self.begin_edit();
            if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                display.reset_colors(molecule);
                for style in self.named_selection_styles.values_mut() {
                    style.color = None;
                }
                display.replace_named_layers(&named_display_layers(
                    &self.named_selections,
                    &self.named_selection_styles,
                ));
                self.refresh_display_attributes();
                self.ui.latest_error = None;
                self.commit_edit(before);
            }
        }
        if let Some(command) = actions.execute {
            match parse_command(&command).map_err(anyhow::Error::from) {
                Ok(command) => {
                    let before = self.begin_edit();
                    match self.apply_command(command) {
                        Ok(()) => {
                            self.commit_edit(before);
                            self.ui.latest_error = None;
                        }
                        Err(error) => self.ui.latest_error = Some(error.to_string()),
                    }
                }
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            }
        }
        if let Some(action) = actions.manager {
            let before = self.begin_edit();
            self.handle_manager_action(action);
            self.commit_edit(before);
        }
        if let Some(update) = actions.camera_update {
            self.apply_camera_update(update);
            self.mark_camera_dirty();
        }
        if let Some(request) = actions.focus_request {
            match self.resolve_focus(request) {
                Ok((point, description)) => {
                    self.camera.depth_of_field.focus_point = point;
                    self.focus_description = description;
                    self.mark_camera_dirty();
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
                    self.mark_camera_dirty();
                    self.ui.latest_error = None;
                }
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            }
        }
    }

    fn session_tabs(&self) -> Vec<SessionTab> {
        self.session_order
            .iter()
            .filter_map(|id| {
                if *id == self.active_session_id && self.loaded_filename.is_some() {
                    Some(SessionTab {
                        id: *id,
                        label: session_label(
                            self.molecule_id.as_deref(),
                            self.loaded_filename.as_deref(),
                        ),
                        dirty: self.dirty,
                    })
                } else {
                    self.inactive_sessions
                        .iter()
                        .find(|session| session.id == *id)
                        .map(DocumentSession::tab)
                }
            })
            .collect()
    }

    fn begin_new_session(&mut self) {
        if self.molecule.is_some() || self.loaded_filename.is_some() {
            let mut previous = DocumentSession::empty(self.active_session_id, self.viewport);
            self.swap_active_document(&mut previous);
            self.inactive_sessions.push(previous);
            self.active_session_id = self.next_session_id;
            self.next_session_id = self.next_session_id.saturating_add(1);
        }
        if !self.session_order.contains(&self.active_session_id) {
            self.session_order.push(self.active_session_id);
        }
    }

    fn activate_session(&mut self, id: u64) {
        if id == self.active_session_id || !self.session_order.contains(&id) {
            return;
        }
        let Some(position) = self
            .inactive_sessions
            .iter()
            .position(|session| session.id == id)
        else {
            return;
        };
        let mut target = self.inactive_sessions.remove(position);
        let mut previous = DocumentSession::empty(self.active_session_id, self.viewport);
        self.swap_active_document(&mut previous);
        if previous.molecule.is_some() || previous.loaded_filename.is_some() {
            self.inactive_sessions.push(previous);
        }
        self.active_session_id = id;
        self.swap_active_document(&mut target);
        self.sync_active_document();
    }

    fn close_session(&mut self, id: u64) {
        let Some(order_position) = self
            .session_order
            .iter()
            .position(|candidate| *candidate == id)
        else {
            return;
        };
        self.session_order.remove(order_position);
        if id != self.active_session_id {
            if let Some(position) = self
                .inactive_sessions
                .iter()
                .position(|session| session.id == id)
            {
                self.inactive_sessions.remove(position);
            }
            return;
        }

        let next_id = if self.session_order.is_empty() {
            None
        } else {
            Some(self.session_order[order_position.min(self.session_order.len() - 1)])
        };
        let mut discarded = DocumentSession::empty(self.active_session_id, self.viewport);
        self.swap_active_document(&mut discarded);
        if let Some(next_id) = next_id
            && let Some(position) = self
                .inactive_sessions
                .iter()
                .position(|session| session.id == next_id)
        {
            let mut next = self.inactive_sessions.remove(position);
            self.active_session_id = next_id;
            self.swap_active_document(&mut next);
        } else {
            self.active_session_id = self.next_session_id;
            self.next_session_id = self.next_session_id.saturating_add(1);
        }
        self.sync_active_document();
    }

    fn swap_active_document(&mut self, session: &mut DocumentSession) {
        std::mem::swap(&mut self.molecule, &mut session.molecule);
        std::mem::swap(&mut self.hierarchy, &mut session.hierarchy);
        std::mem::swap(
            &mut self.secondary_structure,
            &mut session.secondary_structure,
        );
        std::mem::swap(&mut self.atom_bvh, &mut session.atom_bvh);
        std::mem::swap(&mut self.display, &mut session.display);
        std::mem::swap(&mut self.named_selections, &mut session.named_selections);
        std::mem::swap(
            &mut self.named_selection_expressions,
            &mut session.named_selection_expressions,
        );
        std::mem::swap(
            &mut self.named_selection_styles,
            &mut session.named_selection_styles,
        );
        std::mem::swap(
            &mut self.named_selection_statuses,
            &mut session.named_selection_statuses,
        );
        std::mem::swap(&mut self.measurement_lines, &mut session.measurement_lines);
        std::mem::swap(
            &mut self.next_measurement_id,
            &mut session.next_measurement_id,
        );
        std::mem::swap(&mut self.hierarchy_names, &mut session.hierarchy_names);
        std::mem::swap(&mut self.inspection, &mut session.inspection);
        std::mem::swap(
            &mut self.hierarchy_selection,
            &mut session.hierarchy_selection,
        );
        std::mem::swap(
            &mut self.hierarchy_selection_anchor,
            &mut session.hierarchy_selection_anchor,
        );
        std::mem::swap(&mut self.focus_description, &mut session.focus_description);
        std::mem::swap(&mut self.pivot_description, &mut session.pivot_description);
        std::mem::swap(&mut self.loaded_filename, &mut session.loaded_filename);
        std::mem::swap(&mut self.molecule_id, &mut session.molecule_id);
        std::mem::swap(&mut self.scene_path, &mut session.scene_path);
        std::mem::swap(&mut self.dirty, &mut session.dirty);
        std::mem::swap(&mut self.recovery_path, &mut session.recovery_path);
        std::mem::swap(&mut self.autosave_due, &mut session.autosave_due);
        std::mem::swap(&mut self.document_version, &mut session.document_version);
        std::mem::swap(
            &mut self.cartoon_generation,
            &mut session.cartoon_generation,
        );
        std::mem::swap(
            &mut self.needs_cartoon_refresh,
            &mut session.needs_cartoon_refresh,
        );
        std::mem::swap(&mut self.camera, &mut session.camera);
        std::mem::swap(&mut self.undo_history, &mut session.undo_history);
        std::mem::swap(&mut self.redo_history, &mut session.redo_history);
    }

    fn sync_active_document(&mut self) {
        self.camera.set_viewport(self.viewport);
        if self.molecule.is_some() && self.display.is_some() {
            self.needs_cartoon_refresh = true;
            self.schedule_cartoon_job();
        } else {
            let molecule = Molecule::default();
            let display = DisplayState::for_molecule(&molecule);
            self.renderer.update_instances(&molecule, &display);
        }
        self.renderer.update_measurements(&self.measurement_lines);
        self.ui.document_changed();
        self.ui.latest_error = None;
    }

    fn start_load_path(&mut self, path: &Path) -> Result<()> {
        self.start_load_path_with_kind(path, JobKind::Load)
    }

    fn start_load_path_with_kind(&mut self, path: &Path, kind: JobKind) -> Result<()> {
        let filename = path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        self.begin_new_session();
        self.molecule = None;
        self.hierarchy = None;
        self.secondary_structure = None;
        self.atom_bvh = None;
        self.display = None;
        self.named_selections.clear();
        self.named_selection_expressions.clear();
        self.named_selection_styles.clear();
        self.named_selection_statuses.clear();
        self.measurement_lines.clear();
        self.hierarchy_names.clear();
        self.inspection = None;
        self.hierarchy_selection.clear();
        self.hierarchy_selection_anchor = None;
        self.loaded_filename = Some(filename);
        self.molecule_id = Some("Loading…".into());
        self.scene_path = None;
        self.dirty = false;
        self.document_version = self.document_version.wrapping_add(1);
        self.renderer.update_measurements(&[]);

        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let job = JobRequest::Load {
            id,
            session_id: self.active_session_id,
            version: self.document_version,
            path: path.to_owned(),
            cancel: cancel.clone(),
        };
        self.submit_job(job)?;
        self.background_jobs.insert(
            id,
            BackgroundJob {
                session_id: self.active_session_id,
                version: self.document_version,
                kind,
                stage: "Queued for loading".into(),
                progress: 0.0,
                cancel,
            },
        );
        self.ui.latest_error = None;
        Ok(())
    }

    fn submit_job(&self, request: JobRequest) -> Result<()> {
        self.job_sender
            .try_send(request)
            .map_err(|error| match error {
                TrySendError::Full(_) => anyhow::anyhow!(
                    "background worker queue is full; wait for the current operation or cancel it"
                ),
                TrySendError::Disconnected(_) => {
                    anyhow::anyhow!("background worker stopped unexpectedly")
                }
            })
    }

    fn cancel_background_job(&mut self, id: u64) {
        if let Some(job) = self.background_jobs.get(&id) {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn poll_background_jobs(&mut self) {
        while let Ok(event) = self.job_receiver.try_recv() {
            match event {
                JobEvent::RecoveryScanned(result) => match result {
                    Ok(paths) => self.recovery_candidates = paths.into(),
                    Err(error) => self.ui.latest_error = Some(error),
                },
                JobEvent::RecoveryDiscarded(result) => {
                    if let Err(error) = result {
                        self.ui.latest_error = Some(error);
                    }
                }
                JobEvent::Progress {
                    id,
                    stage,
                    progress,
                } => {
                    if let Some(job) = self.background_jobs.get_mut(&id) {
                        job.stage = stage.into();
                        job.progress = progress.clamp(0.0, 1.0);
                    }
                }
                JobEvent::Complete { id, result } => {
                    let Some(job) = self.background_jobs.remove(&id) else {
                        continue;
                    };
                    match *result {
                        Ok(JobOutput::Loaded(payload)) => {
                            self.apply_loaded_job(
                                job.session_id,
                                job.version,
                                *payload,
                                job.kind == JobKind::Recover,
                            );
                        }
                        Ok(JobOutput::Saved(path)) => {
                            self.apply_saved_job(job.session_id, job.version, path);
                        }
                        Ok(JobOutput::Cartoon(prepared)) => {
                            if job.session_id == self.active_session_id
                                && job.version == self.cartoon_generation
                                && let (Some(molecule), Some(display)) =
                                    (&self.molecule, &self.display)
                            {
                                self.renderer
                                    .update_prepared_cartoon(molecule, display, prepared);
                            }
                        }
                        Ok(JobOutput::Cancelled) => {}
                        Err(error) if error == "background operation canceled" => {}
                        Err(error) => self.ui.latest_error = Some(error),
                    }
                }
            }
        }
        if self.needs_cartoon_refresh
            && !self
                .background_jobs
                .values()
                .any(|job| job.kind == JobKind::Cartoon && job.session_id == self.active_session_id)
        {
            self.schedule_cartoon_job();
        }
        self.advance_close();
    }

    fn schedule_cartoon_job(&mut self) {
        let (Some(molecule), Some(display), Some(hierarchy), Some(secondary_structure)) = (
            &self.molecule,
            &self.display,
            &self.hierarchy,
            &self.secondary_structure,
        ) else {
            return;
        };
        for job in self
            .background_jobs
            .values()
            .filter(|job| job.kind == JobKind::Cartoon && job.session_id == self.active_session_id)
        {
            job.cancel.store(true, Ordering::Relaxed);
        }
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        self.cartoon_generation = self.cartoon_generation.wrapping_add(1);
        let generation = self.cartoon_generation;
        let cancel = Arc::new(AtomicBool::new(false));
        let request = JobRequest::Cartoon {
            id,
            molecule: Box::new(molecule.clone()),
            display: Box::new(display.clone()),
            hierarchy: Box::new(hierarchy.clone()),
            secondary_structure: secondary_structure.clone(),
            cancel: cancel.clone(),
        };
        if self.submit_job(request).is_err() {
            self.needs_cartoon_refresh = true;
            return;
        }
        self.needs_cartoon_refresh = false;
        self.background_jobs.insert(
            id,
            BackgroundJob {
                session_id: self.active_session_id,
                version: generation,
                kind: JobKind::Cartoon,
                stage: "Queued ribbon geometry".into(),
                progress: 0.0,
                cancel,
            },
        );
    }

    fn apply_loaded_job(
        &mut self,
        session_id: u64,
        version: u64,
        payload: LoadedPayload,
        recovering: bool,
    ) {
        let original_session = self.active_session_id;
        if session_id != original_session {
            self.activate_session(session_id);
        }
        if self.active_session_id == session_id && self.document_version == version {
            let recovered_path = match &payload {
                LoadedPayload::Scene { path, .. } if recovering => Some(path.clone()),
                _ => None,
            };
            match payload {
                LoadedPayload::Structure {
                    filename,
                    molecule_id,
                    molecule,
                    hierarchy,
                    secondary_structure,
                    atom_bvh,
                    display,
                } => {
                    self.loaded_filename = Some(filename);
                    self.molecule_id = Some(molecule_id);
                    self.scene_path = None;
                    self.molecule = Some(molecule);
                    self.hierarchy = Some(hierarchy);
                    self.secondary_structure = Some(secondary_structure);
                    self.atom_bvh = Some(atom_bvh);
                    self.display = Some(*display);
                    self.named_selections.clear();
                    self.named_selection_expressions.clear();
                    self.named_selection_styles.clear();
                    self.named_selection_statuses.clear();
                    self.measurement_lines.clear();
                    self.next_measurement_id = 1;
                    self.hierarchy_names.clear();
                    self.inspection = None;
                    self.hierarchy_selection.clear();
                    self.hierarchy_selection_anchor = None;
                    self.undo_history.clear();
                    self.redo_history.clear();
                    self.fit();
                    self.camera.depth_of_field.focus_point = self.camera.target;
                    self.focus_description = "Molecule center".into();
                    self.pivot_description = "Molecule center".into();
                }
                LoadedPayload::Scene {
                    filename,
                    path,
                    document,
                    hierarchy,
                    secondary_structure,
                    atom_bvh,
                } => {
                    let SceneDocument {
                        source_name,
                        molecule,
                        display,
                        named_selections,
                        named_selection_expressions,
                        named_selection_styles,
                        measurement_lines,
                        hierarchy_names,
                        inspection,
                        hierarchy_selection,
                        hierarchy_selection_anchor,
                        focus_description,
                        pivot_description,
                        mut camera,
                    } = *document;
                    self.molecule_id = Some(if source_name.trim().is_empty() {
                        molecule_id_from_filename(&filename)
                    } else {
                        molecule_id_from_filename(&source_name)
                    });
                    camera.set_viewport(self.viewport);
                    self.next_measurement_id = measurement_lines
                        .iter()
                        .map(|line| line.id)
                        .max()
                        .unwrap_or(0)
                        .saturating_add(1);
                    self.loaded_filename = Some(if recovering {
                        let label = if source_name.trim().is_empty() {
                            &filename
                        } else {
                            &source_name
                        };
                        format!("{label} (Recovered)")
                    } else {
                        filename
                    });
                    self.scene_path = (!recovering).then_some(path);
                    self.molecule = Some(molecule);
                    self.hierarchy = Some(hierarchy);
                    self.secondary_structure = Some(secondary_structure);
                    self.atom_bvh = Some(atom_bvh);
                    self.display = Some(display);
                    self.named_selections = named_selections;
                    self.named_selection_expressions = named_selection_expressions;
                    self.named_selection_styles = named_selection_styles;
                    self.measurement_lines = measurement_lines;
                    self.hierarchy_names = hierarchy_names
                        .into_iter()
                        .map(|(target, name)| (inspection_target(target), name))
                        .collect();
                    self.inspection = inspection.map(inspection_target);
                    self.hierarchy_selection = hierarchy_selection
                        .into_iter()
                        .map(inspection_target)
                        .collect();
                    self.hierarchy_selection_anchor =
                        hierarchy_selection_anchor.map(inspection_target);
                    self.focus_description = focus_description;
                    self.pivot_description = pivot_description;
                    self.camera = camera;
                    self.undo_history.clear();
                    self.redo_history.clear();
                    self.recalculate_named_selections();
                }
            }
            self.dirty = recovered_path.is_some();
            self.recovery_path = recovered_path;
            self.autosave_due = None;
            self.needs_cartoon_refresh = true;
            self.renderer.update_measurements(&self.measurement_lines);
            self.schedule_cartoon_job();
            self.ui.document_changed();
            self.ui.latest_error = None;
        }
        if session_id != original_session && self.session_order.contains(&original_session) {
            self.activate_session(original_session);
        }
    }

    fn apply_saved_job(&mut self, session_id: u64, version: u64, path: PathBuf) {
        let original_session = self.active_session_id;
        if session_id != original_session {
            self.activate_session(session_id);
        }
        if self.active_session_id == session_id {
            self.loaded_filename = Some(path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            ));
            self.scene_path = Some(path);
            if self.document_version == version {
                self.dirty = false;
                self.autosave_due = None;
                self.remove_recovery_file();
            }
            self.ui.latest_error = None;
        }
        if session_id != original_session && self.session_order.contains(&original_session) {
            self.activate_session(original_session);
        }
    }

    fn write_scene(&self, path: &Path) -> Result<()> {
        let document = self.scene_document()?;
        let contents = encode_scene(&document).context("could not encode scene")?;
        atomic_write(path, &contents)
            .with_context(|| format!("could not atomically write scene {}", path.display()))
    }

    fn scene_document(&self) -> Result<SceneDocument> {
        let molecule = self
            .molecule
            .as_ref()
            .context("open a structure before saving a scene")?;
        let display = self
            .display
            .as_ref()
            .context("display state is unavailable")?;
        Ok(SceneDocument {
            source_name: self
                .molecule_id
                .clone()
                .or_else(|| self.loaded_filename.clone())
                .unwrap_or_default(),
            molecule: molecule.clone(),
            display: display.clone(),
            named_selections: self.named_selections.clone(),
            named_selection_expressions: self.named_selection_expressions.clone(),
            named_selection_styles: self.named_selection_styles.clone(),
            measurement_lines: self.measurement_lines.clone(),
            hierarchy_names: self
                .hierarchy_names
                .iter()
                .map(|(target, name)| (scene_target(*target), name.clone()))
                .collect(),
            inspection: self.inspection.map(scene_target),
            hierarchy_selection: self
                .hierarchy_selection
                .iter()
                .copied()
                .map(scene_target)
                .collect(),
            hierarchy_selection_anchor: self.hierarchy_selection_anchor.map(scene_target),
            focus_description: self.focus_description.clone(),
            pivot_description: self.pivot_description.clone(),
            camera: self.camera.clone(),
        })
    }

    fn save_scene_as(&mut self) -> Result<()> {
        let suggested_name = self
            .loaded_filename
            .as_deref()
            .and_then(|name| Path::new(name).file_stem())
            .map_or_else(
                || format!("scene.{SCENE_EXTENSION}"),
                |stem| format!("{}.{}", stem.to_string_lossy(), SCENE_EXTENSION),
            );
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(SCENE_FORMAT_NAME, &[SCENE_EXTENSION])
            .set_file_name(suggested_name)
            .save_file()
        {
            self.save_scene_to(path)?;
        }
        Ok(())
    }

    fn save_scene_to(&mut self, path: PathBuf) -> Result<()> {
        let path = molecule_path(path);
        if self
            .background_jobs
            .values()
            .any(|job| job.kind == JobKind::Save && job.session_id == self.active_session_id)
        {
            anyhow::bail!("a Save operation is already running for this tab");
        }
        let document = self.scene_document()?;
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.submit_job(JobRequest::Save {
            id,
            session_id: self.active_session_id,
            version: self.document_version,
            path: path.clone(),
            document: Box::new(document),
            cancel: cancel.clone(),
        })?;
        self.background_jobs.insert(
            id,
            BackgroundJob {
                session_id: self.active_session_id,
                version: self.document_version,
                kind: JobKind::Save,
                stage: "Queued for saving".into(),
                progress: 0.0,
                cancel,
            },
        );
        self.ui.latest_error = None;
        Ok(())
    }

    fn start_fetch(&mut self, requested_id: &str) -> Result<()> {
        if self.fetch_receiver.is_some() {
            anyhow::bail!("a PDB download is already in progress");
        }
        let id = normalize_pdb_id(requested_id)?;
        let directory = pdb_download_directory()?;
        let (sender, receiver) = mpsc::channel();
        let window = self.window.clone();
        let worker_id = id.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        thread::spawn(move || {
            let progress_sender = sender.clone();
            let progress_window = window.clone();
            let result = download_pdb_with_progress(
                &worker_id,
                &directory,
                &worker_cancel,
                move |progress| {
                    let _ = progress_sender.send(FetchEvent::Progress(progress));
                    progress_window.request_redraw();
                },
            )
            .map_err(|error| error.to_string());
            let _ = sender.send(FetchEvent::Finished(result));
            window.request_redraw();
        });
        self.fetch_receiver = Some(receiver);
        self.fetching_pdb_id = Some(id);
        self.fetch_progress = FetchProgress::default();
        self.fetch_cancel = Some(cancel);
        self.ui.latest_error = None;
        Ok(())
    }

    fn cancel_fetch(&mut self) {
        if let Some(cancel) = &self.fetch_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    fn poll_fetch_result(&mut self) {
        let mut finished = None;
        loop {
            let event = match self.fetch_receiver.as_ref().map(Receiver::try_recv) {
                Some(Ok(event)) => Some(event),
                Some(Err(TryRecvError::Disconnected)) => Some(FetchEvent::Finished(Err(
                    "PDB download worker stopped unexpectedly".into(),
                ))),
                Some(Err(TryRecvError::Empty)) | None => None,
            };
            match event {
                Some(FetchEvent::Progress(progress)) => self.fetch_progress = progress,
                Some(FetchEvent::Finished(result)) => {
                    finished = Some(result);
                    break;
                }
                None => break,
            }
        }
        let Some(result) = finished else {
            return;
        };
        self.fetch_receiver = None;
        self.fetching_pdb_id = None;
        self.fetch_cancel = None;
        self.ui.close_fetch();
        match result {
            Ok(path) => {
                if let Err(error) = self.start_load_path(&path) {
                    self.ui.latest_error = Some(error.to_string());
                }
            }
            Err(error) if error == FETCH_CANCELLED => self.ui.latest_error = None,
            Err(error) => self.ui.latest_error = Some(error),
        }
    }

    fn begin_edit(&self) -> EditTransaction {
        let mut names = BTreeSet::new();
        names.extend(self.named_selections.keys().cloned());
        names.extend(self.named_selection_expressions.keys().cloned());
        names.extend(self.named_selection_styles.keys().cloned());
        let named_selections = names
            .into_iter()
            .filter_map(|name| {
                self.named_selections.get(&name).cloned().map(|selection| {
                    (
                        name.clone(),
                        NamedSelectionRecord {
                            selection,
                            expression: self.named_selection_expressions.get(&name).cloned(),
                            style: self.named_selection_styles.get(&name).copied(),
                        },
                    )
                })
            })
            .collect();
        let measurements = self
            .measurement_lines
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, line)| (line.id, IndexedMeasurement { index, line }))
            .collect();
        EditTransaction {
            display: self.display.as_ref().map(DisplayState::edit_state),
            named_selections,
            measurements,
            hierarchy_names: self.hierarchy_names.clone(),
            workspace: WorkspaceSelection {
                inspection: self.inspection,
                hierarchy_selection: self.hierarchy_selection.clone(),
                hierarchy_selection_anchor: self.hierarchy_selection_anchor,
            },
        }
    }

    fn commit_edit(&mut self, before: EditTransaction) {
        let operation = EditOperation::between(before, self.begin_edit());
        if operation.changes.is_empty() {
            self.flush_cartoon_refresh();
            return;
        }
        push_history(&mut self.undo_history, operation);
        self.redo_history.clear();
        self.mark_dirty();
        self.flush_cartoon_refresh();
    }

    fn undo(&mut self) -> bool {
        let Some(operation) = self.undo_history.pop_back() else {
            return false;
        };
        self.apply_edit_operation(&operation, HistoryDirection::Undo);
        push_history(&mut self.redo_history, operation);
        self.mark_dirty();
        self.flush_cartoon_refresh();
        true
    }

    fn redo(&mut self) -> bool {
        let Some(operation) = self.redo_history.pop_back() else {
            return false;
        };
        self.apply_edit_operation(&operation, HistoryDirection::Redo);
        push_history(&mut self.undo_history, operation);
        self.mark_dirty();
        self.flush_cartoon_refresh();
        true
    }

    fn apply_edit_operation(&mut self, operation: &EditOperation, direction: HistoryDirection) {
        for change in &operation.changes {
            match change {
                EditChange::Display(change) => {
                    let state = history_value(change, direction).clone();
                    match (state, &self.molecule, &mut self.display) {
                        (Some(state), Some(molecule), Some(display)) => {
                            display.restore_edit_state(molecule, state);
                        }
                        (None, _, _) => self.display = None,
                        _ => {}
                    }
                }
                EditChange::NamedSelection(change) => {
                    let value = history_value(&change.value, direction).clone();
                    if let Some(value) = value {
                        self.named_selections
                            .insert(change.name.clone(), value.selection);
                        set_optional_map_value(
                            &mut self.named_selection_expressions,
                            change.name.clone(),
                            value.expression,
                        );
                        set_optional_map_value(
                            &mut self.named_selection_styles,
                            change.name.clone(),
                            value.style,
                        );
                    } else {
                        self.named_selections.remove(&change.name);
                        self.named_selection_expressions.remove(&change.name);
                        self.named_selection_styles.remove(&change.name);
                        self.named_selection_statuses.remove(&change.name);
                    }
                }
                EditChange::Measurement(change) => {
                    self.measurement_lines.retain(|line| line.id != change.id);
                    if let Some(value) = history_value(&change.value, direction) {
                        let index = value.index.min(self.measurement_lines.len());
                        self.measurement_lines.insert(index, value.line.clone());
                    }
                }
                EditChange::HierarchyName(change) => {
                    set_optional_map_value(
                        &mut self.hierarchy_names,
                        change.target,
                        history_value(&change.value, direction).clone(),
                    );
                }
                EditChange::Workspace(change) => {
                    let value = history_value(change, direction);
                    self.inspection = value.inspection;
                    self.hierarchy_selection = value.hierarchy_selection.clone();
                    self.hierarchy_selection_anchor = value.hierarchy_selection_anchor;
                }
            }
        }
        self.recalculate_named_selections();
        self.renderer.update_measurements(&self.measurement_lines);
        self.ui.latest_error = None;
    }

    fn fit(&mut self) {
        if let Some((minimum, maximum)) = self.molecule.as_ref().and_then(Molecule::bounds) {
            self.camera.fit_bounds(minimum, maximum);
        }
    }

    fn mark_dirty(&mut self) {
        if self.molecule.is_none() {
            return;
        }
        self.dirty = true;
        self.document_version = self.document_version.wrapping_add(1);
        self.autosave_due = Some(Instant::now() + AUTOSAVE_DELAY);
    }

    /// Camera state is serialized with the scene, but is deliberately outside `EditTransaction`.
    /// Camera interaction therefore marks the document for saving without touching undo/redo.
    fn mark_camera_dirty(&mut self) {
        self.mark_dirty();
    }

    /// Starts topology work only after an edit has received its final document version.
    /// This keeps background results from being rejected as stale immediately after enqueueing.
    fn flush_cartoon_refresh(&mut self) {
        if self.needs_cartoon_refresh {
            self.schedule_cartoon_job();
        }
    }

    fn remove_recovery_file(&mut self) {
        if let Some(path) = self.recovery_path.take() {
            let _ = fs::remove_file(path);
        }
    }

    fn next_autosave_deadline(&self) -> Option<Instant> {
        std::iter::once(self.autosave_due)
            .chain(
                self.inactive_sessions
                    .iter()
                    .map(|session| session.autosave_due),
            )
            .flatten()
            .min()
    }

    fn autosave_due_documents(&mut self) {
        let now = Instant::now();
        let mut errors = Vec::new();
        if self.dirty && self.autosave_due.is_some_and(|deadline| deadline <= now) {
            let path = self
                .recovery_path
                .clone()
                .map(Ok)
                .unwrap_or_else(|| recovery_path_for(self.active_session_id));
            match path.and_then(|path| {
                self.write_scene(&path)?;
                Ok(path)
            }) {
                Ok(path) => {
                    self.recovery_path = Some(path);
                    self.autosave_due = None;
                }
                Err(error) => {
                    errors.push(error.to_string());
                    self.autosave_due = Some(now + Duration::from_secs(30));
                }
            }
        }
        for session in &mut self.inactive_sessions {
            if !session.dirty || !session.autosave_due.is_some_and(|deadline| deadline <= now) {
                continue;
            }
            let path = session
                .recovery_path
                .clone()
                .map(Ok)
                .unwrap_or_else(|| recovery_path_for(session.id));
            match path.and_then(|path| {
                session.write_scene(&path)?;
                Ok(path)
            }) {
                Ok(path) => {
                    session.recovery_path = Some(path);
                    session.autosave_due = None;
                }
                Err(error) => {
                    errors.push(error.to_string());
                    session.autosave_due = Some(now + Duration::from_secs(30));
                }
            }
        }
        if !errors.is_empty() {
            self.ui.latest_error = Some(format!("autosave failed: {}", errors.join("; ")));
        }
    }

    fn apply_camera_update(&mut self, update: CameraUpdate) {
        self.camera.set_clip_planes(update.near, update.far);
        self.camera
            .set_lens(update.focal_length_mm, update.sensor_height_mm);
        self.camera.depth_of_field.enabled = update.dof_enabled;
        self.camera.depth_of_field.f_stop = update.f_stop.clamp(0.1, 22.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use astra::{
        molecule::{Atom, Element},
        selection::{evaluate, parse_selection},
    };

    fn measurement_molecule() -> Molecule {
        let atom = |serial, residue_number, position| Atom {
            serial,
            name: format!("A{serial}"),
            element: Element::C,
            residue_name: "GLY".into(),
            residue_number,
            insertion_code: None,
            chain_id: "A".into(),
            position,
            occupancy: 1.0,
            b_factor: 0.0,
            hetero: false,
        };
        Molecule {
            atoms: vec![
                atom(1, 1, Vec3::ZERO),
                atom(2, 1, Vec3::new(2.0, 0.0, 0.0)),
                atom(3, 2, Vec3::new(5.0, 0.0, 0.0)),
            ],
            bonds: Vec::new(),
        }
    }

    #[test]
    fn shift_selects_range_and_command_toggles_one_target() {
        let chain_0 = InspectionTarget::Chain(0);
        let chain_1 = InspectionTarget::Chain(1);
        let chain_2 = InspectionTarget::Chain(2);
        let chain_3 = InspectionTarget::Chain(3);
        let mut selection = BTreeSet::new();
        let mut anchor = None;
        assert_eq!(
            update_hierarchy_selection(
                &mut selection,
                &mut anchor,
                chain_0,
                HierarchySelectionGesture::Replace,
                None,
            ),
            Some(chain_0)
        );
        let range = [chain_0, chain_1, chain_2, chain_3];
        assert_eq!(
            update_hierarchy_selection(
                &mut selection,
                &mut anchor,
                chain_3,
                HierarchySelectionGesture::Range,
                Some(&range),
            ),
            Some(chain_3)
        );
        assert_eq!(selection.len(), 4);
        update_hierarchy_selection(
            &mut selection,
            &mut anchor,
            chain_1,
            HierarchySelectionGesture::Toggle,
            None,
        );
        assert!(!selection.contains(&chain_1));
        assert_eq!(anchor, Some(chain_0));
    }

    #[test]
    fn hierarchy_range_is_inclusive_in_both_directions() {
        let ordered = [
            InspectionTarget::Atom(4),
            InspectionTarget::Atom(8),
            InspectionTarget::Atom(12),
        ];
        assert_eq!(
            inclusive_target_range(&ordered, ordered[2], ordered[0]),
            Some(ordered.to_vec())
        );
    }

    #[test]
    fn measurement_endpoint_accepts_one_atom_or_one_residue() {
        let molecule = measurement_molecule();
        let residue = evaluate(&parse_selection("resi 1").unwrap(), &molecule);
        let endpoint = measurement_endpoint(&molecule, "residue", &residue).unwrap();
        assert_eq!(endpoint.position, Vec3::new(1.0, 0.0, 0.0));

        let atom = evaluate(&parse_selection("serial 3").unwrap(), &molecule);
        let endpoint = measurement_endpoint(&molecule, "atom", &atom).unwrap();
        assert_eq!(endpoint.position, Vec3::new(5.0, 0.0, 0.0));

        let multiple = evaluate(&parse_selection("all").unwrap(), &molecule);
        assert!(measurement_endpoint(&molecule, "multiple", &multiple).is_err());
    }

    #[test]
    fn edit_history_keeps_only_the_newest_fifty_events() {
        let mut history = VecDeque::new();
        for index in 0..60 {
            push_history(
                &mut history,
                EditOperation {
                    changes: vec![EditChange::Workspace(Box::new(ValueChange {
                        before: WorkspaceSelection {
                            inspection: None,
                            hierarchy_selection: BTreeSet::new(),
                            hierarchy_selection_anchor: None,
                        },
                        after: WorkspaceSelection {
                            inspection: Some(InspectionTarget::Atom(index)),
                            hierarchy_selection: BTreeSet::new(),
                            hierarchy_selection_anchor: None,
                        },
                    }))],
                },
            );
        }
        assert_eq!(history.len(), EDIT_HISTORY_LIMIT);
        assert_eq!(
            history.front().and_then(operation_inspection),
            Some(InspectionTarget::Atom(10))
        );
        assert_eq!(
            history.back().and_then(operation_inspection),
            Some(InspectionTarget::Atom(59))
        );
    }

    fn operation_inspection(operation: &EditOperation) -> Option<InspectionTarget> {
        operation.changes.iter().find_map(|change| match change {
            EditChange::Workspace(change) => change.after.inspection,
            _ => None,
        })
    }

    #[test]
    fn pdb_ids_are_normalized_without_allowing_path_components() {
        assert_eq!(normalize_pdb_id(" 4r8p ").unwrap(), "4R8P");
        assert_eq!(normalize_pdb_id("pdb_00004hhb").unwrap(), "PDB_00004HHB");
        assert!(normalize_pdb_id("../../4r8p").is_err());
        assert!(normalize_pdb_id("abc").is_err());
        assert_eq!(
            pdb_download_url("4R8P"),
            "https://files.rcsb.org/download/4R8P.cif"
        );
    }

    #[test]
    fn molecule_ids_prefer_structure_metadata_and_fall_back_to_filename() {
        let pdb = b"HEADER    TEST                                    01-JAN-00   1ABC\n";
        assert_eq!(molecule_id_from_structure(pdb, "renamed.pdb"), "1ABC");
        let cif = b"data_2xyz\n_entry.id 2xyz\n";
        assert_eq!(molecule_id_from_structure(cif, "renamed.cif"), "2XYZ");
        assert_eq!(molecule_id_from_structure(b"ATOM", "4r8p.pdb.gz"), "4R8P");
        assert_eq!(
            molecule_id_from_filename("custom-model.mol"),
            "custom-model"
        );
    }

    #[test]
    fn save_as_enforces_the_molecule_extension() {
        assert_eq!(
            molecule_path(PathBuf::from("scene.pdb")),
            PathBuf::from("scene.mol")
        );
        assert_eq!(
            molecule_path(PathBuf::from("scene.MOL")),
            PathBuf::from("scene.MOL")
        );
    }

    #[test]
    fn atomic_write_replaces_complete_files_and_limited_read_checks_metadata() {
        let directory = env::temp_dir().join(format!(
            "astra-p0-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("scene.mol");
        atomic_write(&path, b"first complete scene").unwrap();
        atomic_write(&path, b"second complete scene").unwrap();
        assert_eq!(
            read_local_file_limited(&path).unwrap(),
            b"second complete scene"
        );

        let oversized = directory.join("oversized.cif");
        fs::File::create(&oversized)
            .unwrap()
            .set_len(MAX_LOCAL_FILE_SIZE + 1)
            .unwrap();
        assert!(read_local_file_limited(&oversized).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires access to files.rcsb.org"]
    fn fetches_and_validates_a_real_rcsb_entry() {
        let directory = env::temp_dir().join(format!("astra-fetch-test-{}", std::process::id()));
        let path = download_pdb("4R8P", &directory).unwrap();
        let contents = fs::read(&path).unwrap();
        let (molecule, _) = parse_structure(&contents, "4R8P.cif").unwrap();
        assert!(!molecule.atoms.is_empty());
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
